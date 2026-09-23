import threading
from pyrobotiqgripper import RobotiqGripper

class RobotiqWrapper:
    def __init__(self, robot):
        portname='/dev/ttyUSB0'
        self.gripper = RobotiqGripper(portname=portname)
        # self.gripper.reset()
        self.gripper.activate()
        
        self.current_state = 'open'
        self.current_position = 0  # 0-255, 0为张开，255为闭合
        self._command_condition = threading.Condition()
        self._command_version = 1
        self._completed_version = 0
        self._last_error_version = 0
        self._last_command_error = None
        
        self.gripper_thread = threading.Thread(target=self._monitor_gripper)
        self.gripper_thread.daemon = True
        self.gripper_thread.start()

    def _monitor_gripper(self):
        while True:
            with self._command_condition:
                self._command_condition.wait_for(
                    lambda: self._command_version > self._completed_version)
                command_version = self._command_version
                command_state = self.current_state
                command_position = self.current_position

            command_error = None
            try:
                if command_state == 'open':
                    self.gripper.open()
                elif command_state == 'open_half':
                    self.gripper.goTo(145)
                elif command_state == 'close':
                    self.gripper.close()
                elif command_state == 'position':
                    self.gripper.goTo(command_position)
            except Exception as exc:
                command_error = exc

            with self._command_condition:
                self._completed_version = max(
                    self._completed_version, command_version)
                self._last_error_version = command_version
                self._last_command_error = command_error
                self._command_condition.notify_all()

            if command_error is not None:
                print(f'Robotiq command failed: {command_error}')

    def _request_command(self, state, position=None, wait=False, timeout=15.0):
        with self._command_condition:
            self.current_state = state
            if position is not None:
                self.current_position = position
            self._command_version += 1
            command_version = self._command_version
            self._command_condition.notify_all()

            if not wait:
                return

            completed = self._command_condition.wait_for(
                lambda: self._completed_version >= command_version,
                timeout=timeout,
            )
            if not completed:
                raise TimeoutError(
                    f'Robotiq {state} command did not complete within {timeout}s')
            if (self._last_error_version >= command_version
                    and self._last_command_error is not None):
                raise RuntimeError(
                    f'Robotiq {state} command failed') from self._last_command_error

    def open(self, wait=False):
        self._request_command('open', wait=wait)

    def open_half(self):
        self._request_command('open_half')
    
    def close(self):
        self._request_command('close')

    def set_position(self, position):
        """连续控制夹爪位置
        Args:
            position (int): 夹爪位置，0-255 (0为完全张开，255为完全闭合)
        """
        position = max(0, min(255, int(position)))  # 确保在0-255范围内
        with self._command_condition:
            if abs(self.current_position - position) > 1:  # 添加死区避免频繁更新
                self.current_position = position
                self.current_state = 'position'
                self._command_version += 1
                self._command_condition.notify_all()

    def get_state(self):
        with self._command_condition:
            if self.current_state == 'position':
                return self.current_position / 255.0  # 返回0-1的归一化值
            return 1 if self.current_state == 'close' else 0
