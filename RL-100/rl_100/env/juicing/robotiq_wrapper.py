import time
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
        self.change_state = True
        
        self.gripper_thread = threading.Thread(target=self._monitor_gripper)
        self.gripper_thread.daemon = True
        self.gripper_thread.start()

    def _monitor_gripper(self):
        while True:
            if self.change_state:
                if self.current_state == 'open':
                    self.gripper.open()
                elif self.current_state == 'open_half':
                    self.gripper.goTo(145)
                elif self.current_state == 'close':
                    self.gripper.close()
                elif self.current_state == 'position':
                    self.gripper.goTo(self.current_position)
                self.change_state = False
            time.sleep(1 / 20)

    def open(self):
        if self.current_state != 'open':
            self.current_state = 'open'
            self.change_state = True

    def open_half(self):
        if self.current_state != 'open_half':
            self.current_state = 'open_half'
            self.change_state = True
    
    def close(self):
        if self.current_state != 'close':
            self.current_state = 'close'
            self.change_state = True

    def set_position(self, position):
        """连续控制夹爪位置
        Args:
            position (int): 夹爪位置，0-255 (0为完全张开，255为完全闭合)
        """
        position = max(0, min(255, int(position)))  # 确保在0-255范围内
        if abs(self.current_position - position) > 1:  # 添加死区避免频繁更新
            self.current_position = position
            self.current_state = 'position'
            self.change_state = True

    def get_state(self):
        if self.current_state == 'position':
            return self.current_position / 255.0  # 返回0-1的归一化值
        else:
            return 1 if self.current_state == 'close' else 0
