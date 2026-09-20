import queue
import threading
import time
from math import pi

import numpy as np
from gym import spaces

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

from .realsense import RealSense
from .robotiq_wrapper import RobotiqWrapper
from .xarm_wrapper import XArmWrapper


manual_done = False
manual_failure = False


def _on_key_press(key):
    """A number finishes successfully; a letter finishes as failure."""
    global manual_done, manual_failure
    try:
        char = getattr(key, 'char', None)
        if char and char.isdigit():
            manual_done = True
            manual_failure = False
            print(f"\n[Peg] Key '{char}': success")
        elif char and char.isalpha():
            manual_done = True
            manual_failure = True
            print(f"\n[Peg] Key '{char}': failure")
    except AttributeError:
        pass


keyboard_listener = None
if keyboard is not None:
    keyboard_listener = keyboard.Listener(on_press=_on_key_press)
    keyboard_listener.start()


def precise_wait(end_time, slack_time=0.001):
    wait_time = end_time - time.monotonic()
    if wait_time <= 0:
        return
    if wait_time > slack_time:
        time.sleep(wait_time - slack_time)
    while time.monotonic() < end_time:
        pass


class PegEnv:
    """Single-stage xArm + Robotiq real-robot environment for peg insertion."""

    # Reset pose and control frequency used by teleop_original.py.
    INIT_JOINTS = np.array([2.6, -17.6, -38.0, 1.7, 58.3, -88.9])

    def __init__(
        self,
        robot_ip='192.168.1.202',
        dt=1 / 20,
        num_point_cloud=1024,
        init_joints=None,
        max_episode_steps=500,
        smooth_penalty=0.001,
        success_reward=2.0,
        failure_reward=-1.0,
        require_reset_confirmation=True,
    ):
        self.dt = float(dt)
        self.num_point_cloud = int(num_point_cloud)
        self.max_episode_steps = int(max_episode_steps)
        self.smooth_penalty = float(smooth_penalty)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.require_reset_confirmation = bool(require_reset_confirmation)
        self.init_joints = np.asarray(
            self.INIT_JOINTS if init_joints is None else init_joints,
            dtype=np.float64,
        )
        if self.init_joints.shape != (6,):
            raise ValueError(f'init_joints must have shape (6,), got {self.init_joints.shape}')

        self.xarm = XArmWrapper(
            joints_init=self.init_joints,
            ip=robot_ip,
        )
        self.gripper = RobotiqWrapper(robot='xarm')
        self.camera = RealSense(num_points=self.num_point_cloud)
        self.camera.start()

        self.camera_queue = queue.Queue(maxsize=1)
        self._camera_running = True
        self._closed = False
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

        self.action_space = spaces.Box(
            low=np.array([-np.inf] * 6 + [0.0, 0.0], dtype=np.float64),
            high=np.array([np.inf] * 6 + [1.0, 1.0], dtype=np.float64),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 1, shape=(3, 84, 84), dtype=np.float32),
            'agent_pos': spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            'ee_pose': spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            'point_cloud': spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.num_point_cloud, 3),
                dtype=np.float32,
            ),
        })

        self.env_step = 0
        self.done = False
        self.pre_action = None
        self.t_start = time.monotonic()
        print('Peg Env Init Done')

    def _camera_loop(self):
        while self._camera_running:
            try:
                frame = self.camera.get_frame(require_pc=True)
            except Exception:
                if self._camera_running:
                    raise
                break
            if self.camera_queue.full():
                self.camera_queue.get_nowait()
            self.camera_queue.put(frame)

    def _get_frame(self):
        while self._camera_running:
            try:
                return self.camera_queue.get(timeout=0.1)
            except queue.Empty:
                pass
        raise RuntimeError('Camera stopped before a frame was available')

    def _get_obs(self):
        joints = self.xarm.get_joint()
        tcp_pose = self.xarm.get_position()
        gripper_state = self.gripper.get_state()
        frame = self._get_frame()

        return {
            'agent_pos': np.concatenate([joints, [gripper_state]]).astype(np.float32),
            'ee_pose': np.concatenate([tcp_pose, [gripper_state]]).astype(np.float32),
            'point_cloud': np.asarray(frame['point_cloud'], dtype=np.float32),
            'image': np.zeros((3, 84, 84), dtype=np.float32),
        }

    def reset(self):
        global manual_done, manual_failure
        self.env_step = 0
        self.done = False
        self.pre_action = None

        self.xarm.reset()
        self.gripper.open()
        time.sleep(0.5)
        if self.require_reset_confirmation:
            input('Press Enter after resetting the peg scene...')

        # Ignore any label key pressed while the scene was being reset.
        manual_done = False
        manual_failure = False
        self.t_start = time.monotonic()
        return self._get_obs()

    def step(self, action):
        global manual_done, manual_failure
        action = np.asarray(action, dtype=np.float64)
        if action.shape not in ((7,), (8,)):
            raise ValueError(f'action must have shape (7,) or (8,), got {action.shape}')
        if not np.all(np.isfinite(action)):
            raise ValueError('action contains NaN or infinity')

        self.env_step += 1
        target = np.empty(6, dtype=np.float64)
        target[:3] = action[:3] * 1000.0
        target[3:] = action[3:6] * 180.0 / pi
        gripper_action = float(np.clip(action[6], 0.0, 1.0))

        self.xarm.set_servo_cartesian(target)
        self.gripper.set_position(int(gripper_action * 255))
        precise_wait(self.t_start + self.env_step * self.dt)

        self.done, timeout = self.terminate()

        success = bool(manual_done and not manual_failure)
        physical_action = action[:7]
        reward = -1.0 / self.max_episode_steps
        if self.pre_action is not None:
            reward -= self.smooth_penalty * np.linalg.norm(
                physical_action - self.pre_action
            )
        if self.done:
            reward += self.success_reward if success else self.failure_reward
        self.pre_action = physical_action.copy()
        obs = self._get_obs()

        if self.done:
            self.gripper.open()
            manual_done = False
            manual_failure = False

        info = {
            'is_success': success,
            'timeout': timeout,
        }
        return obs, reward, self.done, info

    def terminate(self):
        return bool(manual_done), False

    def render(self, mode='rgb_array'):
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._camera_running = False
        self.xarm.close()
        self.gripper.open()
        self.camera.stop()
        self.camera_thread.join(timeout=2.0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
