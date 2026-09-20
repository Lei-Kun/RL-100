import os
import sys
import time
import enum
from math import pi
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import json
import datetime
import pickle

import io
from contextlib import redirect_stdout
from pathlib import Path

from diffusion_policy_3d.env.juicing.xarm_wrapper import XArmWrapper
from diffusion_policy_3d.env.juicing.robotiq_wrapper import RobotiqWrapper
from diffusion_policy_3d.env.juicing.realsense import RealSense
from diffusion_policy_3d.env.juicing.trajectory_recorder import TrajectoryRecorder
from typing import Any, NamedTuple, List, Dict, Optional
from dm_env import StepType, specs
from collections import OrderedDict
from gym import spaces
import threading
from collections import deque
import queue
from pynput import keyboard
from copy import deepcopy

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)
    

is_done = False
manual_failure_requested = False
# is_success = False
def on_press(key):
    global is_done, manual_failure_requested
    try:
        if hasattr(key, 'vk') and key.vk == 65437:
            # Numpad Enter key – treat as success / continuation signal.
            is_done = True
            manual_failure_requested = False
            return

        if hasattr(key, 'char'):
            key_char = key.char.lower()
            if '0' <= key_char <= '9':
                # Numeric key – interpret as explicit success confirmation.
                is_done = True
                manual_failure_requested = False
            elif 'a' <= key_char <= 'z':
                # Alphabetic key – interpret as manual failure / abort.
                is_done = True
                manual_failure_requested = True
    except AttributeError:
        # Ignore special keys that don't carry a `char` attribute.
        pass

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()    


def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


class JuicingEnv:
    # INIT_JOINTS = np.array([-22.1, -21, -36.3, -0.3, 58.3, -22.7])  # Initial joint positions in degrees, vertical
    # END_JOINTS = np.array([-10.4, -76.3, -8.1, 59.5, 53.4, 109.2])   # End joint positions in degrees
    INIT_JOINTS = np.array([-91.4, 10.6, -18.8, -276.6, 76.4, -79.1])  # Initial joint positions in degrees 2, parallel
    END_JOINTS = np.array([-47.4, -2.8, -32.3, -287.6, 107.2, 128])   # End joint positions in degrees 2
    # INIT_JOINTS = np.array([-16.3, 2, -85.7, -323.8, 95.1, 74.1])  # Initial joint positions in degrees for stage 2
    # END_JOINTS = np.array([-22.9, -41.2, -49, -316.5, 108.1, -18.8])   # End joint positions in degrees 2 for stage 2

    def __init__(
        self, 
        robot_ip='192.168.1.202',
        dt=1/30,
        use_camera=True,
        num_point_cloud=1024,
        use_point_cloud=True,
        episode_log_enabled=False,
        episode_log_dir=None,
        episode_log_prefix='juicing',
        stop_mode='auto',
        enable_safety_guard=True,
        max_pos_step_mm=12.0,
        max_rot_step_deg=12.0,
        fault_cmd_pos_mm=50.0,
        fault_cmd_rot_deg=45.0,
        max_measured_pos_jump_m=0.003,
        fault_measured_rot_deg=15.0,
        stage3_max_rot_step_deg=12.0,
        stage3_orientation_gate_enabled=True,
        stage3_orientation_burst_deg=30.0,
        stage3_orientation_burst_window_deg=20.0,
        stage3_orientation_burst_window_steps=3,
        stage3_orientation_burst_hits=2,
        stage3_orientation_freeze_steps=6,
        stage3_guarded_rot_step_deg=4.0,
        stage3_orientation_release_deg=12.0,
        stage3_orientation_release_stable_steps=3,
        replay_bypass_safety=True,
    ):
        print("Juicing Env Init")
        
        # Initialize XArm and gripper
        self.xarm = XArmWrapper(
            joints_init=self.INIT_JOINTS,
            ip=robot_ip,
            enable_safety_guard=enable_safety_guard,
            max_pos_step_mm=max_pos_step_mm,
            max_rot_step_deg=max_rot_step_deg,
            fault_cmd_pos_mm=fault_cmd_pos_mm,
            fault_cmd_rot_deg=fault_cmd_rot_deg,
        )
        self.gripper = RobotiqWrapper(robot='xarm')
        
        self.dt = dt
        self.use_camera = use_camera
        self.stop_mode = str(stop_mode).strip().lower()
        if self.stop_mode not in ('auto', 'manual'):
            raise ValueError(f"Unsupported stop_mode: {stop_mode}")
        self.enable_safety_guard = enable_safety_guard
        self.max_pos_step_mm = float(max_pos_step_mm)
        self.max_rot_step_deg = float(max_rot_step_deg)
        self.fault_cmd_pos_mm = float(fault_cmd_pos_mm)
        self.fault_cmd_rot_deg = float(fault_cmd_rot_deg)
        self.max_measured_pos_jump_m = float(max_measured_pos_jump_m)
        self.fault_measured_rot_deg = float(fault_measured_rot_deg)
        self.stage3_max_rot_step_deg = float(stage3_max_rot_step_deg)
        self.stage3_orientation_gate_enabled = bool(stage3_orientation_gate_enabled)
        self.stage3_orientation_burst_deg = float(stage3_orientation_burst_deg)
        self.stage3_orientation_burst_window_deg = float(stage3_orientation_burst_window_deg)
        self.stage3_orientation_burst_window_steps = max(1, int(stage3_orientation_burst_window_steps))
        self.stage3_orientation_burst_hits = max(1, int(stage3_orientation_burst_hits))
        self.stage3_orientation_freeze_steps = max(1, int(stage3_orientation_freeze_steps))
        self.stage3_guarded_rot_step_deg = float(stage3_guarded_rot_step_deg)
        self.stage3_orientation_release_deg = float(stage3_orientation_release_deg)
        self.stage3_orientation_release_stable_steps = max(1, int(stage3_orientation_release_stable_steps))
        self.replay_bypass_safety = bool(replay_bypass_safety)
        self.execution_profile = 'stage1_policy'
        self.profile_safety_enabled = False
        self.stage3_orientation_mode = 'normal'
        self.stage3_frozen_orientation_deg = None
        self.stage3_freeze_steps_remaining = 0
        self.stage3_release_stable_count = 0
        self.stage3_recent_rotation_bursts = deque(maxlen=self.stage3_orientation_burst_window_steps)
        self.stage3_last_trigger_reason = None
        self.episode_log_enabled = episode_log_enabled
        default_episode_dir = Path(__file__).resolve().parent / 'episodes'
        self.episode_log_dir = Path(episode_log_dir) if episode_log_dir else default_episode_dir
        self.episode_log_prefix = self._sanitize_log_token(episode_log_prefix) or 'juicing'
        self.episode_log_task_name = self.episode_log_prefix
        self.episode_log_stage_label = None
        self._episode_counter = 0
        self._episode_log_file = None
        self._episode_log_path = None
        self._episode_log_step_count = 0
        self._episode_log_last_action = None
        self._episode_log_last_action_rpy_deg = None
        self._episode_log_last_action_rpy_unwrapped_deg = None
        self._episode_log_last_cmd = None
        self._episode_log_last_cmd_rpy_deg = None
        self._episode_log_last_cmd_rpy_unwrapped_deg = None
        self._episode_log_last_measured_pose = None
        self._episode_log_last_measured_rpy_deg = None
        self._episode_log_last_measured_rpy_unwrapped_deg = None
        self._episode_log_metrics = {}
        self._episode_log_open_monotonic = None
        self._episode_log_num_clipped_steps = 0
        self._episode_log_fault_step = None
        self._episode_log_fault_reason = None
        
        self._cur_qpos = None
        self._cur_tcp = None  # (xyz, euler)
        
        # Initialize camera
        self.camera = RealSense(num_points=num_point_cloud)
        self.camera.start()
        self.camera_thread = threading.Thread(target=self._get_camera_frame)
        self.camera_thread.start()
        self.camera_queue = queue.Queue(maxsize=1)
        self.pre_action = None
        
        # Initialize trajectory recorder
        self.trajectory_recorder = TrajectoryRecorder()
        
        # Environment parameters
        number_channel = 3
        obs_sensor_dim = 7  # 6 DOF pose + 1 gripper state
        act_dim = 7         # 6 DOF pose + 1 gripper action
        use_point_cloud = True
        self.num_point_cloud = num_point_cloud
        
        # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float64
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(number_channel, 84, 84),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=1,
                shape=(84, 84),
                dtype=np.float32
            ),
            'depth_scale': spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_sensor_dim,),
                dtype=np.float32
            ),
            'ee_pose': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_sensor_dim,),
                dtype=np.float32
            ),
        })
        
        if use_point_cloud:
            self.observation_space['point_cloud'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_point_cloud, 3),
                dtype=np.float32
            )
        
        self.demo_data = []
        print("Juicing Env Init Done")

    def _get_camera_frame(self):
        while True:
            frame = self.camera.get_frame(require_pc=True)
            if self.camera_queue.full():
                self.camera_queue.get()
            self.camera_queue.put(frame)

    def get_frame(self):
        while True:
            if self.camera_queue.full():
                frame = self.camera_queue.get()
                assert not self.camera_queue.full()
                return frame
            time.sleep(1 / 300)

    def set_episode_log_context(self, task_name=None, stage_label=None):
        if task_name is not None:
            sanitized = self._sanitize_log_token(task_name)
            if sanitized:
                self.episode_log_task_name = sanitized
        if stage_label is not None:
            self.episode_log_stage_label = self._sanitize_log_token(stage_label)

    def set_execution_profile(self, profile_name):
        self.execution_profile = profile_name
        self._reset_stage3_orientation_gate()
        if profile_name == 'stage3_policy':
            self.profile_safety_enabled = self.enable_safety_guard
            self.xarm.configure_safety(
                enable_safety_guard=self.profile_safety_enabled,
                max_pos_step_mm=self.max_pos_step_mm,
                max_rot_step_deg=self.stage3_max_rot_step_deg,
                fault_cmd_pos_mm=self.fault_cmd_pos_mm,
                fault_cmd_rot_deg=self.fault_cmd_rot_deg,
            )
        elif profile_name == 'fixed_trajectory':
            self.profile_safety_enabled = False
            self.xarm.configure_safety(enable_safety_guard=not self.replay_bypass_safety)
        else:
            self.profile_safety_enabled = False
            self.xarm.configure_safety(
                enable_safety_guard=False,
                max_pos_step_mm=self.max_pos_step_mm,
                max_rot_step_deg=self.max_rot_step_deg,
                fault_cmd_pos_mm=self.fault_cmd_pos_mm,
                fault_cmd_rot_deg=self.fault_cmd_rot_deg,
            )

    def _reset_stage3_orientation_gate(self):
        self.stage3_orientation_mode = 'normal'
        self.stage3_frozen_orientation_deg = None
        self.stage3_freeze_steps_remaining = 0
        self.stage3_release_stable_count = 0
        self.stage3_recent_rotation_bursts = deque(maxlen=self.stage3_orientation_burst_window_steps)
        self.stage3_last_trigger_reason = None

    def _sanitize_log_token(self, value):
        if value is None:
            return None
        token = str(value).strip()
        if not token:
            return None
        sanitized = []
        for ch in token:
            if ch.isalnum() or ch in ('-', '_'):
                sanitized.append(ch)
            else:
                sanitized.append('_')
        return ''.join(sanitized).strip('_')

    def _unwrap_angles_deg(self, angles_deg, reference_deg):
        angles_deg = np.asarray(angles_deg, dtype=np.float64)
        if reference_deg is None:
            return angles_deg.copy()
        reference_deg = np.asarray(reference_deg, dtype=np.float64)
        return reference_deg + ((angles_deg - reference_deg + 180.0) % 360.0 - 180.0)

    def _rotation_distance_deg(self, reference_deg, target_deg):
        reference_rot = R.from_euler('xyz', reference_deg, degrees=True)
        target_rot = R.from_euler('xyz', target_deg, degrees=True)
        return float(np.rad2deg((target_rot * reference_rot.inv()).magnitude()))

    def _step_orientation_towards_target(self, reference_deg, target_deg, max_step_deg):
        reference_rot = R.from_euler('xyz', reference_deg, degrees=True)
        target_rot = R.from_euler('xyz', target_deg, degrees=True)
        delta_rot = target_rot * reference_rot.inv()
        delta_angle_deg = float(np.rad2deg(delta_rot.magnitude()))
        if max_step_deg <= 0 or delta_angle_deg <= max_step_deg:
            return np.asarray(target_deg, dtype=np.float64), delta_angle_deg
        step_scale = max_step_deg / delta_angle_deg
        stepped_rot = R.from_rotvec(delta_rot.as_rotvec() * step_scale) * reference_rot
        return stepped_rot.as_euler('xyz', degrees=True), delta_angle_deg

    def _apply_stage3_orientation_gate(self, target_mm_deg, current_orientation_deg):
        target_mm_deg = np.asarray(target_mm_deg, dtype=np.float64)
        current_orientation_deg = np.asarray(current_orientation_deg, dtype=np.float64)
        gate_info = {
            'mode_before': self.stage3_orientation_mode,
            'mode_after': self.stage3_orientation_mode,
            'triggered': False,
            'trigger_reason': None,
            'frozen': False,
            'freeze_steps_remaining': int(self.stage3_freeze_steps_remaining),
            'raw_rotation_distance_deg': 0.0,
            'gated_target': target_mm_deg.copy(),
        }

        if self.execution_profile != 'stage3_policy' or not self.stage3_orientation_gate_enabled:
            gate_info['raw_rotation_distance_deg'] = self._rotation_distance_deg(
                current_orientation_deg, target_mm_deg[3:]
            )
            return gate_info

        reference_orientation_deg = current_orientation_deg
        if self.stage3_orientation_mode == 'frozen' and self.stage3_frozen_orientation_deg is not None:
            reference_orientation_deg = self.stage3_frozen_orientation_deg

        desired_rotation_distance_deg = self._rotation_distance_deg(
            reference_orientation_deg, target_mm_deg[3:]
        )
        gate_info['raw_rotation_distance_deg'] = desired_rotation_distance_deg

        burst_hit = desired_rotation_distance_deg >= self.stage3_orientation_burst_window_deg
        self.stage3_recent_rotation_bursts.append(bool(burst_hit))
        burst_count = sum(self.stage3_recent_rotation_bursts)
        should_freeze = (
            self.stage3_orientation_mode != 'frozen'
            and (
                desired_rotation_distance_deg >= self.stage3_orientation_burst_deg
                or burst_count >= self.stage3_orientation_burst_hits
            )
        )
        if should_freeze:
            self.stage3_orientation_mode = 'frozen'
            self.stage3_frozen_orientation_deg = current_orientation_deg.copy()
            self.stage3_freeze_steps_remaining = self.stage3_orientation_freeze_steps
            self.stage3_release_stable_count = 0
            if desired_rotation_distance_deg >= self.stage3_orientation_burst_deg:
                self.stage3_last_trigger_reason = (
                    f'burst_deg:{desired_rotation_distance_deg:.3f}'
                )
            else:
                self.stage3_last_trigger_reason = (
                    f'burst_window:{burst_count}/{self.stage3_orientation_burst_window_steps}'
                )
            gate_info['triggered'] = True
            gate_info['trigger_reason'] = self.stage3_last_trigger_reason

        if self.stage3_orientation_mode == 'frozen' and self.stage3_frozen_orientation_deg is not None:
            gate_info['frozen'] = True
            if self.stage3_freeze_steps_remaining > 0:
                self.stage3_freeze_steps_remaining -= 1
                self.stage3_release_stable_count = 0
                gated_orientation_deg = self.stage3_frozen_orientation_deg.copy()
            else:
                if desired_rotation_distance_deg < self.stage3_orientation_release_deg:
                    self.stage3_release_stable_count += 1
                else:
                    self.stage3_release_stable_count = 0
                if self.stage3_release_stable_count >= self.stage3_orientation_release_stable_steps:
                    self.stage3_orientation_mode = 'normal'
                    self.stage3_frozen_orientation_deg = None
                    self.stage3_release_stable_count = 0
                    self.stage3_recent_rotation_bursts.clear()
                    self.stage3_last_trigger_reason = None
                    gate_info['frozen'] = False
                else:
                    gated_orientation_deg, _ = self._step_orientation_towards_target(
                        current_orientation_deg,
                        target_mm_deg[3:],
                        self.stage3_guarded_rot_step_deg,
                    )
                    self.stage3_frozen_orientation_deg = np.asarray(gated_orientation_deg, dtype=np.float64)
            if gate_info['frozen']:
                gate_info['gated_target'][3:] = self.stage3_frozen_orientation_deg.copy()

        gate_info['mode_after'] = self.stage3_orientation_mode
        gate_info['freeze_steps_remaining'] = int(self.stage3_freeze_steps_remaining)
        if gate_info['trigger_reason'] is None:
            gate_info['trigger_reason'] = self.stage3_last_trigger_reason
        return gate_info

    def _pose_m_rad_to_mm_deg(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        converted = np.empty_like(pose, dtype=np.float64)
        converted[:3] = pose[:3] * 1000.0
        converted[3:] = np.rad2deg(pose[3:])
        return converted

    def _json_default(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _record_metric(self, name, value):
        if value is None:
            return
        value = float(value)
        if not np.isfinite(value):
            return
        self._episode_log_metrics.setdefault(name, []).append(value)

    def _write_episode_log_record(self, payload):
        if not self.episode_log_enabled or self._episode_log_file is None:
            return
        self._episode_log_file.write(
            json.dumps(payload, ensure_ascii=True, default=self._json_default) + '\n'
        )
        self._episode_log_file.flush()

    def _start_episode_log(self, initial_pose, initial_gripper_state):
        if not self.episode_log_enabled:
            return

        self._finalize_episode_log(status='reset_interrupted', return_success=False, timeout=False)

        self.episode_log_dir.mkdir(parents=True, exist_ok=True)
        self._episode_counter += 1
        self._episode_log_step_count = 0
        self._episode_log_metrics = {}
        self._episode_log_open_monotonic = time.monotonic()

        self._episode_log_last_action = None
        self._episode_log_last_action_rpy_deg = None
        self._episode_log_last_action_rpy_unwrapped_deg = None
        self._episode_log_last_cmd = None
        self._episode_log_last_cmd_rpy_deg = None
        self._episode_log_last_cmd_rpy_unwrapped_deg = None
        self._episode_log_num_clipped_steps = 0
        self._episode_log_fault_step = None
        self._episode_log_fault_reason = None

        initial_pose = np.asarray(initial_pose, dtype=np.float64)
        initial_pose_mm_deg = self._pose_m_rad_to_mm_deg(initial_pose)
        self._episode_log_last_measured_pose = initial_pose.copy()
        self._episode_log_last_measured_rpy_deg = initial_pose_mm_deg[3:].copy()
        self._episode_log_last_measured_rpy_unwrapped_deg = initial_pose_mm_deg[3:].copy()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = [
            f"episode_{timestamp}",
            self.episode_log_task_name or self.episode_log_prefix,
        ]
        if self.episode_log_stage_label:
            filename_parts.append(self.episode_log_stage_label)
        filename_parts.append(f"ep{self._episode_counter:03d}")
        filename = '_'.join(filename_parts) + '.jsonl'

        self._episode_log_path = self.episode_log_dir / filename
        self._episode_log_file = open(self._episode_log_path, 'a', encoding='utf-8')

        self._write_episode_log_record({
            'record_type': 'meta',
            'episode_index': self._episode_counter,
            'task_name': self.episode_log_task_name,
            'stage_label': self.episode_log_stage_label,
            'dt_sec': self.dt,
            'stop_mode': self.stop_mode,
            'num_point_cloud': self.num_point_cloud,
            'log_path': str(self._episode_log_path),
            'initial_tcp_pose_m_rad': initial_pose,
            'initial_tcp_pose_mm_deg': initial_pose_mm_deg,
            'initial_gripper_state': float(initial_gripper_state),
        })

    def _finalize_episode_log(self, status, return_success=None, timeout=None):
        if not self.episode_log_enabled or self._episode_log_file is None:
            return

        summary_stats = {}
        for name, values in self._episode_log_metrics.items():
            if not values:
                continue
            values_np = np.asarray(values, dtype=np.float64)
            summary_stats[name] = {
                'mean': float(np.mean(values_np)),
                'p95': float(np.percentile(values_np, 95)),
                'p99': float(np.percentile(values_np, 99)),
                'max': float(np.max(values_np)),
            }

        duration_sec = None
        if self._episode_log_open_monotonic is not None:
            duration_sec = float(time.monotonic() - self._episode_log_open_monotonic)

        self._write_episode_log_record({
            'record_type': 'summary',
            'episode_index': self._episode_counter,
            'task_name': self.episode_log_task_name,
            'stage_label': self.episode_log_stage_label,
            'status': status,
            'steps_logged': self._episode_log_step_count,
            'duration_sec': duration_sec,
            'return_success': return_success,
            'timeout': timeout,
            'num_clipped_steps': self._episode_log_num_clipped_steps,
            'fault_trigger_step': self._episode_log_fault_step,
            'fault_reason': self._episode_log_fault_reason,
            'summary_stats': summary_stats,
        })

        self._episode_log_file.close()
        self._episode_log_file = None
        self._episode_log_path = None
        self._episode_log_open_monotonic = None

    def reset(self):
        global is_done, is_success, manual_failure_requested
        print("<RESET>")
        self.env_step = 0
        self.done = False
        is_done = False
        manual_failure_requested = False
        is_success = False
        
        # Reset robot to initial position
        self.xarm.reset()
        self.gripper.open()
        time.sleep(0.5)
        
        # Press enter to continue
        input("Press Enter to continue after resetting the robot...")
        
        # Get current robot state
        xarm_pose = self.xarm.get_position()  # [x, y, z, roll, pitch, yaw] in meters and radians
        xarm_q = self.xarm.get_joint()      # Joint angles in degrees
        gripper_state = self.gripper.get_state() / 255.0  # Normalize to [0, 1]
        
        # Get camera frame
        frame = self.get_frame()
        
        # # Process color image
        # color_img = frame['color']
        # color_img = cv2.resize(color_img, (84, 84))
        # color_img = color_img.astype(np.float32) / 255.0
        # color_img = np.transpose(color_img, (2, 0, 1))  # HWC to CHW
        
        # Process depth image
        # depth_img = frame['depth'].astype(np.float32) * frame['depth_scale']
        # depth_img = cv2.resize(depth_img, (84, 84))
        # depth_img = depth_img / 2.0  # Normalize depth to ~[0, 1]
        
        # Agent position: [x, y, z, roll, pitch, yaw, gripper_state]
        q_pos = np.concatenate([xarm_q, [gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [gripper_state]])
        
        obs = {
            'agent_pos': np.array(q_pos),
            'point_cloud': np.array(frame['point_cloud']),  # (size, num_points, 3)
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.array(frame['depth']).astype(np.int32),
            'depth_scale': np.array([frame['depth_scale']]).astype(np.float32),
            'ee_pose': np.array(ee_pose),
        }
        self.demo_data = []
        self._start_episode_log(initial_pose=xarm_pose, initial_gripper_state=gripper_state)
        
        self.t_start = time.monotonic()
        return obs.copy()

    def reset_end(self):
        print("<RESET END>")
        self.xarm.stop_servo_mode()
        
        # Switch to position control mode
        self.xarm.xarm.set_mode(0)
        self.xarm.xarm.set_state(0)
        
        # Pad END_JOINTS to 7-DOF (already in degrees)
        end_joints_7dof = np.append(self.END_JOINTS, 0)
        
        # Move to end position
        current_joints_7dof = self.xarm.xarm.get_servo_angle()[1]
        out_joints_7dof = end_joints_7dof.copy()
        out_joints_7dof[5] = current_joints_7dof[5]
        self.xarm.xarm.set_servo_angle(angle=out_joints_7dof, speed=128, wait=True)
        self.xarm.xarm.set_servo_angle(angle=end_joints_7dof, speed=128, wait=True)
        
        # Update position but don't restart control loop
        # The next episode's reset() will handle that
        curr_pos = np.array(self.xarm.xarm.get_position()[1])
        self.xarm._target = curr_pos.copy()
        self.xarm._last_target = curr_pos.copy()
        self.xarm.first_reset = False  # Mark that we've done at least one reset
        
        # Press enter to continue
        self.gripper.open()
        time.sleep(0.5)
        input("Press Enter to continue after resetting the robot...")
    
    def reset_for_manual_mode(self):
        """重置到拖拽教学模式，不启动控制循环"""
        print("<RESET FOR MANUAL MODE>")
        self.xarm.stop_servo_mode()
        
        # Switch to position control mode
        self.xarm.xarm.set_mode(0)
        self.xarm.xarm.set_state(0)
        
        # Pad INIT_JOINTS to 7-DOF (already in degrees)
        init_joints_7dof = np.append(self.INIT_JOINTS, 0)
        
        # Move to initial position
        self.xarm.xarm.set_servo_angle(angle=init_joints_7dof, speed=128, wait=True)
        
        # Update position but don't restart control loop
        # The next episode's reset() will handle that
        curr_pos = np.array(self.xarm.xarm.get_position()[1])
        self.xarm._target = curr_pos.copy()
        self.xarm._last_target = curr_pos.copy()
        self.xarm.first_reset = False  # Mark that we've done at least one reset
        
        print("机器人已准备好进入手动拖拽模式")

    def step(self, action):
        global is_done, is_success, manual_failure_requested
        start_time = time.monotonic()
        self.env_step += 1
        t_cycle_end = self.t_start + self.env_step * self.dt
        
        # Get current robot state (like in teleop.py lines 165-172)
        xarm_q = self.xarm.get_joint()
        xarm_pose = self.xarm.get_position()
        xarm_gripper_state = self.gripper.get_state()
        gripper_state_pre = float(xarm_gripper_state)
        
        q_pos = np.concatenate([xarm_q, [xarm_gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state]])
        xarm_pose_before = np.array(xarm_pose, dtype=np.float64)
        xarm_pose_before_mm_deg = self._pose_m_rad_to_mm_deg(xarm_pose_before)

        action_np = np.asarray(action, dtype=np.float64)
        action_pos_m = action_np[:3].copy()
        action_rpy_deg = np.rad2deg(action_np[3:6])
        action_rpy_unwrapped_deg = self._unwrap_angles_deg(
            action_rpy_deg, self._episode_log_last_action_rpy_unwrapped_deg
        )

        prev_action = self._episode_log_last_action
        if prev_action is None:
            action_delta_pos_m = np.zeros(3, dtype=np.float64)
            action_delta_rpy_deg_raw = np.zeros(3, dtype=np.float64)
            action_delta_rpy_deg_unwrapped = np.zeros(3, dtype=np.float64)
        else:
            action_delta_pos_m = action_pos_m - prev_action[:3]
            action_delta_rpy_deg_raw = action_rpy_deg - self._episode_log_last_action_rpy_deg
            action_delta_rpy_deg_unwrapped = (
                action_rpy_unwrapped_deg - self._episode_log_last_action_rpy_unwrapped_deg
            )

        xarm_target = np.zeros(6, dtype=np.float32)
        xarm_target[:3] = action_np[:3] * 1000  # Position in mm (like teleop.py line 179)
        xarm_target[3:] = action_np[3:6] * 180 / pi  # Orientation in degrees (like teleop.py line 180)
        cmd_target = np.asarray(xarm_target, dtype=np.float64)
        cmd_rpy_unwrapped_deg = self._unwrap_angles_deg(
            cmd_target[3:], self._episode_log_last_cmd_rpy_unwrapped_deg
        )
        prev_cmd = self._episode_log_last_cmd
        if prev_cmd is None:
            cmd_delta_pos_mm = np.zeros(3, dtype=np.float64)
            cmd_delta_rpy_deg = np.zeros(3, dtype=np.float64)
            cmd_delta_rpy_deg_unwrapped = np.zeros(3, dtype=np.float64)
        else:
            cmd_delta_pos_mm = cmd_target[:3] - prev_cmd[:3]
            cmd_delta_rpy_deg = cmd_target[3:] - self._episode_log_last_cmd_rpy_deg
            cmd_delta_rpy_deg_unwrapped = (
                cmd_rpy_unwrapped_deg - self._episode_log_last_cmd_rpy_unwrapped_deg
            )

        gripper_target = int((1.0 - action_np[6]) * 255)
        command_info = {
            'raw_target': cmd_target.copy(),
            'safe_target': cmd_target.copy(),
            'raw_delta_pos_mm': cmd_delta_pos_mm.copy(),
            'raw_delta_rot_deg': cmd_delta_rpy_deg_unwrapped.copy(),
            'raw_rotation_distance_deg': float(np.linalg.norm(cmd_delta_rpy_deg_unwrapped)),
            'clipped_delta_pos_mm': np.zeros(3, dtype=np.float64),
            'clipped_delta_rot_deg': np.zeros(3, dtype=np.float64),
            'clipped_rotation_distance_deg': 0.0,
            'faulted': False,
            'fault_reason': None,
        }
        stage3_gate_info = {
            'mode_before': self.stage3_orientation_mode,
            'mode_after': self.stage3_orientation_mode,
            'triggered': False,
            'trigger_reason': None,
            'frozen': False,
            'freeze_steps_remaining': int(self.stage3_freeze_steps_remaining),
            'raw_rotation_distance_deg': 0.0,
            'gated_target': cmd_target.copy(),
        }
        cmd_target_to_send = cmd_target.copy()
        if self.execution_profile == 'stage3_policy':
            stage3_gate_info = self._apply_stage3_orientation_gate(
                cmd_target, xarm_pose_before_mm_deg[3:]
            )
            cmd_target_to_send = np.asarray(stage3_gate_info['gated_target'], dtype=np.float64)

        if not self.done:
            command_info = self.xarm.set_servo_cartesian(cmd_target_to_send)
            print(action_np[6])
            
            self.gripper.set_position(gripper_target)

        precise_wait(t_cycle_end)
        
        xarm_q = self.xarm.get_joint()
        xarm_pose = self.xarm.get_position()
        xarm_gripper_state = self.gripper.get_state()
        
        q_pos = np.concatenate([xarm_q, [xarm_gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state]])
        xarm_pose_after = np.array(xarm_pose, dtype=np.float64)
        xarm_pose_after_mm_deg = self._pose_m_rad_to_mm_deg(xarm_pose_after)
        measured_post_rpy_unwrapped_deg = self._unwrap_angles_deg(
            xarm_pose_after_mm_deg[3:], self._episode_log_last_measured_rpy_unwrapped_deg
        )
        measured_delta_pos_m = xarm_pose_after[:3] - xarm_pose_before[:3]
        measured_delta_rpy_deg_raw = xarm_pose_after_mm_deg[3:] - xarm_pose_before_mm_deg[3:]
        measured_delta_rpy_deg_unwrapped = (
            measured_post_rpy_unwrapped_deg - self._episode_log_last_measured_rpy_unwrapped_deg
        )

        safe_cmd_target = np.asarray(command_info.get('safe_target', cmd_target), dtype=np.float64)
        cmd_clip_pos_mm = np.asarray(command_info.get('clipped_delta_pos_mm', np.zeros(3)), dtype=np.float64)
        cmd_clip_rpy_deg = np.asarray(command_info.get('clipped_delta_rot_deg', np.zeros(3)), dtype=np.float64)
        cmd_clip_pos_norm_mm = float(np.linalg.norm(cmd_clip_pos_mm))
        cmd_clip_rpy_norm_deg = float(np.linalg.norm(cmd_clip_rpy_deg))
        cmd_rotation_distance_deg = float(command_info.get('raw_rotation_distance_deg', np.linalg.norm(cmd_delta_rpy_deg_unwrapped)))
        cmd_clipped_rotation_distance_deg = float(command_info.get('clipped_rotation_distance_deg', 0.0))
        if cmd_clip_pos_norm_mm > 1e-6 or cmd_clip_rpy_norm_deg > 1e-6:
            self._episode_log_num_clipped_steps += 1

        measured_delta_pos_norm_m = float(np.linalg.norm(measured_delta_pos_m))
        measured_delta_rpy_unwrapped_norm = float(np.linalg.norm(measured_delta_rpy_deg_unwrapped))

        if self.profile_safety_enabled:
            if measured_delta_pos_norm_m > self.max_measured_pos_jump_m:
                self.xarm.trigger_fault(
                    f'measured_position_jump_m:{measured_delta_pos_norm_m:.6f}'
                )
            elif (
                self.execution_profile == 'stage3_policy'
                and stage3_gate_info['frozen']
                and stage3_gate_info['raw_rotation_distance_deg'] > self.fault_cmd_rot_deg
                and measured_delta_rpy_unwrapped_norm > self.fault_measured_rot_deg
            ):
                self.xarm.trigger_fault(
                    f'stage3_frozen_rotation_jump_deg:{measured_delta_rpy_unwrapped_norm:.3f}'
                )

        fault_status = self.xarm.get_fault_status()
        safety_fault = bool(fault_status['faulted'])
        safety_fault_reason = fault_status['reason']
        if safety_fault and self._episode_log_fault_step is None:
            self._episode_log_fault_step = self.env_step
            self._episode_log_fault_reason = safety_fault_reason
        
        frame = self.get_frame()
        reward = 0
        
        obs = {
            'agent_pos': np.array(q_pos),
            'point_cloud': np.array(frame['point_cloud']),
            'image': np.random.rand(3, 84, 84),  # set fake image
            'depth': np.array(frame['depth']).astype(np.int32),
            'depth_scale': np.array([frame['depth_scale']]).astype(np.float32),
            'ee_pose': np.array(ee_pose),
        }

        if safety_fault:
            self.done, timeout = True, False
        elif action.shape[-1] == 8:
            self.done, timeout = self.terminate(is_done, action[7] > 0.5)
            print("Predict done value:", action[7], "Predicted done:", action[7] > 0.5)
        else:
            self.done, timeout = self.terminate(is_done)

        return_success = False
        if self.done:
            self.gripper.open()
            time.sleep(0.5)

            if safety_fault:
                print(f"Episode ended due to safety fault: {safety_fault_reason}")
                return_success = False
                reward = 0
            elif timeout:
                print("Episode ended due to timeout. Marking as failure.")
                return_success = False
                reward = 0
            else:
                if self.stop_mode == 'manual':
                    print("\nEpisode ended. Was it successful?")
                    user_input = input("Press Enter or type number for success, letter for failure: ").strip()
                    user_input = user_input[-1] if user_input else ''
                    if (not user_input) or user_input.isdigit():
                        print("Success recorded.")
                        return_success = True
                        reward = 1
                    else:
                        print("Failure recorded.")
                        return_success = False
                        reward = 0
                else:
                    manual_failure = manual_failure_requested
                    if manual_failure:
                        print("Episode interrupted via letter key. Marking as failure.")
                        return_success = False
                        reward = 0
                    else:
                        print("Episode completed without manual failure. Marking as success.")
                        return_success = True
                        reward = 1

            # Reset manual flags for the next step sequence.
            is_done = False
            manual_failure_requested = False
            # is_success = False

        self._episode_log_step_count += 1
        step_wall_time_sec = time.monotonic() - start_time
        self._write_episode_log_record({
            'record_type': 'step',
            'episode_index': self._episode_counter,
            'step_idx': self.env_step,
            't_cycle_end_monotonic': t_cycle_end,
            'action': action_np,
            'action_pos_m': action_pos_m,
            'action_rpy_deg': action_rpy_deg,
            'action_rpy_unwrapped_deg': action_rpy_unwrapped_deg,
            'action_delta_pos_m': action_delta_pos_m,
            'action_delta_pos_norm_m': float(np.linalg.norm(action_delta_pos_m)),
            'action_delta_rpy_deg_raw': action_delta_rpy_deg_raw,
            'action_delta_rpy_deg_raw_norm': float(np.linalg.norm(action_delta_rpy_deg_raw)),
            'action_delta_rpy_deg_unwrapped': action_delta_rpy_deg_unwrapped,
            'action_delta_rpy_deg_unwrapped_norm': float(np.linalg.norm(action_delta_rpy_deg_unwrapped)),
            'cmd_target_mm_deg': cmd_target,
            'cmd_target_gated_mm_deg': cmd_target_to_send,
            'cmd_target_safe_mm_deg': safe_cmd_target,
            'cmd_delta_pos_mm': cmd_delta_pos_mm,
            'cmd_delta_pos_norm_mm': float(np.linalg.norm(cmd_delta_pos_mm)),
            'cmd_delta_rpy_deg_raw': cmd_delta_rpy_deg,
            'cmd_delta_rpy_deg_raw_norm': float(np.linalg.norm(cmd_delta_rpy_deg)),
            'cmd_delta_rpy_deg_unwrapped': cmd_delta_rpy_deg_unwrapped,
            'cmd_delta_rpy_deg_unwrapped_norm': float(np.linalg.norm(cmd_delta_rpy_deg_unwrapped)),
            'cmd_rotation_distance_deg': cmd_rotation_distance_deg,
            'cmd_clip_pos_mm': cmd_clip_pos_mm,
            'cmd_clip_pos_norm_mm': cmd_clip_pos_norm_mm,
            'cmd_clip_rpy_deg': cmd_clip_rpy_deg,
            'cmd_clip_rpy_norm_deg': cmd_clip_rpy_norm_deg,
            'cmd_clipped_rotation_distance_deg': cmd_clipped_rotation_distance_deg,
            'stage3_raw_rotation_distance_deg': float(stage3_gate_info['raw_rotation_distance_deg']),
            'stage3_orientation_mode_before': stage3_gate_info['mode_before'],
            'stage3_orientation_mode': stage3_gate_info['mode_after'],
            'stage3_orientation_frozen': bool(stage3_gate_info['frozen']),
            'stage3_orientation_triggered': bool(stage3_gate_info['triggered']),
            'stage3_orientation_trigger_reason': stage3_gate_info['trigger_reason'],
            'stage3_orientation_freeze_steps_remaining': int(stage3_gate_info['freeze_steps_remaining']),
            'measured_pre_pose_m_rad': xarm_pose_before,
            'measured_pre_pose_mm_deg': xarm_pose_before_mm_deg,
            'measured_post_pose_m_rad': xarm_pose_after,
            'measured_post_pose_mm_deg': xarm_pose_after_mm_deg,
            'measured_post_rpy_unwrapped_deg': measured_post_rpy_unwrapped_deg,
            'measured_delta_pos_m': measured_delta_pos_m,
            'measured_delta_pos_norm_m': measured_delta_pos_norm_m,
            'measured_delta_rpy_deg_raw': measured_delta_rpy_deg_raw,
            'measured_delta_rpy_deg_raw_norm': float(np.linalg.norm(measured_delta_rpy_deg_raw)),
            'measured_delta_rpy_deg_unwrapped': measured_delta_rpy_deg_unwrapped,
            'measured_delta_rpy_deg_unwrapped_norm': measured_delta_rpy_unwrapped_norm,
            'gripper_action': float(action_np[6]),
            'gripper_target': gripper_target,
            'gripper_state_pre': gripper_state_pre,
            'gripper_state_post': float(xarm_gripper_state),
            'reward': reward,
            'done': self.done,
            'timeout': timeout,
            'is_success': return_success,
            'safety_fault': safety_fault,
            'safety_fault_reason': safety_fault_reason,
            'predicted_done': bool(action_np[7] > 0.5) if action_np.shape[-1] == 8 else None,
            'step_wall_time_sec': step_wall_time_sec,
        })

        self._record_metric('action_delta_pos_norm_m', np.linalg.norm(action_delta_pos_m))
        self._record_metric('action_delta_rpy_deg_raw_norm', np.linalg.norm(action_delta_rpy_deg_raw))
        self._record_metric('action_delta_rpy_deg_unwrapped_norm', np.linalg.norm(action_delta_rpy_deg_unwrapped))
        self._record_metric('cmd_delta_pos_norm_mm', np.linalg.norm(cmd_delta_pos_mm))
        self._record_metric('cmd_delta_rpy_deg_raw_norm', np.linalg.norm(cmd_delta_rpy_deg))
        self._record_metric('cmd_delta_rpy_deg_unwrapped_norm', np.linalg.norm(cmd_delta_rpy_deg_unwrapped))
        self._record_metric('cmd_rotation_distance_deg', cmd_rotation_distance_deg)
        self._record_metric('cmd_clip_pos_norm_mm', cmd_clip_pos_norm_mm)
        self._record_metric('cmd_clip_rpy_norm_deg', cmd_clip_rpy_norm_deg)
        self._record_metric('cmd_clipped_rotation_distance_deg', cmd_clipped_rotation_distance_deg)
        self._record_metric('measured_delta_pos_norm_m', measured_delta_pos_norm_m)
        self._record_metric('measured_delta_rpy_deg_raw_norm', np.linalg.norm(measured_delta_rpy_deg_raw))
        self._record_metric('measured_delta_rpy_deg_unwrapped_norm', measured_delta_rpy_unwrapped_norm)

        self._episode_log_last_action = np.concatenate([action_pos_m, action_rpy_deg])
        self._episode_log_last_action_rpy_deg = action_rpy_deg.copy()
        self._episode_log_last_action_rpy_unwrapped_deg = action_rpy_unwrapped_deg.copy()
        self._episode_log_last_cmd = cmd_target.copy()
        self._episode_log_last_cmd_rpy_deg = cmd_target[3:].copy()
        self._episode_log_last_cmd_rpy_unwrapped_deg = cmd_rpy_unwrapped_deg.copy()
        self._episode_log_last_measured_pose = xarm_pose_after.copy()
        self._episode_log_last_measured_rpy_deg = xarm_pose_after_mm_deg[3:].copy()
        self._episode_log_last_measured_rpy_unwrapped_deg = measured_post_rpy_unwrapped_deg.copy()

        if self.done:
            status = 'success' if return_success else 'failure'
            if timeout:
                status = 'timeout'
            elif safety_fault:
                status = 'safety_fault'
            self._finalize_episode_log(status=status, return_success=return_success, timeout=timeout)
        
        return obs, reward, self.done, {
            'is_success': return_success,
            'timeout': timeout,
            'safety_fault': safety_fault,
            'safety_fault_reason': safety_fault_reason,
        }

    def terminate(self, is_done, predict_done=None):
        if is_done or predict_done == True:
        # if is_done:
            return True, False
        else:
            if self.env_step >= 1000:
                return True, True
            else:
                return False, False

    def close(self):
        print("Closing Juicing Environment")
        self._finalize_episode_log(status='closed', return_success=False, timeout=False)
        self.xarm.close()
        self.gripper.open()
        self.camera.stop()

    def render(self, mode='rgb_array'):
        if hasattr(self, 'camera_queue') and not self.camera_queue.empty():
            frame = self.camera_queue.queue[0]
            return frame['color']
        else:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def start_manual_recording(self):
        """开始手动录制轨迹模式（机器人已在INIT位置）"""
        print("设置机器人为拖拽教学模式...")
        
        # 清除错误状态
        self.xarm.xarm.clean_error()
        self.xarm.xarm.clean_warn()
        
        # 设置为教学模式（不移动机器人，保持在当前位置）
        code = self.xarm.xarm.set_mode(2)
        print(f"设置教学模式返回码: {code}")
        
        code = self.xarm.xarm.set_state(0)  
        print(f"设置机器人状态返回码: {code}")
        
        # 检查当前状态
        state = self.xarm.xarm.get_state()
        print(f"当前机器人状态: {state}")
        
        if state[0] == 0:
            print("✅ 教学模式设置成功，机器人现在可以手动拖拽")
        else:
            print(f"⚠️ 机器人状态码: {state}, 但继续尝试...")
        
        self.trajectory_recorder = TrajectoryRecorder()
        print("手动录制模式已启动")
        print("机器人已在初始位置并切换到拖拽教学模式")
        print("控制说明:")
        print("  'r' - 开始/停止录制")
        print("  'p' - 开始/停止重放") 
        print("  's' - 保存轨迹")
        print("  'l' - 加载轨迹")
        print("  'c' - 清除轨迹")
        
        # 进入手动控制循环
        try:
            while True:
                # 获取当前机器人状态
                xarm_q = self.xarm.get_joint()
                xarm_pose = self.xarm.get_position()
                gripper_state = self.gripper.get_state()
                
                # 录制当前状态
                self.trajectory_recorder.record_point(
                    joint_positions=xarm_q,
                    tcp_position=xarm_pose,
                    gripper_state=gripper_state
                )
                
                # 检查是否有重放目标
                playback_target = self.trajectory_recorder.get_playback_target()
                if playback_target is not None:
                    # 重放时切换回伺服模式
                    if not hasattr(self, '_in_playback_mode'):
                        print("开始重放，切换到伺服控制模式...")
                        self.set_execution_profile('fixed_trajectory')
                        self.xarm.clear_fault()
                        self.xarm.start_servo_mode()
                        self._in_playback_mode = True
                    
                    target_joints, target_tcp, target_gripper = playback_target
                    
                    # 执行重放动作 - 需要转换单位
                    tcp_for_xarm = target_tcp[:6].copy()
                    tcp_for_xarm[:3] *= 1000  # m -> mm
                    tcp_for_xarm[3:] = tcp_for_xarm[3:] * 180 / np.pi  # 弧度 -> 度
                    
                    command_info = self.xarm.set_servo_cartesian(tcp_for_xarm)
                    if command_info.get('faulted'):
                        print(f"重放触发安全保护: {command_info.get('fault_reason')}")
                        self.trajectory_recorder.is_playing = False
                        continue
                    gripper_target = int((1.0 - target_gripper) * 255)
                    self.gripper.set_position(gripper_target)
                else:
                    # 不在重放时，确保处于教学模式
                    if hasattr(self, '_in_playback_mode'):
                        print("重放结束，切换回拖拽教学模式...")
                        self.xarm.stop_servo_mode()
                        self.xarm.xarm.set_mode(2)  # 教学模式
                        self.xarm.xarm.set_state(0)
                        delattr(self, '_in_playback_mode')
                
                time.sleep(1/30)  # 30Hz控制频率
                
        except KeyboardInterrupt:
            print("\n手动录制模式结束")
            # 恢复正常控制模式
            print("恢复机器人控制模式...")
            self.xarm.xarm.set_mode(0)  # 位置控制模式
            self.xarm.xarm.set_state(0)
            self.trajectory_recorder.stop()
    
    def replay_saved_trajectory(self, trajectory_file: str = None) -> bool:
        """重放保存的轨迹"""
        if not hasattr(self, 'trajectory_recorder'):
            self.trajectory_recorder = TrajectoryRecorder()

        # 加载轨迹
        self.trajectory_recorder.load_trajectory(trajectory_file)

        if not self.trajectory_recorder.saved_trajectory:
            print("没有可重放的轨迹")
            return False

        print(f"开始重放轨迹，共{len(self.trajectory_recorder.saved_trajectory)}个点")

        # 停止现有控制循环并设置为伺服模式
        print("准备机器人进行轨迹重放...")
        self.set_execution_profile('fixed_trajectory')
        self.xarm.clear_fault()
        self.xarm.start_servo_mode()

        # 强制启动重放
        self.trajectory_recorder.is_playing = True
        self.trajectory_recorder.playback_index = 0
        self.trajectory_recorder.playback_start_time = time.time()

        playback_success = True

        try:
            while self.trajectory_recorder.is_playing:
                playback_target = self.trajectory_recorder.get_playback_target()
                if playback_target is None:
                    break

                target_joints, target_tcp, target_gripper = playback_target

                # 执行重放动作 - 需要转换单位
                tcp_for_xarm = target_tcp[:6].copy()
                tcp_for_xarm[:3] *= 1000  # m -> mm
                tcp_for_xarm[3:] = tcp_for_xarm[3:] * 180 / np.pi  # 弧度 -> 度

                command_info = self.xarm.set_servo_cartesian(tcp_for_xarm)
                if command_info.get('faulted'):
                    print(f"轨迹重放触发安全保护: {command_info.get('fault_reason')}")
                    playback_success = False
                    break
                gripper_target = int((1.0 - target_gripper) * 255)
                self.gripper.set_position(gripper_target)

                time.sleep(1/30)  # 30Hz控制频率

        except KeyboardInterrupt:
            print("\n轨迹重放被中断")
            playback_success = False
        except Exception as exc:
            print(f"轨迹重放出现异常: {exc}")
            playback_success = False
        finally:
            # 确保播放状态被重置
            self.trajectory_recorder.is_playing = False

        print("轨迹重放完成")
        print("恢复机器人到正常控制模式...")
        self.xarm.stop_servo_mode()
        self.xarm.xarm.set_mode(0)  # 位置控制模式
        self.xarm.xarm.set_state(0)

        return playback_success

    def __del__(self):
        try:
            if hasattr(self, 'trajectory_recorder'):
                self.trajectory_recorder.stop()
            self.close()
        except:
            pass
