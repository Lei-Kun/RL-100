import time
import numpy as np
import threading
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R, Slerp


def pose_interp(pose1, pose2, alpha):
    # pose: [x, y, z, roll, pitch, yaw], angle unit is deg
    pos1 = np.asarray(pose1[:3], dtype=np.float64)
    pos2 = np.asarray(pose2[:3], dtype=np.float64)
    pos_interp = pos1 + (pos2 - pos1) * alpha

    rot1 = R.from_euler('xyz', pose1[3:], degrees=True)
    rot2 = R.from_euler('xyz', pose2[3:], degrees=True)
    slerp = Slerp([0, 1], R.from_matrix(np.stack([rot1.as_matrix(), rot2.as_matrix()])))
    euler_interp = slerp([alpha])[0].as_euler('xyz', degrees=True)
    return np.concatenate([pos_interp, euler_interp])


class XArmWrapper:
    def __init__(
        self,
        joints_init,
        ip='192.168.1.202',
        enable_safety_guard=True,
        max_pos_step_mm=12.0,
        max_rot_step_deg=12.0,
        fault_cmd_pos_mm=50.0,
        fault_cmd_rot_deg=45.0,
    ):
        self.xarm = XArmAPI(ip)
        self.joints_init = joints_init
        self.first_reset = True
        self._running = False
        self._lock = threading.Lock()
        self.enable_safety_guard = enable_safety_guard
        self.max_pos_step_mm = float(max_pos_step_mm)
        self.max_rot_step_deg = float(max_rot_step_deg)
        self.fault_cmd_pos_mm = float(fault_cmd_pos_mm)
        self.fault_cmd_rot_deg = float(fault_cmd_rot_deg)
        self._thread = None
        self._target = None
        self._last_target = None
        self._last_safe_target = None
        self._last_measured_pose = None
        self._last_measured_rpy_unwrapped_deg = None
        self._faulted = False
        self._fault_reason = None
        self.reset()

    def configure_safety(
        self,
        enable_safety_guard=None,
        max_pos_step_mm=None,
        max_rot_step_deg=None,
        fault_cmd_pos_mm=None,
        fault_cmd_rot_deg=None,
    ):
        with self._lock:
            if enable_safety_guard is not None:
                self.enable_safety_guard = bool(enable_safety_guard)
            if max_pos_step_mm is not None:
                self.max_pos_step_mm = float(max_pos_step_mm)
            if max_rot_step_deg is not None:
                self.max_rot_step_deg = float(max_rot_step_deg)
            if fault_cmd_pos_mm is not None:
                self.fault_cmd_pos_mm = float(fault_cmd_pos_mm)
            if fault_cmd_rot_deg is not None:
                self.fault_cmd_rot_deg = float(fault_cmd_rot_deg)

    def _limit_norm(self, delta, max_norm):
        delta = np.asarray(delta, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if max_norm <= 0 or norm <= max_norm:
            return delta.copy()
        return delta * (max_norm / norm)

    def _compute_rotation_delta(self, reference_euler_deg, target_euler_deg):
        reference_rot = R.from_euler('xyz', reference_euler_deg, degrees=True)
        target_rot = R.from_euler('xyz', target_euler_deg, degrees=True)
        delta_rot = target_rot * reference_rot.inv()
        delta_rotvec_deg = np.rad2deg(delta_rot.as_rotvec())
        delta_angle_deg = float(np.rad2deg(delta_rot.magnitude()))
        return {
            'reference_rot': reference_rot,
            'target_rot': target_rot,
            'delta_rotvec_deg': delta_rotvec_deg,
            'delta_angle_deg': delta_angle_deg,
        }

    def _clear_fault_state_locked(self):
        self._faulted = False
        self._fault_reason = None

    def clear_fault(self):
        with self._lock:
            self._clear_fault_state_locked()

    def _set_fault_locked(self, reason, hold_pose=None):
        if not self._faulted:
            self._fault_reason = str(reason)
        self._faulted = True
        if hold_pose is None:
            hold_pose = self._last_safe_target if self._last_safe_target is not None else self._target
        if hold_pose is not None:
            hold_pose = np.asarray(hold_pose, dtype=np.float64)
            self._last_target = hold_pose.copy()
            self._target = hold_pose.copy()
            self._last_safe_target = hold_pose.copy()
            self._interp_count = self._interp_steps

    def trigger_fault(self, reason):
        with self._lock:
            hold_pose = None
            if self._last_measured_pose is not None:
                hold_pose = self._last_measured_pose.copy()
                if self._last_measured_rpy_unwrapped_deg is not None:
                    hold_pose[3:] = self._last_measured_rpy_unwrapped_deg.copy()
            self._set_fault_locked(reason, hold_pose=hold_pose)

    def get_fault_status(self):
        with self._lock:
            return {
                'faulted': bool(self._faulted),
                'reason': self._fault_reason,
            }

    def _init_pose_state_locked(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        self._target = pose.copy()
        self._last_target = pose.copy()
        self._last_safe_target = pose.copy()
        self._last_measured_pose = pose.copy()
        self._last_measured_rpy_unwrapped_deg = pose[3:].copy()
        self._interp_steps = 25
        self._interp_count = self._interp_steps

    def start_servo_mode(self):
        with self._lock:
            if self._running and self._thread is not None and self._thread.is_alive():
                return
            current_pose = np.asarray(self.xarm.get_position()[1], dtype=np.float64)
            self._init_pose_state_locked(current_pose)
            self._running = True
            self._thread = threading.Thread(target=self._control_loop)
            self._thread.start()

    def stop_servo_mode(self):
        thread = None
        with self._lock:
            if not self._running:
                thread = self._thread
            else:
                self._running = False
                thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join()

    def reset(self):
        if not self.first_reset:
            self.close()
        self.xarm.set_mode(0)
        self.xarm.set_state(0)
        joints_7dof = np.append(self.joints_init, 0) if len(self.joints_init) == 6 else self.joints_init
        self.xarm.set_servo_angle(angle=joints_7dof, speed=80, wait=True)
        curr_pos = np.asarray(self.xarm.get_position()[1], dtype=np.float64)
        with self._lock:
            self._clear_fault_state_locked()
            self._init_pose_state_locked(curr_pos)
        self.start_servo_mode()
        self.first_reset = False

    def set_servo_cartesian(self, target):
        target = np.asarray(target, dtype=np.float64)
        command_info = {
            'raw_target': target.copy(),
            'safe_target': target.copy(),
            'raw_delta_pos_mm': np.zeros(3, dtype=np.float64),
            'raw_delta_rot_deg': np.zeros(3, dtype=np.float64),
            'raw_rotation_distance_deg': 0.0,
            'clipped_delta_pos_mm': np.zeros(3, dtype=np.float64),
            'clipped_delta_rot_deg': np.zeros(3, dtype=np.float64),
            'clipped_rotation_distance_deg': 0.0,
            'faulted': False,
            'fault_reason': None,
        }
        with self._lock:
            current_pose = np.asarray(self.xarm.get_position()[1], dtype=np.float64)
            if self._last_safe_target is None:
                self._init_pose_state_locked(current_pose)

            reference_pose = self._last_safe_target.copy()
            raw_delta_pos = target[:3] - reference_pose[:3]
            rotation_delta = self._compute_rotation_delta(reference_pose[3:], target[3:])
            raw_delta_rot = rotation_delta['delta_rotvec_deg']
            raw_rot_norm = rotation_delta['delta_angle_deg']

            command_info['raw_delta_pos_mm'] = raw_delta_pos.copy()
            command_info['raw_delta_rot_deg'] = raw_delta_rot.copy()
            command_info['raw_rotation_distance_deg'] = raw_rot_norm

            raw_pos_norm = float(np.linalg.norm(raw_delta_pos))
            if self.enable_safety_guard:
                if raw_pos_norm > self.fault_cmd_pos_mm:
                    self._set_fault_locked(
                        f'command_position_jump_mm:{raw_pos_norm:.3f}',
                        hold_pose=reference_pose,
                    )

            limited_delta_pos = raw_delta_pos.copy()
            limited_delta_rot = raw_delta_rot.copy()
            if self.enable_safety_guard and not self._faulted:
                limited_delta_pos = self._limit_norm(raw_delta_pos, self.max_pos_step_mm)
                limited_delta_rot = self._limit_norm(raw_delta_rot, self.max_rot_step_deg)

            safe_target = reference_pose.copy()
            safe_target[:3] += limited_delta_pos
            safe_rot = R.from_rotvec(np.deg2rad(limited_delta_rot)) * rotation_delta['reference_rot']
            safe_target[3:] = safe_rot.as_euler('xyz', degrees=True)

            command_info['safe_target'] = safe_target.copy()
            command_info['clipped_delta_pos_mm'] = raw_delta_pos - limited_delta_pos
            command_info['clipped_delta_rot_deg'] = raw_delta_rot - limited_delta_rot
            command_info['clipped_rotation_distance_deg'] = max(
                0.0, raw_rot_norm - float(np.linalg.norm(limited_delta_rot))
            )
            command_info['faulted'] = bool(self._faulted)
            command_info['fault_reason'] = self._fault_reason

            if not self._faulted:
                self._last_target = current_pose.copy()
                self._target = safe_target.copy()
                self._last_safe_target = safe_target.copy()
            else:
                self._last_target = current_pose.copy()
                self._target = reference_pose.copy()
                self._last_safe_target = reference_pose.copy()
            self._interp_count = 0
        return command_info
    
    def _control_loop(self):
        self.xarm.set_mode(1)
        self.xarm.set_state(state=0)
        control_freq = 500
        dt = 1.0 / control_freq
        while self._running:
            with self._lock:
                if self._interp_count < self._interp_steps:
                    alpha = (self._interp_count + 1) / self._interp_steps
                    interp = pose_interp(self._last_target, self._target, alpha)
                    self._interp_count += 1
                else:
                    interp = self._target
            self.xarm.set_servo_cartesian(interp)
            time.sleep(dt)

    def get_position(self):
        code, pose = self.xarm.get_position()
        if code != 0:
            raise RuntimeError("Abnormal code returned by XArm!")
        pose = np.array(pose)
        pose[:3] /= 1000  # mm -> m
        pose[3:] = pose[3:] / 180 * np.pi
        return pose
    
    def get_joint(self):
        code, angle = self.xarm.get_servo_angle()
        return np.array(angle[:6]) / 180 * np.pi
    
    def set_joint(self, angles, speed=32, wait=False):
        angles_deg = np.array(angles) * 180 / np.pi
        if len(angles_deg) == 6:
            angles_deg = np.append(angles_deg, 0)
        self.xarm.set_servo_angle(angle=angles_deg, speed=speed, wait=wait)

    def close(self):
        self.stop_servo_mode()
