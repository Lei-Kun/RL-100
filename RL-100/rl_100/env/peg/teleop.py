import os
import sys
import time
import math
import numpy as np
# from pynput import keyboard
from avp_stream import VisionProStreamer
from scipy.spatial.transform import Rotation as R

from xarm_wrapper import XArmWrapper
from robotiq_wrapper import RobotiqWrapper
from realsense import RealSense


AVP_IP = "192.168.31.155"

def SE3_inv(SE3):
    SE3_inv = np.eye(4, dtype=np.float32)
    SE3_inv[:3, :3] = SE3[:3, :3].T
    SE3_inv[:3, 3] = -SE3[:3, :3].T @ SE3[:3, 3]
    return SE3_inv

import time

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


class AVPTeleop():
    def __init__(self):
        self.vps = VisionProStreamer(ip=AVP_IP, record=True)
        self.is_rotate = False
    
    def set_init_pose(self, init_pose):
        self.robot_init = np.eye(4)
        self.robot_init[:3, 3] = init_pose[:3]
        self.robot_init[:3, :3] = R.from_euler('xyz', init_pose[3:6], degrees=False).as_matrix()
        self.hand_init = self.get_wrist_data()
        
    def print_frame(self, SE3):
        print("Translation:", SE3[:3, 3])
        print("Rotation (Euler xyz):", R.from_matrix(SE3[:3, :3]).as_euler('xyz', degrees=True))
        print("Joint angles:", R.from_matrix(SE3[:3, :3]).as_rotvec() * 180 / np.pi)
        print("*" * 20)
        
    def get_avp_data(self):
        avp_dict = {}
        avp_dict['arm'] = self.get_arm_pose()
        # 不再需要二进制的gripper状态，直接在主循环中处理连续值
        avp_dict['rotate'] = self.get_rotate()
        return avp_dict
    
    def is_terminate(self):
        right_pinch = self.get_ring_pinch_data() < 0.03
        return right_pinch
    
    def get_wrist_data(self):
        r = self.vps.latest
        return r['right_wrist'][0]
    
    def get_pinch_data(self):
        r = self.vps.latest
        return r['right_pinch_distance']

    def get_left_pinch_data(self):
        r = self.vps.latest
        return r['left_pinch_distance']

    def get_rotate(self):
        if self.get_left_pinch_data() < 0.03 and not self.is_rotate:
            self.is_rotate = True
            return True
        else:
            return False
    
    def get_ring_pinch_data(self):
        r = self.vps.latest
        fingers = r['right_fingers']
        pinch = np.linalg.norm(fingers[4][:3, 3] - fingers[24][:3, 3])
        return pinch
        
    def get_arm_pose(self):
        robot_init = self.robot_init
        hand_init = self.hand_init

        X_VR2Robot = np.array([
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1],
        ])
        
        hand_pose = self.get_wrist_data()
        hand_transform_VR = np.eye(4, dtype=np.float32)
        hand_transform_VR[:3, :3] = hand_pose[:3, :3] @ hand_init[:3, :3].T
        hand_transform_VR[:3, 3] = hand_pose[:3, 3] - hand_init[:3, 3]
        hand_transform_robot = X_VR2Robot @ hand_transform_VR @ SE3_inv(X_VR2Robot)
        robot_pose = np.eye(4, dtype=np.float32)
        robot_pose[:3, :3] = hand_transform_robot[:3, :3] @ robot_init[:3, :3]
        robot_pose[:3, 3] = robot_init[:3, 3] + hand_transform_robot[:3, 3]
        
        if self.is_rotate:
            rot_180 = R.from_euler('xyz', [0, 0, 120], degrees=True).as_matrix()
            robot_pose[:3, :3] = robot_pose[:3, :3] @ rot_180
        
        self.print_frame(robot_pose)
        return robot_pose


if __name__ == '__main__':
    camera = RealSense()
    camera.start()
    print("********** Camera is initialized **********")

    # xarm = XArmWrapper(joints_init=[3.3, -44.4, -70, 1.3, 113.5, 4.3])
    xarm = XArmWrapper(joints_init=[-22.1, -21, -36.3, -0.3, 58.3, -22.7])
    xarm_gripper = RobotiqWrapper(robot='xarm')
    print("********** Robots are initialized **********")

    avp_teleop = AVPTeleop()
    print("********** AVP is initialized **********")
    
    demo_idx = 0
    demo_dir = f'data/demo_{demo_idx:03d}.npy'
    while os.path.exists(demo_dir):
        demo_idx += 1
        demo_dir = f'data/demo_{demo_idx:03d}.npy'

    time.sleep(3)
    xarm_init_pose = xarm.get_position()
    avp_teleop.set_init_pose(xarm_init_pose)
    demo_data = []
    
    dt = 1 / 20
    t_start = time.monotonic()
    frame_idx = 0
    # while True:
    #     avp_dict = avp_teleop.get_avp_data(enable_left=False)
    while True:
        s = time.time()
        t_cycle_end = t_start + (frame_idx + 1) * dt
        t_command_target = t_cycle_end + dt

        xarm_q = xarm.get_joint()
        xarm_pose = xarm.get_position()
        xarm_gripper_state = xarm_gripper.get_state()
        
        q_pos = np.concatenate([xarm_q, [xarm_gripper_state]])
        print(f"XArm Joint Angles: {xarm_q}, Gripper State: {xarm_gripper_state}")
        ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state]])
        
        frame = camera.get_frame()
        
        avp_dict = avp_teleop.get_avp_data()

        # XArm
        xarm_target = np.zeros(6, dtype=np.float32)
        xarm_target[:3] = avp_dict['arm'][:3, 3] * 1000
        xarm_target[3:] = R.from_matrix(avp_dict['arm'][:3, :3]).as_euler('xyz', degrees=True)
        xarm.set_servo_cartesian(xarm_target)
        # 连续夹爪控制：基于拇指食指距离
        pinch_distance = avp_teleop.get_pinch_data()
        
        # 将距离映射到夹爪开合度 (0-255, 0为完全张开, 255为完全闭合)
        # 假设pinch_distance范围是0.0-0.1m，需要根据实际情况调整
        min_distance = 0.0  # 最小距离对应夹爪闭合
        max_distance = 0.08  # 最大距离对应夹爪张开
        
        # 先归一化到0-1，然后反向映射到0-255
        normalized_distance = np.clip((pinch_distance - min_distance) / (max_distance - min_distance), 0.0, 1.0)
        # 反向映射：距离小(手指靠近)→夹爪闭合(255)，距离大(手指分开)→夹爪张开(0)
        gripper_target = int((1.0 - normalized_distance) * 255)
        
        # 使用连续控制
        xarm_gripper.set_position(gripper_target)
        # if is_rotate: rotate gripper by 180 degrees
        xarm_target[:3] /= 1000
        xarm_target[3:] = xarm_target[3:] / 180 * np.pi
        
        # 将归一化的夹爪值（0-1）加入action数组，这里使用反向映射后的值
        gripper_action = normalized_distance
        action = np.concatenate([xarm_target, [gripper_action]])
        demo_data.append({
            'demo_frame_idx': frame_idx,
            'depth': frame['depth'],
            'depth_scale': frame['depth_scale'],
            'qpos': q_pos,
            'eepose': ee_pose,
            'action': action
        })

        if avp_teleop.is_terminate():
            np.save(demo_dir, demo_data, allow_pickle=True)
            print("#" * 64)
            print(f"Success! Demo saved to {demo_dir}.")
            print(f"Demo length: {len(demo_data)}")
            print("#" * 64)
            break
        else:
            frame_idx += 1

        # while time.time() - iter_start_time < dt:
        #     time.sleep(dt / 20)
        precise_wait(t_cycle_end)
        # print(f'Iter: {frame_idx}, Frequency: {1 / (time.time() - iter_start_time):.3f} Hz')
        print(f'Iter: {frame_idx}, Frequency: {1 / (time.time() - s):.3f} Hz')

    xarm_gripper.open()
