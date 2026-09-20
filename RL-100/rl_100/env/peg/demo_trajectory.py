#!/usr/bin/env python3
"""
演示轨迹录制和重放功能的示例脚本

使用方法:
1. 录制轨迹: python demo_trajectory.py --mode record
2. 重放轨迹: python demo_trajectory.py --mode replay [--file trajectory_file.json]

录制模式控制:
- 'r': 开始/停止录制
- 'p': 开始/停止重放  
- 's': 保存轨迹
- 'l': 加载轨迹
- 'c': 清除轨迹
- Ctrl+C: 退出
"""

import argparse
import sys
import os
import time

# 添加3D-Diffusion-Policy目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
diffusion_policy_root = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.insert(0, diffusion_policy_root)

# 直接导入当前目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from juicing_env import JuicingEnv

def record_mode():
    """录制模式 - 手动拖拽录制轨迹"""
    print("=" * 50)
    print("轨迹录制模式")
    print("=" * 50)
    print("请确保机器人已正确连接并处于安全状态")
    input("按回车键继续...")
    
    # 初始化环境
    env = JuicingEnv(
        robot_ip='192.168.1.202',
        dt=1/30,
        use_camera=True,
        num_point_cloud=1024
    )
    
    try:
        # 重置机器人到初始位置（手动模式）
        print("重置机器人到初始位置...")
        env.reset_for_manual_mode()
        
        print("开始手动录制模式...")
        print("现在可以手动拖拽机器人进行轨迹录制")
        
        # 启动手动录制模式
        env.start_manual_recording()
        
    except KeyboardInterrupt:
        print("\n录制模式被用户中断")
    except Exception as e:
        print(f"录制过程中发生错误: {e}")
    finally:
        print("关闭环境...")
        env.close()

def replay_mode(trajectory_file=None):
    """重放模式 - 自动重放保存的轨迹"""
    print("=" * 50)
    print("轨迹重放模式")
    print("=" * 50)
    
    if trajectory_file:
        print(f"指定重放轨迹文件: {trajectory_file}")
    else:
        print("将重放最新保存的轨迹文件")
    
    print("请确保机器人已正确连接并处于安全状态")
    print("请确保机器人周围没有障碍物")
    input("按回车键继续...")
    
    # 初始化环境
    env = JuicingEnv(
        robot_ip='192.168.1.202',
        dt=1/30,
        use_camera=True,
        num_point_cloud=1024
    )
    
    try:
        # 重置机器人到初始位置
        print("重置机器人到初始位置...")
        env.reset()
        
        print("开始重放轨迹...")
        
        # 重放轨迹
        env.replay_saved_trajectory(trajectory_file)
        
        print("轨迹重放完成!")
        
    except KeyboardInterrupt:
        print("\n重放模式被用户中断")
    except Exception as e:
        print(f"重放过程中发生错误: {e}")
    finally:
        print("关闭环境...")
        env.close()

def list_trajectories():
    """列出所有保存的轨迹文件"""
    from trajectory_recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder()
    files = recorder.list_saved_trajectories()
    
    if not files:
        print("没有找到保存的轨迹文件")
        return
        
    print("保存的轨迹文件:")
    for i, filename in enumerate(files, 1):
        print(f"  {i}. {filename}")

def main():
    parser = argparse.ArgumentParser(description="轨迹录制和重放演示")
    parser.add_argument(
        "--mode", 
        choices=["record", "replay", "list"], 
        required=True,
        help="运行模式: record(录制), replay(重放), list(列出文件)"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="重放模式下指定轨迹文件名"
    )
    
    args = parser.parse_args()
    
    if args.mode == "record":
        record_mode()
    elif args.mode == "replay":
        replay_mode(args.file)
    elif args.mode == "list":
        list_trajectories()

if __name__ == "__main__":
    main()
