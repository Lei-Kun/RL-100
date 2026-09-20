# 轨迹录制和重放功能使用说明

## 功能概述

为juicing环境添加了手动拖拽录制和轨迹重放功能，包含：

1. **TrajectoryRecorder** - 独立的轨迹录制器模块
2. **JuicingEnv集成** - 在环境中添加录制和重放方法  
3. **演示脚本** - 便于使用的命令行工具

## 文件说明

- `trajectory_recorder.py` - 轨迹录制器核心模块
- `juicing_env.py` - 已集成录制功能的环境（新增方法）
- `demo_trajectory.py` - 演示使用脚本
- `trajectories/` - 轨迹文件保存目录

## 使用方法

### 1. 命令行使用

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dp3

# 进入juicing目录
cd 3D-Diffusion-Policy/diffusion_policy_3d/env/juicing/

# 录制轨迹
python demo_trajectory.py --mode record

# 重放轨迹（最新文件）
python demo_trajectory.py --mode replay

# 重放指定轨迹文件
python demo_trajectory.py --mode replay --file trajectory_20241201_143022.json

# 列出所有轨迹文件
python demo_trajectory.py --mode list
```

### 2. 代码中直接使用

```python
from juicing_env import JuicingEnv

# 创建环境
env = JuicingEnv(robot_ip='192.168.1.202')
env.reset()

# 方法1: 手动录制模式（带键盘控制）
env.start_manual_recording()

# 方法2: 重放轨迹
env.replay_saved_trajectory("trajectory_file.json")
```

### 3. 录制模式控制键

在手动录制模式(`start_manual_recording()`)中：

- `r` - 开始/停止录制
- `p` - 开始/停止重放
- `s` - 保存当前轨迹  
- `l` - 加载最新轨迹文件
- `c` - 清除当前轨迹
- `Ctrl+C` - 退出录制模式

## 轨迹文件格式

轨迹保存为JSON格式，包含：

```json
{
  "metadata": {
    "recorded_at": "2025-09-02T12:22:43.825548",
    "num_points": 150,
    "duration": 5.0
  },
  "trajectory": [
    {
      "timestamp": 1756786963.8255143,
      "joint_positions": [-6.948, -167.076, ...],  // 6个关节角度(度)
      "tcp_position": [0.450, 0.649, ...],         // TCP位置和姿态
      "gripper_state": 0.692                       // 抓手状态(0-1)
    },
    ...
  ]
}
```

## 工作流程

### 录制演示轨迹

1. 运行 `python demo_trajectory.py --mode record`
2. 按 `r` 开始录制
3. 手动拖拽机器人执行所需动作
4. 按 `r` 停止录制  
5. 按 `s` 保存轨迹
6. `Ctrl+C` 退出

### 重放演示轨迹

1. 运行 `python demo_trajectory.py --mode replay`
2. 机器人将自动执行录制的轨迹
3. 重放完成后自动结束

## 技术特点

- **30Hz控制频率** - 确保平滑的轨迹录制和重放
- **完整状态记录** - 包含关节角度、TCP位置和抓手状态
- **时间同步重放** - 按照录制时的时间间隔重放
- **安全设计** - 不影响原有环境执行逻辑
- **文件管理** - 自动时间戳命名和加载最新文件

## 注意事项

1. 确保机器人已正确连接并处于安全状态
2. 录制前将机器人移动到合适的初始位置
3. 重放前确认周围没有障碍物
4. 轨迹文件保存在 `./trajectories/` 目录下
5. 如需要，可手动编辑轨迹文件调整参数

## 故障排除

如果遇到导入错误，确保：
- 已激活dp3环境
- 在正确的目录下运行脚本
- 所有依赖模块已安装

如果机器人连接失败，检查：
- 机器人IP地址是否正确
- 网络连接是否正常
- 机器人是否处于正确的控制模式