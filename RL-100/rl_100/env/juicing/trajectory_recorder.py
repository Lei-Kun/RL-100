import os
import time
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from pynput import keyboard
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class TrajectoryPoint:
    """Single trajectory point containing robot state and timestamp"""
    timestamp: float
    joint_positions: List[float]  # 6 joint angles in degrees
    tcp_position: List[float]     # [x, y, z, roll, pitch, yaw] in meters/radians
    gripper_state: float         # Gripper position 0-1
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

class TrajectoryRecorder:
    """
    Trajectory recorder for manual demonstration recording and playback
    
    Controls:
    - 'r': Start/stop recording
    - 'p': Start/stop playback  
    - 's': Save current trajectory
    - 'l': Load saved trajectory
    - 'c': Clear current trajectory
    """
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        base_dir = Path(__file__).resolve().parent
        if save_dir is None:
            resolved_dir = base_dir / "trajectories"
        else:
            resolved_dir = Path(save_dir).expanduser()
            if not resolved_dir.is_absolute():
                resolved_dir = (base_dir / resolved_dir).resolve()

        self.save_dir = resolved_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.is_playing = False
        self.current_trajectory: List[TrajectoryPoint] = []
        self.saved_trajectory: List[TrajectoryPoint] = []
        
        # Playback state
        self.playback_index = 0
        self.playback_start_time = 0
        
        # Keyboard control
        self.setup_keyboard_listener()
        
        print("Trajectory Recorder initialized")
        print("Controls:")
        print("  'r' - Start/stop recording")
        print("  'p' - Start/stop playback")
        print("  's' - Save trajectory")
        print("  'l' - Load trajectory") 
        print("  'c' - Clear trajectory")
    
    def setup_keyboard_listener(self):
        """Setup keyboard listener for recording controls"""
        def on_press(key):
            try:
                if key.char == 'r':
                    self.toggle_recording()
                elif key.char == 'p':
                    self.toggle_playback()
                elif key.char == 's':
                    self.save_trajectory()
                elif key.char == 'l':
                    self.load_trajectory()
                elif key.char == 'c':
                    self.clear_trajectory()
            except AttributeError:
                pass
        
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.is_playing:
            print("Cannot record while playing back trajectory")
            return
            
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            self.current_trajectory = []
            print(f"🔴 Recording started - {len(self.current_trajectory)} points")
        else:
            print(f"⏹️ Recording stopped - {len(self.current_trajectory)} points recorded")
    
    def toggle_playback(self):
        """Toggle playback state"""
        if self.is_recording:
            print("Cannot playback while recording")
            return
            
        if not self.saved_trajectory:
            print("No trajectory loaded for playback")
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.playback_index = 0
            self.playback_start_time = time.time()
            print(f"▶️ Playback started - {len(self.saved_trajectory)} points")
        else:
            print("⏸️ Playback stopped")
    
    def record_point(self, joint_positions: np.ndarray, tcp_position: np.ndarray, gripper_state: float):
        """Record a single trajectory point"""
        if not self.is_recording:
            return
            
        point = TrajectoryPoint(
            timestamp=time.time(),
            joint_positions=joint_positions.tolist(),
            tcp_position=tcp_position.tolist(),
            gripper_state=gripper_state
        )
        
        self.current_trajectory.append(point)
        
        if len(self.current_trajectory) % 30 == 0:  # Print every 30 points (~1 second at 30Hz)
            print(f"🔴 Recording: {len(self.current_trajectory)} points")
    
    def get_playback_target(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Get target position for current playback time"""
        if not self.is_playing or not self.saved_trajectory:
            return None
            
        current_time = time.time()
        elapsed_time = current_time - self.playback_start_time
        
        # Find the trajectory point for current time
        target_index = 0
        if len(self.saved_trajectory) > 1:
            # Find closest timestamp
            trajectory_start_time = self.saved_trajectory[0].timestamp
            target_time = trajectory_start_time + elapsed_time
            
            for i, point in enumerate(self.saved_trajectory):
                if point.timestamp <= target_time:
                    target_index = i
                else:
                    break
        
        # Check if playback is complete
        if target_index >= len(self.saved_trajectory) - 1:
            self.is_playing = False
            print("✅ Playback completed")
            return None
            
        target_point = self.saved_trajectory[target_index]
        
        # Update playback index for progress tracking
        if target_index != self.playback_index:
            self.playback_index = target_index
            progress = (target_index / len(self.saved_trajectory)) * 100
            print(f"▶️ Playback: {target_index}/{len(self.saved_trajectory)} ({progress:.1f}%)")
        
        return (
            np.array(target_point.joint_positions),
            np.array(target_point.tcp_position), 
            target_point.gripper_state
        )
    
    def save_trajectory(self, filename: Optional[str] = None):
        """Save current trajectory to file"""
        if not self.current_trajectory:
            print("No trajectory to save")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"
            
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert trajectory to serializable format
        trajectory_data = {
            "metadata": {
                "recorded_at": datetime.now().isoformat(),
                "num_points": len(self.current_trajectory),
                "duration": self.current_trajectory[-1].timestamp - self.current_trajectory[0].timestamp if len(self.current_trajectory) > 1 else 0
            },
            "trajectory": [point.to_dict() for point in self.current_trajectory]
        }
        
        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
            
        print(f"💾 Trajectory saved: {filepath} ({len(self.current_trajectory)} points)")
        
        # Also save as the currently loaded trajectory
        self.saved_trajectory = self.current_trajectory.copy()
    
    def load_trajectory(self, filename: Optional[str] = None):
        """Load trajectory from file"""
        if filename is None:
            # Load most recent trajectory file
            trajectory_files = [f for f in os.listdir(self.save_dir) if f.endswith('.json')]
            if not trajectory_files:
                print("No trajectory files found")
                return
            trajectory_files.sort()
            filename = trajectory_files[-1]
            
        filepath = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Trajectory file not found: {filepath}")
            return
            
        try:
            with open(filepath, 'r') as f:
                trajectory_data = json.load(f)
                
            self.saved_trajectory = [
                TrajectoryPoint.from_dict(point_data) 
                for point_data in trajectory_data["trajectory"]
            ]
            
            metadata = trajectory_data.get("metadata", {})
            print(f"📁 Trajectory loaded: {filename}")
            print(f"   Points: {len(self.saved_trajectory)}")
            print(f"   Duration: {metadata.get('duration', 0):.2f}s")
            print(f"   Recorded: {metadata.get('recorded_at', 'Unknown')}")
            
        except Exception as e:
            print(f"Error loading trajectory: {e}")
    
    def clear_trajectory(self):
        """Clear current trajectory"""
        self.current_trajectory = []
        self.saved_trajectory = []
        print("🗑️ Trajectory cleared")
    
    def get_trajectory_info(self) -> Dict:
        """Get information about current trajectories"""
        return {
            "is_recording": self.is_recording,
            "is_playing": self.is_playing,
            "current_points": len(self.current_trajectory),
            "loaded_points": len(self.saved_trajectory),
            "playback_progress": self.playback_index / len(self.saved_trajectory) if self.saved_trajectory else 0
        }
    
    def list_saved_trajectories(self) -> List[str]:
        """List all saved trajectory files"""
        trajectory_files = [f for f in os.listdir(self.save_dir) if f.endswith('.json')]
        trajectory_files.sort()
        return trajectory_files
    
    def stop(self):
        """Stop the recorder and clean up"""
        self.is_recording = False
        self.is_playing = False
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        print("Trajectory recorder stopped")
