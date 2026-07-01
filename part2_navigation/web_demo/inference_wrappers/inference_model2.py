#!/usr/bin/env python
"""
Wrapper script for Model 2 inference
Runs in flowdit_v2_py310 conda environment
"""

import sys
import json
import numpy as np
from pathlib import Path
import os
import warnings
from io import StringIO
import cv2

# SUPPRESS ALL LOGGING AND WARNINGS IMMEDIATELY
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress torch warnings
import logging
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "flow_constrained_v2"))

def run_inference(video_path, checkpoint_path, current_obs_mode='middle', device='cuda'):
    """Run Model 2 inference - SILENT MODE with fallback"""
    # Suppress stdout/stderr during model loading
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        import torch
        
        # Load video using opencv (avoiding broken inference.py)
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Normalize to [0, 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        
        if not frames:
            raise ValueError("No frames loaded from video")
        
        goal_video = np.array(frames)
        original_shape = goal_video.shape
        
        # Resize to dimensions compatible with the model (multiples of 14)
        # The model uses patch size 14 for vision transformer
        h, w = goal_video.shape[1], goal_video.shape[2]
        
        # Round down to nearest multiple of 14
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        
        if new_h == 0 or new_w == 0:
            # If too small, use minimum compatible size
            new_h = 14
            new_w = 14
        
        # Resize frames if needed
        if new_h != h or new_w != w:
            resized_frames = []
            for frame in goal_video:
                resized_frame = cv2.resize(frame, (new_w, new_h))
                resized_frames.append(resized_frame)
            goal_video = np.array(resized_frames)
        
        # Select current observation
        if current_obs_mode == 'first':
            current_obs = goal_video[0]
        elif current_obs_mode == 'last':
            current_obs = goal_video[-1]
        else:  # middle (default)
            current_obs = goal_video[len(goal_video) // 2]
        
        # Try to use Model 2 (this may fail due to inference.py syntax error)
        try:
            from models.flowdit_production import create_flowdit_production
            
            # Prepare video for model (convert to tensor format)
            goal_video_tensor = torch.from_numpy(goal_video).permute(0, 3, 1, 2).float()
            current_obs_tensor = torch.from_numpy(current_obs).permute(2, 0, 1).float().unsqueeze(0)
            
            # Load and run model
            model = create_flowdit_production(device=device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Run inference
            with torch.no_grad():
                goal_video_tensor = goal_video_tensor.to(device)
                current_obs_tensor = current_obs_tensor.to(device)
                actions = model(goal_video_tensor.unsqueeze(0), current_obs_tensor)
            
            # Convert to numpy
            if isinstance(actions, torch.Tensor):
                actions = actions.squeeze().cpu().numpy()
            
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            
        except (ImportError, SyntaxError, Exception) as e:
            # Fallback: generate dummy predictions based on video length
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            raise Exception(f"Model 2 loading failed: {str(e)}. Please fix flow_constrained_v2/inference.py line 275 indentation")
        
        # Integrate trajectory
        dt = 1.0 / 16.0
        T = len(actions)
        traj = np.zeros((T + 1, 3))
        
        for i in range(T):
            if i < len(actions):
                vx, vy, yaw_rate = actions[i][:3] if len(actions[i]) >= 3 else (0, 0, 0)
            else:
                vx, vy, yaw_rate = 0, 0, 0
            
            x, y, theta = traj[i]
            theta_new = theta + yaw_rate * dt
            vx_world = vx * np.cos(theta) - vy * np.sin(theta)
            vy_world = vx * np.sin(theta) + vy * np.cos(theta)
            x_new = x + vx_world * dt
            y_new = y + vy_world * dt
            traj[i + 1] = [x_new, y_new, theta_new]
        
        trajectory = traj[1:]
        
        return {
            'predictions': actions.tolist() if hasattr(actions, 'tolist') else actions,
            'trajectory': trajectory.tolist(),
            'num_frames': len(actions),
            'fps': 16.0,
            'goal_video': goal_video.tolist(),
            'current_obs': current_obs.tolist(),
            'original_shape': original_shape,
            'resized_shape': goal_video.shape
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == '__main__':
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    try:
        result = run_inference(**input_data)
        # ONLY OUTPUT JSON - NO OTHER MESSAGES
        sys.stdout.write(json.dumps({'success': True, **result}))
        sys.stdout.flush()
    except Exception as e:
        import traceback
        sys.stdout.write(json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }))
        sys.stdout.flush()

