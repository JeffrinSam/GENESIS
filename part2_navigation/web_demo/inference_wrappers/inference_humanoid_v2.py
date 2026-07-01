#!/usr/bin/env python
"""
Wrapper script for Humanoid V2 inference (FlowDiT - Goal-Conditioned)
Uses the FlowDiT production model trained on humanoid data
"""

import sys
import json
import numpy as np
from pathlib import Path
import os
import warnings
from io import StringIO
import cv2

# SUPPRESS ALL LOGGING AND WARNINGS
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "flow_constrained_v2"))


def run_inference(video_path, checkpoint_path, current_obs_mode='middle', device='cuda'):
    """Run Humanoid V2 inference - FlowDiT goal-conditioned"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    try:
        import torch

        # Load video
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()

        if not frames:
            raise ValueError("No frames loaded from video")

        goal_video = np.array(frames)
        original_shape = goal_video.shape

        # Resize to 224x224 (model expects this)
        h, w = goal_video.shape[1], goal_video.shape[2]
        new_h, new_w = 224, 224

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
        else:
            current_obs = goal_video[len(goal_video) // 2]

        # Load model
        from models.flowdit_production import FlowDiTProduction, FlowDiTConfig

        config = FlowDiTConfig(
            action_dim=3,
            action_horizon=8,
            goal_frames=16,
            use_language=False,
            device=device
        )

        model = FlowDiTProduction(config)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Prepare tensors
        goal_video_tensor = torch.from_numpy(goal_video).permute(0, 3, 1, 2).float()
        current_obs_tensor = torch.from_numpy(current_obs).permute(2, 0, 1).float().unsqueeze(0)

        # Sample goal frames (last 16 or uniform)
        T = len(goal_video_tensor)
        if T >= 16:
            goal_indices = np.linspace(max(0, T - 16), T - 1, 16, dtype=int)
        else:
            goal_indices = np.linspace(0, T - 1, min(T, 16), dtype=int)

        goal_frames = goal_video_tensor[goal_indices]  # [16, 3, H, W]

        # Pad if needed
        if len(goal_frames) < 16:
            padding = 16 - len(goal_frames)
            last_frame = goal_frames[-1:].repeat(padding, 1, 1, 1)
            goal_frames = torch.cat([goal_frames, last_frame], dim=0)

        # Run inference
        with torch.no_grad():
            goal_frames = goal_frames.unsqueeze(0).to(device)  # [1, 16, 3, H, W]
            current_obs_tensor = current_obs_tensor.to(device)  # [1, 3, H, W]

            # Use model's inference method (DDIM sampling)
            if hasattr(model, 'sample_actions'):
                actions = model.sample_actions(goal_frames, current_obs_tensor)
            else:
                # Forward pass for training-style prediction
                actions_gt_dummy = torch.zeros(1, 8, 3).to(device)
                predicted_noise, _ = model(goal_frames, current_obs_tensor, actions_gt_dummy)
                # Use the predicted noise as approximate actions (rough)
                actions = predicted_noise

            if isinstance(actions, torch.Tensor):
                actions = actions.squeeze().cpu().numpy()

            if actions.ndim == 1:
                actions = actions.reshape(1, -1)

            # Ensure 3 columns
            if actions.shape[-1] > 3:
                actions = actions[:, :3]

        # Integrate trajectory
        dt = 1.0 / 16.0
        num_actions = len(actions)
        traj = np.zeros((num_actions + 1, 3))

        for i in range(num_actions):
            vx, vy, yaw_rate = actions[i][:3]
            x, y, theta = traj[i]
            theta_new = theta + yaw_rate * dt
            vx_world = vx * np.cos(theta) - vy * np.sin(theta)
            vy_world = vx * np.sin(theta) + vy * np.cos(theta)
            traj[i + 1] = [x + vx_world * dt, y + vy_world * dt, theta_new]

        trajectory = traj[1:]

        return {
            'predictions': actions.tolist() if hasattr(actions, 'tolist') else actions,
            'trajectory': trajectory.tolist(),
            'num_frames': num_actions,
            'fps': 16.0,
            'goal_video': goal_video[goal_indices.tolist()].tolist() if len(goal_video) > 0 else [],
            'current_obs': current_obs.tolist(),
            'original_shape': list(original_shape),
            'resized_shape': list(goal_video.shape)
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == '__main__':
    input_data = json.loads(sys.stdin.read())
    try:
        result = run_inference(**input_data)
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
