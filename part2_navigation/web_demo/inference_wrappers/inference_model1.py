#!/usr/bin/env python
"""
Wrapper script for Model 1 inference
Runs in flow_training conda environment
"""

import sys
import json
import numpy as np
from pathlib import Path
import os
import warnings
from io import StringIO

# SUPPRESS ALL LOGGING AND WARNINGS IMMEDIATELY
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress torch warnings
import logging
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('torchvision').setLevel(logging.ERROR)

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "flow_constrained"))

def run_inference(video_path, checkpoint_path, config_path, embodiment='wheeled', 
                 trajectory_mode=True, fps=16.0, device='cuda'):
    """Run Model 1 inference - SILENT MODE"""
    # Suppress stdout/stderr during model loading
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        import torch
        import yaml
        from models import FusionNetwork, OpticalFlowExtractor, VDMFeatureExtractor, VisionEncoder
        from models.optical_flow import estimate_ego_motion
        from data.video_loader import load_video
        
        # Load model
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model = FusionNetwork(
            ego_motion_dim=config['model']['ego_motion_dim'],
            vdm_feature_dim=config['model']['vdm_feature_dim'],
            vision_feature_dim=config['model']['vision_feature_dim'],
            embodiment_dim=config['model']['embodiment_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_embodiments=config['model']['num_embodiments'],
            action_dim=config['model']['action_dim'],
            dropout=config['model']['dropout']
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        if trajectory_mode:
            # Full sequence prediction
            video = load_video(video_path)
            T = video.shape[0]
            video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
            video_tensor = video_tensor.unsqueeze(0).to(device)
            
            flow_extractor = OpticalFlowExtractor(device=device)
            vdm_extractor = VDMFeatureExtractor(device=device)
            vision_encoder = VisionEncoder(device=device)
            
            embodiment_map = {'wheeled': 0, 'legged': 1, 'aerial': 2, 'humanoid': 3}
            embodiment_idx = torch.tensor([embodiment_map[embodiment]]).to(device)
            
            window_size = 8
            step_size = max(1, window_size // 2)
            predictions = []
            
            with torch.no_grad():
                for start_idx in range(0, T, step_size):
                    end_idx = min(start_idx + window_size, T)
                    window = video_tensor[:, start_idx:end_idx]
                    
                    if window.shape[1] < window_size:
                        padding = torch.zeros(1, window_size - window.shape[1], 
                                            window.shape[2], window.shape[3], window.shape[4]).to(device)
                        window = torch.cat([window, padding], dim=1)
                    
                    optical_flow = flow_extractor.extract_from_video(window)
                    ego_motion = estimate_ego_motion(optical_flow[0, 0])
                    vdm_features = vdm_extractor.extract_from_video(window)
                    vision_features = vision_encoder.extract_from_video(window)
                    
                    pred = model(
                        ego_motion.unsqueeze(0),
                        vdm_features[0, 0].unsqueeze(0),
                        vision_features[0, 0].unsqueeze(0),
                        embodiment_idx
                    )
                    
                    for _ in range(end_idx - start_idx):
                        predictions.append(pred.squeeze(0).cpu().numpy())
            
            while len(predictions) < T:
                predictions.append(predictions[-1] if predictions else np.zeros(3))
            
            predictions = np.array(predictions[:T])
            
            # Integrate trajectory
            dt = 1.0 / fps
            initial_pose = np.array([0.0, 0.0, 0.0])
            trajectory = np.zeros((T, 3))
            trajectory[0] = initial_pose
            
            for t in range(1, T):
                x_prev, y_prev, theta_prev = trajectory[t-1]
                vx, vy, yaw_rate = predictions[t-1]
                
                theta_new = theta_prev + yaw_rate * dt
                vx_world = vx * np.cos(theta_prev) - vy * np.sin(theta_prev)
                vy_world = vx * np.sin(theta_prev) + vy * np.cos(theta_prev)
                
                x_new = x_prev + vx_world * dt
                y_new = y_prev + vy_world * dt
                
                trajectory[t] = [x_new, y_new, theta_new]
            
            return {
                'predictions': predictions.tolist(),
                'trajectory': trajectory.tolist(),
                'num_frames': T,
                'fps': fps
            }
        else:
            # Single frame - simplified version
            from inference_single_video import extract_features_from_video, predict_actions
            
            features = extract_features_from_video(video_path, device)
            prediction = predict_actions(model, features, device, embodiment)
            
            return {
                'predictions': [prediction.tolist()],
                'trajectory': None,
                'num_frames': 1,
                'fps': fps
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

