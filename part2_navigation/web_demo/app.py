"""
Video-to-Navigation Web Application
Flask backend for video inference with visualization

Author: Jeffrin Sam
Institution: Skoltech
Date: January 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import base64
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Project root (parent of web_app/)
PROJECT_ROOT = Path(__file__).parent.parent
# NOTE: Do NOT add flow_constrained or flow_constrained_v2 to sys.path here.
# flow_constrained/models/__init__.py imports diffusers (VDMFeatureExtractor),
# which is not installed in the flowdit_v2_py310 environment.
# All V1 inference runs via subprocess in flow_training env.
# V2 dual-mode inference uses importlib with explicit paths.

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = PROJECT_ROOT / 'web_app' / 'uploads'
app.config['OUTPUT_FOLDER'] = PROJECT_ROOT / 'web_app' / 'outputs'

# Create directories
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Global model cache (not used anymore - models run in separate processes)
_model_cache = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_json_from_output(output):
    """Extract JSON object from mixed text output"""
    import re
    # Try to find JSON object in output
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # If no JSON found, try to parse entire output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {'success': False, 'error': 'Could not extract JSON from output'}


def run_inference_model1(video_path, checkpoint_path, config_path, embodiment='wheeled', 
                         trajectory_mode=True, fps=16.0, device='cuda'):
    """Run inference with Model 1 using subprocess (runs in flow_training conda env)"""
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / 'run_inference_model1.sh'
    inference_script = script_dir / 'inference_wrappers' / 'inference_model1.py'
    
    # Prepare input data
    input_data = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'config_path': str(config_path),
        'embodiment': embodiment,
        'trajectory_mode': trajectory_mode,
        'fps': fps,
        'device': device
    }
    
    # Run via subprocess with conda environment activation
    try:
        # Use bash to run the wrapper script which activates conda
        cmd = ['bash', str(wrapper_script)]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(script_dir)
        )
        
        stdout, stderr = process.communicate(input=json.dumps(input_data))
        
        if process.returncode != 0:
            raise Exception(f"Model 1 inference failed: {stderr}")
        
        # Extract JSON from stdout (it may be mixed with other output)
        result = extract_json_from_output(stdout)
        
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))
        
        return {
            'predictions': result['predictions'],
            'trajectory': result.get('trajectory'),
            'num_frames': result['num_frames'],
            'fps': result['fps']
        }
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Model 1 output: {e}\nOutput: {stdout}\nError: {stderr}")
    except Exception as e:
        raise Exception(f"Model 1 inference error: {str(e)}")


def integrate_trajectory_model1(velocities, dt=1.0/16.0, initial_pose=None):
    """Integrate velocities to get trajectory"""
    if initial_pose is None:
        initial_pose = np.array([0.0, 0.0, 0.0])
    
    T = len(velocities)
    trajectory = np.zeros((T, 3))
    trajectory[0] = initial_pose
    
    for t in range(1, T):
        x_prev, y_prev, theta_prev = trajectory[t-1]
        vx, vy, yaw_rate = velocities[t-1]
        
        theta_new = theta_prev + yaw_rate * dt
        vx_world = vx * np.cos(theta_prev) - vy * np.sin(theta_prev)
        vy_world = vx * np.sin(theta_prev) + vy * np.cos(theta_prev)
        
        x_new = x_prev + vx_world * dt
        y_new = y_prev + vy_world * dt
        
        trajectory[t] = [x_new, y_new, theta_new]
    
    return trajectory


def run_inference_humanoid_v1(video_path, checkpoint_path, trajectory_mode=True,
                              fps=16.0, device='cuda'):
    """Run inference with Humanoid V1 model (DINOv2 + FusionNetwork)"""
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / 'run_inference_humanoid_v1.sh'

    input_data = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'trajectory_mode': trajectory_mode,
        'fps': fps,
        'device': device
    }

    try:
        cmd = ['bash', str(wrapper_script)]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=str(script_dir)
        )
        stdout, stderr = process.communicate(input=json.dumps(input_data))

        if process.returncode != 0:
            raise Exception(f"Humanoid V1 inference failed: {stderr}")

        result = extract_json_from_output(stdout)
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))

        return {
            'predictions': result['predictions'],
            'trajectory': result.get('trajectory'),
            'num_frames': result['num_frames'],
            'fps': result['fps']
        }
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Humanoid V1 output: {e}\nOutput: {stdout}\nError: {stderr}")
    except Exception as e:
        raise Exception(f"Humanoid V1 inference error: {str(e)}")


def run_inference_humanoid_v1_1(video_path, checkpoint_path, trajectory_mode=True,
                                 fps=16.0, device='cuda'):
    """Run inference with Humanoid V1.1 model (RAFT + CLIP + DINOv2 + FusionNetwork)"""
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / 'run_inference_humanoid_v1_1.sh'

    input_data = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'trajectory_mode': trajectory_mode,
        'fps': fps,
        'device': device
    }

    try:
        cmd = ['bash', str(wrapper_script)]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=str(script_dir)
        )
        stdout, stderr = process.communicate(input=json.dumps(input_data), timeout=300)

        if process.returncode != 0:
            raise Exception(f"Humanoid V1.1 inference failed: {stderr}")

        result = extract_json_from_output(stdout)
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))

        return {
            'predictions': result['predictions'],
            'trajectory': result.get('trajectory'),
            'num_frames': result['num_frames'],
            'fps': result['fps']
        }
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Humanoid V1.1 output: {e}\nOutput: {stdout}\nError: {stderr}")
    except Exception as e:
        raise Exception(f"Humanoid V1.1 inference error: {str(e)}")


def run_inference_humanoid_v1_2(video_path, checkpoint_path, trajectory_mode=True,
                                 fps=16.0, device='cuda'):
    """Run inference with Humanoid V1.2 model (RAFT + CLIP + DINOv2 + FusionNetwork, combined data)"""
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / 'run_inference_humanoid_v1_2.sh'

    input_data = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'trajectory_mode': trajectory_mode,
        'fps': fps,
        'device': device
    }

    try:
        cmd = ['bash', str(wrapper_script)]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=str(script_dir)
        )
        stdout, stderr = process.communicate(input=json.dumps(input_data), timeout=300)

        if process.returncode != 0:
            raise Exception(f"Humanoid V1.2 inference failed: {stderr}")

        result = extract_json_from_output(stdout)
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))

        return {
            'predictions': result['predictions'],
            'trajectory': result.get('trajectory'),
            'num_frames': result['num_frames'],
            'fps': result['fps']
        }
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Humanoid V1.2 output: {e}\nOutput: {stdout}\nError: {stderr}")
    except Exception as e:
        raise Exception(f"Humanoid V1.2 inference error: {str(e)}")


def run_inference_humanoid_v2(video_path, checkpoint_path, current_obs_mode='middle',
                               device='cuda'):
    """Run inference with Humanoid V2 model (FlowDiT goal-conditioned)"""
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / 'run_inference_humanoid_v2.sh'

    input_data = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'current_obs_mode': current_obs_mode,
        'device': device
    }

    try:
        cmd = ['bash', str(wrapper_script)]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=str(script_dir)
        )
        stdout, stderr = process.communicate(input=json.dumps(input_data))

        if process.returncode != 0:
            raise Exception(f"Humanoid V2 inference failed: {stderr}")

        result = json.loads(stdout)
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))

        return {
            'predictions': result['predictions'],
            'trajectory': result.get('trajectory'),
            'num_frames': result['num_frames'],
            'fps': result['fps'],
            'goal_video': result.get('goal_video'),
            'current_obs': result.get('current_obs')
        }
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Humanoid V2 output: {e}\nOutput: {stdout}\nError: {stderr}")
    except Exception as e:
        raise Exception(f"Humanoid V2 inference error: {str(e)}")


def run_inference_humanoid_v2_2(video_path, checkpoint_path, current_obs_mode='middle',
                                 device='cuda'):
    """Run inference with Humanoid V2.2 model (FlowDiT goal-conditioned, combined data)"""
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / 'run_inference_humanoid_v2_2.sh'

    input_data = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'current_obs_mode': current_obs_mode,
        'device': device
    }

    try:
        cmd = ['bash', str(wrapper_script)]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=str(script_dir)
        )
        stdout, stderr = process.communicate(input=json.dumps(input_data))

        if process.returncode != 0:
            raise Exception(f"Humanoid V2.2 inference failed: {stderr}")

        result = json.loads(stdout)
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))

        return {
            'predictions': result['predictions'],
            'trajectory': result.get('trajectory'),
            'num_frames': result['num_frames'],
            'fps': result['fps'],
            'goal_video': result.get('goal_video'),
            'current_obs': result.get('current_obs')
        }
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Humanoid V2.2 output: {e}\nOutput: {stdout}\nError: {stderr}")
    except Exception as e:
        raise Exception(f"Humanoid V2.2 inference error: {str(e)}")


def run_inference_humanoid_v2_2_dual_mode(video_path, checkpoint_path,
                                           inference_mode='mode1', video_fps=16,
                                           current_obs_mode='middle', device='cuda'):
    """Run dual-mode inference with Humanoid V2.2 model.

    Mode 1 (Full Trajectory): Processes every frame of the video, producing
    exactly FPS * duration velocity points. Goal video is encoded once.

    Mode 2 (Realtime): Uses a single observation frame with cached goal features.
    Returns 8-step action horizon for closed-loop control.
    """
    import torch
    import cv2
    import time
    import importlib.util

    # Import from flow_constrained_v2 using explicit path (avoids flow_constrained/diffusers conflict)
    v2_models_path = str(PROJECT_ROOT / "flow_constrained_v2" / "models" / "flowdit_production.py")
    spec = importlib.util.spec_from_file_location("flowdit_production", v2_models_path)
    flowdit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flowdit_module)
    create_flowdit_production = flowdit_module.create_flowdit_production

    # Load model
    model = create_flowdit_production(device=device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load video frames
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if not frames:
        raise Exception("Could not read any frames from video")

    video = np.stack(frames, axis=0)
    total_frames = len(frames)

    if inference_mode == 'mode1':
        # Mode 1: subsample to target FPS if needed, then predict per-frame
        if original_fps > video_fps * 1.5:
            subsample = max(1, int(round(original_fps / video_fps)))
            video_sub = video[::subsample]
            effective_fps = original_fps / subsample
        else:
            video_sub = video
            effective_fps = original_fps

        t0 = time.time()
        result = model.predict_full_trajectory(video_sub, video_fps=int(round(effective_fps)))
        t1 = time.time()

        total_time = t1 - t0
        n_points = result['n_frames']

        return {
            'predictions': result['velocities'].tolist(),
            'trajectory': result['trajectory'].tolist(),
            'num_frames': total_frames,
            'fps': int(round(effective_fps)),
            'inference_mode': 'mode1',
            'total_distance': result['total_distance'],
            'mean_speed': result['mean_speed'],
            'duration_sec': result['duration_sec'],
            'final_position': result['final_position'],
            'speed_profile': result['speed_profile'].tolist(),
            'n_velocity_points': n_points,
            'inference_hz': n_points / total_time if total_time > 0 else 0,
            'total_inference_sec': total_time,
        }

    else:
        # Mode 2: Single observation frame with cached goal features
        if current_obs_mode == 'first':
            obs_idx = 0
        elif current_obs_mode == 'last':
            obs_idx = len(video) - 1
        else:
            obs_idx = len(video) // 2

        current_obs = video[obs_idx]

        t0 = time.time()
        actions, cache = model.predict_realtime(video, current_obs, goal_features_cache=None)
        t1 = time.time()

        # Integrate trajectory from actions
        trajectory = integrate_trajectory_model2(actions.tolist())

        return {
            'predictions': actions.tolist(),
            'trajectory': trajectory.tolist(),
            'num_frames': total_frames,
            'fps': int(round(original_fps)),
            'inference_mode': 'mode2',
            'inference_hz': 1000.0 / ((t1 - t0) * 1000),
            'total_inference_sec': t1 - t0,
        }


def run_inference_model2(video_path, checkpoint_path, current_obs_mode='middle',
                         device='cuda'):
    """Run inference with Model 2 using subprocess (runs in flowdit_v2_py310 conda env)"""
    script_dir = Path(__file__).parent
    wrapper_script = script_dir / 'run_inference_model2.sh'
    
    # Prepare input data
    input_data = {
        'video_path': str(video_path),
        'checkpoint_path': str(checkpoint_path),
        'current_obs_mode': current_obs_mode,
        'device': device
    }
    
    # Run via subprocess with conda environment activation
    try:
        # Use bash to run the wrapper script which activates conda
        cmd = ['bash', str(wrapper_script)]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(script_dir)
        )
        
        stdout, stderr = process.communicate(input=json.dumps(input_data))
        
        if process.returncode != 0:
            raise Exception(f"Model 2 inference failed: {stderr}")
        
        result = json.loads(stdout)
        
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))
        
        return {
            'predictions': result['predictions'],
            'trajectory': result.get('trajectory'),
            'num_frames': result['num_frames'],
            'fps': result['fps'],
            'goal_video': result.get('goal_video'),  # For visualization
            'current_obs': result.get('current_obs')  # For visualization
        }
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse Model 2 output: {e}\nOutput: {stdout}\nError: {stderr}")
    except Exception as e:
        raise Exception(f"Model 2 inference error: {str(e)}")


def integrate_trajectory_model2(actions, dt=1.0/16.0):
    """Integrate actions to get trajectory for Model 2"""
    T = len(actions)
    traj = np.zeros((T + 1, 3))
    
    for i, (vx, vy, yaw_rate) in enumerate(actions):
        x, y, theta = traj[i]
        theta_new = theta + yaw_rate * dt
        vx_world = vx * np.cos(theta) - vy * np.sin(theta)
        vy_world = vx * np.sin(theta) + vy * np.cos(theta)
        x_new = x + vx_world * dt
        y_new = y + vy_world * dt
        traj[i + 1] = [x_new, y_new, theta_new]
    
    return traj[1:]


def create_visualization_model1(predictions, trajectory, output_path):
    """Create visualization for Model 1"""
    predictions = np.array(predictions)
    trajectory = np.array(trajectory) if trajectory else None
    
    fig = plt.figure(figsize=(16, 10))
    
    if trajectory is not None:
        # 4-panel visualization
        T = len(predictions)
        time_axis = np.arange(T) / 16.0
        
        x, y, theta = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
        
        # Panel 1: 3D Trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        colors = plt.cm.viridis(np.linspace(0, 1, T))
        ax1.scatter(x, y, time_axis, c=colors, s=20, alpha=0.6)
        ax1.plot(x, y, time_axis, 'b-', alpha=0.3, linewidth=1.5)
        ax1.scatter(x[0], y[0], time_axis[0], c='green', s=200, marker='o', edgecolors='black', linewidth=2)
        ax1.scatter(x[-1], y[-1], time_axis[-1], c='red', s=200, marker='s', edgecolors='black', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Time (s)')
        ax1.set_title('3D Trajectory (X-Y-Time)')
        
        # Panel 2: Top-down view
        ax2 = fig.add_subplot(222)
        ax2.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        ax2.scatter(x[0], y[0], c='green', s=200, marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax2.scatter(x[-1], y[-1], c='red', s=200, marker='s', edgecolors='black', linewidth=2, zorder=5)
        
        step = max(1, T // 20)
        for i in range(0, T, step):
            vx, vy, _ = predictions[i]
            vx_world = vx * np.cos(theta[i]) - vy * np.sin(theta[i])
            vy_world = vx * np.sin(theta[i]) + vy * np.cos(theta[i])
            scale = 0.5
            ax2.arrow(x[i], y[i], vx_world * scale, vy_world * scale,
                     head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-Down View with Velocity Vectors')
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Velocity components
        ax3 = fig.add_subplot(223)
        ax3.plot(time_axis, predictions[:, 0], 'r-', linewidth=2, label='vx (forward)', alpha=0.8)
        ax3.plot(time_axis, predictions[:, 1], 'g-', linewidth=2, label='vy (lateral)', alpha=0.8)
        ax3.plot(time_axis, predictions[:, 2], 'b-', linewidth=2, label='yaw_rate (angular)', alpha=0.8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity')
        ax3.set_title('Velocity Components Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Heading
        ax4 = fig.add_subplot(224, projection='polar')
        scatter = ax4.scatter(theta, np.ones_like(theta), c=time_axis, cmap='viridis', s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.plot(theta, np.ones_like(theta), 'b-', alpha=0.3, linewidth=1.5)
        ax4.set_title('Heading Direction Over Time', pad=20)
        ax4.set_theta_zero_location('E')
        plt.colorbar(scatter, ax=ax4, label='Time (s)', pad=0.1)
    else:
        # Simple bar chart for single prediction
        ax = fig.add_subplot(111)
        action_names = ['vx (m/s)', 'vy (m/s)', 'yaw (rad/s)']
        values = predictions[0]
        ax.bar(action_names, values, color=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Predicted Robot Actions')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_visualization_model2(goal_video, current_obs, actions, trajectory, output_path):
    """Create visualization for Model 2"""
    fig = plt.figure(figsize=(16, 10))
    
    # Panel 1: Goal video frames
    ax1 = fig.add_subplot(221)
    if len(goal_video) > 0:
        goal_indices = [0, len(goal_video) // 2, -1]
        frames_to_show = [goal_video[i] for i in goal_indices]
        # Combine frames horizontally
        combined = np.hstack(frames_to_show)
        ax1.imshow(combined)
        ax1.set_title('Goal Video (First, Middle, Last)')
    else:
        ax1.text(0.5, 0.5, 'Video frames not available', ha='center', va='center')
        ax1.set_title('Goal Video')
    ax1.axis('off')
    
    # Panel 2: Current observation
    ax2 = fig.add_subplot(222)
    if len(current_obs) > 0:
        ax2.imshow(current_obs)
        ax2.set_title('Current Observation')
    else:
        ax2.text(0.5, 0.5, 'Current observation not available', ha='center', va='center')
        ax2.set_title('Current Observation')
    ax2.axis('off')
    
    # Panel 3: Actions
    ax3 = fig.add_subplot(223)
    steps = np.arange(len(actions))
    actions = np.array(actions)
    ax3.plot(steps, actions[:, 0], 'r-o', label='vx (forward)', linewidth=2)
    ax3.plot(steps, actions[:, 1], 'g-s', label='vy (lateral)', linewidth=2)
    ax3.plot(steps, actions[:, 2], 'b-^', label='yaw_rate (angular)', linewidth=2)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Predicted Actions (8 Steps)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Trajectory
    ax4 = fig.add_subplot(224)
    trajectory = np.array(trajectory)
    x, y = trajectory[:, 0], trajectory[:, 1]
    ax4.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax4.scatter(x[0], y[0], c='green', s=200, marker='o', edgecolors='black', linewidth=2, zorder=5, label='Start')
    ax4.scatter(x[-1], y[-1], c='red', s=200, marker='s', edgecolors='black', linewidth=2, zorder=5, label='End')
    
    # Add velocity vectors
    for i in range(len(actions)):
        vx, vy, _ = actions[i]
        theta = trajectory[i, 2] if i < len(trajectory) else 0
        vx_world = vx * np.cos(theta) - vy * np.sin(theta)
        vy_world = vx * np.sin(theta) + vy * np.cos(theta)
        scale = 0.3
        ax4.arrow(x[i], y[i], vx_world * scale, vy_world * scale,
                 head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Integrated Trajectory')
    ax4.legend()
    ax4.set_aspect('equal', adjustable='box')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_visualization_mode1(velocities, trajectory, speed_profile, fps, output_path):
    """Create visualization for Mode 1 full trajectory output."""
    velocities = np.array(velocities)
    trajectory = np.array(trajectory)
    speed_profile = np.array(speed_profile) if len(speed_profile) > 0 else np.linalg.norm(velocities[:, :2], axis=1)
    T = len(velocities)
    t_axis = np.arange(T) / float(fps)

    fig = plt.figure(figsize=(18, 10))

    # Panel 1: Bird-eye trajectory
    ax1 = fig.add_subplot(231)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5)
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Mode 1: Bird-Eye Trajectory ({T} points)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Velocity profile
    ax2 = fig.add_subplot(232)
    ax2.plot(t_axis, velocities[:, 0], 'r-', label='vx', alpha=0.8)
    ax2.plot(t_axis, velocities[:, 1], 'g-', label='vy', alpha=0.8)
    ax2.plot(t_axis, velocities[:, 2], 'b-', label='yaw', alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity')
    ax2.set_title(f'Velocity Profile ({T} velocity points)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Speed profile
    ax3 = fig.add_subplot(233)
    ax3.fill_between(t_axis, speed_profile, alpha=0.3, color='blue')
    ax3.plot(t_axis, speed_profile, 'b-', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('Speed Profile')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Heading over time
    ax4 = fig.add_subplot(234)
    ax4.plot(t_axis, np.degrees(trajectory[1:, 2]), 'purple', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Heading (deg)')
    ax4.set_title('Heading Angle')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Top-down with velocity arrows
    ax5 = fig.add_subplot(235)
    ax5.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1, alpha=0.5)
    step = max(1, T // 15)
    for i in range(0, T, step):
        vx, vy, _ = velocities[i]
        theta = trajectory[i+1, 2] if i+1 < len(trajectory) else 0
        vx_w = vx * np.cos(theta) - vy * np.sin(theta)
        vy_w = vx * np.sin(theta) + vy * np.cos(theta)
        scale = 0.3
        ax5.arrow(trajectory[i+1, 0], trajectory[i+1, 1], vx_w*scale, vy_w*scale,
                  head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.6)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Trajectory with Velocity Vectors')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Summary stats
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    total_dist = float(np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)))
    stats_text = (
        f"Inference Mode: Full Trajectory (Mode 1)\n"
        f"Video FPS: {fps}\n"
        f"Velocity Points: {T}\n"
        f"Duration: {T/fps:.2f}s\n"
        f"Total Distance: {total_dist:.3f}m\n"
        f"Mean Speed: {np.mean(speed_profile):.3f} m/s\n"
        f"Mean vx: {np.mean(velocities[:,0]):.4f} m/s\n"
        f"Mean vy: {np.mean(velocities[:,1]):.5f} m/s\n"
        f"Mean yaw: {np.mean(velocities[:,2]):.4f} rad/s\n"
        f"Final Pos: ({trajectory[-1,0]:.3f}, {trajectory[-1,1]:.3f})\n"
        f"Final Heading: {np.degrees(trajectory[-1,2]):.1f} deg"
    )
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle(f'FlowDiT V2.2 - Mode 1: Full Trajectory from Video ({T} velocity points @ {fps}fps)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


def create_simple_visualization_model2(actions, trajectory, output_path):
    """Create simple visualization for Model 2 without video frames"""
    fig = plt.figure(figsize=(12, 8))
    
    # Panel 1: Actions
    ax1 = fig.add_subplot(121)
    steps = np.arange(len(actions))
    actions = np.array(actions)
    ax1.plot(steps, actions[:, 0], 'r-o', label='vx (forward)', linewidth=2)
    ax1.plot(steps, actions[:, 1], 'g-s', label='vy (lateral)', linewidth=2)
    ax1.plot(steps, actions[:, 2], 'b-^', label='yaw_rate (angular)', linewidth=2)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Velocity')
    ax1.set_title('Predicted Actions (8 Steps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Trajectory
    ax2 = fig.add_subplot(122)
    trajectory = np.array(trajectory)
    x, y = trajectory[:, 0], trajectory[:, 1]
    ax2.plot(x, y, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax2.scatter(x[0], y[0], c='green', s=200, marker='o', edgecolors='black', linewidth=2, zorder=5, label='Start')
    ax2.scatter(x[-1], y[-1], c='red', s=200, marker='s', edgecolors='black', linewidth=2, zorder=5, label='End')
    
    # Add velocity vectors
    for i in range(len(actions)):
        vx, vy, _ = actions[i]
        theta = trajectory[i, 2] if i < len(trajectory) else 0
        vx_world = vx * np.cos(theta) - vy * np.sin(theta)
        vy_world = vx * np.sin(theta) + vy * np.cos(theta)
        scale = 0.3
        ax2.arrow(x[i], y[i], vx_world * scale, vy_world * scale,
                 head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Integrated Trajectory')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle video upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(str(filepath))
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': str(filepath)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/preview_video', methods=['POST'])
def preview_video():
    """Generate a video preview (first frame)"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        import cv2
        import tempfile
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        # Extract first frame
        cap = cv2.VideoCapture(temp_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(temp_path)
        
        if not ret:
            return jsonify({'error': 'Could not read video'}), 400
        
        # Resize and convert to base64
        frame = cv2.resize(frame, (320, 180))
        ret, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'preview': f'data:image/jpeg;base64,{img_base64}',
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inference', methods=['POST'])
def run_inference():
    """Run inference on uploaded video"""
    data = request.json
    filename = data.get('filename')
    model_type = data.get('model_type', 'model1')  # 'model1' or 'model2'
    trajectory_mode = data.get('trajectory_mode', True)
    embodiment = data.get('embodiment', 'wheeled')
    current_obs_mode = data.get('current_obs_mode', 'middle')
    device = data.get('device', 'cuda')
    
    filepath = app.config['UPLOAD_FOLDER'] / filename
    
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    
    inference_mode = data.get('inference_mode', 'mode2')
    video_fps = data.get('video_fps', 16)

    try:
        if model_type == 'model1':
            checkpoint_path = PROJECT_ROOT / 'flow_constrained' / 'checkpoints' / 'wheeled' / 'best_model.pth'
            config_path = PROJECT_ROOT / 'flow_constrained' / 'configs' / 'wheeled.yaml'

            result = run_inference_model1(
                str(filepath),
                str(checkpoint_path),
                str(config_path),
                embodiment=embodiment,
                trajectory_mode=trajectory_mode,
                fps=16.0,
                device=device
            )

            # Create visualization
            viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_model1_viz.png"
            create_visualization_model1(
                result['predictions'],
                result['trajectory'],
                str(viz_path)
            )

        elif model_type == 'humanoid_v1':
            checkpoint_path = Path(os.getenv("CHECKPOINT_HUMANOID_V1", "./checkpoints/humanoid_v1/best_model.pth"))

            result = run_inference_humanoid_v1(
                str(filepath),
                str(checkpoint_path),
                trajectory_mode=trajectory_mode,
                fps=16.0,
                device=device
            )

            viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_humanoid_v1_viz.png"
            create_visualization_model1(
                result['predictions'],
                result['trajectory'],
                str(viz_path)
            )

        elif model_type == 'humanoid_v1_1':
            checkpoint_path = Path(os.getenv("CHECKPOINT_HUMANOID_V1_1", "./checkpoints/humanoid_v1_1/best_model.pth"))

            result = run_inference_humanoid_v1_1(
                str(filepath),
                str(checkpoint_path),
                trajectory_mode=trajectory_mode,
                fps=16.0,
                device=device
            )

            viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_humanoid_v1_1_viz.png"
            create_visualization_model1(
                result['predictions'],
                result['trajectory'],
                str(viz_path)
            )

        elif model_type == 'humanoid_v1_2':
            checkpoint_path = Path(os.getenv("CHECKPOINT_HUMANOID_V1_2", "./checkpoints/humanoid_v1_2/best_model.pth"))

            result = run_inference_humanoid_v1_2(
                str(filepath),
                str(checkpoint_path),
                trajectory_mode=trajectory_mode,
                fps=16.0,
                device=device
            )

            viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_humanoid_v1_2_viz.png"
            create_visualization_model1(
                result['predictions'],
                result['trajectory'],
                str(viz_path)
            )

        elif model_type == 'humanoid_v2_2':
            checkpoint_path = Path(os.getenv("CHECKPOINT_HUMANOID_V2_2", "./checkpoints/humanoid_v2_2/best.pth"))

            if inference_mode in ('mode1', 'mode2'):
                # Dual-mode inference (direct, no subprocess)
                result = run_inference_humanoid_v2_2_dual_mode(
                    str(filepath),
                    str(checkpoint_path),
                    inference_mode=inference_mode,
                    video_fps=video_fps,
                    current_obs_mode=current_obs_mode,
                    device=device
                )

                viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_humanoid_v2_2_{inference_mode}_viz.png"
                if inference_mode == 'mode1':
                    create_visualization_mode1(
                        result['predictions'],
                        result['trajectory'],
                        result.get('speed_profile', []),
                        result.get('fps', 16),
                        str(viz_path)
                    )
                else:
                    create_simple_visualization_model2(
                        result['predictions'],
                        result['trajectory'],
                        str(viz_path)
                    )
            else:
                # Legacy subprocess-based inference
                result = run_inference_humanoid_v2_2(
                    str(filepath),
                    str(checkpoint_path),
                    current_obs_mode=current_obs_mode,
                    device=device
                )

                goal_video = np.array(result.get('goal_video', []))
                current_obs_arr = np.array(result.get('current_obs', []))

                viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_humanoid_v2_2_viz.png"
                if len(goal_video) > 0 and len(current_obs_arr) > 0:
                    create_visualization_model2(
                        goal_video,
                        current_obs_arr,
                        result['predictions'],
                        result['trajectory'],
                        str(viz_path)
                    )
                else:
                    create_simple_visualization_model2(
                        result['predictions'],
                        result['trajectory'],
                        str(viz_path)
                    )

        elif model_type == 'humanoid_v2':
            checkpoint_path = Path(os.getenv("CHECKPOINT_HUMANOID_V2", "./checkpoints/humanoid_v2/best.pth"))

            result = run_inference_humanoid_v2(
                str(filepath),
                str(checkpoint_path),
                current_obs_mode=current_obs_mode,
                device=device
            )

            goal_video = np.array(result.get('goal_video', []))
            current_obs = np.array(result.get('current_obs', []))

            viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_humanoid_v2_viz.png"
            if len(goal_video) > 0 and len(current_obs) > 0:
                create_visualization_model2(
                    goal_video,
                    current_obs,
                    result['predictions'],
                    result['trajectory'],
                    str(viz_path)
                )
            else:
                create_simple_visualization_model2(
                    result['predictions'],
                    result['trajectory'],
                    str(viz_path)
                )

        else:  # model2
            checkpoint_path = PROJECT_ROOT / 'flow_constrained_v2' / 'checkpoints' / 'best.pth'

            result = run_inference_model2(
                str(filepath),
                str(checkpoint_path),
                current_obs_mode=current_obs_mode,
                device=device
            )

            # Get video data from result (already loaded in wrapper)
            goal_video = np.array(result.get('goal_video', []))
            current_obs = np.array(result.get('current_obs', []))

            # Create visualization
            viz_path = app.config['OUTPUT_FOLDER'] / f"{filename}_model2_viz.png"
            if len(goal_video) > 0 and len(current_obs) > 0:
                create_visualization_model2(
                    goal_video,
                    current_obs,
                    result['predictions'],
                    result['trajectory'],
                    str(viz_path)
                )
            else:
                # Fallback: create simple visualization without video frames
                create_simple_visualization_model2(
                    result['predictions'],
                    result['trajectory'],
                    str(viz_path)
                )
        
        # Convert visualization to base64
        with open(viz_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        
        response_data = {
            'success': True,
            'predictions': result['predictions'],
            'trajectory': result['trajectory'],
            'num_frames': result['num_frames'],
            'fps': result.get('fps', 16),
            'visualization': img_data,
            'viz_filename': viz_path.name,
        }
        # Add dual-mode specific fields
        for key in ('inference_mode', 'total_distance', 'mean_speed',
                     'duration_sec', 'final_position', 'speed_profile',
                     'n_velocity_points', 'inference_hz'):
            if key in result:
                response_data[key] = result[key]

        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/download/<format>', methods=['POST'])
def download_data(format):
    """Download predictions in various formats"""
    data = request.json
    predictions = data.get('predictions', [])
    trajectory = data.get('trajectory')
    filename = data.get('filename', 'predictions')
    
    predictions = np.array(predictions)
    
    if format == 'csv':
        # Create DataFrame
        df_data = {
            'frame': range(len(predictions)),
            'vx_m_s': predictions[:, 0],
            'vy_m_s': predictions[:, 1],
            'yaw_rate_rad_s': predictions[:, 2]
        }
        
        if trajectory:
            traj = np.array(trajectory)
            df_data['x_m'] = traj[:, 0]
            df_data['y_m'] = traj[:, 1]
            df_data['heading_rad'] = traj[:, 2]
        
        df = pd.DataFrame(df_data)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{filename}.csv'
        )
    
    elif format == 'excel':
        df_data = {
            'frame': range(len(predictions)),
            'vx_m_s': predictions[:, 0],
            'vy_m_s': predictions[:, 1],
            'yaw_rate_rad_s': predictions[:, 2]
        }
        
        if trajectory:
            traj = np.array(trajectory)
            df_data['x_m'] = traj[:, 0]
            df_data['y_m'] = traj[:, 1]
            df_data['heading_rad'] = traj[:, 2]
        
        df = pd.DataFrame(df_data)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'{filename}.xlsx'
        )
    
    elif format == 'txt':
        output = io.StringIO()
        output.write("Frame\tvx (m/s)\tvy (m/s)\tyaw_rate (rad/s)")
        if trajectory:
            output.write("\tx (m)\ty (m)\theading (rad)")
        output.write("\n")
        
        for i, pred in enumerate(predictions):
            output.write(f"{i}\t{pred[0]:.6f}\t{pred[1]:.6f}\t{pred[2]:.6f}")
            if trajectory:
                traj = np.array(trajectory)
                if i < len(traj):
                    output.write(f"\t{traj[i, 0]:.6f}\t{traj[i, 1]:.6f}\t{traj[i, 2]:.6f}")
            output.write("\n")
        
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name=f'{filename}.txt'
        )
    
    return jsonify({'error': 'Invalid format'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)

