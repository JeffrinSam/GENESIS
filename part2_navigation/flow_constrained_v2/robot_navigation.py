"""
FlowDiT V2 - Closed-Loop Robot Navigation
==========================================

Simulates real robot deployment with continuous feedback loop.

Usage:
    python robot_navigation.py --checkpoint checkpoints/best.pth \
                               --goal_video reference.mp4 \
                               --camera 0

Author: Jeffrin Sam
Date: January 2026
"""

import torch
import numpy as np
import cv2
import argparse
import time
from pathlib import Path
from models.flowdit_production import create_flowdit_production
from inference import load_model, load_video


class RobotNavigator:
    """
    Closed-loop navigation controller.

    In real deployment:
    - Replace simulate_camera() with robot.camera.capture()
    - Replace simulate_execute() with robot.send_velocity_command()
    """

    def __init__(
        self,
        model,
        goal_video: np.ndarray,
        control_hz: float = 2.0,  # Control frequency (Hz)
        action_horizon: int = 8,
        execute_steps: int = 3  # Execute first N actions before replanning
    ):
        self.model = model
        self.goal_video = goal_video
        self.control_period = 1.0 / control_hz
        self.action_horizon = action_horizon
        self.execute_steps = execute_steps

        print(f"\n{'='*70}")
        print("Robot Navigator Initialized")
        print(f"{'='*70}")
        print(f"Control frequency: {control_hz} Hz ({self.control_period*1000:.0f} ms)")
        print(f"Action horizon: {action_horizon} steps")
        print(f"Execute per cycle: {execute_steps} actions")
        print(f"{'='*70}\n")

    def get_current_observation(self, camera_id=0):
        """
        Get current observation from camera.

        In real robot: Replace with robot.camera.capture()
        """
        # For simulation: capture from webcam or video
        cap = cv2.VideoCapture(camera_id)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        # Preprocess
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.float32) / 255.0

        return frame

    def predict_actions(self, current_obs: np.ndarray):
        """Predict actions given current observation."""
        with torch.no_grad():
            actions = self.model.predict(self.goal_video, current_obs)
        return actions

    def execute_actions(self, actions: np.ndarray):
        """
        Execute actions on robot.

        In real robot: Replace with robot.send_velocity_command()
        """
        print(f"\nExecuting {len(actions)} actions:")
        for i, (vx, vy, yaw) in enumerate(actions):
            print(f"  Step {i+1}: vx={vx:6.3f}, vy={vy:6.3f}, yaw={yaw:6.3f}")

            # Simulate execution
            # In real robot:
            # robot.send_velocity_command(vx, vy, yaw)
            # time.sleep(1.0 / 16)  # 16 Hz action execution

        print(f"✓ Executed {len(actions)} actions")

    def check_goal_reached(self, current_obs: np.ndarray):
        """
        Check if goal is reached.

        Can be implemented as:
        - Visual similarity with goal frames
        - Distance threshold
        - User input
        """
        # Simple implementation: return False for now
        # In real robot: implement proper goal checking
        return False

    def run(self, max_steps: int = 100, camera_id: int = 0):
        """
        Run closed-loop navigation.

        Args:
            max_steps: Maximum navigation steps
            camera_id: Camera device ID or video path
        """
        print(f"\n{'='*70}")
        print("Starting Closed-Loop Navigation")
        print(f"{'='*70}\n")

        step = 0
        total_actions_executed = 0

        try:
            while step < max_steps:
                step += 1
                print(f"\n--- Navigation Step {step}/{max_steps} ---")

                # 1. Get current observation
                start_time = time.time()
                current_obs = self.get_current_observation(camera_id)
                obs_time = time.time() - start_time
                print(f"Observation captured: {obs_time*1000:.1f} ms")

                # 2. Predict actions
                start_time = time.time()
                actions = self.predict_actions(current_obs)
                pred_time = time.time() - start_time
                print(f"Actions predicted: {pred_time*1000:.1f} ms")

                # 3. Execute first N actions
                actions_to_execute = actions[:self.execute_steps]
                self.execute_actions(actions_to_execute)
                total_actions_executed += len(actions_to_execute)

                # 4. Check if goal reached
                if self.check_goal_reached(current_obs):
                    print("\n✓ GOAL REACHED!")
                    break

                # 5. Wait for next control cycle
                cycle_time = time.time() - start_time + obs_time + pred_time
                sleep_time = max(0, self.control_period - cycle_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                total_time = time.time() - start_time + obs_time + pred_time
                print(f"Total cycle time: {total_time*1000:.1f} ms")

        except KeyboardInterrupt:
            print("\n\nNavigation interrupted by user")

        print(f"\n{'='*70}")
        print("Navigation Complete")
        print(f"{'='*70}")
        print(f"Total steps: {step}")
        print(f"Total actions executed: {total_actions_executed}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="FlowDiT V2 Closed-Loop Navigation")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    # Goal
    parser.add_argument("--goal_video", type=str, required=True,
                        help="Path to goal/reference video")

    # Camera
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID (0 for webcam) or video path")

    # Control
    parser.add_argument("--control_hz", type=float, default=2.0,
                        help="Control frequency in Hz")
    parser.add_argument("--execute_steps", type=int, default=3,
                        help="Number of actions to execute per cycle")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum navigation steps")

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load goal video
    goal_video = load_video(args.goal_video)

    # Create navigator
    navigator = RobotNavigator(
        model=model,
        goal_video=goal_video,
        control_hz=args.control_hz,
        execute_steps=args.execute_steps
    )

    # Run navigation
    navigator.run(max_steps=args.max_steps, camera_id=args.camera)


if __name__ == "__main__":
    main()
