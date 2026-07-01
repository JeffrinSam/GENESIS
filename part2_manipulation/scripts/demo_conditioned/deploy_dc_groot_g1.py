#!/usr/bin/env python3
"""
Deploy DC-GR00T on Unitree G1 Robot

This script deploys the trained Demo-Conditioned GR00T model
on your physical Unitree G1 robot for real-world operation.

The workflow is:
1. Load trained DC-GR00T model
2. Set demo video (human/robot/Cosmos) to define the task
3. Start control loop that:
   - Captures live camera feed
   - Gets robot state
   - Predicts actions
   - Sends commands to robot

Usage:
    python deploy_dc_groot_g1.py \
        --checkpoint ./dc_checkpoints/final \
        --demo_video ./demos/pick_up_cup.mp4 \
        --demo_type human

Requirements:
    - Unitree SDK installed
    - Camera connected and accessible
    - Robot in ready state
"""

import argparse
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import queue

import numpy as np
import torch


@dataclass
class G1Config:
    """Configuration for Unitree G1 robot."""
    # Control
    control_freq: float = 30.0  # Hz
    action_horizon: int = 16
    action_execution_steps: int = 4  # Execute first N steps

    # Safety limits
    max_joint_velocity: float = 1.0  # rad/s
    max_joint_acceleration: float = 5.0  # rad/s^2

    # Joint limits (example values - adjust to actual G1 specs)
    left_arm_limits: tuple = ((-2.0, 2.0),) * 6
    right_arm_limits: tuple = ((-2.0, 2.0),) * 6
    hand_limits: tuple = ((0.0, 1.0),) * 6

    # Camera
    camera_id: int = 0
    camera_resolution: tuple = (640, 480)


class CameraCapture:
    """Threaded camera capture for low-latency frames."""

    def __init__(self, camera_id: int = 0, resolution: tuple = (640, 480)):
        import cv2
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        import cv2
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


class G1RobotInterface:
    """
    Interface to Unitree G1 robot.

    NOTE: This is a template. You need to implement the actual
    communication with your robot using the Unitree SDK.
    """

    def __init__(self, config: G1Config):
        self.config = config
        self.connected = False

        # State buffers
        self.current_state = np.zeros(29, dtype=np.float32)

        # Initialize SDK (placeholder)
        self._init_robot()

    def _init_robot(self):
        """Initialize connection to robot."""
        # TODO: Implement actual Unitree SDK initialization
        # Example:
        # from unitree_sdk import G1Robot
        # self.robot = G1Robot()
        # self.robot.connect()
        print("Robot interface initialized (simulation mode)")
        self.connected = True

    def get_state(self) -> np.ndarray:
        """Get current robot state."""
        # TODO: Implement actual state reading
        # Example:
        # joint_positions = self.robot.get_joint_positions()
        # return np.concatenate([
        #     joint_positions["left_arm"],
        #     joint_positions["right_arm"],
        #     joint_positions["left_hand"],
        #     joint_positions["right_hand"],
        #     joint_positions["waist"],
        # ])
        return self.current_state

    def send_action(self, action: np.ndarray) -> bool:
        """
        Send action to robot.

        Args:
            action: Joint position targets [action_dim]

        Returns:
            True if successful
        """
        # Parse action components
        left_arm = action[:6]
        right_arm = action[6:12]
        left_hand = action[12:18]
        right_hand = action[18:24]
        waist = action[24:29]

        # Apply safety limits
        left_arm = self._apply_limits(left_arm, self.config.left_arm_limits)
        right_arm = self._apply_limits(right_arm, self.config.right_arm_limits)

        # TODO: Implement actual command sending
        # Example:
        # self.robot.set_joint_positions({
        #     "left_arm": left_arm,
        #     "right_arm": right_arm,
        #     "left_hand": left_hand,
        #     "right_hand": right_hand,
        #     "waist": waist,
        # })

        # Update simulated state
        self.current_state = action

        return True

    def _apply_limits(self, values: np.ndarray, limits: tuple) -> np.ndarray:
        """Apply joint limits to values."""
        clipped = np.zeros_like(values)
        for i, (val, (lo, hi)) in enumerate(zip(values, limits)):
            clipped[i] = np.clip(val, lo, hi)
        return clipped

    def stop(self):
        """Stop robot and disconnect."""
        # TODO: Implement safe stop
        self.connected = False


class DCGr00TController:
    """Main controller for DC-GR00T deployment on G1."""

    def __init__(
        self,
        checkpoint_path: str,
        config: G1Config,
        device: str = "cuda:0",
    ):
        self.config = config
        self.device = device

        # Initialize components
        print("Initializing DC-GR00T controller...")

        # Load model
        from gr00t.model.demo_conditioned import DCGr00t, DCGr00tConfig
        import json

        config_path = Path(checkpoint_path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                model_config = DCGr00tConfig(**json.load(f))
        else:
            model_config = DCGr00tConfig()

        self.model = DCGr00t(model_config)

        # Load weights
        weight_files = list(Path(checkpoint_path).glob("*.safetensors"))
        if weight_files:
            from safetensors.torch import load_file
            state_dict = load_file(weight_files[0])
            self.model.load_state_dict(state_dict, strict=False)
        else:
            weight_files = list(Path(checkpoint_path).glob("*.bin"))
            if weight_files:
                state_dict = torch.load(weight_files[0], map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)

        self.model.to(device)
        self.model.eval()

        # Initialize camera
        self.camera = CameraCapture(
            camera_id=config.camera_id,
            resolution=config.camera_resolution,
        )

        # Initialize robot
        self.robot = G1RobotInterface(config)

        # Task embedding (set when demo is provided)
        self.task_embedding = None

        # Control state
        self.running = False
        self.action_queue = queue.Queue(maxsize=10)

        print("Controller initialized!")

    def set_demo(self, demo_path: str, demo_type: str = "robot"):
        """Set the demonstration video."""
        print(f"Loading demo: {demo_path}")

        # Load demo video
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(demo_path, ctx=cpu(0))
            T = len(vr)
            indices = np.linspace(0, T - 1, 16, dtype=np.int64)
            frames = vr.get_batch(indices).asnumpy()
        except ImportError:
            import cv2
            cap = cv2.VideoCapture(demo_path)
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            all_frames = np.stack(all_frames)
            indices = np.linspace(0, len(all_frames) - 1, 16, dtype=np.int64)
            frames = all_frames[indices]

        # Convert to tensor
        demo_tensor = torch.from_numpy(frames).float()
        demo_tensor = demo_tensor.permute(0, 3, 1, 2) / 255.0
        demo_tensor = demo_tensor.unsqueeze(0).to(self.device)

        demo_type_map = {"human": 0, "robot": 1, "cosmos": 2, "own": 3}
        demo_type_idx = demo_type_map.get(demo_type, 1)
        demo_type_tensor = torch.tensor([demo_type_idx]).to(self.device)

        with torch.no_grad():
            self.task_embedding = self.model.encode_demo(demo_tensor, demo_type_tensor)

        print(f"Demo encoded! Task embedding: {self.task_embedding.shape}")

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess camera frame for model."""
        import cv2
        resized = cv2.resize(frame, (224, 224))
        tensor = torch.from_numpy(resized).float()
        tensor = tensor.permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def _get_action(self, observation: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Get action from model."""
        obs_tensor = self._preprocess_frame(observation)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        inputs = {
            "pixel_values": obs_tensor,
            "state": state_tensor,
            "embodiment_id": torch.tensor([0]).to(self.device),
        }

        with torch.no_grad():
            outputs = self.model.forward(inputs, task_embedding=self.task_embedding)

        return outputs["action"].cpu().numpy()[0]  # [horizon, action_dim]

    def run(self, duration: float = 60.0, visualize: bool = True):
        """
        Run the control loop.

        Args:
            duration: Maximum run duration in seconds
            visualize: Whether to show camera feed
        """
        if self.task_embedding is None:
            raise RuntimeError("No demo set! Call set_demo() first.")

        # Start camera
        self.camera.start()
        time.sleep(0.5)  # Let camera warm up

        print(f"Starting control loop for {duration}s...")
        print("Press Ctrl+C to stop")

        self.running = True
        start_time = time.time()
        control_dt = 1.0 / self.config.control_freq

        # Metrics
        loop_times = []
        inference_times = []

        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()

                # Get observation
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Get state
                state = self.robot.get_state()

                # Get action
                t0 = time.time()
                action_chunk = self._get_action(frame, state)
                inference_times.append(time.time() - t0)

                # Execute first few steps
                for i in range(min(self.config.action_execution_steps, len(action_chunk))):
                    self.robot.send_action(action_chunk[i])
                    time.sleep(control_dt / self.config.action_execution_steps)

                # Visualization
                if visualize:
                    import cv2
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Add info overlay
                    elapsed = time.time() - start_time
                    cv2.putText(display_frame, f"Time: {elapsed:.1f}s / {duration}s",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if inference_times:
                        avg_inf = np.mean(inference_times[-100:]) * 1000
                        cv2.putText(display_frame, f"Inference: {avg_inf:.1f}ms",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("DC-GR00T Control", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Track loop time
                loop_times.append(time.time() - loop_start)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            self.running = False
            self.camera.stop()
            self.robot.stop()
            if visualize:
                import cv2
                cv2.destroyAllWindows()

        # Print metrics
        print("\n" + "=" * 50)
        print("Control Loop Statistics")
        print("=" * 50)
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Loops executed: {len(loop_times)}")
        print(f"Avg loop time: {np.mean(loop_times)*1000:.1f}ms")
        print(f"Avg inference time: {np.mean(inference_times)*1000:.1f}ms")
        print(f"Control frequency: {len(loop_times) / (time.time() - start_time):.1f} Hz")


def main():
    parser = argparse.ArgumentParser(description="Deploy DC-GR00T on G1")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Demo
    parser.add_argument("--demo_video", type=str, required=True,
                       help="Demo video path")
    parser.add_argument("--demo_type", type=str, default="robot",
                       choices=["human", "robot", "cosmos", "own"])

    # Control
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Maximum run duration (seconds)")
    parser.add_argument("--control_freq", type=float, default=30.0,
                       help="Control frequency (Hz)")

    # Camera
    parser.add_argument("--camera_id", type=int, default=0)

    # Display
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    # Create config
    config = G1Config(
        control_freq=args.control_freq,
        camera_id=args.camera_id,
    )

    # Create controller
    controller = DCGr00TController(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )

    # Set demo
    controller.set_demo(args.demo_video, args.demo_type)

    # Run
    controller.run(
        duration=args.duration,
        visualize=not args.no_visualize,
    )


if __name__ == "__main__":
    main()
