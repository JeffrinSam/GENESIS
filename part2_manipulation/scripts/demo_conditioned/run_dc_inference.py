#!/usr/bin/env python3
"""
DC-GR00T Inference Script

Run inference with a trained Demo-Conditioned GR00T model.
The model watches a demo video to understand the task,
then generates actions based on live observations.

Usage:
    # Single inference with demo video
    python run_dc_inference.py \
        --checkpoint ./dc_checkpoints/final \
        --demo_video ./demos/pick_up_cup.mp4 \
        --demo_type human \
        --camera_id 0

    # Start inference server
    python run_dc_inference.py \
        --checkpoint ./dc_checkpoints/final \
        --server \
        --port 8080

    # Test with saved video (no live camera)
    python run_dc_inference.py \
        --checkpoint ./dc_checkpoints/final \
        --demo_video ./demos/pick_up_cup.mp4 \
        --test_video ./test_observations/episode_000.mp4
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch


def load_video_frames(video_path: str, num_frames: int = 16) -> np.ndarray:
    """Load and uniformly sample frames from video."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        T = len(vr)
        indices = np.linspace(0, T - 1, num_frames, dtype=np.int64)
        frames = vr.get_batch(indices).asnumpy()
        return frames
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = np.stack(frames)
        T = len(frames)
        indices = np.linspace(0, T - 1, num_frames, dtype=np.int64)
        return frames[indices]


def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess frame for model input."""
    import cv2
    if frame.shape[:2] != target_size:
        frame = cv2.resize(frame, target_size)
    return frame


class DCGr00TInference:
    """DC-GR00T inference handler."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        num_demo_keyframes: int = 16,
    ):
        self.device = device
        self.num_demo_keyframes = num_demo_keyframes

        print(f"Loading DC-GR00T from {checkpoint_path}...")

        # Load model
        from gr00t.model.demo_conditioned import DCGr00t, DCGr00tConfig

        # Load config
        config_path = Path(checkpoint_path) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = DCGr00tConfig(**config_dict)
        else:
            config = DCGr00tConfig()

        # Load model weights
        self.model = DCGr00t(config)

        # Try loading state dict
        weight_files = list(Path(checkpoint_path).glob("*.bin")) + \
                      list(Path(checkpoint_path).glob("*.pt")) + \
                      list(Path(checkpoint_path).glob("*.safetensors"))

        if weight_files:
            if weight_files[0].suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(weight_files[0])
            else:
                state_dict = torch.load(weight_files[0], map_location="cpu")

            # Handle "model." prefix
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(device)
        self.model.eval()

        print("Model loaded successfully!")

        # Current task embedding (set when demo is provided)
        self.task_embedding = None
        self.demo_type_idx = 1  # Default to robot

    def set_demo(
        self,
        demo_video_path: str,
        demo_type: str = "robot",
    ) -> None:
        """Set the demonstration video that defines the task."""
        print(f"Loading demo: {demo_video_path} (type: {demo_type})")

        # Load demo frames
        demo_frames = load_video_frames(demo_video_path, self.num_demo_keyframes)

        # Convert to tensor
        demo_tensor = torch.from_numpy(demo_frames).float()
        demo_tensor = demo_tensor.permute(0, 3, 1, 2) / 255.0  # [K, C, H, W]
        demo_tensor = demo_tensor.unsqueeze(0).to(self.device)  # [1, K, C, H, W]

        # Demo type
        demo_type_map = {"human": 0, "robot": 1, "cosmos": 2, "own": 3}
        self.demo_type_idx = demo_type_map.get(demo_type, 1)
        demo_type_tensor = torch.tensor([self.demo_type_idx]).to(self.device)

        # Encode demo
        with torch.no_grad():
            self.task_embedding = self.model.encode_demo(demo_tensor, demo_type_tensor)

        print(f"Task embedding shape: {self.task_embedding.shape}")
        print("Demo encoded! Ready for inference.")

    def get_action(
        self,
        observation: np.ndarray,
        state: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get action given current observation and state.

        Args:
            observation: Current camera frame [H, W, C] (uint8)
            state: Current robot state [state_dim] (float32)

        Returns:
            Dict with action predictions
        """
        if self.task_embedding is None:
            raise RuntimeError("No demo set! Call set_demo() first.")

        # Preprocess observation
        obs = preprocess_frame(observation)
        obs_tensor = torch.from_numpy(obs).float()
        obs_tensor = obs_tensor.permute(2, 0, 1) / 255.0  # [C, H, W]
        obs_tensor = obs_tensor.unsqueeze(0).to(self.device)  # [1, C, H, W]

        # Prepare state
        state_tensor = torch.from_numpy(state.astype(np.float32))
        state_tensor = state_tensor.unsqueeze(0).to(self.device)  # [1, state_dim]

        # Prepare inputs
        inputs = {
            "pixel_values": obs_tensor,
            "state": state_tensor,
            "embodiment_id": torch.tensor([0]).to(self.device),  # G1_DC
        }

        # Get action
        with torch.no_grad():
            outputs = self.model.forward(inputs, task_embedding=self.task_embedding)

        action = outputs["action"].cpu().numpy()[0]  # [horizon, action_dim]

        # Parse action components (G1 specific)
        result = {
            "action": action,
            "left_arm": action[:, :6],      # 6 DOF
            "right_arm": action[:, 6:12],   # 6 DOF
            "left_hand": action[:, 12:18],  # 6 DOF
            "right_hand": action[:, 18:24], # 6 DOF
            "waist": action[:, 24:29],      # 5 DOF (or adjust as needed)
        }

        return result


class RealTimeInference:
    """Real-time inference with live camera."""

    def __init__(
        self,
        inference_handler: DCGr00TInference,
        camera_id: int = 0,
        display: bool = True,
    ):
        self.handler = inference_handler
        self.camera_id = camera_id
        self.display = display
        self.running = False

    def run(self, state_callback=None, action_callback=None):
        """
        Run real-time inference loop.

        Args:
            state_callback: Function that returns current robot state
            action_callback: Function that receives predicted actions
        """
        import cv2

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        print("Starting real-time inference. Press 'q' to quit.")
        self.running = True

        fps_counter = 0
        fps_start = time.time()
        fps = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get current state
            if state_callback:
                state = state_callback()
            else:
                state = np.zeros(29, dtype=np.float32)

            # Get action
            try:
                t0 = time.time()
                result = self.handler.get_action(frame_rgb, state)
                inference_time = (time.time() - t0) * 1000

                # Send action
                if action_callback:
                    action_callback(result)

                # Display
                if self.display:
                    # Draw info
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Show action values
                    action = result["action"][0]  # First timestep
                    cv2.putText(frame, f"L-arm: {action[:6].round(2)}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame, f"R-arm: {action[6:12].round(2)}", (10, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    cv2.imshow("DC-GR00T Inference", frame)

            except Exception as e:
                print(f"Inference error: {e}")

            # FPS calculation
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start = time.time()

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False


def run_on_test_video(
    handler: DCGr00TInference,
    test_video_path: str,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """Run inference on a test video and optionally save results."""
    import cv2

    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {test_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Test video: {total_frames} frames @ {fps} FPS")

    # Setup output video
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        state = np.zeros(29, dtype=np.float32)

        try:
            result = handler.get_action(frame_rgb, state)
            results.append({
                "frame_idx": frame_idx,
                "action": result["action"].tolist(),
            })

            # Draw on frame
            action = result["action"][0]
            cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if out:
                out.write(frame)

        except Exception as e:
            print(f"Frame {frame_idx}: {e}")

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames}")

    cap.release()
    if out:
        out.release()
        print(f"Output saved to {output_path}")

    return results


def start_server(
    handler: DCGr00TInference,
    port: int = 8080,
):
    """Start REST API server for inference."""
    from flask import Flask, request, jsonify
    import base64

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/set_demo", methods=["POST"])
    def set_demo():
        data = request.json
        demo_path = data.get("demo_path")
        demo_type = data.get("demo_type", "robot")

        if not demo_path:
            return jsonify({"error": "demo_path required"}), 400

        try:
            handler.set_demo(demo_path, demo_type)
            return jsonify({"status": "ok", "message": "Demo set successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/get_action", methods=["POST"])
    def get_action():
        data = request.json

        # Get observation (base64 encoded image or path)
        if "observation_base64" in data:
            import cv2
            img_data = base64.b64decode(data["observation_base64"])
            nparr = np.frombuffer(img_data, np.uint8)
            observation = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        elif "observation_path" in data:
            import cv2
            observation = cv2.imread(data["observation_path"])
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        else:
            return jsonify({"error": "observation required"}), 400

        # Get state
        state = np.array(data.get("state", [0] * 29), dtype=np.float32)

        try:
            result = handler.get_action(observation, state)
            return jsonify({
                "action": result["action"].tolist(),
                "left_arm": result["left_arm"].tolist(),
                "right_arm": result["right_arm"].tolist(),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)


def main():
    parser = argparse.ArgumentParser(description="DC-GR00T Inference")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on")

    # Demo
    parser.add_argument("--demo_video", type=str, default=None,
                       help="Demo video path")
    parser.add_argument("--demo_type", type=str, default="robot",
                       choices=["human", "robot", "cosmos", "own"],
                       help="Type of demo")

    # Input source
    parser.add_argument("--camera_id", type=int, default=0,
                       help="Camera device ID for live inference")
    parser.add_argument("--test_video", type=str, default=None,
                       help="Test video path (instead of live camera)")
    parser.add_argument("--output_video", type=str, default=None,
                       help="Output video path (for test mode)")

    # Server mode
    parser.add_argument("--server", action="store_true",
                       help="Start REST API server")
    parser.add_argument("--port", type=int, default=8080,
                       help="Server port")

    # Display
    parser.add_argument("--no_display", action="store_true",
                       help="Disable display")

    args = parser.parse_args()

    # Create inference handler
    handler = DCGr00TInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Set demo if provided
    if args.demo_video:
        handler.set_demo(args.demo_video, args.demo_type)

    # Run mode
    if args.server:
        start_server(handler, args.port)

    elif args.test_video:
        results = run_on_test_video(
            handler,
            args.test_video,
            args.output_video,
        )
        print(f"Processed {len(results)} frames")

        # Save results
        if args.output_video:
            results_path = args.output_video.replace(".mp4", "_actions.json")
            with open(results_path, "w") as f:
                json.dump(results, f)
            print(f"Actions saved to {results_path}")

    else:
        # Live camera
        if not args.demo_video:
            print("Warning: No demo video set. Use --demo_video to specify task.")

        rt = RealTimeInference(
            handler,
            camera_id=args.camera_id,
            display=not args.no_display,
        )
        rt.run()


if __name__ == "__main__":
    main()
