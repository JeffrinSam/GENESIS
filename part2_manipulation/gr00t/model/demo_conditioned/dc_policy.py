"""
DC Policy: Demo-Conditioned GR00T Inference Pipeline

Provides easy-to-use interface for:
1. Loading demo video and extracting task embedding (offline)
2. Running closed-loop control using robot's own observation (online)

Usage:
    policy = DCPolicy.from_pretrained("path/to/checkpoint")

    # Show robot a demo video
    policy.set_demo("path/to/demo.mp4", demo_type="human")

    # Run closed-loop control
    while not done:
        action = policy.get_action(robot_observation, robot_state)
        robot.execute(action)
        robot_observation = robot.get_observation()
"""

from pathlib import Path
from typing import Dict, Optional, Union, List
import numpy as np
import torch

from .dc_gr00t import DCGr00t, DCGr00tConfig


class DCPolicy:
    """
    Demo-Conditioned Policy for real-world deployment.

    The robot watches a demo to understand the task, then executes
    using its own sensors with closed-loop control.
    """

    def __init__(
        self,
        model: DCGr00t,
        device: str = "cuda:0",
        num_demo_keyframes: int = 16,
    ):
        """
        Args:
            model: DCGr00t model
            device: Device for inference
            num_demo_keyframes: Number of keyframes to sample from demo
        """
        self.model = model.to(device).eval()
        self.device = device
        self.num_demo_keyframes = num_demo_keyframes

        # Cached task embedding from demo
        self._task_embedding: Optional[torch.Tensor] = None
        self._demo_path: Optional[str] = None
        self._demo_type: Optional[str] = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda:0",
        **kwargs,
    ) -> "DCPolicy":
        """Load policy from pretrained checkpoint."""
        model = DCGr00t.from_pretrained(model_path, **kwargs)
        return cls(model, device=device)

    @classmethod
    def from_groot(
        cls,
        groot_path: str,
        device: str = "cuda:0",
        config: Optional[DCGr00tConfig] = None,
        **kwargs,
    ) -> "DCPolicy":
        """Create DC policy from pretrained GR00T model."""
        model = DCGr00t.from_pretrained_groot(groot_path, config=config, **kwargs)
        return cls(model, device=device)

    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load video from file.

        Args:
            video_path: Path to video file

        Returns:
            [T, H, W, C] video frames as uint8
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Try torchcodec
        try:
            import torchcodec
            frames = torchcodec.decoders.VideoDecoder(str(video_path))
            return np.stack([f.numpy() for f in frames], axis=0)
        except ImportError:
            pass

        # Try decord
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(video_path), ctx=cpu(0))
            return vr.get_batch(list(range(len(vr)))).asnumpy()
        except ImportError:
            pass

        # Fallback to opencv
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames, axis=0)

    def sample_keyframes(
        self,
        video: np.ndarray,
        num_frames: Optional[int] = None,
    ) -> np.ndarray:
        """Sample keyframes uniformly from video."""
        num_frames = num_frames or self.num_demo_keyframes
        T = len(video)
        indices = np.linspace(0, T - 1, num_frames, dtype=np.int64)
        return video[indices]

    def set_demo(
        self,
        demo: Union[str, np.ndarray, torch.Tensor],
        demo_type: str = "robot",
    ) -> torch.Tensor:
        """
        Set demonstration video and compute task embedding.

        Args:
            demo: Path to demo video, or video array [T, H, W, C]
            demo_type: "human", "robot", "cosmos", or "own"

        Returns:
            Task embedding [1, num_tokens, d_model]
        """
        # Load video if path
        if isinstance(demo, str):
            self._demo_path = demo
            demo = self.load_video(demo)

        # Convert to tensor
        if isinstance(demo, np.ndarray):
            demo = torch.from_numpy(demo)

        # Sample keyframes
        if demo.shape[0] > self.num_demo_keyframes:
            keyframes = self.sample_keyframes(demo.numpy(), self.num_demo_keyframes)
            demo = torch.from_numpy(keyframes)

        # Add batch dimension
        if demo.dim() == 4:
            demo = demo.unsqueeze(0)  # [1, T, H, W, C]

        # Get demo type index
        type_map = {"human": 0, "robot": 1, "cosmos": 2, "own": 3}
        demo_type_idx = torch.tensor([type_map.get(demo_type, 1)], device=self.device)

        # Encode demo
        demo = demo.to(self.device)
        with torch.no_grad():
            self._task_embedding = self.model.encode_demo(demo, demo_type_idx)

        self._demo_type = demo_type
        print(f"Task embedding computed from {demo_type} demo")
        print(f"  Shape: {self._task_embedding.shape}")

        return self._task_embedding

    def set_task_embedding(self, task_embedding: torch.Tensor):
        """Directly set task embedding (for precomputed embeddings)."""
        self._task_embedding = task_embedding.to(self.device)

    @property
    def has_demo(self) -> bool:
        """Check if demo has been set."""
        return self._task_embedding is not None

    @torch.no_grad()
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        state: Optional[Dict[str, np.ndarray]] = None,
        language: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get action from current observation conditioned on demo.

        Args:
            observation: Robot's current camera observation
                {"ego_view": [H, W, C] or [T, H, W, C]}
            state: Robot's proprioceptive state (optional)
                {"joints": [D]} etc.
            language: Optional language instruction

        Returns:
            Action dictionary with predicted actions
        """
        if not self.has_demo:
            raise RuntimeError("No demo set! Call set_demo() first.")

        # Prepare observation
        obs_dict = {}

        # Process video observation
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if value.dim() == 3:
                value = value.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, C]
            elif value.dim() == 4:
                value = value.unsqueeze(0)  # [1, T, H, W, C]
            obs_dict[key] = value.to(self.device)

        # Process state
        if state is not None:
            state_tensors = {}
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                if value.dim() == 1:
                    value = value.unsqueeze(0)  # [1, D]
                state_tensors[key] = value.to(self.device, dtype=torch.float32)
            obs_dict["state"] = state_tensors

        # Add language if provided
        if language is not None:
            obs_dict["language"] = language

        # Add embodiment ID (use 0 for G1_DC)
        obs_dict["embodiment_id"] = torch.tensor([0], device=self.device)

        # Get action
        outputs = self.model.get_action(obs_dict, self._task_embedding)
        action_pred = outputs["action_pred"].cpu().numpy()[0]  # [horizon, action_dim]

        return {"action": action_pred}

    def reset(self):
        """Reset policy state (clear cached task embedding)."""
        self._task_embedding = None
        self._demo_path = None
        self._demo_type = None


class DCPolicyServer:
    """
    REST API server for DC-GR00T inference.

    Endpoints:
    - POST /set_demo: Set demonstration video
    - POST /get_action: Get action from observation
    - GET /health: Health check
    """

    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        device: str = "cuda:0",
    ):
        self.policy = DCPolicy.from_pretrained(model_path, device=device)
        self.host = host
        self.port = port

    def run(self):
        """Start the server."""
        try:
            from flask import Flask, request, jsonify
            import tempfile
            import os
        except ImportError:
            raise ImportError("Flask required. Install with: pip install flask")

        app = Flask(__name__)

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "healthy",
                "has_demo": self.policy.has_demo,
            })

        @app.route("/set_demo", methods=["POST"])
        def set_demo():
            try:
                demo_type = request.form.get("demo_type", "robot")

                if "video" not in request.files:
                    return jsonify({"error": "No video file"}), 400

                video_file = request.files["video"]

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    video_path = f.name
                    video_file.save(video_path)

                try:
                    self.policy.set_demo(video_path, demo_type)
                    return jsonify({
                        "status": "success",
                        "demo_type": demo_type,
                        "embedding_shape": list(self.policy._task_embedding.shape),
                    })
                finally:
                    os.unlink(video_path)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route("/get_action", methods=["POST"])
        def get_action():
            try:
                if not self.policy.has_demo:
                    return jsonify({"error": "No demo set"}), 400

                data = request.get_json()

                # Parse observation
                observation = {}
                if "ego_view" in data:
                    observation["ego_view"] = np.array(data["ego_view"], dtype=np.uint8)

                # Parse state
                state = None
                if "state" in data:
                    state = {k: np.array(v, dtype=np.float32) for k, v in data["state"].items()}

                # Get action
                result = self.policy.get_action(observation, state)

                return jsonify({
                    "action": result["action"].tolist(),
                })

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        print(f"Starting DC Policy Server on {self.host}:{self.port}")
        app.run(host=self.host, port=self.port)


def main():
    """CLI for DC Policy."""
    import argparse

    parser = argparse.ArgumentParser(description="DC-GR00T Policy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Demo command - extract task embedding
    demo_parser = subparsers.add_parser("encode_demo", help="Encode demo to task embedding")
    demo_parser.add_argument("--model", required=True, help="Model path")
    demo_parser.add_argument("--video", required=True, help="Demo video path")
    demo_parser.add_argument("--demo_type", default="robot", help="Demo type")
    demo_parser.add_argument("--output", default="task_embedding.pt", help="Output file")
    demo_parser.add_argument("--device", default="cuda:0", help="Device")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start REST API server")
    server_parser.add_argument("--model", required=True, help="Model path")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host")
    server_parser.add_argument("--port", type=int, default=8000, help="Port")
    server_parser.add_argument("--device", default="cuda:0", help="Device")

    args = parser.parse_args()

    if args.command == "encode_demo":
        policy = DCPolicy.from_pretrained(args.model, device=args.device)
        task_embedding = policy.set_demo(args.video, args.demo_type)
        torch.save(task_embedding, args.output)
        print(f"Task embedding saved to {args.output}")

    elif args.command == "server":
        server = DCPolicyServer(
            args.model,
            host=args.host,
            port=args.port,
            device=args.device,
        )
        server.run()


if __name__ == "__main__":
    main()
