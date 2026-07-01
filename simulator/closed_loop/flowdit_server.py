#!/usr/bin/env python3
"""
FlowDiT V2+ TCP server for closed-loop Isaac Sim integration.

Runs in the FlowDiT venv. Accepts POV frames via TCP, returns velocity commands.

Protocol (length-prefixed JSON + binary):
  Client → Server:
    WARMUP:  4-byte len + JSON {"cmd":"warmup","video_path":"...","prompt":"..."}
    STEP:    4-byte len + JSON {"cmd":"step","width":224,"height":224,"channels":3}
             + raw frame bytes (H*W*C uint8)
    STOP:    4-byte len + JSON {"cmd":"stop"}

  Server → Client:
    4-byte len + JSON response

Usage:
    python flowdit_server.py --port 5555 --checkpoint /path/to/best.pth
"""

import argparse
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add FlowDiT to path
FLOWDIT_DIR = os.getenv(
    "FLOWDIT_DIR",
    str(Path(__file__).resolve().parents[2] / "part2_navigation" / "flow_constrained_v2")
)
sys.path.insert(0, FLOWDIT_DIR)

from models.flowdit_v2_plus import create_flowdit_v2_plus


def recv_exact(sock, n):
    """Receive exactly n bytes."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf += chunk
    return buf


def recv_message(sock):
    """Receive length-prefixed message."""
    raw_len = recv_exact(sock, 4)
    msg_len = struct.unpack(">I", raw_len)[0]
    return recv_exact(sock, msg_len)


def send_message(sock, data):
    """Send length-prefixed message."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


class FlowDiTServer:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self.model = None
        self.cache = None

        print(f"Loading checkpoint: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            config = ckpt.get("config", {})
        else:
            state_dict = ckpt
            config = {}

        if isinstance(config, dict) and "use_raft" in config:
            use_raft = bool(config["use_raft"])
        else:
            raft_tokens = ("flow_encoder._raft", "flow_encoder.pool", "flow_encoder.fc")
            use_raft = any(any(tok in k for tok in raft_tokens) for k in state_dict.keys())

        print(f"Creating model (use_raft={use_raft})...", flush=True)
        self.model = create_flowdit_v2_plus(device=device, use_raft=use_raft)
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("Model loaded.", flush=True)

    def handle_warmup(self, msg):
        """Load goal video and warmup the model."""
        video_path = msg["video_path"]
        prompt = msg.get("prompt", None)
        requested_fps = msg.get("fps", 16)

        print(f"Warmup: {video_path}", flush=True)

        cap = cv2.VideoCapture(video_path)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()

        if not frames:
            return {"status": "error", "message": "No frames in video"}

        video = np.stack(frames, axis=0)

        # Stride subsampling
        stride = 1
        if source_fps > requested_fps * 1.5:
            stride = max(1, int(round(source_fps / requested_fps)))
        video = video[::stride]

        print(f"  Video: {len(frames)} frames @ {source_fps} fps, "
              f"stride={stride} → {video.shape[0]} frames", flush=True)

        t0 = time.time()
        self.cache = self.model.warmup_realtime(video, prompt=prompt)
        warmup_ms = (time.time() - t0) * 1000

        # Store video for reference
        self.goal_video = video

        print(f"  Warmup done: {warmup_ms:.0f}ms", flush=True)
        return {"status": "ready", "warmup_ms": warmup_ms,
                "n_frames": video.shape[0]}

    def handle_step(self, msg, frame_data):
        """Process one observation frame and return velocity command."""
        if self.cache is None:
            return {"status": "error", "message": "Not warmed up"}

        h, w, c = msg["height"], msg["width"], msg["channels"]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(h, w, c)
        # Convert to float32 [0,1]
        frame = frame.astype(np.float32) / 255.0

        t0 = time.time()
        command, actions_horizon, self.cache, diagnostics = self.model.predict_realtime(
            goal_video=self.goal_video,
            current_obs=frame,
            goal_features_cache=self.cache,
            return_info=True,
            stop_speed_threshold=0.05,
            stop_yaw_threshold=0.08,
            stop_consecutive_steps=3,
            stop_confidence_threshold=0.15,
            min_steps_before_stop=3,
            smoothing_alpha=0.65,
            horizon_decay=0.65,
            num_action_samples=3,
            max_vx=1.0,
            max_vy=1.0,
            max_yaw_rate=1.0,
        )
        step_ms = (time.time() - t0) * 1000

        # Convert numpy to Python types for JSON
        result = {
            "status": "ok",
            "command": [float(command[0]), float(command[1]), float(command[2])],
            "should_stop": bool(diagnostics.get("should_stop", False)),
            "step_ms": step_ms,
            "diagnostics": {
                "confidence": float(diagnostics.get("confidence", 0)),
                "visual_similarity": float(diagnostics.get("visual_similarity", 0)),
                "arrival_score": float(diagnostics.get("arrival_score", 0)),
                "translational_speed": float(diagnostics.get("translational_speed", 0)),
                "yaw_speed": float(diagnostics.get("yaw_speed", 0)),
                "stop_counter": int(diagnostics.get("stop_counter", 0)),
                "stop_reason": str(diagnostics.get("stop_reason", "continue")),
            },
        }
        return result

    def serve(self, port):
        """Run TCP server."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(1)
        print(f"FlowDiT server listening on port {port}", flush=True)
        print("FLOWDIT_SERVER_READY", flush=True)

        conn, addr = server.accept()
        print(f"Client connected: {addr}", flush=True)

        try:
            while True:
                raw = recv_message(conn)
                msg = json.loads(raw.decode("utf-8"))
                cmd = msg.get("cmd", "")

                if cmd == "warmup":
                    resp = self.handle_warmup(msg)
                    send_message(conn, json.dumps(resp))

                elif cmd == "step":
                    # Read frame data after JSON header
                    n_bytes = msg["height"] * msg["width"] * msg["channels"]
                    frame_data = recv_exact(conn, n_bytes)
                    resp = self.handle_step(msg, frame_data)
                    send_message(conn, json.dumps(resp))

                elif cmd == "stop":
                    send_message(conn, json.dumps({"status": "stopped"}))
                    break
                else:
                    send_message(conn, json.dumps(
                        {"status": "error", "message": f"Unknown cmd: {cmd}"}))
        except (ConnectionError, BrokenPipeError):
            print("Client disconnected.", flush=True)
        finally:
            conn.close()
            server.close()
            print("Server shut down.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    srv = FlowDiTServer(args.checkpoint, device=args.device)
    srv.serve(args.port)


if __name__ == "__main__":
    main()
