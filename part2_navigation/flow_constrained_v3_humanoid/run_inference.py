"""
FlowDiT V3 Humanoid — inference only.
Load checkpoint, load video (folder of frames), run model, output velocity + trajectory.

Usage:
  python run_inference.py --checkpoint checkpoints/flowdit_v3_humanoid_best.pt --video /path/to/episode_dir
  python run_inference.py --checkpoint checkpoints/flowdit_v3_humanoid_best.pt --video /path/to/frames_dir --instruction "humanoid robot pick and place"

Expects video path to be a directory containing frames (e.g. episode_dir/frames/*.jpg)
or an episode directory that has a frames/ subdir.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ImageNet normalization (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_NUM_FRAMES = 16
FRAME_SIZE = 224


def has_transformers() -> bool:
    """Check whether transformers is available in the current runtime."""
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def load_video_tensor(video_path: Path, num_frames: int = DEFAULT_NUM_FRAMES) -> torch.Tensor:
    """Load video from directory of frames -> [1, T, 3, H, W] float32, normalized."""
    video_path = Path(video_path)
    frames_dir = video_path / "frames" if (video_path / "frames").exists() else video_path
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(sorted(frames_dir.glob(ext)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No frames in {video_path}")
    frames_np = []
    for fp in files:
        img = Image.open(fp).convert("RGB")
        frames_np.append(np.array(img))
    frames_np = np.stack(frames_np)  # [T, H, W, 3]
    T_num = frames_np.shape[0]
    if T_num < num_frames:
        indices = list(range(T_num)) + [T_num - 1] * (num_frames - T_num)
        frames_np = frames_np[indices[:num_frames]]
    else:
        indices = np.linspace(0, T_num - 1, num_frames).astype(int)
        frames_np = frames_np[indices]
    transform = T.Compose([
        T.Resize((FRAME_SIZE, FRAME_SIZE)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    out = []
    for t in range(frames_np.shape[0]):
        x = torch.from_numpy(frames_np[t].astype(np.float32) / 255.0).permute(2, 0, 1)
        x = transform(x)
        out.append(x)
    video = torch.stack(out)  # [num_frames, 3, 224, 224]
    return video.unsqueeze(0)  # [1, T, 3, 224, 224]


def main():
    parser = argparse.ArgumentParser(description="FlowDiT V3 Humanoid inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/flowdit_v3_humanoid_best.pt")
    parser.add_argument("--video", type=str, required=True, help="Path to episode dir or frames dir")
    parser.add_argument("--instruction", type=str, default="humanoid robot navigate to goal")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from flow_constrained_v3 import create_flowdit_v3, FlowDiTv3Config

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config") or FlowDiTv3Config(
        variable_horizons=True, max_action_horizon=20, max_trajectory_horizon=40
    )
    if isinstance(config, dict):
        config = FlowDiTv3Config(**config)
    if hasattr(config, "variable_horizons") and not config.variable_horizons:
        config = FlowDiTv3Config(
            variable_horizons=True, max_action_horizon=20, max_trajectory_horizon=40
        )
    if hasattr(config, "use_language") and config.use_language and not has_transformers():
        print("transformers not found: disabling language encoder for fallback inference.")
        config.use_language = False

    model = create_flowdit_v3(config)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    video = load_video_tensor(args.video, num_frames=args.num_frames).to(device)
    prompt = [args.instruction]

    with torch.no_grad():
        actions, trajectory, _, _ = model(video, prompt, mode="inference")

    actions = actions.cpu().numpy()[0]   # [horizon, 4]  vx, vy, vz, yaw
    if trajectory is not None:
        trajectory = trajectory.cpu().numpy()[0]  # [horizon, 3]  x, y, z
    else:
        trajectory = None

    print("Velocity (vx, vy, vz, yaw) per step:")
    print(actions)
    if trajectory is not None:
        print("Trajectory (x, y, z) waypoints:")
        print(trajectory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
