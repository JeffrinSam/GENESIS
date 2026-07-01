"""
FlowDiT V2 - Goal-Conditioned Navigation Inference
===================================================

Usage:
    python inference.py --checkpoint checkpoints/best.pth \
                        --goal_video path/to/reference.mp4 \
                        --current_obs path/to/current_frame.jpg \
                        --output actions.npy

Author: Jeffrin Sam
Date: January 2026
"""

import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from models.flowdit_production import FlowDiTProduction, FlowDiTConfig, create_flowdit_production


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained FlowDiT model from checkpoint."""
    print(f"\n{'='*70}")
    print("Loading FlowDiT V2 Goal-Conditioned Navigation Model")
    print(f"{'='*70}")

    # Create model
    model = create_flowdit_production(device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded checkpoint from: {checkpoint_path}")
    print(f"✓ Trained for {checkpoint['epoch']} epochs")
    print(f"✓ Best validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else "")
    print(f"{'='*70}\n")

    return model


def load_video(video_path: str, target_size=(224, 224), max_frames=None):
    """
    Load video from file.

    Args:
        video_path: Path to video file (mp4, avi, etc.)
        target_size: Resize frames to this size
        max_frames: Maximum number of frames to load (None = all)

    Returns:
        video: numpy array [T, H, W, 3] in range [0, 1]
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, target_size)

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")

    video = np.stack(frames, axis=0)  # [T, H, W, 3]
    print(f"✓ Loaded video: {video.shape[0]} frames from {video_path}")

    return video


def load_image(image_path: str, target_size=(224, 224)):
    """
    Load image from file.

    Args:
        image_path: Path to image file (jpg, png, etc.)
        target_size: Resize to this size

    Returns:
        image: numpy array [H, W, 3] in range [0, 1]
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, target_size)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    print(f"✓ Loaded observation: {image.shape} from {image_path}")

    return image


def select_current_obs_from_video(
    video: np.ndarray,
    mode: str = "middle",
    index: int | None = None,
) -> np.ndarray:
    """
    Pick a single frame from a goal/reference video to serve as current observation.

    Args:
        video: [T, H, W, 3] float32 in [0, 1]
        mode: one of {"first", "middle", "last", "index"}
        index: used when mode == "index"

    Returns:
        current_obs: [H, W, 3] float32 in [0, 1]
    """
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected video shape [T,H,W,3], got {video.shape}")
    T = video.shape[0]
    if T == 0:
        raise ValueError("Video has 0 frames")

    mode = (mode or "middle").lower()
    if mode == "first":
        idx = 0
    elif mode == "middle":
        idx = T // 2
    elif mode == "last":
        idx = T - 1
    elif mode == "index":
        if index is None:
            raise ValueError("mode='index' requires --current_obs_index")
        if index < 0:
            idx = T + index
        else:
            idx = index
        if idx < 0 or idx >= T:
            raise ValueError(f"current_obs_index out of range: {index} for T={T}")
    else:
        raise ValueError(f"Unknown current_obs_from_video mode: {mode}")

    obs = video[idx]
    print(f"✓ Using current_obs from goal_video frame {idx}/{T-1} (mode={mode})")
    return obs


def integrate_trajectory(actions: np.ndarray, dt: float = 1.0 / 16.0) -> np.ndarray:
    """
    Integrate [vx, vy, yaw_rate] into a simple planar trajectory.

    Args:
        actions: [N, 3] velocities in robot frame
        dt: timestep seconds

    Returns:
        traj: [N+1, 3] of [x, y, theta]
    """
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"Expected actions shape [N,3], got {actions.shape}")

    N = actions.shape[0]
    traj = np.zeros((N + 1, 3), dtype=np.float32)
    x, y, theta = 0.0, 0.0, 0.0
    traj[0] = [x, y, theta]

    for i in range(N):
        vx, vy, yaw_rate = actions[i]
        # world-frame velocity
        vx_w = float(vx) * np.cos(theta) - float(vy) * np.sin(theta)
        vy_w = float(vx) * np.sin(theta) + float(vy) * np.cos(theta)

        x += vx_w * dt
        y += vy_w * dt
        theta += float(yaw_rate) * dt
        traj[i + 1] = [x, y, theta]

    return traj


def predict_actions(
    model: FlowDiTProduction,
    goal_video: np.ndarray,
    current_obs: np.ndarray,
    prompt: str = None
):
    """
    Predict actions given goal video and current observation.

    Args:
        model: Trained FlowDiT model
        goal_video: Reference video [T, H, W, 3]
        current_obs: Current observation [H, W, 3]
        prompt: Optional language prompt (not used yet)

    Returns:
        actions: Predicted actions [action_horizon, 3] = [vx, vy, yaw_rate]
    """
    print(f"\n{'='*70}")
    print("Predicting Actions")
    print(f"{'='*70}")

    # Run inference
    actions = model.predict(goal_video, current_obs, prompt)

    print(f"✓ Predicted {len(actions)} action steps")
    print(f"\nAction sequence:")
    print(f"{'Step':<6} {'vx':>8} {'vy':>8} {'yaw':>8}")
    print("-" * 32)
    for i, (vx, vy, yaw) in enumerate(actions):
        print(f"{i+1:<6} {vx:>8.3f} {vy:>8.3f} {yaw:>8.3f}")

    print(f"{'='*70}\n")

    return actions


def main():
    parser = argparse.ArgumentParser(description="FlowDiT V2 Inference")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    # Inputs
    parser.add_argument("--goal_video", type=str, required=True,
                        help="Path to goal/reference video (from video gen model)")
    parser.add_argument("--current_obs", type=str, default=None,
                        help="Path to current observation image. If omitted, a frame will be taken from --goal_video.")
    parser.add_argument("--current_obs_from_video", type=str, default="middle",
                        choices=["first", "middle", "last", "index"],
                        help="If --current_obs is omitted, choose which frame of --goal_video to use.")
    parser.add_argument("--current_obs_index", type=int, default=None,
                        help="Frame index to use when --current_obs_from_video=index (supports negative indexing).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Optional language prompt (e.g., 'go to the table')")

    # Output
    parser.add_argument("--output", type=str, default="actions.npy",
                        help="Output file to save actions (.npy)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the inputs and outputs")
    parser.add_argument("--viz_prefix", type=str, default="visualization",
                        help="Prefix for visualization files (without extension)")
    parser.add_argument("--dt", type=float, default=1.0 / 16.0,
                        help="Timestep (seconds) for integrating actions into a trajectory")

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load inputs
    goal_video = load_video(args.goal_video)
    if args.current_obs is not None:
        current_obs = load_image(args.current_obs)
    else:
        current_obs = select_current_obs_from_video(
            goal_video, mode=args.current_obs_from_video, index=args.current_obs_index
        )

    # Predict actions
    actions = predict_actions(model, goal_video, current_obs, args.prompt)

    # Save actions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, actions)
    print(f"✓ Saved actions to: {output_path}")

    # Visualize if requested
    if args.visualize:
        prefix = output_path.parent / args.viz_prefix
        visualize_inference(goal_video, current_obs, actions, prefix, dt=args.dt)


def visualize_inference(goal_video, current_obs, actions, output_prefix: Path, dt: float = 1.0 / 16.0):
    """
    Create two detailed visualizations:
      1) inputs + action curves
      2) integrated trajectory + velocity vectors + heading
    """
    import matplotlib.pyplot as plt

    # --- Viz 1: Inputs + action curves
    fig1, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Goal frames (4 snapshots)
    goal_indices = [0, len(goal_video) // 3, (2 * len(goal_video)) // 3, -1]
    for i, idx in enumerate(goal_indices):
        axes[0, i].imshow(goal_video[idx])
        title_idx = idx if idx >= 0 else (len(goal_video) - 1)
        axes[0, i].set_title(f"Goal Frame {title_idx}", fontsize=11)
        axes[0, i].axis("off")

    # Current obs
    axes[1, 0].imshow(current_obs)
    axes[1, 0].set_title("Current Observation (derived)" if current_obs is not None else "Current Observation", fontsize=11)
    axes[1, 0].axis("off")

    steps = np.arange(len(actions))
    tsec = steps * dt

    # Linear velocities
    axes[1, 1].plot(tsec, actions[:, 0], "r-o", label="vx (forward)", alpha=0.85)
    axes[1, 1].plot(tsec, actions[:, 1], "g-s", label="vy (lateral)", alpha=0.85)
    axes[1, 1].axhline(0, color="k", linestyle="--", alpha=0.25)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("m/s")
    axes[1, 1].set_title("Linear Velocities")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Angular velocity
    axes[1, 2].plot(tsec, actions[:, 2], "b-^", label="yaw_rate", alpha=0.85)
    axes[1, 2].axhline(0, color="k", linestyle="--", alpha=0.25)
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("rad/s")
    axes[1, 2].set_title("Angular Velocity")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    # Speed magnitude
    speed = np.linalg.norm(actions[:, :2], axis=1)
    axes[1, 3].plot(tsec, speed, "m-o", label="||v||", alpha=0.85)
    axes[1, 3].set_xlabel("Time (s)")
    axes[1, 3].set_ylabel("m/s")
    axes[1, 3].set_title("Speed Magnitude")
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].legend()

    plt.tight_layout()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    out1 = Path(f"{output_prefix}_inputs_actions.jpg")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to: {out1}")
    plt.close(fig1)

    # --- Viz 2: Integrated trajectory + velocity vectors + heading
    traj = integrate_trajectory(actions, dt=dt)  # [N+1, 3]
    x, y, theta = traj[:, 0], traj[:, 1], traj[:, 2]

    fig2 = plt.figure(figsize=(18, 12))

    # (1) XY trajectory with velocity vectors
    ax_xy = fig2.add_subplot(221)
    ax_xy.plot(x, y, "b-", linewidth=2, alpha=0.7, label="Integrated trajectory")
    ax_xy.scatter([x[0]], [y[0]], c="green", s=180, marker="o", edgecolors="black", linewidth=1.5, label="Start", zorder=5)
    ax_xy.scatter([x[-1]], [y[-1]], c="red", s=180, marker="s", edgecolors="black", linewidth=1.5, label="End", zorder=5)

    # velocity vectors (actions are N; pose points are N+1 -> use pose[0:N])
    N = actions.shape[0]
    step = max(1, N // 8)  # show up to ~8 arrows
    for i in range(0, N, step):
        vx, vy, _ = actions[i]
        th = theta[i]
        vx_w = float(vx) * np.cos(th) - float(vy) * np.sin(th)
        vy_w = float(vx) * np.sin(th) + float(vy) * np.cos(th)
        scale = 0.5
        ax_xy.arrow(
            x[i], y[i], vx_w * scale, vy_w * scale,
            head_width=0.03, head_length=0.05, fc="orange", ec="orange",
            alpha=0.8, linewidth=1.5, length_includes_head=True
        )

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-down Trajectory + Velocity Vectors")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis("equal")
    ax_xy.legend()

    # (2) Heading over time
    ax_head = fig2.add_subplot(222)
    t_pose = np.arange(traj.shape[0]) * dt
    ax_head.plot(t_pose, theta, "c-o", alpha=0.85)
    ax_head.set_xlabel("Time (s)")
    ax_head.set_ylabel("theta (rad)")
    ax_head.set_title("Integrated Heading")
    ax_head.grid(True, alpha=0.3)

    # (3) Per-step action table-like plot (stem)
    ax_stem = fig2.add_subplot(223)
    ax_stem.stem(steps, actions[:, 0], linefmt="r-", markerfmt="ro", basefmt="k-", label="vx")
    ax_stem.stem(steps + 0.05, actions[:, 1], linefmt="g-", markerfmt="gs", basefmt="k-", label="vy")
    ax_stem.stem(steps + 0.10, actions[:, 2], linefmt="b-", markerfmt="b^", basefmt="k-", label="yaw")
    ax_stem.axhline(0, color="k", linestyle="--", alpha=0.25)
    ax_stem.set_xlabel("Step")
    ax_stem.set_ylabel("Value")
    ax_stem.set_title("Actions per Step (horizon)")
    ax_stem.grid(True, alpha=0.25)
    ax_stem.legend()

    # (4) Trajectory in X vs time and Y vs time
    ax_xt = fig2.add_subplot(224)
    ax_xt.plot(t_pose, x, "b-o", alpha=0.85, label="x(t)")
    ax_xt.plot(t_pose, y, "m-s", alpha=0.85, label="y(t)")
    ax_xt.set_xlabel("Time (s)")
    ax_xt.set_ylabel("Position (m)")
    ax_xt.set_title("Integrated Position vs Time")
    ax_xt.grid(True, alpha=0.3)
    ax_xt.legend()

    plt.tight_layout()
    out2 = Path(f"{output_prefix}_trajectory.jpg")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"✓ Saved trajectory visualization to: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
