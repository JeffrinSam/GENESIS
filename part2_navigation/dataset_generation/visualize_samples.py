"""
Flow-Constrained Video-to-Action Model for Cross-Embodiment Navigation

Author: Jeffrin Sam
Institution: Skoltech
Year: 2025
License: MIT

Description: Visualize dataset samples and statistics
"""

import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


def visualize_sample(video_path: Path, action_path: Path, clip_meta: dict):
    """Visualize a single clip with actions."""
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    # Load actions
    actions = np.load(action_path)

    # Select 4 frames to display
    num_frames = len(frames)
    indices = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    action_names = ['vx', 'vy', 'yaw'] if actions.shape[1] == 3 else ['vx', 'vy', 'vz', 'yaw']

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(frames[idx])
        ax.axis('off')

        # Add action text
        action_text = f"Frame {idx}/{num_frames-1}\n"
        for j, name in enumerate(action_names):
            action_text += f"{name}: {actions[idx, j]:.2f} "
            if j == 1:
                action_text += "\n"

        ax.set_title(action_text, fontsize=10, family='monospace')

    # Add metadata
    clip_id = video_path.stem
    fig.suptitle(
        f"Clip: {clip_id} | Embodiment: {clip_meta.get('embodiment', 'N/A')} | "
        f"Duration: {clip_meta.get('duration', 0):.1f}s @ {clip_meta.get('fps', 0)}fps",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()


def visualize_action_distribution(dataset_dir: Path):
    """Plot action distribution across dataset."""
    # Load metadata
    with open(dataset_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Collect all actions
    all_actions = []
    action_dir = dataset_dir / "actions"

    for clip_id in metadata.keys():
        action_path = action_dir / f"{clip_id}.npy"
        if action_path.exists():
            actions = np.load(action_path)
            all_actions.append(actions)

    all_actions = np.concatenate(all_actions, axis=0)  # [N, action_dim]

    # Plot distributions
    action_dim = all_actions.shape[1]
    action_names = ['vx (m/s)', 'vy (m/s)', 'yaw (rad/s)'] if action_dim == 3 else \
                   ['vx (m/s)', 'vy (m/s)', 'vz (m/s)', 'yaw (rad/s)']

    fig, axes = plt.subplots(1, action_dim, figsize=(4 * action_dim, 4))

    if action_dim == 1:
        axes = [axes]

    for i in range(action_dim):
        axes[i].hist(all_actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(action_names[i])
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution of {action_names[i]}')
        axes[i].grid(True, alpha=0.3)

        # Add statistics
        mean = np.mean(all_actions[:, i])
        std = np.std(all_actions[:, i])
        axes[i].axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--dataset_dir", type=str, default="../dataset",
                        help="Path to dataset directory")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--show_distribution", action="store_true",
                        help="Show action distribution")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    video_dir = dataset_path / "videos"
    action_dir = dataset_path / "actions"

    # Load metadata
    with open(dataset_path / "metadata.json", 'r') as f:
        metadata = json.load(f)

    # Sample random clips
    clip_ids = list(metadata.keys())
    sampled_ids = random.sample(clip_ids, min(args.num_samples, len(clip_ids)))

    print(f"Visualizing {len(sampled_ids)} samples...\n")

    for clip_id in sampled_ids:
        video_path = video_dir / f"{clip_id}.mp4"
        action_path = action_dir / f"{clip_id}.npy"

        if video_path.exists() and action_path.exists():
            print(f"Showing: {clip_id}")
            visualize_sample(video_path, action_path, metadata[clip_id])
        else:
            print(f"Skipping {clip_id} (missing files)")

    if args.show_distribution:
        print("\nShowing action distribution...")
        visualize_action_distribution(dataset_path)


if __name__ == "__main__":
    main()
