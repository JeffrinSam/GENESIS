"""
Test inference with actual dataset samples.

Usage:
    python test_inference.py --checkpoint checkpoints/best.pth
"""

import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from inference import load_model, load_video, load_image, predict_actions


def test_with_dataset_sample(checkpoint_path: str, dataset_dir: str):
    """Test inference with a sample from the dataset."""

    print("\n" + "="*70)
    print("Testing Inference with Dataset Sample")
    print("="*70 + "\n")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)

    # Find a sample video
    dataset_path = Path(dataset_dir)
    video_files = sorted(list((dataset_path / "videos").glob("*.mp4")))

    if len(video_files) == 0:
        print("ERROR: No video files found in dataset")
        return

    sample_video_path = video_files[0]
    sample_name = sample_video_path.stem

    print(f"Using sample: {sample_name}")

    # Load video
    video = load_video(sample_video_path)

    # Extract goal (last 16 frames) and observation (random middle frame)
    goal_video = video[-16:]  # Last 16 frames
    obs_idx = len(video) // 2  # Middle frame
    current_obs = video[obs_idx]

    print(f"\nGoal video: {goal_video.shape} (last 16 frames)")
    print(f"Current observation: {current_obs.shape} (frame {obs_idx})")

    # Load ground truth actions
    action_path = dataset_path / "actions" / f"{sample_name}.npy"
    if action_path.exists():
        gt_actions = np.load(action_path)
        print(f"Ground truth actions: {gt_actions.shape}")
        print(f"GT actions at frame {obs_idx}:")
        print(f"  vx={gt_actions[obs_idx, 0]:.3f}, vy={gt_actions[obs_idx, 1]:.3f}, yaw={gt_actions[obs_idx, 2]:.3f}")

    # Predict actions
    predicted_actions = predict_actions(model, goal_video, current_obs)

    # Compare if GT available
    if action_path.exists():
        print("\nComparison (Predicted vs Ground Truth):")
        print(f"{'Step':<6} {'Pred vx':>9} {'GT vx':>9} {'Pred vy':>9} {'GT vy':>9} {'Pred yaw':>9} {'GT yaw':>9}")
        print("-" * 70)
        for i in range(min(8, len(gt_actions) - obs_idx)):
            pred = predicted_actions[i]
            gt = gt_actions[obs_idx + i]
            print(f"{i+1:<6} {pred[0]:>9.3f} {gt[0]:>9.3f} {pred[1]:>9.3f} {gt[1]:>9.3f} {pred[2]:>9.3f} {gt[2]:>9.3f}")

        # Compute MSE
        pred_chunk = predicted_actions[:min(8, len(gt_actions) - obs_idx)]
        gt_chunk = gt_actions[obs_idx:obs_idx + len(pred_chunk)]
        mse = np.mean((pred_chunk - gt_chunk) ** 2)
        print(f"\nMSE: {mse:.6f}")

    print("\n" + "="*70)
    print("✓ Inference test complete!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test FlowDiT V2 Inference")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--dataset_dir", type=str,
                        default="./data/sample",
                        help="Path to dataset directory")

    args = parser.parse_args()

    test_with_dataset_sample(args.checkpoint, args.dataset_dir)


if __name__ == "__main__":
    main()
