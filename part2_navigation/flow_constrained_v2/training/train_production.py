"""
Production Training Script for FlowDiT
=======================================

Ready-to-run training script with:
1. RECON dataset loader
2. Diffusion loss
3. Proper logging
4. Checkpointing
5. Validation

Author: Jeffrin Sam
Date: January 2026
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.flowdit_production import FlowDiTProduction, FlowDiTConfig


# ============================================================================
# DATASET LOADER
# ============================================================================

class NavigationDataset(Dataset):
    """
    Dataset for goal-conditioned navigation.

    Expected directory structure:
    dataset/
        videos/
            00000.mp4 or 00000.npy
            00001.mp4
            ...
        actions/
            00000.npy  # Shape: [T, 3] where 3 = [vx, vy, yaw]
            00001.npy
            ...
        metadata.json

    Training strategy:
        - goal_video: Full trajectory or frames near the end (showing destination)
        - current_obs: Random frame sampled from trajectory
        - actions: Next 8 actions from that observation point
    """

    def __init__(
        self,
        dataset_dir: str,
        action_horizon: int = 8,
        video_length: int = 80,  # Number of frames
        goal_frames: int = 16,   # Number of frames to use as goal
        split: str = "train"
    ):
        self.dataset_dir = Path(dataset_dir)
        self.action_horizon = action_horizon
        self.video_length = video_length
        self.goal_frames = goal_frames
        self.split = split

        # Load metadata
        metadata_path = self.dataset_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

                # Check if metadata has splits structure
                if 'splits' in self.metadata and isinstance(self.metadata['splits'], dict):
                    if split in self.metadata['splits']:
                        self.indices = self.metadata['splits'][split]
                    else:
                        print(f"WARNING: No split '{split}' in metadata")
                        self.indices = None
                else:
                    # Metadata is dict of filenames -> create splits
                    all_files = list(self.metadata.keys())
                    print(f"Creating {split} split from {len(all_files)} total samples")
                    self.indices = None
        else:
            print(f"WARNING: No metadata.json found")
            self.indices = None
            self.metadata = {}

        # Get list of video files
        video_dir = self.dataset_dir / "videos"
        if self.indices is None:
            # Use all videos and create train/val split
            video_files = sorted(list(video_dir.glob("*.mp4")))
            all_indices = [f.stem for f in video_files]

            # Create 80/20 train/val split
            np.random.seed(42)  # Reproducible split
            num_train = int(0.8 * len(all_indices))
            shuffled_indices = np.random.permutation(all_indices)

            if split == "train":
                self.indices = shuffled_indices[:num_train].tolist()
            elif split == "val":
                self.indices = shuffled_indices[num_train:].tolist()
            else:
                self.indices = all_indices

        print(f"Loaded {len(self.indices)} {split} samples from {dataset_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Load goal video, current observation, and action sequence.

        Returns:
            dict with:
                'goal': [goal_frames, 3, H, W] tensor - reference video
                'current_obs': [3, H, W] tensor - current observation
                'actions': [action_horizon, 3] tensor
                'index': int
        """
        sample_idx = self.indices[idx]

        # Load video (try mp4 first, then npy)
        video_path = self.dataset_dir / "videos" / f"{sample_idx}.mp4"
        if not video_path.exists():
            video_path = self.dataset_dir / "videos" / f"{sample_idx}.npy"

        if video_path.suffix == '.npy':
            video = np.load(video_path)  # [T, H, W, 3] or [T, 3, H, W]
        else:
            # Load video with cv2 or decord
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            video = np.stack(frames)  # [T, H, W, 3]

        # Convert to torch
        video = torch.from_numpy(video).float()

        # Ensure format is [T, 3, H, W]
        if video.shape[-1] == 3:
            video = video.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        if video.max() > 1.0:
            video = video / 255.0

        # Resize if needed
        if video.shape[-2:] != (224, 224):
            video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)

        T = video.shape[0]

        # Load actions
        action_path = self.dataset_dir / "actions" / f"{sample_idx}.npy"
        actions = np.load(action_path)  # [T, 3]
        actions = torch.from_numpy(actions).float()

        # Sample current observation (random point in trajectory)
        # Leave room for action_horizon
        max_obs_idx = min(T, len(actions)) - self.action_horizon - 1
        if max_obs_idx < 0:
            max_obs_idx = 0
        obs_idx = np.random.randint(0, max(1, max_obs_idx + 1))
        current_obs = video[obs_idx]  # [3, H, W]

        # Goal video: frames near the end (showing destination)
        goal_start = max(0, T - self.goal_frames)
        goal = video[goal_start:T]  # [<=goal_frames, 3, H, W]

        # Pad goal if needed
        if goal.shape[0] < self.goal_frames:
            padding = self.goal_frames - goal.shape[0]
            last_frame = goal[-1:].repeat(padding, 1, 1, 1)
            goal = torch.cat([goal, last_frame], dim=0)

        # Actions from current observation point
        action_start = min(obs_idx, len(actions) - self.action_horizon)
        if action_start < 0:
            action_start = 0
        action_sequence = actions[action_start:action_start + self.action_horizon]

        # Pad actions if needed
        if len(action_sequence) < self.action_horizon:
            padding = self.action_horizon - len(action_sequence)
            action_sequence = torch.cat([
                action_sequence,
                torch.zeros(padding, 3)
            ], dim=0)

        return {
            'goal': goal,              # [goal_frames, 3, H, W]
            'current_obs': current_obs,  # [3, H, W]
            'actions': action_sequence,  # [action_horizon, 3]
            'index': sample_idx
        }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: FlowDiTProduction,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
):
    """Train for one epoch with goal-conditioned inputs."""
    model.train()

    total_loss = 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        goal = batch['goal'].to(device)            # [B, goal_frames, 3, H, W]
        current_obs = batch['current_obs'].to(device)  # [B, 3, H, W]
        actions_gt = batch['actions'].to(device)   # [B, action_horizon, 3]

        # Forward pass with goal and observation
        predicted_noise, true_noise = model(goal, current_obs, actions_gt)

        # Diffusion loss
        loss = F.mse_loss(predicted_noise, true_noise)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        total_loss += loss.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
        })

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: FlowDiTProduction,
    val_loader: DataLoader,
    device: str
):
    """Validate model with goal-conditioned inputs."""
    model.eval()

    total_loss = 0.0
    num_batches = len(val_loader)

    for batch in tqdm(val_loader, desc="Validating"):
        goal = batch['goal'].to(device)
        current_obs = batch['current_obs'].to(device)
        actions_gt = batch['actions'].to(device)

        predicted_noise, true_noise = model(goal, current_obs, actions_gt)
        loss = F.mse_loss(predicted_noise, true_noise)

        total_loss += loss.item()

    return total_loss / num_batches


def save_checkpoint(
    model: FlowDiTProduction,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.config.__dict__
    }

    # Save latest
    torch.save(checkpoint, checkpoint_dir / "latest.pth")

    # Save best
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best.pth")

    # Save periodic
    if epoch % 10 == 0:
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pth")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main(args):
    """Main training function for goal-conditioned navigation."""

    print("\n" + "="*70)
    print("FlowDiT V2 - Goal-Conditioned Navigation Training")
    print("="*70)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Goal frames: {args.goal_frames}")
    print("="*70 + "\n")

    # Create model
    config = FlowDiTConfig(
        action_dim=3,
        action_horizon=args.action_horizon,
        goal_frames=args.goal_frames,
        use_language=False,  # Disabled for deadline
        device=args.device
    )

    model = FlowDiTProduction(config)
    model = model.to(args.device)

    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Create datasets
    train_dataset = NavigationDataset(
        args.dataset_dir,
        action_horizon=args.action_horizon,
        video_length=80,
        goal_frames=args.goal_frames,
        split="train"
    )

    val_dataset = NavigationDataset(
        args.dataset_dir,
        action_horizon=args.action_horizon,
        video_length=80,
        goal_frames=args.goal_frames,
        split="val"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device, epoch)
        print(f"\nTrain Loss: {train_loss:.4f}")

        # Validate
        if epoch % args.val_every == 0:
            val_loss = validate(model, val_loader, args.device)
            print(f"Val Loss: {val_loss:.4f}")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"✓ New best model! Val loss: {val_loss:.4f}")

            save_checkpoint(
                model, optimizer, epoch, val_loss,
                checkpoint_dir, is_best
            )

        # Step scheduler
        scheduler.step()

        # Print learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")

    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*70 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FlowDiT Production Model")

    # Dataset
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset directory")

    # Model
    parser.add_argument("--action_horizon", type=int, default=8,
                        help="Action horizon (number of future actions)")
    parser.add_argument("--goal_frames", type=int, default=16,
                        help="Number of frames to use as goal video")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")

    # Validation
    parser.add_argument("--val_every", type=int, default=1,
                        help="Validate every N epochs")

    # System
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")

    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run training
    main(args)
