"""
Flow-Constrained Video-to-Action Model for Cross-Embodiment Navigation

Author: Jeffrin Sam
Institution: Skoltech
Year: 2025
License: MIT

Description: Split dataset into train/val/test sets
"""

import argparse
import json
import random
from pathlib import Path


def split_dataset(
    dataset_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> dict:
    """
    Split dataset into train/val/test.

    Args:
        dataset_dir: Path to dataset directory
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed

    Returns:
        split_counts: Dictionary with split counts
    """
    # Set seed
    random.seed(seed)

    # Load metadata
    metadata_path = Path(dataset_dir) / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get all clip IDs
    clip_ids = list(metadata.keys())
    random.shuffle(clip_ids)

    # Calculate split sizes
    total = len(clip_ids)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    # Split
    train_ids = clip_ids[:train_size]
    val_ids = clip_ids[train_size:train_size + val_size]
    test_ids = clip_ids[train_size + val_size:]

    # Update metadata
    for clip_id in train_ids:
        metadata[clip_id]['split'] = 'train'

    for clip_id in val_ids:
        metadata[clip_id]['split'] = 'val'

    for clip_id in test_ids:
        metadata[clip_id]['split'] = 'test'

    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    split_counts = {
        'train': len(train_ids),
        'val': len(val_ids),
        'test': len(test_ids),
        'total': total
    }

    print(f"\nDataset split completed:")
    print(f"  Train: {split_counts['train']} clips ({split_counts['train']/total*100:.1f}%)")
    print(f"  Val:   {split_counts['val']} clips ({split_counts['val']/total*100:.1f}%)")
    print(f"  Test:  {split_counts['test']} clips ({split_counts['test']/total*100:.1f}%)")
    print(f"  Total: {split_counts['total']} clips")
    print(f"\nMetadata updated: {metadata_path}")

    return split_counts


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--dataset_dir", type=str, default="../dataset",
                        help="Path to dataset directory")
    parser.add_argument("--train", type=float, default=0.7,
                        help="Training set ratio")
    parser.add_argument("--val", type=float, default=0.15,
                        help="Validation set ratio")
    parser.add_argument("--test", type=float, default=0.15,
                        help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return

    split_dataset(
        dataset_dir=args.dataset_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
