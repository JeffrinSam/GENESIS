"""
Flow-Constrained Video-to-Action Model for Cross-Embodiment Navigation

Author: Jeffrin Sam
Institution: Skoltech
Year: 2025
License: MIT

Description: Verify dataset integrity and completeness
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def verify_dataset(dataset_dir: str) -> dict:
    """
    Verify dataset integrity.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        stats: Dictionary with verification statistics
    """
    dataset_path = Path(dataset_dir)
    video_dir = dataset_path / "videos"
    action_dir = dataset_path / "actions"
    metadata_path = dataset_path / "metadata.json"

    # Check directories exist
    if not video_dir.exists():
        print(f"✗ Video directory not found: {video_dir}")
        return {}

    if not action_dir.exists():
        print(f"✗ Action directory not found: {action_dir}")
        return {}

    if not metadata_path.exists():
        print(f"✗ Metadata file not found: {metadata_path}")
        return {}

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"✓ Found metadata with {len(metadata)} entries")

    # Verification stats
    stats = {
        'total_clips': len(metadata),
        'valid_clips': 0,
        'corrupted_videos': [],
        'missing_videos': [],
        'missing_actions': [],
        'length_mismatches': [],
        'total_duration': 0.0,
        'total_frames': 0,
        'embodiments': {},
        'resolutions': set()
    }

    # Verify each clip
    print("\nVerifying clips...")
    for clip_id, clip_meta in tqdm(metadata.items()):
        video_path = video_dir / f"{clip_id}.mp4"
        action_path = action_dir / f"{clip_id}.npy"

        # Check video exists
        if not video_path.exists():
            stats['missing_videos'].append(clip_id)
            continue

        # Check action file exists
        if not action_path.exists():
            stats['missing_actions'].append(clip_id)
            continue

        # Verify video
        try:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Check video is not corrupted
            if frame_count == 0:
                stats['corrupted_videos'].append(clip_id)
                continue

        except Exception as e:
            stats['corrupted_videos'].append(clip_id)
            continue

        # Verify actions
        try:
            actions = np.load(action_path)

            # Check length matches
            if len(actions) != frame_count:
                stats['length_mismatches'].append(
                    f"{clip_id}: video={frame_count}, actions={len(actions)}"
                )

        except Exception as e:
            stats['missing_actions'].append(clip_id)
            continue

        # Update stats
        stats['valid_clips'] += 1
        stats['total_duration'] += clip_meta.get('duration', 0)
        stats['total_frames'] += frame_count

        # Track embodiments
        embodiment = clip_meta.get('embodiment', 'unknown')
        stats['embodiments'][embodiment] = stats['embodiments'].get(embodiment, 0) + 1

        # Track resolutions
        res = tuple(clip_meta.get('resolution', [height, width]))
        stats['resolutions'].add(res)

    # Print results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    print(f"\nTotal clips: {stats['total_clips']}")
    print(f"Valid clips: {stats['valid_clips']} ({stats['valid_clips']/stats['total_clips']*100:.1f}%)")

    if stats['missing_videos']:
        print(f"\n✗ Missing videos: {len(stats['missing_videos'])}")
        for clip in stats['missing_videos'][:5]:
            print(f"  - {clip}")
        if len(stats['missing_videos']) > 5:
            print(f"  ... and {len(stats['missing_videos']) - 5} more")

    if stats['missing_actions']:
        print(f"\n✗ Missing actions: {len(stats['missing_actions'])}")
        for clip in stats['missing_actions'][:5]:
            print(f"  - {clip}")

    if stats['corrupted_videos']:
        print(f"\n✗ Corrupted videos: {len(stats['corrupted_videos'])}")
        for clip in stats['corrupted_videos'][:5]:
            print(f"  - {clip}")

    if stats['length_mismatches']:
        print(f"\n⚠ Length mismatches: {len(stats['length_mismatches'])}")
        for mismatch in stats['length_mismatches'][:5]:
            print(f"  - {mismatch}")

    print(f"\n✓ Total duration: {stats['total_duration']/60:.1f} minutes")
    print(f"✓ Total frames: {stats['total_frames']:,}")

    print("\nEmbodiments:")
    for embodiment, count in stats['embodiments'].items():
        print(f"  - {embodiment}: {count} clips")

    print("\nResolutions:")
    for res in stats['resolutions']:
        print(f"  - {res[0]}×{res[1]}")

    # Final verdict
    print("\n" + "=" * 60)
    if stats['valid_clips'] == stats['total_clips']:
        print("✓ DATASET IS VALID")
    elif stats['valid_clips'] > stats['total_clips'] * 0.95:
        print("⚠ DATASET MOSTLY VALID (>95%)")
    else:
        print("✗ DATASET HAS ISSUES")
    print("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Verify dataset integrity")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--fix", action="store_true",
                        help="Try to fix issues automatically")

    args = parser.parse_args()

    stats = verify_dataset(args.dataset_dir)

    if args.fix and stats:
        print("\nFixing issues...")
        # TODO: Implement automatic fixing
        print("Auto-fix not implemented yet")


if __name__ == "__main__":
    main()
