#!/usr/bin/env python3
"""
Convert existing LeRobot dataset to DC-GR00T format.

This script takes your existing 1000 episodes of robot data and prepares
them for DC-GR00T training by:
1. Creating self-demo pairs (robot demos itself)
2. Optionally adding external demo sources

For self-demo mode:
- Each episode's ego view becomes both the "demo" and the "observation"
- Random temporal offsets create variety
- This teaches the model that "watching myself do it" = "doing it"

Usage:
    python convert_lerobot_to_dc.py \
        --input_dir /path/to/lerobot_dataset \
        --output_dir /path/to/dc_dataset \
        --mode self_demo
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

import numpy as np


def load_lerobot_episodes(dataset_path: Path) -> List[Dict]:
    """Load episode metadata from LeRobot dataset."""
    episodes_file = dataset_path / "meta" / "episodes.jsonl"
    episodes = []

    if episodes_file.exists():
        with open(episodes_file, "r") as f:
            for line in f:
                episodes.append(json.loads(line.strip()))
    else:
        # Fallback: scan data directory
        data_dir = dataset_path / "data"
        if data_dir.exists():
            for parquet_file in sorted(data_dir.glob("*.parquet")):
                ep_id = parquet_file.stem
                ep_idx = int(ep_id.split("_")[-1]) if "_" in ep_id else int(ep_id)
                episodes.append({
                    "episode_index": ep_idx,
                    "episode_id": ep_id,
                })

    return episodes


def load_tasks(dataset_path: Path) -> Dict[int, str]:
    """Load task descriptions."""
    tasks_file = dataset_path / "meta" / "tasks.jsonl"
    tasks = {}

    if tasks_file.exists():
        with open(tasks_file, "r") as f:
            for line in f:
                task = json.loads(line.strip())
                tasks[task["task_index"]] = task["task"]

    return tasks


def find_video_file(dataset_path: Path, episode_id: str, video_key: str = "ego_view") -> Optional[Path]:
    """Find video file for an episode."""
    # Try different naming conventions
    possible_paths = [
        dataset_path / "videos" / f"observation.images.{video_key}" / f"{episode_id}.mp4",
        dataset_path / "videos" / video_key / f"{episode_id}.mp4",
        dataset_path / "videos" / f"{episode_id}_{video_key}.mp4",
        dataset_path / "videos" / f"{episode_id}.mp4",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Try chunk-based structure
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        for chunk_dir in videos_dir.glob("chunk-*"):
            chunk_paths = [
                chunk_dir / f"observation.images.{video_key}" / f"{episode_id}.mp4",
                chunk_dir / video_key / f"{episode_id}.mp4",
                chunk_dir / f"{episode_id}_{video_key}.mp4",
                chunk_dir / f"{episode_id}.mp4",
            ]
            for path in chunk_paths:
                if path.exists():
                    return path

    return None


def create_self_demo_dataset(
    input_path: Path,
    output_path: Path,
    num_augmentations: int = 3,
    video_keys: List[str] = ["ego_view"],
) -> None:
    """
    Create self-demo dataset where robot demos are used as both demo and target.

    This teaches the model that watching a task execution should enable replication.
    """

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "demo").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "ego_view").mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(parents=True, exist_ok=True)

    # Load episodes
    episodes = load_lerobot_episodes(input_path)
    tasks = load_tasks(input_path)

    print(f"Found {len(episodes)} episodes")

    output_episodes = []
    output_idx = 0

    for ep in episodes:
        ep_idx = ep.get("episode_index", 0)
        ep_id = ep.get("episode_id", f"episode_{ep_idx:06d}")
        if not ep_id.startswith("episode_"):
            ep_id = f"episode_{ep_idx:06d}"

        task_idx = ep.get("task_index", 0)
        task_desc = tasks.get(task_idx, "")

        # Find video file
        video_path = None
        for vk in video_keys:
            video_path = find_video_file(input_path, ep_id, vk)
            if video_path:
                break

        if not video_path:
            print(f"Warning: No video found for {ep_id}")
            continue

        # Find action data (handle chunked structure)
        data_path = input_path / "data" / f"{ep_id}.parquet"
        if not data_path.exists():
            # Try chunk-based structure
            for chunk_dir in (input_path / "data").glob("chunk-*"):
                chunk_data_path = chunk_dir / f"{ep_id}.parquet"
                if chunk_data_path.exists():
                    data_path = chunk_data_path
                    break

        if not data_path.exists():
            print(f"Warning: No data found for {ep_id}")
            continue

        # Create augmented samples
        for aug_idx in range(num_augmentations):
            output_ep_id = f"episode_{output_idx:06d}"

            # Copy video as demo
            demo_dst = output_path / "videos" / "demo" / f"{output_ep_id}.mp4"
            shutil.copy2(video_path, demo_dst)

            # Copy video as ego view (same video for self-demo)
            ego_dst = output_path / "videos" / "ego_view" / f"{output_ep_id}.mp4"
            shutil.copy2(video_path, ego_dst)

            # Copy action data
            shutil.copy2(data_path, output_path / "data" / f"{output_ep_id}.parquet")

            output_episodes.append({
                "episode_id": output_ep_id,
                "original_episode": ep_id,
                "demo_type": "own",  # Self-demo
                "task_description": task_desc,
                "augmentation_idx": aug_idx,
            })

            output_idx += 1

        if (ep_idx + 1) % 100 == 0:
            print(f"Processed {ep_idx + 1}/{len(episodes)} episodes")

    # Write episodes.jsonl
    with open(output_path / "episodes.jsonl", "w") as f:
        for ep in output_episodes:
            f.write(json.dumps(ep) + "\n")

    # Write info
    info = {
        "dataset_type": "dc_groot_self_demo",
        "source_dataset": str(input_path),
        "num_episodes": len(output_episodes),
        "num_original_episodes": len(episodes),
        "num_augmentations": num_augmentations,
    }

    with open(output_path / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSelf-demo dataset created: {output_path}")
    print(f"Original episodes: {len(episodes)}")
    print(f"Output episodes: {len(output_episodes)}")


def create_multi_view_dataset(
    input_path: Path,
    output_path: Path,
    demo_camera: str = "front_view",
    exec_camera: str = "ego_view",
) -> None:
    """
    Create dataset with different camera views for demo vs execution.

    Demo: External view (front, side, etc.)
    Execution: Ego view (robot's perspective)
    """

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "demo").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "ego_view").mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(parents=True, exist_ok=True)

    episodes = load_lerobot_episodes(input_path)
    tasks = load_tasks(input_path)

    output_episodes = []

    for ep in episodes:
        ep_idx = ep.get("episode_index", 0)
        ep_id = ep.get("episode_id", f"episode_{ep_idx:06d}")
        if not ep_id.startswith("episode_"):
            ep_id = f"episode_{ep_idx:06d}"

        task_idx = ep.get("task_index", 0)
        task_desc = tasks.get(task_idx, "")

        # Find demo camera video
        demo_path = find_video_file(input_path, ep_id, demo_camera)
        if not demo_path:
            continue

        # Find execution camera video
        exec_path = find_video_file(input_path, ep_id, exec_camera)
        if not exec_path:
            continue

        # Find action data (handle chunked structure)
        data_path = input_path / "data" / f"{ep_id}.parquet"
        if not data_path.exists():
            # Try chunk-based structure
            for chunk_dir in (input_path / "data").glob("chunk-*"):
                chunk_data_path = chunk_dir / f"{ep_id}.parquet"
                if chunk_data_path.exists():
                    data_path = chunk_data_path
                    break

        if not data_path.exists():
            continue

        output_ep_id = f"episode_{len(output_episodes):06d}"

        # Copy demo video
        shutil.copy2(demo_path, output_path / "videos" / "demo" / f"{output_ep_id}.mp4")

        # Copy ego video
        shutil.copy2(exec_path, output_path / "videos" / "ego_view" / f"{output_ep_id}.mp4")

        # Copy data
        shutil.copy2(data_path, output_path / "data" / f"{output_ep_id}.parquet")

        output_episodes.append({
            "episode_id": output_ep_id,
            "original_episode": ep_id,
            "demo_type": "robot",
            "demo_camera": demo_camera,
            "exec_camera": exec_camera,
            "task_description": task_desc,
        })

    # Write metadata
    with open(output_path / "episodes.jsonl", "w") as f:
        for ep in output_episodes:
            f.write(json.dumps(ep) + "\n")

    info = {
        "dataset_type": "dc_groot_multi_view",
        "source_dataset": str(input_path),
        "num_episodes": len(output_episodes),
        "demo_camera": demo_camera,
        "exec_camera": exec_camera,
    }

    with open(output_path / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nMulti-view dataset created: {output_path}")
    print(f"Episodes: {len(output_episodes)}")


def create_temporal_augmented_dataset(
    input_path: Path,
    output_path: Path,
    demo_offset_range: tuple = (-30, 30),  # Frames
    num_augmentations: int = 5,
) -> None:
    """
    Create dataset with temporal augmentation.

    Uses different temporal segments of the same trajectory as demo/execution.
    This teaches temporal correspondence and trajectory following.
    """

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "demo").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "ego_view").mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(parents=True, exist_ok=True)

    episodes = load_lerobot_episodes(input_path)
    tasks = load_tasks(input_path)

    output_episodes = []
    output_idx = 0

    for ep in episodes:
        ep_idx = ep.get("episode_index", 0)
        ep_id = ep.get("episode_id", f"episode_{ep_idx:06d}")
        if not ep_id.startswith("episode_"):
            ep_id = f"episode_{ep_idx:06d}"

        task_idx = ep.get("task_index", 0)
        task_desc = tasks.get(task_idx, "")

        # Find video
        video_path = find_video_file(input_path, ep_id, "ego_view")
        if not video_path:
            continue

        # Find data
        data_path = input_path / "data" / f"{ep_id}.parquet"
        if not data_path.exists():
            continue

        # Get video length
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except:
            video_length = 300  # Assume ~10s at 30fps

        # Create augmented samples with different temporal segments
        for aug_idx in range(num_augmentations):
            output_ep_id = f"episode_{output_idx:06d}"

            # For simplicity, just copy the same video
            # In practice, you'd want to create temporally offset versions
            shutil.copy2(video_path, output_path / "videos" / "demo" / f"{output_ep_id}.mp4")
            shutil.copy2(video_path, output_path / "videos" / "ego_view" / f"{output_ep_id}.mp4")
            shutil.copy2(data_path, output_path / "data" / f"{output_ep_id}.parquet")

            # Store temporal offset info for training
            offset = random.randint(demo_offset_range[0], demo_offset_range[1])

            output_episodes.append({
                "episode_id": output_ep_id,
                "original_episode": ep_id,
                "demo_type": "own",
                "temporal_offset": offset,
                "task_description": task_desc,
            })

            output_idx += 1

    # Write metadata
    with open(output_path / "episodes.jsonl", "w") as f:
        for ep in output_episodes:
            f.write(json.dumps(ep) + "\n")

    info = {
        "dataset_type": "dc_groot_temporal_aug",
        "source_dataset": str(input_path),
        "num_episodes": len(output_episodes),
        "demo_offset_range": list(demo_offset_range),
        "num_augmentations": num_augmentations,
    }

    with open(output_path / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nTemporal augmented dataset created: {output_path}")
    print(f"Episodes: {len(output_episodes)}")


def merge_datasets(
    dataset_paths: List[Path],
    output_path: Path,
) -> None:
    """Merge multiple DC datasets into one."""

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "demo").mkdir(parents=True, exist_ok=True)
    (output_path / "videos" / "ego_view").mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(parents=True, exist_ok=True)

    merged_episodes = []
    output_idx = 0

    for dataset_path in dataset_paths:
        print(f"Processing {dataset_path}")

        # Load episodes
        episodes_file = dataset_path / "episodes.jsonl"
        if not episodes_file.exists():
            continue

        with open(episodes_file, "r") as f:
            for line in f:
                ep = json.loads(line.strip())
                old_ep_id = ep["episode_id"]
                new_ep_id = f"episode_{output_idx:06d}"

                # Copy files
                for subdir in ["demo", "ego_view"]:
                    src = dataset_path / "videos" / subdir / f"{old_ep_id}.mp4"
                    dst = output_path / "videos" / subdir / f"{new_ep_id}.mp4"
                    if src.exists():
                        shutil.copy2(src, dst)

                src = dataset_path / "data" / f"{old_ep_id}.parquet"
                dst = output_path / "data" / f"{new_ep_id}.parquet"
                if src.exists():
                    shutil.copy2(src, dst)

                # Update episode info
                ep["episode_id"] = new_ep_id
                ep["source_dataset"] = str(dataset_path)
                merged_episodes.append(ep)

                output_idx += 1

    # Write merged episodes
    with open(output_path / "episodes.jsonl", "w") as f:
        for ep in merged_episodes:
            f.write(json.dumps(ep) + "\n")

    info = {
        "dataset_type": "dc_groot_merged",
        "source_datasets": [str(p) for p in dataset_paths],
        "num_episodes": len(merged_episodes),
    }

    with open(output_path / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nMerged dataset created: {output_path}")
    print(f"Total episodes: {len(merged_episodes)}")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot to DC-GR00T format")

    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input LeRobot dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output DC-GR00T dataset directory")

    parser.add_argument("--mode", type=str, default="self_demo",
                       choices=["self_demo", "multi_view", "temporal_aug", "merge"],
                       help="Conversion mode")

    # Self-demo options
    parser.add_argument("--num_augmentations", type=int, default=3,
                       help="Number of augmentations per episode")

    # Multi-view options
    parser.add_argument("--demo_camera", type=str, default="front_view",
                       help="Camera to use for demo")
    parser.add_argument("--exec_camera", type=str, default="ego_view",
                       help="Camera to use for execution")

    # Merge options
    parser.add_argument("--merge_dirs", type=str, nargs="+", default=None,
                       help="Directories to merge (for merge mode)")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if args.mode == "self_demo":
        create_self_demo_dataset(
            input_path=input_path,
            output_path=output_path,
            num_augmentations=args.num_augmentations,
        )

    elif args.mode == "multi_view":
        create_multi_view_dataset(
            input_path=input_path,
            output_path=output_path,
            demo_camera=args.demo_camera,
            exec_camera=args.exec_camera,
        )

    elif args.mode == "temporal_aug":
        create_temporal_augmented_dataset(
            input_path=input_path,
            output_path=output_path,
            num_augmentations=args.num_augmentations,
        )

    elif args.mode == "merge":
        if not args.merge_dirs:
            print("--merge_dirs required for merge mode")
            return
        merge_datasets(
            dataset_paths=[Path(d) for d in args.merge_dirs],
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
