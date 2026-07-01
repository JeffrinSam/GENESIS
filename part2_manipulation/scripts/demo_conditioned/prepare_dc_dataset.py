#!/usr/bin/env python3
"""
Prepare Dataset for Demo-Conditioned GR00T Training

This script helps convert your existing robot data + demo videos into
the format required for DC-GR00T training.

Supports multiple demo sources:
- Human hand demos (from phone camera)
- Robot demos (from another robot or different viewpoint)
- Cosmos-generated videos
- Your own G1 demos (different viewpoint)

Usage:
    # Basic usage with existing LeRobot dataset
    python prepare_dc_dataset.py \
        --robot_data /path/to/lerobot_dataset \
        --demo_videos /path/to/demo_videos \
        --output_dir /path/to/dc_dataset

    # With Cosmos-generated videos
    python prepare_dc_dataset.py \
        --robot_data /path/to/lerobot_dataset \
        --cosmos_videos /path/to/cosmos_videos \
        --output_dir /path/to/dc_dataset

    # With human hand demos
    python prepare_dc_dataset.py \
        --robot_data /path/to/lerobot_dataset \
        --human_demos /path/to/human_videos \
        --output_dir /path/to/dc_dataset
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

import numpy as np


@dataclass
class DemoMapping:
    """Mapping between demo video and robot execution."""
    demo_path: str
    demo_type: str  # "human", "robot", "cosmos", "own"
    episode_id: str
    task_description: str = ""


def find_video_files(directory: Path, extensions: List[str] = [".mp4", ".avi", ".mov", ".webm"]) -> List[Path]:
    """Find all video files in directory."""
    videos = []
    for ext in extensions:
        videos.extend(directory.glob(f"**/*{ext}"))
    return sorted(videos)


def extract_episode_id(filename: str) -> Optional[str]:
    """Extract episode ID from filename."""
    # Try common patterns
    patterns = [
        r"episode[_-]?(\d+)",
        r"ep[_-]?(\d+)",
        r"(\d{6})",
        r"(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            return f"episode_{num:06d}"

    return None


def load_lerobot_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Load LeRobot v2 dataset metadata."""
    meta_path = dataset_path / "meta"

    info = {}

    # Load info.json
    info_file = meta_path / "info.json"
    if info_file.exists():
        with open(info_file, "r") as f:
            info["info"] = json.load(f)

    # Load episodes.jsonl
    episodes_file = meta_path / "episodes.jsonl"
    episodes = []
    if episodes_file.exists():
        with open(episodes_file, "r") as f:
            for line in f:
                episodes.append(json.loads(line.strip()))
    info["episodes"] = episodes

    # Load tasks.jsonl
    tasks_file = meta_path / "tasks.jsonl"
    tasks = {}
    if tasks_file.exists():
        with open(tasks_file, "r") as f:
            for line in f:
                task = json.loads(line.strip())
                tasks[task["task_index"]] = task["task"]
    info["tasks"] = tasks

    return info


def create_demo_mappings(
    robot_data_path: Path,
    demo_videos_path: Optional[Path] = None,
    cosmos_videos_path: Optional[Path] = None,
    human_demos_path: Optional[Path] = None,
    own_demos_path: Optional[Path] = None,
    task_mapping_file: Optional[Path] = None,
) -> List[DemoMapping]:
    """Create mappings between demos and robot executions."""

    mappings = []

    # Load robot dataset to get episode list
    robot_info = load_lerobot_dataset(robot_data_path)
    episodes = robot_info.get("episodes", [])
    tasks = robot_info.get("tasks", {})

    # Load custom task mapping if provided
    custom_mapping = {}
    if task_mapping_file and task_mapping_file.exists():
        with open(task_mapping_file, "r") as f:
            custom_mapping = json.load(f)

    # Process each demo source
    demo_sources = [
        (demo_videos_path, "robot"),
        (cosmos_videos_path, "cosmos"),
        (human_demos_path, "human"),
        (own_demos_path, "own"),
    ]

    for demo_path, demo_type in demo_sources:
        if demo_path is None or not demo_path.exists():
            continue

        demo_videos = find_video_files(demo_path)
        print(f"Found {len(demo_videos)} {demo_type} demo videos")

        for video_path in demo_videos:
            video_name = video_path.stem

            # Try to match with episode
            episode_id = extract_episode_id(video_name)

            if episode_id:
                # Get task description
                task_desc = ""
                if video_name in custom_mapping:
                    task_desc = custom_mapping[video_name]
                elif episode_id in custom_mapping:
                    task_desc = custom_mapping[episode_id]
                else:
                    # Try to get from episode metadata
                    for ep in episodes:
                        if ep.get("episode_index") == int(episode_id.split("_")[-1]):
                            task_idx = ep.get("task_index", 0)
                            task_desc = tasks.get(task_idx, "")
                            break

                mappings.append(DemoMapping(
                    demo_path=str(video_path),
                    demo_type=demo_type,
                    episode_id=episode_id,
                    task_description=task_desc,
                ))

    return mappings


def convert_video_to_standard_format(
    input_path: Path,
    output_path: Path,
    target_fps: int = 30,
    target_resolution: Tuple[int, int] = (224, 224),
) -> bool:
    """Convert video to standard format for training."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False

        # Get source properties
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame sampling
        frame_step = max(1, int(src_fps / target_fps))

        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            target_fps,
            target_resolution,
        )

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                # Resize frame
                resized = cv2.resize(frame, target_resolution)
                out.write(resized)

            frame_idx += 1

        cap.release()
        out.release()
        return True

    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def prepare_dc_dataset(
    robot_data_path: Path,
    output_dir: Path,
    mappings: List[DemoMapping],
    convert_videos: bool = True,
    video_resolution: Tuple[int, int] = (224, 224),
) -> None:
    """Prepare the DC-GR00T dataset."""

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "demo").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "ego_view").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # Load robot data
    robot_info = load_lerobot_dataset(robot_data_path)

    # Process each mapping
    episodes_data = []

    for i, mapping in enumerate(mappings):
        print(f"Processing {i+1}/{len(mappings)}: {mapping.episode_id}")

        # Copy/convert demo video
        demo_src = Path(mapping.demo_path)
        demo_dst = output_dir / "videos" / "demo" / f"{mapping.episode_id}.mp4"

        if convert_videos:
            convert_video_to_standard_format(demo_src, demo_dst, target_resolution=video_resolution)
        else:
            shutil.copy2(demo_src, demo_dst)

        # Copy ego view video from robot data
        ego_src = robot_data_path / "videos" / "observation.images.ego_view" / f"{mapping.episode_id}.mp4"
        if not ego_src.exists():
            # Try alternative naming
            ego_src = robot_data_path / "videos" / "ego_view" / f"{mapping.episode_id}.mp4"

        ego_dst = output_dir / "videos" / "ego_view" / f"{mapping.episode_id}.mp4"

        if ego_src.exists():
            if convert_videos:
                convert_video_to_standard_format(ego_src, ego_dst, target_resolution=video_resolution)
            else:
                shutil.copy2(ego_src, ego_dst)

        # Copy action/state data
        data_src = robot_data_path / "data" / f"{mapping.episode_id}.parquet"
        if data_src.exists():
            shutil.copy2(data_src, output_dir / "data" / f"{mapping.episode_id}.parquet")

        # Record episode info
        episodes_data.append({
            "episode_id": mapping.episode_id,
            "demo_type": mapping.demo_type,
            "task_description": mapping.task_description,
            "demo_path": str(demo_dst.relative_to(output_dir)),
            "ego_path": str(ego_dst.relative_to(output_dir)),
        })

    # Write episodes.jsonl
    with open(output_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_data:
            f.write(json.dumps(ep) + "\n")

    # Write dataset info
    info = {
        "dataset_type": "dc_groot",
        "num_episodes": len(episodes_data),
        "demo_types": list(set(m.demo_type for m in mappings)),
        "video_resolution": list(video_resolution),
        "source_dataset": str(robot_data_path),
    }

    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset prepared: {output_dir}")
    print(f"Total episodes: {len(episodes_data)}")
    print(f"Demo types: {info['demo_types']}")


def create_cross_embodiment_pairs(
    human_demos_path: Path,
    robot_data_path: Path,
    output_dir: Path,
    task_matching: str = "filename",  # "filename", "task_label", "manual"
    manual_mapping_file: Optional[Path] = None,
) -> None:
    """
    Create training pairs for cross-embodiment learning.

    This pairs human demos with robot executions of the same task.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "human_demo").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "robot_ego").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # Load human demos
    human_videos = find_video_files(human_demos_path)

    # Load robot data
    robot_info = load_lerobot_dataset(robot_data_path)

    # Load manual mapping if provided
    manual_mapping = {}
    if manual_mapping_file and manual_mapping_file.exists():
        with open(manual_mapping_file, "r") as f:
            manual_mapping = json.load(f)

    pairs = []

    if task_matching == "filename":
        # Match by extracting task name from filename
        for human_video in human_videos:
            task_name = human_video.stem.lower()
            task_name = re.sub(r'[_\-\d]+', ' ', task_name).strip()

            # Find matching robot episodes
            for ep in robot_info.get("episodes", []):
                ep_task_idx = ep.get("task_index", 0)
                ep_task = robot_info["tasks"].get(ep_task_idx, "").lower()

                if task_name in ep_task or ep_task in task_name:
                    pairs.append({
                        "human_demo": str(human_video),
                        "robot_episode": f"episode_{ep['episode_index']:06d}",
                        "task": robot_info["tasks"].get(ep_task_idx, ""),
                    })

    elif task_matching == "manual":
        # Use manual mapping file
        # Format: {"human_video_name": ["robot_episode_1", "robot_episode_2"]}
        for human_name, robot_eps in manual_mapping.items():
            human_video = human_demos_path / f"{human_name}.mp4"
            if not human_video.exists():
                human_video = human_demos_path / human_name

            if human_video.exists():
                for robot_ep in robot_eps:
                    pairs.append({
                        "human_demo": str(human_video),
                        "robot_episode": robot_ep,
                        "task": manual_mapping.get(f"{human_name}_task", ""),
                    })

    # Process pairs
    episodes_data = []

    for i, pair in enumerate(pairs):
        episode_id = f"pair_{i:06d}"
        print(f"Processing pair {i+1}/{len(pairs)}: {episode_id}")

        # Copy human demo
        human_src = Path(pair["human_demo"])
        human_dst = output_dir / "videos" / "human_demo" / f"{episode_id}.mp4"
        convert_video_to_standard_format(human_src, human_dst)

        # Copy robot ego view
        robot_ep = pair["robot_episode"]
        ego_src = robot_data_path / "videos" / "observation.images.ego_view" / f"{robot_ep}.mp4"
        ego_dst = output_dir / "videos" / "robot_ego" / f"{episode_id}.mp4"

        if ego_src.exists():
            convert_video_to_standard_format(ego_src, ego_dst)

        # Copy robot action data
        data_src = robot_data_path / "data" / f"{robot_ep}.parquet"
        if data_src.exists():
            shutil.copy2(data_src, output_dir / "data" / f"{episode_id}.parquet")

        episodes_data.append({
            "episode_id": episode_id,
            "demo_type": "human",
            "original_human_demo": pair["human_demo"],
            "original_robot_episode": robot_ep,
            "task": pair["task"],
        })

    # Write episodes
    with open(output_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_data:
            f.write(json.dumps(ep) + "\n")

    print(f"\nCross-embodiment dataset prepared: {output_dir}")
    print(f"Total pairs: {len(pairs)}")


def augment_with_cosmos(
    robot_data_path: Path,
    cosmos_videos_path: Path,
    output_dir: Path,
) -> None:
    """
    Augment dataset with Cosmos-generated videos.

    For each robot episode, if there's a corresponding Cosmos video,
    create an augmented training sample.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "cosmos_demo").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "robot_ego").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # Find Cosmos videos
    cosmos_videos = find_video_files(cosmos_videos_path)
    cosmos_map = {}
    for video in cosmos_videos:
        episode_id = extract_episode_id(video.stem)
        if episode_id:
            cosmos_map[episode_id] = video

    # Load robot data
    robot_info = load_lerobot_dataset(robot_data_path)

    episodes_data = []

    for ep in robot_info.get("episodes", []):
        episode_id = f"episode_{ep['episode_index']:06d}"

        if episode_id in cosmos_map:
            print(f"Processing Cosmos augmentation for {episode_id}")

            # Copy Cosmos video as demo
            cosmos_src = cosmos_map[episode_id]
            cosmos_dst = output_dir / "videos" / "cosmos_demo" / f"{episode_id}.mp4"
            convert_video_to_standard_format(cosmos_src, cosmos_dst)

            # Copy robot ego view
            ego_src = robot_data_path / "videos" / "observation.images.ego_view" / f"{episode_id}.mp4"
            ego_dst = output_dir / "videos" / "robot_ego" / f"{episode_id}.mp4"

            if ego_src.exists():
                convert_video_to_standard_format(ego_src, ego_dst)

            # Copy action data
            data_src = robot_data_path / "data" / f"{episode_id}.parquet"
            if data_src.exists():
                shutil.copy2(data_src, output_dir / "data" / f"{episode_id}.parquet")

            task_idx = ep.get("task_index", 0)
            task_desc = robot_info["tasks"].get(task_idx, "")

            episodes_data.append({
                "episode_id": episode_id,
                "demo_type": "cosmos",
                "task": task_desc,
            })

    # Write episodes
    with open(output_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_data:
            f.write(json.dumps(ep) + "\n")

    print(f"\nCosmos-augmented dataset prepared: {output_dir}")
    print(f"Total episodes: {len(episodes_data)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DC-GR00T dataset")

    parser.add_argument("--robot_data", type=str, required=True,
                       help="Path to LeRobot format robot dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for DC-GR00T dataset")

    # Demo sources
    parser.add_argument("--demo_videos", type=str, default=None,
                       help="Path to robot demo videos")
    parser.add_argument("--cosmos_videos", type=str, default=None,
                       help="Path to Cosmos-generated videos")
    parser.add_argument("--human_demos", type=str, default=None,
                       help="Path to human hand demo videos")
    parser.add_argument("--own_demos", type=str, default=None,
                       help="Path to own G1 demos (different viewpoint)")

    # Options
    parser.add_argument("--task_mapping", type=str, default=None,
                       help="JSON file mapping videos to task descriptions")
    parser.add_argument("--convert_videos", action="store_true", default=True,
                       help="Convert videos to standard format")
    parser.add_argument("--video_resolution", type=int, nargs=2, default=[224, 224],
                       help="Target video resolution (width, height)")

    # Modes
    parser.add_argument("--mode", type=str, default="basic",
                       choices=["basic", "cross_embodiment", "cosmos_augment"],
                       help="Dataset preparation mode")
    parser.add_argument("--cross_matching", type=str, default="filename",
                       choices=["filename", "task_label", "manual"],
                       help="Task matching method for cross-embodiment")
    parser.add_argument("--manual_mapping", type=str, default=None,
                       help="Manual mapping file for cross-embodiment")

    args = parser.parse_args()

    robot_data_path = Path(args.robot_data)
    output_dir = Path(args.output_dir)

    if args.mode == "basic":
        # Create mappings
        mappings = create_demo_mappings(
            robot_data_path=robot_data_path,
            demo_videos_path=Path(args.demo_videos) if args.demo_videos else None,
            cosmos_videos_path=Path(args.cosmos_videos) if args.cosmos_videos else None,
            human_demos_path=Path(args.human_demos) if args.human_demos else None,
            own_demos_path=Path(args.own_demos) if args.own_demos else None,
            task_mapping_file=Path(args.task_mapping) if args.task_mapping else None,
        )

        if not mappings:
            print("No demo mappings found. Please check your paths.")
            return

        prepare_dc_dataset(
            robot_data_path=robot_data_path,
            output_dir=output_dir,
            mappings=mappings,
            convert_videos=args.convert_videos,
            video_resolution=tuple(args.video_resolution),
        )

    elif args.mode == "cross_embodiment":
        if not args.human_demos:
            print("--human_demos required for cross_embodiment mode")
            return

        create_cross_embodiment_pairs(
            human_demos_path=Path(args.human_demos),
            robot_data_path=robot_data_path,
            output_dir=output_dir,
            task_matching=args.cross_matching,
            manual_mapping_file=Path(args.manual_mapping) if args.manual_mapping else None,
        )

    elif args.mode == "cosmos_augment":
        if not args.cosmos_videos:
            print("--cosmos_videos required for cosmos_augment mode")
            return

        augment_with_cosmos(
            robot_data_path=robot_data_path,
            cosmos_videos_path=Path(args.cosmos_videos),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
