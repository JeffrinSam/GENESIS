"""
Flow-Constrained Video-to-Action Model for Cross-Embodiment Navigation

Author: Jeffrin Sam
Institution: Skoltech
Year: 2025
License: MIT

Description: Convert RECON dataset to VideotoNav format
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional
import shutil
import h5py


def load_recon_trajectory(hdf5_path: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load frames and actions from a RECON HDF5 trajectory file.

    Args:
        hdf5_path: Path to HDF5 trajectory file

    Returns:
        frames: List of RGB frames [H, W, 3]
        actions: Action array [T, 3] with (vx, vy, yaw_rate)
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Validate required fields exist
        required_fields = ['images/rgb_left', 'commands/linear_velocity', 'commands/angular_velocity']
        missing_fields = [field for field in required_fields if field not in f]
        if missing_fields:
            raise ValueError(f"Missing required HDF5 fields: {missing_fields}")

        # Load JPEG-compressed images
        jpeg_images = f['images/rgb_left'][:]
        linear_velocity = f['commands/linear_velocity'][:]
        angular_velocity = f['commands/angular_velocity'][:]

        # Validate length consistency
        T = len(jpeg_images)
        if T == 0:
            raise ValueError("Empty trajectory")
        if len(linear_velocity) != T or len(angular_velocity) != T:
            raise ValueError(f"Length mismatch: images={T}, linear_vel={len(linear_velocity)}, angular_vel={len(angular_velocity)}")

        # Decode JPEG images
        frames = []
        for i, jpeg_bytes in enumerate(jpeg_images):
            try:
                img_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise RuntimeError(f"Failed to decode JPEG at frame {i}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                frames.append(img_rgb)
            except Exception as e:
                raise RuntimeError(f"JPEG decode error at frame {i}: {e}")

        # Construct 3D action array for differential drive
        actions = np.zeros((T, 3), dtype=np.float64)
        actions[:, 0] = linear_velocity  # vx (m/s)
        actions[:, 1] = 0.0              # vy (m/s) - differential drive
        actions[:, 2] = angular_velocity # yaw_rate (rad/s)

    return frames, actions


def frames_to_video(frames: List[np.ndarray], output_path: Path, fps: int = 16):
    """Convert frames to MP4 video.

    Args:
        frames: List of RGB frames [H, W, 3]
        output_path: Output video path
        fps: Frames per second
    """
    if len(frames) == 0:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


def resample_trajectory(frames: List[np.ndarray], actions: np.ndarray,
                       target_fps: int = 16, source_fps: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
    """Resample trajectory to target FPS.

    RECON is typically recorded at 10 FPS, we want 16 FPS.

    Args:
        frames: Original frames
        actions: Original actions [T, 3]
        target_fps: Target frames per second
        source_fps: Source frames per second

    Returns:
        resampled_frames: Resampled frames
        resampled_actions: Resampled actions [T', 3]
    """
    num_frames = len(frames)
    duration = num_frames / source_fps
    target_num_frames = int(duration * target_fps)

    # Resample indices
    source_indices = np.arange(num_frames)
    target_indices = np.linspace(0, num_frames - 1, target_num_frames)

    # Resample frames (nearest neighbor)
    resampled_frames = []
    for idx in target_indices:
        nearest_idx = int(np.round(idx))
        resampled_frames.append(frames[nearest_idx])

    # Resample actions (linear interpolation)
    resampled_actions = np.zeros((target_num_frames, actions.shape[1]))
    for i in range(actions.shape[1]):
        resampled_actions[:, i] = np.interp(target_indices, source_indices, actions[:, i])

    return resampled_frames, resampled_actions


def clip_trajectory(frames: List[np.ndarray], actions: np.ndarray,
                    min_duration: float, max_duration: float, fps: int) -> List[Tuple[List[np.ndarray], np.ndarray]]:
    """Split trajectory into clips of specified duration.

    Args:
        frames: Full trajectory frames
        actions: Full trajectory actions [T, 3]
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        fps: Frames per second

    Returns:
        clips: List of (frames, actions) tuples
    """
    min_frames = int(min_duration * fps)
    max_frames = int(max_duration * fps)
    total_frames = len(frames)

    clips = []

    # If trajectory is shorter than min_duration, skip
    if total_frames < min_frames:
        return clips

    # If trajectory is shorter than max_duration, use entire trajectory
    if total_frames <= max_frames:
        return [(frames, actions)]

    # Split into non-overlapping clips of max_duration
    for start_idx in range(0, total_frames - min_frames + 1, max_frames):
        end_idx = min(start_idx + max_frames, total_frames)

        clip_frames = frames[start_idx:end_idx]
        clip_actions = actions[start_idx:end_idx]

        if len(clip_frames) >= min_frames:
            clips.append((clip_frames, clip_actions))

    return clips


def convert_recon_dataset(recon_dir: Path, output_dir: Path,
                         min_duration: float = 2.0, max_duration: float = 12.0,
                         fps: int = 16, source_fps: int = 10):
    """Convert RECON dataset to VideotoNav format.

    Args:
        recon_dir: Path to RECON dataset root
        output_dir: Output directory
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        fps: Target frames per second
        source_fps: RECON source FPS (typically 10)
    """
    # Find all HDF5 files
    hdf5_files = sorted(recon_dir.glob("*.hdf5"))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {recon_dir}")

    # Create output directories
    video_dir = output_dir / "videos"
    action_dir = output_dir / "actions"
    video_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if present
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    print(f"Found {len(hdf5_files)} HDF5 files in {recon_dir}")
    print(f"Converting to clips ({min_duration}s - {max_duration}s @ {fps}fps)...")

    clip_count = 0
    skipped_count = 0
    error_count = 0
    error_details = {}

    for hdf5_file in tqdm(hdf5_files, desc="Converting RECON"):
        traj_name = hdf5_file.stem

        try:
            # Load trajectory
            frames, actions = load_recon_trajectory(hdf5_file)

            # Resample to target FPS
            frames, actions = resample_trajectory(frames, actions, fps, source_fps)

            # Split into clips
            clips = clip_trajectory(frames, actions, min_duration, max_duration, fps)

            if len(clips) == 0:
                skipped_count += 1
                continue

            # Save each clip
            for clip_idx, (clip_frames, clip_actions) in enumerate(clips):
                clip_id = f"recon_{traj_name}_clip_{clip_idx:03d}"

                # Save video
                video_path = video_dir / f"{clip_id}.mp4"
                frames_to_video(clip_frames, video_path, fps)

                # Save actions
                action_path = action_dir / f"{clip_id}.npy"
                np.save(action_path, clip_actions)

                # Add to metadata
                duration = len(clip_frames) / fps
                metadata[clip_id] = {
                    "source": "recon",
                    "original_trajectory": traj_name,
                    "original_file": hdf5_file.name,
                    "clip_index": clip_idx,
                    "duration": duration,
                    "fps": fps,
                    "num_frames": len(clip_frames),
                    "embodiment": "wheeled",
                    "robot_type": "differential_drive",
                    "action_dim": 3,
                    "action_names": ["vx", "vy", "yaw_rate"],
                    "action_units": ["m/s", "m/s", "rad/s"],
                    "video_path": f"videos/{clip_id}.mp4",
                    "action_path": f"actions/{clip_id}.npy"
                }

                clip_count += 1

        except ValueError as e:
            error_type = "ValidationError"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            if error_count < 5:
                print(f"\nValidation error in {traj_name}: {e}")
            error_count += 1
            continue
        except RuntimeError as e:
            error_type = "JPEGDecodeError"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            if error_count < 5:
                print(f"\nJPEG decode error in {traj_name}: {e}")
            error_count += 1
            continue
        except Exception as e:
            error_type = "UnexpectedError"
            error_details[error_type] = error_details.get(error_type, 0) + 1
            if error_count < 5:
                print(f"\nUnexpected error in {traj_name}: {e}")
            error_count += 1
            continue

    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Total clips created: {clip_count}")
    print(f"  Trajectories skipped (too short): {skipped_count}")
    print(f"  Trajectories with errors: {error_count}")
    if error_details:
        print(f"  Error breakdown:")
        for error_type, count in error_details.items():
            print(f"    - {error_type}: {count}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert RECON dataset to VideotoNav format")
    parser.add_argument("--recon_dir", type=str, required=True,
                       help="Path to RECON dataset root directory")
    parser.add_argument("--output_dir", type=str, default="../dataset",
                       help="Output directory for converted dataset")
    parser.add_argument("--min_duration", type=float, default=2.0,
                       help="Minimum clip duration in seconds")
    parser.add_argument("--max_duration", type=float, default=12.0,
                       help="Maximum clip duration in seconds")
    parser.add_argument("--fps", type=int, default=16,
                       help="Target frames per second")
    parser.add_argument("--source_fps", type=int, default=10,
                       help="RECON source FPS (default: 10)")

    args = parser.parse_args()

    recon_dir = Path(args.recon_dir)
    output_dir = Path(args.output_dir)

    if not recon_dir.exists():
        raise FileNotFoundError(f"RECON directory not found: {recon_dir}")

    convert_recon_dataset(
        recon_dir=recon_dir,
        output_dir=output_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        fps=args.fps,
        source_fps=args.source_fps
    )


if __name__ == "__main__":
    main()
