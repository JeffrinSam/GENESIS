"""
Flow-Constrained Video-to-Action Model for Cross-Embodiment Navigation

Author: Jeffrin Sam
Institution: Skoltech
Year: 2025
License: MIT

Description: Convert TartanAir dataset to VideotoNav format
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation


def load_tartanair_trajectory(traj_dir: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load frames and poses from a TartanAir trajectory.

    Args:
        traj_dir: Path to trajectory directory (e.g., P000)

    Returns:
        frames: List of RGB frames [H, W, 3]
        poses: Pose array [T, 7] with (x, y, z, qx, qy, qz, qw)
    """
    # Load RGB frames
    image_dir = traj_dir / "image_left"
    if not image_dir.exists():
        image_dir = traj_dir / "image_lcam_front"  # Alternative naming

    frame_paths = sorted(image_dir.glob("*.png"))

    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # Load poses
    pose_path = traj_dir / "pose_left.txt"
    if not pose_path.exists():
        pose_path = traj_dir / "pose_lcam_front.txt"  # Alternative naming

    poses = np.loadtxt(pose_path)  # [T, 7]

    return frames, poses


def poses_to_velocities(poses: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """Convert poses to velocity commands.

    Args:
        poses: Pose array [T, 7] with (x, y, z, qx, qy, qz, qw)
        dt: Time step between frames (seconds)

    Returns:
        velocities: Velocity array [T, 4] with (vx, vy, vz, yaw_rate)
    """
    T = poses.shape[0]
    velocities = np.zeros((T, 4))

    for t in range(T - 1):
        # Position difference
        pos_current = poses[t, :3]
        pos_next = poses[t + 1, :3]

        # Rotation (quaternion)
        quat_current = poses[t, 3:]
        quat_next = poses[t + 1, 3:]

        # Linear velocity in world frame
        linear_vel_world = (pos_next - pos_current) / dt

        # Convert to body frame
        rot_current = Rotation.from_quat(quat_current)
        linear_vel_body = rot_current.inv().apply(linear_vel_world)

        # Angular velocity (yaw rate)
        rot_diff = Rotation.from_quat(quat_next) * rot_current.inv()
        euler_diff = rot_diff.as_euler('xyz')
        yaw_rate = euler_diff[2] / dt  # Only yaw component

        velocities[t] = [linear_vel_body[0], linear_vel_body[1], linear_vel_body[2], yaw_rate]

    # Last frame: copy previous velocity
    velocities[-1] = velocities[-2]

    return velocities


def resample_trajectory(frames: List[np.ndarray], velocities: np.ndarray,
                       target_fps: int = 16, source_fps: int = 10) -> Tuple[List[np.ndarray], np.ndarray]:
    """Resample trajectory to target FPS.

    TartanAir is typically recorded at 10 FPS.

    Args:
        frames: Original frames
        velocities: Original velocities [T, 4]
        target_fps: Target frames per second
        source_fps: Source frames per second

    Returns:
        resampled_frames: Resampled frames
        resampled_velocities: Resampled velocities [T', 4]
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

    # Resample velocities (linear interpolation)
    resampled_velocities = np.zeros((target_num_frames, velocities.shape[1]))
    for i in range(velocities.shape[1]):
        resampled_velocities[:, i] = np.interp(target_indices, source_indices, velocities[:, i])

    return resampled_frames, resampled_velocities


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


def clip_trajectory(frames: List[np.ndarray], velocities: np.ndarray,
                    duration: float, fps: int) -> List[Tuple[List[np.ndarray], np.ndarray]]:
    """Split trajectory into clips of specified duration.

    Args:
        frames: Full trajectory frames
        velocities: Full trajectory velocities [T, 4]
        duration: Clip duration in seconds
        fps: Frames per second

    Returns:
        clips: List of (frames, velocities) tuples
    """
    clip_frames = int(duration * fps)
    total_frames = len(frames)

    clips = []

    # If trajectory is shorter than clip duration, skip
    if total_frames < clip_frames:
        return clips

    # Split into non-overlapping clips
    for start_idx in range(0, total_frames - clip_frames + 1, clip_frames):
        end_idx = start_idx + clip_frames

        clip_frame_list = frames[start_idx:end_idx]
        clip_velocities = velocities[start_idx:end_idx]

        clips.append((clip_frame_list, clip_velocities))

    return clips


def convert_tartanair_dataset(tartanair_dir: Path, output_dir: Path,
                              duration: float = 8.0, fps: int = 16,
                              source_fps: int = 10, embodiment: str = "aerial"):
    """Convert TartanAir dataset to VideotoNav format.

    Args:
        tartanair_dir: Path to TartanAir dataset root
        output_dir: Output directory
        duration: Clip duration in seconds
        fps: Target frames per second
        source_fps: TartanAir source FPS (typically 10)
        embodiment: Robot embodiment type (aerial or wheeled)
    """
    if not tartanair_dir.exists():
        raise FileNotFoundError(f"TartanAir directory not found: {tartanair_dir}")

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

    # Find all environments and trajectories
    env_dirs = sorted([d for d in tartanair_dir.iterdir() if d.is_dir()])

    print(f"Found {len(env_dirs)} environments in {tartanair_dir}")

    clip_count = 0
    skipped_count = 0

    for env_dir in env_dirs:
        env_name = env_dir.name

        # Find difficulty levels
        difficulty_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir()])

        for difficulty_dir in difficulty_dirs:
            difficulty = difficulty_dir.name

            # Find trajectories (P000, P001, ...)
            traj_dirs = sorted([d for d in difficulty_dir.iterdir()
                              if d.is_dir() and d.name.startswith('P')])

            for traj_dir in tqdm(traj_dirs, desc=f"Converting {env_name}/{difficulty}"):
                traj_name = traj_dir.name

                try:
                    # Load trajectory
                    frames, poses = load_tartanair_trajectory(traj_dir)

                    # Convert poses to velocities
                    velocities = poses_to_velocities(poses, dt=1.0/source_fps)

                    # Resample to target FPS
                    frames, velocities = resample_trajectory(frames, velocities, fps, source_fps)

                    # Split into clips
                    clips = clip_trajectory(frames, velocities, duration, fps)

                    if len(clips) == 0:
                        skipped_count += 1
                        continue

                    # Save each clip
                    for clip_idx, (clip_frames, clip_velocities) in enumerate(clips):
                        clip_id = f"tartanair_{env_name}_{difficulty}_{traj_name}_clip_{clip_idx:03d}"

                        # Convert to wheeled format if needed (drop vz)
                        if embodiment == "wheeled":
                            # Use only vx, vy, yaw_rate
                            clip_actions = np.zeros((clip_velocities.shape[0], 3))
                            clip_actions[:, 0] = clip_velocities[:, 0]  # vx
                            clip_actions[:, 1] = clip_velocities[:, 1]  # vy
                            clip_actions[:, 2] = clip_velocities[:, 3]  # yaw_rate
                            action_dim = 3
                            action_names = ["vx", "vy", "yaw"]
                        else:
                            # Aerial: use all 4 components
                            clip_actions = clip_velocities
                            action_dim = 4
                            action_names = ["vx", "vy", "vz", "yaw"]

                        # Save video
                        video_path = video_dir / f"{clip_id}.mp4"
                        frames_to_video(clip_frames, video_path, fps)

                        # Save actions
                        action_path = action_dir / f"{clip_id}.npy"
                        np.save(action_path, clip_actions)

                        # Add to metadata
                        metadata[clip_id] = {
                            "source": "tartanair",
                            "environment": env_name,
                            "difficulty": difficulty,
                            "original_trajectory": traj_name,
                            "clip_index": clip_idx,
                            "duration": duration,
                            "fps": fps,
                            "num_frames": len(clip_frames),
                            "embodiment": embodiment,
                            "action_dim": action_dim,
                            "action_names": action_names,
                            "action_units": ["m/s"] * (action_dim - 1) + ["rad/s"],
                            "video_path": f"videos/{clip_id}.mp4",
                            "action_path": f"actions/{clip_id}.npy"
                        }

                        clip_count += 1

                except Exception as e:
                    print(f"\nError processing {env_name}/{difficulty}/{traj_name}: {e}")
                    skipped_count += 1
                    continue

    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Total clips created: {clip_count}")
    print(f"  Trajectories skipped: {skipped_count}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert TartanAir dataset to VideotoNav format")
    parser.add_argument("--tartanair_dir", type=str, required=True,
                       help="Path to TartanAir dataset root directory")
    parser.add_argument("--output_dir", type=str, default="../dataset",
                       help="Output directory for converted dataset")
    parser.add_argument("--duration", type=float, default=8.0,
                       help="Clip duration in seconds")
    parser.add_argument("--fps", type=int, default=16,
                       help="Target frames per second")
    parser.add_argument("--source_fps", type=int, default=10,
                       help="TartanAir source FPS (default: 10)")
    parser.add_argument("--embodiment", type=str, default="aerial",
                       choices=["wheeled", "aerial"],
                       help="Robot embodiment type (wheeled: 3D actions, aerial: 4D actions)")

    args = parser.parse_args()

    tartanair_dir = Path(args.tartanair_dir)
    output_dir = Path(args.output_dir)

    if not tartanair_dir.exists():
        raise FileNotFoundError(f"TartanAir directory not found: {tartanair_dir}")

    convert_tartanair_dataset(
        tartanair_dir=tartanair_dir,
        output_dir=output_dir,
        duration=args.duration,
        fps=args.fps,
        source_fps=args.source_fps,
        embodiment=args.embodiment
    )


if __name__ == "__main__":
    main()
