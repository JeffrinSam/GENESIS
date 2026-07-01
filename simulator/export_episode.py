#!/usr/bin/env python3
"""
Export recorded sessions to FlowDiT V2+ compatible dataset format.

Usage:
    python export_episode.py recordings/20260302_143000_warehouse/
    python export_episode.py recordings/20260302_143000_warehouse/ --dataset-dir dataset/
    python export_episode.py --all   # export all unprocessed sessions

Output format (matching NavigationDatasetV2Plus):
    dataset/
        videos/episode_XXXXXX.mp4   (224x224, 16fps, RGB)
        actions/episode_XXXXXX.npy  ([T, 3] — vx, vy, yaw_rate)
        metadata.json               (train/val splits)
"""

import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

SCRIPT_DIR  = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset"
REC_ROOT    = SCRIPT_DIR / "recordings"


def load_trajectory(csv_path: Path) -> dict:
    """Load trajectory.csv into structured dict."""
    frames, ts, vxs, vys, yrs, xs, ys, hs = [], [], [], [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame"]))
            ts.append(float(row["t"]))
            vxs.append(float(row["vx"]))
            vys.append(float(row["vy"]))
            yrs.append(float(row["yaw_rate"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            hs.append(float(row["heading"]))
    return {
        "frames": frames,
        "t": np.array(ts),
        "vx": np.array(vxs),
        "vy": np.array(vys),
        "yaw_rate": np.array(yrs),
        "x": np.array(xs),
        "y": np.array(ys),
        "heading": np.array(hs),
    }


def get_next_episode_id(dataset_dir: Path) -> int:
    """Find next available episode number."""
    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        return 0
    existing = sorted(videos_dir.glob("episode_*.mp4"))
    if not existing:
        return 0
    last = existing[-1].stem  # episode_000005
    return int(last.split("_")[1]) + 1


def frames_to_mp4_cv2(frames_dir: Path, output_path: Path, fps: int, resolution: tuple):
    """Create MP4 video from PNG frames using OpenCV."""
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        # Try PPM fallback
        frame_files = sorted(frames_dir.glob("frame_*.ppm"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)

    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[:2] != (resolution[1], resolution[0]):
            img = cv2.resize(img, resolution)
        writer.write(img)

    writer.release()
    return len(frame_files)


def frames_to_mp4_ffmpeg(frames_dir: Path, output_path: Path, fps: int):
    """Create MP4 from frames using ffmpeg (fallback if no OpenCV)."""
    # Check for PNG or PPM
    ext = "png" if list(frames_dir.glob("frame_*.png")) else "ppm"
    pattern = str(frames_dir / f"frame_%06d.{ext}")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale=224:224",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def export_session(session_dir: Path, dataset_dir: Path) -> str:
    """Export one recording session to dataset format. Returns episode_id string."""
    session_dir = Path(session_dir)
    dataset_dir = Path(dataset_dir)

    # Validate session
    traj_csv = session_dir / "trajectory.csv"
    frames_dir = session_dir / "frames"
    meta_path = session_dir / "metadata.json"

    if not traj_csv.exists():
        raise FileNotFoundError(f"No trajectory.csv in {session_dir}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"No frames/ directory in {session_dir}")

    # Load session metadata
    session_meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            session_meta = json.load(f)
    fps = session_meta.get("fps", 16)

    # Load trajectory
    traj = load_trajectory(traj_csv)
    n_frames = len(traj["frames"])
    if n_frames < 2:
        raise ValueError(f"Too few frames ({n_frames}) in {session_dir}")

    # Build actions array [T, 3]
    actions = np.stack([traj["vx"], traj["vy"], traj["yaw_rate"]], axis=1)  # [T, 3]

    # Determine episode ID
    videos_dir = dataset_dir / "videos"
    actions_dir = dataset_dir / "actions"
    videos_dir.mkdir(parents=True, exist_ok=True)
    actions_dir.mkdir(parents=True, exist_ok=True)

    ep_num = get_next_episode_id(dataset_dir)
    ep_id = f"episode_{ep_num:06d}"

    # Save actions as .npy
    npy_path = actions_dir / f"{ep_id}.npy"
    np.save(str(npy_path), actions.astype(np.float32))
    print(f"  Actions: {npy_path} — shape {actions.shape}")

    # Create MP4 video
    mp4_path = videos_dir / f"{ep_id}.mp4"
    if HAS_CV2:
        n = frames_to_mp4_cv2(frames_dir, mp4_path, fps, (224, 224))
        print(f"  Video:   {mp4_path} — {n} frames @ {fps} fps")
    else:
        frames_to_mp4_ffmpeg(frames_dir, mp4_path, fps)
        print(f"  Video:   {mp4_path} (via ffmpeg) @ {fps} fps")

    # Update dataset metadata
    ds_meta_path = dataset_dir / "metadata.json"
    if ds_meta_path.exists():
        with open(ds_meta_path) as f:
            ds_meta = json.load(f)
    else:
        ds_meta = {"episodes": {}, "splits": {"train": [], "val": []}}

    ds_meta["episodes"][ep_id] = {
        "source_session": session_dir.name,
        "environment": session_meta.get("environment", "unknown"),
        "robot": session_meta.get("robot", "limo"),
        "n_frames": n_frames,
        "duration_sec": round(n_frames / fps, 3),
        "fps": fps,
        "action_dim": 3,
    }

    # Update splits (80/20)
    all_eps = sorted(ds_meta["episodes"].keys())
    np.random.seed(42)
    perm = np.random.permutation(len(all_eps))
    n_train = max(1, int(0.8 * len(all_eps)))
    ds_meta["splits"]["train"] = [all_eps[i] for i in perm[:n_train]]
    ds_meta["splits"]["val"] = [all_eps[i] for i in perm[n_train:]]

    with open(ds_meta_path, "w") as f:
        json.dump(ds_meta, f, indent=2)
    print(f"  Metadata updated: {len(all_eps)} episodes "
          f"({len(ds_meta['splits']['train'])} train / {len(ds_meta['splits']['val'])} val)")

    return ep_id


def main():
    parser = argparse.ArgumentParser(description="Export recordings to FlowDiT dataset format")
    parser.add_argument("session", nargs="?", type=str,
                        help="Path to recording session directory")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR),
                        help="Output dataset directory")
    parser.add_argument("--all", action="store_true",
                        help="Export all unprocessed sessions")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    if args.all:
        # Find sessions not yet exported
        if not REC_ROOT.exists():
            print("No recordings directory found.")
            return
        ds_meta_path = dataset_dir / "metadata.json"
        exported = set()
        if ds_meta_path.exists():
            with open(ds_meta_path) as f:
                meta = json.load(f)
            exported = {v["source_session"] for v in meta.get("episodes", {}).values()}

        sessions = sorted(d for d in REC_ROOT.iterdir()
                          if d.is_dir() and d.name not in exported)
        if not sessions:
            print("All sessions already exported.")
            return
        print(f"Exporting {len(sessions)} session(s)...")
        for sess in sessions:
            print(f"\n--- {sess.name} ---")
            try:
                export_session(sess, dataset_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
    elif args.session:
        session_dir = Path(args.session)
        if not session_dir.exists():
            print(f"ERROR: {session_dir} does not exist")
            sys.exit(1)
        print(f"Exporting {session_dir.name}...")
        export_session(session_dir, dataset_dir)
        print("\nDone.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
