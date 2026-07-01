#!/usr/bin/env python3
"""
Run FlowAgent experiment pipeline for drone tasks.

Pipeline:
    1. Load task.json from task folder
    2. Run FlowDiT V2+ inference on goal_video.mp4 → predicted_actions.npy
    3. Convert .npy → .csv for drone_sim.py replay
    4. Run drone_sim.py headless replay in Isaac Sim
    5. Compute metrics (SR, SPL, ATE, Final Error, Path Length)
    6. Save results.json

Usage:
    python run_experiment.py --task drone/tasks/task_01_wh_forward_5m
    python run_experiment.py --all                         # run all tasks with videos
    python run_experiment.py --task drone/tasks/task_07_wh_gate --skip-inference
    python run_experiment.py --all --env warehouse

Requires two Python environments:
    - FlowDiT:   flowdit_v3_humanoid_inference/.venv/bin/python3
    - Isaac Sim:  isaacsim/_build/linux-x86_64/release/python.sh
"""

import os
import sys
import json
import math
import csv
import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DRONE_TASKS_DIR = SCRIPT_DIR / "drone" / "tasks"
DRONE_SIM = SCRIPT_DIR / "drone" / "drone_sim.py"

FLOWDIT_DIR = Path(os.getenv(
    "FLOWDIT_DIR",
    str(Path(__file__).resolve().parents[1] / "part2_navigation" / "flow_constrained_v2")
))
FLOWDIT_PYTHON = Path(os.getenv("FLOWDIT_PYTHON", sys.executable))
FLOWDIT_CHECKPOINT = Path(os.getenv(
    "FLOWDIT_CHECKPOINT",
    str(FLOWDIT_DIR / "checkpoints" / "best.pth")
))

ISAAC_PYTHON = Path(os.getenv("ISAAC_SIM_PYTHON", "/opt/isaacsim/python.sh"))

ACTION_FPS = 16
SIM_HZ = 60


# ── FlowDiT inference (subprocess) ──────────────────────────────────────────

FLOWDIT_INFERENCE_SCRIPT = """
import sys, json, numpy as np
sys.path.insert(0, "{flowdit_dir}")

import torch
import cv2
from models.flowdit_v2_plus import create_flowdit_v2_plus

video_path = sys.argv[1]
checkpoint_path = sys.argv[2]
output_path = sys.argv[3]

# Load video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 16
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frames.append(frame)
cap.release()

if not frames:
    print(json.dumps({{"error": "No frames in video"}}))
    sys.exit(1)

video = np.stack(frames).astype(np.float32) / 255.0
print(f"Video: {{video.shape[0]}} frames @ {{fps}} fps", flush=True)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_flowdit_v2_plus(use_raft=False, device=device)
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# Inference
with torch.no_grad():
    results = model.predict_full_trajectory(video, video_fps=fps)

velocities = results["velocities"]
if isinstance(velocities, torch.Tensor):
    velocities = velocities.cpu().numpy()

np.save(output_path, velocities.astype(np.float32))

info = {{
    "n_frames": int(velocities.shape[0]),
    "action_dim": int(velocities.shape[1]),
    "mean_speed": float(np.mean(np.linalg.norm(velocities[:, :2], axis=1))),
    "total_distance": float(results.get("total_distance", 0)),
    "video_fps": fps,
}}
print("FLOWDIT_RESULT:" + json.dumps(info), flush=True)
"""


def run_flowdit_inference(video_path: Path, output_npy: Path) -> dict:
    """Run FlowDiT inference as a subprocess with its own Python env."""
    script_content = FLOWDIT_INFERENCE_SCRIPT.format(flowdit_dir=str(FLOWDIT_DIR))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        tmp_script = f.name

    try:
        result = subprocess.run(
            [str(FLOWDIT_PYTHON), tmp_script,
             str(video_path), str(FLOWDIT_CHECKPOINT), str(output_npy)],
            capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            print(f"  FlowDiT stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"FlowDiT inference failed (exit {result.returncode})")

        # Parse result from stdout
        info = {}
        for line in result.stdout.splitlines():
            if line.startswith("FLOWDIT_RESULT:"):
                info = json.loads(line[len("FLOWDIT_RESULT:"):])
            else:
                print(f"  [FlowDiT] {line}")

        if not output_npy.exists():
            raise RuntimeError("FlowDiT did not produce output .npy")

        return info

    finally:
        Path(tmp_script).unlink(missing_ok=True)


# ── Convert .npy actions → .csv for drone_sim.py ────────────────────────────

def actions_to_csv(npy_path: Path, csv_path: Path):
    """Convert [T, 3] npy (vx, vy, yaw_rate) → CSV with vz_m_s=0 column."""
    actions = np.load(str(npy_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vx_m_s", "vy_m_s", "yaw_rate_rad_s", "vz_m_s"])
        for row in actions:
            writer.writerow([
                f"{float(row[0]):.6f}",
                f"{float(row[1]):.6f}",
                f"{float(row[2]):.6f}",
                "0.000000",
            ])
    return len(actions)


# ── Isaac Sim replay (subprocess) ────────────────────────────────────────────

def run_isaac_replay(csv_path: Path, env: str, num_steps: int = 0) -> Path:
    """Run drone_sim.py headless CSV replay, return recording directory."""
    cmd = [
        str(ISAAC_PYTHON), str(DRONE_SIM),
        "--env", env,
        "--headless",
        "--record",
        "--csv-replay", str(csv_path),
    ]
    if num_steps > 0:
        cmd += ["--num-steps", str(num_steps)]

    print(f"  Running Isaac Sim replay...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Isaac Sim stderr (last 2000 chars):\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Isaac Sim replay failed (exit {result.returncode})")

    # Find the recording directory from output
    rec_dir = None
    for line in result.stdout.splitlines():
        print(f"  [IsaacSim] {line}")
        if "Recording saved:" in line:
            # e.g. "Recording saved: /path/to/recordings/20260303_..."
            parts = line.split("Recording saved:")
            if len(parts) > 1:
                rec_dir = Path(parts[1].strip())

    # Fallback: find most recent recording
    if rec_dir is None or not rec_dir.exists():
        rec_root = SCRIPT_DIR / "recordings"
        if rec_root.exists():
            dirs = sorted(rec_root.iterdir(), key=lambda d: d.name, reverse=True)
            drone_dirs = [d for d in dirs if "drone" in d.name]
            if drone_dirs:
                rec_dir = drone_dirs[0]

    if rec_dir is None or not rec_dir.exists():
        raise RuntimeError("Could not find replay recording directory")

    return rec_dir


# ── Load replay trajectory ───────────────────────────────────────────────────

def load_trajectory(csv_path: Path) -> dict:
    """Load trajectory.csv from a recording session."""
    xs, ys, zs, hs = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            hs.append(float(row["heading"]))
    return {
        "x": np.array(xs), "y": np.array(ys),
        "z": np.array(zs), "heading": np.array(hs),
    }


# ── Metrics computation ─────────────────────────────────────────────────────

def compute_metrics(pred_traj: dict, task: dict) -> dict:
    """Compute SR, SPL, ATE, Final Error against task start/goal."""
    start = task["start"]
    goal = task["goal"]
    radius = task["success_radius_m"]

    # Final position from replay
    final_x = float(pred_traj["x"][-1])
    final_y = float(pred_traj["y"][-1])
    final_z = float(pred_traj["z"][-1])

    # Goal in world frame (drone starts at origin, task start is the origin)
    # drone_sim.py starts drone at (0,0,z), so goal relative to start:
    goal_dx = goal["x"] - start["x"]
    goal_dy = goal["y"] - start["y"]
    goal_dz = goal["z"] - start["z"]

    # Final error (3D)
    final_err_3d = math.sqrt(
        (final_x - goal_dx)**2 +
        (final_y - goal_dy)**2 +
        (final_z - goal_dz)**2
    )
    # Final error (2D, ignoring altitude)
    final_err_2d = math.sqrt(
        (final_x - goal_dx)**2 +
        (final_y - goal_dy)**2
    )

    # Success
    success = final_err_2d <= radius

    # Ground-truth path length (straight line from start to goal)
    gt_path_len = math.sqrt(goal_dx**2 + goal_dy**2 + goal_dz**2)

    # Predicted path length
    dx = np.diff(pred_traj["x"])
    dy = np.diff(pred_traj["y"])
    dz = np.diff(pred_traj["z"])
    pred_path_len = float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))

    # SPL
    if success and pred_path_len > 0:
        spl = gt_path_len / max(pred_path_len, gt_path_len)
    else:
        spl = 0.0

    # ATE: average distance from predicted trajectory to straight-line GT
    n = len(pred_traj["x"])
    t_frac = np.linspace(0, 1, n)
    gt_x = t_frac * goal_dx
    gt_y = t_frac * goal_dy
    gt_z = start["z"] + t_frac * goal_dz  # interpolate altitude
    gt_z_rel = t_frac * goal_dz

    errors = np.sqrt(
        (pred_traj["x"] - gt_x)**2 +
        (pred_traj["y"] - gt_y)**2 +
        (pred_traj["z"] - (start["z"] + gt_z_rel))**2
    )
    ate = float(np.mean(errors))

    # Direction accuracy: heading vs direction to goal
    goal_heading = math.atan2(goal_dy, goal_dx)
    heading_errors = np.cos(pred_traj["heading"] - goal_heading)
    dir_acc = float(np.mean(heading_errors > 0.75))

    return {
        "success": success,
        "final_error_2d_m": round(final_err_2d, 4),
        "final_error_3d_m": round(final_err_3d, 4),
        "ate_m": round(ate, 4),
        "spl": round(spl, 4),
        "direction_accuracy": round(dir_acc, 4),
        "pred_path_length_m": round(pred_path_len, 4),
        "gt_path_length_m": round(gt_path_len, 4),
        "success_radius_m": radius,
        "n_steps": n,
        "final_position": {"x": round(final_x, 4), "y": round(final_y, 4), "z": round(final_z, 4)},
        "goal_relative": {"x": round(goal_dx, 4), "y": round(goal_dy, 4), "z": round(goal_dz, 4)},
    }


# ── Task processing ─────────────────────────────────────────────────────────

def process_task(task_dir: Path, skip_inference: bool = False) -> dict:
    """Run full pipeline for one task. Returns results dict."""
    task_file = task_dir / "task.json"
    with open(task_file) as f:
        task = json.load(f)

    name = task["name"]
    env = task["environment"]
    print(f"\n{'─'*65}")
    print(f"  Task {task['task_id']}: {name} ({env})")
    print(f"{'─'*65}")

    # Check for video
    video_dir = task_dir / "video"
    video_files = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []
    if not video_files and not skip_inference:
        npy_file = task_dir / "predicted_actions.npy"
        if not npy_file.exists():
            print(f"  [SKIP] No video in {video_dir} and no predicted_actions.npy")
            return {"status": "skipped", "reason": "no_video"}

    # Step 1: FlowDiT inference
    npy_path = task_dir / "predicted_actions.npy"
    inference_info = {}

    if skip_inference and npy_path.exists():
        print(f"  Skipping inference — using existing {npy_path.name}")
    elif video_files:
        video_path = video_files[0]  # Use first .mp4 found
        print(f"  Video: {video_path.name}")
        print(f"  Running FlowDiT inference...", flush=True)
        inference_info = run_flowdit_inference(video_path, npy_path)
        print(f"  Predicted {inference_info.get('n_frames', '?')} actions")
    elif not npy_path.exists():
        print(f"  [SKIP] No video and no cached actions")
        return {"status": "skipped", "reason": "no_video"}

    # Step 2: Convert .npy → .csv
    csv_path = task_dir / "predicted_actions.csv"
    n_actions = actions_to_csv(npy_path, csv_path)
    print(f"  Actions CSV: {n_actions} rows")

    # Step 3: Isaac Sim replay
    rec_dir = run_isaac_replay(csv_path, env, num_steps=n_actions + 10)
    print(f"  Recording: {rec_dir.name}")

    # Step 4: Load trajectory and compute metrics
    traj_csv = rec_dir / "trajectory.csv"
    if not traj_csv.exists():
        print(f"  [FAIL] No trajectory.csv in {rec_dir}")
        return {"status": "failed", "reason": "no_trajectory"}

    pred_traj = load_trajectory(traj_csv)
    metrics = compute_metrics(pred_traj, task)

    # Step 5: Save results
    results = {
        "task_id": task["task_id"],
        "name": name,
        "environment": env,
        "task_type": task["task_type"],
        "status": "completed",
        "metrics": metrics,
        "inference": inference_info,
        "replay_dir": str(rec_dir),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = task_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print metrics
    m = metrics
    status = "PASS" if m["success"] else "FAIL"
    print(f"\n  [{status}] Final Error: {m['final_error_2d_m']:.3f}m "
          f"(radius: {m['success_radius_m']}m)")
    print(f"  ATE: {m['ate_m']:.3f}m | SPL: {m['spl']:.3f} | "
          f"Dir Acc: {m['direction_accuracy']:.1%}")
    print(f"  Path: pred={m['pred_path_length_m']:.2f}m gt={m['gt_path_length_m']:.2f}m")

    return results


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    """Print aggregate metrics across all tasks."""
    completed = [r for r in all_results if r.get("status") == "completed"]
    skipped = [r for r in all_results if r.get("status") == "skipped"]
    failed = [r for r in all_results if r.get("status") == "failed"]

    print(f"\n{'='*65}")
    print(f" EXPERIMENT SUMMARY")
    print(f"{'='*65}")
    print(f"  Completed: {len(completed)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")

    if not completed:
        print(f"{'='*65}")
        return

    successes = [r for r in completed if r["metrics"]["success"]]
    sr = len(successes) / len(completed)
    avg_ate = np.mean([r["metrics"]["ate_m"] for r in completed])
    avg_spl = np.mean([r["metrics"]["spl"] for r in completed])
    avg_final = np.mean([r["metrics"]["final_error_2d_m"] for r in completed])
    avg_dir = np.mean([r["metrics"]["direction_accuracy"] for r in completed])

    print(f"\n  Aggregate Metrics ({len(completed)} tasks):")
    print(f"  {'─'*40}")
    print(f"  SR:              {sr:.1%} ({len(successes)}/{len(completed)})")
    print(f"  SPL:             {avg_spl:.3f}")
    print(f"  ATE:             {avg_ate:.3f} m")
    print(f"  Final Error:     {avg_final:.3f} m")
    print(f"  Dir Accuracy:    {avg_dir:.1%}")

    # Per-task breakdown
    print(f"\n  Per-Task Results:")
    print(f"  {'Task':<25} {'SR':>4} {'Final':>7} {'ATE':>7} {'SPL':>6}")
    print(f"  {'─'*55}")
    for r in completed:
        m = r["metrics"]
        ok = "Y" if m["success"] else "N"
        print(f"  {r['name']:<25} {ok:>4} {m['final_error_2d_m']:>6.3f}m "
              f"{m['ate_m']:>6.3f}m {m['spl']:>5.3f}")

    print(f"{'='*65}")

    # Save aggregate results
    agg_path = SCRIPT_DIR / "experiment_results.json"
    agg = {
        "timestamp": datetime.now().isoformat(),
        "n_tasks": len(all_results),
        "n_completed": len(completed),
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "aggregate": {
            "success_rate": round(sr, 4),
            "mean_spl": round(float(avg_spl), 4),
            "mean_ate_m": round(float(avg_ate), 4),
            "mean_final_error_m": round(float(avg_final), 4),
            "mean_direction_accuracy": round(float(avg_dir), 4),
        },
        "per_task": {r["task_id"]: r["metrics"] for r in completed},
    }
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Saved: {agg_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run FlowAgent drone experiment")
    parser.add_argument("--task", type=str, default=None,
                        help="Path to a single task folder")
    parser.add_argument("--all", action="store_true",
                        help="Run all tasks that have goal videos")
    parser.add_argument("--env", choices=["warehouse", "hospital"], default=None,
                        help="Filter tasks by environment")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip FlowDiT inference, use cached predicted_actions.npy")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.print_help()
        sys.exit(1)

    # Validate external dependencies
    if not FLOWDIT_PYTHON.exists():
        print(f"WARNING: FlowDiT Python not found: {FLOWDIT_PYTHON}")
    if not ISAAC_PYTHON.exists():
        print(f"ERROR: Isaac Sim python.sh not found: {ISAAC_PYTHON}")
        sys.exit(1)
    if not FLOWDIT_CHECKPOINT.exists() and not args.skip_inference:
        print(f"WARNING: FlowDiT checkpoint not found: {FLOWDIT_CHECKPOINT}")

    print(f"{'='*65}")
    print(f" FlowAgent Drone Experiment Pipeline")
    print(f"{'='*65}")
    print(f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  FlowDiT:    {FLOWDIT_CHECKPOINT.name}")
    print(f"  Isaac Sim:  {ISAAC_PYTHON}")
    print(f"{'='*65}")

    all_results = []

    if args.task:
        task_dir = Path(args.task)
        if not task_dir.is_absolute():
            task_dir = SCRIPT_DIR / task_dir
        if not (task_dir / "task.json").exists():
            print(f"ERROR: No task.json in {task_dir}")
            sys.exit(1)
        result = process_task(task_dir, skip_inference=args.skip_inference)
        all_results.append(result)
    else:
        # Run all tasks
        for task_dir in sorted(DRONE_TASKS_DIR.iterdir()):
            if not task_dir.is_dir():
                continue
            if not (task_dir / "task.json").exists():
                continue

            # Filter by environment
            if args.env:
                with open(task_dir / "task.json") as f:
                    t = json.load(f)
                if t["environment"] != args.env:
                    continue

            result = process_task(task_dir, skip_inference=args.skip_inference)
            all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()
