#!/usr/bin/env python3
"""
Run Mode 1 simulation validation for a drone task.

Pipeline:
  1. Read task.json + predicted_actions.csv
  2. Run drone_sim.py --csv-replay in Isaac Sim (headless)
  3. Convert recorded frames → sim_video.mp4
  4. Plot predicted vs simulated vs GT trajectory
  5. Compute metrics: SR, ATE, SPL, final error
  6. Save everything to task_dir/mode1_sim/

Usage:
    python run_mode1_validation.py --task drone/tasks/task_01_wh_forward_5m
    python run_mode1_validation.py --task drone/tasks/task_01_wh_forward_5m --no-headless
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ISAAC_PYTHON = Path(os.getenv("ISAAC_SIM_PYTHON", "/opt/isaacsim/python.sh"))
DRONE_SIM = SCRIPT_DIR / "drone" / "drone_sim.py"


def read_task(task_dir):
    """Load task.json."""
    with open(task_dir / "task.json") as f:
        return json.load(f)


def read_prediction_info(task_dir):
    """Load prediction_info.json."""
    with open(task_dir / "prediction_info.json") as f:
        return json.load(f)


def run_simulation(task_dir, task, pred_info, headless=True):
    """Run drone_sim.py with CSV replay. Returns output_dir path."""
    output_dir = task_dir / "mode1_sim"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = task_dir / "predicted_actions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No predicted_actions.csv in {task_dir}")

    # Use the effective FPS from the prediction
    fps = int(round(pred_info.get("effective_fps", 24)))

    # Start position from task
    start = task["start"]
    env = task.get("environment", "warehouse")

    cmd = [
        str(ISAAC_PYTHON), str(DRONE_SIM),
        "--env", env,
        "--csv-replay", str(csv_path),
        "--fps", str(fps),
        "--start-x", str(start["x"]),
        "--start-y", str(start["y"]),
        "--start-z", str(start["z"]),
        "--start-heading", str(start["heading_deg"]),
        "--output-dir", str(output_dir),
    ]
    if headless:
        cmd.append("--headless")

    print(f"  Running Isaac Sim (headless={headless}, fps={fps})...")
    print(f"  Start: ({start['x']}, {start['y']}, {start['z']}, {start['heading_deg']}deg)")
    print(f"  CMD: {' '.join(cmd[-8:])}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=600,
    )
    elapsed = time.time() - t0

    # Print sim output (last part)
    for line in result.stdout.splitlines()[-15:]:
        print(f"    [sim] {line}")
    if result.returncode != 0:
        err = result.stderr[-1000:] if result.stderr else ""
        print(f"  Sim stderr: {err}")
        raise RuntimeError(f"Simulation failed (exit {result.returncode})")

    print(f"  Simulation done in {elapsed:.1f}s")
    return output_dir


def frames_to_video(frames_dir, video_path, fps=24):
    """Convert PNG frames to MP4 using ffmpeg."""
    if not frames_dir.exists():
        print(f"  WARNING: No frames directory at {frames_dir}")
        return False

    frame_count = len(list(frames_dir.glob("frame_*.png")))
    if frame_count == 0:
        print(f"  WARNING: No frames found in {frames_dir}")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  ffmpeg failed: {result.stderr[-300:]}")
        return False

    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  Video: {video_path.name} ({frame_count} frames, {size_mb:.1f} MB)")
    return True


def read_sim_trajectory(csv_path):
    """Read simulation trajectory CSV → dict of arrays."""
    data = {"frame": [], "t": [], "vx": [], "vy": [], "yaw_rate": [],
            "vz": [], "x": [], "y": [], "z": [], "heading": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                if key in row:
                    data[key].append(float(row[key]))
    return {k: np.array(v) for k, v in data.items()}


def compute_metrics(task, sim_traj, pred_traj):
    """Compute SR, ATE, SPL, final error."""
    start = task["start"]
    goal = task["goal"]
    success_radius = task.get("success_radius_m", 0.5)

    # Final sim position (world frame)
    sim_final_x = sim_traj["x"][-1] if len(sim_traj["x"]) else start["x"]
    sim_final_y = sim_traj["y"][-1] if len(sim_traj["y"]) else start["y"]

    # GT displacement
    gt_dx = goal["x"] - start["x"]
    gt_dy = goal["y"] - start["y"]
    gt_dist = math.sqrt(gt_dx**2 + gt_dy**2)

    # Final error (2D distance from goal)
    final_err_x = sim_final_x - goal["x"]
    final_err_y = sim_final_y - goal["y"]
    final_error = math.sqrt(final_err_x**2 + final_err_y**2)

    # Success
    success = final_error <= success_radius

    # Path length (sim)
    if len(sim_traj["x"]) > 1:
        dx = np.diff(sim_traj["x"])
        dy = np.diff(sim_traj["y"])
        sim_path_len = float(np.sum(np.sqrt(dx**2 + dy**2)))
    else:
        sim_path_len = 0.0

    # SPL = success * gt_dist / max(sim_path_len, gt_dist)
    spl = 0.0
    if success and gt_dist > 0:
        spl = gt_dist / max(sim_path_len, gt_dist)

    # ATE: mean distance from straight-line GT trajectory
    # GT trajectory: linear interpolation from start to goal
    N = len(sim_traj["x"])
    if N > 0 and gt_dist > 0:
        fracs = np.linspace(0, 1, N)
        gt_x = start["x"] + fracs * gt_dx
        gt_y = start["y"] + fracs * gt_dy
        errors = np.sqrt((sim_traj["x"] - gt_x)**2 + (sim_traj["y"] - gt_y)**2)
        ate = float(np.mean(errors))
    else:
        ate = 0.0

    return {
        "success": success,
        "success_radius_m": success_radius,
        "final_error_m": round(final_error, 4),
        "spl": round(spl, 4),
        "ate_m": round(ate, 4),
        "gt_distance_m": round(gt_dist, 4),
        "sim_path_length_m": round(sim_path_len, 4),
        "sim_final_position": {
            "x": round(float(sim_final_x), 4),
            "y": round(float(sim_final_y), 4),
        },
        "goal_position": {"x": goal["x"], "y": goal["y"]},
        "sim_frames": N,
    }


def make_validation_plot(task_dir, task, sim_traj, metrics, pred_info):
    """Plot predicted vs simulated vs GT trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = task["start"]
    goal = task["goal"]

    # Load predicted trajectory
    pred_traj_path = task_dir / "predicted_trajectory.npy"
    pred_actions_path = task_dir / "predicted_actions.npy"

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"Mode 1 Simulation Validation — {task['task_id']} — {task['name']}\n"
        f"{task['description']}",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Top-down trajectory comparison ──
    ax = axes[0]

    # GT line (start → goal)
    ax.plot([start["x"], goal["x"]], [start["y"], goal["y"]],
            "k--", linewidth=2, alpha=0.5, label="GT path", zorder=1)

    # Predicted trajectory (from FlowDiT, shifted to start position)
    if pred_traj_path.exists():
        pred_traj = np.load(str(pred_traj_path))
        pred_x = pred_traj[:, 0] + start["x"]
        pred_y = pred_traj[:, 1] + start["y"]
        ax.plot(pred_x, pred_y, color="#2563eb", linewidth=1.5,
                alpha=0.7, label="predicted", zorder=2)

    # Simulated trajectory
    if len(sim_traj["x"]) > 0:
        ax.plot(sim_traj["x"], sim_traj["y"], color="#dc2626",
                linewidth=2, label="simulated", zorder=3)

    # Markers
    ax.scatter(start["x"], start["y"], color="#16a34a", s=120,
               marker="o", zorder=5, label="start")
    ax.scatter(goal["x"], goal["y"], color="#f59e0b", s=120,
               marker="*", zorder=5, label="goal")
    if len(sim_traj["x"]) > 0:
        ax.scatter(sim_traj["x"][-1], sim_traj["y"][-1], color="#dc2626",
                   s=100, marker="x", zorder=5, label="sim end")

    # Success circle around goal
    circle = plt.Circle((goal["x"], goal["y"]),
                         task.get("success_radius_m", 0.5),
                         fill=False, color="#f59e0b", linestyle=":",
                         linewidth=1.5, label=f"radius={task.get('success_radius_m', 0.5)}m")
    ax.add_patch(circle)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Comparison (top-down)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    # ── Panel 2: Velocity profile from sim ──
    ax = axes[1]
    if len(sim_traj["t"]) > 0:
        ax.plot(sim_traj["t"], sim_traj["vx"], label="vx", color="#ef4444")
        ax.plot(sim_traj["t"], sim_traj["vy"], label="vy", color="#10b981")
        ax.plot(sim_traj["t"], sim_traj["yaw_rate"], label="yaw_rate", color="#3b82f6")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title("Sim Velocity Commands")
    ax.legend()
    ax.grid(alpha=0.25)

    # ── Panel 3: Metrics summary ──
    ax = axes[2]
    ax.axis("off")
    sr_str = "YES" if metrics["success"] else "NO"
    sr_color = "#16a34a" if metrics["success"] else "#dc2626"
    summary = (
        f"FlowDiT V2+ Mode 1 → Isaac Sim\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Success: {sr_str}\n"
        f"Final Error: {metrics['final_error_m']:.3f} m\n"
        f"Success Radius: {metrics['success_radius_m']} m\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"SPL: {metrics['spl']:.3f}\n"
        f"ATE: {metrics['ate_m']:.3f} m\n"
        f"GT Distance: {metrics['gt_distance_m']:.2f} m\n"
        f"Sim Path Length: {metrics['sim_path_length_m']:.3f} m\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Sim Final: ({metrics['sim_final_position']['x']:.3f}, "
        f"{metrics['sim_final_position']['y']:.3f})\n"
        f"Goal: ({goal['x']:.1f}, {goal['y']:.1f})\n"
        f"Sim Frames: {metrics['sim_frames']}\n"
        f"FPS: {pred_info.get('effective_fps', 24):.0f}"
    )
    ax.text(
        0.05, 0.95, summary,
        va="top", fontfamily="monospace", fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#dbeafe" if metrics["success"] else "#fee2e2",
              "alpha": 0.7},
        transform=ax.transAxes,
    )

    plt.tight_layout()
    out_path = task_dir / "mode1_sim" / "validation_plot.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: validation_plot.png")


def main():
    parser = argparse.ArgumentParser(description="Run Mode 1 sim validation")
    parser.add_argument("--task", required=True,
                        help="Task directory (relative to Simulator/)")
    parser.add_argument("--no-headless", action="store_true",
                        help="Run with GUI (default: headless)")
    args = parser.parse_args()

    task_dir = SCRIPT_DIR / args.task
    if not task_dir.exists():
        print(f"ERROR: Task directory not found: {task_dir}")
        sys.exit(1)

    task = read_task(task_dir)
    pred_info = read_prediction_info(task_dir)

    print(f"\n{'='*65}")
    print(f" Mode 1 Simulation Validation")
    print(f"{'='*65}")
    print(f"  Task: {task['task_id']} — {task['name']}")
    print(f"  Description: {task['description']}")
    print(f"  Environment: {task.get('environment', 'warehouse')}")
    print(f"  Start: ({task['start']['x']}, {task['start']['y']}, {task['start']['z']})")
    print(f"  Goal:  ({task['goal']['x']}, {task['goal']['y']}, {task['goal']['z']})")
    print(f"  Success radius: {task.get('success_radius_m', 0.5)}m")
    print(f"{'='*65}\n")

    # Step 1: Run simulation
    headless = not args.no_headless
    output_dir = run_simulation(task_dir, task, pred_info, headless=headless)

    # Step 2: Read sim trajectory
    traj_csv = output_dir / "trajectory.csv"
    if not traj_csv.exists():
        print(f"  ERROR: No trajectory.csv produced by simulation")
        sys.exit(1)
    sim_traj = read_sim_trajectory(traj_csv)
    print(f"  Sim trajectory: {len(sim_traj['x'])} points")

    # Step 3: Convert frames → video
    fps = int(round(pred_info.get("effective_fps", 24)))
    frames_dir = output_dir / "frames"
    video_path = output_dir / "sim_video.mp4"
    frames_to_video(frames_dir, video_path, fps=fps)

    # Step 4: Compute metrics
    pred_traj = None
    pred_traj_path = task_dir / "predicted_trajectory.npy"
    if pred_traj_path.exists():
        pred_traj = np.load(str(pred_traj_path))
    metrics = compute_metrics(task, sim_traj, pred_traj)

    print(f"\n  --- Metrics ---")
    print(f"  Success:      {'YES' if metrics['success'] else 'NO'}")
    print(f"  Final Error:  {metrics['final_error_m']:.3f} m")
    print(f"  SPL:          {metrics['spl']:.3f}")
    print(f"  ATE:          {metrics['ate_m']:.3f} m")
    print(f"  GT Distance:  {metrics['gt_distance_m']:.2f} m")
    print(f"  Path Length:  {metrics['sim_path_length_m']:.3f} m")

    # Step 5: Save metrics
    metrics["task_id"] = task["task_id"]
    metrics["task_name"] = task["name"]
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: metrics.json")

    # Step 6: Validation plot
    make_validation_plot(task_dir, task, sim_traj, metrics, pred_info)

    print(f"\n{'='*65}")
    print(f" VALIDATION COMPLETE — {task['task_id']}")
    print(f"{'='*65}")
    sr = "PASS" if metrics["success"] else "FAIL"
    print(f"  Result: {sr} (error={metrics['final_error_m']:.3f}m, "
          f"radius={metrics['success_radius_m']}m)")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
