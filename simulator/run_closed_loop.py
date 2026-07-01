#!/usr/bin/env python3
"""
Run FlowDiT V2+ closed-loop control with Isaac Sim drone.

Architecture:
  1. Starts FlowDiT TCP server (FlowDiT venv) → warmup with goal video
  2. Starts Isaac Sim drone (Isaac Sim Python) → sends live POV frames to server
  3. FlowDiT returns velocity commands → drone applies them in real-time
  4. Loop until should_stop or max steps → compute metrics

Usage:
    python run_closed_loop.py --task drone/tasks/task_01_wh_forward_5m
    python run_closed_loop.py --all-tasks
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

SCRIPT_DIR = Path(__file__).parent

# ── Paths ─────────────────────────────────────────────────────────────────────
FLOWDIT_DIR = Path(os.getenv(
    "FLOWDIT_DIR",
    str(Path(__file__).resolve().parents[1] / "part2_navigation" / "flow_constrained_v2")
))
FLOWDIT_PYTHON = os.getenv("FLOWDIT_PYTHON", sys.executable)
FLOWDIT_CHECKPOINT = os.getenv(
    "FLOWDIT_CHECKPOINT",
    str(FLOWDIT_DIR / "checkpoints" / "best.pth")
)
FLOWDIT_SERVER = str(SCRIPT_DIR / "closed_loop" / "flowdit_server.py")

ISAAC_PYTHON = os.getenv("ISAAC_SIM_PYTHON", "/opt/isaacsim/python.sh")
DRONE_SIM = str(SCRIPT_DIR / "drone" / "drone_sim.py")

PORT = 5555


def discover_tasks(robot="drone"):
    """Find all task directories with goal videos."""
    tasks_root = SCRIPT_DIR / robot / "tasks"
    if not tasks_root.exists():
        return []
    valid = []
    for td in sorted(tasks_root.glob("task_*")):
        if (td / "task.json").exists() and (td / "video" / "goal_video.mp4").exists():
            valid.append(td)
    return valid


def start_flowdit_server(port=PORT):
    """Start FlowDiT TCP server in background. Returns subprocess."""
    cmd = [
        FLOWDIT_PYTHON, FLOWDIT_SERVER,
        "--port", str(port),
        "--checkpoint", FLOWDIT_CHECKPOINT,
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    # Wait for server ready
    print("Starting FlowDiT server...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 120:
        line = proc.stdout.readline()
        if line:
            print(f"  [server] {line.rstrip()}", flush=True)
            if "FLOWDIT_SERVER_READY" in line:
                print(f"  Server ready in {time.time()-t0:.1f}s", flush=True)
                return proc
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with code {proc.returncode}")
    raise RuntimeError("Server startup timeout (120s)")


def run_sim_closed_loop(task, video_path, output_dir, port=PORT, max_steps=200):
    """Run drone_sim in closed-loop mode."""
    start = task["start"]
    env = task.get("environment", "warehouse")

    cmd = [
        ISAAC_PYTHON, DRONE_SIM,
        "--env", env, "--headless",
        "--start-x", str(start["x"]),
        "--start-y", str(start["y"]),
        "--start-z", str(start["z"]),
        "--start-heading", str(start["heading_deg"]),
        "--output-dir", str(output_dir),
        "--no-frames",
        "--closed-loop", str(port),
        "--closed-loop-video", str(video_path),
        "--max-closed-loop-steps", str(max_steps),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    for line in result.stdout.splitlines():
        if line.strip():
            print(f"    [sim] {line.rstrip()}", flush=True)
    if result.returncode != 0:
        err = result.stderr[-1000:] if result.stderr else ""
        raise RuntimeError(f"Sim failed (exit {result.returncode}): {err}")
    return output_dir


def read_sim_trajectory(csv_path):
    """Read trajectory CSV from sim output."""
    data = {"x": [], "y": [], "z": [], "heading": [], "t": [],
            "vx": [], "vy": [], "yaw_rate": [], "vz": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                if key in row:
                    data[key].append(float(row[key]))
    return {k: np.array(v) for k, v in data.items()}


def compute_metrics(task, sim_traj):
    """Compute SR, ATE, SPL, final error."""
    start, goal = task["start"], task["goal"]
    radius = task.get("success_radius_m", 0.5)

    sim_fx = sim_traj["x"][-1] if len(sim_traj["x"]) else start["x"]
    sim_fy = sim_traj["y"][-1] if len(sim_traj["y"]) else start["y"]

    gt_dx = goal["x"] - start["x"]
    gt_dy = goal["y"] - start["y"]
    gt_dist = math.sqrt(gt_dx**2 + gt_dy**2)
    final_error = math.sqrt((sim_fx - goal["x"])**2 + (sim_fy - goal["y"])**2)
    success = final_error <= radius

    if len(sim_traj["x"]) > 1:
        sim_path = float(np.sum(np.sqrt(
            np.diff(sim_traj["x"])**2 + np.diff(sim_traj["y"])**2)))
    else:
        sim_path = 0.0

    spl = (gt_dist / max(sim_path, gt_dist)) if (success and gt_dist > 0) else 0.0

    N = len(sim_traj["x"])
    if N > 0 and gt_dist > 0:
        fracs = np.linspace(0, 1, N)
        ate = float(np.mean(np.sqrt(
            (sim_traj["x"] - (start["x"] + fracs * gt_dx))**2 +
            (sim_traj["y"] - (start["y"] + fracs * gt_dy))**2
        )))
    else:
        ate = 0.0

    return {
        "success": success,
        "final_error_m": round(final_error, 4),
        "spl": round(spl, 4),
        "ate_m": round(ate, 4),
        "gt_distance_m": round(gt_dist, 4),
        "sim_path_length_m": round(sim_path, 4),
        "sim_final": {"x": round(float(sim_fx), 4), "y": round(float(sim_fy), 4)},
    }


def make_closed_loop_plot(task, sim_traj, metrics, diag, output_path):
    """Plot closed-loop results: trajectory + diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start, goal = task["start"], task["goal"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f"Closed-Loop V2+ — {task.get('task_id', '')} — {task.get('name', '')}\n"
        f"{task.get('description', '')}",
        fontsize=14, fontweight="bold"
    )

    # Panel 1: Trajectory
    ax = axes[0]
    ax.plot([start["x"], goal["x"]], [start["y"], goal["y"]],
            "k--", linewidth=2, alpha=0.4, label="GT path")
    if len(sim_traj["x"]) > 0:
        ax.plot(sim_traj["x"], sim_traj["y"],
                color="#16a34a", linewidth=2, label="Closed-loop")
        ax.scatter(sim_traj["x"][-1], sim_traj["y"][-1],
                   color="#16a34a", s=100, marker="x", zorder=5)
    ax.scatter(start["x"], start["y"], color="black", s=150, marker="o",
               zorder=6, label="Start")
    ax.scatter(goal["x"], goal["y"], color="#f59e0b", s=150, marker="*",
               zorder=6, label="Goal")
    circle = plt.Circle((goal["x"], goal["y"]),
                         task.get("success_radius_m", 0.5),
                         fill=False, color="#f59e0b", linestyle=":", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 2: Velocity + confidence
    ax = axes[1]
    steps_data = diag.get("steps", [])
    if steps_data:
        steps = [s["step"] for s in steps_data]
        vx_vals = [s.get("vx", 0) for s in steps_data]
        vy_vals = [s.get("vy", 0) for s in steps_data]
        yr_vals = [s.get("yaw_rate", 0) for s in steps_data]
        conf_vals = [s.get("confidence", 0) for s in steps_data]

        ax.plot(steps, vx_vals, color="#ef4444", linewidth=1.2, label="vx", alpha=0.8)
        ax.plot(steps, vy_vals, color="#10b981", linewidth=1.2, label="vy", alpha=0.8)
        ax.plot(steps, yr_vals, color="#3b82f6", linewidth=1.2, label="yaw", alpha=0.8)

        ax2 = ax.twinx()
        ax2.fill_between(steps, conf_vals, alpha=0.15, color="#8b5cf6")
        ax2.plot(steps, conf_vals, color="#8b5cf6", linewidth=1, alpha=0.5,
                 label="confidence")
        ax2.set_ylabel("Confidence", color="#8b5cf6")
        ax2.set_ylim(0, 1.1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Velocity (m/s, rad/s)")
    ax.set_title("Commands & Confidence")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 3: Metrics + diagnostics summary
    ax = axes[2]
    ax.axis("off")
    sr_str = "YES" if metrics["success"] else "NO"
    n_steps = diag.get("total_steps", 0)
    stopped = diag.get("should_stop", False)

    text = (
        f"Success:         {sr_str}\n"
        f"Final Error:     {metrics['final_error_m']:.3f} m\n"
        f"SPL:             {metrics['spl']:.3f}\n"
        f"ATE:             {metrics['ate_m']:.3f} m\n"
        f"GT Distance:     {metrics['gt_distance_m']:.3f} m\n"
        f"Path Length:     {metrics['sim_path_length_m']:.3f} m\n"
        f"─────────────────────────\n"
        f"Steps:           {n_steps}\n"
        f"Stopped by model: {stopped}\n"
        f"Radius:          {task.get('success_radius_m', 0.5)} m\n"
    )

    if steps_data:
        avg_conf = np.mean([s.get("confidence", 0) for s in steps_data])
        avg_vis = np.mean([s.get("visual_similarity", 0) for s in steps_data])
        text += (
            f"Avg Confidence:  {avg_conf:.3f}\n"
            f"Avg Visual Sim:  {avg_vis:.3f}\n"
        )

    ax.text(0.05, 0.95, text, va="top", fontfamily="monospace", fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "#f0fdf4", "alpha": 0.8},
            transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_one_task(task_dir, port=PORT, max_steps=200):
    """Run closed-loop on a single task. Returns (task, metrics, diag)."""
    task_dir = Path(task_dir)
    with open(task_dir / "task.json") as f:
        task = json.load(f)

    video_path = task_dir / "video" / "goal_video.mp4"
    output_dir = task_dir / "closed_loop"
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_output = output_dir / "sim"

    # Start FlowDiT server
    server_proc = start_flowdit_server(port)

    try:
        # Run Isaac Sim with closed-loop (sim handles warmup via TCP)
        print(f"  Running sim closed-loop...", flush=True)
        time.sleep(1)
        run_sim_closed_loop(task, video_path, sim_output, port, max_steps)

    finally:
        # Kill server if still running
        if server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=5)

    # Read results
    traj_csv = sim_output / "trajectory.csv"
    if not traj_csv.exists():
        raise RuntimeError(f"No trajectory CSV at {traj_csv}")

    sim_traj = read_sim_trajectory(str(traj_csv))
    metrics = compute_metrics(task, sim_traj)

    diag_path = sim_output / "closed_loop_diagnostics.json"
    diag = {}
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    make_closed_loop_plot(task, sim_traj, metrics, diag,
                          output_dir / "closed_loop_plot.png")

    sr = "PASS" if metrics["success"] else "FAIL"
    print(f"  Result: {sr} | err={metrics['final_error_m']:.3f}m | "
          f"path={metrics['sim_path_length_m']:.3f}m | "
          f"steps={diag.get('total_steps', '?')}", flush=True)

    return task, metrics, diag


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop FlowDiT V2+ with Isaac Sim drone")
    parser.add_argument("--task", default=None,
                        help="Single task (e.g. drone/tasks/task_01_wh_forward_5m)")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--robot", default="drone")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    if args.all_tasks:
        task_dirs = discover_tasks(args.robot)
        print(f"\n{'#'*70}")
        print(f" CLOSED-LOOP VALIDATION: {len(task_dirs)} tasks")
        print(f"{'#'*70}\n")

        all_results = []
        for i, td in enumerate(task_dirs):
            print(f"\n[{i+1}/{len(task_dirs)}] {td.name}")
            try:
                task, metrics, diag = run_one_task(td, args.port, args.max_steps)
                all_results.append((task, metrics))
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        # Aggregate
        if all_results:
            n = len(all_results)
            successes = sum(1 for _, m in all_results if m["success"])
            avg_err = np.mean([m["final_error_m"] for _, m in all_results])
            avg_spl = np.mean([m["spl"] for _, m in all_results])
            avg_ate = np.mean([m["ate_m"] for _, m in all_results])

            agg = {
                "model": "FlowDiT V2+ Closed-Loop",
                "tasks_evaluated": n,
                "success_rate": round(successes / n, 4),
                "successes": successes,
                "mean_final_error_m": round(float(avg_err), 4),
                "mean_spl": round(float(avg_spl), 4),
                "mean_ate_m": round(float(avg_ate), 4),
            }

            agg_dir = SCRIPT_DIR / args.robot / "closed_loop_results"
            agg_dir.mkdir(parents=True, exist_ok=True)
            with open(agg_dir / "aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)

            print(f"\n{'#'*70}")
            print(f" CLOSED-LOOP COMPLETE: {successes}/{n} tasks passed")
            print(f"  SR: {agg['success_rate']*100:.1f}%  |  "
                  f"Error: {avg_err:.3f}m  |  "
                  f"SPL: {avg_spl:.3f}  |  ATE: {avg_ate:.3f}m")
            print(f"  Results: {agg_dir}")
            print(f"{'#'*70}")

    elif args.task:
        task_dir = SCRIPT_DIR / args.task
        task, metrics, diag = run_one_task(task_dir, args.port, args.max_steps)

        print(f"\n{'='*60}")
        sr = "PASS" if metrics["success"] else "FAIL"
        print(f" {sr} | err={metrics['final_error_m']:.3f}m | "
              f"SPL={metrics['spl']:.3f} | ATE={metrics['ate_m']:.3f}m")
        print(f" Steps: {diag.get('total_steps', '?')} | "
              f"Stopped: {diag.get('should_stop', '?')}")
        print(f"{'='*60}")
    else:
        parser.error("Specify --task or --all-tasks")


if __name__ == "__main__":
    main()
