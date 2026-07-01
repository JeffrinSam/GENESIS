#!/usr/bin/env python3
"""
Run FlowDiT V2+ Mode 1 inference on all task videos.

Mirrors the inference pipeline from flow_constrained_v2_plus/website/app.py:
  - Video: BGR→RGB, resize 224×224, normalize [0,1], stride subsample
  - Model: auto-detect use_raft from checkpoint config
  - Output: velocities [T, 3], trajectory [T, 3] (integrated)
  - CSV: frame, vx_m_s, vy_m_s, yaw_rate_rad_s, x_m, y_m, heading_rad
  - Plot: 4-panel (trajectory+arrows, velocity components, speed, summary)

Usage:
    python predict_all_actions.py
    python predict_all_actions.py --robot drone
    python predict_all_actions.py --robot drone --tasks task_01_wh_forward_5m
    python predict_all_actions.py --skip-existing
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

FLOWDIT_DIR = Path(os.getenv(
    "FLOWDIT_DIR",
    str(Path(__file__).resolve().parents[1] / "part2_navigation" / "flow_constrained_v2")
))
FLOWDIT_PYTHON = Path(os.getenv("FLOWDIT_PYTHON", sys.executable))
FLOWDIT_CHECKPOINT = Path(os.getenv(
    "FLOWDIT_CHECKPOINT",
    str(FLOWDIT_DIR / "checkpoints" / "best.pth")
))

ROBOT_DIRS = {
    "drone": "drone",
    "humanoid": "humanoid",
    "mobile_robot": "mobile_robot",
}

# ── Inference subprocess script ──────────────────────────────────────────────
# Matches website/app.py: _read_video_rgb, _load_model, run_v2_plus_inference
# Runs in FlowDiT's venv. Outputs JSON result on stdout.

INFERENCE_SCRIPT = '''
import sys, json, numpy as np
sys.path.insert(0, "{flowdit_dir}")

import torch
import cv2
from models.flowdit_v2_plus import create_flowdit_v2_plus

video_path = sys.argv[1]
checkpoint_path = sys.argv[2]
output_npy = sys.argv[3]
output_traj = sys.argv[4]
requested_fps = float(sys.argv[5])

# ── Read video (matches website _read_video_rgb) ──
cap = cv2.VideoCapture(video_path)
source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frames.append(frame.astype(np.float32) / 255.0)
cap.release()

if not frames:
    print(json.dumps({{"error": "No frames in video"}}))
    sys.exit(1)

if source_fps <= 1e-3:
    source_fps = 16.0

video = np.stack(frames, axis=0)
print(f"Video: {{video.shape[0]}} frames @ {{source_fps}} fps (source)", flush=True)

# ── Stride subsample (matches website) ──
stride = 1
if source_fps > requested_fps * 1.5:
    stride = max(1, int(round(source_fps / requested_fps)))

video_for_model = video[::stride]
effective_fps = source_fps / stride
print(f"After stride={{stride}}: {{video_for_model.shape[0]}} frames @ {{effective_fps:.1f}} fps", flush=True)

# ── Load model (matches website _load_model with auto use_raft) ──
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_obj = torch.load(checkpoint_path, map_location=device, weights_only=False)

if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
    state_dict = ckpt_obj["model_state_dict"]
    ckpt_meta = ckpt_obj
else:
    state_dict = ckpt_obj
    ckpt_meta = {{}}

# Auto-detect use_raft from checkpoint config
config = ckpt_meta.get("config")
if isinstance(config, dict) and "use_raft" in config:
    use_raft = bool(config["use_raft"])
else:
    raft_tokens = ("flow_encoder._raft", "flow_encoder.pool", "flow_encoder.fc")
    use_raft = any(any(tok in k for tok in raft_tokens) for k in state_dict.keys())

print(f"Model: use_raft={{use_raft}}, device={{device}}", flush=True)

model = create_flowdit_v2_plus(device=device, use_raft=use_raft)
try:
    model.load_state_dict(state_dict, strict=True)
except Exception:
    model.load_state_dict(state_dict, strict=False)
model.eval()

# ── Run Mode 1 inference ──
with torch.no_grad():
    result = model.predict_full_trajectory(
        video_for_model, video_fps=max(1, int(round(effective_fps)))
    )

velocities = np.asarray(result["velocities"], dtype=np.float32)
trajectory = np.asarray(result["trajectory"], dtype=np.float32)
speed_profile = np.asarray(result.get("speed_profile"), dtype=np.float32)

np.save(output_npy, velocities)
np.save(output_traj, trajectory)

# ── Compute stats (matches website return format) ──
total_distance = float(np.sum(np.linalg.norm(
    np.diff(trajectory[:, :2], axis=0), axis=1
))) if len(trajectory) > 1 else 0.0

info = {{
    "n_velocity_points": int(len(velocities)),
    "effective_fps": float(effective_fps),
    "source_fps": float(source_fps),
    "stride": int(stride),
    "mean_speed": float(np.mean(speed_profile)) if len(speed_profile) else 0.0,
    "total_distance": total_distance,
    "duration_sec": float(len(velocities) / max(effective_fps, 1e-6)),
    "final_position": trajectory[-1].tolist() if len(trajectory) else [0.0, 0.0, 0.0],
    "num_source_frames": int(len(video)),
}}

print("FLOWDIT_RESULT:" + json.dumps(info), flush=True)
'''


def run_inference(video_path, output_npy, output_traj, requested_fps=16.0):
    """Run FlowDiT Mode 1 as subprocess. Returns info dict."""
    script = INFERENCE_SCRIPT.format(flowdit_dir=str(FLOWDIT_DIR))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        result = subprocess.run(
            [str(FLOWDIT_PYTHON), tmp,
             str(video_path), str(FLOWDIT_CHECKPOINT),
             str(output_npy), str(output_traj),
             str(requested_fps)],
            capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            err = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
            raise RuntimeError(f"FlowDiT failed (exit {result.returncode}):\n{err}")

        info = {}
        for line in result.stdout.splitlines():
            if line.startswith("FLOWDIT_RESULT:"):
                info = json.loads(line[len("FLOWDIT_RESULT:"):])
            else:
                print(f"    [FlowDiT] {line}")

        if not Path(output_npy).exists():
            raise RuntimeError("No output .npy produced")

        return info
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_csv(npy_path, traj_path, csv_path, fps):
    """Save CSV with trajectory columns (matches website format)."""
    actions = np.load(str(npy_path))
    traj = np.load(str(traj_path)) if Path(traj_path).exists() else None

    # Align trajectory to velocity points
    if traj is not None:
        if len(traj) == len(actions) + 1:
            traj_aligned = traj[1:]
        else:
            traj_aligned = traj[:len(actions)]
    else:
        # Integrate from velocities
        dt = 1.0 / max(fps, 1e-6)
        traj_aligned = np.zeros((len(actions), 3), dtype=np.float32)
        x, y, theta = 0.0, 0.0, 0.0
        for i, (vx, vy, yaw_rate) in enumerate(actions):
            vx_w = float(vx) * np.cos(theta) - float(vy) * np.sin(theta)
            vy_w = float(vx) * np.sin(theta) + float(vy) * np.cos(theta)
            x += vx_w * dt
            y += vy_w * dt
            theta += float(yaw_rate) * dt
            traj_aligned[i] = [x, y, theta]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vx_m_s", "vy_m_s", "yaw_rate_rad_s",
                          "x_m", "y_m", "heading_rad"])
        for i, row in enumerate(actions):
            tx, ty, th = traj_aligned[i] if i < len(traj_aligned) else (0, 0, 0)
            writer.writerow([
                i,
                f"{float(row[0]):.6f}",
                f"{float(row[1]):.6f}",
                f"{float(row[2]):.6f}",
                f"{float(tx):.6f}",
                f"{float(ty):.6f}",
                f"{float(th):.6f}",
            ])
    return len(actions)


def make_plots(task_dir, task_json, info):
    """Generate 4-panel visualization matching website _create_visualization_mode1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npy_path = task_dir / "predicted_actions.npy"
    traj_path = task_dir / "predicted_trajectory.npy"
    velocities = np.load(str(npy_path))
    trajectory = np.load(str(traj_path)) if traj_path.exists() else None

    fps = info.get("effective_fps", 16)
    T = len(velocities)
    t = np.arange(T) / max(float(fps), 1e-6)
    speed_profile = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)

    # Align trajectory
    if trajectory is not None:
        traj_aligned = trajectory[1:] if len(trajectory) == T + 1 else trajectory[:T]
    else:
        dt = 1.0 / max(fps, 1e-6)
        traj_aligned = np.zeros((T, 3), dtype=np.float32)
        x, y, theta = 0.0, 0.0, 0.0
        for i in range(T):
            vx_w = float(velocities[i, 0]) * np.cos(theta) - float(velocities[i, 1]) * np.sin(theta)
            vy_w = float(velocities[i, 0]) * np.sin(theta) + float(velocities[i, 1]) * np.cos(theta)
            x += vx_w * dt
            y += vy_w * dt
            theta += float(velocities[i, 2]) * dt
            traj_aligned[i] = [x, y, theta]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"FlowDiT V2+ Mode 1 — {task_json['task_id']} — {task_json['name']}\n"
                 f"{task_json['description']}", fontsize=13, fontweight="bold")

    # Panel 1: Integrated trajectory with velocity arrows
    ax1 = fig.add_subplot(221)
    if trajectory is not None:
        ax1.plot(trajectory[:, 0], trajectory[:, 1], color="#2563eb", linewidth=1.8)
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], color="#16a34a", s=80, label="start")
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], color="#dc2626", s=80, label="end")
    else:
        ax1.plot(traj_aligned[:, 0], traj_aligned[:, 1], color="#2563eb", linewidth=1.8)
        ax1.scatter(traj_aligned[0, 0], traj_aligned[0, 1], color="#16a34a", s=80, label="start")
        ax1.scatter(traj_aligned[-1, 0], traj_aligned[-1, 1], color="#dc2626", s=80, label="end")

    # GT direction line
    s = task_json["start"]
    g = task_json["goal"]
    ax1.plot([0, g["x"] - s["x"]], [0, g["y"] - s["y"]],
             "k--", alpha=0.4, linewidth=1, label="GT direction")

    # Velocity arrows (every ~20 points)
    step = max(1, T // 20)
    for i in range(0, T, step):
        vx, vy, _ = velocities[i]
        th = traj_aligned[i, 2]
        vx_w = vx * np.cos(th) - vy * np.sin(th)
        vy_w = vx * np.sin(th) + vy * np.cos(th)
        ax1.arrow(
            traj_aligned[i, 0], traj_aligned[i, 1],
            vx_w * 0.25, vy_w * 0.25,
            head_width=0.02, head_length=0.03,
            fc="#ef4444", ec="#ef4444", alpha=0.45,
        )

    ax1.set_title("Integrated trajectory")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    # Panel 2: Velocity components
    ax2 = fig.add_subplot(222)
    ax2.plot(t, velocities[:, 0], label="vx", color="#ef4444")
    ax2.plot(t, velocities[:, 1], label="vy", color="#10b981")
    ax2.plot(t, velocities[:, 2], label="yaw_rate", color="#3b82f6")
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax2.set_title("Velocity components")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("value")
    ax2.grid(alpha=0.25)
    ax2.legend()

    # Panel 3: Speed profile
    ax3 = fig.add_subplot(223)
    ax3.fill_between(t, speed_profile, alpha=0.22, color="#2563eb")
    ax3.plot(t, speed_profile, color="#1d4ed8", linewidth=1.6)
    ax3.set_title("Speed profile")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("|v| (m/s)")
    ax3.grid(alpha=0.25)

    # Panel 4: Summary stats
    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    total_dist = info.get("total_distance", 0)
    final_pos = info.get("final_position", [0, 0, 0])
    summary = (
        f"FlowDiT V2+ (Mode 1)\n"
        f"FPS used: {fps:.2f}\n"
        f"Velocity points: {T}\n"
        f"Duration: {T / max(fps, 1e-6):.2f}s\n"
        f"Mean speed: {np.mean(speed_profile):.4f} m/s\n"
        f"Total distance: {total_dist:.4f} m\n"
        f"Final pose: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})\n"
        f"Stride: {info.get('stride', 1)} | Source FPS: {info.get('source_fps', 0):.0f}"
    )
    ax4.text(
        0.04, 0.95, summary,
        va="top", fontfamily="monospace", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#dbeafe", "alpha": 0.7},
    )

    plt.tight_layout()
    plt.savefig(str(task_dir / "action_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def discover_tasks(robot_filter=None, task_filter=None):
    """Find all tasks with goal_video.mp4."""
    tasks = []
    for name, robot_dir in ROBOT_DIRS.items():
        if robot_filter and name != robot_filter:
            continue
        tasks_dir = BASE_DIR / robot_dir / "tasks"
        if not tasks_dir.is_dir():
            continue
        for folder in sorted(tasks_dir.iterdir()):
            if not folder.is_dir():
                continue
            if task_filter and folder.name not in task_filter:
                continue
            video = folder / "video" / "goal_video.mp4"
            task_file = folder / "task.json"
            if not video.exists() or not task_file.exists():
                continue
            with open(task_file) as f:
                task_json = json.load(f)
            tasks.append({
                "dir": folder,
                "video": video,
                "task": task_json,
                "robot_dir": name,
            })
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run FlowDiT Mode 1 on all task videos")
    parser.add_argument("--robot", choices=list(ROBOT_DIRS.keys()))
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tasks that already have predicted_actions.npy")
    parser.add_argument("--fps", type=float, default=16.0,
                        help="Requested FPS for inference (default: 16)")
    args = parser.parse_args()

    tasks = discover_tasks(robot_filter=args.robot, task_filter=args.tasks)
    if not tasks:
        print("No tasks with goal_video.mp4 found.")
        sys.exit(0)

    if args.skip_existing:
        tasks = [t for t in tasks if not (t["dir"] / "predicted_actions.npy").exists()]

    print(f"\n{'='*65}")
    print(f" FlowDiT V2+ Mode 1 — Batch Action Prediction")
    print(f"{'='*65}")
    print(f"  Checkpoint: {FLOWDIT_CHECKPOINT}")
    print(f"  Request FPS: {args.fps}")
    print(f"  Tasks:      {len(tasks)}")
    print(f"{'='*65}\n")

    ok, fail = 0, 0
    total_time = 0.0

    for i, t in enumerate(tasks, 1):
        task = t["task"]
        task_dir = t["dir"]
        print(f"[{i}/{len(tasks)}] {t['robot_dir']}/{task_dir.name}")
        print(f"  Task: {task['task_id']} — {task['name']}")

        npy_out = task_dir / "predicted_actions.npy"
        traj_out = task_dir / "predicted_trajectory.npy"
        csv_out = task_dir / "predicted_actions.csv"

        try:
            # Step 1: Run FlowDiT inference
            print(f"  Running FlowDiT inference...", flush=True)
            t0 = time.time()
            info = run_inference(t["video"], npy_out, traj_out, args.fps)
            elapsed = time.time() - t0
            total_time += elapsed

            n = info.get("n_velocity_points", 0)
            spd = info.get("mean_speed", 0)
            dist = info.get("total_distance", 0)
            efps = info.get("effective_fps", args.fps)
            print(f"  Inference: {elapsed:.1f}s | {n} pts | "
                  f"mean_speed={spd:.3f} m/s | distance={dist:.2f}m | "
                  f"fps={efps:.1f}")

            # Step 2: Save CSV (with trajectory columns)
            n_rows = save_csv(npy_out, traj_out, csv_out, efps)
            print(f"  Saved: predicted_actions.csv ({n_rows} rows)")

            # Step 3: Generate plots (website-style)
            make_plots(task_dir, task, info)
            print(f"  Saved: action_plot.png")

            # Step 4: Save info
            info["task_id"] = task["task_id"]
            info["inference_time_s"] = elapsed
            info["checkpoint"] = str(FLOWDIT_CHECKPOINT)
            with open(task_dir / "prediction_info.json", "w") as f:
                json.dump(info, f, indent=2)

            ok += 1
            print()

        except Exception as e:
            print(f"  FAILED: {e}\n")
            fail += 1

    print(f"{'='*65}")
    print(f" PREDICTION SUMMARY")
    print(f"{'='*65}")
    print(f"  OK:     {ok}")
    print(f"  Failed: {fail}")
    print(f"  Total:  {len(tasks)}")
    if total_time > 0:
        print(f"  Time:   {total_time:.0f}s ({total_time/60:.1f} min)")
        if ok > 0:
            print(f"  Avg:    {total_time/ok:.1f}s per task")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
