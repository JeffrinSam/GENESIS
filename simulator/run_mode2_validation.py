#!/usr/bin/env python3
"""
Run FlowDiT V2+ Mode 2 inference + Isaac Sim validation for a drone task.

Mode 2 = realtime closed-loop: warmup_realtime() → predict_realtime() per step.
Uses goal video frames as observations (simulating what the robot camera would see).

Pipeline:
  1. Run FlowDiT Mode 2 inference (subprocess in FlowDiT venv)
  2. Save predicted actions + trajectory + diagnostics
  3. Run drone_sim.py --csv-replay in Isaac Sim (headless)
  4. Plot predicted vs simulated vs GT trajectory
  5. Compute metrics: SR, ATE, SPL, final error
  6. Save everything to task_dir/mode2_sim/

Usage:
    python run_mode2_validation.py --task drone/tasks/task_01_wh_forward_5m
    python run_mode2_validation.py --task drone/tasks/task_01_wh_forward_5m --no-headless
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent

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
DRONE_SIM = SCRIPT_DIR / "drone" / "drone_sim.py"

# ── Mode 2 inference subprocess script ───────────────────────────────────────
MODE2_SCRIPT = '''
import sys, json, time, numpy as np
sys.path.insert(0, "{flowdit_dir}")

import torch
import cv2
from models.flowdit_v2_plus import create_flowdit_v2_plus

video_path = sys.argv[1]
checkpoint_path = sys.argv[2]
output_npy = sys.argv[3]
output_traj = sys.argv[4]
output_diag = sys.argv[5]
requested_fps = float(sys.argv[6])
obs_mode = sys.argv[7]  # "first", "middle", "last"

# ── Read video ──
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
    print(json.dumps({{"error": "No frames"}}))
    sys.exit(1)
if source_fps <= 1e-3:
    source_fps = 16.0

video = np.stack(frames, axis=0)
print(f"Video: {{video.shape[0]}} frames @ {{source_fps}} fps", flush=True)

# ── Stride subsample ──
stride = 1
if source_fps > requested_fps * 1.5:
    stride = max(1, int(round(source_fps / requested_fps)))
video_for_model = video[::stride]
effective_fps = source_fps / stride
print(f"After stride={{stride}}: {{video_for_model.shape[0]}} frames @ {{effective_fps:.1f}} fps", flush=True)

# ── Load model ──
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
    state_dict = ckpt_obj["model_state_dict"]
    ckpt_meta = ckpt_obj
else:
    state_dict = ckpt_obj
    ckpt_meta = {{}}

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

# ── Mode 2 inference ──
V = video_for_model
N = len(V)

if obs_mode == "first":
    start_obs_idx = 0
elif obs_mode == "last":
    start_obs_idx = N - 1
else:
    start_obs_idx = N // 2

max_steps = N - start_obs_idx
obs_indices = list(range(start_obs_idx, start_obs_idx + max_steps))

print(f"Mode 2: obs_mode={{obs_mode}}, start_idx={{start_obs_idx}}, max_steps={{max_steps}}", flush=True)

t0 = time.time()
with torch.no_grad():
    cache = model.warmup_realtime(V, prompt=None)
t_warmup = time.time()
print(f"Warmup: {{(t_warmup - t0)*1000:.0f}} ms", flush=True)

commands = []
diag_list = []
stopped = False
stop_step = None
stop_reason = "not_reached"

fps_int = max(1, int(round(effective_fps)))

for step_i, obs_idx in enumerate(obs_indices):
    current_obs = V[obs_idx]
    command, horizon_actions, cache, diagnostics = model.predict_realtime(
        goal_video=V,
        current_obs=current_obs,
        prompt=None,
        goal_features_cache=cache,
        video_fps=fps_int,
        stop_speed_threshold=0.05,
        stop_yaw_threshold=0.08,
        stop_consecutive_steps=3,
        stop_confidence_threshold=0.15,
        min_steps_before_stop=3,
        smoothing_alpha=0.65,
        horizon_decay=0.65,
        num_action_samples=3,
        max_vx=1.0,
        max_vy=1.0,
        max_yaw_rate=1.0,
        return_info=True,
    )
    commands.append(np.asarray(command, dtype=np.float32))
    diag_list.append({{
        "step": step_i,
        "obs_idx": obs_idx,
        "confidence": float(diagnostics.get("confidence", 0.0)),
        "uncertainty": float(diagnostics.get("uncertainty", 0.0)),
        "should_stop": bool(diagnostics.get("should_stop", False)),
        "stop_reason": diagnostics.get("stop_reason", ""),
    }})

    if step_i % 20 == 0:
        c = diagnostics.get("confidence", 0)
        print(f"  Step {{step_i}}/{{max_steps}}: vx={{command[0]:.3f}} vy={{command[1]:.3f}} "
              f"yaw={{command[2]:.3f}} conf={{c:.3f}}", flush=True)

    if diagnostics.get("should_stop", False):
        stopped = True
        stop_step = step_i
        stop_reason = diagnostics.get("stop_reason", "velocity_threshold")
        print(f"  STOPPED at step {{step_i}}: {{stop_reason}}", flush=True)
        break

t_done = time.time()

# ── Process results ──
actions = np.stack(commands, axis=0) if commands else np.zeros((0, 3), dtype=np.float32)

# Integrate trajectory
dt = 1.0 / effective_fps
traj = np.zeros((len(actions), 3), dtype=np.float32)
x, y, theta = 0.0, 0.0, 0.0
for i, (vx, vy, yaw_rate) in enumerate(actions):
    vx_w = float(vx) * np.cos(theta) - float(vy) * np.sin(theta)
    vy_w = float(vx) * np.sin(theta) + float(vy) * np.cos(theta)
    x += vx_w * dt
    y += vy_w * dt
    theta += float(yaw_rate) * dt
    traj[i] = [x, y, theta]

np.save(output_npy, actions)
np.save(output_traj, traj)

# Save diagnostics
with open(output_diag, "w") as f:
    json.dump(diag_list, f, indent=2)

# Stats
total_dist = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))) if len(traj) > 1 else 0.0
speed_profile = np.linalg.norm(actions[:, :2], axis=1) if len(actions) > 0 else np.array([0.0])

control_elapsed = max(1e-6, t_done - t_warmup)
total_elapsed = max(1e-6, t_done - t0)

info = {{
    "inference_mode": "mode2",
    "n_velocity_points": int(len(actions)),
    "effective_fps": float(effective_fps),
    "source_fps": float(source_fps),
    "stride": int(stride),
    "mean_speed": float(np.mean(speed_profile)),
    "total_distance": total_dist,
    "duration_sec": float(len(actions) / max(effective_fps, 1e-6)),
    "final_position": traj[-1].tolist() if len(traj) > 0 else [0.0, 0.0, 0.0],
    "num_source_frames": int(len(video)),
    "obs_mode": obs_mode,
    "start_obs_idx": start_obs_idx,
    "mode2_steps_executed": int(len(actions)),
    "mode2_stopped": stopped,
    "mode2_stop_step": stop_step,
    "mode2_stop_reason": stop_reason,
    "inference_hz": float(len(actions) / control_elapsed),
    "warmup_ms": float((t_warmup - t0) * 1000.0),
    "confidence_mean": float(np.mean([d["confidence"] for d in diag_list])) if diag_list else 0.0,
    "uncertainty_mean": float(np.mean([d["uncertainty"] for d in diag_list])) if diag_list else 0.0,
}}

print("FLOWDIT_RESULT:" + json.dumps(info), flush=True)
'''


def run_mode2_inference(video_path, output_dir, requested_fps=16.0, obs_mode="first"):
    """Run FlowDiT Mode 2 as subprocess. Returns info dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_out = output_dir / "predicted_actions.npy"
    traj_out = output_dir / "predicted_trajectory.npy"
    diag_out = output_dir / "diagnostics.json"

    script = MODE2_SCRIPT.format(flowdit_dir=str(FLOWDIT_DIR))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        result = subprocess.run(
            [str(FLOWDIT_PYTHON), tmp,
             str(video_path), str(FLOWDIT_CHECKPOINT),
             str(npy_out), str(traj_out), str(diag_out),
             str(requested_fps), obs_mode],
            capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            err = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
            raise RuntimeError(f"FlowDiT Mode 2 failed (exit {result.returncode}):\n{err}")

        info = {}
        for line in result.stdout.splitlines():
            if line.startswith("FLOWDIT_RESULT:"):
                info = json.loads(line[len("FLOWDIT_RESULT:"):])
            else:
                print(f"    [FlowDiT] {line}")

        if not npy_out.exists():
            raise RuntimeError("No output .npy produced")

        return info
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_csv(output_dir, fps):
    """Save CSV with trajectory columns."""
    actions = np.load(str(output_dir / "predicted_actions.npy"))
    traj = np.load(str(output_dir / "predicted_trajectory.npy"))

    traj_aligned = traj[:len(actions)]
    csv_path = output_dir / "predicted_actions.csv"

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


def run_simulation(task_dir, task, pred_info, output_dir, headless=True):
    """Run drone_sim.py with CSV replay."""
    csv_path = output_dir / "predicted_actions.csv"
    fps = int(round(pred_info.get("effective_fps", 24)))
    start = task["start"]
    env = task.get("environment", "warehouse")

    sim_output = output_dir / "sim"
    sim_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ISAAC_PYTHON), str(DRONE_SIM),
        "--env", env,
        "--csv-replay", str(csv_path),
        "--fps", str(fps),
        "--start-x", str(start["x"]),
        "--start-y", str(start["y"]),
        "--start-z", str(start["z"]),
        "--start-heading", str(start["heading_deg"]),
        "--output-dir", str(sim_output),
        "--headless",
    ]

    print(f"  Running Isaac Sim (fps={fps})...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    for line in result.stdout.splitlines()[-10:]:
        print(f"    [sim] {line}")
    if result.returncode != 0:
        err = result.stderr[-500:] if result.stderr else ""
        raise RuntimeError(f"Simulation failed (exit {result.returncode}): {err}")

    print(f"  Simulation done in {elapsed:.1f}s")
    return sim_output


def read_sim_trajectory(csv_path):
    """Read sim trajectory CSV."""
    data = {"frame": [], "t": [], "vx": [], "vy": [], "yaw_rate": [],
            "vz": [], "x": [], "y": [], "z": [], "heading": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                if key in row:
                    data[key].append(float(row[key]))
    return {k: np.array(v) for k, v in data.items()}


def compute_metrics(task, sim_traj):
    """Compute SR, ATE, SPL, final error."""
    start = task["start"]
    goal = task["goal"]
    radius = task.get("success_radius_m", 0.5)

    sim_fx = sim_traj["x"][-1] if len(sim_traj["x"]) else start["x"]
    sim_fy = sim_traj["y"][-1] if len(sim_traj["y"]) else start["y"]

    gt_dx = goal["x"] - start["x"]
    gt_dy = goal["y"] - start["y"]
    gt_dist = math.sqrt(gt_dx**2 + gt_dy**2)

    final_error = math.sqrt((sim_fx - goal["x"])**2 + (sim_fy - goal["y"])**2)
    success = final_error <= radius

    if len(sim_traj["x"]) > 1:
        dx = np.diff(sim_traj["x"])
        dy = np.diff(sim_traj["y"])
        sim_path_len = float(np.sum(np.sqrt(dx**2 + dy**2)))
    else:
        sim_path_len = 0.0

    spl = (gt_dist / max(sim_path_len, gt_dist)) if (success and gt_dist > 0) else 0.0

    N = len(sim_traj["x"])
    if N > 0 and gt_dist > 0:
        fracs = np.linspace(0, 1, N)
        gt_x = start["x"] + fracs * gt_dx
        gt_y = start["y"] + fracs * gt_dy
        ate = float(np.mean(np.sqrt((sim_traj["x"] - gt_x)**2 + (sim_traj["y"] - gt_y)**2)))
    else:
        ate = 0.0

    return {
        "success": success,
        "success_radius_m": radius,
        "final_error_m": round(final_error, 4),
        "spl": round(spl, 4),
        "ate_m": round(ate, 4),
        "gt_distance_m": round(gt_dist, 4),
        "sim_path_length_m": round(sim_path_len, 4),
        "sim_final_position": {"x": round(float(sim_fx), 4), "y": round(float(sim_fy), 4)},
        "goal_position": {"x": goal["x"], "y": goal["y"]},
        "sim_frames": N,
    }


def make_validation_plot(task_dir, task, output_dir, sim_traj, metrics, pred_info):
    """3-panel validation plot: trajectory, velocities+confidence, metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = task["start"]
    goal = task["goal"]

    # Load predicted trajectory (relative → absolute)
    pred_traj = np.load(str(output_dir / "predicted_trajectory.npy"))
    pred_actions = np.load(str(output_dir / "predicted_actions.npy"))

    # Load diagnostics
    diag_path = output_dir / "diagnostics.json"
    diag = []
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)

    fps = pred_info.get("effective_fps", 24)

    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(
        f"Mode 2 Simulation Validation — {task['task_id']} — {task['name']}\n"
        f"{task['description']}",
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Top-down trajectory ──
    ax = fig.add_subplot(131)
    ax.plot([start["x"], goal["x"]], [start["y"], goal["y"]],
            "k--", linewidth=2, alpha=0.5, label="GT path")

    # Predicted (shifted to start)
    pred_x = pred_traj[:, 0] + start["x"]
    pred_y = pred_traj[:, 1] + start["y"]
    ax.plot(pred_x, pred_y, color="#2563eb", linewidth=1.5, alpha=0.7, label="predicted")

    # Simulated
    if len(sim_traj["x"]) > 0:
        ax.plot(sim_traj["x"], sim_traj["y"], color="#dc2626", linewidth=2, label="simulated")

    ax.scatter(start["x"], start["y"], color="#16a34a", s=120, marker="o", zorder=5, label="start")
    ax.scatter(goal["x"], goal["y"], color="#f59e0b", s=120, marker="*", zorder=5, label="goal")
    if len(sim_traj["x"]) > 0:
        ax.scatter(sim_traj["x"][-1], sim_traj["y"][-1], color="#dc2626", s=100, marker="x", zorder=5, label="sim end")

    circle = plt.Circle((goal["x"], goal["y"]), task.get("success_radius_m", 0.5),
                         fill=False, color="#f59e0b", linestyle=":", linewidth=1.5)
    ax.add_patch(circle)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory Comparison")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Panel 2: Velocity + confidence ──
    ax2 = fig.add_subplot(132)
    T = len(pred_actions)
    t = np.arange(T) / max(fps, 1e-6)
    ax2.plot(t, pred_actions[:, 0], label="vx", color="#ef4444", alpha=0.8)
    ax2.plot(t, pred_actions[:, 1], label="vy", color="#10b981", alpha=0.8)
    ax2.plot(t, pred_actions[:, 2], label="yaw", color="#3b82f6", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)

    # Confidence overlay (secondary axis)
    if diag:
        ax2b = ax2.twinx()
        conf = [d["confidence"] for d in diag]
        t_conf = np.arange(len(conf)) / max(fps, 1e-6)
        ax2b.fill_between(t_conf, conf, alpha=0.15, color="#f59e0b")
        ax2b.plot(t_conf, conf, color="#f59e0b", linewidth=1, alpha=0.6, label="confidence")
        ax2b.set_ylabel("Confidence", color="#f59e0b")
        ax2b.set_ylim(0, 1)

    # Mark stop point
    if pred_info.get("mode2_stopped") and pred_info.get("mode2_stop_step") is not None:
        stop_t = pred_info["mode2_stop_step"] / max(fps, 1e-6)
        ax2.axvline(stop_t, color="red", linestyle="--", alpha=0.7, label="stopped")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity")
    ax2.set_title("Mode 2 Commands + Confidence")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.25)

    # ── Panel 3: Metrics ──
    ax3 = fig.add_subplot(133)
    ax3.axis("off")
    sr_str = "YES" if metrics["success"] else "NO"
    stop_info = ""
    if pred_info.get("mode2_stopped"):
        stop_info = (f"\nStopped: step {pred_info['mode2_stop_step']}"
                     f"\nReason: {pred_info['mode2_stop_reason']}")
    else:
        stop_info = "\nStopped: NO (ran all steps)"

    summary = (
        f"FlowDiT V2+ (Mode 2) → Isaac Sim\n"
        f"{'━'*32}\n"
        f"Success: {sr_str}\n"
        f"Final Error: {metrics['final_error_m']:.3f} m\n"
        f"Success Radius: {metrics['success_radius_m']} m\n"
        f"{'━'*32}\n"
        f"SPL: {metrics['spl']:.3f}\n"
        f"ATE: {metrics['ate_m']:.3f} m\n"
        f"GT Distance: {metrics['gt_distance_m']:.2f} m\n"
        f"Sim Path Length: {metrics['sim_path_length_m']:.3f} m\n"
        f"{'━'*32}\n"
        f"Steps Executed: {pred_info.get('mode2_steps_executed', 0)}"
        f"{stop_info}\n"
        f"Inference Hz: {pred_info.get('inference_hz', 0):.1f}\n"
        f"Warmup: {pred_info.get('warmup_ms', 0):.0f} ms\n"
        f"Confidence: {pred_info.get('confidence_mean', 0):.3f}\n"
        f"FPS: {fps:.0f}"
    )
    ax3.text(
        0.05, 0.95, summary, va="top", fontfamily="monospace", fontsize=10,
        bbox={"boxstyle": "round",
              "facecolor": "#dbeafe" if metrics["success"] else "#fee2e2",
              "alpha": 0.7},
        transform=ax3.transAxes,
    )

    plt.tight_layout()
    plt.savefig(str(output_dir / "validation_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: validation_plot.png")


def frames_to_video(frames_dir, video_path, fps=24):
    """Convert PNG frames to MP4."""
    if not frames_dir.exists():
        return False
    frame_count = len(list(frames_dir.glob("frame_*.png")))
    if frame_count == 0:
        return False
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return False
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  Video: {video_path.name} ({frame_count} frames, {size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Mode 2 sim validation")
    parser.add_argument("--task", required=True, help="Task directory (relative to Simulator/)")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--obs-mode", default="first",
                        choices=["first", "middle", "last"],
                        help="Where to start observing in the video (default: first)")
    parser.add_argument("--fps", type=float, default=16.0)
    args = parser.parse_args()

    task_dir = SCRIPT_DIR / args.task
    if not task_dir.exists():
        print(f"ERROR: Task not found: {task_dir}")
        sys.exit(1)

    with open(task_dir / "task.json") as f:
        task = json.load(f)

    video_path = task_dir / "video" / "goal_video.mp4"
    if not video_path.exists():
        print(f"ERROR: No goal_video.mp4 in {task_dir / 'video'}")
        sys.exit(1)

    output_dir = task_dir / "mode2_sim"

    print(f"\n{'='*65}")
    print(f" Mode 2 Simulation Validation")
    print(f"{'='*65}")
    print(f"  Task: {task['task_id']} — {task['name']}")
    print(f"  Description: {task['description']}")
    print(f"  Obs mode: {args.obs_mode}")
    print(f"  Start: ({task['start']['x']}, {task['start']['y']}, {task['start']['z']})")
    print(f"  Goal:  ({task['goal']['x']}, {task['goal']['y']}, {task['goal']['z']})")
    print(f"{'='*65}\n")

    # Step 1: Mode 2 inference
    print("Step 1: FlowDiT Mode 2 inference")
    t0 = time.time()
    pred_info = run_mode2_inference(video_path, output_dir, args.fps, args.obs_mode)
    inference_time = time.time() - t0

    n = pred_info.get("n_velocity_points", 0)
    hz = pred_info.get("inference_hz", 0)
    stopped = pred_info.get("mode2_stopped", False)
    print(f"  Result: {n} steps | {hz:.1f} Hz | stopped={stopped}")
    print(f"  Inference time: {inference_time:.1f}s\n")

    # Step 2: Save CSV
    print("Step 2: Save CSV")
    fps = pred_info.get("effective_fps", 24)
    n_rows = save_csv(output_dir, fps)
    print(f"  Saved: predicted_actions.csv ({n_rows} rows)\n")

    # Step 3: Run sim
    print("Step 3: Isaac Sim replay")
    sim_output = run_simulation(task_dir, task, pred_info, output_dir,
                                headless=not args.no_headless)

    # Step 4: Read sim trajectory
    traj_csv = sim_output / "trajectory.csv"
    if not traj_csv.exists():
        print("  ERROR: No trajectory.csv from simulation")
        sys.exit(1)
    sim_traj = read_sim_trajectory(traj_csv)
    print(f"  Sim trajectory: {len(sim_traj['x'])} points\n")

    # Step 5: Convert frames → video
    sim_video = output_dir / "sim_video.mp4"
    frames_to_video(sim_output / "frames", sim_video, fps=int(round(fps)))

    # Step 6: Metrics
    print("Step 4: Metrics")
    metrics = compute_metrics(task, sim_traj)
    print(f"  Success:     {'YES' if metrics['success'] else 'NO'}")
    print(f"  Final Error: {metrics['final_error_m']:.3f} m")
    print(f"  SPL:         {metrics['spl']:.3f}")
    print(f"  ATE:         {metrics['ate_m']:.3f} m")

    # Save
    metrics["task_id"] = task["task_id"]
    metrics["inference_mode"] = "mode2"
    pred_info["inference_time_s"] = inference_time
    pred_info["checkpoint"] = str(FLOWDIT_CHECKPOINT)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "prediction_info.json", "w") as f:
        json.dump(pred_info, f, indent=2)
    print(f"  Saved: metrics.json, prediction_info.json\n")

    # Step 7: Validation plot
    print("Step 5: Validation plot")
    make_validation_plot(task_dir, task, output_dir, sim_traj, metrics, pred_info)

    print(f"\n{'='*65}")
    print(f" MODE 2 VALIDATION COMPLETE — {task['task_id']}")
    print(f"{'='*65}")
    sr = "PASS" if metrics["success"] else "FAIL"
    print(f"  Result: {sr} (error={metrics['final_error_m']:.3f}m)")
    print(f"  Steps: {pred_info.get('mode2_steps_executed', 0)} | "
          f"Hz: {pred_info.get('inference_hz', 0):.1f} | "
          f"Stopped: {pred_info.get('mode2_stopped', False)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
