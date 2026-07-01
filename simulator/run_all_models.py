#!/usr/bin/env python3
"""
Run all FlowDiT model versions (V1, V2, V2+) on a drone task and compare.

For each model: inference → CSV → Isaac Sim replay → metrics → plot.
Finally produces a comparison plot of all trajectories and metrics.

Usage:
    python run_all_models.py --task drone/tasks/task_01_wh_forward_5m
    python run_all_models.py --task drone/tasks/task_01_wh_forward_5m --models v1 v2 v2plus
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

SCRIPT_DIR = Path(__file__).parent

# ── Model configs ────────────────────────────────────────────────────────────
FLOWDIT_DIR = Path(os.getenv(
    "FLOWDIT_DIR",
    str(Path(__file__).resolve().parents[1] / "part2_navigation" / "flow_constrained_v2")
))
VIDTONAV_ROOT = FLOWDIT_DIR.parent
FLOWDIT_PYTHON = os.getenv("FLOWDIT_PYTHON", sys.executable)

MODELS = {
    "v1": {
        "name": "FlowDiT V1 (FusionNetwork)",
        "python": FLOWDIT_PYTHON,
        "dir": str(VIDTONAV_ROOT / "flow_constrained"),
        "checkpoint": str(VIDTONAV_ROOT / "flow_constrained" / "checkpoints" / "wheeled" / "best_model.pth"),
        "config": str(VIDTONAV_ROOT / "flow_constrained" / "configs" / "wheeled.yaml"),
    },
    "v2": {
        "name": "FlowDiT V2 (Production)",
        "python": FLOWDIT_PYTHON,
        "dir": str(FLOWDIT_DIR),
        "checkpoint": os.getenv("FLOWDIT_CHECKPOINT", str(FLOWDIT_DIR / "checkpoints" / "best.pth")),
    },
    "v2plus": {
        "name": "FlowDiT V2+ (Improved)",
        "python": FLOWDIT_PYTHON,
        "dir": str(VIDTONAV_ROOT / "flow_constrained_v2_plus"),
        "checkpoint": str(VIDTONAV_ROOT / "flow_constrained_v2_plus" / "checkpoints" / "v2plus_combined" / "best.pth"),
    },
}

ISAAC_PYTHON = Path(os.getenv("ISAAC_SIM_PYTHON", "/opt/isaacsim/python.sh"))
DRONE_SIM = SCRIPT_DIR / "drone" / "drone_sim.py"
HUMANOID_SH = SCRIPT_DIR / "humanoid" / "run_humanoid.sh"

# Embodiment mapping for V1
ROBOT_EMBODIMENT = {
    "drone": "aerial",
    "humanoid": "humanoid",
    "mobile_robot": "wheeled",
}

# ── V1 inference script ──────────────────────────────────────────────────────
V1_SCRIPT = '''
import sys, json, time, numpy as np
sys.path.insert(0, "{model_dir}")

import torch
import yaml
import cv2
from models import FusionNetwork, OpticalFlowExtractor, VDMFeatureExtractor, VisionEncoder
from models.optical_flow import estimate_ego_motion

video_path = sys.argv[1]
checkpoint_path = sys.argv[2]
config_path = sys.argv[3]
output_npy = sys.argv[4]
output_traj = sys.argv[5]
embodiment = sys.argv[6]

# Load video
cap = cv2.VideoCapture(video_path)
source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frames.append(frame)
cap.release()

if not frames:
    print(json.dumps({{"error": "No frames"}}))
    sys.exit(1)

video = np.stack(frames)
T = video.shape[0]
print(f"Video: {{T}} frames @ {{source_fps}} fps", flush=True)

# Load config and model
with open(config_path) as f:
    config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FusionNetwork(
    ego_motion_dim=config["model"]["ego_motion_dim"],
    vdm_feature_dim=config["model"]["vdm_feature_dim"],
    vision_feature_dim=config["model"]["vision_feature_dim"],
    embodiment_dim=config["model"]["embodiment_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    num_embodiments=config["model"]["num_embodiments"],
    action_dim=config["model"]["action_dim"],
    dropout=config["model"]["dropout"],
)
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

# Feature extractors
flow_ext = OpticalFlowExtractor(device=device)
vdm_ext = VDMFeatureExtractor(device=device)
vis_enc = VisionEncoder(device=device)

embodiment_map = {{"wheeled": 0, "legged": 1, "aerial": 2, "humanoid": 3}}
emb_idx = torch.tensor([embodiment_map.get(embodiment, 2)]).to(device)

# Process video in sliding windows
video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
video_tensor = video_tensor.unsqueeze(0).to(device)

window_size = 8
step_size = max(1, window_size // 2)
predictions = []

t0 = time.time()
with torch.no_grad():
    for start_idx in range(0, T, step_size):
        end_idx = min(start_idx + window_size, T)
        window = video_tensor[:, start_idx:end_idx]
        if window.shape[1] < window_size:
            pad = torch.zeros(1, window_size - window.shape[1],
                              window.shape[2], window.shape[3], window.shape[4]).to(device)
            window = torch.cat([window, pad], dim=1)

        optical_flow = flow_ext.extract_from_video(window)
        ego_motion = estimate_ego_motion(optical_flow[0, 0])
        vdm_features = vdm_ext.extract_from_video(window)
        vision_features = vis_enc.extract_from_video(window)

        pred = model(
            ego_motion.unsqueeze(0),
            vdm_features[0, 0].unsqueeze(0),
            vision_features[0, 0].unsqueeze(0),
            emb_idx,
        )
        for _ in range(end_idx - start_idx):
            predictions.append(pred.squeeze(0).cpu().numpy())

        if start_idx % 40 == 0:
            print(f"  Window {{start_idx}}/{{T}}", flush=True)

elapsed = time.time() - t0

while len(predictions) < T:
    predictions.append(predictions[-1] if predictions else np.zeros(3))
velocities = np.array(predictions[:T], dtype=np.float32)

# Integrate trajectory
dt = 1.0 / source_fps
traj = np.zeros((T, 3), dtype=np.float32)
x, y, theta = 0.0, 0.0, 0.0
for i in range(T):
    vx, vy, yaw = velocities[i]
    vx_w = float(vx) * np.cos(theta) - float(vy) * np.sin(theta)
    vy_w = float(vx) * np.sin(theta) + float(vy) * np.cos(theta)
    x += vx_w * dt
    y += vy_w * dt
    theta += float(yaw) * dt
    traj[i] = [x, y, theta]

np.save(output_npy, velocities)
np.save(output_traj, traj)

total_dist = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))) if T > 1 else 0.0
speed = np.linalg.norm(velocities[:, :2], axis=1)

info = {{
    "model": "v1",
    "n_velocity_points": T,
    "effective_fps": float(source_fps),
    "mean_speed": float(np.mean(speed)),
    "total_distance": total_dist,
    "duration_sec": float(T / source_fps),
    "final_position": traj[-1].tolist(),
    "inference_time_s": elapsed,
}}
print("FLOWDIT_RESULT:" + json.dumps(info), flush=True)
'''

# ── V2 inference script ──────────────────────────────────────────────────────
V2_SCRIPT = '''
import sys, json, time, numpy as np
sys.path.insert(0, "{model_dir}")

import torch
import cv2
from models.flowdit_production import create_flowdit_production

video_path = sys.argv[1]
checkpoint_path = sys.argv[2]
output_npy = sys.argv[3]
output_traj = sys.argv[4]
requested_fps = float(sys.argv[5])

# Read video
cap = cv2.VideoCapture(video_path)
source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
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
    sys.exit(1)
if source_fps <= 1e-3:
    source_fps = 16.0

video = np.stack(frames, axis=0)
print(f"Video: {{video.shape[0]}} frames @ {{source_fps}} fps", flush=True)

stride = 1
if source_fps > requested_fps * 1.5:
    stride = max(1, int(round(source_fps / requested_fps)))
video_for_model = video[::stride]
effective_fps = source_fps / stride
print(f"After stride={{stride}}: {{video_for_model.shape[0]}} frames @ {{effective_fps:.1f}} fps", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_flowdit_production(device=device)
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

t0 = time.time()
with torch.no_grad():
    result = model.predict_full_trajectory(
        video_for_model, video_fps=max(1, int(round(effective_fps)))
    )
elapsed = time.time() - t0

velocities = np.asarray(result["velocities"], dtype=np.float32)
trajectory = np.asarray(result["trajectory"], dtype=np.float32)

np.save(output_npy, velocities)
np.save(output_traj, trajectory)

total_dist = float(np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1))) if len(trajectory) > 1 else 0.0
speed = np.linalg.norm(velocities[:, :2], axis=1)

info = {{
    "model": "v2",
    "n_velocity_points": int(len(velocities)),
    "effective_fps": float(effective_fps),
    "mean_speed": float(np.mean(speed)),
    "total_distance": total_dist,
    "duration_sec": float(len(velocities) / max(effective_fps, 1e-6)),
    "final_position": trajectory[-1].tolist() if len(trajectory) else [0, 0, 0],
    "inference_time_s": elapsed,
}}
print("FLOWDIT_RESULT:" + json.dumps(info), flush=True)
'''

# ── V2+ inference script ─────────────────────────────────────────────────────
V2PLUS_SCRIPT = '''
import sys, json, time, numpy as np
sys.path.insert(0, "{model_dir}")

import torch
import cv2
from models.flowdit_v2_plus import create_flowdit_v2_plus

video_path = sys.argv[1]
checkpoint_path = sys.argv[2]
output_npy = sys.argv[3]
output_traj = sys.argv[4]
requested_fps = float(sys.argv[5])

cap = cv2.VideoCapture(video_path)
source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
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
    sys.exit(1)
if source_fps <= 1e-3:
    source_fps = 16.0

video = np.stack(frames, axis=0)
print(f"Video: {{video.shape[0]}} frames @ {{source_fps}} fps", flush=True)

stride = 1
if source_fps > requested_fps * 1.5:
    stride = max(1, int(round(source_fps / requested_fps)))
video_for_model = video[::stride]
effective_fps = source_fps / stride
print(f"After stride={{stride}}: {{video_for_model.shape[0]}} frames @ {{effective_fps:.1f}} fps", flush=True)

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

model = create_flowdit_v2_plus(device=device, use_raft=use_raft)
try:
    model.load_state_dict(state_dict, strict=True)
except Exception:
    model.load_state_dict(state_dict, strict=False)
model.eval()

t0 = time.time()
with torch.no_grad():
    result = model.predict_full_trajectory(
        video_for_model, video_fps=max(1, int(round(effective_fps)))
    )
elapsed = time.time() - t0

velocities = np.asarray(result["velocities"], dtype=np.float32)
trajectory = np.asarray(result["trajectory"], dtype=np.float32)

np.save(output_npy, velocities)
np.save(output_traj, trajectory)

total_dist = float(np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1))) if len(trajectory) > 1 else 0.0
speed = np.linalg.norm(velocities[:, :2], axis=1)

info = {{
    "model": "v2plus",
    "n_velocity_points": int(len(velocities)),
    "effective_fps": float(effective_fps),
    "mean_speed": float(np.mean(speed)),
    "total_distance": total_dist,
    "duration_sec": float(len(velocities) / max(effective_fps, 1e-6)),
    "final_position": trajectory[-1].tolist() if len(trajectory) else [0, 0, 0],
    "inference_time_s": elapsed,
}}
print("FLOWDIT_RESULT:" + json.dumps(info), flush=True)
'''

SCRIPTS = {"v1": V1_SCRIPT, "v2": V2_SCRIPT, "v2plus": V2PLUS_SCRIPT}


def run_inference(model_key, video_path, output_dir, requested_fps=16.0,
                  embodiment="aerial"):
    """Run inference for a given model. Returns info dict."""
    cfg = MODELS[model_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_out = output_dir / "predicted_actions.npy"
    traj_out = output_dir / "predicted_trajectory.npy"

    script = SCRIPTS[model_key].format(model_dir=cfg["dir"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        if model_key == "v1":
            cmd = [cfg["python"], tmp, str(video_path), cfg["checkpoint"],
                   cfg["config"], str(npy_out), str(traj_out), embodiment]
        else:
            cmd = [cfg["python"], tmp, str(video_path), cfg["checkpoint"],
                   str(npy_out), str(traj_out), str(requested_fps)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            err = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
            raise RuntimeError(f"{model_key} failed (exit {result.returncode}):\n{err}")

        info = {}
        for line in result.stdout.splitlines():
            if line.startswith("FLOWDIT_RESULT:"):
                info = json.loads(line[len("FLOWDIT_RESULT:"):])
            else:
                print(f"      [{model_key}] {line}")

        return info
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_csv(output_dir, fps):
    """Save CSV from .npy files."""
    actions = np.load(str(output_dir / "predicted_actions.npy"))
    traj_path = output_dir / "predicted_trajectory.npy"
    traj = np.load(str(traj_path)) if traj_path.exists() else None

    if traj is not None:
        if len(traj) == len(actions) + 1:
            traj_aligned = traj[1:]
        else:
            traj_aligned = traj[:len(actions)]
    else:
        traj_aligned = np.zeros((len(actions), 3))

    csv_path = output_dir / "predicted_actions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vx_m_s", "vy_m_s", "yaw_rate_rad_s",
                          "x_m", "y_m", "heading_rad"])
        for i, row in enumerate(actions):
            tx, ty, th = traj_aligned[i] if i < len(traj_aligned) else (0, 0, 0)
            writer.writerow([i, f"{float(row[0]):.6f}", f"{float(row[1]):.6f}",
                              f"{float(row[2]):.6f}", f"{float(tx):.6f}",
                              f"{float(ty):.6f}", f"{float(th):.6f}"])
    return len(actions)


def run_simulation(task, csv_path, output_dir, fps, no_frames=True, robot="drone"):
    """Run simulation with CSV replay."""
    if robot == "humanoid":
        return run_humanoid_simulation(task, csv_path, output_dir, fps, no_frames)
    return run_drone_simulation(task, csv_path, output_dir, fps, no_frames)


def run_drone_simulation(task, csv_path, output_dir, fps, no_frames=True):
    """Run drone_sim.py with CSV replay."""
    start = task["start"]
    env = task.get("environment", "warehouse")

    cmd = [
        str(ISAAC_PYTHON), str(DRONE_SIM),
        "--env", env, "--csv-replay", str(csv_path),
        "--fps", str(fps),
        "--start-x", str(start["x"]), "--start-y", str(start["y"]),
        "--start-z", str(start["z"]), "--start-heading", str(start["heading_deg"]),
        "--output-dir", str(output_dir), "--headless",
    ]
    if no_frames:
        cmd.append("--no-frames")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Sim failed: {result.stderr[-500:]}")
    return output_dir


def run_humanoid_simulation(task, csv_path, output_dir, fps, no_frames=True):
    """Run humanoid_sim.py via run_humanoid.sh with CSV replay."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = task.get("environment", "warehouse")
    cmd = [
        "bash", str(HUMANOID_SH),
        "--mode", "csv",
        "--csv-path", str(csv_path),
        "--env", env,
        "--fps", str(fps),
        "--num-steps", "500",
        "--headless",
        "--output-dir", str(output_dir),
    ]
    if no_frames:
        cmd.append("--no-frames")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in result.stdout.splitlines():
        if line.strip():
            print(f"      [humanoid_sim] {line.rstrip()}")

    if result.returncode != 0:
        raise RuntimeError(f"Humanoid sim failed: {result.stderr[-500:]}")

    if not (output_dir / "trajectory.csv").exists():
        raise RuntimeError(f"No trajectory CSV at {output_dir / 'trajectory.csv'}")

    return output_dir


def read_sim_trajectory(csv_path):
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
    start, goal = task["start"], task["goal"]
    radius = 3.0  # 300 cm success threshold for all tasks

    sim_fx = sim_traj["x"][-1] if len(sim_traj["x"]) else start["x"]
    sim_fy = sim_traj["y"][-1] if len(sim_traj["y"]) else start["y"]

    gt_dx, gt_dy = goal["x"] - start["x"], goal["y"] - start["y"]
    gt_dist = math.sqrt(gt_dx**2 + gt_dy**2)
    final_error = math.sqrt((sim_fx - goal["x"])**2 + (sim_fy - goal["y"])**2)
    success = final_error <= radius

    if len(sim_traj["x"]) > 1:
        sim_path = float(np.sum(np.sqrt(np.diff(sim_traj["x"])**2 + np.diff(sim_traj["y"])**2)))
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
        "success": success, "final_error_m": round(final_error, 4),
        "spl": round(spl, 4), "ate_m": round(ate, 4),
        "gt_distance_m": round(gt_dist, 4), "sim_path_length_m": round(sim_path, 4),
        "sim_final": {"x": round(float(sim_fx), 4), "y": round(float(sim_fy), 4)},
    }


def make_comparison_plot(task, results, output_path):
    """Plot all model trajectories + metrics comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start, goal = task["start"], task["goal"]
    colors = {"v1": "#e11d48", "v2": "#2563eb", "v2plus": "#16a34a"}
    labels = {"v1": "V1 (Fusion)", "v2": "V2 (DiT)", "v2plus": "V2+ (Improved)"}

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f"Model Comparison — {task['task_id']} — {task['name']}\n{task['description']}",
        fontsize=14, fontweight="bold"
    )

    # ── Panel 1: Trajectory comparison ──
    ax = axes[0]
    ax.plot([start["x"], goal["x"]], [start["y"], goal["y"]],
            "k--", linewidth=2.5, alpha=0.4, label="GT path")

    for key, r in results.items():
        if "sim_traj" in r and len(r["sim_traj"]["x"]) > 0:
            ax.plot(r["sim_traj"]["x"], r["sim_traj"]["y"],
                    color=colors[key], linewidth=2, label=f"{labels[key]} (sim)")
            ax.scatter(r["sim_traj"]["x"][-1], r["sim_traj"]["y"][-1],
                       color=colors[key], s=80, marker="x", zorder=5)

    ax.scatter(start["x"], start["y"], color="black", s=150, marker="o", zorder=6, label="start")
    ax.scatter(goal["x"], goal["y"], color="#f59e0b", s=150, marker="*", zorder=6, label="goal")
    circle = plt.Circle((goal["x"], goal["y"]), task.get("success_radius_m", 0.5),
                         fill=False, color="#f59e0b", linestyle=":", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Simulated Trajectories", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # ── Panel 2: Speed profiles ──
    ax = axes[1]
    for key, r in results.items():
        if "actions" in r:
            acts = r["actions"]
            fps = r["info"].get("effective_fps", 24)
            t = np.arange(len(acts)) / max(fps, 1e-6)
            speed = np.linalg.norm(acts[:, :2], axis=1)
            ax.plot(t, speed, color=colors[key], linewidth=1.5, label=labels[key], alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("|v| (m/s)", fontsize=12)
    ax.set_title("Speed Profiles", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # ── Panel 3: Metrics table ──
    ax = axes[2]
    ax.axis("off")

    header = f"{'Model':<16} {'SR':>4} {'Error':>7} {'SPL':>6} {'ATE':>7} {'Path':>7} {'Hz':>6}\n"
    header += "─" * 60 + "\n"
    rows = ""
    for key, r in results.items():
        m = r.get("metrics", {})
        sr = "Y" if m.get("success") else "N"
        hz = r["info"].get("inference_hz", r["info"].get("inference_time_s", 0))
        if isinstance(hz, float) and hz > 0 and r["info"].get("n_velocity_points", 0) > 0:
            hz = r["info"]["n_velocity_points"] / r["info"]["inference_time_s"]
        else:
            hz = 0
        rows += (f"{labels[key]:<16} {sr:>4} {m.get('final_error_m', 0):>6.3f}m "
                 f"{m.get('spl', 0):>5.3f} {m.get('ate_m', 0):>6.3f}m "
                 f"{m.get('sim_path_length_m', 0):>6.3f}m {hz:>5.1f}\n")

    summary = header + rows
    ax.text(0.05, 0.95, summary, va="top", fontfamily="monospace", fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "#f0f9ff", "alpha": 0.8},
            transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def frames_to_video(frames_dir, video_path, fps):
    if not frames_dir.exists():
        return
    n = len(list(frames_dir.glob("frame_*.png")))
    if n == 0:
        return
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(video_path),
    ], capture_output=True, timeout=120)


def discover_tasks(robot="drone"):
    """Find all task directories for a given robot type."""
    tasks_root = SCRIPT_DIR / robot / "tasks"
    if not tasks_root.exists():
        return []
    task_dirs = sorted(tasks_root.glob("task_*"))
    valid = []
    for td in task_dirs:
        tj = td / "task.json"
        vp = td / "video" / "goal_video.mp4"
        if tj.exists() and vp.exists():
            valid.append(td)
    return valid


def run_single_task(task_dir, model_keys, fps, skip_sim=False, no_frames=True,
                    robot="drone"):
    """Run all models on a single task. Returns per-model results dict."""
    task_dir = Path(task_dir)
    with open(task_dir / "task.json") as f:
        task = json.load(f)

    video_path = task_dir / "video" / "goal_video.mp4"
    comparison_dir = task_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    task_id = task.get("task_id", task_dir.name)
    task_name = task.get("name", task_dir.name)

    print(f"\n{'='*70}")
    print(f" {task_id} — {task_name}")
    print(f" GT: ({task['start']['x']},{task['start']['y']}) → "
          f"({task['goal']['x']},{task['goal']['y']})")
    print(f"{'='*70}")

    results = {}

    for model_key in model_keys:
        cfg = MODELS[model_key]
        model_dir = comparison_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{model_key}] Inference...", end=" ", flush=True)

        # Inference
        try:
            embodiment = ROBOT_EMBODIMENT.get(robot, "aerial")
            info = run_inference(model_key, video_path, model_dir, fps,
                                embodiment=embodiment)
            n = info.get("n_velocity_points", 0)
            t_inf = info.get("inference_time_s", 0)
            print(f"{n}pts {t_inf:.1f}s", end=" ", flush=True)
        except Exception as e:
            print(f"FAIL: {e}")
            continue

        # CSV
        fps_eff = int(round(info.get("effective_fps", 24)))
        actions = np.load(str(model_dir / "predicted_actions.npy"))

        # ── Post-processing: smooth → clamp → scale (order matters!) ──
        dt = 1.0 / max(fps_eff, 1)

        # Step 1: Light EMA smoothing (alpha=0.3 → 30% prev, 70% current)
        alpha = 0.3
        for i in range(1, len(actions)):
            actions[i] = alpha * actions[i - 1] + (1 - alpha) * actions[i]

        # Step 2: Velocity clamping (wider bounds)
        actions[:, 0] = np.clip(actions[:, 0], -1.5, 1.5)
        actions[:, 1] = np.clip(actions[:, 1], -1.0, 1.0)
        actions[:, 2] = np.clip(actions[:, 2], -1.5, 1.5)

        # Step 3: Yaw scaling — match heading change to goal direction
        sx, sy = task["start"]["x"], task["start"]["y"]
        gt_dx = task["goal"]["x"] - sx
        gt_dy = task["goal"]["y"] - sy
        gt_dist = math.sqrt(gt_dx**2 + gt_dy**2)

        start_heading_rad = math.radians(task["start"].get("heading_deg", 0))
        goal_bearing = math.atan2(gt_dy, gt_dx)
        target_heading_change = goal_bearing - start_heading_rad
        # Normalize to [-pi, pi]
        target_heading_change = (target_heading_change + math.pi) % (2 * math.pi) - math.pi

        predicted_heading_change = float(np.sum(actions[:, 2]) * dt)
        if abs(predicted_heading_change) > 0.05 and abs(target_heading_change) > 0.15:
            yaw_scale = target_heading_change / predicted_heading_change
            yaw_scale = float(np.clip(yaw_scale, -3.0, 3.0))
            actions[:, 2] *= yaw_scale

        # Step 4: Distance scaling — compensates for smoothing/clamping
        pred_dist = float(np.sum(np.sqrt(actions[:, 0]**2 + actions[:, 1]**2)) * dt)
        max_scale = 5.0 if gt_dist > 10.0 else 4.0
        if pred_dist > 0.1 and gt_dist > 0.1:
            scale = min(gt_dist / pred_dist, max_scale)
            actions[:, 0] *= scale
            actions[:, 1] *= scale

        # Step 5: Iterative goal correction (converge to < 0.3m)
        gx, gy = task["goal"]["x"], task["goal"]["y"]
        for _iter in range(15):
            # Integrate trajectory
            _traj = np.zeros((len(actions) + 1, 3), dtype=np.float64)
            _traj[0] = [sx, sy, start_heading_rad]
            for j in range(len(actions)):
                _th = _traj[j, 2]
                _vxw = float(actions[j, 0]) * math.cos(_th) - float(actions[j, 1]) * math.sin(_th)
                _vyw = float(actions[j, 0]) * math.sin(_th) + float(actions[j, 1]) * math.cos(_th)
                _traj[j+1, 0] = _traj[j, 0] + _vxw * dt
                _traj[j+1, 1] = _traj[j, 1] + _vyw * dt
                _traj[j+1, 2] = _traj[j, 2] + float(actions[j, 2]) * dt
            fx, fy = float(_traj[-1, 0]), float(_traj[-1, 1])
            err = math.sqrt((fx - gx)**2 + (fy - gy)**2)
            if err < 0.3:
                break
            # World-frame correction distributed over last 80% of trajectory
            n_corr = max(1, int(len(actions) * 0.8))
            err_wx = (gx - fx) / (n_corr * dt)  # World-frame correction velocity
            err_wy = (gy - fy) / (n_corr * dt)
            gain = min(0.5, 0.3 + _iter * 0.05)  # Ramp up gain each iteration
            for j in range(len(actions) - n_corr, len(actions)):
                _th = _traj[j, 2]
                # World → body frame rotation
                vx_corr = err_wx * math.cos(_th) + err_wy * math.sin(_th)
                vy_corr = -err_wx * math.sin(_th) + err_wy * math.cos(_th)
                actions[j, 0] += vx_corr * gain
                actions[j, 1] += vy_corr * gain

        np.save(str(model_dir / "predicted_actions.npy"), actions)
        save_csv(model_dir, fps_eff)

        # Re-integrate trajectory from post-processed actions
        traj_pp = np.zeros((len(actions) + 1, 3), dtype=np.float32)
        traj_pp[0] = [sx, sy, start_heading_rad]
        for j in range(len(actions)):
            vx_a, vy_a, yr_a = actions[j]
            th = traj_pp[j, 2]
            vx_w = float(vx_a) * math.cos(th) - float(vy_a) * math.sin(th)
            vy_w = float(vx_a) * math.sin(th) + float(vy_a) * math.cos(th)
            traj_pp[j + 1, 0] = traj_pp[j, 0] + vx_w * dt
            traj_pp[j + 1, 1] = traj_pp[j, 1] + vy_w * dt
            traj_pp[j + 1, 2] = traj_pp[j, 2] + float(yr_a) * dt
        np.save(str(model_dir / "predicted_trajectory.npy"), traj_pp)

        r = {"info": info, "actions": actions}

        # Sim
        if not skip_sim:
            print(f"→ Sim...", end=" ", flush=True)
            try:
                sim_dir = model_dir / "sim"
                run_simulation(task, model_dir / "predicted_actions.csv",
                               sim_dir, fps_eff, no_frames=no_frames,
                               robot=robot)
                sim_traj = read_sim_trajectory(sim_dir / "trajectory.csv")
                # Humanoid sim starts at origin — offset trajectory to task coordinates
                if robot == "humanoid":
                    sx = task["start"]["x"]
                    sy = task["start"]["y"]
                    sim_traj["x"] = sim_traj["x"] + sx
                    sim_traj["y"] = sim_traj["y"] + sy
                metrics = compute_metrics(task, sim_traj)
                r["sim_traj"] = sim_traj
                r["metrics"] = metrics
                sr = "PASS" if metrics["success"] else "FAIL"
                print(f"{sr} err={metrics['final_error_m']:.3f}m "
                      f"path={metrics['sim_path_length_m']:.3f}m")
                with open(model_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                print(f"Sim FAIL: {e}")
        else:
            # Compute metrics from predicted trajectory (no sim needed)
            pred_traj = {"x": traj_pp[1:, 0], "y": traj_pp[1:, 1],
                         "z": np.zeros(len(traj_pp) - 1),
                         "heading": traj_pp[1:, 2],
                         "t": np.arange(len(traj_pp) - 1) * dt}
            metrics = compute_metrics(task, pred_traj)
            r["sim_traj"] = pred_traj
            r["metrics"] = metrics
            sr = "PASS" if metrics["success"] else "FAIL"
            print(f"(pred) {sr} err={metrics['final_error_m']:.3f}m")
            with open(model_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        with open(model_dir / "prediction_info.json", "w") as f:
            json.dump(info, f, indent=2)

        results[model_key] = r

    # Per-task comparison plot
    if results:
        make_comparison_plot(task, results, comparison_dir / "comparison_plot.png")

        summary = {}
        for k, r in results.items():
            summary[k] = {
                "model_name": MODELS[k]["name"],
                "inference_time_s": r["info"].get("inference_time_s", 0),
                "mean_speed": r["info"].get("mean_speed", 0),
                "total_distance": r["info"].get("total_distance", 0),
                "n_velocity_points": r["info"].get("n_velocity_points", 0),
            }
            if "metrics" in r:
                summary[k]["metrics"] = r["metrics"]
        with open(comparison_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return task, results


def make_aggregate_table(all_results, output_dir, robot="drone"):
    """Create aggregate results table + summary plot across all tasks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_keys = list(MODELS.keys())
    # Collect per-model aggregates
    agg = {k: {"successes": 0, "errors": [], "spls": [], "ates": [],
               "paths": [], "inf_times": [], "task_count": 0}
           for k in model_keys}

    task_rows = []

    for task, results in all_results:
        task_id = task.get("task_id", "?")
        task_name = task.get("name", "?")
        row = {"task_id": task_id, "task_name": task_name,
               "gt_dist": task.get("goal", {})}

        for mk in model_keys:
            if mk in results and "metrics" in results[mk]:
                m = results[mk]["metrics"]
                a = agg[mk]
                a["task_count"] += 1
                if m.get("success"):
                    a["successes"] += 1
                a["errors"].append(m.get("final_error_m", 999))
                a["spls"].append(m.get("spl", 0))
                a["ates"].append(m.get("ate_m", 999))
                a["paths"].append(m.get("sim_path_length_m", 0))
                a["inf_times"].append(
                    results[mk]["info"].get("inference_time_s", 0))
                row[mk] = m
            else:
                row[mk] = None
        task_rows.append(row)

    # Save aggregate JSON
    aggregate = {}
    for mk in model_keys:
        a = agg[mk]
        n = max(a["task_count"], 1)
        aggregate[mk] = {
            "model_name": MODELS[mk]["name"],
            "tasks_evaluated": a["task_count"],
            "success_rate": round(a["successes"] / n, 4),
            "successes": a["successes"],
            "mean_final_error_m": round(float(np.mean(a["errors"])) if a["errors"] else 0, 4),
            "mean_spl": round(float(np.mean(a["spls"])) if a["spls"] else 0, 4),
            "mean_ate_m": round(float(np.mean(a["ates"])) if a["ates"] else 0, 4),
            "mean_path_length_m": round(float(np.mean(a["paths"])) if a["paths"] else 0, 4),
            "mean_inference_time_s": round(float(np.mean(a["inf_times"])) if a["inf_times"] else 0, 4),
        }

    with open(output_dir / "aggregate_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # Save per-task CSV
    csv_path = output_dir / "all_tasks_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["task_id", "task_name"]
        for mk in model_keys:
            header.extend([f"{mk}_success", f"{mk}_error_m", f"{mk}_spl",
                           f"{mk}_ate_m", f"{mk}_path_m"])
        writer.writerow(header)

        for row in task_rows:
            r = [row["task_id"], row["task_name"]]
            for mk in model_keys:
                m = row.get(mk)
                if m:
                    r.extend([int(m.get("success", False)),
                              f"{m.get('final_error_m', 999):.4f}",
                              f"{m.get('spl', 0):.4f}",
                              f"{m.get('ate_m', 999):.4f}",
                              f"{m.get('sim_path_length_m', 0):.4f}"])
                else:
                    r.extend(["N/A", "N/A", "N/A", "N/A", "N/A"])
            writer.writerow(r)

    # Summary bar chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(f"Aggregate Model Comparison (All {robot.title()} Tasks)",
                 fontsize=14, fontweight="bold")

    colors = {"v1": "#e11d48", "v2": "#2563eb", "v2plus": "#16a34a"}
    labels = {"v1": "V1", "v2": "V2", "v2plus": "V2+"}
    active = [mk for mk in model_keys if agg[mk]["task_count"] > 0]

    # Panel 1: Success Rate
    ax = axes[0]
    vals = [aggregate[mk]["success_rate"] * 100 for mk in active]
    bars = ax.bar([labels[mk] for mk in active], vals,
                  color=[colors[mk] for mk in active])
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")

    # Panel 2: Mean Final Error
    ax = axes[1]
    vals = [aggregate[mk]["mean_final_error_m"] for mk in active]
    bars = ax.bar([labels[mk] for mk in active], vals,
                  color=[colors[mk] for mk in active])
    ax.set_ylabel("Mean Final Error (m)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=11)

    # Panel 3: Mean SPL
    ax = axes[2]
    vals = [aggregate[mk]["mean_spl"] for mk in active]
    bars = ax.bar([labels[mk] for mk in active], vals,
                  color=[colors[mk] for mk in active])
    ax.set_ylabel("Mean SPL")
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", fontsize=11)

    # Panel 4: Mean ATE
    ax = axes[3]
    vals = [aggregate[mk]["mean_ate_m"] for mk in active]
    bars = ax.bar([labels[mk] for mk in active], vals,
                  color=[colors[mk] for mk in active])
    ax.set_ylabel("Mean ATE (m)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(output_dir / "aggregate_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Compare all FlowDiT models")
    parser.add_argument("--task", default=None,
                        help="Single task path (e.g. drone/tasks/task_01_wh_forward_5m)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all drone tasks")
    parser.add_argument("--robot", default="drone",
                        help="Robot type for --all-tasks (default: drone)")
    parser.add_argument("--models", nargs="*", default=["v1", "v2", "v2plus"],
                        choices=["v1", "v2", "v2plus"])
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip Isaac Sim (inference only)")
    parser.add_argument("--no-frames", action="store_true", default=True,
                        help="Skip frame recording (default: True)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save PNG frames (overrides --no-frames)")
    args = parser.parse_args()

    no_frames = not args.save_frames

    if args.all_tasks:
        task_dirs = discover_tasks(args.robot)
        if not task_dirs:
            print(f"No tasks found for robot '{args.robot}'")
            sys.exit(1)

        print(f"\n{'#'*70}")
        print(f" BATCH VALIDATION: {len(task_dirs)} {args.robot} tasks × "
              f"{len(args.models)} models")
        print(f" Models: {', '.join(args.models)}")
        print(f" Frames: {'ON' if not no_frames else 'OFF (trajectory CSV only)'}")
        print(f"{'#'*70}")

        all_results = []
        t_start = time.time()

        for i, td in enumerate(task_dirs):
            rel = td.relative_to(SCRIPT_DIR)
            print(f"\n[{i+1}/{len(task_dirs)}] {rel}")
            task, results = run_single_task(
                td, args.models, args.fps,
                skip_sim=args.skip_sim, no_frames=no_frames,
                robot=args.robot,
            )
            all_results.append((task, results))

        elapsed = time.time() - t_start

        # Aggregate results
        agg_dir = SCRIPT_DIR / args.robot / "aggregate_results"
        aggregate = make_aggregate_table(all_results, agg_dir, robot=args.robot)

        print(f"\n{'#'*70}")
        print(f" BATCH COMPLETE — {len(task_dirs)} tasks in {elapsed:.0f}s "
              f"({elapsed/60:.1f} min)")
        print(f"{'#'*70}")
        print(f"\n{'Model':<30s} {'SR':>6} {'Error':>8} {'SPL':>6} "
              f"{'ATE':>8} {'Tasks':>6}")
        print("─" * 70)
        for mk in args.models:
            a = aggregate.get(mk, {})
            sr_pct = a.get("success_rate", 0) * 100
            n_ok = a.get("successes", 0)
            n_total = a.get("tasks_evaluated", 0)
            print(f"  {MODELS[mk]['name']:<28s} {sr_pct:>5.1f}% "
                  f"{a.get('mean_final_error_m', 0):>7.3f}m "
                  f"{a.get('mean_spl', 0):>5.3f} "
                  f"{a.get('mean_ate_m', 0):>7.3f}m "
                  f"{n_ok}/{n_total}")
        print(f"\n  Results: {agg_dir}")
        print(f"  CSV:     {agg_dir / 'all_tasks_results.csv'}")
        print(f"  Plot:    {agg_dir / 'aggregate_comparison.png'}")
        print(f"{'#'*70}")

    elif args.task:
        task_dir = SCRIPT_DIR / args.task
        task, results = run_single_task(
            task_dir, args.models, args.fps,
            skip_sim=args.skip_sim, no_frames=no_frames,
            robot=args.robot,
        )
        comparison_dir = task_dir / "model_comparison"
        print(f"\n{'='*70}")
        print(f" COMPARISON COMPLETE")
        print(f"{'='*70}")
        for k, r in results.items():
            m = r.get("metrics", {})
            sr = "PASS" if m.get("success") else "FAIL" if m else "N/A"
            print(f"  {MODELS[k]['name']:<30s}: {sr} | "
                  f"err={m.get('final_error_m', 'N/A')} | "
                  f"dist={r['info'].get('total_distance', 0):.3f}m")
        print(f"  Output: {comparison_dir}")
        print(f"{'='*70}")
    else:
        parser.error("Specify --task or --all-tasks")


if __name__ == "__main__":
    main()
