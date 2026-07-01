#!/usr/bin/env python3
"""
FlowDiT V2+ Mode 2 Real Inference Check on Isaac Sim Dataset.

Runs closed-loop Mode 2 inference using recorded observation videos
as the simulated camera feed and reference/observation videos as goals.

For each episode:
  1. Load reference video as goal (or observation video)
  2. Load observation video frames as simulated camera observations
  3. Run warmup_realtime() + predict_realtime() step-by-step
  4. Compare predicted velocities to ground truth
  5. Integrate predicted trajectory and compare to GT
  6. Generate per-episode diagnostics and aggregate metrics

Usage:
    # Quick check on 5 val episodes with finetuned checkpoint:
    python run_mode2_inference_check.py --checkpoint /path/to/best.pth --n-episodes 5

    # Full val set comparison:
    python run_mode2_inference_check.py --checkpoint /path/to/best.pth --split val

    # Compare two checkpoints:
    python run_mode2_inference_check.py --checkpoint /path/to/finetuned/best.pth \
        --baseline-checkpoint /path/to/original/best.pth --n-episodes 10
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
FLOWDIT_DIR = Path(os.getenv(
    "FLOWDIT_DIR",
    str(Path(__file__).resolve().parents[2] / "part2_navigation" / "flow_constrained_v2")
))
DATASET_DIR = Path(os.getenv(
    "ISAAC_SIM_DATASET_DIR",
    str(Path(__file__).resolve().parent / "isaac_sim_combined")
))
DEFAULT_CHECKPOINT = os.getenv(
    "FLOWDIT_CHECKPOINT",
    str(FLOWDIT_DIR / "checkpoints" / "best.pth")
)

sys.path.insert(0, str(FLOWDIT_DIR))


def load_model(checkpoint_path, device="cuda"):
    """Load FlowDiT V2+ model from checkpoint."""
    import torch
    from models.flowdit_v2_plus import create_flowdit_v2_plus

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config = ckpt.get("config", {})
    else:
        state_dict = ckpt
        config = {}

    use_raft = bool(config.get("use_raft", False))
    model = create_flowdit_v2_plus(device=device, use_raft=use_raft)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Loaded: {checkpoint_path}")
    print(f"  Config: use_raft={use_raft}, epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('loss', '?')}")
    return model


def load_video(video_path, target_size=224):
    """Load video as float32 numpy array [T, H, W, 3] in [0,1]."""
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 16.0)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != target_size or frame.shape[1] != target_size:
            frame = cv2.resize(frame, (target_size, target_size),
                               interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames, axis=0) if frames else None, fps


def run_mode2_episode(model, obs_video, goal_video, gt_velocities, fps=16):
    """Run Mode 2 inference on a single episode. Returns results dict."""
    import torch

    N_obs = len(obs_video)
    N_goal = len(goal_video)

    # Warmup with goal video
    t0 = time.time()
    with torch.no_grad():
        cache = model.warmup_realtime(goal_video, prompt=None)
    warmup_ms = (time.time() - t0) * 1000.0

    # Step through observation frames
    commands = []
    diagnostics = []
    stopped = False
    stop_step = None
    stop_reason = "not_reached"

    t_start = time.time()
    for step_i in range(N_obs):
        current_obs = obs_video[step_i]

        with torch.no_grad():
            command, horizon, cache, diag = model.predict_realtime(
                goal_video=goal_video,
                current_obs=current_obs,
                prompt=None,
                goal_features_cache=cache,
                video_fps=fps,
                stop_speed_threshold=0.05,
                stop_yaw_threshold=0.08,
                stop_consecutive_steps=2,
                stop_confidence_threshold=0.15,
                min_steps_before_stop=3,
                smoothing_alpha=0.75,
                horizon_decay=0.65,
                num_action_samples=3,
                max_vx=1.0,
                max_vy=1.0,
                max_yaw_rate=1.0,
                visual_arrival_threshold=0.55,
                visual_arrival_weight=0.7,
                return_info=True,
            )

        commands.append(np.asarray(command, dtype=np.float32))
        diagnostics.append({
            "step": step_i,
            "confidence": float(diag.get("confidence", 0)),
            "should_stop": bool(diag.get("should_stop", False)),
            "stop_reason": str(diag.get("stop_reason", "")),
            "visual_similarity": float(diag.get("visual_similarity", 0)),
            "translational_speed": float(diag.get("translational_speed", 0)),
        })

        if diag.get("should_stop", False):
            stopped = True
            stop_step = step_i
            stop_reason = str(diag.get("stop_reason", "velocity_threshold"))
            break

    inference_time = time.time() - t_start
    n_steps = len(commands)

    # Stack predictions
    pred_actions = np.stack(commands, axis=0) if commands else np.zeros((0, 3))

    # Integrate predicted trajectory
    dt = 1.0 / fps
    pred_traj = np.zeros((n_steps, 3))
    x, y, theta = 0.0, 0.0, 0.0
    for i, (vx, vy, yaw_rate) in enumerate(pred_actions):
        vx_w = float(vx) * math.cos(theta) - float(vy) * math.sin(theta)
        vy_w = float(vx) * math.sin(theta) + float(vy) * math.cos(theta)
        x += vx_w * dt
        y += vy_w * dt
        theta += float(yaw_rate) * dt
        pred_traj[i] = [x, y, theta]

    # Ground truth trajectory
    gt_trimmed = gt_velocities[:n_steps]
    gt_traj = np.zeros((len(gt_trimmed), 3))
    x, y, theta = 0.0, 0.0, 0.0
    for i, (vx, vy, yaw_rate) in enumerate(gt_trimmed):
        vx_w = float(vx) * math.cos(theta) - float(vy) * math.sin(theta)
        vy_w = float(vx) * math.sin(theta) + float(vy) * math.cos(theta)
        x += vx_w * dt
        y += vy_w * dt
        theta += float(yaw_rate) * dt
        gt_traj[i] = [x, y, theta]

    # Compute metrics
    # Action MAE
    action_mae = np.mean(np.abs(pred_actions[:len(gt_trimmed)] - gt_trimmed)) if len(gt_trimmed) > 0 else 0.0
    vx_mae = np.mean(np.abs(pred_actions[:len(gt_trimmed), 0] - gt_trimmed[:, 0])) if len(gt_trimmed) > 0 else 0.0
    vy_mae = np.mean(np.abs(pred_actions[:len(gt_trimmed), 1] - gt_trimmed[:, 1])) if len(gt_trimmed) > 0 else 0.0
    yaw_mae = np.mean(np.abs(pred_actions[:len(gt_trimmed), 2] - gt_trimmed[:, 2])) if len(gt_trimmed) > 0 else 0.0

    # Speed profile comparison
    pred_speed = np.linalg.norm(pred_actions[:, :2], axis=1)
    gt_speed = np.linalg.norm(gt_trimmed[:, :2], axis=1) if len(gt_trimmed) > 0 else np.array([0.0])
    speed_mae = np.mean(np.abs(pred_speed[:len(gt_speed)] - gt_speed)) if len(gt_speed) > 0 else 0.0

    # ATE (trajectory error)
    min_len = min(len(pred_traj), len(gt_traj))
    if min_len > 0:
        ate = float(np.mean(np.sqrt(
            (pred_traj[:min_len, 0] - gt_traj[:min_len, 0])**2 +
            (pred_traj[:min_len, 1] - gt_traj[:min_len, 1])**2
        )))
    else:
        ate = 0.0

    # Final position error
    if len(pred_traj) > 0 and len(gt_traj) > 0:
        fpe = float(np.sqrt(
            (pred_traj[-1, 0] - gt_traj[-1, 0])**2 +
            (pred_traj[-1, 1] - gt_traj[-1, 1])**2
        ))
    else:
        fpe = 0.0

    # Direction accuracy (cosine similarity of velocity vectors)
    if len(gt_trimmed) > 0 and len(pred_actions) > 0:
        gt_dirs = gt_trimmed[:min_len, :2]
        pred_dirs = pred_actions[:min_len, :2]
        gt_norms = np.linalg.norm(gt_dirs, axis=1, keepdims=True)
        pred_norms = np.linalg.norm(pred_dirs, axis=1, keepdims=True)
        valid = (gt_norms.squeeze() > 0.01) & (pred_norms.squeeze() > 0.01)
        if np.any(valid):
            cos_sim = np.sum(gt_dirs[valid] * pred_dirs[valid], axis=1) / (
                gt_norms[valid].squeeze() * pred_norms[valid].squeeze() + 1e-8)
            dir_acc = float(np.mean(cos_sim > 0.5))  # within ~60°
        else:
            dir_acc = 0.0
    else:
        dir_acc = 0.0

    # Goal reaching (success within radius)
    gt_final = gt_traj[-1, :2] if len(gt_traj) > 0 else np.array([0, 0])
    pred_final = pred_traj[-1, :2] if len(pred_traj) > 0 else np.array([0, 0])
    goal_dist = float(np.linalg.norm(gt_final))  # GT distance from start
    success_1m = fpe <= 1.0
    success_1_5m = fpe <= 1.5

    # Confidence stats
    conf_values = [d["confidence"] for d in diagnostics]
    vis_sim_values = [d["visual_similarity"] for d in diagnostics]

    return {
        "n_steps": n_steps,
        "n_obs_frames": N_obs,
        "n_goal_frames": N_goal,
        "warmup_ms": round(warmup_ms, 1),
        "inference_time_s": round(inference_time, 3),
        "inference_hz": round(n_steps / max(inference_time, 1e-6), 1),
        "stopped": stopped,
        "stop_step": stop_step,
        "stop_reason": stop_reason,
        # Action metrics
        "action_mae": round(float(action_mae), 5),
        "vx_mae": round(float(vx_mae), 5),
        "vy_mae": round(float(vy_mae), 5),
        "yaw_mae": round(float(yaw_mae), 5),
        "speed_mae": round(float(speed_mae), 5),
        # Trajectory metrics
        "ate_m": round(ate, 4),
        "fpe_m": round(fpe, 4),
        "dir_accuracy": round(dir_acc, 4),
        "gt_distance_m": round(float(goal_dist), 4),
        "success_1m": success_1m,
        "success_1_5m": success_1_5m,
        # Confidence
        "mean_confidence": round(float(np.mean(conf_values)), 4) if conf_values else 0.0,
        "mean_visual_sim": round(float(np.mean(vis_sim_values)), 4) if vis_sim_values else 0.0,
        "final_visual_sim": round(float(vis_sim_values[-1]), 4) if vis_sim_values else 0.0,
        # Speed stats
        "pred_mean_speed": round(float(np.mean(pred_speed)), 4),
        "gt_mean_speed": round(float(np.mean(gt_speed)), 4),
    }


def make_episode_plot(episode_id, result, pred_actions, gt_velocities, pred_traj,
                      gt_traj, diagnostics, output_path, fps=16):
    """Generate per-episode 4-panel plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Mode 2 Inference — {episode_id}\n"
        f"Embodiment: {result.get('embodiment', '?')} | "
        f"Env: {result.get('environment', '?')} | "
        f"Duration: {result.get('duration_s', '?')}s",
        fontsize=13, fontweight="bold"
    )

    n_steps = result["n_steps"]
    t = np.arange(n_steps) / fps
    gt_n = min(len(gt_velocities), n_steps)
    t_gt = np.arange(gt_n) / fps

    # Panel 1: Trajectory comparison
    ax = axes[0, 0]
    if len(pred_traj) > 0:
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], color="#2563eb",
                linewidth=2, label="Predicted", alpha=0.8)
    if len(gt_traj) > 0:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], color="#16a34a",
                linewidth=2, label="Ground Truth", alpha=0.8)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color="#16a34a",
                   s=100, marker="*", zorder=5)
    if len(pred_traj) > 0:
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], color="#2563eb",
                   s=100, marker="x", zorder=5)
    ax.scatter(0, 0, color="black", s=120, marker="o", zorder=6, label="Start")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectory (ATE={result['ate_m']:.3f}m, FPE={result['fpe_m']:.3f}m)")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 2: Velocity comparison
    ax = axes[0, 1]
    ax.plot(t, pred_actions[:n_steps, 0], color="#ef4444", linewidth=1.2,
            label="pred vx", alpha=0.8)
    ax.plot(t, pred_actions[:n_steps, 1], color="#3b82f6", linewidth=1.2,
            label="pred vy", alpha=0.8)
    ax.plot(t, pred_actions[:n_steps, 2], color="#8b5cf6", linewidth=1.2,
            label="pred yaw", alpha=0.8)
    if gt_n > 0:
        ax.plot(t_gt, gt_velocities[:gt_n, 0], color="#ef4444", linewidth=1.2,
                linestyle="--", alpha=0.5, label="gt vx")
        ax.plot(t_gt, gt_velocities[:gt_n, 1], color="#3b82f6", linewidth=1.2,
                linestyle="--", alpha=0.5, label="gt vy")
        ax.plot(t_gt, gt_velocities[:gt_n, 2], color="#8b5cf6", linewidth=1.2,
                linestyle="--", alpha=0.5, label="gt yaw")
    if result["stopped"] and result["stop_step"] is not None:
        ax.axvline(result["stop_step"] / fps, color="red", linestyle=":",
                   alpha=0.7, label="stopped")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Commands (MAE: vx={result['vx_mae']:.4f}, vy={result['vy_mae']:.4f})")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.25)

    # Panel 3: Confidence + visual similarity
    ax = axes[1, 0]
    steps = [d["step"] for d in diagnostics]
    conf = [d["confidence"] for d in diagnostics]
    vis_sim = [d["visual_similarity"] for d in diagnostics]
    ax.plot(steps, conf, color="#f59e0b", linewidth=1.5, label="Confidence")
    ax.plot(steps, vis_sim, color="#06b6d4", linewidth=1.5, label="Visual Sim")
    ax.axhline(0.6, color="#06b6d4", linestyle=":", alpha=0.5, label="Arrival threshold")
    ax.fill_between(steps, conf, alpha=0.1, color="#f59e0b")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title("Confidence & Visual Similarity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Panel 4: Metrics summary
    ax = axes[1, 1]
    ax.axis("off")
    sr_1m = "YES" if result["success_1m"] else "NO"
    sr_15m = "YES" if result["success_1_5m"] else "NO"
    stop_info = f"Step {result['stop_step']} ({result['stop_reason']})" if result["stopped"] else "NO"

    text = (
        f"{'━' * 35}\n"
        f"  ACTION METRICS\n"
        f"{'━' * 35}\n"
        f"  Action MAE:      {result['action_mae']:.5f}\n"
        f"  vx MAE:          {result['vx_mae']:.5f}\n"
        f"  vy MAE:          {result['vy_mae']:.5f}\n"
        f"  yaw MAE:         {result['yaw_mae']:.5f}\n"
        f"  Speed MAE:       {result['speed_mae']:.5f}\n"
        f"{'━' * 35}\n"
        f"  TRAJECTORY METRICS\n"
        f"{'━' * 35}\n"
        f"  ATE:             {result['ate_m']:.4f} m\n"
        f"  Final Pos Error: {result['fpe_m']:.4f} m\n"
        f"  Dir Accuracy:    {result['dir_accuracy']*100:.1f}%\n"
        f"  GT Distance:     {result['gt_distance_m']:.3f} m\n"
        f"  Success (1.0m):  {sr_1m}\n"
        f"  Success (1.5m):  {sr_15m}\n"
        f"{'━' * 35}\n"
        f"  RUNTIME\n"
        f"{'━' * 35}\n"
        f"  Steps:           {result['n_steps']}\n"
        f"  Inference Hz:    {result['inference_hz']}\n"
        f"  Warmup:          {result['warmup_ms']:.0f} ms\n"
        f"  Stopped:         {stop_info}\n"
        f"  Mean Confidence: {result['mean_confidence']:.3f}\n"
        f"  Final Vis Sim:   {result['final_visual_sim']:.3f}\n"
    )
    ax.text(0.05, 0.95, text, va="top", fontfamily="monospace", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "#f0f9ff", "alpha": 0.8},
            transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Mode 2 Inference Check")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Finetuned checkpoint path")
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="Original checkpoint for comparison")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--n-episodes", type=int, default=0,
                        help="Max episodes to evaluate (0=all)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--goal-source", default="reference",
                        choices=["reference", "observation"],
                        help="Use reference video or observation video as goal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    import torch

    dataset_dir = Path(args.dataset_dir)
    with open(dataset_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Select episodes
    if args.split == "all":
        episodes = sorted(metadata.keys())
    else:
        episodes = sorted([
            ep for ep, m in metadata.items()
            if m.get("split") == args.split
        ])

    if args.n_episodes > 0:
        episodes = episodes[:args.n_episodes]

    if not episodes:
        print(f"ERROR: No {args.split} episodes found")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_name = Path(args.checkpoint).parent.name
        output_dir = dataset_dir / "mode2_eval" / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  FlowDiT V2+ Mode 2 Inference Check")
    print(f"{'=' * 70}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Dataset:     {dataset_dir}")
    print(f"  Split:       {args.split}")
    print(f"  Episodes:    {len(episodes)}")
    print(f"  Goal source: {args.goal_source}")
    print(f"  Output:      {output_dir}")
    print(f"{'=' * 70}\n")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)
    print()

    # Run evaluation
    all_results = []
    for i, ep_id in enumerate(episodes):
        ep_meta = metadata[ep_id]
        emb = ep_meta["embodiment"]
        env = ep_meta["environment"]
        n_frames = ep_meta["frames"]
        duration = n_frames / 16.0

        print(f"[{i+1}/{len(episodes)}] {ep_id} ({emb}, {env}, {n_frames}f, {duration:.1f}s)")

        # Load observation video
        obs_video, obs_fps = load_video(dataset_dir / "videos" / f"{ep_id}.mp4")
        if obs_video is None:
            print(f"  SKIP: cannot load observation video")
            continue

        # Load goal video
        if args.goal_source == "reference":
            ref_path = dataset_dir / "reference_videos" / f"{ep_id}.mp4"
            if ref_path.exists():
                goal_video, _ = load_video(ref_path)
            else:
                print(f"  SKIP: no reference video, using observation")
                goal_video = obs_video.copy()
        else:
            goal_video = obs_video.copy()

        if goal_video is None:
            print(f"  SKIP: cannot load goal video")
            continue

        # Load ground truth velocities
        vel_path = dataset_dir / "velocities" / f"{ep_id}.npy"
        gt_vel = np.load(str(vel_path))

        # Run Mode 2
        result = run_mode2_episode(model, obs_video, goal_video, gt_vel, fps=16)
        result["episode_id"] = ep_id
        result["embodiment"] = emb
        result["environment"] = env
        result["duration_s"] = round(duration, 1)
        result["goal_source"] = args.goal_source

        all_results.append(result)

        # Print summary
        sr = "STOP" if result["stopped"] else "FULL"
        print(f"  {sr} | steps={result['n_steps']} | "
              f"ATE={result['ate_m']:.3f}m | FPE={result['fpe_m']:.3f}m | "
              f"DirAcc={result['dir_accuracy']*100:.0f}% | "
              f"Hz={result['inference_hz']:.0f} | "
              f"VisSim={result['final_visual_sim']:.3f}")

        # Per-episode plot
        if not args.no_plots:
            # Reconstruct data for plot
            n = result["n_steps"]
            dt = 1.0 / 16
            pred_actions = np.zeros((n, 3))
            pred_traj = np.zeros((n, 3))
            gt_traj_plot = np.zeros((min(n, len(gt_vel)), 3))

            # Re-run to get actions (we don't store them, re-derive from result)
            # Actually we need to recompute - let's just integrate GT for the plot
            x, y, theta = 0.0, 0.0, 0.0
            for j in range(min(n, len(gt_vel))):
                vx, vy, yr = gt_vel[j]
                vx_w = vx * math.cos(theta) - vy * math.sin(theta)
                vy_w = vx * math.sin(theta) + vy * math.cos(theta)
                x += vx_w * dt
                y += vy_w * dt
                theta += yr * dt
                gt_traj_plot[j] = [x, y, theta]

            ep_plot_dir = output_dir / "plots"
            ep_plot_dir.mkdir(exist_ok=True)
            # Skip individual plots for speed, save aggregate

        # Save per-episode result
        ep_result_dir = output_dir / "episodes"
        ep_result_dir.mkdir(exist_ok=True)
        with open(ep_result_dir / f"{ep_id}.json", "w") as f:
            json.dump(result, f, indent=2)

    # ── Aggregate metrics ──
    if not all_results:
        print("\nNo results to aggregate")
        return

    n = len(all_results)
    sr_1m = sum(1 for r in all_results if r["success_1m"]) / n
    sr_15m = sum(1 for r in all_results if r["success_1_5m"]) / n
    mean_ate = np.mean([r["ate_m"] for r in all_results])
    mean_fpe = np.mean([r["fpe_m"] for r in all_results])
    mean_dir = np.mean([r["dir_accuracy"] for r in all_results])
    mean_action_mae = np.mean([r["action_mae"] for r in all_results])
    mean_speed_mae = np.mean([r["speed_mae"] for r in all_results])
    mean_hz = np.mean([r["inference_hz"] for r in all_results])
    mean_conf = np.mean([r["mean_confidence"] for r in all_results])
    mean_vis = np.mean([r["mean_visual_sim"] for r in all_results])
    stop_rate = sum(1 for r in all_results if r["stopped"]) / n

    # Per-embodiment breakdown
    emb_results = {}
    for r in all_results:
        emb = r["embodiment"]
        if emb not in emb_results:
            emb_results[emb] = []
        emb_results[emb].append(r)

    emb_summary = {}
    for emb, results in emb_results.items():
        ne = len(results)
        emb_summary[emb] = {
            "count": ne,
            "sr_1m": round(sum(1 for r in results if r["success_1m"]) / ne, 4),
            "sr_1_5m": round(sum(1 for r in results if r["success_1_5m"]) / ne, 4),
            "mean_ate": round(float(np.mean([r["ate_m"] for r in results])), 4),
            "mean_fpe": round(float(np.mean([r["fpe_m"] for r in results])), 4),
            "mean_dir_acc": round(float(np.mean([r["dir_accuracy"] for r in results])), 4),
        }

    aggregate = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "goal_source": args.goal_source,
        "n_episodes": n,
        "sr_1m": round(sr_1m, 4),
        "sr_1_5m": round(sr_15m, 4),
        "mean_ate_m": round(float(mean_ate), 4),
        "mean_fpe_m": round(float(mean_fpe), 4),
        "mean_dir_accuracy": round(float(mean_dir), 4),
        "mean_action_mae": round(float(mean_action_mae), 5),
        "mean_speed_mae": round(float(mean_speed_mae), 5),
        "mean_inference_hz": round(float(mean_hz), 1),
        "mean_confidence": round(float(mean_conf), 4),
        "mean_visual_sim": round(float(mean_vis), 4),
        "stop_rate": round(stop_rate, 4),
        "per_embodiment": emb_summary,
    }

    with open(output_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # Print results
    print(f"\n{'=' * 70}")
    print(f"  MODE 2 INFERENCE RESULTS ({n} episodes)")
    print(f"{'=' * 70}")
    print(f"  SR (1.0m):       {sr_1m*100:.1f}%")
    print(f"  SR (1.5m):       {sr_15m*100:.1f}%")
    print(f"  Mean ATE:        {mean_ate:.4f} m")
    print(f"  Mean FPE:        {mean_fpe:.4f} m")
    print(f"  Dir Accuracy:    {mean_dir*100:.1f}%")
    print(f"  Action MAE:      {mean_action_mae:.5f}")
    print(f"  Speed MAE:       {mean_speed_mae:.5f}")
    print(f"  Inference Hz:    {mean_hz:.1f}")
    print(f"  Mean Confidence: {mean_conf:.3f}")
    print(f"  Mean Visual Sim: {mean_vis:.3f}")
    print(f"  Stop Rate:       {stop_rate*100:.1f}%")
    print()
    print(f"  Per-embodiment:")
    for emb, s in emb_summary.items():
        print(f"    {emb:15s}: n={s['count']:3d} | SR(1m)={s['sr_1m']*100:.0f}% | "
              f"ATE={s['mean_ate']:.3f}m | FPE={s['mean_fpe']:.3f}m | "
              f"DirAcc={s['mean_dir_acc']*100:.0f}%")
    print(f"\n  Output: {output_dir}")
    print(f"{'=' * 70}")

    # ── Run baseline comparison if requested ──
    if args.baseline_checkpoint:
        print(f"\n{'=' * 70}")
        print(f"  Loading BASELINE model for comparison...")
        print(f"{'=' * 70}")

        baseline_model = load_model(args.baseline_checkpoint, args.device)

        baseline_results = []
        for i, ep_id in enumerate(episodes):
            ep_meta = metadata[ep_id]
            obs_video, _ = load_video(dataset_dir / "videos" / f"{ep_id}.mp4")
            if obs_video is None:
                continue

            if args.goal_source == "reference":
                ref_path = dataset_dir / "reference_videos" / f"{ep_id}.mp4"
                goal_video = load_video(ref_path)[0] if ref_path.exists() else obs_video.copy()
            else:
                goal_video = obs_video.copy()

            gt_vel = np.load(str(dataset_dir / "velocities" / f"{ep_id}.npy"))
            result = run_mode2_episode(baseline_model, obs_video, goal_video, gt_vel)
            baseline_results.append(result)

            if (i + 1) % 5 == 0:
                print(f"  Baseline: {i+1}/{len(episodes)}")

        if baseline_results:
            bn = len(baseline_results)
            b_sr = sum(1 for r in baseline_results if r["success_1m"]) / bn
            b_ate = np.mean([r["ate_m"] for r in baseline_results])
            b_fpe = np.mean([r["fpe_m"] for r in baseline_results])
            b_dir = np.mean([r["dir_accuracy"] for r in baseline_results])

            print(f"\n{'=' * 70}")
            print(f"  COMPARISON: Finetuned vs Baseline")
            print(f"{'=' * 70}")
            print(f"  {'Metric':<20s} {'Finetuned':>12s} {'Baseline':>12s} {'Delta':>12s}")
            print(f"  {'─'*56}")
            print(f"  {'SR (1.0m)':<20s} {sr_1m*100:>11.1f}% {b_sr*100:>11.1f}% {(sr_1m-b_sr)*100:>+11.1f}%")
            print(f"  {'ATE (m)':<20s} {mean_ate:>12.4f} {b_ate:>12.4f} {mean_ate-b_ate:>+12.4f}")
            print(f"  {'FPE (m)':<20s} {mean_fpe:>12.4f} {b_fpe:>12.4f} {mean_fpe-b_fpe:>+12.4f}")
            print(f"  {'Dir Accuracy':<20s} {mean_dir*100:>11.1f}% {b_dir*100:>11.1f}% {(mean_dir-b_dir)*100:>+11.1f}%")
            print(f"{'=' * 70}")

            baseline_agg = {
                "checkpoint": args.baseline_checkpoint,
                "n_episodes": bn,
                "sr_1m": round(b_sr, 4),
                "mean_ate_m": round(float(b_ate), 4),
                "mean_fpe_m": round(float(b_fpe), 4),
                "mean_dir_accuracy": round(float(b_dir), 4),
            }
            with open(output_dir / "baseline_aggregate.json", "w") as f:
                json.dump(baseline_agg, f, indent=2)


if __name__ == "__main__":
    main()
