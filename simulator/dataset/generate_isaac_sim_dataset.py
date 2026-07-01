#!/usr/bin/env python3
"""
Isaac Sim Dataset Generator for FlowDiT V2+ Mode 2
====================================================

Two-phase pipeline:
  Phase 1: Isaac Sim recordings (photorealistic observation videos)
  Phase 2: LTX reference video generation (start+goal image → video)

Embodiments:
  - Drone (Crazyflie CF2X)       z=1.5-2.0m   max 0.8 m/s
  - Human (drone cam at z=1.6m)  z=1.6m        max 0.8 m/s  (human-like walking POV)
  - Mobile Wheel (AgileX Limo)   z=ground      max 0.8 m/s

Each episode produces:
  - Observation video (224x224 @16fps MP4) — from Isaac Sim (downsampled from 720p)
  - Reference video (LTX conditioned on 720p start+goal)
  - Actions [T,3] normalized [-1,1]
  - Trajectory [T,3] world frame [x,y,theta]
  - Velocities [T,3] real m/s
  - Start image + Goal image (1280x720 PNG)
  - Prompt (language description)

Usage:
  # Phase 1 only (sim recordings):
  python generate_isaac_sim_dataset.py --phase sim --total_episodes 100

  # Phase 2 only (LTX reference videos, after Phase 1):
  python generate_isaac_sim_dataset.py --phase ltx

  # Both phases:
  python generate_isaac_sim_dataset.py --total_episodes 100

  # Dry run:
  python generate_isaac_sim_dataset.py --dry-run --total_episodes 5

Author: Jeffrin Sam
Date: March 2026
"""

import os
import sys
import json
import csv
import math
import argparse
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
SIM_ROOT = SCRIPT_DIR.parent
ISAAC_PYTHON = os.getenv("ISAAC_SIM_PYTHON", "/opt/isaacsim/python.sh")

DRONE_SIM = str(SIM_ROOT / "drone" / "drone_sim.py")
LIMO_SIM = str(SIM_ROOT / "mobile_robot" / "limo_sim.py")

RECORDINGS_DIR = SIM_ROOT / "recordings"
DATASET_DIR = SCRIPT_DIR

# LTX model (downloaded on first use)
LTX_MODEL_ID = "Lightricks/LTX-Video-0.9.5"
LTX_CACHE_DIR = os.getenv("LTX_CACHE_DIR", str(Path(__file__).resolve().parents[2] / "models" / "ltx"))


# ============================================================================
# EMBODIMENT CONFIGS
# ============================================================================

@dataclass
class EmbodimentConfig:
    name: str
    sim_type: str           # "drone" or "limo" (what simulator to use)
    max_vx: float
    max_vy: float
    max_yaw: float
    holonomic: bool
    environments: List[str]
    start_positions: Dict[str, List[dict]]  # env -> list of {x, y, z, heading_deg}
    smoothing: float = 0.85
    vx_noise: float = 0.02
    vy_noise: float = 0.01
    yaw_noise: float = 0.015


EMBODIMENTS = {
    "drone": EmbodimentConfig(
        name="drone", sim_type="drone",
        max_vx=0.8, max_vy=0.5, max_yaw=0.6,
        holonomic=True,
        environments=["warehouse", "hospital"],
        start_positions={
            "warehouse": [
                {"x": -2.5, "y": 0.0, "z": 1.5, "heading_deg": 0.0},
                {"x": 0.0, "y": -6.0, "z": 1.5, "heading_deg": 90.0},
                {"x": -3.0, "y": -3.0, "z": 1.5, "heading_deg": 45.0},
                {"x": 2.0, "y": 2.0, "z": 1.5, "heading_deg": -90.0},
                {"x": -1.0, "y": -8.0, "z": 2.0, "heading_deg": 0.0},
                {"x": 3.0, "y": -2.0, "z": 1.5, "heading_deg": 180.0},
                {"x": -4.0, "y": 4.0, "z": 1.5, "heading_deg": -45.0},
                {"x": 0.0, "y": 0.0, "z": 1.5, "heading_deg": 0.0},
                {"x": -2.0, "y": -5.0, "z": 1.8, "heading_deg": 30.0},
                {"x": 1.0, "y": -3.0, "z": 1.5, "heading_deg": -60.0},
            ],
            "hospital": [
                {"x": 5.0, "y": 5.0, "z": 1.8, "heading_deg": 0.0},
                {"x": 3.0, "y": 8.0, "z": 1.8, "heading_deg": -90.0},
                {"x": 7.0, "y": 3.0, "z": 1.8, "heading_deg": 90.0},
                {"x": 5.0, "y": 10.0, "z": 1.5, "heading_deg": 180.0},
                {"x": 2.0, "y": 5.0, "z": 1.8, "heading_deg": 45.0},
                {"x": 8.0, "y": 6.0, "z": 1.8, "heading_deg": -45.0},
                {"x": 4.0, "y": 3.0, "z": 2.0, "heading_deg": 0.0},
                {"x": 6.0, "y": 8.0, "z": 1.8, "heading_deg": -135.0},
                {"x": 3.0, "y": 3.0, "z": 1.5, "heading_deg": 60.0},
                {"x": 7.0, "y": 7.0, "z": 1.8, "heading_deg": -30.0},
            ],
        },
        smoothing=0.82, vx_noise=0.02, vy_noise=0.02, yaw_noise=0.015,
    ),
    "human": EmbodimentConfig(
        name="human", sim_type="drone",  # Use drone sim at human eye height
        max_vx=0.8, max_vy=0.3, max_yaw=0.5,
        holonomic=True,
        environments=["warehouse", "hospital"],
        start_positions={
            "warehouse": [
                # z=1.6m = human eye height, camera moves like walking person
                {"x": -2.5, "y": 0.0, "z": 1.6, "heading_deg": 0.0},
                {"x": 0.0, "y": -6.0, "z": 1.6, "heading_deg": 90.0},
                {"x": -3.0, "y": -3.0, "z": 1.6, "heading_deg": 45.0},
                {"x": 2.0, "y": 2.0, "z": 1.6, "heading_deg": -90.0},
                {"x": -1.0, "y": -8.0, "z": 1.6, "heading_deg": 0.0},
                {"x": 3.0, "y": -2.0, "z": 1.6, "heading_deg": 180.0},
                {"x": -4.0, "y": 4.0, "z": 1.6, "heading_deg": -45.0},
                {"x": 0.0, "y": 0.0, "z": 1.6, "heading_deg": 0.0},
                {"x": -2.0, "y": -5.0, "z": 1.6, "heading_deg": 30.0},
                {"x": 1.0, "y": -3.0, "z": 1.6, "heading_deg": -60.0},
            ],
            "hospital": [
                {"x": 5.0, "y": 5.0, "z": 1.6, "heading_deg": 0.0},
                {"x": 3.0, "y": 8.0, "z": 1.6, "heading_deg": -90.0},
                {"x": 7.0, "y": 3.0, "z": 1.6, "heading_deg": 90.0},
                {"x": 5.0, "y": 10.0, "z": 1.6, "heading_deg": 180.0},
                {"x": 2.0, "y": 5.0, "z": 1.6, "heading_deg": 45.0},
                {"x": 8.0, "y": 6.0, "z": 1.6, "heading_deg": -45.0},
                {"x": 4.0, "y": 3.0, "z": 1.6, "heading_deg": 0.0},
                {"x": 6.0, "y": 8.0, "z": 1.6, "heading_deg": -135.0},
                {"x": 3.0, "y": 3.0, "z": 1.6, "heading_deg": 60.0},
                {"x": 7.0, "y": 7.0, "z": 1.6, "heading_deg": -30.0},
            ],
        },
        smoothing=0.88, vx_noise=0.03, vy_noise=0.015, yaw_noise=0.02,
    ),
    "mobile_wheel": EmbodimentConfig(
        name="mobile_wheel", sim_type="limo",
        max_vx=0.8, max_vy=0.0, max_yaw=0.4,
        holonomic=False,
        environments=["warehouse", "hospital"],
        start_positions={
            "warehouse": [
                {"x": -2.5, "y": 0.0, "z": 0.0, "heading_deg": 0.0},
                {"x": 0.0, "y": -6.0, "z": 0.0, "heading_deg": 90.0},
                {"x": -3.0, "y": -3.0, "z": 0.0, "heading_deg": 45.0},
                {"x": 2.0, "y": 2.0, "z": 0.0, "heading_deg": -90.0},
                {"x": -1.0, "y": -8.0, "z": 0.0, "heading_deg": 0.0},
                {"x": 3.0, "y": -2.0, "z": 0.0, "heading_deg": 180.0},
                {"x": -4.0, "y": 4.0, "z": 0.0, "heading_deg": -45.0},
                {"x": 0.0, "y": 0.0, "z": 0.0, "heading_deg": 0.0},
                {"x": -2.0, "y": -5.0, "z": 0.0, "heading_deg": 30.0},
                {"x": 1.0, "y": -3.0, "z": 0.0, "heading_deg": -60.0},
            ],
            "hospital": [
                {"x": 5.0, "y": 5.0, "z": 0.0, "heading_deg": 0.0},
                {"x": 3.0, "y": 8.0, "z": 0.0, "heading_deg": -90.0},
                {"x": 7.0, "y": 3.0, "z": 0.0, "heading_deg": 90.0},
                {"x": 5.0, "y": 10.0, "z": 0.0, "heading_deg": 180.0},
                {"x": 2.0, "y": 5.0, "z": 0.0, "heading_deg": 45.0},
                {"x": 8.0, "y": 6.0, "z": 0.0, "heading_deg": -45.0},
                {"x": 4.0, "y": 3.0, "z": 0.0, "heading_deg": 0.0},
                {"x": 6.0, "y": 8.0, "z": 0.0, "heading_deg": -135.0},
                {"x": 3.0, "y": 3.0, "z": 0.0, "heading_deg": 60.0},
                {"x": 7.0, "y": 7.0, "z": 0.0, "heading_deg": -30.0},
            ],
        },
        smoothing=0.90, vx_noise=0.015, vy_noise=0.0, yaw_noise=0.01,
    ),
}


# ============================================================================
# PROMPTS
# ============================================================================

PROMPTS = {
    "warehouse": [
        # Object-referenced tasks (visible landmarks in Simple_Warehouse)
        "go to the yellow wall section near the metal pillar",
        "move toward the white roller shutter door ahead",
        "navigate to the grey concrete floor area by the far shelf",
        "approach the steel support column on the left",
        "go to the loading bay with the orange safety markings",
        "move forward to the metal shelving rack at the end",
        "navigate past the concrete pillar to the open floor area",
        "head to the warehouse wall with the yellow paint strip",
        "approach the industrial door on the far side",
        "move to the pallet storage zone near the wall",
        "go to the skylight area under the roof panels",
        "navigate toward the grey floor marker near the pillar",
        "move to the warehouse corner by the ventilation duct",
        "approach the bright section under the overhead lights",
        "head toward the wall-mounted equipment panel",
        "go to the open space between the two support beams",
        "navigate to the concrete floor section with the drain grate",
        "move to the yellow-striped safety boundary",
        "approach the far end of the warehouse aisle",
        "go forward to the painted floor line",
    ],
    "hospital": [
        # Object-referenced tasks (visible landmarks in Hospital)
        "go to the white corridor junction ahead",
        "navigate to the green exit sign at the end of the hall",
        "move toward the medical cart near the wall",
        "approach the double doors at the corridor end",
        "head to the nurses station counter on the right",
        "go to the blue equipment cabinet along the wall",
        "navigate past the handrail to the room entrance",
        "move to the tiled floor area near the elevator",
        "approach the window at the end of the hallway",
        "go to the bench seating area on the left",
        "navigate toward the overhead fluorescent lights",
        "move to the fire extinguisher station on the wall",
        "head to the reception desk area ahead",
        "approach the glass partition near the waiting area",
        "go to the corridor intersection with the floor marking",
        "navigate to the medication dispensing area",
        "move toward the patient room door on the right",
        "approach the hand sanitizer station on the wall",
        "head to the linoleum floor section by the doorway",
        "go forward to the corridor bend ahead",
    ],
}


# ============================================================================
# TRAJECTORY GENERATION (CSV format for Isaac Sim replay)
# ============================================================================

OBSTACLE_BOXES = {
    "warehouse": [
        # (x_min, y_min, x_max, y_max) — approximate obstacle footprints
        (-6.0, -1.5, -5.0, 1.5),   # shelf rack left
        (4.0, -1.5, 6.0, 1.5),     # shelf rack right
        (-3.0, -10.0, -2.0, -7.0), # pallet stack back
        (1.5, 3.0, 3.5, 5.0),      # crate cluster
        (-1.0, -4.0, 1.0, -3.0),   # center pillar zone
    ],
    "hospital": [
        (4.2, 6.2, 5.3, 7.3),      # nurses station
        (6.8, 4.2, 7.8, 5.3),      # equipment cart
        (1.5, 9.0, 2.5, 10.5),     # bed zone (moved away from starts)
        (5.2, 1.5, 6.3, 2.8),      # reception desk
        (9.0, 8.0, 10.5, 9.5),     # far corridor obstacle
    ],
}

# Walls / bounding box per environment
ENV_BOUNDS = {
    "warehouse": {"x_min": -8.0, "x_max": 8.0, "y_min": -12.0, "y_max": 8.0},
    "hospital":  {"x_min": 0.0,  "x_max": 12.0, "y_min": 0.0,   "y_max": 14.0},
}

COLLISION_RADIUS = 0.35  # robot clearance in meters


def _check_collision(x: float, y: float, env_name: str) -> bool:
    """Return True if (x, y) collides with obstacles or walls."""
    bounds = ENV_BOUNDS.get(env_name, ENV_BOUNDS["warehouse"])
    margin = COLLISION_RADIUS
    if (x < bounds["x_min"] + margin or x > bounds["x_max"] - margin or
            y < bounds["y_min"] + margin or y > bounds["y_max"] - margin):
        return True
    for (bx0, by0, bx1, by1) in OBSTACLE_BOXES.get(env_name, []):
        if (bx0 - margin < x < bx1 + margin and
                by0 - margin < y < by1 + margin):
            return True
    return False


def generate_trajectory_csv(
    emb: EmbodimentConfig,
    output_path: Path,
    duration_sec: float,
    rng: np.random.RandomState,
    start_pos: dict,
    env_name: str = "warehouse",
    max_distance_m: float = 4.0,
    max_heading_change_rad: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a collision-free velocity trajectory CSV for Isaac Sim CSV replay.

    CRITICAL CONSTRAINTS:
      - Goal visible from start (max distance + heading limit)
      - No collisions with walls or obstacles
      - Speed 0.2 to 0.8 m/s
      - Duration: 5, 10 or 15 seconds

    Returns:
        velocities: [T, 3] real m/s [vx, vy, yaw_rate]
        trajectory: [T, 3] integrated [x, y, theta] in world frame (relative to start)
    """
    fps = 16
    T = int(duration_sec * fps)

    velocities = np.zeros((T, 3), dtype=np.float32)

    # Phase durations
    accel_end = int(T * rng.uniform(0.05, 0.15))
    cruise_end = int(T * rng.uniform(0.65, 0.85))

    # Target cruise speed — 0.2 to 0.8 m/s
    cruise_vx = rng.uniform(0.2, emb.max_vx)
    cruise_vy = 0.0
    if emb.holonomic:
        cruise_vy = rng.uniform(-0.15, 0.15) * emb.max_vy

    # Gentle heading changes — keep total heading within ±max_heading_change_rad
    n_turns = rng.randint(0, 3)
    turn_frames = sorted(rng.randint(
        accel_end, max(cruise_end, accel_end + 1), size=max(n_turns, 1)
    ))
    turn_magnitudes = rng.uniform(-0.2, 0.2, size=len(turn_frames)) * emb.max_yaw
    current_yaw_target = 0.0

    for t in range(T):
        if t < accel_end:
            progress = (t + 1) / max(accel_end, 1)
            vx = cruise_vx * progress
            vy = cruise_vy * progress
            yaw = current_yaw_target * progress
        elif t < cruise_end:
            vx = cruise_vx
            vy = cruise_vy
            yaw = current_yaw_target
            for i, tf in enumerate(turn_frames):
                if abs(t - tf) < 5:
                    yaw = turn_magnitudes[i]
                    current_yaw_target = turn_magnitudes[i] * 0.2
                    if emb.holonomic:
                        vy += rng.uniform(-0.05, 0.05) * emb.max_vy
        else:
            remaining = T - t
            total_decel = T - cruise_end
            progress = remaining / max(total_decel, 1)
            vx = cruise_vx * progress * 0.8
            vy = cruise_vy * progress * 0.5
            yaw = current_yaw_target * progress * 0.3

        if not emb.holonomic:
            vy = 0.0

        vx += rng.normal(0, emb.vx_noise)
        vy += rng.normal(0, emb.vy_noise)
        yaw += rng.normal(0, emb.yaw_noise)
        velocities[t] = [vx, vy, yaw]

    # Smooth
    alpha = emb.smoothing
    for t in range(1, T):
        velocities[t] = alpha * velocities[t - 1] + (1 - alpha) * velocities[t]

    # Terminal deceleration
    decel_frames = min(12, T)
    for i in range(decel_frames):
        t = T - decel_frames + i
        decay = (decel_frames - i) / decel_frames
        velocities[t] *= decay * 0.25

    # Clip to physical limits
    velocities[:, 0] = np.clip(velocities[:, 0], -emb.max_vx, emb.max_vx)
    velocities[:, 1] = np.clip(velocities[:, 1], -emb.max_vy, emb.max_vy)
    velocities[:, 2] = np.clip(velocities[:, 2], -emb.max_yaw, emb.max_yaw)

    # Integrate trajectory in WORLD frame (starting from start_pos)
    trajectory = np.zeros((T, 3), dtype=np.float32)  # relative [x, y, theta]
    dt = 1.0 / fps
    x, y, theta = 0.0, 0.0, 0.0
    start_heading_rad = math.radians(start_pos.get("heading_deg", 0.0))
    world_x = start_pos.get("x", 0.0)
    world_y = start_pos.get("y", 0.0)
    world_theta = start_heading_rad

    collision_truncate = T  # frame at which to truncate
    grace_frames = 16  # skip collision check for first 1s (start pos is known-valid)
    for t in range(T):
        vx_t, vy_t, yr_t = velocities[t]
        theta += yr_t * dt
        x += (vx_t * math.cos(theta) - vy_t * math.sin(theta)) * dt
        y += (vx_t * math.sin(theta) + vy_t * math.cos(theta)) * dt
        trajectory[t] = [x, y, theta]

        # Check world-frame collision (after grace period)
        if t >= grace_frames:
            wx = world_x + x * math.cos(start_heading_rad) - y * math.sin(start_heading_rad)
            wy = world_y + x * math.sin(start_heading_rad) + y * math.cos(start_heading_rad)
            if _check_collision(wx, wy, env_name):
                collision_truncate = max(t - 2, grace_frames)
                break

    # Truncate at collision
    if collision_truncate < T:
        decel_len = min(8, collision_truncate)
        for i in range(decel_len):
            tt = collision_truncate - decel_len + i
            if tt >= 0:
                decay = (decel_len - i) / decel_len
                velocities[tt] *= decay * 0.3
        T = collision_truncate
        velocities = velocities[:T]
        trajectory = trajectory[:T]

    # CONSTRAINT: Enforce max distance
    distances = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)
    over_idx = np.where(distances > max_distance_m)[0]
    if len(over_idx) > 0:
        cutoff = over_idx[0]
        decel_len = min(8, cutoff)
        for i in range(decel_len):
            t = cutoff - decel_len + i
            if t >= 0:
                decay = (decel_len - i) / decel_len
                velocities[t] *= decay * 0.3
        T = cutoff
        velocities = velocities[:T]
        trajectory = trajectory[:T]

    # CONSTRAINT: Enforce max heading change
    total_heading = abs(trajectory[-1, 2]) if len(trajectory) > 0 else 0
    if total_heading > max_heading_change_rad:
        scale = max_heading_change_rad / total_heading
        velocities[:, 2] *= scale
        x, y, theta = 0.0, 0.0, 0.0
        for t in range(len(velocities)):
            vx_t, vy_t, yr_t = velocities[t]
            theta += yr_t * dt
            x += (vx_t * math.cos(theta) - vy_t * math.sin(theta)) * dt
            y += (vx_t * math.sin(theta) + vy_t * math.cos(theta)) * dt
            trajectory[t] = [x, y, theta]

    # ── Stop padding: append 16 zero-velocity frames so model learns to stop ──
    STOP_PAD_FRAMES = 16
    zero_pad = np.zeros((STOP_PAD_FRAMES, 3), dtype=np.float32)
    velocities = np.concatenate([velocities, zero_pad], axis=0)
    # Extend trajectory with the final pose held constant
    if len(trajectory) > 0:
        final_pose = trajectory[-1:].copy()
        traj_pad = np.repeat(final_pose, STOP_PAD_FRAMES, axis=0)
        trajectory = np.concatenate([trajectory, traj_pad], axis=0)

    # Write CSV for Isaac Sim replay
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["vx_m_s", "vy_m_s", "yaw_rate_rad_s"])
        for t in range(len(velocities)):
            writer.writerow([
                f"{velocities[t, 0]:.6f}",
                f"{velocities[t, 1]:.6f}",
                f"{velocities[t, 2]:.6f}",
            ])

    return velocities, trajectory


def generate_action_prompt(velocities: np.ndarray) -> str:
    """Build synthetic prompt from action sequence (matches V2+ training)."""
    terminal_speed = float(np.linalg.norm(velocities[-1, :2]))
    terminal_yaw = float(abs(velocities[-1, 2]))
    mean_vx = float(np.mean(velocities[:, 0]))
    mean_vy = float(np.mean(velocities[:, 1]))
    mean_yaw = float(np.mean(velocities[:, 2]))

    if terminal_speed < 0.05 and terminal_yaw < 0.08:
        return "go to the goal and stop smoothly"
    if abs(mean_yaw) > 0.1:
        return "turn left toward the goal" if mean_yaw > 0 else "turn right toward the goal"
    if abs(mean_vy) > 0.05:
        return "move laterally while approaching the goal"
    if mean_vx < -0.02:
        return "back up slightly and align with the goal"
    return "move forward to the goal with stable speed"


# ============================================================================
# ISAAC SIM LAUNCHER
# ============================================================================

def find_latest_recording() -> str:
    """Find the most recently created recording directory."""
    if not RECORDINGS_DIR.exists():
        return ""
    rec_dirs = sorted(
        [d for d in RECORDINGS_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return str(rec_dirs[-1]) if rec_dirs else ""


def run_isaac_sim_episode(
    sim_type: str,
    env: str,
    csv_path: str,
    start_pos: dict,
    output_dir: str = None,
    timeout_sec: int = 300,
) -> Tuple[bool, str]:
    """
    Launch Isaac Sim in headless mode with CSV replay and recording.

    sim_type:
      - "drone": drone_sim.py with --start-x/y/z/heading, --output-dir
      - "limo":  limo_sim.py (no --output-dir, no --start-*)

    Returns: (success, recording_dir_path)
    """
    pre_launch_latest = find_latest_recording()

    if sim_type == "drone":
        cmd = [
            ISAAC_PYTHON, DRONE_SIM,
            "--env", env,
            "--csv-replay", csv_path,
            "--headless",
            "--record",
            "--cam-width", "1280",
            "--cam-height", "720",
            "--start-x", str(start_pos["x"]),
            "--start-y", str(start_pos["y"]),
            "--start-heading", str(start_pos["heading_deg"]),
        ]
        if "z" in start_pos:
            cmd += ["--start-z", str(start_pos["z"])]
        if output_dir:
            cmd += ["--output-dir", output_dir]

    elif sim_type == "limo":
        cmd = [
            ISAAC_PYTHON, LIMO_SIM,
            "--env", env,
            "--csv-replay", csv_path,
            "--headless",
            "--cam-width", "1280",
            "--cam-height", "720",
            "--start-x", str(start_pos["x"]),
            "--start-y", str(start_pos["y"]),
            "--start-heading", str(start_pos["heading_deg"]),
        ]
        if output_dir:
            cmd += ["--output-dir", output_dir]

    else:
        raise ValueError(f"Unknown sim type: {sim_type}")

    print(f"    Launching: {sim_type} {env}...", flush=True)

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_sec,
            capture_output=True,
            text=True,
            cwd=str(SIM_ROOT),
        )

        if result.returncode not in (0, 1):
            print(f"    WARNING: Isaac Sim returned code {result.returncode}", flush=True)
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-3:]:
                    print(f"      {line}", flush=True)
            return False, ""

        # Find the recording directory
        if output_dir and Path(output_dir).exists():
            rec_dir = output_dir
        else:
            post_latest = find_latest_recording()
            if post_latest and post_latest != pre_launch_latest:
                rec_dir = post_latest
            else:
                print("    WARNING: No new recording directory found", flush=True)
                return False, ""

        # Verify recording has frames
        frames_dir = Path(rec_dir) / "frames"
        if frames_dir.exists():
            n_frames = len(list(frames_dir.glob("frame_*")))
            if n_frames < 5:
                print(f"    WARNING: Only {n_frames} frames captured", flush=True)
                return False, rec_dir

        return True, rec_dir

    except subprocess.TimeoutExpired:
        print(f"    WARNING: Timed out after {timeout_sec}s", flush=True)
        return False, ""
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        return False, ""


# ============================================================================
# EXPORT TO FLOWDIT FORMAT (Phase 1)
# ============================================================================

def export_recording_to_flowdit(
    recording_dir: str,
    episode_id: str,
    output_base: Path,
    velocities: np.ndarray,
    trajectory: np.ndarray,
    emb_name: str,
    env_name: str,
    prompt: str,
    action_prompt: str,
    split: str,
) -> dict:
    """
    Convert Isaac Sim recording to FlowDiT V2+ dataset format.
    Saves observation video, start/goal images, actions, trajectories, velocities.
    Reference video is generated in Phase 2 (LTX).

    Returns: episode metadata dict
    """
    rec_path = Path(recording_dir)
    frames_dir = rec_path / "frames"
    T = len(velocities)

    # Create output directories
    for subdir in ["videos", "reference_videos", "actions", "trajectories",
                   "velocities", "start_images", "goal_images"]:
        (output_base / subdir).mkdir(parents=True, exist_ok=True)

    # Read frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("frame_*.ppm"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    n_frames = min(len(frame_files), T)

    if not HAS_CV2:
        raise ImportError("cv2 required for video export")

    # Create observation video (224x224) — downsample from 720p frames
    video_path = output_base / "videos" / f"{episode_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 16, (224, 224))

    first_frame_720p = None
    last_frame_720p = None

    for i in range(n_frames):
        frame = cv2.imread(str(frame_files[i]))
        if frame is None:
            continue
        # Keep 720p for start/goal, downsample for observation video
        if i == 0:
            first_frame_720p = frame.copy()
        last_frame_720p = frame.copy()
        # Downsample to 224x224 for observation video
        frame_224 = cv2.resize(frame, (224, 224))
        writer.write(frame_224)
    writer.release()

    # Save start and goal images at 720p (1280x720)
    if first_frame_720p is not None:
        cv2.imwrite(str(output_base / "start_images" / f"{episode_id}.png"), first_frame_720p)
    if last_frame_720p is not None:
        cv2.imwrite(str(output_base / "goal_images" / f"{episode_id}.png"), last_frame_720p)

    # Trim actions/trajectory/velocities to match frame count
    velocities_trimmed = velocities[:n_frames]
    trajectory_trimmed = trajectory[:n_frames]

    # Normalize actions to [-1, 1]
    emb = EMBODIMENTS[emb_name]
    actions_norm = np.zeros_like(velocities_trimmed)
    actions_norm[:, 0] = velocities_trimmed[:, 0] / max(emb.max_vx, 1e-6)
    if emb.max_vy > 0:
        actions_norm[:, 1] = velocities_trimmed[:, 1] / max(emb.max_vy, 1e-6)
    actions_norm[:, 2] = velocities_trimmed[:, 2] / max(emb.max_yaw, 1e-6)
    actions_norm = np.clip(actions_norm, -1.0, 1.0)

    # Save arrays
    np.save(str(output_base / "actions" / f"{episode_id}.npy"), actions_norm.astype(np.float32))
    np.save(str(output_base / "trajectories" / f"{episode_id}.npy"), trajectory_trimmed.astype(np.float32))
    np.save(str(output_base / "velocities" / f"{episode_id}.npy"), velocities_trimmed.astype(np.float32))

    # Velocity stats
    speeds = np.linalg.norm(velocities_trimmed[:, :2], axis=1)
    meta = {
        "embodiment": emb_name,
        "environment": env_name,
        "split": split,
        "fps": 16,
        "source": emb.sim_type,
        "task": "navigate_to_goal",
        "frames": int(n_frames),
        "prompt": prompt,
        "action_prompt": action_prompt,
        "velocity_stats": {
            "max_speed_ms": float(np.max(speeds)),
            "mean_speed_ms": float(np.mean(speeds)),
            "total_distance_m": float(np.sum(speeds) / 16.0),
            "final_position_m": [float(trajectory_trimmed[-1, 0]), float(trajectory_trimmed[-1, 1])],
            "final_heading_rad": float(trajectory_trimmed[-1, 2]),
        },
        "max_velocity_ms": emb.max_vx,
        "holonomic": emb.holonomic,
        "renderer": "isaac_sim",
        "has_reference_video": False,  # Updated in Phase 2
    }

    return meta


# ============================================================================
# LTX REFERENCE VIDEO GENERATION (Phase 2)
# ============================================================================

def generate_ltx_reference_videos(dataset_dir: Path, metadata: dict, batch_size: int = 1):
    """
    Generate reference videos using LTX conditioned on start + goal images.

    For each episode:
      1. Load start_image and goal_image
      2. Use LTXConditionPipeline with start at frame_index=0, goal at frame_index=15
      3. Save 16-frame reference video
    """
    import torch
    from PIL import Image

    # Try importing from the videonav env's diffusers
    try:
        from diffusers import LTXImageToVideoPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        print("[ERROR] diffusers not available. Install with: pip install diffusers transformers")
        return

    # Try the condition pipeline first (supports start+goal conditioning)
    try:
        from diffusers.pipelines.ltx.pipeline_ltx_condition import (
            LTXConditionPipeline, LTXVideoCondition
        )
        use_condition_pipeline = True
        print("[LTX] Using LTXConditionPipeline (start+goal conditioning)")
    except ImportError:
        use_condition_pipeline = False
        print("[LTX] Falling back to LTXImageToVideoPipeline (start image only)")

    # Load model
    print(f"[LTX] Loading model: {LTX_MODEL_ID}...")
    os.makedirs(LTX_CACHE_DIR, exist_ok=True)

    if use_condition_pipeline:
        pipe = LTXConditionPipeline.from_pretrained(
            LTX_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=LTX_CACHE_DIR,
        )
    else:
        pipe = LTXImageToVideoPipeline.from_pretrained(
            LTX_MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=LTX_CACHE_DIR,
        )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("[LTX] Model loaded and ready.")

    ref_dir = dataset_dir / "reference_videos"
    ref_dir.mkdir(parents=True, exist_ok=True)
    start_dir = dataset_dir / "start_images"
    goal_dir = dataset_dir / "goal_images"

    episodes = list(metadata.keys())
    total = len(episodes)
    success = 0

    for idx, episode_id in enumerate(episodes):
        meta = metadata[episode_id]

        # Skip if reference video already exists
        ref_path = ref_dir / f"{episode_id}.mp4"
        if ref_path.exists():
            meta["has_reference_video"] = True
            success += 1
            continue

        start_path = start_dir / f"{episode_id}.png"
        goal_path = goal_dir / f"{episode_id}.png"

        if not start_path.exists() or not goal_path.exists():
            print(f"  [{idx+1}/{total}] {episode_id}: SKIP (missing start/goal images)")
            continue

        try:
            start_img = Image.open(start_path).convert("RGB")
            goal_img = Image.open(goal_path).convert("RGB")

            prompt = meta.get("prompt", "navigate to the goal location")

            # Match reference video frames to observation video duration
            ep_frames = meta.get("frames", 16)
            # LTX generates at 16fps; pick num_frames matching episode
            # LTX requires num_frames in {9, 17, 25, 33, 41, ...} = 1 + 8*k
            # Pick closest valid frame count, capped at episode frames
            target_frames = min(ep_frames, 81)  # max 81 for LTX
            valid_counts = [1 + 8 * k for k in range(20) if 1 + 8 * k <= 161]
            num_frames = min(valid_counts, key=lambda x: abs(x - target_frames))
            if num_frames < 9:
                num_frames = 9
            goal_frame_idx = num_frames - 1

            # LTX width/height must be multiples of 32; use 512x288 for speed
            # (720p input images are high-res but LTX generation at full 1280x720
            #  is very slow; 512x288 preserves 16:9 and is practical)
            ltx_width = 512
            ltx_height = 288

            if use_condition_pipeline:
                condition_start = LTXVideoCondition(image=start_img, frame_index=0)
                condition_goal = LTXVideoCondition(image=goal_img, frame_index=goal_frame_idx)

                generator = torch.Generator("cuda").manual_seed(idx)
                output = pipe(
                    conditions=[condition_start, condition_goal],
                    prompt=prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    width=ltx_width,
                    height=ltx_height,
                    num_frames=num_frames,
                    num_inference_steps=30,
                    generator=generator,
                )
            else:
                generator = torch.Generator("cuda").manual_seed(idx)
                output = pipe(
                    image=start_img,
                    prompt=prompt,
                    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    width=ltx_width,
                    height=ltx_height,
                    num_frames=num_frames,
                    num_inference_steps=30,
                    generator=generator,
                )

            # Save reference video (resize to 224x224 for FlowDiT input)
            video_frames = output.frames[0]  # list of PIL images
            if HAS_CV2:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                ref_writer = cv2.VideoWriter(str(ref_path), fourcc, 16, (224, 224))
                for frame_pil in video_frames:
                    frame_np = np.array(frame_pil)
                    frame_np = cv2.resize(frame_np, (224, 224))
                    ref_writer.write(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
                ref_writer.release()
            else:
                export_to_video(video_frames, str(ref_path), fps=16)

            meta["has_reference_video"] = True
            meta["reference_frames"] = len(video_frames)
            success += 1
            print(f"  [{idx+1}/{total}] {episode_id}: OK ({len(video_frames)} frames, {ltx_width}x{ltx_height})")

        except Exception as e:
            print(f"  [{idx+1}/{total}] {episode_id}: FAILED ({e})")

        # Clear GPU cache periodically
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    print(f"\n[LTX] Reference video generation complete: {success}/{total} successful")
    return metadata


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Isaac Sim + LTX dataset generator for FlowDiT V2+")
    parser.add_argument("--output_dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--embodiments", nargs="+", default=["drone", "human", "mobile_wheel"],
                        choices=list(EMBODIMENTS.keys()))
    parser.add_argument("--total_episodes", type=int, default=100,
                        help="Total episodes across all embodiments (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "sim", "ltx"],
                        help="Run phase: sim (recordings only), ltx (reference videos only), all (both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only generate CSVs, don't launch Isaac Sim or LTX")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-episode Isaac Sim timeout in seconds")
    parser.add_argument("--max-distance", type=float, default=4.0,
                        help="Max travel distance in meters (goal must be visible from start)")
    args = parser.parse_args()

    # Compute episodes per embodiment from total
    n_embs = len(args.embodiments)
    eps_per_emb = args.total_episodes // n_embs
    eps_remainder = args.total_episodes % n_embs

    rng = np.random.RandomState(args.seed)
    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "temp_csvs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = output_dir / "isaac_sim_combined"

    print(f"\n{'='*60}")
    print(f"FlowDiT V2+ Dataset Generator (Isaac Sim + LTX)")
    print(f"{'='*60}")
    print(f"Embodiments: {args.embodiments}")
    print(f"Total episodes: {args.total_episodes} ({eps_per_emb} per emb)")
    print(f"Durations: randomly from [5, 10, 15] seconds")
    print(f"Resolution: 720p (1280x720) start/goal, 224x224 observation")
    print(f"Phase: {args.phase}")
    print(f"Dry run: {args.dry_run}")
    print(f"Output: {combined_dir}")
    print(f"{'='*60}\n")

    # ── Phase 1: Isaac Sim recordings ──────────────────────────────────────
    if args.phase in ("all", "sim"):
        print("=" * 60)
        print("PHASE 1: Isaac Sim Recordings")
        print("=" * 60)

        all_meta = {}
        global_counter = 0
        success_count = 0
        fail_count = 0

        for emb_idx, emb_name in enumerate(args.embodiments):
            emb = EMBODIMENTS[emb_name]
            envs = emb.environments
            n_episodes_this_emb = eps_per_emb + (1 if emb_idx < eps_remainder else 0)
            eps_per_env = n_episodes_this_emb // len(envs)
            remainder = n_episodes_this_emb % len(envs)

            print(f"\n--- {emb_name.upper()} (sim={emb.sim_type}) ---")
            print(f"  Environments: {envs}")
            print(f"  Episodes per env: {eps_per_env} (+{remainder} in first)")

            for env_idx, env_name in enumerate(envs):
                n_eps = eps_per_env + (1 if env_idx < remainder else 0)
                starts = emb.start_positions[env_name]

                print(f"\n  [{env_name}] Generating {n_eps} episodes...")

                for ep_idx in range(n_eps):
                    episode_id = f"episode_{global_counter:06d}"

                    # Random duration from [5, 10, 15] seconds
                    duration = float(rng.choice([5, 10, 15]))

                    # Random start position
                    start_pos = starts[ep_idx % len(starts)].copy()
                    start_pos["x"] += rng.uniform(-0.3, 0.3)
                    start_pos["y"] += rng.uniform(-0.3, 0.3)
                    start_pos["heading_deg"] += rng.uniform(-10, 10)

                    # Generate collision-free trajectory CSV
                    csv_path = csv_dir / f"{episode_id}.csv"
                    velocities, trajectory = generate_trajectory_csv(
                        emb, csv_path, duration, rng,
                        start_pos=start_pos,
                        env_name=env_name,
                        max_distance_m=args.max_distance,
                    )

                    # Pick prompts
                    prompt = rng.choice(PROMPTS[env_name])
                    action_prompt = generate_action_prompt(velocities)

                    # Split
                    split = "train" if rng.random() < 0.8 else "val"

                    if args.dry_run:
                        T = len(velocities)
                        speeds = np.linalg.norm(velocities[:, :2], axis=1)
                        meta = {
                            "embodiment": emb_name,
                            "environment": env_name,
                            "split": split,
                            "fps": 16,
                            "source": emb.sim_type,
                            "task": "navigate_to_goal",
                            "frames": T,
                            "prompt": prompt,
                            "action_prompt": action_prompt,
                            "velocity_stats": {
                                "max_speed_ms": float(np.max(speeds)),
                                "mean_speed_ms": float(np.mean(speeds)),
                                "total_distance_m": float(np.sum(speeds) / 16.0),
                            },
                            "csv_path": str(csv_path),
                            "start_position": start_pos,
                            "duration_sec": duration,
                            "renderer": "isaac_sim (dry-run)",
                            "has_reference_video": False,
                        }
                        all_meta[episode_id] = meta
                        print(f"    [{ep_idx+1}/{n_eps}] {episode_id}: "
                              f"{T} frames, {duration:.1f}s, "
                              f"max_speed={np.max(speeds):.3f}m/s [DRY RUN]")

                    else:
                        # Run Isaac Sim
                        rec_output = str(RECORDINGS_DIR / f"{episode_id}_{emb_name}_{env_name}")
                        success, rec_dir = run_isaac_sim_episode(
                            sim_type=emb.sim_type,
                            env=env_name,
                            csv_path=str(csv_path),
                            start_pos=start_pos,
                            output_dir=rec_output,
                            timeout_sec=args.timeout,
                        )

                        if success and rec_dir:
                            try:
                                meta = export_recording_to_flowdit(
                                    recording_dir=rec_dir,
                                    episode_id=episode_id,
                                    output_base=combined_dir,
                                    velocities=velocities,
                                    trajectory=trajectory,
                                    emb_name=emb_name,
                                    env_name=env_name,
                                    prompt=prompt,
                                    action_prompt=action_prompt,
                                    split=split,
                                )
                                all_meta[episode_id] = meta
                                success_count += 1
                                print(f"    [{ep_idx+1}/{n_eps}] {episode_id}: OK "
                                      f"({meta['frames']} frames)")
                            except Exception as e:
                                print(f"    [{ep_idx+1}/{n_eps}] {episode_id}: EXPORT FAILED: {e}")
                                fail_count += 1
                        else:
                            fail_count += 1
                            print(f"    [{ep_idx+1}/{n_eps}] {episode_id}: SIM FAILED")

                    global_counter += 1

        # Write metadata
        if args.dry_run:
            meta_path = output_dir / "dry_run_plan.json"
        else:
            meta_path = combined_dir / "metadata.json"
            combined_dir.mkdir(parents=True, exist_ok=True)

        with open(meta_path, 'w') as f:
            json.dump(all_meta, f, indent=2)

        # Phase 1 summary
        print(f"\n{'='*60}")
        print(f"Phase 1 {'(DRY RUN) ' if args.dry_run else ''}complete!")
        print(f"{'='*60}")
        print(f"Total: {global_counter}")
        if not args.dry_run:
            print(f"Successful: {success_count}")
            print(f"Failed: {fail_count}")
        for emb_name in args.embodiments:
            eps = [k for k, v in all_meta.items() if v["embodiment"] == emb_name]
            print(f"  {emb_name}: {len(eps)} episodes")

        if args.dry_run:
            print(f"\nCSVs saved to: {csv_dir}")
            print(f"Plan saved to: {meta_path}")
            return

    # ── Phase 2: LTX reference videos ─────────────────────────────────────
    if args.phase in ("all", "ltx"):
        print(f"\n{'='*60}")
        print("PHASE 2: LTX Reference Video Generation")
        print(f"{'='*60}")

        meta_path = combined_dir / "metadata.json"
        if not meta_path.exists():
            print(f"[ERROR] No metadata found at {meta_path}")
            print("  Run Phase 1 first: python generate_isaac_sim_dataset.py --phase sim")
            return

        with open(meta_path) as f:
            all_meta = json.load(f)

        print(f"  Episodes to process: {len(all_meta)}")
        all_meta = generate_ltx_reference_videos(combined_dir, all_meta)

        # Update metadata
        with open(meta_path, 'w') as f:
            json.dump(all_meta, f, indent=2)

        # Final summary
        has_ref = sum(1 for v in all_meta.values() if v.get("has_reference_video"))
        print(f"\n{'='*60}")
        print("Dataset generation complete!")
        print(f"{'='*60}")
        print(f"Total episodes: {len(all_meta)}")
        print(f"With reference videos: {has_ref}")
        for emb_name in args.embodiments:
            eps = [k for k, v in all_meta.items() if v["embodiment"] == emb_name]
            print(f"  {emb_name}: {len(eps)} episodes")
        print(f"\nDataset: {combined_dir}")
        print(f"\nContents:")
        print(f"  videos/          — observation videos (224x224 @16fps, downsampled)")
        print(f"  reference_videos/ — LTX-generated goal videos (duration-matched)")
        print(f"  start_images/    — first frame at 720p (1280x720)")
        print(f"  goal_images/     — last frame at 720p (1280x720)")
        print(f"  actions/         — normalized [-1,1] velocity commands")
        print(f"  trajectories/    — world-frame [x,y,theta] paths")
        print(f"  velocities/      — real m/s velocity measurements")
        print(f"  metadata.json    — per-episode metadata with prompts")


if __name__ == "__main__":
    main()
