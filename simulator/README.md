# GENESIS Simulation Validator

Simulation-based validation pipeline for the **FlowDiT V2+** video-to-navigation model using **NVIDIA Isaac Sim 5.1** and **Isaac Lab**.

**Primary platform: Unitree G1 Humanoid** with whole-body control. Drone and mobile robot examples are in `archive/` for reference.

---

## Directory Structure

```
Simulator/
├── humanoid/                   # PRIMARY — G1 humanoid data generation
│   ├── collect_dataset.py      #   Single-pass data collector
│   ├── run_collect.sh          #   Launcher script
│   └── README.md               #   Detailed usage instructions
│
├── dataset/                    # FlowDiT validation dataset tools
│   ├── generate_isaac_sim_dataset.py
│   └── run_mode2_inference_check.py
│
├── closed_loop/                # Realtime control server
│   └── flowdit_server.py
│
├── predict_all_actions.py      # Mode 1 batch inference
├── run_mode1_validation.py     # Mode 1 validation
├── run_mode2_validation.py     # Mode 2 (realtime) validation
├── run_closed_loop.py          # Closed-loop Isaac Sim + FlowDiT
├── run_experiment.py           # Full pipeline orchestrator
├── run_all_models.py           # V1 vs V2 vs V2+ comparison
├── export_episode.py           # Convert recordings → FlowDiT format
│
└── archive/                    # Legacy examples (drone, mobile_robot, pipeline)
```

---

## Quick Start — Humanoid Data Generation

```bash
cd simulator/humanoid

# Collect one episode (robot walks to table in warehouse, stops, records 3 cameras)
bash run_collect.sh --output-dir proof/episode_001 --headless --enable_cameras

# Custom speed / duration
bash run_collect.sh --output-dir proof/episode_002 --headless --enable_cameras \
  --max-speed 0.4 --walk-duration 20

# Encode MP4 from frames
ffmpeg -y -framerate 16 -i proof/episode_001/frames_fpv/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 proof/episode_001/fpv_video.mp4
```

See [humanoid/README.md](humanoid/README.md) for full documentation.

---

## Humanoid (G1) — Primary Platform

| Parameter | Value |
|-----------|-------|
| Physics | Real (Isaac Lab ManagerBasedEnv, PhysX) |
| Controller | G1WBCController (stand.onnx + walk.onnx) |
| Control input | [vx, vy, vyaw] velocity commands |
| DOFs | 43 (29 body + 14 hand) |
| Standing height | 0.78 m |
| Sim rate | 60 Hz physics, 30 Hz control |
| Cameras | Ego (head, down), FPV (head, forward), Third-person |

### What `collect_dataset.py` produces

```
proof/episode_NNN/
  frames_pov/           # Ego camera (720x720)
  frames_fpv/           # Eye-level FPV (720x720)
  frames_third_person/  # Following camera (720x720)
  velocity_commands.csv # Input velocities
  trajectory.csv        # Robot state per frame
  joint_states.npy      # 43-DOF joint positions
  metadata.json         # Episode info
```

---

## Validation Pipeline

### Mode 1 — Batch Inference

```bash
# Run FlowDiT V2+ on recorded episodes
python predict_all_actions.py --robot humanoid --checkpoint path/to/best.pth

# Validate predictions in Isaac Sim
python run_mode1_validation.py --robot humanoid --predictions path/to/actions/
```

### Mode 2 — Realtime Closed-Loop

```bash
# Start FlowDiT inference server
python closed_loop/flowdit_server.py --checkpoint path/to/best.pth

# Run realtime validation
python run_mode2_validation.py --robot humanoid
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **SR** | Success Rate — within radius of goal |
| **SPL** | Success weighted by Path Length |
| **ATE** | Average Trajectory Error (mean L2) |
| **Direction Accuracy** | Heading cosine similarity > 0.75 |
| **Final Error** | L2 distance at last timestep |

---

## Dependencies

- **Isaac Sim 5.1** + **Isaac Lab**
- **IsaacLab-Arena** (`$ISAACLAB_ARENA_DIR` — set in `.env`)
- **Digitaltwin** WBC models (`$DIGITALTWIN_DIR` — set in `.env`)
- **FlowDiT V2+** (`flow_constrained_v2_plus/`)
- NVIDIA GPU with Vulkan (tested: RTX 5090)

---

## Archive

Legacy drone (Crazyflie CF2X) and mobile robot (AgileX Limo) examples are in `archive/`. These use kinematic control (no real physics) and are kept for reference. The humanoid module is the primary platform for GENESIS.
