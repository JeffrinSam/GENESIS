# FlowDiT V3 Humanoid — Inference Bundle

Clean inference-only package for the FlowDiT v3 humanoid model. No training or evaluation code.

## Contents

- **flow_constrained_v3/** — Model code only (`models/flowdit_v3.py`)
- **checkpoints/flowdit_v3_humanoid_best.pt** — Final humanoid finetuned weights
- **unitree_data/** — Unitree G1 data (GR00T + Arena) for testing
- **run_inference.py** — Single script to run inference from video frames

## Setup (new PC)

```bash
cd flowdit_v3_humanoid_inference
pip install torch torchvision transformers Pillow numpy
```

Optional: use a venv or conda env with Python 3.10+.

## Run inference

```bash
# From the bundle root
python run_inference.py --checkpoint checkpoints/flowdit_v3_humanoid_best.pt --video /path/to/episode

# Example using unitree_data (after converting to frames if needed):
# Point --video to a directory that contains a "frames" subdir with .jpg/.png, or directly to that frames dir.
python run_inference.py --checkpoint checkpoints/flowdit_v3_humanoid_best.pt --video unitree_data/unitree_g1.LMPnPAppleToPlateDC/.../episode_xxx
```

- **--video**: Path to an episode directory (with `frames/` inside) or directly to a directory of images.
- **--instruction**: Optional text prompt (default: "humanoid robot navigate to goal").
- **--device**: Optional `cuda` or `cpu` (default: auto).

Output: velocity commands `[vx, vy, vz, yaw]` and trajectory waypoints `[x, y, z]` printed to stdout.

## Model

- **FlowDiT v3** with variable horizons (1 vel/sec, 2 traj/sec), finetuned for humanoid (vx, vy, yaw; vz=0 at base).
- Checkpoint trained on unitree_data (GR00T + Arena G1).

## Data

- **unitree_data**: Original Unitree G1 datasets only. For inference you need frames (e.g. convert with the conversion scripts on the training machine if you need FlowDiT-format episodes; or point `--video` at any directory of frames).
