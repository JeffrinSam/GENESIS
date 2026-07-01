# FlowDiT — Video-to-Navigation

**Author**: Jeffrin Sam | Skoltech | 2025-2026 | MIT License

Video-conditioned robot navigation: give the robot a reference video of where to go, it navigates there in closed-loop.

---

## Three Models

| Model | Folder | Robot | Status | Use |
|-------|--------|-------|--------|-----|
| **V1** | `flow_constrained/` | Wheeled / Aerial / Humanoid | Legacy | Research / baseline only |
| **V2** | `flow_constrained_v2/` | Any | **Production** | Real robot deployment |
| **V3** | `flowdit_v3_humanoid_inference/` | Unitree G1 humanoid | Inference-only | Humanoid deployment |

**New person → go straight to V2 if wheeled/general, V3 if humanoid (G1).**

---

## Quick Start

### V2 — General navigation (recommended)

```bash
conda activate flowdit_v2_py310
cd flow_constrained_v2

# Verify the trained model works
python test_inference.py --checkpoint checkpoints/best.pth
# Expected: MSE ~0.046

# Single inference
python inference.py \
    --checkpoint checkpoints/best.pth \
    --goal_video path/to/reference.mp4 \
    --current_obs path/to/frame.jpg \
    --output actions.npy

# Closed-loop on real robot @ 2 Hz
python robot_navigation.py \
    --checkpoint checkpoints/best.pth \
    --goal_video reference.mp4 \
    --camera 0 --control_hz 2.0
```

### V3 — Unitree G1 humanoid

```bash
cd flowdit_v3_humanoid_inference
source .venv/bin/activate   # torch 2.9.1+cu128, already set up

python run_inference.py \
    --checkpoint checkpoints/flowdit_v3_humanoid_best.pt \
    --video /path/to/episode/frames/ \
    --instruction "humanoid robot navigate to goal"
# Output: [vx, vy, vz, yaw] x 8 steps + [x,y,z] x 16 waypoints
```

---

## Repository Structure

```
VideotoNav_JeffrinSam_Skoltech2025/
│
├── README.md                         ← You are here
├── COMPARISON.md                     ← V1 vs V2 vs V3 architecture comparison
├── PROJECT_STATUS.md                 ← What is done, what is next
├── TESTING_GUIDE.md                  ← How to test each model
├── LICENSE
│
├── flow_constrained/                 ← V1 (legacy, research baseline)
│   ├── models/                       (fusion_network, optical_flow, vdm_features,
│   │                                  vision_encoder, constraints, resolution_adapter)
│   ├── training/ (train.py, losses.py)
│   ├── inference/ (policy.py)
│   ├── configs/ (wheeled.yaml, aerial.yaml, humanoid_unitree_g1.yaml)
│   └── README.md
│
├── flow_constrained_v2/              ← V2 PRODUCTION (use this for real robots)
│   ├── models/
│   │   └── flowdit_production.py     ← main model (GoalEncoder + DiT)
│   ├── training/
│   │   └── train_production.py
│   ├── checkpoints/
│   │   └── best.pth                  ← trained weights (epoch 67, MSE 0.046)
│   ├── evaluation/
│   │   └── iros_metrics.py           (SR, SPL, ATE, collision rate)
│   ├── inference.py                  ← primary inference API
│   ├── robot_navigation.py           ← closed-loop control @ 2 Hz
│   ├── test_inference.py
│   ├── ARCHITECTURE.md
│   └── INFERENCE_GUIDE.md
│
├── flowdit_v3_humanoid_inference/    ← V3 HUMANOID (Unitree G1)
│   ├── flow_constrained_v3/
│   │   └── models/flowdit_v3.py      ← full V3 model
│   ├── checkpoints/
│   │   └── flowdit_v3_humanoid_best.pt
│   ├── unitree_data/                 ← Unitree G1 test episodes
│   ├── run_inference.py
│   ├── .venv/                        ← pre-built venv (torch 2.9.1+cu128)
│   └── README.md
│
├── dataset/                          ← converted training data
│   ├── recon/                        (8,948 navigation videos)
│   └── tartanair/
│
├── dataset_generation/               ← tools to download & convert datasets
│   ├── convert_recon.py
│   ├── convert_tartanair.py
│   ├── split_dataset.py
│   ├── verify_dataset.py
│   └── README.md
│
├── web_app/                          ← Flask inference UI
│   ├── app.py
│   ├── inference_wrappers/           (humanoid_v2, model1, model2)
│   └── start.sh
│
├── flowdit_codebase.zip              ← code-only zip (for AI agents / handoff)
│                                        no checkpoints, no data — pure architecture
└── archive/                          ← old versions, stale docs, intermediate ckpts
```

---

## Architecture Summary

### V1 — Reactive MLP
```
Video → [RAFT flow + SVD VDM + DINOv2] → Concat (1536-dim) → FusionMLP → [vx, vy, yaw]
         (all frozen, ~391M)               (trainable, 2.8M)
```
Open-loop. No goal. Fast to train (~33h, 50 epochs).

### V2 — Goal-conditioned Diffusion Transformer
```
Reference video (16 frames) → GoalEncoder (DINOv2 + Flow CNN + TemporalAttn) → goal feat (512)
Current frame               → DINOv2                                          → obs feat (768)
                              Concat (1792) → DiT (8 blocks, 512-dim) → [vx,vy,yaw] × 8 steps
```
Closed-loop @ 2 Hz. Trained: RECON 8,948 videos, 75 epochs, best MSE 0.046.
Total 43.5M params (~3M trainable). Inference ~50ms.

### V3 — Humanoid DiT with trajectory output
```
Reference video → VideoEncoder (DINOv2 + VideoMAE) → goal feat
Current frame   → DINOv2 (shared)                  → obs feat
                  CrossVideoAttention + CLIP language + EmbodimentDetector
                  → DiT → [vx,vy,vz,yaw]×8 + [x,y,z]×16 waypoints
```
Finetuned on Unitree G1 GR00T + Arena data. Inference-only (no training code included).

---

## Training V2 from scratch

```bash
conda activate flowdit_v2_py310
cd flow_constrained_v2

python training/train_production.py \
    --dataset_root ../dataset/recon \
    --epochs 75 --batch_size 8 --lr 1e-4

# Checkpoint saves to checkpoints/best.pth at best val MSE
# Expected: ~5.3 GB GPU, ~12h on RTX 5090
```

## Training V1 from scratch

```bash
conda activate flowdit_v2_py310
cd flow_constrained

# Edit data path in config first
python training/train.py --config configs/wheeled.yaml
# ~4-8h on RTX 5090 for 1000 clips
```

---

## Dataset

RECON dataset (real-world navigation, 8,948 clips) already converted and in `dataset/recon/`.

To download fresh:
```bash
cd dataset_generation
wget https://lmb.informatik.uni-freiburg.de/data/RECON/recon_dataset_v1.tar.gz
tar -xzf recon_dataset_v1.tar.gz
python convert_recon.py --recon_dir ./recon_dataset --output_dir ../dataset/recon
python verify_dataset.py --dataset_dir ../dataset/recon
```

---

## For AI agents (Codex / GPT / Claude)

Use `flowdit_codebase.zip` — 138 KB, pure code, all three architectures, no data/checkpoints.
The zip contains its own `README.md` with full architecture documentation.

---

## Key metrics (V2)

| Metric | Value |
|--------|-------|
| Val MSE | 0.046 (epoch 67) |
| Inference speed | ~50ms / cycle |
| Control frequency | 2 Hz |
| Training data | 8,948 RECON videos |
| Model size | 43.5M params (~3M trainable) |
| GPU memory (inference) | ~1 GB |
