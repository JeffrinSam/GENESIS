# Architecture Comparison — V1 vs V2 vs V3

---

## At a glance

| | V1 | V2 | V3 |
|---|---|---|---|
| **Folder** | `flow_constrained/` | `flow_constrained_v2/` | `flowdit_v3_humanoid_inference/` |
| **Paradigm** | Open-loop, reactive | Closed-loop, goal-conditioned | Closed-loop, goal-conditioned |
| **Input** | Video clip | Reference video + current frame | Reference video + current frame |
| **Output** | `[vx, vy, yaw]` × 1 | `[vx, vy, yaw]` × 8 steps | `[vx, vy, vz, yaw]` × 8 + `[x,y,z]` × 16 waypoints |
| **Action head** | MLP (2.8M trainable) | Diffusion Transformer (DiT) | Diffusion Transformer (DiT) |
| **Trainable params** | 2.8M (of 393M total) | ~3M (of 43.5M total) | ~55M |
| **Language** | No | Optional CLIP | Optional CLIP |
| **Embodiment** | Wheeled / Aerial / Humanoid | Any | Humanoid (Unitree G1) |
| **Inference** | ~100ms | ~50ms | ~80ms |
| **Training code** | Yes | Yes | No (inference-only bundle) |
| **Status** | Legacy / baseline | **Production** | **Production (humanoid)** |

---

## V1 — Flow-Constrained Reactive Policy

### Concept
Three frozen pretrained models extract complementary features from video.
A small 2.8M trainable MLP fuses them into velocity commands.
No goal — pure imitation of training trajectories.

### Data flow
```
Input video (T frames, any resolution)
    │
    ├─► RAFT optical flow           → ego-motion features  (256-dim)   [frozen]
    ├─► SVD / CLIP VDM features     → implicit dynamics    (512-dim)   [frozen]
    └─► DINOv2 ViT-B/14             → semantic features    (768-dim)   [frozen]
                │
                Concat → 1536-dim
                │
            FusionNetwork (MLP, residual blocks, 2.8M params)   [trainable]
                │
            [vx, vy, yaw]  (per-embodiment velocity-scaled)
```

### Key design choices
- Relies entirely on pretrained foundation models → trains in hours on 100-1000 clips
- `ResolutionAdapter` auto-detects Cosmos 2.5 / Wan 2.2 video sources
- `PhysicalConstraints` applies per-embodiment velocity/acceleration limits
- `TemporalSmoother` stabilises predictions across frames

### When to use V1
- You want a simple, fast-to-train baseline
- Offline video analysis (not real robot deployment)
- You don't have a reference/goal video

---

## V2 — FlowDiT Goal-Conditioned Navigation

### Concept
A reference video defines the goal.
The model continuously compares current camera observation against the goal
and outputs an 8-step action horizon via a Diffusion Transformer.

### Data flow
```
Reference video (16 frames, 224×224)
    └─► GoalEncoder
          ├─ DINOv2 ViT-B/14 (frozen)         → frame features (768)
          ├─ Flow CNN on RAFT (trainable)      → flow features  (256)
          └─ Temporal Attention                → goal embedding (512)

Current frame (224×224)
    └─► DINOv2 (shared, frozen)               → obs embedding  (768)

Language (optional)
    └─► CLIP ViT-L/14                         → lang embedding (512)

Concat [goal(512) + obs(768)] = 1280-dim condition
    │
Diffusion Transformer (DiT)
    ├─ 8 blocks, 512-dim hidden, 8 heads
    ├─ adaLN-Zero timestep conditioning
    └─ DDIM sampling (10 steps, ~30ms)
    │
[vx, vy, yaw] × 8 steps
```

### Key design choices
- Goal-conditioned: reference video = visual memory, no maps needed
- DiT diffusion → multimodal action distributions, handles uncertainty
- Shared DINOv2 encoder for goal and obs → efficient
- Trained on RECON (8,948 real navigation videos, 75 epochs)
- Best checkpoint: epoch 67, val MSE = 0.046

### When to use V2
- Real robot closed-loop navigation
- Goal specified as a reference video (from video gen model or recorded)
- Any embodiment (wheeled, legged, aerial)

---

## V3 — FlowDiT Humanoid (Dual Output)

### Concept
Extends V2 for humanoid robots with:
- 4-DOF velocity output (adds vz for 3D)
- 16-waypoint trajectory prediction (geometric path planning)
- CrossVideoAttention between reference and observation
- VideoMAE temporal encoder (optional)
- Auto embodiment detection

### Data flow
```
Reference video (48-240 frames, any resolution)
    └─► VideoEncoder
          ├─ DINOv2 ViT-B/14 (spatial, frozen)     → 768/frame
          └─ VideoMAE v2 Base (temporal, optional)  → 768 sequence
                └─ Fusion layer                     → goal feat (512)

Current frame
    └─► DINOv2 (shared)                             → obs feat (768)

Depth frames (optional)
    └─► DepthEncoder (3D CNN)                       → 256-dim

Language (optional)
    └─► CLIP text encoder (frozen)                  → 512-dim

CrossVideoAttention
    Query: obs feat, Key/Value: goal feat            → attended (512)

EmbodimentDetector (auto: aerial/bipedal/wheeled/legged)

DirectActionPredictor (DiT backbone)
    ├─► Velocity:    [vx, vy, vz, yaw] × 8 steps
    └─► Trajectory:  [x, y, z]         × 16 waypoints
```

### Key design choices
- Variable video length (48–240 frames = 3–15 sec at 16fps)
- Dual output: velocity commands (reactive) + trajectory (planning ahead)
- Finetuned on Unitree G1 GR00T + Arena manipulation episodes
- `.venv` pre-built with torch 2.9.1+cu128 — just activate and run

### When to use V3
- Unitree G1 humanoid robot
- Need trajectory waypoints (for a separate path planner)
- Inference-only (no retraining needed)

---

## Progression

```
V1  →  frozen features + MLP          →  simple, fast to train, open-loop
V2  →  + goal video + DiT             →  closed-loop, goal-directed
V3  →  + CrossAttn + dual output      →  humanoid-optimised, trajectory-aware
```

Each version is a proper subset improvement of the previous — V1 code concepts
are preserved in V2 (same feature extractors), V2's DiT backbone is reused in V3.
