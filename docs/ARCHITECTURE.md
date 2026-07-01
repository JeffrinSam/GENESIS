# GENESIS Architecture

## Overview

GENESIS uses **video as the universal task interface** across robot embodiments. A single reference video communicates the desired task to both navigation and manipulation controllers, removing the need for hand-engineered reward functions or task-specific representations.

---

## Part 1 — ClaudeOpusBrain (Agentic Video Generation)

```
User: "G1 picks up a bottle" + workspace image
                │
                ▼
        ┌──────────────────┐
        │   Qwen3-VL-2B    │  expand → 200-300 word physics-aware prompt
        │  prompt extender  │  + negative prompt
        └────────┬─────────┘
                 │
         ┌───────┴────────┐
         ▼                ▼
   ┌──────────┐    ┌──────────────┐
   │  WAN 2.2 │    │ Cosmos 2.5   │
   │ TI2V-5B  │    │    14B       │
   │(nav tasks)│   │(manip tasks) │
   └────┬─────┘    └──────┬───────┘
        └────────┬─────────┘
                 │ video (61 frames)
                 ▼
        ┌──────────────────┐
        │  Cosmos-Reason2  │  validate: Adherence / Physics / Quality
        │   validator      │  scores 0-100 each
        └────────┬─────────┘
                 │ scores + analysis
                 ▼
        ┌──────────────────┐
        │  Claude Opus 4.6 │  optimize prompts
        │  (or Sonnet,     │  guided by short memory + long memory
        │   Llama, Qwen)   │
        └────────┬─────────┘
                 │ improved prompts
                 └──────────────→ back to Qwen3-VL (next iteration)
```

**Self-tuning loop**: up to 5 iterations, stops when average score ≥ 80/100.

**Memory system**:
- *Short memory* — tracks strategy effectiveness within a single task (5 iterations)
- *Long memory* — persistent cross-task rules (JSON), injected into every optimizer call

**Supported task types**: `drone`, `ground`, `g1`, `g1_nav`, `ur3`

---

## Part 2a — FlowDiT V2 (Video-to-Navigation)

```
Goal video [B, T, 3, 224, 224]
       │
       ├── DINOv2 ViT-B/14 (frozen)  ── per-frame features [B, T, 768]
       │                                       │
       └── Optical Flow (RAFT/lightweight) ─── flow features [B, T-1, 256]
                                               │
                              TemporalAttention (multi-head)
                                               │
                              goal_embedding [B, 1024]
                                               │
Current observation [B, 3, 224, 224]           │
       │                                       │
       └── DINOv2 ViT-B/14 (frozen)           │
               obs_embedding [B, 768]          │
                                               │
                     Concat [B, 2304] ─────────┘
                              │
                    Diffusion Transformer
                      (8 DiT blocks, adaLN)
                              │
                     Actions [B, 8, 3]
                     (vx, vy, yaw_rate)
```

**Training**: 8,948 RECON episodes, 75 epochs, AdamW + cosine LR, MSE diffusion loss.
Best checkpoint: epoch 67, val MSE = 0.046.

**Inference modes**:
- *Offline*: `predict_full_trajectory()` — process full goal video at once
- *Realtime (2 Hz)*: `predict_realtime()` — cache goal features, update obs each step

---

## Part 2b — DC-GR00T (Video-to-Manipulation)

```
Demo video [B, T, 224, 224, 3]
       │
       └── SigLIP ViT (frozen, per-frame)
                  │
       Temporal Transformer (4 layers, trainable)
                  │
       Perceiver Resampler (2 layers, trainable)
                  │
       task_embedding [B, 16, 768]
                  │
       Task Cross-Attention (2 layers, trainable)
                  │
       Eagle VLM Backbone [B, seq, 2048]  ←── from GR00T-N1.6-3B pretrain
       (language-conditioned, frozen)
                  │
       DiT Action Head (32 layers)         ←── from GR00T-N1.6-3B pretrain
                  │
       Actions [B, 43]  (Unitree G1 joint targets)
```

**Base model**: NVIDIA GR00T-N1.6-3B (Eagle VLM + DiT action head from pretrain).
**DC addition**: Demo encoder (SigLIP + temporal transformer + perceiver resampler + task cross-attention).
**Fine-tuning**: LoRA on language_model layers + full gradient on DC modules.

---

## Simulator

Validates the full pipeline in Isaac Sim + Isaac Lab physics simulation.

**Mode 1 (open-loop replay)**:
```
goal_video → FlowDiT V2 → predicted_actions.csv
                                │
                    Isaac Sim CSV replay
                                │
                    trajectory.csv + metrics
```

**Mode 2 (closed-loop realtime)**:
```
goal_video → FlowDiT warmup_realtime()
                │
        [ per-step @ 2 Hz ]
        Isaac Sim current_obs → FlowDiT predict_realtime() → execute action
                │
        trajectory.csv + diagnostics.json
```

**Metrics**: SR (success rate at radius), SPL (success weighted by path length), ATE (average trajectory error), Direction Accuracy (cosine similarity > 0.75).

---

## Data Flow (End-to-End)

```
1. User provides: task description + workspace image
2. Part 1 generates: reference_video.mp4 (61 frames @ 16 fps)
3. Part 2a produces: actions.npy [T, 3] — velocity commands
4. Part 2b produces: joint_targets.npy [T, 43] — arm/body joints
5. Simulator evaluates: SR, SPL, ATE across 41 tasks
```
