# Project Status

**Author**: Jeffrin Sam | Skoltech | Updated: March 2026

---

## What exists and works

| Component | Status | Location |
|-----------|--------|----------|
| V1 model (reactive MLP) | Complete — legacy | `flow_constrained/` |
| V2 model (goal-conditioned DiT) | **Complete — production** | `flow_constrained_v2/` |
| V3 model (humanoid DiT) | **Complete — inference-only** | `flowdit_v3_humanoid_inference/` |
| RECON dataset (8,948 videos) | Ready | `dataset/recon/` |
| V2 trained checkpoint | epoch 67, MSE 0.046 | `flow_constrained_v2/checkpoints/best.pth` |
| V3 trained checkpoint | humanoid finetuned | `flowdit_v3_humanoid_inference/checkpoints/flowdit_v3_humanoid_best.pt` |
| Web inference UI | Working | `web_app/` |
| Code-only zip (for AI agents) | Ready | `flowdit_codebase.zip` |

---

## V2 Architecture (production model)

```
Reference video (16 frames) + Current frame → GoalEncoder + ObsEncoder → DiT → [vx,vy,yaw] × 8 steps
```

- Trained on RECON (8,948 videos, 7,158 train / 1,790 val)
- 75 epochs, AdamW, cosine LR decay
- Best: epoch 67, val MSE = 0.046
- 43.5M total params, ~3M trainable
- ~50ms inference, 2 Hz closed-loop

---

## V3 Architecture (humanoid)

```
Reference video + Current frame → VideoEncoder (DINOv2 + VideoMAE) → CrossAttn → DiT → velocities + trajectory
```

- Finetuned on Unitree G1 data (GR00T + Arena episodes)
- Dual output: [vx, vy, vz, yaw] × 8 + [x,y,z] × 16 waypoints
- Inference-only bundle (no training code — training was done separately)

---

## What is next

1. **Integrate V2 with Part 1 (video generation)**
   - Part 1 generates reference videos via ClaudeOpusBrain (Cosmos 2.5)
   - V2 takes that reference video as goal input
   - Currently tested on RECON; needs end-to-end test with generated videos

2. **Real robot deployment**
   - `robot_navigation.py` is ready — adapt camera capture + velocity command functions for your robot API
   - See `flow_constrained_v2/INFERENCE_GUIDE.md`

3. **V2+ improvements (experimental, archived)**
   - The experimental V2+ code is in `archive/v2plus_research/`
   - Key pending fixes: early stopping bug, speed magnitude prediction, CLIP training
   - Not production-ready — do not deploy

---

## Environments

| Env | Use |
|-----|-----|
| `conda activate flowdit_v2_py310` | V2 training + inference |
| `flowdit_v3_humanoid_inference/.venv/` | V3 inference (torch 2.9.1+cu128) |
