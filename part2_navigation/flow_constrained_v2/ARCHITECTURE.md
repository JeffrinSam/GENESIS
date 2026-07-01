# FlowDiT V2 - Goal-Conditioned Navigation Architecture

## 🏗️ Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Goal Definition (ONE-TIME per navigation task)                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Input                                                              │
│  ┌──────────────┐                                                        │
│  │ Start Image  │  ──┐                                                   │
│  └──────────────┘    │                                                   │
│                      ├──→  Video Generation Model                        │
│  ┌──────────────┐    │     (YOUR separate model)                        │
│  │ Text Prompt  │  ──┘           ↓                                       │
│  │"go to table" │          Reference Video                               │
│  └──────────────┘          [3-15 sec, 48-240 frames]                    │
│                            Shows: Start → Goal path                      │
│                                  ↓                                       │
│                            Store reference_video.mp4                     │
│                            (Used throughout navigation)                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Closed-Loop Navigation (CONTINUOUS @ 2 Hz)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Perception                                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  Robot Camera  ──→  Capture Frame  ──→  Preprocess              │   │
│  │  (Live video)       [640x480]           [224x224, normalized]   │   │
│  │                                                ↓                 │   │
│  │                                         current_obs.jpg          │   │
│  │                                         [224, 224, 3]            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Action Prediction (FlowDiT V2 Model)                    │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  ┌────────────────┐         ┌────────────────┐                  │   │
│  │  │ Reference Video│         │ Current Obs    │                  │   │
│  │  │ [16, 3, 224,224]│         │ [3, 224, 224]  │                  │   │
│  │  └────────┬───────┘         └────────┬───────┘                  │   │
│  │           │                          │                          │   │
│  │           ↓                          ↓                          │   │
│  │  ┌────────────────┐         ┌────────────────┐                  │   │
│  │  │ Goal Encoder   │         │  Obs Encoder   │                  │   │
│  │  │ (DINOv2+Flow)  │         │  (DINOv2)      │                  │   │
│  │  │ + Temporal Attn│         │  Single Frame  │                  │   │
│  │  └────────┬───────┘         └────────┬───────┘                  │   │
│  │           │                          │                          │   │
│  │           │ [768 vision]             │ [768 obs]                │   │
│  │           │ [256 flow]               │                          │   │
│  │           └──────────┬───────────────┘                          │   │
│  │                      ↓                                          │   │
│  │              ┌──────────────┐                                   │   │
│  │              │ Concatenate  │                                   │   │
│  │              │ Condition    │                                   │   │
│  │              │ [2304 dim]   │                                   │   │
│  │              └──────┬───────┘                                   │   │
│  │                     │                                           │   │
│  │                     ↓                                           │   │
│  │         ┌───────────────────────┐                               │   │
│  │         │ Diffusion Transformer │                               │   │
│  │         │ (DiT)                 │                               │   │
│  │         │ - 8 blocks            │                               │   │
│  │         │ - Adaptive LayerNorm  │                               │   │
│  │         │ - Multi-head Attention│                               │   │
│  │         └───────────┬───────────┘                               │   │
│  │                     │                                           │   │
│  │                     ↓                                           │   │
│  │              ┌──────────────┐                                   │   │
│  │              │ DDIM Sampling│                                   │   │
│  │              │ (10 steps)   │                                   │   │
│  │              └──────┬───────┘                                   │   │
│  │                     │                                           │   │
│  │                     ↓                                           │   │
│  │              ┌──────────────┐                                   │   │
│  │              │   Actions    │                                   │   │
│  │              │ [8, 3]       │                                   │   │
│  │              │ vx, vy, yaw  │                                   │   │
│  │              └──────────────┘                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Execution                                                │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  Actions [8, 3] ──→ Execute first 3 ──→ Robot Control           │   │
│  │   ┌──────────┐                          ┌────────────────┐      │   │
│  │   │ vx: 0.55 │ ──────────────────────→  │ Motor Commands │      │   │
│  │   │ vy: 0.00 │ @ 16 Hz execution       │ Wheel velocities│      │   │
│  │   │ yaw: 0.12│                          └────────────────┘      │   │
│  │   └──────────┘                                   ↓              │   │
│  │                                            Robot Moves           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: Loop Back                                                │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                  │   │
│  │  ┌──────────────┐                                                │   │
│  │  │ Goal Reached?│ ──No──→ Go to STEP 1 (get new frame)          │   │
│  │  └──────┬───────┘                                                │   │
│  │         │                                                        │   │
│  │        Yes                                                       │   │
│  │         │                                                        │   │
│  │         ↓                                                        │   │
│  │   ┌────────────┐                                                 │   │
│  │   │ STOP & Done│                                                 │   │
│  │   └────────────┘                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture Details

### Goal Encoder (Reference Video → Features)

```
Reference Video [B, 16, 3, 224, 224]
           ↓
┌──────────────────────────────────────┐
│ Sample 16 frames uniformly           │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ DINOv2 ViT-B/14 (Per Frame)         │
│ - Pretrained vision encoder          │
│ - Frozen weights                     │
│ - Output: [B, 16, 768]               │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Temporal Attention                   │
│ - Multi-head attention across time   │
│ - Aggregates temporal info           │
│ - Output: [B, 768] (vision features) │
└──────────────┬───────────────────────┘
               │
               ├─→ Goal Vision Features [768]
               │
┌──────────────┴───────────────────────┐
│ Optical Flow Computation             │
│ - Compute flow between frames        │
│ - CNN encoder for flow               │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Flow Temporal Attention              │
│ - Aggregate flow over time           │
│ - Output: [B, 256] (flow features)   │
└──────────────┬───────────────────────┘
               │
               └─→ Goal Flow Features [256]
```

### Observation Encoder (Current Frame → Features)

```
Current Observation [B, 3, 224, 224]
           ↓
┌──────────────────────────────────────┐
│ DINOv2 ViT-B/14                      │
│ - Same encoder as goal               │
│ - Shared weights                     │
│ - Single frame processing            │
│ - Output: [B, 768]                   │
└──────────────┬───────────────────────┘
               │
               └─→ Observation Features [768]
```

### Diffusion Transformer (Condition → Actions)

```
Condition Vector [B, 2304]
  = [Goal Vision: 768] + [Goal Flow: 256] + [Obs: 768] + [Lang: 512]
           ↓
┌──────────────────────────────────────┐
│ Noisy Actions [B, 8, 3]              │
│ (Training: GT + noise)               │
│ (Inference: Random noise)            │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Action Embedding                     │
│ Linear: [B, 8, 3] → [B, 8, 512]      │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Condition Projection                 │
│ Linear: [B, 2304] → [B, 512]         │
└──────────────┬───────────────────────┘
               │
               ├─→ Conditioning [B, 512]
               │
┌──────────────┴───────────────────────┐
│ Timestep Embedding                   │
│ Sinusoidal encoding                  │
└──────────────┬───────────────────────┘
               │
               └─→ Add to Conditioning
               ↓
┌──────────────────────────────────────┐
│ DiT Block 1-8 (Repeated)             │
│ ┌──────────────────────────────────┐ │
│ │ Adaptive LayerNorm (adaLN)       │ │
│ │ - Scale & shift based on         │ │
│ │   condition                      │ │
│ └──────────────────────────────────┘ │
│ ┌──────────────────────────────────┐ │
│ │ Multi-head Self-Attention        │ │
│ │ - 8 heads                        │ │
│ │ - Attention across action steps  │ │
│ └──────────────────────────────────┘ │
│ ┌──────────────────────────────────┐ │
│ │ Adaptive LayerNorm (adaLN)       │ │
│ └──────────────────────────────────┘ │
│ ┌──────────────────────────────────┐ │
│ │ MLP (4x expansion)               │ │
│ │ - 512 → 2048 → 512               │ │
│ └──────────────────────────────────┘ │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Output Layer                         │
│ Linear: [B, 8, 512] → [B, 8, 3]      │
└──────────────┬───────────────────────┘
               ↓
         Noise Prediction [B, 8, 3]
               ↓
┌──────────────────────────────────────┐
│ DDIM Denoising (Inference)           │
│ - 10 iterative steps                 │
│ - Gradually denoise to clean actions │
└──────────────┬───────────────────────┘
               ↓
         Actions [B, 8, 3]
```

---

## 🔄 Training vs Inference

### Training Data Flow

```
Dataset Sample
├─ Full trajectory video [T, 3, 224, 224]
├─ Actions [T, 3]
└─ Sample:
    ├─ Goal: Last 16 frames → Goal features
    ├─ Obs: Random frame t → Obs features
    └─ Target: Actions[t:t+8] → Ground truth
                      ↓
         Add Gaussian noise to actions
                      ↓
              Predict noise with model
                      ↓
           Compute MSE(predicted, true_noise)
                      ↓
               Backprop & Update
```

### Inference Data Flow

```
User provides:
├─ Reference video (from video gen)
└─ Current camera frame
           ↓
    Extract features
           ↓
    Start from random noise
           ↓
    DDIM denoising (10 steps)
           ↓
    Clean actions [8, 3]
           ↓
    Execute on robot
```

---

## 📊 Key Dimensions

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| **Goal Encoder** | [B, 16, 3, 224, 224] | [B, 1024] | ~86M (frozen) |
| - DINOv2 | [B, 3, 224, 224] | [B, 768] | 86M |
| - Flow CNN | [B, 2, 224, 224] | [B, 256] | 2.5M |
| - Temporal Attn | [B, 16, 768] | [B, 768] | 2.4M |
| **Obs Encoder** | [B, 3, 224, 224] | [B, 768] | 86M (shared) |
| **Language Encoder** | None | [B, 512] | <1K (dummy) |
| **DiT** | [B, 8, 3] | [B, 8, 3] | 43.5M |
| - 8 DiT Blocks | [B, 8, 512] | [B, 8, 512] | 38M |
| - Projections | Various | Various | 5.5M |
| **Total Model** | - | - | **43.5M trainable** |

---

## ⚡ Performance Metrics

### Inference Speed (GPU: RTX 3090)

| Step | Time (ms) | Notes |
|------|-----------|-------|
| Goal encoding | 15 ms | Once per task |
| Obs encoding | 8 ms | Per control cycle |
| DiT forward (1 step) | 3 ms | × 10 for DDIM |
| DDIM sampling (10 steps) | 30 ms | Total denoising |
| **Total per cycle** | **~50 ms** | **20 Hz capable** |

### Memory Usage

- Model parameters: 43.5M × 4 bytes = **174 MB**
- Activation memory: ~500 MB (batch_size=1)
- Total GPU memory: **~1 GB**

### Control Frequency

- Recommended: **2 Hz** (500 ms/cycle)
- Achievable: **4 Hz** (250 ms/cycle)
- Maximum: **20 Hz** (50 ms/cycle) with optimizations

---

## 🎯 Why This Architecture Works

### 1. Goal-Conditioned Design
- **Reference video** = Where to go (FIXED memory)
- **Current observation** = Where I am (CHANGES)
- **Actions** = How to get there (COMPUTED)

Just like humans: Remember destination, look at surroundings, adjust path.

### 2. Temporal Reasoning
- Goal encoder processes video over time
- Captures motion patterns and trajectory
- Flow features encode dynamics

### 3. Diffusion for Robustness
- Iterative refinement → smoother actions
- Handles uncertainty in predictions
- More stable than direct regression

### 4. Efficient Frozen Encoders
- DINOv2 pretrained on vision tasks
- Only train action prediction (DiT)
- Faster training, better generalization

---

## 🔬 Comparison with Alternatives

| Approach | Our FlowDiT V2 | Alternative |
|----------|----------------|-------------|
| **Input** | Goal video + Current obs | Only current obs |
| **Planning** | Implicit in goal encoding | Explicit planning |
| **Memory** | Reference video (constant) | Recurrent state |
| **Control** | Closed-loop (2 Hz) | Open-loop |
| **Adaptation** | Real-time replanning | Fixed trajectory |

---

## 📝 Implementation Notes

### Why DINOv2?
- Self-supervised learning on diverse data
- Strong visual representations
- Good for navigation/spatial reasoning

### Why Diffusion?
- Better than regression for multi-modal actions
- Handles uncertainty naturally
- Smooth, realistic trajectories

### Why 8 Action Steps?
- Balance between:
  - Planning horizon (longer = better planning)
  - Reactivity (shorter = faster adaptation)
- Execute 3, replan with new observation

### Why 2 Hz Control?
- Fast enough for navigation
- Allows model inference time
- Safer than faster control (can react to obstacles)

---

**Next:** See `COMPARISON.md` for differences between v1 and v2!
