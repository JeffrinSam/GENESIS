# DC-GR00T: Demo-Conditioned GR00T — In-Depth Technical Documentation

## Table of Contents

1. [What is DC-GR00T?](#1-what-is-dc-groot)
2. [Motivation: Why Demo-Conditioning?](#2-motivation-why-demo-conditioning)
3. [Architecture Overview](#3-architecture-overview)
4. [Component-by-Component Deep Dive](#4-component-by-component-deep-dive)
   - 4.1 [Eagle Backbone (VLM)](#41-eagle-backbone-vlm)
   - 4.2 [Demo Encoder](#42-demo-encoder)
   - 4.3 [Task Cross-Attention](#43-task-cross-attention)
   - 4.4 [Action Head (Diffusion Transformer)](#44-action-head-diffusion-transformer)
   - 4.5 [Flow Matching](#45-flow-matching)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Loss Functions](#8-loss-functions)
9. [Configuration Reference](#9-configuration-reference)
10. [File Map](#10-file-map)

---

## 1. What is DC-GR00T?

DC-GR00T (Demo-Conditioned GR00T) is a modified version of NVIDIA's GR00T N1.6 Vision-Language-Action (VLA) model. It extends the base GR00T by adding the ability to **condition action prediction on a demonstration video**.

**In plain terms:** You show the robot a video of someone (a human, another robot, or even a synthetically generated video) performing a task. The robot watches that video, understands *what needs to be done*, and then executes the task using its own body and its own camera feed in a closed-loop manner.

The base GR00T N1.6 uses **language instructions** (e.g., "pick up the cup") to tell the robot what to do. DC-GR00T replaces or augments this with **video demonstrations** — which can convey far more nuanced task information (contact points, motion trajectories, object relationships) than language alone.

**Key properties:**
- **Cross-embodiment**: The demo video doesn't need to come from the same robot. A human hand demo can drive a Unitree G1 humanoid.
- **Closed-loop**: At execution time, the robot uses its own sensors, not replay. It reacts to the actual environment.
- **One-shot**: A single demo video is enough. No re-training needed per new task (in theory, after sufficient pre-training).

**Research lineage:** Based on ideas from Vid2Robot (Google), CrossFormer, and Track2Act.

---

## 2. Motivation: Why Demo-Conditioning?

| Approach | Pros | Cons |
|----------|------|------|
| **Language-conditioned** (base GR00T) | Easy to specify tasks; composable | Ambiguous for precise motions; can't convey contact details |
| **Demo-conditioned** (DC-GR00T) | Rich task specification; cross-embodiment; captures motion patterns | Requires demo video; harder to generalize across very different tasks |
| **Teleoperation replay** | Exact execution | No generalization; open-loop; same embodiment required |

DC-GR00T sits in the sweet spot: richer than language, more flexible than replay.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DC-GR00T Architecture                        │
│                                                                     │
│  ┌──────────────────────┐       ┌─────────────────────────────┐    │
│  │   DEMO VIDEO         │       │   ROBOT OBSERVATION         │    │
│  │   [B, T, H, W, C]   │       │   (ego camera + language)   │    │
│  └──────────┬───────────┘       └──────────────┬──────────────┘    │
│             │                                   │                   │
│             ▼                                   ▼                   │
│  ┌──────────────────────┐       ┌─────────────────────────────┐    │
│  │   DEMO ENCODER       │       │   EAGLE BACKBONE (VLM)      │    │
│  │                      │       │   Cosmos-Reason 2B           │    │
│  │  1. SigLIP ViT       │       │   (frozen or top-4 tuned)   │    │
│  │     (per-frame)      │       │                             │    │
│  │  2. Temporal          │       │   → backbone_features       │    │
│  │     Transformer       │       │     [B, seq_len, 2048]     │    │
│  │     (4 layers)        │       │                             │    │
│  │  3. Perceiver         │       └──────────────┬──────────────┘    │
│  │     Resampler         │                      │                   │
│  │     (2 layers)        │                      │                   │
│  │                      │                      │                   │
│  │  → task_embedding    │                      │                   │
│  │    [B, 16, 768]      │                      │                   │
│  └──────────┬───────────┘                      │                   │
│             │                                   │                   │
│             └──────────────┬────────────────────┘                   │
│                            ▼                                        │
│             ┌──────────────────────────────┐                        │
│             │   TASK CROSS-ATTENTION       │                        │
│             │   (2 layers)                 │                        │
│             │                              │                        │
│             │   backbone_features attends  │                        │
│             │   to task_embedding          │                        │
│             │                              │                        │
│             │   + optional: concatenate    │                        │
│             │     projected task tokens    │                        │
│             └──────────────┬───────────────┘                        │
│                            │                                        │
│                            ▼                                        │
│  ┌────────────────┐  ┌────────────────────────────────────────┐    │
│  │ ROBOT STATE    │  │  NOISY ACTION (flow matching)          │    │
│  │ [B, 1, 29]    │  │  [B, 16, 29]                           │    │
│  │       │        │  │       │                                 │    │
│  │  state_encoder │  │  action_encoder + pos_embed             │    │
│  │  (CategoryMLP) │  │  (MultiEmbodimentActionEncoder)        │    │
│  │       │        │  │       │                                 │    │
│  │  [B,1,1536]   │  │  [B,16,1536]                           │    │
│  └───────┬────────┘  └───────┬────────────────────────────────┘    │
│          │                    │                                      │
│          └────────┬───────────┘                                      │
│                   ▼                                                  │
│          ┌────────────────────────────────┐                          │
│          │   sa_embs [B, 17, 1536]       │                          │
│          └────────────────┬───────────────┘                          │
│                           ▼                                          │
│          ┌────────────────────────────────┐                          │
│          │   DIFFUSION TRANSFORMER (DiT) │                          │
│          │   AlternateVLDiT              │                          │
│          │   32 layers, 32 heads         │                          │
│          │   head_dim=48 → 1536 hidden   │                          │
│          │                               │                          │
│          │   cross-attends to fused      │                          │
│          │   backbone+task features      │                          │
│          │   every 2 blocks              │                          │
│          └────────────────┬───────────────┘                          │
│                           ▼                                          │
│          ┌────────────────────────────────┐                          │
│          │   ACTION DECODER              │                          │
│          │   CategorySpecificMLP         │                          │
│          │   → pred_velocity [B, 16, 29] │                          │
│          └────────────────────────────────┘                          │
│                                                                      │
│   Training: MSE(pred_velocity, true_velocity) with action_mask      │
│   Inference: iterative denoising (4 steps) from Gaussian noise      │
└──────────────────────────────────────────────────────────────────────┘
```

**Total parameters:** ~3.85 billion (3.55B from base GR00T + ~300M from demo encoder and task cross-attention).

---

## 4. Component-by-Component Deep Dive

### 4.1 Eagle Backbone (VLM)

**File:** `gr00t/model/modules/eagle_backbone.py`

The Eagle backbone is NVIDIA's Cosmos-Reason-2B VLM variant. It processes the robot's **current observation** — the ego camera image and any language prompt — and produces rich visual-language features.

**What it does:**
1. Takes the robot's ego camera frame (448×448 RGB) and optional language text
2. Encodes the image through a vision encoder (ViT) into patch tokens
3. Processes through a 2B-parameter language model that fuses vision and language
4. Outputs `backbone_features` of shape `[B, seq_len, 2048]` — a sequence of 2048-dim feature vectors

**Key details:**
- Uses **SDPA** (Scaled Dot Product Attention) instead of Flash-Attention for RTX 5090 Blackwell compatibility
- Supports flexible resolution — encodes images in native aspect ratio without padding
- The backbone is mostly **frozen** during DC-GR00T training (`tune_llm=False`, `tune_visual=False`)
- Optionally the top 4 LLM layers can be unfrozen (`tune_top_llm_layers`)

**Why frozen?** The backbone already has strong visual-language understanding from GR00T's massive pretraining. DC-GR00T only needs to learn the *new* demo-conditioning pathway.

---

### 4.2 Demo Encoder

**File:** `gr00t/model/demo_conditioned/demo_encoder.py`

The Demo Encoder is the core novelty of DC-GR00T. It takes a raw demonstration video and compresses it into a fixed-size **task embedding** — a compact representation of "what needs to be done."

**Three-stage pipeline:**

#### Stage 1: Frame Encoder (SigLIP ViT)
```
Input:  [B, T, C, H, W]  — T keyframes from demo video (default T=16)
Output: [B, T, 768]       — per-frame embeddings
```

- Uses a pretrained **SigLIP** (google/siglip-base-patch16-224) vision transformer
- Each frame is independently encoded to a 768-dim vector
- The ViT is **frozen** — its parameters don't change during training
- Frames are resized to 224×224 and normalized to [0, 1]

**Why SigLIP?** It's trained with contrastive image-text learning, so it produces semantically meaningful frame embeddings out of the box.

#### Stage 2: Temporal Transformer (4 layers)
```
Input:  [B, T, 768]  — independent frame embeddings
Output: [B, T, 768]  — temporally-aware frame embeddings
```

- Adds **sinusoidal positional encoding** so the model knows frame ordering
- 4-layer transformer encoder with:
  - 8 attention heads
  - 2048 FFN hidden dim
  - Pre-norm (LayerNorm before attention/FFN, not after)
  - GELU activation
  - 0.1 dropout
- **Self-attention** across frames lets the model learn:
  - Motion patterns (how objects move over time)
  - State transitions (before → after relationships)
  - Temporal dynamics (speed, pauses, key moments)

**This is what makes it more than just "average frame features."** A frame of a cup on a table looks the same whether the task is "pick up the cup" or "push the cup" — but the temporal context disambiguates.

#### Stage 3: Perceiver Resampler (2 layers)
```
Input:  [B, T, 768]   — variable-length temporal embeddings (T can vary)
Output: [B, 16, 768]  — fixed 16 task tokens
```

Based on Flamingo's Perceiver Resampler architecture:

1. Maintains 16 **learned query tokens** (nn.Parameter, randomly initialized)
2. Each resampler layer applies:
   - **Cross-attention**: queries attend to frame embeddings (queries ask "what objects?", "what motion?", etc.)
   - **Self-attention**: queries attend to each other (they specialize and coordinate)
   - **FFN**: nonlinear transformation
3. All with pre-norm residual connections

**Why 16 tokens?** It's a hyperparameter. 16 tokens × 768 dims = 12,288 floats to describe an entire task. This is a massive compression (from potentially thousands of image patches across dozens of frames) that forces the model to extract only task-relevant information.

#### Demo Type Embedding

The encoder also adds a **demo type embedding** — a learned 768-dim vector that tells the model what kind of demo it's watching:

| ID | Type | Description |
|----|------|-------------|
| 0 | human | Human hand performing the task |
| 1 | robot | Another robot performing the task |
| 2 | cosmos | Synthetically generated video (e.g., from NVIDIA Cosmos) |
| 3 | own | The same robot from a different viewpoint |

This is added to every frame embedding before the temporal transformer, allowing the model to account for embodiment differences (a human hand grasps differently than a robot gripper).

---

### 4.3 Task Cross-Attention

**File:** `gr00t/model/demo_conditioned/dc_gr00t.py` — `TaskCrossAttention` class (line 134) and `process_backbone_output_with_task` method (line 317)

This is where the **demo understanding meets the robot's current observation**. The task embedding from the Demo Encoder is fused into the backbone features through two mechanisms:

#### Mechanism 1: Cross-Attention Layers (2 layers)

Each `TaskCrossAttention` layer has three sub-layers:

```
backbone_features [B, seq_len, 2048]
        │
        ▼
   Cross-Attention ── query: backbone_features
        │              key/value: task_embedding (projected from 768→2048)
        │
        ▼ (residual connection)
   Self-Attention ── among backbone features
        │
        ▼ (residual connection)
   FFN (Linear→GELU→Dropout→Linear→Dropout)
        │
        ▼ (residual connection)
   task-conditioned backbone_features [B, seq_len, 2048]
```

**What's happening conceptually:** The robot's visual features are "asking questions" to the task embedding. For example:
- "I see a red cup — is that the object I should manipulate?"
- "My gripper is near the table — should I reach down?"

The cross-attention learns to selectively pull task-relevant information from the demo embedding into the observation features.

#### Mechanism 2: Token Concatenation (when `fuse_task_with_language=True`)

After cross-attention, the task tokens are **projected to 2048 dims** and **concatenated** to the backbone features:

```
backbone_features:  [B, seq_len, 2048]
task_proj:          [B, 16, 2048]        (projected from 768→2048)
────────────────────────────────────
concatenated:       [B, seq_len+16, 2048]
```

The attention masks are extended accordingly:
- `backbone_attention_mask` gets 16 extra `True` entries
- `image_mask` gets 16 extra `False` entries (task tokens are not image tokens)

This dual fusion (cross-attention + concatenation) gives the DiT two ways to use the task information:
1. Through the already-modified backbone features (implicit)
2. Through direct cross-attention to the appended task tokens in the DiT (explicit)

---

### 4.4 Action Head (Diffusion Transformer)

**File:** `gr00t/model/demo_conditioned/dc_gr00t.py` — `DCGr00tActionHead` class (line 216)

The action head takes the fused features and produces robot actions. It's built around a **Diffusion Transformer (DiT)** — the same architecture used in modern image and video generation models, but here it generates *action trajectories*.

#### State Encoder
```
robot_state [B, 1, 29]  →  CategorySpecificMLP  →  state_features [B, 1, 1536]
```

- `CategorySpecificMLP`: A different MLP per embodiment ID, so the same 29-dim state vector can mean different things for different robots
- `max_state_dim=29` (for Unitree G1's 29 DOF)
- During training, state features are randomly dropped out (`state_dropout_prob`) and replaced with a learned mask token — this teaches the model to work even with unreliable state information

#### Action Encoder
```
noisy_actions [B, 16, 29]  +  timestep [B]  +  embodiment_id [B]
        │
   MultiEmbodimentActionEncoder
        │
   action_features [B, 16, 1536]  +  positional_embedding
```

- The action horizon is 16 — the model predicts 16 future timesteps at once (action chunking)
- Each action vector is 29-dim (matching G1's DOF)
- Positional embeddings are added so the model knows the temporal ordering of action steps

#### DiT (AlternateVLDiT)

```
sa_embs = concat(state_features, action_features)  →  [B, 17, 1536]
                                                       (1 state + 16 action tokens)

DiT processes sa_embs while cross-attending to fused backbone+task features
```

Configuration:
- **32 layers** (2× larger than N1.5's 16 layers)
- **32 attention heads**, head_dim=48 → hidden_size=1536
- **Adaptive layer norm** (AdaLN) conditioned on diffusion timestep
- Cross-attention to VL features **every 2 blocks** (`attend_text_every_n_blocks=2`)
- `interleave_self_attention=True`
- Dropout: 0.2

The DiT output is decoded through another `CategorySpecificMLP` to get the predicted velocity field:
```
DiT output [B, 17, 1536]  →  action_decoder  →  [B, 17, 29]  →  take last 16  →  pred_velocity [B, 16, 29]
```

---

### 4.5 Flow Matching

DC-GR00T uses **flow matching** (also called "rectified flow") instead of DDPM-style diffusion. This is a more efficient approach to generative modeling.

#### Training

1. **Sample noise:** `noise ~ N(0, I)` with same shape as actions `[B, 16, 29]`
2. **Sample time:** `t ~ Beta(1.5, 1.0)` then `t = (1 - t) * 0.999` — biased toward t≈1 (near the data)
3. **Interpolate:** `noisy_trajectory = (1 - t) * noise + t * actions`
   - At t=0: pure noise
   - At t=1: clean actions
4. **Target velocity:** `velocity = actions - noise` (the straight-line direction from noise to data)
5. **Predict velocity** through the model
6. **Loss:** MSE between predicted and true velocity, masked by `action_mask`

#### Inference (Denoising)

1. Start with `actions = random Gaussian noise [B, 16, 29]`
2. Iterate for `num_inference_timesteps=4` steps:
   - Compute timestep `t = step / 4`
   - Predict velocity `v = model(actions, t)`
   - Euler step: `actions = actions + (1/4) * v`
3. After 4 steps, `actions` has been transported from noise to a plausible action trajectory

**Why only 4 steps?** Flow matching with straight paths requires far fewer steps than DDPM (which typically needs 50-1000 steps). The Beta(1.5, 1.0) time distribution biases training toward the data end, which is where accuracy matters most.

The `num_timestep_buckets=1000` discretizes the continuous time into 1000 bins for the timestep embedding.

---

## 5. Data Pipeline

**File:** `scripts/demo_conditioned/train_dc_groot.py` — `DCDataset` and `DCDataCollator`

### Dataset Structure

```
dataset/
├── episodes.jsonl              # Episode metadata (one JSON per line)
├── videos/
│   ├── demo/                   # Demonstration videos
│   │   ├── episode_000000.mp4  # What to do (can be human/robot/cosmos)
│   │   └── ...
│   └── ego_view/               # Robot's first-person execution videos
│       ├── episode_000000.mp4  # What the robot saw while executing
│       └── ...
└── data/
    ├── episode_000000.parquet  # Timestep-level action/state data
    └── ...
```

### Per-Sample Data Flow

1. **Demo video** → sample 16 keyframes uniformly → `[16, H, W, C]`
2. **Ego video** → pick a random timestep → extract single frame → resize to 448×448 → `[448, 448, 3]`
3. **Actions** → extract 16-step chunk starting from the random timestep → concatenate joint groups → `[16, 29]`
4. **State** → robot joint positions at the sampled timestep → `[29]`
5. **Demo type** → integer label (0=human, 1=robot, 2=cosmos, 3=own)

### Collation

The `DCDataCollator`:
1. Resizes demo frames to 224×224 for SigLIP
2. Processes ego frames through the Eagle VLM's data collator (creates `pixel_values`, `input_ids`, `attention_mask`)
3. Batches everything into tensors

---

## 6. Training Pipeline

**File:** `scripts/demo_conditioned/train_dc_groot.py` — `DCTrainer` and `main()`

### Step-by-Step Training Loop

For each batch:

1. **Extract demo frames** from the batch
2. **Encode demo** → `task_embedding = model.encode_demo(demo_frames, demo_type)` → `[B, 16, 768]`
3. **Process robot observation** through Eagle backbone → `backbone_features [B, seq_len, 2048]`
4. **Fuse task + observation** via cross-attention and concatenation
5. **Encode robot state** → `state_features [B, 1, 1536]`
6. **Sample noise and time** for flow matching
7. **Create noisy actions** by interpolating between noise and ground-truth actions
8. **Encode noisy actions** → `action_features [B, 16, 1536]`
9. **Run DiT** on concatenated state+action features, cross-attending to fused backbone features
10. **Decode** to predicted velocity
11. **Compute MSE loss** between predicted and true velocity, weighted by action mask
12. **Optionally** compute **alignment loss** (contrastive, see Section 8)
13. **Backpropagate** through demo encoder + task cross-attention + action head (backbone mostly frozen)

### LoRA Optimization

For RTX 5090 32GB training, **LoRA** is applied to the language model layers:

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
)
```

This reduces trainable parameters from 3.35B → 7.3M (99.78% reduction) and memory from 30GB+ → 19.5GB.

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Gradient accumulation | 8 (effective batch 32) |
| Learning rate | 1e-4 |
| Scheduler | Cosine with 1000-step warmup |
| Max steps | 30,000 |
| Precision | BF16 |
| Gradient checkpointing | Enabled |

---

## 7. Inference Pipeline

**File:** `gr00t/model/demo_conditioned/dc_policy.py`

### Two-Phase Inference

#### Phase 1: Offline — Encode Demo (once per task)

```python
policy = DCPolicy.from_pretrained("path/to/checkpoint")
policy.set_demo("demo_video.mp4", demo_type="human")
# Internally: load video → sample 16 keyframes → DemoEncoder → task_embedding [1, 16, 768]
# This embedding is cached and reused for every control step
```

#### Phase 2: Online — Closed-Loop Control (every timestep)

```python
while not done:
    action = policy.get_action(
        observation={"ego_view": camera_frame},    # [H, W, C]
        state={"joints": joint_positions},          # [29]
    )
    robot.execute(action["action"])                 # [16, 29] action chunk
    camera_frame = robot.get_observation()
```

At each control step:
1. Eagle backbone processes the current ego camera frame → backbone features
2. Task cross-attention fuses in the cached task embedding
3. State encoder processes current joint state
4. DiT denoises from Gaussian noise in 4 steps → predicted action trajectory `[16, 29]`
5. Robot executes the first few actions, then re-plans

### REST API Server

DC-GR00T also provides a Flask-based REST API server (`DCPolicyServer`) for network deployment:

- `POST /set_demo` — Upload demo video, compute task embedding
- `POST /get_action` — Send observation JSON, receive action JSON
- `GET /health` — Health check

---

## 8. Loss Functions

### Primary: Action Flow Matching Loss

```python
action_loss = MSE(pred_velocity, true_velocity) * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

- Per-element MSE between predicted and ground-truth velocity fields
- Masked by `action_mask` — only valid action dimensions contribute to the loss
- The mask handles variable DOF across different embodiments

### Auxiliary: Video Alignment Loss (InfoNCE)

**File:** `gr00t/model/demo_conditioned/demo_encoder.py` — `VideoAlignmentLoss` class

When the dataset includes both a third-party demo and the robot's own execution video for the same task:

1. Encode both videos through the demo encoder → `demo_embedding`, `robot_embedding`
2. Pool each to a single vector by averaging across tokens
3. L2-normalize both
4. Compute cosine similarity matrix: `sim = demo_pooled @ robot_pooled.T / temperature`
5. Apply symmetric InfoNCE:
   - `loss_d2r = CrossEntropy(sim, diagonal_labels)` — demo→robot direction
   - `loss_r2d = CrossEntropy(sim.T, diagonal_labels)` — robot→demo direction
   - `loss = (loss_d2r + loss_r2d) / 2`

**Purpose:** Ensures that embeddings for the *same task* (whether seen as a human demo or robot execution) are similar, while embeddings for *different tasks* are pushed apart. Temperature τ=0.07.

**Total loss:**
```
total_loss = action_loss + 0.1 * alignment_loss
```

---

## 9. Configuration Reference

From `DCGr00tConfig` (dataclass):

### Base Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `nvidia/Eagle-Block2A-2B-v2` | VLM backbone |
| `backbone_embedding_dim` | 2048 | VLM output feature dimension |
| `tune_llm` | False | Whether to train the LLM layers |
| `tune_visual` | False | Whether to train the vision encoder |
| `tune_top_llm_layers` | 0 | Number of top LLM layers to unfreeze |

### Action Head

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_state_dim` | 29 | Robot state vector dimension |
| `max_action_dim` | 29 | Robot action vector dimension |
| `action_horizon` | 16 | Number of future timesteps to predict |
| `hidden_size` | 1024 | Hidden dim for state/action MLPs |
| `input_embedding_dim` | 1536 | Input dim to DiT (32 heads × 48 head_dim) |

### DiT

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 32 | Transformer layers |
| `num_attention_heads` | 32 | Attention heads |
| `attention_head_dim` | 48 | Per-head dimension |
| `norm_type` | `ada_norm` | Adaptive LayerNorm (conditioned on timestep) |
| `dropout` | 0.2 | Dropout rate |
| `attend_text_every_n_blocks` | 2 | Cross-attention frequency to VL features |

### Demo Encoder

| Parameter | Default | Description |
|-----------|---------|-------------|
| `demo_encoder_d_model` | 768 | Demo encoder hidden dimension |
| `num_task_tokens` | 16 | Number of output task tokens from Perceiver |
| `demo_temporal_layers` | 4 | Temporal transformer depth |
| `demo_resampler_layers` | 2 | Perceiver resampler depth |
| `demo_nhead` | 8 | Attention heads in demo encoder |
| `demo_dim_feedforward` | 2048 | FFN hidden dim in demo encoder |
| `max_demo_frames` | 64 | Maximum input frames |

### Flow Matching

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inference_timesteps` | 4 | Denoising steps at inference |
| `noise_beta_alpha` | 1.5 | Beta distribution α for time sampling |
| `noise_beta_beta` | 1.0 | Beta distribution β for time sampling |
| `noise_s` | 0.999 | Time scaling factor |
| `num_timestep_buckets` | 1000 | Discretization bins for timestep embedding |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `state_dropout_prob` | 0.0 | Probability of dropping state input |
| `max_num_embodiments` | 32 | Maximum supported robot types |
| `use_video_alignment_loss` | True | Enable contrastive alignment loss |
| `alignment_loss_weight` | 0.1 | Weight of alignment loss |

---

## 10. File Map

```
gr00t/model/demo_conditioned/
├── __init__.py              — Package exports (DCGr00t, DemoEncoder, DCPolicy)
├── dc_gr00t.py              — Main model: DCGr00tConfig, TaskCrossAttention,
│                               DCGr00tActionHead, DCGr00t
├── demo_encoder.py          — DemoEncoder (SigLIP + Temporal Transformer +
│                               Perceiver Resampler), VideoAlignmentLoss
└── dc_policy.py             — DCPolicy (inference wrapper), DCPolicyServer (REST API)

scripts/demo_conditioned/
└── train_dc_groot.py        — Training: DCDataset, DCDataCollator, DCTrainer, main()

gr00t/model/modules/
├── eagle_backbone.py        — Eagle VLM wrapper (Cosmos-Reason 2B)
├── dit.py                   — Diffusion Transformer (DiT, AlternateVLDiT)
└── embodiment_conditioned_mlp.py — CategorySpecificMLP, MultiEmbodimentActionEncoder

inference_demo.py            — End-to-end inference test on Unitree G1
```

---

## Summary

DC-GR00T is a **video-conditioned robot policy** that:

1. **Watches** a demo video through a SigLIP→Temporal Transformer→Perceiver Resampler pipeline to extract 16 task tokens
2. **Observes** the world through an Eagle VLM backbone processing the robot's ego camera
3. **Fuses** demo understanding with current observation via cross-attention
4. **Generates** action trajectories using a 32-layer Diffusion Transformer with flow matching (4 denoising steps)
5. **Executes** in closed-loop, re-planning at every timestep using fresh observations

The entire system can be fine-tuned on consumer GPUs (RTX 5090 32GB) using LoRA, with only 0.22% of parameters trainable, achieving stable training in under 2 hours.
