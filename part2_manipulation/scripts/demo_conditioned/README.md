# DC-GR00T: Demo-Conditioned GR00T for Unitree G1

**Watch a demo video → Robot intelligently executes the task**

DC-GR00T enables your Unitree G1 robot to learn tasks from demonstration videos (human hands, other robots, or Cosmos-generated videos) and execute them using closed-loop control with its own camera.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Data Requirements](#data-requirements)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Training](#training)
5. [Inference & Deployment](#inference--deployment)
6. [Tips & Troubleshooting](#tips--troubleshooting)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                          │
├─────────────────────────────────────────────────────────────────┤
│  Demo Video ──→ [Demo Encoder] ──→ Task Embedding               │
│       +                                  ↓                      │
│  Robot Execution ──→ [GR00T Backbone] + [Cross-Attention]       │
│       +                                  ↓                      │
│  Actions ←────────── [DiT Action Head] ←─┘                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Show Cosmos video (once) → Model understands "what to do"   │
│  2. Robot uses OWN camera → Sees actual environment             │
│  3. Generates actions → Adapts to real object positions         │
│  4. Closed-loop control → Corrects errors in real-time          │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: The demo provides TASK UNDERSTANDING, not exact trajectories. The robot executes intelligently, not blindly.

---

## Data Requirements

### Minimum Data

| Data Type | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Robot episodes | **500** | **2,000+** | Your G1 doing various manipulation tasks |
| Tasks/skills | **5** | **20+** | Different manipulation primitives |
| Episodes per task | **50** | **100+** | Variety in object positions, lighting |

### What You Need

```
Your Data (LeRobot format):
├── videos/
│   └── observation.images.ego_view/
│       ├── episode_000000.mp4    # Robot's ego-view camera
│       ├── episode_000001.mp4
│       └── ...
├── data/
│   ├── episode_000000.parquet    # Actions & states
│   ├── episode_000001.parquet
│   └── ...
└── meta/
    ├── info.json
    ├── episodes.jsonl
    └── tasks.jsonl
```

### Data Quality Checklist

- [ ] **Consistent camera position** - Ego-view from robot's perspective
- [ ] **Good lighting** - Avoid shadows, reflections
- [ ] **Diverse object positions** - Don't always place objects in same spot
- [ ] **Clean demonstrations** - Remove failed/messy episodes
- [ ] **Task labels** - Each episode should have task description

### Cross-Embodiment Data (Optional but Recommended)

For better generalization to Cosmos/human demos:

| Source | Episodes | Purpose |
|--------|----------|---------|
| Human hand demos | 100-500 | Learn task structure from humans |
| Different viewpoints | 200+ | Learn viewpoint invariance |
| Cosmos videos | 100+ | Learn to interpret generated videos |

---

## Step-by-Step Guide

### Step 1: Prepare Your Environment

```bash
# Navigate to project
cd Isaac-GR00T

# Install dependencies
pip install torch torchvision transformers accelerate safetensors
pip install opencv-python decord flask
pip install wandb  # For training logs (optional)

# Verify installation
python -c "from gr00t.model.demo_conditioned import DCGr00t; print('OK')"
```

### Step 2: Organize Your Data

**Option A: You have LeRobot format data (1000 episodes)**

```bash
# Convert to DC-GR00T format (self-demo mode)
python scripts/demo_conditioned/convert_lerobot_to_dc.py \
    --input_dir /path/to/your/lerobot_dataset \
    --output_dir ./data/dc_dataset \
    --mode self_demo \
    --num_augmentations 3
```

This creates ~3000 training pairs where each episode's video is both demo and execution.

**Option B: You have separate demo videos**

```bash
# Prepare with external demos
python scripts/demo_conditioned/prepare_dc_dataset.py \
    --robot_data /path/to/lerobot_dataset \
    --demo_videos /path/to/demo_videos \
    --output_dir ./data/dc_dataset
```

**Option C: You have human hand demos**

```bash
# Cross-embodiment with human demos
python scripts/demo_conditioned/prepare_dc_dataset.py \
    --robot_data /path/to/lerobot_dataset \
    --human_demos /path/to/human_videos \
    --output_dir ./data/dc_dataset \
    --mode cross_embodiment
```

### Step 3: Verify Data

```bash
# Check dataset structure
ls ./data/dc_dataset/
# Should show: videos/, data/, episodes.jsonl, info.json

# Check episode count
wc -l ./data/dc_dataset/episodes.jsonl
# Should show number of training episodes

# Preview an episode
python -c "
import json
with open('./data/dc_dataset/episodes.jsonl') as f:
    ep = json.loads(f.readline())
    print(json.dumps(ep, indent=2))
"
```

### Step 4: Train DC-GR00T

**Basic Training (Single GPU)**

```bash
python scripts/demo_conditioned/train_dc_groot.py \
    --dataset_path ./data/dc_dataset \
    --output_dir ./checkpoints/dc_groot \
    --pretrained_groot nvidia/GR00T-N1.6-3B \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4
```

**Multi-GPU Training**

```bash
accelerate launch --num_processes 4 \
    scripts/demo_conditioned/train_dc_groot.py \
    --dataset_path ./data/dc_dataset \
    --output_dir ./checkpoints/dc_groot \
    --pretrained_groot nvidia/GR00T-N1.6-3B \
    --num_epochs 50 \
    --batch_size 16
```

**With Wandb Logging**

```bash
wandb login  # First time only

python scripts/demo_conditioned/train_dc_groot.py \
    --dataset_path ./data/dc_dataset \
    --output_dir ./checkpoints/dc_groot \
    --pretrained_groot nvidia/GR00T-N1.6-3B \
    --num_epochs 50 \
    --wandb_project dc_groot_g1
```

### Training Time Estimates

| Episodes | GPU | Batch Size | Time (50 epochs) |
|----------|-----|------------|------------------|
| 1,000 | 1x RTX 4090 | 4 | ~8 hours |
| 1,000 | 4x RTX 4090 | 16 | ~2 hours |
| 3,000 | 1x RTX 4090 | 4 | ~24 hours |
| 3,000 | 4x RTX 4090 | 16 | ~6 hours |

### Step 5: Test Inference

**Quick Test (Offline)**

```bash
# Test on a saved video
python scripts/demo_conditioned/run_dc_inference.py \
    --checkpoint ./checkpoints/dc_groot/final \
    --demo_video ./demos/pick_cup.mp4 \
    --demo_type robot \
    --test_video ./test/observation.mp4 \
    --output_video ./test/output_with_actions.mp4
```

**Live Camera Test**

```bash
python scripts/demo_conditioned/run_dc_inference.py \
    --checkpoint ./checkpoints/dc_groot/final \
    --demo_video ./demos/pick_cup.mp4 \
    --demo_type robot \
    --camera_id 0
```

### Step 6: Deploy on Robot

```bash
python scripts/demo_conditioned/deploy_dc_groot_g1.py \
    --checkpoint ./checkpoints/dc_groot/final \
    --demo_video ./cosmos_videos/new_task.mp4 \
    --demo_type cosmos \
    --duration 60 \
    --control_freq 30
```

---

## Training

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | required | Path to DC dataset |
| `--output_dir` | required | Where to save checkpoints |
| `--pretrained_groot` | `nvidia/GR00T-N1.6-3B` | Base GR00T model |
| `--num_epochs` | 50 | Training epochs |
| `--batch_size` | 4 | Per-GPU batch size |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--warmup_steps` | 500 | LR warmup steps |
| `--alignment_weight` | 0.1 | Cross-embodiment loss weight |
| `--freeze_backbone` | True | Freeze vision backbone |
| `--wandb_project` | None | Wandb project name |

### Training Stages (Recommended)

**Stage 1: Self-Demo (Required)**
```bash
# Train on robot's own demos first
python train_dc_groot.py \
    --dataset_path ./data/self_demo \
    --output_dir ./checkpoints/stage1 \
    --num_epochs 30
```

**Stage 2: Cross-Embodiment (If you have human demos)**
```bash
# Fine-tune with human demos
python train_dc_groot.py \
    --dataset_path ./data/cross_embodiment \
    --output_dir ./checkpoints/stage2 \
    --resume_from ./checkpoints/stage1/final \
    --alignment_weight 0.2 \
    --num_epochs 20
```

**Stage 3: Cosmos Adaptation (If you have Cosmos videos)**
```bash
# Adapt to Cosmos-generated videos
python train_dc_groot.py \
    --dataset_path ./data/cosmos_augmented \
    --output_dir ./checkpoints/stage3 \
    --resume_from ./checkpoints/stage2/final \
    --num_epochs 10
```

### Monitoring Training

```python
# Check loss curves in wandb or tensorboard
# Good signs:
# - Action loss decreasing steadily
# - Alignment loss stable or decreasing
# - No sudden spikes

# Bad signs:
# - Loss plateaus early → increase data or reduce LR
# - Loss oscillates → reduce LR or increase batch size
# - NaN loss → check data preprocessing
```

---

## Inference & Deployment

### Using DC-GR00T in Your Code

```python
from gr00t.model.demo_conditioned import DCPolicy

# Load model
policy = DCPolicy("./checkpoints/dc_groot/final", device="cuda:0")

# Set demo (do this ONCE per task)
policy.set_demo(
    demo_path="./cosmos_videos/pick_object.mp4",
    demo_type="cosmos"  # or "human", "robot", "own"
)

# Control loop
while running:
    # Get current observation from robot's camera
    frame = camera.get_frame()  # RGB numpy array [H, W, 3]

    # Get current robot state
    state = robot.get_joint_positions()  # numpy array [29]

    # Get action
    action = policy.get_action(frame, state)
    # action["left_arm"]  - [horizon, 6]
    # action["right_arm"] - [horizon, 6]
    # action["left_hand"] - [horizon, 6]
    # action["right_hand"]- [horizon, 6]

    # Execute first action
    robot.set_joint_positions(action["action"][0])
```

### REST API Server

```bash
# Start server
python run_dc_inference.py \
    --checkpoint ./checkpoints/dc_groot/final \
    --server \
    --port 8080
```

```python
# Client usage
import requests
import base64

# Set demo
requests.post("http://localhost:8080/set_demo", json={
    "demo_path": "/path/to/demo.mp4",
    "demo_type": "cosmos"
})

# Get action
with open("observation.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8080/get_action", json={
    "observation_base64": img_b64,
    "state": [0.0] * 29
})
action = response.json()["action"]
```

---

## Tips & Troubleshooting

### Data Collection Tips

1. **Vary object positions** - Don't always place objects in the same spot
2. **Vary lighting** - Collect some data with different lighting conditions
3. **Include failures** - Some recovery behaviors help generalization
4. **Clean transitions** - Remove frames where robot is resetting

### Training Tips

1. **Start small** - Train on 500 episodes first, verify it works, then scale
2. **Monitor validation loss** - Use 10% holdout to detect overfitting
3. **Checkpoint frequently** - Save every 5 epochs
4. **Use mixed precision** - Add `--fp16` for faster training

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Robot doesn't move | Actions near zero | Check action normalization in data |
| Jerky movements | Action discontinuity | Increase `action_execution_steps` |
| Ignores demo | Poor task encoding | Train longer, add more diverse demos |
| Wrong task | Demo too similar to another | Add task descriptions, train longer |
| Slow inference | Large model | Use TensorRT, reduce action horizon |

### Performance Benchmarks

| Metric | Target | Notes |
|--------|--------|-------|
| Inference time | <50ms | On RTX 4090 |
| Control frequency | 20-30 Hz | Depends on task |
| Success rate (seen tasks) | >80% | After 50 epochs |
| Success rate (new Cosmos) | >50% | With cross-embodiment training |

---

## File Structure

```
Isaac-GR00T/
├── gr00t/
│   ├── model/
│   │   └── demo_conditioned/
│   │       ├── __init__.py
│   │       ├── demo_encoder.py      # Temporal transformer, perceiver
│   │       ├── dc_gr00t.py          # Main model with cross-attention
│   │       └── dc_policy.py         # Easy-to-use policy wrapper
│   ├── configs/
│   │   └── data/
│   │       └── embodiment_configs.py  # DC_MODALITY_CONFIGS
│   └── data/
│       └── embodiment_tags.py         # UNITREE_G1_DC
│
└── scripts/
    └── demo_conditioned/
        ├── README.md                  # This file
        ├── train_dc_groot.py          # Training script
        ├── prepare_dc_dataset.py      # Dataset preparation
        ├── convert_lerobot_to_dc.py   # LeRobot conversion
        ├── run_dc_inference.py        # Inference script
        └── deploy_dc_groot_g1.py      # G1 deployment
```

---

## Citation

If you use DC-GR00T in your research:

```bibtex
@misc{dc_groot_2024,
  title={DC-GR00T: Demo-Conditioned GR00T for Cross-Embodiment Robot Learning},
  author={Your Name},
  year={2024},
  note={Built on NVIDIA GR00T and Isaac-GR00T}
}
```

---

## Support

- Issues: Check troubleshooting section above
- Bugs: Open issue in repository
- Questions: Contact maintainer

**Good luck with your G1 robot!** 🤖
