# GENESIS Setup Guide

## Prerequisites

- Linux (Ubuntu 22.04 recommended)
- NVIDIA GPU with CUDA 12.x
- Anaconda or Miniconda
- ~50 GB free disk space

---

## Step 1: Clone the repository

```bash
git clone https://github.com/jeffrinsam/GENESIS.git
cd GENESIS
```

---

## Step 2: Create conda environments

```bash
bash scripts/setup_environments.sh
```

This creates three environments:

| Environment | Purpose | Activate with |
|-------------|---------|---------------|
| `genesis-generation` | Part 1 — video generation | `conda activate genesis-generation` |
| `genesis-navigation` | Part 2a — FlowDiT navigation | `conda activate genesis-navigation` |
| `genesis-simulation` | Simulator — Isaac Sim eval | `conda activate genesis-simulation` |

---

## Step 3: Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
# Required for Part 1 with Claude optimizer
ANTHROPIC_API_KEY=sk-ant-...

# Required for Part 1 video generation
WAN_ROOT=/path/to/Wan2.2
COSMOS_ROOT=/path/to/cosmos-predict2.5
COSMOS_REASON2_ROOT=/path/to/cosmos-reason2
QWEN_ROOT=/path/to/Qwen3-VL

# Required for Simulator
ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh
DIGITALTWIN_DIR=/path/to/Digitaltwin
```

---

## Step 4: Download checkpoints

```bash
bash scripts/download_checkpoints.sh
```

---

## Step 5: Set up upstream models (Part 1)

Part 1 depends on external video generation models. Clone them separately:

### WAN 2.2 (navigation videos)
```bash
git clone https://github.com/Wan-AI/Wan2.2.git /path/to/Wan2.2
# Follow WAN 2.2 README for model weight download
# Set: export WAN_ROOT=/path/to/Wan2.2
```

### Cosmos Predict 2.5 (manipulation videos)
```bash
git clone https://github.com/NVIDIA/cosmos-predict2.git /path/to/cosmos-predict2.5
# Follow NVIDIA README for weights
# Set: export COSMOS_ROOT=/path/to/cosmos-predict2.5
```

### Qwen3-VL-2B (prompt expansion)
```bash
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir /path/to/Qwen3-VL
# Set: export QWEN_ROOT=/path/to/Qwen3-VL
```

---

## Step 6: Isaac Sim (Simulator only)

Isaac Sim 5.1+ is required for the simulator validation pipeline. It is not pip-installable.

1. Download from [NVIDIA Omniverse](https://developer.nvidia.com/isaac-sim)
2. Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/)
3. Set `ISAAC_SIM_PYTHON` in `.env` to your `python.sh` path

The Simulator can be used **without** Isaac Sim for offline trajectory analysis (just `predict_all_actions.py`).

---

## Verify installation

```bash
# Part 2a — quick navigation inference test
conda activate genesis-navigation
bash scripts/quick_start.sh
```

---

## Troubleshooting

**`ANTHROPIC_API_KEY not set`** — export it or add to `.env`

**`conda not found`** — install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

**`CUDA out of memory`** — see VRAM requirements in README.md; Cosmos 2.5 needs ~28 GB

**`module not found: wan`** — check `WAN_ROOT` is set and WAN 2.2 is cloned

**Isaac Sim not launching** — check `ISAAC_SIM_PYTHON` points to the correct `python.sh`
