# AgentLLM - Unified Robotics Video Generation Interface

**Engineered by Jeffrin Sam (jeffrinsam.a@gmail.com)**

Part of: Self-Tuning Robotics Video Generation System

## Overview

Unified web interface (port 5002) for robotics video generation. Supports 4 tasks: Drone navigation, Ground Robot navigation, UR3 manipulation, Humanoid G1 manipulation.

## Prerequisites

- NVIDIA GPU with 32GB VRAM (RTX 5090 recommended)
- CUDA 12.x
- conda (Miniconda/Anaconda)
- uv (for venv-based components)
- Python 3.12+

## Environment Setup

This project uses multiple environments. Each component loads sequentially to stay within VRAM limits.

### 1. wan2.2 conda env (Flask server, WAN generation, Qwen extender)

```bash
# This should already exist from the WAN 2.2 setup
conda activate wan2.2
pip install flask pillow
```

### 2. Qwen3-VL venv (prompt enhancement)

```bash
cd $QWEN_ROOT
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "transformers>=5.0.0" accelerate pillow
```

### 3. cosmos-predict2.5 venv (Cosmos video generation)

```bash
cd $COSMOS_ROOT
uv sync --extra cu128
```

### 4. cosmos-reason2 venv (video validation)

```bash
cd $COSMOS_REASON2_ROOT
uv sync --extra cu128
```

## Running

```bash
cd part1_generation/agentllm
./start_unified.sh
# Open http://localhost:5002
```

Or manually:

```bash
conda activate wan2.2
python3 unified_app.py
```

## Notes

- Sequential model loading: Load -> Use -> Unload -> Repeat (peak 16GB VRAM)
- WAN generation: 2-4 min per video
- Cosmos generation: 3-12 min per video
- Validation auto-fallback to 70% if Cosmos-Reason2 fails
