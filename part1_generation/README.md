# Part 1 — Agentic Video Generation

This module implements **ClaudeOpusBrain**: an LLM-in-the-loop pipeline that generates, validates, and iteratively refines goal-conditioned robot videos. The videos are used downstream as navigation targets (Part 2a) and manipulation demonstrations (Part 2b).

**Papers**:
- [Action Agent: Agentic Video Generation Meets Flow-Constrained Diffusion](https://arxiv.org/abs/2605.01477) — IROS 2026 (navigation videos → Part 2a FlowDiT)
- [PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models](https://arxiv.org/abs/2509.13903) — arXiv (manipulation videos → Part 2b DC-GR00T)

**Conda environment**: `genesis-generation`

## Contents

| Directory | Purpose |
|-----------|---------|
| [`claudeopusbrain/`](claudeopusbrain/) | Self-tuning brain: Claude API optimizer + WAN/Cosmos generator + Cosmos-Reason2 validator |
| [`agentllm/`](agentllm/) | Unified 4-task web UI (port 5002) for drone, ground, manipulation, and humanoid nav tasks |

## Quick Start

```bash
conda activate genesis-generation
cd part1_generation/claudeopusbrain

export ANTHROPIC_API_KEY="sk-ant-..."

# Single task, self-tuning (~$2, ~5 iterations)
python run_self_tuning.py --task "G1 picks up bottle" --task-type g1 \
  --image workspace.jpg --model opus --max-iterations 5

# Batch experiments (100 tasks, checkpointed, resumable)
python run_batch_experiments.py --tasks tasks.json --model opus \
  --max-iterations 5 --cost-budget 200
```

See [`claudeopusbrain/README.md`](claudeopusbrain/README.md) and [`agentllm/README.md`](agentllm/README.md) for full setup and usage.

## External Dependencies

The generation pipeline requires these upstream model repos (not included — install separately):

| Model | Env var | Used for |
|-------|---------|---------|
| WAN 2.2 | `WAN_ROOT` | Navigation video generation (T2V/TI2V) |
| Cosmos Predict 2.5 | `COSMOS_ROOT` | Manipulation video generation (14B) |
| Cosmos-Reason2 | `COSMOS_REASON2_ROOT` | Video quality validation |
| Qwen3-VL-2B | `QWEN_ROOT` | Prompt expansion |

See [docs/SETUP.md](../docs/SETUP.md) for installation instructions.
