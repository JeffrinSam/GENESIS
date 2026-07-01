# Claudeopusbrain - Self-Tuning Optimization Brain

**Engineered by Jeffrin Sam (jeffrinsam.a@gmail.com)**

Part of: Self-Tuning Robotics Video Generation System

## Overview

Claude Opus 4.6 serves as the optimization brain for iterative prompt refinement. Over 5 iterations, it analyzes validation scores and refines prompts to improve video quality from ~60-75 to ~80-85/100.

## Prerequisites

- Anthropic API key (Claude Opus 4.6 access)
- Python 3.12+
- AgentLLM pipeline functional (called via subprocess)
- NVIDIA GPU with 32GB VRAM (RTX 5090 recommended) for the generation pipeline

## Environment Setup

```bash
# Create environment
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python anthropic httpx

# Set API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

**Important**: Never commit `.env` - it contains your API key.

## Running

```bash
source .venv/bin/activate
python3 claude_brain.py --task "Humanoid picks up bottle" --iterations 5
```

## How It Works

1. Takes initial prompt + task description
2. Calls AgentLLM pipeline to generate and validate video
3. Claude analyzes scores (adherence, physics, quality)
4. Generates improved prompt with reasoning
5. Repeats for up to 5 iterations or until scores exceed threshold

## Cost

- ~$2 per video (5 iterations)
- 75-85% success rate in reaching target scores
