# AgentLLM: Self-Tuning Robotics Video Generation System

## 🎯 What It Does

**AgentLLM** is an AI-powered system that generates high-quality robotics videos from simple text descriptions and images. It combines multiple AI models in an intelligent pipeline with **self-tuning optimization** to progressively improve video quality.

### Core Capability

```
Input:
  - Simple prompt: "Humanoid picks up bottle"
  - Workspace image: [photo of environment]

Output:
  - High-quality 5-second video
  - Physics-realistic motion
  - Task-accurate behavior
  - Validation scores + analysis
```

---

## 🏗️ System Architecture

### 1. **Prompt Enhancement** (Qwen3-VL)
- Takes simple user prompts
- Extends them with physics, kinematics, visual details
- Example:
  - Input: `"Drone flies forward"`
  - Output: `"Cinematic drone flight, smooth forward motion at 2m/s, maintaining 1.5m altitude, gimbal-stabilized camera with gradual descent, natural environmental lighting..."`

### 2. **Video Generation** (WAN 2.2 / Cosmos 2B/14B)
- Generates photorealistic videos
- Two model families:
  - **WAN 2.2**: Fast, optimized for navigation (drones, ground robots)
  - **Cosmos 2B/14B**: High-quality, physics-aware, for manipulation tasks

### 3. **Validation** (Cosmos-Reason2)
- Analyzes generated videos
- Scores 3 dimensions:
  - **Prompt Adherence**: Does it follow the task? (0-100)
  - **Physics Realism**: Is motion physically plausible? (0-100)
  - **Visual Quality**: Is it high-quality, consistent? (0-100)
- Provides detailed reasoning for each score

### 4. **Self-Tuning Brain** (Claude Opus 4.6) 🧠
- Reads validation feedback
- Analyzes root causes of failures
- Improves system prompts and negative prompts
- Iteratively refines until perfect (≤5 iterations)

---

## 🤖 Supported Tasks

### Navigation (Blue Category)
1. **Drone Aerial Navigation**
   - FPV drone flights
   - Outdoor/indoor environments
   - Cinematic camera work

2. **Ground Robot Navigation**
   - Wheeled/legged robots (Unitree G1)
   - Obstacle avoidance
   - Terrain traversal

### Manipulation (Purple Category)
3. **Bimanual UR3 Manipulation**
   - Dual-arm coordination
   - Pick-and-place tasks
   - Tabletop manipulation

4. **Humanoid G1 Manipulation**
   - Full-body humanoid tasks
   - Whole-body manipulation
   - Complex object interactions

---

## ✨ Key Features

### Current System (v1.0)
- ✅ **Unified Web Interface**: Single website for all 4 tasks
- ✅ **Real-Time Logs**: See generation progress live
- ✅ **QwenVL Visibility**: View and edit enhanced prompts
- ✅ **Multi-Model Support**: WAN, Cosmos 2B, Cosmos 14B
- ✅ **Automatic Validation**: Cosmos-Reason2 analysis
- ✅ **Advanced Parameters**: Full control over generation

### Self-Tuning System (v2.0 - In Development)
- 🔧 **Claude Opus Brain**: AI-driven prompt optimization
- 🔧 **Iterative Refinement**: Auto-improve over 5 iterations
- 🔧 **Multi-Objective Optimization**: Balance adherence, physics, quality
- 🔧 **Explainable Reasoning**: See why changes are made
- 🔧 **Prompt Library**: Learn from successful generations

---

## 📊 Performance

### Generation Times
| Task Type | Model | Frames | Time |
|-----------|-------|--------|------|
| Drone/Ground | WAN 2.2 | 61 | 2-4 min |
| UR3/G1 | Cosmos 2B | 77 | 3-5 min |
| UR3/G1 | Cosmos 14B | 121 | 10-12 min |

### Expected Quality (Without Self-Tuning)
- Average scores: 60-75/100
- Single-shot generation
- Manual prompt tuning needed

### Expected Quality (With Self-Tuning)
- Average scores: 80-85/100
- 75-85% success in ≤5 iterations
- Automatic optimization
- Cost: ~$2/video (Claude API)

---

## 🎓 Use Cases

### Research
- Robotics video generation studies
- Prompt optimization research
- Multi-modal AI pipelines
- Reinforcement learning from video

### Development
- Robot simulation data generation
- Training data augmentation
- Behavior visualization
- Rapid prototyping

### Education
- Teaching robotics concepts
- Demonstrating tasks visually
- Interactive learning tools
- Concept validation

---

## 🔬 Technology Stack

**AI Models:**
- **Qwen3.5-9B**: Vision-language prompt enhancement
- **WAN 2.2**: Fast video diffusion for navigation
- **Cosmos 2B/14B**: Physics-aware video generation
- **Cosmos-Reason2**: Video validation and reasoning
- **Claude Opus 4.6**: Optimization brain (v2.0)

**Framework:**
- **Backend**: Python 3.12, Flask
- **Frontend**: HTML5, Tailwind CSS, Alpine.js
- **Inference**: PyTorch, Transformers, Diffusers
- **GPU**: NVIDIA CUDA (RTX 4090 / RTX 5090)

---

## 📁 Directory Structure

```
AgentLLM/
├── README.md                    # This file (what it does)
├── THEORY.md                    # How it works (science)
├── QUICKSTART.md                # How to run (practical guide)
│
├── unified_app.py               # Main web server
├── templates/unified_index.html # Web interface
├── start_unified.sh             # Startup script
│
├── Navigation/                  # Navigation tasks
│   ├── navigation_pipeline.py
│   └── video_validator.py
│
├── Manipulation/                # Manipulation tasks
│   ├── manipulation_pipeline.py
│   └── video_validator.py
│
├── outputs/                     # Generated videos + validations
├── uploads/                     # User-uploaded images
│
└── archive/                     # Archived research docs
    ├── root/
    │   ├── RESEARCH_FINDINGS_2026.md
    │   ├── CLAUDE_OPUS_BRAIN_APPROACH.md
    │   └── SELF_TUNING_AGENT.md
    └── old_docs/
```

---

## 🚀 Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

**TL;DR:**
```bash
cd part1_generation/agentllm
./start_unified.sh
# Open http://localhost:5002
```

---

## 📚 Documentation

1. **[README.md](README.md)** ← You are here (Overview)
2. **[THEORY.md](THEORY.md)** - How it works (Science & Approach)
3. **[QUICKSTART.md](QUICKSTART.md)** - How to run (Step-by-step guide)

**Research Documents** (in archive/):
- [RESEARCH_FINDINGS_2026.md](archive/root/RESEARCH_FINDINGS_2026.md) - Latest 2026 techniques
- [CLAUDE_OPUS_BRAIN_APPROACH.md](archive/root/CLAUDE_OPUS_BRAIN_APPROACH.md) - Self-tuning design
- [SELF_TUNING_AGENT.md](archive/root/SELF_TUNING_AGENT.md) - Original RL approach

---

## 🎯 Current Status

**Version 1.0** (Production Ready ✅)
- Unified web interface working
- All 4 tasks operational
- Manual prompt engineering
- Cosmos-Reason2 validation integrated

**Version 2.0** (In Development 🔧)
- Claude Opus 4.6 brain
- Self-tuning optimization loop
- Prompt library with warm start
- Multi-agent debate optimization

---

## 🤝 Contributing

This is a research project. For questions or contributions:
- Contact: [Your contact info]
- Issues: Document in archive/
- Research: See archive/root/ for latest findings

---

## 📄 License

[Specify license]

---

## 🙏 Acknowledgments

**Models Used:**
- Qwen3-VL by Alibaba DAMO Academy
- Cosmos by NVIDIA
- Claude by Anthropic

**Built for:** Robotics Research - Video Generation Pipeline

---

**Status**: Active Development
**Last Updated**: 2026-02-07
**Version**: 1.0 (v2.0 in progress)
