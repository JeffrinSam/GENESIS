# GENESIS: Video-Conditioned Robot Learning

[![arXiv](https://img.shields.io/badge/arXiv-2605.01477-b31b1b.svg)](https://arxiv.org/abs/2605.01477)
[![arXiv](https://img.shields.io/badge/arXiv-2509.13903-b31b1b.svg)](https://arxiv.org/abs/2509.13903)
[![arXiv](https://img.shields.io/badge/arXiv-2606.13856-b31b1b.svg)](https://arxiv.org/abs/2606.13856)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-JeffrinSam-yellow.svg)](https://huggingface.co/JeffrinSam)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--8635--5379-green.svg)](https://orcid.org/0009-0000-8635-5379)

**GENESIS** is a three-part framework that uses **video as the universal task interface** for robot learning — from agentic video generation through navigation and manipulation.

> **"Action Agent: Agentic Video Generation Meets Flow-Constrained Diffusion"** — arXiv:2605.01477  
> Jeffrin Sam, Nguyen Khang, Yara Mahmoud, Miguel Altamirano Cabrera, Dzmitry Tsetserukou

> **"PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models"** — arXiv:2509.13903  
> Artem Lykov, Jeffrin Sam, Hung Khang Nguyen, Vladislav Kozlovskiy, Yara Mahmoud, Valerii Serpiva, Miguel Altamirano Cabrera, Mikhail Konenkov, Dzmitry Tsetserukou

---

## Pipeline Overview

```
Text instruction + workspace image
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Part 1 — ClaudeOpusBrain (agentic video generation)            │
│  LLM self-tuning loop: Qwen3-VL → WAN 2.2 / Cosmos 2.5        │
│  → Cosmos-Reason2 validate → Claude optimize → iterate          │
└────────────────────────────┬────────────────────────────────────┘
                             │ reference video
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│  Part 2a — Navigation   │   │  Part 2b — Manipulation         │
│  FlowDiT V2             │   │  DC-GR00T                       │
│  DINOv2 + optical flow  │   │  SigLIP + Temporal Transformer  │
│  → DiT → [vx, vy, yaw] │   │  → GR00T N1.6 DiT → 43-DOF    │
└───────────┬─────────────┘   └──────────────┬──────────────────┘
            │                                │
            └──────────────┬─────────────────┘
                           ▼
              ┌────────────────────────┐
              │  Simulator             │
              │  Isaac Sim + Isaac Lab │
              │  SR / SPL / ATE eval   │
              └────────────────────────┘
```

---

## Repository Structure

| Directory | Description | Conda env |
|-----------|-------------|-----------|
| [`part1_generation/`](part1_generation/) | ClaudeOpusBrain self-tuning video generation — [Action Agent](https://arxiv.org/abs/2605.01477) (IROS 2026) · [PhysicalAgent](https://arxiv.org/abs/2509.13903) | `genesis-generation` |
| [`part2_navigation/`](part2_navigation/) | FlowDiT V2 video-to-navigation — [Action Agent](https://arxiv.org/abs/2605.01477) | `genesis-navigation` |
| [`part2_manipulation/`](part2_manipulation/) | DC-GR00T video-to-manipulation — [PhysicalAgent](https://arxiv.org/abs/2509.13903) | `dc_groot` (see part2_manipulation/README.md) |
| [`simulator/`](simulator/) | Isaac Sim validation pipeline | `genesis-simulation` |
| [`scripts/`](scripts/) | Setup, download, quick-start helpers | — |
| [`environments/`](environments/) | Conda environment YAMLs | — |
| [`docs/`](docs/) | Architecture and setup docs | — |

---

## Quick Start

### 1. Clone and set up environments

```bash
git clone https://github.com/jeffrinsam/GENESIS.git
cd GENESIS

# Create all conda environments (~10 min)
bash scripts/setup_environments.sh

# Copy and fill in your config
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and model paths
```

### 2. Download checkpoints

```bash
bash scripts/download_checkpoints.sh
```

### 3. Part 1 — Generate a trajectory video

```bash
conda activate genesis-generation
cd part1_generation/claudeopusbrain

export ANTHROPIC_API_KEY="your-key-here"

# Single task self-tuning (~$2, ~5 iterations)
python run_self_tuning.py \
  --task "G1 humanoid picks up a bottle" \
  --task-type g1 \
  --image /path/to/workspace.jpg \
  --model opus \
  --max-iterations 5
```

### 4. Part 2a — Navigate from video

```bash
conda activate genesis-navigation
cd part2_navigation

# Quick validation
python flow_constrained_v2/test_inference.py \
  --checkpoint flow_constrained_v2/checkpoints/best.pth

# Single inference
python flow_constrained_v2/inference.py \
  --checkpoint flow_constrained_v2/checkpoints/best.pth \
  --goal_video /path/to/reference.mp4 \
  --current_obs /path/to/frame.jpg \
  --output actions.npy

# Closed-loop on real robot
python flow_constrained_v2/robot_navigation.py \
  --checkpoint flow_constrained_v2/checkpoints/best.pth \
  --goal_video /path/to/reference.mp4 \
  --camera 0 \
  --control_hz 2.0
```

### 5. Simulator evaluation

```bash
conda activate genesis-simulation
cd simulator

# Batch Mode 1 (open-loop replay)
python predict_all_actions.py --robot humanoid \
  --checkpoint ../part2_navigation/flow_constrained_v2/checkpoints/best.pth

# Mode 2 (closed-loop realtime)
python run_mode2_validation.py --robot humanoid
```

---

## Key Results

| Model | Task | Success Rate | SPL |
|-------|------|-------------|-----|
| FlowDiT V2 | Indoor navigation (3.0m threshold) | **100%** (41 tasks) | 0.89 |
| FlowDiT V2 | Indoor navigation (1.0m threshold) | 39% | 0.41 |
| DC-GR00T | G1 manipulation (LIBERO) | training in progress | — |

---

## VRAM Requirements

| Component | Minimum VRAM | Recommended |
|-----------|-------------|-------------|
| WAN 2.2 TI2V-5B (Part 1 nav) | 16 GB | 24 GB |
| Cosmos 2.5 14B (Part 1 manip) | 28 GB | 32 GB |
| FlowDiT V2 inference | 4 GB | 8 GB |
| FlowDiT V2 training | 12 GB | 24 GB |
| DC-GR00T training (LoRA, bf16) | 27 GB | 32 GB |

---

## Upstream Models

GENESIS builds on these publicly available models. Clone them separately and set paths in `.env`:

| Model | Source | Used in |
|-------|--------|---------|
| WAN 2.2 | [Wan-AI/Wan2.2](https://github.com/Wan-AI/Wan2.2) | Part 1 navigation video |
| Cosmos Predict 2.5 | [NVIDIA/cosmos-predict2](https://github.com/NVIDIA/cosmos-predict2) | Part 1 manipulation video |
| Cosmos Reason 2 | NVIDIA (private beta) | Part 1 validation |
| Qwen3-VL-2B | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Part 1 prompt expansion |
| GR00T-N1.6-3B | [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) | Part 2b base model |

---

## Citation

```bibtex
@inproceedings{sam2026actionagent,
  title     = {Action Agent: Agentic Video Generation Meets Flow-Constrained Diffusion},
  author    = {Sam, Jeffrin and Khang, Nguyen and Mahmoud, Yara and
               Altamirano Cabrera, Miguel and Tsetserukou, Dzmitry},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots
               and Systems (IROS)},
  year      = {2026},
  note      = {arXiv:2605.01477}
}

@article{lykov2025physicalagent,
  title     = {PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models},
  author    = {Lykov, Artem and Sam, Jeffrin and Nguyen, Hung Khang and Kozlovskiy, Vladislav and
               Mahmoud, Yara and Serpiva, Valerii and Altamirano Cabrera, Miguel and
               Konenkov, Mikhail and Tsetserukou, Dzmitry},
  journal   = {arXiv preprint arXiv:2509.13903},
  year      = {2025}
}

@article{sam2026outputlevel,
  title     = {Output-Level Regularization Eliminates the Seed Lottery in Single-GPU VLA Fine-Tuning},
  author    = {Sam, Jeffrin and Tsetserukou, Dzmitry},
  journal   = {arXiv preprint arXiv:2606.13856},
  year      = {2026}
}
```

---

## Related citations

```bibtex
@inproceedings{serpiva2026diffusioncinema,
  title     = {DiffusionCinema: Text-to-Aerial Cinematography},
  author    = {Serpiva, Valerii and Lykov, Artem and Sam, Jeffrin and
               Fedoseev, Aleksey and Tsetserukou, Dzmitry},
  booktitle = {Companion Proceedings of the 21st ACM/IEEE International
               Conference on Human-Robot Interaction (HRI)},
  year      = {2026},
  address   = {Edinburgh, UK},
  doi       = {10.1145/3776734.3794353}
}

@article{serpiva2026dreamtonav,
  title     = {DreamToNav: Generalizable Navigation for Robots via Generative Video Planning},
  author    = {Serpiva, Valerii and Sam, Jeffrin and Simon, Chidera and Amjad, Hajira and
               Zhura, Iana and Lykov, Artem and Tsetserukou, Dzmitry},
  journal   = {arXiv preprint arXiv:2603.06190},
  year      = {2026}
}

@inproceedings{fernando2026generativempc,
  title     = {GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance for
               Bimanual Mobile Manipulation},
  author    = {Fernando, Marcelino Julio and Altamirano Cabrera, Miguel and Sam, Jeffrin and
               Mahmoud, Yara and Gubernatorov, Konstantin and Tsetserukou, Dzmitry},
  booktitle = {2026 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2026},
  note      = {arXiv:2604.19522}
}

@article{zhura2026diffusionanything,
  title     = {DiffusionAnything: End-to-End In-context Diffusion Learning for
               Unified Navigation and Pre-Grasp Motion},
  author    = {Zhura, Iana and Mahmoud, Yara and Sam, Jeffrin and Nguyen, Hung Khang and
               Seyidov, Didar and Altamirano Cabrera, Miguel and Tsetserukou, Dzmitry},
  journal   = {arXiv preprint arXiv:2603.26322},
  year      = {2026}
}

@inproceedings{mahmoud2026safehumanoid,
  title     = {SafeHumanoid: VLM-RAG-Driven Impedance Control of Humanoid Robot},
  author    = {Mahmoud, Yara and Sam, Jeffrin and Khang, Nguyen and Fernando, Marcelino and
               Tokmurziyev, Issatay and Altamirano Cabrera, Miguel and Khan, Muhammad Haris and
               Lykov, Artem and Tsetserukou, Dzmitry},
  booktitle = {Companion Proceedings of the 21st ACM/IEEE International
               Conference on Human-Robot Interaction (HRI)},
  year      = {2026},
  doi       = {10.1145/3776734.3794539}
}

@inproceedings{nguyen2026taphri,
  title     = {TapHRI: TCN-Driven Touch Control of Collaborative Robots Using Only Embedded Robot Sensing},
  author    = {Nguyen, Hung Khang and Sam, Jeffrin and Gulyamova, Safina and
               Altamirano Cabrera, Miguel and Lykov, Artem and Tsetserukou, Dzmitry},
  booktitle = {Companion Proceedings of the 21st ACM/IEEE International
               Conference on Human-Robot Interaction (HRI)},
  year      = {2026},
  doi       = {10.1145/3776734.3794561}
}

@inproceedings{habel2026glove2uav,
  title     = {Glove2UAV: A Wearable IMU-Based Glove for Intuitive Control of UAV},
  author    = {Habel, Amir and Snegirev, Ivan and Semenyakina, Elizaveta and
               Altamirano Cabrera, Miguel and Sam, Jeffrin and Mehboob, Fawad and
               Khan, Roohan Ahmed and Mustafa, Muhammad Ahsan and Tsetserukou, Dzmitry},
  booktitle = {Companion Proceedings of the 21st ACM/IEEE International
               Conference on Human-Robot Interaction (HRI)},
  year      = {2026},
  doi       = {10.1145/3776734.3794566}
}

@inproceedings{mehboob2026dronevla,
  title     = {DroneVLA: VLA-Based Aerial Manipulation},
  author    = {Mehboob, Fawad and James, Monijesu and Habel, Amir and Sam, Jeffrin and
               Altamirano Cabrera, Miguel and Tsetserukou, Dzmitry},
  booktitle = {Companion Proceedings of the 21st ACM/IEEE International
               Conference on Human-Robot Interaction (HRI)},
  year      = {2026},
  doi       = {10.1145/3776734.3794572}
}

@inproceedings{habel2025yawsitter,
  title     = {YawSitter: Modeling and Controlling a Tail-Sitter UAV with Enhanced Yaw Control},
  author    = {Habel, Amir and Mehboob, Fawad and Sam, Jeffrin and Fortin, Clement and
               Tsetserukou, Dzmitry},
  booktitle = {2025 IEEE International Conference on Robotics and Biomimetics (ROBIO)},
  year      = {2025},
  note      = {arXiv:2510.02968}
}

@article{sautenkov2025uavcodeagents,
  title     = {UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and
               Vision-Language Reasoning},
  author    = {Sautenkov, Oleg and Yaqoot, Yasheerah and Mustafa, Muhammad Ahsan and
               Batool, Faryal and Sam, Jeffrin and Lykov, Artem and Wen, Chih-Yung and
               Tsetserukou, Dzmitry},
  journal   = {arXiv preprint arXiv:2505.07236},
  year      = {2025}
}
```

---

## Full publication list

All publications — **bold** = first author. Verified from Google Scholar (OyIR64QAAAAJ) and arXiv.

| Year | Paper | Venue | arXiv / DOI |
|------|-------|-------|-------------|
| 2026 | **[Action Agent: Agentic Video Generation Meets Flow-Constrained Diffusion](https://arxiv.org/abs/2605.01477)** | IROS 2026 | [2605.01477](https://arxiv.org/abs/2605.01477) |
| 2026 | **[Output-Level Regularization Eliminates the Seed Lottery in Single-GPU VLA Fine-Tuning](https://arxiv.org/abs/2606.13856)** | arXiv | [2606.13856](https://arxiv.org/abs/2606.13856) |
| 2026 | [GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance for Bimanual Mobile Manipulation](https://arxiv.org/abs/2604.19522) | IEEE SMC 2026 | [2604.19522](https://arxiv.org/abs/2604.19522) |
| 2026 | [DiffusionAnything: End-to-End In-context Diffusion Learning for Unified Navigation and Pre-Grasp Motion](https://arxiv.org/abs/2603.26322) | arXiv | [2603.26322](https://arxiv.org/abs/2603.26322) |
| 2026 | [DreamToNav: Generalizable Navigation for Robots via Generative Video Planning](https://arxiv.org/abs/2603.06190) | arXiv | [2603.06190](https://arxiv.org/abs/2603.06190) |
| 2026 | [TapHRI: TCN-Driven Touch Control of Collaborative Robots Using Only Embedded Robot Sensing](https://dl.acm.org/doi/10.1145/3776734.3794561) | HRI 2026 | [10.1145/3776734.3794561](https://dl.acm.org/doi/10.1145/3776734.3794561) |
| 2026 | [SafeHumanoid: VLM-RAG-Driven Impedance Control of Humanoid Robot](https://dl.acm.org/doi/10.1145/3776734.3794539) | HRI 2026 | [10.1145/3776734.3794539](https://dl.acm.org/doi/10.1145/3776734.3794539) · [arXiv:2511.23300](https://arxiv.org/abs/2511.23300) |
| 2026 | [DiffusionCinema: Text-to-Aerial Cinematography](https://dl.acm.org/doi/10.1145/3776734.3794353) | HRI 2026, Edinburgh | [10.1145/3776734.3794353](https://dl.acm.org/doi/10.1145/3776734.3794353) |
| 2026 | [Glove2UAV: A Wearable IMU-Based Glove for Intuitive Control of UAV](https://dl.acm.org/doi/10.1145/3776734.3794566) | HRI 2026 | [10.1145/3776734.3794566](https://dl.acm.org/doi/10.1145/3776734.3794566) · [arXiv:2601.15775](https://arxiv.org/abs/2601.15775) |
| 2026 | [DroneVLA: VLA-Based Aerial Manipulation](https://dl.acm.org/doi/10.1145/3776734.3794572) | HRI 2026 | [10.1145/3776734.3794572](https://dl.acm.org/doi/10.1145/3776734.3794572) · [arXiv:2601.13809](https://arxiv.org/abs/2601.13809) |
| 2025 | [PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models](https://arxiv.org/abs/2509.13903) | arXiv | [2509.13903](https://arxiv.org/abs/2509.13903) |
| 2025 | [YawSitter: Modeling and Controlling a Tail-Sitter UAV with Enhanced Yaw Control](https://arxiv.org/abs/2510.02968) | IEEE ROBIO 2025 | [arXiv:2510.02968](https://arxiv.org/abs/2510.02968) |
| 2025 | [UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning](https://arxiv.org/abs/2505.07236) | arXiv | [2505.07236](https://arxiv.org/abs/2505.07236) |

---

## Works that use GENESIS

If your work builds on GENESIS, feel free to open a PR to add it here.

| Paper | Venue | Component used |
|-------|-------|----------------|
| [DiffusionCinema: Text-to-Aerial Cinematography](https://dl.acm.org/doi/10.1145/3776734.3794353) | HRI 2026, Edinburgh | Part 1 — video generation |
| [DreamToNav: Generalizable Navigation via Generative Video Planning](https://arxiv.org/abs/2603.06190) | arXiv:2603.06190 | Part 1 — video generation |

---

## Open to collaborate

I am building toward **general-purpose humanoid robot intelligence** — AGI grounded in physical action and embodied understanding. GENESIS is one step: video as the universal task interface across embodiments. The bigger challenges are still open:

- **World models at scale** — learning environment dynamics from simulation and real-world data that generalise across tasks and embodiments
- **Unified action representations** — a single policy backbone that covers manipulation, navigation, and whole-body loco-manipulation
- **Continual learning** — robots that improve from in-deployment experience without catastrophic forgetting
- **Embodied reasoning** — closing the loop between language understanding, visual perception, and physical action in real time at robot speeds

If you work on any of these problems — or want to — reach out. I am particularly interested in collaborations that bridge **simulation** (Isaac Sim, MuJoCo), **foundation models** (VLMs, diffusion policies), and **physical deployment** on humanoid platforms (Unitree G1, H1, Figure, Agility Digit).

| | |
|---|---|
| **LinkedIn** | [jeffrin-s-a-m](https://www.linkedin.com/in/jeffrin-s-a-m/) |
| **HuggingFace** | [JeffrinSam](https://huggingface.co/JeffrinSam) |
| **Google Scholar** | [OyIR64QAAAAJ](https://scholar.google.com/citations?user=OyIR64QAAAAJ) |
| **ORCID** | [0009-0000-8635-5379](https://orcid.org/0009-0000-8635-5379) |

---

## Author

**Jeffrin Sam** — Skoltech, Moscow  
[![HuggingFace](https://img.shields.io/badge/🤗-JeffrinSam-yellow.svg)](https://huggingface.co/JeffrinSam)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0000--8635--5379-green.svg)](https://orcid.org/0009-0000-8635-5379)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jeffrin--s--a--m-blue.svg)](https://www.linkedin.com/in/jeffrin-s-a-m/)
[![Scholar](https://img.shields.io/badge/Google_Scholar-OyIR64QAAAAJ-lightgrey.svg)](https://scholar.google.com/citations?user=OyIR64QAAAAJ)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

This project uses upstream models with their own licenses. Check each model's license before commercial use.
