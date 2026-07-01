# AgentLLM: Theory and Scientific Approach

## 🧬 Core Concept

AgentLLM implements a **multi-stage AI pipeline** that transforms simple user commands into high-quality, physics-realistic robotics videos through iterative self-improvement.

---

## 🏗️ Pipeline Architecture

### Stage 1: Prompt Enhancement (Qwen3-VL)

**Problem**: User prompts are too simple for video generation
- Input: `"Drone flies forward"`
- Missing: Physics, kinematics, visual details, constraints

**Solution**: Vision-Language Model (VLM) enhancement

```
Qwen3-VL Process:
1. Analyze input image (workspace/environment)
2. Understand task semantics
3. Add physics constraints (velocity, acceleration, forces)
4. Add visual details (camera work, lighting, composition)
5. Add motion planning (trajectories, waypoints, timing)

Output: Detailed 200-500 word prompt
```

**Scientific Basis:**
- **Vision-Language Models (VLMs)**: Combine vision encoders + language models
- **Multimodal Fusion**: Image features + text embeddings → enhanced understanding
- **Task-Specific System Prompts**: Guide model toward robotics-specific knowledge

**Key Innovation:**
Different system prompts for different tasks:
- Navigation → Emphasize path planning, obstacle avoidance
- Manipulation → Emphasize grasping, kinematics, contact dynamics

---

### Stage 2: Video Generation (Diffusion Models)

**Problem**: Generate realistic robotics videos from text prompts

**Solution**: Physics-aware video diffusion models

#### WAN 2.2 (World Animator Network)
- **For**: Navigation tasks (drones, ground robots)
- **Architecture**: Latent diffusion in spatiotemporal space
- **Strength**: Fast, smooth motion, good for camera movement
- **Weakness**: Less physics-aware

#### Cosmos 2B/14B (NVIDIA)
- **For**: Manipulation tasks (UR3, humanoid)
- **Architecture**: Transformer-based video diffusion with physics priors
- **Strength**: Physics-realistic, understands object interactions
- **Weakness**: Slower, more compute-intensive

**Diffusion Process:**
```
1. Start with noise: ε ~ N(0, I)
2. Iterative denoising: x_{t-1} = f(x_t, prompt, t)
3. Repeat T steps (typically 50)
4. Final: Clean video x_0
```

**Negative Prompts:**
- Guide model AWAY from undesirable outputs
- Examples: "collision, blurry, teleportation, physics violations"
- Critical for quality control

**Scientific Basis:**
- **Denoising Diffusion Probabilistic Models (DDPM)**: Reverse diffusion process
- **Latent Diffusion**: Compress video to latent space for efficiency
- **Classifier-Free Guidance**: Steer generation toward prompt without separate classifier
- **Physics-Informed Priors**: Cosmos models trained on physics simulation data

---

### Stage 3: Validation (Cosmos-Reason2)

**Problem**: How do we know if the video is good?

**Solution**: Multi-dimensional video reasoning model

**Cosmos-Reason2 Architecture:**
- Vision encoder (video → features)
- Multimodal transformer (video + text reasoning)
- Outputs: Scores + natural language analysis

**Three Validation Dimensions:**

1. **Prompt Adherence (0-100)**
   - Does the video follow the task description?
   - Are all elements from the prompt present?
   - Timing, sequence, objects correct?

2. **Physics Realism (0-100)**
   - Are motions physically plausible?
   - Collision detection
   - Gravity, inertia, friction respected?
   - Joint limits, workspace bounds violated?

3. **Visual Quality (0-100)**
   - Sharpness, consistency, lighting
   - Temporal coherence (no flickering)
   - Natural motion (no jitter, jumps)
   - Professional cinematography

**Output Format:**
```json
{
  "components": [
    {
      "name": "Prompt Adherence",
      "score": 75,
      "analysis": "Task mostly followed, but arm selection unclear..."
    },
    {
      "name": "Physics Realism",
      "score": 65,
      "analysis": "Collision detected at frame 45 between left arm and torso..."
    },
    {
      "name": "Visual Quality",
      "score": 82,
      "analysis": "High visual fidelity, smooth motion, excellent lighting..."
    }
  ],
  "pass": false,
  "average": 74.0
}
```

**Scientific Basis:**
- **Video Question Answering (VideoQA)**: Reason about video content
- **Multi-Task Learning**: Train on adherence, physics, quality simultaneously
- **Chain-of-Thought Reasoning**: Generate explanatory text alongside scores
- **Multimodal Reward Models**: Map (video, text) → scalar rewards per dimension

---

## 🧠 Self-Tuning Loop (v2.0 Architecture)

### The Challenge

Single-shot generation often fails:
- Validation scores: 60-75/100
- Physics violations common
- Task adherence issues
- Manual prompt tuning tedious

### The Solution: Iterative Self-Improvement

```
Iteration 1:
  User prompt → Qwen3-VL (default system prompt)
              → Video model (default negative prompt)
              → Video (score: 65/100)
              → Cosmos validation (detailed feedback)
              → Claude Opus 4.6 brain ← 🧠

Claude analyzes:
  - What went wrong? (collision detected)
  - Why? (trajectory planning didn't account for self-collision)
  - How to fix? (add constraint to system prompt, update negative prompt)

Iteration 2:
  User prompt → Qwen3-VL (IMPROVED system prompt)
              → Video model (IMPROVED negative prompt)
              → Video (score: 78/100)
              → Cosmos validation
              → Claude Opus 4.6 brain

Claude analyzes:
  - Collision fixed! ✓
  - But now motion less natural...
  - Balance constraint vs fluidity

Iteration 3:
  ... (refined prompts)
  → Video (score: 83/100) ✓ SUCCESS
```

---

## 🎯 Claude Opus 4.6 as Optimization Brain

### Why Claude?

1. **Superior Reasoning**
   - Best model for complex multi-objective optimization (Feb 2026)
   - Understands physics, robotics, video quality
   - Natural language reasoning about tradeoffs

2. **Long Context (200K tokens)**
   - Can see ALL previous iterations
   - Learns patterns across attempts
   - Avoids repeating mistakes

3. **Constitutional AI**
   - Self-critique built-in
   - Iterative refinement of own suggestions
   - Considers second-order effects

4. **Explainable**
   - Shows reasoning for every change
   - Debuggable and interpretable
   - Builds trust and understanding

### Optimization Process

**Input to Claude:**
```
{
  "task": "Humanoid picks up bottle",
  "iteration": 2,
  "current_system_prompt": "...",
  "current_negative_prompt": "...",
  "validation": {
    "scores": {
      "adherence": 75,
      "physics": 65,  ← Bottleneck
      "quality": 82
    },
    "analysis": {
      "physics": "Collision at frame 45..."
    }
  },
  "history": [
    {"iteration": 1, "scores": [65, 60, 70], ...}
  ]
}
```

**Claude's Reasoning:**
```
Analysis:
- Physics is the bottleneck (65/100)
- Collision issue → trajectory planning
- Previous iteration improved from 60→65, but stuck
- Quality is good (82), don't hurt it

Root Cause:
- System prompt doesn't emphasize collision avoidance
- Qwen3-VL generates trajectory without self-collision check
- Video model follows prompt literally

Strategy:
1. Add to system prompt:
   "Ensure collision-free inverse kinematics.
    Plan trajectories that maintain minimum 5cm clearance
    between all body parts."

2. Add to negative prompt:
   "arm interpenetration, self-collision, body overlap"

3. Keep existing quality constraints (don't over-constrain)

Expected:
- Physics: 65 → 75 (+10)
- Quality: 82 → 80 (-2, acceptable tradeoff)
- Adherence: 75 → 75 (maintain)

Confidence: 75%
```

**Output from Claude:**
```json
{
  "system_prompt": "...(complete improved prompt)...",
  "negative_prompt": "...(complete improved prompt)...",
  "reasoning": {...},
  "confidence": 0.75
}
```

---

## 📊 Multi-Objective Optimization

### The Tradeoff Problem

Three objectives often conflict:
- **High prompt adherence** ↔️ **Natural motion** (over-constraining hurts fluidity)
- **Physics realism** ↔️ **Visual quality** (more constraints → less aesthetic freedom)
- **Task correctness** ↔️ **Speed** (more frames → better adherence, but slower)

### Pareto Optimality

Can't maximize all objectives simultaneously. Instead, find **Pareto-optimal** solutions:
- No objective can improve without harming another
- Balance based on task priorities

**Claude's Advantage:**
- Natural language reasoning about tradeoffs
- Explicit priority setting
- Adaptive balancing based on current bottlenecks

Example:
```
If physics = 90, adherence = 70, quality = 85:
  → Focus on adherence (bottleneck)
  → Accept slight physics drop (90→88) if adherence improves (70→78)
  → Don't touch quality (already good)
```

---

## 🔄 Iterative Refinement Strategy

### Progressive Improvement

**Iteration 1-2: Broad Improvements**
- Identify major issues (collisions, task misunderstanding)
- Add general constraints
- Large score jumps expected (+10-15 points)

**Iteration 3-4: Fine-Tuning**
- Refine constraints (not too strict, not too loose)
- Balance competing objectives
- Smaller improvements (+5-8 points)

**Iteration 5: Convergence**
- Minor tweaks
- Verify no regression
- Final polish (+2-5 points)

### Stopping Criteria

Stop when:
1. ✅ **Success**: All scores ≥70, average ≥80
2. ❌ **Max iterations**: Reached 5 iterations (return best)
3. ⚠️ **Plateau**: No improvement for 2 consecutive iterations
4. 🎯 **High confidence**: Claude reports 90%+ confidence

---

## 🎓 Scientific Foundations

### Key Research Areas

1. **Vision-Language Models (VLMs)**
   - Multimodal fusion (vision + language)
   - Task-specific prompting
   - Zero-shot generalization

2. **Video Diffusion Models**
   - Latent diffusion for efficiency
   - Physics-informed priors
   - Negative prompt guidance

3. **Multimodal Reward Models**
   - Video quality assessment
   - Multi-dimensional scoring
   - Chain-of-thought reasoning

4. **Prompt Optimization**
   - Automatic prompt engineering
   - Multi-objective optimization
   - Meta-learning for warm start

5. **Constitutional AI**
   - Self-critique and revision
   - Iterative refinement
   - Explainable decision-making

### State-of-the-Art Techniques (2026)

**From Research:**
- **Direct Preference Optimization (DPO)**: 40% more efficient than RLHF
- **TextGrad**: Automatic differentiation via text
- **DSPy**: Programming foundation models
- **Contextual Bandits**: Intelligent action selection (60% regret reduction)
- **Meta-Learning**: Bi-level optimization for fast adaptation
- **Curriculum Learning**: Progressive difficulty training
- **VideoReward**: Multi-dimensional video reward models
- **Diffusion-NPO**: Negative preference optimization

See [archive/root/RESEARCH_FINDINGS_2026.md](archive/root/RESEARCH_FINDINGS_2026.md) for comprehensive research review.

---

## 💻 Implementation Choices

### Sequential Model Loading (RTX 5090)

**Challenge**: 4 models, limited VRAM (32GB)

**Solution**:
```
Step 1: Load Qwen3-VL (8GB) → Generate → Unload
Step 2: Load Video Model (16GB) → Generate → Unload
Step 3: Load Validator (12GB) → Validate → Unload
Step 4: Claude API (0GB VRAM) → Optimize → Done

Peak VRAM: 16GB (well within 32GB limit)
```

**Techniques:**
- **4-bit Quantization**: Reduce model size 4×
- **LRU Eviction**: Automatically unload least recently used
- **Layer-wise Inference**: Load model layer-by-layer if needed
- **Explicit Unloading**: `del model; torch.cuda.empty_cache()`

### API vs Local

**Claude Opus 4.6**: API-based
- ✅ Zero VRAM
- ✅ Latest model
- ✅ No deployment complexity
- ❌ Requires internet
- ❌ ~$0.50 per iteration

**Qwen3-VL, Video Models, Validator**: Local
- ✅ No API costs
- ✅ Full control
- ✅ Offline operation
- ❌ VRAM required
- ❌ Slower inference

**Hybrid approach**: Best of both worlds

---

## 📈 Performance Analysis

### Expected Convergence

```
Iteration | Avg Score | Improvement | Cumulative Cost
----------|-----------|-------------|----------------
0 (base)  | 65        | -           | $0
1         | 72        | +7          | $0.50
2         | 78        | +6          | $1.00
3         | 83        | +5          | $1.50
4         | 85        | +2          | $2.00
5         | 86        | +1          | $2.50

Success: 85 ≥ 80 ✓ at iteration 4
Total cost: $2.00
```

### Success Rate Prediction

**Based on Claude Opus 4.6 capabilities:**
- Simple tasks (pick object): 90% success in ≤3 iterations
- Medium tasks (bimanual): 80% success in ≤4 iterations
- Complex tasks (whole-body): 70% success in ≤5 iterations
- **Overall**: 75-85% success rate in ≤5 iterations

### Comparison: Manual vs Self-Tuning

| Approach | Avg Score | Iterations | Time | Cost |
|----------|-----------|------------|------|------|
| **Manual** | 65-70 | 1 | 5 min | $0 |
| **Manual + Expert** | 75-80 | 3-5 (human) | 30 min | $0 |
| **Self-Tuning** | 80-85 | 3-4 (auto) | 15 min | $2 |

**Self-tuning wins**: Higher quality, faster, scalable

---

## 🔬 Future Research Directions

### Short-Term (v2.1)
- Prompt library with similarity search
- Warm start from successful prompts
- A/B testing of multiple prompt variants
- Confidence-based early stopping

### Medium-Term (v2.5)
- Multi-agent debate (3 Claude instances)
- Constitutional AI self-critique
- Cross-task transfer learning
- Curriculum learning for training

### Long-Term (v3.0)
- Train RL policy from Claude's data
- Hybrid Claude + RL (reasoning + learned patterns)
- World models for simulation
- End-to-end differentiable pipeline

---

## 🎓 Key Insights

### What Makes This System Unique?

1. **Multi-Stage Pipeline**: Not just video generation, but enhancement → generation → validation → optimization

2. **Explainable AI**: Every decision traceable (Claude's reasoning, validation analysis)

3. **Progressive Refinement**: Don't settle for first result, iteratively improve

4. **Multi-Objective Balance**: Explicitly reason about tradeoffs

5. **Domain-Specific**: Tailored for robotics (physics, kinematics, task semantics)

### Why It Works

- **Qwen3-VL**: Bridges gap between simple prompts and detailed requirements
- **Diffusion Models**: State-of-the-art video generation quality
- **Cosmos-Reason2**: Provides actionable, multi-dimensional feedback
- **Claude Opus**: Best-in-class reasoning for optimization
- **Iterative Loop**: Systematically addresses failures

---

## 📚 Further Reading

**Essential Documents:**
1. [README.md](README.md) - Overview
2. [THEORY.md](THEORY.md) - This document
3. [QUICKSTART.md](QUICKSTART.md) - Practical guide

**Research Archive:**
1. [RESEARCH_FINDINGS_2026.md](archive/root/RESEARCH_FINDINGS_2026.md) - Comprehensive 2026 research
2. [CLAUDE_OPUS_BRAIN_APPROACH.md](archive/root/CLAUDE_OPUS_BRAIN_APPROACH.md) - Self-tuning design
3. [SELF_TUNING_AGENT.md](archive/root/SELF_TUNING_AGENT.md) - Original RL approach

**Papers & References:**
- See RESEARCH_FINDINGS_2026.md for 45+ citations

---

**Last Updated**: 2026-02-07
**Status**: Complete
**Version**: 2.0 (theory documented, implementation in progress)
