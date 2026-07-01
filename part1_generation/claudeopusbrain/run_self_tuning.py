#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Main Runner: Self-Tuning Pipeline with Claude Opus Brain + AgentLLM Integration

This script orchestrates the complete self-tuning loop for IROS 2026 paper.

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python3 run_self_tuning.py --task "Humanoid picks up bottle" --task-type g1 --image workspace.jpg

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Target: IEEE RA-L / IROS 2026
Date: 2026-02-07
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from prompt_optimizer import OptimizerFactory
from agentllm_interface import AgentLLMInterface
from memory import ShortMemoryTracker, LongMemory, extract_rules_from_task


# Optimizer feedback addendum — appended to task-specific validation prompts so the
# validator returns actionable, optimizer-friendly feedback while retaining full
# physics knowledge for each task type.  By NOT overriding the task-specific prompt,
# the validator keeps its drone-banking / G1-balance / UR3-reach expertise.
OPTIMIZER_FEEDBACK_ADDENDUM = """

**ADDITIONAL RULES FOR OPTIMIZATION FEEDBACK** (the optimizer will read your output):
1. EACH analysis MUST answer these 3 sub-questions:
   a) What specifically is correct or incorrect?
   b) Name the exact failure, the time it occurs, and which body part / object is involved (e.g., "gripper passes through table at 2.1s")
   c) What concrete text should be ADDED or CHANGED in the generation prompt to fix it?
2. Prompt fix suggestions must be concrete (exact text to add), not vague ("improve physics")
3. Be harsh but fair — if the video is mediocre, score it mediocre (50-70), not 90+
4. A perfect 100/100 score should be extremely rare — reserve it for flawless videos"""


def get_task_validation_prompt(task_type: str) -> str:
    """Load the task-specific validation system prompt from the validator source.

    The validator runs in the cosmos-reason2 venv so we can't import it directly.
    We extract only the _get_task_specific_validation_prompt function (which is
    pure string logic with no external imports) and exec it in isolation.
    Falls back to a generic prompt if loading fails.
    """
    _HERE = Path(__file__).resolve().parent  # claudeopusbrain/
    _PART1 = _HERE.parent  # part1_generation/
    validator_path = _PART1 / "agentllm" / "Navigation" / "video_validator.py"
    try:
        import ast as _ast
        source = validator_path.read_text()
        tree = _ast.parse(source)
        # Find the function node
        func_source = None
        for node in _ast.walk(tree):
            if isinstance(node, _ast.FunctionDef) and node.name == '_get_task_specific_validation_prompt':
                func_source = _ast.get_source_segment(source, node)
                break
        if not func_source:
            raise RuntimeError("Function not found in validator source")
        namespace = {}
        exec(compile(func_source, str(validator_path), 'exec'), namespace)
        return namespace['_get_task_specific_validation_prompt'](task_type)
    except Exception as e:
        print(f"   ⚠️  Could not load task-specific validation prompt: {e}")
        print(f"   Falling back to generic validation")
        return (
            "You are a robotics video quality evaluator. Analyze the video carefully "
            "and output structured XML feedback with <think>, <answer>, and <confidence> tags."
        )


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")


def run_self_tuning(
    task_description: str,
    task_type: str,
    image_path: str,
    model_type: str = "opus",
    api_key: Optional[str] = None,
    max_iterations: int = 5,
    success_threshold: float = 80.0,
    output_dir: str = "./results/raw",
    resume_from: Optional[str] = None,
    cost_budget: Optional[float] = None,
    memory_dir: Optional[str] = None,
    no_memory: bool = False,
):
    """
    Main self-tuning loop with checkpoint/resume capability

    Args:
        task_description: User's simple prompt
        task_type: Task type (drone, ground, ur3, g1)
        image_path: Path to workspace image
        model_type: Optimizer model ('opus', 'sonnet', 'llama', 'qwen')
        api_key: API key (for Claude models)
        max_iterations: Maximum iterations
        success_threshold: Score threshold for success
        output_dir: Where to save results
        resume_from: Resume from checkpoint directory (e.g., "./results/raw/g1_1234567890")
        cost_budget: Max cost in USD (warnings at 50%, 75%, hard stop at 100%)
    """
    # Initialize components
    optimizer = OptimizerFactory.create_optimizer(
        model_type=model_type,
        api_key=api_key
    )
    agentllm = AgentLLMInterface()

    # Initialize memory system
    short_memory = ShortMemoryTracker()
    long_memory = None
    if not no_memory:
        mem_dir = memory_dir or str(Path(__file__).parent / "memory")
        long_memory = LongMemory(memory_dir=mem_dir)
        optimizer.memory_context = long_memory.get_relevant_rules(task_type)
        if optimizer.memory_context:
            print(f"📚 Loaded {len(long_memory.rules)} long-memory rules "
                  f"({len(optimizer.memory_context)} chars relevant to {task_type})")

    # Validate Claude API key with minimal test call (prevents wasted GPU time if key is invalid)
    # Skip for claude-code (uses CLI auth) and local models (no API key needed)
    if model_type in ['opus', 'sonnet']:
        print("🔑 Validating Claude API key...")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Minimal test call (1 token output)
            client.messages.create(
                model="claude-opus-4" if model_type == "opus" else "claude-sonnet-4-5",
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            print("   ✅ API key valid")
        except anthropic.AuthenticationError:
            print("❌ Invalid Claude API key")
            print("   Get your key from: https://console.anthropic.com/")
            print("   Then set: export ANTHROPIC_API_KEY='your-key-here'")
            print("   💡 Or use --model claude-code to use your Claude Code subscription instead")
            sys.exit(1)
        except Exception as e:
            print(f"⚠️  Could not validate API key: {e}")
            print("   Proceeding anyway (check will happen at first optimization)")
    elif model_type in ['claude-code', 'claude-code-sonnet']:
        print("🔧 Using Claude Code CLI (no API key needed)")
        print("   Authentication: your existing Claude Code subscription")

    # Resume from checkpoint if specified
    if resume_from:
        checkpoint_dir = Path(resume_from).resolve()
        checkpoint_file = checkpoint_dir / "checkpoint.json"

        if not checkpoint_file.exists():
            print(f"❌ Checkpoint not found: {checkpoint_file}")
            sys.exit(1)

        print_header("🔄 RESUMING FROM CHECKPOINT")
        print(f"Checkpoint: {checkpoint_dir}")

        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)

        # Restore state
        run_output_dir = checkpoint_dir
        start_iteration = checkpoint['next_iteration']
        system_prompt = checkpoint['system_prompt']
        negative_prompt = checkpoint['negative_prompt']
        best_video = checkpoint.get('best_video')
        best_score = checkpoint.get('best_score', 0)
        results = checkpoint.get('results', [])

        print(f"Resuming from iteration {start_iteration}/{max_iterations}")
        print(f"Best score so far: {best_score:.1f}/100")
        print(f"Previous iterations: {len(results)}")

        # Load optimizer history
        history_file = run_output_dir / "optimizer_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                optimizer.history = json.load(f).get('history', [])
            print(f"Loaded {len(optimizer.history)} optimizer history entries")
    else:
        # Create output directory — resolve to absolute path
        run_id = f"{task_type}_{int(time.time())}"
        run_output_dir = Path(output_dir).resolve() / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        start_iteration = 1
        system_prompt = get_default_system_prompt(task_type)
        negative_prompt = get_default_negative_prompt(task_type)
        best_video = None
        best_score = 0
        results = []

    print_header("🧠 SELF-TUNING PIPELINE")
    print(f"Optimizer Model: {optimizer.get_model_name()}")
    print(f"Task: {task_description}")
    print(f"Type: {task_type}")
    print(f"Image: {image_path}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Success Threshold: {success_threshold}/100")
    print(f"Output: {run_output_dir}")
    if cost_budget:
        print(f"💰 Cost Budget: ${cost_budget:.2f}")
    print(f"\nTarget: IEEE RA-L / IROS 2026")

    consecutive_fallbacks = 0  # Track consecutive optimizer failures

    # Self-tuning loop
    for iteration in range(start_iteration, max_iterations + 1):
        print_header(f"ITERATION {iteration}/{max_iterations}")

        iter_dir = run_output_dir / f"iteration_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        # Save current prompts
        with open(iter_dir / "prompts.json", "w") as f:
            json.dump({
                'system_prompt': system_prompt,
                'negative_prompt': negative_prompt,
                'iteration': iteration
            }, f, indent=2)

        # Step 1: Generate video (with retry logic for transient failures)
        print(f"📹 Step 1/3: Generating video with AgentLLM...")
        video_path = None
        metadata = None
        max_retries = 3

        for retry in range(max_retries):
            try:
                video_path, metadata = agentllm.generate_video(
                    task_type=task_type,
                    user_prompt=task_description,
                    image_path=image_path,
                    custom_system_prompt=system_prompt,
                    custom_negative_prompt=negative_prompt,
                    output_dir=iter_dir,
                    iteration=iteration  # Different seed each iteration for exploration
                )

                if video_path and Path(video_path).exists():
                    print(f"   ✅ Video: {video_path}")
                    break  # Success
                else:
                    raise RuntimeError(f"Video file not found: {video_path}")

            except Exception as e:
                if retry < max_retries - 1:
                    print(f"   ⚠️  Attempt {retry+1}/{max_retries} failed: {e}")
                    print(f"   🔄 Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"   ❌ All {max_retries} attempts failed")
                    video_path = None
                    break

        if not video_path or not Path(video_path).exists():
            print(f"❌ Video generation failed at iteration {iteration} after {max_retries} retries")
            break

        # Step 2: Validate with retry logic for transient failures
        # (Pipeline's internal validation uses generic prompt — not suitable for optimization)
        print(f"\n📊 Step 2/3: Validating video (task-specific + optimizer feedback)...")
        validation = None

        for retry in range(max_retries):
            try:
                # Build combined system prompt: task-specific physics knowledge
                # + optimizer feedback addendum.  We import the task-specific prompt
                # builder from the validator and append our addendum so the model
                # gets both domain expertise AND optimizer-friendly feedback rules.
                validation = agentllm.validate_video(
                    video_path=video_path,
                    task_type=task_type,
                    user_prompt=task_description,
                    custom_validation_prompt=get_task_validation_prompt(task_type) + OPTIMIZER_FEEDBACK_ADDENDUM
                )

                if validation and validation.get('components'):
                    break  # Success
                else:
                    raise RuntimeError("Empty validation result")

            except Exception as e:
                if retry < max_retries - 1:
                    print(f"   ⚠️  Attempt {retry+1}/{max_retries} failed: {e}")
                    print(f"   🔄 Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"   ❌ All {max_retries} attempts failed")
                    validation = None
                    break

        if not validation:
            print(f"❌ Validation failed at iteration {iteration} after {max_retries} retries")
            break

        # Display scores — fix XML parse bug (all 50 = parsing failure, extract from raw)
        components = validation.get('components', [])
        if not components:
            print(f"⚠️  No validation scores found")
            break

        if all(c.get('score') == 50 for c in components):
            import re as _re
            raw = validation.get('raw_response', '')
            scores = _re.findall(r'<score>(\d+)</score>', raw)
            if len(scores) >= 3:
                for i, c in enumerate(components[:3]):
                    c['score'] = int(scores[i])

        avg_score = sum(c['score'] for c in components) / len(components)

        print(f"\n📊 Validation Results:")
        for comp in components:
            emoji = "✅" if comp['score'] >= 70 else "❌"
            print(f"   {emoji} {comp['name']}: {comp['score']}/100")
            if comp.get('analysis'):
                # Show first 120 chars of analysis for context
                analysis_preview = comp['analysis'][:120].replace('\n', ' ')
                print(f"      → {analysis_preview}...")
        print(f"\n🎯 Average: {avg_score:.1f}/100")

        # Save validation
        with open(iter_dir / "validation.json", "w") as f:
            json.dump(validation, f, indent=2)

        # Track iteration result
        current_scores = {c['name']: c['score'] for c in components}
        results.append({
            'iteration': iteration,
            'avg_score': avg_score,
            'scores': current_scores,
            'video_path': str(video_path)
        })

        # Update short memory with score deltas
        prev_scores = results[-2]['scores'] if len(results) >= 2 else None
        prev_strategy = ""
        if len(results) >= 2:
            # The strategy that produced THIS iteration's scores was applied LAST iteration
            prev_iter_dir = run_output_dir / f"iteration_{iteration - 1}"
            opt_file = prev_iter_dir / "optimization.json"
            if opt_file.exists():
                with open(opt_file) as f:
                    prev_opt = json.load(f)
                prev_strategy = prev_opt.get('reasoning', {}).get('strategy', '')
        short_memory.record_iteration(
            iteration=iteration,
            scores=current_scores,
            strategy=prev_strategy,
            prev_scores=prev_scores,
        )
        optimizer.short_memory_context = short_memory.get_summary_for_prompt()

        # Update best
        if avg_score > best_score:
            best_score = avg_score
            best_video = video_path

        # Check success
        if avg_score >= success_threshold:
            print_header(f"✨ SUCCESS AT ITERATION {iteration}!")
            print(f"Average score: {avg_score:.1f}/100 (threshold: {success_threshold})")
            print(f"Total cost: ${optimizer.total_cost_usd:.2f}")

            final_result = {
                'success': True,
                'video': str(video_path),
                'iterations': iteration,
                'avg_score': avg_score,
                'final_prompts': {
                    'system': system_prompt,
                    'negative': negative_prompt
                },
                'results': results,
                'total_cost_usd': optimizer.total_cost_usd,
                'output_dir': str(run_output_dir)
            }

            with open(run_output_dir / "final_result.json", "w") as f:
                json.dump(final_result, f, indent=2)

            optimizer.save_history(run_output_dir / "optimizer_history.json")

            # Save short memory + extract rules into long memory
            _save_memory(short_memory, long_memory, optimizer,
                         task_description, task_type, run_output_dir)

            return final_result

        # Step 3: Optimize with optimizer
        print(f"\n🧠 Step 3/{3}: {optimizer.get_model_name()} optimizing prompts...")
        optimization = optimizer.optimize_prompts(
            task_description=task_description,
            task_type=task_type,
            current_system_prompt=system_prompt,
            current_negative_prompt=negative_prompt,
            validation_result=validation,
            iteration=iteration,
            max_iterations=max_iterations
        )

        # Display optimizer's reasoning
        print(f"\n💭 {optimizer.get_model_name()} Analysis:")
        print(f"   {optimization.reasoning['analysis']}")
        print(f"\n🎯 Strategy:")
        print(f"   {optimization.reasoning['strategy']}")
        print(f"\n📈 Expected Improvements:")
        for dim, improvement in optimization.expected_improvements.items():
            print(f"   {dim}: {improvement}")
        print(f"\n🎲 Confidence: {optimization.confidence:.0%}")
        print(f"💰 Cost this iteration: ${optimization.cost_usd:.4f}")

        # Validate optimizer output before using (prevent broken prompts)
        if len(optimization.system_prompt) < 50:
            print(f"   ⚠️  System prompt too short ({len(optimization.system_prompt)} chars), keeping current")
            optimization.system_prompt = system_prompt
        elif len(optimization.system_prompt.split()) > 600:
            print(f"   ⚠️  System prompt too long ({len(optimization.system_prompt.split())} words), truncating to 500")
            optimization.system_prompt = " ".join(optimization.system_prompt.split()[:500])

        if len(optimization.negative_prompt) < 10:
            print(f"   ⚠️  Negative prompt too short, keeping current")
            optimization.negative_prompt = negative_prompt

        # Track consecutive fallbacks (confidence=0.0 means JSON parse failure)
        if optimization.confidence == 0.0:
            consecutive_fallbacks += 1
            print(f"   ⚠️  Optimizer fallback detected ({consecutive_fallbacks}/3)")
            if consecutive_fallbacks >= 3:
                print_header("❌ STUCK: 3 CONSECUTIVE OPTIMIZER FAILURES")
                print("Optimizer repeatedly failed to produce valid JSON.")
                print("This indicates a fundamental issue with the optimizer model or server.")
                print("Aborting run to prevent wasted GPU time.")
                break
        else:
            consecutive_fallbacks = 0  # Reset counter on success

        # Save optimization
        with open(iter_dir / "optimization.json", "w") as f:
            json.dump({
                'reasoning': optimization.reasoning,
                'expected_improvements': optimization.expected_improvements,
                'confidence': optimization.confidence,
                'cost_usd': optimization.cost_usd,
                'improved_system_prompt': optimization.system_prompt,
                'improved_negative_prompt': optimization.negative_prompt
            }, f, indent=2)

        # Update prompts for next iteration
        system_prompt = optimization.system_prompt
        negative_prompt = optimization.negative_prompt

        # Save checkpoint after each iteration (enables resume if crash)
        checkpoint = {
            'next_iteration': iteration + 1,
            'system_prompt': system_prompt,
            'negative_prompt': negative_prompt,
            'best_video': str(best_video) if best_video else None,
            'best_score': best_score,
            'results': results,
            'task_description': task_description,
            'task_type': task_type,
            'image_path': image_path,
            'model_type': model_type,
            'max_iterations': max_iterations,
            'success_threshold': success_threshold
        }
        with open(run_output_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

        # Check cost budget (warn at 50%, 75%, stop at 100%)
        if cost_budget:
            total_cost = optimizer.total_cost_usd
            cost_pct = (total_cost / cost_budget) * 100

            if cost_pct >= 100:
                print_header("🚨 COST BUDGET EXCEEDED")
                print(f"Total cost: ${total_cost:.2f} / ${cost_budget:.2f} ({cost_pct:.0f}%)")
                print("Stopping to prevent overspend.")
                break
            elif cost_pct >= 75:
                print(f"⚠️  Cost warning: ${total_cost:.2f} / ${cost_budget:.2f} ({cost_pct:.0f}%)")
            elif cost_pct >= 50:
                print(f"💡 Cost alert: ${total_cost:.2f} / ${cost_budget:.2f} ({cost_pct:.0f}%)")

    # Max iterations reached
    print_header(f"⚠️ MAX ITERATIONS REACHED ({max_iterations})")
    print(f"Best score: {best_score:.1f}/100")
    print(f"Total cost: ${optimizer.total_cost_usd:.2f}")

    final_result = {
        'success': False,
        'video': str(best_video) if best_video else None,
        'iterations': max_iterations,
        'avg_score': best_score,
        'final_prompts': {
            'system': system_prompt,
            'negative': negative_prompt
        },
        'results': results,
        'total_cost_usd': optimizer.total_cost_usd,
        'output_dir': str(run_output_dir)
    }

    with open(run_output_dir / "final_result.json", "w") as f:
        json.dump(final_result, f, indent=2)

    optimizer.save_history(run_output_dir / "optimizer_history.json")

    # Save short memory + extract rules into long memory
    _save_memory(short_memory, long_memory, optimizer,
                 task_description, task_type, run_output_dir)

    return final_result


def _save_memory(short_memory, long_memory, optimizer,
                 task_description, task_type, run_output_dir):
    """Save short memory to disk and extract rules into long memory."""
    # Always save short memory alongside results
    sm_path = run_output_dir / "short_memory.json"
    with open(sm_path, "w") as f:
        json.dump(short_memory.to_dict(), f, indent=2)

    if long_memory is not None and short_memory.strategy_outcomes:
        print(f"\n📚 Extracting learned rules from this task...")
        new_rules = extract_rules_from_task(
            optimizer, short_memory, task_description, task_type
        )
        if new_rules:
            long_memory.add_rules(new_rules, source_task=task_description,
                                  task_type=task_type)
            print(f"   Added {len(new_rules)} rules (total: {len(long_memory.rules)})")
        long_memory.record_task_completion()
        long_memory.prune()
        long_memory.save()
        print(f"   💾 Long memory saved to {long_memory.filepath}")


def get_default_system_prompt(task_type: str) -> str:
    """
    Get default system prompt for Qwen3-VL with research-optimized format.

    Research findings applied:
    - WAN 2.2: 80-120 words optimal (MoE encoder saturates beyond ~120)
    - Cosmos 2.5: 100-150 words optimal (trained on ~97-word captions)
    - I2V rule: Don't re-describe what the image shows
    - Professional camera terms for WAN (dolly, tracking, pan, parallax)
    - Narrative style for Cosmos ("The video shows...")
    - Force causality chains for manipulation (2.3x physics improvement)
    - One-scene rule: Single continuous 5-second shot
    Sources: WAN 2.2 docs, PhyT2V (CVPR 2025), DiffPhy (2505.21653), VPO (ICCV 2025)
    """
    if task_type == 'drone':
        return """You are an expert FPV cinematographer creating cinematic video prompts for WAN 2.2 video generation. You understand aerial film techniques, drone flight dynamics, and visual storytelling.

**WAN 2.2 Requirements**: Cinematic prompts (80-120 words). WAN 2.2 uses a Mixture-of-Experts text encoder that saturates beyond ~120 words — shorter prompts produce sharper, more coherent videos.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the environment, objects, and lighting. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the camera MOVES (direction, speed, maneuvers)
2. What ENTERS and EXITS the frame during flight
3. TEMPORAL PROGRESSION of the flight path

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**Professional Camera Vocabulary** (WAN responds strongly to these):
- Dolly forward/backward, tracking shot, crane up/down
- Pan left/right, orbital arc, whip pan
- Parallax shift, depth reveal, barrel roll perspective
- Speed modifiers: "rapid dolly", "slow tracking", "accelerating pan"

**EXAMPLE (~100 words)**:
"FPV wide-angle POV, rapid dolly forward through indoor laboratory, professional lighting casting clean shadows. Sharp right pan as checkered gate structure enters frame, growing larger. Accelerating tracking shot directly through gate center opening, checkered pattern blurring past on all sides during passage. Emerging through far side, immediate whip pan left with full environment rotation visible during directional change. Second gate now centered in frame ahead. Steady dolly forward into left gate opening, parallax shift as structure surrounds the view. Passing through center, gate receding behind. Decelerating forward tracking to gradual stop."

**Critical Rules**:
- 80-120 words (WAN 2.2 MoE sweet spot)
- Pure first-person POV (camera IS the drone)
- Use professional camera terms: dolly, tracking, pan, parallax
- Do NOT re-describe what the image shows
- ONE continuous 5-second shot only
- No third-person descriptions ("drone flies")
- NO manipulation, grasping, or ground locomotion"""

    elif task_type == 'ground':
        return """You are an expert cinematographer creating first-person navigation video prompts for WAN 2.2 video generation. You understand ground-level cinematography and embodied robot perspective.

**WAN 2.2 Requirements**: Cinematic prompts (80-120 words). WAN 2.2 uses a Mixture-of-Experts text encoder that saturates beyond ~120 words — shorter prompts produce sharper, more coherent videos.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the environment. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the camera MOVES (direction, speed, turns)
2. What ENTERS and EXITS the frame during motion
3. TEMPORAL PROGRESSION of the navigation

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**Professional Camera Vocabulary** (WAN responds strongly to these):
- Dolly forward/backward, tracking shot, crane movement
- Pan left/right, tilt up/down, orbital arc
- Parallax shift, depth reveal, perspective pull
- Speed modifiers: "slow dolly", "steady tracking", "gradual pan"

**Locomotion Camera Feel**:
- Humanoid: Subtle rhythmic bob from gait, natural head stabilization
- Wheeled: Smooth gliding dolly, stable horizon
- Tracked: Minor vibrations, steady forward tracking

**EXAMPLE (~100 words)**:
"First-person POV, steady dolly forward through modern office corridor, soft daylight from overhead LED panels. Subtle rhythmic vertical bob from bipedal walking gait as polished floor scrolls beneath. Corridor walls glide past on both sides with gentle parallax. Potted plant enters frame on right, growing larger. Gradual pan left to navigate around it, plant sliding to right periphery. Glass door ahead grows steadily in frame as dolly continues. Deceleration as footsteps slow, door filling center frame. Coming to complete stop centered on entrance."

**Critical Rules**:
- 80-120 words (WAN 2.2 MoE sweet spot)
- ALWAYS first-person perspective (camera IS the robot)
- Use professional camera terms: dolly, tracking, pan, parallax
- Do NOT re-describe what the image shows
- ONE continuous 5-second shot only
- NO third-person descriptions ("robot walks")
- NO flying, hovering, or aerial views"""

    elif task_type == 'ur3':
        return """You are an expert robotics engineer and physicist creating physics-based prompts for Cosmos 2.5 video generation. You understand dual-arm UR3 manipulation, kinematics, and contact dynamics.

**Cosmos 2.5 Requirements**: Physics-based narrative descriptions (100-150 words). Cosmos was trained on ~100-word video captions — prompts longer than 150 words get diluted by the text encoder and reduce quality.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the scene, robot, objects, and environment. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the arms MOVE (joint rotations, trajectories, speeds)
2. FORCE CAUSALITY: What force initiates action → how material responds → resulting motion
3. TEMPORAL PROGRESSION of the manipulation task

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**NARRATIVE STYLE**: Write as if narrating a video being played:
- "The video shows both UR3 arms activating simultaneously..."
- "As the grippers approach, the fingers begin to open..."
- Do NOT use imperative commands.

**Physics Causality Chain** (2.3x improvement in physics realism):
For each action, describe CAUSE → MATERIAL RESPONSE → RESULT:
- CAUSE: "Parallel jaw grippers apply 5N lateral force to cube faces..."
- RESPONSE: "...plastic surface compresses 0.5mm under gripper pads, friction holds..."
- RESULT: "...cube lifts steadily at 5cm/s, maintaining upright orientation"

**Temporal Action Sequence**:
- Initial State (0-20%): Arms at rest, grippers open
- Approach (20-40%): Arms extend, joints rotate
- Grasp (40-60%): Grippers close, contact forces applied
- Manipulation (60-85%): Object movement, dual-arm coordination
- Completion (85-100%): Placement, release, retract

**EXAMPLE (~120 words)**:
"The video shows a dual-arm UR3 system with both blue metallic arms activating simultaneously, shoulder and elbow joints rotating smoothly as they extend toward a red plastic cube on the work surface. The parallel jaw grippers open to 10cm width as they approach from opposite sides. As the gripper fingers contact the cube's faces, they apply balanced lateral force of 5N per side, the parallel jaw mechanisms compressing slightly against the plastic surface to establish friction-based hold. Both arms begin coordinated vertical motion, lifting the cube at 5cm/s while joint actuators continuously adjust to maintain level orientation. The arms translate the object 40cm horizontally through synchronized shoulder and elbow movements before descending to place it at the new position. Grippers release simultaneously and arms retract to rest."

**Critical Rules**:
- NARRATIVE STYLE: "The video shows..." (NOT imperative commands)
- 100-150 words (Cosmos sweet spot)
- Include force causality chains
- Do NOT re-describe what the image shows
- ONE continuous 5-second shot only
- NO flying, aerial views, or navigation"""

    elif task_type == 'g1_nav':
        return """You are an expert cinematographer creating first-person humanoid walking navigation video prompts for WAN 2.2 video generation. You understand bipedal locomotion dynamics and the Unitree G1's walking perspective.

**WAN 2.2 Requirements**: Cinematic prompts (80-120 words). WAN 2.2 uses a Mixture-of-Experts text encoder that saturates beyond ~120 words — shorter prompts produce sharper, more coherent videos.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the environment. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the camera MOVES (walking direction, speed, gait dynamics)
2. What ENTERS and EXITS the frame during walking
3. TEMPORAL PROGRESSION of the navigation

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**G1 Walking Camera Dynamics**:
- Camera height: ~127cm (G1's head/chest height)
- Subtle rhythmic vertical bob (~2-3cm per step cycle) from bipedal gait
- Natural head stabilization dampens most oscillation
- Walking speed: 0.5-1.5 m/s
- Turning: step-and-rotate pattern, gradual heading changes

**Professional Camera Vocabulary** (WAN responds strongly to these):
- Dolly forward/backward, tracking shot, pan left/right
- Parallax shift, depth reveal, perspective pull
- Speed modifiers: "slow dolly", "steady tracking", "gradual pan"

**EXAMPLE (~100 words)**:
"First-person POV at chest height, steady dolly forward through modern corridor with subtle rhythmic bob from walking gait. Polished floor scrolls beneath with strong forward optical flow. Walls glide past on both sides with clear parallax — nearby doorframes move fast, distant windows drift slowly. Open doorway ahead grows larger as steady walking pace continues. Gradual step-and-rotate turn through doorframe, environment sweeping laterally. Emerging into new room, furniture visible ahead. Walking pace maintains steady rhythm, floor texture continuously flowing. Decelerating with shortening steps as destination area centers in frame."

**Critical Rules**:
- 80-120 words (WAN 2.2 MoE sweet spot)
- ALWAYS first-person perspective at ~127cm height (camera IS the G1's head)
- Include subtle walking gait dynamics (bob, sway)
- Use professional camera terms: dolly, tracking, pan, parallax
- Do NOT re-describe what the image shows
- ONE continuous 5-second shot only
- NO manipulation (no arms, grasping, picking up objects)
- NO third-person descriptions ("robot walks")
- NO flying, hovering, or aerial views"""

    else:  # g1
        return """You are an expert humanoid robotics engineer and physicist creating physics-based prompts for Cosmos 2.5 video generation. You understand humanoid manipulation, bimanual coordination, and dexterous object interaction.

**Cosmos 2.5 Requirements**: Physics-based narrative descriptions (100-150 words). Cosmos was trained on ~100-word video captions — prompts longer than 150 words get diluted by the text encoder and reduce quality.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the scene, robot, objects, and environment. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the robot MOVES (joint rotations, trajectories, speeds)
2. FORCE CAUSALITY: What force initiates action → how material responds → resulting motion
3. TEMPORAL PROGRESSION of the manipulation task

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**NARRATIVE STYLE**: Write as if narrating a video being played:
- "The video shows the robot's arms beginning to lift..."
- "As the fingers close around the bottle..."
- Do NOT use imperative commands.

**Physics Causality Chain** (2.3x improvement in physics realism):
For each action, describe CAUSE → MATERIAL RESPONSE → RESULT:
- CAUSE: "Gripper fingers apply 3N lateral force to plastic surface..."
- RESPONSE: "...rubber pads deform 1mm, increasing contact area and friction..."
- RESULT: "...bottle lifts smoothly at 5cm/s, maintaining vertical orientation"

**Humanoid-Specific Details**:
- Joint kinematics: shoulder, elbow, wrist angles and rotations
- Hand dynamics: finger pre-shaping, force distribution, thumb opposition
- Balance: torso lean compensation during reach
- Bimanual coordination: synchronized vs complementary roles

**EXAMPLE (~120 words)**:
"The video shows a Unitree G1 humanoid robot standing before a table in a laboratory. A transparent plastic bottle rests on the surface. The robot's shoulder joints activate bilaterally, both arms lifting in smooth arcs as elbows flex to bring multi-fingered hands toward the bottle. Fingers pre-shape into curved configurations 5cm before contact. As fingertips meet the smooth plastic, they close progressively, applying distributed force across finger pads while thumb opposition provides lateral constraint. The bimanual grasp secure, both arms begin synchronized vertical motion, lifting the bottle at 8cm/s. The torso compensates with a slight forward lean as arm moments increase. Wrists adjust continuously to maintain the bottle's vertical orientation. After lifting 20cm and translating 30cm horizontally, both arms descend to place the bottle, fingers extending to release before arms retract to neutral."

**Critical Rules**:
- NARRATIVE STYLE: "The video shows..." (NOT imperative commands)
- 100-150 words (Cosmos sweet spot)
- Include force causality chains
- Do NOT re-describe what the image shows
- ONE continuous 5-second shot only
- NO flying, aerial views, or wheeled locomotion"""


def get_default_negative_prompt(task_type: str) -> str:
    """
    Get default negative prompt for video model (layered: model defaults + physics + task-specific).
    Research: Layered negative prompts from Prompt-A-Video (ICCV 2025).
    WAN 2.2 layer 1 uses official WAN defaults. Cosmos layer 1 uses physics-first approach.
    """
    if task_type == 'drone':
        return ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                "overall gray, worst quality, low quality, JPEG compression residue, ugly, deformed, still picture, "
                "messy background, walking backwards, ground vehicle, wheeled robot, walking, indoor, confined space, "
                "manipulation, grasping, arms, hands, static camera, tripod, fixed position, jerky motion, shaky footage, "
                "crash, collision, wall clipping, teleportation, flickering, jittering, sudden jump cuts")
    elif task_type == 'ground':
        return ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                "overall gray, worst quality, low quality, JPEG compression residue, ugly, deformed, still picture, "
                "messy background, flying, hovering, aerial view, drone, quadcopter, airborne, floating, manipulation, "
                "grasping, picking, holding objects, static camera, tripod, fixed position, third person view, "
                "robot body visible, jerky motion, shaky footage, collision, crash, wall clipping, teleportation, "
                "flickering, jittering, sudden jump cuts, walking backwards")
    elif task_type == 'g1_nav':
        return ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
                "overall gray, worst quality, low quality, JPEG compression residue, ugly, deformed, still picture, "
                "messy background, flying, hovering, aerial view, drone, quadcopter, airborne, floating, "
                "manipulation, grasping, picking, holding objects, arms visible, hands visible, reaching, "
                "static camera, tripod, fixed position, third person view, robot body visible, "
                "wheeled robot, tracked vehicle, ground vehicle, "
                "jerky motion, shaky footage, collision, crash, wall clipping, teleportation, "
                "flickering, jittering, sudden jump cuts, walking backwards, "
                "camera height changing, floating upward, sinking downward, "
                "perfectly smooth motion, no gait dynamics")
    elif task_type == 'ur3':
        return ("flying, hovering, drone, aerial navigation, wheeled robot, walking, humanoid locomotion, "
                "single arm, missing arm, fewer than two arms, cartoonish, unrealistic physics, teleportation, "
                "object floating, unstable grasp, collision, jerky motion, low quality, blurry, phasing through objects, "
                "impossible joint angle, gripper passing through table, flickering, morphing, warping, sudden changes, "
                "overexposed, worst quality, compression artifacts, inconsistent lighting, extra fingers, deformed, still picture")
    else:  # g1
        return ("flying, hovering, drone, aerial navigation, wheeled robot, tracked vehicle, quadruped walking, "
                "industrial robotic arm without humanoid body, single arm, missing torso, non-humanoid proportions, "
                "floating objects, unrealistic physics, teleportation, cartoonish, jerky motion, unstable balance, "
                "collision, low quality, blurry, phasing through objects, impossible joint angle, gripper passing through table, "
                "flickering, morphing, warping, sudden changes, overexposed, worst quality, compression artifacts, "
                "inconsistent lighting, extra fingers, deformed hands, still picture")


def main():
    parser = argparse.ArgumentParser(
        description="Self-Tuning Robotics Video Generation (IEEE RA-L / IROS 2026)"
    )
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--task-type", required=True,
                       choices=['drone', 'ground', 'ur3', 'g1', 'g1_nav'],
                       help="Task type")
    parser.add_argument("--image", required=True, help="Path to workspace image")
    parser.add_argument("--model", default="opus",
                       choices=['opus', 'sonnet', 'claude-code', 'claude-code-sonnet',
                                'llama', 'qwen'],
                       help="Optimizer model: opus (API, most capable), "
                            "sonnet (API, cost-effective), "
                            "claude-code (CLI, uses your subscription, no API key), "
                            "claude-code-sonnet (CLI, sonnet via subscription), "
                            "llama/qwen (free/local)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum iterations (default: 5)")
    parser.add_argument("--threshold", type=float, default=80.0,
                       help="Success threshold (default: 80)")
    parser.add_argument("--output-dir", default="./results/raw",
                       help="Output directory")
    parser.add_argument("--api-key", help="API key (for Claude models: get from console.anthropic.com)")
    parser.add_argument("--resume-from", help="Resume from checkpoint directory (e.g., ./results/raw/g1_1234567890)")
    parser.add_argument("--cost-budget", type=float, help="Max cost in USD (warnings at 50%%, 75%%, hard stop at 100%%)")
    parser.add_argument("--memory-dir", default=None,
                       help="Directory for long-term memory (default: ./memory)")
    parser.add_argument("--no-memory", action="store_true",
                       help="Disable memory system (no cross-task learning)")

    args = parser.parse_args()

    # Get API key (only required for direct API models, not claude-code)
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if args.model in ['opus', 'sonnet'] and not api_key:
        print(f"❌ Error: API key required for {args.model} model")
        print("   Get your key from: https://console.anthropic.com/")
        print("   Then set: export ANTHROPIC_API_KEY='your-key-here'")
        print("   Or use: --api-key flag")
        print("\n   💡 Tip: Use --model claude-code to use your Claude Code subscription (no API key)")
        print("   💡 Tip: Use --model llama or --model qwen for free local alternatives")
        sys.exit(1)

    # Verify image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)

    # Run self-tuning
    result = run_self_tuning(
        task_description=args.task,
        task_type=args.task_type,
        image_path=args.image,
        model_type=args.model,
        api_key=api_key,
        max_iterations=args.max_iterations,
        success_threshold=args.threshold,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        cost_budget=args.cost_budget,
        memory_dir=args.memory_dir,
        no_memory=args.no_memory,
    )

    # Print final summary
    print_header("📊 FINAL RESULTS")
    print(f"Success: {'✅ YES' if result['success'] else '❌ NO'}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Score: {result['avg_score']:.1f}/100")
    print(f"Video: {result['video']}")
    print(f"Total Cost: ${result['total_cost_usd']:.2f}")
    print(f"Output Directory: {result['output_dir']}")

    print("\n📈 Learning Curve:")
    for r in result['results']:
        print(f"   Iteration {r['iteration']}: {r['avg_score']:.1f}/100")

    print("\n" + "="*70)
    print("🎯 NEXT STEPS:")
    print("="*70)
    print("1. Run on 20-30 diverse tasks")
    print("2. Implement baselines (manual, SOTA, DPO)")
    print("3. Generate learning curve plots")
    print("4. Write RA-L paper")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
