# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Claude Code Brain: Uses the Claude Code CLI as the optimization brain

Instead of calling the Anthropic API directly (which requires a paid API key),
this uses the `claude` CLI tool in print mode (-p). This leverages whatever
authentication the user has set up for Claude Code (Pro, Max, or API key).

Usage:
    python3 run_self_tuning.py --model claude-code --task "..." --task-type g1 --image workspace.jpg

No API key needed — uses your existing Claude Code subscription.

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-02-14
"""

import json
import os
import re
import shutil
import subprocess
import time
from typing import Dict, List, Optional

from prompt_optimizer import PromptOptimizer, OptimizationResult


class ClaudeCodeBrain(PromptOptimizer):
    """
    Claude Code CLI-based optimization brain.

    Uses `claude -p` (print mode) instead of the Anthropic API.
    Works with Claude Pro, Max, or any Claude Code authentication.

    Same optimization logic as ClaudeBrain (system prompt, prompt
    construction, JSON parsing) but fully standalone — does NOT
    require the `anthropic` package.

    Cost: $0.00 per call (included in Claude Code subscription).
    """

    # CLI model flag -> display name
    MODEL_NAMES = {
        "opus": "Claude Opus 4.6 (via Code CLI)",
        "sonnet": "Claude Sonnet 4.5 (via Code CLI)",
        "haiku": "Claude Haiku 4.5 (via Code CLI)",
    }

    def __init__(
        self,
        model: str = "opus",
        max_tokens: int = 8000,
        timeout: int = 300,
        **kwargs,
    ):
        """
        Initialize Claude Code Brain.

        Args:
            model: Claude model short name for CLI ("opus", "sonnet", "haiku")
            max_tokens: Maximum tokens to generate
            timeout: CLI call timeout in seconds (default 5 min)
        """
        super().__init__()

        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Optimizer role system prompt (same as ClaudeBrain)
        self.system_prompt = self._get_system_prompt()

        # Verify the claude CLI is available
        self._claude_path = self._find_claude_cli()

    # ------------------------------------------------------------------
    # PromptOptimizer interface
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return self.MODEL_NAMES.get(self.model, f"Claude Code ({self.model})")

    def optimize_prompts(
        self,
        task_description: str,
        task_type: str,
        current_system_prompt: str,
        current_negative_prompt: str,
        validation_result: Dict,
        iteration: int,
        max_iterations: int = 5,
    ) -> OptimizationResult:
        """
        Optimize prompts by calling the Claude Code CLI in print mode.
        """
        # Build the user prompt
        user_prompt = self._build_optimization_prompt(
            task_description=task_description,
            task_type=task_type,
            current_system_prompt=current_system_prompt,
            current_negative_prompt=current_negative_prompt,
            validation_result=validation_result,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        # Combine system + user prompt for CLI
        full_prompt = (
            f"<instructions>\n{self.system_prompt}\n</instructions>\n\n"
            f"{user_prompt}\n\n"
            "CRITICAL: Your response must be ONLY valid JSON. "
            "No markdown code blocks, no explanatory text before or after. "
            "Just the raw JSON object."
        )

        print(f"🧠 Calling Claude Code CLI --model {self.model} (iteration {iteration})...")
        start_time = time.time()

        try:
            response_text = self._call_cli(full_prompt)
            elapsed = time.time() - start_time

            print(f"   Response: {len(response_text)} chars")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Cost: $0.00 (included in subscription)")

            # Parse JSON response
            result = self._parse_response(response_text, cost_usd=0.0)

            if result is None:
                print("   ⚠️  Creating fallback result (prompts unchanged)")
                result = OptimizationResult(
                    system_prompt=current_system_prompt,
                    negative_prompt=current_negative_prompt,
                    reasoning={
                        "analysis": "Claude Code CLI response could not be parsed as JSON",
                        "root_causes": ["JSON parsing failure"],
                        "strategy": "Keeping current prompts unchanged",
                        "tradeoffs": "No optimization this iteration",
                    },
                    expected_improvements={
                        "adherence": "maintain",
                        "physics": "maintain",
                        "quality": "maintain",
                    },
                    confidence=0.0,
                    raw_response=response_text,
                    cost_usd=0.0,
                    model_name=self.get_model_name(),
                )

            # Save to history
            self._add_to_history(
                iteration=iteration,
                task_description=task_description,
                task_type=task_type,
                current_prompts={
                    "system_prompt": current_system_prompt,
                    "negative_prompt": current_negative_prompt,
                    "validation": validation_result,
                },
                result=result,
            )

            return result

        except subprocess.TimeoutExpired:
            print(f"   ❌ Claude Code CLI timed out after {self.timeout}s")
            raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    def _find_claude_cli(self) -> str:
        """Find the claude CLI executable on PATH."""
        path = shutil.which("claude")
        if path:
            return path
        for candidate in [
            os.path.expanduser("~/.claude/bin/claude"),
            "/usr/local/bin/claude",
        ]:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        raise RuntimeError(
            "claude CLI not found in PATH.\n"
            "Install Claude Code: npm install -g @anthropic-ai/claude-code\n"
            "Or check that it is on your PATH."
        )

    def _call_cli(self, prompt: str) -> str:
        """
        Call `claude -p` and return the response text.

        Uses stdin to pass the prompt (handles any prompt length).
        """
        cmd = [
            self._claude_path,
            "-p",                       # print mode (non-interactive one-shot)
            "--model", self.model,       # opus / sonnet / haiku
        ]

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            stderr_preview = (result.stderr or "")[:300]
            raise RuntimeError(
                f"Claude Code CLI exited with code {result.returncode}: {stderr_preview}"
            )

        response = result.stdout.strip()
        if not response:
            raise RuntimeError("Claude Code CLI returned an empty response")

        return response

    # ------------------------------------------------------------------
    # Prompt construction (same logic as ClaudeBrain, standalone)
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """System prompt that defines Claude's role as optimization agent."""
        return """You are an expert optimization agent for robotics video generation.

Your role: Analyze video validation feedback and improve two types of prompts:

1. **System Prompt** (for Qwen3-VL language model):
   - Guides how the LLM extends user commands into detailed video generation prompts
   - Should be clear, specific, address identified failure modes
   - Focuses on: task understanding, physics constraints, motion planning, visual description

2. **Negative Prompt** (for video diffusion model):
   - Specifies what to AVOID in generated videos
   - Should include: visual artifacts, physics violations, motion errors observed in validation
   - Focuses on: preventing failures seen in previous iterations

Your optimization process:

1. **Analyze Validation Feedback**:
   - Read all validation scores (prompt adherence, physics realism, visual quality)
   - Identify lowest-scoring dimensions (bottlenecks)
   - Understand root causes from detailed analysis
   - Consider multi-objective tradeoffs

2. **Root Cause Analysis**:
   - Why did each failure occur?
   - Is it a prompt clarity issue? -> improve system prompt
   - Is it a diffusion model issue? -> improve negative prompt
   - Are objectives conflicting? -> balance carefully

3. **Strategic Improvement**:
   - Make TARGETED changes to address specific failures
   - Don't over-constrain (can hurt naturalness)
   - Don't remove what's working
   - Be surgical, not broad

4. **Multi-Objective Balancing**:
   - If improving physics hurts quality, find middle ground
   - Prioritize based on current bottlenecks
   - Consider task requirements

5. **Progressive Refinement**:
   - Early iterations (1-2): Broad improvements
   - Later iterations (3-4): Fine-tuning
   - If stuck: Try different approach

6. **Output Format**:
   You MUST respond with ONLY a JSON object in this exact format:

{
  "reasoning": {
    "analysis": "What went wrong and why? (2-3 sentences)",
    "root_causes": ["cause 1", "cause 2", ...],
    "strategy": "How will you improve it? (2-3 sentences)",
    "tradeoffs": "Any competing objectives to balance? (1-2 sentences)"
  },
  "system_prompt": "COMPLETE improved system prompt here (full text, 200-400 words)",
  "negative_prompt": "COMPLETE improved negative prompt here (full text, comma-separated)",
  "expected_improvements": {
    "adherence": "+X points" or "maintain",
    "physics": "+X points" or "maintain",
    "quality": "+X points" or "maintain"
  },
  "confidence": 0.75
}

**Critical Rules**:
- Return ONLY valid JSON - no markdown, no extra text
- System prompt must be COMPLETE (not a diff) - 200-400 words
- Negative prompt must be COMPLETE (not a diff) - comma-separated list
- Focus on lowest-scoring dimensions first
- Consider iteration number (early vs late stage)
- Aim for 80+ average score as success threshold
- Be explicit about tradeoffs

**Remember**: You're optimizing for ROBOTICS video generation. Physics realism, collision-free motion, and task correctness are critical."""

    def _build_optimization_prompt(
        self,
        task_description: str,
        task_type: str,
        current_system_prompt: str,
        current_negative_prompt: str,
        validation_result: Dict,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Build the user prompt that provides all context."""
        components = validation_result.get("components", [])
        scores_dict = {c["name"]: c["score"] for c in components}
        avg_score = sum(scores_dict.values()) / len(scores_dict) if scores_dict else 0

        score_summary = "\n".join([
            f"  - {comp['name']}: {comp['score']}/100 "
            f"{'✅' if comp['score'] >= 70 else '❌'}"
            for comp in components
        ])

        # Bottleneck priority signal
        priority_signal = ""
        if scores_dict:
            worst_name = min(scores_dict, key=scores_dict.get)
            best_name = max(scores_dict, key=scores_dict.get)
            gap = scores_dict[best_name] - scores_dict[worst_name]
            priority_signal = (
                f"\n⚠️ **BOTTLENECK**: '{worst_name}' = {scores_dict[worst_name]}/100 "
                f"(gap: {gap} pts below '{best_name}'). Allocate primary focus here.\n"
            )

        # Detailed analysis (worst-scoring first)
        sorted_components = sorted(components, key=lambda c: c["score"])
        detailed_analysis = "\n\n".join([
            f"**{comp['name']}** ({comp['score']}/100):\n{comp.get('analysis', 'N/A')}"
            for comp in sorted_components
        ])

        memory_section = self._get_full_memory_section()

        category = "Navigation" if task_type in ["drone", "ground", "g1_nav"] else "Manipulation"

        return f"""# Video Generation Optimization Task

## Task Description
User Request: **"{task_description}"**
Task Type: **{task_type}** ({category})

## Current Iteration: {iteration}/{max_iterations}

## Validation Results from Cosmos-Reason2

### Score Summary
{score_summary}

**Average Score: {avg_score:.1f}/100**
**Pass Threshold: 80/100**
**Status: {'✅ PASS' if avg_score >= 80 else '❌ FAIL - Needs Improvement'}**
{priority_signal}
### Detailed Analysis (worst-scoring first)
{detailed_analysis}

## Current Prompts (Iteration {iteration})

### System Prompt (for Qwen3-VL):
```
{current_system_prompt}
```

### Negative Prompt (for Video Model):
```
{current_negative_prompt}
```

{memory_section}

---

## Your Task

Analyze the validation feedback and improve both prompts for iteration {iteration + 1}.

**Focus on**:
1. Lowest-scoring dimensions (biggest bottlenecks)
2. Root causes of failures
3. Targeted, surgical improvements
4. Multi-objective balance
5. Progressive refinement strategy (iteration {iteration} of {max_iterations})

**Output ONLY valid JSON** in the specified format. No markdown code blocks, no extra text."""

    # ------------------------------------------------------------------
    # Response parsing (same logic as ClaudeBrain, standalone)
    # ------------------------------------------------------------------

    def _parse_response(self, response_text: str, cost_usd: float) -> Optional[OptimizationResult]:
        """
        Parse Claude's JSON response with robust fallback.

        Returns None if parsing fails (signals caller to use safe fallback).
        """
        try:
            text = response_text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # Try direct parse
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: extract largest JSON-like structure
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                matches = re.findall(json_pattern, text, re.DOTALL)
                if matches:
                    for match in sorted(matches, key=len, reverse=True):
                        try:
                            data = json.loads(match)
                            break
                        except json.JSONDecodeError:
                            continue
                    else:
                        raise ValueError("No valid JSON found in response")
                else:
                    raise ValueError("No JSON structure detected")

            # Validate required fields
            required = ["reasoning", "system_prompt", "negative_prompt",
                        "expected_improvements", "confidence"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            return OptimizationResult(
                system_prompt=data["system_prompt"],
                negative_prompt=data["negative_prompt"],
                reasoning=data["reasoning"],
                expected_improvements=data["expected_improvements"],
                confidence=float(data["confidence"]),
                raw_response=response_text,
                cost_usd=cost_usd,
                model_name=self.get_model_name(),
            )

        except Exception as e:
            print(f"⚠️  Failed to parse Claude Code response: {e}")
            print(f"   Response preview: {response_text[:300]}...")
            print(f"   🔄 Using SAFE FALLBACK: keeping current prompts unchanged")
            return None


# Quick self-test
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Claude Code Brain (CLI-based)")
    print("=" * 60)

    try:
        brain = ClaudeCodeBrain(model="sonnet")
        print(f"\n✅ Claude CLI found: {brain._claude_path}")
        print(f"   Model: {brain.get_model_name()}")
    except RuntimeError as e:
        print(f"\n❌ {e}")
        exit(1)

    # Mock validation
    mock_validation = {
        "pass": False,
        "confidence": 65,
        "components": [
            {
                "name": "Prompt Adherence",
                "score": 72,
                "analysis": "Task mostly followed, but arm selection unclear.",
            },
            {
                "name": "Physical Plausibility",
                "score": 58,
                "analysis": "Collision detected at frame 45.",
            },
            {
                "name": "Visual Quality",
                "score": 75,
                "analysis": "Good fidelity, minor flickering.",
            },
        ],
    }

    print("\nRunning test optimization...")
    result = brain.optimize_prompts(
        task_description="Humanoid picks up bottle",
        task_type="g1",
        current_system_prompt="You are a robotics video generation assistant...",
        current_negative_prompt="blurry, low quality, collision",
        validation_result=mock_validation,
        iteration=1,
        max_iterations=5,
    )

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Analysis: {result.reasoning.get('analysis', 'N/A')[:200]}")
    print(f"Strategy: {result.reasoning.get('strategy', 'N/A')[:200]}")
    print(f"System prompt preview: {result.system_prompt[:100]}...")
    print(f"Cost: ${result.cost_usd:.4f}")
