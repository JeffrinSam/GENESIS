# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Claude Brain: LLM-based optimization agent for robotics video generation

This module implements the Claude-based optimization brain that uses Claude models
(Opus 4.6 or Sonnet 4.5) to iteratively improve video generation prompts based
on validation feedback.

Key Features:
- Multi-objective optimization (adherence, physics, quality)
- Explainable reasoning
- 200K context window (sees all iteration history)
- Constitutional AI (self-critique)
- Supports both Opus (most capable) and Sonnet (cost-effective)

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-02-07
"""

import anthropic
import json
import os
import time
from typing import Dict, List, Optional

# Import base class
from prompt_optimizer import PromptOptimizer, OptimizationResult


class ClaudeBrain(PromptOptimizer):
    """
    Claude-based optimization brain for prompt refinement

    This class encapsulates the LLM-based optimization logic that analyzes
    validation feedback and improves prompts iteratively.

    Supports both Claude Opus 4 (most capable) and Sonnet 4.5 (cost-effective).
    """

    # Model configurations
    MODEL_CONFIGS = {
        "claude-opus-4-20250514": {
            "name": "Claude Opus 4.6",
            "input_cost": 15.0,   # per 1M tokens
            "output_cost": 75.0   # per 1M tokens
        },
        "claude-sonnet-4-5-20250929": {
            "name": "Claude Sonnet 4.5",
            "input_cost": 3.0,    # per 1M tokens
            "output_cost": 15.0   # per 1M tokens
        }
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-20250514",  # Default: Opus 4.6
        temperature: float = 0.7,
        max_tokens: int = 8000
    ):
        """
        Initialize Claude Brain

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use:
                   - "claude-opus-4-20250514" (Opus 4.6, most capable)
                   - "claude-sonnet-4-5-20250929" (Sonnet 4.5, cost-effective)
            temperature: Temperature for generation (0.7 = balanced)
            max_tokens: Maximum tokens to generate
        """
        # Initialize base class
        super().__init__()

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Verify model is supported
        if model not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported Claude model: {model}. "
                f"Supported: {list(self.MODEL_CONFIGS.keys())}"
            )

        # System prompt for Claude (defines its role as optimizer)
        self.system_prompt = self._get_claude_system_prompt()

    def get_model_name(self) -> str:
        """Return the name of the model being used"""
        return self.MODEL_CONFIGS[self.model]["name"]

    def optimize_prompts(
        self,
        task_description: str,
        task_type: str,
        current_system_prompt: str,
        current_negative_prompt: str,
        validation_result: Dict,
        iteration: int,
        max_iterations: int = 5
    ) -> OptimizationResult:
        """
        Main optimization function: Claude analyzes validation and improves prompts

        Args:
            task_description: User's simple prompt ("Humanoid picks up bottle")
            task_type: Task type (drone, ground, ur3, g1)
            current_system_prompt: Current system prompt for Qwen3-VL
            current_negative_prompt: Current negative prompt for video model
            validation_result: Full validation output from Cosmos-Reason2
            iteration: Current iteration number (1-5)
            max_iterations: Maximum iterations allowed

        Returns:
            OptimizationResult with improved prompts and reasoning
        """
        # Build comprehensive user prompt for Claude
        user_prompt = self._build_optimization_prompt(
            task_description=task_description,
            task_type=task_type,
            current_system_prompt=current_system_prompt,
            current_negative_prompt=current_negative_prompt,
            validation_result=validation_result,
            iteration=iteration,
            max_iterations=max_iterations
        )

        print(f"🧠 Calling Claude Opus (iteration {iteration})...")
        start_time = time.time()

        try:
            # Call Claude Opus API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            elapsed_time = time.time() - start_time

            # Calculate cost (approximate)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost_usd = self._calculate_cost(input_tokens, output_tokens)
            self.total_cost_usd += cost_usd

            print(f"   Tokens: {input_tokens} input, {output_tokens} output")
            print(f"   Cost: ${cost_usd:.4f} (total: ${self.total_cost_usd:.4f})")
            print(f"   Time: {elapsed_time:.1f}s")

            # Parse Claude's response (returns None if parsing failed)
            result = self._parse_response(
                response.content[0].text,
                cost_usd=cost_usd
            )

            # If parsing failed, create safe fallback (keeps prompts unchanged)
            if result is None:
                print(f"   ⚠️  Creating fallback result (prompts unchanged)")
                result = OptimizationResult(
                    system_prompt=current_system_prompt,  # Keep unchanged
                    negative_prompt=current_negative_prompt,  # Keep unchanged
                    reasoning={
                        'analysis': 'Claude response could not be parsed',
                        'root_causes': ['JSON parsing failure'],
                        'strategy': 'Keeping current prompts unchanged to avoid breaking the pipeline',
                        'tradeoffs': 'No optimization this iteration'
                    },
                    expected_improvements={
                        'adherence': 'maintain',
                        'physics': 'maintain',
                        'quality': 'maintain'
                    },
                    confidence=0.0,
                    raw_response=response.content[0].text,
                    cost_usd=cost_usd,
                    model_name=self.get_model_name()
                )

            # Save to history using base class method
            self._add_to_history(
                iteration=iteration,
                task_description=task_description,
                task_type=task_type,
                current_prompts={
                    'system_prompt': current_system_prompt,
                    'negative_prompt': current_negative_prompt,
                    'validation': validation_result
                },
                result=result
            )

            return result

        except Exception as e:
            print(f"❌ Error calling Claude API: {e}")
            raise

    def _get_claude_system_prompt(self) -> str:
        """
        System prompt that defines Claude's role as optimization agent

        This is critical - it tells Claude how to be an optimizer.
        """
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
   - Is it a prompt clarity issue? → improve system prompt
   - Is it a diffusion model issue? → improve negative prompt
   - Are objectives conflicting? → balance carefully

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

```json
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
```

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
        max_iterations: int
    ) -> str:
        """
        Build the user prompt that provides all context to Claude
        """
        # Extract validation scores
        components = validation_result.get('components', [])
        scores_dict = {c['name']: c['score'] for c in components}
        avg_score = sum(scores_dict.values()) / len(scores_dict) if scores_dict else 0

        score_summary = "\n".join([
            f"  - {comp['name']}: {comp['score']}/100 "
            f"{'✅' if comp['score'] >= 70 else '❌'}"
            for comp in components
        ])

        # Bottleneck priority signal — explicit attention directive
        if scores_dict:
            worst_name = min(scores_dict, key=scores_dict.get)
            best_name = max(scores_dict, key=scores_dict.get)
            gap = scores_dict[best_name] - scores_dict[worst_name]
            priority_signal = (
                f"\n⚠️ **BOTTLENECK**: '{worst_name}' = {scores_dict[worst_name]}/100 "
                f"(gap: {gap} pts below '{best_name}'). Allocate primary focus here.\n"
            )
        else:
            priority_signal = ""

        # Detailed analysis from Cosmos-Reason2 — put worst-scoring first
        sorted_components = sorted(components, key=lambda c: c['score'])
        detailed_analysis = "\n\n".join([
            f"**{comp['name']}** ({comp['score']}/100):\n{comp.get('analysis', 'N/A')}"
            for comp in sorted_components
        ])

        # Include full memory context (long memory + short memory + iteration history)
        memory_section = self._get_full_memory_section()

        # Build prompt
        prompt = f"""# Video Generation Optimization Task

## Task Description
User Request: **"{task_description}"**
Task Type: **{task_type}** ({'Navigation' if task_type in ['drone', 'ground', 'g1_nav'] else 'Manipulation'})

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

        return prompt

    def _parse_response(self, response_text: str, cost_usd: float) -> OptimizationResult:
        """
        Parse Claude's JSON response with robust fallback

        If parsing fails, returns None to signal fallback needed.
        """
        try:
            # Remove markdown code blocks if present
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]  # Remove ```json
            if text.startswith("```"):
                text = text[3:]  # Remove ```
            if text.endswith("```"):
                text = text[:-3]  # Remove trailing ```

            text = text.strip()

            # Try direct parse
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from mixed text
                import re
                # Find largest JSON-like structure
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                matches = re.findall(json_pattern, text, re.DOTALL)
                if matches:
                    # Try each match, use first that parses
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
            required_fields = ['reasoning', 'system_prompt', 'negative_prompt',
                             'expected_improvements', 'confidence']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            return OptimizationResult(
                system_prompt=data['system_prompt'],
                negative_prompt=data['negative_prompt'],
                reasoning=data['reasoning'],
                expected_improvements=data['expected_improvements'],
                confidence=float(data['confidence']),
                raw_response=response_text,
                cost_usd=cost_usd,
                model_name=self.get_model_name()
            )

        except Exception as e:
            print(f"⚠️  Failed to parse Claude response: {e}")
            print(f"   Response preview: {response_text[:300]}...")
            print(f"   🔄 Using SAFE FALLBACK: keeping current prompts unchanged")
            return None  # Signal to caller that parsing failed

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for Claude API call based on model

        Uses model-specific pricing from MODEL_CONFIGS
        """
        config = self.MODEL_CONFIGS[self.model]
        input_cost = (input_tokens / 1_000_000) * config["input_cost"]
        output_cost = (output_tokens / 1_000_000) * config["output_cost"]

        return input_cost + output_cost



# Example usage
if __name__ == "__main__":
    # Test Claude Brain with mock data
    import sys

    if len(sys.argv) < 2:
        print("Usage: python claude_brain.py <ANTHROPIC_API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]

    # Create brain
    brain = ClaudeBrain(api_key=api_key)

    # Mock validation result
    mock_validation = {
        'pass': False,
        'confidence': 65,
        'components': [
            {
                'name': 'Prompt Adherence',
                'score': 72,
                'analysis': 'Task mostly followed, but arm selection unclear in grasping phase.'
            },
            {
                'name': 'Physical Plausibility',
                'score': 58,
                'analysis': 'Collision detected at frame 45 between left arm and torso during reach.'
            },
            {
                'name': 'Visual Quality',
                'score': 75,
                'analysis': 'Good visual fidelity, some minor flickering in background.'
            }
        ]
    }

    # Test optimization
    result = brain.optimize_prompts(
        task_description="Humanoid picks up bottle",
        task_type="g1",
        current_system_prompt="You are a robotics video generation assistant...",
        current_negative_prompt="blurry, low quality, collision",
        validation_result=mock_validation,
        iteration=1,
        max_iterations=5
    )

    print("\n" + "="*70)
    print("OPTIMIZATION RESULT")
    print("="*70)
    print(f"\nReasoning:")
    print(f"  Analysis: {result.reasoning['analysis']}")
    print(f"  Strategy: {result.reasoning['strategy']}")
    print(f"\nExpected Improvements:")
    for dim, improvement in result.expected_improvements.items():
        print(f"  {dim}: {improvement}")
    print(f"\nConfidence: {result.confidence:.0%}")
    print(f"\nSystem Prompt (preview):")
    print(f"  {result.system_prompt[:200]}...")
    print(f"\nNegative Prompt:")
    print(f"  {result.negative_prompt[:150]}...")
    print(f"\nCost: ${result.cost_usd:.4f}")
    print("\n" + "="*70)
