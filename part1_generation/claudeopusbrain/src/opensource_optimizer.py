# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Open-Source LLM Optimizers: Free alternatives to Claude

This module implements prompt optimizers using open-source LLMs via the custom
Ollama server at the OLLAMA_SERVER env var (default localhost:11434).

Server API:
    GET  /health   - Check server status
    GET  /models   - List available models
    POST /generate - Generate response
    GET  /config   - Server configuration

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-02-07
"""

import os
import json
import time
import requests
from typing import Dict, Optional
from prompt_optimizer import PromptOptimizer, OptimizationResult


# Custom Ollama server address
OLLAMA_SERVER = os.getenv("OLLAMA_SERVER", "http://localhost:11434")


def check_server() -> bool:
    """Check if the Ollama server is running"""
    try:
        resp = requests.get(f"{OLLAMA_SERVER}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def list_models() -> list:
    """List available models on the server"""
    try:
        resp = requests.get(f"{OLLAMA_SERVER}/models", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass
    return []


class LlamaOptimizer(PromptOptimizer):
    """
    Optimizer using Meta's Llama 3.1 via custom Ollama server at the OLLAMA_SERVER env var (default localhost:11434)

    Server API format:
        POST /generate
        Body: {"prompt": "...", "model": "llama3.1:70b"}
        Response: {"response": "...", "tokens_per_sec": ...}
    """

    def __init__(
        self,
        model_name: str = "llama3.1:70b",
        server: str = OLLAMA_SERVER,
        temperature: float = 0.7,
        timeout: int = 600
    ):
        """
        Initialize Llama optimizer

        Args:
            model_name: Model name on the server (e.g., "llama3.1:70b")
            server: Server address (default: the OLLAMA_SERVER env var (default http://localhost:11434))
            temperature: Temperature for generation
            timeout: Request timeout in seconds (600 = 10 minutes for large models)
        """
        super().__init__()

        self.model_name = model_name
        self.server = server
        self.temperature = temperature
        self.timeout = timeout

        # Health check: verify server is reachable
        if not check_server():
            raise RuntimeError(
                f"❌ Ollama server not reachable at {self.server}\n"
                f"   Make sure the server is running and accessible.\n"
                f"   Test with: curl {self.server}/health"
            )

        # System prompt for Llama (injected into prompt text)
        self.system_prompt = self._get_llama_system_prompt()

    def get_model_name(self) -> str:
        """Return the name of the model being used"""
        return f"Llama 3.1 ({self.model_name})"

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
        Optimize prompts using Llama model
        """
        # Build optimization prompt
        user_prompt = self._build_optimization_prompt(
            task_description=task_description,
            task_type=task_type,
            current_system_prompt=current_system_prompt,
            current_negative_prompt=current_negative_prompt,
            validation_result=validation_result,
            iteration=iteration,
            max_iterations=max_iterations
        )

        print(f"🦙 Calling {self.model_name} @ {self.server} (iteration {iteration})...")
        start_time = time.time()

        try:
            # Call via custom Ollama server API
            response, tokens_per_sec = self._call_server(user_prompt)
            elapsed_time = time.time() - start_time

            print(f"   Time: {elapsed_time:.1f}s  |  Speed: {tokens_per_sec:.1f} tokens/sec")
            print(f"   Cost: $0.00 (free/local)")

            # Parse response (returns None if parsing failed)
            result = self._parse_response(response, cost_usd=0.0)

            # If parsing failed, create safe fallback (keeps prompts unchanged)
            if result is None:
                print(f"   ⚠️  Creating fallback result (prompts unchanged)")
                result = OptimizationResult(
                    system_prompt=current_system_prompt,  # Keep unchanged
                    negative_prompt=current_negative_prompt,  # Keep unchanged
                    reasoning={
                        'analysis': f'{self.get_model_name()} response could not be parsed',
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
                    raw_response=response,
                    cost_usd=0.0,
                    model_name=self.get_model_name()
                )

            # Save to history
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
            print(f"❌ Error calling Llama: {e}")
            raise

    def _call_server(self, prompt: str):
        """
        Call the custom Ollama server at the OLLAMA_SERVER env var (default localhost:11434)

        API format:
            POST /generate
            Body: {"prompt": "...", "model": "llama3.1:70b"}
            Response: {"response": "...", "tokens_per_sec": ...}
        """
        # Combine system prompt + user prompt (server has no system field)
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        payload = {
            "prompt": full_prompt,
            "model": self.model_name
        }

        try:
            resp = requests.post(
                f"{self.server}/generate",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()

            data = resp.json()
            response_text = data.get("response", "")
            tokens_per_sec = data.get("tokens_per_sec", 0.0)

            if not response_text:
                raise ValueError("Empty response from server")

            return response_text, tokens_per_sec

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Server timeout after {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Server call failed: {e}")

    def _get_llama_system_prompt(self) -> str:
        """
        System prompt for Llama (adapted from Claude's version)
        """
        return """You are an expert optimization agent for robotics video generation.

Your role: Analyze video validation feedback and improve two types of prompts:
1. System prompt (for Qwen3-VL): Guides prompt enhancement
2. Negative prompt (for video model): Specifies what to avoid

Process:
1. Analyze validation scores (adherence, physics, quality)
2. Identify root causes of failures
3. Reason about multi-objective tradeoffs
4. Generate improved prompts (targeted, surgical changes)
5. Explain reasoning clearly

Output ONLY valid JSON in this exact format:
{
  "reasoning": {
    "analysis": "What went wrong and why? (2-3 sentences)",
    "root_causes": ["cause 1", "cause 2"],
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

Critical Rules:
- Return ONLY valid JSON - no markdown, no extra text
- System prompt must be COMPLETE (not a diff)
- Negative prompt must be COMPLETE (not a diff)
- Focus on lowest-scoring dimensions first
- Aim for 80+ average score as success threshold"""

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
        Build optimization prompt for Llama/Qwen
        """
        components = validation_result.get('components', [])
        scores_dict = {c['name']: c['score'] for c in components}
        avg_score = sum(scores_dict.values()) / len(scores_dict) if scores_dict else 0

        # Score summary with pass/fail indicator
        score_summary = "\n".join([
            f"  - {comp['name']}: {comp['score']}/100 {'OK' if comp['score'] >= 70 else 'FAIL'}"
            for comp in components
        ])

        # Bottleneck priority signal
        if scores_dict:
            worst_name = min(scores_dict, key=scores_dict.get)
            best_name = max(scores_dict, key=scores_dict.get)
            gap = scores_dict[best_name] - scores_dict[worst_name]
            priority_signal = (
                f"\nBOTTLENECK: '{worst_name}' = {scores_dict[worst_name]}/100 "
                f"(gap: {gap} pts). Fix this first.\n"
            )
        else:
            priority_signal = ""

        # Detailed analysis from validator
        detailed_analysis = "\n\n".join([
            f"{comp['name']} ({comp['score']}/100):\n{comp.get('analysis', 'N/A')}"
            for comp in components
        ])

        # Full memory context (long memory + short memory + iteration history)
        memory_section = self._get_full_memory_section()

        return f"""# Video Prompt Optimization Task

Task: "{task_description}" (type: {task_type})
Iteration: {iteration}/{max_iterations}

## Scores
{score_summary}
Average: {avg_score:.1f}/100  Target: 80/100
{priority_signal}
## Detailed Feedback
{detailed_analysis}

## Current Prompts
System: {current_system_prompt}

Negative: {current_negative_prompt}
{memory_section}
---
Fix the bottleneck. Output ONLY JSON:
{{"reasoning": {{"analysis": "...", "root_causes": ["..."], "strategy": "...", "tradeoffs": "..."}}, "system_prompt": "COMPLETE prompt here", "negative_prompt": "comma-separated list", "expected_improvements": {{"adherence": "+X or maintain", "physics": "+X or maintain", "quality": "+X or maintain"}}, "confidence": 0.75}}"""

    def _parse_response(self, response_text: str, cost_usd: float) -> OptimizationResult:
        """
        Parse Llama's JSON response with robust fallback

        If parsing fails, returns a safe result that keeps prompts unchanged
        rather than crashing the entire self-tuning loop.
        """
        try:
            # Remove markdown code blocks if present
            text = response_text.strip()
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
            print(f"⚠️  Failed to parse {self.get_model_name()} response: {e}")
            print(f"   Response preview: {response_text[:300]}...")
            print(f"   🔄 Using SAFE FALLBACK: keeping current prompts unchanged")

            # Return safe fallback — keeps prompts unchanged, loop continues
            return None  # Signal to caller that parsing failed


class QwenOptimizer(LlamaOptimizer):
    """
    Optimizer using Alibaba's Qwen 2.5 via the same custom Ollama server.

    Inherits from LlamaOptimizer since the server API is identical.
    Just uses a different model name in the request.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:72b",
        server: str = OLLAMA_SERVER,
        temperature: float = 0.7,
        timeout: int = 600
    ):
        super().__init__(
            model_name=model_name,
            server=server,
            temperature=temperature,
            timeout=timeout
        )

    def get_model_name(self) -> str:
        return f"Qwen 2.5 ({self.model_name})"


# Test / example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Testing custom Ollama server connection")
    print(f"Server: {OLLAMA_SERVER}")
    print("=" * 60)

    # 1. Check server health
    print("\n1. Health check...")
    if check_server():
        print("   ✅ Server is running")
    else:
        print(f"   ❌ Server not reachable at {OLLAMA_SERVER}")
        print("   Make sure the server is running!")
        exit(1)

    # 2. List available models
    print("\n2. Available models:")
    models = list_models()
    if models:
        for m in models:
            print(f"   - {m}")
    else:
        print("   (none found)")

    # 3. Test optimization with mock validation
    print("\n3. Testing optimization with mock data...")
    optimizer = LlamaOptimizer(model_name="llama3.1:70b")

    mock_validation = {
        'pass': False,
        'confidence': 65,
        'components': [
            {'name': 'Prompt Adherence', 'score': 72,
             'analysis': 'Task mostly followed, arm selection unclear.'},
            {'name': 'Physical Plausibility', 'score': 58,
             'analysis': 'Collision at frame 45.'},
            {'name': 'Visual Quality', 'score': 75,
             'analysis': 'Good quality, minor flickering.'}
        ]
    }

    try:
        result = optimizer.optimize_prompts(
            task_description="Humanoid picks up bottle",
            task_type="g1",
            current_system_prompt="You are a robotics video generation assistant...",
            current_negative_prompt="blurry, low quality, collision",
            validation_result=mock_validation,
            iteration=1,
            max_iterations=5
        )

        print(f"\n   ✅ Optimization complete!")
        print(f"   Model: {result.model_name}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Strategy: {result.reasoning['strategy'][:100]}...")

    except Exception as e:
        print(f"\n   ❌ Test failed: {e}")
