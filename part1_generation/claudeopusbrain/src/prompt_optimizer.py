# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Abstract Base Class for Prompt Optimizers

This module defines the interface that all prompt optimizer implementations
must follow, whether using Claude, open-source LLMs, or RL-based methods.

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-02-07
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OptimizationResult:
    """Result from prompt optimization"""
    system_prompt: str
    negative_prompt: str
    reasoning: Dict[str, str]
    expected_improvements: Dict[str, str]
    confidence: float
    raw_response: str
    cost_usd: float
    model_name: str


class PromptOptimizer(ABC):
    """
    Abstract base class for all prompt optimizer implementations

    All optimizers (Claude Opus, Sonnet, Llama, etc.) must implement this interface.
    """

    def __init__(self, **kwargs):
        """
        Initialize optimizer

        Args:
            **kwargs: Implementation-specific parameters
        """
        self.iteration_history: List[Dict] = []
        self.total_cost_usd: float = 0.0
        # Memory context strings — set by orchestrator, injected into prompts
        self.memory_context: str = ""          # Long memory (cross-task rules)
        self.short_memory_context: str = ""    # Short memory (this task's strategy history)

    @abstractmethod
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
        Main optimization function: Analyze validation and improve prompts

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
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model being used"""
        pass

    def get_summary(self) -> Dict:
        """
        Get summary of optimization session
        """
        return {
            'model': self.get_model_name(),
            'total_iterations': len(self.iteration_history),
            'total_cost_usd': self.total_cost_usd,
            'history': self.iteration_history
        }

    def save_history(self, filepath: str):
        """
        Save optimization history to JSON file
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"💾 Saved optimization history to {filepath}")

    def _add_to_history(self, iteration: int, task_description: str,
                       task_type: str, current_prompts: Dict,
                       result: OptimizationResult):
        """
        Add optimization result to history
        """
        import time
        self.iteration_history.append({
            'iteration': iteration,
            'task': task_description,
            'task_type': task_type,
            'model': self.get_model_name(),
            'input': current_prompts,
            'output': {
                'system_prompt': result.system_prompt,
                'negative_prompt': result.negative_prompt,
                'reasoning': result.reasoning,
                'expected_improvements': result.expected_improvements,
                'confidence': result.confidence,
                'cost_usd': result.cost_usd
            },
            'timestamp': time.time()
        })


    def _summarize_history(self) -> str:
        """Summarize previous iterations for context (shared across all optimizers)."""
        if not self.iteration_history:
            return ""

        summary = "\n## Iteration History\n\n"
        for i, hist in enumerate(self.iteration_history):
            iter_num = hist['iteration']
            validation = hist['input']['validation']
            components = validation.get('components', [])
            avg = sum(c['score'] for c in components) / len(components) if components else 0

            summary += f"### Iteration {iter_num}\n"
            summary += f"- Average Score: {avg:.1f}/100\n"

            if 'output' in hist and 'reasoning' in hist['output']:
                strategy = hist['output']['reasoning'].get('strategy', 'N/A')
                summary += f"- Strategy Applied: {strategy}\n"

            if i < len(self.iteration_history) - 1:
                next_val = self.iteration_history[i + 1]['input']['validation']
                next_comps = next_val.get('components', [])
                next_avg = sum(c['score'] for c in next_comps) / len(next_comps) if next_comps else 0
                improvement = next_avg - avg
                summary += f"- Result: {'+' if improvement > 0 else ''}{improvement:.1f} points\n"
            else:
                summary += "- Result: Current iteration\n"

            summary += "\n"

        summary += "**Pattern Recognition**: Look for what worked and what didn't across iterations.\n"
        return summary

    def _get_full_memory_section(self) -> str:
        """Combine long memory, short memory, and iteration history into one prompt section."""
        sections = []
        if self.memory_context:
            sections.append(self.memory_context)
        if self.short_memory_context:
            sections.append(self.short_memory_context)
        history = self._summarize_history()
        if history:
            sections.append(history)
        return "\n".join(sections)


class OptimizerFactory:
    """
    Factory for creating optimizer instances
    """

    @staticmethod
    def create_optimizer(
        model_type: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> PromptOptimizer:
        """
        Create an optimizer instance based on model type

        Args:
            model_type: Type of model ('opus', 'sonnet', 'llama', 'qwen', etc.)
            api_key: API key if needed (for Claude models)
            **kwargs: Additional model-specific parameters

        Returns:
            Optimizer instance
        """
        model_type = model_type.lower()

        if model_type in ['opus', 'claude-opus', 'opus-4']:
            from claude_brain import ClaudeBrain
            return ClaudeBrain(
                api_key=api_key,
                model="claude-opus-4-20250514",
                **kwargs
            )

        elif model_type in ['sonnet', 'claude-sonnet', 'sonnet-4']:
            from claude_brain import ClaudeBrain
            return ClaudeBrain(
                api_key=api_key,
                model="claude-sonnet-4-5-20250929",
                **kwargs
            )

        elif model_type in ['claude-code', 'code']:
            from claude_code_brain import ClaudeCodeBrain
            return ClaudeCodeBrain(model="opus", **kwargs)

        elif model_type in ['claude-code-sonnet', 'code-sonnet']:
            from claude_code_brain import ClaudeCodeBrain
            return ClaudeCodeBrain(model="sonnet", **kwargs)

        elif model_type in ['llama', 'llama-3.1', 'llama-70b']:
            from opensource_optimizer import LlamaOptimizer
            return LlamaOptimizer(model_name="llama3.1:70b", **kwargs)

        elif model_type in ['qwen', 'qwen-2.5', 'qwen-72b']:
            from opensource_optimizer import QwenOptimizer
            return QwenOptimizer(model_name="qwen2.5:72b", **kwargs)

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported: opus, sonnet, claude-code, claude-code-sonnet, llama, qwen"
            )
