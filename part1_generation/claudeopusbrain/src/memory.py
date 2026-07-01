# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Two-Tier Memory System for Prompt Optimization

Short Memory: Tracks strategy effectiveness within a single task's iterations.
Long Memory:  Persists learned rules across tasks (JSON file).

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-03-29
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Short Memory – per-task, lives for 5 iterations
# ---------------------------------------------------------------------------

@dataclass
class StrategyOutcome:
    iteration: int
    strategy: str
    score_delta: float
    per_dim_delta: Dict[str, float]
    helped: List[str]
    hurt: List[str]


class ShortMemoryTracker:
    """Track strategy effectiveness within one task's optimization loop."""

    def __init__(self):
        self.iteration_scores: List[Dict] = []
        self.strategy_outcomes: List[StrategyOutcome] = []
        self.candidate_rules: List[str] = []

    def record_iteration(
        self,
        iteration: int,
        scores: Dict[str, float],
        strategy: str = "",
        prev_scores: Optional[Dict[str, float]] = None,
    ):
        """Record scores for an iteration and compute deltas vs previous."""
        avg = sum(scores.values()) / len(scores) if scores else 0
        entry = {"iteration": iteration, "avg": round(avg, 1), **scores}
        self.iteration_scores.append(entry)

        if prev_scores and strategy:
            per_dim = {
                k: round(scores.get(k, 0) - prev_scores.get(k, 0), 1)
                for k in scores
            }
            delta = round(avg - (sum(prev_scores.values()) / len(prev_scores)), 1)
            helped = [k for k, v in per_dim.items() if v > 2]
            hurt = [k for k, v in per_dim.items() if v < -2]
            outcome = StrategyOutcome(
                iteration=iteration,
                strategy=strategy,
                score_delta=delta,
                per_dim_delta=per_dim,
                helped=helped,
                hurt=hurt,
            )
            self.strategy_outcomes.append(outcome)

            # Auto-detect strong patterns as candidate rules
            if delta >= 10:
                self.candidate_rules.append(
                    f"Strategy '{strategy[:120]}' improved score by +{delta} pts "
                    f"(helped: {', '.join(helped) or 'overall'})"
                )
            elif delta <= -5:
                self.candidate_rules.append(
                    f"ANTI-PATTERN: Strategy '{strategy[:120]}' hurt score by {delta} pts "
                    f"(hurt: {', '.join(hurt) or 'overall'})"
                )

    def get_summary_for_prompt(self) -> str:
        """Formatted text block for injection into optimizer prompt."""
        if not self.strategy_outcomes:
            return ""

        lines = ["## This Task's Strategy History\n"]
        for o in self.strategy_outcomes:
            emoji = "+" if o.score_delta > 0 else ""
            lines.append(f"### Iteration {o.iteration - 1} → {o.iteration}")
            lines.append(f"- Strategy: \"{o.strategy}\"")
            lines.append(f"- Score delta: {emoji}{o.score_delta} pts "
                         f"({', '.join(f'{k}: {emoji}{v}' for k, v in o.per_dim_delta.items())})")
            if o.helped:
                lines.append(f"- Helped: {', '.join(o.helped)}")
            if o.hurt:
                lines.append(f"- Hurt: {', '.join(o.hurt)}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "iteration_scores": self.iteration_scores,
            "strategy_outcomes": [asdict(o) for o in self.strategy_outcomes],
            "candidate_rules": self.candidate_rules,
        }


# ---------------------------------------------------------------------------
# Long Memory – persistent across tasks
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    id: str
    text: str
    category: str  # task_type_pattern | model_pattern | negative_prompt_pattern | anti_pattern
    task_types: List[str]
    confidence: float
    source_tasks: List[Dict]
    times_tested: int = 0
    times_helped: int = 0
    times_hurt: int = 0
    created: float = field(default_factory=time.time)
    last_confirmed: float = field(default_factory=time.time)


class LongMemory:
    """Persistent cross-task memory stored as JSON."""

    def __init__(self, memory_dir: str = "./memory"):
        self.memory_dir = Path(memory_dir)
        self.filepath = self.memory_dir / "long_memory.json"
        self.rules: List[Dict] = []
        self.stats: Dict = {"total_tasks_completed": 0, "total_rules": 0}
        self._load()

    # -- persistence --

    def _load(self):
        if self.filepath.exists():
            try:
                data = json.loads(self.filepath.read_text())
                self.rules = data.get("rules", [])
                self.stats = data.get("stats", self.stats)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Long memory corrupted ({e}), starting fresh")
                self.rules = []

    def save(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.stats["total_rules"] = len(self.rules)
        data = {
            "version": 1,
            "last_updated": time.time(),
            "rules": self.rules,
            "stats": self.stats,
        }
        self.filepath.write_text(json.dumps(data, indent=2))

    # -- query --

    def get_relevant_rules(self, task_type: str, top_k: int = 10,
                           min_confidence: float = 0.4) -> str:
        """Return formatted text of rules relevant to task_type for prompt injection."""
        relevant = [
            r for r in self.rules
            if task_type in r.get("task_types", []) or "all" in r.get("task_types", [])
        ]
        relevant = [r for r in relevant if r.get("confidence", 0) >= min_confidence]
        relevant.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        relevant = relevant[:top_k]

        if not relevant:
            return ""

        positive = [r for r in relevant if r.get("category") != "anti_pattern"]
        negative = [r for r in relevant if r.get("category") == "anti_pattern"]

        lines = [f"## Learned Rules from Previous Tasks ({len(relevant)} relevant)\n"]

        if positive:
            lines.append(f"**Relevant to '{task_type}' tasks:**")
            for i, r in enumerate(positive, 1):
                conf = r.get("confidence", 0)
                tested = r.get("times_tested", 0)
                lines.append(f"{i}. [{conf:.2f}] {r['text']} (tested {tested}x)")
            lines.append("")

        if negative:
            lines.append("**Anti-patterns to AVOID:**")
            for i, r in enumerate(negative, 1):
                conf = r.get("confidence", 0)
                lines.append(f"{i}. [{conf:.2f}] {r['text']}")
            lines.append("")

        lines.append("Use these rules to guide optimization. "
                      "Do not blindly follow rules with confidence < 0.6.\n")
        return "\n".join(lines)

    # -- mutation --

    def add_rules(self, new_rules: List[Dict], source_task: str, task_type: str):
        """Add new rules, merging with existing if text is similar."""
        for nr in new_rules:
            merged = False
            nr_text_lower = nr.get("text", "").lower()
            for existing in self.rules:
                # Simple overlap check: if >60% words overlap, merge
                ex_words = set(existing.get("text", "").lower().split())
                nr_words = set(nr_text_lower.split())
                if len(ex_words & nr_words) > 0.6 * max(len(ex_words), len(nr_words), 1):
                    # Merge: bump confidence, add source
                    existing["times_tested"] = existing.get("times_tested", 0) + 1
                    existing["times_helped"] = existing.get("times_helped", 0) + 1
                    existing["confidence"] = min(
                        0.99,
                        existing.get("confidence", 0.5) + 0.05
                    )
                    existing.setdefault("source_tasks", []).append(
                        {"task": source_task, "task_type": task_type}
                    )
                    existing["last_confirmed"] = time.time()
                    merged = True
                    break

            if not merged:
                rule = {
                    "id": f"rule_{int(time.time())}_{len(self.rules)}",
                    "text": nr.get("text", ""),
                    "category": nr.get("category", "task_type_pattern"),
                    "task_types": nr.get("task_types", [task_type]),
                    "confidence": nr.get("confidence", 0.5),
                    "source_tasks": [{"task": source_task, "task_type": task_type}],
                    "times_tested": 1,
                    "times_helped": 1,
                    "times_hurt": 0,
                    "created": time.time(),
                    "last_confirmed": time.time(),
                }
                self.rules.append(rule)

    def record_task_completion(self):
        self.stats["total_tasks_completed"] = self.stats.get("total_tasks_completed", 0) + 1

    def prune(self, max_rules: int = 50, min_confidence: float = 0.25,
              min_tests: int = 2):
        """Remove low-confidence rules; enforce hard cap."""
        # Remove rules with enough tests but low confidence
        self.rules = [
            r for r in self.rules
            if not (r.get("times_tested", 0) >= min_tests
                    and r.get("confidence", 0) < min_confidence)
        ]
        # Hard cap: keep highest confidence
        if len(self.rules) > max_rules:
            self.rules.sort(key=lambda r: r.get("confidence", 0), reverse=True)
            self.rules = self.rules[:max_rules]


# ---------------------------------------------------------------------------
# Rule Extraction – after task completion
# ---------------------------------------------------------------------------

def extract_rules_heuristic(short_memory: ShortMemoryTracker,
                            task_type: str) -> List[Dict]:
    """Extract rules from short memory using simple heuristics (no LLM call)."""
    rules = []
    for candidate in short_memory.candidate_rules:
        is_anti = candidate.startswith("ANTI-PATTERN")
        rules.append({
            "text": candidate,
            "category": "anti_pattern" if is_anti else "task_type_pattern",
            "task_types": [task_type],
            "confidence": 0.5,
        })

    # If the same dimension was worst in >60% of iterations, note it
    if len(short_memory.iteration_scores) >= 3:
        dim_worst_counts: Dict[str, int] = {}
        for entry in short_memory.iteration_scores:
            scores_only = {k: v for k, v in entry.items()
                          if k not in ("iteration", "avg")}
            if scores_only:
                worst = min(scores_only, key=scores_only.get)
                dim_worst_counts[worst] = dim_worst_counts.get(worst, 0) + 1
        for dim, count in dim_worst_counts.items():
            if count >= len(short_memory.iteration_scores) * 0.6:
                rules.append({
                    "text": f"'{dim}' is a persistent bottleneck for {task_type} tasks — "
                            f"focus optimization effort here",
                    "category": "task_type_pattern",
                    "task_types": [task_type],
                    "confidence": 0.6,
                })
    return rules


def extract_rules_via_llm(optimizer, short_memory: ShortMemoryTracker,
                          task_description: str, task_type: str) -> List[Dict]:
    """Ask the optimizer LLM to extract generalizable rules from a completed task.

    Falls back to heuristic extraction on any failure.
    """
    history_text = short_memory.get_summary_for_prompt()
    scores_text = json.dumps(short_memory.iteration_scores, indent=2)

    extraction_prompt = f"""Given this completed optimization task, extract 1-5 generalizable rules
that could help with FUTURE tasks of the same or similar type.

Task: "{task_description}" (type: {task_type})

Score progression:
{scores_text}

Strategy outcomes:
{history_text}

Rules should be:
- Specific enough to be actionable (name exact prompt changes)
- General enough to apply beyond this single task
- Include score evidence

Output ONLY a JSON array:
[{{"text": "...", "category": "task_type_pattern|model_pattern|negative_prompt_pattern|anti_pattern", "task_types": ["{task_type}"], "confidence": 0.0-1.0}}]"""

    try:
        # Use the optimizer's own LLM to extract rules (cheapest viable call)
        if hasattr(optimizer, 'client'):
            # Claude API path
            response = optimizer.client.messages.create(
                model=optimizer.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": extraction_prompt}],
            )
            text = response.content[0].text.strip()
            cost = optimizer._calculate_cost(
                response.usage.input_tokens, response.usage.output_tokens
            )
            optimizer.total_cost_usd += cost
        elif hasattr(optimizer, '_call_claude_code'):
            # Claude Code CLI path
            text = optimizer._call_claude_code(extraction_prompt)
        elif hasattr(optimizer, 'server_url'):
            # Ollama path
            import requests
            resp = requests.post(
                f"{optimizer.server_url}/api/generate",
                json={"model": optimizer.model_name, "prompt": extraction_prompt,
                      "stream": False},
                timeout=120,
            )
            text = resp.json().get("response", "[]")
        else:
            return extract_rules_heuristic(short_memory, task_type)

        # Parse JSON from response
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        rules = json.loads(text)
        if not isinstance(rules, list):
            raise ValueError("Expected JSON array")

        # Validate and clean
        cleaned = []
        for r in rules:
            if isinstance(r, dict) and "text" in r:
                cleaned.append({
                    "text": str(r["text"]),
                    "category": r.get("category", "task_type_pattern"),
                    "task_types": r.get("task_types", [task_type]),
                    "confidence": min(1.0, max(0.0, float(r.get("confidence", 0.5)))),
                })
        return cleaned if cleaned else extract_rules_heuristic(short_memory, task_type)

    except Exception as e:
        print(f"   ⚠️  LLM rule extraction failed ({e}), using heuristic fallback")
        return extract_rules_heuristic(short_memory, task_type)


def extract_rules_from_task(optimizer, short_memory: ShortMemoryTracker,
                            task_description: str, task_type: str,
                            use_llm: bool = True) -> List[Dict]:
    """Main entry point: extract rules after a task completes."""
    # Always get heuristic rules (free)
    heuristic = extract_rules_heuristic(short_memory, task_type)

    if use_llm and short_memory.strategy_outcomes:
        llm_rules = extract_rules_via_llm(
            optimizer, short_memory, task_description, task_type
        )
        # Merge: LLM rules take priority, heuristic fills gaps
        all_texts = {r["text"].lower() for r in llm_rules}
        for hr in heuristic:
            if hr["text"].lower() not in all_texts:
                llm_rules.append(hr)
        return llm_rules

    return heuristic
