#!/usr/bin/env python3
"""
Comprehensive integration test for the two-tier memory system.

Tests:
1. ShortMemoryTracker: recording iterations, computing deltas, candidate rules
2. LongMemory: save/load, add rules, merge duplicates, prune, relevance filter
3. Rule extraction: heuristic fallback
4. Base class: _summarize_history, _get_full_memory_section
5. End-to-end: simulate a 5-iteration task with memory flowing through

Run: python3 tests/test_memory_system.py
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from memory import ShortMemoryTracker, LongMemory, extract_rules_heuristic
from prompt_optimizer import PromptOptimizer, OptimizationResult, OptimizerFactory

PASSED = 0
FAILED = 0

def test(name):
    def decorator(fn):
        def wrapper():
            global PASSED, FAILED
            try:
                fn()
                print(f"  ✅ {name}")
                PASSED += 1
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                FAILED += 1
        return wrapper
    return decorator


# -----------------------------------------------------------------------
# 1. ShortMemoryTracker tests
# -----------------------------------------------------------------------
print("\n=== ShortMemoryTracker ===")

@test("Record first iteration (no delta)")
def test_sm_first():
    sm = ShortMemoryTracker()
    sm.record_iteration(1, {"Adherence": 60, "Physics": 45, "Quality": 70})
    assert len(sm.iteration_scores) == 1
    assert sm.iteration_scores[0]["avg"] == 58.3  # (60+45+70)/3
    assert len(sm.strategy_outcomes) == 0  # No prev to compare
test_sm_first()

@test("Record second iteration with delta tracking")
def test_sm_second():
    sm = ShortMemoryTracker()
    sm.record_iteration(1, {"Adherence": 60, "Physics": 45, "Quality": 70})
    sm.record_iteration(
        2,
        {"Adherence": 65, "Physics": 60, "Quality": 72},
        strategy="Added force causality chain",
        prev_scores={"Adherence": 60, "Physics": 45, "Quality": 70},
    )
    assert len(sm.strategy_outcomes) == 1
    o = sm.strategy_outcomes[0]
    assert o.score_delta == 7.3  # (65+60+72)/3 - (60+45+70)/3
    assert "Physics" in o.helped  # +15 > 2
    assert o.hurt == []
test_sm_second()

@test("Large delta generates candidate rule")
def test_sm_candidate_rule():
    sm = ShortMemoryTracker()
    sm.record_iteration(1, {"A": 40, "B": 40, "C": 40})
    sm.record_iteration(
        2, {"A": 55, "B": 55, "C": 55},
        strategy="Big improvement strategy",
        prev_scores={"A": 40, "B": 40, "C": 40},
    )
    assert len(sm.candidate_rules) == 1
    assert "+15.0 pts" in sm.candidate_rules[0]
test_sm_candidate_rule()

@test("Negative delta generates anti-pattern rule")
def test_sm_anti_pattern():
    sm = ShortMemoryTracker()
    sm.record_iteration(1, {"A": 70, "B": 70, "C": 70})
    sm.record_iteration(
        2, {"A": 60, "B": 60, "C": 60},
        strategy="Bad move",
        prev_scores={"A": 70, "B": 70, "C": 70},
    )
    assert len(sm.candidate_rules) == 1
    assert "ANTI-PATTERN" in sm.candidate_rules[0]
test_sm_anti_pattern()

@test("get_summary_for_prompt returns formatted text")
def test_sm_summary():
    sm = ShortMemoryTracker()
    sm.record_iteration(1, {"A": 50, "B": 50})
    sm.record_iteration(2, {"A": 60, "B": 70}, strategy="Try X", prev_scores={"A": 50, "B": 50})
    text = sm.get_summary_for_prompt()
    assert "Strategy History" in text
    assert "Try X" in text
    assert "Helped:" in text
test_sm_summary()

@test("to_dict serializes correctly")
def test_sm_to_dict():
    sm = ShortMemoryTracker()
    sm.record_iteration(1, {"A": 50})
    d = sm.to_dict()
    assert "iteration_scores" in d
    assert "strategy_outcomes" in d
    assert "candidate_rules" in d
    json.dumps(d)  # Must be JSON-serializable
test_sm_to_dict()


# -----------------------------------------------------------------------
# 2. LongMemory tests
# -----------------------------------------------------------------------
print("\n=== LongMemory ===")

@test("Create empty long memory (no file)")
def test_lm_empty():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        assert lm.rules == []
        assert lm.stats["total_tasks_completed"] == 0
test_lm_empty()

@test("Save and reload long memory")
def test_lm_save_load():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        lm.add_rules([{"text": "Test rule", "category": "task_type_pattern",
                        "task_types": ["g1"], "confidence": 0.7}],
                      source_task="test", task_type="g1")
        lm.save()

        lm2 = LongMemory(memory_dir=td)
        assert len(lm2.rules) == 1
        assert lm2.rules[0]["text"] == "Test rule"
test_lm_save_load()

@test("Merge duplicate rules (>60% word overlap)")
def test_lm_merge():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        lm.add_rules([{"text": "Adding force causality chains improves physics scores",
                        "task_types": ["g1"], "confidence": 0.5}],
                      source_task="task1", task_type="g1")
        lm.add_rules([{"text": "Adding force causality chains improves physics significantly",
                        "task_types": ["g1"], "confidence": 0.6}],
                      source_task="task2", task_type="g1")
        assert len(lm.rules) == 1  # Merged, not duplicated
        assert lm.rules[0]["times_tested"] == 2
        assert lm.rules[0]["confidence"] > 0.5  # Bumped
test_lm_merge()

@test("get_relevant_rules filters by task_type")
def test_lm_filter():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        lm.add_rules([{"text": "G1 rule", "task_types": ["g1"], "confidence": 0.7}],
                      source_task="t1", task_type="g1")
        lm.add_rules([{"text": "Drone rule", "task_types": ["drone"], "confidence": 0.8}],
                      source_task="t2", task_type="drone")
        g1_text = lm.get_relevant_rules("g1")
        assert "G1 rule" in g1_text
        assert "Drone rule" not in g1_text
test_lm_filter()

@test("get_relevant_rules returns empty for no matches")
def test_lm_no_match():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        lm.add_rules([{"text": "G1 rule", "task_types": ["g1"], "confidence": 0.7}],
                      source_task="t1", task_type="g1")
        assert lm.get_relevant_rules("ur3") == ""
test_lm_no_match()

@test("Anti-patterns shown separately in prompt")
def test_lm_anti_pattern():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        lm.add_rules([{"text": "Don't remove one-scene rule", "category": "anti_pattern",
                        "task_types": ["g1"], "confidence": 0.9}],
                      source_task="t1", task_type="g1")
        text = lm.get_relevant_rules("g1")
        assert "Anti-patterns to AVOID" in text
test_lm_anti_pattern()

@test("Prune removes low-confidence rules with enough tests")
def test_lm_prune():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        lm.rules = [
            {"text": "Bad rule", "confidence": 0.1, "times_tested": 5,
             "task_types": ["g1"]},
            {"text": "Good rule", "confidence": 0.9, "times_tested": 3,
             "task_types": ["g1"]},
        ]
        lm.prune(min_confidence=0.25, min_tests=2)
        assert len(lm.rules) == 1
        assert lm.rules[0]["text"] == "Good rule"
test_lm_prune()

@test("Prune enforces max_rules cap")
def test_lm_prune_cap():
    with tempfile.TemporaryDirectory() as td:
        lm = LongMemory(memory_dir=td)
        lm.rules = [
            {"text": f"Rule {i}", "confidence": i / 10.0, "times_tested": 1,
             "task_types": ["g1"]}
            for i in range(20)
        ]
        lm.prune(max_rules=5)
        assert len(lm.rules) == 5
        assert lm.rules[0]["confidence"] == 1.9  # Highest first
test_lm_prune_cap()

@test("Corrupted JSON doesn't crash")
def test_lm_corrupt():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "long_memory.json"
        p.write_text("{invalid json!!!")
        lm = LongMemory(memory_dir=td)
        assert lm.rules == []  # Graceful fallback
test_lm_corrupt()


# -----------------------------------------------------------------------
# 3. Heuristic rule extraction
# -----------------------------------------------------------------------
print("\n=== Heuristic Rule Extraction ===")

@test("Extract rules from strong improvements")
def test_heuristic_positive():
    sm = ShortMemoryTracker()
    sm.record_iteration(1, {"A": 40, "B": 40})
    sm.record_iteration(2, {"A": 55, "B": 55}, strategy="Good move",
                        prev_scores={"A": 40, "B": 40})
    rules = extract_rules_heuristic(sm, "g1")
    assert len(rules) >= 1
    assert any("Good move" in r["text"] for r in rules)
test_heuristic_positive()

@test("Extract persistent bottleneck rule")
def test_heuristic_bottleneck():
    sm = ShortMemoryTracker()
    for i in range(1, 6):
        sm.record_iteration(i, {"Adherence": 80, "Physics": 40, "Quality": 75})
    rules = extract_rules_heuristic(sm, "g1")
    assert any("Physics" in r["text"] and "bottleneck" in r["text"] for r in rules)
test_heuristic_bottleneck()


# -----------------------------------------------------------------------
# 4. Base class methods
# -----------------------------------------------------------------------
print("\n=== PromptOptimizer Base Class ===")

@test("_summarize_history with no history returns empty")
def test_base_empty_history():
    # Create a concrete subclass for testing
    class DummyOptimizer(PromptOptimizer):
        def optimize_prompts(self, **kwargs): pass
        def get_model_name(self): return "dummy"
    opt = DummyOptimizer()
    assert opt._summarize_history() == ""
test_base_empty_history()

@test("_get_full_memory_section combines all three")
def test_base_full_section():
    class DummyOptimizer(PromptOptimizer):
        def optimize_prompts(self, **kwargs): pass
        def get_model_name(self): return "dummy"
    opt = DummyOptimizer()
    opt.memory_context = "## Long Memory Rules\nRule 1: test"
    opt.short_memory_context = "## Short Memory\nIteration 1->2: +5"
    # Add a fake history entry
    opt.iteration_history = [{
        'iteration': 1,
        'input': {'validation': {'components': [{'score': 65}]}},
        'output': {'reasoning': {'strategy': 'try harder'}},
    }]
    section = opt._get_full_memory_section()
    assert "Long Memory Rules" in section
    assert "Short Memory" in section
    assert "Iteration History" in section
    assert "try harder" in section
test_base_full_section()

@test("memory_context and short_memory_context default to empty")
def test_base_defaults():
    class DummyOptimizer(PromptOptimizer):
        def optimize_prompts(self, **kwargs): pass
        def get_model_name(self): return "dummy"
    opt = DummyOptimizer()
    assert opt.memory_context == ""
    assert opt.short_memory_context == ""
test_base_defaults()


# -----------------------------------------------------------------------
# 5. End-to-end: simulate 5 iterations
# -----------------------------------------------------------------------
print("\n=== End-to-End Simulation ===")

@test("Full 5-iteration simulation with memory")
def test_e2e():
    with tempfile.TemporaryDirectory() as td:
        # 1. Start with empty long memory
        lm = LongMemory(memory_dir=td)
        assert lm.rules == []

        # 2. Simulate Task 1: 5 iterations
        sm = ShortMemoryTracker()
        scores_sequence = [
            {"Adherence": 50, "Physics": 35, "Quality": 55},
            {"Adherence": 65, "Physics": 55, "Quality": 70},  # +16.7 avg
            {"Adherence": 68, "Physics": 62, "Quality": 70},  # 0 avg
            {"Adherence": 72, "Physics": 70, "Quality": 75},  # +5.7 avg
            {"Adherence": 78, "Physics": 76, "Quality": 80},  # +5.7 avg
        ]
        strategies = [
            "",
            "Added force causality chain for grasp",
            "Added morphing to negative prompt",
            "Increased physics detail in prompt",
            "Fine-tuned balance compensation",
        ]

        for i, (scores, strat) in enumerate(zip(scores_sequence, strategies)):
            prev = scores_sequence[i - 1] if i > 0 else None
            sm.record_iteration(i + 1, scores, strategy=strat, prev_scores=prev)

        # 3. Verify short memory tracked everything
        assert len(sm.iteration_scores) == 5
        assert len(sm.strategy_outcomes) == 4  # 4 deltas (iter 2-5)
        assert sm.strategy_outcomes[0].score_delta >= 10  # First big jump (+16.7)
        assert len(sm.candidate_rules) >= 1  # The +16.7 jump should produce a rule

        # 4. Extract rules (heuristic only, no LLM)
        rules = extract_rules_heuristic(sm, "g1")
        assert len(rules) >= 1

        # 5. Save to long memory
        lm.add_rules(rules, source_task="G1 picks up bottle", task_type="g1")
        lm.record_task_completion()
        lm.save()

        # 6. Reload and verify
        lm2 = LongMemory(memory_dir=td)
        assert len(lm2.rules) >= 1
        assert lm2.stats["total_tasks_completed"] == 1

        # 7. Second task reads the rules
        relevant = lm2.get_relevant_rules("g1")
        assert len(relevant) > 0
        assert "force causality" in relevant.lower() or "pts" in relevant.lower()

        # 8. Verify prompt injection format
        class DummyOpt(PromptOptimizer):
            def optimize_prompts(self, **kw): pass
            def get_model_name(self): return "dummy"
        opt = DummyOpt()
        opt.memory_context = relevant
        opt.short_memory_context = sm.get_summary_for_prompt()
        section = opt._get_full_memory_section()
        assert "Learned Rules" in section
        assert "Strategy History" in section
test_e2e()


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {PASSED} passed, {FAILED} failed")
print(f"{'='*50}")
sys.exit(1 if FAILED > 0 else 0)
