#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Batch Experiment Runner: Run 100-task experiments for IROS 2026

This script runs self-tuning experiments on multiple tasks sequentially,
with automatic checkpoint/resume, progress tracking, and cost monitoring.

Usage:
    # Prepare tasks.json with your 100 tasks
    # Format: [{"task": "...", "task_type": "...", "image": "..."}, ...]

    export ANTHROPIC_API_KEY="your-key-here"
    python3 run_batch_experiments.py --tasks tasks.json --model opus --cost-budget 200

Features:
    - Automatic checkpoint/resume (crash-safe)
    - Progress tracking with ETA
    - Cost monitoring and warnings
    - Parallel execution on multiple GPUs (future)
    - Results aggregation and learning curves

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Target: IROS 2026 (Deadline: March 1, 2026)
Date: 2026-02-09
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import run_self_tuning from main script
from run_self_tuning import run_self_tuning


def load_tasks(tasks_file: str) -> List[Dict]:
    """
    Load tasks from JSON file

    Expected format:
    [
        {
            "task": "Humanoid picks up bottle",
            "task_type": "g1",
            "image": "./images/workspace1.jpg"
        },
        ...
    ]
    """
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)

    # Validate task format
    required_fields = ['task', 'task_type', 'image']
    for i, task in enumerate(tasks):
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Task {i} missing required field: {field}")

        # Verify image exists
        if not Path(task['image']).exists():
            raise ValueError(f"Task {i} image not found: {task['image']}")

    return tasks


def save_checkpoint(checkpoint_file: str, data: Dict):
    """Save experiment checkpoint"""
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load experiment checkpoint"""
    if not Path(checkpoint_file).exists():
        return None

    with open(checkpoint_file, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))


def run_batch_experiments(
    tasks_file: str,
    model_type: str = "opus",
    api_key: str = None,
    max_iterations: int = 5,
    success_threshold: float = 80.0,
    output_dir: str = "./results/batch",
    cost_budget: float = None,
    resume: bool = False
):
    """
    Run self-tuning experiments on multiple tasks

    Args:
        tasks_file: Path to tasks JSON file
        model_type: Optimizer model
        api_key: API key (for Claude)
        max_iterations: Max iterations per task
        success_threshold: Success threshold
        output_dir: Output directory
        cost_budget: Max total cost for all tasks (USD)
        resume: Resume from checkpoint if exists
    """
    # Load tasks
    tasks = load_tasks(tasks_file)

    # Create output directory
    batch_id = f"batch_{int(time.time())}"
    batch_dir = Path(output_dir).resolve() / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = batch_dir / "batch_checkpoint.json"

    # Resume from checkpoint if exists
    if resume:
        checkpoint = load_checkpoint(checkpoint_file)
        if checkpoint:
            print("=" * 70)
            print("🔄 RESUMING FROM CHECKPOINT")
            print("=" * 70)
            start_idx = checkpoint['next_task_idx']
            completed_tasks = checkpoint['completed_tasks']
            total_cost = checkpoint['total_cost']
            print(f"Resuming from task {start_idx}/{len(tasks)}")
            print(f"Completed: {len(completed_tasks)}/{len(tasks)}")
            print(f"Total cost so far: ${total_cost:.2f}")
            print()
        else:
            start_idx = 0
            completed_tasks = []
            total_cost = 0.0
    else:
        start_idx = 0
        completed_tasks = []
        total_cost = 0.0

    print("=" * 70)
    print(f"🧪 BATCH EXPERIMENTS FOR IROS 2026")
    print("=" * 70)
    print(f"Total tasks: {len(tasks)}")
    print(f"Optimizer: {model_type}")
    print(f"Max iterations per task: {max_iterations}")
    print(f"Success threshold: {success_threshold}/100")
    if cost_budget:
        print(f"Cost budget: ${cost_budget:.2f}")
    print(f"Output: {batch_dir}")
    print("=" * 70)
    print()

    start_time = time.time()

    # Run experiments
    for idx in range(start_idx, len(tasks)):
        task = tasks[idx]

        print("\n" + "=" * 70)
        print(f"📋 TASK {idx + 1}/{len(tasks)}")
        print("=" * 70)
        print(f"Description: {task['task']}")
        print(f"Type: {task['task_type']}")
        print(f"Image: {task['image']}")

        # Calculate remaining budget for this task
        if cost_budget:
            remaining_tasks = len(tasks) - idx
            remaining_budget = cost_budget - total_cost
            task_budget = remaining_budget / remaining_tasks

            print(f"💰 Budget for this task: ${task_budget:.2f}")
            print(f"   (Remaining: ${remaining_budget:.2f} for {remaining_tasks} tasks)")

            if remaining_budget <= 0:
                print("\n🚨 COST BUDGET EXHAUSTED")
                print(f"Completed {idx}/{len(tasks)} tasks")
                break
        else:
            task_budget = None

        print()

        task_start = time.time()

        try:
            # Run self-tuning for this task
            result = run_self_tuning(
                task_description=task['task'],
                task_type=task['task_type'],
                image_path=task['image'],
                model_type=model_type,
                api_key=api_key,
                max_iterations=max_iterations,
                success_threshold=success_threshold,
                output_dir=str(batch_dir / "tasks"),
                cost_budget=task_budget
            )

            task_elapsed = time.time() - task_start
            total_cost += result['total_cost_usd']

            # Save task result
            task_result = {
                'task_idx': idx,
                'task': task['task'],
                'task_type': task['task_type'],
                'success': result['success'],
                'iterations': result['iterations'],
                'avg_score': result['avg_score'],
                'video': result['video'],
                'cost_usd': result['total_cost_usd'],
                'time_seconds': task_elapsed,
                'output_dir': result['output_dir']
            }

            completed_tasks.append(task_result)

            # Print task summary
            print("\n" + "=" * 70)
            print(f"✅ TASK {idx + 1} COMPLETE")
            print("=" * 70)
            print(f"Success: {'✅ YES' if result['success'] else '❌ NO'}")
            print(f"Final score: {result['avg_score']:.1f}/100")
            print(f"Iterations: {result['iterations']}")
            print(f"Cost: ${result['total_cost_usd']:.2f}")
            print(f"Time: {format_time(task_elapsed)}")
            print("=" * 70)

        except Exception as e:
            print(f"\n❌ TASK {idx + 1} FAILED: {e}")
            print("Continuing to next task...")

            # Log failure
            task_result = {
                'task_idx': idx,
                'task': task['task'],
                'task_type': task['task_type'],
                'success': False,
                'error': str(e)
            }
            completed_tasks.append(task_result)

        # Save checkpoint after each task
        checkpoint = {
            'next_task_idx': idx + 1,
            'completed_tasks': completed_tasks,
            'total_cost': total_cost,
            'start_time': start_time,
            'tasks_file': tasks_file,
            'model_type': model_type
        }
        save_checkpoint(checkpoint_file, checkpoint)

        # Print progress
        elapsed = time.time() - start_time
        avg_time_per_task = elapsed / (idx - start_idx + 1)
        remaining_tasks = len(tasks) - (idx + 1)
        eta_seconds = avg_time_per_task * remaining_tasks

        print(f"\n📊 PROGRESS: {idx + 1}/{len(tasks)} tasks")
        print(f"   Elapsed: {format_time(elapsed)}")
        print(f"   ETA: {format_time(eta_seconds)}")
        print(f"   Total cost: ${total_cost:.2f}")
        if cost_budget:
            print(f"   Budget: {(total_cost/cost_budget)*100:.1f}% used")
        print()

    # Final summary
    total_elapsed = time.time() - start_time
    success_count = sum(1 for t in completed_tasks if t.get('success'))

    print("\n" + "=" * 70)
    print("🎉 BATCH EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Total tasks: {len(completed_tasks)}/{len(tasks)}")
    print(f"Success rate: {success_count}/{len(completed_tasks)} ({100*success_count/len(completed_tasks):.1f}%)")
    print(f"Total time: {format_time(total_elapsed)}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Avg cost per task: ${total_cost/len(completed_tasks):.2f}")
    print(f"Output: {batch_dir}")
    print("=" * 70)

    # Save final results
    final_results = {
        'batch_id': batch_id,
        'tasks_file': tasks_file,
        'model_type': model_type,
        'max_iterations': max_iterations,
        'success_threshold': success_threshold,
        'total_tasks': len(completed_tasks),
        'success_count': success_count,
        'success_rate': success_count / len(completed_tasks),
        'total_time_seconds': total_elapsed,
        'total_cost_usd': total_cost,
        'avg_cost_per_task': total_cost / len(completed_tasks),
        'completed_tasks': completed_tasks,
        'timestamp': datetime.now().isoformat()
    }

    with open(batch_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n📊 Results saved to: {batch_dir}/final_results.json")
    print("\n🎯 NEXT STEPS FOR IROS 2026:")
    print("   1. Analyze learning curves (use notebooks/02_visualize_learning_curves.ipynb)")
    print("   2. Run baseline experiments for comparison")
    print("   3. Generate paper figures and tables")
    print("   4. Write IROS 2026 paper (6-8 pages)")
    print("   5. Submit by March 1, 2026")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Batch Experiments for IROS 2026 Self-Tuning Video Generation"
    )
    parser.add_argument("--tasks", required=True, help="Path to tasks JSON file")
    parser.add_argument("--model", default="opus",
                       choices=['opus', 'sonnet', 'llama', 'qwen'],
                       help="Optimizer model")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Max iterations per task")
    parser.add_argument("--threshold", type=float, default=80.0,
                       help="Success threshold")
    parser.add_argument("--output-dir", default="./results/batch",
                       help="Output directory")
    parser.add_argument("--api-key", help="API key (for Claude models)")
    parser.add_argument("--cost-budget", type=float,
                       help="Max total cost for all tasks (USD)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if exists")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if args.model in ['opus', 'sonnet'] and not api_key:
        print("❌ Error: API key required for Claude models")
        print("   Get your key from: https://console.anthropic.com/")
        print("   Then set: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Verify tasks file exists
    if not Path(args.tasks).exists():
        print(f"❌ Error: Tasks file not found: {args.tasks}")
        sys.exit(1)

    # Run batch experiments
    run_batch_experiments(
        tasks_file=args.tasks,
        model_type=args.model,
        api_key=api_key,
        max_iterations=args.max_iterations,
        success_threshold=args.threshold,
        output_dir=args.output_dir,
        cost_budget=args.cost_budget,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
