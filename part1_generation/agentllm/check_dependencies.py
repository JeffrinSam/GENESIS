#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Quick dependency and path checker for Unified AgentLLM Interface
"""

import os
import sys
from pathlib import Path

def check_python_packages():
    """Check if required Python packages are installed."""
    print("Checking Python packages...")
    packages = ['flask', 'werkzeug']
    missing = []

    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)

    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip3 install {' '.join(missing)}")
        return False
    return True


def check_paths():
    """Check if required directories and files exist."""
    print("\nChecking paths...")

    _here = Path(__file__).resolve().parent  # agentllm/
    _wan = Path(os.getenv("WAN_ROOT", ""))
    _qwen = Path(os.getenv("QWEN_ROOT", ""))
    _cosmos_reason2 = Path(os.getenv("COSMOS_REASON2_ROOT", ""))
    checks = [
        (_wan / 'generate.py', 'WAN 2.2 generator (set WAN_ROOT)'),
        (_qwen / 'prompt_extenders' / 'wan22' / 'prompt_extender_drone.py', 'Drone extender (set QWEN_ROOT)'),
        (_qwen / 'prompt_extenders' / 'wan22' / 'prompt_extender_ground_robot.py', 'Ground extender'),
        (_qwen / 'prompt_extenders' / 'cosmos25' / 'prompt_extender_bimanual_ur3.py', 'UR3 extender'),
        (_qwen / 'prompt_extenders' / 'cosmos25' / 'prompt_extender_unitree_g1.py', 'G1 extender'),
        (_cosmos_reason2 / '.venv' / 'bin' / 'python3', 'Cosmos-Reason2 venv (set COSMOS_REASON2_ROOT)'),
        (_here / 'Navigation' / 'video_validator.py', 'Navigation validator'),
        (_here / 'Manipulation' / 'video_validator.py', 'Manipulation validator'),
        (_here / 'Manipulation' / 'cosmos_generate.py', 'Cosmos generator'),
    ]

    all_good = True
    for path, name in checks:
        if path.exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (NOT FOUND: {path})")
            all_good = False

    return all_good


def check_flask_app():
    """Check if Flask app can be imported."""
    print("\nChecking Flask app...")

    try:
        # Add AgentLLM to path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import unified_app

        print("  ✓ unified_app.py imports successfully")
        print(f"  ✓ Task configs: {len(unified_app.TASK_CONFIGS)} tasks")

        for task_id, config in unified_app.TASK_CONFIGS.items():
            print(f"    - {task_id}: {config['name']}")

        return True
    except Exception as e:
        print(f"  ✗ Failed to import unified_app.py: {e}")
        return False


def main():
    print("="*60)
    print("  Unified AgentLLM Dependency Checker")
    print("="*60)
    print()

    results = []

    results.append(check_python_packages())
    results.append(check_paths())
    results.append(check_flask_app())

    print()
    print("="*60)
    if all(results):
        print("  ✅ ALL CHECKS PASSED")
        print("  Ready to start: ./start_unified.sh")
    else:
        print("  ❌ SOME CHECKS FAILED")
        print("  Fix the issues above before starting")
    print("="*60)

    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
