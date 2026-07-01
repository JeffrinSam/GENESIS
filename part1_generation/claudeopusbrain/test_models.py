#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Test Script: Verify Multi-Model Optimizer Architecture

This script tests all optimizer models to ensure they work correctly before
running full experiments for IROS 2026.

Usage:
    # Test all models (requires API key and Ollama)
    python3 test_models.py --all

    # Test specific model
    python3 test_models.py --model opus
    python3 test_models.py --model sonnet
    python3 test_models.py --model llama
    python3 test_models.py --model qwen

    # Test with API key
    python3 test_models.py --model opus --api-key sk-ant-...

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-02-07
"""

import os
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from prompt_optimizer import OptimizerFactory


def test_optimizer(model_type: str, api_key: str = None):
    """
    Test a specific optimizer model

    Args:
        model_type: Model type to test
        api_key: API key (for Claude models)
    """
    print("\n" + "="*70)
    print(f"TESTING {model_type.upper()} OPTIMIZER")
    print("="*70 + "\n")

    try:
        # Create optimizer
        print(f"1. Creating {model_type} optimizer...")
        optimizer = OptimizerFactory.create_optimizer(
            model_type=model_type,
            api_key=api_key
        )
        print(f"   ✅ Optimizer created: {optimizer.get_model_name()}")

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
        print(f"\n2. Testing optimization with mock validation...")
        result = optimizer.optimize_prompts(
            task_description="Humanoid picks up bottle from table",
            task_type="g1",
            current_system_prompt="You are an expert in physics-based robotics manipulation...",
            current_negative_prompt="blurry, low quality, distorted, collision",
            validation_result=mock_validation,
            iteration=1,
            max_iterations=5
        )

        print(f"   ✅ Optimization complete!")
        print(f"\n3. Verification:")
        print(f"   Model Name: {result.model_name}")
        print(f"   System Prompt Length: {len(result.system_prompt)} chars")
        print(f"   Negative Prompt Length: {len(result.negative_prompt)} chars")
        print(f"   Reasoning Keys: {list(result.reasoning.keys())}")
        print(f"   Expected Improvements: {list(result.expected_improvements.keys())}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Cost: ${result.cost_usd:.4f}")

        # Show sample reasoning
        print(f"\n4. Sample Reasoning:")
        print(f"   Analysis: {result.reasoning['analysis'][:150]}...")
        print(f"   Strategy: {result.reasoning['strategy'][:150]}...")

        # Verify structure
        print(f"\n5. Structure Verification:")
        errors = []

        if not result.system_prompt or len(result.system_prompt) < 100:
            errors.append("System prompt too short or empty")

        if not result.negative_prompt or len(result.negative_prompt) < 20:
            errors.append("Negative prompt too short or empty")

        required_reasoning_keys = ['analysis', 'strategy']
        for key in required_reasoning_keys:
            if key not in result.reasoning:
                errors.append(f"Missing reasoning key: {key}")

        required_improvement_keys = ['adherence', 'physics', 'quality']
        for key in required_improvement_keys:
            if key not in result.expected_improvements:
                errors.append(f"Missing improvement key: {key}")

        if not (0 <= result.confidence <= 1):
            errors.append(f"Confidence out of range: {result.confidence}")

        if errors:
            print(f"   ❌ Errors found:")
            for error in errors:
                print(f"      - {error}")
            return False
        else:
            print(f"   ✅ All structure checks passed!")

        # Test history tracking
        print(f"\n6. History Tracking:")
        summary = optimizer.get_summary()
        print(f"   Total iterations: {summary['total_iterations']}")
        print(f"   Total cost: ${summary['total_cost_usd']:.4f}")
        print(f"   Model: {summary['model']}")
        print(f"   ✅ History tracking works!")

        print("\n" + "="*70)
        print(f"✅ {model_type.upper()} OPTIMIZER TEST PASSED!")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\n" + "="*70)
        print(f"❌ {model_type.upper()} OPTIMIZER TEST FAILED!")
        print("="*70 + "\n")

        # Show helpful error messages
        if model_type in ['opus', 'sonnet']:
            print("💡 Troubleshooting for Claude models:")
            print("   - Make sure you have an Anthropic API key")
            print("   - Get one from: https://console.anthropic.com/")
            print("   - Set: export ANTHROPIC_API_KEY='your-key-here'")
            print("   - Or use: --api-key flag\n")

        elif model_type in ['llama', 'qwen']:
            print(f"💡 Troubleshooting for {model_type}:")
            print("   - Make sure the Ollama server is running (default: localhost:11434)")
            print("   - Check: curl http://localhost:11434/health")
            print("   - List models: curl http://localhost:11434/models")
            print("   - Override with: export OLLAMA_SERVER=http://<host>:<port>\n")

        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Multi-Model Optimizer Architecture"
    )
    parser.add_argument("--model", choices=['opus', 'sonnet', 'llama', 'qwen'],
                       help="Test specific model")
    parser.add_argument("--all", action="store_true",
                       help="Test all models")
    parser.add_argument("--api-key", help="Anthropic API key (for Claude models)")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")

    # Determine which models to test
    if args.all:
        models = ['opus', 'sonnet', 'llama', 'qwen']
    elif args.model:
        models = [args.model]
    else:
        print("Error: Specify --model or --all")
        parser.print_help()
        sys.exit(1)

    # Check Ollama server if testing free models
    if any(m in models for m in ['llama', 'qwen']):
        from opensource_optimizer import check_server, list_models, OLLAMA_SERVER
        print(f"\nChecking Ollama server at {OLLAMA_SERVER}...")
        if check_server():
            print(f"   ✅ Server reachable")
            available = list_models()
            if available:
                print(f"   Available models: {', '.join(str(m) for m in available)}")
        else:
            print(f"   ❌ Server not reachable at {OLLAMA_SERVER}")

    # Test models
    results = {}
    for model in models:
        # Check if API key is available for Claude models
        if model in ['opus', 'sonnet'] and not api_key:
            print(f"\n⚠️  Skipping {model}: No API key available")
            print("   Set ANTHROPIC_API_KEY or use --api-key flag")
            results[model] = False
            continue

        # Test model
        success = test_optimizer(model, api_key)
        results[model] = success

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")

    for model, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model.upper():12} {status}")

    # Overall result
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nReady to run experiments! 🚀")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix errors before running experiments.")
    print("="*70 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
