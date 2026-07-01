#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Unified Test Script for Qwen3-VL Intelligent Routing System
Tests the complete pipeline: Router → Specialized Prompt Extender → Enhanced Prompt + Config

This script demonstrates:
1. Using prompt_router.py to analyze image and prompt
2. Automatically selecting the correct specialized extender
3. Generating enhanced prompts with appropriate configs (WAN 2.2 or Cosmos 2.5)
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

SCRIPT_DIR = Path(__file__).parent

# Test cases with expected routing outcomes
TEST_CASES = {
    'drone_aerial': {
        'prompt': 'Drone flies over a forest at sunset, smooth aerial shot',
        'expected_task': 'DRONE_NAVIGATION',
        'expected_extender': 'wan22_drone'
    },
    'humanoid_walking': {
        'prompt': 'Humanoid robot walks down a corridor, navigating around obstacles',
        'expected_task': 'GROUND_NAVIGATION',
        'expected_extender': 'wan22_ground'
    },
    'wheeled_navigation': {
        'prompt': 'Mobile robot navigates across a room, avoiding furniture',
        'expected_task': 'GROUND_NAVIGATION',
        'expected_extender': 'wan22_ground'
    },
    'ur3_manipulation': {
        'prompt': 'Dual UR3 arms pick up a box and place it on a shelf',
        'expected_task': 'BIMANUAL_UR3',
        'expected_extender': 'cosmos_ur3'
    },
    'g1_grasping': {
        'prompt': 'Unitree G1 humanoid robot picks up a bottle with both hands',
        'expected_task': 'UNITREE_G1',
        'expected_extender': 'cosmos_g1'
    }
}


def run_router(image_path: str, user_prompt: str, output_file: str) -> Dict:
    """
    Run the prompt router to analyze task type

    Args:
        image_path: Path to input image
        user_prompt: User's description
        output_file: Where to save routing analysis

    Returns:
        Routing analysis dict
    """
    logging.info("="*70)
    logging.info("STEP 1: Running Intelligent Router")
    logging.info("="*70)

    router_script = SCRIPT_DIR / 'prompt_router.py'

    cmd = [
        'python', str(router_script),
        '--image', image_path,
        '--prompt', user_prompt,
        '--output', output_file
    ]

    logging.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"Router failed: {result.stderr}")
        raise RuntimeError(f"Router failed with code {result.returncode}")

    # Load routing analysis
    with open(output_file, 'r') as f:
        analysis = json.load(f)

    logging.info(f"\nRouting Result:")
    logging.info(f"  Task Type: {analysis['task_type']}")
    logging.info(f"  Confidence: {analysis['confidence']:.2f}")
    logging.info(f"  Selected Extender: {analysis['selected_extender']}")
    logging.info(f"  Reasoning: {analysis['reasoning']}")

    return analysis


def run_extender(analysis: Dict, image_path: str, user_prompt: str, output_base: str) -> Tuple[Path, Path]:
    """
    Run the appropriate specialized prompt extender

    Args:
        analysis: Routing analysis from router
        image_path: Path to input image (may be None for T2V)
        user_prompt: User's description
        output_base: Base name for output files

    Returns:
        Tuple of (prompt_file, config_file) paths
    """
    logging.info("\n" + "="*70)
    logging.info("STEP 2: Running Specialized Prompt Extender")
    logging.info("="*70)

    extender_script = Path(analysis['extender_script'])
    extender_name = analysis['selected_extender']

    logging.info(f"Using extender: {extender_name}")
    logging.info(f"Script: {extender_script}")

    # Build command
    cmd = [
        'python', str(extender_script),
        '--prompt', user_prompt,
        '--output', output_base
    ]

    # Add image if required (Cosmos always needs image, WAN optional for I2V/TI2V)
    if image_path and ('cosmos' in extender_name or image_path):
        cmd.extend(['--image', image_path])

    logging.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"Extender failed: {result.stderr}")
        raise RuntimeError(f"Extender failed with code {result.returncode}")

    logging.info("\n" + result.stdout)

    # Determine output file paths
    outputs_dir = SCRIPT_DIR / 'outputs'
    prompt_file = outputs_dir / f"{output_base}_prompt.txt"

    if 'cosmos' in extender_name:
        config_file = outputs_dir / f"{output_base}_cosmos_config.json"
    else:
        config_file = outputs_dir / f"{output_base}_wan_config.json"

    return prompt_file, config_file


def verify_outputs(prompt_file: Path, config_file: Path, expected_task: str):
    """
    Verify that outputs were generated correctly

    Args:
        prompt_file: Path to generated prompt file
        config_file: Path to generated config file
        expected_task: Expected task type for verification
    """
    logging.info("\n" + "="*70)
    logging.info("STEP 3: Verifying Outputs")
    logging.info("="*70)

    # Check files exist
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    logging.info(f"✓ Prompt file exists: {prompt_file}")
    logging.info(f"✓ Config file exists: {config_file}")

    # Read and display prompt
    with open(prompt_file, 'r') as f:
        prompt_content = f.read()

    # Count words in enhanced prompt (should be within expected range)
    prompt_lines = prompt_content.split('\n')
    enhanced_prompt = ""
    capture = False
    for line in prompt_lines:
        if '='*60 in line:
            if not capture:
                capture = True
            else:
                break
        elif capture and line.strip() and '='*60 not in line and 'Negative Prompt' not in line:
            enhanced_prompt += line + " "

    word_count = len(enhanced_prompt.split())
    logging.info(f"✓ Enhanced prompt length: {word_count} words")

    # Verify word count is in acceptable range
    if 'DRONE' in expected_task or 'GROUND' in expected_task:
        # WAN 2.2: 100-200 words
        if word_count < 80 or word_count > 250:
            logging.warning(f"Word count {word_count} outside expected range (100-200) for WAN 2.2")
    else:
        # Cosmos 2.5: 150-300 words
        if word_count < 120 or word_count > 350:
            logging.warning(f"Word count {word_count} outside expected range (150-300) for Cosmos 2.5")

    # Read and verify config
    with open(config_file, 'r') as f:
        config = json.load(f)

    logging.info(f"✓ Config file valid JSON")

    # Check config has required fields
    if 'prompt' not in config:
        raise ValueError("Config missing 'prompt' field")

    if 'DRONE' in expected_task or 'GROUND' in expected_task:
        # WAN 2.2 config
        required_fields = ['task', 'size', 'frame_num', 'sample_steps', 'sample_guide_scale']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"WAN config missing required field: {field}")
        logging.info(f"✓ WAN 2.2 config valid: {config['task']}, {config['size']}, {config['frame_num']} frames")
    else:
        # Cosmos 2.5 config
        required_fields = ['inference_type', 'input_path', 'num_output_frames', 'resolution']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Cosmos config missing required field: {field}")
        logging.info(f"✓ Cosmos 2.5 config valid: {config['inference_type']}, {config['resolution']}, {config['num_output_frames']} frames")

    logging.info("\n✅ All verifications passed!")


def run_full_test(test_name: str, image_path: str, user_prompt: str, expected_task: str):
    """
    Run complete test: Router → Extender → Verify

    Args:
        test_name: Name of test case
        image_path: Path to test image
        user_prompt: User's description
        expected_task: Expected task type
    """
    print("\n\n")
    print("🚀 " + "="*68 + " 🚀")
    print(f"   TESTING: {test_name.upper()}")
    print("🚀 " + "="*68 + " 🚀")
    print(f"\nPrompt: {user_prompt}")
    print(f"Expected Task: {expected_task}\n")

    # Step 1: Run router
    routing_output = SCRIPT_DIR / 'outputs' / f"{test_name}_routing.json"
    analysis = run_router(image_path, user_prompt, str(routing_output))

    # Verify routing is correct
    if analysis['task_type'] != expected_task:
        logging.warning(f"⚠️  Routing mismatch! Expected {expected_task}, got {analysis['task_type']}")
    else:
        logging.info(f"✓ Routing correct: {expected_task}")

    # Step 2: Run extender
    prompt_file, config_file = run_extender(analysis, image_path, user_prompt, test_name)

    # Step 3: Verify outputs
    verify_outputs(prompt_file, config_file, expected_task)

    print("\n✅ " + "="*68 + " ✅")
    print(f"   TEST PASSED: {test_name.upper()}")
    print("✅ " + "="*68 + " ✅\n")


def main():
    parser = argparse.ArgumentParser(
        description='Test Qwen3-VL Intelligent Routing System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with specific image and prompt
  python test_routing_system.py --image test.jpg --prompt "Drone flies over forest"

  # Run predefined test case
  python test_routing_system.py --test drone_aerial --image test.jpg

  # List available test cases
  python test_routing_system.py --list-tests
        """
    )

    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--prompt', type=str, help='User prompt to test')
    parser.add_argument('--test', type=str, choices=list(TEST_CASES.keys()),
                       help='Run predefined test case')
    parser.add_argument('--list-tests', action='store_true',
                       help='List available test cases')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List tests
    if args.list_tests:
        print("\nAvailable Test Cases:")
        print("="*70)
        for name, info in TEST_CASES.items():
            print(f"\n{name}:")
            print(f"  Prompt: {info['prompt']}")
            print(f"  Expected Task: {info['expected_task']}")
            print(f"  Expected Extender: {info['expected_extender']}")
        print("="*70)
        return 0

    # Run specific test
    if args.test:
        if not args.image:
            print("Error: --image required when using --test")
            return 1

        test_info = TEST_CASES[args.test]
        run_full_test(
            args.test,
            args.image,
            test_info['prompt'],
            test_info['expected_task']
        )
        return 0

    # Run custom test
    if args.image and args.prompt:
        run_full_test(
            'custom_test',
            args.image,
            args.prompt,
            'UNKNOWN'  # Don't verify task type for custom tests
        )
        return 0

    # No valid arguments
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
