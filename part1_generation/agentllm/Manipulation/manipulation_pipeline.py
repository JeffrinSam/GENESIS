#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Complete Manipulation Pipeline with Validation
For UR3 Bimanual and G1 Humanoid manipulation using Cosmos 2.5

Pipeline Flow:
1. Input: Image + Simple Prompt
2. Qwen3-VL enhances prompt with physics-based details
3. Cosmos 2.5 generates manipulation video
4. Cosmos-Reason2 validates if video accomplished user's goal
5. Output: Video + Validation Report

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-03-21
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# Base paths — resolved from environment variables (set in .env or shell)
COSMOS_BASE = Path(os.getenv("COSMOS_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "cosmos-predict2.5")))
QWEN_EXTENDERS = Path(os.getenv("QWEN_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "Qwen3-VL"))) / "prompt_extenders"
COSMOS_REASON2_BASE = Path(os.getenv("COSMOS_REASON2_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "cosmos-reason2")))
VALIDATOR_SCRIPT = Path(__file__).parent / 'video_validator.py'  # Direct Cosmos-Reason2 validator
OUTPUT_DIR = Path(__file__).parent / 'outputs'

# Task configurations
MANIPULATION_TASKS = {
    'ur3': {
        'name': 'Bimanual UR3 Manipulation',
        'extender': QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_bimanual_ur3.py',
        'description': 'Dual-arm manipulation with UR3 robots'
    },
    'g1': {
        'name': 'Unitree G1 Humanoid Manipulation',
        'extender': QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_unitree_g1.py',
        'description': 'Humanoid robot manipulation tasks'
    }
}


def print_header(title: str):
    """Print formatted section header"""
    print("")
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("")


def enhance_prompt(task_type: str, user_prompt: str, image_path: str) -> str:
    """Step 1: Enhance prompt using Qwen3-VL with physics-based details"""
    print_header("STEP 1/4: PROMPT ENHANCEMENT (QWEN3-VL)")

    task_config = MANIPULATION_TASKS[task_type]
    extender_script = task_config['extender']
    output_base = f"manip_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logging.info(f"Task: {task_config['name']}")
    logging.info(f"Extender: {extender_script.name}")
    logging.info(f"Focus: Physics-based manipulation")

    # Use Qwen3.5 venv (needs transformers>=5.0 for qwen3_5 model type)
    qwen35_python = os.getenv("QWEN_PYTHON", str(Path(os.getenv("QWEN_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "Qwen3-VL"))) / ".venv" / "bin" / "python"))
    cmd = [
        qwen35_python, str(extender_script),
        '--prompt', user_prompt,
        '--image', image_path,
        '--output', output_base
    ]

    logging.info("Enhancing prompt with physics details...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"Enhancement failed: {result.stderr}")
        raise RuntimeError("Qwen3-VL enhancement failed")

    # Read enhanced prompt
    outputs_dir = QWEN_EXTENDERS / 'outputs'
    prompt_file = outputs_dir / f"{output_base}_prompt.txt"

    with open(prompt_file, 'r') as f:
        content = f.read()

    # Extract enhanced prompt (between separator lines)
    lines = content.split('\n')
    enhanced_prompt = ""
    capture = False

    for line in lines:
        if '=' * 60 in line:
            if not capture:
                capture = True
            else:
                break
        elif capture and line.strip() and 'Negative Prompt' not in line:
            enhanced_prompt += line + " "

    enhanced_prompt = enhanced_prompt.strip()

    logging.info(f"✓ Enhanced: {len(enhanced_prompt.split())} words")
    logging.info(f"Preview: {enhanced_prompt[:100]}...")

    return enhanced_prompt


def generate_video(prompt: str, image_path: str, output_path: str, **kwargs) -> None:
    """Step 2: Generate manipulation video using Cosmos 2.5"""
    print_header("STEP 2/4: VIDEO GENERATION (COSMOS 2.5)")

    model = kwargs.get('model', '2B')

    # Use Cosmos venv python directly
    cosmos_python = COSMOS_BASE / '.venv' / 'bin' / 'python'
    wrapper_script = Path(__file__).parent / 'cosmos_generate.py'

    # Convert to absolute paths
    image_abs = str(Path(image_path).resolve())
    output_abs = str(Path(output_path).resolve())

    cmd = [
        str(cosmos_python),
        str(wrapper_script),
        '--model', model,
        '--input_path', image_abs,
        '--prompt', prompt,
        '--output_path', output_abs,
        '--num_output_frames', str(kwargs.get('frames', 77)),
        '--guidance', str(kwargs.get('guidance', 7)),
        '--seed', str(kwargs.get('seed', 42)),
    ]

    logging.info(f"Model: Cosmos 2.5 {model}")
    logging.info(f"Input Image: {image_path}")
    logging.info(f"Frames: {kwargs.get('frames', 77)}")
    logging.info("")
    logging.info("Generating video... (3-5 min for 2B, 8-12 min for 14B)")

    # Run from Cosmos directory
    result = subprocess.run(cmd, cwd=str(COSMOS_BASE))

    if result.returncode != 0:
        raise RuntimeError("Cosmos video generation failed")

    logging.info(f"✓ Video saved: {output_path}")


def validate_video(video_path: str, task_type: str, original_prompt: str) -> dict:
    """Step 3: Validate video using Cosmos-Reason2"""
    print_header("STEP 3/4: VIDEO VALIDATION (COSMOS-REASON2)")

    # Convert to absolute paths
    video_abs = str(Path(video_path).resolve())

    # Use cosmos-reason2 venv python
    reason2_python = COSMOS_REASON2_BASE / '.venv' / 'bin' / 'python'
    cmd = [
        str(reason2_python),
        str(VALIDATOR_SCRIPT),
        '--video', video_abs,
        '--task', task_type,
        '--prompt', original_prompt,
        '--output', str(OUTPUT_DIR / f"validation_{Path(video_path).stem}.json")
    ]

    logging.info(f"Validator: Cosmos-Reason2-2B (via wrapper)")
    logging.info(f"Checking: Prompt adherence, physics, quality")
    logging.info("")
    logging.info("Validating... (1-3 minutes)")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"Validation failed: {result.stderr}")
        raise RuntimeError("Video validation failed")

    # Read validation report
    report_path = OUTPUT_DIR / f"validation_{Path(video_path).stem}.json"
    with open(report_path, 'r') as f:
        validation_result = json.load(f)

    logging.info(f"✓ Validation complete")

    return validation_result


def print_validation_summary(result: dict):
    """Step 4: Print validation summary"""
    print_header("STEP 4/4: VALIDATION RESULTS")

    print(f"Overall Result: {'✅ PASS' if result['pass'] else '❌ FAIL'}")
    print(f"Confidence: {result['confidence']}%")
    print("")

    print("Component Scores:")
    for component in result['components']:
        score = component['score']
        status = "✓" if score >= 70 else "✗"
        print(f"  {status} {component['name']}: {score}/100")
        print(f"     {component['analysis'][:80]}...")

    print("")
    print(f"Detailed Report: {result.get('report_path', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Manipulation Pipeline with Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete Closed-Loop Manipulation Pipeline:
  1. Input: Image + Simple Prompt
  2. Qwen3-VL enhances with physics-based details
  3. Cosmos 2.5 generates manipulation video
  4. Cosmos-Reason2 validates video quality

Task Types:
  ur3  - Bimanual UR3 arm manipulation
  g1   - Unitree G1 humanoid manipulation

Examples:
  # UR3 bimanual manipulation
  python3 manipulation_pipeline.py \\
    --task ur3 \\
    --image workspace.jpg \\
    --prompt "Pick red cube with left arm, blue cube with right" \\
    --output outputs/ur3_task.mp4

  # G1 humanoid with 14B model (higher quality)
  python3 manipulation_pipeline.py \\
    --task g1 \\
    --image kitchen.jpg \\
    --prompt "Pick up bottle from table" \\
    --cosmos-model 14B \\
    --output outputs/g1_task.mp4

  # Quick test (2B model, fewer frames)
  python3 manipulation_pipeline.py \\
    --task ur3 \\
    --image test.jpg \\
    --prompt "Grasp object" \\
    --cosmos-frames 77 \\
    --output outputs/test.mp4
        """
    )

    # Core arguments
    parser.add_argument('--task', type=str, required=True,
                       choices=['ur3', 'g1'],
                       help='Manipulation task type')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image showing workspace/scene')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Simple prompt describing manipulation goal')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video path')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip video validation (faster)')

    # Cosmos 2.5 parameters
    cosmos_group = parser.add_argument_group('Cosmos 2.5 Parameters')
    cosmos_group.add_argument('--cosmos-model', type=str, default='2B',
                             choices=['2B', '14B'],
                             help='Model size (default: 2B for speed)')
    cosmos_group.add_argument('--cosmos-frames', type=int, default=77,
                             help='Frame count (default: 77, max: 121)')
    cosmos_group.add_argument('--cosmos-guidance', type=float, default=7,
                             help='Guidance scale (default: 7)')
    cosmos_group.add_argument('--cosmos-seed', type=int, default=42,
                             help='Random seed (default: 42)')

    args = parser.parse_args()

    # Validate inputs
    if args.task not in MANIPULATION_TASKS:
        print(f"Error: Invalid task '{args.task}'")
        return 1

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print pipeline summary
    task_config = MANIPULATION_TASKS[args.task]
    print("")
    print("=" * 70)
    print("COMPLETE MANIPULATION PIPELINE WITH VALIDATION")
    print("=" * 70)
    print(f"Task: {task_config['name']}")
    print(f"Description: {task_config['description']}")
    print(f"Input Image: {args.image}")
    print(f"User Prompt: {args.prompt}")
    print(f"Model: Cosmos 2.5 {args.cosmos_model}")
    print(f"Output Video: {args.output}")
    print(f"Validation: {'ENABLED' if not args.no_validation else 'DISABLED'}")
    print("=" * 70)

    try:
        # Step 1: Enhance prompt
        enhanced_prompt = enhance_prompt(args.task, args.prompt, args.image)

        # Step 2: Generate video
        generate_video(
            prompt=enhanced_prompt,
            image_path=args.image,
            output_path=str(output_path),
            model=args.cosmos_model,
            frames=args.cosmos_frames,
            guidance=args.cosmos_guidance,
            seed=args.cosmos_seed
        )

        # Step 3: Validate video (optional)
        validation_result = None
        if not args.no_validation:
            validation_result = validate_video(
                video_path=str(output_path),
                task_type=args.task,
                original_prompt=args.prompt
            )

            # Step 4: Print results
            print_validation_summary(validation_result)

        # Final summary
        print("")
        print("=" * 70)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Task: {task_config['name']}")
        print(f"Video: {output_path}")

        if validation_result:
            status = "✅ PASSED" if validation_result['pass'] else "❌ FAILED"
            print(f"Validation: {status} ({validation_result['confidence']}% confidence)")
            print(f"Report: {OUTPUT_DIR}/validation_{output_path.stem}.json")

        print("")
        print(f"Play with: vlc {output_path}")
        print("=" * 70)

        return 0

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
