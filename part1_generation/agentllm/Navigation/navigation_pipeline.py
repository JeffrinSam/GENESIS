#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Complete Navigation Pipeline with Validation
For Drone and Ground Robot navigation using WAN 2.2

Pipeline Flow:
1. Input: Image + Simple Prompt
2. Qwen3-VL enhances prompt with first-person perspective
3. WAN 2.2 (TI2V-5B) generates navigation video
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
WAN_BASE = Path(os.getenv("WAN_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "Wan2.2")))
QWEN_EXTENDERS = Path(os.getenv("QWEN_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "Qwen3-VL"))) / "prompt_extenders"
COSMOS_REASON2_BASE = Path(os.getenv("COSMOS_REASON2_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "cosmos-reason2")))
VALIDATOR_SCRIPT = Path(__file__).parent / 'video_validator.py'  # Direct Cosmos-Reason2 validator
OUTPUT_DIR = Path(__file__).parent / 'outputs'

# Task configurations
NAVIGATION_TASKS = {
    'drone': {
        'name': 'Drone Aerial Navigation',
        'extender': QWEN_EXTENDERS / 'wan22' / 'prompt_extender_drone.py',
        'description': 'First-person drone flight perspective'
    },
    'ground': {
        'name': 'Ground Robot Navigation',
        'extender': QWEN_EXTENDERS / 'wan22' / 'prompt_extender_ground_robot.py',
        'description': 'First-person ground robot POV'
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
    """Step 1: Enhance prompt using Qwen3-VL with first-person perspective"""
    print_header("STEP 1/4: PROMPT ENHANCEMENT (QWEN3-VL)")

    task_config = NAVIGATION_TASKS[task_type]
    extender_script = task_config['extender']
    output_base = f"nav_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logging.info(f"Task: {task_config['name']}")
    logging.info(f"Extender: {extender_script.name}")
    logging.info(f"Perspective: First-person embodied")

    # Use Qwen3.5 venv (needs transformers>=5.0 for qwen3_5 model type)
    qwen35_python = os.getenv("QWEN_PYTHON", str(Path(os.getenv("QWEN_ROOT", str(Path(__file__).resolve().parents[4] / "Part1" / "Qwen3-VL"))) / ".venv" / "bin" / "python"))
    cmd = [
        qwen35_python, str(extender_script),
        '--prompt', user_prompt,
        '--image', image_path,
        '--output', output_base
    ]

    logging.info("Enhancing prompt with first-person perspective...")
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
    """Step 2: Generate navigation video using WAN 2.2 TI2V-5B"""
    print_header("STEP 2/4: VIDEO GENERATION (WAN 2.2)")

    wan_task = 'ti2v-5B'
    ckpt_dir = WAN_BASE / 'Wan2.2-TI2V-5B'

    # Use wan2.2 conda env python (has torch + WAN dependencies)
    wan22_python = Path(os.getenv("WAN_PYTHON", shutil.which("python3") or "python3"))
    cmd = [
        str(wan22_python), str(WAN_BASE / 'generate.py'),
        '--task', wan_task,
        '--ckpt_dir', str(ckpt_dir),
        '--prompt', prompt,
        '--image', image_path,
        '--size', kwargs.get('size', '1280*704'),
        '--frame_num', str(kwargs.get('frames', 61)),
        '--sample_steps', str(kwargs.get('steps', 30)),
        '--sample_guide_scale', str(kwargs.get('guidance', 7.5)),
        '--base_seed', str(kwargs.get('seed', -1)),
        '--save_file', output_path
    ]

    if 'shift' in kwargs and kwargs['shift'] is not None:
        cmd.extend(['--sample_shift', str(kwargs['shift'])])

    logging.info(f"Model: WAN 2.2 {wan_task}")
    logging.info(f"Input Image: {image_path}")
    logging.info(f"Resolution: {kwargs.get('size', '1280*704')}")
    logging.info(f"Frames: {kwargs.get('frames', 61)}")
    logging.info("")
    logging.info("Generating video... (2-8 minutes)")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError("WAN video generation failed")

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
        description='Complete Navigation Pipeline with Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete Closed-Loop Navigation Pipeline:
  1. Input: Image + Simple Prompt
  2. Qwen3-VL enhances with first-person perspective
  3. WAN 2.2 generates navigation video
  4. Cosmos-Reason2 validates video quality

Task Types:
  drone   - Aerial navigation from drone's perspective
  ground  - Ground robot navigation from robot's POV

Examples:
  # Drone navigation with validation
  python3 navigation_pipeline.py \\
    --task drone \\
    --image aerial_view.jpg \\
    --prompt "Flying over snowy mountains at sunrise" \\
    --output outputs/drone_flight.mp4

  # Ground robot with custom WAN parameters
  python3 navigation_pipeline.py \\
    --task ground \\
    --image corridor.jpg \\
    --prompt "Navigate through office hallway" \\
    --wan-frames 121 \\
    --wan-guidance 9.0 \\
    --output outputs/ground_nav.mp4

  # Quick test (fast generation)
  python3 navigation_pipeline.py \\
    --task drone \\
    --image test.jpg \\
    --prompt "Gliding forward over forest" \\
    --wan-frames 61 \\
    --output outputs/test.mp4
        """
    )

    # Core arguments
    parser.add_argument('--task', type=str, required=True,
                       choices=['drone', 'ground'],
                       help='Navigation task type')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image showing scene/environment')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Simple prompt describing navigation goal')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video path')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip video validation (faster)')

    # WAN 2.2 parameters
    wan_group = parser.add_argument_group('WAN 2.2 Parameters')
    wan_group.add_argument('--wan-size', type=str, default='1280*704',
                          help='Resolution (default: 1280*704)')
    wan_group.add_argument('--wan-frames', type=int, default=61,
                          help='Frame count (default: 61, max: 121)')
    wan_group.add_argument('--wan-steps', type=int, default=30,
                          help='Sample steps (default: 30)')
    wan_group.add_argument('--wan-guidance', type=float, default=7.5,
                          help='Guidance scale (default: 7.5)')
    wan_group.add_argument('--wan-shift', type=float,
                          help='Sample shift (optional)')
    wan_group.add_argument('--wan-seed', type=int, default=-1,
                          help='Random seed (-1 = random)')

    args = parser.parse_args()

    # Validate inputs
    if args.task not in NAVIGATION_TASKS:
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
    task_config = NAVIGATION_TASKS[args.task]
    print("")
    print("=" * 70)
    print("COMPLETE NAVIGATION PIPELINE WITH VALIDATION")
    print("=" * 70)
    print(f"Task: {task_config['name']}")
    print(f"Description: {task_config['description']}")
    print(f"Input Image: {args.image}")
    print(f"User Prompt: {args.prompt}")
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
            size=args.wan_size,
            frames=args.wan_frames,
            steps=args.wan_steps,
            guidance=args.wan_guidance,
            shift=args.wan_shift,
            seed=args.wan_seed
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
