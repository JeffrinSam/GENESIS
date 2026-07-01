#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Cosmos 2.5 Video Generation Wrapper
MUST be run via Cosmos venv python: cosmos-predict2.5/.venv/bin/python

Uses official Cosmos Predict 2.5 Python API with proper checkpoint handling.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from PIL import Image

# Import Cosmos modules (script MUST run in Cosmos venv)
from cosmos_predict2.config import InferenceArguments, SetupArguments, InferenceType
from cosmos_predict2.inference import Inference


def ensure_rgb(image_path: str) -> str:
    """Convert image to RGB if it has alpha channel. Returns path to use."""
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'LA', 'P'):
        rgb = img.convert('RGB')
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False, prefix='cosmos_rgb_')
        rgb.save(tmp.name)
        return tmp.name
    return image_path


def main():
    parser = argparse.ArgumentParser(description='Generate video with Cosmos 2.5')
    parser.add_argument('--model', type=str, required=True, choices=['2B', '14B'],
                       help='Model size (2B or 14B)')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt for generation')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save output video')
    parser.add_argument('--num_output_frames', type=int, default=77,
                       help='Number of frames to generate')
    parser.add_argument('--guidance', type=float, default=7.0,
                       help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--negative_prompt', type=str, default=None,
                       help='Negative prompt for generation')

    args = parser.parse_args()

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure input image is RGB (Cosmos does not handle RGBA/alpha channels)
    input_path = ensure_rgb(args.input_path)

    # Model parameter format: "2B/post-trained" or "14B/post-trained"
    model_name = f"{args.model}/post-trained"

    print(f"Cosmos 2.5 {args.model} Video Generation")
    print(f"Model: {model_name}")
    print(f"Input: {input_path}")
    print(f"Output: {args.output_path}")
    print("")

    # Setup Cosmos with offloading for memory efficiency
    # checkpoint_path=None means use default checkpoint for the model
    setup_args = SetupArguments(
        model=model_name,
        output_dir=output_dir,
        offload_diffusion_model=True,
        offload_tokenizer=True,
        offload_text_encoder=True,
        disable_guardrails=True,
    )

    # Create inference pipeline
    print("Loading Cosmos model...")
    pipeline = Inference(setup_args)

    # Prepare inference arguments
    # name is required - use output filename without extension
    output_name = Path(args.output_path).stem

    inference_kwargs = dict(
        name=output_name,
        inference_type=InferenceType.IMAGE2WORLD,
        input_path=input_path,
        prompt=args.prompt,
        num_output_frames=args.num_output_frames,
        guidance=args.guidance,
        seed=args.seed,
        resolution="432,768",  # H,W format for Cosmos
    )
    if args.negative_prompt:
        inference_kwargs['negative_prompt'] = args.negative_prompt

    inference_args = [InferenceArguments(**inference_kwargs)]

    # Generate video
    print(f"Generating {args.num_output_frames} frames...")
    print(f"Prompt: {args.prompt[:100]}...")
    print("")

    output_videos = pipeline.generate(inference_args, output_dir)

    # Cosmos generates video with auto-generated name, rename to requested path
    if output_videos and len(output_videos) > 0:
        generated_path = Path(output_videos[0])
        target_path = Path(args.output_path)

        if generated_path != target_path:
            if target_path.exists():
                target_path.unlink()
            generated_path.rename(target_path)

        print(f"✓ Video saved to: {target_path}")
        return 0
    else:
        print("ERROR: No video was generated")
        return 1


if __name__ == '__main__':
    sys.exit(main())
