#!/usr/bin/env python3
"""
WAN 2.2 Ground Robot Navigation Prompt Extender
Generates cinematic, detailed prompts for ground-based robot navigation using WAN 2.2 film director approach
Supports: Humanoid robots (walking), Wheeled robots, Tracked robots, Mobile platforms
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
QWEN_MODEL_PATH = SCRIPT_DIR.parent / 'Qwen3.5-9B'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'

# WAN 2.2 Ground Robot System Prompt (Film Director Approach — Research-Optimized v2)
# Key changes: output length 100-200→80-120, I2V rule, professional camera vocabulary, one-scene rule
# Sources: WAN 2.2 docs, VPO (ICCV 2025), Prompt-A-Video (ICCV 2025), community testing
GROUND_ROBOT_SYSTEM_PROMPT = """You are an expert cinematographer and robotics specialist creating cinematic video prompts for WAN 2.2 video generation. You understand film theory, ground-based robot locomotion, and visual storytelling.

**CRITICAL PERSPECTIVE RULE**: The camera IS the robot's eyes. This is FIRST-PERSON embodied navigation — the viewer sees through the robot's perspective. Never describe the robot externally.

**WAN 2.2 Requirements**: Cinematic prompts optimized for WAN 2.2's MoE text encoder (80-120 words). WAN processes prompts through a Mixture-of-Experts encoder that saturates beyond ~120 words — shorter prompts produce sharper, more coherent videos.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the environment, objects, and lighting. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the camera MOVES (direction, speed, turns)
2. What ENTERS and EXITS the frame during motion
3. TEMPORAL PROGRESSION of the navigation

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**Your Task**: Transform simple ground robot navigation prompts into FIRST-PERSON cinematic descriptions (80-120 words) for WAN 2.2.

**Professional Camera Vocabulary** (WAN 2.2 responds strongly to these):
- Dolly forward/backward, tracking shot, crane movement
- Pan left/right, tilt up/down, orbital arc
- Parallax shift, depth reveal, perspective pull
- Rack focus, motion blur, rolling shutter feel
- Speed modifiers: "slow dolly", "rapid tracking", "gradual pan"

**Locomotion-Specific Camera Feel**:
- Humanoid walking: Subtle rhythmic bob, natural head stabilization, dolly-forward with gait oscillation
- Wheeled motion: Smooth tracking shot, stable horizon, gliding dolly
- Tracked motion: Minor vibrations, steady forward tracking with subtle shake

**Examples (FIRST-PERSON, ~100 words each)**:

**Humanoid Robot** (~100 words):
"First-person POV, steady dolly forward through modern office corridor, soft daylight from overhead LED panels. Subtle rhythmic vertical bob from bipedal walking gait as polished floor scrolls beneath. Corridor walls glide past on both sides with gentle parallax. A potted plant enters frame on the right, growing larger. Gradual pan left to navigate around it, the plant sliding to right periphery. Glass door ahead grows steadily larger as the dolly continues forward. Deceleration as footsteps slow, door filling frame center. Coming to a complete stop centered on the entrance."

**Wheeled Mobile Robot** (~100 words):
"Low-angle first-person POV, smooth tracking shot across wooden floor in residential living room, ambient indoor lighting with cool tones. Furniture visible ahead — couch, coffee table, bookshelf. Floor flows beneath at low vantage point, sunlight from right windows casting dynamic shadow parallax. Coffee table approaches center frame. Gradual pan left, perspective rotating, table sliding right as clear path opens. Steady forward dolly toward bookshelf, details growing larger with natural depth reveal. Deceleration, floor motion slowing, coming to rest with bookshelf dominating forward view."

**Critical Rules**:
- 80-120 words total (WAN 2.2 MoE sweet spot — longer prompts reduce quality)
- ALWAYS first-person perspective (YOU are the robot)
- Environment moves relative to YOUR motion
- Use professional camera terms: dolly, tracking, pan, parallax, depth reveal
- Do NOT re-describe what the image already shows
- ONE continuous 5-second shot only
- Do NOT add objects not in the original prompt
- NO flying, hovering, or aerial views (ground-based only)
- NO third-person descriptions ("robot walks", "robot moves")"""

# Negative prompt for ground robot navigation (layered: WAN 2.2 default + physics + task-specific)
# Layer 1: WAN 2.2 official defaults, Layer 2: Physics violations, Layer 3: Task-specific exclusions
GROUND_ROBOT_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, deformed, still picture, messy background, flying, hovering, aerial view, drone, quadcopter, airborne, floating, manipulation, grasping, picking, arms extended toward objects, holding objects, static camera, tripod, fixed position, third person view, external robot view, robot body visible, jerky motion, shaky footage, collision, crash, wall clipping, teleportation, flickering, jittering, sudden jump cuts, walking backwards"


class GroundRobotPromptExtender:
    """Generates cinematic ground robot navigation prompts for WAN 2.2"""

    def __init__(self, model_path: Path = QWEN_MODEL_PATH, device: str = 'cuda'):
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else 'cpu'

        logging.info(f"Loading Qwen3.5-9B from: {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            trust_remote_code=True
        )

        if self.device != 'cuda':
            self.model = self.model.to(self.device)

        logging.info("Model loaded successfully")

    def generate_prompt(self, user_prompt: str, image_path: Optional[str] = None,
                       custom_system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate enhanced cinematic prompt

        Args:
            user_prompt: Simple user description
            image_path: Optional input image for I2V
            custom_system_prompt: Custom system prompt (optional, uses default if not provided)

        Returns:
            Tuple of (enhanced_prompt, negative_prompt)
        """
        logging.info("Generating enhanced ground robot navigation prompt...")

        # Use custom system prompt if provided, otherwise use default
        system_prompt = custom_system_prompt if custom_system_prompt else GROUND_ROBOT_SYSTEM_PROMPT

        # Build query
        query = f"""User wants to generate a ground robot navigation video with this description:
"{user_prompt}"

Generate a detailed, cinematic prompt for WAN 2.2 following the film director approach. Focus on ground-level perspective, robot locomotion (walking/rolling/tracked), and beautiful cinematography."""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": []}
        ]

        # Add image if provided
        if image_path:
            image = Image.open(image_path).convert('RGB')
            messages[1]["content"].append({"type": "image", "image": image})
            messages[1]["content"].append({"type": "text", "text": f"Analyze this image and: {query}"})
        else:
            messages[1]["content"].append({"type": "text", "text": query})

        # Process (Qwen3.5: apply_chat_template handles tokenization + vision)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Bypass strict kwarg validation for vision inputs (transformers 5.x)
        orig_validate = self.model._validate_model_kwargs
        def _skip_vision_validate(model_kwargs):
            for k in ['pixel_values', 'image_grid_thw', 'mm_token_type_ids',
                       'pixel_values_videos', 'video_grid_thw']:
                model_kwargs.pop(k, None)
            return orig_validate(model_kwargs)
        self.model._validate_model_kwargs = _skip_vision_validate

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]

        enhanced_prompt = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        logging.info(f"Generated prompt ({len(enhanced_prompt.split())} words)")

        return enhanced_prompt, GROUND_ROBOT_NEGATIVE_PROMPT

    def create_wan_config(self, enhanced_prompt: str, negative_prompt: str,
                         image_path: Optional[str] = None, output_name: str = "ground_robot_nav") -> dict:
        """Create WAN 2.2 JSON configuration"""

        config = {
            "task": "ti2v-5B" if image_path else "t2v-A14B",
            "size": "1280*704",
            "frame_num": 61,
            "prompt": enhanced_prompt,
            "sample_steps": 30,
            "sample_guide_scale": 7.5,
            "save_file": f"{output_name}.mp4"
        }

        if negative_prompt:
            config["negative_prompt"] = negative_prompt

        if image_path:
            config["image"] = str(Path(image_path).absolute())

        return config

    def save_outputs(self, enhanced_prompt: str, config: dict, output_base: str):
        """Save prompt and config files"""
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Save human-readable prompt
        prompt_file = OUTPUT_DIR / f"{output_base}_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(f"WAN 2.2 Ground Robot Navigation Prompt\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(enhanced_prompt)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Negative Prompt:\n{config.get('negative_prompt', 'None')}\n")

        # Save WAN config
        config_file = OUTPUT_DIR / f"{output_base}_wan_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logging.info(f"Saved outputs:")
        logging.info(f"  Prompt: {prompt_file}")
        logging.info(f"  Config: {config_file}")

        return prompt_file, config_file

def main():
    parser = argparse.ArgumentParser(description='WAN 2.2 Ground Robot Navigation Prompt Extender')
    parser.add_argument('--prompt', type=str, required=True, help='User prompt describing robot navigation')
    parser.add_argument('--image', type=str, help='Optional input image for I2V')
    parser.add_argument('--system_prompt', type=str, default=None,
                       help='Custom system prompt (optional, uses default if not provided)')
    parser.add_argument('--output', type=str, default='ground_robot_nav', help='Output filename base')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize extender
    extender = GroundRobotPromptExtender()

    # Generate prompt
    enhanced_prompt, negative_prompt = extender.generate_prompt(args.prompt, args.image, args.system_prompt)

    # Create config
    config = extender.create_wan_config(enhanced_prompt, negative_prompt, args.image, args.output)

    # Save outputs
    prompt_file, config_file = extender.save_outputs(enhanced_prompt, config, args.output)

    # Print summary
    print("\n" + "="*70)
    print("WAN 2.2 GROUND ROBOT NAVIGATION PROMPT GENERATED")
    print("="*70)
    print(f"\nUser Prompt:\n{args.prompt}")
    print(f"\nEnhanced Prompt ({len(enhanced_prompt.split())} words):\n{enhanced_prompt}")
    print(f"\nNegative Prompt:\n{negative_prompt}")
    print(f"\nTask: {config['task']}")
    print(f"Resolution: {config['size']}")
    print(f"Frames: {config['frame_num']}")
    print(f"\nFiles saved to:")
    print(f"  - {prompt_file}")
    print(f"  - {config_file}")
    print("="*70)

    print(f"\nTo generate video with WAN 2.2:")
    print(f"cd $WAN_ROOT")
    print(f"python generate.py --task {config['task']} --ckpt_dir ./Wan2.2-TI2V-5B \\")
    print(f"  --prompt \"{enhanced_prompt}\" \\")
    print(f"  --size {config['size']} --frame_num {config['frame_num']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
