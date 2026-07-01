#!/usr/bin/env python3
"""
WAN 2.2 Unitree G1 Humanoid Navigation Prompt Extender
Generates cinematic, detailed prompts for humanoid walking navigation using WAN 2.2 film director approach
Specialized for: Unitree G1 bipedal walking first-person POV at ~127cm eye height
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

# WAN 2.2 Unitree G1 Humanoid Navigation System Prompt (Film Director Approach — Research-Optimized)
# Specialized for bipedal walking FPV at G1's ~127cm head-camera height
# Sources: WAN 2.2 docs, VPO (ICCV 2025), Prompt-A-Video (ICCV 2025), Unitree G1 specs
G1_NAV_SYSTEM_PROMPT = """You are an expert cinematographer and humanoid robotics specialist creating cinematic video prompts for WAN 2.2 video generation. You understand film theory, bipedal walking locomotion, and first-person visual storytelling.

**CRITICAL PERSPECTIVE RULE**: The camera IS the Unitree G1 humanoid robot's head-mounted camera at approximately 127cm height (chest-height to an adult human). This is FIRST-PERSON embodied walking navigation — the viewer sees through the robot's eyes. Never describe the robot externally. Never show the robot's body, arms, or legs.

**WAN 2.2 Requirements**: Cinematic prompts optimized for WAN 2.2's MoE text encoder (80-120 words). WAN processes prompts through a Mixture-of-Experts encoder that saturates beyond ~120 words — shorter prompts produce sharper, more coherent videos.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the environment, objects, and lighting. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the camera MOVES (direction, speed, gait dynamics)
2. What ENTERS and EXITS the frame during walking motion
3. TEMPORAL PROGRESSION of the navigation path

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**Your Task**: Transform simple humanoid navigation prompts into FIRST-PERSON walking POV cinematic descriptions (80-120 words) for WAN 2.2.

**Professional Camera Vocabulary** (WAN 2.2 responds strongly to these):
- Dolly forward/backward, tracking shot, steady advance
- Pan left/right, tilt up/down, gradual heading change
- Parallax shift, depth reveal, perspective pull
- Speed modifiers: "slow dolly", "steady tracking", "brisk walking pace"

**Bipedal Walking Camera Dynamics** (CRITICAL for realism):
- Subtle rhythmic vertical bob (~2-3cm per step cycle) from bipedal gait
- Natural head stabilization dampens most oscillation — NOT exaggerated bouncing
- Slight lateral sway synchronized with step rhythm
- Camera height stays approximately constant at ~127cm (G1 head height)
- Walking speed: slow (0.5 m/s), steady (1.0 m/s), brisk (1.5 m/s)
- Deceleration: steps shorten and slow before stopping

**Examples (FIRST-PERSON WALKING POV, ~100 words each)**:

**Walking Through Doorway** (~100 words):
"First-person POV at chest height, steady dolly forward through open doorway, soft indoor lighting from overhead fluorescents. Subtle rhythmic walking bob as polished floor scrolls beneath with strong parallax. Door frame grows larger in approach, edges sliding to periphery as the camera passes through. Beyond the threshold, new room opens up — desk and chair ahead with gentle depth reveal. Floor texture transitions from tile to carpet. Gradual deceleration as steps shorten, forward motion slowing to a stop with the desk centered in frame at comfortable viewing distance."

**Navigating Corridor** (~100 words):
"First-person walking POV at 127cm height, tracking forward through wide corridor, cool fluorescent lighting. Walls glide past on both sides with steady parallax — near wall details move fast, far end moves slowly. Subtle vertical bob from bipedal gait, horizon staying level through natural head stabilization. Approaching T-junction ahead, gradual pan right to preview the turn. Smooth heading change as environment sweeps left, new corridor opening up with depth reveal. Continuing forward at steady pace, floor flowing beneath, ceiling lights passing overhead rhythmically. Forward motion maintaining consistent elevation throughout."

**Critical Rules**:
- 80-120 words total (WAN 2.2 MoE sweet spot — longer prompts reduce quality)
- ALWAYS first-person walking perspective at ~127cm height
- Environment moves relative to YOUR walking motion
- Use professional camera terms: dolly, tracking, pan, parallax, depth reveal
- Include subtle bipedal walking bob — NOT exaggerated bouncing
- Do NOT re-describe what the image already shows
- ONE continuous 5-second shot only
- Do NOT add objects not in the original prompt
- NO flying, hovering, or aerial views (walking only)
- NO third-person descriptions ("robot walks", "humanoid moves")
- NO manipulation (no reaching, grasping, picking up objects — navigation ONLY)
- Camera height stays approximately constant throughout"""

# Negative prompt for G1 humanoid navigation (layered: WAN 2.2 default + physics + task-specific)
G1_NAV_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, deformed, still picture, messy background, flying, hovering, aerial view, drone, quadcopter, airborne, floating, manipulation, grasping, picking, arms extended toward objects, holding objects, robot arms visible, robot body visible, hands reaching, static camera, tripod, fixed position, third person view, external robot view, robot body visible, jerky motion, shaky footage, collision, crash, wall clipping, teleportation, flickering, jittering, sudden jump cuts, walking backwards, camera height changing, camera rising, camera dropping, ascending, descending, vertical movement, dynamic moving objects, furniture sliding, walls moving, floor warping, object drift, object morphing, scene breathing, jelly artifacts, geometry deformation, non-rigid background, camera zoom, focal-length shift, scene cut, jump cut"


class G1NavPromptExtender:
    """Generates cinematic G1 humanoid walking navigation prompts for WAN 2.2"""

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
        Generate enhanced cinematic prompt for G1 humanoid walking navigation

        Args:
            user_prompt: Simple user description
            image_path: Optional input image for I2V
            custom_system_prompt: Custom system prompt (optional, uses default if not provided)

        Returns:
            Tuple of (enhanced_prompt, negative_prompt)
        """
        logging.info("Generating enhanced G1 humanoid navigation prompt...")

        system_prompt = custom_system_prompt if custom_system_prompt else G1_NAV_SYSTEM_PROMPT

        query = f"""User wants to generate a humanoid robot walking navigation video with this description:
"{user_prompt}"

Generate a detailed, cinematic prompt for WAN 2.2 following the film director approach. Focus on first-person walking perspective at ~127cm height with subtle bipedal gait dynamics and beautiful cinematography. This is NAVIGATION ONLY — no manipulation."""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": []}
        ]

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

        return enhanced_prompt, G1_NAV_NEGATIVE_PROMPT

    def create_wan_config(self, enhanced_prompt: str, negative_prompt: str,
                         image_path: Optional[str] = None, output_name: str = "g1_nav") -> dict:
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

        prompt_file = OUTPUT_DIR / f"{output_base}_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(f"WAN 2.2 Unitree G1 Humanoid Navigation Prompt\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(enhanced_prompt)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Negative Prompt:\n{config.get('negative_prompt', 'None')}\n")

        config_file = OUTPUT_DIR / f"{output_base}_wan_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logging.info(f"Saved outputs:")
        logging.info(f"  Prompt: {prompt_file}")
        logging.info(f"  Config: {config_file}")

        return prompt_file, config_file

def main():
    parser = argparse.ArgumentParser(description='WAN 2.2 Unitree G1 Humanoid Navigation Prompt Extender')
    parser.add_argument('--prompt', type=str, required=True, help='User prompt describing humanoid navigation')
    parser.add_argument('--image', type=str, help='Optional input image for I2V')
    parser.add_argument('--system_prompt', type=str, default=None,
                       help='Custom system prompt (optional, uses default if not provided)')
    parser.add_argument('--output', type=str, default='g1_nav', help='Output filename base')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    extender = G1NavPromptExtender()

    enhanced_prompt, negative_prompt = extender.generate_prompt(args.prompt, args.image, args.system_prompt)

    config = extender.create_wan_config(enhanced_prompt, negative_prompt, args.image, args.output)

    prompt_file, config_file = extender.save_outputs(enhanced_prompt, config, args.output)

    print("\n" + "="*70)
    print("WAN 2.2 UNITREE G1 HUMANOID NAVIGATION PROMPT GENERATED")
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
