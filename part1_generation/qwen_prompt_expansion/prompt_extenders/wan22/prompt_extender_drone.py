#!/usr/bin/env python3
"""
WAN 2.2 Drone Navigation Prompt Extender
Generates cinematic, detailed prompts for aerial drone navigation using WAN 2.2 film director approach
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

# WAN 2.2 Drone System Prompt (Film Director Approach — Research-Optimized v2)
# Key changes: output 100-200→80-120 words, I2V rule, professional camera vocab, speed modifiers
# Sources: InstaSD, Story321, PromptSloth, ViewComfy, HuggingFace WAN guide, WAN 2.2 GitHub
DRONE_SYSTEM_PROMPT = """You are an expert aerial cinematographer and drone pilot specializing in creating cinematic video prompts for WAN 2.2 video generation. You understand film theory, aerial photography, and UAV flight dynamics.

**CRITICAL PERSPECTIVE RULE**: The camera IS the drone's eyes. This is FIRST-PERSON embodied navigation - the viewer sees through the drone's perspective as it flies. Never describe the drone externally. The world moves relative to the camera's motion.

**WAN 2.2 Requirements**: Cinematic aesthetics with detailed visual elements (80-120 words). WAN 2.2's MoE architecture performs best at 80-120 words — shorter under-specifies, longer dilutes intent and introduces contradictions.

**IMAGE-TO-VIDEO CRITICAL RULE**: The source image already shows the environment, objects, and scene. DO NOT re-describe what is visible. Focus ONLY on:
1. HOW the camera MOVES (direction, speed, trajectory)
2. What ENTERS the view as motion progresses
3. CAMERA technique (dolly, tracking, crane, pan)

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts.

**Your Task**: Transform simple drone navigation prompts into FIRST-PERSON cinematic descriptions (80-120 words) for WAN 2.2.

**Required Elements** (Choose 2-3 from each):

1. **Lighting & Atmosphere**:
   - Time: Daylight, Dawn, Dusk, Night
   - Quality: Soft lighting, Hard lighting, Volumetric light
   - Tone: Warm colors, Cool colors, Teal-and-orange

2. **Professional Camera Vocabulary** (WAN 2.2 responds strongly to these):
   - Forward: "dolly forward", "tracking shot forward", "smooth glide"
   - Turning: "pan left/right", "orbital arc", "banking turn"
   - Vertical: "crane up/down", "ascending smoothly", "descending"
   - Complex: "parallax reveal" (foreground moves, background steady)
   - Speed: "slow dolly", "rapid tracking", "gentle crane"
   - Depth: "shallow depth of field", "deep focus"
   - AVOID: "whip pan" (WAN 2.2 struggles with this)

3. **First-Person Motion** (CRITICAL):
   - Start: "Starting at X meters altitude, hovering above..."
   - Motion: "Gliding forward smoothly, dolly tracking along the path..."
   - Reveal: "Banking gently right, parallax reveal of river below..."
   - End: "Slowing to hover, crane descending toward the surface..."
   - Environment responds: "Canopy flowing beneath", "Horizon tilting", "Ground rising closer"

**Structure**: [Lighting] + [Starting position] + [First-person motion with camera terms] + [Reveal/ending]

**Example** (~100 words):
"Daylight, soft warm lighting, first-person aerial POV, wide shot. Hovering above a lush green forest, dense emerald canopy stretching ahead. Smooth dolly forward as the forest flows steadily beneath. Sunlight filters through scattered clouds, casting dynamic shadows on passing foliage. Gentle banking turn right with parallax effect — foreground treetops shift rapidly while distant mountains remain steady. A winding river reveals itself from behind the treeline, growing larger in frame. Slow crane descent, bringing the shimmering water surface closer into view. Shallow depth of field softens the distant forest edge."

**WRONG**: "A quadcopter drone flies over forest. The drone banks right."
**RIGHT**: "Dolly forward over forest canopy. Gentle banking right, parallax revealing river below."

**Critical Rules**:
- ✅ ALWAYS first-person (YOU are the drone)
- ✅ Use professional camera terms: dolly, tracking, crane, pan, orbital arc
- ✅ Include speed modifiers: "slowly", "smoothly", "gently", "rapidly"
- ✅ Include parallax/depth cues when turning
- ✅ Describe what enters YOUR view as you move
- ❌ NEVER mention "drone" as external object
- ❌ NEVER re-describe what the image already shows
- ❌ NEVER use "whip pan" (WAN 2.2 limitation)
- 80-120 words total (WAN sweet spot)
- ONE continuous shot, no scene cuts"""

# Negative prompt for drone navigation (WAN 2.2 default + task-specific)
DRONE_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, deformed, still picture, messy background, walking backwards, ground vehicle, wheeled robot, walking, indoor, confined space, manipulation, grasping, arms, hands, static camera, tripod, fixed position, jerky motion, shaky footage, crash, collision, wall clipping, teleportation, flickering, jittering, sudden jump cuts"


class DronePromptExtender:
    """Generates cinematic drone prompts for WAN 2.2"""

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
        logging.info("Generating enhanced drone navigation prompt...")

        # Use custom system prompt if provided, otherwise use default
        system_prompt = custom_system_prompt if custom_system_prompt else DRONE_SYSTEM_PROMPT

        # Build query with explicit image understanding step
        if image_path:
            query = f"""STEP 1: First, carefully analyze the provided image and describe:
- What environment/scene is shown (forest, city, ocean, mountains, etc.)
- Lighting conditions (daylight, sunset, overcast, etc.)
- Key visual elements and landmarks
- Spatial layout and depth

STEP 2: User wants to create a drone navigation video starting from this scene with the following intent:
"{user_prompt}"

STEP 3: Now, based on:
- The actual environment you see in the image
- The user's navigation intent: "{user_prompt}"
- WAN 2.2 film director approach requirements

Generate a detailed, cinematic FIRST-PERSON prompt that:
1. Starts from the actual scene shown in the image
2. Executes the user's intended motion: "{user_prompt}"
3. Uses proper cinematography (lighting, colors, camera work)
4. Maintains first-person drone perspective throughout
5. Is 100-200 words

Output ONLY the final cinematic prompt, nothing else."""
        else:
            query = f"""User wants to generate a drone navigation video with this description:
"{user_prompt}"

Generate a detailed, cinematic prompt for WAN 2.2 following the film director approach. Focus on aerial perspective, smooth flight dynamics, and beautiful cinematography."""

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": []}
        ]

        # Add image if provided
        if image_path:
            image = Image.open(image_path).convert('RGB')
            messages[1]["content"].append({"type": "image", "image": image})
            messages[1]["content"].append({"type": "text", "text": query})
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

        return enhanced_prompt, DRONE_NEGATIVE_PROMPT

    def create_wan_config(self, enhanced_prompt: str, negative_prompt: str,
                         image_path: Optional[str] = None, output_name: str = "drone_navigation") -> dict:
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
            f.write(f"WAN 2.2 Drone Navigation Prompt\n")
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
    parser = argparse.ArgumentParser(description='WAN 2.2 Drone Navigation Prompt Extender')
    parser.add_argument('--prompt', type=str, required=True, help='User prompt describing drone action')
    parser.add_argument('--image', type=str, help='Optional input image for I2V')
    parser.add_argument('--system_prompt', type=str, default=None,
                       help='Custom system prompt (optional, uses default if not provided)')
    parser.add_argument('--output', type=str, default='drone_nav', help='Output filename base')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize extender
    extender = DronePromptExtender()

    # Generate prompt
    enhanced_prompt, negative_prompt = extender.generate_prompt(args.prompt, args.image, args.system_prompt)

    # Create config
    config = extender.create_wan_config(enhanced_prompt, negative_prompt, args.image, args.output)

    # Save outputs
    prompt_file, config_file = extender.save_outputs(enhanced_prompt, config, args.output)

    # Print summary
    print("\n" + "="*70)
    print("WAN 2.2 DRONE NAVIGATION PROMPT GENERATED")
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
