#!/usr/bin/env python3
"""
Cosmos 2.5 Unitree G1 Humanoid Manipulation Prompt Extender
Generates physics-based, detailed prompts for Unitree G1 humanoid bimanual manipulation using Cosmos 2.5 physics approach
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

# Cosmos 2.5 Unitree G1 System Prompt (Physics Engineer Approach — Research-Optimized v2)
# Key changes from v1 (backed by 45+ research papers, NVIDIA docs, community guides):
# 1. Output length: 150-300 → 100-150 words (Cosmos trained on ~97-word captions)
# 2. Narrative style: "The video shows..." (matches Cosmos training distribution)
# 3. I2V rule: Don't re-describe what image already shows
# 4. Force causality chains: Describe CAUSES not just outcomes (PhyT2V: 2.3x improvement)
# 5. Examples shortened to ~120 words (match training distribution)
# 6. One-scene rule: Single continuous 5-second shot
# Sources: arxiv 2501.03575, DiffPhy (2505.21653), PhyT2V (CVPR 2025), NVIDIA Cosmos docs
G1_SYSTEM_PROMPT = """You are an expert humanoid robotics engineer and physicist specializing in creating detailed, physics-based prompts for Cosmos 2.5 video generation. You understand humanoid manipulation, bimanual coordination, anthropomorphic kinematics, and dexterous object interaction.

**Cosmos 2.5 Requirements**: Physics-based narrative descriptions (100-150 words). Cosmos was trained on ~100-word video captions — prompts longer than 150 words get diluted by the text encoder and reduce quality.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the scene, robot, objects, and environment. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the robot MOVES (joint rotations, trajectories, speeds)
2. FORCE CAUSALITY: What force initiates action → how material responds → resulting motion
3. TEMPORAL PROGRESSION of the manipulation task

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**NARRATIVE STYLE**: Write as if narrating a video being played:
- "The video shows the robot's arms beginning to lift..."
- "As the sequence progresses, the gripper fingers close..."
- Do NOT use imperative commands like "The robot picks up the bottle."

**Your Task**: Transform simple Unitree G1 manipulation prompts into physics-grounded narrative descriptions (100-150 words) for Cosmos 2.5.

**Required Elements**:

1. **Temporal Action Sequence** (Critical — drives video coherence):
   - **Initial State** (0-20%): Brief setup, arms at rest
   - **Approach** (20-40%): Arms extend, joints rotate, hands near object
   - **Grasp** (40-60%): Fingers pre-shape, contact, force application
   - **Manipulation** (60-85%): Object movement with physics causality
   - **Completion** (85-100%): Release, retraction to neutral

2. **Physics Causality Chain** (Critical — 2.3x improvement in physics realism):
   For each action, describe CAUSE → MATERIAL RESPONSE → RESULT:
   - CAUSE: "Gripper fingers apply 3N lateral force to plastic surface..."
   - RESPONSE: "...rubber pads deform 1mm, increasing contact area and friction..."
   - RESULT: "...bottle lifts smoothly at 5cm/s, maintaining vertical orientation"

3. **Humanoid-Specific Details**:
   - Joint kinematics: shoulder, elbow, wrist angles and rotations
   - Hand dynamics: finger pre-shaping, force distribution, thumb opposition
   - Balance: torso lean compensation during reach
   - Bimanual coordination: synchronized vs complementary roles

**Examples**:

**Bimanual Object Pickup** (~120 words):
"The video shows a Unitree G1 humanoid robot standing before a table in a laboratory. A transparent plastic bottle rests on the surface. The robot's shoulder joints activate bilaterally, both arms lifting in smooth arcs as elbows flex to bring multi-fingered hands toward the bottle. Fingers pre-shape into curved configurations 5cm before contact. As fingertips meet the smooth plastic, they close progressively, applying distributed force across finger pads while thumb opposition provides lateral constraint. The bimanual grasp secure, both arms begin synchronized vertical motion, lifting the bottle at 8cm/s. The torso compensates with a slight forward lean as arm moments increase. Wrists adjust continuously to maintain the bottle's vertical orientation. After lifting 20cm and translating 30cm horizontally, both arms descend to place the bottle, fingers extending to release before arms retract to neutral."

**Single-Arm Precision Pick** (~110 words):
"The video captures the G1's right arm initiating movement, shoulder rotating forward as the elbow extends toward a ceramic mug on the counter. The multi-fingered hand pre-shapes into a wrap grasp, fingers curving to match the mug's cylindrical profile. Contact occurs as rubber fingertips press against the glazed ceramic surface, friction coefficient increasing as the grip tightens with 4N of distributed radial force. The mug lifts cleanly from the surface, the arm's elbow flexing to raise it 15cm while the torso shifts slightly left to maintain center-of-mass balance. The wrist rotates to keep the mug level, preventing liquid spillage. The arm translates the mug 25cm rightward before lowering and releasing with controlled finger extension."

**Critical Rules**:
- NARRATIVE STYLE: "The video shows..." (NOT imperative commands)
- 100-150 words total (Cosmos sweet spot — longer prompts reduce quality)
- Include force causality chains (what force → material response → resulting motion)
- Temporal progression: initial → approach → grasp → manipulate → complete
- Do NOT re-describe what the image already shows
- Do NOT add objects not in the original prompt
- ONE continuous 5-second shot only
- Physics language: forces, friction, deformation, gravity compensation
- Humanoid specifics: joint angles, bimanual coordination, balance
- NO flying, aerial views, wheeled locomotion, or pure industrial robotics"""

# Negative prompt for Unitree G1 (layered: technical + physics + task-specific)
G1_NEGATIVE_PROMPT = "flying, hovering, drone, aerial navigation, wheeled robot, tracked vehicle, quadruped walking, industrial robotic arm without humanoid body, single arm, missing torso, non-humanoid proportions, floating objects, unrealistic physics, teleportation, cartoonish, jerky motion, unstable balance, collision, low quality, blurry, phasing through objects, impossible joint angle, gripper passing through table, flickering, morphing, warping, sudden changes, overexposed, worst quality, compression artifacts, inconsistent lighting, extra fingers, deformed hands, still picture"


class UnitreeG1PromptExtender:
    """Generates physics-based Unitree G1 humanoid manipulation prompts for Cosmos 2.5"""

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
        Generate enhanced physics-based prompt

        Args:
            user_prompt: Simple user description
            image_path: Optional input image for I2V
            custom_system_prompt: Custom system prompt (optional, uses default if not provided)

        Returns:
            Tuple of (enhanced_prompt, negative_prompt)
        """
        logging.info("Generating enhanced Unitree G1 humanoid manipulation prompt...")

        # Use custom system prompt if provided, otherwise use default
        system_prompt = custom_system_prompt if custom_system_prompt else G1_SYSTEM_PROMPT

        # Build query
        query = f"""User wants to generate a Unitree G1 humanoid bimanual manipulation video with this description:
"{user_prompt}"

Generate a detailed, physics-based prompt for Cosmos 2.5 following the physics engineer approach. Focus on humanoid kinematics, dexterous bimanual coordination, anthropomorphic motion, temporal progression, and realistic physical interactions."""

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
                max_new_tokens=800,
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

        return enhanced_prompt, G1_NEGATIVE_PROMPT

    def create_cosmos_config(self, enhanced_prompt: str, negative_prompt: str,
                            image_path: str, output_name: str = "g1_manipulation") -> dict:
        """Create Cosmos 2.5 JSON configuration"""

        config = {
            "inference_type": "image2world",
            "name": output_name,
            "input_path": str(Path(image_path).absolute()),
            "prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "num_output_frames": 77,
            "resolution": "432,768",
            "seed": 42,
            "guidance": 7
        }

        return config

    def save_outputs(self, enhanced_prompt: str, config: dict, output_base: str):
        """Save prompt and config files"""
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Save human-readable prompt
        prompt_file = OUTPUT_DIR / f"{output_base}_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(f"Cosmos 2.5 Unitree G1 Humanoid Manipulation Prompt\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.write(enhanced_prompt)
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Negative Prompt:\n{config.get('negative_prompt', 'None')}\n")

        # Save Cosmos config
        config_file = OUTPUT_DIR / f"{output_base}_cosmos_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logging.info(f"Saved outputs:")
        logging.info(f"  Prompt: {prompt_file}")
        logging.info(f"  Config: {config_file}")

        return prompt_file, config_file


def main():
    parser = argparse.ArgumentParser(description='Cosmos 2.5 Unitree G1 Humanoid Manipulation Prompt Extender')
    parser.add_argument('--prompt', type=str, required=True, help='User prompt describing humanoid manipulation task')
    parser.add_argument('--image', type=str, required=True, help='Input image showing initial scene (required for Cosmos)')
    parser.add_argument('--system_prompt', type=str, default=None,
                       help='Custom system prompt (optional, uses default if not provided)')
    parser.add_argument('--output', type=str, default='g1_manipulation', help='Output filename base')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize extender
    extender = UnitreeG1PromptExtender()

    # Generate prompt
    enhanced_prompt, negative_prompt = extender.generate_prompt(args.prompt, args.image, args.system_prompt)

    # Create config
    config = extender.create_cosmos_config(enhanced_prompt, negative_prompt, args.image, args.output)

    # Save outputs
    prompt_file, config_file = extender.save_outputs(enhanced_prompt, config, args.output)

    # Print summary
    print("\n" + "="*70)
    print("COSMOS 2.5 UNITREE G1 HUMANOID MANIPULATION PROMPT GENERATED")
    print("="*70)
    print(f"\nUser Prompt:\n{args.prompt}")
    print(f"\nEnhanced Prompt ({len(enhanced_prompt.split())} words):\n{enhanced_prompt}")
    print(f"\nNegative Prompt:\n{negative_prompt}")
    print(f"\nInference Type: {config['inference_type']}")
    print(f"Resolution: {config['resolution']}")
    print(f"Frames: {config['num_output_frames']}")
    print(f"Guidance: {config['guidance']}")
    print(f"\nFiles saved to:")
    print(f"  - {prompt_file}")
    print(f"  - {config_file}")
    print("="*70)

    print(f"\nTo generate video with Cosmos 2.5:")
    print(f"cd $COSMOS_ROOT")
    print(f"python inference_i2w.py \\")
    print(f"  --checkpoint_dir ./checkpoints/Cosmos-2.5-Predict-14B \\")
    print(f"  --input_path {config['input_path']} \\")
    print(f"  --prompt \"{enhanced_prompt}\" \\")
    print(f"  --guidance {config['guidance']} \\")
    print(f"  --num_output_frames {config['num_output_frames']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
