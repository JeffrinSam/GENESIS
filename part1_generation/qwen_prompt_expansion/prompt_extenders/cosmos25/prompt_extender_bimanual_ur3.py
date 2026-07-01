#!/usr/bin/env python3
"""
Cosmos 2.5 Bimanual UR3 Manipulation Prompt Extender
Generates physics-based, detailed prompts for dual-arm UR3 manipulation using Cosmos 2.5 physics approach
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

# Cosmos 2.5 Bimanual UR3 System Prompt (Physics Engineer Approach — Research-Optimized v2)
# Key changes: output length 150-300→100-150, narrative style, I2V rule, force causality, shorter examples
# Sources: arxiv 2501.03575, DiffPhy (2505.21653), PhyT2V (CVPR 2025), NVIDIA Cosmos docs
UR3_SYSTEM_PROMPT = """You are an expert robotics engineer and physicist specializing in creating detailed, physics-based prompts for Cosmos 2.5 video generation. You understand industrial robotics, dual-arm manipulation, kinematics, and object interaction dynamics.

**Cosmos 2.5 Requirements**: Physics-based narrative descriptions (100-150 words). Cosmos was trained on ~100-word video captions — prompts longer than 150 words get diluted by the text encoder and reduce quality.

**IMAGE-TO-VIDEO CRITICAL RULE**: When an image is provided, it already shows the scene, robot, objects, and environment. DO NOT waste words re-describing visible elements. Focus ONLY on:
1. HOW the arms MOVE (joint rotations, trajectories, speeds)
2. FORCE CAUSALITY: What force initiates action → how material responds → resulting motion
3. TEMPORAL PROGRESSION of the manipulation task

**ONE SCENE RULE**: Describe ONE continuous 5-second shot. No scene cuts, no time skips.

**NARRATIVE STYLE**: Write as if narrating a video being played:
- "The video shows both UR3 arms activating simultaneously..."
- "As the grippers approach, the fingers begin to open..."
- Do NOT use imperative commands.

**Your Task**: Transform simple bimanual UR3 manipulation prompts into physics-grounded narrative descriptions (100-150 words) for Cosmos 2.5.

**Required Elements**:

1. **Temporal Action Sequence** (Critical — drives video coherence):
   - **Initial State** (0-20%): Arms at rest, grippers open, object visible
   - **Approach** (20-40%): Arms extend, joints rotate, grippers near object
   - **Grasp** (40-60%): Grippers close, contact forces applied, stable hold
   - **Manipulation** (60-85%): Object movement with dual-arm coordination
   - **Completion** (85-100%): Placement, gripper release, arms retract

2. **Physics Causality Chain** (Critical — 2.3x improvement in physics realism):
   For each action, describe CAUSE → MATERIAL RESPONSE → RESULT:
   - CAUSE: "Parallel jaw grippers apply 5N lateral force to cube faces..."
   - RESPONSE: "...plastic surface compresses 0.5mm under gripper pads, friction holds..."
   - RESULT: "...cube lifts steadily at 5cm/s, maintaining upright orientation"

3. **Dual-Arm Coordination**:
   - Synchronized vs sequential motion
   - Load sharing and complementary roles
   - Joint-level coordination (6 DOF per arm)

**Examples**:

**Pick and Place Task** (~120 words):
"The video shows a dual-arm UR3 system with both blue metallic arms activating simultaneously, shoulder and elbow joints rotating smoothly as they extend toward a red plastic cube on the work surface. The parallel jaw grippers open to 10cm width as they approach from opposite sides. As the gripper fingers contact the cube's faces, they apply balanced lateral force of 5N per side, the parallel jaw mechanisms compressing slightly against the plastic surface to establish friction-based hold. Both arms begin coordinated vertical motion, lifting the cube at 5cm/s while joint actuators continuously adjust to maintain level orientation. The arms translate the object 40cm horizontally through synchronized shoulder and elbow movements before descending to place it at the new position. Grippers release simultaneously and arms retract to rest."

**Coordinated Assembly** (~120 words):
"The video captures the left UR3 arm descending first, its gripper fingers closing around a threaded aluminum base cylinder with controlled radial force. The arm lifts and positions the base centrally, orienting the threaded top upward. The right arm simultaneously grasps the smaller cap cylinder from above. With both components secured, the right arm executes a precise descent, aligning the cap's internal threads with the base's external threads through subtle 6-DOF adjustments. As threading surfaces make contact, the right arm begins controlled clockwise rotation while applying gentle downward force, threads engaging progressively with each turn. The left arm provides reactive stabilization, counteracting transmitted torques. After full seating, both grippers release and arms withdraw to rest positions."

**Critical Rules**:
- NARRATIVE STYLE: "The video shows..." (NOT imperative commands)
- 100-150 words total (Cosmos sweet spot — longer prompts reduce quality)
- Include force causality chains (what force → material response → resulting motion)
- Temporal progression: initial → approach → grasp → manipulate → complete
- Do NOT re-describe what the image already shows
- Do NOT add objects not in the original prompt
- ONE continuous 5-second shot only
- Physics language: forces, friction, deformation, contact dynamics
- Dual-arm specifics: synchronized motion, load sharing, complementary roles
- NO flying, aerial views, navigation, or humanoid locomotion"""

# Negative prompt for bimanual UR3 (layered: technical + physics + task-specific)
UR3_NEGATIVE_PROMPT = "flying, hovering, drone, aerial navigation, wheeled robot, walking, humanoid locomotion, single arm, missing arm, fewer than two arms, cartoonish, unrealistic physics, teleportation, object floating, unstable grasp, collision, jerky motion, low quality, blurry, phasing through objects, impossible joint angle, gripper passing through table, flickering, morphing, warping, sudden changes, overexposed, worst quality, compression artifacts, inconsistent lighting, extra fingers, deformed, still picture"


class BimanualUR3PromptExtender:
    """Generates physics-based bimanual UR3 manipulation prompts for Cosmos 2.5"""

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
        logging.info("Generating enhanced bimanual UR3 manipulation prompt...")

        # Use custom system prompt if provided, otherwise use default
        system_prompt = custom_system_prompt if custom_system_prompt else UR3_SYSTEM_PROMPT

        # Build query
        query = f"""User wants to generate a bimanual UR3 robotic manipulation video with this description:
"{user_prompt}"

Generate a detailed, physics-based prompt for Cosmos 2.5 following the physics engineer approach. Focus on dual-arm coordination, object manipulation dynamics, temporal progression, and realistic physical interactions."""

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

        return enhanced_prompt, UR3_NEGATIVE_PROMPT

    def create_cosmos_config(self, enhanced_prompt: str, negative_prompt: str,
                            image_path: str, output_name: str = "ur3_manipulation") -> dict:
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
            f.write(f"Cosmos 2.5 Bimanual UR3 Manipulation Prompt\n")
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
    parser = argparse.ArgumentParser(description='Cosmos 2.5 Bimanual UR3 Manipulation Prompt Extender')
    parser.add_argument('--prompt', type=str, required=True, help='User prompt describing manipulation task')
    parser.add_argument('--image', type=str, required=True, help='Input image showing initial scene (required for Cosmos)')
    parser.add_argument('--system_prompt', type=str, default=None,
                       help='Custom system prompt (optional, uses default if not provided)')
    parser.add_argument('--output', type=str, default='ur3_manipulation', help='Output filename base')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize extender
    extender = BimanualUR3PromptExtender()

    # Generate prompt
    enhanced_prompt, negative_prompt = extender.generate_prompt(args.prompt, args.image, args.system_prompt)

    # Create config
    config = extender.create_cosmos_config(enhanced_prompt, negative_prompt, args.image, args.output)

    # Save outputs
    prompt_file, config_file = extender.save_outputs(enhanced_prompt, config, args.output)

    # Print summary
    print("\n" + "="*70)
    print("COSMOS 2.5 BIMANUAL UR3 MANIPULATION PROMPT GENERATED")
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
