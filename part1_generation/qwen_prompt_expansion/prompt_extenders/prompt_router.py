#!/usr/bin/env python3
"""
Qwen3.5-9B Intelligent Prompt Router
Analyzes image and user prompt to automatically route to the appropriate specialized prompt extender
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Dict

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# Paths
SCRIPT_DIR = Path(__file__).parent
QWEN_MODEL_PATH = Path(os.getenv('QWEN_MODEL_PATH', 'Qwen/Qwen3.5-9B'))

# Task categories and their indicators
TASK_INDICATORS = {
    'drone_navigation': {
        'keywords': ['drone', 'aerial', 'flying', 'hovering', 'quadcopter', 'uav', 'fly', 'air'],
        'description': 'Aerial drone navigation and flight tasks'
    },
    'ground_navigation': {
        'keywords': ['navigate', 'move', 'walk', 'roll', 'drive', 'wheeled', 'mobile', 'robot',
                     'humanoid', 'go to', 'reach', 'approach', 'avoid', 'path', 'corridor',
                     'room', 'floor', 'ground', 'locomotion'],
        'description': 'Ground-based robot navigation (wheeled, tracked, humanoid)'
    },
    'bimanual_ur3': {
        'keywords': ['ur3', 'dual arm', 'two arms', 'bimanual', 'pick and place', 'grasp',
                     'manipulate', 'industrial robot', 'robotic arm', 'gripper', 'hold'],
        'description': 'Bimanual UR3 dual-arm manipulation tasks'
    },
    'unitree_g1': {
        'keywords': ['unitree', 'g1', 'humanoid manipulation', 'humanoid grasp',
                     'humanoid pick', 'humanoid arm', 'humanoid hand'],
        'description': 'Unitree G1 humanoid bimanual manipulation'
    }
}

# Router system prompt
ROUTER_SYSTEM_PROMPT = """You are an expert robotic vision system analyzer. Your task is to analyze an image and user prompt to determine the type of robotics task being requested.

Task Categories:
1. DRONE_NAVIGATION: Aerial drone flight, hovering, aerial navigation, UAV missions
2. GROUND_NAVIGATION: Wheeled robots, tracked robots, mobile robots, humanoid walking/locomotion
3. BIMANUAL_UR3: Industrial dual-arm manipulation, pick-and-place, UR3 robot arms
4. UNITREE_G1: Humanoid robot manipulation, humanoid grasping, Unitree G1 specific tasks

Analysis Instructions:
- Look at the image: Is there a drone/aerial vehicle, ground robot, robotic arms, or humanoid?
- Read the prompt: What action is requested? Navigation or manipulation?
- Consider context: Indoor/outdoor, objects to manipulate, movement patterns

Response Format (JSON only):
{
  "task_type": "DRONE_NAVIGATION|GROUND_NAVIGATION|BIMANUAL_UR3|UNITREE_G1",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this task type was chosen",
  "scene_description": "Description of what you see in the image",
  "detected_elements": ["element1", "element2", ...]
}

Be precise and confident in your classification. If unsure, default to GROUND_NAVIGATION for ground-based tasks or BIMANUAL_UR3 for manipulation."""


class PromptRouter:
    """Intelligent router that analyzes images and prompts to select appropriate extender"""

    def __init__(self, model_path: Path = QWEN_MODEL_PATH, device: str = 'cuda'):
        """Initialize Qwen3-VL model for routing"""
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else 'cpu'

        logging.info(f"Loading Qwen3.5-9B model from: {self.model_path}")
        logging.info(f"Using device: {self.device}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            trust_remote_code=True
        )

        if self.device != 'cuda':
            self.model = self.model.to(self.device)

        logging.info("Qwen3.5-9B model loaded successfully")

    def analyze_task(self, image_path: str, user_prompt: str) -> Dict:
        """
        Analyze image and prompt to determine task type

        Args:
            image_path: Path to input image
            user_prompt: User's description of desired action

        Returns:
            Dict with task_type, confidence, reasoning, etc.
        """
        logging.info("Analyzing task type...")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Create analysis prompt
        analysis_query = f"""User Prompt: "{user_prompt}"

Analyze the image and user prompt to determine the robotics task type. Respond with JSON only."""

        # Prepare messages (Qwen3.5: system content must be list of dicts)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": ROUTER_SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": analysis_query
                    }
                ]
            }
        ]

        # Process with Qwen3.5 (apply_chat_template handles tokenization + vision)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
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

        # Generate analysis
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        logging.info(f"Qwen3-VL Analysis Response:\n{response}")

        # Parse JSON response
        try:
            # Extract JSON from response (may have markdown formatting)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                # Fallback: try to parse entire response
                result = json.loads(response)

            # Validate result
            if 'task_type' not in result:
                raise ValueError("No task_type in response")

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Failed to parse JSON response: {e}")
            # Fallback: keyword-based analysis
            return self._fallback_analysis(user_prompt, image_path)

    def _fallback_analysis(self, user_prompt: str, image_path: str) -> Dict:
        """Fallback keyword-based analysis if JSON parsing fails"""
        logging.info("Using fallback keyword-based analysis")

        prompt_lower = user_prompt.lower()

        # Count matches for each category
        scores = {}
        for task_type, info in TASK_INDICATORS.items():
            score = sum(1 for keyword in info['keywords'] if keyword in prompt_lower)
            scores[task_type] = score

        # Get best match
        best_task = max(scores, key=scores.get)
        confidence = min(scores[best_task] / 5.0, 1.0)  # Normalize to 0-1

        return {
            'task_type': best_task.upper(),
            'confidence': confidence,
            'reasoning': f'Matched {scores[best_task]} keywords for {best_task}',
            'scene_description': 'Analysis based on text keywords',
            'detected_elements': [kw for kw in TASK_INDICATORS[best_task]['keywords'] if kw in prompt_lower]
        }

    def route_to_extender(self, analysis: Dict) -> Tuple[str, Path]:
        """
        Determine which prompt extender to use based on analysis

        Args:
            analysis: Task analysis dict from analyze_task()

        Returns:
            Tuple of (extender_name, extender_script_path)
        """
        task_type = analysis['task_type'].upper()

        extender_mapping = {
            'DRONE_NAVIGATION': ('wan22_drone', SCRIPT_DIR / 'wan22' / 'prompt_extender_drone.py'),
            'GROUND_NAVIGATION': ('wan22_ground', SCRIPT_DIR / 'wan22' / 'prompt_extender_ground_robot.py'),
            'BIMANUAL_UR3': ('cosmos_ur3', SCRIPT_DIR / 'cosmos25' / 'prompt_extender_bimanual_ur3.py'),
            'UNITREE_G1': ('cosmos_g1', SCRIPT_DIR / 'cosmos25' / 'prompt_extender_unitree_g1.py')
        }

        if task_type not in extender_mapping:
            logging.warning(f"Unknown task type: {task_type}, defaulting to ground navigation")
            task_type = 'GROUND_NAVIGATION'

        return extender_mapping[task_type]

def main():
    parser = argparse.ArgumentParser(
        description='Intelligent Prompt Router using Qwen3.5-9B'
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--prompt', type=str, required=True,
                       help='User prompt describing the desired action')
    parser.add_argument('--output', type=str, default='router_analysis.json',
                       help='Output JSON file with routing decision')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize router
    router = PromptRouter()

    # Analyze task
    analysis = router.analyze_task(args.image, args.prompt)

    # Determine extender
    extender_name, extender_path = router.route_to_extender(analysis)

    # Add extender info to analysis
    analysis['selected_extender'] = extender_name
    analysis['extender_script'] = str(extender_path)

    # Save result
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("QWEN3-VL ROUTING ANALYSIS")
    print("="*60)
    print(f"Task Type: {analysis['task_type']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Reasoning: {analysis['reasoning']}")
    print(f"Selected Extender: {extender_name}")
    print(f"Extender Script: {extender_path}")
    print(f"\nScene Description:\n{analysis.get('scene_description', 'N/A')}")
    print(f"\nDetected Elements: {', '.join(analysis.get('detected_elements', []))}")
    print("="*60)
    print(f"\nAnalysis saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
