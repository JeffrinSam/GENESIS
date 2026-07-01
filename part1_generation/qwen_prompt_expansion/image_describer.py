#!/usr/bin/env python3
"""
Qwen3-VL Image Describer Interface
Engineered by Jeffrin Sam / Modified for custom prompts

A standalone interface for Qwen3-VL that allows:
- Customizable system prompts
- Image analysis and description
- Flexible user queries

Uses existing Qwen3-VL/.venv and Qwen3.5-9B model.
Does not modify any existing codebase files.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Union, Optional

try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM
except ImportError as e:
    print(f"Error: Missing dependencies - {e}")
    print("Install with: pip install torch transformers pillow")
    sys.exit(1)


# Default system prompt for image description
DEFAULT_SYSTEM_PROMPT = """You are an expert visual analysis system. Describe what you see in the image accurately and concisely.

Guidelines:
- Focus on key objects, their positions, and spatial relationships
- Note any robots, humans, or relevant equipment
- Describe the environment (indoor/outdoor, lighting, layout)
- Mention any actions or events occurring
- Be objective and factual
- Keep descriptions clear and structured
"""


def load_system_prompt(prompt_str: Optional[str], prompt_file: Optional[str]) -> str:
    """Load system prompt from string or file."""
    if prompt_file:
        path = Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_file}")
        return path.read_text().strip()
    return prompt_str or DEFAULT_SYSTEM_PROMPT


def build_messages(
    system_prompt: str,
    user_query: str,
    image_path: Optional[str] = None
) -> List[Dict[str, Union[str, List[Dict]]]]:
    """Build message list for Qwen3-VL."""
    
    # Build user content
    if image_path:
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": user_query}
        ]
    else:
        content = [{"type": "text", "text": user_query}]
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def run_inference(
    model_path: str,
    messages: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.4,
    top_p: float = 0.9,
    flash_attn2: bool = False,
) -> str:
    """Run inference with Qwen3-VL model."""
    
    attn_impl = "flash_attention_2" if flash_attn2 else None
    
    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Prepare inputs using apply_chat_template with tokenize=True
    # This is the correct way for Qwen3-VL multimodal inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Filter inputs to only include keys the model expects
    # Qwen3-VL may include extra keys that cause errors with generate()
    # Note: pixel_values is processed internally by the model's vision encoder
    allowed_keys = ['input_ids', 'attention_mask']
    filtered_inputs = {k: v for k, v in inputs.items() if k in allowed_keys}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **filtered_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    
    # Decode output
    trimmed = generated_ids[:, filtered_inputs["input_ids"].shape[1]:]
    output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return output.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Image Describer with Customizable System Prompt"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        default=os.getenv('QWEN_MODEL_PATH', 'Qwen/Qwen3.5-9B'),
        help="Path to Qwen3-VL model or HuggingFace model ID (set QWEN_MODEL_PATH env var)"
    )
    
    # System prompt options
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt as a string"
    )
    prompt_group.add_argument(
        "--system-prompt-file",
        type=str,
        help="Path to file containing system prompt"
    )
    
    # Input options
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file to analyze"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Describe what you see in this image in detail.",
        help="User query/question about the image"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        help="Use Flash Attention 2"
    )
    
    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON with metadata"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    
    args = parser.parse_args()
    
    # Validate image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    # Load system prompt
    try:
        system_prompt = load_system_prompt(args.system_prompt, args.system_prompt_file)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Build messages
    messages = build_messages(system_prompt, args.query, args.image)
    
    # Run inference
    try:
        result = run_inference(
            model_path=args.model_path,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            flash_attn2=args.flash_attn2,
        )
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Format output
    if args.json:
        output = {
            "description": result,
            "image": args.image,
            "query": args.query,
            "system_prompt_used": args.system_prompt_file or ("custom" if args.system_prompt else "default"),
            "generation_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        }
        output_str = json.dumps(output, indent=2)
    else:
        output_str = result
    
    # Output result
    if args.output:
        Path(args.output).write_text(output_str)
        print(f"Output saved to: {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()