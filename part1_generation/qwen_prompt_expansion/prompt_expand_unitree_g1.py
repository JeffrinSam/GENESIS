import argparse
import os

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


SYSTEM_PROMPT = """You are an expert robotic prompter for Unitree G1 manipulation.
Your job is to rewrite a short human instruction into the best possible manipulation prompt that a robot policy/controller can follow.

Priorities:
- Safety-first: avoid harming humans, the robot, and the environment.
- Robustness: include preconditions, perception checks, recovery behaviors, stop conditions.
- Clarity: explicit object, grasp strategy, approach direction, force limits, and placement target if implied.
- Grounding: specify what to look for (bottle pose, free space, obstacles) and what success means.
- Output only the rewritten prompt (no explanations, no extra commentary).
"""


def _build_messages(task: str):
    user_text = f"""Short task instruction: \"{task}\"

Rewrite it into a single, detailed “robot execution prompt” for Unitree G1 arm manipulation.
Include:
- assumptions to verify
- step-by-step plan (numbered)
- grasp strategy (where to grasp, approach vector, closure/force guidance)
- safety constraints
- failure handling / retries
- clear success criteria
"""

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default=os.getenv('QWEN_MODEL_PATH', 'Qwen/Qwen3.5-9B'),
        help="Local path or HuggingFace model ID (set QWEN_MODEL_PATH env var)",
    )
    ap.add_argument("--task", default="pick up the bottle")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--flash-attn2", action="store_true")
    args = ap.parse_args()

    attn_impl = "flash_attention_2" if args.flash_attn2 else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    messages = _build_messages(args.task)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
    out = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(out.strip())


if __name__ == "__main__":
    main()
