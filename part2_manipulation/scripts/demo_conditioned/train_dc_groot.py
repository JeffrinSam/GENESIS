#!/usr/bin/env python3
"""
Train Demo-Conditioned GR00T (DC-GR00T)

Train GR00T to execute tasks based on demonstration videos.
The robot watches a demo (human/robot/Cosmos) to understand what to do,
then executes using its own observation (closed-loop control).

Usage:
    python train_dc_groot.py \
        --dataset_path /path/to/dataset \
        --output_dir ./dc_checkpoints \
        --pretrained_groot nvidia/GR00T-N1.6-3B

Dataset format:
    dataset/
    ├── episodes.jsonl
    ├── videos/
    │   ├── ego_view/          # Robot's ego view during execution
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   └── demo/              # Demo videos (can be human, robot, cosmos)
    │       ├── episode_000000.mp4
    │       └── ...
    └── data/
        ├── episode_000000.parquet  # Actions, states, etc.
        └── ...
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from gr00t.model.demo_conditioned import DCGr00t, DCGr00tConfig, DemoEncoder
from gr00t.model.demo_conditioned.demo_encoder import VideoAlignmentLoss
from gr00t.data.embodiment_tags import EmbodimentTag


class DCDataset(Dataset):
    """
    Dataset for Demo-Conditioned GR00T training.

    Each sample contains:
    - Demo video (what to do)
    - Robot observation during execution
    - Robot state
    - Actions taken
    """

    def __init__(
        self,
        dataset_path: str,
        num_demo_keyframes: int = 16,
        num_obs_frames: int = 1,
        action_horizon: int = 16,
        demo_subdir: str = "demo",
        ego_subdir: str = "ego_view",
    ):
        self.dataset_path = Path(dataset_path)
        self.num_demo_keyframes = num_demo_keyframes
        self.num_obs_frames = num_obs_frames
        self.action_horizon = action_horizon
        self.demo_subdir = demo_subdir
        self.ego_subdir = ego_subdir

        # Load episodes
        self.episodes = self._load_episodes()
        print(f"Loaded {len(self.episodes)} episodes from {dataset_path}")

    def _load_episodes(self) -> List[Dict]:
        episodes_file = self.dataset_path / "episodes.jsonl"
        episodes = []

        if episodes_file.exists():
            with open(episodes_file, "r") as f:
                for line in f:
                    episodes.append(json.loads(line.strip()))
        else:
            # Scan for videos
            demo_dir = self.dataset_path / "videos" / self.demo_subdir
            if demo_dir.exists():
                for video_file in sorted(demo_dir.glob("*.mp4")):
                    episodes.append({
                        "episode_id": video_file.stem,
                    })

        return episodes

    def _load_video(self, path: Path, num_frames: int) -> np.ndarray:
        """Load and sample video frames."""
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(path), ctx=cpu(0))
            T = len(vr)
            indices = np.linspace(0, T - 1, num_frames, dtype=np.int64)
            frames = vr.get_batch(indices).asnumpy()
            return frames
        except ImportError:
            import cv2
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            frames = np.stack(frames)
            T = len(frames)
            indices = np.linspace(0, T - 1, num_frames, dtype=np.int64)
            return frames[indices]

    def _load_actions(self, episode_id: str) -> Dict[str, np.ndarray]:
        """Load action data."""
        parquet_path = self.dataset_path / "data" / f"{episode_id}.parquet"
        if parquet_path.exists():
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            return {col: df[col].values for col in df.columns}
        return {}

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode = self.episodes[idx]
        episode_id = episode.get("episode_id", f"episode_{idx:06d}")

        # Load demo video
        demo_path = self.dataset_path / "videos" / self.demo_subdir / f"{episode_id}.mp4"
        demo_frames = self._load_video(demo_path, self.num_demo_keyframes)

        # Load robot ego view (sample a random timestep)
        ego_path = self.dataset_path / "videos" / self.ego_subdir / f"{episode_id}.mp4"
        try:
            from decord import VideoReader, cpu
            import cv2
            vr = VideoReader(str(ego_path), ctx=cpu(0))
            T = len(vr)
            # Random starting point
            start_idx = np.random.randint(0, max(1, T - self.action_horizon))
            ego_frame = vr[start_idx].asnumpy()
            # Resize to 448x448 (divisible by patch_size=14)
            ego_frame = cv2.resize(ego_frame, (448, 448), interpolation=cv2.INTER_LINEAR)
        except:
            ego_frame = np.zeros((448, 448, 3), dtype=np.uint8)
            start_idx = 0

        # Load actions
        data = self._load_actions(episode_id)

        # Get action chunk
        actions = []
        for key in ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]:
            if key in data:
                arr = data[key]
                if isinstance(arr[0], np.ndarray):
                    arr = np.stack(arr)
                chunk = arr[start_idx:start_idx + self.action_horizon]
                if len(chunk) < self.action_horizon:
                    chunk = np.pad(chunk, ((0, self.action_horizon - len(chunk)), (0, 0)))
                actions.append(chunk)

        if actions:
            actions = np.concatenate(actions, axis=-1).astype(np.float32)
        else:
            actions = np.zeros((self.action_horizon, 29), dtype=np.float32)

        # Action mask
        action_mask = np.ones_like(actions)

        # Get state
        state = np.zeros(29, dtype=np.float32)
        for i, key in enumerate(["left_arm", "right_arm", "left_hand", "right_hand", "waist"]):
            if key in data:
                arr = data[key]
                if isinstance(arr[start_idx], np.ndarray):
                    state_part = arr[start_idx]
                else:
                    state_part = np.array([arr[start_idx]])
                # Simplified: just use first few dims
                end_idx = min(i * 6 + len(state_part), 29)
                state[i * 6:end_idx] = state_part[:end_idx - i * 6]

        # Demo type (assume robot demos for now, can be randomized)
        demo_type = episode.get("demo_type", "robot")
        demo_type_map = {"human": 0, "robot": 1, "cosmos": 2, "own": 3}
        demo_type_idx = demo_type_map.get(demo_type, 1)

        return {
            "demo_frames": demo_frames,  # [K, H, W, C]
            "demo_type": demo_type_idx,
            "ego_frame": ego_frame,  # [H, W, C]
            "state": state,  # [state_dim]
            "action": actions,  # [horizon, action_dim]
            "action_mask": action_mask,  # [horizon, action_dim]
            "embodiment_id": 0,  # G1_DC
        }


class DCDataCollator:
    """Collate DC-GR00T training samples."""

    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F

        # Resize demo frames to 224x224 for SigLIP encoder
        demo_frames_list = []
        for b in batch:
            frames = b["demo_frames"]  # [K, H, W, C]
            # Convert to tensor and permute to [K, C, H, W]
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            # Resize to 224x224
            frames_resized = F.interpolate(frames_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            demo_frames_list.append(frames_resized)

        demo_frames = torch.stack(demo_frames_list)  # [B, K, C, H, W]
        demo_type = torch.tensor([b["demo_type"] for b in batch])
        ego_frames = torch.from_numpy(np.stack([b["ego_frame"] for b in batch]))
        states = torch.from_numpy(np.stack([b["state"] for b in batch])).unsqueeze(1)  # [B, 1, state_dim] - add time dimension
        actions = torch.from_numpy(np.stack([b["action"] for b in batch]))
        action_masks = torch.from_numpy(np.stack([b["action_mask"] for b in batch]))
        embodiment_ids = torch.tensor([b["embodiment_id"] for b in batch])

        batch_size = len(batch)

        # Process VLM inputs using the model's collator
        # The key insight: We need to provide the VLM inputs that will be passed directly
        # to the backbone, NOT vlm_content (which prepare_input expects to process)
        if self.processor is not None:
            from PIL import Image
            vlm_content_list = []
            for i in range(batch_size):
                # Convert ego frame to PIL Image
                ego_frame_np = ego_frames[i].numpy().astype(np.uint8)
                ego_image = Image.fromarray(ego_frame_np)

                # Create vlm_content dict
                # For Eagle, the conversation content must be a LIST containing image elements
                # AND the text must use numbered placeholders like <image-1>, <image-2>, etc.
                vlm_content = {
                    "text": "<image-1>",  # Eagle expects numbered placeholders: <image-1>, <image-2>, etc.
                    "images": [ego_image],  # List of PIL images (for non-Eagle models)
                    "conversation": [  # Conversation format for Eagle
                        {
                            "role": "user",
                            "content": [  # Content must be a LIST for extract_vision_info
                                {
                                    "type": "image",
                                    "image": ego_image,  # PIL Image object
                                }
                            ],
                        }
                    ],
                }
                vlm_content_list.append(vlm_content)

            # Process vlm_content through the collator to get pixel_values, input_ids, etc.
            try:
                import sys

                # Debug: Check what we're passing to the processor
                print(f"DEBUG: Passing {len(vlm_content_list)} vlm_content items to processor", file=sys.stderr)
                print(f"DEBUG: First vlm_content keys: {vlm_content_list[0].keys()}", file=sys.stderr)
                print(f"DEBUG: First vlm_content has {len(vlm_content_list[0]['images'])} images", file=sys.stderr)

                processed = self.processor([{"vlm_content": vlm} for vlm in vlm_content_list])

                # Debug: Check what the processor returned
                print(f"DEBUG: Processor returned type: {type(processed)}", file=sys.stderr)
                print(f"DEBUG: Processor returned: {processed}", file=sys.stderr)

                # The processor returns BatchFeature(data={"inputs": {...}})
                # We need to extract the actual inputs dict
                if hasattr(processed, 'data') and "inputs" in processed.data:
                    vlm_inputs = processed.data["inputs"]
                    print(f"DEBUG: Extracted from processed.data['inputs']", file=sys.stderr)
                elif isinstance(processed, dict) and "inputs" in processed:
                    vlm_inputs = processed["inputs"]
                    print(f"DEBUG: Extracted from processed['inputs']", file=sys.stderr)
                else:
                    vlm_inputs = processed
                    print(f"DEBUG: Using processed directly", file=sys.stderr)

                # Debug output
                print(f"DEBUG: VLM inputs type: {type(vlm_inputs)}", file=sys.stderr)
                print(f"DEBUG: VLM inputs keys: {vlm_inputs.keys() if hasattr(vlm_inputs, 'keys') else 'N/A'}", file=sys.stderr)

                if hasattr(vlm_inputs, 'keys'):
                    print(f"DEBUG: pixel_values present: {'pixel_values' in vlm_inputs}", file=sys.stderr)
                    print(f"DEBUG: input_ids present: {'input_ids' in vlm_inputs}", file=sys.stderr)
                    print(f"DEBUG: attention_mask present: {'attention_mask' in vlm_inputs}", file=sys.stderr)

                    # If pixel_values is missing, check if it's in the original processed object
                    if 'pixel_values' not in vlm_inputs and hasattr(processed, 'keys'):
                        print(f"DEBUG: Checking processed keys: {processed.keys()}", file=sys.stderr)

                sys.stderr.flush()
            except Exception as e:
                print(f"ERROR processing VLM content: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                vlm_inputs = {}
        else:
            print("WARNING: No processor available in DCDataCollator", file=sys.stderr)
            vlm_inputs = {}

        # Construct the output batch
        # Important: Unpack vlm_inputs at the top level so pixel_values, input_ids, etc.
        # are directly accessible (not nested under "inputs")
        output_batch = {
            "demo_frames": demo_frames,
            "demo_type": demo_type,
            "state": states,
            "action": actions,
            "action_mask": action_masks,
            "embodiment_id": embodiment_ids,
        }

        # Add VLM inputs directly to the batch
        output_batch.update(vlm_inputs)

        return output_batch


class DCTrainer(Trainer):
    """Custom trainer for DC-GR00T."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract demo frames
        demo_frames = inputs.pop("demo_frames")
        demo_type = inputs.pop("demo_type")

        # Encode demo to task embedding
        task_embedding = model.encode_demo(demo_frames, demo_type)

        # Forward pass
        outputs = model.forward(inputs, task_embedding=task_embedding)
        loss = outputs["loss"]

        if return_outputs:
            return loss, outputs
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train DC-GR00T")

    # Data
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--val_dataset_path", type=str, default=None)

    # Model
    parser.add_argument("--pretrained_groot", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--from_scratch", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, default="./dc_checkpoints")
    parser.add_argument("--run_name", type=str, default="dc_groot_g1")

    # Training
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # DC-specific
    parser.add_argument("--num_demo_keyframes", type=int, default=16)
    parser.add_argument("--num_task_tokens", type=int, default=16)
    parser.add_argument("--alignment_loss_weight", type=float, default=0.1)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create config
    config = DCGr00tConfig(
        num_task_tokens=args.num_task_tokens,
        alignment_loss_weight=args.alignment_loss_weight,
    )

    # Create model
    print("=" * 60)
    print("Creating DC-GR00T model...")
    print("=" * 60)

    if args.from_scratch:
        model = DCGr00t(config)
    else:
        model = DCGr00t.from_pretrained_groot(args.pretrained_groot, config=config)

    # Apply LoRA for RTX 5090 memory optimization (REQUIRED for 32GB VRAM)
    try:
        from peft import LoraConfig, get_peft_model
        print("\n" + "=" * 60)
        print("Applying LoRA for RTX 5090 Memory Optimization")
        print("=" * 60)
        lora_config = LoraConfig(
            r=8,  # Rank (lower = more memory savings)
            lora_alpha=16,  # Scaling factor
            # Use regex to target only language model layers, excluding vision encoder
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print("\n✓ LoRA Applied Successfully!")
        model.print_trainable_parameters()
        print("=" * 60 + "\n")
    except ImportError:
        print("\n" + "!" * 60)
        print("WARNING: PEFT not installed!")
        print("Install with: pip install peft")
        print("Training will likely OOM on RTX 5090 without LoRA")
        print("!" * 60 + "\n")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Create datasets
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    train_dataset = DCDataset(
        args.dataset_path,
        num_demo_keyframes=args.num_demo_keyframes,
    )

    val_dataset = None
    if args.val_dataset_path:
        val_dataset = DCDataset(
            args.val_dataset_path,
            num_demo_keyframes=args.num_demo_keyframes,
        )

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    # Create proper VLM collator
    from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6DataCollator
    vlm_collator = Gr00tN1d6DataCollator(
        model_name=config.model_name,
        model_type=config.backbone_model_type,
        transformers_loading_kwargs={"trust_remote_code": True},
    )
    collator = DCDataCollator(processor=vlm_collator)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_total_limit=5,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else [],
    )

    # Check for checkpoint
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint is None:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint:
            print(f"Found checkpoint: {last_checkpoint}")
            resume_checkpoint = last_checkpoint

    # Create trainer
    trainer = DCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print("=" * 60)
    print("Saving final model...")
    print("=" * 60)

    trainer.save_model(os.path.join(args.output_dir, "final"))
    print("Training complete!")


if __name__ == "__main__":
    main()
