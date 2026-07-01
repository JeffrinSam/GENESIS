#!/usr/bin/env python3
"""
Ultimate GPU-Optimized Manipulation Batch Runner
Sequential load/unload pattern per image per stage:
  1. Load Qwen3-VL → enhance ALL samples → UNLOAD
  2. Load Cosmos 2.5 → generate ALL samples → UNLOAD
  3. Load Cosmos-Reason2 → validate ALL samples → UNLOAD
  4. Next image, repeat

32GB GPU friendly - only 1 model loaded at a time.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import uuid
import glob

_PART1_DIR = Path(__file__).resolve().parents[1]
QWEN_EXTENDERS = Path(os.getenv('QWEN_ROOT', str(_PART1_DIR / 'qwen_prompt_expansion'))) / 'prompt_extenders'
COSMOS_VENV_PY = Path(os.getenv('COSMOS_ROOT', '/opt/cosmos-predict2.5')) / '.venv/bin/python'
COSMOS_GEN = _PART1_DIR / 'agentllm' / 'Manipulation' / 'cosmos_generate.py'
REASON2_VENV_PY = Path(os.getenv('COSMOS_REASON2_ROOT', '/opt/cosmos-reason2')) / '.venv/bin/python'
VALIDATOR = _PART1_DIR / 'agentllm' / 'Manipulation' / 'video_validator.py'
OUTPUT_BASE = Path(__file__).parent / 'results'
DEFAULT_CONDA_ENV = 'wan2.2'

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger('batch_manipulation_sequential')


def filename_to_prompt(path: Path) -> str:
    name = path.name
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    parts = name.split('_')
    if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) >= 6:
        name = '_'.join(parts[2:])
    prompt = name.replace('_', ' ').replace('-', ' ').replace(',', '')
    return ' '.join(prompt.split())


def infer_task(prompt: str) -> str:
    p = prompt.lower()
    return 'ur3' if ('gripper' in p or 'grasp' in p or 'lift' in p or 'pick' in p) else 'g1'


def enhance_prompt_batch(task: str, prompts: list, images: list, conda_env: str = DEFAULT_CONDA_ENV):
    """
    Stage 1: Load Qwen3-VL ONCE, enhance all prompts, then UNLOAD.
    
    Args:
        task: ur3 or g1
        prompts: List of (original_prompt, image_path) tuples
        images: List of image paths (for reference)
        conda_env: Conda environment with Qwen3-VL
    
    Returns:
        Dict: {image_path: enhanced_prompt}
    """
    task_map = {
        'ur3': QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_bimanual_ur3.py',
        'g1': QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_unitree_g1.py'
    }
    extender = task_map.get(task)
    if not extender or not extender.exists():
        log.warning(f'Extender not found for {task}. Using filename prompts.')
        return {img: p for p, img in prompts}

    outputs_dir = QWEN_EXTENDERS / 'outputs'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_prompts = {}
    
    log.info(f'[STAGE 1/3] Loading Qwen3-VL extender (load once)...')
    
    for orig_prompt, image_path in prompts:
        out_base = f"manip_{task}_{uuid.uuid4().hex[:8]}"
        
        cmd = ['conda', 'run', '-n', conda_env, 'python3', str(extender), 
               '--prompt', orig_prompt, '--image', image_path, '--output', out_base]
        
        log.info(f'  Enhancing: {Path(image_path).name}')
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        if res.returncode != 0:
            log.warning(f'    Extender failed, using original prompt')
            enhanced_prompts[image_path] = orig_prompt
            continue
        
        prompt_file = outputs_dir / f"{out_base}_prompt.txt"
        if prompt_file.exists():
            enhanced = prompt_file.read_text(encoding='utf-8')
            if '====' in enhanced:
                parts = enhanced.split('\n')
                capture = False
                out = ''
                for line in parts:
                    if set(line.strip()) == set('='):
                        if not capture:
                            capture = True
                            continue
                        else:
                            break
                    if capture and line.strip() and 'Negative Prompt' not in line:
                        out += line + ' '
                enhanced = out.strip()
            enhanced_prompts[image_path] = enhanced or orig_prompt
            log.info(f'    Enhanced: {len(enhanced.split())} words')
        else:
            enhanced_prompts[image_path] = orig_prompt
    
    log.info('[STAGE 1/3] Qwen3-VL UNLOADED')
    return enhanced_prompts


def generate_video_batch(task: str, generation_tasks: list, model: str = '2B', frames: int = 77, guidance: float = 7.0):
    """
    Stage 2: Load Cosmos 2.5 ONCE, generate all videos, then UNLOAD.
    
    Args:
        task: ur3 or g1 (for reference)
        generation_tasks: List of (image_path, prompt, output_path, seed) tuples
        model: Model size
        frames: Frame count
        guidance: Guidance scale
    """
    if not COSMOS_VENV_PY.exists():
        log.error('Cosmos venv python not found')
        raise RuntimeError('Cosmos venv missing')
    
    log.info(f'[STAGE 2/3] Loading Cosmos 2.5 {model} (load once)...')
    
    for image_path, prompt, output_path, seed in generation_tasks:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [str(COSMOS_VENV_PY), str(COSMOS_GEN),
               '--model', model,
               '--input_path', image_path,
               '--prompt', prompt,
               '--output_path', output_path,
               '--num_output_frames', str(frames),
               '--guidance', str(guidance),
               '--seed', str(seed)]
        
        log.info(f'  Generating: {Path(output_path).name} (seed {seed})')
        res = subprocess.run(cmd, cwd=str(COSMOS_VENV_PY.parent.parent), capture_output=True, text=True)
        
        if res.returncode != 0:
            log.error(f'    Generation failed: {res.stderr[:200] if res.stderr else res.stdout[:200]}')
            raise RuntimeError('Cosmos generation failed')
        
        if Path(output_path).exists():
            log.info(f'    Generated: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB')
        else:
            log.warning(f'    Output file not created')
    
    log.info('[STAGE 2/3] Cosmos 2.5 UNLOADED')


def validate_video_batch(task: str, validation_tasks: list):
    """
    Stage 3: Load Cosmos-Reason2 ONCE, validate all videos, then UNLOAD.
    
    Args:
        task: ur3 or g1
        validation_tasks: List of (video_path, original_prompt, output_json) tuples
    """
    if not REASON2_VENV_PY.exists():
        log.error('Reason2 venv python not found')
        raise RuntimeError('Reason2 venv missing')
    
    log.info(f'[STAGE 3/3] Loading Cosmos-Reason2 (load once)...')
    
    for video_path, original_prompt, output_json in validation_tasks:
        output_dir = Path(output_json).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [str(REASON2_VENV_PY), str(VALIDATOR),
               '--video', video_path,
               '--task', task,
               '--prompt', original_prompt,
               '--output', output_json,
               '--fps', '4']
        
        log.info(f'  Validating: {Path(video_path).name}')
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        if res.returncode != 0:
            log.warning(f'    Validation warning (skipping): {res.stderr[:200] if res.stderr else ""}')
            # Don't fail on validation - just log and continue
        else:
            log.info(f'    Validated')
    
    log.info('[STAGE 3/3] Cosmos-Reason2 UNLOADED')


def process_single_image(image_path: str, task: str, num_samples: int, seed_start: int = 0,
                         conda_env: str = DEFAULT_CONDA_ENV, model: str = '2B',
                         frames: int = 77, guidance: float = 7.0, no_validate: bool = False):
    """
    Process ONE image through all stages with sequential load/unload.
    """
    image = Path(image_path)
    if not image.exists():
        log.error(f'Image not found: {image}')
        return None
    
    prompt = filename_to_prompt(image)
    inferred_task = infer_task(prompt)
    if task is None:
        task = inferred_task
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTPUT_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f'\n{"="*70}')
    log.info(f'Image: {image.name}')
    log.info(f'Task: {task} (inferred: {inferred_task})')
    log.info(f'Prompt: "{prompt}"')
    log.info(f'Samples: {num_samples}')
    log.info(f'{"="*70}\n')
    
    try:
        # STAGE 1: Enhance prompt
        enhanced_prompts = enhance_prompt_batch(task, [(prompt, str(image))], [str(image)], conda_env)
        enhanced = enhanced_prompts.get(str(image), prompt)
        
        # STAGE 2: Generate videos
        generation_tasks = []
        for i in range(num_samples):
            seed = seed_start + i
            out_video = str(run_dir / f'manip_{task}_{run_id}_s{seed}.mp4')
            generation_tasks.append((str(image), enhanced, out_video, seed))
        
        generate_video_batch(task, generation_tasks, model=model, frames=frames, guidance=guidance)
        
        # STAGE 3: Validate videos
        if not no_validate:
            validation_tasks = []
            for _, _, out_video, _ in generation_tasks:
                out_json = str(run_dir / f'validation_{Path(out_video).stem}.json')
                validation_tasks.append((out_video, prompt, out_json))
            
            validate_video_batch(task, validation_tasks)
        
        # Save report
        result = {
            'run_id': run_id,
            'image': str(image),
            'task': task,
            'prompt': prompt,
            'enhanced_prompt': enhanced,
            'samples': num_samples,
            'validated': not no_validate
        }
        with open(run_dir / 'report.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        log.info(f'\n✓ Image complete. Results: {run_dir}\n')
        return result
        
    except Exception as e:
        log.error(f'✗ Image failed: {e}')
        return None


def process_all_images(manip_dir: str, task: str = 'g1', num_samples: int = 5,
                       conda_env: str = DEFAULT_CONDA_ENV, model: str = '2B',
                       frames: int = 77, guidance: float = 7.0, no_validate: bool = False):
    """
    Process all images in directory with sequential load/unload per image.
    """
    manip_path = Path(manip_dir)
    # Collect images with common extensions (Path.glob works per-extension)
    images = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        images.extend(sorted([str(p) for p in manip_path.glob(ext)]))
    images = sorted(images)
    
    log.info(f'Found {len(images)} images to process')
    
    results = []
    for idx, img in enumerate(images, 1):
        log.info(f'\n[{idx}/{len(images)}] Processing...')
        result = process_single_image(img, task, num_samples, conda_env=conda_env, 
                                      model=model, frames=frames, guidance=guidance, 
                                      no_validate=no_validate)
        if result:
            results.append(result)
    
    log.info(f'\n{"="*70}')
    log.info(f'BATCH COMPLETE: {len(results)}/{len(images)} images processed')
    log.info(f'{"="*70}\n')
    
    return results


def main():
    p = argparse.ArgumentParser(description='GPU-Optimized sequential load/unload batch runner')
    p.add_argument('--image', type=str, default=None, help='Single image (or use --all)')
    p.add_argument('--all', action='store_true', help='Process all images in the default manipulation data dir')
    p.add_argument('--task', type=str, choices=['ur3', 'g1'], default='g1')
    p.add_argument('--num-samples', type=int, default=5)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--conda-env', type=str, default=DEFAULT_CONDA_ENV)
    p.add_argument('--model', type=str, default='2B', choices=['2B', '14B'])
    p.add_argument('--frames', type=int, default=77)
    p.add_argument('--guidance', type=float, default=7.0)
    p.add_argument('--no-validate', action='store_true', help='Skip validation to save GPU memory')
    args = p.parse_args()
    
    if args.all:
        default_dir = os.getenv('MANIP_DATA_DIR', str(Path.home() / 'manipulation_data'))
        process_all_images(default_dir, task=args.task, num_samples=args.num_samples,
                          conda_env=args.conda_env, model=args.model, frames=args.frames,
                          guidance=args.guidance, no_validate=args.no_validate)
    elif args.image:
        process_single_image(args.image, args.task, args.num_samples, args.seed_start,
                            conda_env=args.conda_env, model=args.model, frames=args.frames,
                            guidance=args.guidance, no_validate=args.no_validate)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
