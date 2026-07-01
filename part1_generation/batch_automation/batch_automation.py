#!/usr/bin/env python3
"""
Batch Video Generation Automation System - V2
Uses IMAGE FILENAMES as PROMPTS directly
Supports both Navigation and Manipulation tasks

Works with AgentLLM pipelines

Setup:
1. Save images with descriptive names (filename = prompt)
2. Navigation images go in: part1_generation/agentllm/Navigation/uploads/
3. Manipulation images go in: part1_generation/agentllm/Manipulation/uploads/
4. Run: python3 batch_automation.py

Example image names:
- Navigation: "Smooth_drone_flight_over_mountains.jpg"
- Manipulation: "Robot_picks_up_bottle_and_places_it_down.jpg"

Author: Automation System
Date: 2026-02-10
"""

import argparse
import json
import logging
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('batch_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base paths — resolved relative to this file's location in part1_generation/batch_automation/
_PART1_DIR = Path(__file__).resolve().parents[1]
NAVIGATION_UPLOADS = _PART1_DIR / 'agentllm' / 'Navigation' / 'uploads'
MANIPULATION_UPLOADS = _PART1_DIR / 'agentllm' / 'Manipulation' / 'uploads'
NAVIGATION_PIPELINE = _PART1_DIR / 'agentllm' / 'Navigation' / 'navigation_pipeline.py'
MANIPULATION_PIPELINE = _PART1_DIR / 'agentllm' / 'Manipulation' / 'manipulation_pipeline.py'
OUTPUT_DIR = Path(__file__).parent / 'results'
BATCH_LOG = Path(__file__).parent / 'batch_automation.log'


class BatchTask:
    """Represents a single batch task"""
    def __init__(self, task_id: str, task_type: str, image_path: str, 
                 prompt: str, category: str):
        self.task_id = task_id
        self.task_type = task_type  # drone, ground, ur3, g1
        self.image_path = image_path
        self.prompt = prompt  # Extracted from filename
        self.category = category  # navigation or manipulation
        self.status = 'pending'  # pending, completed, failed
        self.start_time = None
        self.end_time = None
        self.duration_seconds = None
        self.video_output = None
        self.error = None
        self.log_output = None

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'image_name': Path(self.image_path).name,
            'prompt': self.prompt,
            'category': self.category,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'video_output': self.video_output,
            'error': self.error,
        }


class BatchAutomation:
    """Main batch automation system - uses filenames as prompts"""
    
    def __init__(self):
        self.tasks: List[BatchTask] = []
        self.results = {}
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = OUTPUT_DIR / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
    def filename_to_prompt(self, filename: str) -> str:
        """Convert filename to prompt by removing timestamp and extension
        
        Examples:
        - "20251229_130841_Worlds-longest-drone-fpv-one-shot.jpeg"
          → "Worlds longest drone fpv one shot"
        - "Align_the_right_gripper_with_the_plant.png"
          → "Align the right gripper with the plant"
        """
        # Remove file extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Remove timestamp prefix if present (YYYYMMDD_HHMMSS_ format)
        if len(name_without_ext) > 15 and name_without_ext[:8].isdigit():
            # Has timestamp prefix - remove it
            parts = name_without_ext.split('_', 2)  # Split on first 2 underscores
            if len(parts) >= 3:
                name_without_ext = parts[2]
            else:
                name_without_ext = '_'.join(parts)
        
        # Replace underscores/hyphens with spaces
        prompt = name_without_ext.replace('_', ' ').replace('-', ' ')
        
        # Clean up multiple spaces
        prompt = ' '.join(prompt.split())
        
        return prompt
    
    def discover_images(self) -> Tuple[int, int]:
        """Discover all images in navigation and manipulation folders
        
        Returns: (navigation_count, manipulation_count)
        """
        logger.info("=" * 80)
        logger.info("DISCOVERING IMAGES FROM FILENAMES")
        logger.info("=" * 80)
        
        nav_count = 0
        manip_count = 0
        
        # Discover navigation images (Auto-detect Drone vs Ground)
        nav_dir = NAVIGATION_UPLOADS
        if nav_dir.exists():
            nav_images = sorted(
                list(nav_dir.glob('*.jpg')) + 
                list(nav_dir.glob('*.jpeg')) + 
                list(nav_dir.glob('*.png'))
            )
            
            for idx, img_path in enumerate(nav_images):
                # Extract prompt from filename
                prompt = self.filename_to_prompt(img_path.name)
                
                # Auto-detect task type from prompt
                prompt_lower = prompt.lower()
                if 'ground' in prompt_lower or 'robot walk' in prompt_lower or 'corridor' in prompt_lower:
                    task_type = 'ground'
                else:
                    task_type = 'drone'  # Default to drone for most prompts
                
                task_id = f'nav_{task_type}_{idx:03d}'
                task = BatchTask(task_id, task_type, str(img_path), prompt, 'navigation')
                self.tasks.append(task)
                nav_count += 1
                
                logger.info(f"  ✓ {task_id}")
                logger.info(f"    Image: {img_path.name}")
                logger.info(f"    Prompt: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
        
        # Discover manipulation images (Auto-detect UR3 vs G1)
        manip_dir = MANIPULATION_UPLOADS
        if manip_dir.exists():
            manip_images = sorted(
                list(manip_dir.glob('*.jpg')) + 
                list(manip_dir.glob('*.jpeg')) + 
                list(manip_dir.glob('*.png'))
            )
            
            for idx, img_path in enumerate(manip_images):
                # Extract prompt from filename
                prompt = self.filename_to_prompt(img_path.name)
                
                # Auto-detect task type from prompt
                prompt_lower = prompt.lower()
                if 'ur3' in prompt_lower or 'bimanual' in prompt_lower or 'dual' in prompt_lower:
                    task_type = 'ur3'
                elif 'g1' in prompt_lower or 'humanoid' in prompt_lower or 'walk' in prompt_lower:
                    task_type = 'g1'
                else:
                    # Alternate if unclear
                    task_type = 'ur3' if idx % 2 == 0 else 'g1'
                
                task_id = f'manip_{task_type}_{idx:03d}'
                task = BatchTask(task_id, task_type, str(img_path), prompt, 'manipulation')
                self.tasks.append(task)
                manip_count += 1
                
                logger.info(f"  ✓ {task_id}")
                logger.info(f"    Image: {img_path.name}")
                logger.info(f"    Prompt: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
        
        logger.info("")
        logger.info(f"Total discovered: {len(self.tasks)} tasks")
        logger.info(f"  Navigation (WAN 2.2): {nav_count}")
        logger.info(f"  Manipulation (Cosmos 2.5): {manip_count}")
        
        return nav_count, manip_count
    
    def process_batch(self, skip_errors: bool = True, max_tasks: Optional[int] = None) -> None:
        """Process all tasks in batch"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("STARTING BATCH PROCESSING")
        logger.info("=" * 80)
        
        tasks_to_process = self.tasks[:max_tasks] if max_tasks else self.tasks
        total = len(tasks_to_process)
        
        if total == 0:
            logger.warning("No tasks to process!")
            return
        
        completed = 0
        failed = 0
        
        for idx, task in enumerate(tasks_to_process, 1):
            logger.info("")
            logger.info(f"[{idx}/{total}] TASK: {task.task_id}")
            logger.info("-" * 80)
            logger.info(f"Image: {Path(task.image_path).name}")
            logger.info(f"Prompt: {task.prompt}")
            logger.info(f"Type: {task.task_type.upper()}")
            logger.info(f"Category: {task.category.upper()}")
            logger.info("")
            
            start = datetime.now()
            task.start_time = start.isoformat()
            
            try:
                self.process_task(task)
                task.status = 'completed'
                completed += 1
                logger.info(f"✅ COMPLETED: {task.task_id}")
            except Exception as e:
                logger.error(f"❌ FAILED: {task.task_id}")
                logger.error(f"Error: {str(e)}")
                task.status = 'failed'
                task.error = str(e)
                failed += 1
                
                if not skip_errors:
                    raise
            
            end = datetime.now()
            task.end_time = end.isoformat()
            task.duration_seconds = (end - start).total_seconds()
            
            # Save progress after each task
            self.save_progress()
            
            # Show progress
            logger.info(f"Duration: {task.duration_seconds:.1f}s")
    
    def process_task(self, task: BatchTask) -> None:
        """Process a single task using appropriate pipeline"""
        
        if task.category == 'navigation':
            self._process_navigation_task(task)
        else:
            self._process_manipulation_task(task)
    
    def _process_navigation_task(self, task: BatchTask) -> None:
        """Run navigation pipeline (WAN 2.2)"""
        output_video = self.run_dir / f'{task.task_id}.mp4'
        
        cmd = [
            'python3', str(NAVIGATION_PIPELINE),
            '--task_type', task.task_type,
            '--image', task.image_path,
            '--prompt', task.prompt,
            '--output', str(output_video),
        ]
        
        logger.info(f"Running WAN 2.2 pipeline...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"Pipeline stdout: {result.stdout[-500:]}")
            logger.error(f"Pipeline stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"Navigation pipeline failed (exit code {result.returncode})")
        
        if output_video.exists():
            task.video_output = str(output_video)
            logger.info(f"✓ Video saved: {output_video.name}")
            logger.info(f"  Size: {output_video.stat().st_size / (1024*1024):.1f} MB")
        else:
            raise RuntimeError(f"Output video not created: {output_video}")
    
    def _process_manipulation_task(self, task: BatchTask) -> None:
        """Run manipulation pipeline (Cosmos 2.5)"""
        output_video = self.run_dir / f'{task.task_id}.mp4'
        
        cmd = [
            'python3', str(MANIPULATION_PIPELINE),
            '--task_type', task.task_type,
            '--image', task.image_path,
            '--prompt', task.prompt,
            '--output', str(output_video),
            '--model', '2B',  # Use 2B for speed, can change to 14B
        ]
        
        logger.info(f"Running Cosmos 2.5 pipeline...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        
        if result.returncode != 0:
            logger.error(f"Pipeline stdout: {result.stdout[-500:]}")
            logger.error(f"Pipeline stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"Manipulation pipeline failed (exit code {result.returncode})")
        
        if output_video.exists():
            task.video_output = str(output_video)
            logger.info(f"✓ Video saved: {output_video.name}")
            logger.info(f"  Size: {output_video.stat().st_size / (1024*1024):.1f} MB")
        else:
            raise RuntimeError(f"Output video not created: {output_video}")
    
    def save_progress(self) -> None:
        """Save batch progress to JSON"""
        progress_file = self.run_dir / 'progress.json'
        progress_data = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'total_tasks': len(self.tasks),
            'completed': sum(1 for t in self.tasks if t.status == 'completed'),
            'failed': sum(1 for t in self.tasks if t.status == 'failed'),
            'pending': sum(1 for t in self.tasks if t.status == 'pending'),
            'tasks': [t.to_dict() for t in self.tasks],
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def generate_summary(self) -> None:
        """Generate batch summary report"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("BATCH SUMMARY REPORT")
        logger.info("=" * 80)
        logger.info("")
        
        completed = sum(1 for t in self.tasks if t.status == 'completed')
        failed = sum(1 for t in self.tasks if t.status == 'failed')
        pending = sum(1 for t in self.tasks if t.status == 'pending')
        total = len(self.tasks)
        
        if total > 0:
            success_rate = 100 * completed / total
        else:
            success_rate = 0
        
        logger.info(f"Total tasks: {total}")
        logger.info(f"✅ Completed: {completed} ({success_rate:.1f}%)")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⏳ Pending: {pending}")
        
        # Calculate total time
        total_time = sum(t.duration_seconds or 0 for t in self.tasks if t.status == 'completed')
        if completed > 0:
            avg_time = total_time / completed
            logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
            logger.info(f"⏱️  Avg per video: {avg_time:.1f} seconds")
        
        logger.info(f"📁 Results: {self.run_dir}")
        
        if failed > 0:
            logger.info("")
            logger.info("Failed tasks:")
            for task in self.tasks:
                if task.status == 'failed':
                    logger.info(f"  ❌ {task.task_id}: {task.error[:80]}")
        
        # Save summary to file
        summary_file = self.run_dir / 'SUMMARY.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BATCH VIDEO GENERATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(f"Total tasks: {total}\n")
            f.write(f"✅ Completed: {completed} ({success_rate:.1f}%)\n")
            f.write(f"❌ Failed: {failed}\n")
            f.write(f"⏳ Pending: {pending}\n\n")
            f.write(f"⏱️  Total time: {total_time/60:.1f} minutes\n")
            f.write(f"📁 Results directory: {self.run_dir}\n\n")
            
            if failed > 0:
                f.write("FAILED TASKS:\n")
                f.write("-" * 80 + "\n")
                for task in self.tasks:
                    if task.status == 'failed':
                        f.write(f"{task.task_id}: {task.error}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("TASK DETAILS:\n")
            f.write("=" * 80 + "\n")
            for task in self.tasks:
                f.write(f"\n{task.task_id}\n")
                f.write(f"  Status: {task.status}\n")
                f.write(f"  Image: {Path(task.image_path).name}\n")
                f.write(f"  Prompt: {task.prompt}\n")
                if task.duration_seconds:
                    f.write(f"  Duration: {task.duration_seconds:.1f}s\n")
                if task.video_output:
                    f.write(f"  Output: {Path(task.video_output).name}\n")
                if task.error:
                    f.write(f"  Error: {task.error}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch Video Generation - Uses Image Filenames as Prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover and preview all tasks
  python3 batch_automation.py --discover-only

  # Process first 5 tasks
  python3 batch_automation.py --max-tasks=5

  # Process all tasks, skip errors
  python3 batch_automation.py --skip-errors

  # Process all tasks, stop on first error
  python3 batch_automation.py
        """
    )
    parser.add_argument('--skip-errors', action='store_true', 
                       help='Skip failed tasks and continue with next')
    parser.add_argument('--max-tasks', type=int, default=None,
                       help='Maximum number of tasks to process')
    parser.add_argument('--discover-only', action='store_true',
                       help='Only discover and list tasks, do not process')
    
    args = parser.parse_args()
    
    # Create automation system
    automation = BatchAutomation()
    
    # Discover images
    nav_count, manip_count = automation.discover_images()
    
    if args.discover_only:
        logger.info("")
        logger.info("Discovery complete! Use batch_automation.py to process.")
        return
    
    if nav_count + manip_count == 0:
        logger.warning("No images found in Navigation/uploads or Manipulation/uploads")
        logger.warning("Please check directory paths and add images")
        return
    
    # Process batch
    try:
        automation.process_batch(
            skip_errors=args.skip_errors,
            max_tasks=args.max_tasks
        )
    finally:
        # Always generate summary
        automation.generate_summary()


if __name__ == '__main__':
    main()
