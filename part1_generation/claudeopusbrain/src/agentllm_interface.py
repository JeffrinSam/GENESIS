# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
AgentLLM Interface: Subprocess calls to existing working pipeline

This module provides clean interfaces to call the existing AgentLLM components
without modifying them. All integration happens through subprocess calls.

Author: Jeffrin Sam (jeffrinsam.a@gmail.com)
Date: 2026-02-07
"""

import subprocess
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import re

_HERE = Path(__file__).resolve().parent  # src/
_PART1 = _HERE.parent.parent  # part1_generation/
_GENESIS = _PART1.parent  # GENESIS/

_CONDA_BIN = shutil.which("conda") or os.getenv("CONDA_EXE", "conda")
_QWEN_ROOT = Path(os.getenv("QWEN_ROOT", str(_GENESIS.parent / "Part1" / "Qwen3-VL")))
_COSMOS_ROOT = Path(os.getenv("COSMOS_ROOT", str(_GENESIS.parent / "Part1" / "cosmos-predict2.5")))
_COSMOS_REASON2_ROOT = Path(os.getenv("COSMOS_REASON2_ROOT", str(_GENESIS.parent / "Part1" / "cosmos-reason2")))
_WAN_ROOT = Path(os.getenv("WAN_ROOT", str(_GENESIS.parent / "Part1" / "Wan2.2")))
_QWEN_PYTHON = os.getenv("QWEN_PYTHON", str(_QWEN_ROOT / ".venv" / "bin" / "python"))
_COSMOS_PYTHON = os.getenv("COSMOS_PYTHON", str(_COSMOS_ROOT / ".venv" / "bin" / "python"))
_COSMOS_REASON2_PYTHON = os.getenv("COSMOS_REASON2_PYTHON", str(_COSMOS_REASON2_ROOT / ".venv" / "bin" / "python"))


class AgentLLMInterface:
    """
    Interface to existing AgentLLM pipeline components

    Calls existing scripts via subprocess without modifying them.
    """

    def __init__(
        self,
        agentllm_root: str = str(_PART1 / "agentllm"),
        qwen_root: str = str(_QWEN_ROOT)
    ):
        """
        Initialize interface with paths to existing components

        Args:
            agentllm_root: Path to AgentLLM directory
            qwen_root: Path to Qwen3-VL directory
        """
        self.agentllm_root = Path(agentllm_root)
        self.qwen_root = Path(qwen_root)

        # Task configurations (matches AgentLLM)
        self.task_configs = {
            'drone': {
                'category': 'navigation',
                'pipeline': self.agentllm_root / 'Navigation' / 'navigation_pipeline.py',
                'extender': self.qwen_root / 'prompt_extenders' / 'wan22' / 'prompt_extender_drone.py'
            },
            'ground': {
                'category': 'navigation',
                'pipeline': self.agentllm_root / 'Navigation' / 'navigation_pipeline.py',
                'extender': self.qwen_root / 'prompt_extenders' / 'wan22' / 'prompt_extender_ground_robot.py'
            },
            'ur3': {
                'category': 'manipulation',
                'pipeline': self.agentllm_root / 'Manipulation' / 'manipulation_pipeline.py',
                'extender': self.qwen_root / 'prompt_extenders' / 'cosmos25' / 'prompt_extender_bimanual_ur3.py'
            },
            'g1': {
                'category': 'manipulation',
                'pipeline': self.agentllm_root / 'Manipulation' / 'manipulation_pipeline.py',
                'extender': self.qwen_root / 'prompt_extenders' / 'cosmos25' / 'prompt_extender_unitree_g1.py'
            },
            'g1_nav': {
                'category': 'navigation',
                'pipeline': self.agentllm_root / 'Navigation' / 'navigation_pipeline.py',
                'extender': self.qwen_root / 'prompt_extenders' / 'wan22' / 'prompt_extender_unitree_g1_nav.py'
            }
        }

    # Paths to components (read-only, never modified)
    COSMOS_BASE = _COSMOS_ROOT
    QWEN_EXTENDERS = _QWEN_ROOT / 'prompt_extenders'

    def generate_video(
        self,
        task_type: str,
        user_prompt: str,
        image_path: str,
        custom_system_prompt: Optional[str] = None,
        custom_negative_prompt: Optional[str] = None,
        output_dir: Optional[Path] = None,
        iteration: int = 1
    ) -> Tuple[Optional[Path], Dict]:
        """
        Generate video using existing AgentLLM pipeline.

        When custom_system_prompt is provided for manipulation tasks, calls the
        Qwen3-VL extender and Cosmos directly so the optimizer's system_prompt
        actually reaches the extender. Otherwise uses the standard pipeline.

        Args:
            task_type: Task type (drone, ground, ur3, g1)
            user_prompt: Simple user prompt
            image_path: Path to workspace image
            custom_system_prompt: Custom system prompt for Qwen3-VL extender
            custom_negative_prompt: Negative prompt for video model (passed to WAN/Cosmos)
            output_dir: Optional output directory

        Returns:
            (video_path, metadata) tuple
        """
        if task_type not in self.task_configs:
            raise ValueError(f"Unknown task type: {task_type}")

        config = self.task_configs[task_type]

        # Determine output video path — use absolute path so subprocess CWD doesn't matter
        if output_dir:
            output_path = Path(output_dir).resolve() / f"{task_type}_output.mp4"
        else:
            output_path = config['pipeline'].parent / "outputs" / f"{task_type}_output.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # For tasks with custom system prompt: call components directly
        # so the optimizer's system_prompt actually reaches Qwen3-VL
        if custom_system_prompt:
            if config['category'] == 'manipulation':
                return self._generate_manipulation_direct(
                    task_type, user_prompt, image_path,
                    custom_system_prompt, output_path, iteration,
                    custom_negative_prompt=custom_negative_prompt
                )
            elif config['category'] == 'navigation':
                return self._generate_navigation_direct(
                    task_type, user_prompt, image_path,
                    custom_system_prompt, output_path, iteration,
                    custom_negative_prompt=custom_negative_prompt
                )

        # Standard path: call the pipeline script (no custom prompt injection)
        pipeline_script = config['pipeline']
        conda_bin = _CONDA_BIN
        cmd = [
            conda_bin, 'run', '-n', 'wan2.2', '--no-capture-output',
            'python3', str(pipeline_script),
            '--task', task_type,
            '--prompt', user_prompt,
            '--image', image_path,
            '--output', str(output_path),
            '--no-validation'  # Skip pipeline's internal validation (we run our own)
        ]

        print(f"   Calling pipeline: {pipeline_script.name}")
        print(f"   Task: {task_type}")
        print(f"   Prompt: {user_prompt[:50]}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=pipeline_script.parent
            )

            if result.returncode != 0:
                if output_path.exists():
                    print(f"⚠️  Pipeline exited {result.returncode} but video exists — continuing")
                else:
                    print(f"❌ Pipeline failed (no video generated):")
                    print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                    return None, {'error': result.stderr}

            if output_path.exists():
                video_path = output_path
            else:
                video_path = self._extract_video_path(result.stdout, config['category'])

            if not video_path:
                print(f"⚠️  Video file not found at {output_path}")
                return None, {'error': 'Video not generated'}

            return video_path, {'task_type': task_type, 'category': config['category'],
                                 'user_prompt': user_prompt, 'video_path': str(video_path)}

        except subprocess.TimeoutExpired:
            print(f"❌ Pipeline timeout (>10 minutes)")
            return None, {'error': 'Timeout'}
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return None, {'error': str(e)}

    def _generate_manipulation_direct(
        self,
        task_type: str,
        user_prompt: str,
        image_path: str,
        custom_system_prompt: str,
        output_path: Path,
        iteration: int = 1,
        custom_negative_prompt: Optional[str] = None
    ) -> Tuple[Optional[Path], Dict]:
        """
        Direct path: call Qwen3-VL extender then Cosmos separately.
        Used when optimizer provides a custom system prompt for the extender.
        Stays entirely within Claudeopusbrain — no AgentLLM modification needed.
        """
        config = self.task_configs[task_type]
        extender_script = config['extender']
        cosmos_script = config['pipeline'].parent / 'cosmos_generate.py'
        cosmos_python = Path(_COSMOS_PYTHON)
        conda_bin = _CONDA_BIN

        print(f"   [Direct] Extender: {extender_script.name} (custom system prompt)")

        # Step A: Run Qwen3.5 extender with custom system prompt
        import time as _time
        output_base = f"selftuning_{task_type}_{int(_time.time())}"
        qwen35_python = _QWEN_PYTHON
        extender_cmd = [
            qwen35_python, str(extender_script),
            '--prompt', user_prompt,
            '--image', str(Path(image_path).resolve()),
            '--system_prompt', custom_system_prompt,
            '--output', output_base
        ]

        try:
            ext_result = subprocess.run(
                extender_cmd, capture_output=True, text=True, timeout=300
            )
            if ext_result.returncode != 0:
                print(f"❌ Extender failed: {ext_result.stderr[-300:]}")
                return None, {'error': 'Extender failed'}

            # Read enhanced prompt from output file
            prompt_file = self.QWEN_EXTENDERS / 'outputs' / f"{output_base}_prompt.txt"
            if not prompt_file.exists():
                print(f"❌ Extender output not found: {prompt_file}")
                return None, {'error': 'Extender output missing'}

            with open(prompt_file, 'r') as f:
                content = f.read()

            # Extract enhanced prompt and negative prompt from extender output
            # Format:
            #   ============================================================
            #   [Enhanced prompt text here]
            #   ============================================================
            #   Negative Prompt:
            #   [Negative prompt text]

            enhanced_prompt = ""
            negative_prompt_from_extender = ""

            # Split by "Negative Prompt:" marker
            parts = content.split('Negative Prompt:', 1)

            if len(parts) == 2:
                # Extract enhanced prompt (between ======= and "Negative Prompt:")
                enhanced_section = parts[0]
                lines = enhanced_section.split('\n')
                capture = False
                for line in lines:
                    if '=' * 60 in line:
                        if not capture:
                            capture = True
                        else:
                            break  # Second ===, end of enhanced prompt
                    elif capture and line.strip():
                        enhanced_prompt += line + " "

                # Extract negative prompt (after "Negative Prompt:")
                negative_section = parts[1].strip()
                # Remove leading/trailing === if present
                for line in negative_section.split('\n'):
                    if '=' * 60 not in line and line.strip():
                        negative_prompt_from_extender += line.strip() + " "

                negative_prompt_from_extender = negative_prompt_from_extender.strip()

            enhanced_prompt = enhanced_prompt.strip()

            if not enhanced_prompt:
                print(f"❌ Could not parse enhanced prompt from extender output")
                return None, {'error': 'Empty enhanced prompt'}

            print(f"   [Direct] Enhanced: {len(enhanced_prompt.split())} words")
            if negative_prompt_from_extender:
                print(f"   [Direct] Negative (extender): {len(negative_prompt_from_extender.split())} terms")

        except subprocess.TimeoutExpired:
            print(f"❌ Extender timeout")
            return None, {'error': 'Extender timeout'}

        # Step B: Run Cosmos directly with enhanced prompt
        # Use iteration-dependent seed for exploration (each iteration tries different noise)
        seed = 42 + (iteration - 1) * 17  # 42, 59, 76, 93, 110...
        print(f"   [Direct] Cosmos generation (seed={seed})...")
        # Build negative prompt: prefer optimizer's custom prompt, fall back to extender's
        effective_negative = custom_negative_prompt or negative_prompt_from_extender or ""

        cosmos_cmd = [
            str(cosmos_python),
            str(cosmos_script),
            '--model', '2B',
            '--input_path', str(Path(image_path).resolve()),
            '--prompt', enhanced_prompt,
            '--output_path', str(output_path),
            '--num_output_frames', '61',
            '--guidance', '7',
            '--seed', str(seed),
        ]
        if effective_negative:
            cosmos_cmd.extend(['--negative_prompt', effective_negative])

        try:
            subprocess.run(cosmos_cmd, timeout=600, cwd=str(self.COSMOS_BASE), check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Cosmos failed: {e}")
            return None, {'error': 'Cosmos failed'}
        except subprocess.TimeoutExpired:
            print(f"❌ Cosmos timeout")
            return None, {'error': 'Cosmos timeout'}

        if not output_path.exists():
            print(f"❌ Video not generated at {output_path}")
            return None, {'error': 'Video not generated'}

        print(f"   [Direct] Video: {output_path}")
        return output_path, {
            'task_type': task_type,
            'category': 'manipulation',
            'user_prompt': user_prompt,
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt_from_extender': negative_prompt_from_extender,
            'custom_system_prompt': True,
            'video_path': str(output_path)
        }

    def _generate_navigation_direct(
        self,
        task_type: str,
        user_prompt: str,
        image_path: str,
        custom_system_prompt: str,
        output_path: Path,
        iteration: int = 1,
        custom_negative_prompt: Optional[str] = None
    ) -> Tuple[Optional[Path], Dict]:
        """
        Direct path for navigation: call WAN extender then WAN 2.2 separately.
        Used when optimizer provides a custom system prompt for the extender.
        Stays entirely within Claudeopusbrain — no AgentLLM modification needed.
        """
        config = self.task_configs[task_type]
        extender_script = config['extender']
        wan_base = _WAN_ROOT
        conda_bin = _CONDA_BIN

        print(f"   [Direct] Extender: {extender_script.name} (custom system prompt)")

        # Step A: Run WAN extender with custom system prompt (Qwen3.5 venv)
        import time as _time
        output_base = f"selftuning_{task_type}_{int(_time.time())}"
        qwen35_python = _QWEN_PYTHON
        extender_cmd = [
            qwen35_python, str(extender_script),
            '--prompt', user_prompt,
            '--image', str(Path(image_path).resolve()),
            '--system_prompt', custom_system_prompt,
            '--output', output_base
        ]

        try:
            ext_result = subprocess.run(
                extender_cmd, capture_output=True, text=True, timeout=300
            )
            if ext_result.returncode != 0:
                print(f"❌ Extender failed: {ext_result.stderr[-300:]}")
                return None, {'error': 'Extender failed'}

            # Read enhanced prompt from output file
            prompt_file = self.QWEN_EXTENDERS / 'outputs' / f"{output_base}_prompt.txt"
            if not prompt_file.exists():
                print(f"❌ Extender output not found: {prompt_file}")
                return None, {'error': 'Extender output missing'}

            with open(prompt_file, 'r') as f:
                content = f.read()

            # Extract enhanced prompt (between ======= lines)
            enhanced_prompt = ""
            negative_prompt_from_extender = ""

            parts = content.split('Negative Prompt:', 1)
            if len(parts) == 2:
                enhanced_section = parts[0]
                lines = enhanced_section.split('\n')
                capture = False
                for line in lines:
                    if '=' * 60 in line:
                        if not capture:
                            capture = True
                        else:
                            break
                    elif capture and line.strip():
                        enhanced_prompt += line + " "

                negative_section = parts[1].strip()
                for line in negative_section.split('\n'):
                    if '=' * 60 not in line and line.strip():
                        negative_prompt_from_extender += line.strip() + " "
                negative_prompt_from_extender = negative_prompt_from_extender.strip()

            enhanced_prompt = enhanced_prompt.strip()

            if not enhanced_prompt:
                print(f"❌ Could not parse enhanced prompt from extender output")
                return None, {'error': 'Empty enhanced prompt'}

            print(f"   [Direct] Enhanced: {len(enhanced_prompt.split())} words")
            if negative_prompt_from_extender:
                print(f"   [Direct] Negative (extender): {len(negative_prompt_from_extender.split())} terms")

        except subprocess.TimeoutExpired:
            print(f"❌ Extender timeout")
            return None, {'error': 'Extender timeout'}

        # Step B: Run WAN 2.2 directly with enhanced prompt
        # Use iteration-dependent seed for exploration (each iteration tries different noise)
        seed = 42 + (iteration - 1) * 17  # 42, 59, 76, 93, 110...
        print(f"   [Direct] WAN 2.2 generation (seed={seed})...")
        # Build negative prompt: prefer optimizer's custom prompt, fall back to extender's
        effective_negative = custom_negative_prompt or negative_prompt_from_extender or ""

        wan_cmd = [
            conda_bin, 'run', '-n', 'wan2.2', '--no-capture-output',
            'python3', str(wan_base / 'generate.py'),
            '--task', 'ti2v-5B',
            '--ckpt_dir', str(wan_base / 'Wan2.2-TI2V-5B'),
            '--prompt', enhanced_prompt,
            '--image', str(Path(image_path).resolve()),
            '--size', '1280*704',
            '--frame_num', '61',
            '--sample_steps', '30',
            '--sample_guide_scale', '7.5',
            '--base_seed', str(seed),
            '--save_file', str(output_path)
        ]
        if effective_negative:
            wan_cmd.extend(['--negative_prompt', effective_negative])

        try:
            subprocess.run(wan_cmd, timeout=600, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ WAN failed: {e}")
            return None, {'error': 'WAN failed'}
        except subprocess.TimeoutExpired:
            print(f"❌ WAN timeout")
            return None, {'error': 'WAN timeout'}

        if not output_path.exists():
            print(f"❌ Video not generated at {output_path}")
            return None, {'error': 'Video not generated'}

        print(f"   [Direct] Video: {output_path}")
        return output_path, {
            'task_type': task_type,
            'category': 'navigation',
            'user_prompt': user_prompt,
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt_from_extender': negative_prompt_from_extender,
            'custom_system_prompt': True,
            'video_path': str(output_path)
        }

    def validate_video(
        self,
        video_path: Path,
        task_type: str,
        user_prompt: str,
        custom_validation_prompt: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Validate video using Cosmos-Reason2 validator

        Args:
            video_path: Path to generated video
            task_type: Task type
            user_prompt: Original user prompt
            custom_validation_prompt: Optional custom system prompt for validator

        Returns:
            Validation result dict or None if failed
        """
        config = self.task_configs[task_type]
        category = config['category']

        # Get validator script path
        validator_script = self.agentllm_root / category.capitalize() / 'video_validator.py'

        if not validator_script.exists():
            print(f"❌ Validator not found: {validator_script}")
            return None

        # Use cosmos-reason2 venv
        reason2_python = Path(_COSMOS_REASON2_PYTHON)
        python_bin = str(reason2_python) if reason2_python.exists() else 'python3'

        # Build command — validator writes JSON to --output file
        import tempfile
        output_file = Path(tempfile.mktemp(suffix='.json', prefix='validation_'))
        cmd = [
            python_bin,
            str(validator_script),
            '--video', str(video_path),
            '--task', task_type,
            '--prompt', user_prompt,
            '--output', str(output_file),
        ]

        if custom_validation_prompt:
            cmd.extend(['--system-prompt', custom_validation_prompt])

        print(f"   Calling validator: {validator_script.name}")

        try:
            # Run validator — force offline mode so it uses cached model weights
            # without attempting to re-verify access on the gated HF Hub repo
            import os as _os
            validator_env = _os.environ.copy()
            validator_env['TRANSFORMERS_OFFLINE'] = '1'
            validator_env['HF_HUB_OFFLINE'] = '1'
            validator_env['HF_DATASETS_OFFLINE'] = '1'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=validator_script.parent,
                env=validator_env,
            )

            # returncode 1 = video failed validation (not a crash)
            if result.returncode not in (0, 1):
                print(f"❌ Validator crashed:")
                print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                return None

            # Read validation JSON from output file (validator writes to file, not stdout)
            if output_file.exists():
                import json as _json
                with open(output_file) as f:
                    return _json.load(f)
            else:
                print(f"⚠️  Validation output file not found: {output_file}")
                return None

        except subprocess.TimeoutExpired:
            print(f"❌ Validation timeout (>10 minutes)")
            return None
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return None

    def _extract_video_path(self, output: str, category: str) -> Optional[Path]:
        """
        Extract video path from pipeline output

        Looks for patterns like:
        - "Video saved: /path/to/video.mp4"
        - "Output: /path/to/video.mp4"
        """
        # Try multiple patterns
        patterns = [
            r'Video saved:\s*(.+\.mp4)',
            r'Output:\s*(.+\.mp4)',
            r'Generated:\s*(.+\.mp4)',
            r'(?:^|\n)(/[^\s]+\.mp4)',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.MULTILINE)
            if match:
                video_path = Path(match.group(1).strip())
                if video_path.exists():
                    return video_path

        # If not found, look in output directory
        output_dir = self.agentllm_root / category.capitalize() / 'outputs'
        if output_dir.exists():
            # Get most recent .mp4 file
            videos = sorted(output_dir.glob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
            if videos:
                return videos[0]

        return None

    def _extract_json(self, output: str) -> Optional[Dict]:
        """
        Extract JSON from output text

        Handles cases where JSON is embedded in other output
        """
        # Try direct JSON parse
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Look for validation JSON file
        # Check if there's a reference to JSON file in output
        json_file_match = re.search(r'Validation saved:\s*(.+\.json)', output)
        if json_file_match:
            json_file = Path(json_file_match.group(1).strip())
            if json_file.exists():
                with open(json_file, 'r') as f:
                    return json.load(f)

        return None


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python agentllm_interface.py <task_type> <prompt> <image_path>")
        print("Example: python agentllm_interface.py g1 'Humanoid picks up bottle' workspace.jpg")
        sys.exit(1)

    task_type = sys.argv[1]
    user_prompt = sys.argv[2]
    image_path = sys.argv[3]

    # Create interface
    interface = AgentLLMInterface()

    print("\n" + "="*70)
    print("TESTING AGENTLLM INTERFACE")
    print("="*70 + "\n")

    # Test video generation
    print("Step 1: Generating video...")
    video_path, metadata = interface.generate_video(
        task_type=task_type,
        user_prompt=user_prompt,
        image_path=image_path
    )

    if not video_path:
        print("❌ Video generation failed")
        sys.exit(1)

    print(f"✅ Video generated: {video_path}")

    # Test validation
    print("\nStep 2: Validating video...")
    validation = interface.validate_video(
        video_path=video_path,
        task_type=task_type,
        user_prompt=user_prompt
    )

    if not validation:
        print("❌ Validation failed")
        sys.exit(1)

    print(f"✅ Validation complete")
    print(json.dumps(validation, indent=2))

    print("\n" + "="*70)
    print("INTERFACE TEST COMPLETE")
    print("="*70 + "\n")
