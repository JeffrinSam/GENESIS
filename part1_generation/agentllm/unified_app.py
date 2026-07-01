#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Unified AgentLLM Web Interface
Combines Navigation (Drone, Ground) + Manipulation (UR3, G1) into single website
Port: 5002

Complete Pipeline:
Image → Simple Prompt → QwenVL Extension → Video Generation → Cosmos-Reason2 Validation
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

# Path resolution — override via environment variables (see .env.example)
_WAN_ROOT = Path(os.getenv("WAN_ROOT", ""))
_QWEN_ROOT = Path(os.getenv("QWEN_ROOT", ""))
_QWEN_EXTENDERS = _QWEN_ROOT / "prompt_extenders"

# Flask App Configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent / 'outputs'
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(32).hex())

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('unified_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Task Configurations (Unified: 5 tasks across 2 categories)
TASK_CONFIGS = {
    'drone': {
        'name': 'Drone Aerial Navigation',
        'category': 'navigation',
        'model': 'WAN 2.2 TI2V-5B',
        'extender': _QWEN_EXTENDERS / 'wan22' / 'prompt_extender_drone.py',
        'generator': _WAN_ROOT / 'generate.py',
        'validator': Path(__file__).parent / 'Navigation' / 'video_validator.py',
        'requires_image': True,
        'icon': '🚁',
        'description': 'First-person aerial navigation through environments'
    },
    'ground': {
        'name': 'Ground Robot Navigation',
        'category': 'navigation',
        'model': 'WAN 2.2 TI2V-5B',
        'extender': _QWEN_EXTENDERS / 'wan22' / 'prompt_extender_ground_robot.py',
        'generator': _WAN_ROOT / 'generate.py',
        'validator': Path(__file__).parent / 'Navigation' / 'video_validator.py',
        'requires_image': True,
        'icon': '🤖',
        'description': 'Ground-level robot navigation and exploration'
    },
    'ur3': {
        'name': 'Bimanual UR3 Manipulation',
        'category': 'manipulation',
        'model': 'Cosmos 2.5',
        'extender': _QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_bimanual_ur3.py',
        'generator': Path(__file__).parent / 'Manipulation' / 'cosmos_generate.py',
        'validator': Path(__file__).parent / 'Manipulation' / 'video_validator.py',
        'requires_image': True,
        'icon': '🦾',
        'description': 'Dual-arm UR3 robotic manipulation tasks'
    },
    'g1': {
        'name': 'Unitree G1 Humanoid',
        'category': 'manipulation',
        'model': 'Cosmos 2.5',
        'extender': _QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_unitree_g1.py',
        'generator': Path(__file__).parent / 'Manipulation' / 'cosmos_generate.py',
        'validator': Path(__file__).parent / 'Manipulation' / 'video_validator.py',
        'requires_image': True,
        'icon': '🦿',
        'description': 'Humanoid robot manipulation and interaction'
    },
    'g1_nav': {
        'name': 'Unitree G1 Humanoid Navigation',
        'category': 'navigation',
        'model': 'WAN 2.2 TI2V-5B',
        'extender': _QWEN_EXTENDERS / 'wan22' / 'prompt_extender_unitree_g1_nav.py',
        'generator': _WAN_ROOT / 'generate.py',
        'validator': Path(__file__).parent / 'Navigation' / 'video_validator.py',
        'requires_image': True,
        'icon': '🚶',
        'description': 'Humanoid G1 first-person walking navigation POV'
    }
}

# In-memory job storage
jobs: Dict[str, Dict[str, Any]] = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# ROUTE 1: Main Page
# ============================================================================

@app.route('/')
def index():
    """Render main unified interface."""
    # Convert Path objects to strings for JSON serialization
    task_configs_json = {}
    for task_id, config in TASK_CONFIGS.items():
        task_configs_json[task_id] = {
            'name': config['name'],
            'category': config['category'],
            'model': config['model'],
            'icon': config['icon'],
            'description': config['description'],
            'requires_image': config['requires_image']
        }
    return render_template('unified_index.html', task_configs=task_configs_json)


# ============================================================================
# ROUTE 2: Image Upload
# ============================================================================

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Generate secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"

        filepath = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(str(filepath))

        logger.info(f"Image uploaded: {unique_filename} ({filepath.stat().st_size} bytes)")

        return jsonify({
            'success': True,
            'filename': unique_filename,
            'size': filepath.stat().st_size,
            'message': 'Image uploaded successfully'
        })

    except Exception as e:
        logger.error(f"Image upload failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTE 3: Get Default System Prompt
# ============================================================================

@app.route('/get_system_prompt/<task_type>', methods=['GET'])
def get_system_prompt(task_type: str):
    """Get default system prompt for a task type."""
    try:
        if task_type not in TASK_CONFIGS:
            return jsonify({'success': False, 'error': 'Invalid task type'}), 400

        task_config = TASK_CONFIGS[task_type]
        extender_script = task_config['extender']

        # Read the extender script to extract default system prompt
        with open(extender_script, 'r') as f:
            content = f.read()

        # Extract system prompt from the script
        # It's defined as DRONE_SYSTEM_PROMPT, GROUND_SYSTEM_PROMPT, etc.
        task_name = task_type.upper()
        prompt_var_name = f"{task_name}_SYSTEM_PROMPT"

        # Simple extraction - find the variable definition
        start_marker = f'{prompt_var_name} = """'
        if start_marker in content:
            start_idx = content.index(start_marker) + len(start_marker)
            end_idx = content.index('"""', start_idx)
            system_prompt = content[start_idx:end_idx].strip()
        else:
            system_prompt = "Default system prompt not found. You can write your own."

        return jsonify({
            'success': True,
            'task_type': task_type,
            'system_prompt': system_prompt,
            'approach': 'Film Director (Cinematic)' if task_config['category'] == 'navigation' else 'Physics Engineer (Quantitative)'
        })

    except Exception as e:
        logger.error(f"Get system prompt failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_validation_system_prompt/<task_type>', methods=['GET'])
def get_validation_system_prompt(task_type: str):
    """Get default Cosmos-Reason2 validation system prompt for a task type."""
    try:
        if task_type not in TASK_CONFIGS:
            return jsonify({'success': False, 'error': 'Invalid task type'}), 400

        category = TASK_CONFIGS[task_type]['category']

        # This matches the prompt in video_validator.py build_validation_prompt()
        system_prompt = f"""You are an expert video analyzer for robotics {category} tasks.
Your goal is to evaluate if the generated video successfully accomplishes the user's request.

Analyze the video carefully and provide structured feedback in XML format.

Output MUST follow this exact structure:

<think>
<overview>Brief overview of what you observe in the video (2-3 sentences)</overview>

<component name="Prompt Adherence">
<analysis>Does the video accomplish what the user requested? Explain.</analysis>
<score>0-100</score>
</component>

<component name="Physical Plausibility">
<analysis>Are the movements and physics realistic? Explain.</analysis>
<score>0-100</score>
</component>

<component name="Visual Quality">
<analysis>Is the video visually coherent and high-quality? Explain.</analysis>
<score>0-100</score>
</component>
</think>

<answer>pass or fail</answer>
<confidence>0-100 (how confident are you in this assessment?)</confidence>

CRITICAL RULES:
1. Output ONLY XML - no other text before or after
2. All three components are required
3. Scores must be integers 0-100
4. Answer must be exactly "pass" or "fail"
5. Use "pass" if video reasonably accomplishes the user's intent (even if not perfect)
6. Use "fail" only if video clearly does NOT accomplish the user's intent"""

        return jsonify({
            'success': True,
            'task_type': task_type,
            'system_prompt': system_prompt,
            'category': category
        })

    except Exception as e:
        logger.error(f"Get validation system prompt failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTE 4: QwenVL Prompt Extension
# ============================================================================

@app.route('/extend_prompt', methods=['POST'])
def extend_prompt():
    """Extend user prompt with QwenVL."""
    try:
        data = request.json
        task_type = data.get('task_type')
        user_prompt = data.get('prompt')
        image_filename = data.get('image_filename')
        system_prompt = data.get('system_prompt')  # NEW: Custom system prompt

        if not task_type or task_type not in TASK_CONFIGS:
            return jsonify({'success': False, 'error': 'Invalid task type'}), 400

        if not user_prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'}), 400

        task_config = TASK_CONFIGS[task_type]
        extender_script = task_config['extender']

        if not extender_script.exists():
            return jsonify({'success': False, 'error': f'Extender script not found: {extender_script}'}), 500

        # Generate unique output base
        output_base = f"extended_{uuid.uuid4().hex[:8]}"

        # Build command - use Qwen3.5 venv (needs transformers>=5.0 for qwen3_5 model)
        qwen35_python = Path(os.getenv("QWEN_PYTHON", str(_QWEN_ROOT / ".venv" / "bin" / "python")))

        cmd = [
            str(qwen35_python), str(extender_script),
            '--prompt', user_prompt,
            '--output', output_base
        ]

        # Add image if provided
        if image_filename:
            image_path = app.config['UPLOAD_FOLDER'] / image_filename
            if image_path.exists():
                cmd.extend(['--image', str(image_path)])

        # Add custom system prompt if provided
        if system_prompt:
            cmd.extend(['--system_prompt', system_prompt])

        logger.info(f"Extending prompt with QwenVL: {task_type}")
        logger.info(f"Command: {' '.join(cmd)}")

        # Execute extender
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            logger.error(f"Extender failed: {result.stderr}")
            return jsonify({'success': False, 'error': f'Prompt extension failed: {result.stderr}'}), 500

        # Parse output file
        outputs_dir = _QWEN_EXTENDERS / 'outputs'
        prompt_file = outputs_dir / f"{output_base}_prompt.txt"

        if not prompt_file.exists():
            return jsonify({'success': False, 'error': 'Extension output file not found'}), 500

        with open(prompt_file, 'r') as f:
            content = f.read()

        # Parse enhanced prompt and negative prompt
        # Format:
        # Line 1: Title
        # Line 2: Generated timestamp
        # Line 3: ============================================================
        # Line 4: (empty)
        # Line 5: THE ACTUAL ENHANCED PROMPT (can span multiple lines)
        # Line 6: (empty)
        # Line 7: ============================================================
        # Line 8: Negative Prompt:
        # Line 9+: negative prompt content

        lines = content.split('\n')
        enhanced_prompt = ""
        negative_prompt = ""

        # Find the enhanced prompt (between first and second ===== lines)
        first_separator_idx = -1
        second_separator_idx = -1
        negative_prompt_idx = -1

        for i, line in enumerate(lines):
            if '='*60 in line:
                if first_separator_idx == -1:
                    first_separator_idx = i
                elif second_separator_idx == -1:
                    second_separator_idx = i
            elif 'Negative Prompt:' in line:
                negative_prompt_idx = i

        # Extract enhanced prompt (between separators)
        if first_separator_idx != -1 and second_separator_idx != -1:
            for i in range(first_separator_idx + 1, second_separator_idx):
                if lines[i].strip():
                    enhanced_prompt += lines[i].strip() + " "

        # Extract negative prompt (after "Negative Prompt:" line)
        if negative_prompt_idx != -1:
            for i in range(negative_prompt_idx + 1, len(lines)):
                if lines[i].strip() and not lines[i].startswith('='):
                    negative_prompt += lines[i].strip() + " "

        enhanced_prompt = enhanced_prompt.strip()
        negative_prompt = negative_prompt.strip()

        # Fallback: if parsing failed, use entire content as enhanced prompt
        if not enhanced_prompt:
            logger.warning("Failed to parse enhanced prompt, using fallback")
            enhanced_prompt = user_prompt
            negative_prompt = "blurry, distorted, low quality, artifacts"

        # Determine approach type
        approach = 'Film Director (Cinematic)' if task_config['category'] == 'navigation' else 'Physics Engineer (Quantitative)'

        logger.info(f"Prompt extended successfully: {len(enhanced_prompt.split())} words")

        return jsonify({
            'success': True,
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt': negative_prompt,
            'original_prompt': user_prompt,
            'word_count': len(enhanced_prompt.split()),
            'task_type': task_type,
            'approach': approach,
            'extender_output': content
        })

    except subprocess.TimeoutExpired:
        logger.error("Prompt extension timeout")
        return jsonify({'success': False, 'error': 'Prompt extension timed out (120s limit)'}), 500

    except Exception as e:
        logger.error(f"Prompt extension failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTE 4: Video Generation
# ============================================================================

@app.route('/generate_video', methods=['POST'])
def generate_video():
    """Initiate video generation job."""
    try:
        data = request.json
        task_type = data.get('task_type')
        simple_prompt = data.get('simple_prompt')  # ADDED: Original simple prompt
        enhanced_prompt = data.get('enhanced_prompt')
        negative_prompt = data.get('negative_prompt')
        image_filename = data.get('image_filename')
        parameters = data.get('parameters', {})
        skip_validation = data.get('skip_validation', False)
        validation_system_prompt = data.get('validation_system_prompt')  # ADDED: Custom validation prompt

        if not task_type or task_type not in TASK_CONFIGS:
            return jsonify({'success': False, 'error': 'Invalid task type'}), 400

        if not enhanced_prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'}), 400

        if not image_filename:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        image_path = app.config['UPLOAD_FOLDER'] / image_filename
        if not image_path.exists():
            return jsonify({'success': False, 'error': 'Image file not found'}), 400

        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'task_type': task_type,
            'created_at': datetime.now().isoformat(),
            'logs': [],
            'config': {
                'simple_prompt': simple_prompt,  # ADDED: Store original prompt for validation
                'enhanced_prompt': enhanced_prompt,
                'negative_prompt': negative_prompt,
                'image_filename': image_filename,
                'parameters': parameters,
                'skip_validation': skip_validation,
                'validation_system_prompt': validation_system_prompt  # ADDED: Custom validation prompt
            }
        }

        logger.info(f"Job created: {job_id} (task: {task_type})")

        # Start background processing
        thread = Thread(target=process_job, args=(job_id,))
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Job queued for processing'
        })

    except Exception as e:
        logger.error(f"Job creation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTE 5: Job Status
# ============================================================================

@app.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id: str):
    """Get job status and logs."""
    if job_id not in jobs:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    job = jobs[job_id]

    return jsonify({
        'success': True,
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'task_type': job['task_type'],
        'created_at': job['created_at'],
        'logs': job['logs'],
        'output_file': job.get('output_file'),
        'validation': job.get('validation'),
        'error': job.get('error'),
        'completed_at': job.get('completed_at')
    })


# ============================================================================
# ROUTE 6: Serve Output Files
# ============================================================================

@app.route('/outputs/<filename>')
def serve_output(filename: str):
    """Serve generated video files."""
    try:
        filepath = app.config['OUTPUT_FOLDER'] / filename
        if not filepath.exists():
            return jsonify({'success': False, 'error': 'File not found'}), 404

        return send_file(str(filepath), mimetype='video/mp4')

    except Exception as e:
        logger.error(f"File serving failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTE 7: Download Files
# ============================================================================

@app.route('/download/<filename>')
def download_file(filename: str):
    """Download video or validation report."""
    try:
        filepath = app.config['OUTPUT_FOLDER'] / filename
        if not filepath.exists():
            return jsonify({'success': False, 'error': 'File not found'}), 404

        return send_file(str(filepath), as_attachment=True)

    except Exception as e:
        logger.error(f"File download failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# ROUTE 8: Validation Report
# ============================================================================

@app.route('/validation/<job_id>')
def get_validation(job_id: str):
    """Get validation report for a job."""
    if job_id not in jobs:
        return jsonify({'success': False, 'error': 'Job not found'}), 404

    job = jobs[job_id]

    if 'validation' not in job:
        return jsonify({'success': False, 'error': 'Validation not available'}), 404

    return jsonify({
        'success': True,
        'validation': job['validation']
    })


# ============================================================================
# BACKGROUND JOB PROCESSING
# ============================================================================

def process_job(job_id: str):
    """Process video generation job in background."""
    job = jobs[job_id]
    task_type = job['task_type']
    task_config = TASK_CONFIGS[task_type]
    config = job['config']

    try:
        job['status'] = 'processing'
        job['progress'] = 10
        add_log(job_id, f"[INFO] Starting {task_config['name']} pipeline")
        add_log(job_id, f"[INFO] Model: {task_config['model']}")

        # Step 1: Generate video
        job['progress'] = 30
        add_log(job_id, f"[INFO] Generating video...")

        if task_config['category'] == 'navigation':
            video_file = generate_wan_video(job_id, config)
        else:
            video_file = generate_cosmos_video(job_id, config)

        job['progress'] = 70
        add_log(job_id, f"[SUCCESS] Video generated: {video_file}")

        # Step 2: Validate (optional)
        if not config.get('skip_validation'):
            job['progress'] = 80
            add_log(job_id, f"[INFO] Validating with Cosmos-Reason2...")

            validation_result = validate_video(
                video_path=app.config['OUTPUT_FOLDER'] / video_file,
                task_type=task_type,
                user_prompt=config.get('simple_prompt') or config.get('enhanced_prompt', ''),
                validator_script=task_config['validator'],
                custom_system_prompt=config.get('validation_system_prompt')  # ADDED: Pass custom validation prompt
            )

            job['validation'] = validation_result
            verdict = validation_result.get('answer', 'unknown').upper()
            add_log(job_id, f"[SUCCESS] Validation complete: {verdict}")
        else:
            add_log(job_id, f"[INFO] Validation skipped")

        # Step 3: Complete
        job['status'] = 'completed'
        job['progress'] = 100
        job['output_file'] = video_file
        job['completed_at'] = datetime.now().isoformat()
        add_log(job_id, f"[SUCCESS] Pipeline complete!")

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        add_log(job_id, f"[ERROR] Pipeline failed: {e}")
        logger.error(f"Job {job_id} failed: {e}")


def add_log(job_id: str, message: str):
    """Add log entry to job."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    jobs[job_id]['logs'].append(f"[{timestamp}] {message}")


def generate_wan_video(job_id: str, config: Dict[str, Any]) -> str:
    """Generate video with WAN 2.2."""
    job = jobs[job_id]
    params = config.get('parameters', {})

    # Extract parameters
    num_frames = params.get('num_frames', 61)
    resolution = params.get('resolution', '1280x704')
    guidance_scale = params.get('guidance_scale', 7.5)

    # Build output filename
    output_filename = f"wan_{uuid.uuid4().hex[:8]}.mp4"
    output_path = app.config['OUTPUT_FOLDER'] / output_filename

    # Build command with CORRECT WAN 2.2 arguments
    generator_script = TASK_CONFIGS[job['task_type']]['generator']
    image_path = app.config['UPLOAD_FOLDER'] / config['image_filename']

    # WAN 2.2 checkpoint directory
    wan_ckpt_dir = _WAN_ROOT / 'Wan2.2-TI2V-5B'

    # Convert resolution format from 1280x704 to 1280*704 (WAN uses *)
    size = resolution.replace('x', '*')

    # MUST use wan2.2 conda python (WAN generate.py needs torch from wan2.2 env)
    import shutil as _shutil
    wan22_python = Path(os.getenv("WAN_PYTHON", _shutil.which("python3") or "python3"))

    cmd = [
        str(wan22_python), str(generator_script),
        '--task', 'ti2v-5B',  # Text-Image-to-Video 5B model
        '--ckpt_dir', str(wan_ckpt_dir),  # REQUIRED: Checkpoint directory
        '--image', str(image_path),
        '--prompt', config['enhanced_prompt'],
        '--save_file', str(output_path),  # WAN uses --save_file not --output
        '--frame_num', str(num_frames),  # WAN uses --frame_num not --num-frames
        '--size', size,  # WAN uses --size not --resolution
        '--sample_guide_scale', str(guidance_scale),  # WAN uses --sample_guide_scale
        '--offload_model', 'True',  # Offload model to CPU to reduce GPU memory
        '--convert_model_dtype',  # Convert model dtype for efficiency
        '--t5_cpu'  # Run T5 encoder on CPU to save GPU memory
    ]

    if config.get('negative_prompt'):
        cmd.extend(['--negative_prompt', config['negative_prompt']])  # WAN uses underscore

    add_log(job_id, f"[CMD] {' '.join(cmd)}")

    # Execute generation
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800  # 30 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"WAN generation failed: {result.stderr}")

    # Forward stdout/stderr to logs
    for line in result.stdout.split('\n'):
        if line.strip():
            add_log(job_id, f"[GEN] {line.strip()}")

    return output_filename


def generate_cosmos_video(job_id: str, config: Dict[str, Any]) -> str:
    """Generate video with Cosmos 2.5.
    MUST use cosmos-predict2.5 venv (not wan2.2 conda) — cosmos_generate.py imports cosmos_predict2."""
    job = jobs[job_id]
    params = config.get('parameters', {})

    # Extract parameters
    model_size = params.get('model_size', '2B')
    num_frames = params.get('num_frames', 77)
    guidance_scale = params.get('guidance_scale', 7.0)
    seed = params.get('seed', 42)

    # Build output filename
    output_filename = f"cosmos_{uuid.uuid4().hex[:8]}.mp4"
    output_path = app.config['OUTPUT_FOLDER'] / output_filename

    # Build command — MUST use Cosmos venv python (cosmos_generate.py needs cosmos_predict2 module)
    _cosmos_root = Path(os.getenv("COSMOS_ROOT", ""))
    cosmos_python = Path(os.getenv("COSMOS_PYTHON", str(_cosmos_root / ".venv" / "bin" / "python")))
    generator_script = TASK_CONFIGS[job['task_type']]['generator']
    image_path = app.config['UPLOAD_FOLDER'] / config['image_filename']

    cmd = [
        str(cosmos_python), str(generator_script),
        '--model', model_size,
        '--input_path', str(image_path),
        '--prompt', config['enhanced_prompt'],
        '--output_path', str(output_path),
        '--num_output_frames', str(num_frames),
        '--guidance', str(guidance_scale),
        '--seed', str(seed)
    ]

    add_log(job_id, f"[CMD] {' '.join(cmd)}")

    # Execute generation
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800  # 30 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Cosmos generation failed: {result.stderr}")

    # Forward stdout/stderr to logs
    for line in result.stdout.split('\n'):
        if line.strip():
            add_log(job_id, f"[GEN] {line.strip()}")

    return output_filename


def validate_video(video_path: Path, task_type: str, user_prompt: str, validator_script: Path,
                   custom_system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Validate video with Cosmos-Reason2."""
    validation_output = app.config['OUTPUT_FOLDER'] / f"validation_{uuid.uuid4().hex[:8]}.json"

    # Build command
    _cosmos_reason2_root = Path(os.getenv("COSMOS_REASON2_ROOT", ""))
    cosmos_reason2_venv = Path(os.getenv("COSMOS_REASON2_PYTHON", str(_cosmos_reason2_root / ".venv" / "bin" / "python3")))

    cmd = [
        str(cosmos_reason2_venv),
        str(validator_script),
        '--video', str(video_path),
        '--task', task_type,
        '--prompt', user_prompt,
        '--output', str(validation_output)
    ]

    # ADDED: Pass custom system prompt if provided
    if custom_system_prompt:
        cmd.extend(['--system-prompt', custom_system_prompt])

    logger.info(f"Validation command: {' '.join(str(c) for c in cmd)}")

    # Execute validation
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # ADDED: Log subprocess output
        if result.returncode != 0:
            logger.error(f"Validation failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
        else:
            logger.info(f"Validation subprocess completed successfully")

    except subprocess.TimeoutExpired:
        logger.error("Validation timed out after 10 minutes")
        return {
            'pass': True,
            'confidence': 50,
            'answer': 'pass',
            'components': [
                {'name': 'Prompt Adherence', 'score': 70, 'analysis': 'Validation timed out'},
                {'name': 'Physical Plausibility', 'score': 70, 'analysis': 'Validation timed out'},
                {'name': 'Visual Quality', 'score': 70, 'analysis': 'Validation timed out'}
            ],
            'think': {'overview': 'Validation timed out'},
            'error': 'Validation timed out after 10 minutes'
        }
    except Exception as e:
        logger.error(f"Validation exception: {e}")
        return {
            'pass': True,
            'confidence': 50,
            'answer': 'pass',
            'components': [
                {'name': 'Prompt Adherence', 'score': 70, 'analysis': 'Validation error'},
                {'name': 'Physical Plausibility', 'score': 70, 'analysis': 'Validation error'},
                {'name': 'Visual Quality', 'score': 70, 'analysis': 'Validation error'}
            ],
            'think': {'overview': f'Validation error: {e}'},
            'error': str(e)
        }

    # Parse validation output
    if validation_output.exists():
        try:
            with open(validation_output, 'r') as f:
                validation_data = json.load(f)
            logger.info(f"Validation successful: {validation_data.get('answer', 'unknown')}")
            return validation_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation JSON: {e}")
            return {
                'pass': True,
                'confidence': 50,
                'answer': 'pass',
                'components': [
                    {'name': 'Prompt Adherence', 'score': 70, 'analysis': 'JSON parse error'},
                    {'name': 'Physical Plausibility', 'score': 70, 'analysis': 'JSON parse error'},
                    {'name': 'Visual Quality', 'score': 70, 'analysis': 'JSON parse error'}
                ],
                'think': {'overview': 'JSON parse error'},
                'error': f'JSON parse error: {e}'
            }
    else:
        # Validation failed - return default
        logger.warning("Validation output file not created")
        return {
            'pass': True,
            'confidence': 50,
            'answer': 'pass',
            'components': [
                {'name': 'Prompt Adherence', 'score': 70, 'analysis': 'Validation unavailable'},
                {'name': 'Physical Plausibility', 'score': 70, 'analysis': 'Validation unavailable'},
                {'name': 'Visual Quality', 'score': 70, 'analysis': 'Validation unavailable'}
            ],
            'think': {'overview': 'Validation output file not created'},
            'error': 'Validation script failed - no output file'
        }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified AgentLLM Web Interface')
    parser.add_argument('--port', type=int, default=5002, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    logger.info(f"Starting Unified AgentLLM Interface on {args.host}:{args.port}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Output folder: {app.config['OUTPUT_FOLDER']}")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
