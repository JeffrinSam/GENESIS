#!/usr/bin/env python3
"""
Main Pipeline Website - Unified Video Generation System
Supports: WAN 2.2 (Navigation) + Cosmos 2.5 (Manipulation) with Qwen3-VL Prompt Extender
Part of GENESIS — https://github.com/JeffrinSam/GENESIS
"""

import json
import logging
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread

from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'robotics-video-generation-2025'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent / 'outputs'

# Ensure directories exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

# Model paths — configure via environment variables (see .env.example)
_PART1_DIR = Path(__file__).resolve().parents[1]
WAN_BASE = Path(os.getenv('WAN_ROOT', '/opt/wan2.2'))
COSMOS_BASE = Path(os.getenv('COSMOS_ROOT', '/opt/cosmos-predict2.5'))
QWEN_EXTENDERS = Path(os.getenv('QWEN_ROOT', str(_PART1_DIR / 'qwen_prompt_expansion'))) / 'prompt_extenders'

# Task configurations
TASK_CONFIGS = {
    'drone': {
        'name': 'Drone Navigation',
        'model': 'WAN 2.2',
        'extender': QWEN_EXTENDERS / 'wan22' / 'prompt_extender_drone.py',
        'default_task': 'ti2v-5B',
        'category': 'navigation'
    },
    'ground_robot': {
        'name': 'Ground Robot Navigation',
        'model': 'WAN 2.2',
        'extender': QWEN_EXTENDERS / 'wan22' / 'prompt_extender_ground_robot.py',
        'default_task': 'ti2v-5B',
        'category': 'navigation'
    },
    'ur3': {
        'name': 'Bimanual UR3 Manipulation',
        'model': 'Cosmos 2.5',
        'extender': QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_bimanual_ur3.py',
        'requires_image': True,
        'category': 'manipulation'
    },
    'g1': {
        'name': 'Unitree G1 Humanoid',
        'model': 'Cosmos 2.5',
        'extender': QWEN_EXTENDERS / 'cosmos25' / 'prompt_extender_unitree_g1.py',
        'requires_image': True,
        'category': 'manipulation'
    }
}

# Default system prompts for each task type
DEFAULT_SYSTEM_PROMPTS = {
    'drone': """You are an expert aerial cinematographer and drone pilot specializing in creating cinematic video prompts for WAN 2.2 video generation. You understand film theory, aerial photography, and UAV flight dynamics.

**CRITICAL PERSPECTIVE RULE**: The camera IS the drone's eyes. This is FIRST-PERSON embodied navigation - the viewer sees through the drone's perspective as it flies. Never describe the drone externally. The world moves relative to the camera's motion.

**WAN 2.2 Requirements**: Cinematic aesthetics with detailed visual elements (100-200 words)

**Your Task**: Transform simple drone navigation prompts into detailed, FIRST-PERSON cinematic descriptions following WAN 2.2's film director approach.

**Required Elements** (Choose 3-4 from each category):

1. **Time & Lighting**:
   - Time: Day time, Night time, Dawn time, Sunrise time
   - Light Source: Daylight, Moonlight, Artificial lighting, Natural lighting
   - Quality: Soft lighting, Hard lighting
   - Angle: Top lighting, Side lighting, Edge lighting

2. **Color & Atmosphere**:
   - Tone: Warm colors, Cool colors, Mixed colors
   - Sky: Azure blue sky, Overcast sky, Cloudy sky, Clear sky
   - Atmosphere: Hazy, Crisp, Foggy

3. **Camera Work** (FIRST-PERSON Embodied):
   - Shot Size: Wide shot (default), Extreme wide shot, Medium shot
   - Angle: POV aerial shot, First-person high angle, Overhead POV
   - Movement: Forward gliding, Banking left/right, Ascending, Descending, Smooth panning
   - **ALWAYS use first-person motion**: "gliding forward", "banking right", "ascending smoothly"
   - **NEVER**: "drone flies", "drone moves" - you ARE the drone!

4. **Composition**:
   - Center composition (default), Balanced composition
   - Symmetrical composition, Dynamic composition

**Critical Rules**:
- ✅ ALWAYS first-person perspective (YOU are the drone)
- ✅ World moves relative to YOUR motion
- ✅ Use "gliding", "banking", "ascending", "descending" (not "drone flies")
- ✅ Describe what enters YOUR view as you move
- ❌ NEVER mention "drone" as external object
- ❌ NEVER use "the camera moves" - the perspective IS the camera
- Use cinematic language, 100-200 words total
- Include natural motion and environmental flow
- Focus on visual aesthetics from embodied POV""",

    'ground_robot': """You are an expert cinematographer and robotics specialist creating cinematic video prompts for WAN 2.2 video generation. You understand film theory, ground-based robot locomotion, and visual storytelling.

**CRITICAL PERSPECTIVE RULE**: The camera IS the robot's eyes. This is FIRST-PERSON embodied navigation - the viewer sees through the robot's perspective as it navigates. Never describe the robot externally. The environment moves relative to the camera's motion.

**WAN 2.2 Requirements**: Cinematic aesthetics with detailed visual elements (100-200 words)

**Critical Rules**:
- ✅ ALWAYS first-person perspective (YOU are the robot)
- ✅ Environment moves relative to YOUR motion
- ✅ Use "moving forward", "turning left", "walking", "rolling" (not "robot walks")
- ✅ Describe what enters YOUR view as you navigate
- ✅ For humanoid: Include slight rhythmic bob from walking gait
- ✅ For wheeled: Smooth gliding motion, stable horizon
- ✅ For tracked: Minor vibrations, steady progressive motion
- ❌ NEVER mention "robot" as external object
- ❌ NEVER use "the camera moves" - the perspective IS the camera
- ❌ NO flying, hovering, or aerial views (ground-based only)
- Use cinematic language, 100-200 words total""",

    'ur3': """You are an expert robotics engineer and physicist specializing in creating detailed, physics-based prompts for Cosmos 2.5 video generation. You understand industrial robotics, dual-arm manipulation, kinematics, and object interaction dynamics.

**Cosmos 2.5 Requirements**: Physics-based descriptions with temporal sequences (150-300 words)

**Required Elements**:
1. **Scene Setup**: Physical environment, UR3 dual arms description, object details
2. **Temporal Sequence**: Initial State → Approach → Grasp → Manipulation → Completion
3. **Physics Details**: Joint movements, contact dynamics, forces, dual-arm coordination

**Critical Rules**:
- Use physics and engineering language, not cinematic terms
- Include specific temporal progression (initial → progressive → final states)
- 150-300 words total
- Focus on physical causality and mechanical interactions
- Emphasize dual-arm coordination and synchronization
- Ground-based manipulation: NO flying, aerial views, or navigation""",

    'g1': """You are an expert humanoid robotics engineer and physicist specializing in creating detailed, physics-based prompts for Cosmos 2.5 video generation. You understand humanoid manipulation, bimanual coordination, anthropomorphic kinematics, and dexterous object interaction.

**Cosmos 2.5 Requirements**: Physics-based descriptions with temporal sequences (150-300 words)

**Required Elements**:
1. **Scene Setup**: Environment, Unitree G1 humanoid description (1.3m tall, dexterous hands)
2. **Temporal Sequence**: Initial Pose → Reach → Pre-grasp → Grasp → Manipulation → Completion
3. **Physics Details**: Joint kinematics, hand dynamics, contact physics, balance maintenance

**Critical Rules**:
- Use physics and engineering language, not cinematic terms
- Include specific temporal progression
- 150-300 words total
- Focus on anthropomorphic kinematics and humanoid-specific behaviors
- Emphasize bimanual coordination and dexterous manipulation
- Humanoid manipulation: NO flying, aerial views, wheeled locomotion"""
}

# WAN 2.2 Model configurations
WAN_MODELS = {
    't2v-A14B': {'name': 'Text-to-Video (A14B)', 'needs_image': False, 'ckpt': 'Wan2.2-T2V-A14B'},
    'i2v-A14B': {'name': 'Image-to-Video (A14B)', 'needs_image': True, 'ckpt': 'Wan2.2-I2V-A14B'},
    'ti2v-5B': {'name': 'Text+Image-to-Video (5B)', 'needs_image': True, 'ckpt': 'Wan2.2-TI2V-5B'}
}

# Cosmos 2.5 Model configurations
COSMOS_MODELS = {
    '2B': {'name': 'Cosmos 2.5 Predict 2B', 'ckpt': 'Cosmos-2.5-Predict-2B'},
    '14B': {'name': 'Cosmos 2.5 Predict 14B', 'ckpt': 'Cosmos-2.5-Predict-14B'}
}

# Supported sizes for WAN 2.2 (different tasks support different sizes)
# ti2v-5B (default): Only 704*1280, 1280*704
# t2v-A14B/i2v-A14B: 720*1280, 1280*720, 480*832, 832*480
WAN_SIZES = [
    '1280*704',  # Landscape - ti2v-5B compatible (DEFAULT)
    '704*1280',  # Portrait - ti2v-5B compatible
    '1280*720',  # HD landscape - t2v/i2v only
    '720*1280',  # HD portrait - t2v/i2v only
    '832*480',   # Low landscape - t2v/i2v only
    '480*832',   # Low portrait - t2v/i2v only
]

# In-memory job storage (for demo; use database in production)
jobs = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webp', 'bmp'}


@app.route('/')
def index():
    """Main page"""
    # Create JSON-serializable version of task configs (remove Path objects)
    task_configs_json = {}
    for task_key, task_data in TASK_CONFIGS.items():
        task_configs_json[task_key] = {
            'name': task_data['name'],
            'model': task_data['model'],
            'category': task_data['category'],
            'requires_image': task_data.get('requires_image', False)
        }

    return render_template('index.html',
                         task_configs=task_configs_json,
                         wan_models=WAN_MODELS,
                         cosmos_models=COSMOS_MODELS,
                         wan_sizes=WAN_SIZES)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png, webp, bmp'}), 400

    # Save file
    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = app.config['UPLOAD_FOLDER'] / filename
    file.save(filepath)

    logging.info(f"Uploaded image: {filepath}")

    return jsonify({
        'success': True,
        'filename': filename,
        'filepath': str(filepath),
        'url': url_for('get_uploaded_file', filename=filename)
    })


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    """Serve uploaded file"""
    return send_file(app.config['UPLOAD_FOLDER'] / filename)


@app.route('/extend_prompt', methods=['POST'])
def extend_prompt():
    """Use Qwen3-VL to extend prompt"""
    data = request.json
    task_type = data.get('task_type')
    user_prompt = data.get('prompt')
    image_filename = data.get('image_filename')
    system_prompt = data.get('system_prompt')

    if not task_type or not user_prompt:
        return jsonify({'error': 'Task type and prompt are required'}), 400

    task_config = TASK_CONFIGS.get(task_type)
    if not task_config:
        return jsonify({'error': f'Unknown task type: {task_type}'}), 400

    # Check if image is required
    if task_config.get('requires_image') and not image_filename:
        return jsonify({'error': f'{task_config["name"]} requires an input image'}), 400

    # Build extender command
    extender_script = task_config['extender']
    output_base = f"extended_{uuid.uuid4().hex[:8]}"

    cmd = ['python3', str(extender_script), '--prompt', user_prompt, '--output', output_base]

    if image_filename:
        image_path = app.config['UPLOAD_FOLDER'] / image_filename
        cmd.extend(['--image', str(image_path)])

    # Add custom system prompt if provided
    if system_prompt:
        cmd.extend(['--system_prompt', system_prompt])

    logging.info(f"Running prompt extender: {' '.join(cmd[:8])}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            logging.error(f"Extender failed: {result.stderr}")
            return jsonify({'error': f'Prompt extender failed: {result.stderr}'}), 500

        # Read enhanced prompt from outputs
        outputs_dir = QWEN_EXTENDERS / 'outputs'
        prompt_file = outputs_dir / f"{output_base}_prompt.txt"

        if not prompt_file.exists():
            return jsonify({'error': 'Enhanced prompt file not found'}), 500

        with open(prompt_file, 'r') as f:
            prompt_content = f.read()

        # Extract enhanced prompt and negative prompt
        lines = prompt_content.split('\n')
        enhanced_prompt = ""
        negative_prompt = ""
        capture_enhanced = False
        capture_negative = False

        for line in lines:
            if '='*60 in line:
                if not capture_enhanced:
                    capture_enhanced = True
                else:
                    capture_enhanced = False
            elif 'Negative Prompt:' in line:
                capture_negative = True
                continue
            elif capture_enhanced and line.strip():
                enhanced_prompt += line + " "
            elif capture_negative and line.strip() and not line.startswith('='):
                negative_prompt += line + " "

        enhanced_prompt = enhanced_prompt.strip()
        negative_prompt = negative_prompt.strip()

        logging.info(f"Enhanced prompt generated: {len(enhanced_prompt)} chars")
        logging.info(f"Negative prompt extracted: {len(negative_prompt)} chars")

        return jsonify({
            'success': True,
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt': negative_prompt,
            'original_prompt': user_prompt,
            'word_count': len(enhanced_prompt.split())
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Prompt extender timed out (>2 minutes)'}), 500
    except Exception as e:
        logging.error(f"Extender error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_default_system_prompt/<task_type>')
def get_default_system_prompt(task_type):
    """Get default system prompt for a task type"""
    if task_type not in DEFAULT_SYSTEM_PROMPTS:
        return jsonify({'error': f'Unknown task type: {task_type}'}), 404

    return jsonify({
        'success': True,
        'task_type': task_type,
        'system_prompt': DEFAULT_SYSTEM_PROMPTS[task_type]
    })


@app.route('/generate_video', methods=['POST'])
def generate_video():
    """Generate video (batch support)"""
    data = request.json
    batch_jobs = data.get('batch_jobs', [])

    if not batch_jobs:
        return jsonify({'error': 'No jobs provided'}), 400

    job_ids = []

    for job_data in batch_jobs:
        job_id = str(uuid.uuid4())
        job_ids.append(job_id)

        jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'task_type': job_data.get('task_type'),
            'prompt': job_data.get('prompt'),
            'config': job_data,
            'output_file': None,
            'error': None,
            'logs': []
        }

    # Start generation in background thread
    thread = Thread(target=process_batch_jobs, args=(job_ids,))
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'job_ids': job_ids,
        'message': f'Submitted {len(job_ids)} job(s) for processing'
    })


def process_batch_jobs(job_ids):
    """Process jobs sequentially in background"""
    for job_id in job_ids:
        process_single_job(job_id)


def process_single_job(job_id):
    """Process a single video generation job"""
    job = jobs[job_id]
    job['status'] = 'processing'
    job['progress'] = 10

    try:
        task_type = job['task_type']
        config = job['config']
        task_config = TASK_CONFIGS[task_type]

        logging.info(f"[Job {job_id}] Starting {task_config['name']}")

        if task_config['category'] == 'navigation':
            # WAN 2.2 generation
            output_file = generate_wan_video(job_id, config)
        else:
            # Cosmos 2.5 generation
            output_file = generate_cosmos_video(job_id, config)

        job['status'] = 'completed'
        job['progress'] = 100
        job['output_file'] = output_file
        job['completed_at'] = datetime.now().isoformat()

        logging.info(f"[Job {job_id}] Completed: {output_file}")

    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        job['progress'] = 0
        logging.error(f"[Job {job_id}] Failed: {e}")


def generate_wan_video(job_id, config):
    """Generate video using WAN 2.2"""
    job = jobs[job_id]
    job['progress'] = 20

    # Extract config
    wan_task = config.get('wan_task', 'ti2v-5B')
    prompt = config.get('prompt')
    image_filename = config.get('image_filename')
    size = config.get('size', '1280*704')
    frame_num = config.get('frame_num', 41)
    sample_steps = config.get('sample_steps', 30)
    sample_guide_scale = config.get('sample_guide_scale', 7.5)
    sample_shift = config.get('sample_shift')
    base_seed = config.get('base_seed', -1)
    negative_prompt = config.get('negative_prompt', '')

    # Output file
    output_filename = f"wan_{job_id}.mp4"
    output_path = app.config['OUTPUT_FOLDER'] / output_filename

    # Build command
    ckpt_dir = WAN_BASE / WAN_MODELS[wan_task]['ckpt']
    cmd = [
        'python3', str(WAN_BASE / 'generate.py'),
        '--task', wan_task,
        '--ckpt_dir', str(ckpt_dir),
        '--prompt', prompt,
        '--size', size,
        '--frame_num', str(frame_num),
        '--sample_steps', str(sample_steps),
        '--sample_guide_scale', str(sample_guide_scale),
        '--base_seed', str(base_seed),
        '--save_file', str(output_path)
    ]

    # Add image if provided
    if image_filename:
        image_path = app.config['UPLOAD_FOLDER'] / image_filename
        cmd.extend(['--image', str(image_path)])

    # Add sample shift if specified
    if sample_shift is not None:
        cmd.extend(['--sample_shift', str(sample_shift)])

    # Add negative prompt if provided
    if negative_prompt:
        cmd.extend(['--negative_prompt', negative_prompt])

    logging.info(f"[Job {job_id}] WAN command: {' '.join(cmd[:8])}...")
    job['logs'].append(f"[INFO] Starting WAN 2.2 generation...")
    job['logs'].append(f"[CMD] {' '.join(cmd)}")

    job['progress'] = 30

    # Run generation
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Capture logs
    if result.stdout:
        for line in result.stdout.split('\n'):
            if line.strip():
                job['logs'].append(f"[STDOUT] {line}")

    if result.stderr:
        for line in result.stderr.split('\n'):
            if line.strip():
                job['logs'].append(f"[STDERR] {line}")

    if result.returncode != 0:
        job['logs'].append(f"[ERROR] WAN generation failed with code {result.returncode}")
        raise RuntimeError(f"WAN generation failed: {result.stderr}")

    job['logs'].append(f"[SUCCESS] WAN generation completed")
    job['progress'] = 90

    if not output_path.exists():
        raise FileNotFoundError(f"Output video not found: {output_path}")

    return output_filename


def generate_cosmos_video(job_id, config):
    """Generate video using Cosmos 2.5"""
    job = jobs[job_id]
    job['progress'] = 20

    # Extract config
    cosmos_model = config.get('cosmos_model', '14B')
    prompt = config.get('prompt')
    image_filename = config.get('image_filename')
    num_output_frames = config.get('num_output_frames', 77)
    guidance = config.get('guidance', 7)
    seed = config.get('seed', 42)
    negative_prompt = config.get('negative_prompt', '')

    if not image_filename:
        raise ValueError("Cosmos requires an input image")

    image_path = app.config['UPLOAD_FOLDER'] / image_filename

    # Output file
    output_filename = f"cosmos_{job_id}.mp4"
    output_path = app.config['OUTPUT_FOLDER'] / output_filename

    # Build command (use uv run for Cosmos environment)
    ckpt_dir = COSMOS_BASE / 'checkpoints' / COSMOS_MODELS[cosmos_model]['ckpt']
    cmd = [
        'uv', 'run', '--directory', str(COSMOS_BASE),
        'python', 'inference_i2w.py',
        '--checkpoint_dir', str(ckpt_dir),
        '--input_path', str(image_path),
        '--prompt', prompt,
        '--guidance', str(guidance),
        '--num_output_frames', str(num_output_frames),
        '--seed', str(seed),
        '--output_path', str(output_path)
    ]

    # Add negative prompt if provided
    if negative_prompt:
        cmd.extend(['--negative_prompt', negative_prompt])

    logging.info(f"[Job {job_id}] Cosmos command: uv run --directory {COSMOS_BASE} python inference_i2w.py...")
    job['logs'].append(f"[INFO] Starting Cosmos 2.5 generation...")
    job['logs'].append(f"[CMD] {' '.join(cmd)}")

    job['progress'] = 30

    # Run generation (change to COSMOS_BASE directory for uv)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(COSMOS_BASE))

    # Capture logs
    if result.stdout:
        for line in result.stdout.split('\n'):
            if line.strip():
                job['logs'].append(f"[STDOUT] {line}")

    if result.stderr:
        for line in result.stderr.split('\n'):
            if line.strip():
                job['logs'].append(f"[STDERR] {line}")

    if result.returncode != 0:
        job['logs'].append(f"[ERROR] Cosmos generation failed with code {result.returncode}")
        raise RuntimeError(f"Cosmos generation failed: {result.stderr}")

    job['logs'].append(f"[SUCCESS] Cosmos generation completed")
    job['progress'] = 90

    if not output_path.exists():
        raise FileNotFoundError(f"Output video not found: {output_path}")

    return output_filename


@app.route('/job_status/<job_id>')
def job_status(job_id):
    """Get job status"""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    response = {
        'id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
        'task_type': job['task_type'],
        'created_at': job['created_at'],
        'logs': job.get('logs', [])
    }

    if job['status'] == 'completed':
        response['output_file'] = job['output_file']
        response['download_url'] = url_for('download_video', filename=job['output_file'])
        response['completed_at'] = job.get('completed_at')

    if job['status'] == 'failed':
        response['error'] = job['error']

    return jsonify(response)


@app.route('/download/<filename>')
def download_video(filename):
    """Download generated video"""
    filepath = app.config['OUTPUT_FOLDER'] / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_file(filepath, as_attachment=True)


@app.route('/outputs/<filename>')
def view_video(filename):
    """View generated video"""
    filepath = app.config['OUTPUT_FOLDER'] / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    return send_file(filepath)


if __name__ == '__main__':
    print("="*70)
    print("MAIN PIPELINE - ROBOTICS VIDEO GENERATION SYSTEM")
    print("="*70)
    print(f"WAN 2.2 Base: {WAN_BASE}")
    print(f"Cosmos 2.5 Base: {COSMOS_BASE}")
    print(f"Qwen3-VL Extenders: {QWEN_EXTENDERS}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output Folder: {app.config['OUTPUT_FOLDER']}")
    print("="*70)
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
