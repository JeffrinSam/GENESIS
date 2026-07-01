# Qwen3-VL Prompt Extenders

AI-powered prompt enhancement system for robotics video generation using **Qwen3.5-9B** vision-language model.

---

## How It Works

```
┌─────────────────────────────────────────┐
│  INPUT                                  │
│  • Image (robot/scene/workspace)        │
│  • Simple user prompt (10-30 words)     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Qwen3.5-9B                   │
│  • Analyzes image content               │
│  • Applies task-specific system prompt  │
│  • Generates detailed description       │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  OUTPUT                                 │
│  • Extended prompt (100-300 words)      │
│  • Model configuration (JSON)           │
└─────────────────────────────────────────┘
```

**Simple user input** → **VL model understands** → **Extended detailed prompt**

---

## The 4 Extenders

### Navigation Tasks (WAN 2.2)

#### 1. Drone Navigation
**File**: [`wan22/prompt_extender_drone.py`](wan22/prompt_extender_drone.py)

- **For**: Aerial drone flight, UAV navigation
- **Perspective**: **First-person** (camera IS the drone's eyes)
- **Style**: Cinematic, 100-200 words
- **System Prompt**: Film director approach with aerial cinematography
- **Output**: WAN 2.2 config + enhanced prompt

**Example**:
```bash
python3 wan22/prompt_extender_drone.py \
  --prompt "Flying over mountains" \
  --image aerial_view.jpg \
  --output mountain_flight
```

**Enhanced Output**:
> "Starting at 100 meters altitude, hovering above rugged mountain peaks. Snow-capped summits stretch endlessly ahead under golden morning light. Gliding forward smoothly, the rocky terrain flows steadily beneath. Banking gently to the right, the horizon tilts, revealing a deep valley. Sunlight glints off distant glacier ice..."

#### 2. Ground Robot Navigation
**File**: [`wan22/prompt_extender_ground_robot.py`](wan22/prompt_extender_ground_robot.py)

- **For**: Humanoid, wheeled, or tracked robot navigation
- **Perspective**: **First-person** (camera IS the robot's eyes)
- **Style**: Cinematic, 100-200 words
- **System Prompt**: Ground-level cinematography with locomotion patterns
- **Output**: WAN 2.2 config + enhanced prompt

**Example**:
```bash
python3 wan22/prompt_extender_ground_robot.py \
  --prompt "Navigate down corridor" \
  --image hallway.jpg \
  --output corridor_nav
```

**Enhanced Output**:
> "Starting in modern office corridor, facing down the hallway. Polished floor stretches ahead, fluorescent lights creating even illumination. Beginning to move forward with smooth rolling motion, floor flowing beneath steadily. Corridor walls sliding past on both sides. Doorway approaching ahead on the left. Turning slightly left, view rotating smoothly..."

---

### Manipulation Tasks (Cosmos 2.5)

#### 3. Bimanual UR3 Manipulation
**File**: [`cosmos25/prompt_extender_bimanual_ur3.py`](cosmos25/prompt_extender_bimanual_ur3.py)

- **For**: Dual-arm UR3 industrial manipulation
- **Style**: Physics-based temporal sequences, 150-300 words
- **System Prompt**: Physics engineer approach with contact dynamics
- **Output**: Cosmos 2.5 config + enhanced prompt
- **Image**: **Required** (Cosmos always needs initial state)

**Example**:
```bash
python3 cosmos25/prompt_extender_bimanual_ur3.py \
  --prompt "Pick up box and place on shelf" \
  --image workspace.jpg \
  --output box_placement
```

**Enhanced Output**:
> "A dual-arm UR3 robotic system mounted on a sturdy workbench in an industrial laboratory. Two blue UR3 arms with parallel jaw grippers positioned symmetrically. Between the arms rests a cardboard box (15cm × 15cm × 20cm, ~500g). Initial state (0-20%): Both arms in home position, grippers open. Approach phase (20-40%): Left arm remains stationary while right arm extends..."

#### 4. Unitree G1 Humanoid Manipulation
**File**: [`cosmos25/prompt_extender_unitree_g1.py`](cosmos25/prompt_extender_unitree_g1.py)

- **For**: Unitree G1 humanoid bimanual manipulation
- **Style**: Physics-based temporal sequences, 150-300 words
- **System Prompt**: Anthropomorphic kinematics with dexterous grasping
- **Output**: Cosmos 2.5 config + enhanced prompt
- **Image**: **Required** (Cosmos always needs initial state)

**Example**:
```bash
python3 cosmos25/prompt_extender_unitree_g1.py \
  --prompt "Pick up bottle with both hands" \
  --image kitchen.jpg \
  --output bottle_grasp
```

**Enhanced Output**:
> "A Unitree G1 humanoid robot (~1.3m tall) stands at kitchen counter. White and black chassis, anthropomorphic proportions with dual arms. On the counter: plastic water bottle (500ml, standard grip diameter). Initial state (0-20%): G1 stationary, both arms at sides, five-fingered hands open. Approach phase (20-40%): Both arms lift simultaneously..."

---

## System Prompt Architecture

Each extender uses a specialized system prompt that guides Qwen3-VL's output:

### Navigation (First-Person Cinematic)
- **Perspective Rule**: Camera IS the embodiment's eyes (first-person POV)
- **Motion Description**: "Gliding forward" NOT "drone flies"
- **World Movement**: Environment moves relative to camera
- **Style**: Cinematography (lighting, color, camera work)
- **Length**: 100-200 words

### Manipulation (Physics Temporal)
- **Scene Setup**: Robot configuration, object properties, spatial relationships
- **Temporal Structure**: Initial → Approach → Grasp → Manipulate → Complete
- **Physics Details**: Joint movements, contact dynamics, forces, collision avoidance
- **Style**: Engineering specification (precise, physical, causal)
- **Length**: 150-300 words

---

## Output Files

Each extender creates two files in [`outputs/`](outputs/):

### 1. Enhanced Prompt (`*_prompt.txt`)
Human-readable text file:
```
Generated: 2025-12-27 14:32:15

Enhanced Prompt:
[100-300 word detailed description]

Negative Prompt:
[Task-specific exclusions]
```

### 2. Model Configuration (`*_wan_config.json` or `*_cosmos_config.json`)

**WAN 2.2 Config**:
```json
{
  "task": "ti2v-5B",
  "size": "1280*720",
  "frame_num": 61,
  "prompt": "Enhanced prompt...",
  "negative_prompt": "...",
  "sample_steps": 30,
  "sample_guide_scale": 7.5,
  "save_file": "output.mp4"
}
```

**Cosmos 2.5 Config**:
```json
{
  "inference_type": "image2world",
  "name": "job_name",
  "input_path": "/absolute/path/to/image.jpg",
  "prompt": "Enhanced prompt...",
  "negative_prompt": "...",
  "num_output_frames": 77,
  "resolution": "432,768",
  "seed": 42,
  "guidance": 7
}
```

---

## Critical: First-Person Navigation

**IMPORTANT**: Navigation tasks (drone, ground) use **FIRST-PERSON perspective**.

### ✅ Correct (First-Person)
```
"Gliding forward over forest. Treetops rushing beneath. Banking right,
revealing river ahead. Horizon tilting as view rotates."
```
- Camera IS the embodiment's eyes
- World moves relative to camera
- Viewer sees through robot/drone's perspective

### ❌ Wrong (Third-Person)
```
"A drone flies over forest. The drone's rotors spin. The drone banks
right. A river appears below the drone."
```
- Describes drone as external object
- Like filming the drone from outside
- NOT what we want

**Why**: First-person generates embodied POV videos, third-person generates external views.

---

## Usage Examples

### Drone with Image
```bash
python3 wan22/prompt_extender_drone.py \
  --image aerial_scene.jpg \
  --prompt "Flying over coastal cliffs at sunset" \
  --output coastal_flight

# Check output
cat outputs/coastal_flight_prompt.txt
```

### Ground Robot without Image (generates generic scene)
```bash
python3 wan22/prompt_extender_ground_robot.py \
  --prompt "Navigate warehouse aisles" \
  --output warehouse_nav
```

### UR3 Manipulation (image required)
```bash
python3 cosmos25/prompt_extender_bimanual_ur3.py \
  --image workspace_setup.jpg \
  --prompt "Assemble two parts together" \
  --output assembly
```

### G1 Manipulation (image required)
```bash
python3 cosmos25/prompt_extender_unitree_g1.py \
  --image kitchen_scene.jpg \
  --prompt "Pour water into cup" \
  --output pouring
```

---

## Integration with Video Generation

### WAN 2.2 (Navigation)
```bash
cd $WAN_ROOT

# Use enhanced prompt from outputs/
python3 generate.py \
  --task ti2v-5B \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --prompt "$(cat /path/to/outputs/mountain_flight_prompt.txt)" \
  --image aerial_view.jpg \
  --size 1280*720 \
  --frame_num 61 \
  --save_file mountain_flight.mp4
```

### Cosmos 2.5 (Manipulation)
```bash
cd $COSMOS_ROOT

# Use enhanced prompt from outputs/
python3 inference_i2w.py \
  --checkpoint_dir ./checkpoints/Cosmos-2.5-Predict-2B \
  --input_path workspace_setup.jpg \
  --prompt "$(cat /path/to/outputs/assembly_prompt.txt)" \
  --num_output_frames 77 \
  --guidance 7
```

---

## Integration with Manual Pipeline

The manual pipeline ([`part1_generation/mainpipeline/`](../../Manuelpipeline/)) automatically uses these extenders:

```bash
cd part1_generation/mainpipeline

# Pipeline calls appropriate extender automatically
python3 complete_pipeline.py --task drone \
  --image aerial.jpg \
  --prompt "Flying over mountains" \
  --enhance \
  --output drone.mp4
```

**What happens**:
1. Pipeline detects task type (drone)
2. Calls `wan22/prompt_extender_drone.py` automatically
3. Uses enhanced prompt for WAN 2.2 generation
4. Outputs video with first-person perspective

---

## Directory Structure

```
prompt_extenders/
├── README.md                          # This file
│
├── wan22/                             # WAN 2.2 extenders
│   ├── prompt_extender_drone.py       # Aerial navigation (first-person)
│   └── prompt_extender_ground_robot.py # Ground navigation (first-person)
│
├── cosmos25/                          # Cosmos 2.5 extenders
│   ├── prompt_extender_bimanual_ur3.py # UR3 manipulation (physics)
│   └── prompt_extender_unitree_g1.py   # G1 manipulation (physics)
│
└── outputs/                           # Generated prompts and configs
    ├── *_prompt.txt                   # Enhanced prompts
    ├── *_wan_config.json              # WAN 2.2 configs
    └── *_cosmos_config.json           # Cosmos 2.5 configs
```

---

## Requirements

- **Python**: 3.8+
- **PyTorch**: 2.11.0+ with CUDA 12.8
- **Transformers**: Latest (Hugging Face)
- **Qwen3.5-9B**: Model at `$QWEN_MODEL_PATH` (set to local path or use HF model ID `Qwen/Qwen3.5-9B`)
- **VRAM**: 16GB+ recommended (RTX 4090/5090, A6000)
- **Environment**: `conda activate wan2.2`

---

## Common Parameters

All extenders support:

```bash
--prompt "User task description"  # Required: Simple 10-30 word description
--image path/to/image.jpg         # Optional for WAN, Required for Cosmos
--output job_name                 # Output filename prefix
```

**Note**: Cosmos extenders (UR3, G1) always require `--image` since Cosmos 2.5 uses image2world mode.

---

## Performance

- **Qwen3-VL Inference**: 5-15 seconds on RTX 5090
- **VRAM Usage**: ~16GB (bfloat16 precision)
- **Generation Quality**: Enhanced prompts significantly improve video coherence

---

## Task-Specific Negative Prompts

Each extender includes optimized negative prompts:

- **Drone**: Excludes ground vehicles, manipulation, indoor scenes, walking, running
- **Ground Robot**: Excludes flying, aerial views, manipulation tasks, hovering
- **UR3**: Excludes flying, navigation, single arm, humanoid walking
- **G1**: Excludes flying, wheeled robots, industrial arms without humanoid body

---

## Quick Reference

| Task | Extender | Image | Perspective | Output Length | Model |
|------|----------|-------|-------------|---------------|-------|
| Drone | `wan22/prompt_extender_drone.py` | Optional | First-person | 100-200 words | WAN 2.2 |
| Ground | `wan22/prompt_extender_ground_robot.py` | Optional | First-person | 100-200 words | WAN 2.2 |
| UR3 | `cosmos25/prompt_extender_bimanual_ur3.py` | **Required** | Physics | 150-300 words | Cosmos 2.5 |
| G1 | `cosmos25/prompt_extender_unitree_g1.py` | **Required** | Physics | 150-300 words | Cosmos 2.5 |

---

**Last Updated**: December 27, 2025
**Status**: Production ready - All 4 extenders operational
**Mode**: Image+text for all tasks (navigation + manipulation)
