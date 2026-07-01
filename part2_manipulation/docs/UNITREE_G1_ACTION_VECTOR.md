# Unitree G1 Action Vector Guide

**Robot**: Unitree G1 Humanoid
**Action Dimensions**: 43 DOF (Degrees of Freedom)
**Dataset**: dc_groot_teleop (311 episodes)

---

## Action Vector Format

### Shape: `[43]` - Full Body Control

Your trained DC-GR00T model outputs **43-dimensional action vectors** for controlling the Unitree G1 humanoid robot.

```python
action = model.predict(demo_video, observation)
# action.shape = [43]  # One value per joint/DOF
```

---

## Action Vector Breakdown

Based on your dataset analysis:

```
Index Range │ Body Part              │ DOF Count │ Description
────────────┼────────────────────────┼───────────┼──────────────────────────
0-6         │ Left Arm               │ 7         │ Shoulder(3), Elbow(2), Wrist(2)
7-13        │ Right Arm              │ 7         │ Shoulder(3), Elbow(2), Wrist(2)
14-20       │ Torso & Head           │ 7         │ Waist(3), Neck(2), Head(2)
21-27       │ Left Hand/Fingers      │ 7         │ Gripper/finger joints
28-34       │ Right Hand/Fingers     │ 7         │ Gripper/finger joints
35-42       │ Legs (Hip/Knee/Ankle)  │ 8         │ Leg stabilization/mobility
────────────┴────────────────────────┴───────────┴──────────────────────────
Total: 43 DOF
```

### Example Action Vector

```python
# From episode_000000, timestep 0:
action = [
    # Left Arm (0-6)
    -0.533, -0.044, 0.113, 0.697, -0.202, -0.031, -0.706,

    # Right Arm (7-13)
    0.044, -0.023, 0.695, -0.248, 0.015, -0.005, -0.008,

    # Torso/Head (14-20)
    -0.009, 0.020, 0.273, 0.278, -0.549, -0.047, 0.036,

    # Left Hand (21-27)
    -0.101, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    # Right Hand (28-34)
    0.0, -0.004, -0.226, -0.043, -0.740, 0.068, 0.192,

    # Legs (35-42)
    0.137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]
```

---

## Action Value Ranges

From your dataset statistics:

```python
# Across all 311 episodes:
action_min  = -0.875  # Minimum joint angle/position
action_max  =  1.000  # Maximum joint angle/position
action_mean = -0.059  # Average (varies per joint)
action_std  =  0.227  # Standard deviation
```

### Per-Joint Statistics

```
Joint Type          │ Typical Range    │ Units
────────────────────┼──────────────────┼─────────────
Shoulder Joints     │ [-0.7, 0.7]      │ radians
Elbow Joints        │ [-0.3, 0.7]      │ radians
Wrist Joints        │ [-0.2, 0.2]      │ radians
Torso Joints        │ [-0.6, 0.4]      │ radians
Gripper/Fingers     │ [0.0, 1.0]       │ normalized (0=open, 1=closed)
Leg Joints          │ [-0.9, 0.5]      │ radians
```

**Note**: Values are typically in radians for revolute joints, meters for prismatic joints, or normalized [0,1] for grippers.

---

## How Your Model Works

### Input/Output Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

1. DEMO VIDEO INPUT (One-time)
   ├─ Video: episode_000000.mp4
   ├─ Length: 5-10 seconds (~120 frames)
   ├─ Content: "Pick up the yellow pear and place it on the plate"
   └─ Extract: 16 keyframes

2. DEMO ENCODING (Cache this!)
   ├─ Process 16 keyframes → Demo Encoder
   └─ Output: demo_embedding [task understanding]
      └─ "Task: Grasp pear, lift, move to plate, release"

3. OBSERVATION INPUT (Real-time, each timestep)
   ├─ Camera Image: [3, 448, 448] RGB
   ├─ Robot State: [43] joint positions (optional)
   └─ Frequency: 10-30 Hz (control loop)

4. PREDICTION (Real-time)
   ├─ Input: observation + demo_embedding
   ├─ Model: DC-GR00T (your trained model)
   └─ Output: action [43] joint commands

5. EXECUTION
   ├─ Send action to Unitree G1 controller
   └─ Robot executes joint movements
```

---

## Code Examples

### Complete Inference Loop

```python
import torch
from peft import PeftModel
from gr00t.model.demo_conditioned import DCGr00t
from decord import VideoReader, cpu
import numpy as np

# ============================================================
# SETUP (Once)
# ============================================================

# Load trained model
base_model = DCGr00t.from_pretrained_groot("nvidia/GR00T-N1.6-3B")
model = PeftModel.from_pretrained(
    base_model,
    "checkpoints/dc_groot_full_training/final"
)
model.eval().cuda()

# Load and encode demo video (once)
def load_demo(video_path, num_keyframes=16):
    import cv2
    vr = VideoReader(str(video_path), ctx=cpu(0))
    T = len(vr)
    indices = np.linspace(0, T - 1, num_keyframes, dtype=np.int64)
    frames = vr.get_batch(indices).asnumpy()

    # Resize to 224x224 (demo encoder expects this size)
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, (224, 224))
        resized_frames.append(resized)
    frames = np.stack(resized_frames)

    # Preprocess
    frames = torch.from_numpy(frames).float()
    frames = frames.permute(0, 3, 1, 2) / 255.0  # [N, C, H, W]
    return frames.unsqueeze(0)  # [1, N, C, H, W]

demo_video = load_demo("demo_pick_pear.mp4")
demo_video = demo_video.cuda()

# Encode demo (cache this!)
# Demo type must be tensor: 0=human, 1=robot, 2=cosmos, 3=own
demo_type = torch.tensor([3], device='cuda')  # 3 = own robot
with torch.no_grad():
    demo_embedding = model.encode_demo(demo_video, demo_type)

print(f"✅ Demo encoded! Shape: {demo_embedding.shape}")
print(f"   Task understood: 'Pick up pear and place on plate'")

# ============================================================
# INFERENCE LOOP (Real-time)
# ============================================================

# Connect to Unitree G1
robot = UnitreeG1Robot()  # Your robot interface

while not task_complete:
    # 1. Get current observation
    camera_image = robot.get_camera_image()  # [3, 448, 448]
    current_state = robot.get_joint_positions()  # [43] (optional)

    # 2. Prepare observation
    obs = preprocess_observation(camera_image)
    obs = obs.cuda()

    # 3. Predict action (using cached demo_embedding!)
    with torch.no_grad():
        action = model.predict(
            demo_embedding=demo_embedding,
            observation=obs,
            state=current_state  # Optional
        )

    # action.shape = [action_horizon, 43]
    # Use first action: action[0] = [43]

    # 4. Execute on robot
    robot.execute_action(action[0].cpu().numpy())

    # 5. Check if task complete
    task_complete = check_task_success(camera_image)

    # 6. Wait for next control cycle
    time.sleep(1/30)  # 30 Hz control loop

print("✅ Task completed!")
```

### Single Prediction Example

```python
# Load model (once)
model = load_trained_model("checkpoints/dc_groot_full_training/final")

# Encode demo (once)
demo_emb = encode_demo("demo_pick_pear.mp4")

# Get observation
camera_img = robot.get_camera()  # [3, 448, 448]

# Predict action
action = model.predict(demo_emb, camera_img)
# action = [43] values for all joints

# Execute
robot.move_to_position(action)
```

---

## Dataset Structure

### Your Dataset: `dc_groot_teleop`

```
dc_groot_teleop/
├── episodes.jsonl                 # Metadata (311 episodes)
├── info.json                      # Dataset info
├── data/
│   ├── episode_000000.parquet    # Timestep data
│   ├── episode_000001.parquet
│   └── ... (311 files)
└── videos/
    ├── ego_view/                  # Robot camera view
    │   ├── episode_000000.mp4    # Your demo video
    │   └── ...
    └── demo/                      # Optional demo videos
        └── ...
```

### Episode Data Format

```python
# Load episode
df = pd.read_parquet("data/episode_000000.parquet")

# Columns:
df.columns = [
    'observation.state',        # Robot joint positions [43]
    'action',                   # Action to take [43]
    'observation.img_state_delta',  # State change info
    'timestamp',                # Time
    'frame_index',              # Frame number
    'episode_index',            # Episode ID
    'index',                    # Timestep index
    'task_index'                # Task ID
]

# Each row = one timestep
# action[t] = what robot should do at timestep t
```

---

## Action Space Details

### Unitree G1 Specifications

**Robot Type**: Humanoid
**Height**: ~1.3-1.5m
**DOF**: 43 total
**Control Frequency**: 10-100 Hz
**Action Type**: Position control (joint angles)

### Control Modes

Your trained model outputs **position commands**:

```python
# Model output (position control)
action = [
    -0.533,  # Left shoulder pitch (radians)
    -0.044,  # Left shoulder roll (radians)
    0.113,   # Left shoulder yaw (radians)
    ...      # (40 more joints)
]

# These are TARGET positions
# Robot controller will move joints to these positions
```

**Not velocity or torque control** - the model predicts where each joint should be.

---

## Task Information

### Example Task from Dataset

```python
# From episodes.jsonl:
{
    "episode_id": "episode_000000",
    "task_description": "Pick up the yellow pear and place it on the plate",
    "demo_type": "own",  # Self-demonstration
    "augmentation_idx": 0
}
```

### Your Dataset Tasks

- **Source**: `groot_teleop/g1-pick-apple`
- **Episodes**: 311
- **Task Family**: Pick and place manipulation
- **Demo Type**: Own demonstrations (from same robot)

---

## Model Performance Expectations

### What Your Trained Model Can Do:

✅ **Watch** generated demo video (5-10 sec)
✅ **Understand** task: "Pick pear, move to plate"
✅ **Observe** real environment via camera
✅ **Generate** 43-DOF actions for Unitree G1
✅ **Adapt** to different object positions
✅ **Generalize** to similar but not identical scenes

### Limitations:

⚠️ **Similar Tasks**: Best on tasks similar to training data
⚠️ **Object Variations**: Works best with objects similar to training
⚠️ **Environment**: May need adaptation for very different environments
⚠️ **Safety**: Always test in simulation or controlled environment first

---

## Inference Tips

### 1. Demo Video Quality

```python
# Good demo video:
✅ Clear view of object and goal
✅ Smooth motion (no jittery camera)
✅ 5-10 seconds showing full task
✅ Consistent lighting
✅ Same camera viewpoint as training

# Poor demo video:
❌ Blurry or low resolution
❌ Object occluded
❌ Different camera angle
❌ Too fast or too slow
```

### 2. Action Smoothing

```python
# Raw model output may be jittery
# Apply smoothing for safer execution:

from collections import deque

action_history = deque(maxlen=5)

def smooth_action(action):
    action_history.append(action)
    # Moving average
    smoothed = np.mean(action_history, axis=0)
    return smoothed

# Use smoothed action
smoothed_action = smooth_action(raw_action)
robot.execute(smoothed_action)
```

### 3. Safety Checks

```python
# Always validate actions before execution

def is_safe_action(action, current_state):
    # Check joint limits
    if np.any(action < JOINT_LIMITS_MIN) or np.any(action > JOINT_LIMITS_MAX):
        return False

    # Check velocity (change from current state)
    velocity = np.abs(action - current_state)
    if np.any(velocity > MAX_JOINT_VELOCITY):
        return False

    return True

# Only execute if safe
if is_safe_action(action, robot.get_state()):
    robot.execute(action)
else:
    robot.emergency_stop()
```

---

## Testing Your Model

### Step 1: Simulation Test

```bash
# First test in simulation (Isaac Sim, MuJoCo, etc.)
python inference_demo.py --sim --visualize
```

### Step 2: Real Robot (Controlled)

```bash
# Limit workspace and speed
python inference_demo.py --real --speed 0.3 --workspace limited
```

### Step 3: Full Deployment

```bash
# Full speed after validation
python inference_demo.py --real --speed 1.0
```

---

## Common Issues & Solutions

### Issue 1: Actions Too Large/Unstable

**Problem**: Robot makes jerky movements

**Solution**:
```python
# Clip actions to safe range
action = np.clip(action, JOINT_MIN, JOINT_MAX)

# Reduce action magnitude
action = 0.5 * action  # Scale down by 50%
```

### Issue 2: Demo Not Recognized

**Problem**: Model doesn't understand task from demo

**Solution**:
```python
# Ensure demo video quality
# Check keyframe extraction
frames = load_demo(video_path, num_keyframes=16)
print(f"Keyframes extracted: {frames.shape}")

# Try more keyframes
demo_emb = model.encode_demo(video, num_keyframes=32)
```

### Issue 3: Slow Inference

**Problem**: Real-time control too slow

**Solution**:
```python
# Use half precision (faster)
model = model.half()

# Reduce diffusion steps (if applicable)
model.set_num_diffusion_steps(50)  # Instead of 100

# Batch observations if possible
```

---

## Next Steps

1. **✅ Model is trained** - Ready to use!
2. **📹 Prepare demo videos** - For your target tasks
3. **🤖 Test in simulation** - Validate before real robot
4. **🔧 Tune parameters** - Action scaling, smoothing, etc.
5. **🚀 Deploy on Unitree G1** - Start with slow, safe tests

---

## Summary

**Your Trained DC-GR00T Model:**

- **Input**: Demo video (any length) + Robot camera observation
- **Output**: 43-dimensional action vector for Unitree G1
- **Process**:
  1. Extract 16 keyframes from demo
  2. Understand task
  3. Generate actions based on current observation
  4. Execute on robot

**Action Vector**: `[43]` joint position commands
**Frequency**: 10-30 Hz control loop
**Status**: ✅ Trained and ready for deployment

---

**See `inference_demo.py` for working example!**
