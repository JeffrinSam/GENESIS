# FlowDiT V2 - Goal-Conditioned Navigation Inference Guide

## 🎯 Overview

Your trained FlowDiT V2 model performs **goal-conditioned navigation**:
- **Input 1**: Reference video (from video generation model - shows where to go)
- **Input 2**: Current observation (from robot camera - where robot is now)
- **Output**: Actions (vx, vy, yaw_rate) for next 8 timesteps

## 📋 Quick Start

### 1. Test Inference (Verify Model Works)

```bash
cd flow_constrained_v2
conda activate genesis-navigation

# Test with dataset sample
python test_inference.py --checkpoint checkpoints/best.pth
```

**Expected Output:**
```
✓ Predicted 8 action steps
MSE: 0.045 (good!)
```

---

## 🚀 Deployment Workflow

### **Full Pipeline (Video Gen → Your Model → Robot)**

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Generate Reference Video (ONE TIME)            │
├─────────────────────────────────────────────────────────┤
│ User: Image + "go to the table"                        │
│   ↓                                                     │
│ Video Gen Model → reference_video.mp4 (3-15 sec)       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 2: Closed-Loop Navigation (CONTINUOUS)            │
├─────────────────────────────────────────────────────────┤
│ Loop @ 2 Hz:                                            │
│   1. Robot Camera → current_frame.jpg                   │
│   2. FlowDiT Model:                                     │
│      - Input: reference_video + current_frame           │
│      - Output: [8 actions]                              │
│   3. Execute first 3 actions                            │
│   4. Repeat (get new frame, predict, execute...)        │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Usage Examples

### Example 1: Single Inference (Testing)

```bash
# You have:
# - reference.mp4 (from video gen model)
# - current_obs.jpg (from robot camera)

python inference.py \
    --checkpoint checkpoints/best.pth \
    --goal_video reference.mp4 \
    --current_obs current_obs.jpg \
    --output actions.npy \
    --visualize
```

**Output:**
- `actions.npy` - Predicted actions [8, 3]
- `visualization.jpg` - Visual summary

### Example 2: Closed-Loop Navigation (Simulation)

```bash
# Simulate continuous navigation
python robot_navigation.py \
    --checkpoint checkpoints/best.pth \
    --goal_video reference.mp4 \
    --camera 0 \
    --control_hz 2.0 \
    --max_steps 100
```

**What it does:**
- Captures frames from webcam (camera 0)
- Predicts actions every 0.5 seconds (2 Hz)
- Simulates action execution
- Repeats until goal reached or max steps

---

## 🤖 Real Robot Integration

### Python API

```python
import torch
import numpy as np
from inference import load_model, load_video, load_image

# 1. Load model (ONCE at startup)
model = load_model("checkpoints/best.pth", device="cuda")

# 2. Load reference video (ONCE per task)
reference_video = load_video("reference.mp4")  # From video gen model

# 3. Navigation loop (CONTINUOUS)
while not goal_reached:
    # Get current observation
    current_frame = robot.camera.capture()  # Your robot's camera API
    current_obs = preprocess_image(current_frame)  # Resize to 224x224, normalize

    # Predict actions
    actions = model.predict(reference_video, current_obs)
    # actions.shape = [8, 3] = [[vx, vy, yaw], ...]

    # Execute first 3 actions
    for i in range(3):
        vx, vy, yaw = actions[i]
        robot.send_velocity_command(vx, vy, yaw)  # Your robot's control API
        time.sleep(1/16)  # 16 Hz execution

    # Check if goal reached
    if distance_to_goal < threshold:
        break
```

### Key Functions to Replace

In `robot_navigation.py`, replace these simulation functions:

```python
# Replace this:
def get_current_observation(self, camera_id=0):
    cap = cv2.VideoCapture(camera_id)  # Webcam
    ...

# With your robot's camera:
def get_current_observation(self):
    frame = self.robot.camera.capture()
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255.0
    return frame
```

```python
# Replace this:
def execute_actions(self, actions):
    print(f"Simulated execution: {actions}")

# With your robot's control:
def execute_actions(self, actions):
    for vx, vy, yaw in actions:
        self.robot.send_velocity_command(vx, vy, yaw)
        time.sleep(1/16)  # 16 Hz
```

---

## 📊 Model Performance

**Test Results:**
- Checkpoint: `best.pth` (epoch 67)
- MSE: 0.046 (on test sample)
- Inference time: ~50 ms on GPU
- Control frequency: 2 Hz achievable

**Action Outputs:**
- `vx`: Forward velocity (m/s) - typically 0.5-0.7
- `vy`: Lateral velocity (m/s) - typically near 0
- `yaw`: Angular velocity (rad/s) - typically -0.7 to +0.7

---

## 🔧 Advanced Usage

### Custom Control Frequency

```bash
# Faster control (4 Hz)
python robot_navigation.py \
    --checkpoint checkpoints/best.pth \
    --goal_video reference.mp4 \
    --control_hz 4.0 \
    --execute_steps 2
```

### Batch Inference (Multiple Observations)

```python
# Process multiple observations in parallel
batch_obs = [obs1, obs2, obs3, ...]  # List of images
batch_obs = torch.stack([torch.from_numpy(o) for o in batch_obs])

# Batch forward pass
batch_actions = model(
    goal_video.unsqueeze(0).repeat(len(batch_obs), 1, 1, 1, 1),
    batch_obs,
    actions_gt=None
)
```

### Goal Video from Video Generation Model

Your video generation model should output:
- **Duration**: 3-15 seconds (48-240 frames @ 16 fps)
- **Resolution**: Any (will be resized to 224x224)
- **Format**: MP4, AVI, or any OpenCV-readable format
- **Content**: Show the path from starting point to destination

Example workflow:
```python
# User input
user_image = capture_starting_position()
prompt = "go to the table"

# Generate reference video
reference_video = video_gen_model.generate(
    image=user_image,
    prompt=prompt,
    duration=5.0,  # 5 seconds
    fps=16
)

# Save and use
reference_video.save("reference.mp4")
```

---

## 📁 File Structure

```
flow_constrained_v2/
├── checkpoints/
│   ├── best.pth           ← Best model (use this!)
│   ├── latest.pth         ← Final model (epoch 70+)
│   └── epoch_*.pth        ← Intermediate checkpoints
├── models/
│   └── flowdit_production.py  ← Model definition
├── inference.py           ← Single inference script
├── robot_navigation.py    ← Closed-loop navigation
├── test_inference.py      ← Test with dataset
└── INFERENCE_GUIDE.md     ← This file
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Use CPU or reduce batch size
```bash
python inference.py --device cpu ...
```

### Issue: Actions seem incorrect
**Check:**
1. Reference video shows the correct path?
2. Current observation is preprocessed correctly? (224x224, normalized to [0,1])
3. Using `best.pth` checkpoint?

### Issue: Slow inference
**Speed up:**
1. Use GPU: `--device cuda`
2. Reduce diffusion steps in config (default: 10 inference steps)
3. Batch multiple observations together

---

## 📞 Next Steps

1. **Test with your video gen model:**
   - Generate reference videos
   - Test inference with them

2. **Integrate with robot:**
   - Adapt `robot_navigation.py` to your robot's API
   - Test in simulation first

3. **Tune for your robot:**
   - Adjust control frequency
   - Scale action values if needed
   - Add safety checks (collision detection, bounds checking)

4. **Evaluate performance:**
   - Measure success rate
   - Track navigation errors
   - Log actions for analysis

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{flowdit_v2_2026,
  title={FlowDiT V2: Goal-Conditioned Navigation with Diffusion Transformers},
  author={Jeffrin Sam},
  institution={Skoltech},
  year={2026}
}
```

---

**Questions?** Check the code comments or test scripts for more examples!
