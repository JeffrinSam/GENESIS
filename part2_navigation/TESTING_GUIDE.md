# Testing Guide

**Three models, one guide.** Pick the section for your robot type.

---

## V2 — General / Wheeled (production)

```bash
conda activate flowdit_v2_py310
cd flow_constrained_v2
```

### Sanity check (30 seconds)
```bash
python test_inference.py --checkpoint checkpoints/best.pth
# Expected: MSE ~0.046, 8 actions predicted per sample
```

### Single inference
```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --goal_video path/to/reference.mp4 \
    --current_obs path/to/frame.jpg \
    --output actions.npy
```

### Closed-loop with webcam
```bash
python robot_navigation.py \
    --checkpoint checkpoints/best.pth \
    --goal_video reference.mp4 \
    --camera 0 --control_hz 2.0
```

### Real robot integration pattern
```python
from inference import load_model, predict_actions

model = load_model("checkpoints/best.pth", device="cuda")

# Replace these two functions with your robot's API
def get_frame(): ...          # → np.ndarray (H, W, 3)
def send_velocity(vx, vy, yaw): ...

while not goal_reached:
    actions = predict_actions(model, goal_video, get_frame())
    # actions.shape = [8, 3] = [[vx, vy, yaw], ...]
    for vx, vy, yaw in actions[:3]:      # execute first 3, then replan
        send_velocity(vx, vy, yaw)
        time.sleep(1/6)
```

---

## V3 — Unitree G1 humanoid

```bash
cd flowdit_v3_humanoid_inference
source .venv/bin/activate
```

### Run inference on episode frames
```bash
python run_inference.py \
    --checkpoint checkpoints/flowdit_v3_humanoid_best.pt \
    --video unitree_data/unitree_g1.LMPnPAppleToPlateDC/<episode>/frames/ \
    --instruction "humanoid robot navigate to goal"
# Output: vx vy vz yaw (velocities) + x y z waypoints x16
```

---

## V1 — Legacy baseline (research only)

```bash
conda activate flowdit_v2_py310
cd flow_constrained
```

```python
from inference.policy import NavigationPolicy
import cv2

policy = NavigationPolicy(
    checkpoint_path='checkpoints/wheeled/final_model.pth',
    embodiment='wheeled', device='cuda'
)

cap = cv2.VideoCapture('path/to/video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    action = policy.predict(frame)
    print(f"vx={action[0]:.3f}  vy={action[1]:.3f}  yaw={action[2]:.3f}")
cap.release()
```

---

## Choosing the right model

| Situation | Use |
|-----------|-----|
| Wheeled robot, real deployment | **V2** |
| Unitree G1 humanoid | **V3** |
| Comparing to a simple baseline | **V1** |
| Need to retrain from scratch | **V2** (has training code) |
| Inference-only, humanoid | **V3** (fastest setup, .venv ready) |

---

## Full pipeline (Part 1 → Part 2)

```
ClaudeOpusBrain (Part 1)
    → generates reference.mp4 from (image + text prompt)

FlowDiT V2 (Part 2)
    → takes reference.mp4 + live camera
    → outputs [vx, vy, yaw] at 2 Hz

Robot executes actions
```

```bash
# Part 1 (in wan2.2 env)
cd Part1/Claudeopusbrain
python run_self_tuning.py --task "go to the table" --task-type g1 --image workspace.jpg

# Part 2 (in flowdit_v2_py310 env)
cd flow_constrained_v2
python robot_navigation.py \
    --checkpoint checkpoints/best.pth \
    --goal_video ../Part1/output/reference.mp4 \
    --camera 0
```
