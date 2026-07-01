# G1 Humanoid Dataset Collector

Collects walking episodes of the Unitree G1 humanoid robot in Isaac Sim. The robot walks from origin to a goal table (~5m away) in a warehouse, stops when it arrives, and records multi-view camera frames + trajectory data.

## Pipeline

1. **Velocity commands** are generated (or loaded from CSV): ramp-up -> cruise -> brake -> stop
2. **WBC controller** (stand.onnx + walk.onnx) converts velocity commands to joint position targets
3. **Heading correction** steers the robot toward the goal (compensates for WBC drift)
4. **Proximity stopping** brakes when close to the table and stops within 0.4m
5. **Three cameras** capture frames every control step:
   - **Ego cam** (POV): Mounted on head_link, looks down at hands (Arena-style, for dataset training)
   - **FPV cam**: Eye-level forward view (human-like first-person perspective)
   - **Third-person cam**: Follows behind the robot (for proof/verification videos)

## Files

```
collect_dataset.py   # Main collection script (single-pass: physics + capture)
run_collect.sh       # Launcher (activates env_isaaclab, sets PYTHONPATH)
proof/               # Recorded episodes
```

## Dependencies

- **Isaac Sim** + **Isaac Lab** (set `ISAAC_SIM_PYTHON` in `.env`)
- **IsaacLab-Arena** repo (`$ISAACLAB_ARENA_DIR`) — on PYTHONPATH
- **Digitaltwin** (`$DIGITALTWIN_DIR`) — WBC controller, velocity sources, env config
  - `models/stand.onnx` — standing policy
  - `models/walk.onnx` — walking policy
  - `wbc_controller.py` — G1WBCController
  - `velocity_sources.py` — CSVVelocitySource
  - `g1_env.py` — G1FlatEnvCfg (robot config, actuators, physics)

## Usage

### Basic run (auto-generates walk-forward velocity CSV)

```bash
cd simulator/humanoid
bash run_collect.sh --output-dir proof/episode_001 --headless --enable_cameras
```

### Custom velocity CSV

```bash
bash run_collect.sh --csv-path my_velocities.csv --output-dir proof/episode_001 --headless --enable_cameras
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | (required) | Where to save frames, trajectory, metadata |
| `--headless` | off | Run without GUI (required for servers) |
| `--enable_cameras` | off | Enable camera rendering (required for frames) |
| `--csv-path` | auto-gen | Custom velocity CSV (columns: `frame,vx_m_s,vy_m_s,yaw_rate_rad_s`) |
| `--walk-duration` | 15.0 | Duration in seconds (when auto-generating CSV) |
| `--max-speed` | 0.3 | Max forward speed m/s |
| `--fps` | 16 | Frame capture rate |
| `--res` | 720 | Camera resolution (square, pixels) |
| `--env` | warehouse | Environment (`flat` or `warehouse`) |

### Encode MP4 from frames

```bash
ffmpeg -y -framerate 16 -i proof/episode_001/frames_fpv/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 proof/episode_001/fpv_video.mp4

ffmpeg -y -framerate 16 -i proof/episode_001/frames_pov/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 proof/episode_001/pov_video.mp4

ffmpeg -y -framerate 16 -i proof/episode_001/frames_third_person/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 proof/episode_001/third_person_video.mp4
```

## Output Structure

Each episode produces:

```
proof/episode_NNN/
  frames_pov/           # Ego camera PNGs (720x720)
  frames_fpv/           # FPV camera PNGs (720x720)
  frames_third_person/  # Third-person camera PNGs (720x720)
  velocity_commands.csv # Input velocity commands
  trajectory.csv        # Robot state per frame (pos, heading, vel)
  joint_states.npy      # Joint positions array (N_frames x 43)
  metadata.json         # Episode info (start/end pos, frames, etc.)
  start_image.png       # First ego frame
  goal_image.png        # Last ego frame
```

## Key Parameters (in collect_dataset.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CONTROL_HZ` | 30 | Physics control rate (60Hz sim / decimation=2) |
| `HEADING_KP` | 3.0 | Proportional gain for yaw correction toward goal |
| `BRAKE_DIST` | 1.5m | Distance from goal to start braking |
| `STOP_DIST` | 0.4m | Distance from goal to fully stop |
| `STAND_FRAMES` | 32 | Frames of standing recorded after arrival (~2s) |
| `GOAL_X, GOAL_Y` | 5.0, 0.0 | Table/goal position in world frame |

## How It Works

The script uses a **single-pass** approach — physics simulation and frame capture happen in the same loop. This avoids the "flying robot" problem that occurs when rendering is done in a separate pass (extra `simulation_app.update()` calls add uncontrolled physics steps).

Cameras are **Isaac Lab native sensors** (`TiledCameraCfg`) integrated into the scene config. When `env.step()` is called, it automatically renders and populates camera data — no extra physics steps needed.

The WBC (Whole Body Controller) has an intrinsic heading drift of ~0.5 rad/s. A proportional controller corrects heading toward the goal each step, keeping the robot on course.
