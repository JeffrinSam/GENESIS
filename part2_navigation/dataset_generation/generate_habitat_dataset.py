"""
Flow-Constrained Video-to-Action Model for Cross-Embodiment Navigation

Author: Jeffrin Sam
Institution: Skoltech
Year: 2025
License: MIT

Description: Generate synthetic navigation datasets using Habitat-Sim
"""

import os
os.environ['HABITAT_SIM_GPU_DEVICE_ID'] = '-1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.6'

import habitat_sim
import numpy as np
import cv2
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import random
from typing import Tuple, Dict, List


class NavigationDataGenerator:
    """Generate navigation clips using Habitat-Sim."""

    def __init__(
        self,
        scene_dir: str = "data/scene_datasets",
        resolution: Tuple[int, int] = (720, 1280),
        fps: int = 16,
        embodiment: str = "wheeled",
        max_linear_vel: float = 2.0,
        max_angular_vel: float = 1.5,
        headless: bool = False
    ):
        """
        Initialize dataset generator.

        Args:
            scene_dir: Path to Habitat scenes
            resolution: (height, width) for video frames
            fps: Frames per second
            embodiment: Robot type (wheeled, legged, aerial, humanoid)
            max_linear_vel: Maximum linear velocity (m/s)
            max_angular_vel: Maximum angular velocity (rad/s)
            headless: Run in headless mode (no display)
        """
        self.scene_dir = Path(scene_dir)
        self.resolution = resolution
        self.fps = fps
        self.embodiment = embodiment
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.headless = headless

        # Action dimensions based on embodiment
        self.action_dim = 4 if embodiment == "aerial" else 3

        # Load available scenes
        self.scenes = self._find_scenes()
        print(f"Found {len(self.scenes)} scenes")

        # Initialize simulator
        self.sim = None

    def _find_scenes(self) -> List[str]:
        """Find available Habitat scenes."""
        scenes = []
        scene_dir = Path(self.scene_dir)
        
        # Look for GLB files in habitat-test-scenes subdirectory
        test_scenes_dir = Path("data/scene_datasets/habitat-test-scenes")
        if test_scenes_dir.exists():
            for glb_file in test_scenes_dir.rglob("*.glb"):
                scenes.append(str(glb_file))
                print(f"Found scene: {glb_file}")
        
        if not scenes:
            print("No GLB scenes found! Download with:")
            print("  python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes")

        return scenes

    def _create_simulator(self, scene_path: str) -> habitat_sim.Simulator:
        """Create Habitat simulator for a scene."""
        # Create simulator config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = str(scene_path)
        backend_cfg.enable_physics = False
        backend_cfg.gpu_device_id = -1  # Use CPU rendering to avoid GPU shader issues
        backend_cfg.enable_gfx_replay_save = False
        backend_cfg.enable_gfx_replay_save = False
        # Force CPU rendering to avoid shader issues
        backend_cfg.create_renderer = True

        # Create sensor specs
        sensor_specs = []

        # RGB camera - working minimal configuration
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [320, 240]  # Small resolution that works
        rgb_sensor.position = [0.0, 1.5, 0.0]
        rgb_sensor.orientation = [0.0, 0.0, 0.0]
        rgb_sensor.hfov = 45  # Narrow FOV that works
        sensor_specs.append(rgb_sensor)

        # Agent config
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }

        # Create simulator
        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        return sim

    def _random_navigable_position(self, sim: habitat_sim.Simulator) -> np.ndarray:
        """Get a random navigable position in the scene."""
        # Try to find navigable position
        for _ in range(100):
            pos = sim.pathfinder.get_random_navigable_point()
            if sim.pathfinder.is_navigable(pos):
                return pos

        # Fallback
        return np.array([0.0, 0.0, 0.0])

    def _sample_action(self) -> np.ndarray:
        """Sample a random navigation action."""
        if self.embodiment == "aerial":
            # (vx, vy, vz, yaw)
            vx = np.random.uniform(-self.max_linear_vel, self.max_linear_vel)
            vy = np.random.uniform(-self.max_linear_vel * 0.5, self.max_linear_vel * 0.5)
            vz = np.random.uniform(-1.0, 1.0)  # Slower vertical
            yaw = np.random.uniform(-self.max_angular_vel, self.max_angular_vel)
            return np.array([vx, vy, vz, yaw])
        else:
            # (vx, vy, yaw)
            vx = np.random.uniform(-self.max_linear_vel, self.max_linear_vel)
            vy = np.random.uniform(-self.max_linear_vel * 0.3, self.max_linear_vel * 0.3)
            yaw = np.random.uniform(-self.max_angular_vel, self.max_angular_vel)
            return np.array([vx, vy, yaw])

    def _apply_action(
        self,
        sim: habitat_sim.Simulator,
        action: np.ndarray,
        dt: float = 1/16
    ) -> np.ndarray:
        """
        Apply continuous velocity action to simulator.

        Args:
            sim: Habitat simulator
            action: Velocity command (vx, vy, yaw) or (vx, vy, vz, yaw)
            dt: Time step

        Returns:
            actual_action: Action that was actually applied
        """
        agent = sim.get_agent(0)
        state = agent.get_state()

        # Extract velocities
        vx = action[0]
        vy = action[1] if len(action) > 2 else 0.0
        yaw = action[-1]

        # Update orientation
        new_rotation = state.rotation * habitat_sim.utils.quat_from_angle_axis(
            yaw * dt, np.array([0, 1, 0])
        )

        # Update position
        forward = habitat_sim.utils.quat_rotate_vector(
            new_rotation, np.array([0, 0, -1])
        )
        right = habitat_sim.utils.quat_rotate_vector(
            new_rotation, np.array([1, 0, 0])
        )

        new_position = state.position + (forward * vx + right * vy) * dt

        # Check if new position is navigable
        if sim.pathfinder.is_navigable(new_position):
            state.position = new_position
            state.rotation = new_rotation
            agent.set_state(state)
            return action
        else:
            # Collision - return zero action
            return np.zeros_like(action)

    def generate_clip(
        self,
        duration: float = 8.0,
        scene_idx: int = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a single navigation clip.

        Args:
            duration: Clip duration in seconds
            scene_idx: Index of scene to use (random if None)

        Returns:
            frames: [T, H, W, 3] - RGB video frames (0-255)
            actions: [T, action_dim] - Velocity actions
            metadata: Dict with clip metadata
        """
        # Select scene - use the correct path
        scene_path = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"

        # Create simulator if needed
        if self.sim is None or self.current_scene != scene_path:
            if self.sim is not None:
                self.sim.close()
            self.sim = self._create_simulator(scene_path)
            self.current_scene = scene_path

        # Reset to random position
        start_pos = self._random_navigable_position(self.sim)
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        state.position = start_pos
        state.rotation = habitat_sim.utils.quat_from_angle_axis(
            np.random.uniform(0, 2 * np.pi), np.array([0, 1, 0])
        )
        agent.set_state(state)

        # Generate trajectory
        num_frames = int(duration * self.fps)
        frames = []
        actions = []

        dt = 1.0 / self.fps
        collision_count = 0

        for t in range(num_frames):
            # Get observation
            obs = self.sim.get_sensor_observations()
            frame = obs["color_sensor"]  # [H, W, 4] RGBA
            
            # Convert RGBA to RGB
            frame_rgb = frame[:, :, :3]
            
            # Resize to target resolution if needed
            if frame_rgb.shape[:2] != self.resolution[::-1]:
                frame_rgb = cv2.resize(frame_rgb, (self.resolution[1], self.resolution[0]))
            
            frames.append(frame_rgb)

            # Sample action
            action = self._sample_action()

            # Apply action
            actual_action = self._apply_action(self.sim, action, dt)
            actions.append(actual_action)

            # Check for collision
            if np.allclose(actual_action, 0.0) and not np.allclose(action, 0.0):
                collision_count += 1

        # Convert to numpy arrays
        frames = np.array(frames, dtype=np.uint8)  # [T, H, W, 3]
        actions = np.array(actions, dtype=np.float32)  # [T, action_dim]

        # Metadata
        metadata = {
            "scene": str(Path(scene_path).relative_to(self.scene_dir)),
            "duration": duration,
            "fps": self.fps,
            "num_frames": num_frames,
            "resolution": list(self.resolution),
            "embodiment": self.embodiment,
            "collisions": collision_count,
            "start_position": start_pos.tolist()
        }

        return frames, actions, metadata

    def generate_dataset(
        self,
        num_clips: int,
        output_dir: str,
        duration: float = 8.0,
        min_quality: float = 0.7  # Minimum quality (collision ratio)
    ) -> Dict:
        """
        Generate complete dataset.

        Args:
            num_clips: Number of clips to generate
            output_dir: Output directory
            duration: Clip duration in seconds
            min_quality: Minimum quality threshold (0-1)

        Returns:
            dataset_metadata: Complete dataset metadata
        """
        output_path = Path(output_dir)
        video_dir = output_path / "videos"
        action_dir = output_path / "actions"

        # Create directories
        video_dir.mkdir(parents=True, exist_ok=True)
        action_dir.mkdir(parents=True, exist_ok=True)

        dataset_metadata = {}
        generated_count = 0
        attempts = 0

        pbar = tqdm(total=num_clips, desc="Generating clips")

        while generated_count < num_clips and attempts < num_clips * 3:
            attempts += 1

            try:
                # Generate clip
                frames, actions, metadata = self.generate_clip(duration)

                # Quality check: reject clips with too many collisions
                collision_ratio = metadata["collisions"] / metadata["num_frames"]
                if collision_ratio > (1 - min_quality):
                    continue  # Skip low-quality clip

                # Save video
                clip_id = f"habitat_clip_{generated_count:04d}"
                video_path = video_dir / f"{clip_id}.mp4"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    self.fps,
                    (self.resolution[1], self.resolution[0])
                )

                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()

                # Save actions
                action_path = action_dir / f"{clip_id}.npy"
                np.save(action_path, actions)

                # Update metadata
                dataset_metadata[clip_id] = {
                    "embodiment": self.embodiment,
                    "split": "train",  # Will be updated later
                    "fps": self.fps,
                    "duration": duration,
                    "resolution": list(self.resolution),
                    "source": "habitat_sim",
                    "num_frames": metadata["num_frames"],
                    "environment": "indoor",
                    "scene": metadata["scene"],
                    "collisions": metadata["collisions"]
                }

                generated_count += 1
                pbar.update(1)

            except Exception as e:
                print(f"\nWarning: Failed to generate clip: {e}")
                continue

        pbar.close()

        # Save metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)

        print(f"\nGenerated {generated_count} clips in {attempts} attempts")
        print(f"Success rate: {generated_count/attempts*100:.1f}%")
        print(f"Dataset saved to: {output_path}")

        # Cleanup
        if self.sim is not None:
            self.sim.close()

        return dataset_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate Habitat navigation dataset")

    # Dataset parameters
    parser.add_argument("--num_clips", type=int, default=100, help="Number of clips to generate")
    parser.add_argument("--duration", type=float, default=8.0, help="Clip duration (seconds)")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument("--output_dir", type=str, default="../dataset", help="Output directory")

    # Scene parameters
    parser.add_argument("--scene_dir", type=str, default="data/scene_datasets",
                        help="Path to Habitat scenes")

    # Robot parameters
    parser.add_argument("--embodiment", type=str, default="wheeled",
                        choices=["wheeled", "legged", "aerial", "humanoid"],
                        help="Robot embodiment type")
    parser.add_argument("--max_linear_vel", type=float, default=2.0,
                        help="Max linear velocity (m/s)")
    parser.add_argument("--max_angular_vel", type=float, default=1.5,
                        help="Max angular velocity (rad/s)")

    # Video parameters
    parser.add_argument("--resolution", type=str, default="720p",
                        choices=["480p", "720p", "1080p"],
                        help="Video resolution")

    # Quality parameters
    parser.add_argument("--min_quality", type=float, default=0.7,
                        help="Minimum quality threshold (0-1)")

    # Other
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse resolution
    resolution_map = {
        "480p": (480, 854),
        "720p": (720, 1280),
        "1080p": (1080, 1920)
    }
    resolution = resolution_map[args.resolution]

    # Create generator
    print("Initializing Habitat dataset generator...")
    generator = NavigationDataGenerator(
        scene_dir=args.scene_dir,
        resolution=resolution,
        fps=args.fps,
        embodiment=args.embodiment,
        max_linear_vel=args.max_linear_vel,
        max_angular_vel=args.max_angular_vel,
        headless=args.headless
    )

    # Generate dataset
    print(f"\nGenerating {args.num_clips} clips...")
    print(f"Duration: {args.duration}s @ {args.fps}fps")
    print(f"Resolution: {resolution}")
    print(f"Embodiment: {args.embodiment}")
    print(f"Output: {args.output_dir}\n")

    metadata = generator.generate_dataset(
        num_clips=args.num_clips,
        output_dir=args.output_dir,
        duration=args.duration,
        min_quality=args.min_quality
    )

    print("\n✓ Dataset generation complete!")
    print(f"Total clips: {len(metadata)}")
    print(f"Location: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Verify: python verify_dataset.py --dataset_dir", args.output_dir)
    print("  2. Split: python split_dataset.py --dataset_dir", args.output_dir)
    print("  3. Train: cd ../flow_constrained && python training/train.py")


if __name__ == "__main__":
    main()
