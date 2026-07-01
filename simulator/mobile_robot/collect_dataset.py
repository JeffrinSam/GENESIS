#!/usr/bin/env python3
"""
Dataset collector: Jetbot mobile robot navigates to a goal pose using REAL dynamics.

Uses Isaac Sim's WheeledRobot + DifferentialController + WheelBasePoseController
for proper physics-based wheel rotation and differential drive dynamics.

Usage:
  ./run_collect.sh --output-dir proof/episode_003 --headless --enable_cameras
  ./run_collect.sh --output-dir proof/episode_003 --headless --enable_cameras \
    --env hospital --table-x 4.0 --table-y 0.0
"""

import sys
import os
import csv
import json
import math
import pathlib
import argparse
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mobile robot dataset collector (real dynamics)")
parser.add_argument("--robot", choices=["jetbot", "carter"], default="carter",
                    help="Robot type: jetbot (tiny) or carter (Nova Carter, full-size)")
parser.add_argument("--max-speed", type=float, default=0.5, help="Max linear speed (m/s)")
parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
parser.add_argument("--env", choices=["flat", "warehouse", "hospital"], default="hospital")
parser.add_argument("--res", type=int, default=512, help="Camera resolution (square)")
parser.add_argument("--max-time", type=float, default=60.0, help="Max episode time (sec)")
# Start pose
parser.add_argument("--start-x", type=float, default=0.0)
parser.add_argument("--start-y", type=float, default=0.0)
parser.add_argument("--start-yaw", type=float, default=0.0, help="Start heading (rad)")
# Target
parser.add_argument("--table-x", type=float, default=4.0, help="Target center X")
parser.add_argument("--table-y", type=float, default=0.0, help="Target center Y")
parser.add_argument("--table-yaw", type=float, default=90.0, help="Target rotation (deg)")
parser.add_argument("--stand-off", type=float, default=0.40, help="Stop distance before target (m)")
parser.add_argument("--no-table", action="store_true", help="Skip procedural table")
parser.add_argument("--rtx", action="store_true", help="Enable RTX path tracing")
parser.add_argument("--waypoints", type=str, default="",
                    help="Waypoints as 'x1,y1;x2,y2;...' — robot visits each before final goal")
parser.add_argument("--direct-goal", action="store_true",
                    help="Use --table-x/--table-y directly as goal (skip table offset calculation)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Parse waypoints
_waypoint_list = []
if args_cli.waypoints:
    for wp_str in args_cli.waypoints.split(";"):
        parts = wp_str.strip().split(",")
        if len(parts) == 2:
            _waypoint_list.append((float(parts[0]), float(parts[1])))

# Compute goal from table position
if getattr(args_cli, 'direct_goal', False):
    args_cli.goal_x = args_cli.table_x
    args_cli.goal_y = args_cli.table_y
    _dx = args_cli.table_x - args_cli.start_x
    _dy = args_cli.table_y - args_cli.start_y
    args_cli.goal_yaw = math.atan2(_dy, _dx)
else:
    _dx = args_cli.table_x - args_cli.start_x
    _dy = args_cli.table_y - args_cli.start_y
    _approach_yaw = math.atan2(_dy, _dx)
    _tyr = math.radians(args_cli.table_yaw)
    _table_hx = 0.6 * abs(math.cos(_tyr)) + 0.4 * abs(math.sin(_tyr))
    _table_hy = 0.6 * abs(math.sin(_tyr)) + 0.4 * abs(math.cos(_tyr))
    _adx, _ady = math.cos(_approach_yaw), math.sin(_approach_yaw)
    _table_reach = abs(_adx) * _table_hx + abs(_ady) * _table_hy
    _total_offset = _table_reach + args_cli.stand_off
    args_cli.goal_x = args_cli.table_x - _total_offset * math.cos(_approach_yaw)
    args_cli.goal_y = args_cli.table_y - _total_offset * math.sin(_approach_yaw)
    args_cli.goal_yaw = _approach_yaw

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, Sdf, UsdShade

# Enable wheeled_robots extension for WheeledRobot + DifferentialController
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.robot.wheeled_robots", True)

# ── Constants ────────────────────────────────────────────────────────────────
CONTROL_HZ = 30

# Robot configs: {robot_type: (wheel_radius, wheel_base, spawn_height, wheel_dof_names, usd_subpath, cam_height)}
ROBOT_CONFIGS = {
    "jetbot": {
        "wheel_radius": 0.03,
        "wheel_base": 0.1125,
        "spawn_height": 0.05,
        "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
        "usd_path": "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
        "cam_height": 0.15,       # FPV camera Z offset
        "cam_forward": 0.05,      # FPV camera X offset
        "chassis_link": "chassis",
    },
    "carter": {
        "wheel_radius": 0.08,     # Nova Carter wheel radius ~8cm
        "wheel_base": 0.413,      # distance between drive wheels
        "spawn_height": 0.0,
        "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
        "usd_path": "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        "cam_height": 0.55,       # FPV camera at top of robot
        "cam_forward": 0.15,      # FPV camera on front
        "chassis_link": "chassis_link",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def yaw_to_quat_wxyz(yaw):
    w = math.cos(yaw / 2)
    z = math.sin(yaw / 2)
    return w, 0.0, 0.0, z


def quat_to_heading(quat_wxyz):
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), \
                  float(quat_wxyz[2]), float(quat_wxyz[3])
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def _create_procedural_table(stage, goal_x, goal_y, yaw_deg=90.0):
    """Same procedural table as humanoid script."""
    table_path = "/World/GoalTable"
    if stage.GetPrimAtPath(table_path).IsValid():
        return

    xform = UsdGeom.Xform.Define(stage, table_path)
    xf = UsdGeom.Xformable(xform.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(goal_x, goal_y, 0.0))
    xf.AddRotateZOp().Set(float(yaw_deg))

    top = UsdGeom.Cube.Define(stage, f"{table_path}/Top")
    txf = UsdGeom.Xformable(top.GetPrim())
    txf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.38))
    txf.AddScaleOp().Set(Gf.Vec3f(0.6, 0.4, 0.02))

    all_parts = [f"{table_path}/Top"]
    for i, (lx, ly) in enumerate([(-0.5, -0.3), (0.5, -0.3), (-0.5, 0.3), (0.5, 0.3)]):
        lp = f"{table_path}/Leg{i}"
        leg = UsdGeom.Cube.Define(stage, lp)
        lxf = UsdGeom.Xformable(leg.GetPrim())
        lxf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, 0.18))
        lxf.AddScaleOp().Set(Gf.Vec3f(0.025, 0.025, 0.18))
        all_parts.append(lp)

    for part_path in all_parts:
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(part_path))

    mat = UsdShade.Material.Define(stage, f"{table_path}/TableMaterial")
    shader = UsdShade.Shader.Define(stage, f"{table_path}/TableMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.30, 0.15))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for part_path in all_parts:
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(part_path)).Bind(mat)
    print(f"[collect] Created table at ({goal_x:.1f}, {goal_y:.1f})")


def load_scene(env_type, table_x, table_y, table_yaw_deg, no_table=False):
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    if env_type == "warehouse":
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.core.utils.nucleus import get_assets_root_path
            assets = get_assets_root_path()
            add_reference_to_stage(
                usd_path=assets + "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                prim_path="/World/Environment")
            print("[collect] Loaded warehouse")
        except Exception as e:
            print(f"[WARN] Warehouse: {e}")

    elif env_type == "hospital":
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Hospital/hospital.usd",
                prim_path="/World/Environment")
            print("[collect] Loaded hospital")
        except Exception as e:
            print(f"[WARN] Hospital: {e}")

    if not no_table:
        _create_procedural_table(stage, table_x, table_y, table_yaw_deg)

    light_path = "/World/ExtraDomeLight"
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = stage.DefinePrim(light_path, "DomeLight")
        UsdLux.DomeLight(light).CreateIntensityAttr(3000.0)
        UsdLux.DomeLight(light).CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))


def setup_fpv_camera(stage, robot_path, robot_cfg):
    """Create FPV camera on robot chassis facing forward (+X) with Z-up."""
    chassis = robot_cfg["chassis_link"]
    cam_path = f"{robot_path}/{chassis}/FPVCam"
    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam_xf = UsdGeom.Xformable(cam.GetPrim())
    cam_xf.AddTranslateOp().Set(Gf.Vec3d(robot_cfg["cam_forward"], 0.0, robot_cfg["cam_height"]))
    # USD camera default: looks -Z with +Y up
    # RotateXYZ(90, 0, -90) transforms -Z→+X and +Y→+Z (look forward, Z-up)
    # +3° on X tilts slightly upward to reduce ground visibility
    cam_xf.AddRotateXYZOp().Set(Gf.Vec3f(93, 0, -90))
    cam.GetFocalLengthAttr().Set(18.0)
    cam.GetHorizontalApertureAttr().Set(20.955)
    return cam_path


def update_third_person_camera(stage, tp_cam_path, rx, ry, ryaw, behind=2.5, height=1.5):
    """Position third-person camera behind+above robot, looking forward+down."""
    tp_xf = UsdGeom.Xformable(stage.GetPrimAtPath(tp_cam_path))
    tp_xf.ClearXformOpOrder()
    cam_x = rx - behind * math.cos(ryaw)
    cam_y = ry - behind * math.sin(ryaw)
    tp_xf.AddTranslateOp().Set(Gf.Vec3d(cam_x, cam_y, height))
    yaw_deg = math.degrees(ryaw)
    tilt_down = 30
    tp_xf.AddRotateXYZOp().Set(Gf.Vec3f(90 - tilt_down, 0, -90 + yaw_deg))


def enable_ray_tracing():
    import carb.settings
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    print("[collect] RTX Path Tracing enabled")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.robot.wheeled_robots.robots.wheeled_robot import WheeledRobot
    from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()

    # Load environment
    load_scene(args_cli.env, args_cli.table_x, args_cli.table_y, args_cli.table_yaw,
               no_table=args_cli.no_table)

    # Robot config
    rcfg = ROBOT_CONFIGS[args_cli.robot]
    from isaacsim.core.utils.nucleus import get_assets_root_path
    assets = get_assets_root_path()
    robot_path = f"/World/{args_cli.robot.capitalize()}"

    w, qx, qy, qz = yaw_to_quat_wxyz(args_cli.start_yaw)

    robot = world.scene.add(
        WheeledRobot(
            prim_path=robot_path,
            name=args_cli.robot,
            wheel_dof_names=rcfg["wheel_dof_names"],
            create_robot=True,
            usd_path=assets + rcfg["usd_path"],
            position=np.array([args_cli.start_x, args_cli.start_y, rcfg["spawn_height"]]),
            orientation=np.array([w, qx, qy, qz]),
        )
    )
    print(f"[collect] Spawned {args_cli.robot} (WheeledRobot) at {robot_path}")

    # Setup FPV camera
    fpv_cam_path = setup_fpv_camera(stage, robot_path, rcfg)

    # Create third-person camera
    tp_cam_path = "/World/ThirdPersonCam"
    tp_cam = UsdGeom.Camera.Define(stage, tp_cam_path)
    tp_cam.GetFocalLengthAttr().Set(18.0)
    tp_cam.GetHorizontalApertureAttr().Set(20.955)

    if args_cli.rtx:
        enable_ray_tracing()

    # Reset world — this triggers WheeledRobot.initialize() + post_reset()
    # post_reset() calls switch_control_mode("velocity") for real dynamics
    world.reset()

    # Print joint info
    print(f"[collect] {args_cli.robot} joints: {robot.dof_names}")
    print(f"[collect] Wheel DOF indices: {robot.wheel_dof_indices}")

    # Official Isaac Sim DifferentialController for real wheel dynamics
    diff_controller = DifferentialController(
        name="diff_controller",
        wheel_radius=rcfg["wheel_radius"],
        wheel_base=rcfg["wheel_base"],
    )
    # Proportional heading gain for smooth simultaneous drive+steer
    KP_YAW = 2.0

    # Build waypoint list: intermediate waypoints + final goal
    all_goals = []
    for wx, wy in _waypoint_list:
        all_goals.append(np.array([wx, wy, 0.0]))
    all_goals.append(np.array([args_cli.goal_x, args_cli.goal_y, 0.0]))
    current_goal_idx = 0
    goal_position = all_goals[current_goal_idx]
    print(f"[collect] Waypoints: {len(all_goals)} total ({len(_waypoint_list)} intermediate + 1 final)")

    # ── Frame dirs ────────────────────────────────────────────────────────
    has_cameras = getattr(args_cli, "enable_cameras", False)
    frames_fpv = out_dir / "frames_fpv"
    frames_tp = out_dir / "frames_third_person"
    if has_cameras:
        frames_fpv.mkdir(parents=True, exist_ok=True)
        frames_tp.mkdir(parents=True, exist_ok=True)

    # Camera render products
    fpv_rp = None
    tp_rp = None
    if has_cameras:
        try:
            import omni.replicator.core as rep
            fpv_rp = rep.create.render_product(fpv_cam_path,
                                                (args_cli.res, args_cli.res))
            tp_rp = rep.create.render_product(tp_cam_path,
                                               (args_cli.res, args_cli.res))
            annot_fpv = rep.AnnotatorRegistry.get_annotator("rgb")
            annot_fpv.attach([fpv_rp])
            annot_tp = rep.AnnotatorRegistry.get_annotator("rgb")
            annot_tp.attach([tp_rp])
            print("[collect] Cameras ready")
        except Exception as e:
            print(f"[WARN] Cameras: {e}")
            has_cameras = False

    # ── CSV logger ────────────────────────────────────────────────────────
    csv_path = out_dir / "velocity_commands.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "time", "x", "y", "yaw",
                         "lin_vel_cmd", "ang_vel_cmd",
                         "wheel_left_vel", "wheel_right_vel", "phase"])

    # ── Run ───────────────────────────────────────────────────────────────
    max_steps = int(args_cli.max_time * CONTROL_HZ)
    print(f"[collect] Start: ({args_cli.start_x:.1f}, {args_cli.start_y:.1f}, yaw={args_cli.start_yaw:.2f})")
    print(f"[collect] Goal:  ({args_cli.goal_x:.1f}, {args_cli.goal_y:.1f}, yaw={args_cli.goal_yaw:.2f})")
    print(f"[collect] Running (max {max_steps} steps = {args_cli.max_time:.1f}s) with REAL dynamics...")

    frame_count = 0
    done = False
    stand_count = 0
    STAND_FRAMES = 60  # frames to stand still after reaching goal

    for step in range(max_steps):
        # Get robot pose from physics simulation
        pos, quat = robot.get_world_pose()
        rx, ry, rz = float(pos[0]), float(pos[1]), float(pos[2])
        ryaw = quat_to_heading(quat)

        dist = math.sqrt((goal_position[0] - rx)**2 + (goal_position[1] - ry)**2)

        if not done:
            # Check if we reached current waypoint
            if dist < 0.20 and current_goal_idx < len(all_goals) - 1:
                current_goal_idx += 1
                goal_position = all_goals[current_goal_idx]
                print(f"  [waypoint] Reached WP {current_goal_idx}, next: ({goal_position[0]:.1f}, {goal_position[1]:.1f})")
                dist = math.sqrt((goal_position[0] - rx)**2 + (goal_position[1] - ry)**2)

            # Proportional controller: drive forward + steer simultaneously
            desired_yaw = math.atan2(goal_position[1] - ry, goal_position[0] - rx)
            yaw_err = (desired_yaw - ryaw + math.pi) % (2 * math.pi) - math.pi
            ang_vel = KP_YAW * yaw_err
            ang_vel = max(-1.0, min(1.0, ang_vel))  # clamp angular velocity
            # Reduce forward speed when heading is way off
            fwd_factor = max(0.0, math.cos(yaw_err))
            lin_vel = args_cli.max_speed * fwd_factor
            actions = diff_controller.forward(np.array([lin_vel, ang_vel]))

            robot.apply_wheel_actions(actions)

            # Extract commanded velocities for logging
            wv = actions.joint_velocities
            if wv is not None:
                wl, wr = float(wv[0]), float(wv[1])
                lin_cmd = (wl + wr) * rcfg["wheel_radius"] / 2.0
                ang_cmd = (wr - wl) * rcfg["wheel_radius"] / rcfg["wheel_base"]
            else:
                wl, wr, lin_cmd, ang_cmd = 0, 0, 0, 0

            phase = f"wp{current_goal_idx}" if current_goal_idx < len(all_goals) - 1 else "walk"
            if dist < 0.15 and current_goal_idx == len(all_goals) - 1:
                done = True
                stand_count = 0
                phase = "stand"
        else:
            # Robot reached goal — stop wheels
            from isaacsim.core.utils.types import ArticulationAction
            robot.apply_wheel_actions(
                ArticulationAction(joint_velocities=np.array([0.0, 0.0]))
            )
            stand_count += 1
            wl, wr, lin_cmd, ang_cmd = 0, 0, 0, 0
            phase = "stand"
            if stand_count >= STAND_FRAMES:
                phase = "done"

        # Update third-person camera
        update_third_person_camera(stage, tp_cam_path, rx, ry, ryaw)

        # Physics + render step
        world.step(render=True)

        # Log
        csv_writer.writerow([step, f"{step/CONTROL_HZ:.3f}", f"{rx:.3f}", f"{ry:.3f}",
                             f"{ryaw:.3f}", f"{lin_cmd:.4f}", f"{ang_cmd:.4f}",
                             f"{wl:.3f}", f"{wr:.3f}", phase])

        # Capture frames
        if has_cameras:
            try:
                fpv_data = annot_fpv.get_data()
                tp_data = annot_tp.get_data()
                if fpv_data is not None and len(fpv_data.shape) >= 2:
                    import PIL.Image
                    fpv_img = PIL.Image.fromarray(fpv_data[:, :, :3])
                    fpv_img.save(frames_fpv / f"frame_{frame_count:06d}.png")
                    tp_img = PIL.Image.fromarray(tp_data[:, :, :3])
                    tp_img.save(frames_tp / f"frame_{frame_count:06d}.png")
                    frame_count += 1
            except Exception:
                pass

        if step % 50 == 0:
            # Also read actual wheel velocities from physics
            actual_wv = robot.get_wheel_velocities()
            print(f"  step {step:4d}  [{phase:6s}]  cmd_lin={lin_cmd:+.3f} cmd_ang={ang_cmd:+.3f}"
                  f"  wheel=[{wl:+.2f},{wr:+.2f}]  actual_wheel={actual_wv}"
                  f"  pos=[{rx:+.2f},{ry:+.2f}]  dist={dist:.2f}  frames={frame_count}")

        if phase == "done":
            print(f"[collect] DONE at step {step}")
            break

    csv_file.close()

    # Final pose
    pos, quat = robot.get_world_pose()
    fx, fy = float(pos[0]), float(pos[1])
    fyaw = quat_to_heading(quat)

    # Save metadata
    meta = {
        "robot": args_cli.robot,
        "dynamics": "real_physics (WheeledRobot + DifferentialController)",
        "env": args_cli.env,
        "wheel_radius": rcfg["wheel_radius"],
        "wheel_base": rcfg["wheel_base"],
        "start": {"x": args_cli.start_x, "y": args_cli.start_y, "yaw": args_cli.start_yaw},
        "goal": {"x": float(args_cli.goal_x), "y": float(args_cli.goal_y), "yaw": float(args_cli.goal_yaw)},
        "final": {"x": fx, "y": fy, "yaw": fyaw},
        "frames": frame_count,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    pos_err = math.sqrt((fx - args_cli.goal_x)**2 + (fy - args_cli.goal_y)**2)
    yaw_err = abs(math.degrees(math.atan2(math.sin(fyaw - args_cli.goal_yaw),
                                           math.cos(fyaw - args_cli.goal_yaw))))

    print(f"\n[collect] Done. {frame_count} frames saved.")
    print(f"  Dynamics: REAL physics (WheeledRobot velocity control)")
    print(f"  Start: ({args_cli.start_x:+.2f}, {args_cli.start_y:+.2f}, yaw={args_cli.start_yaw:.2f})")
    print(f"  Final: ({fx:+.2f}, {fy:+.2f}, yaw={fyaw:.2f})")
    print(f"  Goal:  ({args_cli.goal_x:+.2f}, {args_cli.goal_y:+.2f}, yaw={args_cli.goal_yaw:.2f})")
    print(f"  Pos error: {pos_err:.3f}m | Yaw error: {yaw_err:.1f}deg")
    print(f"  Output: {args_cli.output_dir}")

    simulation_app.close()


if __name__ == "__main__":
    run()
