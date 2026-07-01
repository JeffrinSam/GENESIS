#!/usr/bin/env python3
"""
Manual WASD+QE control for G1 humanoid in warehouse.

Uses carb.input keyboard events (same as NVIDIA G1 Digital Twin example).

Controls (one command at a time):
  W / Up / Numpad8     — forward  (vx)
  S                    — backward (vx)
  A / Left / Numpad4   — turn left  (vyaw)
  D / Right / Numpad6  — turn right (vyaw)
  Q                    — strafe left  (vy)
  E                    — strafe right (vy)
  Release key          — stop

Usage:
  ./run_manual.sh --enable_cameras
"""

import sys
import os
import math
import pathlib
import argparse

DIGITALTWIN_DIR = pathlib.Path(os.getenv("DIGITALTWIN_DIR", "/opt/Digitaltwin"))
if str(DIGITALTWIN_DIR) not in sys.path:
    sys.path.insert(0, str(DIGITALTWIN_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 manual WASD+QE control")
parser.add_argument("--speed", type=float, default=0.3, help="Forward/backward speed (m/s)")
parser.add_argument("--strafe-speed", type=float, default=0.3, help="Strafe speed (m/s)")
parser.add_argument("--turn-speed", type=float, default=0.5, help="Turn speed (rad/s)")
parser.add_argument("--env", choices=["flat", "warehouse", "hospital", "office", "digital_warehouse"], default="warehouse")
parser.add_argument("--res", type=int, default=720, help="Camera resolution (square)")
parser.add_argument("--start-x", type=float, default=0.0)
parser.add_argument("--start-y", type=float, default=0.0)
parser.add_argument("--start-yaw", type=float, default=0.0)
parser.add_argument("--table-x", type=float, default=5.0, help="Table center X")
parser.add_argument("--table-y", type=float, default=0.0, help="Table center Y")
parser.add_argument("--table-yaw", type=float, default=90.0, help="Table rotation (deg)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input
import numpy as np
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from pxr import Gf, UsdGeom, UsdLux

from g1_env import G1FlatEnvCfg, G1FlatSceneCfg
from wbc_controller import G1WBCController

CONTROL_HZ = 30
SPAWN_HEIGHT = 0.74


# ── Scene config ──────────────────────────────────────────────────────────────
@configclass
class G1SceneManualCfg(G1FlatSceneCfg):
    ego_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link/EgoCam",
        update_period=0.0,
        height=args_cli.res, width=args_cli.res,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=20.955,
            vertical_aperture=20.955, clipping_range=(0.1, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.04485, 0.0, 0.35325),
            rot=(0.32651, -0.62721, 0.62721, -0.32651),
            convention="ros",
        ),
    )


@configclass
class G1EnvManualCfg(G1FlatEnvCfg):
    scene: G1SceneManualCfg = G1SceneManualCfg(num_envs=1, env_spacing=4.0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def yaw_to_quat_wxyz(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def quat_to_heading(quat_wxyz):
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), \
                  float(quat_wxyz[2]), float(quat_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def teleport_robot(robot, x, y, yaw, device):
    w, qx, qy, qz = yaw_to_quat_wxyz(yaw)
    root_pose = torch.tensor([[x, y, SPAWN_HEIGHT, w, qx, qy, qz]],
                             dtype=torch.float32, device=device)
    root_vel = torch.zeros((1, 6), dtype=torch.float32, device=device)
    robot.write_root_link_pose_to_sim(root_pose)
    robot.write_root_link_velocity_to_sim(root_vel)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel))
    robot.reset()


def load_scene(env_type, table_x, table_y, table_yaw_deg):
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
            print("[manual] Loaded warehouse")
        except Exception as e:
            print(f"[WARN] Warehouse: {e}")

    elif env_type == "hospital":
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Hospital/hospital.usd",
                prim_path="/World/Environment")
            print("[manual] Loaded hospital")
        except Exception as e:
            print(f"[WARN] Hospital: {e}")

    elif env_type == "office":
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd",
                prim_path="/World/Environment")
            print("[manual] Loaded office")
        except Exception as e:
            print(f"[WARN] Office: {e}")

    elif env_type == "digital_warehouse":
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
                prim_path="/World/Environment")
            print("[manual] Loaded digital twin warehouse")
        except Exception as e:
            print(f"[WARN] Digital Twin Warehouse: {e}")

    # Table
    from pxr import Sdf, UsdShade, UsdPhysics
    table_path = "/World/GoalTable"
    existing = stage.GetPrimAtPath(table_path)
    if existing.IsValid():
        stage.RemovePrim(table_path)
    table_prim = stage.DefinePrim(table_path, "Xform")
    xf = UsdGeom.Xformable(table_prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(table_x, table_y, 0.0))
    xf.AddRotateZOp().Set(float(table_yaw_deg))
    top = UsdGeom.Cube.Define(stage, f"{table_path}/top")
    top_xf = UsdGeom.Xformable(top.GetPrim())
    top_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.75))
    top_xf.AddScaleOp().Set(Gf.Vec3f(0.6, 0.4, 0.02))
    for i, (lx, ly) in enumerate([(-0.55, -0.35), (-0.55, 0.35), (0.55, -0.35), (0.55, 0.35)]):
        leg = UsdGeom.Cube.Define(stage, f"{table_path}/leg_{i}")
        leg_xf = UsdGeom.Xformable(leg.GetPrim())
        leg_xf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, 0.365))
        leg_xf.AddScaleOp().Set(Gf.Vec3f(0.02, 0.02, 0.365))
    all_parts = [f"{table_path}/top"] + [f"{table_path}/leg_{i}" for i in range(4)]
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
    print(f"[manual] Table at ({table_x:.1f}, {table_y:.1f})")

    light_path = "/World/ExtraDomeLight"
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = stage.DefinePrim(light_path, "DomeLight")
        UsdLux.DomeLight(light).CreateIntensityAttr(3000.0)
        UsdLux.DomeLight(light).CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    has_cameras = getattr(args_cli, "enable_cameras", False)
    if has_cameras:
        env_cfg = G1EnvManualCfg()
    else:
        env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedEnv(cfg=env_cfg)
    device = env.device

    load_scene(args_cli.env, args_cli.table_x, args_cli.table_y, args_cli.table_yaw)

    wbc = G1WBCController(num_envs=1)
    obs, _ = env.reset()
    wbc.reset()
    robot = env.scene["robot"]

    teleport_robot(robot, args_cli.start_x, args_cli.start_y, args_cli.start_yaw, device)

    # Find waist joints for posture lock
    jnames = robot.data.joint_names
    waist_pitch_idx = None
    waist_roll_idx = None
    for i, n in enumerate(jnames):
        if n == "waist_pitch_joint":
            waist_pitch_idx = i
        elif n == "waist_roll_joint":
            waist_roll_idx = i
    print(f"[manual] waist_pitch={waist_pitch_idx} waist_roll={waist_roll_idx}")

    # Warmup
    for _ in range(60):
        wbc.set_velocity(0.0, 0.0, 0.0)
        targets = wbc.step(robot.data, device=str(device))
        obs, _ = env.step(targets)

    # ── Keyboard via carb.input (same as NVIDIA G1 Digital Twin) ──────────
    base_command = np.array([0.0, 0.0, 0.0])  # [vx, vy, vyaw]

    # Key → [vx, vy, vyaw] — one command at a time
    input_keyboard_mapping = {
        "W":        [args_cli.speed, 0.0, 0.0],
        "UP":       [args_cli.speed, 0.0, 0.0],
        "NUMPAD_8": [args_cli.speed, 0.0, 0.0],
        "S":        [-args_cli.speed, 0.0, 0.0],
        "A":        [0.0, 0.0, args_cli.turn_speed],
        "LEFT":     [0.0, 0.0, args_cli.turn_speed],
        "NUMPAD_4": [0.0, 0.0, args_cli.turn_speed],
        "D":        [0.0, 0.0, -args_cli.turn_speed],
        "RIGHT":    [0.0, 0.0, -args_cli.turn_speed],
        "NUMPAD_6": [0.0, 0.0, -args_cli.turn_speed],
        "Q":        [0.0, args_cli.strafe_speed, 0.0],
        "E":        [0.0, -args_cli.strafe_speed, 0.0],
    }

    def on_keyboard_event(event, *args, **kwargs):
        nonlocal base_command
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in input_keyboard_mapping:
                base_command = np.array(input_keyboard_mapping[event.input.name])
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in input_keyboard_mapping:
                base_command[:] = 0.0
        return True

    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    sub_keyboard = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("=" * 50)
    print(" G1 Manual Control (carb.input)")
    print("=" * 50)
    print(f" W/S      = fwd/back   ({args_cli.speed:.2f} m/s)")
    print(f" Q/E      = strafe L/R ({args_cli.strafe_speed:.2f} m/s)")
    print(f" A/D      = turn  L/R  ({args_cli.turn_speed:.2f} rad/s)")
    print(f" Release  = stop")
    print("=" * 50)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            vx, vy, vyaw = float(base_command[0]), float(base_command[1]), float(base_command[2])
            wbc.set_velocity(vx, vy, vyaw)
            targets = wbc.step(robot.data, device=str(device))
            # Lock waist upright — prevent forward lean
            if waist_pitch_idx is not None:
                targets[0, waist_pitch_idx] = 0.0
            if waist_roll_idx is not None:
                targets[0, waist_roll_idx] = 0.0
            obs, _ = env.step(targets)

        if step % 90 == 0:  # every 3s
            bp = robot.data.root_link_pos_w[0].cpu().numpy()
            heading = quat_to_heading(robot.data.root_link_quat_w[0].cpu().numpy())
            cmd = "fwd" if vx > 0 else "back" if vx < 0 else \
                  "left" if vy > 0 else "right" if vy < 0 else \
                  "turnL" if vyaw > 0 else "turnR" if vyaw < 0 else "stop"
            print(f"  [{cmd:5s}]  pos=({bp[0]:+.2f}, {bp[1]:+.2f})  yaw={math.degrees(heading):+.1f}deg")

        step += 1

    input_iface.unsubscribe_to_keyboard_events(keyboard, sub_keyboard)
    env.close()
    print("[manual] Done.")


if __name__ == "__main__":
    run()
    simulation_app.close()
