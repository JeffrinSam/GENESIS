#!/usr/bin/env python3
"""
Dataset collector: G1 humanoid navigates to a goal pose in warehouse.

Continuous heading+cross-track controller:
  WALK  → full speed with PD heading correction + lateral vy cross-track
  STAND → record standing frames at goal
  DONE  → save and exit

Supports arbitrary start/goal poses for data generation.

Usage:
  ./run_collect.sh --output-dir proof/episode_001 --headless --enable_cameras
  ./run_collect.sh --output-dir proof/episode_001 --headless --enable_cameras \
    --start-x -2 --start-y 1 --start-yaw 0.5 \
    --goal-x 5 --goal-y 0 --goal-yaw 3.14
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

DIGITALTWIN_DIR = pathlib.Path(os.getenv("DIGITALTWIN_DIR", "/opt/Digitaltwin"))
if str(DIGITALTWIN_DIR) not in sys.path:
    sys.path.insert(0, str(DIGITALTWIN_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 humanoid dataset collector")
parser.add_argument("--max-speed", type=float, default=0.3, help="Max walking speed (m/s)")
parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
parser.add_argument("--env", choices=["flat", "warehouse", "hospital", "office", "digital_warehouse"], default="warehouse")
parser.add_argument("--res", type=int, default=1080, help="Camera resolution (square)")
parser.add_argument("--max-time", type=float, default=60.0, help="Max episode time (sec)")
# Start pose
parser.add_argument("--start-x", type=float, default=0.0)
parser.add_argument("--start-y", type=float, default=0.0)
parser.add_argument("--start-yaw", type=float, default=0.0, help="Start heading (rad)")
# Target object: robot stops stand-off meters in front of it
parser.add_argument("--table-x", type=float, default=5.0, help="Target object center X")
parser.add_argument("--table-y", type=float, default=0.0, help="Target object center Y")
parser.add_argument("--table-yaw", type=float, default=90.0, help="Target object rotation (deg)")
parser.add_argument("--stand-off", type=float, default=0.60, help="Stop distance before target edge (m)")
parser.add_argument("--no-table", action="store_true", help="Skip procedural table (navigate to existing object)")
parser.add_argument("--rtx", action="store_true", help="Enable RTX path tracing for realistic rendering")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Robot approaches from start toward table; compute heading and stop point
_dx = args_cli.table_x - args_cli.start_x
_dy = args_cli.table_y - args_cli.start_y
_approach_yaw = math.atan2(_dy, _dx)
# Table half-extents rotated into world frame (top cube scale = 0.6 x 0.4)
_tyr = math.radians(args_cli.table_yaw)
_table_hx = 0.6 * abs(math.cos(_tyr)) + 0.4 * abs(math.sin(_tyr))
_table_hy = 0.6 * abs(math.sin(_tyr)) + 0.4 * abs(math.cos(_tyr))
# Half-extent along approach direction
_adx, _ady = math.cos(_approach_yaw), math.sin(_approach_yaw)
_table_reach = abs(_adx) * _table_hx + abs(_ady) * _table_hy
# Stop stand_off meters from table EDGE (not center)
_total_offset = _table_reach + args_cli.stand_off
args_cli.goal_x = args_cli.table_x - _total_offset * math.cos(_approach_yaw)
args_cli.goal_y = args_cli.table_y - _total_offset * math.sin(_approach_yaw)
args_cli.goal_yaw = _approach_yaw

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from pxr import Gf, UsdGeom, UsdLux

from g1_env import G1FlatEnvCfg, G1FlatSceneCfg
from wbc_controller import G1WBCController

try:
    import omni.replicator.core as rep
    _HAS_REPLICATOR = True
except (ImportError, ModuleNotFoundError):
    rep = None
    _HAS_REPLICATOR = False

# ── Constants ────────────────────────────────────────────────────────────────
CONTROL_HZ = 30
CAPTURE_INTERVAL = max(1, int(round(CONTROL_HZ / args_cli.fps)))
SPAWN_HEIGHT = 0.74


# ══════════════════════════════════════════════════════════════════════════════
# GoalPoseController — WALK → ALIGN → STAND navigation to (x, y, yaw) goal
# ══════════════════════════════════════════════════════════════════════════════
class GoalPoseController:
    """Two-stage controller: position first, then orientation.

    WALK:   Navigate to goal position (PD heading + cross-track).
    ORIENT: Turn in place to face goal_yaw (small vx for WBC turn).
    STAND:  Hold zero velocity.

    Table AABB safety prevents collision.
    """

    WALK   = "walk"
    ORIENT = "orient"
    STAND  = "stand"
    DONE   = "done"

    def __init__(self, goal_x, goal_y, goal_yaw,
                 start_x=0.0, start_y=0.0,
                 table_x=None, table_y=None, table_yaw_deg=90.0,
                 table_half_x=0.6, table_half_y=0.4,
                 max_vx=0.3, max_vyaw=1.0, max_vy=0.15,
                 stop_dist=0.60, brake_dist=2.5,
                 yaw_tol=0.10,           # ~5.7deg for orient
                 orient_vx=0.0,
                 robot_radius=0.45,      # robot body + arm reach (m)
                 apf_influence=2.0,      # APF influence zone from surface (m)
                 apf_hard_stop=0.35,     # hard stop distance from surface (m)
                 crawl_dist=1.0,         # crawl zone: cap speed when this close (m)
                 crawl_speed=0.05,       # max speed in crawl zone (m/s)
                 stand_steps=60):
        self.goal_x, self.goal_y, self.goal_yaw = goal_x, goal_y, goal_yaw
        self.start_x, self.start_y = start_x, start_y
        self.table_x, self.table_y = table_x, table_y
        self.max_vx, self.max_vyaw, self.max_vy = max_vx, max_vyaw, max_vy
        self.stop_dist, self.brake_dist = stop_dist, brake_dist
        self.yaw_tol, self.orient_vx = yaw_tol, orient_vx
        self.robot_radius = robot_radius
        self.apf_influence = apf_influence      # slow down within this distance
        self.apf_hard_stop = apf_hard_stop      # full stop within this distance
        self.crawl_dist = crawl_dist            # crawl when surface dist < this
        self.crawl_speed = crawl_speed          # max speed in crawl zone
        self.stand_steps = stand_steps

        # Table AABB in world frame
        if table_x is not None:
            yr = math.radians(table_yaw_deg)
            cy, sy = abs(math.cos(yr)), abs(math.sin(yr))
            self._thw = table_half_x * cy + table_half_y * sy
            self._thd = table_half_x * sy + table_half_y * cy
        else:
            self._thw, self._thd = 0.0, 0.0

        # PD gains
        self.kp_yaw, self.kd_yaw, self.kp_cross = 2.5, 0.4, 0.5
        # Smoothing
        self._max_dvx = 0.010       # vx rate limit
        self._max_dvyaw = 0.05      # vyaw rate limit for orient
        self._ema_vyaw, self._ema_vy = 0.85, 0.80

        # Cross-track line
        ldx, ldy = goal_x - start_x, goal_y - start_y
        ll = math.sqrt(ldx * ldx + ldy * ldy)
        if ll > 0.01:
            self._lnx, self._lny = -ldy / ll, ldx / ll
        else:
            self._lnx, self._lny = 0.0, 1.0

        # State
        self.phase = self.WALK
        self._stand_cnt = 0
        self._prev_herr = 0.0
        self._out_vx = 0.0
        self._out_vyaw = 0.0
        self._sm_vy = 0.0
        self._sm_vyaw = 0.0

    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    def _rlim(self, tgt, cur, md):
        return cur + max(-md, min(md, tgt - cur))

    def _surface_dist(self, rx, ry):
        """Distance from robot surface to table surface.
        Robot = circle (robot_radius), Table = AABB."""
        if self.table_x is None:
            return float('inf')
        # Distance from robot center to table AABB edge
        gx = abs(rx - self.table_x) - self._thw
        gy = abs(ry - self.table_y) - self._thd
        if gx > 0 and gy > 0:
            center_dist = math.sqrt(gx * gx + gy * gy)
        else:
            center_dist = max(gx, gy)
        # Subtract robot radius to get surface-to-surface
        return center_dist - self.robot_radius

    def _apf_scale(self, rx, ry):
        """APF repulsive speed scale: 1.0 = no effect, 0.0 = full stop.
        Smooth cosine ramp between apf_influence and apf_hard_stop."""
        sd = self._surface_dist(rx, ry)
        if sd >= self.apf_influence:
            return 1.0      # outside influence zone — no effect
        if sd <= self.apf_hard_stop:
            return 0.0      # too close — full stop
        # Cosine ramp: smooth 1→0
        t = (sd - self.apf_hard_stop) / (self.apf_influence - self.apf_hard_stop)
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def compute(self, robot_x, robot_y, robot_yaw):
        """Returns: (vx, vy, vyaw, phase, done)"""
        dx = self.goal_x - robot_x
        dy = self.goal_y - robot_y
        dist = math.sqrt(dx * dx + dy * dy)
        apf = self._apf_scale(robot_x, robot_y)
        sd = self._surface_dist(robot_x, robot_y)

        # ── PROXIMITY STOP — force STAND when APF says danger ────────
        # APF hard stop OR very low APF (robot drifts forward during turns)
        if apf < 0.15 and self.phase in (self.WALK, self.ORIENT):
            self.phase = self.STAND
            self._stand_cnt = 0

        # ── WALK — navigate to goal position ─────────────────────────
        if self.phase == self.WALK:
            if dist < self.stop_dist:
                yerr = self._wrap(self.goal_yaw - robot_yaw)
                if abs(yerr) < 0.035:
                    self.phase = self.STAND
                else:
                    self.phase = self.ORIENT
                self._stand_cnt = 0
                self._out_vx = 0.0
                self._out_vyaw = 0.0
                self._sm_vy = 0.0
                self._sm_vyaw = 0.0
                # fall through
            else:
                # Heading toward goal
                gh = math.atan2(dy, dx)
                herr = self._wrap(gh - robot_yaw)
                derr = herr - self._prev_herr
                self._prev_herr = herr

                # Cross-track error
                ct = (robot_x - self.start_x) * self._lnx + (robot_y - self.start_y) * self._lny

                # Crawl zone: cap ALL speeds when close to table
                speed_cap = self.crawl_speed if sd < self.crawl_dist else 999.0

                # One command at a time: vyaw > vy > vx
                # APF gates ALL commands (WBC drifts forward even during turns)
                if abs(herr) > 0.15:
                    tvyaw = float(np.clip(self.kp_yaw * herr + self.kd_yaw * derr,
                                           -self.max_vyaw, self.max_vyaw))
                    tvyaw *= apf  # APF scales turn speed too
                    tvyaw = max(-speed_cap * 5, min(speed_cap * 5, tvyaw))  # crawl cap
                    self._sm_vyaw = self._ema_vyaw * self._sm_vyaw + (1 - self._ema_vyaw) * tvyaw
                    return 0.0, 0.0, self._sm_vyaw, self.WALK, False

                if abs(ct) > 0.10:
                    tvy = float(np.clip(-self.kp_cross * ct, -self.max_vy, self.max_vy))
                    tvy *= apf  # APF scales strafe too
                    tvy = max(-speed_cap, min(speed_cap, tvy))
                    self._sm_vy = self._ema_vy * self._sm_vy + (1 - self._ema_vy) * tvy
                    return 0.0, self._sm_vy, 0.0, self.WALK, False

                # Walk forward — cosine brake for goal + APF for table
                if dist < self.brake_dist:
                    f = max(0, min(1, (dist - self.stop_dist) / (self.brake_dist - self.stop_dist)))
                    f = 0.5 * (1.0 - math.cos(math.pi * f))
                    tvx = 0.02 + (self.max_vx - 0.02) * f  # min 0.02 (was 0.08)
                else:
                    tvx = self.max_vx

                # APF: scale speed by repulsive field
                tvx *= apf
                # Crawl zone cap
                tvx = min(tvx, speed_cap)

                self._out_vx = self._rlim(tvx, self._out_vx, self._max_dvx)
                return self._out_vx, 0.0, 0.0, self.WALK, False

        # ── ORIENT — turn in place to face goal_yaw ──────────────────
        if self.phase == self.ORIENT:
            yerr = self._wrap(self.goal_yaw - robot_yaw)
            if abs(yerr) < self.yaw_tol:
                self.phase = self.STAND
                self._stand_cnt = 0
                # fall through to STAND
            else:
                tvyaw = float(np.clip(2.0 * yerr, -self.max_vyaw, self.max_vyaw))
                tvyaw *= apf  # APF gates orient turns too
                self._out_vyaw = self._rlim(tvyaw, self._out_vyaw, self._max_dvyaw)
                return 0.0, 0.0, self._out_vyaw, self.ORIENT, False

        # ── STAND — hold zero ────────────────────────────────────────
        if self.phase == self.STAND:
            self._stand_cnt += 1
            if self._stand_cnt >= self.stand_steps:
                self.phase = self.DONE
                return 0.0, 0.0, 0.0, self.DONE, True
            return 0.0, 0.0, 0.0, self.STAND, False

        return 0.0, 0.0, 0.0, self.DONE, True


# ── Scene config with cameras on head_link ───────────────────────────────────
@configclass
class G1SceneWithCamCfg(G1FlatSceneCfg):
    """G1 scene + ego camera + FPV camera on robot's head_link."""

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

    fpv_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/FPVCam",
        update_period=0.0,
        height=args_cli.res, width=args_cli.res,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=20.955,
            vertical_aperture=20.955, clipping_range=(0.1, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.30),             # pelvis: forward + up to chest height
            rot=(0.9848, 0.0, 0.1736, 0.0),    # ~20deg upward (pelvis tilts less than torso)
            convention="world",
        ),
    )


@configclass
class G1EnvWithCamCfg(G1FlatEnvCfg):
    scene: G1SceneWithCamCfg = G1SceneWithCamCfg(num_envs=1, env_spacing=4.0)


# ── Helpers ──────────────────────────────────────────────────────────────────
def qrot(q, v):
    w, x, y, z = [float(a) for a in q]
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return R @ np.asarray(v, dtype=np.float64)


def quat_to_heading(quat_wxyz):
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), \
                  float(quat_wxyz[2]), float(quat_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat_wxyz(yaw):
    """Convert yaw angle (rad) to quaternion (w, x, y, z)."""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def look_at(eye, tgt, up=None):
    if up is None:
        up = np.array([0., 0., 1.])
    eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
    if np.linalg.norm(tgt - eye) < 1e-6:
        tgt = eye + np.array([1., 0., 0.])
    v = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*tgt.tolist()), Gf.Vec3d(*up.tolist()))
    return v.GetInverse()


def set_camera_xform(stage, cam_path, eye, tgt):
    cp = stage.GetPrimAtPath(cam_path)
    if not cp.IsValid():
        return
    xf = UsdGeom.Xformable(cp)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(look_at(eye, tgt))


# ── Scene setup ──────────────────────────────────────────────────────────────
def _create_procedural_table(stage, goal_x, goal_y, yaw_deg=90.0):
    """Create a physical table at (goal_x, goal_y) rotated by yaw_deg around Z."""
    from pxr import Sdf, UsdShade, UsdPhysics
    table_path = "/World/GoalTable"
    existing = stage.GetPrimAtPath(table_path)
    if existing.IsValid():
        stage.RemovePrim(table_path)
    table_prim = stage.DefinePrim(table_path, "Xform")
    xf = UsdGeom.Xformable(table_prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(goal_x, goal_y, 0.0))
    xf.AddRotateZOp().Set(float(yaw_deg))

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

    # Add physics collision to all table parts
    for part_path in all_parts:
        part_prim = stage.GetPrimAtPath(part_path)
        UsdPhysics.CollisionAPI.Apply(part_prim)

    mat = UsdShade.Material.Define(stage, f"{table_path}/TableMaterial")
    shader = UsdShade.Shader.Define(stage, f"{table_path}/TableMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.30, 0.15))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for part_path in all_parts:
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(part_path)).Bind(mat)
    print(f"[collect] Created table at ({goal_x:.1f}, {goal_y:.1f}) rotated {yaw_deg:.0f}deg [with collision]")


def _build_hospital_scene(stage):
    """Build a hospital corridor scene using procedural geometry + Nucleus props."""
    from pxr import Sdf, UsdShade, UsdPhysics

    root = "/World/Hospital"
    stage.DefinePrim(root, "Xform")

    # --- Floor (white tiles) ---
    floor = UsdGeom.Cube.Define(stage, f"{root}/Floor")
    fxf = UsdGeom.Xformable(floor.GetPrim())
    fxf.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, -0.01))
    fxf.AddScaleOp().Set(Gf.Vec3f(8.0, 5.0, 0.01))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    # Floor material (light gray tiles)
    fmat = UsdShade.Material.Define(stage, f"{root}/FloorMat")
    fsh = UsdShade.Shader.Define(stage, f"{root}/FloorMat/Shader")
    fsh.CreateIdAttr("UsdPreviewSurface")
    fsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.87, 0.85))
    fsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    fmat.CreateSurfaceOutput().ConnectToSource(fsh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(floor.GetPrim()).Bind(fmat)

    # --- Walls ---
    wall_mat = UsdShade.Material.Define(stage, f"{root}/WallMat")
    wsh = UsdShade.Shader.Define(stage, f"{root}/WallMat/Shader")
    wsh.CreateIdAttr("UsdPreviewSurface")
    wsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.92, 0.92, 0.90))
    wsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    wall_mat.CreateSurfaceOutput().ConnectToSource(wsh.ConnectableAPI(), "surface")

    walls = [
        ("BackWall",  (3.0, -5.0, 1.5), (8.0, 0.05, 1.5)),
        ("FrontWall", (3.0, 5.0, 1.5),  (8.0, 0.05, 1.5)),
        ("LeftWall",  (-5.0, 0.0, 1.5), (0.05, 5.0, 1.5)),
        ("RightWall", (11.0, 0.0, 1.5), (0.05, 5.0, 1.5)),
    ]
    for name, pos, scale in walls:
        w = UsdGeom.Cube.Define(stage, f"{root}/{name}")
        wxf = UsdGeom.Xformable(w.GetPrim())
        wxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        wxf.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdPhysics.CollisionAPI.Apply(w.GetPrim())
        UsdShade.MaterialBindingAPI(w.GetPrim()).Bind(wall_mat)

    # --- Baseboard (green stripe at bottom of walls, typical hospital) ---
    bb_mat = UsdShade.Material.Define(stage, f"{root}/BaseboardMat")
    bbsh = UsdShade.Shader.Define(stage, f"{root}/BaseboardMat/Shader")
    bbsh.CreateIdAttr("UsdPreviewSurface")
    bbsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.35, 0.55, 0.45))
    bbsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    bb_mat.CreateSurfaceOutput().ConnectToSource(bbsh.ConnectableAPI(), "surface")
    baseboards = [
        ("BB_Back",  (3.0, -4.95, 0.1), (8.0, 0.02, 0.1)),
        ("BB_Front", (3.0, 4.95, 0.1),  (8.0, 0.02, 0.1)),
        ("BB_Left",  (-4.95, 0.0, 0.1), (0.02, 5.0, 0.1)),
        ("BB_Right", (10.95, 0.0, 0.1), (0.02, 5.0, 0.1)),
    ]
    for name, pos, scale in baseboards:
        bb = UsdGeom.Cube.Define(stage, f"{root}/{name}")
        bxf = UsdGeom.Xformable(bb.GetPrim())
        bxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        bxf.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdShade.MaterialBindingAPI(bb.GetPrim()).Bind(bb_mat)

    # --- Ceiling ---
    ceil = UsdGeom.Cube.Define(stage, f"{root}/Ceiling")
    cxf = UsdGeom.Xformable(ceil.GetPrim())
    cxf.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 3.01))
    cxf.AddScaleOp().Set(Gf.Vec3f(8.0, 5.0, 0.01))
    ceil_mat = UsdShade.Material.Define(stage, f"{root}/CeilMat")
    csh = UsdShade.Shader.Define(stage, f"{root}/CeilMat/Shader")
    csh.CreateIdAttr("UsdPreviewSurface")
    csh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))
    ceil_mat.CreateSurfaceOutput().ConnectToSource(csh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(ceil.GetPrim()).Bind(ceil_mat)

    # --- Hospital props from Nucleus ---
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.nucleus import get_assets_root_path
        assets = get_assets_root_path()
        props = [
            ("/Isaac/Environments/Hospital/Props/SM_HospitalBed_01b.usd",
             f"{root}/Bed1", (8.0, -3.5, 0.0), 90.0),
            ("/Isaac/Environments/Hospital/Props/SM_WheelChair_01a.usd",
             f"{root}/Wheelchair", (-3.0, -3.0, 0.0), 45.0),
            ("/Isaac/Environments/Hospital/Props/SM_MedicalCabinet_01a.usd",
             f"{root}/Cabinet", (-4.5, 0.0, 0.0), 90.0),
            ("/Isaac/Environments/Hospital/Props/SM_SupplyCart_01e.usd",
             f"{root}/Cart", (8.0, 3.0, 0.0), -90.0),
            ("/Isaac/Environments/Hospital/Props/SM_GasCart_01b.usd",
             f"{root}/GasCart", (-3.5, 3.5, 0.0), 0.0),
        ]
        for usd_path, prim_path, pos, yaw_deg in props:
            try:
                add_reference_to_stage(usd_path=assets + usd_path, prim_path=prim_path)
                pxf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
                pxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
                pxf.AddRotateZOp().Set(float(yaw_deg))
            except Exception as e:
                print(f"[WARN] Prop {usd_path}: {e}")
        print("[collect] Loaded hospital props")
    except Exception as e:
        print(f"[WARN] Hospital props: {e}")

    # --- Fluorescent ceiling lights ---
    for i, lx in enumerate([-1.0, 3.0, 7.0]):
        lpath = f"{root}/CeilingLight_{i}"
        lp = stage.DefinePrim(lpath, "RectLight")
        lxf = UsdGeom.Xformable(lp)
        lxf.AddTranslateOp().Set(Gf.Vec3d(lx, 0.0, 2.95))
        lxf.AddRotateXOp().Set(-90.0)
        UsdLux.RectLight(lp).CreateIntensityAttr(5000.0)
        UsdLux.RectLight(lp).CreateWidthAttr(0.3)
        UsdLux.RectLight(lp).CreateHeightAttr(2.0)
        UsdLux.RectLight(lp).CreateColorAttr(Gf.Vec3f(0.95, 0.98, 1.0))

    print("[collect] Built hospital scene (procedural room + props)")


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
            print(f"[collect] Loaded warehouse")
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

    elif env_type == "office":
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd",
                prim_path="/World/Environment")
            print("[collect] Loaded office")
        except Exception as e:
            print(f"[WARN] Office: {e}")

    elif env_type == "digital_warehouse":
        try:
            from isaacsim.core.utils.stage import add_reference_to_stage
            add_reference_to_stage(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd",
                prim_path="/World/Environment")
            print("[collect] Loaded digital twin warehouse")
        except Exception as e:
            print(f"[WARN] Digital Twin Warehouse: {e}")

    if not no_table:
        _create_procedural_table(stage, table_x, table_y, table_yaw_deg)

    light_path = "/World/ExtraDomeLight"
    if not stage.GetPrimAtPath(light_path).IsValid():
        light = stage.DefinePrim(light_path, "DomeLight")
        UsdLux.DomeLight(light).CreateIntensityAttr(3000.0)
        UsdLux.DomeLight(light).CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))


def enable_ray_tracing():
    """Enable RTX path tracing with default Isaac Sim settings."""
    import carb.settings
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    print("[collect] RTX Path Tracing enabled (default settings)")


def teleport_robot(robot, x, y, yaw, device):
    """Teleport robot to (x, y, SPAWN_HEIGHT) with given yaw. Reset joints."""
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Environment ──────────────────────────────────────────────────────
    has_cameras = getattr(args_cli, "enable_cameras", False)
    if has_cameras:
        env_cfg = G1EnvWithCamCfg()
    else:
        env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedEnv(cfg=env_cfg)
    device = env.device
    print(f"[collect] Environment ready | device={device} | cameras={has_cameras}")

    try:
        import PIL.Image
        _HAS_PIL = True
    except ImportError:
        _HAS_PIL = False

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    load_scene(args_cli.env, args_cli.table_x, args_cli.table_y, args_cli.table_yaw,
               no_table=args_cli.no_table)

    if args_cli.rtx:
        enable_ray_tracing()

    # ── WBC controller ───────────────────────────────────────────────────
    wbc = G1WBCController(num_envs=1)

    # ── Reset + teleport to start pose ───────────────────────────────────
    obs, _ = env.reset()
    wbc.reset()
    robot = env.scene["robot"]

    start_x, start_y, start_yaw = args_cli.start_x, args_cli.start_y, args_cli.start_yaw
    goal_x, goal_y, goal_yaw = args_cli.goal_x, args_cli.goal_y, args_cli.goal_yaw

    # Print sim joint names + find waist indices for posture correction
    jnames = robot.data.joint_names
    waist_pitch_idx = None
    waist_roll_idx = None
    for i, n in enumerate(jnames):
        if n == "waist_pitch_joint":
            waist_pitch_idx = i
        elif n == "waist_roll_joint":
            waist_roll_idx = i
    print(f"[collect] Joints: {len(jnames)} | waist_pitch={waist_pitch_idx} waist_roll={waist_roll_idx}")

    # ── Task camera (1080px) for start/goal images ─────────────────────
    rgb_task = None
    if has_cameras and _HAS_REPLICATOR and _HAS_PIL:
        task_cam_path = "/World/TaskCamera"
        task_prim = stage.DefinePrim(task_cam_path, "Camera")
        task_cam = UsdGeom.Camera(task_prim)
        task_cam.CreateFocalLengthAttr(24.0)
        task_cam.CreateHorizontalApertureAttr(36.0)
        task_cam.CreateVerticalApertureAttr(36.0)
        task_cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
        rp_task = rep.create.render_product(task_cam_path, (1080, 1080))
        rgb_task = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_task.attach([rp_task])

    teleport_robot(robot, start_x, start_y, start_yaw, device)
    # Warmup: 60 steps with stand policy — lets physics + rendering fully settle
    for _ in range(60):
        wbc.set_velocity(0.0, 0.0, 0.0)
        targets = wbc.step(robot.data, device=str(device))
        obs, _ = env.step(targets)

    # Capture start image: make robot invisible, run WBC stand to keep physics stable
    if rgb_task is not None and _HAS_PIL:
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        UsdGeom.Imageable(robot_prim).MakeInvisible()
        eye_h = 1.0
        start_eye = np.array([start_x, start_y, eye_h])
        start_tgt = np.array([args_cli.table_x, args_cli.table_y, 0.75])
        set_camera_xform(stage, task_cam_path, start_eye, start_tgt)
        # Run 30 steps with WBC stand to let renderer converge without robot falling
        for _ in range(30):
            wbc.set_velocity(0.0, 0.0, 0.0)
            targets = wbc.step(robot.data, device=str(device))
            env.step(targets)
        task_data = rgb_task.get_data()
        if task_data is not None and task_data.size > 0:
            PIL.Image.fromarray(task_data[:, :, :3]).save(str(out_dir / "start_image.png"))
            print("[collect] Saved start_image.png (1080px, task cam)")
        UsdGeom.Imageable(robot_prim).MakeVisible()

    print(f"[collect] Start pose: ({start_x:.1f}, {start_y:.1f}, yaw={start_yaw:.2f})")
    print(f"[collect] Goal pose:  ({goal_x:.1f}, {goal_y:.1f}, yaw={goal_yaw:.2f})")

    # ── Goal controller ──────────────────────────────────────────────────
    nav = GoalPoseController(
        goal_x, goal_y, goal_yaw,
        start_x=start_x, start_y=start_y,
        table_x=args_cli.table_x, table_y=args_cli.table_y,
        table_yaw_deg=args_cli.table_yaw,
        max_vx=args_cli.max_speed,
    )

    # ── Third-person camera ──────────────────────────────────────────────
    rgb_tp = None
    if has_cameras and _HAS_REPLICATOR:
        tp_path = "/World/ThirdPersonCamera"
        tp_prim = stage.DefinePrim(tp_path, "Camera")
        tp_cam = UsdGeom.Camera(tp_prim)
        tp_cam.CreateFocalLengthAttr(24.0)
        tp_cam.CreateHorizontalApertureAttr(36.0)
        tp_cam.CreateVerticalApertureAttr(36.0)
        tp_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 2000.0))
        rp_tp = rep.create.render_product(tp_path, (args_cli.res, args_cli.res))
        rgb_tp = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_tp.attach([rp_tp])

    pov_dir = out_dir / "frames_pov"
    fpv_dir = out_dir / "frames_fpv"
    tp_dir = out_dir / "frames_third_person"
    if has_cameras:
        pov_dir.mkdir(parents=True, exist_ok=True)
        fpv_dir.mkdir(parents=True, exist_ok=True)
        tp_dir.mkdir(parents=True, exist_ok=True)

    # ── Data storage ─────────────────────────────────────────────────────
    traj_csv_f = open(out_dir / "trajectory.csv", "w", newline="")
    writer = csv.writer(traj_csv_f)
    writer.writerow(["frame", "t", "vx", "vy", "yaw_rate",
                      "x", "y", "heading", "qw", "qx", "qy", "qz", "z", "phase"])
    joint_states_list = []
    vel_log = []  # for velocity_commands.csv
    frame_count = 0

    # ── Main loop ────────────────────────────────────────────────────────
    MAX_STEPS = int(args_cli.max_time * CONTROL_HZ)
    print(f"[collect] Running (max {MAX_STEPS} steps = {args_cli.max_time}s)...")

    bp = robot.data.root_link_pos_w[0].cpu().numpy()
    start_pos = bp.copy()

    for step in range(MAX_STEPS):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            bp = robot.data.root_link_pos_w[0].cpu().numpy()
            bq = robot.data.root_link_quat_w[0].cpu().numpy()
            heading = quat_to_heading(bq)

            vx, vy, vyaw, phase, done = nav.compute(
                float(bp[0]), float(bp[1]), heading)

            wbc.set_velocity(vx, vy, vyaw)
            targets = wbc.step(robot.data, device=str(device))
            # Lock waist pitch during WALK and ORIENT — prevents forward lean
            # Leave unlocked during STAND so balance policy can work
            if phase in (GoalPoseController.WALK, GoalPoseController.ORIENT) and waist_pitch_idx is not None:
                targets[0, waist_pitch_idx] = 0.0

            obs, _ = env.step(targets)

        # ── Capture ──────────────────────────────────────────────────
        if step % CAPTURE_INTERVAL == 0:
            bp = robot.data.root_link_pos_w[0].cpu().numpy()
            bq = robot.data.root_link_quat_w[0].cpu().numpy()
            heading = quat_to_heading(bq)
            jp = robot.data.joint_pos[0].cpu().numpy()

            t = frame_count / args_cli.fps
            writer.writerow([
                frame_count, f"{t:.4f}",
                f"{vx:.4f}", f"{vy:.4f}", f"{vyaw:.4f}",
                f"{bp[0]:.4f}", f"{bp[1]:.4f}", f"{heading:.4f}",
                f"{bq[0]:.6f}", f"{bq[1]:.6f}", f"{bq[2]:.6f}", f"{bq[3]:.6f}",
                f"{bp[2]:.4f}", phase,
            ])
            vel_log.append([frame_count, f"{vx:.4f}", f"{vy:.4f}", f"{vyaw:.4f}"])
            joint_states_list.append(jp.copy())

            if has_cameras and _HAS_PIL:
                ego_cam = env.scene.sensors["ego_cam"]
                ego_rgb = ego_cam.data.output["rgb"]
                if ego_rgb is not None and ego_rgb.numel() > 0:
                    img = ego_rgb[0].cpu().numpy()
                    if img.shape[-1] == 4:
                        img = img[:, :, :3]
                    PIL.Image.fromarray(img).save(
                        str(pov_dir / f"frame_{frame_count:06d}.png"))

                fpv_cam = env.scene.sensors["fpv_cam"]
                fpv_rgb = fpv_cam.data.output["rgb"]
                if fpv_rgb is not None and fpv_rgb.numel() > 0:
                    img = fpv_rgb[0].cpu().numpy()
                    if img.shape[-1] == 4:
                        img = img[:, :, :3]
                    PIL.Image.fromarray(img).save(
                        str(fpv_dir / f"frame_{frame_count:06d}.png"))

                if rgb_tp is not None:
                    pos = np.array([float(bp[0]), float(bp[1]), float(bp[2])])
                    fwd = qrot(bq, [1., 0., 0.])
                    tp_eye = pos - fwd * 2.0 + np.array([0., 0., 1.0])
                    tp_tgt = pos + np.array([0., 0., 0.5])
                    set_camera_xform(stage, "/World/ThirdPersonCamera", tp_eye, tp_tgt)
                    tp_data = rgb_tp.get_data()
                    if tp_data is not None and tp_data.size > 0:
                        PIL.Image.fromarray(tp_data[:, :, :3]).save(
                            str(tp_dir / f"frame_{frame_count:06d}.png"))

            frame_count += 1

        if step % 50 == 0:
            bp = robot.data.root_link_pos_w[0].cpu().numpy()
            dist = math.sqrt((goal_x - bp[0])**2 + (goal_y - bp[1])**2)
            print(f"  step {step:4d}  [{phase:5s}]  cmd=[{vx:+.3f},{vy:+.3f},{vyaw:+.3f}]"
                  f"  pos=[{bp[0]:+.2f},{bp[1]:+.2f},{bp[2]:+.2f}]"
                  f"  dist={dist:.2f}  frames={frame_count}")

        if done:
            print(f"[collect] DONE at step {step}")
            break

    # ── Post-task recording: 2 more seconds of standing ──────────────────
    POST_FRAMES = int(2.0 * CONTROL_HZ)
    print(f"[collect] Recording {POST_FRAMES} post-task frames...")
    for pf in range(POST_FRAMES):
        wbc.set_velocity(0.0, 0.0, 0.0)
        targets = wbc.step(robot.data, device=str(device))
        obs, _ = env.step(targets)

        if pf % CAPTURE_INTERVAL == 0:
            bp = robot.data.root_link_pos_w[0].cpu().numpy()
            bq = robot.data.root_link_quat_w[0].cpu().numpy()
            jp = robot.data.joint_pos[0].cpu().numpy()
            heading = quat_to_heading(bq)

            writer.writerow([
                f"{(step + 1 + pf) / CONTROL_HZ:.4f}",
                "0.0000", "0.0000", "0.0000",
                f"{bp[0]:.4f}", f"{bp[1]:.4f}",
                f"{bq[0]:.6f}", f"{bq[1]:.6f}", f"{bq[2]:.6f}", f"{bq[3]:.6f}",
                f"{bp[2]:.4f}", "done",
            ])
            vel_log.append([frame_count, "0.0000", "0.0000", "0.0000"])
            joint_states_list.append(jp.copy())

            if has_cameras and _HAS_PIL:
                ego_rgb = env.scene.sensors["ego_cam"].data.output["rgb"]
                if ego_rgb is not None and ego_rgb.numel() > 0:
                    img = ego_rgb[0].cpu().numpy()
                    if img.shape[-1] == 4: img = img[:, :, :3]
                    PIL.Image.fromarray(img).save(str(pov_dir / f"frame_{frame_count:06d}.png"))

                fpv_rgb = env.scene.sensors["fpv_cam"].data.output["rgb"]
                if fpv_rgb is not None and fpv_rgb.numel() > 0:
                    img = fpv_rgb[0].cpu().numpy()
                    if img.shape[-1] == 4: img = img[:, :, :3]
                    PIL.Image.fromarray(img).save(str(fpv_dir / f"frame_{frame_count:06d}.png"))

                if rgb_tp is not None:
                    pos = np.array([float(bp[0]), float(bp[1]), float(bp[2])])
                    fwd = qrot(bq, [1., 0., 0.])
                    tp_eye = pos - fwd * 2.0 + np.array([0., 0., 1.0])
                    tp_tgt = pos + np.array([0., 0., 0.5])
                    set_camera_xform(stage, "/World/ThirdPersonCamera", tp_eye, tp_tgt)
                    tp_data = rgb_tp.get_data()
                    if tp_data is not None and tp_data.size > 0:
                        PIL.Image.fromarray(tp_data[:, :, :3]).save(
                            str(tp_dir / f"frame_{frame_count:06d}.png"))

            frame_count += 1

    # ── Save ─────────────────────────────────────────────────────────────
    traj_csv_f.close()
    np.save(str(out_dir / "joint_states.npy"), np.array(joint_states_list))

    # Velocity commands CSV
    with open(out_dir / "velocity_commands.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "vx_m_s", "vy_m_s", "yaw_rate_rad_s"])
        for row in vel_log:
            w.writerow(row)

    # Read final position BEFORE hiding robot for goal image
    final_pos = robot.data.root_link_pos_w[0].cpu().numpy()
    final_heading = quat_to_heading(robot.data.root_link_quat_w[0].cpu().numpy())

    # Goal image: 1080px, from 0.5m before goal, robot made invisible
    if rgb_task is not None and _HAS_PIL:
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        UsdGeom.Imageable(robot_prim).MakeInvisible()
        approach_dx = goal_x - start_x
        approach_dy = goal_y - start_y
        approach_len = math.sqrt(approach_dx**2 + approach_dy**2)
        if approach_len > 0.01:
            dir_x = approach_dx / approach_len
            dir_y = approach_dy / approach_len
            cam_x = goal_x - 0.5 * dir_x
            cam_y = goal_y - 0.5 * dir_y
        else:
            cam_x, cam_y = goal_x - 0.5, goal_y
        eye_h = 1.0
        goal_eye = np.array([cam_x, cam_y, eye_h])
        goal_tgt = np.array([args_cli.table_x, args_cli.table_y, 0.75])
        set_camera_xform(stage, "/World/TaskCamera", goal_eye, goal_tgt)
        for _ in range(30):
            wbc.set_velocity(0.0, 0.0, 0.0)
            targets = wbc.step(robot.data, device=str(device))
            env.step(targets)
        task_data = rgb_task.get_data()
        if task_data is not None and task_data.size > 0:
            PIL.Image.fromarray(task_data[:, :, :3]).save(str(out_dir / "goal_image.png"))
            print("[collect] Saved goal_image.png (1080px, 0.5m before goal)")
    pos_err = math.sqrt((goal_x - final_pos[0])**2 + (goal_y - final_pos[1])**2)
    yaw_err = abs(math.atan2(math.sin(goal_yaw - final_heading),
                              math.cos(goal_yaw - final_heading)))

    metadata = {
        "robot": "unitree_g1",
        "controller": "GoalPoseController + wbc_onnx",
        "environment": args_cli.env,
        "start_pose": {"x": start_x, "y": start_y, "yaw": start_yaw},
        "goal_pose": {"x": goal_x, "y": goal_y, "yaw": goal_yaw},
        "final_pos": final_pos.tolist(),
        "final_heading": round(final_heading, 4),
        "position_error_m": round(pos_err, 4),
        "yaw_error_rad": round(yaw_err, 4),
        "total_frames": frame_count,
        "fps": args_cli.fps,
        "max_speed": args_cli.max_speed,
        "duration_sec": round(frame_count / args_cli.fps, 3),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    env.close()
    dist_walked = np.linalg.norm(final_pos[:2] - start_pos[:2])
    print(f"\n[collect] Done. {frame_count} frames saved.")
    print(f"  Start: ({start_x:+.2f}, {start_y:+.2f}, yaw={start_yaw:.2f})")
    print(f"  Final: ({final_pos[0]:+.2f}, {final_pos[1]:+.2f}, yaw={final_heading:.2f})")
    print(f"  Goal:  ({goal_x:+.2f}, {goal_y:+.2f}, yaw={goal_yaw:.2f})")
    print(f"  Pos error: {pos_err:.3f}m | Yaw error: {math.degrees(yaw_err):.1f}deg")
    print(f"  Distance walked: {dist_walked:.2f}m")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    run()
    simulation_app.close()
