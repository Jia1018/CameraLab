#!/usr/bin/env python3
"""Generate a broad Blender-only combinatorial preview bank.

This creates many unique clips plus 100-200 pair relationships. It is designed
for breadth and fast review, not final visual quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "site" / "assets" / "runs"
BLENDER_VERSION = "3.6.5"
FPS = 24
WIDTH = 320
HEIGHT = 240
SAMPLES = 1
SEED = 20260530

CAMERA_PRIMITIVES = [
    "Dolly_In", "Dolly_Out", "Pedestal_Up", "Pedestal_Down", "Truck_Right", "Truck_Left",
    "Pan_Right", "Pan_Left", "Tilt_Up", "Tilt_Down", "Roll_Clockwise", "Roll_Counterclockwise",
    "Zoom_In", "Zoom_Out", "Static",
]

CAMERA_PROFILES = [
    {"id": "cam_static", "primitives": ["Static"], "speed": "none", "base_pos": (0.0, -5.6, 2.1), "base_target": (0.0, 0.0, 0.55), "start_roll_deg": -3.0, "lens": 34.0, "motion": {}},
    {"id": "cam_dolly_in", "primitives": ["Dolly_In"], "speed": "ease_in_out", "base_pos": (0.7, -6.1, 2.0), "base_target": (0.15, 0.0, 0.50), "start_roll_deg": 1.5, "lens": 34.0, "motion": {"dy": (1.35, 0.30, 0.65, 2.00), "dtx": (0.02, 0.08, -0.16, 0.16), "droll": (0.0, 1.4, -3.0, 3.0), "dlens": (0.0, 1.2, -2.0, 3.0)}},
    {"id": "cam_dolly_out", "primitives": ["Dolly_Out"], "speed": "ease_in_out", "base_pos": (-0.7, -4.1, 2.05), "base_target": (-0.10, 0.0, 0.55), "start_roll_deg": -1.0, "lens": 35.0, "motion": {"dy": (-1.45, 0.35, -2.15, -0.70), "dtx": (0.0, 0.08, -0.16, 0.16), "droll": (0.0, 1.3, -3.0, 3.0), "dlens": (0.0, 1.0, -2.0, 2.0)}},
    {"id": "cam_truck_left_pan", "primitives": ["Truck_Left", "Pan_Right"], "speed": "ease_in_out", "base_pos": (1.9, -5.4, 2.1), "base_target": (-0.25, 0.0, 0.58), "start_roll_deg": 0.5, "lens": 34.0, "motion": {"dx": (-1.55, 0.35, -2.20, -0.70), "dtx": (0.70, 0.18, 0.25, 1.10), "droll": (0.0, 1.2, -2.5, 2.5)}},
    {"id": "cam_truck_right_pan", "primitives": ["Truck_Right", "Pan_Left"], "speed": "ease_in_out", "base_pos": (-1.9, -5.4, 2.1), "base_target": (0.25, 0.0, 0.58), "start_roll_deg": -0.5, "lens": 34.0, "motion": {"dx": (1.55, 0.35, 0.70, 2.20), "dtx": (-0.70, 0.18, -1.10, -0.25), "droll": (0.0, 1.2, -2.5, 2.5)}},
    {"id": "cam_crane_up_tilt", "primitives": ["Pedestal_Up", "Tilt_Down"], "speed": "slow", "base_pos": (0.8, -5.4, 1.55), "base_target": (0.15, 0.0, 0.78), "start_roll_deg": 0.0, "lens": 34.0, "motion": {"dz": (0.95, 0.25, 0.45, 1.45), "dtz": (-0.28, 0.10, -0.52, -0.08), "droll": (0.0, 1.0, -2.0, 2.0)}},
    {"id": "cam_pedestal_down_zoom", "primitives": ["Pedestal_Down", "Zoom_In"], "speed": "ease_in", "base_pos": (-0.9, -5.5, 3.0), "base_target": (-0.18, 0.0, 0.50), "start_roll_deg": 2.0, "lens": 31.0, "motion": {"dz": (-0.85, 0.22, -1.35, -0.35), "dtz": (0.18, 0.08, 0.02, 0.36), "dlens": (9.0, 2.5, 3.0, 14.0), "droll": (0.0, 1.0, -2.0, 2.0)}},
    {"id": "cam_orbit_soft", "primitives": ["Truck_Right", "Pan_Left", "Dolly_In"], "speed": "ease_in_out", "base_pos": (-1.3, -5.3, 2.15), "base_target": (0.0, 0.0, 0.58), "start_roll_deg": -1.0, "lens": 35.0, "motion": {"dx": (1.15, 0.30, 0.45, 1.80), "dy": (0.45, 0.18, 0.05, 0.85), "dtx": (-0.35, 0.16, -0.75, 0.05), "droll": (0.0, 1.4, -3.0, 3.0)}},
    {"id": "cam_top_down_drift", "primitives": ["Pedestal_Down", "Tilt_Down", "Truck_Right"], "speed": "ease_in_out", "base_pos": (-1.2, -0.65, 6.4), "base_target": (0.0, -0.05, 0.12), "start_roll_deg": 4.0, "lens": 42.0, "motion": {"dx": (1.45, 0.35, 0.65, 2.10), "dy": (0.22, 0.12, -0.05, 0.45), "dz": (-0.45, 0.16, -0.85, -0.10), "dtx": (0.0, 0.16, -0.35, 0.35), "dty": (0.18, 0.10, -0.05, 0.40), "droll": (0.0, 1.5, -3.0, 3.0)}},
    {"id": "cam_gentle_roll_cw", "primitives": ["Roll_Clockwise", "Pan_Left"], "speed": "ease_in_out", "base_pos": (0.7, -5.5, 2.05), "base_target": (0.12, 0.0, 0.56), "start_roll_deg": -4.0, "lens": 34.0, "motion": {"dx": (-0.45, 0.18, -0.85, -0.10), "dtx": (-0.18, 0.10, -0.40, 0.05), "droll": (5.0, 1.8, 1.5, 8.5), "roll_wobble": (0.6, 0.35, 0.0, 1.4)}},
    {"id": "cam_gentle_roll_ccw", "primitives": ["Roll_Counterclockwise", "Truck_Left"], "speed": "slow", "base_pos": (-0.7, -5.5, 2.25), "base_target": (-0.12, 0.0, 0.58), "start_roll_deg": 4.0, "lens": 34.0, "motion": {"dx": (0.45, 0.18, 0.10, 0.85), "dtx": (0.16, 0.10, -0.05, 0.38), "droll": (-5.0, 1.8, -8.5, -1.5), "roll_wobble": (0.6, 0.35, 0.0, 1.4)}},
    {"id": "cam_wobble_small", "primitives": ["Pan_Left", "Pan_Right", "Roll_Counterclockwise", "Zoom_Out"], "speed": "variable", "base_pos": (-0.4, -5.4, 2.05), "base_target": (0.0, 0.0, 0.55), "start_roll_deg": 2.0, "lens": 39.0, "motion": {"dtx": (0.0, 0.18, -0.35, 0.35), "dtz": (0.0, 0.10, -0.20, 0.20), "droll": (0.0, 1.6, -3.5, 3.5), "roll_wobble": (1.2, 0.45, 0.2, 2.2), "dlens": (-4.0, 1.4, -7.0, -1.0)}},
    {"id": "cam_out_of_frame_push", "primitives": ["Dolly_In", "Tilt_Up"], "speed": "fast", "base_pos": (1.0, -5.4, 1.95), "base_target": (0.25, 0.20, 0.66), "start_roll_deg": -2.0, "lens": 34.0, "motion": {"dy": (1.65, 0.35, 0.85, 2.35), "dz": (0.30, 0.12, 0.05, 0.60), "dty": (0.85, 0.24, 0.35, 1.35), "dtz": (0.24, 0.10, 0.05, 0.45), "droll": (0.0, 1.2, -2.5, 2.5)}},
    {"id": "cam_rare_fast_roll", "primitives": ["Roll_Clockwise", "Pan_Right"], "speed": "ease_in_out", "base_pos": (0.2, -5.3, 2.1), "base_target": (0.0, 0.0, 0.56), "start_roll_deg": -6.0, "lens": 34.0, "motion": {"dx": (0.35, 0.16, 0.05, 0.70), "dtx": (0.25, 0.12, 0.02, 0.52), "droll": (13.0, 2.5, 9.0, 18.0)}},
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sample_gaussian(rng: random.Random, spec: tuple[float, float, float, float] | None) -> float:
    if spec is None:
        return 0.0
    mean, std, low, high = spec
    return round(clamp(rng.gauss(mean, std), low, high), 5)


def sample_camera_specs(seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed + 17)
    cameras = []
    for profile in CAMERA_PROFILES:
        motion_dist = profile.get("motion", {})
        motion = {key: sample_gaussian(rng, motion_dist.get(key)) for key in ["dx", "dy", "dz", "dtx", "dty", "dtz", "droll", "roll_wobble", "dlens"]}
        camera = {
            "id": profile["id"],
            "primitives": profile["primitives"],
            "speed": profile["speed"],
            "start_position": [round(v, 5) for v in profile["base_pos"]],
            "start_target": [round(v, 5) for v in profile["base_target"]],
            "start_roll_deg": profile["start_roll_deg"],
            "start_lens": profile["lens"],
            "motion_sample": motion,
            "motion_distribution": motion_dist,
        }
        cameras.append(camera)
    return cameras


CAMERA_SPECS = sample_camera_specs(SEED)

PHYSICS_SPECS = [
    {"id": "phys_static_center", "kind": "static", "objects": 1, "speed": "none"},
    {"id": "phys_move_away", "kind": "move_away", "objects": 1, "speed": "medium"},
    {"id": "phys_move_toward", "kind": "move_toward", "objects": 1, "speed": "medium"},
    {"id": "phys_fast_diag_bounce", "kind": "diag_bounce", "objects": 1, "speed": "fast"},
    {"id": "phys_slow_roll_left", "kind": "roll_left", "objects": 1, "speed": "slow"},
    {"id": "phys_roll_right_fast", "kind": "roll_right", "objects": 1, "speed": "fast"},
    {"id": "phys_two_cross_collision", "kind": "two_cross", "objects": 2, "speed": "medium"},
    {"id": "phys_two_same_direction", "kind": "two_same_direction", "objects": 2, "speed": "mixed"},
    {"id": "phys_multi_swirl", "kind": "multi_swirl", "objects": 3, "speed": "mixed"},
    {"id": "phys_hit_static_block", "kind": "hit_static_block", "objects": 2, "speed": "fast"},
    {"id": "phys_out_of_frame", "kind": "out_of_frame", "objects": 1, "speed": "fast"},
    {"id": "phys_vertical_drop", "kind": "vertical_drop", "objects": 1, "speed": "variable"},
    {"id": "phys_confusing_move_back", "kind": "confusing_move_back", "objects": 1, "speed": "medium"},
    {"id": "phys_static_with_dynamic_bg", "kind": "static_with_dynamic_bg", "objects": 3, "speed": "slow"},
]

SCENE_SPECS = [
    {"id": "scene_gray_low_wall", "ground": (0.66, 0.68, 0.64, 1), "wall": (0.78, 0.79, 0.76, 1), "side": (0.70, 0.72, 0.72, 1), "wall_height": 2.0, "style": "plain"},
    {"id": "scene_blue_tall_wall", "ground": (0.62, 0.67, 0.69, 1), "wall": (0.70, 0.77, 0.82, 1), "side": (0.64, 0.70, 0.74, 1), "wall_height": 2.8, "style": "horizontal_bands"},
    {"id": "scene_warm_panel_wall", "ground": (0.69, 0.66, 0.60, 1), "wall": (0.80, 0.75, 0.67, 1), "side": (0.73, 0.69, 0.64, 1), "wall_height": 2.35, "style": "wall_panels"},
    {"id": "scene_green_floor_marks", "ground": (0.60, 0.68, 0.62, 1), "wall": (0.74, 0.80, 0.73, 1), "side": (0.67, 0.73, 0.67, 1), "wall_height": 2.15, "style": "floor_marks"},
    {"id": "scene_dark_side_blocks", "ground": (0.58, 0.60, 0.62, 1), "wall": (0.72, 0.73, 0.75, 1), "side": (0.52, 0.54, 0.57, 1), "wall_height": 2.55, "style": "side_blocks"},
    {"id": "scene_red_wall_stripes", "ground": (0.66, 0.64, 0.62, 1), "wall": (0.80, 0.70, 0.68, 1), "side": (0.72, 0.66, 0.65, 1), "wall_height": 2.25, "style": "vertical_stripes"},
]
COLORS = [(0.86, 0.24, 0.18, 1), (0.12, 0.52, 0.46, 1), (0.18, 0.35, 0.75, 1), (0.95, 0.62, 0.18, 1), (0.48, 0.28, 0.74, 1)]


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    camera: dict
    physics: dict
    scene: dict
    frames: int


def ease(t: float, mode: str) -> float:
    if mode in {"slow", "medium", "fast", "none"}:
        return t
    if mode == "ease_in":
        return t * t
    if mode == "ease_in_out":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if mode == "variable":
        return max(0.0, min(1.0, t + 0.08 * math.sin(4.0 * math.pi * t)))
    return t


def make_mat(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def reset_scene(frames: int) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames
    bpy.context.scene.render.fps = FPS
    bpy.context.scene.render.resolution_x = WIDTH
    bpy.context.scene.render.resolution_y = HEIGHT
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.samples = SAMPLES
    bpy.context.scene.world.color = (0.76, 0.80, 0.84)
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


def look_at(obj: bpy.types.Object, target: Vector, roll: float = 0.0) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    obj.rotation_euler.rotate_axis("Z", roll)


def add_cube(name: str, loc: tuple[float, float, float], scale: tuple[float, float, float], color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    obj.data.materials.append(make_mat(f"mat_{name}", color))
    return obj


def add_sphere(name: str, loc: tuple[float, float, float], radius: float, color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(make_mat(f"mat_{name}", color))
    return obj


def add_cylinder(name: str, loc: tuple[float, float, float], radius: float, depth: float, color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(make_mat(f"mat_{name}", color))
    return obj


def add_world(scene: dict) -> None:
    wall_height = scene["wall_height"]
    add_cube("ground", (0, 0, -0.03), (9, 9, 0.06), scene["ground"])
    add_cube("back_wall", (0, 3.35, wall_height / 2), (9, 0.08, wall_height), scene["wall"])
    add_cube("left_wall", (-4.25, 0, wall_height / 2), (0.08, 6.5, wall_height), scene["side"])

    style = scene["style"]
    if style == "horizontal_bands":
        for i, z in enumerate([0.55, 1.15, 1.75, 2.35]):
            add_cube(f"wall_band_{i}", (0, 3.295, z), (8.3, 0.035, 0.035), (0.48, 0.57, 0.62, 1))
    elif style == "wall_panels":
        for i, x in enumerate([-2.8, -1.4, 0.0, 1.4, 2.8]):
            add_cube(f"back_panel_{i}", (x, 3.29, 1.05), (0.72, 0.035, 1.45), (0.70, 0.63, 0.54, 1))
    elif style == "floor_marks":
        for i, (x, y) in enumerate([(-3.2, -2.3), (-2.4, 2.35), (2.6, -2.1), (3.2, 1.85)]):
            add_cylinder(f"floor_mark_{i}", (x, y, 0.012), 0.07, 0.018, (0.34, 0.43, 0.36, 1))
    elif style == "side_blocks":
        for i, (x, y, sx, sy) in enumerate([(-3.55, -2.15, 0.45, 0.75), (3.45, -1.85, 0.55, 0.55), (-3.35, 2.35, 0.50, 0.45), (3.55, 2.25, 0.42, 0.70)]):
            add_cube(f"side_low_block_{i}", (x, y, 0.14), (sx, sy, 0.28), (0.38, 0.40, 0.42, 1))
    elif style == "vertical_stripes":
        for i, x in enumerate([-3.6, -2.4, -1.2, 0.0, 1.2, 2.4, 3.6]):
            add_cube(f"wall_stripe_{i}", (x, 3.29, wall_height / 2), (0.08, 0.035, wall_height * 0.84), (0.64, 0.43, 0.42, 1))

    bpy.ops.object.light_add(type="AREA", location=(0, -4.0, 5.0))
    light = bpy.context.object
    light.name = "key_light"
    light.data.energy = 520
    light.data.size = 5.5
    bpy.ops.object.light_add(type="POINT", location=(-3.2, 1.2, 3.5))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 70


def create_dynamic_objects(physics: dict) -> list[bpy.types.Object]:
    kind = physics["kind"]
    objects = []
    if kind in {"static", "move_away", "move_toward", "diag_bounce", "out_of_frame", "vertical_drop", "confusing_move_back"}:
        objects.append(add_sphere("main_sphere", (0, 0, 0.28), 0.28, COLORS[0]))
    elif kind in {"roll_left", "roll_right"}:
        objects.append(add_cube("main_cube", (0, 0, 0.25), (0.5, 0.5, 0.5), COLORS[1]))
    elif kind == "two_cross":
        objects.append(add_sphere("red_cross", (-2, -0.9, 0.25), 0.24, COLORS[0]))
        objects.append(add_sphere("blue_cross", (2, 0.9, 0.25), 0.24, COLORS[2]))
    elif kind == "two_same_direction":
        objects.append(add_sphere("slow_lead", (-2, -0.6, 0.25), 0.24, COLORS[3]))
        objects.append(add_cube("fast_follow", (-3.0, 0.2, 0.25), (0.45, 0.45, 0.45), COLORS[1]))
    elif kind == "multi_swirl":
        objects.append(add_sphere("orbit_gold", (1, 0, 0.24), 0.22, COLORS[3]))
        objects.append(add_cube("sweep_teal", (-1.2, 0, 0.24), (0.42, 0.42, 0.42), COLORS[1]))
        objects.append(add_sphere("bounce_violet", (0, -1.3, 0.22), 0.21, COLORS[4]))
    elif kind == "hit_static_block":
        objects.append(add_sphere("incoming_ball", (-2.8, -0.2, 0.24), 0.24, COLORS[0]))
        block = add_cube("target_static_block", (0.5, -0.2, 0.3), (0.55, 0.55, 0.6), (0.25, 0.27, 0.29, 1))
        objects.append(block)
    elif kind == "static_with_dynamic_bg":
        objects.append(add_sphere("foreground_static", (0, -0.7, 0.28), 0.28, COLORS[0]))
        objects.append(add_cube("background_slider", (-2.2, 1.4, 0.25), (0.45, 0.45, 0.45), COLORS[1]))
        objects.append(add_sphere("background_bouncer", (1.5, 1.2, 0.22), 0.22, COLORS[2]))
    else:
        raise ValueError(kind)
    return objects


def object_pose(kind: str, idx: int, t: float) -> tuple[Vector, tuple[float, float, float]]:
    if kind == "static":
        return Vector((0, -0.4, 0.28)), (0, 0, 0)
    if kind == "move_away":
        return Vector((0, -0.7 + 2.2 * t, 0.28)), (0, 0, 0)
    if kind == "move_toward":
        return Vector((0, 1.5 - 2.2 * t, 0.28)), (0, 0, 0)
    if kind == "confusing_move_back":
        return Vector((0, -0.2 + 1.8 * t, 0.28)), (0, 0, 0)
    if kind == "diag_bounce":
        return Vector((-2.7 + 5.4 * t, -1.6 + 3.2 * t, 0.28 + 0.9 * abs(math.sin(2.5 * math.pi * t)))), (0, 5 * math.pi * t, 0)
    if kind == "roll_left":
        return Vector((2.4 - 4.8 * t, -0.7, 0.25)), (0, -6 * math.pi * t, 0.2 * math.sin(2 * math.pi * t))
    if kind == "roll_right":
        return Vector((-2.4 + 4.8 * t, 0.55, 0.25)), (0, 8 * math.pi * t, 0)
    if kind == "out_of_frame":
        return Vector((-1.0 + 6.0 * t, -0.5 + 1.0 * t, 0.28)), (0, 0, 0)
    if kind == "vertical_drop":
        return Vector((0.3 * math.sin(2 * math.pi * t), 0.2, 1.8 - 1.5 * min(1.0, 1.4 * t) + 0.55 * abs(math.sin(4 * math.pi * t)) * (1 - t))), (0, 0, 0)
    if kind == "two_cross":
        if idx == 0:
            return Vector((-2.5 + 5.0 * t, -0.9 + 0.25 * math.sin(2 * math.pi * t), 0.25 + 0.35 * abs(math.sin(2 * math.pi * t)))), (0, 0, 0)
        return Vector((2.5 - 5.0 * t, 0.9 - 0.25 * math.sin(2 * math.pi * t), 0.25 + 0.25 * abs(math.sin(2 * math.pi * (t + 0.2))))), (0, 0, 0)
    if kind == "two_same_direction":
        if idx == 0:
            return Vector((-1.8 + 2.7 * t, -0.6, 0.24)), (0, 0, 0)
        x = -3.0 + 4.5 * t
        if 0.45 < t < 0.62:
            x -= 0.4 * math.sin(math.pi * (t - 0.45) / 0.17)
        return Vector((x, 0.2, 0.25)), (0, 5 * math.pi * t, 0)
    if kind == "multi_swirl":
        a = 2 * math.pi * t
        if idx == 0:
            return Vector((1.1 * math.cos(a), 1.1 * math.sin(a), 0.24)), (0, 0, 0)
        if idx == 1:
            return Vector((-1.8 + 3.6 * t, 0.6 * math.cos(a), 0.25)), (0.3 * a, 2.8 * a, 0)
        return Vector((0.6 * math.sin(a), -1.5 + 3.0 * t, 0.22 + 0.65 * abs(math.sin(3 * math.pi * t)))), (0, 0, 0)
    if kind == "hit_static_block":
        contact_t = 0.68
        contact_x = -0.02  # block min x (0.225) minus sphere radius (0.24), with a small visual gap
        if idx == 0:
            if t <= contact_t:
                u = t / contact_t
                x = -2.8 + (contact_x + 2.8) * u
            else:
                u = (t - contact_t) / (1.0 - contact_t)
                x = contact_x - 1.05 * u
            z = 0.24 + 0.13 * abs(math.sin(5 * math.pi * t)) * (1.0 - 0.35 * max(0.0, t - contact_t))
            return Vector((x, -0.2, z)), (0, 0, 0)
        block_shift = 0.32 * max(0.0, t - contact_t) / (1.0 - contact_t)
        return Vector((0.5 + block_shift, -0.2, 0.3)), (0, 0, 0.35 * block_shift)
    if kind == "static_with_dynamic_bg":
        if idx == 0:
            return Vector((0, -0.7, 0.28)), (0, 0, 0)
        if idx == 1:
            return Vector((-2.2 + 4.4 * t, 1.4, 0.25)), (0, 4 * math.pi * t, 0)
        return Vector((1.5, 1.2 - 1.8 * t, 0.22 + 0.55 * abs(math.sin(3 * math.pi * t)))), (0, 0, 0)
    raise ValueError(kind)


def camera_state(camera: dict, t: float) -> tuple[Vector, Vector, float, float]:
    u = ease(t, camera["speed"])
    motion = camera["motion_sample"]
    pos = Vector(camera["start_position"])
    target = Vector(camera["start_target"])
    pos += Vector((motion["dx"], motion["dy"], motion["dz"])) * u
    target += Vector((motion["dtx"], motion["dty"], motion["dtz"])) * u
    roll_deg = camera["start_roll_deg"] + motion["droll"] * u
    if motion["roll_wobble"]:
        roll_deg += motion["roll_wobble"] * math.sin(2.0 * math.pi * u)
    lens = camera["start_lens"] + motion["dlens"] * u
    return pos, target, math.radians(roll_deg), lens

def scene_for_physics(physics: dict) -> dict:
    # Same-physics pairs must keep object motion and scene identical. Binding scene
    # to physics_id makes that invariant explicit and easy to audit in manifests.
    idx = PHYSICS_SPECS.index(physics) % len(SCENE_SPECS)
    return SCENE_SPECS[idx]

def default_frames_for_physics(physics: dict) -> int:
    return [48, 60, 72, 84][PHYSICS_SPECS.index(physics) % 4]

def build_clip_specs(max_clips: int, seed: int) -> list[ClipSpec]:
    specs = []
    diagnostic_scene = SCENE_SPECS[0]
    forced = [
        ("ambiguous_cam_dolly_static", "cam_dolly_in", "phys_static_center", diagnostic_scene),
        ("ambiguous_static_obj_toward_camera", "cam_static", "phys_move_toward", diagnostic_scene),
        ("outframe_cam_push_obj_cross", "cam_out_of_frame_push", "phys_out_of_frame", None),
    ]
    for clip_id, cam_id, phys_id, scene in forced:
        phys = next(p for p in PHYSICS_SPECS if p["id"] == phys_id)
        specs.append(ClipSpec(clip_id, next(c for c in CAMERA_SPECS if c["id"] == cam_id), phys, scene or scene_for_physics(phys), default_frames_for_physics(phys)))
    used = {(spec.camera["id"], spec.physics["id"]) for spec in specs}
    round_idx = 0
    while len(specs) < max_clips:
        added = False
        for cam_idx, cam in enumerate(CAMERA_SPECS):
            if len(specs) >= max_clips:
                break
            phys = PHYSICS_SPECS[(cam_idx + round_idx) % len(PHYSICS_SPECS)]
            key = (cam["id"], phys["id"])
            if key in used:
                continue
            used.add(key)
            clip_id = f"clip_{len(specs):03d}_{cam['id']}_{phys['id']}"
            specs.append(ClipSpec(clip_id, cam, phys, scene_for_physics(phys), default_frames_for_physics(phys)))
            added = True
        if not added:
            break
        round_idx += 1
    return specs


def render_clip(spec: ClipSpec, run_dir: Path) -> dict[str, object]:
    reset_scene(spec.frames)
    add_world(spec.scene)
    objects = create_dynamic_objects(spec.physics)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = spec.camera["id"]
    bpy.context.scene.camera = camera
    records = []
    for frame in range(1, spec.frames + 1):
        t = (frame - 1) / max(1, spec.frames - 1)
        obj_records = []
        for idx, obj in enumerate(objects):
            loc, rot = object_pose(spec.physics["kind"], idx, t)
            obj.location = loc
            obj.rotation_euler = rot
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_euler", frame=frame)
            obj_records.append({"name": obj.name, "position": [round(v, 5) for v in obj.location], "rotation_euler": [round(v, 5) for v in obj.rotation_euler]})
        pos, target, roll, lens = camera_state(spec.camera, t)
        camera.location = pos
        camera.data.lens = lens
        look_at(camera, target, roll)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
        records.append({"frame": frame - 1, "t": round(t, 5), "camera": {"position": [round(v, 5) for v in pos], "look_at": [round(v, 5) for v in target], "roll": round(roll, 5), "lens": round(lens, 4), "primitives": spec.camera["primitives"]}, "objects": obj_records})

    frame_dir = run_dir / "frames" / spec.clip_id
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(frame_dir / "frame_")
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True, write_still=False)

    video_rel = Path("videos") / f"{spec.clip_id}.mp4"
    video_path = run_dir / video_rel
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frame_dir / "frame_%04d.png"), "-pix_fmt", "yuv420p", "-crf", "23", str(video_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    metadata_rel = Path("metadata") / f"{spec.clip_id}.json"
    metadata_path = run_dir / metadata_rel
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"clip_id": spec.clip_id, "camera_id": spec.camera["id"], "physics_id": spec.physics["id"], "scene_id": spec.scene["id"], "scene_style": spec.scene["style"], "fps": FPS, "frames_count": spec.frames, "resolution": [WIDTH, HEIGHT], "camera_primitives": spec.camera["primitives"], "camera_speed": spec.camera["speed"], "camera_start_position": spec.camera.get("start_position"), "camera_start_target": spec.camera.get("start_target"), "camera_start_roll_deg": spec.camera.get("start_roll_deg"), "camera_start_lens": spec.camera.get("start_lens"), "camera_motion_sample": spec.camera.get("motion_sample"), "camera_motion_distribution": spec.camera.get("motion_distribution"), "physics_kind": spec.physics["kind"], "physics_objects": spec.physics["objects"], "physics_speed": spec.physics["speed"], "video": str(video_rel), "frames": records}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"clip_id": spec.clip_id, "camera_id": spec.camera["id"], "physics_id": spec.physics["id"], "scene_id": spec.scene["id"], "scene_style": spec.scene["style"], "camera_primitives": spec.camera["primitives"], "pair_groups": [], "video": str(video_rel), "metadata": str(metadata_rel)}


def make_pairs(clips: list[dict[str, object]], target_pairs: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed + 99)
    by_cam = {}
    by_phys_scene = {}
    for clip in clips:
        by_cam.setdefault(clip["camera_id"], []).append(clip)
        by_phys_scene.setdefault((clip["physics_id"], clip["scene_id"]), []).append(clip)

    pairs = []
    seen_pairs = set()

    def add_pair(kind, title, a, b, controlled, varied, tags):
        if a["clip_id"] == b["clip_id"]:
            return
        key = tuple(sorted([a["clip_id"], b["clip_id"]]))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        group_id = f"pair_{len(pairs):03d}_{kind}"
        pairs.append({"group_id": group_id, "title": title, "controlled_factor": controlled, "varied_factor": varied, "clip_ids": [a["clip_id"], b["clip_id"]], "tags": tags})
        a["pair_groups"].append(group_id)
        b["pair_groups"].append(group_id)

    for cam, group in by_cam.items():
        candidates = [(a, b) for i, a in enumerate(group) for b in group[i + 1:] if a["physics_id"] != b["physics_id"] or a["scene_id"] != b["scene_id"]]
        rng.shuffle(candidates)
        for a, b in candidates[:10]:
            if len(pairs) >= target_pairs:
                return pairs
            add_pair("same_camera", f"Same camera {cam}, different physics/scene", a, b, "camera_id", "physics_id/scene_id", ["same_camera", "different_physics_or_scene"])

    for (phys, scene), group in by_phys_scene.items():
        candidates = [(a, b) for i, a in enumerate(group) for b in group[i + 1:] if a["camera_id"] != b["camera_id"]]
        rng.shuffle(candidates)
        for a, b in candidates[:10]:
            if len(pairs) >= target_pairs:
                return pairs
            add_pair("same_physics_scene", f"Same physics and scene {phys} / {scene}, different camera", a, b, "physics_id/scene_id", "camera_id", ["same_physics", "same_scene", "different_camera"])

    attempts = 0
    while len(pairs) < target_pairs and attempts < target_pairs * 50:
        attempts += 1
        a, b = rng.sample(clips, 2)
        tags = ["mixed_combo"]
        if a["camera_id"] != b["camera_id"]:
            tags.append("different_camera")
        if a["physics_id"] != b["physics_id"]:
            tags.append("different_physics")
        if a["scene_id"] != b["scene_id"]:
            tags.append("different_scene")
        add_pair("mixed", "Mixed camera/physics/scene comparison", a, b, "none", "camera_id/physics_id/scene_id", tags)
    return pairs


def update_index(run_id: str) -> None:
    index_path = RUN_ROOT / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        runs = [run for run in index.get("runs", []) if run.get("run_id") != run_id]
    else:
        runs = []
    runs.append({"run_id": run_id, "manifest": f"{run_id}/manifest.json"})
    index_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="blender_bank_v0")
    parser.add_argument("--clips", type=int, default=72)
    parser.add_argument("--pairs", type=int, default=180)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    run_dir = RUN_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = build_clip_specs(args.clips, SEED)
    clips = [render_clip(spec, run_dir) for spec in specs]
    pair_groups = make_pairs(clips, args.pairs, SEED)
    ambiguous_equivalence_groups = [{"group_id": "ambiguous_000_dolly_vs_object_toward_camera", "title": "Ambiguous appearance: camera dolly-in vs object moving toward camera", "clip_ids": ["ambiguous_cam_dolly_static", "ambiguous_static_obj_toward_camera"], "reason": "These clips can look similar in a single view: one changes apparent scale through camera motion, the other through physical object motion. They are diagnostic examples, not supervised pair samples, because both camera_id and physics_id differ.", "hidden_factor_difference": ["camera_motion", "physical_motion"], "intended_use": "diagnostic/evaluation example; disambiguate with additional same-camera or same-physics/scene pairs."}]
    manifest = {"project": "camera_motion_disentangle", "run_id": args.run_id, "generator": "blender_combinatorial_bank", "description": "Balanced longer-duration Blender-only preview bank with explicit scene_id. Same-physics pairs keep physics_id and scene_id fixed, while camera-only pairs keep camera_id fixed and vary physics/scene.", "blender_version": BLENDER_VERSION, "resolution": [WIDTH, HEIGHT], "fps": FPS, "cycles_samples": SAMPLES, "camera_primitives_reference": CAMERA_PRIMITIVES, "camera_motion_model": "per-axis Gaussian displacement/target/roll/lens samples with clipped tails", "camera_motion_profiles": CAMERA_PROFILES, "scene_reference": SCENE_SPECS, "clips": clips, "pair_groups": pair_groups, "ambiguous_equivalence_groups": ambiguous_equivalence_groups}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_index(args.run_id)
    print(f"Wrote {len(clips)} clips and {len(pair_groups)} pairs to {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
