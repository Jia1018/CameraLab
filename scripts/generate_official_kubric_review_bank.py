#!/usr/bin/env python3
"""Generate a small review bank with the official PyPI Kubric package.

This script keeps the existing GitHub Pages manifest shape, but uses official
Kubric objects, Kubric's PyBullet wrapper, and Kubric's Blender renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "site" / "assets" / "runs"
DEFAULT_BLENDER = Path("/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender")
DEFAULT_KUBRIC_SITE_PACKAGES = Path(
    "/workspace/writeable/environments/kubric_official/lib/python3.10/site-packages"
)

FPS = 24
PHYSICS_HZ = 240
FRAMES = 96
WIDTH = 640
HEIGHT = 480
SEED = 20260612

BODY_COLORS = {
    "red": [0.88, 0.20, 0.16, 1.0],
    "teal": [0.10, 0.56, 0.50, 1.0],
    "blue": [0.16, 0.34, 0.78, 1.0],
    "gold": [0.95, 0.62, 0.16, 1.0],
    "violet": [0.48, 0.30, 0.78, 1.0],
    "dark": [0.23, 0.25, 0.28, 1.0],
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def trunc_gauss(rng: random.Random, mean: float, std: float, low: float, high: float) -> float:
    return round(clamp(rng.gauss(mean, std), low, high), 5)


def vec(values: Any) -> list[float]:
    return [round(float(v), 5) for v in values]


def color_tuple(values: list[float]) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def world_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": "scene_gray_low_wall",
            "ground_color": [0.64, 0.66, 0.62, 1.0],
            "back_wall_color": [0.77, 0.78, 0.75, 1.0],
            "side_wall_color": [0.69, 0.71, 0.70, 1.0],
            "wall_height": 2.1,
            "wall_style": "plain",
            "floor_style": "subtle_marks",
            "static_props": [],
        },
        {
            "id": "scene_blue_tall_wall",
            "ground_color": [0.60, 0.66, 0.68, 1.0],
            "back_wall_color": [0.69, 0.77, 0.82, 1.0],
            "side_wall_color": [0.63, 0.69, 0.74, 1.0],
            "wall_height": 2.8,
            "wall_style": "horizontal_bands",
            "floor_style": "plain",
            "static_props": [],
        },
        {
            "id": "scene_warm_panel_wall",
            "ground_color": [0.68, 0.65, 0.59, 1.0],
            "back_wall_color": [0.80, 0.74, 0.66, 1.0],
            "side_wall_color": [0.72, 0.68, 0.63, 1.0],
            "wall_height": 2.35,
            "wall_style": "wall_panels",
            "floor_style": "plain",
            "static_props": [
                {
                    "name": "side_reference_block",
                    "shape": "box",
                    "half_extents": [0.26, 0.26, 0.15],
                    "position": [-3.25, 1.95, 0.15],
                    "color": [0.42, 0.39, 0.35, 1.0],
                    "collider": True,
                }
            ],
        },
    ]


def sphere(
    name: str,
    radius: float,
    position: tuple[float, float, float],
    color: str,
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    restitution: float = 0.55,
    friction: float = 0.35,
    static: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": "sphere",
        "radius": round(radius, 5),
        "scale": [round(radius, 5), round(radius, 5), round(radius, 5)],
        "position": vec(position),
        "velocity": vec(velocity),
        "angular_velocity": vec(angular_velocity),
        "mass": round(mass, 5),
        "restitution": round(restitution, 5),
        "friction": round(friction, 5),
        "static": static,
        "color": BODY_COLORS[color],
        "role": "visible_static_obstacle" if static else "dynamic",
    }


def cube(
    name: str,
    half_extents: tuple[float, float, float],
    position: tuple[float, float, float],
    color: str,
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    restitution: float = 0.45,
    friction: float = 0.55,
    static: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": "box",
        "half_extents": vec(half_extents),
        "size": vec((2.0 * half_extents[0], 2.0 * half_extents[1], 2.0 * half_extents[2])),
        "scale": vec(half_extents),
        "position": vec(position),
        "velocity": vec(velocity),
        "angular_velocity": vec(angular_velocity),
        "mass": round(mass, 5),
        "restitution": round(restitution, 5),
        "friction": round(friction, 5),
        "static": static,
        "color": BODY_COLORS[color],
        "role": "visible_static_obstacle" if static else "dynamic",
    }


def physics_specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 101)

    def low_friction() -> float:
        return trunc_gauss(rng, 0.055, 0.018, 0.015, 0.11)

    def medium_friction() -> float:
        return trunc_gauss(rng, 0.22, 0.06, 0.08, 0.40)

    def bounce() -> float:
        return trunc_gauss(rng, 0.78, 0.10, 0.55, 0.94)

    def mass() -> float:
        return trunc_gauss(rng, 1.0, 0.25, 0.55, 1.65)

    r_static = trunc_gauss(rng, 0.28, 0.035, 0.22, 0.34)
    r_drop = trunc_gauss(rng, 0.25, 0.025, 0.21, 0.30)
    r_a = trunc_gauss(rng, 0.26, 0.025, 0.22, 0.31)
    r_b = trunc_gauss(rng, 0.27, 0.025, 0.22, 0.32)
    r_hit = trunc_gauss(rng, 0.24, 0.025, 0.20, 0.30)
    r_cube_sphere = trunc_gauss(rng, 0.26, 0.025, 0.22, 0.32)

    two_speed = trunc_gauss(rng, 1.65, 0.22, 1.20, 2.15)
    block_speed = trunc_gauss(rng, 1.95, 0.25, 1.45, 2.50)
    cube_sphere_speed = trunc_gauss(rng, 1.70, 0.24, 1.20, 2.25)

    return [
        {
            "id": "phys_static_sphere",
            "kind": "single_static",
            "speed_class": "none",
            "description": "One visible object at rest; useful camera-only control.",
            "sample_model": "radius, mass, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [],
            "bodies": [
                sphere(
                    "red_resting_sphere",
                    r_static,
                    (0.0, -0.30, r_static),
                    "red",
                    mass=mass(),
                    restitution=trunc_gauss(rng, 0.45, 0.08, 0.25, 0.65),
                    friction=medium_friction(),
                )
            ],
        },
        {
            "id": "phys_drop_bounce",
            "kind": "gravity_bounce",
            "speed_class": "mixed",
            "description": "Sphere falls and bounces under gravity with lateral drift.",
            "sample_model": "radius, mass, lateral velocity, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["gold_drop_bounce", "ground"]],
            "bodies": [
                sphere(
                    "gold_drop_bounce",
                    r_drop,
                    (
                        trunc_gauss(rng, -0.35, 0.16, -0.65, -0.05),
                        trunc_gauss(rng, 0.08, 0.10, -0.12, 0.28),
                        trunc_gauss(rng, 1.50, 0.16, 1.20, 1.85),
                    ),
                    "gold",
                    velocity=(trunc_gauss(rng, 0.50, 0.14, 0.22, 0.85), 0.08, -0.35),
                    mass=mass(),
                    restitution=trunc_gauss(rng, 0.86, 0.06, 0.72, 0.96),
                    friction=low_friction(),
                )
            ],
        },
        {
            "id": "phys_two_sphere_collision",
            "kind": "two_body_collision",
            "speed_class": "medium",
            "description": "Two spheres cross the center and collide.",
            "sample_model": "radii, masses, velocities, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["red_cross_sphere", "blue_cross_sphere"]],
            "bodies": [
                sphere(
                    "red_cross_sphere",
                    r_a,
                    (-1.05, -0.18, r_a),
                    "red",
                    velocity=(two_speed, 0.20, 0.0),
                    mass=mass(),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                sphere(
                    "blue_cross_sphere",
                    r_b,
                    (1.05, 0.18, r_b),
                    "blue",
                    velocity=(-two_speed, -0.20, 0.0),
                    mass=mass(),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
            ],
        },
        {
            "id": "phys_ball_hits_visible_block",
            "kind": "dynamic_static_collision",
            "speed_class": "medium",
            "description": "Sphere collides with a visible static cube obstacle.",
            "sample_model": "sphere radius, block size, velocity, mass, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["teal_block_target", "violet_hit_sphere"]],
            "bodies": [
                sphere(
                    "violet_hit_sphere",
                    r_hit,
                    (-0.85, 0.0, r_hit),
                    "violet",
                    velocity=(block_speed, 0.0, 0.0),
                    mass=mass(),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                cube(
                    "teal_block_target",
                    (
                        trunc_gauss(rng, 0.24, 0.03, 0.18, 0.31),
                        trunc_gauss(rng, 0.30, 0.03, 0.23, 0.38),
                        trunc_gauss(rng, 0.28, 0.03, 0.22, 0.35),
                    ),
                    (-0.05, 0.0, 0.28),
                    "teal",
                    mass=0.0,
                    restitution=trunc_gauss(rng, 0.68, 0.08, 0.45, 0.82),
                    friction=medium_friction(),
                    static=True,
                ),
            ],
        },
        {
            "id": "phys_cube_sphere_collision",
            "kind": "cube_sphere_collision",
            "speed_class": "medium",
            "description": "A cube and a sphere meet near the center and rebound.",
            "sample_model": "cube size, sphere radius, masses, velocities, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["teal_moving_cube", "gold_moving_sphere"]],
            "bodies": [
                cube(
                    "teal_moving_cube",
                    (
                        trunc_gauss(rng, 0.23, 0.025, 0.19, 0.30),
                        trunc_gauss(rng, 0.23, 0.025, 0.19, 0.30),
                        trunc_gauss(rng, 0.23, 0.025, 0.19, 0.30),
                    ),
                    (-1.05, -0.16, 0.31),
                    "teal",
                    velocity=(cube_sphere_speed, 0.18, 0.0),
                    angular_velocity=(0.0, -2.0, 0.25),
                    mass=mass(),
                    restitution=trunc_gauss(rng, 0.62, 0.08, 0.42, 0.78),
                    friction=low_friction(),
                ),
                sphere(
                    "gold_moving_sphere",
                    r_cube_sphere,
                    (1.05, 0.16, r_cube_sphere),
                    "gold",
                    velocity=(-cube_sphere_speed, -0.18, 0.0),
                    mass=mass(),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
            ],
        },
        {
            "id": "phys_three_body_scatter",
            "kind": "three_body_collision",
            "speed_class": "mixed",
            "description": "Three objects interact near the center for multi-body review.",
            "sample_model": "sizes, masses, velocities, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["red_scatter_sphere", "teal_scatter_cube"]],
            "bodies": [
                sphere(
                    "red_scatter_sphere",
                    0.24,
                    (-1.25, -0.28, 0.24),
                    "red",
                    velocity=(trunc_gauss(rng, 1.75, 0.22, 1.25, 2.25), 0.38, 0.0),
                    mass=mass(),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                cube(
                    "teal_scatter_cube",
                    (0.24, 0.24, 0.24),
                    (0.05, 0.05, 0.24),
                    "teal",
                    velocity=(0.05, 0.00, 0.0),
                    angular_velocity=(0.0, 1.4, 0.4),
                    mass=mass(),
                    restitution=trunc_gauss(rng, 0.58, 0.08, 0.38, 0.74),
                    friction=low_friction(),
                ),
                sphere(
                    "blue_scatter_sphere",
                    0.23,
                    (1.20, 0.30, 0.23),
                    "blue",
                    velocity=(trunc_gauss(rng, -1.55, 0.20, -2.05, -1.05), -0.35, 0.0),
                    mass=mass(),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
            ],
        },
    ]


def camera_specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 303)

    def camera(
        camera_id: str,
        primitives: list[str],
        speed_class: str,
        start_position: tuple[float, float, float],
        start_target: tuple[float, float, float],
        velocity: tuple[float, float, float],
        target_velocity: tuple[float, float, float],
        *,
        roll_start_deg: float = 0.0,
        roll_velocity_deg_s: float = 0.0,
        lens_mm: float = 35.0,
        lens_velocity_mm_s: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "id": camera_id,
            "primitives": primitives,
            "speed_class": speed_class,
            "start_position": vec(start_position),
            "start_target": vec(start_target),
            "linear_velocity_mps": vec(velocity),
            "target_velocity_mps": vec(target_velocity),
            "roll_start_deg": round(roll_start_deg, 5),
            "roll_velocity_deg_s": round(roll_velocity_deg_s, 5),
            "lens_mm": round(lens_mm, 5),
            "lens_velocity_mm_s": round(lens_velocity_mm_s, 5),
            "motion_distribution": "camera position, target, roll, and lens velocities are clipped Gaussian samples",
        }

    dolly = trunc_gauss(rng, 0.58, 0.16, 0.28, 0.95)
    truck = trunc_gauss(rng, 0.50, 0.14, 0.22, 0.84)
    crane = trunc_gauss(rng, 0.30, 0.10, 0.10, 0.55)
    top_drift = trunc_gauss(rng, 0.24, 0.08, 0.08, 0.42)

    return [
        camera(
            "cam_static_left_high",
            ["static", "left_start", "slight_roll"],
            "none",
            (-2.45, -4.60, 1.80),
            (-0.10, -0.05, 0.42),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            roll_start_deg=trunc_gauss(rng, -1.8, 1.0, -4.0, 1.0),
            lens_mm=trunc_gauss(rng, 34.0, 2.0, 30.0, 40.0),
        ),
        camera(
            "cam_dolly_in_right_start",
            ["dolly_in", "right_start", "target_drift"],
            "medium",
            (2.35, -5.05, 1.55),
            (0.05, 0.00, 0.38),
            (-0.10, dolly, 0.02),
            (trunc_gauss(rng, 0.03, 0.03, -0.04, 0.10), 0.0, 0.0),
            roll_start_deg=trunc_gauss(rng, 1.2, 1.0, -1.0, 3.8),
            roll_velocity_deg_s=trunc_gauss(rng, 0.5, 0.5, -0.4, 1.6),
            lens_mm=trunc_gauss(rng, 33.0, 2.0, 29.0, 39.0),
            lens_velocity_mm_s=trunc_gauss(rng, 0.8, 0.4, 0.0, 1.6),
        ),
        camera(
            "cam_truck_left_pan_right",
            ["truck_left", "pan_right", "off_center_start"],
            "medium",
            (2.85, -4.10, 1.35),
            (0.35, 0.00, 0.38),
            (-truck, 0.03, 0.0),
            (-0.08, 0.02, 0.0),
            roll_start_deg=trunc_gauss(rng, -0.4, 1.0, -2.8, 2.0),
            roll_velocity_deg_s=trunc_gauss(rng, -0.25, 0.45, -1.2, 0.7),
            lens_mm=trunc_gauss(rng, 36.0, 2.5, 31.0, 42.0),
        ),
        camera(
            "cam_crane_up_tilt_down",
            ["crane_up", "tilt_down", "front_start"],
            "slow",
            (-0.75, -4.60, 1.10),
            (-0.05, -0.05, 0.32),
            (0.05, 0.02, crane),
            (0.0, 0.0, -0.05),
            roll_start_deg=trunc_gauss(rng, 0.0, 1.0, -2.2, 2.2),
            roll_velocity_deg_s=trunc_gauss(rng, 0.15, 0.35, -0.6, 0.9),
            lens_mm=trunc_gauss(rng, 37.0, 2.0, 32.0, 43.0),
        ),
        camera(
            "cam_top_down_drift",
            ["top_down", "ceiling_start", "drift"],
            "slow",
            (0.75, -0.80, 5.15),
            (0.05, -0.02, 0.20),
            (-top_drift, top_drift * 0.55, -0.04),
            (0.02, 0.0, 0.0),
            roll_start_deg=trunc_gauss(rng, 6.0, 2.0, 2.0, 11.0),
            roll_velocity_deg_s=trunc_gauss(rng, 0.8, 0.6, -0.2, 2.2),
            lens_mm=trunc_gauss(rng, 38.0, 2.5, 33.0, 45.0),
        ),
    ]


def look_at_quat_with_roll(position: list[float], target: list[float], roll_deg: float) -> list[float]:
    import numpy as np
    import pyquaternion as pyquat
    from kubric.core.objects import look_at_quat

    base = pyquat.Quaternion(*look_at_quat(position, target, up="Y", front="-Z"))
    axis = np.array(target, dtype=float) - np.array(position, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6 or abs(roll_deg) < 1e-8:
        return vec(base)
    roll = pyquat.Quaternion(axis=axis / norm, degrees=roll_deg)
    return vec(roll * base)


def camera_records(camera: dict[str, Any], frames: int) -> list[dict[str, Any]]:
    records = []
    for frame in range(frames):
        elapsed = frame / FPS
        position = [
            camera["start_position"][axis] + camera["linear_velocity_mps"][axis] * elapsed
            for axis in range(3)
        ]
        target = [
            camera["start_target"][axis] + camera["target_velocity_mps"][axis] * elapsed
            for axis in range(3)
        ]
        roll = camera["roll_start_deg"] + camera["roll_velocity_deg_s"] * elapsed
        lens = clamp(camera["lens_mm"] + camera["lens_velocity_mm_s"] * elapsed, 28.0, 50.0)
        records.append(
            {
                "frame": frame,
                "time_s": round(elapsed, 5),
                "position": vec(position),
                "look_at": vec(target),
                "quaternion_wxyz": look_at_quat_with_roll(vec(position), vec(target), roll),
                "roll_deg": round(roll, 5),
                "lens_mm": round(lens, 5),
            }
        )
    return records


def make_material(kb: Any, name: str, rgba: list[float], roughness: float = 0.72) -> Any:
    return kb.PrincipledBSDFMaterial(name=f"mat_{name}", color=kb.Color(*color_tuple(rgba)), roughness=roughness)


def make_kb_body(kb: Any, body: dict[str, Any], material: Any | None = None, background: bool = False) -> Any:
    kwargs = {
        "name": body["name"],
        "scale": tuple(body["scale"]),
        "position": tuple(body["position"]),
        "velocity": tuple(body.get("velocity", [0.0, 0.0, 0.0])),
        "angular_velocity": tuple(body.get("angular_velocity", [0.0, 0.0, 0.0])),
        "mass": float(body.get("mass", 1.0)),
        "friction": float(body.get("friction", 0.35)),
        "restitution": float(body.get("restitution", 0.55)),
        "static": bool(body.get("static", False)),
        "background": background,
    }
    if material is not None:
        kwargs["material"] = material
    if body["shape"] == "sphere":
        return kb.Sphere(**kwargs)
    if body["shape"] == "box":
        return kb.Cube(**kwargs)
    raise ValueError(body["shape"])


def world_bodies(world: dict[str, Any]) -> list[dict[str, Any]]:
    height = float(world["wall_height"])
    bodies = [
        {
            "name": "ground",
            "shape": "box",
            "scale": [4.5, 4.5, 0.04],
            "half_extents": [4.5, 4.5, 0.04],
            "position": [0.0, 0.0, -0.04],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "mass": 0.0,
            "friction": 0.18,
            "restitution": 0.62,
            "static": True,
            "color": world["ground_color"],
        },
        {
            "name": "back_wall",
            "shape": "box",
            "scale": [4.5, 0.05, height / 2.0],
            "half_extents": [4.5, 0.05, height / 2.0],
            "position": [0.0, 3.35, height / 2.0],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "mass": 0.0,
            "friction": 0.25,
            "restitution": 0.58,
            "static": True,
            "color": world["back_wall_color"],
        },
        {
            "name": "left_wall",
            "shape": "box",
            "scale": [0.05, 3.45, height / 2.0],
            "half_extents": [0.05, 3.45, height / 2.0],
            "position": [-4.35, 0.0, height / 2.0],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "mass": 0.0,
            "friction": 0.25,
            "restitution": 0.58,
            "static": True,
            "color": world["side_wall_color"],
        },
        {
            "name": "right_wall",
            "shape": "box",
            "scale": [0.05, 3.45, height / 2.0],
            "half_extents": [0.05, 3.45, height / 2.0],
            "position": [4.35, 0.0, height / 2.0],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "mass": 0.0,
            "friction": 0.25,
            "restitution": 0.58,
            "static": True,
            "color": world["side_wall_color"],
        },
    ]
    for prop in world.get("static_props", []):
        if prop.get("collider"):
            bodies.append(
                {
                    "name": prop["name"],
                    "shape": prop["shape"],
                    "scale": prop["half_extents"],
                    "half_extents": prop["half_extents"],
                    "position": prop["position"],
                    "velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "mass": 0.0,
                    "friction": 0.25,
                    "restitution": 0.45,
                    "static": True,
                    "color": prop["color"],
                }
            )
    return bodies


def add_world_assets(kb: Any, scene: Any, world: dict[str, Any]) -> None:
    for body in world_bodies(world):
        mat = make_material(kb, body["name"], body["color"])
        scene.add(make_kb_body(kb, body, mat, background=True))

    def add_deco(name: str, half_extents: tuple[float, float, float], position: tuple[float, float, float], color: list[float]) -> None:
        body = {
            "name": name,
            "shape": "box",
            "scale": list(half_extents),
            "position": list(position),
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "mass": 0.0,
            "friction": 0.0,
            "restitution": 0.0,
            "static": True,
            "color": color,
        }
        scene.add(make_kb_body(kb, body, make_material(kb, name, color), background=True))

    if world["wall_style"] == "horizontal_bands":
        for idx, z in enumerate([0.65, 1.25, 1.85, 2.45]):
            add_deco(f"wall_band_{idx}", (4.15, 0.018, 0.018), (0.0, 3.285, z), [0.47, 0.56, 0.62, 1.0])
    elif world["wall_style"] == "wall_panels":
        for idx, x in enumerate([-2.8, -1.4, 0.0, 1.4, 2.8]):
            add_deco(f"wall_panel_{idx}", (0.35, 0.018, 0.71), (x, 3.285, 1.05), [0.69, 0.62, 0.54, 1.0])

    if world["floor_style"] == "subtle_marks":
        for idx, (x, y) in enumerate([(-2.9, -2.3), (2.7, -2.1), (-2.5, 2.2), (2.6, 2.0)]):
            add_deco(f"floor_mark_{idx}", (0.21, 0.018, 0.009), (x, y, 0.012), [0.48, 0.50, 0.48, 1.0])


def build_sim_scene(kb: Any, physics: dict[str, Any], world: dict[str, Any], frames: int, width: int, height: int) -> tuple[Any, dict[str, Any]]:
    scene = kb.Scene(frame_start=0, frame_end=frames - 1, frame_rate=FPS, step_rate=PHYSICS_HZ, resolution=(width, height))
    body_assets: dict[str, Any] = {}
    for body in world_bodies(world):
        asset = make_kb_body(kb, body, None, background=True)
        scene.add(asset)
    for body in physics["bodies"]:
        asset = make_kb_body(kb, body, None, background=False)
        scene.add(asset)
        body_assets[body["name"]] = asset
    return scene, body_assets


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def collision_name(instance: Any) -> str:
    if isinstance(instance, tuple) and len(instance) == 1:
        instance = instance[0]
    return getattr(instance, "name", None) or "none"


def collision_summary(collisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for collision in collisions:
        names = [collision_name(instance) for instance in collision.get("instances", [])]
        if len(names) != 2:
            continue
        key = pair_key(names[0], names[1])
        item = by_pair.setdefault(
            key,
            {
                "pair": list(key),
                "samples": 0,
                "first_frame": round(float(collision["frame"]), 5),
                "max_force": 0.0,
            },
        )
        item["samples"] += 1
        item["first_frame"] = min(item["first_frame"], round(float(collision["frame"]), 5))
        item["max_force"] = round(max(item["max_force"], float(collision.get("force", 0.0))), 5)
    return {"pairs": list(by_pair.values())}


def expected_contacts_passed(summary: dict[str, Any], expected_contacts: list[list[str]]) -> bool:
    seen = {tuple(item["pair"]) for item in summary["pairs"]}
    for a, b in expected_contacts:
        if pair_key(a, b) in seen:
            continue
        if "ground" in (a, b):
            other = b if a == "ground" else a
            if pair_key(other, "none") in seen:
                continue
        return False
    return True


def sphere_sphere_touch(a: dict[str, Any], b: dict[str, Any], margin: float) -> bool:
    distance = math.dist(a["position"], b["position"])
    return distance <= float(a["radius"] + b["radius"] + margin)


def sphere_box_touch(sphere_obj: dict[str, Any], box_obj: dict[str, Any], margin: float) -> bool:
    center = sphere_obj["position"]
    box_center = box_obj["position"]
    half = box_obj.get("half_extents") or [v / 2.0 for v in box_obj["size"]]
    closest = [clamp(center[i], box_center[i] - half[i], box_center[i] + half[i]) for i in range(3)]
    return math.dist(center, closest) <= float(sphere_obj["radius"] + margin)


def ground_touch(obj: dict[str, Any], margin: float) -> bool:
    if obj["shape"] == "sphere":
        return obj["position"][2] <= float(obj["radius"] + margin)
    half = obj.get("half_extents") or [v / 2.0 for v in obj["size"]]
    return obj["position"][2] - float(half[2]) <= margin


def visual_contact_exists(records: list[dict[str, Any]], name_a: str, name_b: str, margin: float = 0.035) -> bool:
    for frame_record in records:
        objects = {obj["name"]: obj for obj in frame_record["objects"]}
        if name_a == "ground" and name_b in objects and ground_touch(objects[name_b], margin):
            return True
        if name_b == "ground" and name_a in objects and ground_touch(objects[name_a], margin):
            return True
        if name_a not in objects or name_b not in objects:
            continue
        a = objects[name_a]
        b = objects[name_b]
        if a["shape"] == "sphere" and b["shape"] == "sphere" and sphere_sphere_touch(a, b, margin):
            return True
        if a["shape"] == "sphere" and b["shape"] == "box" and sphere_box_touch(a, b, margin):
            return True
        if a["shape"] == "box" and b["shape"] == "sphere" and sphere_box_touch(b, a, margin):
            return True
    return False


def expected_visual_contacts_passed(records: list[dict[str, Any]], expected_contacts: list[list[str]]) -> bool:
    return all(visual_contact_exists(records, a, b) for a, b in expected_contacts)


def simulate_physics(physics: dict[str, Any], world: dict[str, Any], frames: int, width: int, height: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pybullet as pb
    import kubric as kb
    from kubric.simulator import PyBullet

    PyBullet.__del__ = lambda self: None
    while pb.isConnected():
        pb.disconnect()

    scene, body_assets = build_sim_scene(kb, physics, world, frames, width, height)
    simulator = PyBullet(scene)
    animation, collisions = simulator.run()
    if pb.isConnected(simulator.physics_client):
        pb.disconnect(simulator.physics_client)
    summary = collision_summary(collisions)

    records: list[dict[str, Any]] = []
    for frame in range(frames):
        objects = []
        for body in physics["bodies"]:
            asset = body_assets[body["name"]]
            anim = animation[asset]
            objects.append(
                {
                    "name": body["name"],
                    "shape": body["shape"],
                    "radius": body.get("radius"),
                    "size": body.get("size"),
                    "half_extents": body.get("half_extents"),
                    "color": body["color"],
                    "role": body["role"],
                    "position": vec(anim["position"][frame]),
                    "orientation_wxyz": vec(anim["quaternion"][frame]),
                    "linear_velocity": vec(anim["velocity"][frame]),
                    "angular_velocity": vec(anim["angular_velocity"][frame]),
                }
            )
        records.append({"frame": frame, "time_s": round(frame / FPS, 5), "objects": objects})

    expected = physics.get("expected_contacts", [])
    quality = {
        "expected_contacts": expected,
        "expected_contacts_passed": expected_contacts_passed(summary, expected)
        or expected_visual_contacts_passed(records, expected),
        "collision_log_contacts_passed": expected_contacts_passed(summary, expected),
        "visual_contacts_passed": expected_visual_contacts_passed(records, expected),
        "collision_summary": summary,
    }
    return records, quality


def clip_id_for(index: int, camera_id: str, physics_id: str) -> str:
    return f"clip_{index:03d}_{camera_id}_{physics_id}"


def make_pairs(clips: list[dict[str, Any]], target_pairs: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 909)
    pairs = []
    seen: set[tuple[str, str, str]] = set()
    by_camera: dict[str, list[dict[str, Any]]] = {}
    by_physics_scene: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for clip in clips:
        by_camera.setdefault(clip["camera_id"], []).append(clip)
        by_physics_scene.setdefault((clip["physics_id"], clip["scene_id"]), []).append(clip)

    def add(kind: str, title: str, a: dict[str, Any], b: dict[str, Any], controlled: str, varied: str, tags: list[str]) -> None:
        key = (kind, a["clip_id"], b["clip_id"])
        reverse = (kind, b["clip_id"], a["clip_id"])
        if key in seen or reverse in seen or a["clip_id"] == b["clip_id"]:
            return
        seen.add(key)
        group_id = f"pair_{len(pairs):03d}_{kind}"
        pairs.append(
            {
                "group_id": group_id,
                "title": title,
                "controlled_factor": controlled,
                "varied_factor": varied,
                "clip_ids": [a["clip_id"], b["clip_id"]],
                "tags": tags,
            }
        )
        a["pair_groups"].append(group_id)
        b["pair_groups"].append(group_id)

    same_camera_budget = target_pairs // 2
    camera_candidates = []
    for camera_id, group in by_camera.items():
        for idx, a in enumerate(group):
            for b in group[idx + 1 :]:
                if a["physics_id"] != b["physics_id"]:
                    camera_candidates.append((camera_id, a, b))
    rng.shuffle(camera_candidates)
    for camera_id, a, b in camera_candidates:
        if len(pairs) >= same_camera_budget:
            break
        add(
            "same_camera",
            f"Same camera trajectory {camera_id}, different physical program",
            a,
            b,
            "camera_id",
            "physics_id/scene_id",
            ["same_camera", "different_physics", "possibly_different_scene"],
        )

    physics_target = target_pairs - len(pairs)
    start = len(pairs)
    physics_candidates = []
    for (physics_id, scene_id), group in by_physics_scene.items():
        for idx, a in enumerate(group):
            for b in group[idx + 1 :]:
                if a["camera_id"] != b["camera_id"]:
                    physics_candidates.append((physics_id, scene_id, a, b))
    rng.shuffle(physics_candidates)
    for physics_id, scene_id, a, b in physics_candidates:
        if len(pairs) - start >= physics_target:
            break
        add(
            "same_physics_scene",
            f"Same physics and scene {physics_id} / {scene_id}, different camera",
            a,
            b,
            "physics_id/scene_id",
            "camera_id",
            ["same_physics", "same_scene", "different_camera"],
        )
    return pairs


def update_index(run_root: Path, run_id: str) -> None:
    index_path = run_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        runs = [run for run in index.get("runs", []) if run.get("run_id") != run_id]
    else:
        runs = []
    runs.append({"run_id": run_id, "manifest": f"{run_id}/manifest.json"})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")


def write_run(
    run_id: str,
    run_dir: Path,
    run_root: Path,
    cameras: list[dict[str, Any]],
    physics_programs: list[dict[str, Any]],
    worlds: list[dict[str, Any]],
    pair_target: int,
    seed: int,
    frames: int,
    width: int,
    height: int,
) -> Path:
    metadata_dir = run_dir / "metadata"
    frames_root = run_dir / "frames"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []

    sim_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for physics_index, physics in enumerate(physics_programs):
        world = worlds[physics_index % len(worlds)]
        sim_key = f"{physics['id']}__{world['id']}"
        sim_cache[sim_key] = simulate_physics(physics, world, frames, width, height)
        if not sim_cache[sim_key][1]["expected_contacts_passed"]:
            expected = sim_cache[sim_key][1]["expected_contacts"]
            seen = sim_cache[sim_key][1]["collision_summary"]["pairs"]
            raise RuntimeError(f"{physics['id']} failed expected contact audit: expected={expected}, seen={seen}")

    clip_index = 0
    for physics_index, physics in enumerate(physics_programs):
        world = worlds[physics_index % len(worlds)]
        sim_key = f"{physics['id']}__{world['id']}"
        physics_frames, quality = sim_cache[sim_key]
        for camera in cameras:
            clip_id = clip_id_for(clip_index, camera["id"], physics["id"])
            camera_frames = camera_records(camera, frames)
            metadata_rel = Path("metadata") / f"{clip_id}.json"
            video_rel = Path("videos") / f"{clip_id}.mp4"
            frames_rel = Path("frames") / clip_id
            metadata = {
                "clip_id": clip_id,
                "camera_id": camera["id"],
                "physics_id": physics["id"],
                "scene_id": world["id"],
                "fps": FPS,
                "frames_count": frames,
                "duration_s": round((frames - 1) / FPS, 5),
                "resolution": [width, height],
                "generator": "official_kubric_review",
                "simulator": "official kubric.simulator.PyBullet",
                "renderer": "official kubric.renderer.Blender",
                "camera_spec": camera,
                "physics_spec": physics,
                "scene_spec": world,
                "camera_frames": camera_frames,
                "physics_frames": physics_frames,
                "quality_audit": quality,
                "orientation_format": "wxyz",
                "video": str(video_rel),
                "frames_dir": str(frames_rel),
            }
            (run_dir / metadata_rel).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            clip = {
                "clip_id": clip_id,
                "camera_id": camera["id"],
                "physics_id": physics["id"],
                "scene_id": world["id"],
                "camera_primitives": camera["primitives"],
                "physics_kind": physics["kind"],
                "physics_speed_class": physics["speed_class"],
                "camera_speed_class": camera["speed_class"],
                "pair_groups": [],
                "video": str(video_rel),
                "metadata": str(metadata_rel),
            }
            clips.append(clip)
            render_jobs.append(
                {
                    "clip_id": clip_id,
                    "metadata": str(run_dir / metadata_rel),
                    "frames_dir": str(run_dir / frames_rel),
                    "video": str(run_dir / video_rel),
                }
            )
            clip_index += 1

    pair_groups = make_pairs(clips, pair_target, seed)
    ambiguous = []
    dolly_static = next(
        (c["clip_id"] for c in clips if c["camera_id"] == "cam_dolly_in_right_start" and c["physics_id"] == "phys_static_sphere"),
        None,
    )
    static_toward = None
    if dolly_static and static_toward:
        ambiguous.append(
            {
                "group_id": "ambiguous_000_dolly_in_vs_object_toward_camera",
                "title": "Ambiguous appearance: camera dolly-in vs object moving toward camera",
                "clip_ids": [dolly_static, static_toward],
                "reason": "A forward camera move and an object moving toward the camera can both increase image scale. This is a diagnostic comparison, not a controlled training pair.",
                "hidden_factor_difference": ["camera_motion", "physical_motion"],
                "intended_use": "manual review / ambiguity audit",
            }
        )

    manifest = {
        "project": "camera_motion_disentangle",
        "run_id": run_id,
        "generator": "official_kubric_review",
        "description": "Small official-Kubric review bank. Same-physics pairs reuse identical physics_frames and scene_id; same-camera pairs reuse identical camera_frames.",
        "fps": FPS,
        "frames_count": frames,
        "duration_s": round((frames - 1) / FPS, 5),
        "resolution": [width, height],
        "seed": seed,
        "simulator": "official kubric.simulator.PyBullet",
        "renderer": "official kubric.renderer.Blender",
        "render_note": "Kubric renderer outputs PNG frames, then system ffmpeg encodes review MP4s for GitHub Pages.",
        "sampling_model": "object radii/sizes, masses, initial speeds, friction, restitution, and camera velocities are sampled from clipped Gaussian templates; each sampled value is written into per-clip metadata.",
        "quality_filters": "Expected non-ground contacts are audited before rendering; runs fail if required contact pairs are absent.",
        "camera_reference": cameras,
        "physics_reference": physics_programs,
        "scene_reference": worlds,
        "clips": clips,
        "pair_groups": pair_groups,
        "ambiguous_equivalence_groups": ambiguous,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    jobs_path = run_dir / "render_jobs.json"
    jobs_path.write_text(json.dumps({"run_id": run_id, "fps": FPS, "jobs": render_jobs}, indent=2), encoding="utf-8")
    update_index(run_root, run_id)
    return jobs_path


def blender_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def patch_blender_denoiser(blender_cls: Any) -> None:
    def set_denoising(self: Any, value: bool) -> None:
        self.blender_scene.cycles.use_denoising = bool(value)
        if not value:
            return
        enum_items = list(self.blender_scene.cycles.bl_rna.properties["denoiser"].enum_items.keys())
        if "OPENIMAGEDENOISE" in enum_items:
            self.blender_scene.cycles.denoiser = "OPENIMAGEDENOISE"
        elif enum_items:
            self.blender_scene.cycles.denoiser = next(iter(enum_items))

    blender_cls.use_denoising = property(blender_cls.use_denoising.fget, set_denoising)


def set_linear_interpolation() -> None:
    import bpy

    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"
        data = getattr(obj, "data", None)
        if data and data.animation_data and data.animation_data.action:
            for fcurve in data.animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = "LINEAR"


def render_one_job(job: dict[str, Any]) -> None:
    import kubric as kb
    from kubric.renderer import Blender

    patch_blender_denoiser(Blender)
    metadata = json.loads(Path(job["metadata"]).read_text(encoding="utf-8"))
    frames = int(metadata["frames_count"])
    width, height = metadata["resolution"]
    scene = kb.Scene(frame_start=1, frame_end=frames, frame_rate=FPS, step_rate=PHYSICS_HZ, resolution=(width, height))
    add_world_assets(kb, scene, metadata["scene_spec"])

    body_assets = {}
    first_frame = metadata["physics_frames"][0]
    body_specs = {body["name"]: body for body in metadata["physics_spec"]["bodies"]}
    for object_record in first_frame["objects"]:
        spec = body_specs[object_record["name"]]
        material = make_material(kb, spec["name"], spec["color"])
        asset = make_kb_body(kb, spec, material, background=False)
        scene.add(asset)
        body_assets[spec["name"]] = asset

    first_camera = metadata["camera_frames"][0]
    camera = kb.PerspectiveCamera(
        name=metadata["camera_id"],
        position=tuple(first_camera["position"]),
        quaternion=tuple(first_camera["quaternion_wxyz"]),
        focal_length=float(first_camera["lens_mm"]),
        sensor_width=32.0,
    )
    scene.camera = camera
    scene.add(camera)
    sun = kb.DirectionalLight(name="sun", position=(3.5, -4.5, 6.0), intensity=1.4)
    sun.look_at((0.0, 0.0, 0.0))
    key = kb.RectAreaLight(name="key_area", position=(-2.5, -3.2, 4.5), width=3.0, height=3.0, intensity=130.0)
    key.look_at((0.0, 0.0, 0.0))
    scene.add([sun, key])

    renderer = Blender(
        scene,
        scratch_dir=Path(job["frames_dir"]),
        samples_per_pixel=16,
        use_denoising=True,
        adaptive_sampling=True,
        verbose=False,
    )
    renderer.blender_scene.render.engine = "CYCLES"
    renderer.blender_scene.cycles.samples = 16
    renderer.blender_scene.cycles.use_denoising = True
    renderer.blender_scene.render.image_settings.file_format = "PNG"
    renderer.blender_scene.render.image_settings.color_mode = "RGB"
    renderer.blender_scene.view_settings.view_transform = "Standard"
    renderer.blender_scene.view_settings.look = "Medium High Contrast"

    for frame_record in metadata["physics_frames"]:
        frame_no = frame_record["frame"] + 1
        for object_record in frame_record["objects"]:
            asset = body_assets[object_record["name"]]
            asset.position = tuple(object_record["position"])
            asset.quaternion = tuple(object_record["orientation_wxyz"])
            asset.keyframe_insert("position", frame_no)
            asset.keyframe_insert("quaternion", frame_no)
    for camera_record in metadata["camera_frames"]:
        frame_no = camera_record["frame"] + 1
        camera.position = tuple(camera_record["position"])
        camera.quaternion = tuple(camera_record["quaternion_wxyz"])
        camera.focal_length = float(camera_record["lens_mm"])
        camera.keyframe_insert("position", frame_no)
        camera.keyframe_insert("quaternion", frame_no)
        camera.keyframe_insert("focal_length", frame_no)
    set_linear_interpolation()

    frames_dir = Path(job["frames_dir"])
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    renderer.render(png_filepath=str(frames_dir / "frame_"), exr_filepath=None)


def render_jobs_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-jobs", type=Path, required=True)
    args = parser.parse_args(blender_argv())
    payload = json.loads(args.render_jobs.read_text(encoding="utf-8"))
    for idx, job in enumerate(payload["jobs"], start=1):
        print(f"[{idx}/{len(payload['jobs'])}] rendering {job['clip_id']}", flush=True)
        render_one_job(job)


def render_with_blender(blender_bin: Path, jobs_path: Path, kubric_site_packages: Path) -> None:
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(kubric_site_packages) if not existing else f"{kubric_site_packages}:{existing}"
    log_path = jobs_path.with_name("render.log")
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [
                str(blender_bin),
                "--background",
                "--python",
                str(Path(__file__).resolve()),
                "--",
                "--render-jobs",
                str(jobs_path),
            ],
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )


def encode_videos(jobs_path: Path) -> None:
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    for job in payload["jobs"]:
        frames_dir = Path(job["frames_dir"])
        video_path = Path(job["video"])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(payload["fps"]),
                "-start_number",
                "1",
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            check=True,
        )


def generate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="kubric_review_v2_official")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--kubric-site-packages", type=Path, default=DEFAULT_KUBRIC_SITE_PACKAGES)
    parser.add_argument("--camera-limit", type=int, default=4)
    parser.add_argument("--physics-limit", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_root / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} already exists; pass --overwrite to replace generated assets.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cameras = camera_specs(args.seed)[: args.camera_limit]
    programs = physics_specs(args.seed)[: args.physics_limit]
    worlds = world_specs()
    jobs_path = write_run(
        args.run_id,
        run_dir,
        args.run_root,
        cameras,
        programs,
        worlds,
        args.pairs,
        args.seed,
        args.frames,
        args.width,
        args.height,
    )
    if not args.no_render:
        render_with_blender(args.blender_bin, jobs_path, args.kubric_site_packages)
        encode_videos(jobs_path)
    print(f"Wrote {len(cameras) * len(programs)} official Kubric clips to {run_dir}")


if __name__ == "__main__":
    if "--render-jobs" in blender_argv():
        render_jobs_main()
    else:
        generate_main()
