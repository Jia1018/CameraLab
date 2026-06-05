#!/usr/bin/env python3
"""Generate a small review bank with real PyBullet physics and Blender renders.

This is a lightweight Kubric-style pipeline for the local container:
PyBullet owns rigid-body motion, Blender owns rendering, and the manifest keeps
camera_id, physics_id, and scene_id as separate auditable factors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "site" / "assets" / "runs"
DEFAULT_BLENDER = Path("/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender")

FPS = 24
PHYSICS_HZ = 240
FRAMES = 60
WIDTH = 320
HEIGHT = 240
SEED = 20260603

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


def vec(values: tuple[float, float, float] | list[float]) -> list[float]:
    return [round(float(v), 5) for v in values]


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
                    "size": [0.52, 0.52, 0.30],
                    "position": [-3.25, 1.95, 0.15],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "color": [0.42, 0.39, 0.35, 1.0],
                    "collider": True,
                }
            ],
        },
        {
            "id": "scene_green_floor_marks",
            "ground_color": [0.58, 0.67, 0.61, 1.0],
            "back_wall_color": [0.73, 0.80, 0.73, 1.0],
            "side_wall_color": [0.66, 0.73, 0.67, 1.0],
            "wall_height": 2.2,
            "wall_style": "plain",
            "floor_style": "corner_marks",
            "static_props": [],
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
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": "sphere",
        "radius": radius,
        "position": vec(position),
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "velocity": vec(velocity),
        "angular_velocity": vec(angular_velocity),
        "mass": mass,
        "restitution": restitution,
        "friction": friction,
        "color": BODY_COLORS[color],
        "role": "dynamic" if mass > 0 else "visible_static_obstacle",
    }


def cube(
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: str,
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    restitution: float = 0.45,
    friction: float = 0.55,
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": "box",
        "size": vec(size),
        "position": vec(position),
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "velocity": vec(velocity),
        "angular_velocity": vec(angular_velocity),
        "mass": mass,
        "restitution": restitution,
        "friction": friction,
        "color": BODY_COLORS[color],
        "role": "dynamic" if mass > 0 else "visible_static_obstacle",
    }


def physics_specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 101)
    toward_speed = trunc_gauss(rng, 0.92, 0.20, 0.48, 1.35)
    away_speed = trunc_gauss(rng, 0.78, 0.18, 0.40, 1.15)
    drop_x = trunc_gauss(rng, 0.38, 0.16, -0.05, 0.75)
    roll_speed = trunc_gauss(rng, 0.86, 0.20, 0.45, 1.28)
    cross_speed = trunc_gauss(rng, 1.02, 0.20, 0.68, 1.45)
    block_speed = trunc_gauss(rng, 1.12, 0.20, 0.72, 1.55)
    scatter_speed = trunc_gauss(rng, 0.88, 0.25, 0.45, 1.45)
    cube_sphere_speed = trunc_gauss(rng, 0.95, 0.18, 0.60, 1.35)

    return [
        {
            "id": "phys_static_sphere",
            "kind": "single_static",
            "speed_class": "none",
            "description": "One visible object at rest; useful camera-only control.",
            "motion_distribution": {"linear_speed_mps": [0.0, 0.0, 0.0, 0.0]},
            "bodies": [sphere("red_resting_sphere", 0.28, (0.0, -0.35, 0.28), "red", friction=0.65)],
        },
        {
            "id": "phys_sphere_toward_camera",
            "kind": "single_translation",
            "speed_class": "medium",
            "description": "Sphere moves toward the usual front camera position.",
            "motion_distribution": {"linear_speed_mps": [0.92, 0.20, 0.48, 1.35]},
            "bodies": [
                sphere(
                    "red_toward_camera",
                    0.27,
                    (0.05, 1.45, 0.27),
                    "red",
                    velocity=(0.0, -toward_speed, 0.0),
                    friction=0.25,
                )
            ],
        },
        {
            "id": "phys_sphere_away_camera",
            "kind": "single_translation",
            "speed_class": "medium",
            "description": "Sphere moves away from the usual front camera position.",
            "motion_distribution": {"linear_speed_mps": [0.78, 0.18, 0.40, 1.15]},
            "bodies": [
                sphere(
                    "blue_away_camera",
                    0.27,
                    (-0.15, -1.35, 0.27),
                    "blue",
                    velocity=(0.05, away_speed, 0.0),
                    friction=0.28,
                )
            ],
        },
        {
            "id": "phys_drop_bounce",
            "kind": "gravity_bounce",
            "speed_class": "mixed",
            "description": "Sphere falls and bounces under gravity with slight lateral velocity.",
            "motion_distribution": {"x_velocity_mps": [0.38, 0.16, -0.05, 0.75]},
            "bodies": [
                sphere(
                    "gold_drop_bounce",
                    0.25,
                    (-0.45, 0.15, 1.95),
                    "gold",
                    velocity=(drop_x, 0.18, 0.0),
                    restitution=0.78,
                    friction=0.22,
                )
            ],
        },
        {
            "id": "phys_cube_roll_left",
            "kind": "single_roll",
            "speed_class": "medium",
            "description": "Cube slides and tumbles left with angular velocity.",
            "motion_distribution": {"linear_speed_mps": [0.86, 0.20, 0.45, 1.28]},
            "bodies": [
                cube(
                    "teal_rolling_cube",
                    (0.48, 0.48, 0.48),
                    (2.05, -0.65, 0.24),
                    "teal",
                    velocity=(-roll_speed, 0.04, 0.0),
                    angular_velocity=(0.0, -3.2, 0.25),
                    restitution=0.35,
                    friction=0.48,
                )
            ],
        },
        {
            "id": "phys_two_cross_collision",
            "kind": "two_body_collision",
            "speed_class": "medium",
            "description": "Two spheres collide near the center and scatter visibly.",
            "motion_distribution": {"linear_speed_mps": [1.02, 0.20, 0.68, 1.45]},
            "bodies": [
                sphere(
                    "red_cross_left",
                    0.25,
                    (-1.55, -0.46, 0.25),
                    "red",
                    velocity=(cross_speed, 0.26, 0.0),
                    restitution=0.72,
                    friction=0.20,
                ),
                sphere(
                    "blue_cross_right",
                    0.25,
                    (1.55, 0.38, 0.25),
                    "blue",
                    velocity=(-cross_speed, -0.23, 0.0),
                    restitution=0.72,
                    friction=0.20,
                ),
            ],
        },
        {
            "id": "phys_ball_hits_visible_block",
            "kind": "visible_static_obstacle",
            "speed_class": "medium_fast",
            "description": "Ball hits a visible static block; there are no hidden colliders.",
            "motion_distribution": {"impact_speed_mps": [1.12, 0.20, 0.72, 1.55]},
            "bodies": [
                sphere(
                    "red_incoming_ball",
                    0.24,
                    (-1.85, -0.2, 0.24),
                    "red",
                    velocity=(block_speed, 0.0, 0.0),
                    restitution=0.68,
                    friction=0.22,
                ),
                cube(
                    "dark_visible_block",
                    (0.58, 0.58, 0.56),
                    (0.35, -0.2, 0.28),
                    "dark",
                    mass=0.0,
                    restitution=0.55,
                    friction=0.62,
                ),
            ],
        },
        {
            "id": "phys_three_body_scatter",
            "kind": "multi_body_collision",
            "speed_class": "mixed",
            "description": "Three visible bodies interact with different masses and shapes.",
            "motion_distribution": {"linear_speed_mps": [0.88, 0.25, 0.45, 1.45]},
            "bodies": [
                sphere(
                    "gold_left_sphere",
                    0.22,
                    (-1.35, -0.95, 0.22),
                    "gold",
                    velocity=(scatter_speed, 0.48, 0.0),
                    restitution=0.66,
                    friction=0.24,
                ),
                cube(
                    "teal_middle_cube",
                    (0.42, 0.42, 0.42),
                    (0.05, -0.05, 0.21),
                    "teal",
                    velocity=(0.06, 0.0, 0.0),
                    angular_velocity=(0.0, 1.4, 0.6),
                    restitution=0.44,
                    friction=0.55,
                ),
                sphere(
                    "violet_right_sphere",
                    0.22,
                    (1.35, 0.70, 0.22),
                    "violet",
                    velocity=(-0.72 * scatter_speed, -0.32, 0.0),
                    restitution=0.62,
                    friction=0.22,
                ),
            ],
        },
        {
            "id": "phys_cube_sphere_collision",
            "kind": "mixed_shape_collision",
            "speed_class": "medium",
            "description": "A moving cube and a moving sphere collide without overlap starts.",
            "motion_distribution": {"linear_speed_mps": [0.95, 0.18, 0.60, 1.35]},
            "bodies": [
                cube(
                    "blue_left_cube",
                    (0.42, 0.42, 0.42),
                    (-1.40, 0.55, 0.21),
                    "blue",
                    velocity=(cube_sphere_speed, -0.22, 0.0),
                    angular_velocity=(0.0, -2.4, 0.4),
                    restitution=0.48,
                    friction=0.46,
                ),
                sphere(
                    "red_right_sphere",
                    0.24,
                    (1.35, -0.18, 0.24),
                    "red",
                    velocity=(-0.88 * cube_sphere_speed, 0.25, 0.0),
                    restitution=0.64,
                    friction=0.24,
                ),
            ],
        },
    ]


def camera_specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 211)

    def linear_camera(
        camera_id: str,
        primitives: list[str],
        start_position: tuple[float, float, float],
        start_target: tuple[float, float, float],
        velocity: tuple[float, float, float],
        target_velocity: tuple[float, float, float],
        *,
        roll_start_deg: float = 0.0,
        roll_velocity_deg_s: float = 0.0,
        lens_mm: float = 35.0,
        lens_velocity_mm_s: float = 0.0,
        speed_class: str = "medium",
        description: str,
    ) -> dict[str, Any]:
        return {
            "id": camera_id,
            "mode": "linear",
            "primitives": primitives,
            "speed_class": speed_class,
            "description": description,
            "start_position": vec(start_position),
            "start_target": vec(start_target),
            "linear_velocity_mps": vec(velocity),
            "target_velocity_mps": vec(target_velocity),
            "roll_start_deg": round(roll_start_deg, 5),
            "roll_velocity_deg_s": round(roll_velocity_deg_s, 5),
            "lens_mm": round(lens_mm, 5),
            "lens_velocity_mm_s": round(lens_velocity_mm_s, 5),
            "motion_distribution": "component velocities sampled from clipped Gaussian templates; velocity is in real units per second",
        }

    dolly_speed = trunc_gauss(rng, 0.62, 0.18, 0.28, 1.05)
    dolly_out_speed = trunc_gauss(rng, 0.56, 0.16, 0.25, 0.95)
    truck_speed = trunc_gauss(rng, 0.55, 0.18, 0.22, 1.00)
    crane_speed = trunc_gauss(rng, 0.34, 0.12, 0.12, 0.68)
    top_drift = trunc_gauss(rng, 0.28, 0.10, 0.08, 0.55)
    orbit_deg_s = trunc_gauss(rng, 8.0, 2.2, 3.5, 13.0)
    roll_deg_s = trunc_gauss(rng, 3.0, 1.3, 0.8, 6.5)

    return [
        linear_camera(
            "cam_static_front_left",
            ["Static"],
            (-1.05, -5.45, 2.05),
            (0.0, -0.05, 0.55),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            roll_start_deg=-2.0,
            speed_class="none",
            description="Fixed oblique front-left camera.",
        ),
        linear_camera(
            "cam_dolly_in_left_start",
            ["Dolly_In", "Small_Pan"],
            (-1.45, -6.05, 2.10),
            (-0.10, -0.05, 0.58),
            (0.12, dolly_speed, -0.03),
            (0.05, 0.02, 0.01),
            roll_start_deg=1.2,
            roll_velocity_deg_s=trunc_gauss(rng, 0.6, 0.5, -0.6, 1.6),
            lens_mm=34.0,
            lens_velocity_mm_s=trunc_gauss(rng, 0.8, 0.6, -0.3, 2.0),
            speed_class="medium",
            description="Forward dolly from a left-of-center starting composition.",
        ),
        linear_camera(
            "cam_dolly_out_right_start",
            ["Dolly_Out", "Small_Tilt"],
            (1.20, -3.85, 2.00),
            (0.10, -0.02, 0.52),
            (-0.08, -dolly_out_speed, 0.03),
            (-0.04, 0.0, 0.02),
            roll_start_deg=-1.0,
            roll_velocity_deg_s=trunc_gauss(rng, -0.4, 0.5, -1.6, 0.6),
            lens_mm=36.0,
            lens_velocity_mm_s=trunc_gauss(rng, -0.8, 0.5, -2.0, 0.3),
            speed_class="medium",
            description="Dolly out from a right-of-center close starting view.",
        ),
        linear_camera(
            "cam_truck_right_pan_left",
            ["Truck_Right", "Pan_Left"],
            (-2.15, -5.20, 2.15),
            (0.25, -0.08, 0.58),
            (truck_speed, 0.04, 0.0),
            (-0.16, 0.02, 0.0),
            roll_start_deg=0.8,
            roll_velocity_deg_s=trunc_gauss(rng, 0.2, 0.6, -1.2, 1.6),
            lens_mm=34.0,
            speed_class="medium",
            description="Horizontal truck with compensating pan.",
        ),
        linear_camera(
            "cam_crane_up_tilt_down",
            ["Pedestal_Up", "Tilt_Down"],
            (0.85, -5.25, 1.42),
            (0.12, 0.0, 0.82),
            (-0.03, 0.02, crane_speed),
            (0.0, 0.0, -0.11),
            roll_start_deg=0.0,
            roll_velocity_deg_s=trunc_gauss(rng, 0.1, 0.5, -1.0, 1.1),
            lens_mm=35.0,
            speed_class="slow_medium",
            description="Camera rises while tilting down toward the moving objects.",
        ),
        linear_camera(
            "cam_top_down_drift",
            ["Top_Down", "Truck_Right", "Small_Tilt"],
            (-0.95, -0.55, 5.85),
            (0.02, -0.04, 0.12),
            (top_drift, 0.10, -0.08),
            (0.05, 0.05, 0.0),
            roll_start_deg=4.0,
            roll_velocity_deg_s=trunc_gauss(rng, -0.4, 0.7, -1.8, 0.9),
            lens_mm=43.0,
            speed_class="slow_medium",
            description="High ceiling-like view drifting across the scene.",
        ),
        {
            "id": "cam_orbit_soft_arc",
            "mode": "orbit",
            "primitives": ["Orbit", "Pan_Left"],
            "speed_class": "medium",
            "description": "A short, readable orbit arc rather than a rapid spin.",
            "center": [0.0, -0.05, 0.55],
            "radius": 5.25,
            "height": 2.22,
            "start_angle_deg": -18.0,
            "angular_velocity_deg_s": orbit_deg_s,
            "target_velocity_mps": [0.04, 0.0, 0.0],
            "roll_start_deg": -1.5,
            "roll_velocity_deg_s": trunc_gauss(rng, 0.3, 0.6, -0.8, 1.6),
            "lens_mm": 35.0,
            "lens_velocity_mm_s": 0.0,
            "motion_distribution": "orbit angular velocity sampled from clipped Gaussian in degrees per second",
        },
        linear_camera(
            "cam_gentle_roll_front",
            ["Roll_Clockwise", "Small_Pan"],
            (0.55, -5.35, 2.05),
            (0.05, -0.03, 0.56),
            (-0.05, 0.09, 0.0),
            (-0.04, 0.0, 0.02),
            roll_start_deg=-3.5,
            roll_velocity_deg_s=roll_deg_s,
            lens_mm=34.0,
            speed_class="slow_medium",
            description="Readable roll with a few degrees per second, avoiding uncomfortable fast spins.",
        ),
    ]


def camera_records(camera: dict[str, Any], frames: int) -> list[dict[str, Any]]:
    records = []
    duration_s = (frames - 1) / FPS
    for frame in range(frames):
        elapsed = frame / FPS
        if camera["mode"] == "orbit":
            angle = math.radians(camera["start_angle_deg"] + camera["angular_velocity_deg_s"] * elapsed)
            center = camera["center"]
            position = [
                center[0] + camera["radius"] * math.sin(angle),
                center[1] - camera["radius"] * math.cos(angle),
                camera["height"],
            ]
            target = [
                center[0] + camera["target_velocity_mps"][0] * elapsed,
                center[1] + camera["target_velocity_mps"][1] * elapsed,
                center[2] + camera["target_velocity_mps"][2] * elapsed,
            ]
        else:
            position = [
                camera["start_position"][axis] + camera["linear_velocity_mps"][axis] * elapsed
                for axis in range(3)
            ]
            target = [
                camera["start_target"][axis] + camera["target_velocity_mps"][axis] * elapsed
                for axis in range(3)
            ]
        roll = camera["roll_start_deg"] + camera["roll_velocity_deg_s"] * elapsed
        lens = clamp(camera["lens_mm"] + camera["lens_velocity_mm_s"] * elapsed, 26.0, 55.0)
        records.append(
            {
                "frame": frame,
                "time_s": round(elapsed, 5),
                "duration_s": round(duration_s, 5),
                "position": vec(position),
                "look_at": vec(target),
                "roll_deg": round(roll, 5),
                "lens_mm": round(lens, 5),
            }
        )
    return records


def add_collision_body(p: Any, body: dict[str, Any], client: int) -> int:
    if body["shape"] == "sphere":
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=body["radius"], physicsClientId=client)
    elif body["shape"] == "box":
        half = [v / 2.0 for v in body["size"]]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
    else:
        raise ValueError(body["shape"])
    body_id = p.createMultiBody(
        baseMass=body.get("mass", 0.0),
        baseCollisionShapeIndex=collision,
        basePosition=body["position"],
        baseOrientation=body.get("orientation", [0.0, 0.0, 0.0, 1.0]),
        physicsClientId=client,
    )
    p.changeDynamics(
        body_id,
        -1,
        lateralFriction=body.get("friction", 0.5),
        spinningFriction=0.02,
        rollingFriction=0.02,
        restitution=body.get("restitution", 0.45),
        physicsClientId=client,
    )
    if body.get("mass", 0.0) > 0:
        p.resetBaseVelocity(
            body_id,
            linearVelocity=body.get("velocity", [0.0, 0.0, 0.0]),
            angularVelocity=body.get("angular_velocity", [0.0, 0.0, 0.0]),
            physicsClientId=client,
        )
    return body_id


def static_box(
    name: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    *,
    restitution: float = 0.55,
    friction: float = 0.75,
) -> dict[str, Any]:
    return {
        "name": name,
        "shape": "box",
        "size": vec(size),
        "position": vec(position),
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "mass": 0.0,
        "restitution": restitution,
        "friction": friction,
        "color": [0.6, 0.6, 0.6, 1.0],
    }


def world_collision_bodies(world: dict[str, Any]) -> list[dict[str, Any]]:
    height = world["wall_height"]
    bodies = [
        static_box("ground_collider", (9.0, 9.0, 0.08), (0.0, 0.0, -0.04), restitution=0.55, friction=0.78),
        static_box("back_wall_collider", (9.0, 0.10, height), (0.0, 3.35, height / 2.0), restitution=0.58, friction=0.72),
        static_box("left_wall_collider", (0.10, 6.9, height), (-4.35, 0.0, height / 2.0), restitution=0.58, friction=0.72),
        static_box("right_wall_collider", (0.10, 6.9, height), (4.35, 0.0, height / 2.0), restitution=0.58, friction=0.72),
    ]
    for prop in world.get("static_props", []):
        if prop.get("collider"):
            item = dict(prop)
            item["mass"] = 0.0
            item["restitution"] = 0.45
            item["friction"] = 0.70
            bodies.append(item)
    return bodies


def simulate_physics(physics: dict[str, Any], world: dict[str, Any], frames: int) -> list[dict[str, Any]]:
    import pybullet as p

    client = p.connect(p.DIRECT)
    body_ids: list[tuple[int, dict[str, Any]]] = []
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=client)
        p.setPhysicsEngineParameter(numSolverIterations=80, physicsClientId=client)

        for body in world_collision_bodies(world):
            add_collision_body(p, body, client)
        for body in physics["bodies"]:
            body_ids.append((add_collision_body(p, body, client), body))

        records = []
        steps_per_frame = PHYSICS_HZ // FPS
        for frame in range(frames):
            if frame > 0:
                for _ in range(steps_per_frame):
                    p.stepSimulation(physicsClientId=client)
            object_records = []
            for body_id, body in body_ids:
                position, orientation = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
                linear_velocity, angular_velocity = p.getBaseVelocity(body_id, physicsClientId=client)
                object_records.append(
                    {
                        "name": body["name"],
                        "shape": body["shape"],
                        "radius": body.get("radius"),
                        "size": body.get("size"),
                        "role": body.get("role", "dynamic"),
                        "mass": body.get("mass", 0.0),
                        "color": body["color"],
                        "position": vec(position),
                        "orientation": vec(orientation),
                        "linear_velocity": vec(linear_velocity),
                        "angular_velocity": vec(angular_velocity),
                    }
                )
            records.append({"frame": frame, "time_s": round(frame / FPS, 5), "objects": object_records})
        return records
    finally:
        p.disconnect(client)


def clip_id_for(index: int, camera_id: str, physics_id: str) -> str:
    return f"clip_{index:03d}_{camera_id}_{physics_id}"


def make_pairs(clips: list[dict[str, Any]], target_pairs: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 307)
    by_camera: dict[str, list[dict[str, Any]]] = {}
    by_physics_scene: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for clip in clips:
        by_camera.setdefault(clip["camera_id"], []).append(clip)
        by_physics_scene.setdefault((clip["physics_id"], clip["scene_id"]), []).append(clip)

    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, title: str, a: dict[str, Any], b: dict[str, Any], controlled: str, varied: str, tags: list[str]) -> None:
        key = tuple(sorted([a["clip_id"], b["clip_id"]]))
        if key in seen:
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

    same_camera_budget = max(1, target_pairs // 2)
    for camera_id, group in by_camera.items():
        candidates = [
            (a, b)
            for idx, a in enumerate(group)
            for b in group[idx + 1 :]
            if a["physics_id"] != b["physics_id"]
        ]
        rng.shuffle(candidates)
        per_camera = max(2, math.ceil(same_camera_budget / max(1, len(by_camera))))
        for a, b in candidates[:per_camera]:
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

    same_physics_budget = target_pairs - len(pairs)
    start_count = len(pairs)
    for (physics_id, scene_id), group in by_physics_scene.items():
        candidates = [
            (a, b)
            for idx, a in enumerate(group)
            for b in group[idx + 1 :]
            if a["camera_id"] != b["camera_id"]
        ]
        rng.shuffle(candidates)
        per_physics = max(2, math.ceil(same_physics_budget / max(1, len(by_physics_scene))))
        for a, b in candidates[:per_physics]:
            if len(pairs) - start_count >= same_physics_budget:
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

    attempts = 0
    while len(pairs) < target_pairs and attempts < target_pairs * 40:
        attempts += 1
        a, b = rng.sample(clips, 2)
        add(
            "mixed_review",
            "Mixed review comparison",
            a,
            b,
            "none",
            "camera_id/physics_id/scene_id",
            ["mixed_review"],
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
) -> Path:
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []

    sim_cache: dict[str, list[dict[str, Any]]] = {}
    for physics_index, physics in enumerate(physics_programs):
        world = worlds[physics_index % len(worlds)]
        sim_key = f"{physics['id']}__{world['id']}"
        sim_cache[sim_key] = simulate_physics(physics, world, FRAMES)

    clip_index = 0
    for physics_index, physics in enumerate(physics_programs):
        world = worlds[physics_index % len(worlds)]
        sim_key = f"{physics['id']}__{world['id']}"
        for camera in cameras:
            clip_id = clip_id_for(clip_index, camera["id"], physics["id"])
            camera_frames = camera_records(camera, FRAMES)
            metadata_rel = Path("metadata") / f"{clip_id}.json"
            video_rel = Path("videos") / f"{clip_id}.mp4"
            metadata = {
                "clip_id": clip_id,
                "camera_id": camera["id"],
                "physics_id": physics["id"],
                "scene_id": world["id"],
                "fps": FPS,
                "frames_count": FRAMES,
                "duration_s": round((FRAMES - 1) / FPS, 5),
                "resolution": [WIDTH, HEIGHT],
                "generator": "kubric_style_pybullet_blender_review",
                "simulator": "PyBullet",
                "renderer": "Blender EEVEE",
                "camera_spec": camera,
                "physics_spec": physics,
                "scene_spec": world,
                "camera_frames": camera_frames,
                "physics_frames": sim_cache[sim_key],
                "video": str(video_rel),
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
            render_jobs.append({"clip_id": clip_id, "metadata": str(run_dir / metadata_rel), "video": str(run_dir / video_rel)})
            clip_index += 1

    pair_groups = make_pairs(clips, pair_target, seed)
    ambiguous = []
    dolly_static = next(
        (c["clip_id"] for c in clips if c["camera_id"] == "cam_dolly_in_left_start" and c["physics_id"] == "phys_static_sphere"),
        None,
    )
    static_toward = next(
        (c["clip_id"] for c in clips if c["camera_id"] == "cam_static_front_left" and c["physics_id"] == "phys_sphere_toward_camera"),
        None,
    )
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
        "generator": "kubric_style_pybullet_blender_review",
        "description": "Small review bank with real PyBullet rigid-body simulation and Blender rendering. Same-physics pairs reuse identical physics_frames and scene_id; same-camera pairs reuse identical camera_frames.",
        "important_note": "The full PyPI kubric package was not used here because dependency resolution failed in this container; this bank uses the same PyBullet/Blender split intended for Kubric-style data review and avoids hand-authored kinematic object motion.",
        "fps": FPS,
        "frames_count": FRAMES,
        "duration_s": round((FRAMES - 1) / FPS, 5),
        "resolution": [WIDTH, HEIGHT],
        "seed": seed,
        "simulator": "PyBullet",
        "renderer": "Blender 3.6.5 EEVEE",
        "camera_motion_model": "camera start positions and per-second velocities are sampled from clipped Gaussian templates; duration does not scale the sampled velocity down",
        "physics_motion_model": "rigid-body initial velocities, gravity, contacts, friction, and restitution are simulated by PyBullet at 240 Hz",
        "camera_reference": cameras,
        "physics_reference": physics_programs,
        "scene_reference": worlds,
        "clips": clips,
        "pair_groups": pair_groups,
        "ambiguous_equivalence_groups": ambiguous,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    jobs_path = run_dir / "render_jobs.json"
    jobs_path.write_text(json.dumps({"run_id": run_id, "jobs": render_jobs}, indent=2), encoding="utf-8")
    update_index(run_root, run_id)
    return jobs_path


def render_with_blender(blender_bin: Path, jobs_path: Path) -> None:
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
        )


def blender_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def make_blender_material(bpy: Any, name: str, color: list[float]) -> Any:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def blender_add_cube(bpy: Any, name: str, position: list[float], size: list[float], color: list[float]) -> Any:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0], size[1], size[2])
    obj.data.materials.append(make_blender_material(bpy, f"mat_{name}", color))
    return obj


def blender_add_sphere(bpy: Any, name: str, position: list[float], radius: float, color: list[float]) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(make_blender_material(bpy, f"mat_{name}", color))
    return obj


def blender_set_linear_interpolation(obj: Any) -> None:
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def blender_look_at(obj: Any, target_values: list[float], roll_deg: float) -> None:
    from mathutils import Vector

    target = Vector(target_values)
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    obj.rotation_euler.rotate_axis("Z", math.radians(roll_deg))


def blender_reset_scene(bpy: Any, frames: int) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.fps = FPS
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 4
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3
    scene.eevee.gtao_factor = 1.4
    scene.world.color = (0.76, 0.80, 0.84)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"


def blender_add_world(bpy: Any, world: dict[str, Any]) -> None:
    height = world["wall_height"]
    blender_add_cube(bpy, "ground", [0.0, 0.0, -0.04], [9.0, 9.0, 0.08], world["ground_color"])
    blender_add_cube(bpy, "back_wall", [0.0, 3.35, height / 2.0], [9.0, 0.10, height], world["back_wall_color"])
    blender_add_cube(bpy, "left_wall", [-4.35, 0.0, height / 2.0], [0.10, 6.9, height], world["side_wall_color"])
    blender_add_cube(bpy, "right_wall", [4.35, 0.0, height / 2.0], [0.10, 6.9, height], world["side_wall_color"])

    if world["wall_style"] == "horizontal_bands":
        for idx, z in enumerate([0.65, 1.25, 1.85, 2.45]):
            blender_add_cube(bpy, f"wall_band_{idx}", [0.0, 3.285, z], [8.35, 0.035, 0.035], [0.47, 0.56, 0.62, 1.0])
    elif world["wall_style"] == "wall_panels":
        for idx, x in enumerate([-2.8, -1.4, 0.0, 1.4, 2.8]):
            blender_add_cube(bpy, f"wall_panel_{idx}", [x, 3.285, 1.05], [0.70, 0.035, 1.42], [0.69, 0.62, 0.54, 1.0])

    if world["floor_style"] == "subtle_marks":
        for idx, (x, y) in enumerate([(-2.9, -2.3), (2.7, -2.1), (-2.5, 2.2), (2.6, 2.0)]):
            blender_add_cube(bpy, f"floor_mark_{idx}", [x, y, 0.012], [0.42, 0.035, 0.018], [0.48, 0.50, 0.48, 1.0])
    elif world["floor_style"] == "corner_marks":
        for idx, (x, y) in enumerate([(-3.1, -2.4), (3.0, -2.2), (-3.0, 2.25), (3.1, 2.05)]):
            blender_add_cube(bpy, f"corner_mark_{idx}", [x, y, 0.012], [0.34, 0.08, 0.018], [0.36, 0.43, 0.36, 1.0])

    for prop in world.get("static_props", []):
        if prop["shape"] == "box":
            blender_add_cube(bpy, prop["name"], prop["position"], prop["size"], prop["color"])

    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 5.2))
    key = bpy.context.object
    key.name = "key_light"
    key.data.energy = 520
    key.data.size = 5.5
    bpy.ops.object.light_add(type="POINT", location=(-3.2, 1.0, 3.8))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 70


def render_one_job(bpy: Any, job: dict[str, Any]) -> None:
    metadata = json.loads(Path(job["metadata"]).read_text(encoding="utf-8"))
    blender_reset_scene(bpy, metadata["frames_count"])
    blender_add_world(bpy, metadata["scene_spec"])

    object_meshes: dict[str, Any] = {}
    for frame_record in metadata["physics_frames"]:
        frame_no = frame_record["frame"] + 1
        for object_record in frame_record["objects"]:
            name = object_record["name"]
            if name not in object_meshes:
                if object_record["shape"] == "sphere":
                    obj = blender_add_sphere(bpy, name, object_record["position"], object_record["radius"], object_record["color"])
                elif object_record["shape"] == "box":
                    obj = blender_add_cube(bpy, name, object_record["position"], object_record["size"], object_record["color"])
                else:
                    raise ValueError(object_record["shape"])
                object_meshes[name] = obj
            obj = object_meshes[name]
            obj.location = object_record["position"]
            quat = object_record["orientation"]
            from mathutils import Quaternion

            obj.rotation_euler = Quaternion((quat[3], quat[0], quat[1], quat[2])).to_euler()
            obj.keyframe_insert("location", frame=frame_no)
            obj.keyframe_insert("rotation_euler", frame=frame_no)

    for obj in object_meshes.values():
        blender_set_linear_interpolation(obj)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = metadata["camera_id"]
    bpy.context.scene.camera = camera
    for camera_record in metadata["camera_frames"]:
        frame_no = camera_record["frame"] + 1
        camera.location = camera_record["position"]
        camera.data.lens = camera_record["lens_mm"]
        blender_look_at(camera, camera_record["look_at"], camera_record["roll_deg"])
        camera.keyframe_insert("location", frame=frame_no)
        camera.keyframe_insert("rotation_euler", frame=frame_no)
        camera.data.keyframe_insert("lens", frame=frame_no)
    blender_set_linear_interpolation(camera)
    if camera.data.animation_data and camera.data.animation_data.action:
        for fcurve in camera.data.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"

    video_path = Path(job["video"])
    video_path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(video_path)
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"
    bpy.ops.render.render(animation=True, write_still=False)


def render_jobs_main() -> None:
    import bpy

    parser = argparse.ArgumentParser()
    parser.add_argument("--render-jobs", type=Path, required=True)
    args = parser.parse_args(blender_argv())
    payload = json.loads(args.render_jobs.read_text(encoding="utf-8"))
    for idx, job in enumerate(payload["jobs"], start=1):
        print(f"[{idx}/{len(payload['jobs'])}] rendering {job['clip_id']}")
        render_one_job(bpy, job)


def generate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="kubric_review_v0")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--camera-limit", type=int, default=6)
    parser.add_argument("--physics-limit", type=int, default=7)
    parser.add_argument("--pairs", type=int, default=72)
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
    jobs_path = write_run(args.run_id, run_dir, args.run_root, cameras, programs, worlds, args.pairs, args.seed)
    if not args.no_render:
        render_with_blender(args.blender_bin, jobs_path)
    print(f"Wrote {len(cameras) * len(programs)} clips to {run_dir}")


def main() -> None:
    if "--render-jobs" in blender_argv():
        render_jobs_main()
    else:
        generate_main()


if __name__ == "__main__":
    main()
