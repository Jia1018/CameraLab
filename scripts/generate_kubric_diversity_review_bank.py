#!/usr/bin/env python3
"""Generate a compact diversity review bank for camera and object motion.

This script reuses the official Kubric/PyBullet/Blender pipeline from
``generate_official_kubric_review_bank.py`` but supplies broader camera and
physics templates for manual review before a larger batch_v2.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import generate_official_kubric_review_bank as gen


RUN_ROOT = gen.PROJECT_ROOT / "site" / "assets" / "runs"
DEFAULT_RUN_ID = "kubric_diversity_review_v0"
DEFAULT_SEED = 20260722


def camera_records(camera: dict[str, Any], frames: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame in range(frames):
        elapsed = frame / gen.FPS
        target = [
            camera["start_target"][axis] + camera["target_velocity_mps"][axis] * elapsed
            for axis in range(3)
        ]
        if camera.get("path_model") == "orbit":
            center = camera.get("orbit_center", camera["start_target"])
            angle = math.radians(camera["orbit_start_deg"] + camera["orbit_velocity_deg_s"] * elapsed)
            radius = camera["orbit_radius_m"]
            height = camera["orbit_height_m"] + camera.get("orbit_height_velocity_mps", 0.0) * elapsed
            position = [center[0] + radius * math.sin(angle), center[1] - radius * math.cos(angle), height]
        else:
            position = [
                camera["start_position"][axis] + camera["linear_velocity_mps"][axis] * elapsed
                for axis in range(3)
            ]
        roll = camera["roll_start_deg"] + camera["roll_velocity_deg_s"] * elapsed
        lens = gen.clamp(camera["lens_mm"] + camera["lens_velocity_mm_s"] * elapsed, 28.0, 50.0)
        records.append(
            {
                "frame": frame,
                "time_s": round(elapsed, 5),
                "position": gen.vec(position),
                "look_at": gen.vec(target),
                "quaternion_wxyz": gen.look_at_quat_with_roll(gen.vec(position), gen.vec(target), roll),
                "roll_deg": round(roll, 5),
                "lens_mm": round(lens, 5),
            }
        )
    return records


def camera_specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 707)

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
        path_model: str = "linear",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = {
            "id": camera_id,
            "primitives": primitives,
            "speed_class": speed_class,
            "path_model": path_model,
            "start_position": gen.vec(start_position),
            "start_target": gen.vec(start_target),
            "linear_velocity_mps": gen.vec(velocity),
            "target_velocity_mps": gen.vec(target_velocity),
            "roll_start_deg": round(roll_start_deg, 5),
            "roll_velocity_deg_s": round(roll_velocity_deg_s, 5),
            "lens_mm": round(lens_mm, 5),
            "lens_velocity_mm_s": round(lens_velocity_mm_s, 5),
            "motion_distribution": "camera position, target, roll, lens, and orbit velocities are clipped Gaussian samples",
        }
        if extra:
            spec.update(extra)
        return spec

    dolly_in = gen.trunc_gauss(rng, 0.58, 0.15, 0.28, 0.92)
    dolly_out = gen.trunc_gauss(rng, 0.40, 0.12, 0.18, 0.68)
    truck = gen.trunc_gauss(rng, 0.48, 0.13, 0.20, 0.80)
    crane = gen.trunc_gauss(rng, 0.30, 0.10, 0.10, 0.55)
    top_drift = gen.trunc_gauss(rng, 0.24, 0.08, 0.08, 0.42)
    orbit_radius = gen.trunc_gauss(rng, 4.10, 0.28, 3.55, 4.70)
    orbit_start_deg = gen.trunc_gauss(rng, -38.0, 8.0, -55.0, -20.0)
    orbit_velocity = gen.trunc_gauss(rng, 7.5, 2.3, 3.0, 13.0)
    orbit_angle = math.radians(orbit_start_deg)
    orbit_position = (
        orbit_radius * math.sin(orbit_angle),
        -orbit_radius * math.cos(orbit_angle),
        1.48,
    )

    return [
        camera(
            "cam_static_left_high",
            ["static", "left_start", "high_angle", "slight_roll"],
            "none",
            (-2.45, -4.60, 1.80),
            (-0.10, -0.05, 0.42),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, -1.2, 0.9, -3.2, 1.2),
            lens_mm=gen.trunc_gauss(rng, 34.0, 2.0, 30.0, 40.0),
        ),
        camera(
            "cam_dolly_in_right_start",
            ["dolly_in", "right_start", "target_drift", "mild_zoom"],
            "medium",
            (2.35, -5.05, 1.55),
            (0.05, 0.00, 0.38),
            (-0.10, dolly_in, 0.02),
            (gen.trunc_gauss(rng, 0.03, 0.03, -0.04, 0.10), 0.0, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.8, 0.9, -1.0, 3.0),
            roll_velocity_deg_s=gen.trunc_gauss(rng, 0.25, 0.35, -0.35, 1.05),
            lens_mm=gen.trunc_gauss(rng, 33.0, 2.0, 29.0, 39.0),
            lens_velocity_mm_s=gen.trunc_gauss(rng, 0.55, 0.30, 0.0, 1.20),
        ),
        camera(
            "cam_truck_left_pan_right",
            ["truck_left", "pan_right", "off_center_start"],
            "medium",
            (2.85, -4.10, 1.35),
            (0.35, 0.00, 0.38),
            (-truck, 0.03, 0.0),
            (-0.08, 0.02, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, -0.4, 0.9, -2.6, 1.8),
            roll_velocity_deg_s=gen.trunc_gauss(rng, -0.16, 0.30, -0.85, 0.45),
            lens_mm=gen.trunc_gauss(rng, 36.0, 2.5, 31.0, 42.0),
        ),
        camera(
            "cam_crane_up_tilt_down",
            ["crane_up", "tilt_down", "front_start"],
            "slow",
            (-0.75, -4.60, 1.10),
            (-0.05, -0.05, 0.32),
            (0.05, 0.02, crane),
            (0.0, 0.0, -0.05),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 0.9, -2.0, 2.0),
            roll_velocity_deg_s=gen.trunc_gauss(rng, 0.10, 0.25, -0.45, 0.65),
            lens_mm=gen.trunc_gauss(rng, 37.0, 2.0, 32.0, 43.0),
        ),
        camera(
            "cam_top_down_drift",
            ["top_down", "ceiling_start", "drift"],
            "slow",
            (0.75, -0.80, 5.15),
            (0.05, -0.02, 0.20),
            (-top_drift, top_drift * 0.55, -0.04),
            (0.02, 0.0, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 5.5, 1.8, 2.0, 10.0),
            roll_velocity_deg_s=gen.trunc_gauss(rng, 0.55, 0.40, -0.15, 1.45),
            lens_mm=gen.trunc_gauss(rng, 38.0, 2.5, 33.0, 45.0),
        ),
        camera(
            "cam_orbit_left_arc",
            ["orbit", "arc_left_to_front", "look_at_center"],
            "slow",
            orbit_position,
            (0.00, 0.00, 0.34),
            (0.0, 0.0, 0.0),
            (0.015, 0.00, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 0.8, -1.8, 1.8),
            roll_velocity_deg_s=gen.trunc_gauss(rng, 0.05, 0.20, -0.35, 0.50),
            lens_mm=gen.trunc_gauss(rng, 36.0, 2.2, 31.0, 43.0),
            path_model="orbit",
            extra={
                "orbit_center": [0.0, 0.0, 0.34],
                "orbit_radius_m": orbit_radius,
                "orbit_start_deg": orbit_start_deg,
                "orbit_velocity_deg_s": orbit_velocity,
                "orbit_height_m": 1.48,
                "orbit_height_velocity_mps": gen.trunc_gauss(rng, 0.015, 0.015, -0.01, 0.045),
            },
        ),
        camera(
            "cam_dolly_out_low_start",
            ["dolly_out", "low_start", "tilt_up"],
            "slow",
            (-1.55, -2.65, 0.78),
            (-0.10, 0.00, 0.28),
            (0.04, -dolly_out, 0.02),
            (0.02, 0.00, gen.trunc_gauss(rng, 0.045, 0.02, 0.00, 0.09)),
            roll_start_deg=gen.trunc_gauss(rng, -0.5, 0.9, -2.4, 1.6),
            roll_velocity_deg_s=gen.trunc_gauss(rng, 0.08, 0.28, -0.45, 0.65),
            lens_mm=gen.trunc_gauss(rng, 34.0, 2.2, 29.0, 41.0),
        ),
        camera(
            "cam_low_truck_right_pan_left",
            ["truck_right", "pan_left", "low_start"],
            "medium",
            (-2.45, -3.65, 0.92),
            (-0.35, -0.02, 0.32),
            (gen.trunc_gauss(rng, 0.38, 0.12, 0.18, 0.68), 0.02, 0.0),
            (0.10, 0.01, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.2, 0.8, -1.8, 2.2),
            roll_velocity_deg_s=gen.trunc_gauss(rng, -0.12, 0.28, -0.75, 0.45),
            lens_mm=gen.trunc_gauss(rng, 32.0, 2.0, 28.0, 38.0),
        ),
    ]


def physics_specs(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 808)

    def low_friction() -> float:
        return gen.trunc_gauss(rng, 0.055, 0.020, 0.015, 0.12)

    def medium_friction() -> float:
        return gen.trunc_gauss(rng, 0.24, 0.07, 0.08, 0.44)

    def high_friction() -> float:
        return gen.trunc_gauss(rng, 0.58, 0.12, 0.32, 0.82)

    def bounce() -> float:
        return gen.trunc_gauss(rng, 0.76, 0.10, 0.52, 0.94)

    def mass(mean: float = 1.0) -> float:
        return gen.trunc_gauss(rng, mean, 0.25, 0.45, 1.85)

    appearance_index = 0

    def appearance(role: str, *, profile: str | None = None) -> dict[str, Any]:
        nonlocal appearance_index
        sampled = gen.sample_appearance(rng, role, appearance_index, profile=profile)
        appearance_index += 1
        return sampled

    r_rest = gen.trunc_gauss(rng, 0.25, 0.035, 0.18, 0.33)
    rest_box = (
        gen.trunc_gauss(rng, 0.18, 0.03, 0.12, 0.25),
        gen.trunc_gauss(rng, 0.32, 0.04, 0.22, 0.42),
        gen.trunc_gauss(rng, 0.16, 0.025, 0.11, 0.22),
    )
    r_drop = gen.trunc_gauss(rng, 0.25, 0.025, 0.20, 0.31)
    r_a = gen.trunc_gauss(rng, 0.25, 0.025, 0.20, 0.31)
    r_b = gen.trunc_gauss(rng, 0.29, 0.03, 0.22, 0.36)
    r_hit = gen.trunc_gauss(rng, 0.23, 0.025, 0.19, 0.29)
    tall_block = (
        gen.trunc_gauss(rng, 0.16, 0.025, 0.11, 0.22),
        gen.trunc_gauss(rng, 0.18, 0.030, 0.12, 0.25),
        gen.trunc_gauss(rng, 0.52, 0.060, 0.40, 0.66),
    )
    moving_box = (
        gen.trunc_gauss(rng, 0.34, 0.045, 0.24, 0.46),
        gen.trunc_gauss(rng, 0.18, 0.035, 0.11, 0.28),
        gen.trunc_gauss(rng, 0.20, 0.030, 0.14, 0.28),
    )
    r_box_sphere = gen.trunc_gauss(rng, 0.27, 0.035, 0.20, 0.35)
    scatter_cube = (
        gen.trunc_gauss(rng, 0.23, 0.035, 0.16, 0.33),
        gen.trunc_gauss(rng, 0.23, 0.035, 0.16, 0.33),
        gen.trunc_gauss(rng, 0.22, 0.030, 0.16, 0.30),
    )
    scatter_small_box = (
        gen.trunc_gauss(rng, 0.15, 0.025, 0.10, 0.21),
        gen.trunc_gauss(rng, 0.28, 0.040, 0.18, 0.38),
        gen.trunc_gauss(rng, 0.14, 0.025, 0.09, 0.20),
    )
    scatter_r_a = gen.trunc_gauss(rng, 0.23, 0.03, 0.18, 0.31)
    scatter_r_b = gen.trunc_gauss(rng, 0.20, 0.025, 0.16, 0.27)

    return [
        {
            "id": "phys_static_mixed_pair",
            "kind": "multi_object_static",
            "speed_class": "none",
            "description": "A resting sphere and rectangular box for appearance, size, and camera-only review.",
            "sample_model": "sphere radius, box aspect ratio, mass, friction, restitution, color, and material are clipped Gaussian samples.",
            "expected_contacts": [],
            "bodies": [
                gen.sphere(
                    "resting_sphere",
                    r_rest,
                    (-0.45, -0.28, r_rest),
                    appearance("resting_sphere", profile="satin"),
                    mass=mass(0.8),
                    restitution=gen.trunc_gauss(rng, 0.42, 0.08, 0.25, 0.62),
                    friction=medium_friction(),
                ),
                gen.cube(
                    "resting_rect_box",
                    rest_box,
                    (0.46, 0.20, rest_box[2]),
                    appearance("resting_rect_box", profile="matte"),
                    mass=mass(1.1),
                    restitution=gen.trunc_gauss(rng, 0.34, 0.08, 0.18, 0.55),
                    friction=high_friction(),
                ),
            ],
        },
        {
            "id": "phys_drop_bounce",
            "kind": "gravity_bounce",
            "speed_class": "mixed",
            "description": "Sphere falls and bounces under gravity with lateral drift.",
            "sample_model": "radius, mass, lateral velocity, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["drop_bounce_sphere", "ground"]],
            "bodies": [
                gen.sphere(
                    "drop_bounce_sphere",
                    r_drop,
                    (
                        gen.trunc_gauss(rng, -0.36, 0.16, -0.70, -0.05),
                        gen.trunc_gauss(rng, 0.08, 0.10, -0.12, 0.30),
                        gen.trunc_gauss(rng, 1.50, 0.17, 1.18, 1.88),
                    ),
                    appearance("drop_bounce_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 0.52, 0.15, 0.22, 0.88), 0.08, -0.35),
                    mass=mass(0.8),
                    restitution=gen.trunc_gauss(rng, 0.86, 0.06, 0.72, 0.96),
                    friction=low_friction(),
                )
            ],
        },
        {
            "id": "phys_two_sphere_collision",
            "kind": "two_body_collision",
            "speed_class": "medium",
            "description": "Two differently sized spheres cross the center and collide.",
            "sample_model": "radii, masses, velocities, friction, restitution, colors, and materials are clipped Gaussian samples.",
            "expected_contacts": [["cross_sphere_a", "cross_sphere_b"]],
            "bodies": [
                gen.sphere(
                    "cross_sphere_a",
                    r_a,
                    (-1.05, -0.18, r_a),
                    appearance("cross_sphere_a", profile="glossy"),
                    velocity=(gen.trunc_gauss(rng, 1.60, 0.22, 1.15, 2.10), 0.20, 0.0),
                    mass=mass(0.85),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                gen.sphere(
                    "cross_sphere_b",
                    r_b,
                    (1.05, 0.18, r_b),
                    appearance("cross_sphere_b", profile="matte"),
                    velocity=(gen.trunc_gauss(rng, -1.50, 0.22, -2.05, -1.05), -0.20, 0.0),
                    mass=mass(1.25),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
            ],
        },
        {
            "id": "phys_sphere_hits_tall_block",
            "kind": "dynamic_static_collision",
            "speed_class": "medium",
            "description": "Sphere collides with a tall visible block obstacle.",
            "sample_model": "sphere radius, tall-block aspect ratio, velocity, mass, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["hit_sphere", "tall_block_target"]],
            "bodies": [
                gen.sphere(
                    "hit_sphere",
                    r_hit,
                    (-0.92, 0.0, r_hit),
                    appearance("hit_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 1.90, 0.24, 1.40, 2.42), 0.0, 0.0),
                    mass=mass(0.9),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                gen.cube(
                    "tall_block_target",
                    tall_block,
                    (0.04, 0.0, tall_block[2]),
                    appearance("tall_block_target", profile="brushed_metal"),
                    mass=0.0,
                    restitution=gen.trunc_gauss(rng, 0.60, 0.08, 0.40, 0.78),
                    friction=medium_friction(),
                    static=True,
                ),
            ],
        },
        {
            "id": "phys_box_sphere_collision",
            "kind": "box_sphere_collision",
            "speed_class": "medium",
            "description": "A rectangular box and a sphere meet near the center and rebound.",
            "sample_model": "box aspect ratio, sphere radius, masses, velocities, friction, and restitution are clipped Gaussian samples.",
            "expected_contacts": [["moving_rect_box", "moving_sphere"]],
            "bodies": [
                gen.cube(
                    "moving_rect_box",
                    moving_box,
                    (-1.12, -0.14, moving_box[2]),
                    appearance("moving_rect_box", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, 1.42, 0.20, 1.00, 1.88), 0.16, 0.0),
                    angular_velocity=(0.0, -1.4, 0.25),
                    mass=mass(1.25),
                    restitution=gen.trunc_gauss(rng, 0.58, 0.08, 0.38, 0.74),
                    friction=low_friction(),
                ),
                gen.sphere(
                    "moving_sphere",
                    r_box_sphere,
                    (0.95, 0.15, r_box_sphere),
                    appearance("moving_sphere", profile="glossy"),
                    velocity=(gen.trunc_gauss(rng, -1.28, 0.20, -1.78, -0.88), -0.16, 0.0),
                    mass=mass(0.95),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
            ],
        },
        {
            "id": "phys_four_body_scatter",
            "kind": "four_body_collision",
            "speed_class": "mixed",
            "description": "Four objects with mixed size and shape scatter near the center.",
            "sample_model": "object count, sizes, masses, velocities, friction, restitution, colors, and materials are sampled from clipped Gaussian templates.",
            "expected_contacts": [["scatter_sphere_a", "scatter_cube"]],
            "bodies": [
                gen.sphere(
                    "scatter_sphere_a",
                    scatter_r_a,
                    (-1.30, -0.30, scatter_r_a),
                    appearance("scatter_sphere_a", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 1.72, 0.22, 1.22, 2.20), 0.38, 0.0),
                    mass=mass(0.85),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                gen.cube(
                    "scatter_cube",
                    scatter_cube,
                    (-0.05, 0.05, scatter_cube[2]),
                    appearance("scatter_cube", profile="matte"),
                    velocity=(0.04, 0.00, 0.0),
                    angular_velocity=(0.0, 1.2, 0.3),
                    mass=mass(1.15),
                    restitution=gen.trunc_gauss(rng, 0.56, 0.08, 0.36, 0.74),
                    friction=low_friction(),
                ),
                gen.sphere(
                    "scatter_sphere_b",
                    scatter_r_b,
                    (1.34, 0.42, scatter_r_b),
                    appearance("scatter_sphere_b", profile="glossy"),
                    velocity=(gen.trunc_gauss(rng, -1.08, 0.16, -1.48, -0.78), -0.26, 0.0),
                    mass=mass(0.65),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                gen.cube(
                    "scatter_small_box",
                    scatter_small_box,
                    (0.88, -0.58, scatter_small_box[2]),
                    appearance("scatter_small_box", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, -0.24, 0.10, -0.50, -0.06), 0.28, 0.0),
                    angular_velocity=(0.0, -0.8, 0.7),
                    mass=mass(0.75),
                    restitution=gen.trunc_gauss(rng, 0.62, 0.08, 0.42, 0.80),
                    friction=low_friction(),
                ),
            ],
        },
    ]


def world_specs() -> list[dict[str, Any]]:
    worlds = gen.world_specs()
    worlds.extend(
        [
            {
                "id": "scene_green_low_wall",
                "ground_color": [0.56, 0.64, 0.57, 1.0],
                "back_wall_color": [0.67, 0.76, 0.67, 1.0],
                "side_wall_color": [0.61, 0.69, 0.62, 1.0],
                "wall_height": 2.20,
                "wall_style": "horizontal_bands",
                "floor_style": "subtle_marks",
                "static_props": [],
            },
            {
                "id": "scene_neutral_tall_panel",
                "ground_color": [0.62, 0.61, 0.65, 1.0],
                "back_wall_color": [0.74, 0.72, 0.78, 1.0],
                "side_wall_color": [0.67, 0.66, 0.72, 1.0],
                "wall_height": 2.75,
                "wall_style": "wall_panels",
                "floor_style": "plain",
                "static_props": [
                    {
                        "name": "far_side_reference_block",
                        "shape": "box",
                        "half_extents": [0.22, 0.20, 0.14],
                        "position": [3.25, 2.05, 0.14],
                        "color": [0.40, 0.42, 0.48, 1.0],
                        "collider": True,
                    }
                ],
            },
        ]
    )
    return worlds


def update_manifest_for_diversity(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator"] = "official_kubric_diversity_review"
    manifest["description"] = (
        "Diversity review bank for jointly inspecting camera trajectory variety, object appearance, "
        "object shape proxies, material sampling, and physical event variety before batch_v2."
    )
    manifest["diversity_axes"] = [
        "camera path family",
        "camera speed class",
        "camera start viewpoint",
        "object count",
        "object size/aspect ratio",
        "object color/material",
        "collision type",
        "scene color/wall style",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=gen.DEFAULT_BLENDER)
    parser.add_argument("--kubric-site-packages", type=Path, default=gen.DEFAULT_KUBRIC_SITE_PACKAGES)
    parser.add_argument("--camera-limit", type=int, default=6)
    parser.add_argument("--physics-limit", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--frames", type=int, default=gen.FRAMES)
    parser.add_argument("--width", type=int, default=gen.WIDTH)
    parser.add_argument("--height", type=int, default=gen.HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_root / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} already exists; pass --overwrite to replace generated assets.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gen.camera_records = camera_records
    cameras = camera_specs(args.seed)[: args.camera_limit]
    programs = physics_specs(args.seed)[: args.physics_limit]
    worlds = world_specs()
    jobs_path = gen.write_run(
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
    update_manifest_for_diversity(run_dir)
    if not args.no_render:
        gen.render_with_blender(args.blender_bin, jobs_path, args.kubric_site_packages)
        gen.encode_videos(jobs_path)
    print(f"Wrote {len(cameras) * len(programs)} diversity Kubric clips to {run_dir}")


if __name__ == "__main__":
    generate_main()
