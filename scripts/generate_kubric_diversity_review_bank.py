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
    brisk_dolly_multiplier = gen.trunc_gauss(rng, 1.34, 0.10, 1.20, 1.50)
    brisk_dolly_in = dolly_in * brisk_dolly_multiplier
    dolly_out = gen.trunc_gauss(rng, 0.40, 0.12, 0.18, 0.68)
    truck = gen.trunc_gauss(rng, 0.48, 0.13, 0.20, 0.80)
    brisk_truck_multiplier = gen.trunc_gauss(rng, 1.32, 0.10, 1.20, 1.50)
    brisk_truck = truck * brisk_truck_multiplier
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
            extra={"speed_sampling_band": "static", "speed_multiplier": 1.0},
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
            extra={"speed_sampling_band": "baseline", "speed_multiplier": 1.0},
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
            extra={"speed_sampling_band": "baseline", "speed_multiplier": 1.0},
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
            extra={"speed_sampling_band": "baseline", "speed_multiplier": 1.0},
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
            extra={"speed_sampling_band": "baseline", "speed_multiplier": 1.0},
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
                "speed_sampling_band": "baseline",
                "speed_multiplier": 1.0,
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
            extra={"speed_sampling_band": "baseline", "speed_multiplier": 1.0},
        ),
        camera(
            "cam_brisk_dolly_in_right_start",
            ["dolly_in", "right_start", "target_drift", "brisk_speed"],
            "brisk",
            (2.15, -5.10, 1.48),
            (0.02, 0.00, 0.36),
            (-0.08, brisk_dolly_in, 0.02),
            (gen.trunc_gauss(rng, 0.04, 0.03, -0.02, 0.11), 0.0, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.6, 0.8, -0.8, 2.6),
            roll_velocity_deg_s=gen.trunc_gauss(rng, 0.35, 0.28, -0.15, 0.95),
            lens_mm=gen.trunc_gauss(rng, 33.0, 2.0, 29.0, 39.0),
            lens_velocity_mm_s=gen.trunc_gauss(rng, 0.70, 0.26, 0.10, 1.20),
            extra={"speed_sampling_band": "brisk", "speed_multiplier": round(brisk_dolly_multiplier, 5)},
        ),
        camera(
            "cam_brisk_truck_left_pan_right",
            ["truck_left", "pan_right", "off_center_start", "brisk_speed"],
            "brisk",
            (2.95, -4.00, 1.30),
            (0.32, 0.02, 0.36),
            (-brisk_truck, 0.04, 0.0),
            (-0.10, 0.02, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, -0.3, 0.8, -2.0, 1.6),
            roll_velocity_deg_s=gen.trunc_gauss(rng, -0.22, 0.26, -0.78, 0.30),
            lens_mm=gen.trunc_gauss(rng, 35.0, 2.2, 30.0, 41.0),
            extra={"speed_sampling_band": "brisk", "speed_multiplier": round(brisk_truck_multiplier, 5)},
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
    cyl_r = gen.trunc_gauss(rng, 0.18, 0.025, 0.13, 0.24)
    cyl_depth = gen.trunc_gauss(rng, 0.46, 0.06, 0.34, 0.62)
    cone_r = gen.trunc_gauss(rng, 0.22, 0.03, 0.16, 0.30)
    cone_depth = gen.trunc_gauss(rng, 0.48, 0.06, 0.36, 0.64)
    cap_r = gen.trunc_gauss(rng, 0.14, 0.02, 0.10, 0.19)
    cap_mid = gen.trunc_gauss(rng, 0.34, 0.05, 0.22, 0.48)
    dyn_cyl_r = gen.trunc_gauss(rng, 0.18, 0.025, 0.13, 0.23)
    dyn_cyl_depth = gen.trunc_gauss(rng, 0.52, 0.06, 0.38, 0.68)
    dyn_cap_r = gen.trunc_gauss(rng, 0.13, 0.02, 0.10, 0.18)
    dyn_cap_mid = gen.trunc_gauss(rng, 0.44, 0.06, 0.30, 0.60)
    dyn_cone_r = gen.trunc_gauss(rng, 0.20, 0.025, 0.15, 0.26)
    dyn_cone_depth = gen.trunc_gauss(rng, 0.42, 0.05, 0.30, 0.56)
    shape_hit_sphere_r = gen.trunc_gauss(rng, 0.20, 0.025, 0.15, 0.26)
    multi_drop_r_a = gen.trunc_gauss(rng, 0.21, 0.025, 0.16, 0.28)
    multi_drop_r_b = gen.trunc_gauss(rng, 0.18, 0.025, 0.14, 0.24)
    multi_drop_box = (
        gen.trunc_gauss(rng, 0.18, 0.025, 0.13, 0.25),
        gen.trunc_gauss(rng, 0.20, 0.030, 0.14, 0.28),
        gen.trunc_gauss(rng, 0.16, 0.025, 0.11, 0.22),
    )
    drop_hit_sphere_r = gen.trunc_gauss(rng, 0.20, 0.025, 0.16, 0.27)
    drop_hit_box = (
        gen.trunc_gauss(rng, 0.26, 0.035, 0.18, 0.35),
        gen.trunc_gauss(rng, 0.22, 0.030, 0.15, 0.30),
        gen.trunc_gauss(rng, 0.17, 0.025, 0.12, 0.23),
    )
    air_drop_r_a = gen.trunc_gauss(rng, 0.20, 0.020, 0.16, 0.25)
    air_drop_r_b = gen.trunc_gauss(rng, 0.20, 0.020, 0.16, 0.25)
    air_drop_box = (
        gen.trunc_gauss(rng, 0.16, 0.025, 0.11, 0.22),
        gen.trunc_gauss(rng, 0.18, 0.030, 0.12, 0.25),
        gen.trunc_gauss(rng, 0.15, 0.025, 0.10, 0.21),
    )
    air_sphere_box_r = gen.trunc_gauss(rng, 0.20, 0.020, 0.16, 0.25)
    air_sphere_box_half = (
        gen.trunc_gauss(rng, 0.17, 0.025, 0.12, 0.23),
        gen.trunc_gauss(rng, 0.18, 0.030, 0.12, 0.25),
        gen.trunc_gauss(rng, 0.15, 0.025, 0.10, 0.21),
    )
    air_box_a = (
        gen.trunc_gauss(rng, 0.16, 0.025, 0.11, 0.22),
        gen.trunc_gauss(rng, 0.18, 0.030, 0.12, 0.25),
        gen.trunc_gauss(rng, 0.14, 0.020, 0.10, 0.19),
    )
    air_box_b = (
        gen.trunc_gauss(rng, 0.18, 0.025, 0.12, 0.24),
        gen.trunc_gauss(rng, 0.16, 0.030, 0.11, 0.23),
        gen.trunc_gauss(rng, 0.15, 0.020, 0.10, 0.20),
    )
    air_shape_sphere_r = gen.trunc_gauss(rng, 0.19, 0.020, 0.15, 0.24)
    air_shape_cyl_r = gen.trunc_gauss(rng, 0.16, 0.020, 0.12, 0.21)
    air_shape_cyl_depth = gen.trunc_gauss(rng, 0.46, 0.050, 0.34, 0.60)
    air_chain_r_a = gen.trunc_gauss(rng, 0.18, 0.020, 0.14, 0.23)
    air_chain_r_b = gen.trunc_gauss(rng, 0.18, 0.020, 0.14, 0.23)
    air_chain_r_c = gen.trunc_gauss(rng, 0.17, 0.020, 0.13, 0.22)

    def randomized_mixed_drop_scene() -> dict[str, Any]:
        object_count = rng.choices([3, 4, 5], weights=[0.20, 0.55, 0.25], k=1)[0]
        max_ground = max(0, object_count - 2)
        ground_count = min(max_ground, rng.choices([1, 2], weights=[0.60, 0.40], k=1)[0])
        airborne_count = object_count - ground_count
        ground_states = rng.choices(
            ["ground_moving", "ground_rest_dynamic", "ground_static"],
            weights=[0.45, 0.35, 0.20],
            k=ground_count,
        )
        states = ["airborne"] * airborne_count + ground_states
        rng.shuffle(states)
        def sample_shape_for_state(state: str) -> str:
            if state == "airborne":
                return rng.choices(["sphere", "box", "cylinder", "capsule"], weights=[0.46, 0.42, 0.10, 0.02], k=1)[0]
            return rng.choices(["sphere", "box", "cylinder", "capsule"], weights=[0.30, 0.35, 0.22, 0.13], k=1)[0]

        shapes = [sample_shape_for_state(state) for state in states]
        bodies: list[dict[str, Any]] = []
        airborne_names: list[str] = []

        def footprint_radius(body: dict[str, Any]) -> float:
            half = body.get("half_extents")
            if half:
                return math.sqrt(float(half[0]) ** 2 + float(half[1]) ** 2)
            return float(body.get("radius", 0.22))

        def start_separation_ok(candidate: dict[str, Any]) -> bool:
            for other in bodies:
                dx = float(candidate["position"][0]) - float(other["position"][0])
                dy = float(candidate["position"][1]) - float(other["position"][1])
                xy_gap = math.sqrt(dx * dx + dy * dy) - footprint_radius(candidate) - footprint_radius(other)
                z_gap = abs(float(candidate["position"][2]) - float(other["position"][2])) - float(candidate["bottom_offset"]) - float(other["bottom_offset"])
                if xy_gap < 0.14 and z_gap < 0.10:
                    return False
            return True

        def sample_velocity(state: str, x: float, y: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
            if state == "ground_static":
                return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            if state == "ground_rest_dynamic":
                return (
                    (gen.trunc_gauss(rng, 0.0, 0.06, -0.12, 0.12), gen.trunc_gauss(rng, 0.0, 0.06, -0.12, 0.12), 0.0),
                    (gen.trunc_gauss(rng, 0.0, 0.35, -0.8, 0.8), gen.trunc_gauss(rng, 0.0, 0.35, -0.8, 0.8), gen.trunc_gauss(rng, 0.0, 0.25, -0.6, 0.6)),
                )
            if state == "ground_moving":
                return (
                    (gen.trunc_gauss(rng, 0.0, 0.40, -0.90, 0.90), gen.trunc_gauss(rng, 0.0, 0.36, -0.82, 0.82), 0.0),
                    (gen.trunc_gauss(rng, 0.0, 1.10, -2.6, 2.6), gen.trunc_gauss(rng, 0.0, 1.10, -2.6, 2.6), gen.trunc_gauss(rng, 0.0, 0.65, -1.5, 1.5)),
                )
            center_bias = rng.random() < 0.25
            vx_mean = -0.18 * x if center_bias else 0.0
            vy_mean = -0.15 * y if center_bias else 0.0
            return (
                (
                    gen.trunc_gauss(rng, vx_mean, 0.42, -0.98, 0.98),
                    gen.trunc_gauss(rng, vy_mean, 0.38, -0.90, 0.90),
                    gen.trunc_gauss(rng, -0.14, 0.36, -1.05, 0.35),
                ),
                (
                    gen.trunc_gauss(rng, 0.0, 1.25, -3.0, 3.0),
                    gen.trunc_gauss(rng, 0.0, 1.25, -3.0, 3.0),
                    gen.trunc_gauss(rng, 0.0, 0.80, -2.0, 2.0),
                ),
            )

        def sample_body(index: int, state: str, shape: str, x: float, y: float) -> dict[str, Any]:
            name = f"random_drop_{index:02d}_{shape}"
            is_static = state == "ground_static"
            restitution = gen.trunc_gauss(rng, 0.56, 0.18, 0.18, 0.92)
            friction = gen.trunc_gauss(rng, 0.28, 0.18, 0.03, 0.78)
            body_mass = 0.0 if is_static else mass(gen.trunc_gauss(rng, 0.95, 0.20, 0.55, 1.45))
            if shape == "sphere":
                radius = gen.trunc_gauss(rng, 0.20, 0.045, 0.12, 0.32)
                bottom = radius
            elif shape == "box":
                half = (
                    gen.trunc_gauss(rng, 0.18, 0.055, 0.10, 0.32),
                    gen.trunc_gauss(rng, 0.18, 0.055, 0.10, 0.32),
                    gen.trunc_gauss(rng, 0.16, 0.045, 0.08, 0.28),
                )
                bottom = half[2]
            else:
                radius = gen.trunc_gauss(rng, 0.14, 0.030, 0.09, 0.22)
                depth = gen.trunc_gauss(rng, 0.42, 0.090, 0.24, 0.66)
                long_axis = rng.choice(["x", "y"])
                bottom = radius
            height_above_ground = 0.0
            if state == "airborne":
                height_above_ground = gen.trunc_gauss(rng, 1.15, 0.65, 0.40, 2.45)
            z = bottom + height_above_ground
            velocity, angular_velocity = sample_velocity(state, x, y)
            if shape == "sphere":
                body = gen.sphere(
                    name,
                    radius,
                    (x, y, z),
                    appearance(name),
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    mass=body_mass,
                    restitution=restitution,
                    friction=friction,
                    static=is_static,
                )
            elif shape == "box":
                body = gen.cube(
                    name,
                    half,
                    (x, y, z),
                    appearance(name),
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    mass=body_mass,
                    restitution=restitution,
                    friction=friction,
                    static=is_static,
                )
            elif shape == "cylinder":
                body = gen.cylinder(
                    name,
                    radius,
                    depth,
                    (x, y, z),
                    appearance(name),
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    mass=body_mass,
                    restitution=restitution,
                    friction=friction,
                    static=is_static,
                    long_axis=long_axis,
                )
            else:
                body = gen.capsule(
                    name,
                    radius,
                    depth,
                    (x, y, z),
                    appearance(name),
                    velocity=velocity,
                    angular_velocity=angular_velocity,
                    mass=body_mass,
                    restitution=restitution,
                    friction=friction,
                    static=is_static,
                    long_axis=long_axis,
                )
            body["initial_state"] = state
            body["sampled_height_above_ground_m"] = round(height_above_ground, 5)
            body["initialization_model"] = "independent_per_object_clipped_gaussian"
            return body

        for index, (state, shape) in enumerate(zip(states, shapes)):
            for attempt in range(180):
                if attempt < 120:
                    x = gen.trunc_gauss(rng, 0.0, 0.82, -1.75, 1.75)
                    y = gen.trunc_gauss(rng, 0.04, 0.76, -1.35, 1.75)
                else:
                    x = round(rng.uniform(-2.0, 2.0), 5)
                    y = round(rng.uniform(-1.65, 2.0), 5)
                candidate = sample_body(index, state, shape, x, y)
                if start_separation_ok(candidate):
                    bodies.append(candidate)
                    if state == "airborne":
                        airborne_names.append(candidate["name"])
                    break
            else:
                bodies.append(candidate)
                if state == "airborne":
                    airborne_names.append(candidate["name"])

        state_counts = {state: states.count(state) for state in sorted(set(states))}
        return {
            "id": "phys_randomized_mixed_drop_scene",
            "kind": "randomized_mixed_drop_scene",
            "speed_class": "mixed",
            "description": "Independently initialized multi-object scene mixing staggered falling objects with optional ground moving, dynamic resting, or static objects; contacts may or may not occur.",
            "sample_model": "object count, per-object state, shape, size, position, height, velocity, angular velocity, friction, restitution, color, and material are sampled independently from clipped Gaussian distributions, with only loose in-frame and initial non-overlap constraints.",
            "expected_contacts": [],
            "allow_unlabeled_contacts": True,
            "drop_timing_audit": {
                "airborne_names": airborne_names,
                "min_spread_frames": 14,
                "ground_margin_m": 0.055,
                "purpose": "reject samples where independently falling objects still land almost simultaneously",
            },
            "initial_state_counts": state_counts,
            "bodies": bodies,
        }

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
            "id": "phys_multi_drop_bounce",
            "kind": "multi_object_gravity_bounce",
            "speed_class": "mixed",
            "description": "Two spheres and one box fall with staggered heights and bounce or tumble on the floor.",
            "sample_model": "multiple object sizes, masses, drop heights, lateral velocities, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["drop_sphere_a", "ground"], ["drop_sphere_b", "ground"], ["drop_box", "ground"]],
            "bodies": [
                gen.sphere(
                    "drop_sphere_a",
                    multi_drop_r_a,
                    (-0.78, -0.42, gen.trunc_gauss(rng, 1.06, 0.10, 0.88, 1.28)),
                    appearance("drop_sphere_a", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 0.18, 0.07, 0.02, 0.36), 0.04, gen.trunc_gauss(rng, -0.18, 0.06, -0.34, -0.04)),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, 1.3, 0.45, 0.2, 2.4), 0.2),
                    mass=mass(0.75),
                    restitution=gen.trunc_gauss(rng, 0.64, 0.07, 0.44, 0.80),
                    friction=low_friction(),
                ),
                gen.sphere(
                    "drop_sphere_b",
                    multi_drop_r_b,
                    (0.78, -0.34, gen.trunc_gauss(rng, 0.94, 0.09, 0.78, 1.16)),
                    appearance("drop_sphere_b", profile="glossy"),
                    velocity=(gen.trunc_gauss(rng, -0.14, 0.06, -0.30, -0.02), 0.03, gen.trunc_gauss(rng, -0.14, 0.05, -0.28, 0.0)),
                    angular_velocity=(0.1, gen.trunc_gauss(rng, -1.0, 0.45, -2.2, -0.1), -0.2),
                    mass=mass(0.65),
                    restitution=gen.trunc_gauss(rng, 0.60, 0.07, 0.42, 0.76),
                    friction=low_friction(),
                ),
                gen.cube(
                    "drop_box",
                    multi_drop_box,
                    (0.00, 0.78, gen.trunc_gauss(rng, 0.98, 0.10, 0.80, 1.22)),
                    appearance("drop_box", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, 0.06, 0.06, -0.08, 0.20), gen.trunc_gauss(rng, -0.12, 0.06, -0.28, 0.0), -0.14),
                    angular_velocity=(gen.trunc_gauss(rng, 0.35, 0.25, -0.15, 0.95), gen.trunc_gauss(rng, -0.45, 0.25, -1.05, 0.10), 0.18),
                    mass=mass(1.05),
                    restitution=gen.trunc_gauss(rng, 0.42, 0.08, 0.24, 0.62),
                    friction=medium_friction(),
                ),
            ],
        },
        {
            "id": "phys_airborne_drop_collision",
            "kind": "airborne_drop_collision",
            "speed_class": "mixed",
            "description": "Two falling spheres collide before either reaches the ground, with a third falling box for multi-object gravity context.",
            "sample_model": "airborne sphere sizes, converging lateral velocities, drop heights, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["airdrop_left_sphere", "airdrop_right_sphere"], ["airdrop_left_sphere", "ground"], ["airdrop_right_sphere", "ground"], ["airdrop_follow_box", "ground"]],
            "expected_airborne_contacts": [["airdrop_left_sphere", "airdrop_right_sphere"]],
            "bodies": [
                gen.sphere(
                    "airdrop_left_sphere",
                    air_drop_r_a,
                    (-0.38, -0.10, gen.trunc_gauss(rng, 1.52, 0.06, 1.40, 1.68)),
                    appearance("airdrop_left_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 0.68, 0.06, 0.54, 0.84), gen.trunc_gauss(rng, 0.02, 0.02, -0.03, 0.07), gen.trunc_gauss(rng, -0.20, 0.06, -0.34, -0.08)),
                    angular_velocity=(gen.trunc_gauss(rng, 0.4, 0.30, -0.2, 1.0), gen.trunc_gauss(rng, 1.4, 0.45, 0.3, 2.5), 0.15),
                    mass=mass(0.72),
                    restitution=gen.trunc_gauss(rng, 0.72, 0.06, 0.56, 0.86),
                    friction=low_friction(),
                ),
                gen.sphere(
                    "airdrop_right_sphere",
                    air_drop_r_b,
                    (0.38, -0.08, gen.trunc_gauss(rng, 1.50, 0.06, 1.38, 1.66)),
                    appearance("airdrop_right_sphere", profile="glossy"),
                    velocity=(gen.trunc_gauss(rng, -0.66, 0.06, -0.82, -0.52), gen.trunc_gauss(rng, -0.02, 0.02, -0.07, 0.03), gen.trunc_gauss(rng, -0.18, 0.06, -0.32, -0.06)),
                    angular_velocity=(gen.trunc_gauss(rng, -0.3, 0.30, -0.9, 0.3), gen.trunc_gauss(rng, -1.3, 0.45, -2.4, -0.2), -0.15),
                    mass=mass(0.72),
                    restitution=gen.trunc_gauss(rng, 0.72, 0.06, 0.56, 0.86),
                    friction=low_friction(),
                ),
                gen.cube(
                    "airdrop_follow_box",
                    air_drop_box,
                    (0.05, 0.58, gen.trunc_gauss(rng, 1.18, 0.08, 1.02, 1.36)),
                    appearance("airdrop_follow_box", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, -0.04, 0.06, -0.18, 0.10), gen.trunc_gauss(rng, -0.18, 0.06, -0.34, -0.04), gen.trunc_gauss(rng, -0.28, 0.07, -0.48, -0.12)),
                    angular_velocity=(gen.trunc_gauss(rng, 0.55, 0.30, -0.10, 1.25), gen.trunc_gauss(rng, -0.50, 0.30, -1.20, 0.10), gen.trunc_gauss(rng, 0.25, 0.20, -0.20, 0.70)),
                    mass=mass(0.95),
                    restitution=gen.trunc_gauss(rng, 0.46, 0.08, 0.28, 0.66),
                    friction=medium_friction(),
                ),
            ],
        },
        {
            "id": "phys_airborne_sphere_box_collision",
            "kind": "airborne_sphere_box_collision",
            "speed_class": "mixed",
            "description": "A falling sphere contacts a falling box before either reaches the ground.",
            "sample_model": "sphere radius, box extents, converging velocities, drop heights, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["airsphere_box_sphere", "airsphere_box_target"], ["airsphere_box_sphere", "ground"], ["airsphere_box_target", "ground"]],
            "expected_airborne_contacts": [["airsphere_box_sphere", "airsphere_box_target"]],
            "bodies": [
                gen.sphere(
                    "airsphere_box_sphere",
                    air_sphere_box_r,
                    (-0.40, -0.10, gen.trunc_gauss(rng, 1.54, 0.06, 1.42, 1.70)),
                    appearance("airsphere_box_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 0.74, 0.06, 0.58, 0.90), gen.trunc_gauss(rng, 0.01, 0.02, -0.04, 0.06), gen.trunc_gauss(rng, -0.18, 0.06, -0.34, -0.04)),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, 1.5, 0.45, 0.4, 2.7), 0.1),
                    mass=mass(0.72),
                    restitution=gen.trunc_gauss(rng, 0.68, 0.07, 0.50, 0.84),
                    friction=low_friction(),
                ),
                gen.cube(
                    "airsphere_box_target",
                    air_sphere_box_half,
                    (0.24, -0.08, gen.trunc_gauss(rng, 1.52, 0.06, 1.40, 1.68)),
                    appearance("airsphere_box_target", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, -0.24, 0.05, -0.38, -0.10), gen.trunc_gauss(rng, -0.01, 0.02, -0.06, 0.04), gen.trunc_gauss(rng, -0.16, 0.06, -0.32, -0.02)),
                    angular_velocity=(gen.trunc_gauss(rng, 0.45, 0.30, -0.10, 1.10), gen.trunc_gauss(rng, -0.65, 0.35, -1.40, 0.10), gen.trunc_gauss(rng, 0.25, 0.20, -0.20, 0.70)),
                    mass=mass(0.95),
                    restitution=gen.trunc_gauss(rng, 0.50, 0.08, 0.30, 0.70),
                    friction=medium_friction(),
                ),
            ],
        },
        {
            "id": "phys_airborne_box_box_collision",
            "kind": "airborne_box_box_collision",
            "speed_class": "mixed",
            "description": "Two rectangular boxes collide while falling, then tumble toward the ground.",
            "sample_model": "box extents, converging velocities, drop heights, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["airbox_left", "airbox_right"], ["airbox_left", "ground"], ["airbox_right", "ground"]],
            "expected_airborne_contacts": [["airbox_left", "airbox_right"]],
            "bodies": [
                gen.cube(
                    "airbox_left",
                    air_box_a,
                    (-0.36, 0.10, gen.trunc_gauss(rng, 1.05, 0.05, 0.94, 1.18)),
                    appearance("airbox_left", profile="matte"),
                    velocity=(gen.trunc_gauss(rng, 0.56, 0.07, 0.40, 0.74), gen.trunc_gauss(rng, -0.02, 0.03, -0.09, 0.05), gen.trunc_gauss(rng, -0.02, 0.025, -0.08, 0.03)),
                    angular_velocity=(gen.trunc_gauss(rng, 0.40, 0.25, 0.0, 1.15), gen.trunc_gauss(rng, 0.4, 0.30, -0.2, 1.0), gen.trunc_gauss(rng, -0.3, 0.25, -0.9, 0.2)),
                    mass=mass(0.90),
                    restitution=gen.trunc_gauss(rng, 0.24, 0.05, 0.12, 0.40),
                    friction=high_friction(),
                ),
                gen.cube(
                    "airbox_right",
                    air_box_b,
                    (0.36, 0.04, gen.trunc_gauss(rng, 1.03, 0.05, 0.92, 1.16)),
                    appearance("airbox_right", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, -0.56, 0.07, -0.74, -0.40), gen.trunc_gauss(rng, 0.02, 0.03, -0.05, 0.09), gen.trunc_gauss(rng, -0.02, 0.025, -0.08, 0.03)),
                    angular_velocity=(gen.trunc_gauss(rng, -0.35, 0.25, -1.05, 0.1), gen.trunc_gauss(rng, -0.5, 0.30, -1.1, 0.1), gen.trunc_gauss(rng, 0.35, 0.25, -0.2, 0.9)),
                    mass=mass(0.95),
                    restitution=gen.trunc_gauss(rng, 0.24, 0.05, 0.12, 0.40),
                    friction=high_friction(),
                ),
            ],
        },
        {
            "id": "phys_airborne_sphere_cylinder_collision",
            "kind": "airborne_sphere_cylinder_collision",
            "speed_class": "mixed",
            "description": "A falling sphere contacts a falling horizontal cylinder before ground contact, producing visible spin.",
            "sample_model": "sphere radius, cylinder radius/depth, converging velocities, drop heights, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["airshape_sphere", "airshape_cylinder"], ["airshape_sphere", "ground"], ["airshape_cylinder", "ground"]],
            "expected_airborne_contacts": [["airshape_sphere", "airshape_cylinder"]],
            "bodies": [
                gen.sphere(
                    "airshape_sphere",
                    air_shape_sphere_r,
                    (-0.44, 0.06, gen.trunc_gauss(rng, 1.72, 0.07, 1.56, 1.88)),
                    appearance("airshape_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 0.66, 0.07, 0.50, 0.84), gen.trunc_gauss(rng, -0.01, 0.03, -0.08, 0.06), gen.trunc_gauss(rng, -0.18, 0.06, -0.34, -0.04)),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, 1.5, 0.45, 0.4, 2.7), 0.1),
                    mass=mass(0.72),
                    restitution=gen.trunc_gauss(rng, 0.66, 0.07, 0.48, 0.82),
                    friction=low_friction(),
                ),
                gen.cylinder(
                    "airshape_cylinder",
                    air_shape_cyl_r,
                    air_shape_cyl_depth,
                    (0.28, 0.02, gen.trunc_gauss(rng, 1.70, 0.07, 1.54, 1.86)),
                    appearance("airshape_cylinder", profile="glossy"),
                    velocity=(gen.trunc_gauss(rng, -0.28, 0.06, -0.44, -0.12), gen.trunc_gauss(rng, 0.01, 0.03, -0.06, 0.08), gen.trunc_gauss(rng, -0.16, 0.06, -0.32, -0.02)),
                    angular_velocity=(gen.trunc_gauss(rng, 0.8, 0.35, 0.0, 1.6), gen.trunc_gauss(rng, -1.1, 0.40, -2.0, -0.2), gen.trunc_gauss(rng, 0.35, 0.25, -0.2, 0.9)),
                    mass=mass(0.95),
                    restitution=gen.trunc_gauss(rng, 0.56, 0.08, 0.36, 0.74),
                    friction=medium_friction(),
                    long_axis="y",
                ),
            ],
        },
        {
            "id": "phys_airborne_chain_collision",
            "kind": "airborne_chain_collision",
            "speed_class": "mixed",
            "description": "Three falling spheres form a short airborne chain: one pair contacts first, then another pair contacts before ground contact.",
            "sample_model": "sphere radii, staged lateral velocities, drop heights, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["airchain_left_sphere", "airchain_mid_sphere"], ["airchain_mid_sphere", "airchain_right_sphere"], ["airchain_left_sphere", "ground"], ["airchain_mid_sphere", "ground"], ["airchain_right_sphere", "ground"]],
            "expected_airborne_contacts": [["airchain_left_sphere", "airchain_mid_sphere"], ["airchain_mid_sphere", "airchain_right_sphere"]],
            "bodies": [
                gen.sphere(
                    "airchain_left_sphere",
                    air_chain_r_a,
                    (-0.48, 0.00, gen.trunc_gauss(rng, 1.82, 0.07, 1.66, 1.98)),
                    appearance("airchain_left_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 0.86, 0.07, 0.70, 1.02), 0.00, gen.trunc_gauss(rng, -0.10, 0.05, -0.24, 0.0)),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, 1.5, 0.45, 0.4, 2.7), 0.1),
                    mass=mass(0.75),
                    restitution=gen.trunc_gauss(rng, 0.74, 0.06, 0.58, 0.88),
                    friction=low_friction(),
                ),
                gen.sphere(
                    "airchain_mid_sphere",
                    air_chain_r_b,
                    (0.00, 0.02, gen.trunc_gauss(rng, 1.80, 0.07, 1.64, 1.96)),
                    appearance("airchain_mid_sphere", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, 0.08, 0.035, 0.0, 0.16), 0.00, gen.trunc_gauss(rng, -0.10, 0.05, -0.24, 0.0)),
                    angular_velocity=(0.1, gen.trunc_gauss(rng, -0.3, 0.30, -0.9, 0.3), 0.0),
                    mass=mass(0.70),
                    restitution=gen.trunc_gauss(rng, 0.72, 0.06, 0.56, 0.86),
                    friction=low_friction(),
                ),
                gen.sphere(
                    "airchain_right_sphere",
                    air_chain_r_c,
                    (0.50, -0.02, gen.trunc_gauss(rng, 1.78, 0.07, 1.62, 1.94)),
                    appearance("airchain_right_sphere", profile="glossy"),
                    velocity=(gen.trunc_gauss(rng, -0.34, 0.05, -0.48, -0.20), 0.00, gen.trunc_gauss(rng, -0.10, 0.05, -0.24, 0.0)),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, -1.1, 0.40, -2.0, -0.2), -0.1),
                    mass=mass(0.68),
                    restitution=gen.trunc_gauss(rng, 0.70, 0.06, 0.54, 0.84),
                    friction=low_friction(),
                ),
            ],
        },
        {
            "id": "phys_drop_hits_dynamic_box",
            "kind": "drop_then_object_collision",
            "speed_class": "mixed",
            "description": "A falling sphere strikes a visible dynamic box, combining gravity bounce with object-object contact.",
            "sample_model": "falling sphere and target box sizes, masses, velocities, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["falling_hit_sphere", "drop_target_box"], ["drop_target_box", "ground"]],
            "bodies": [
                gen.sphere(
                    "falling_hit_sphere",
                    drop_hit_sphere_r,
                    (-0.12, -0.04, gen.trunc_gauss(rng, 1.22, 0.13, 0.98, 1.52)),
                    appearance("falling_hit_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 0.16, 0.08, -0.02, 0.34), 0.04, gen.trunc_gauss(rng, -0.44, 0.10, -0.70, -0.20)),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, 1.0, 0.35, 0.2, 1.8), 0.0),
                    mass=mass(0.75),
                    restitution=gen.trunc_gauss(rng, 0.74, 0.08, 0.52, 0.92),
                    friction=low_friction(),
                ),
                gen.cube(
                    "drop_target_box",
                    drop_hit_box,
                    (0.02, 0.00, drop_hit_box[2]),
                    appearance("drop_target_box", profile="matte"),
                    velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, 0.0, 0.0),
                    mass=mass(1.25),
                    restitution=gen.trunc_gauss(rng, 0.48, 0.09, 0.28, 0.70),
                    friction=medium_friction(),
                ),
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
            "id": "phys_shape_mixed_primitives",
            "kind": "shape_diversity_collision",
            "speed_class": "medium",
            "description": "Sphere hits a visible cylinder while a cone and a horizontal floor capsule provide audited shape diversity.",
            "sample_model": "procedural cylinder/cone/capsule dimensions, masses, velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["shape_driver_sphere", "shape_center_cylinder"]],
            "bodies": [
                gen.sphere(
                    "shape_driver_sphere",
                    cap_r,
                    (-0.92, -0.18, cap_r),
                    appearance("shape_driver_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 1.48, 0.18, 1.10, 1.92), 0.18, 0.0),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, 1.2, 0.4, 0.4, 2.0), 0.2),
                    mass=mass(0.85),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                gen.cylinder(
                    "shape_center_cylinder",
                    cyl_r,
                    cyl_depth,
                    (0.02, 0.00, cyl_depth / 2.0),
                    appearance("shape_center_cylinder", profile="satin"),
                    velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, 0.0, 0.0),
                    mass=0.0,
                    restitution=gen.trunc_gauss(rng, 0.54, 0.10, 0.34, 0.78),
                    friction=medium_friction(),
                    static=True,
                ),
                gen.cone(
                    "shape_side_cone",
                    cone_r,
                    cone_depth,
                    (0.88, 0.48, cone_depth / 2.0),
                    appearance("shape_side_cone", profile="matte"),
                    velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, 0.0, 0.0),
                    mass=0.0,
                    restitution=gen.trunc_gauss(rng, 0.42, 0.08, 0.22, 0.64),
                    friction=high_friction(),
                    static=True,
                ),
                gen.capsule(
                    "shape_back_capsule",
                    cap_r,
                    cap_mid,
                    (-0.25, 0.78, cap_r),
                    appearance("shape_back_capsule", profile="glossy"),
                    velocity=(0.0, 0.0, 0.0),
                    angular_velocity=(0.0, 0.0, 0.0),
                    mass=0.0,
                    restitution=gen.trunc_gauss(rng, 0.46, 0.10, 0.24, 0.70),
                    friction=medium_friction(),
                    static=True,
                    long_axis="x",
                ),
            ],
        },
        {
            "id": "phys_dynamic_cylinder_roll",
            "kind": "dynamic_cylinder_ground_motion",
            "speed_class": "medium",
            "description": "A horizontal procedural cylinder rolls and slides across the floor under PyBullet dynamics.",
            "sample_model": "cylinder radius/depth, mass, velocity, angular velocity, friction, restitution, color, and material profile are clipped Gaussian samples.",
            "expected_contacts": [["rolling_cylinder", "ground"]],
            "bodies": [
                gen.cylinder(
                    "rolling_cylinder",
                    dyn_cyl_r,
                    dyn_cyl_depth,
                    (-0.95, -0.26, dyn_cyl_r),
                    appearance("rolling_cylinder", profile="satin"),
                    velocity=(gen.trunc_gauss(rng, 1.05, 0.16, 0.70, 1.45), 0.18, 0.0),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, -4.2, 0.8, -6.2, -2.5), 0.12),
                    mass=mass(1.05),
                    restitution=gen.trunc_gauss(rng, 0.42, 0.08, 0.24, 0.62),
                    friction=medium_friction(),
                    long_axis="y",
                ),
            ],
        },
        {
            "id": "phys_dynamic_capsule_roll",
            "kind": "dynamic_capsule_ground_motion",
            "speed_class": "medium",
            "description": "A horizontal procedural capsule rolls and slides across the floor instead of standing upright.",
            "sample_model": "capsule radius/depth, mass, velocity, angular velocity, friction, restitution, color, and material profile are clipped Gaussian samples.",
            "expected_contacts": [["rolling_capsule", "ground"]],
            "bodies": [
                gen.capsule(
                    "rolling_capsule",
                    dyn_cap_r,
                    dyn_cap_mid,
                    (-0.75, 0.36, dyn_cap_r),
                    appearance("rolling_capsule", profile="glossy"),
                    velocity=(0.12, gen.trunc_gauss(rng, -0.92, 0.14, -1.25, -0.58), 0.0),
                    angular_velocity=(gen.trunc_gauss(rng, 3.8, 0.7, 2.2, 5.6), 0.0, 0.20),
                    mass=mass(0.95),
                    restitution=gen.trunc_gauss(rng, 0.45, 0.08, 0.25, 0.66),
                    friction=medium_friction(),
                    long_axis="x",
                ),
            ],
        },
        {
            "id": "phys_sphere_hits_dynamic_cylinder",
            "kind": "sphere_dynamic_cylinder_collision",
            "speed_class": "medium",
            "description": "A moving sphere contacts a horizontal dynamic cylinder, testing non-box dynamic collision behavior.",
            "sample_model": "sphere and cylinder sizes, masses, velocities, angular velocities, friction, restitution, colors, and material profiles are clipped Gaussian samples.",
            "expected_contacts": [["shape_hit_sphere", "dynamic_hit_cylinder"]],
            "bodies": [
                gen.sphere(
                    "shape_hit_sphere",
                    shape_hit_sphere_r,
                    (-0.95, -0.08, shape_hit_sphere_r),
                    appearance("shape_hit_sphere", profile="rubber"),
                    velocity=(gen.trunc_gauss(rng, 1.36, 0.16, 1.02, 1.75), 0.08, 0.0),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, 1.4, 0.35, 0.7, 2.2), 0.0),
                    mass=mass(0.80),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                gen.cylinder(
                    "dynamic_hit_cylinder",
                    dyn_cyl_r,
                    dyn_cyl_depth,
                    (0.08, 0.02, dyn_cyl_r),
                    appearance("dynamic_hit_cylinder", profile="matte"),
                    velocity=(gen.trunc_gauss(rng, -0.12, 0.06, -0.24, 0.0), 0.0, 0.0),
                    angular_velocity=(0.0, gen.trunc_gauss(rng, -0.5, 0.25, -1.1, 0.0), 0.0),
                    mass=mass(1.15),
                    restitution=gen.trunc_gauss(rng, 0.56, 0.08, 0.36, 0.76),
                    friction=medium_friction(),
                    long_axis="y",
                ),
            ],
        },
        {
            "id": "phys_dynamic_cone_slide",
            "kind": "dynamic_cone_ground_motion",
            "speed_class": "slow",
            "description": "A horizontal procedural cone slides and gently tumbles on the floor as a stress test for cone dynamics.",
            "sample_model": "cone radius/depth, mass, velocity, angular velocity, friction, restitution, color, and material profile are clipped Gaussian samples.",
            "expected_contacts": [["sliding_cone", "ground"]],
            "bodies": [
                gen.cone(
                    "sliding_cone",
                    dyn_cone_r,
                    dyn_cone_depth,
                    (-0.55, -0.42, dyn_cone_r),
                    appearance("sliding_cone", profile="matte"),
                    velocity=(gen.trunc_gauss(rng, 0.56, 0.12, 0.28, 0.86), 0.22, 0.0),
                    angular_velocity=(gen.trunc_gauss(rng, 1.2, 0.35, 0.4, 2.0), 0.2, 0.35),
                    mass=mass(0.90),
                    restitution=gen.trunc_gauss(rng, 0.34, 0.08, 0.16, 0.52),
                    friction=high_friction(),
                    long_axis="x",
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
        {
            "id": "phys_three_body_chain",
            "kind": "three_body_chain_collision",
            "speed_class": "mixed",
            "description": "Three objects form a chained collision with randomized material profiles.",
            "sample_model": "three-object chain layout, sizes, masses, velocities, friction, restitution, colors, and material profiles are sampled from clipped Gaussian templates.",
            "expected_contacts": [["chain_driver_sphere", "chain_center_box"]],
            "bodies": [
                gen.sphere(
                    "chain_driver_sphere",
                    scatter_r_a,
                    (-1.25, -0.28, scatter_r_a),
                    appearance("chain_driver_sphere"),
                    velocity=(gen.trunc_gauss(rng, 1.65, 0.20, 1.20, 2.15), 0.20, 0.0),
                    mass=mass(0.95),
                    restitution=bounce(),
                    friction=gen.trunc_gauss(rng, 0.09, 0.04, 0.02, 0.20),
                ),
                gen.cube(
                    "chain_center_box",
                    scatter_cube,
                    (-0.08, -0.12, scatter_cube[2]),
                    appearance("chain_center_box"),
                    velocity=(0.06, 0.00, 0.0),
                    angular_velocity=(0.0, 1.0, 0.35),
                    mass=mass(1.20),
                    restitution=gen.trunc_gauss(rng, 0.50, 0.11, 0.28, 0.74),
                    friction=medium_friction(),
                ),
                gen.sphere(
                    "chain_target_sphere",
                    scatter_r_b,
                    (0.92, 0.16, scatter_r_b),
                    appearance("chain_target_sphere"),
                    velocity=(gen.trunc_gauss(rng, -0.28, 0.10, -0.55, -0.06), -0.08, 0.0),
                    mass=mass(0.75),
                    restitution=bounce(),
                    friction=gen.trunc_gauss(rng, 0.14, 0.05, 0.04, 0.28),
                ),
            ],
        },
        {
            "id": "phys_four_body_crossfire",
            "kind": "four_body_crossfire_collision",
            "speed_class": "mixed",
            "description": "Four objects enter from different directions around a central box.",
            "sample_model": "four-object crossfire layout with randomized material profiles, colors, masses, velocities, friction, and restitution.",
            "expected_contacts": [["crossfire_left_sphere", "crossfire_center_box"]],
            "bodies": [
                gen.sphere(
                    "crossfire_left_sphere",
                    scatter_r_a,
                    (-1.25, 0.10, scatter_r_a),
                    appearance("crossfire_left_sphere"),
                    velocity=(gen.trunc_gauss(rng, 1.72, 0.22, 1.25, 2.25), -0.10, 0.0),
                    mass=mass(0.85),
                    restitution=bounce(),
                    friction=low_friction(),
                ),
                gen.cube(
                    "crossfire_center_box",
                    scatter_cube,
                    (0.02, 0.02, scatter_cube[2]),
                    appearance("crossfire_center_box"),
                    velocity=(0.02, 0.02, 0.0),
                    angular_velocity=(0.0, -1.1, 0.45),
                    mass=mass(1.15),
                    restitution=gen.trunc_gauss(rng, 0.54, 0.10, 0.32, 0.78),
                    friction=medium_friction(),
                ),
                gen.sphere(
                    "crossfire_right_sphere",
                    scatter_r_b,
                    (1.20, -0.35, scatter_r_b),
                    appearance("crossfire_right_sphere"),
                    velocity=(gen.trunc_gauss(rng, -1.18, 0.18, -1.60, -0.78), 0.32, 0.0),
                    mass=mass(0.75),
                    restitution=bounce(),
                    friction=gen.trunc_gauss(rng, 0.11, 0.04, 0.03, 0.24),
                ),
                gen.cube(
                    "crossfire_small_box",
                    scatter_small_box,
                    (-0.35, 0.80, scatter_small_box[2]),
                    appearance("crossfire_small_box"),
                    velocity=(0.35, gen.trunc_gauss(rng, -0.75, 0.14, -1.05, -0.42), 0.0),
                    angular_velocity=(0.0, 0.7, -0.9),
                    mass=mass(0.75),
                    restitution=gen.trunc_gauss(rng, 0.58, 0.11, 0.34, 0.82),
                    friction=gen.trunc_gauss(rng, 0.16, 0.06, 0.04, 0.32),
                ),
            ],
        },
        randomized_mixed_drop_scene(),
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
        "object shape diversity, camera speed bands, material sampling, and physical event variety before batch_v2."
    )
    manifest["diversity_axes"] = [
        "camera path family",
        "camera speed class",
        "camera speed sampling band",
        "camera start viewpoint",
        "object count",
        "object size/aspect ratio",
        "object primitive shape: sphere, box, cylinder, cone, capsule",
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
    parser.add_argument("--camera-limit", type=int, default=9)
    parser.add_argument("--physics-limit", type=int, default=9)
    parser.add_argument("--camera-ids", default="", help="Comma-separated camera ids to include after camera_specs sampling.")
    parser.add_argument("--physics-ids", default="", help="Comma-separated physics ids to include after physics_specs sampling.")
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
    cameras_all = camera_specs(args.seed)
    programs_all = physics_specs(args.seed)
    if args.camera_ids:
        wanted = [item.strip() for item in args.camera_ids.split(",") if item.strip()]
        by_id = {camera["id"]: camera for camera in cameras_all}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            raise SystemExit(f"unknown camera ids: {missing}")
        cameras = [by_id[item] for item in wanted]
    else:
        cameras = cameras_all[: args.camera_limit]
    if args.physics_ids:
        wanted = [item.strip() for item in args.physics_ids.split(",") if item.strip()]
        by_id = {program["id"]: program for program in programs_all}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            raise SystemExit(f"unknown physics ids: {missing}")
        programs = [by_id[item] for item in wanted]
    else:
        programs = programs_all[: args.physics_limit]
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
