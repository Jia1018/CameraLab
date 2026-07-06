#!/usr/bin/env python3
"""Generate pair-centric official-Kubric batch/review runs.

The older review scripts render a full camera x physics grid.  This script
samples pair groups first, so the review set can cover many camera and object
families without showing every possible combination.  It still preserves the
pair contracts:

* same-camera pairs reuse one exact camera trajectory and vary physics.
* same-physics pairs reuse one exact physics simulation and scene, then vary
  the camera trajectory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import generate_kubric_diversity_review_bank as diversity
import generate_official_kubric_review_bank as gen
from update_kubric_progress import write_progress


RUN_ROOT = gen.PROJECT_ROOT / "site" / "assets" / "runs"
DEFAULT_RUN_ID = "kubric_batch_v2_review_0000"
DEFAULT_SEED = 20260724

CAMERA_FAMILIES = [
    "static_view",
    "dolly_in",
    "dolly_out",
    "truck_pan",
    "crane_tilt",
    "top_down_drift",
    "orbit_arc",
    "low_truck_roll",
    "diagonal_combo",
]

PHYSICS_FAMILIES = [
    "phys_static_mixed_pair",
    "phys_drop_bounce",
    "phys_two_sphere_collision",
    "phys_sphere_hits_tall_block",
    "phys_box_sphere_collision",
    "phys_four_body_scatter",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def frame_count(rng: random.Random, mean: float, std: float, low: int, high: int, multiple: int) -> int:
    value = int(round(clamp(rng.gauss(mean, std), low, high)))
    if multiple > 1:
        value = int(round(value / multiple) * multiple)
    return max(low, min(high, value))


def sample_curve(rng: random.Random, *, static: bool = False) -> dict[str, Any]:
    if static:
        return {"type": "linear"}
    kind = rng.choices(
        ["linear", "ease_in_out", "ease_in", "ease_out", "two_stage"],
        weights=[0.20, 0.34, 0.14, 0.14, 0.18],
        k=1,
    )[0]
    if kind in {"ease_in", "ease_out"}:
        return {"type": kind, "power": gen.trunc_gauss(rng, 1.65, 0.30, 1.15, 2.35)}
    if kind == "two_stage":
        return {
            "type": kind,
            "split": gen.trunc_gauss(rng, 0.52, 0.09, 0.34, 0.70),
            "first_fraction": gen.trunc_gauss(rng, 0.43, 0.18, 0.18, 0.82),
        }
    return {"type": kind}


def curve_progress(curve: dict[str, Any], u: float) -> float:
    u = clamp(u, 0.0, 1.0)
    kind = curve.get("type", "linear")
    if kind == "ease_in_out":
        return u * u * (3.0 - 2.0 * u)
    if kind == "ease_in":
        return u ** float(curve.get("power", 1.7))
    if kind == "ease_out":
        power = float(curve.get("power", 1.7))
        return 1.0 - (1.0 - u) ** power
    if kind == "two_stage":
        split = clamp(float(curve.get("split", 0.5)), 0.05, 0.95)
        first = clamp(float(curve.get("first_fraction", 0.5)), 0.02, 0.98)
        if u <= split:
            return first * (u / split)
        return first + (1.0 - first) * ((u - split) / (1.0 - split))
    return u


def add_vec(a: list[float], b: list[float], scale: float = 1.0) -> list[float]:
    return gen.vec([float(a[i]) + float(b[i]) * scale for i in range(3)])


def camera_records(camera: dict[str, Any], frames: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    denom = max(1, frames - 1)
    for frame in range(frames):
        u = frame / denom
        progress = curve_progress(camera["speed_curve"], u)
        target = add_vec(camera["start_target"], camera["target_delta"], progress)
        if camera.get("path_model") == "orbit":
            center = camera["orbit_center"]
            angle = math.radians(camera["orbit_start_deg"] + camera["orbit_delta_deg"] * progress)
            radius = float(camera["orbit_radius_m"]) + float(camera.get("orbit_radius_delta_m", 0.0)) * progress
            height = float(camera["orbit_height_m"]) + float(camera.get("orbit_height_delta_m", 0.0)) * progress
            position = gen.vec([center[0] + radius * math.sin(angle), center[1] - radius * math.cos(angle), height])
        else:
            position = add_vec(camera["start_position"], camera["position_delta"], progress)
        roll = float(camera["roll_start_deg"]) + float(camera["roll_delta_deg"]) * progress
        lens = gen.clamp(float(camera["lens_mm"]) + float(camera["lens_delta_mm"]) * progress, 28.0, 50.0)
        records.append(
            {
                "frame": frame,
                "time_s": round(frame / gen.FPS, 5),
                "progress": round(progress, 5),
                "position": gen.vec(position),
                "look_at": gen.vec(target),
                "quaternion_wxyz": gen.look_at_quat_with_roll(gen.vec(position), gen.vec(target), roll),
                "roll_deg": round(roll, 5),
                "lens_mm": round(lens, 5),
            }
        )
    return records


def speed_class(distance: float, duration_s: float) -> str:
    avg = distance / max(duration_s, 1e-6)
    if avg < 0.10:
        return "none"
    if avg < 0.28:
        return "slow"
    if avg < 0.70:
        return "medium"
    return "fast"


def make_camera(
    *,
    camera_id: str,
    family: str,
    primitives: list[str],
    start_position: tuple[float, float, float],
    start_target: tuple[float, float, float],
    position_delta: tuple[float, float, float],
    target_delta: tuple[float, float, float],
    roll_start_deg: float,
    roll_delta_deg: float,
    lens_mm: float,
    lens_delta_mm: float,
    speed_curve: dict[str, Any],
    frames: int,
    path_model: str = "linear",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration_s = max((frames - 1) / gen.FPS, 1e-6)
    distance = math.sqrt(sum(float(v) ** 2 for v in position_delta))
    spec = {
        "id": camera_id,
        "family_id": family,
        "primitives": primitives,
        "speed_class": speed_class(distance, duration_s),
        "path_model": path_model,
        "start_position": gen.vec(start_position),
        "start_target": gen.vec(start_target),
        "position_delta": gen.vec(position_delta),
        "target_delta": gen.vec(target_delta),
        "average_linear_velocity_mps": gen.vec([float(v) / duration_s for v in position_delta]),
        "roll_start_deg": round(roll_start_deg, 5),
        "roll_delta_deg": round(roll_delta_deg, 5),
        "roll_velocity_deg_s_avg": round(roll_delta_deg / duration_s, 5),
        "lens_mm": round(lens_mm, 5),
        "lens_delta_mm": round(lens_delta_mm, 5),
        "lens_velocity_mm_s_avg": round(lens_delta_mm / duration_s, 5),
        "speed_curve": speed_curve,
        "motion_distribution": (
            "camera endpoints and average speeds are clipped Gaussian samples; "
            "per-frame motion follows the stored speed_curve, so most moving "
            "trajectories have within-clip speed variation"
        ),
    }
    if extra:
        spec.update(extra)
    return spec


def side_sign(rng: random.Random) -> int:
    return -1 if rng.random() < 0.5 else 1


def sample_camera(rng: random.Random, family: str, instance_id: int, frames: int) -> dict[str, Any]:
    duration_s = max((frames - 1) / gen.FPS, 1e-6)
    sign = side_sign(rng)
    curve = sample_curve(rng, static=family == "static_view")
    camera_id = f"cam_{family}_{instance_id:04d}"

    if family == "static_view":
        start = (
            sign * gen.trunc_gauss(rng, 2.25, 0.50, 0.65, 3.15),
            gen.trunc_gauss(rng, -4.25, 0.55, -5.25, -3.05),
            gen.trunc_gauss(rng, 1.65, 0.50, 0.78, 2.65),
        )
        target = (
            gen.trunc_gauss(rng, 0.0, 0.18, -0.35, 0.35),
            gen.trunc_gauss(rng, 0.0, 0.15, -0.30, 0.30),
            gen.trunc_gauss(rng, 0.38, 0.09, 0.20, 0.58),
        )
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["static", "off_center_start"],
            start_position=start,
            start_target=target,
            position_delta=(0.0, 0.0, 0.0),
            target_delta=(0.0, 0.0, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.8, -5.0, 5.0),
            roll_delta_deg=0.0,
            lens_mm=gen.trunc_gauss(rng, 35.0, 3.0, 29.0, 44.0),
            lens_delta_mm=0.0,
            speed_curve=curve,
            frames=frames,
        )

    if family == "dolly_in":
        speed = gen.trunc_gauss(rng, 0.54, 0.18, 0.18, 0.95)
        dist = min(gen.trunc_gauss(rng, speed * duration_s, 0.20, 0.65, 2.25), 2.35)
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["dolly_in", "off_axis_start", "mild_zoom"],
            start_position=(sign * gen.trunc_gauss(rng, 1.65, 0.45, 0.55, 2.65), -5.10, gen.trunc_gauss(rng, 1.40, 0.22, 1.02, 1.88)),
            start_target=(gen.trunc_gauss(rng, 0.02, 0.12, -0.22, 0.28), 0.0, gen.trunc_gauss(rng, 0.36, 0.07, 0.22, 0.52)),
            position_delta=(-sign * gen.trunc_gauss(rng, 0.08, 0.07, -0.04, 0.24), dist, gen.trunc_gauss(rng, 0.02, 0.06, -0.10, 0.16)),
            target_delta=(gen.trunc_gauss(rng, 0.03, 0.06, -0.08, 0.14), gen.trunc_gauss(rng, 0.0, 0.04, -0.08, 0.08), 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.2, -3.0, 3.0),
            roll_delta_deg=gen.trunc_gauss(rng, 0.3, 0.8, -1.3, 2.2),
            lens_mm=gen.trunc_gauss(rng, 33.0, 2.5, 28.0, 40.0),
            lens_delta_mm=gen.trunc_gauss(rng, 1.5, 1.0, -0.5, 4.0),
            speed_curve=curve,
            frames=frames,
        )

    if family == "dolly_out":
        speed = gen.trunc_gauss(rng, 0.45, 0.16, 0.16, 0.82)
        dist = min(gen.trunc_gauss(rng, speed * duration_s, 0.18, 0.55, 2.05), 2.15)
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["dolly_out", "low_or_mid_start", "mild_zoom_out"],
            start_position=(sign * gen.trunc_gauss(rng, 1.20, 0.50, 0.25, 2.35), gen.trunc_gauss(rng, -2.85, 0.35, -3.45, -2.05), gen.trunc_gauss(rng, 1.18, 0.28, 0.72, 1.82)),
            start_target=(gen.trunc_gauss(rng, -0.02, 0.12, -0.28, 0.22), 0.0, gen.trunc_gauss(rng, 0.34, 0.07, 0.20, 0.50)),
            position_delta=(sign * gen.trunc_gauss(rng, 0.05, 0.06, -0.08, 0.18), -dist, gen.trunc_gauss(rng, 0.03, 0.07, -0.10, 0.18)),
            target_delta=(gen.trunc_gauss(rng, 0.0, 0.06, -0.12, 0.12), 0.0, gen.trunc_gauss(rng, 0.04, 0.04, -0.04, 0.12)),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.2, -3.0, 3.0),
            roll_delta_deg=gen.trunc_gauss(rng, -0.2, 0.7, -1.8, 1.4),
            lens_mm=gen.trunc_gauss(rng, 38.0, 2.5, 32.0, 45.0),
            lens_delta_mm=gen.trunc_gauss(rng, -1.1, 0.9, -3.5, 0.7),
            speed_curve=curve,
            frames=frames,
        )

    if family == "truck_pan":
        speed = gen.trunc_gauss(rng, 0.50, 0.17, 0.18, 0.92)
        dist = min(gen.trunc_gauss(rng, speed * duration_s, 0.20, 0.70, 2.35), 2.45)
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["truck", "counter_pan", "off_center_start"],
            start_position=(sign * gen.trunc_gauss(rng, 2.65, 0.35, 1.75, 3.35), gen.trunc_gauss(rng, -4.25, 0.35, -5.00, -3.45), gen.trunc_gauss(rng, 1.32, 0.20, 0.92, 1.78)),
            start_target=(sign * gen.trunc_gauss(rng, 0.28, 0.12, 0.05, 0.55), 0.0, gen.trunc_gauss(rng, 0.38, 0.08, 0.22, 0.55)),
            position_delta=(-sign * dist, gen.trunc_gauss(rng, 0.02, 0.06, -0.12, 0.14), 0.0),
            target_delta=(-sign * gen.trunc_gauss(rng, 0.28, 0.11, 0.08, 0.56), gen.trunc_gauss(rng, 0.02, 0.04, -0.06, 0.10), 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.1, -2.6, 2.6),
            roll_delta_deg=gen.trunc_gauss(rng, 0.0, 0.9, -2.0, 2.0),
            lens_mm=gen.trunc_gauss(rng, 35.0, 2.8, 29.0, 43.0),
            lens_delta_mm=gen.trunc_gauss(rng, 0.0, 0.45, -1.0, 1.0),
            speed_curve=curve,
            frames=frames,
        )

    if family == "crane_tilt":
        up = 1 if rng.random() < 0.68 else -1
        dist = up * gen.trunc_gauss(rng, 0.78, 0.25, 0.28, 1.35)
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["crane", "tilt", "height_change"],
            start_position=(sign * gen.trunc_gauss(rng, 0.75, 0.45, -0.10, 1.75), gen.trunc_gauss(rng, -4.55, 0.35, -5.25, -3.70), gen.trunc_gauss(rng, 1.05, 0.18, 0.72, 1.42)),
            start_target=(gen.trunc_gauss(rng, 0.0, 0.12, -0.24, 0.24), 0.0, gen.trunc_gauss(rng, 0.30, 0.06, 0.18, 0.44)),
            position_delta=(gen.trunc_gauss(rng, 0.04, 0.10, -0.18, 0.24), gen.trunc_gauss(rng, 0.04, 0.06, -0.08, 0.16), dist),
            target_delta=(0.0, 0.0, -up * gen.trunc_gauss(rng, 0.14, 0.08, 0.02, 0.32)),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.0, -2.4, 2.4),
            roll_delta_deg=gen.trunc_gauss(rng, 0.0, 0.65, -1.6, 1.6),
            lens_mm=gen.trunc_gauss(rng, 36.5, 2.5, 31.0, 44.0),
            lens_delta_mm=gen.trunc_gauss(rng, 0.2, 0.7, -1.2, 1.8),
            speed_curve=curve,
            frames=frames,
        )

    if family == "top_down_drift":
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["top_down", "ceiling_start", "drift"],
            start_position=(gen.trunc_gauss(rng, 0.55, 0.45, -0.35, 1.45), gen.trunc_gauss(rng, -0.75, 0.45, -1.55, 0.20), gen.trunc_gauss(rng, 5.05, 0.30, 4.45, 5.55)),
            start_target=(gen.trunc_gauss(rng, 0.0, 0.12, -0.24, 0.24), gen.trunc_gauss(rng, 0.0, 0.12, -0.24, 0.24), gen.trunc_gauss(rng, 0.22, 0.05, 0.12, 0.34)),
            position_delta=(sign * gen.trunc_gauss(rng, 0.62, 0.22, 0.18, 1.08), -sign * gen.trunc_gauss(rng, 0.42, 0.18, 0.10, 0.82), gen.trunc_gauss(rng, -0.18, 0.14, -0.55, 0.08)),
            target_delta=(gen.trunc_gauss(rng, 0.05, 0.06, -0.08, 0.18), gen.trunc_gauss(rng, 0.02, 0.06, -0.12, 0.14), 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 5.0, 2.2, 0.0, 10.0),
            roll_delta_deg=gen.trunc_gauss(rng, 1.0, 1.0, -1.2, 3.4),
            lens_mm=gen.trunc_gauss(rng, 38.0, 2.5, 32.0, 45.0),
            lens_delta_mm=gen.trunc_gauss(rng, 0.0, 0.5, -1.0, 1.2),
            speed_curve=curve,
            frames=frames,
        )

    if family == "orbit_arc":
        radius = gen.trunc_gauss(rng, 4.00, 0.35, 3.30, 4.75)
        angle_start = gen.trunc_gauss(rng, sign * 38.0, 9.0, -58.0 if sign < 0 else 18.0, -18.0 if sign < 0 else 58.0)
        angle_delta = -sign * gen.trunc_gauss(rng, 30.0, 12.0, 12.0, 58.0)
        height = gen.trunc_gauss(rng, 1.45, 0.24, 0.98, 2.05)
        angle = math.radians(angle_start)
        start = (radius * math.sin(angle), -radius * math.cos(angle), height)
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["orbit", "arc", "look_at_center"],
            start_position=start,
            start_target=(0.0, 0.0, gen.trunc_gauss(rng, 0.34, 0.06, 0.22, 0.48)),
            position_delta=(0.0, 0.0, 0.0),
            target_delta=(gen.trunc_gauss(rng, 0.02, 0.04, -0.06, 0.10), 0.0, 0.0),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.0, -2.4, 2.4),
            roll_delta_deg=gen.trunc_gauss(rng, 0.0, 0.65, -1.4, 1.4),
            lens_mm=gen.trunc_gauss(rng, 36.0, 2.4, 30.0, 43.0),
            lens_delta_mm=gen.trunc_gauss(rng, 0.0, 0.5, -1.0, 1.0),
            speed_curve=curve,
            frames=frames,
            path_model="orbit",
            extra={
                "speed_class": speed_class(abs(math.radians(angle_delta) * radius), duration_s),
                "orbit_center": [0.0, 0.0, 0.34],
                "orbit_radius_m": radius,
                "orbit_radius_delta_m": gen.trunc_gauss(rng, 0.0, 0.08, -0.18, 0.18),
                "orbit_start_deg": angle_start,
                "orbit_delta_deg": angle_delta,
                "orbit_height_m": height,
                "orbit_height_delta_m": gen.trunc_gauss(rng, 0.02, 0.08, -0.16, 0.22),
            },
        )

    if family == "low_truck_roll":
        return make_camera(
            camera_id=camera_id,
            family=family,
            primitives=["low_start", "truck", "roll"],
            start_position=(-sign * gen.trunc_gauss(rng, 2.25, 0.40, 1.30, 3.05), gen.trunc_gauss(rng, -3.65, 0.35, -4.35, -2.90), gen.trunc_gauss(rng, 0.82, 0.13, 0.58, 1.12)),
            start_target=(-sign * gen.trunc_gauss(rng, 0.28, 0.12, 0.04, 0.56), 0.0, gen.trunc_gauss(rng, 0.30, 0.07, 0.18, 0.45)),
            position_delta=(sign * gen.trunc_gauss(rng, 0.98, 0.30, 0.38, 1.65), gen.trunc_gauss(rng, 0.16, 0.12, -0.08, 0.42), gen.trunc_gauss(rng, 0.05, 0.05, -0.05, 0.16)),
            target_delta=(sign * gen.trunc_gauss(rng, 0.18, 0.08, 0.04, 0.36), 0.0, gen.trunc_gauss(rng, 0.02, 0.04, -0.06, 0.10)),
            roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.4, -3.5, 3.5),
            roll_delta_deg=gen.trunc_gauss(rng, 0.0, 1.8, -4.2, 4.2),
            lens_mm=gen.trunc_gauss(rng, 32.0, 2.2, 28.0, 38.0),
            lens_delta_mm=gen.trunc_gauss(rng, 0.0, 0.5, -1.0, 1.2),
            speed_curve=curve,
            frames=frames,
        )

    if family != "diagonal_combo":
        raise ValueError(f"unknown camera family {family}")

    return make_camera(
        camera_id=camera_id,
        family=family,
        primitives=["diagonal_move", "pan_tilt_combo", "off_axis_start"],
        start_position=(sign * gen.trunc_gauss(rng, 2.35, 0.38, 1.35, 3.10), gen.trunc_gauss(rng, -4.75, 0.38, -5.45, -3.85), gen.trunc_gauss(rng, 1.25, 0.22, 0.82, 1.76)),
        start_target=(sign * gen.trunc_gauss(rng, 0.15, 0.10, -0.05, 0.38), gen.trunc_gauss(rng, 0.0, 0.08, -0.18, 0.18), gen.trunc_gauss(rng, 0.34, 0.07, 0.20, 0.50)),
        position_delta=(-sign * gen.trunc_gauss(rng, 0.62, 0.22, 0.18, 1.12), gen.trunc_gauss(rng, 0.82, 0.24, 0.32, 1.42), gen.trunc_gauss(rng, 0.28, 0.14, -0.05, 0.58)),
        target_delta=(-sign * gen.trunc_gauss(rng, 0.16, 0.08, 0.02, 0.34), gen.trunc_gauss(rng, 0.04, 0.06, -0.08, 0.16), gen.trunc_gauss(rng, -0.04, 0.06, -0.16, 0.08)),
        roll_start_deg=gen.trunc_gauss(rng, 0.0, 1.2, -3.0, 3.0),
        roll_delta_deg=gen.trunc_gauss(rng, 0.0, 1.2, -2.8, 2.8),
        lens_mm=gen.trunc_gauss(rng, 35.5, 2.6, 29.0, 43.0),
        lens_delta_mm=gen.trunc_gauss(rng, 0.4, 0.8, -1.0, 2.2),
        speed_curve=curve,
        frames=frames,
    )


def sample_physics(family: str, instance_id: int, seed: int) -> dict[str, Any]:
    options = {item["id"]: item for item in diversity.physics_specs(seed)}
    if family not in options:
        raise ValueError(f"unknown physics family {family}")
    physics = copy.deepcopy(options[family])
    physics["family_id"] = family
    physics["base_id"] = family
    physics["id"] = f"{family}_{instance_id:04d}"
    physics["sample_seed"] = seed
    return physics


def sample_world(rng: random.Random, pair_index: int) -> dict[str, Any]:
    worlds = diversity.world_specs()
    world = copy.deepcopy(worlds[pair_index % len(worlds)])
    world["selection_note"] = "shared by both clips in a pair; varied across pairs"
    if rng.random() < 0.35:
        # Small color changes keep the scene varied without adding confusing
        # foreground clutter.
        for key in ["ground_color", "back_wall_color", "side_wall_color"]:
            color = list(world[key])
            world[key] = [
                round(clamp(color[i] + rng.gauss(0.0, 0.025), 0.35, 0.90), 5) if i < 3 else color[i]
                for i in range(4)
            ]
        world["id"] = f"{world['id']}_tint_{pair_index:04d}"
    return world


def make_pair_specs(
    rng: random.Random,
    pair_index: int,
    attempt: int,
    frames: int,
    pair_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    world = sample_world(rng, pair_index + attempt)
    instance_base = pair_index * 10 + attempt

    if pair_kind == "same_camera":
        camera_family = CAMERA_FAMILIES[pair_index % len(CAMERA_FAMILIES)]
        camera = sample_camera(rng, camera_family, instance_base, frames)
        physics_family_a = PHYSICS_FAMILIES[pair_index % len(PHYSICS_FAMILIES)]
        physics_family_b = PHYSICS_FAMILIES[(pair_index + 3) % len(PHYSICS_FAMILIES)]
        physics_a = sample_physics(physics_family_a, instance_base * 2, rng.randint(1, 2_000_000_000))
        physics_b = sample_physics(physics_family_b, instance_base * 2 + 1, rng.randint(1, 2_000_000_000))
        specs = [
            {"camera": camera, "physics": physics_a, "world": world, "frames": frames},
            {"camera": camera, "physics": physics_b, "world": world, "frames": frames},
        ]
        group = {
            "kind": "same_camera",
            "title": f"Same camera trajectory {camera['family_id']}, different physics",
            "controlled_factor": "camera_id/scene_id/frames",
            "varied_factor": "physics_id",
            "tags": ["same_camera", "same_scene", "different_physics", "paircentric_review"],
        }
        return specs, group

    if pair_kind != "same_physics_scene":
        raise ValueError(pair_kind)

    physics_family = PHYSICS_FAMILIES[pair_index % len(PHYSICS_FAMILIES)]
    physics = sample_physics(physics_family, instance_base, rng.randint(1, 2_000_000_000))
    camera_family_a = CAMERA_FAMILIES[pair_index % len(CAMERA_FAMILIES)]
    camera_family_b = CAMERA_FAMILIES[(pair_index + 4) % len(CAMERA_FAMILIES)]
    if camera_family_a == camera_family_b:
        camera_family_b = CAMERA_FAMILIES[(pair_index + 1) % len(CAMERA_FAMILIES)]
    camera_a = sample_camera(rng, camera_family_a, instance_base * 2, frames)
    camera_b = sample_camera(rng, camera_family_b, instance_base * 2 + 1, frames)
    specs = [
        {"camera": camera_a, "physics": physics, "world": world, "frames": frames},
        {"camera": camera_b, "physics": physics, "world": world, "frames": frames},
    ]
    group = {
        "kind": "same_physics_scene",
        "title": f"Same physics and scene {physics['family_id']}, different cameras",
        "controlled_factor": "physics_id/scene_id/frames",
        "varied_factor": "camera_id",
        "tags": ["same_physics", "same_scene", "different_camera", "paircentric_review"],
    }
    return specs, group


def quality_or_error(physics: dict[str, Any], quality: dict[str, Any]) -> str | None:
    if not quality.get("expected_contacts_passed", False):
        return "expected contact audit failed"
    if not quality.get("physics_plausibility_passed", False):
        checks = quality.get("physics_plausibility", {}).get("checks", {})
        failed = [name for name, check in checks.items() if not check.get("passed", False)]
        return f"physics plausibility audit failed: {failed}"
    return None


def write_clip_metadata(
    *,
    run_dir: Path,
    clip_index: int,
    pair_group_id: str,
    pair_kind: str,
    spec: dict[str, Any],
    physics_frames: list[dict[str, Any]],
    quality: dict[str, Any],
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    camera = spec["camera"]
    physics = spec["physics"]
    world = spec["world"]
    frames = int(spec["frames"])
    clip_id = f"clip_{clip_index:04d}_{camera['id']}_{physics['id']}"
    metadata_rel = Path("metadata") / f"{clip_id}.json"
    video_rel = Path("videos") / f"{clip_id}.mp4"
    frames_rel = Path("frames") / clip_id
    metadata = {
        "clip_id": clip_id,
        "pair_group_id": pair_group_id,
        "pair_kind": pair_kind,
        "camera_id": camera["id"],
        "camera_family": camera["family_id"],
        "physics_id": physics["id"],
        "physics_family": physics["family_id"],
        "scene_id": world["id"],
        "fps": gen.FPS,
        "frames_count": frames,
        "duration_s": round((frames - 1) / gen.FPS, 5),
        "resolution": [width, height],
        "generator": "official_kubric_batch_v2_pair_sampler",
        "simulator": "official kubric.simulator.PyBullet",
        "renderer": "official kubric.renderer.Blender",
        "camera_spec": camera,
        "physics_spec": physics,
        "scene_spec": world,
        "camera_frames": camera_records(camera, frames),
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
        "camera_family": camera["family_id"],
        "physics_id": physics["id"],
        "physics_family": physics["family_id"],
        "scene_id": world["id"],
        "frames_count": frames,
        "duration_s": round((frames - 1) / gen.FPS, 5),
        "camera_primitives": camera["primitives"],
        "physics_kind": physics["kind"],
        "physics_speed_class": physics["speed_class"],
        "camera_speed_class": camera["speed_class"],
        "speed_curve": camera["speed_curve"],
        "pair_groups": [pair_group_id],
        "video": str(video_rel),
        "metadata": str(metadata_rel),
    }
    job = {
        "clip_id": clip_id,
        "metadata": str(run_dir / metadata_rel),
        "frames_dir": str(run_dir / frames_rel),
        "video": str(run_dir / video_rel),
    }
    return clip, job


def materialize_pair(
    *,
    run_dir: Path,
    clip_index: int,
    pair_index: int,
    pair_specs: list[dict[str, Any]],
    group_spec: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair_group_id = f"pair_{pair_index:04d}_{group_spec['kind']}"
    sim_cache: dict[tuple[str, str, int], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for spec in pair_specs:
        physics = spec["physics"]
        world = spec["world"]
        frames = int(spec["frames"])
        key = (physics["id"], world["id"], frames)
        if key not in sim_cache:
            physics_frames, quality = gen.simulate_physics(physics, world, frames, width, height)
            error = quality_or_error(physics, quality)
            if error:
                raise RuntimeError(f"{physics['id']} / {world['id']}: {error}")
            sim_cache[key] = (physics_frames, quality)

    clips: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for offset, spec in enumerate(pair_specs):
        key = (spec["physics"]["id"], spec["world"]["id"], int(spec["frames"]))
        physics_frames, quality = sim_cache[key]
        clip, job = write_clip_metadata(
            run_dir=run_dir,
            clip_index=clip_index + offset,
            pair_group_id=pair_group_id,
            pair_kind=group_spec["kind"],
            spec=spec,
            physics_frames=physics_frames,
            quality=quality,
            width=width,
            height=height,
        )
        clips.append(clip)
        jobs.append(job)

    pair_group = {
        "group_id": pair_group_id,
        "title": group_spec["title"],
        "controlled_factor": group_spec["controlled_factor"],
        "varied_factor": group_spec["varied_factor"],
        "clip_ids": [clip["clip_id"] for clip in clips],
        "tags": group_spec["tags"],
        "frames_count": clips[0]["frames_count"],
        "duration_s": clips[0]["duration_s"],
    }
    return clips, jobs, pair_group


def summarize_counts(clips: list[dict[str, Any]], pair_groups: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [clip["duration_s"] for clip in clips]
    frames = [clip["frames_count"] for clip in clips]
    return {
        "camera_families": dict(Counter(clip["camera_family"] for clip in clips)),
        "physics_families": dict(Counter(clip["physics_family"] for clip in clips)),
        "pair_kinds": dict(Counter(group["tags"][0] for group in pair_groups)),
        "duration_s_min": min(durations) if durations else None,
        "duration_s_max": max(durations) if durations else None,
        "frames_min": min(frames) if frames else None,
        "frames_max": max(frames) if frames else None,
    }


def write_run(args: argparse.Namespace) -> Path:
    rng = random.Random(args.seed)
    run_dir = args.run_root / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} already exists; pass --overwrite to replace generated assets.")
        shutil.rmtree(run_dir)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    render_jobs: list[dict[str, Any]] = []
    pair_groups: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for pair_index in range(args.pairs):
        pair_kind = "same_camera" if pair_index % 2 == 0 else "same_physics_scene"
        if args.same_camera_fraction >= 0.0:
            pair_kind = "same_camera" if rng.random() < args.same_camera_fraction else "same_physics_scene"
        pair_done = False
        for attempt in range(args.max_pair_attempts):
            frames = frame_count(rng, args.frames_mean, args.frames_std, args.frames_min, args.frames_max, args.frame_multiple)
            try:
                pair_specs, group_spec = make_pair_specs(rng, pair_index, attempt, frames, pair_kind)
                new_clips, new_jobs, pair_group = materialize_pair(
                    run_dir=run_dir,
                    clip_index=len(clips),
                    pair_index=pair_index,
                    pair_specs=pair_specs,
                    group_spec=group_spec,
                    width=args.width,
                    height=args.height,
                )
            except Exception as exc:  # noqa: BLE001 - record failed samples for audit.
                failures.append({"pair_index": pair_index, "attempt": attempt, "kind": pair_kind, "error": str(exc)})
                continue
            clips.extend(new_clips)
            render_jobs.extend(new_jobs)
            pair_groups.append(pair_group)
            pair_done = True
            break
        if not pair_done:
            raise RuntimeError(f"failed to create pair {pair_index} after {args.max_pair_attempts} attempts")

    coverage = summarize_counts(clips, pair_groups)
    manifest = {
        "project": "camera_motion_disentangle",
        "run_id": args.run_id,
        "generator": "official_kubric_batch_v2_pair_sampler",
        "description": (
            "Pair-centric official-Kubric batch/review run. It samples random "
            "camera/physics combinations for coverage instead of rendering a "
            "full camera x physics grid."
        ),
        "fps": gen.FPS,
        "resolution": [args.width, args.height],
        "seed": args.seed,
        "simulator": "official kubric.simulator.PyBullet",
        "renderer": "official kubric.renderer.Blender",
        "review_strategy": "coverage-oriented random pair sampling; not a full camera-by-object grid",
        "pair_contract": {
            "same_camera": "both clips reuse identical camera_frames, scene_id, fps, and frames_count; physics differs",
            "same_physics_scene": "both clips reuse identical physics_frames, scene_id, fps, and frames_count; camera differs",
            "length_policy": "clips inside a pair have identical length; different pairs sample different lengths",
        },
        "duration_sampling": {
            "distribution": "clipped Gaussian rounded to frame_multiple",
            "frames_mean": args.frames_mean,
            "frames_std": args.frames_std,
            "frames_min": args.frames_min,
            "frames_max": args.frames_max,
            "frame_multiple": args.frame_multiple,
        },
        "camera_motion_model": (
            "camera endpoints, roll, lens, target drift, and orbit parameters are "
            "sampled from clipped Gaussian templates; speed_curve controls "
            "within-clip acceleration/deceleration"
        ),
        "physics_sampling_model": (
            "object sizes, aspect ratios, masses, initial velocities, angular "
            "velocities, friction, restitution, colors, and material profiles are "
            "sampled from clipped Gaussian templates"
        ),
        "quality_filters": (
            "Every physics sample must pass expected-contact checks and plausibility "
            "audits before metadata or render jobs are accepted."
        ),
        "camera_family_reference": CAMERA_FAMILIES,
        "physics_family_reference": PHYSICS_FAMILIES,
        "coverage_summary": coverage,
        "sample_failures": failures[:20],
        "clips": clips,
        "pair_groups": pair_groups,
        "ambiguous_equivalence_groups": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    jobs_path = run_dir / "render_jobs.json"
    jobs_path.write_text(json.dumps({"run_id": args.run_id, "fps": gen.FPS, "jobs": render_jobs}, indent=2), encoding="utf-8")
    gen.update_index(args.run_root, args.run_id)
    return jobs_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=gen.DEFAULT_BLENDER)
    parser.add_argument("--kubric-site-packages", type=Path, default=gen.DEFAULT_KUBRIC_SITE_PACKAGES)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--width", type=int, default=gen.WIDTH)
    parser.add_argument("--height", type=int, default=gen.HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frames-mean", type=float, default=108.0)
    parser.add_argument("--frames-std", type=float, default=24.0)
    parser.add_argument("--frames-min", type=int, default=72)
    parser.add_argument("--frames-max", type=int, default=168)
    parser.add_argument("--frame-multiple", type=int, default=12)
    parser.add_argument("--same-camera-fraction", type=float, default=-1.0)
    parser.add_argument("--max-pair-attempts", type=int, default=16)
    parser.add_argument("--progress-watch-interval", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if args.frames_min <= 1 or args.frames_max < args.frames_min:
        raise SystemExit("invalid frame bounds")
    if args.frame_multiple <= 0:
        raise SystemExit("--frame-multiple must be positive")

    jobs_path = write_run(args)
    run_dir = args.run_root / args.run_id
    write_progress(run_dir, Path(sys.executable))
    if not args.no_render:
        progress_watcher: subprocess.Popen[bytes] | None = None
        if args.progress_watch_interval > 0:
            progress_watcher = subprocess.Popen(
                [
                    sys.executable,
                    str(gen.PROJECT_ROOT / "scripts" / "update_kubric_progress.py"),
                    "--run-dir",
                    str(run_dir),
                    "--python-bin",
                    sys.executable,
                    "--watch",
                    str(args.progress_watch_interval),
                    "--until-complete",
                ]
            )
        try:
            gen.render_with_blender(args.blender_bin, jobs_path, args.kubric_site_packages)
            write_progress(run_dir, Path(sys.executable))
            gen.encode_videos(jobs_path)
            write_progress(run_dir, Path(sys.executable))
        finally:
            if progress_watcher is not None and progress_watcher.poll() is None:
                progress_watcher.terminate()
                try:
                    progress_watcher.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    progress_watcher.kill()
                    progress_watcher.wait()
            write_progress(run_dir, Path(sys.executable))
    else:
        write_progress(run_dir, Path(sys.executable))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(
        f"Wrote {len(manifest['clips'])} clips and {len(manifest['pair_groups'])} pairs "
        f"to {run_dir}"
    )


if __name__ == "__main__":
    main()
