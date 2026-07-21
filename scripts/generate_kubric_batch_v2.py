#!/usr/bin/env python3
"""Generate pair/block-centric official-Kubric batch/review runs.

The older review scripts render a full camera x physics grid.  This script
samples shared-factor groups first, so the review set can cover many camera and
object families without showing every possible global combination.  It preserves
the same shared-factor contracts:

* same-camera groups reuse one exact camera trajectory and vary physics.
* same-physics groups reuse one exact physics simulation and scene, then vary
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
    "phys_multi_drop_bounce",
    "phys_airborne_drop_collision",
    "phys_airborne_sphere_box_collision",
    "phys_airborne_box_box_collision",
    "phys_airborne_sphere_cylinder_collision",
    "phys_airborne_chain_collision",
    "phys_drop_hits_dynamic_box",
    "phys_two_sphere_collision",
    "phys_sphere_hits_tall_block",
    "phys_box_sphere_collision",
    "phys_four_body_scatter",
    "phys_three_body_chain",
    "phys_four_body_crossfire",
    "phys_dynamic_cylinder_roll",
    "phys_dynamic_capsule_roll",
    "phys_sphere_hits_dynamic_cylinder",
    "phys_randomized_mixed_drop_scene",
    "phys_randomized_mixed_drop_collision",
    "phys_randomized_mixed_drop_no_collision",
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


def sample_camera_speed_variant(rng: random.Random, family: str) -> dict[str, Any]:
    if family == "static_view":
        return {
            "band": "static",
            "multiplier": 1.0,
            "note": "static camera has no translational speed multiplier",
        }
    band = rng.choices(["baseline", "brisk"], weights=[0.78, 0.22], k=1)[0]
    if band == "brisk":
        multiplier = gen.trunc_gauss(rng, 1.28, 0.12, 1.08, 1.50)
    else:
        multiplier = gen.trunc_gauss(rng, 1.00, 0.06, 0.88, 1.12)
    return {
        "band": band,
        "multiplier": round(multiplier, 5),
        "note": "brisk samples are capped around 1.5x the baseline camera-motion scale",
    }


def scale_tuple(values: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(float(value) * scale for value in values)


def sample_camera(rng: random.Random, family: str, instance_id: int, frames: int) -> dict[str, Any]:
    duration_s = max((frames - 1) / gen.FPS, 1e-6)
    sign = side_sign(rng)
    curve = sample_curve(rng, static=family == "static_view")
    camera_id = f"cam_{family}_{instance_id:04d}"
    speed_variant = sample_camera_speed_variant(rng, family)
    speed_gain = float(speed_variant["multiplier"])

    def build_camera(**kwargs: Any) -> dict[str, Any]:
        extra = dict(kwargs.pop("extra", {}) or {})
        if family != "static_view":
            if kwargs.get("path_model") == "orbit":
                if "orbit_delta_deg" in extra:
                    original_delta = float(extra["orbit_delta_deg"])
                    delta_sign = 1.0 if original_delta >= 0.0 else -1.0
                    scaled_delta = delta_sign * min(abs(original_delta * speed_gain), 78.0)
                    extra["orbit_delta_deg"] = round(scaled_delta, 5)
                    radius = float(extra.get("orbit_radius_m", 0.0))
                    extra["speed_class"] = speed_class(abs(math.radians(scaled_delta) * radius), duration_s)
            else:
                kwargs["position_delta"] = scale_tuple(kwargs["position_delta"], speed_gain)
        extra["speed_sampling_band"] = speed_variant["band"]
        extra["speed_multiplier"] = speed_variant["multiplier"]
        extra["speed_sampling_note"] = speed_variant["note"]
        return make_camera(**kwargs, extra=extra)

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
        return build_camera(
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
        return build_camera(
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
        return build_camera(
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
        return build_camera(
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
        return build_camera(
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
        return build_camera(
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
        return build_camera(
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
        return build_camera(
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

    return build_camera(
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


def family_sequence(families: list[str], start: int, count: int, stride: int) -> list[str]:
    if count <= 0:
        return []
    if not families:
        raise ValueError("empty family pool")
    result: list[str] = []
    index = start
    attempts = 0
    max_attempts = max(len(families) * 4, count * 4)
    while len(result) < count and attempts < max_attempts:
        family = families[index % len(families)]
        if family not in result or count > len(families):
            result.append(family)
        index += stride
        attempts += 1
    if len(result) < count:
        for family in families:
            if len(result) >= count:
                break
            if family not in result or count > len(families):
                result.append(family)
    return result


def make_group_specs(
    rng: random.Random,
    group_index: int,
    attempt: int,
    frames: int,
    group_kind: str,
    group_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    world = sample_world(rng, group_index + attempt)
    instance_base = group_index * 100 + attempt * 10
    structure_tag = "paircentric_review" if group_size == 2 else "shared_factor_block"

    if group_kind == "same_camera":
        camera_family = CAMERA_FAMILIES[group_index % len(CAMERA_FAMILIES)]
        camera = sample_camera(rng, camera_family, instance_base, frames)
        physics_families = family_sequence(PHYSICS_FAMILIES, group_index, group_size, stride=3)
        physics_specs = [
            sample_physics(physics_family, instance_base * 10 + offset, rng.randint(1, 2_000_000_000))
            for offset, physics_family in enumerate(physics_families)
        ]
        specs = [
            {"camera": camera, "physics": physics, "world": world, "frames": frames}
            for physics in physics_specs
        ]
        group = {
            "kind": "same_camera",
            "group_structure": "same_camera_pair" if group_size == 2 else "same_camera_block",
            "title": (
                f"Same camera trajectory {camera['family_id']}, "
                f"{group_size} different physics programs"
            ),
            "controlled_factor": "camera_id/scene_id/frames",
            "varied_factor": "physics_id",
            "tags": ["same_camera", "same_scene", "different_physics", structure_tag],
        }
        return specs, group

    if group_kind != "same_physics_scene":
        raise ValueError(group_kind)

    physics_family = PHYSICS_FAMILIES[group_index % len(PHYSICS_FAMILIES)]
    physics = sample_physics(physics_family, instance_base, rng.randint(1, 2_000_000_000))
    camera_families = family_sequence(CAMERA_FAMILIES, group_index, group_size, stride=4)
    cameras = [
        sample_camera(rng, camera_family, instance_base * 10 + offset, frames)
        for offset, camera_family in enumerate(camera_families)
    ]
    specs = [
        {"camera": camera, "physics": physics, "world": world, "frames": frames}
        for camera in cameras
    ]
    group = {
        "kind": "same_physics_scene",
        "group_structure": "same_physics_pair" if group_size == 2 else "same_physics_block",
        "title": (
            f"Same physics and scene {physics['family_id']}, "
            f"{group_size} different camera trajectories"
        ),
        "controlled_factor": "physics_id/scene_id/frames",
        "varied_factor": "camera_id",
        "tags": ["same_physics", "same_scene", "different_camera", structure_tag],
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


def sub_vec(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def norm(a: list[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in a))


def normalize(a: list[float], fallback: tuple[float, float, float]) -> list[float]:
    length = norm(a)
    if length < 1e-8:
        return [float(v) for v in fallback]
    return [float(v) / length for v in a]


def object_radius_for_framing(obj: dict[str, Any]) -> float:
    radius = obj.get("bounding_radius")
    if radius is not None:
        return float(radius)
    if obj.get("radius") is not None:
        return float(obj["radius"])
    half = obj.get("half_extents") or [v / 2.0 for v in obj.get("size", [0.4, 0.4, 0.4])]
    return math.sqrt(sum(float(v) ** 2 for v in half))


def camera_basis(camera_frame: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    position = [float(v) for v in camera_frame["position"]]
    target = [float(v) for v in camera_frame["look_at"]]
    forward = normalize(sub_vec(target, position), (0.0, 1.0, 0.0))
    world_up = [0.0, 0.0, 1.0]
    right_raw = cross(forward, world_up)
    right = normalize(right_raw, (1.0, 0.0, 0.0))
    if norm(right_raw) < 1e-5:
        right = normalize(cross(forward, [0.0, 1.0, 0.0]), (1.0, 0.0, 0.0))
    up = normalize(cross(right, forward), (0.0, 0.0, 1.0))
    return forward, right, up


def object_in_camera_frame(
    obj: dict[str, Any],
    camera_frame: dict[str, Any],
    width: int,
    height: int,
    *,
    safe_margin: float = 0.92,
    radius_scale: float = 0.80,
) -> bool:
    camera_position = [float(v) for v in camera_frame["position"]]
    obj_position = [float(v) for v in obj["position"]]
    rel = sub_vec(obj_position, camera_position)
    forward, right, up = camera_basis(camera_frame)
    depth = dot(rel, forward)
    radius = object_radius_for_framing(obj) * radius_scale
    if depth <= max(0.05, radius * 0.4):
        return False
    lens = float(camera_frame["lens_mm"])
    sensor_width = 32.0
    sensor_height = sensor_width * float(height) / float(width)
    half_width = math.tan(math.atan(sensor_width / (2.0 * lens))) * safe_margin
    half_height = math.tan(math.atan(sensor_height / (2.0 * lens))) * safe_margin
    x_ndc = dot(rel, right) / depth
    y_ndc = dot(rel, up) / depth
    r_ndc = radius / depth
    return abs(x_ndc) + r_ndc <= half_width and abs(y_ndc) + r_ndc <= half_height


def longest_consecutive_run(frames: list[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for frame in frames:
        if previous is None or frame == previous + 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        previous = frame
    return max(longest, current)


def camera_framing_audit(
    camera_frames: list[dict[str, Any]],
    physics_frames: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    min_fraction: float = 0.08,
) -> dict[str, Any]:
    frame_count = min(len(camera_frames), len(physics_frames))
    if frame_count == 0:
        return {"passed": False, "reason": "empty camera or physics frames"}
    min_contiguous = max(8, min(18, int(round(frame_count * min_fraction))))
    all_visible_frames: list[int] = []
    all_extent_visible_frames: list[int] = []
    visible_counts: list[int] = []
    extent_visible_counts: list[int] = []
    object_count = 0
    worst_frame: dict[str, Any] | None = None
    strict_worst_frame: dict[str, Any] | None = None
    for idx in range(frame_count):
        camera_frame = camera_frames[idx]
        objects = physics_frames[idx]["objects"]
        object_count = max(object_count, len(objects))
        visible = [
            obj["name"]
            for obj in objects
            if object_in_camera_frame(obj, camera_frame, width, height, safe_margin=0.96, radius_scale=0.20)
        ]
        extent_visible = [
            obj["name"]
            for obj in objects
            if object_in_camera_frame(obj, camera_frame, width, height, safe_margin=0.92, radius_scale=0.80)
        ]
        visible_names = set(visible)
        extent_visible_names = set(extent_visible)
        missing = [obj["name"] for obj in objects if obj["name"] not in visible_names]
        extent_missing = [obj["name"] for obj in objects if obj["name"] not in extent_visible_names]
        visible_counts.append(len(visible))
        extent_visible_counts.append(len(extent_visible))
        if not missing and objects:
            all_visible_frames.append(int(physics_frames[idx]["frame"]))
        if not extent_missing and objects:
            all_extent_visible_frames.append(int(physics_frames[idx]["frame"]))
        if worst_frame is None or len(visible) < int(worst_frame["visible_count"]):
            worst_frame = {
                "frame": int(physics_frames[idx]["frame"]),
                "visible_count": len(visible),
                "missing": missing[:8],
            }
        if strict_worst_frame is None or len(extent_visible) < int(strict_worst_frame["visible_count"]):
            strict_worst_frame = {
                "frame": int(physics_frames[idx]["frame"]),
                "visible_count": len(extent_visible),
                "missing": extent_missing[:8],
            }

    longest = longest_consecutive_run(all_visible_frames)
    strict_longest = longest_consecutive_run(all_extent_visible_frames)
    max_visible = max(visible_counts) if visible_counts else 0
    max_extent_visible = max(extent_visible_counts) if extent_visible_counts else 0
    passed = object_count > 0 and longest >= min_contiguous
    return {
        "passed": passed,
        "gate": "establishing_full_scene_window",
        "object_count": object_count,
        "max_visible_objects": max_visible,
        "all_visible_frame_count": len(all_visible_frames),
        "longest_all_visible_run_frames": longest,
        "min_required_contiguous_frames": min_contiguous,
        "all_visible_frame_examples": all_visible_frames[:12],
        "worst_frame": worst_frame,
        "strict_extent_visible": {
            "max_visible_objects": max_extent_visible,
            "all_visible_frame_count": len(all_extent_visible_frames),
            "longest_all_visible_run_frames": strict_longest,
            "all_visible_frame_examples": all_extent_visible_frames[:12],
            "worst_frame": strict_worst_frame,
            "safe_margin": 0.92,
            "radius_scale": 0.80,
        },
        "visibility_gate_params": {
            "safe_margin": 0.96,
            "radius_scale": 0.20,
            "min_fraction": min_fraction,
        },
        "projection_model": (
            "approximate pinhole frustum using camera position/look_at/lens; "
            "the hard gate requires a continuous full-scene window where every "
            "dynamic object center plus a small radius margin is visible, while "
            "strict_extent_visible records a more conservative body-extent check"
        ),
    }


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
    camera_frames = camera_records(camera, frames)
    framing = camera_framing_audit(camera_frames, physics_frames, width, height)
    if not framing["passed"]:
        raise RuntimeError(
            f"{clip_id}: camera framing audit failed: "
            f"max_visible={framing['max_visible_objects']}/{framing['object_count']}, "
            f"longest_all_visible_run={framing['longest_all_visible_run_frames']} "
            f"< {framing['min_required_contiguous_frames']}"
        )
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
        "camera_frames": camera_frames,
        "physics_frames": physics_frames,
        "quality_audit": quality,
        "camera_framing_audit": framing,
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
    group_prefix = "pair" if len(pair_specs) == 2 else "block"
    pair_group_id = f"{group_prefix}_{pair_index:04d}_{group_spec['kind']}"
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
    try:
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
    except Exception:
        for job in jobs:
            Path(job["metadata"]).unlink(missing_ok=True)
        raise

    pair_group = {
        "group_id": pair_group_id,
        "kind": group_spec["kind"],
        "group_structure": group_spec.get("group_structure", group_spec["kind"]),
        "group_size": len(clips),
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
        "group_structures": dict(Counter(group.get("group_structure", group["tags"][0]) for group in pair_groups)),
        "group_sizes": dict(Counter(str(group.get("group_size", len(group.get("clip_ids", [])))) for group in pair_groups)),
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

    group_size = 2 if args.group_mode == "pairs" else args.block_size
    for group_index in range(args.pairs):
        group_kind = "same_camera" if group_index % 2 == 0 else "same_physics_scene"
        if args.same_camera_fraction >= 0.0:
            group_kind = "same_camera" if rng.random() < args.same_camera_fraction else "same_physics_scene"
        group_done = False
        for attempt in range(args.max_pair_attempts):
            frames = frame_count(rng, args.frames_mean, args.frames_std, args.frames_min, args.frames_max, args.frame_multiple)
            try:
                group_specs, group_spec = make_group_specs(rng, group_index, attempt, frames, group_kind, group_size)
                new_clips, new_jobs, pair_group = materialize_pair(
                    run_dir=run_dir,
                    clip_index=len(clips),
                    pair_index=group_index,
                    pair_specs=group_specs,
                    group_spec=group_spec,
                    width=args.width,
                    height=args.height,
                )
            except Exception as exc:  # noqa: BLE001 - record failed samples for audit.
                failures.append({"group_index": group_index, "attempt": attempt, "kind": group_kind, "error": str(exc)})
                continue
            clips.extend(new_clips)
            render_jobs.extend(new_jobs)
            pair_groups.append(pair_group)
            group_done = True
            break
        if not group_done:
            failure_report = {
                "run_id": args.run_id,
                "failed_group_index": group_index,
                "failed_group_kind": group_kind,
                "group_mode": args.group_mode,
                "group_size": group_size,
                "max_group_attempts": args.max_pair_attempts,
                "failure_count": len(failures),
                "recent_failures": failures[-20:],
            }
            (run_dir / "failure_report.json").write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
            raise RuntimeError(f"failed to create group {group_index} after {args.max_pair_attempts} attempts")

    coverage = summarize_counts(clips, pair_groups)
    is_block_run = args.group_mode == "shared_factor_blocks"
    manifest = {
        "project": "camera_motion_disentangle",
        "run_id": args.run_id,
        "generator": "official_kubric_batch_v2_shared_factor_sampler" if is_block_run else "official_kubric_batch_v2_pair_sampler",
        "description": (
            "Shared-factor block official-Kubric batch/review run. It samples small "
            "same-camera and same-physics blocks instead of a global camera x physics grid."
            if is_block_run
            else (
                "Pair-centric official-Kubric batch/review run. It samples random "
                "camera/physics combinations for coverage instead of rendering a "
                "full camera x physics grid."
            )
        ),
        "fps": gen.FPS,
        "resolution": [args.width, args.height],
        "seed": args.seed,
        "simulator": "official kubric.simulator.PyBullet",
        "renderer": "official kubric.renderer.Blender",
        "review_strategy": (
            "coverage-oriented shared-factor block sampling; each group contains multiple clips sharing camera or physics"
            if is_block_run
            else "coverage-oriented random pair sampling; not a full camera-by-object grid"
        ),
        "shared_factor_group_policy": {
            "group_mode": args.group_mode,
            "groups_requested": args.pairs,
            "clips_per_group": group_size,
            "same_camera_groups": "one exact camera trajectory and scene paired with multiple physics programs",
            "same_physics_scene_groups": "one exact physics simulation and scene paired with multiple camera trajectories",
            "global_grid_policy": "small local blocks are sampled for reusable supervision; the full global camera x physics cartesian product is not rendered",
        },
        "pair_contract": {
            "same_camera": "all clips in a same-camera group reuse identical camera_frames, scene_id, fps, and frames_count; physics differs",
            "same_physics_scene": "all clips in a same-physics group reuse identical physics_frames, scene_id, fps, and frames_count; camera differs",
            "length_policy": "clips inside a group have identical length; different groups sample different lengths",
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
            "Every accepted clip must pass expected-contact checks, physics "
            "plausibility audits, and camera framing audits before metadata or "
            "render jobs are accepted."
        ),
        "camera_framing_filter": {
            "gate": "establishing_full_scene_window",
            "requirement": (
                "at least one continuous segment where all dynamic object centers "
                "plus a small radius margin are inside the approximate camera frustum"
            ),
            "min_contiguous_frames": "max(8, min(18, round(frame_count * 0.08)))",
            "diagnostics": "metadata also records strict_extent_visible with a larger body-radius margin",
        },
        "camera_family_reference": CAMERA_FAMILIES,
        "physics_family_reference": PHYSICS_FAMILIES,
        "coverage_summary": coverage,
        "sample_failure_count": len(failures),
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
    parser.add_argument("--pairs", type=int, default=24, help="Number of shared-factor groups to sample; legacy name kept for existing scripts.")
    parser.add_argument("--groups", type=int, default=None, help="Alias for --pairs when thinking in shared-factor groups.")
    parser.add_argument("--group-mode", choices=["pairs", "shared_factor_blocks"], default="pairs")
    parser.add_argument("--block-size", type=int, default=4, help="Clips per group when --group-mode shared_factor_blocks.")
    parser.add_argument("--width", type=int, default=gen.WIDTH)
    parser.add_argument("--height", type=int, default=gen.HEIGHT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frames-mean", type=float, default=108.0)
    parser.add_argument("--frames-std", type=float, default=24.0)
    parser.add_argument("--frames-min", type=int, default=72)
    parser.add_argument("--frames-max", type=int, default=168)
    parser.add_argument("--frame-multiple", type=int, default=12)
    parser.add_argument("--camera-families", default="", help="Comma-separated camera family ids to sample; defaults to the full pool.")
    parser.add_argument("--physics-families", default="", help="Comma-separated physics family ids to sample; defaults to the full pool.")
    parser.add_argument("--same-camera-fraction", type=float, default=-1.0)
    parser.add_argument("--max-pair-attempts", type=int, default=64)
    parser.add_argument("--progress-watch-interval", type=float, default=60.0)
    parser.add_argument("--resource-watch-interval", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if args.groups is not None:
        args.pairs = args.groups
    if args.pairs <= 0:
        raise SystemExit("--pairs/--groups must be positive")
    if args.block_size < 2:
        raise SystemExit("--block-size must be at least 2")
    if args.frames_min <= 1 or args.frames_max < args.frames_min:
        raise SystemExit("invalid frame bounds")
    if args.frame_multiple <= 0:
        raise SystemExit("--frame-multiple must be positive")

    global CAMERA_FAMILIES, PHYSICS_FAMILIES
    if args.camera_families:
        requested = [item.strip() for item in args.camera_families.split(",") if item.strip()]
        missing = [item for item in requested if item not in CAMERA_FAMILIES]
        if missing:
            raise SystemExit(f"unknown camera families: {missing}")
        CAMERA_FAMILIES = requested
    if args.physics_families:
        requested = [item.strip() for item in args.physics_families.split(",") if item.strip()]
        missing = [item for item in requested if item not in PHYSICS_FAMILIES]
        if missing:
            raise SystemExit(f"unknown physics families: {missing}")
        PHYSICS_FAMILIES = requested

    jobs_path = write_run(args)
    run_dir = args.run_root / args.run_id
    write_progress(run_dir, Path(sys.executable))
    if not args.no_render:
        progress_watcher: subprocess.Popen[bytes] | None = None
        resource_watcher: subprocess.Popen[bytes] | None = None
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
        if args.resource_watch_interval > 0:
            resource_watcher = subprocess.Popen(
                [
                    sys.executable,
                    str(gen.PROJECT_ROOT / "scripts" / "watch_run_resources.py"),
                    "--run-dir",
                    str(run_dir),
                    "--interval",
                    str(args.resource_watch_interval),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        try:
            gen.render_with_blender(args.blender_bin, jobs_path, args.kubric_site_packages)
            write_progress(run_dir, Path(sys.executable))
            gen.encode_videos(jobs_path)
            write_progress(run_dir, Path(sys.executable))
        finally:
            if resource_watcher is not None and resource_watcher.poll() is None:
                resource_watcher.terminate()
                try:
                    resource_watcher.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    resource_watcher.kill()
                    resource_watcher.wait()
            if progress_watcher is not None and progress_watcher.poll() is None:
                progress_watcher.terminate()
                try:
                    progress_watcher.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    progress_watcher.kill()
                    progress_watcher.wait()
            write_progress(run_dir, Path(sys.executable))
            subprocess.run(
                [
                    sys.executable,
                    str(gen.PROJECT_ROOT / "scripts" / "watch_run_resources.py"),
                    "--run-dir",
                    str(run_dir),
                    "--once",
                ],
                check=False,
            )
    else:
        write_progress(run_dir, Path(sys.executable))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(
        f"Wrote {len(manifest['clips'])} clips and {len(manifest['pair_groups'])} groups "
        f"to {run_dir}"
    )


if __name__ == "__main__":
    main()
