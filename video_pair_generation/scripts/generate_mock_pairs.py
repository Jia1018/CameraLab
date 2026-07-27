#!/usr/bin/env python3
"""Generate lightweight paired preview videos for disentanglement experiments.

This script does not use Kubric. It produces simple 2D videos with explicit
camera and physical motion metadata so the pair protocol and review UI can be
tested before Blender/Kubric is available.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "site" / "assets" / "runs"


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    name: str
    kind: str


@dataclass(frozen=True)
class PhysicsSpec:
    physics_id: str
    name: str
    kind: str


CAMERAS = {
    "cam_orbit_left": CameraSpec("cam_orbit_left", "Orbit left", "orbit_left"),
    "cam_dolly_in": CameraSpec("cam_dolly_in", "Dolly in", "dolly_in"),
}

PHYSICS = {
    "phys_bounce": PhysicsSpec("phys_bounce", "Ball bounce", "bounce"),
    "phys_roll": PhysicsSpec("phys_roll", "Box roll", "roll"),
}

CLIPS = [
    {
        "clip_id": "A_cam_orbit_phys_bounce",
        "camera_id": "cam_orbit_left",
        "physics_id": "phys_bounce",
        "pair_groups": ["same_camera_diff_physics", "diff_camera_same_physics"],
    },
    {
        "clip_id": "B_cam_orbit_phys_roll",
        "camera_id": "cam_orbit_left",
        "physics_id": "phys_roll",
        "pair_groups": ["same_camera_diff_physics"],
    },
    {
        "clip_id": "C_cam_dolly_phys_bounce",
        "camera_id": "cam_dolly_in",
        "physics_id": "phys_bounce",
        "pair_groups": ["diff_camera_same_physics"],
    },
]

PAIR_GROUPS = [
    {
        "group_id": "same_camera_diff_physics",
        "title": "Same camera, different physics",
        "controlled_factor": "camera_id",
        "varied_factor": "physics_id",
        "clip_ids": ["A_cam_orbit_phys_bounce", "B_cam_orbit_phys_roll"],
    },
    {
        "group_id": "diff_camera_same_physics",
        "title": "Different camera, same physics",
        "controlled_factor": "physics_id",
        "varied_factor": "camera_id",
        "clip_ids": ["A_cam_orbit_phys_bounce", "C_cam_dolly_phys_bounce"],
    },
]


def camera_transform(spec: CameraSpec, t: float) -> dict[str, float]:
    if spec.kind == "orbit_left":
        return {
            "pan_x": -72.0 * math.sin((t - 0.5) * math.pi),
            "pan_y": 18.0 * math.cos(t * math.pi * 2.0),
            "zoom": 1.0 + 0.05 * math.sin(t * math.pi),
            "roll": math.radians(2.0 * math.sin(t * math.pi * 2.0)),
        }
    if spec.kind == "dolly_in":
        return {
            "pan_x": 18.0 * math.sin(t * math.pi * 2.0),
            "pan_y": -28.0 * t,
            "zoom": 0.82 + 0.45 * t,
            "roll": 0.0,
        }
    raise ValueError(f"unknown camera kind: {spec.kind}")


def object_state(spec: PhysicsSpec, t: float) -> dict[str, float | str]:
    if spec.kind == "bounce":
        x = -150.0 + 300.0 * t
        y = -62.0 - 122.0 * abs(math.sin(t * math.pi * 2.0))
        return {"shape": "circle", "x": x, "y": y, "theta": 0.0, "radius": 28.0}
    if spec.kind == "roll":
        x = -165.0 + 310.0 * t
        y = -54.0
        return {
            "shape": "box",
            "x": x,
            "y": y,
            "theta": t * math.pi * 4.0,
            "half_size": 26.0,
        }
    raise ValueError(f"unknown physics kind: {spec.kind}")


def world_to_screen(x: float, y: float, cam: dict[str, float], width: int, height: int) -> tuple[float, float]:
    x = (x + cam["pan_x"]) * cam["zoom"]
    y = (y + cam["pan_y"]) * cam["zoom"]
    roll = cam["roll"]
    xr = x * math.cos(roll) - y * math.sin(roll)
    yr = x * math.sin(roll) + y * math.cos(roll)
    return width / 2.0 + xr, height / 2.0 + yr


def draw_background(draw: ImageDraw.ImageDraw, cam: dict[str, float], width: int, height: int) -> None:
    draw.rectangle((0, 0, width, height), fill=(238, 242, 244))
    for gx in range(-360, 361, 60):
        p0 = world_to_screen(gx, -210, cam, width, height)
        p1 = world_to_screen(gx, 170, cam, width, height)
        draw.line((*p0, *p1), fill=(207, 216, 220), width=1)
    for gy in range(-180, 181, 60):
        p0 = world_to_screen(-360, gy, cam, width, height)
        p1 = world_to_screen(360, gy, cam, width, height)
        draw.line((*p0, *p1), fill=(207, 216, 220), width=1)
    p0 = world_to_screen(-260, -20, cam, width, height)
    p1 = world_to_screen(260, -20, cam, width, height)
    draw.line((*p0, *p1), fill=(76, 86, 92), width=max(3, int(5 * cam["zoom"])))


def draw_object(draw: ImageDraw.ImageDraw, state: dict[str, float | str], cam: dict[str, float], width: int, height: int) -> None:
    cx, cy = world_to_screen(float(state["x"]), float(state["y"]), cam, width, height)
    if state["shape"] == "circle":
        r = float(state["radius"]) * cam["zoom"]
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(221, 87, 70), outline=(110, 35, 30), width=3)
        draw.line((cx, cy, cx + r * 0.85, cy), fill=(110, 35, 30), width=3)
        return

    half = float(state["half_size"]) * cam["zoom"]
    theta = float(state["theta"]) + cam["roll"]
    corners = []
    for sx, sy in [(-half, -half), (half, -half), (half, half), (-half, half)]:
        rx = sx * math.cos(theta) - sy * math.sin(theta)
        ry = sx * math.sin(theta) + sy * math.cos(theta)
        corners.append((cx + rx, cy + ry))
    draw.polygon(corners, fill=(52, 128, 117), outline=(24, 72, 68))
    draw.line((corners[0], corners[2]), fill=(24, 72, 68), width=2)


def render_clip(
    clip: dict[str, object],
    run_dir: Path,
    frames: int,
    fps: int,
    width: int,
    height: int,
) -> dict[str, object]:
    camera = CAMERAS[str(clip["camera_id"])]
    physics = PHYSICS[str(clip["physics_id"])]
    video_path = run_dir / "videos" / f"{clip['clip_id']}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {video_path}")

    frame_records = []
    for frame_idx in range(frames):
        t = frame_idx / max(1, frames - 1)
        cam = camera_transform(camera, t)
        obj = object_state(physics, t)
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        draw_background(draw, cam, width, height)
        draw_object(draw, obj, cam, width, height)
        draw.text((16, 14), str(clip["clip_id"]), fill=(32, 37, 40))
        draw.text((16, 38), f"camera={camera.camera_id}  physics={physics.physics_id}", fill=(32, 37, 40))
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        frame_records.append({"frame": frame_idx, "t": t, "camera": cam, "object": obj})

    writer.release()
    metadata_path = run_dir / "metadata" / f"{clip['clip_id']}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "clip_id": clip["clip_id"],
        "camera": camera.__dict__,
        "physics": physics.__dict__,
        "pair_groups": clip["pair_groups"],
        "video": str(video_path.relative_to(run_dir)),
        "frames": frame_records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "clip_id": clip["clip_id"],
        "camera_id": camera.camera_id,
        "physics_id": physics.physics_id,
        "pair_groups": clip["pair_groups"],
        "video": str(video_path.relative_to(run_dir)),
        "metadata": str(metadata_path.relative_to(run_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="mock_pair_v0")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    run_dir = args.run_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    clips = [render_clip(clip, run_dir, args.frames, args.fps, args.width, args.height) for clip in CLIPS]
    manifest = {
        "project": "camera_motion_disentangle",
        "run_id": args.run_id,
        "generator": "mock_preview",
        "description": "Preview videos for validating paired camera/physics disentanglement design.",
        "clips": clips,
        "pair_groups": PAIR_GROUPS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.run_root / "index.json").write_text(
        json.dumps({"runs": [{"run_id": args.run_id, "manifest": f"{args.run_id}/manifest.json"}]}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(clips)} clips to {run_dir}")


if __name__ == "__main__":
    main()
