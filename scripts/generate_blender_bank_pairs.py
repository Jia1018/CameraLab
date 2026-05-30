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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

CAMERA_SPECS = [
    {"id": "cam_static", "primitives": ["Static"], "speed": "none"},
    {"id": "cam_dolly_in", "primitives": ["Dolly_In"], "speed": "slow"},
    {"id": "cam_dolly_out_fast", "primitives": ["Dolly_Out"], "speed": "fast"},
    {"id": "cam_truck_left_pan_right", "primitives": ["Truck_Left", "Pan_Right"], "speed": "medium"},
    {"id": "cam_truck_right_roll", "primitives": ["Truck_Right", "Roll_Clockwise"], "speed": "medium"},
    {"id": "cam_crane_up_tilt_down", "primitives": ["Pedestal_Up", "Tilt_Down"], "speed": "slow"},
    {"id": "cam_pedestal_down_zoom_in", "primitives": ["Pedestal_Down", "Zoom_In"], "speed": "ease_in"},
    {"id": "cam_orbit_combo", "primitives": ["Truck_Right", "Pan_Left", "Dolly_In"], "speed": "ease_in_out"},
    {"id": "cam_wobble_roll_zoom", "primitives": ["Pan_Left", "Pan_Right", "Roll_Counterclockwise", "Zoom_Out"], "speed": "variable"},
    {"id": "cam_out_of_frame_push", "primitives": ["Dolly_In", "Tilt_Up"], "speed": "fast"},
]

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

BACKGROUNDS = ["plain_wall", "blocks", "pillars", "mixed_shapes", "corridor", "cluttered_studio"]
COLORS = [(0.86, 0.24, 0.18, 1), (0.12, 0.52, 0.46, 1), (0.18, 0.35, 0.75, 1), (0.95, 0.62, 0.18, 1), (0.48, 0.28, 0.74, 1)]


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    camera: dict
    physics: dict
    background: str
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


def add_world(background: str, rng: random.Random) -> None:
    add_cube("ground", (0, 0, -0.03), (9, 9, 0.06), (0.66, 0.68, 0.64, 1))
    add_cube("back_wall", (0, 3.35, 1.2), (9, 0.08, 2.4), (0.78, 0.79, 0.76, 1))
    add_cube("left_wall", (-4.25, 0, 1.1), (0.08, 6.5, 2.2), (0.70, 0.72, 0.72, 1))

    if background in {"blocks", "mixed_shapes", "cluttered_studio"}:
        for i in range(4 if background != "cluttered_studio" else 8):
            x = rng.uniform(-3.0, 3.0)
            y = rng.uniform(-0.6, 2.7)
            h = rng.uniform(0.25, 0.9)
            add_cube(f"static_block_{i}", (x, y, h / 2), (rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.7), h), (0.35 + 0.08 * i, 0.36, 0.39, 1))
    if background in {"pillars", "mixed_shapes", "corridor"}:
        for i, x in enumerate([-2.7, -1.2, 1.2, 2.7]):
            add_cylinder(f"pillar_{i}", (x, 2.0, 0.65), 0.13, 1.3, (0.42, 0.44, 0.46, 1))
    if background in {"mixed_shapes", "cluttered_studio"}:
        for i in range(3):
            add_sphere(f"static_sphere_{i}", (rng.uniform(-3, 3), rng.uniform(0.8, 2.5), 0.18), 0.18, COLORS[(i + 2) % len(COLORS)])

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
        if idx == 0:
            x = -2.8 + 4.5 * min(t, 0.74)
            if t > 0.74:
                x = 0.53 - 1.1 * (t - 0.74)
            return Vector((x, -0.2, 0.24 + 0.15 * abs(math.sin(6 * math.pi * t)))), (0, 0, 0)
        return Vector((0.5 + 0.25 * max(0, t - 0.74), -0.2, 0.3)), (0, 0, 0.5 * max(0, t - 0.74))
    if kind == "static_with_dynamic_bg":
        if idx == 0:
            return Vector((0, -0.7, 0.28)), (0, 0, 0)
        if idx == 1:
            return Vector((-2.2 + 4.4 * t, 1.4, 0.25)), (0, 4 * math.pi * t, 0)
        return Vector((1.5, 1.2 - 1.8 * t, 0.22 + 0.55 * abs(math.sin(3 * math.pi * t)))), (0, 0, 0)
    raise ValueError(kind)


def camera_state(camera: dict, t: float) -> tuple[Vector, Vector, float, float]:
    cid = camera["id"]
    u = ease(t, camera["speed"])
    pos = Vector((0, -5.6, 2.1))
    target = Vector((0, 0, 0.55))
    roll = 0.0
    lens = 34.0
    if cid == "cam_static":
        pass
    elif cid == "cam_dolly_in":
        pos.y = -6.2 + 2.2 * u
    elif cid == "cam_dolly_out_fast":
        pos.y = -3.6 - 3.0 * u
    elif cid == "cam_truck_left_pan_right":
        pos.x = 2.2 - 4.4 * u
        target.x = -0.8 + 1.6 * u
    elif cid == "cam_truck_right_roll":
        pos.x = -2.2 + 4.4 * u
        roll = math.radians(16 * math.sin(math.pi * u))
    elif cid == "cam_crane_up_tilt_down":
        pos.z = 1.5 + 1.7 * u
        target.z = 0.9 - 0.55 * u
    elif cid == "cam_pedestal_down_zoom_in":
        pos.z = 3.1 - 1.4 * u
        lens = 30 + 22 * u
    elif cid == "cam_orbit_combo":
        angle = math.radians(-45 + 95 * u)
        radius = 5.4 - 1.2 * u
        pos = Vector((radius * math.sin(angle), -radius * math.cos(angle), 2.0 + 0.5 * math.sin(math.pi * u)))
        target.x = 0.35 * math.sin(2 * math.pi * u)
    elif cid == "cam_wobble_roll_zoom":
        target.x = 0.8 * math.sin(2 * math.pi * u)
        target.z = 0.55 + 0.25 * math.sin(3 * math.pi * u)
        roll = math.radians(-12 * math.sin(2 * math.pi * u))
        lens = 42 - 12 * u
    elif cid == "cam_out_of_frame_push":
        pos = Vector((0.4, -5.2 + 3.0 * u, 1.8 + 0.8 * u))
        target = Vector((0.3 + 1.2 * u, 0.2 + 1.4 * u, 0.7 + 0.4 * u))
    else:
        raise ValueError(cid)
    return pos, target, roll, lens


def build_clip_specs(max_clips: int, seed: int) -> list[ClipSpec]:
    rng = random.Random(seed)
    specs = []
    forced = [
        ("confusing_cam_push_static", "cam_dolly_in", "phys_static_center"),
        ("confusing_static_obj_back", "cam_static", "phys_confusing_move_back"),
        ("outframe_cam_push_obj_cross", "cam_out_of_frame_push", "phys_out_of_frame"),
    ]
    for clip_id, cam_id, phys_id in forced:
        specs.append(ClipSpec(clip_id, next(c for c in CAMERA_SPECS if c["id"] == cam_id), next(p for p in PHYSICS_SPECS if p["id"] == phys_id), rng.choice(BACKGROUNDS), rng.choice([16, 20, 24, 28])))
    for cam in CAMERA_SPECS:
        for phys in PHYSICS_SPECS:
            if len(specs) >= max_clips:
                return specs
            clip_id = f"clip_{len(specs):03d}_{cam['id']}_{phys['id']}"
            frames = rng.choice([16, 20, 24, 28, 32])
            specs.append(ClipSpec(clip_id, cam, phys, rng.choice(BACKGROUNDS), frames))
    return specs


def render_clip(spec: ClipSpec, run_dir: Path) -> dict[str, object]:
    rng = random.Random(f"{SEED}_{spec.clip_id}")
    reset_scene(spec.frames)
    add_world(spec.background, rng)
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
    metadata = {"clip_id": spec.clip_id, "camera_id": spec.camera["id"], "physics_id": spec.physics["id"], "background": spec.background, "fps": FPS, "frames_count": spec.frames, "resolution": [WIDTH, HEIGHT], "camera_primitives": spec.camera["primitives"], "camera_speed": spec.camera["speed"], "physics_kind": spec.physics["kind"], "physics_objects": spec.physics["objects"], "physics_speed": spec.physics["speed"], "video": str(video_rel), "frames": records}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"clip_id": spec.clip_id, "camera_id": spec.camera["id"], "physics_id": spec.physics["id"], "background": spec.background, "camera_primitives": spec.camera["primitives"], "pair_groups": [], "video": str(video_rel), "metadata": str(metadata_rel)}


def make_pairs(clips: list[dict[str, object]], target_pairs: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed + 99)
    by_cam = {}
    by_phys = {}
    for clip in clips:
        by_cam.setdefault(clip["camera_id"], []).append(clip)
        by_phys.setdefault(clip["physics_id"], []).append(clip)
    pairs = []
    def add_pair(kind, title, a, b, controlled, varied, tags):
        if a["clip_id"] == b["clip_id"]:
            return
        group_id = f"pair_{len(pairs):03d}_{kind}"
        pairs.append({"group_id": group_id, "title": title, "controlled_factor": controlled, "varied_factor": varied, "clip_ids": [a["clip_id"], b["clip_id"]], "tags": tags})
        a["pair_groups"].append(group_id)
        b["pair_groups"].append(group_id)

    clip_by_id = {clip["clip_id"]: clip for clip in clips}
    if "confusing_cam_push_static" in clip_by_id and "confusing_static_obj_back" in clip_by_id:
        add_pair("confusing", "Confusing: camera pushes in vs object moves away", clip_by_id["confusing_cam_push_static"], clip_by_id["confusing_static_obj_back"], "apparent_scale_change", "camera_vs_object_source", ["confusing", "dolly_vs_object_motion"])
    for cam, group in by_cam.items():
        for _ in range(10):
            if len(group) >= 2 and len(pairs) < target_pairs:
                a, b = rng.sample(group, 2)
                add_pair("same_camera", f"Same camera {cam}, different physics/background", a, b, "camera_id", "physics_id/background", ["same_camera", "different_physics"])
    for phys, group in by_phys.items():
        for _ in range(10):
            if len(group) >= 2 and len(pairs) < target_pairs:
                a, b = rng.sample(group, 2)
                add_pair("same_physics", f"Same physics {phys}, different camera/background", a, b, "physics_id", "camera_id/background", ["same_physics", "different_camera"])
    while len(pairs) < target_pairs:
        a, b = rng.sample(clips, 2)
        tags = ["mixed_combo"]
        if a["camera_id"] != b["camera_id"]:
            tags.append("different_camera")
        if a["physics_id"] != b["physics_id"]:
            tags.append("different_physics")
        add_pair("mixed", "Mixed camera/physics/background comparison", a, b, "none", "camera_id/physics_id/background", tags)
    return pairs[:target_pairs]


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
    parser.add_argument("--clips", type=int, default=40)
    parser.add_argument("--pairs", type=int, default=120)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    run_dir = RUN_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = build_clip_specs(args.clips, SEED)
    clips = [render_clip(spec, run_dir) for spec in specs]
    pair_groups = make_pairs(clips, args.pairs, SEED)
    manifest = {"project": "camera_motion_disentangle", "run_id": args.run_id, "generator": "blender_combinatorial_bank", "description": "Broad Blender-only preview bank with many camera primitives, object motions, backgrounds, confusing cases, out-of-frame cases, variable speeds, and 100-200 pair relationships.", "blender_version": BLENDER_VERSION, "resolution": [WIDTH, HEIGHT], "fps": FPS, "cycles_samples": SAMPLES, "camera_primitives_reference": CAMERA_PRIMITIVES, "clips": clips, "pair_groups": pair_groups}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_index(args.run_id)
    print(f"Wrote {len(clips)} clips and {len(pair_groups)} pairs to {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
