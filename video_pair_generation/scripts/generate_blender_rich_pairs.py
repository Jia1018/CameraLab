#!/usr/bin/env python3
"""Generate a richer Blender-only paired dataset preview.

Run with:
  /workspace/writeable/blender-3.6.5-linux-x64/blender --background --python scripts/generate_blender_rich_pairs.py

This remains Blender-only procedural motion, not Kubric/PyBullet. It is meant to
exercise the paired-data contract with richer camera and object motion before the
Kubric runtime is installed.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from mathutils import Vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "site" / "assets" / "runs" / "blender_rich_v0"
FRAMES = 32
FPS = 24
WIDTH = 640
HEIGHT = 480
SAMPLES = 2

CLIPS = [
    ("A_combo_bounce_diag", "cam_combo_crane_orbit", "phys_bounce_diag"),
    ("B_combo_two_cross", "cam_combo_crane_orbit", "phys_two_cross"),
    ("C_orbit_two_cross", "cam_orbit_fast", "phys_two_cross"),
    ("D_dolly_two_cross", "cam_dolly_in_tilt", "phys_two_cross"),
    ("E_truck_roll_reverse", "cam_truck_right", "phys_roll_reverse"),
    ("F_truck_multi_swirl", "cam_truck_right", "phys_multi_swirl"),
]

PAIR_GROUPS = [
    {
        "group_id": "same_camera_combo_diff_physics",
        "title": "Same combo camera, different physics",
        "controlled_factor": "camera_id",
        "varied_factor": "physics_id",
        "clip_ids": ["A_combo_bounce_diag", "B_combo_two_cross"],
    },
    {
        "group_id": "different_camera_same_multibody_physics",
        "title": "Different cameras, same multi-object physics",
        "controlled_factor": "physics_id",
        "varied_factor": "camera_id",
        "clip_ids": ["B_combo_two_cross", "C_orbit_two_cross", "D_dolly_two_cross"],
    },
    {
        "group_id": "same_truck_camera_single_vs_multi",
        "title": "Same truck camera, single vs multi-object physics",
        "controlled_factor": "camera_id",
        "varied_factor": "physics_id",
        "clip_ids": ["E_truck_roll_reverse", "F_truck_multi_swirl"],
    },
]

MATERIALS = {
    "red": (0.86, 0.24, 0.18, 1.0),
    "teal": (0.10, 0.52, 0.46, 1.0),
    "gold": (0.95, 0.62, 0.18, 1.0),
    "blue": (0.18, 0.35, 0.75, 1.0),
    "violet": (0.48, 0.28, 0.74, 1.0),
    "ground": (0.68, 0.70, 0.66, 1.0),
    "wall": (0.78, 0.80, 0.78, 1.0),
    "marker": (0.28, 0.30, 0.30, 1.0),
}


def mat(name: str) -> bpy.types.Material:
    material = bpy.data.materials.new(f"mat_{name}")
    material.diffuse_color = MATERIALS[name]
    return material


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAMES
    bpy.context.scene.render.fps = FPS
    bpy.context.scene.render.resolution_x = WIDTH
    bpy.context.scene.render.resolution_y = HEIGHT
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.samples = SAMPLES
    bpy.context.scene.world.color = (0.78, 0.82, 0.86)
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_world() -> None:
    ground_mat = mat("ground")
    wall_mat = mat("wall")
    marker_mat = mat("marker")

    bpy.ops.mesh.primitive_plane_add(size=9.0, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "matte_ground_no_grid"
    ground.data.materials.append(ground_mat)

    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 3.0, 1.0))
    wall = bpy.context.object
    wall.name = "soft_back_wall"
    wall.dimensions = (8.5, 0.08, 2.0)
    wall.data.materials.append(wall_mat)

    # Sparse low-profile markers give camera-motion reference without a grid-like floor.
    for x, y in [(-3.0, -2.2), (-1.0, 2.2), (1.4, -2.0), (3.1, 1.5)]:
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.06, depth=0.018, location=(x, y, 0.012))
        marker = bpy.context.object
        marker.name = "floor_reference_dot"
        marker.data.materials.append(marker_mat)

    bpy.ops.object.light_add(type="AREA", location=(0, -4.0, 5.0))
    key = bpy.context.object
    key.name = "key_light"
    key.data.energy = 650
    key.data.size = 5.0

    bpy.ops.object.light_add(type="POINT", location=(-3.5, 2.5, 3.0))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 80


def add_sphere(name: str, color: str, radius: float = 0.28) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat(color))
    return obj


def add_cube(name: str, color: str, size: float = 0.48) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=size)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat(color))
    return obj


def create_actors(physics_id: str) -> list[bpy.types.Object]:
    if physics_id == "phys_bounce_diag":
        return [add_sphere("diag_fast_sphere", "red", 0.30)]
    if physics_id == "phys_two_cross":
        return [add_sphere("cross_left_to_right", "red", 0.25), add_sphere("cross_right_to_left", "blue", 0.25)]
    if physics_id == "phys_roll_reverse":
        return [add_cube("reverse_roll_box", "teal", 0.52)]
    if physics_id == "phys_multi_swirl":
        return [add_sphere("swirl_sphere", "gold", 0.23), add_cube("swirl_cube", "teal", 0.42), add_sphere("vertical_bounce", "violet", 0.22)]
    raise ValueError(physics_id)


def actor_pose(physics_id: str, actor_index: int, t: float) -> tuple[Vector, tuple[float, float, float]]:
    if physics_id == "phys_bounce_diag":
        x = -2.5 + 5.0 * t
        y = -1.6 + 3.2 * t
        z = 0.30 + 1.15 * abs(math.sin(2.4 * math.pi * t))
        return Vector((x, y, z)), (0.0, 5.0 * math.pi * t, 0.0)

    if physics_id == "phys_two_cross":
        if actor_index == 0:
            x = -2.4 + 4.8 * t
            y = -1.0 + 0.35 * math.sin(2.0 * math.pi * t)
            z = 0.25 + 0.65 * abs(math.sin(2.0 * math.pi * t))
        else:
            x = 2.4 - 4.8 * t
            y = 1.0 - 0.35 * math.sin(2.0 * math.pi * t)
            z = 0.25 + 0.45 * abs(math.sin(2.0 * math.pi * (t + 0.18)))
        return Vector((x, y, z)), (0.0, 0.0, 0.0)

    if physics_id == "phys_roll_reverse":
        x = 2.5 - 5.0 * t
        y = -0.85 + 0.5 * math.sin(math.pi * t)
        return Vector((x, y, 0.26)), (0.0, -7.0 * math.pi * t, 0.45 * math.sin(2.0 * math.pi * t))

    if physics_id == "phys_multi_swirl":
        angle = 2.0 * math.pi * t
        if actor_index == 0:
            return Vector((1.15 * math.cos(angle), 1.15 * math.sin(angle), 0.26)), (0.0, 0.0, 0.0)
        if actor_index == 1:
            return Vector((-1.3 + 2.6 * t, 0.9 * math.cos(angle), 0.24)), (0.4 * angle, 2.6 * angle, 0.0)
        return Vector((0.7 * math.sin(angle), -1.4 + 2.8 * t, 0.22 + 0.8 * abs(math.sin(3.0 * math.pi * t)))), (0.0, 0.0, 0.0)

    raise ValueError(physics_id)


def camera_position(camera_id: str, t: float) -> Vector:
    if camera_id == "cam_combo_crane_orbit":
        angle = math.radians(-38 + 76 * t)
        radius = 5.8 - 0.8 * t
        return Vector((radius * math.sin(angle), -radius * math.cos(angle), 2.2 + 1.1 * t))
    if camera_id == "cam_orbit_fast":
        angle = math.radians(-58 + 116 * t)
        return Vector((5.2 * math.sin(angle), -5.2 * math.cos(angle), 2.55))
    if camera_id == "cam_dolly_in_tilt":
        return Vector((-0.7 + 1.0 * t, -6.5 + 3.0 * t, 2.0 + 0.35 * math.sin(math.pi * t)))
    if camera_id == "cam_truck_right":
        return Vector((-2.4 + 4.8 * t, -5.1, 2.35))
    raise ValueError(camera_id)


def look_target(camera_id: str, t: float) -> Vector:
    if camera_id == "cam_truck_right":
        return Vector((-0.5 + 1.0 * t, 0.0, 0.55))
    if camera_id == "cam_dolly_in_tilt":
        return Vector((0.0, 0.15, 0.45 + 0.35 * t))
    return Vector((0.0, 0.0, 0.65))


def animate_clip(clip_id: str, camera_id: str, physics_id: str) -> dict[str, object]:
    reset_scene()
    add_world()
    actors = create_actors(physics_id)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = camera_id
    camera.data.lens = 32
    bpy.context.scene.camera = camera

    frame_records = []
    for frame in range(1, FRAMES + 1):
        t = (frame - 1) / max(1, FRAMES - 1)
        objects = []
        for idx, actor in enumerate(actors):
            actor.location, actor.rotation_euler = actor_pose(physics_id, idx, t)
            actor.keyframe_insert("location", frame=frame)
            actor.keyframe_insert("rotation_euler", frame=frame)
            objects.append(
                {
                    "name": actor.name,
                    "position": [round(v, 5) for v in actor.location],
                    "rotation_euler": [round(v, 5) for v in actor.rotation_euler],
                }
            )
        camera.location = camera_position(camera_id, t)
        target = look_target(camera_id, t)
        look_at(camera, target)
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
        frame_records.append(
            {
                "frame": frame - 1,
                "t": round(t, 5),
                "camera": {
                    "position": [round(v, 5) for v in camera.location],
                    "look_at": [round(v, 5) for v in target],
                    "rotation_euler": [round(v, 5) for v in camera.rotation_euler],
                },
                "objects": objects,
            }
        )

    frame_dir = RUN_DIR / "frames" / clip_id
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(frame_dir / "frame_")
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(animation=True, write_still=False)

    video_rel = Path("videos") / f"{clip_id}.mp4"
    video_path = RUN_DIR / video_rel
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    metadata_rel = Path("metadata") / f"{clip_id}.json"
    metadata_path = RUN_DIR / metadata_rel
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "camera_id": camera_id,
                "physics_id": physics_id,
                "resolution": [WIDTH, HEIGHT],
                "fps": FPS,
                "frames": frame_records,
                "video": str(video_rel),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "physics_id": physics_id,
        "pair_groups": [group["group_id"] for group in PAIR_GROUPS if clip_id in group["clip_ids"]],
        "video": str(video_rel),
        "metadata": str(metadata_rel),
    }


def update_index(run_id: str) -> None:
    index_path = RUN_DIR.parent / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        runs = [run for run in index.get("runs", []) if run.get("run_id") != run_id]
    else:
        runs = []
    runs.append({"run_id": run_id, "manifest": f"{run_id}/manifest.json"})
    index_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    clips = [animate_clip(*spec) for spec in CLIPS]
    manifest = {
        "project": "camera_motion_disentangle",
        "run_id": RUN_DIR.name,
        "generator": "blender_rich_preview",
        "description": "Higher-resolution Blender-only paired preview with single/multi-object procedural motion and varied camera direction/speed.",
        "resolution": [WIDTH, HEIGHT],
        "fps": FPS,
        "frames": FRAMES,
        "cycles_samples": SAMPLES,
        "clips": clips,
        "pair_groups": PAIR_GROUPS,
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_index(RUN_DIR.name)
    print(f"Wrote richer Blender paired preview to {RUN_DIR}")


if __name__ == "__main__":
    sys.exit(main())
