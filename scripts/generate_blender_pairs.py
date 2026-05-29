#!/usr/bin/env python3
"""Generate paired 3D preview videos with Blender only.

Run with:
  /workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender --background --python scripts/generate_blender_pairs.py

This is not Kubric/PyBullet yet. It uses Blender rendering plus deterministic
procedural object trajectories to validate the paired data contract in 3D.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "site" / "assets" / "runs" / "blender_pair_v0"
FRAMES = 24
FPS = 24
WIDTH = 320
HEIGHT = 240

CAMERA_IDS = ["cam_orbit_left", "cam_dolly_in"]
PHYSICS_IDS = ["phys_bounce", "phys_roll"]
CLIPS = [
    ("A_cam_orbit_phys_bounce", "cam_orbit_left", "phys_bounce"),
    ("B_cam_orbit_phys_roll", "cam_orbit_left", "phys_roll"),
    ("C_cam_dolly_phys_bounce", "cam_dolly_in", "phys_bounce"),
]


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
    bpy.context.scene.cycles.samples = 4
    
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_world() -> None:
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "ground"
    material = bpy.data.materials.new("mat_ground")
    material.diffuse_color = (0.72, 0.76, 0.76, 1.0)
    plane.data.materials.append(material)

    for x in [-3, -2, -1, 0, 1, 2, 3]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, 2.8, 0.01))
        marker = bpy.context.object
        marker.name = f"grid_x_{x}"
        marker.dimensions = (0.025, 5.6, 0.02)
        marker.data.materials.append(material)
    for y in [-2, -1, 0, 1, 2]:
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, y, 0.012))
        marker = bpy.context.object
        marker.name = f"grid_y_{y}"
        marker.dimensions = (6.0, 0.025, 0.02)
        marker.data.materials.append(material)

    bpy.ops.object.light_add(type="AREA", location=(0, -3.5, 5.0))
    light = bpy.context.object
    light.name = "key_light"
    light.data.energy = 500
    light.data.size = 5


def add_actor(physics_id: str) -> bpy.types.Object:
    if physics_id == "phys_bounce":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.32)
        obj = bpy.context.object
        mat = bpy.data.materials.new("mat_bounce")
        mat.diffuse_color = (0.86, 0.24, 0.18, 1.0)
    elif physics_id == "phys_roll":
        bpy.ops.mesh.primitive_cube_add(size=0.56)
        obj = bpy.context.object
        mat = bpy.data.materials.new("mat_roll")
        mat.diffuse_color = (0.12, 0.52, 0.46, 1.0)
    else:
        raise ValueError(physics_id)
    obj.name = physics_id
    obj.data.materials.append(mat)
    return obj


def object_pose(physics_id: str, t: float) -> tuple[Vector, tuple[float, float, float]]:
    if physics_id == "phys_bounce":
        x = -1.7 + 3.4 * t
        z = 0.32 + 1.2 * abs(math.sin(2.0 * math.pi * t))
        return Vector((x, 0.0, z)), (0.0, 0.0, 0.0)
    if physics_id == "phys_roll":
        x = -1.7 + 3.4 * t
        return Vector((x, 0.0, 0.28)), (0.0, 6.0 * math.pi * t, 0.0)
    raise ValueError(physics_id)


def camera_pose(camera_id: str, t: float) -> Vector:
    if camera_id == "cam_orbit_left":
        angle = math.radians(-32 + 64 * t)
        radius = 5.2
        return Vector((radius * math.sin(angle), -radius * math.cos(angle), 2.5))
    if camera_id == "cam_dolly_in":
        return Vector((0.8 - 0.8 * t, -6.4 + 2.6 * t, 2.4 - 0.25 * t))
    raise ValueError(camera_id)


def animate_clip(clip_id: str, camera_id: str, physics_id: str) -> dict[str, object]:
    reset_scene()
    add_world()
    actor = add_actor(physics_id)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    frames = []
    for frame in range(1, FRAMES + 1):
        t = (frame - 1) / max(1, FRAMES - 1)
        actor.location, actor.rotation_euler = object_pose(physics_id, t)
        actor.keyframe_insert("location", frame=frame)
        actor.keyframe_insert("rotation_euler", frame=frame)
        camera.location = camera_pose(camera_id, t)
        look_at(camera, Vector((0.0, 0.0, 0.6)))
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_euler", frame=frame)
        frames.append(
            {
                "frame": frame - 1,
                "t": t,
                "camera_position": list(camera.location),
                "object_position": list(actor.location),
                "object_rotation_euler": list(actor.rotation_euler),
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
    pattern = str(frame_dir / "frame_%04d.png")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(FPS), "-i", pattern, "-pix_fmt", "yuv420p", str(video_path)],
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
                "video": str(video_rel),
                "frames": frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "clip_id": clip_id,
        "camera_id": camera_id,
        "physics_id": physics_id,
        "pair_groups": [
            group
            for group, ids in {
                "same_camera_diff_physics": ["A_cam_orbit_phys_bounce", "B_cam_orbit_phys_roll"],
                "diff_camera_same_physics": ["A_cam_orbit_phys_bounce", "C_cam_dolly_phys_bounce"],
            }.items()
            if clip_id in ids
        ],
        "video": str(video_rel),
        "metadata": str(metadata_rel),
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    clips = [animate_clip(*spec) for spec in CLIPS]
    manifest = {
        "project": "camera_motion_disentangle",
        "run_id": RUN_DIR.name,
        "generator": "blender_preview",
        "description": "Blender-only 3D preview; not Kubric/PyBullet.",
        "clips": clips,
        "pair_groups": [
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
        ],
    }
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    index_path = RUN_DIR.parent / "index.json"
    runs = [{"run_id": "mock_pair_v0", "manifest": "mock_pair_v0/manifest.json"}, {"run_id": RUN_DIR.name, "manifest": f"{RUN_DIR.name}/manifest.json"}]
    index_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print(f"Wrote Blender paired preview to {RUN_DIR}")


if __name__ == "__main__":
    sys.exit(main())
