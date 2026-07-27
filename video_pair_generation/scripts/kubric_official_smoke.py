#!/usr/bin/env python3
"""Smoke tests for the official PyPI Kubric package.

Run the physics mode with the official Kubric virtualenv. Run the render mode
with Blender and the official Kubric site-packages on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def blender_argv() -> list[str]:
    """Return arguments after Blender's optional '--' separator."""
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def patch_blender_denoiser(blender_cls: Any) -> None:
    """Make Kubric 0.1.1 compatible with Blender 3.6 denoiser enum names."""

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


def run_physics(out_dir: Path) -> None:
    import kubric as kb
    from kubric.simulator import PyBullet

    scene = kb.Scene(frame_start=0, frame_end=24, frame_rate=24, step_rate=240, resolution=(320, 240))
    ground = kb.Cube(
        name="ground",
        scale=(4.0, 4.0, 0.05),
        position=(0.0, 0.0, -0.025),
        static=True,
        friction=0.2,
        restitution=0.6,
    )
    ball = kb.Sphere(
        name="ball",
        scale=(0.25, 0.25, 0.25),
        position=(0.0, 0.0, 1.5),
        velocity=(0.5, 0.0, -0.5),
        friction=0.05,
        restitution=0.8,
    )
    scene.add([ground, ball])

    simulator = PyBullet(scene)
    animation, collisions = simulator.run()
    summary = {
        "kubric_version": kb.__version__,
        "objects": [asset.name for asset in animation],
        "ball_z_first": round(float(animation[ball]["position"][0][2]), 5),
        "ball_z_last": round(float(animation[ball]["position"][-1][2]), 5),
        "collision_count": len(collisions),
        "first_collision_frame": round(float(collisions[0]["frame"]), 5) if collisions else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "physics_smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_render(out_dir: Path, resolution: int) -> None:
    import kubric as kb
    from kubric.renderer import Blender

    patch_blender_denoiser(Blender)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = kb.Scene(
        frame_start=1,
        frame_end=1,
        frame_rate=24,
        step_rate=240,
        resolution=(resolution, resolution),
    )
    red = kb.PrincipledBSDFMaterial(color=kb.Color.from_hexint(0xDD3333), roughness=0.7)
    gray = kb.PrincipledBSDFMaterial(color=kb.Color.from_hexint(0x999999), roughness=0.8)
    ball = kb.Sphere(
        name="ball",
        scale=(0.35, 0.35, 0.35),
        position=(0.0, 0.0, 0.35),
        material=red,
        friction=0.05,
        restitution=0.8,
    )
    ground = kb.Cube(
        name="ground",
        scale=(4.0, 4.0, 0.05),
        position=(0.0, 0.0, -0.025),
        static=True,
        material=gray,
        background=True,
    )
    camera = kb.PerspectiveCamera(
        name="camera",
        position=(3.0, -5.0, 2.4),
        focal_length=35.0,
        sensor_width=32.0,
    )
    camera.look_at((0.0, 0.0, 0.3))
    light = kb.DirectionalLight(name="sun", position=(3.0, -4.0, 6.0), intensity=1.5)
    light.look_at((0.0, 0.0, 0.0))

    scene.camera = camera
    scene.add([ground, ball, camera, light])

    renderer = Blender(
        scene,
        scratch_dir=out_dir,
        samples_per_pixel=1,
        use_denoising=False,
        adaptive_sampling=False,
        verbose=False,
    )
    renderer.render(png_filepath=str(out_dir / "frame_"), exr_filepath=None)
    rendered = sorted(path.name for path in out_dir.glob("frame_*.png"))
    print(json.dumps({"kubric_version": kb.__version__, "rendered": rendered}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["physics", "render"], default="physics")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/kubric_official_smoke"))
    parser.add_argument("--resolution", type=int, default=64)
    args = parser.parse_args(blender_argv())

    if args.mode == "physics":
        run_physics(args.out_dir)
    else:
        run_render(args.out_dir, args.resolution)


if __name__ == "__main__":
    main()
