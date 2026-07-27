#!/usr/bin/env python3
"""Kubric scaffold for paired camera/physics synthetic videos.

The local machine currently needs Kubric, Blender Python, PyBullet, and imageio
before this can render. The important part here is the experiment contract:
camera trajectories and physics seeds are independent ids, so paired clips can
share one factor exactly while varying the other.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "site" / "assets" / "runs"


CAMERA_TRAJECTORIES = {
    "cam_orbit_left": {
        "description": "Fixed-radius leftward orbit around the scene center.",
        "keyframes": [
            {"frame": 0, "position": [3.2, -5.0, 3.0], "look_at": [0.0, 0.0, 0.6]},
            {"frame": 47, "position": [0.2, -5.8, 3.0], "look_at": [0.0, 0.0, 0.6]},
            {"frame": 95, "position": [-3.2, -5.0, 3.0], "look_at": [0.0, 0.0, 0.6]},
        ],
    },
    "cam_dolly_in": {
        "description": "Forward dolly with a slight lateral correction.",
        "keyframes": [
            {"frame": 0, "position": [1.0, -7.2, 2.8], "look_at": [0.0, 0.0, 0.6]},
            {"frame": 47, "position": [0.4, -5.4, 2.6], "look_at": [0.0, 0.0, 0.6]},
            {"frame": 95, "position": [0.0, -3.9, 2.4], "look_at": [0.0, 0.0, 0.6]},
        ],
    },
}

PHYSICS_PROGRAMS = {
    "phys_bounce": {
        "description": "Sphere dropped with horizontal velocity; gravity/collision produce bounce.",
        "seed": 101,
        "objects": [{"asset": "sphere", "position": [-1.7, 0.0, 2.0], "velocity": [1.4, 0.0, 0.0]}],
    },
    "phys_roll": {
        "description": "Box starts with linear and angular velocity and rolls across the plane.",
        "seed": 202,
        "objects": [
            {
                "asset": "cube",
                "position": [-1.7, 0.0, 0.6],
                "velocity": [1.2, 0.0, 0.0],
                "angular_velocity": [0.0, 3.0, 0.0],
            }
        ],
    },
}

CLIP_SPECS = [
    ("A_cam_orbit_phys_bounce", "cam_orbit_left", "phys_bounce"),
    ("B_cam_orbit_phys_roll", "cam_orbit_left", "phys_roll"),
    ("C_cam_dolly_phys_bounce", "cam_dolly_in", "phys_bounce"),
]


def require_kubric() -> None:
    missing = [name for name in ["kubric", "pybullet", "imageio"] if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install Kubric/Blender/PyBullet first, or run scripts/generate_mock_pairs.py for the preview dataset."
        )


def write_contract(run_dir: Path, run_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "run_id": run_id,
        "camera_trajectories": CAMERA_TRAJECTORIES,
        "physics_programs": PHYSICS_PROGRAMS,
        "clip_specs": [
            {"clip_id": clip_id, "camera_id": camera_id, "physics_id": physics_id}
            for clip_id, camera_id, physics_id in CLIP_SPECS
        ],
        "pair_groups": [
            {
                "group_id": "same_camera_diff_physics",
                "clip_ids": ["A_cam_orbit_phys_bounce", "B_cam_orbit_phys_roll"],
                "controlled_factor": "camera_id",
                "varied_factor": "physics_id",
            },
            {
                "group_id": "diff_camera_same_physics",
                "clip_ids": ["A_cam_orbit_phys_bounce", "C_cam_dolly_phys_bounce"],
                "controlled_factor": "physics_id",
                "varied_factor": "camera_id",
            },
        ],
    }
    (run_dir / "kubric_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="kubric_pair_v0")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()

    run_dir = args.run_root / args.run_id
    write_contract(run_dir, args.run_id)
    require_kubric()
    raise SystemExit(
        "Kubric render implementation is the next step after installing the runtime. "
        f"The paired experiment contract was written to {run_dir / 'kubric_contract.json'}."
    )


if __name__ == "__main__":
    main()
