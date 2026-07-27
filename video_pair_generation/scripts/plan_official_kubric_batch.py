#!/usr/bin/env python3
"""Create a reproducible shard plan for official Kubric batch generation."""

from __future__ import annotations

import argparse
import json
import random
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = PROJECT_ROOT / "video_pair_generation" / "scripts" / "generate_official_kubric_review_bank.py"
DEFAULT_PYTHON = Path("/workspace/writeable/environments/kubric_official/bin/python")
DEFAULT_RUN_ROOT = Path("/workspace/writeable/datasets/camera_motion_disentangle")
FPS = 24


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sampled_frame_count(
    rng: random.Random, mean: float, std: float, low: int, high: int, multiple: int
) -> int:
    value = int(round(clamp(rng.gauss(mean, std), low, high)))
    if multiple > 1:
        value = int(round(value / multiple) * multiple)
    return max(low, min(high, value))


def shell_join(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_generator_command(
    *,
    python_bin: Path,
    run_id: str,
    run_root: Path,
    camera_limit: int,
    physics_limit: int,
    pairs: int,
    frames: int,
    width: int,
    height: int,
    seed: int,
    overwrite: bool,
    render: bool,
) -> str:
    command: list[Any] = [
        python_bin,
        GENERATOR,
        "--run-id",
        run_id,
        "--run-root",
        run_root,
        "--camera-limit",
        camera_limit,
        "--physics-limit",
        physics_limit,
        "--pairs",
        pairs,
        "--frames",
        frames,
        "--width",
        width,
        "--height",
        height,
        "--seed",
        seed,
    ]
    if overwrite:
        command.append("--overwrite")
    if not render:
        command.append("--no-render")
    return "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python " + shell_join(command)


def write_shell_script(path: Path, commands: list[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        "",
    ]
    lines.extend(commands)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def plan_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default="kubric_batch_v1")
    parser.add_argument("--plan-dir", type=Path, default=PROJECT_ROOT / "video_pair_generation" / "batch_plans")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=20260706)
    parser.add_argument("--camera-limit", type=int, default=5)
    parser.add_argument("--physics-limit", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames-mean", type=float, default=96.0)
    parser.add_argument("--frames-std", type=float, default=18.0)
    parser.add_argument("--frames-min", type=int, default=72)
    parser.add_argument("--frames-max", type=int, default=144)
    parser.add_argument("--frame-multiple", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.shards <= 0:
        raise SystemExit("--shards must be positive")
    if args.frames_min <= 0 or args.frames_max < args.frames_min:
        raise SystemExit("frame bounds are invalid")
    if args.frame_multiple <= 0:
        raise SystemExit("--frame-multiple must be positive")

    rng = random.Random(args.seed_base + 707)
    batch_run_root = args.run_root / args.batch_id
    plan_dir = args.plan_dir / args.batch_id
    plan_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    audit_commands = []
    render_commands = []
    for shard_index in range(args.shards):
        seed = args.seed_base + shard_index
        frames = sampled_frame_count(
            rng,
            args.frames_mean,
            args.frames_std,
            args.frames_min,
            args.frames_max,
            args.frame_multiple,
        )
        run_id = f"{args.batch_id}_shard_{shard_index:04d}"
        clips = args.camera_limit * args.physics_limit
        duration_s = round((frames - 1) / FPS, 5)
        shard = {
            "shard_index": shard_index,
            "run_id": run_id,
            "seed": seed,
            "run_root": str(batch_run_root),
            "camera_limit": args.camera_limit,
            "physics_limit": args.physics_limit,
            "clips": clips,
            "target_pairs": args.pairs,
            "frames": frames,
            "duration_s": duration_s,
            "resolution": [args.width, args.height],
        }
        shard["audit_command"] = build_generator_command(
            python_bin=args.python_bin,
            run_id=run_id,
            run_root=batch_run_root,
            camera_limit=args.camera_limit,
            physics_limit=args.physics_limit,
            pairs=args.pairs,
            frames=frames,
            width=args.width,
            height=args.height,
            seed=seed,
            overwrite=args.overwrite,
            render=False,
        )
        shard["render_command"] = build_generator_command(
            python_bin=args.python_bin,
            run_id=run_id,
            run_root=batch_run_root,
            camera_limit=args.camera_limit,
            physics_limit=args.physics_limit,
            pairs=args.pairs,
            frames=frames,
            width=args.width,
            height=args.height,
            seed=seed,
            overwrite=args.overwrite,
            render=True,
        )
        shards.append(shard)
        audit_commands.append(shard["audit_command"])
        render_commands.append(shard["render_command"])

    total_clips = sum(shard["clips"] for shard in shards)
    total_pairs = sum(shard["target_pairs"] for shard in shards)
    total_video_seconds = round(sum(shard["clips"] * shard["duration_s"] for shard in shards), 5)
    total_render_frames = sum(shard["clips"] * shard["frames"] for shard in shards)
    plan = {
        "batch_id": args.batch_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(GENERATOR),
        "python_bin": str(args.python_bin),
        "run_root": str(batch_run_root),
        "contract": {
            "same_camera_pairs": "same sampled camera trajectory, varied physics program and scene",
            "same_physics_pairs": "same physics frames and same scene, varied sampled camera trajectory",
            "paired_lengths": "all clips in a shard share frames; different shards sample different frame counts",
        },
        "sampling": {
            "frames": {
                "distribution": "clipped Gaussian rounded to frame_multiple",
                "mean": args.frames_mean,
                "std": args.frames_std,
                "min": args.frames_min,
                "max": args.frames_max,
                "frame_multiple": args.frame_multiple,
            },
            "per_shard_seed": "seed_base + shard_index",
        },
        "totals": {
            "shards": args.shards,
            "clips": total_clips,
            "target_pairs": total_pairs,
            "video_seconds": total_video_seconds,
            "render_frames": total_render_frames,
        },
        "recommended_workflow": [
            "Run audit_shards.sh first; it simulates physics and writes metadata without Blender rendering.",
            "Inspect audit failures and adjust sampling/audit thresholds before running render_shards.sh.",
            "Keep full batch outputs outside the git repo; publish only a small review subset to GitHub Pages docs.",
        ],
        "shards": shards,
    }

    plan_path = plan_dir / "plan.json"
    audit_path = plan_dir / "audit_shards.sh"
    render_path = plan_dir / "render_shards.sh"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    write_shell_script(audit_path, audit_commands)
    write_shell_script(render_path, render_commands)

    print(f"wrote {plan_path}")
    print(f"wrote {audit_path}")
    print(f"wrote {render_path}")
    print(f"planned {total_clips} clips, {total_pairs} target pairs, {total_render_frames} render frames")


if __name__ == "__main__":
    plan_main()
