#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove intermediate checkpoints only when the final is evaluated."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = STEP_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(1))


def discover_evaluated_checkpoints(root: Path) -> dict[Path, list[Path]]:
    evaluated: dict[Path, list[Path]] = defaultdict(list)
    for result_path in root.rglob("*.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        metadata = result.get("metadata", {})
        checkpoints = metadata.get("checkpoints", {})
        models = result.get("models", {})
        if not isinstance(checkpoints, dict) or not isinstance(models, dict):
            continue
        for name, raw_path in checkpoints.items():
            if name not in models or not isinstance(raw_path, str):
                continue
            evaluated[Path(raw_path).expanduser().resolve()].append(result_path)
    return evaluated


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    evaluated = discover_evaluated_checkpoints(root)
    checkpoints_by_run: dict[Path, list[Path]] = defaultdict(list)
    for checkpoint in root.rglob("step_*.pt"):
        if checkpoint.is_file() and STEP_PATTERN.match(checkpoint.name):
            checkpoints_by_run[checkpoint.parent].append(checkpoint)

    candidates: list[Path] = []
    for run_dir, checkpoints in sorted(checkpoints_by_run.items()):
        checkpoints.sort(key=checkpoint_step)
        if len(checkpoints) < 2:
            continue
        final_checkpoint = checkpoints[-1].resolve()
        result_paths = evaluated.get(final_checkpoint, [])
        if final_checkpoint.stat().st_size == 0 or not result_paths:
            print(f"SKIP run={run_dir} reason=final_not_evaluated")
            continue
        intermediates = checkpoints[:-1]
        candidates.extend(intermediates)
        size_bytes = sum(path.stat().st_size for path in intermediates)
        print(
            f"READY run={run_dir} final={final_checkpoint.name} "
            f"evaluated_by={result_paths[0]} intermediates={len(intermediates)} "
            f"bytes={size_bytes}"
        )

    total_bytes = sum(path.stat().st_size for path in candidates)
    print(
        f"SUMMARY mode={'execute' if args.execute else 'dry-run'} "
        f"files={len(candidates)} bytes={total_bytes}"
    )
    if not args.execute:
        return
    for checkpoint in candidates:
        checkpoint.unlink()
        print(f"REMOVED {checkpoint}")


if __name__ == "__main__":
    main()
