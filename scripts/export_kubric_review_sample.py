#!/usr/bin/env python3
"""Export a compact website review sample from a larger Kubric batch run."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST_ROOT = PROJECT_ROOT / "site" / "assets" / "runs"
FEATURE_WEIGHTS = {
    "camera_families": 4.0,
    "physics_families": 5.0,
    "pair_kinds": 3.0,
    "physics_kinds": 2.0,
    "camera_speed_classes": 1.0,
    "physics_speed_classes": 1.0,
    "duration_bins": 1.0,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def duration_bin(duration_s: float) -> str:
    if duration_s < 3.5:
        return "short"
    if duration_s < 5.25:
        return "medium"
    return "long"


def clip_duration_s(clip: dict[str, Any], fps: float) -> float:
    if "duration_s" in clip:
        return float(clip["duration_s"])
    if "frames_count" in clip:
        return max(0.0, (int(clip["frames_count"]) - 1) / max(fps, 1e-6))
    return 0.0


def clip_frames_count(clip: dict[str, Any], fps: float) -> int:
    if "frames_count" in clip:
        return int(clip["frames_count"])
    if "duration_s" in clip:
        return max(1, int(round(float(clip["duration_s"]) * fps)) + 1)
    return 0


def clip_path(run_dir: Path, clip: dict[str, Any], key: str) -> Path:
    return run_dir / Path(str(clip[key]))


def group_is_complete(run_dir: Path, group: dict[str, Any], clips_by_id: dict[str, dict[str, Any]]) -> bool:
    for clip_id in group["clip_ids"]:
        clip = clips_by_id.get(clip_id)
        if clip is None:
            return False
        video = clip_path(run_dir, clip, "video")
        metadata = clip_path(run_dir, clip, "metadata")
        if not video.exists() or video.stat().st_size <= 0:
            return False
        if not metadata.exists() or metadata.stat().st_size <= 0:
            return False
    return True


def group_features(group: dict[str, Any], clips_by_id: dict[str, dict[str, Any]], fps: float) -> dict[str, set[str]]:
    clips = [clips_by_id[clip_id] for clip_id in group["clip_ids"]]
    group_duration = float(group.get("duration_s", clip_duration_s(clips[0], fps)))
    return {
        "camera_families": {str(clip.get("camera_family", "")) for clip in clips},
        "physics_families": {str(clip.get("physics_family", "")) for clip in clips},
        "pair_kinds": {str(group.get("tags", ["unknown"])[0])},
        "physics_kinds": {str(clip.get("physics_kind", "")) for clip in clips},
        "camera_speed_classes": {str(clip.get("camera_speed_class", "")) for clip in clips},
        "physics_speed_classes": {str(clip.get("physics_speed_class", "")) for clip in clips},
        "duration_bins": {duration_bin(group_duration)},
    }


def select_groups(
    groups: list[dict[str, Any]],
    clips_by_id: dict[str, dict[str, Any]],
    *,
    target: int,
    seed: int,
    fps: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = [
        {
            "index": index,
            "group": group,
            "features": group_features(group, clips_by_id, fps),
        }
        for index, group in enumerate(groups)
    ]
    selected: list[dict[str, Any]] = []
    seen = {key: set() for key in FEATURE_WEIGHTS}

    while candidates and len(selected) < target:
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(candidates):
            score = rng.random() * 0.01
            for key, values in candidate["features"].items():
                score += FEATURE_WEIGHTS[key] * len(values - seen[key])
            if score > best_score:
                best_score = score
                best_index = index
        candidate = candidates.pop(best_index)
        selected.append(candidate["group"])
        for key, values in candidate["features"].items():
            seen[key].update(values)

    return selected


def summarize(clips: list[dict[str, Any]], groups: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    durations = [clip_duration_s(clip, fps) for clip in clips]
    frames = [clip_frames_count(clip, fps) for clip in clips]
    return {
        "camera_families": dict(Counter(str(clip.get("camera_family", clip.get("camera_id", ""))) for clip in clips)),
        "physics_families": dict(Counter(str(clip.get("physics_family", clip.get("physics_id", ""))) for clip in clips)),
        "pair_kinds": dict(Counter(str(group.get("tags", ["unknown"])[0]) for group in groups)),
        "duration_s_min": min(durations) if durations else None,
        "duration_s_max": max(durations) if durations else None,
        "frames_min": min(frames) if frames else None,
        "frames_max": max(frames) if frames else None,
    }


def update_index(run_root: Path, run_id: str) -> None:
    index_path = run_root / "index.json"
    if index_path.exists():
        index = read_json(index_path)
        runs = [run for run in index.get("runs", []) if run.get("run_id") != run_id]
    else:
        runs = []
    runs.append({"run_id": run_id, "manifest": f"{run_id}/manifest.json"})
    write_json(index_path, {"runs": runs})


def copy_clip(
    *,
    source_run_dir: Path,
    dest_run_dir: Path,
    source_run_id: str,
    dest_run_id: str,
    clip: dict[str, Any],
) -> dict[str, Any]:
    copied_clip = copy.deepcopy(clip)
    copied_clip["source_run_id"] = source_run_id
    copied_clip["source_clip_id"] = clip["clip_id"]

    for key in ("video", "metadata"):
        rel_path = Path(str(clip[key]))
        source_path = source_run_dir / rel_path
        dest_path = dest_run_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

    metadata_path = dest_run_dir / Path(str(clip["metadata"]))
    metadata = read_json(metadata_path)
    metadata["source_run_id"] = source_run_id
    metadata["source_clip_id"] = clip["clip_id"]
    metadata["review_sample_run_id"] = dest_run_id
    write_json(metadata_path, metadata)
    return copied_clip


def export_review_sample(args: argparse.Namespace) -> Path:
    source_manifest = read_json(args.source_run_dir / "manifest.json")
    source_run_id = str(source_manifest.get("run_id", args.source_run_dir.name))
    dest_run_dir = args.dest_run_root / args.dest_run_id

    if dest_run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{dest_run_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(dest_run_dir)
    dest_run_dir.mkdir(parents=True, exist_ok=True)

    clips_by_id = {clip["clip_id"]: clip for clip in source_manifest["clips"]}
    complete_groups = [
        group
        for group in source_manifest["pair_groups"]
        if group_is_complete(args.source_run_dir, group, clips_by_id)
    ]
    if not complete_groups:
        raise SystemExit(f"No complete pair groups with videos found in {args.source_run_dir}")

    selected_groups = select_groups(
        complete_groups,
        clips_by_id,
        target=min(args.pairs, len(complete_groups)),
        seed=args.seed,
        fps=float(source_manifest.get("fps", 24)),
    )
    selected_clip_ids = []
    for group in selected_groups:
        selected_clip_ids.extend(group["clip_ids"])

    selected_clips = [
        copy_clip(
            source_run_dir=args.source_run_dir,
            dest_run_dir=dest_run_dir,
            source_run_id=source_run_id,
            dest_run_id=args.dest_run_id,
            clip=clips_by_id[clip_id],
        )
        for clip_id in selected_clip_ids
    ]

    selected_clip_id_set = set(selected_clip_ids)
    ambiguous_groups = [
        group
        for group in source_manifest.get("ambiguous_equivalence_groups", [])
        if all(clip_id in selected_clip_id_set for clip_id in group.get("clip_ids", []))
    ]

    dest_manifest = copy.deepcopy(source_manifest)
    dest_manifest["run_id"] = args.dest_run_id
    dest_manifest["generator"] = "kubric_review_sample_export"
    dest_manifest["description"] = (
        "Compact coverage-oriented review sample exported from a larger "
        "official-Kubric batch run."
    )
    dest_manifest["source_run"] = {
        "run_id": source_run_id,
        "run_dir": str(args.source_run_dir),
    }
    dest_manifest["review_sample"] = {
        "selected_pair_groups": len(selected_groups),
        "selected_clips": len(selected_clips),
        "selection_seed": args.seed,
        "selection_policy": (
            "greedy coverage over camera families, physics families, pair kinds, "
            "physics kinds, speed classes, and duration bins; only complete "
            "pair groups with videos are eligible"
        ),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dest_manifest["coverage_summary"] = summarize(selected_clips, selected_groups, float(source_manifest.get("fps", 24)))
    dest_manifest["clips"] = selected_clips
    dest_manifest["pair_groups"] = copy.deepcopy(selected_groups)
    dest_manifest["ambiguous_equivalence_groups"] = copy.deepcopy(ambiguous_groups)
    write_json(dest_run_dir / "manifest.json", dest_manifest)
    update_index(args.dest_run_root, args.dest_run_id)
    return dest_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--dest-run-id", required=True)
    parser.add_argument("--dest-run-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    run_dir = export_review_sample(args)
    manifest = read_json(run_dir / "manifest.json")
    print(
        f"Exported {len(manifest['clips'])} clips / {len(manifest['pair_groups'])} pairs "
        f"to {run_dir}"
    )


if __name__ == "__main__":
    main()
