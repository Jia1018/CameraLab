#!/usr/bin/env python3
"""Repair pre-existing batch-v2 camera paths that cross the scene floor."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import generate_kubric_batch_v2 as batch
import generate_official_kubric_review_bank as gen


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def repaired_camera_metadata(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    camera = dict(metadata["camera_spec"])
    if camera.get("path_model") == "orbit":
        return None

    family = str(camera["family_id"])
    min_required = batch.minimum_camera_height(family)
    old_frames = metadata["camera_frames"]
    old_min_height = min(float(frame["position"][2]) for frame in old_frames)
    if old_min_height >= min_required - 1e-5:
        return None

    start_position = tuple(float(value) for value in camera["start_position"])
    old_delta = tuple(float(value) for value in camera["position_delta"])
    new_delta, height_constraint = batch.constrain_linear_camera_height(
        family,
        start_position,
        old_delta,
    )
    if new_delta == old_delta:
        raise RuntimeError(f"{metadata['clip_id']}: low path was not corrected")

    frames_count = int(metadata["frames_count"])
    duration_s = max((frames_count - 1) / int(metadata["fps"]), 1e-6)
    camera["position_delta"] = gen.vec(new_delta)
    camera["average_linear_velocity_mps"] = gen.vec([value / duration_s for value in new_delta])
    camera["speed_class"] = batch.speed_class(math.sqrt(sum(value**2 for value in new_delta)), duration_s)
    camera["height_constraint"] = height_constraint

    camera_frames = batch.camera_records(camera, frames_count)
    path_audit = batch.camera_path_audit(camera, camera_frames)
    if not path_audit["passed"]:
        raise RuntimeError(f"{metadata['clip_id']}: repaired path audit failed: {path_audit}")
    width, height = (int(value) for value in metadata["resolution"])
    framing_audit = batch.camera_framing_audit(
        camera_frames,
        metadata["physics_frames"],
        width,
        height,
    )
    if not framing_audit["passed"]:
        raise RuntimeError(f"{metadata['clip_id']}: repaired framing audit failed: {framing_audit}")

    repaired = dict(metadata)
    repaired["camera_spec"] = camera
    repaired["camera_frames"] = camera_frames
    repaired["camera_path_audit"] = path_audit
    repaired["camera_framing_audit"] = framing_audit
    record = {
        "clip_id": metadata["clip_id"],
        "camera_id": metadata["camera_id"],
        "camera_family": family,
        "old_minimum_height_m": round(old_min_height, 5),
        "new_minimum_height_m": path_audit["minimum_height_m"],
        "old_end_height_m": round(float(old_frames[-1]["position"][2]), 5),
        "new_end_height_m": round(float(camera_frames[-1]["position"][2]), 5),
        "old_position_delta": gen.vec(old_delta),
        "new_position_delta": gen.vec(new_delta),
        "camera_speed_class": camera["speed_class"],
    }
    return repaired, record


def repair_run(run_dir: Path, *, apply: bool) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    jobs_path = run_dir / "render_jobs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs_payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs_by_clip = {job["clip_id"]: job for job in jobs_payload["jobs"]}

    repaired_payloads: list[tuple[Path, dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    speed_classes: dict[str, str] = {}
    for metadata_path in sorted((run_dir / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = repaired_camera_metadata(metadata)
        if result is None:
            continue
        repaired, record = result
        repaired_payloads.append((metadata_path, repaired))
        records.append(record)
        speed_classes[record["clip_id"]] = record["camera_speed_class"]

    missing_jobs = sorted(record["clip_id"] for record in records if record["clip_id"] not in jobs_by_clip)
    if missing_jobs:
        raise RuntimeError(f"missing render jobs for repaired clips: {missing_jobs}")

    if not apply:
        return records

    for metadata_path, repaired in repaired_payloads:
        write_json_atomic(metadata_path, repaired)
    for clip in manifest["clips"]:
        speed_class = speed_classes.get(clip["clip_id"])
        if speed_class is not None:
            clip["camera_speed_class"] = speed_class
    write_json_atomic(manifest_path, manifest)

    repair_jobs_path = run_dir / "render_jobs.camera_floor_repair.json"
    write_json_atomic(
        repair_jobs_path,
        {
            "run_id": jobs_payload["run_id"],
            "fps": jobs_payload["fps"],
            "repair_kind": "camera_floor_clearance",
            "jobs": [jobs_by_clip[record["clip_id"]] for record in records],
        },
    )
    write_json_atomic(
        run_dir / "camera_floor_repair.json",
        {
            "run_id": jobs_payload["run_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "minimum_height_policy_m": {
                "default": batch.DEFAULT_MIN_CAMERA_HEIGHT_M,
                "crane_tilt": batch.CRANE_TILT_MIN_CAMERA_HEIGHT_M,
            },
            "clips_repaired": len(records),
            "unique_cameras_repaired": len({record["camera_id"] for record in records}),
            "render_jobs": repair_jobs_path.name,
            "records": records,
        },
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrected metadata and per-run rerender job files. The default is a dry run.",
    )
    args = parser.parse_args()

    total = 0
    for run_dir in args.run_dirs:
        records = repair_run(run_dir, apply=args.apply)
        total += len(records)
        print(
            f"{run_dir}: clips={len(records)} "
            f"unique_cameras={len({record['camera_id'] for record in records})} "
            f"mode={'apply' if args.apply else 'dry-run'}"
        )
        for record in records:
            print(
                f"  {record['clip_id']}: "
                f"z={record['old_minimum_height_m']} -> {record['new_minimum_height_m']}"
            )
    print(f"total clips={total}")


if __name__ == "__main__":
    main()
