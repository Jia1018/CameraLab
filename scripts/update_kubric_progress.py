#!/usr/bin/env python3
"""Write progress.json for a generated Kubric run."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/workspace/writeable/environments/kubric_official/bin/python")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_frames(job: dict[str, Any]) -> int:
    metadata_path = Path(job["metadata"])
    if not metadata_path.exists():
        return 0
    return int(read_json(metadata_path).get("frames_count", 0))


def frame_count(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    return sum(1 for _ in frames_dir.glob("frame_*.png"))


def last_render_log_event(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"\[(\d+)/(\d+)\]\s+(rendering|skipping completed)\s+(.+)$")
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line.strip())
        if match:
            last = {
                "index": int(match.group(1)),
                "total": int(match.group(2)),
                "action": match.group(3),
                "clip_id": match.group(4),
            }
    return last


def build_progress(run_dir: Path, python_bin: Path) -> dict[str, Any]:
    jobs_path = run_dir / "render_jobs.json"
    manifest_path = run_dir / "manifest.json"
    jobs_payload = read_json(jobs_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    jobs = jobs_payload.get("jobs", [])
    log_event = last_render_log_event(run_dir / "render.log")

    job_rows = []
    frames_expected_total = 0
    frames_rendered_total = 0
    rendered_jobs_complete = 0
    videos_complete = 0
    active_job = log_event.get("clip_id") if log_event else None

    for job in jobs:
        frames_dir = Path(job["frames_dir"])
        video_path = Path(job["video"])
        expected = expected_frames(job)
        rendered = frame_count(frames_dir)
        final_frame = frames_dir / f"frame_{expected:04d}.png" if expected else frames_dir / "frame_0001.png"
        frames_complete = expected > 0 and rendered >= expected and final_frame.exists()
        video_complete = video_path.exists() and video_path.stat().st_size > 0
        frames_expected_total += expected
        frames_rendered_total += min(rendered, expected)
        rendered_jobs_complete += int(frames_complete)
        videos_complete += int(video_complete)
        job_rows.append(
            {
                "clip_id": job["clip_id"],
                "expected_frames": expected,
                "rendered_frames": rendered,
                "frames_complete": frames_complete,
                "video_complete": video_complete,
                "frames_dir": str(frames_dir),
                "video": str(video_path),
            }
        )

    all_frames_complete = bool(jobs) and rendered_jobs_complete == len(jobs)
    all_videos_complete = bool(jobs) and videos_complete == len(jobs)
    if all_videos_complete:
        phase = "complete"
    elif all_frames_complete:
        phase = "encoding_or_waiting_for_encode"
    elif frames_rendered_total > 0:
        phase = "rendering"
    else:
        phase = "metadata_ready"

    rel_run_dir = run_dir
    try:
        rel_run_dir = run_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        pass

    resume_command = (
        f"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python {python_bin} "
        f"{PROJECT_ROOT / 'scripts' / 'resume_kubric_run.py'} --run-dir {run_dir}"
    )
    progress = {
        "run_id": jobs_payload.get("run_id") or manifest.get("run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "run_dir_relative_to_repo": str(rel_run_dir),
        "phase": phase,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "active_job": active_job,
        "last_render_log_event": log_event,
        "jobs_total": len(jobs),
        "frames_expected_total": frames_expected_total,
        "frames_rendered_total": frames_rendered_total,
        "frames_progress": round(frames_rendered_total / frames_expected_total, 5) if frames_expected_total else 0.0,
        "rendered_jobs_complete": rendered_jobs_complete,
        "videos_complete": videos_complete,
        "complete": all_videos_complete,
        "resume_command": resume_command,
        "note": "Use resume_command after an interrupted run; it skips completed frame directories and completed videos.",
        "jobs": job_rows,
    }
    return progress


def write_progress(run_dir: Path, python_bin: Path) -> dict[str, Any]:
    progress = build_progress(run_dir, python_bin)
    out_path = run_dir / "progress.json"
    tmp_path = run_dir / "progress.json.tmp"
    tmp_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--watch", type=float, default=0.0, help="Update every N seconds; 0 writes once.")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    while True:
        progress = write_progress(args.run_dir, args.python_bin)
        print(
            f"{progress['updated_at_utc']} {progress['phase']} "
            f"frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
            f"videos={progress['videos_complete']}/{progress['jobs_total']}",
            flush=True,
        )
        if args.watch <= 0:
            break
        if args.until_complete and progress["complete"]:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
