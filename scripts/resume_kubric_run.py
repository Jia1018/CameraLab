#!/usr/bin/env python3
"""Resume rendering/encoding an interrupted Kubric run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import generate_official_kubric_review_bank as gen
from update_kubric_progress import build_progress, write_progress


DEFAULT_PYTHON = Path("/workspace/writeable/environments/kubric_official/bin/python")


def read_jobs(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "render_jobs.json").read_text(encoding="utf-8"))


def write_jobs(path: Path, run_id: str, fps: int, jobs: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"run_id": run_id, "fps": fps, "jobs": jobs}, indent=2), encoding="utf-8")


def log_event(run_dir: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with (run_dir / "resume_parent.log").open("a", encoding="utf-8") as log_file:
        print(f"{timestamp} {message}", file=log_file, flush=True)


def frame_job_complete(job: dict[str, Any], row_by_clip: dict[str, dict[str, Any]]) -> bool:
    return bool(row_by_clip[job["clip_id"]]["frames_complete"])


def video_job_complete(job: dict[str, Any], row_by_clip: dict[str, dict[str, Any]]) -> bool:
    return bool(row_by_clip[job["clip_id"]]["video_complete"])


def render_missing(
    *,
    run_dir: Path,
    blender_bin: Path,
    kubric_site_packages: Path,
    jobs_payload: dict[str, Any],
    rows: dict[str, dict[str, Any]],
) -> None:
    missing = [job for job in jobs_payload["jobs"] if not frame_job_complete(job, rows)]
    if not missing:
        log_event(run_dir, "render: no missing frame jobs")
        return
    resume_jobs_path = run_dir / "render_jobs.resume_frames.json"
    write_jobs(resume_jobs_path, jobs_payload["run_id"], int(jobs_payload["fps"]), missing)
    log_event(run_dir, f"render: starting {len(missing)} missing frame jobs via {blender_bin}")
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(kubric_site_packages) if not existing else f"{kubric_site_packages}:{existing}"
    gen.configure_blender_runtime_env(env)
    log_path = run_dir / "resume_render.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [
                str(blender_bin),
                "--background",
                *gen.blender_thread_args(),
                "--python",
                str(Path(gen.__file__).resolve()),
                "--",
                "--render-jobs",
                str(resume_jobs_path),
            ],
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
    log_event(run_dir, f"render: finished {len(missing)} missing frame jobs")


def encode_missing(run_dir: Path, jobs_payload: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    missing = [job for job in jobs_payload["jobs"] if not video_job_complete(job, rows)]
    if not missing:
        log_event(run_dir, "encode: no missing video jobs")
        return
    encode_jobs_path = run_dir / "render_jobs.resume_videos.json"
    write_jobs(encode_jobs_path, jobs_payload["run_id"], int(jobs_payload["fps"]), missing)
    log_event(run_dir, f"encode: starting {len(missing)} video jobs")
    fps = int(jobs_payload["fps"])
    for index, job in enumerate(missing, start=1):
        frames_dir = Path(job["frames_dir"])
        video_path = Path(job["video"])
        tmp_video_path = video_path.with_name(f"{video_path.name}.tmp.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if tmp_video_path.exists():
            tmp_video_path.unlink()
        log_event(run_dir, f"encode: [{index}/{len(missing)}] starting {job['clip_id']}")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-start_number",
                "1",
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(tmp_video_path),
            ],
            check=True,
        )
        tmp_video_path.replace(video_path)
        log_event(run_dir, f"encode: [{index}/{len(missing)}] finished {job['clip_id']} bytes={video_path.stat().st_size}")
    log_event(run_dir, f"encode: finished {len(missing)} video jobs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--blender-bin", type=Path, default=gen.DEFAULT_BLENDER)
    parser.add_argument("--kubric-site-packages", type=Path, default=gen.DEFAULT_KUBRIC_SITE_PACKAGES)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    args = parser.parse_args()

    log_event(args.run_dir, "resume: started")
    jobs_payload = read_jobs(args.run_dir)
    progress = write_progress(args.run_dir, args.python_bin)
    log_event(
        args.run_dir,
        f"resume: initial phase={progress['phase']} frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
        f"videos={progress['videos_complete']}/{progress['jobs_total']}",
    )
    rows = {row["clip_id"]: row for row in progress["jobs"]}
    if not args.skip_render:
        render_missing(
            run_dir=args.run_dir,
            blender_bin=args.blender_bin,
            kubric_site_packages=args.kubric_site_packages,
            jobs_payload=jobs_payload,
            rows=rows,
        )
        progress = write_progress(args.run_dir, args.python_bin)
        log_event(
            args.run_dir,
            f"resume: after render phase={progress['phase']} frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
            f"videos={progress['videos_complete']}/{progress['jobs_total']}",
        )
        rows = {row["clip_id"]: row for row in progress["jobs"]}
    if not args.skip_encode:
        encode_missing(args.run_dir, jobs_payload, rows)
        progress = write_progress(args.run_dir, args.python_bin)
        log_event(
            args.run_dir,
            f"resume: after encode phase={progress['phase']} frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
            f"videos={progress['videos_complete']}/{progress['jobs_total']}",
        )
    log_event(args.run_dir, "resume: finished")
    print(
        f"resume status: {progress['phase']} "
        f"frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
        f"videos={progress['videos_complete']}/{progress['jobs_total']}"
    )


if __name__ == "__main__":
    main()
