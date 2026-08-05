#!/usr/bin/env python3
"""Resume rendering/encoding an interrupted Kubric run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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


def start_progress_watcher(run_dir: Path, python_bin: Path, interval: float) -> subprocess.Popen[bytes] | None:
    if interval <= 0:
        return None
    return subprocess.Popen(
        [
            sys.executable,
            str(gen.PROJECT_ROOT / "video_pair_generation" / "scripts" / "update_kubric_progress.py"),
            "--run-dir",
            str(run_dir),
            "--python-bin",
            str(python_bin),
            "--watch",
            str(interval),
            "--until-complete",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_resource_watcher(run_dir: Path, interval: float) -> subprocess.Popen[bytes] | None:
    if interval <= 0:
        return None
    return subprocess.Popen(
        [
            sys.executable,
            str(gen.PROJECT_ROOT / "video_pair_generation" / "scripts" / "watch_run_resources.py"),
            "--run-dir",
            str(run_dir),
            "--interval",
            str(interval),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_watcher(watcher: subprocess.Popen[bytes] | None) -> None:
    if watcher is not None and watcher.poll() is None:
        watcher.terminate()
        try:
            watcher.wait(timeout=10)
        except subprocess.TimeoutExpired:
            watcher.kill()
            watcher.wait()


def stop_resource_watcher(run_dir: Path, watcher: subprocess.Popen[bytes] | None) -> None:
    stop_watcher(watcher)
    subprocess.run(
        [
            sys.executable,
            str(gen.PROJECT_ROOT / "video_pair_generation" / "scripts" / "watch_run_resources.py"),
            "--run-dir",
            str(run_dir),
            "--once",
        ],
        check=False,
    )


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
    encode_after_render: bool,
    cleanup_frames_after_encode: bool,
) -> None:
    missing = [
        job
        for job in jobs_payload["jobs"]
        if not frame_job_complete(job, rows) and not video_job_complete(job, rows)
    ]
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
        command = [
            str(blender_bin),
            "--background",
            *gen.blender_thread_args(),
            "--python",
            str(Path(gen.__file__).resolve()),
            "--",
            "--render-jobs",
            str(resume_jobs_path),
        ]
        if encode_after_render:
            command.append("--encode-after-render")
        if cleanup_frames_after_encode:
            command.append("--cleanup-frames-after-encode")
        subprocess.run(
            command,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
    log_event(run_dir, f"render: finished {len(missing)} missing frame jobs")


def encode_missing(
    run_dir: Path,
    jobs_payload: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    *,
    cleanup_frames_after_encode: bool,
) -> None:
    ready = [
        job
        for job in jobs_payload["jobs"]
        if not video_job_complete(job, rows) and frame_job_complete(job, rows)
    ]
    if not ready:
        log_event(run_dir, "encode: no completed frame jobs waiting for video encoding")
        return
    encode_jobs_path = run_dir / "render_jobs.resume_videos.json"
    write_jobs(encode_jobs_path, jobs_payload["run_id"], int(jobs_payload["fps"]), ready)
    log_event(run_dir, f"encode: starting {len(ready)} video jobs")
    fps = int(jobs_payload["fps"])
    for index, job in enumerate(ready, start=1):
        log_event(run_dir, f"encode: [{index}/{len(ready)}] starting {job['clip_id']}")
        video_path = gen.encode_video_job(job, fps)
        if cleanup_frames_after_encode:
            gen.cleanup_job_frames(job)
        log_event(
            run_dir,
            f"encode: [{index}/{len(ready)}] finished {job['clip_id']} bytes={video_path.stat().st_size} "
            f"frames_cleaned={cleanup_frames_after_encode}",
        )
    log_event(run_dir, f"encode: finished {len(ready)} video jobs")


def cleanup_completed_frames(
    run_dir: Path,
    jobs_payload: dict[str, Any],
    rows: dict[str, dict[str, Any]],
) -> int:
    cleaned = 0
    for job in jobs_payload["jobs"]:
        frames_dir = Path(job["frames_dir"])
        if not video_job_complete(job, rows) or not frames_dir.exists():
            continue
        gen.validate_video_job(job)
        gen.cleanup_job_frames(job)
        cleaned += 1
    if cleaned:
        log_event(run_dir, f"cleanup: removed retained frames for {cleaned} completed videos")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--blender-bin", type=Path, default=gen.DEFAULT_BLENDER)
    parser.add_argument("--kubric-site-packages", type=Path, default=gen.DEFAULT_KUBRIC_SITE_PACKAGES)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--progress-watch-interval", type=float, default=60.0)
    parser.add_argument("--resource-watch-interval", type=float, default=10.0)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    log_event(args.run_dir, "resume: started")
    progress_watcher = start_progress_watcher(args.run_dir, args.python_bin, args.progress_watch_interval)
    resource_watcher = start_resource_watcher(args.run_dir, args.resource_watch_interval)
    try:
        jobs_payload = read_jobs(args.run_dir)
        progress = write_progress(args.run_dir, args.python_bin)
        log_event(
            args.run_dir,
            f"resume: initial phase={progress['phase']} frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
            f"videos={progress['videos_complete']}/{progress['jobs_total']}",
        )
        rows = {row["clip_id"]: row for row in progress["jobs"]}
        if not args.keep_frames and cleanup_completed_frames(args.run_dir, jobs_payload, rows):
            progress = write_progress(args.run_dir, args.python_bin)
            rows = {row["clip_id"]: row for row in progress["jobs"]}
        if not args.skip_encode:
            encode_missing(
                args.run_dir,
                jobs_payload,
                rows,
                cleanup_frames_after_encode=not args.keep_frames,
            )
            progress = write_progress(args.run_dir, args.python_bin)
            rows = {row["clip_id"]: row for row in progress["jobs"]}
        if not args.skip_render:
            render_missing(
                run_dir=args.run_dir,
                blender_bin=args.blender_bin,
                kubric_site_packages=args.kubric_site_packages,
                jobs_payload=jobs_payload,
                rows=rows,
                encode_after_render=not args.skip_encode,
                cleanup_frames_after_encode=not args.skip_encode and not args.keep_frames,
            )
            progress = write_progress(args.run_dir, args.python_bin)
            log_event(
                args.run_dir,
                f"resume: after render phase={progress['phase']} frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
                f"videos={progress['videos_complete']}/{progress['jobs_total']}",
            )
            rows = {row["clip_id"]: row for row in progress["jobs"]}
        if not args.skip_encode:
            encode_missing(
                args.run_dir,
                jobs_payload,
                rows,
                cleanup_frames_after_encode=not args.keep_frames,
            )
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
    finally:
        stop_resource_watcher(args.run_dir, resource_watcher)
        stop_watcher(progress_watcher)
        write_progress(args.run_dir, args.python_bin)


if __name__ == "__main__":
    main()
