#!/usr/bin/env python3
"""Append lightweight resource snapshots for a long Kubric run."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CGROUP_ROOT = Path("/sys/fs/cgroup")
PROCESS_KEYWORDS = (
    "blender",
    "ffmpeg",
    "generate_kubric_batch_v2.py",
    "resume_kubric_run.py",
)


def read_int(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def read_events(path: Path) -> dict[str, int]:
    events: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            events[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return events


def read_progress(run_dir: Path) -> dict[str, Any]:
    progress_path = run_dir / "progress.json"
    if not progress_path.exists():
        return {}
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "phase": progress.get("phase"),
        "active_job": progress.get("active_job"),
        "frames_rendered_total": progress.get("frames_rendered_total"),
        "frames_expected_total": progress.get("frames_expected_total"),
        "videos_complete": progress.get("videos_complete"),
        "jobs_total": progress.get("jobs_total"),
    }


def proc_rss_kib(pid: str) -> int | None:
    status_path = Path("/proc") / pid / "status"
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else None
    except (OSError, ValueError):
        return None
    return None


def proc_cmdline(pid: str) -> str:
    try:
        raw = (Path("/proc") / pid / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def selected_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        cmdline = proc_cmdline(entry.name)
        if not cmdline or not any(keyword in cmdline for keyword in PROCESS_KEYWORDS):
            continue
        rows.append(
            {
                "pid": int(entry.name),
                "rss_kib": proc_rss_kib(entry.name),
                "cmdline": cmdline[:280],
            }
        )
    rows.sort(key=lambda row: int(row["rss_kib"] or 0), reverse=True)
    return rows[:12]


def snapshot(run_dir: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(run_dir)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "progress": read_progress(run_dir),
        "cgroup": {
            "memory_current_bytes": read_int(CGROUP_ROOT / "memory.current"),
            "memory_peak_bytes": read_int(CGROUP_ROOT / "memory.peak"),
            "memory_max_bytes": read_int(CGROUP_ROOT / "memory.max"),
            "memory_swap_max_bytes": read_int(CGROUP_ROOT / "memory.swap.max"),
            "memory_events": read_events(CGROUP_ROOT / "memory.events"),
        },
        "disk": {
            "path": str(run_dir),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "processes": selected_processes(),
    }


def append_snapshot(run_dir: Path) -> None:
    out_path = run_dir / "resource_usage.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as out_file:
        print(json.dumps(snapshot(run_dir), sort_keys=True), file=out_file, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be positive")

    while True:
        append_snapshot(args.run_dir)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
