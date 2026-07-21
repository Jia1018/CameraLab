#!/usr/bin/env python3
"""Wait for a Kubric run to finish, then export a compact site review sample."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_PYTHON = Path("/workspace/writeable/environments/kubric_review/bin/python")
DEFAULT_OFFICIAL_PYTHON = Path("/workspace/writeable/environments/kubric_official/bin/python")


def log(log_path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        print(f"{timestamp} {message}", file=log_file, flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_progress(run_dir: Path, python_bin: Path, log_path: Path) -> dict[str, Any]:
    subprocess.run(
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "update_kubric_progress.py"),
            "--run-dir",
            str(run_dir),
            "--python-bin",
            str(python_bin),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    progress = read_json(run_dir / "progress.json")
    log(
        log_path,
        f"progress phase={progress['phase']} frames={progress['frames_rendered_total']}/{progress['frames_expected_total']} "
        f"videos={progress['videos_complete']}/{progress['jobs_total']}",
    )
    return progress


def wait_until_complete(run_dir: Path, python_bin: Path, interval: float, log_path: Path) -> None:
    while True:
        progress = update_progress(run_dir, python_bin, log_path)
        if progress.get("complete"):
            return
        time.sleep(interval)


def run_step(command: list[str], log_path: Path) -> None:
    log(log_path, "running " + " ".join(command))
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--dest-run-id", required=True)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--poll-interval", type=float, default=300.0)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_OFFICIAL_PYTHON)
    parser.add_argument("--preview-python-bin", type=Path, default=DEFAULT_REVIEW_PYTHON)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be positive")

    log_path = args.source_run_dir / "finalize_review_sample.log"
    dest_run_dir = PROJECT_ROOT / "site" / "assets" / "runs" / args.dest_run_id
    log(log_path, f"finalize: waiting for {args.source_run_dir}")
    wait_until_complete(args.source_run_dir, args.python_bin, args.poll_interval, log_path)

    export_cmd = [
        str(args.python_bin),
        str(PROJECT_ROOT / "scripts" / "export_kubric_review_sample.py"),
        "--source-run-dir",
        str(args.source_run_dir),
        "--dest-run-id",
        args.dest_run_id,
        "--pairs",
        str(args.pairs),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        export_cmd.append("--overwrite")
    run_step(export_cmd, log_path)

    run_step(
        [
            str(args.preview_python_bin),
            str(PROJECT_ROOT / "scripts" / "make_run_previews.py"),
            "--run-dir",
            str(dest_run_dir),
        ],
        log_path,
    )
    run_step(["bash", str(PROJECT_ROOT / "scripts" / "sync_site_to_docs.sh")], log_path)
    log(log_path, f"finalize: wrote review sample {dest_run_dir}")


if __name__ == "__main__":
    main()
