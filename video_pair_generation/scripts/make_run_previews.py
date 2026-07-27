#!/usr/bin/env python3
"""Create browser-free previews for a generated run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "site" / "assets" / "runs" / "mock_pair_v0"


def sample_video(video_path: Path, samples: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [round(i * max(0, frame_count - 1) / max(1, samples - 1)) for i in range(samples)]
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    capture.release()
    return frames


def make_contact_sheet(video_path: Path, out_path: Path, title: str, samples: int, thumb_width: int) -> None:
    frames = sample_video(video_path, samples)
    if not frames:
        raise RuntimeError(f"no frames sampled from {video_path}")
    thumbs = []
    for frame in frames:
        scale = thumb_width / frame.width
        thumbs.append(frame.resize((thumb_width, round(frame.height * scale))))
    label_h = 34
    width = thumb_width * len(thumbs)
    height = label_h + max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (width, height), (245, 247, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(31, 37, 40))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * thumb_width, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_summary(run_dir: Path, manifest: dict[str, object]) -> None:
    lines = [
        f"# {manifest['run_id']}",
        "",
        f"Generator: `{manifest.get('generator', 'unknown')}`",
        "",
        "## Pair Groups",
        "",
    ]
    for group in manifest["pair_groups"]:
        lines.extend(
            [
                f"- `{group['group_id']}`: controlled `{group['controlled_factor']}`, varied `{group['varied_factor']}`",
                f"  clips: {', '.join(group['clip_ids'])}",
            ]
        )
    lines.extend(["", "## Clips", ""])
    for clip in manifest["clips"]:
        lines.append(f"- `{clip['clip_id']}`: camera `{clip['camera_id']}`, physics `{clip['physics_id']}`")
        lines.append(f"  preview: `previews/{clip['clip_id']}_sheet.png`")
        lines.append(f"  video: `{clip['video']}`")
        lines.append(f"  metadata: `{clip['metadata']}`")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=180)
    args = parser.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for clip in manifest["clips"]:
        video_path = args.run_dir / clip["video"]
        out_path = args.run_dir / "previews" / f"{clip['clip_id']}_sheet.png"
        make_contact_sheet(video_path, out_path, clip["clip_id"], args.samples, args.thumb_width)
    write_summary(args.run_dir, manifest)
    print(f"Wrote previews and SUMMARY.md under {args.run_dir}")


if __name__ == "__main__":
    main()
