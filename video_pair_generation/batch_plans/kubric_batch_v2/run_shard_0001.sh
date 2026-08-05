#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

RUN_ID="${RUN_ID:-kubric_batch_v2_blocks_shard_0001}"
RUN_ROOT="${RUN_ROOT:-/workspace/writeable/datasets/camera_motion_disentangle}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
REVIEW_RUN_ID="${REVIEW_RUN_ID:-${RUN_ID}_review_sample}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/writeable/environments/kubric_official/bin/python}"
PREVIEW_PYTHON_BIN="${PREVIEW_PYTHON_BIN:-/workspace/writeable/environments/kubric_review/bin/python}"
BLENDER_BIN="${BLENDER_BIN:-/workspace/writeable/blender-3.6.5-linux-x64/blender}"
KUBRIC_SITE_PACKAGES="${KUBRIC_SITE_PACKAGES:-/workspace/writeable/environments/kubric_official/lib/python3.10/site-packages}"
DRIVER_LOG="${DRIVER_LOG:-${RUN_ROOT}/${RUN_ID}.driver.log}"

export KUBRIC_BLENDER_THREADS="${KUBRIC_BLENDER_THREADS:-4}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

mkdir -p "${RUN_ROOT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${DRIVER_LOG}"
}

log "driver started run_id=${RUN_ID} blender_threads=${KUBRIC_BLENDER_THREADS}"

if [[ -f "${RUN_DIR}/render_jobs.json" && -f "${RUN_DIR}/manifest.json" ]]; then
  log "audit plan already exists; preserving it for resume"
else
  log "starting no-render physics and framing audit"
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/video_pair_generation/scripts/generate_kubric_batch_v2.py" \
    --run-id "${RUN_ID}" \
    --run-root "${RUN_ROOT}" \
    --groups 200 \
    --group-mode shared_factor_blocks \
    --block-size 4 \
    --width 640 \
    --height 480 \
    --seed 20260731 \
    --frames-mean 108 \
    --frames-std 24 \
    --frames-min 72 \
    --frames-max 168 \
    --frame-multiple 12 \
    --no-render \
    --overwrite \
    >>"${DRIVER_LOG}" 2>&1
  log "no-render audit finished"
fi

"${PYTHON_BIN}" - "${RUN_DIR}/manifest.json" "${RUN_ID}" >>"${DRIVER_LOG}" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_id = sys.argv[2]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
coverage = manifest["coverage_summary"]
groups = manifest["pair_groups"]
clips = manifest["clips"]

assert manifest["run_id"] == run_id
assert manifest["fps"] == 24
assert manifest["resolution"] == [640, 480]
assert manifest["seed"] == 20260731
assert len(groups) == 200
assert len(clips) == 800
assert all(len(group["clip_ids"]) == 4 for group in groups)
assert coverage["pair_kinds"] == {"same_camera": 100, "same_physics": 100}
assert set(coverage["camera_families"]) == set(manifest["camera_family_reference"])
assert set(coverage["physics_families"]) == set(manifest["physics_family_reference"])
assert coverage["frames_min"] >= 72
assert coverage["frames_max"] <= 168

print(
    "audit assertions passed: "
    f"clips={len(clips)} groups={len(groups)} "
    f"camera_families={len(coverage['camera_families'])} "
    f"physics_families={len(coverage['physics_families'])} "
    f"frames={coverage['frames_min']}-{coverage['frames_max']}"
)
PY

log "starting or resuming Blender render with inline MP4 encoding and PNG cleanup"
"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/video_pair_generation/scripts/resume_kubric_run.py" \
  --run-dir "${RUN_DIR}" \
  --blender-bin "${BLENDER_BIN}" \
  --kubric-site-packages "${KUBRIC_SITE_PACKAGES}" \
  --python-bin "${PYTHON_BIN}" \
  --progress-watch-interval 60 \
  --resource-watch-interval 10 \
  >>"${DRIVER_LOG}" 2>&1
log "render and encoding finished"

log "exporting 24 review groups to ${REVIEW_RUN_ID}"
"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/video_pair_generation/scripts/finalize_kubric_review_sample.py" \
  --source-run-dir "${RUN_DIR}" \
  --dest-run-id "${REVIEW_RUN_ID}" \
  --pairs 24 \
  --seed 20260731 \
  --poll-interval 300 \
  --python-bin "${PYTHON_BIN}" \
  --preview-python-bin "${PREVIEW_PYTHON_BIN}" \
  --overwrite \
  >>"${DRIVER_LOG}" 2>&1

log "driver complete; review synced to docs"
