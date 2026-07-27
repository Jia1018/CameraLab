#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/workspace/writeable/environments/camera_motion_disentangle/bin/python"
checkpoint_root="/workspace/writeable/checkpoints/camera_motion_disentangle/first_wave"

run_experiment() {
  local experiment_id="$1"
  local config_name="$2"
  local output_dir="$checkpoint_root/$experiment_id"

  mkdir -p "$output_dir"
  cp "$task_root/configs/$config_name" "$output_dir/config.yaml"
  PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/train_vjepa21_camxtime.py" \
    --config "$task_root/configs/$config_name" \
    2>&1 | tee "$output_dir/train.log"
}

run_experiment "a1_swap_invariance_seed17" "vjepa21_camxtime_a1_500.yaml"
run_experiment "a0_swap_only_seed17" "vjepa21_camxtime_a0_500.yaml"

