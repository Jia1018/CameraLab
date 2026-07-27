#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/workspace/writeable/environments/camera_motion_disentangle/bin/python"
run_dir="/workspace/writeable/checkpoints/camera_motion_disentangle/sixth_wave/u1_identity_ae384_seed17"
config="$task_root/configs/vjepa21_camxtime_u1_ae384_500.yaml"

mkdir -p "$run_dir"
cp "$config" "$run_dir/config.yaml"

PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/train_feature_autoencoder_upper_bound.py" --config "$config" 2>&1 | tee "$run_dir/train.log"

PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/evaluate_camera_interventions.py" --config "$config" --checkpoint "B7=/workspace/writeable/checkpoints/camera_motion_disentangle/fourth_wave/b7_spatial_film_camdelta8_seed17/step_0000500.pt" --checkpoint "U1=$run_dir/step_0000500.pt" --samples 16 --seed 10017 --output "$run_dir/identity_upper_bound_seed10017_n16.json" 2>&1 | tee "$run_dir/identity_upper_bound.log"
