#!/usr/bin/env bash
set -euo pipefail

task_root="/workspace/writeable/code/camera_motion_disentangle/vjepa_training"
python_bin="/workspace/writeable/environments/camera_motion_disentangle/bin/python"
run_dir="/workspace/writeable/checkpoints/camera_motion_disentangle/tenth_wave/u8_identity_ae768_d2_lowlr_seed17"
config="$task_root/configs/vjepa21_camxtime_u8_ae768_d2_lowlr_convergence.yaml"
source_best="/workspace/writeable/checkpoints/camera_motion_disentangle/ninth_wave/u7_identity_ae768_d2_convergence_seed17/best.pt"

mkdir -p "$run_dir"
cp "$config" "$run_dir/config.yaml"
train_args=(--config "$run_dir/config.yaml")
if [[ -s "$run_dir/latest.pt" ]]; then
    train_args+=(--resume "$run_dir/latest.pt")
else
    train_args+=(--initialize "$source_best")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/train_feature_autoencoder_upper_bound.py" \
    "${train_args[@]}" 2>&1 | tee -a "$run_dir/train.log"

PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/evaluate_camera_interventions.py" \
    --config "$run_dir/config.yaml" \
    --checkpoint "U8_BEST=$run_dir/best.pt" \
    --checkpoint "U8_LATEST=$run_dir/latest.pt" \
    --samples 16 --seed 10017 \
    --output "$run_dir/identity_upper_bound_seed10017_n16.json" \
    2>&1 | tee "$run_dir/identity_upper_bound.log"
