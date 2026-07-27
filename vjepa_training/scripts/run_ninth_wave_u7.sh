#!/usr/bin/env bash
set -euo pipefail

task_root="/workspace/writeable/code/camera_motion_disentangle/vjepa_training"
python_bin="/workspace/writeable/environments/camera_motion_disentangle/bin/python"
run_dir="/workspace/writeable/checkpoints/camera_motion_disentangle/ninth_wave/u7_identity_ae768_d2_convergence_seed17"
config="$task_root/configs/vjepa21_camxtime_u7_ae768_d2_convergence.yaml"
baseline="/workspace/writeable/checkpoints/camera_motion_disentangle/seventh_wave/u3_identity_ae768_d2_seed17/step_0002000.pt"

mkdir -p "$run_dir"
cp "$config" "$run_dir/config.yaml"
resume="$baseline"
if [[ -s "$run_dir/latest.pt" ]]; then
    resume="$run_dir/latest.pt"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/train_feature_autoencoder_upper_bound.py" \
    --config "$run_dir/config.yaml" --resume "$resume" 2>&1 | tee -a "$run_dir/train.log"

PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/evaluate_camera_interventions.py" \
    --config "$run_dir/config.yaml" \
    --checkpoint "U7_BEST=$run_dir/best.pt" \
    --checkpoint "U7_LATEST=$run_dir/latest.pt" \
    --samples 16 --seed 10017 \
    --output "$run_dir/identity_upper_bound_seed10017_n16.json" \
    2>&1 | tee "$run_dir/identity_upper_bound.log"
