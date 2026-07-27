#!/usr/bin/env bash
set -euo pipefail

python_bin="/workspace/writeable/environments/camera_motion_disentangle/bin/python"
task_root="/workspace/writeable/code/camera_motion_disentangle/vjepa_training"
root="/workspace/writeable/checkpoints/camera_motion_disentangle/eighth_wave"
timeout_seconds=21600

wait_and_prune() {
    local run_dir="$1"
    local final_model_name="$2"
    local intermediate_model_name="$3"
    local final_checkpoint="$run_dir/step_0002000.pt"
    local intermediate_checkpoint="$run_dir/step_0001000.pt"
    local final_result_json="$run_dir/identity_upper_bound_seed10017_n16.json"
    local intermediate_result_json="$run_dir/identity_upper_bound_step1000_seed10017_n16.json"
    local deadline=$((SECONDS + timeout_seconds))

    while [[ ! -s "$final_checkpoint" || ! -s "$final_result_json" ]]; do
        if (( SECONDS >= deadline )); then
            printf 'cleanup timeout: %s\n' "$run_dir" >&2
            return 1
        fi
        sleep 60
    done

    "$python_bin" -c 'import json, sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert sys.argv[2] in payload["models"]' "$final_result_json" "$final_model_name"
    if [[ ! -s "$intermediate_result_json" ]]; then
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$python_bin" "$task_root/scripts/evaluate_camera_interventions.py" --config "$run_dir/config.yaml" --checkpoint "$intermediate_model_name=$intermediate_checkpoint" --samples 16 --seed 10017 --output "$intermediate_result_json"
    fi
    "$python_bin" -c 'import json, sys; payload=json.load(open(sys.argv[1], encoding="utf-8")); assert sys.argv[2] in payload["models"]' "$intermediate_result_json" "$intermediate_model_name"
    rm -f "$intermediate_checkpoint"
    printf 'pruned=%s\n' "$intermediate_checkpoint"
}

wait_and_prune "$root/u5_identity_ae768_d2_ema_seed17" "U5_EMA" "U5_EMA_1000"
wait_and_prune "$root/u6_identity_ae768_d2_relation_seed17" "U6_REL" "U6_REL_1000"
