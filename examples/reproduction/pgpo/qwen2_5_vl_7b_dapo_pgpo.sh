#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_dapo_pgpo"
ALGO_ARGS=(
    "algorithm.corrupt_image=mask_visual_attention"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.visual_sensitivity_metric=sampled_low_var_kl"
    "algorithm.advantage_scaling_method=pgpo"
    "algorithm.pgpo_token_scaling_threshold=0.4"
    "algorithm.pgpo_token_scaling_boost=2.0"
)
EXTRA_ARGS=(
    "worker.actor.clip_ratio_low=0.2"
    "worker.actor.clip_ratio_high=0.28"
)

launch_pgpo_matrix "$@"
