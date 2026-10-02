#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_cfpo_d"
ALGO_ARGS=(
    "algorithm.corrupt_image=cross_modal_attention_value_mean"
    "algorithm.corrupt_image_kwargs={\"saliency_std_multiplier\":2.0}"
    "algorithm.visual_sensitivity_metric=sampled_low_var_kl"
    "algorithm.visual_sensitivity_reference=current"
    "algorithm.visual_sensitivity_loss_coef=0.01"
    "algorithm.invariant_entropy_coef=0.03"
    "algorithm.entropy_loss_type=sampled"
)
EXTRA_ARGS=(
    "data.mini_rollout_batch_size=128"
    "worker.actor.clip_ratio_high=0.28"
    "algorithm.disable_kl=true"
    "algorithm.use_kl_loss=false"
    "algorithm.kl_coef=0.0"
    "algorithm.online_filtering=true"
    "algorithm.filter_key=accuracy"
    "algorithm.filter_low=0.01"
    "algorithm.filter_high=0.99"
)

launch_original_cfpo_matrix "$@"
