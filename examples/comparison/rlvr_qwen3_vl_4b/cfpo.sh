#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "cfpo"
ALGO_ARGS=(
    "algorithm.corrupt_image=cross_modal_attention_value_mean"
    "algorithm.corrupt_image_kwargs={\"saliency_std_multiplier\":2.0}"
    "algorithm.visual_sensitivity_metric=sampled_low_var_kl"
    "algorithm.visual_sensitivity_reference=current"
    "algorithm.visual_sensitivity_loss_coef=0.02"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
