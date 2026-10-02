#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_dapo_vppo"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_position=response"
    "algorithm.corrupt_image_kwargs={\"patch_size\":14,\"black_prob\":0.5}"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.top_perception_quantile=0.4"
    "algorithm.perception_thr_granularity=response"
    "algorithm.response_advantage_scaling_method=vppo"
    "algorithm.vppo_response_scaling_min=0.9"
    "algorithm.invariant_entropy_coef=0.06"
    "algorithm.entropy_loss_type=sampled"
)
EXTRA_ARGS=()

launch_vppo_matrix "$@"
