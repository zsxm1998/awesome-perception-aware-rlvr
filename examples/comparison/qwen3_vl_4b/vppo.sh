#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "vppo"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_kwargs={\"patch_size\":16,\"black_prob\":0.5}"
    "algorithm.corrupt_image_position=response"
    "algorithm.top_perception_quantile=0.4"
    "algorithm.perception_thr_granularity=response"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.response_advantage_scaling_method=vppo"
    "algorithm.vppo_response_scaling_min=0.9"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
