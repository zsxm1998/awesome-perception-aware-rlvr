#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "pgpo"
ALGO_ARGS=(
    "algorithm.corrupt_image=mask_visual_attention"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.visual_sensitivity_metric=sampled_low_var_kl"
    "algorithm.advantage_scaling_method=pgpo"
    "algorithm.pgpo_token_scaling_threshold=0.4"
    "algorithm.pgpo_token_scaling_boost=2.0"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
