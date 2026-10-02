#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_dapo"
ALGO_ARGS=(
    "algorithm.invariant_entropy_coef=0.06"
    "algorithm.entropy_loss_type=sampled"
)
EXTRA_ARGS=()

launch_vppo_matrix "$@"
