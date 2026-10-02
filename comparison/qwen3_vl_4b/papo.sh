#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "papo"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_kwargs={\"patch_size\":16,\"black_prob\":0.6}"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.visual_sensitivity_loss_coef=0.01"
    "algorithm.visual_sensitivity_reference=current"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
