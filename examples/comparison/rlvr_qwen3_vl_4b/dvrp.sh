#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "dvrp"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_kwargs={\"patch_size\":16,\"black_prob\":0.6}"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.incremental_image_transform=vp_diffusion"
    "algorithm.visual_sensitivity_loss_coef=0.01"
    "algorithm.visual_sensitivity_reference=current"
    "algorithm.visual_robustness_loss_coef=0.01"
    "algorithm.noise_t_init=500"
    "algorithm.noise_gamma=10"
    "algorithm.noise_t_max=1000"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
