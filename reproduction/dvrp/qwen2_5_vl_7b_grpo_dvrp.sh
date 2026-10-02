#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_grpo_dvrp"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.corrupt_image_kwargs={\"patch_size\":14,\"black_prob\":0.6}"
    "algorithm.incremental_image_transform=vp_diffusion"
    "algorithm.incremental_image_kwargs={}"
    "algorithm.visual_sensitivity_loss_coef=0.01"
    "algorithm.visual_robustness_loss_coef=0.01"
    "algorithm.decremental_entropy_coef=0.05"
    "algorithm.incremental_entropy_coef=0.05"
    "algorithm.entropy_loss_type=sampled"
    "algorithm.noise_t_init=500"
    "algorithm.noise_gamma=10"
    "algorithm.noise_t_max=1000"
)
EXTRA_ARGS=(
    "algorithm.use_kl_loss=false"
    "algorithm.disable_kl=true"
)

launch_dvrp_matrix "$@"
