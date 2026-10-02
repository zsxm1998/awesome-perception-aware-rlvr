#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_dapo_dvrp"
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
    "data.mini_rollout_batch_size=128"
    "worker.actor.clip_ratio_low=0.2"
    "worker.actor.clip_ratio_high=0.28"
    "algorithm.use_kl_loss=false"
    "algorithm.disable_kl=true"
    "algorithm.online_filtering=true"
    "algorithm.filter_key=accuracy"
    "algorithm.filter_low=0.01"
    "algorithm.filter_high=0.99"
)

launch_dvrp_matrix "$@"
