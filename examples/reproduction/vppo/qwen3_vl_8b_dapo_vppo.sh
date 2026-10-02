#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen3-VL-8B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_8b_dapo_vppo"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_position=response"
    "algorithm.corrupt_image_kwargs={\"patch_size\":16,\"black_prob\":0.5}"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.top_perception_quantile=0.4"
    "algorithm.perception_thr_granularity=response"
    "algorithm.response_advantage_scaling_method=vppo"
    "algorithm.vppo_response_scaling_min=0.9"
    "algorithm.invariant_entropy_coef=0.12"
    "algorithm.entropy_loss_type=sampled"
)
EXTRA_ARGS=(
    "data.max_response_length=8192"
    "worker.actor.micro_batch_size_per_device_for_update=4"
    "worker.actor.micro_batch_size_per_device_for_experience=8"
    "worker.rollout.max_num_batched_tokens=12289"
    "trainer.max_steps=130"
)

launch_vppo_matrix "$@"
