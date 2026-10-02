#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_grpo_papo"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.corrupt_image_kwargs={\"patch_size\":14,\"black_prob\":0.6}"
    "algorithm.visual_sensitivity_loss_coef=0.02"
)
EXTRA_ARGS=(
    "trainer.n_gpus_per_node=2"
)

launch_papo_matrix "$@"
