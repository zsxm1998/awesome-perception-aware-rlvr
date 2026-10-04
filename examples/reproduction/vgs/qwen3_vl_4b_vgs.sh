#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 4 / Table 1 (4B student): VGS with gamma 2, lambda 0.01 and eta 0.36.
pr_default MODEL_PATH "Qwen/Qwen3-VL-4B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_4b_vgs"
ALGO_ARGS=(
    "algorithm.distill_target=visual_gain"
    "algorithm.distill_contrast_view=no_image"
    "algorithm.vgs_steering_coef=2.0"
    "algorithm.vgs_text_prior_coef=0.01"
    "algorithm.vgs_vds_quantile=0.7"
    "algorithm.vgs_vds_scope=micro_batch"
    "algorithm.vgs_loss_scale=0.36"
)
EXTRA_ARGS=()

launch_vgs_student "$@"
