#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2: 0.7 GRPO + 0.3 VGS (Eq. 22, without eta) with the teacher of stage 1.
default_vgs_teacher
pr_default EXPERIMENT_NAME "qwen3_vl_2b_grpo_vgs"
ALGO_ARGS=(
    "worker.teacher.source=model"
    "worker.teacher.model.model_path=$TEACHER_PATH"
    "algorithm.policy_loss_coef=0.7"
    "algorithm.distill_loss_coef=0.3"
    "algorithm.distill_divergence=reverse_kl"
    "algorithm.distill_target=visual_gain"
    "algorithm.distill_contrast_view=no_image"
    "algorithm.vgs_steering_coef=2.0"
    "algorithm.vgs_text_prior_coef=0.01"
    "algorithm.vgs_vds_quantile=0.7"
    "algorithm.vgs_vds_scope=micro_batch"
    "algorithm.vgs_loss_scale=1.0"
)
EXTRA_ARGS=()

launch_vgs_grpo_combination "$@"
