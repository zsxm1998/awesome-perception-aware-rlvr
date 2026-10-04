#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2: 0.7 GRPO + 0.3 Standard OPD (Eq. 22) with the teacher of stage 1.
default_vgs_teacher
pr_default EXPERIMENT_NAME "qwen3_vl_2b_grpo_opd"
ALGO_ARGS=(
    "worker.teacher.source=model"
    "worker.teacher.model.model_path=$TEACHER_PATH"
    "algorithm.policy_loss_coef=0.7"
    "algorithm.distill_loss_coef=0.3"
    "algorithm.distill_divergence=reverse_kl"
)
EXTRA_ARGS=()

launch_vgs_grpo_combination "$@"
