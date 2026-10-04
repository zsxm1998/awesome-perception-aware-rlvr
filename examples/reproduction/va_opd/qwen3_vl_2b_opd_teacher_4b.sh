#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2 (4B -> 2B, Geometry3K): Standard OPD, the full-vocabulary reverse KL
# averaged per response, then over responses (Eq. 1), so that VA-OPD with uniform weights reduces to it.
pr_default VA_OPD_DATA "geo3k"
pr_default TEACHER_PATH "Qwen/Qwen3-VL-4B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_opd_teacher_4b"
ALGO_ARGS=(
    "worker.actor.loss_avg_mode=seq"
)
EXTRA_ARGS=()

launch_va_opd_distillation "$@"
