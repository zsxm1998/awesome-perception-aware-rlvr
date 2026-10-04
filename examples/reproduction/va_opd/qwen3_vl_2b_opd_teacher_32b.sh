#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2 (32B -> 2B, Geometry3K): Standard OPD. The 32B teacher needs 80 GB GPUs (about 8 GB of it per GPU).
pr_default VA_OPD_DATA "geo3k"
pr_default TEACHER_PATH "Qwen/Qwen3-VL-32B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_opd_teacher_32b"
ALGO_ARGS=(
    "worker.actor.loss_avg_mode=seq"
)
EXTRA_ARGS=()

launch_va_opd_distillation "$@"
