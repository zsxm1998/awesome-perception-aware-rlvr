#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2 (8B -> 2B, ViRL39K, 5 epochs = about 12,150 steps): Standard OPD, the full-vocabulary reverse KL
# averaged per response, then over responses (Eq. 1), so that VA-OPD with uniform weights reduces to it.
pr_default VA_OPD_DATA "virl39k"
pr_default TEACHER_PATH "Qwen/Qwen3-VL-8B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_opd_virl39k"
ALGO_ARGS=(
    "worker.actor.loss_avg_mode=seq"
)
EXTRA_ARGS=()

launch_va_opd_distillation "$@"
