#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2 (8B -> 2B, ViRL39K, 5 epochs = about 12,150 steps): VA-OPD.
pr_default VA_OPD_DATA "virl39k"
pr_default TEACHER_PATH "Qwen/Qwen3-VL-8B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_va_opd_virl39k"
ALGO_ARGS=(
    "worker.actor.loss_avg_mode=token"
    "algorithm.distill_weighting=va_opd"
    "algorithm.corrupt_image=pixelation"
    "algorithm.corrupt_image_kwargs={\"ratio\":0.1}"
    "algorithm.visual_sensitivity_reference=teacher"
    "algorithm.visual_sensitivity_metric=sampled_positive_log_ratio"
    "algorithm.va_opd_softmax_temperature=1.0"
    "algorithm.va_opd_high_fraction=0.2"
    "algorithm.va_opd_high_weight=0.5"
)
EXTRA_ARGS=()

launch_va_opd_distillation "$@"
