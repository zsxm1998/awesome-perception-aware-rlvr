#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# VA-OPD: opd_full.sh with per-token weights from the teacher's visual advantage
# max(log q(y_t | image) - log q(y_t | pixelated image), 0): responses weighted by softmax of the z-scored
# mean advantage within their prompt (tau 1), the top 20% tokens of a response sharing half its weight.
pr_default EXPERIMENT_NAME "va_opd"
ALGO_ARGS=(
    "algorithm.distill_loss_coef=1.0"
    "algorithm.policy_loss_coef=0.0"
    "algorithm.distill_divergence=reverse_kl"
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

launch_opd_comparison "$@"
