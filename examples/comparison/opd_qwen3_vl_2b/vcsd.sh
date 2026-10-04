#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# VCSD: self-distillation from an EMA of the student (rate 0.05, no external teacher), toward the teacher's
# distribution sharpened by its contrast with a black image of the same size (alpha 1, support
# q >= 0.1 max q, end-of-sequence tokens not contrasted), with forward KL at temperature 2 times 4.
pr_default EXPERIMENT_NAME "vcsd"
ALGO_ARGS=(
    "worker.teacher.source=ema"
    "worker.teacher.model.model_path=null"
    "worker.teacher.ema_rate=0.05"
    "algorithm.distill_loss_coef=1.0"
    "algorithm.policy_loss_coef=0.0"
    "algorithm.distill_target=contrast_sharpened"
    "algorithm.distill_contrast_view=black"
    "algorithm.distill_divergence=forward_kl"
    "algorithm.distill_temperature=2.0"
    "algorithm.distill_temperature_scope=all"
    "algorithm.vcsd_alpha=1.0"
    "algorithm.vcsd_support_beta=0.1"
    "algorithm.vcsd_anchor_coef=1.0"
    "algorithm.vcsd_keep_token_ids=auto"
)
EXTRA_ARGS=()

launch_opd_comparison "$@"
