#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Optional ablation, not in the results table: opd_full.sh averaged per response, then over responses
# (seq-mean-token-mean, the papers' formulas) instead of over all tokens. VA-OPD normalizes its weights
# per prompt; this run separates that part of its difference from opd_full.sh.
pr_default EXPERIMENT_NAME "opd_full_seq"
ALGO_ARGS=(
    "algorithm.distill_loss_coef=1.0"
    "algorithm.policy_loss_coef=0.0"
    "algorithm.distill_divergence=reverse_kl"
    "worker.actor.loss_avg_mode=seq"
)
EXTRA_ARGS=()

launch_opd_comparison "$@"
