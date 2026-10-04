#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# OPD on the full next-token distributions: per-token reverse KL(p || q) over the whole vocabulary as the
# only loss (VA-OPD's and VGS's "Standard OPD").
pr_default EXPERIMENT_NAME "opd_full"
ALGO_ARGS=(
    "algorithm.distill_loss_coef=1.0"
    "algorithm.policy_loss_coef=0.0"
    "algorithm.distill_divergence=reverse_kl"
)
EXTRA_ARGS=()

launch_opd_comparison "$@"
