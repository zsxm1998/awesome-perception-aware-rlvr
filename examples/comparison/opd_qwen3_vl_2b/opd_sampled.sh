#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# OPD from sampled tokens: advantage log q(y_t) - log pi_old(y_t) (clamped to +-10) with the PPO loss.
pr_default EXPERIMENT_NAME "opd_sampled"
ALGO_ARGS=(
    "algorithm.adv_estimator=teacher_log_ratio"
    "algorithm.teacher_log_ratio_clip=10"
)
EXTRA_ARGS=()

launch_opd_comparison "$@"
