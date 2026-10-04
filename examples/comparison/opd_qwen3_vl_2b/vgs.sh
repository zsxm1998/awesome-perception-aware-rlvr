#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# VGS: eta * (KL(p || q) + gamma KL(p || q*) + lambda gate KL(p_text || q_text)), q* ~ sg(p_text) q / q_text,
# the gate on the top 30% tokens by the teacher's KL(q || q_text) in each micro-batch; gamma 2, lambda 0.01,
# eta 0.41 (the paper's value for a 2B student).
pr_default EXPERIMENT_NAME "vgs"
ALGO_ARGS=(
    "algorithm.distill_loss_coef=1.0"
    "algorithm.policy_loss_coef=0.0"
    "algorithm.distill_divergence=reverse_kl"
    "algorithm.distill_target=visual_gain"
    "algorithm.distill_contrast_view=no_image"
    "algorithm.vgs_steering_coef=2.0"
    "algorithm.vgs_text_prior_coef=0.01"
    "algorithm.vgs_vds_quantile=0.7"
    "algorithm.vgs_vds_scope=micro_batch"
    "algorithm.vgs_loss_scale=0.41"
)
EXTRA_ARGS=()

launch_opd_comparison "$@"
