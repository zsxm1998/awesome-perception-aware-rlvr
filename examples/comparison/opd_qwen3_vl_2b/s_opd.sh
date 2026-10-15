#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# S-OPD: opd_sampled.sh plus two student-side terms with the weights of the paper's main experiments.
# TPC (0.02): the k3 between the student's log-probs on the image and on a masked copy (16 px blocks, each
# blackened with probability 0.6), weighted per token by the teacher's log-prob drop
# max(log q(y_t | image) - log q(y_t | masked), 0) on the same copy, averaged over the tokens with a drop, and
# maximized. PA (0.02): the k3 to a copy with Gaussian noise (std 0.2 on [0, 1] pixels), minimized. Both copies are
# drawn on the resized image, one per response, as the comparison's other views.
pr_default EXPERIMENT_NAME "s_opd"
ALGO_ARGS=(
    "algorithm.adv_estimator=teacher_log_ratio"
    "algorithm.teacher_log_ratio_clip=10"
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_kwargs={\"patch_size\":16,\"black_prob\":0.6}"
    "algorithm.corrupt_image_position=response"
    "algorithm.visual_sensitivity_metric=sampled_low_var_kl"
    "algorithm.visual_sensitivity_reference=current"
    "algorithm.visual_sensitivity_loss_coef=0.02"
    "algorithm.visual_sensitivity_loss_weighting=teacher"
    "algorithm.incremental_image_transform=rgb_gaussian_noise"
    "algorithm.incremental_image_kwargs={\"std\":0.2}"
    "algorithm.visual_robustness_loss_coef=0.02"
)
EXTRA_ARGS=()

launch_opd_comparison "$@"
