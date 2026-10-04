#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1 baseline (Geometry3K): PAPO on GRPO, with the settings of examples/reproduction/papo (one mask of
# 16-px patches blackened with p=0.6 per prompt, on the original image; KL_prcp coefficient 0.02).
pr_default VA_OPD_DATA "geo3k"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_papo"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.corrupt_image_kwargs={\"patch_size\":16,\"black_prob\":0.6,\"mask_before_resize\":true}"
    "algorithm.visual_sensitivity_loss_coef=0.02"
)
EXTRA_ARGS=()

launch_va_opd_rl "$@"
