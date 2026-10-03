#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_grpo_vepo"
ALGO_ARGS=(
    "algorithm.corrupt_image=gaussian_noise"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.corrupt_image_kwargs={\"std\":6.928}"
    "algorithm.visual_sensitivity_metric=vepo"
    "algorithm.visual_sensitivity_jsd_weight=0.7"
    "algorithm.visual_sensitivity_entropy_gate=normal_entropy"
    "algorithm.top_perception_quantile=0.2"
    "algorithm.perception_thr_granularity=response"
    "algorithm.normalize_pg_loss_by_selected_tokens=true"
)
EXTRA_ARGS=()

launch_full_vocab_sensitivity_geo3k "$@"
