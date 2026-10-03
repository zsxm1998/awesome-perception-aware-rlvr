#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_grpo_tor"
ALGO_ARGS=(
    "algorithm.corrupt_image=no_image"
    "algorithm.top_entropy_quantile=0.3"
    "algorithm.entropy_thr_granularity=batch"
    "algorithm.entropy_top_p=0.95"
    "algorithm.top_perception_quantile=0.3"
    "algorithm.perception_thr_granularity=batch"
    "algorithm.tor_use_token_weighting=true"
    "algorithm.tor_rsn_weight=1.0"
    "algorithm.tor_prcp_weight=0.5"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.visual_sensitivity_metric=sampled_abs_log_ratio"
)
EXTRA_ARGS=(
    "worker.actor.clip_ratio_high=0.2"
)

launch_tor_matrix "$@"
