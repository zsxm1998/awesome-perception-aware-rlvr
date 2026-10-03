#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_dapo_tor"
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
    "data.mini_rollout_batch_size=128"
    "worker.actor.clip_ratio_high=0.28"
    "algorithm.use_kl_loss=false"
    "algorithm.disable_kl=true"
    "algorithm.online_filtering=true"
    "algorithm.filter_key=accuracy"
    "algorithm.filter_low=0.01"
    "algorithm.filter_high=0.99"
    "trainer.max_try_make_batch=-1"
)

launch_tor_matrix "$@"
