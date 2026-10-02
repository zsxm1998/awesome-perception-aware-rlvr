#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "tor"
ALGO_ARGS=(
    "algorithm.corrupt_image=random_patch"
    "algorithm.corrupt_image_kwargs={\"patch_size\":16,\"black_prob\":0.6}"
    "algorithm.corrupt_image_position=prompt"
    "algorithm.tor_use_token_weighting=true"
    "algorithm.tor_rsn_weight=1.0"
    "algorithm.top_entropy_quantile=0.3"
    "algorithm.entropy_thr_granularity=batch"
    "algorithm.tor_prcp_weight=0.5"
    "algorithm.top_perception_quantile=0.3"
    "algorithm.perception_thr_granularity=batch"
    "algorithm.visual_sensitivity_reference=old"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
