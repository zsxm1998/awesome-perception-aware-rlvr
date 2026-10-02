#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_grpo_entropy"
ALGO_ARGS=(
    "algorithm.top_entropy_quantile=0.2"
    "algorithm.entropy_thr_granularity=response"
    "algorithm.normalize_pg_loss_by_selected_tokens=true"
)
EXTRA_ARGS=()

launch_full_vocab_sensitivity_geo3k "$@"
