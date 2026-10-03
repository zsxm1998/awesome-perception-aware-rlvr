#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_grpo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "worker.actor.clip_ratio_high=0.2"
)

launch_tor_matrix "$@"
