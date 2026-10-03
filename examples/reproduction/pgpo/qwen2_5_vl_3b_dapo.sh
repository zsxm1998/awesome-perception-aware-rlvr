#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_dapo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "worker.actor.clip_ratio_low=0.2"
    "worker.actor.clip_ratio_high=0.28"
)

launch_pgpo_matrix "$@"
