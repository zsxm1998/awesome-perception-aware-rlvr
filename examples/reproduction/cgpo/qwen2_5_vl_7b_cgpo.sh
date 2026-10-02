#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_cgpo"
ALGO_ARGS=("${CGPO_ALGO_ARGS[@]}")
EXTRA_ARGS=()

launch_cgpo "$@"
