#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen3-VL-8B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_8b_grpo"
ALGO_ARGS=("${TEXT_ONLY_ARGS[@]}")
EXTRA_ARGS=()

launch_deepeyes "$@"
