#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-7B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_grpo_deepeyes"
ALGO_ARGS=("${DEEPEYES_AGENT_ARGS[@]}" "${QWEN25_VL_ARGS[@]}")
EXTRA_ARGS=()

launch_deepeyes "$@"
