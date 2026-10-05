#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen3-VL-8B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_8b_vapo"
ALGO_ARGS=("${VAPO_ARGS[@]}")
EXTRA_ARGS=()

launch_vapo "$@"
