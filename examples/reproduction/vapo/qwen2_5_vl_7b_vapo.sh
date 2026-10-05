#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_vapo"
ALGO_ARGS=("${VAPO_ARGS[@]}")
EXTRA_ARGS=()

launch_vapo "$@"
