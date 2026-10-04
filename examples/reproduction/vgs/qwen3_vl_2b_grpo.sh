#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2: GRPO alone (alpha = 0).
pr_default EXPERIMENT_NAME "qwen3_vl_2b_grpo"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_vgs_grpo_combination "$@"
