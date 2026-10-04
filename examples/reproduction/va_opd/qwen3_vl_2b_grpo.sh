#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1 baseline (Geometry3K): GRPO.
pr_default VA_OPD_DATA "geo3k"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_grpo"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_va_opd_rl "$@"
