#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1 (2B student, teacher Qwen3-VL-8B-Instruct): OPD.
pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_opd"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_s_opd virl39k 8b "$@"
