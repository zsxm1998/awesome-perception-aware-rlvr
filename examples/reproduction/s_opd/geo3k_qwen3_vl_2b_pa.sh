#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 2 (2B student, teacher Qwen3-VL-8B-Instruct, Geometry3K): OPD + PA.
pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
pr_default EXPERIMENT_NAME "geo3k_qwen3_vl_2b_pa"
s_opd_terms 0 0.02
EXTRA_ARGS=()

launch_s_opd geo3k 8b "$@"
