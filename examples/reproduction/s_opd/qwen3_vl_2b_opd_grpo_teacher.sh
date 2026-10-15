#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1 (2B student, the 8B GRPO teacher of qwen3_vl_8b_grpo_teacher.sh): OPD_G.
pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_opd_grpo_teacher"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_s_opd virl39k grpo "$@"
