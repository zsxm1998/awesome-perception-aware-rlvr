#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1 (2B student, the 8B GRPO teacher of qwen3_vl_8b_grpo_teacher.sh): S-OPD_G.
pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_s_opd_grpo_teacher"
s_opd_terms 0.02 0.02
EXTRA_ARGS=()

launch_s_opd virl39k grpo "$@"
