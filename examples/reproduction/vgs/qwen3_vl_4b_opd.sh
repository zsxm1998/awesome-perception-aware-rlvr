#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 4 / Table 1 (4B student): Standard OPD, KL(p || q) on every response token.
pr_default MODEL_PATH "Qwen/Qwen3-VL-4B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_4b_opd"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_vgs_student "$@"
