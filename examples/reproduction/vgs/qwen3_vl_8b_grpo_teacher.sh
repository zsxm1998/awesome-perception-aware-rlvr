#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Stage 1 (Table 3): the GRPO teacher. Finalize it with scripts/finalize_run.py before stage 2 (README.md).
pr_default EXPERIMENT_NAME "qwen3_vl_8b_grpo_teacher"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_vgs_teacher_grpo "$@"
