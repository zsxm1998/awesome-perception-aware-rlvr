#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 6 ("ViRL39K GRPO"): the GRPO-trained 8B teacher of the _grpo_teacher scripts. Finalize it with
# scripts/finalize_run.py before distilling from it (README.md).
pr_default EXPERIMENT_NAME "qwen3_vl_8b_grpo_teacher"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_s_opd_grpo_teacher "$@"
