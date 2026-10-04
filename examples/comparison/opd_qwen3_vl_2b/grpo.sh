#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../qwen3_vl_4b/common.sh"

# RL reference: the GRPO recipe of the RL comparison (../qwen3_vl_4b/grpo.sh) with the 2B student,
# logged in the OPD comparison's project.
pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
pr_default EXPERIMENT_NAME "grpo"
ALGO_ARGS=()
EXTRA_ARGS=("trainer.project_name=Comparison-OPD-Qwen3-VL-2B")

launch_grpo_comparison "$@"
