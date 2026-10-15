#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "grit"
ALGO_ARGS=(
    "data.format_prompt=null"
    "data.system_prompt=$ROOT_DIR/examples/system_prompt/grit_GR.txt"
    "worker.reward.reward_function=$ROOT_DIR/examples/reward_function/grit.py:compute_score"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
