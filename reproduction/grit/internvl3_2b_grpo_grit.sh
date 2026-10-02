#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "OpenGVLab/InternVL3-2B"
pr_default EXPERIMENT_NAME "internvl3_2b_grpo_grit"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_grit "$@"
