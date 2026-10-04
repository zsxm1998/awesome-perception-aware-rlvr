#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1: Vision-OPD on Qwen3.5-4B, one node of 8 GPUs.
pr_default MODEL_PATH "Qwen/Qwen3.5-4B"
pr_default EXPERIMENT_NAME "qwen3_5_4b_vision_opd"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_vision_opd "$@"
