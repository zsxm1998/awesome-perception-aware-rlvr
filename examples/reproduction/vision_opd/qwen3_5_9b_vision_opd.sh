#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1: Vision-OPD on Qwen3.5-9B, two nodes of 8 GPUs (16, as the paper); start the Ray cluster first, see
# README.md ("Two nodes").
pr_default MODEL_PATH "Qwen/Qwen3.5-9B"
pr_default EXPERIMENT_NAME "qwen3_5_9b_vision_opd"
pr_default NNODES "2"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_vision_opd "$@"
