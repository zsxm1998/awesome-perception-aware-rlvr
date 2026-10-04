#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1: VCSD on Qwen/Qwen3.5-4B.
pr_default MODEL_PATH "Qwen/Qwen3.5-4B"
pr_default EXPERIMENT_NAME "qwen3_5_4b_vcsd"
ALGO_ARGS=()
# The teacher scores the image and the black view in one forward; with Qwen3.5's 248k-token vocabulary their
# logits take about 15 GB per micro-batch of 4 x 7k tokens, so the update uses micro-batches of 2.
EXTRA_ARGS=("worker.actor.micro_batch_size_per_device_for_update=2")

launch_vcsd "$@"
