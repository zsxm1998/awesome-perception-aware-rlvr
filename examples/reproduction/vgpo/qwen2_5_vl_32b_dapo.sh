#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-32B-Instruct"
pr_default NNODES "4"  # the paper trains the 32B model on 32 GPUs; start Ray on every node first
pr_default EXPERIMENT_NAME "qwen2_5_vl_32b_dapo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "worker.rollout.tensor_parallel_size=4"
    "worker.actor.micro_batch_size_per_device_for_update=2"
    "worker.actor.micro_batch_size_per_device_for_experience=4"
)

launch_vgpo "$@"
