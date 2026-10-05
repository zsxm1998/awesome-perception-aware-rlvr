#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-32B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_32b_geo3k_grpo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "worker.rollout.n=8"
    "trainer.total_epochs=10"
    "worker.actor.micro_batch_size_per_device_for_update=1"
    "worker.actor.micro_batch_size_per_device_for_experience=2"
    "worker.rollout.tensor_parallel_size=8"
    "worker.rollout.gpu_memory_utilization=0.5"
)

launch_noisyrollout geo3k "$@"
