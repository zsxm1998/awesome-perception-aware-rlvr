#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-32B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_32b_k12_noisyrollout"
# Table 13: alpha_0 = 450, lambda = 30, gamma = 35 of t_max = 70 steps
mapfile -t ALGO_ARGS < <(noisyrollout_args 450 30 0.5)
EXTRA_ARGS=(
    "worker.rollout.n=8"  # 4 from the clean images + 4 from the noised images
    "trainer.total_epochs=7"
    "trainer.max_steps=70"  # Table 13 lists t_max = 70 with 7 epochs (84 steps at 12 steps per epoch)
    "worker.actor.micro_batch_size_per_device_for_update=1"
    "worker.actor.micro_batch_size_per_device_for_experience=2"
    "worker.rollout.tensor_parallel_size=8"
    "worker.rollout.gpu_memory_utilization=0.5"
)

launch_noisyrollout k12 "$@"
