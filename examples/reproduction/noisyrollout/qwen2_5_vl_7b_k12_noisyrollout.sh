#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_k12_noisyrollout"
# alpha_0 = 450, lambda = 60, gamma = 40 of t_max = 120 steps (6,457 // 512 = 12 steps per epoch, 10 epochs)
mapfile -t ALGO_ARGS < <(noisyrollout_args 450 60 0.3333333333333333)
EXTRA_ARGS=(
    "worker.rollout.n=12"  # 6 from the clean images + 6 from the noised images
    "trainer.total_epochs=10"
)

launch_noisyrollout k12 "$@"
