#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_geo3k_noisyrollout"
# alpha_0 = 500, lambda = 30, gamma = 40 of t_max = 60 steps (2,101 // 512 = 4 steps per epoch, 15 epochs)
mapfile -t ALGO_ARGS < <(noisyrollout_args 500 30 0.6666666666666666)
EXTRA_ARGS=(
    "worker.rollout.n=12"  # 6 from the clean images + 6 from the noised images
)

launch_noisyrollout geo3k "$@"
