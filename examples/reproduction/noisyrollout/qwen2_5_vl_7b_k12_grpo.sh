#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_k12_grpo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "worker.rollout.n=12"
    "trainer.total_epochs=10"
)

launch_noisyrollout k12 "$@"
