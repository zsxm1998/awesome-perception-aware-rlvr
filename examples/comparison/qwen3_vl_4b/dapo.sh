#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# DAPO baseline: the shared recipe with DAPO's own components (no KL, clip-higher, dynamic sampling).
# The shared entropy penalty is kept, so only these components differ from grpo.sh.
pr_default EXPERIMENT_NAME "dapo"
ALGO_ARGS=(
    "algorithm.disable_kl=true"
    "algorithm.use_kl_loss=false"
    "worker.actor.clip_ratio_low=0.2"
    "worker.actor.clip_ratio_high=0.28"
    "algorithm.online_filtering=true"
    "algorithm.filter_key=accuracy"
    "algorithm.filter_low=0.01"
    "algorithm.filter_high=0.99"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
