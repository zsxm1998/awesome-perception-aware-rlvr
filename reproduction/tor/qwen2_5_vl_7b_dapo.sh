#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "qwen2_5_vl_7b_dapo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "data.max_response_length=5120"
    "data.mini_rollout_batch_size=128"
    "worker.actor.clip_ratio_high=0.28"
    "algorithm.use_kl_loss=false"
    "algorithm.disable_kl=true"
    "algorithm.online_filtering=true"
    "algorithm.filter_key=accuracy"
    "algorithm.filter_low=0.01"
    "algorithm.filter_high=0.99"
    "trainer.max_try_make_batch=-1"
)

launch_tor_matrix "$@"
