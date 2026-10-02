#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_dapo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "trainer.n_gpus_per_node=2"
    "data.mini_rollout_batch_size=128"
    "worker.actor.clip_ratio_low=0.2"
    "worker.actor.clip_ratio_high=0.28"
    "algorithm.disable_kl=true"
    "algorithm.online_filtering=true"
    "algorithm.filter_key=accuracy"
    "algorithm.filter_low=0.01"
    "algorithm.filter_high=0.99"
)

launch_papo_matrix "$@"
