#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_dapo"
ALGO_ARGS=(
    "algorithm.online_filtering=true"
    "algorithm.filter_key=accuracy"
    "algorithm.filter_low=0.01"
    "algorithm.filter_high=0.99"
)
EXTRA_ARGS=(
    "worker.actor.loss_avg_mode=token"
    "trainer.val_freq=25"
    "trainer.save_freq=25"
)

launch_pepo_geometry3k "$@"
