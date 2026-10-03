#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_dapo"
ALGO_ARGS=(
    "algorithm.online_filtering=true"
    "algorithm.filter_key=overall"
    "algorithm.filter_criterion=std"
    "algorithm.online_filtering_fallback=first_round"
)
EXTRA_ARGS=(
    "worker.actor.loss_avg_mode=token"
    "trainer.max_try_make_batch=3"
    "trainer.val_freq=25"
    "trainer.save_freq=25"
)

launch_pepo_geometry3k "$@"
