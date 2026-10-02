#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "OpenGVLab/InternVL3-2B-Instruct"
pr_default EXPERIMENT_NAME "internvl3_2b_pepo_d"
ALGO_ARGS=(
    "algorithm.visual_sensitivity_metric=hidden_state_similarity"
    "algorithm.visual_sensitivity_hidden_metric=cosine"
    "algorithm.visual_token=<IMG_CONTEXT>"
    "algorithm.advantage_scaling_method=pepo"
    "algorithm.advantage_scaling_schedule=linear"
    "algorithm.pepo_gate_alpha=0.05"
    "algorithm.pepo_gate_temperature=1.8"
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
