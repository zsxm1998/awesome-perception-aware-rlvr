#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "OpenGVLab/InternVL3-2B-Instruct"
pr_default EXPERIMENT_NAME "internvl3_2b_pepo"
ALGO_ARGS=(
    "algorithm.visual_sensitivity_metric=hidden_state_similarity"
    "algorithm.visual_sensitivity_hidden_metric=cosine"
    "algorithm.visual_token=<IMG_CONTEXT>"
    "algorithm.advantage_scaling_method=pepo"
    "algorithm.advantage_scaling_schedule=linear"
    "algorithm.pepo_gate_alpha=0.05"
    "algorithm.pepo_gate_temperature=1.8"
)
EXTRA_ARGS=(
    "trainer.val_freq=25"
    "trainer.save_freq=25"
)

launch_pepo_geometry3k "$@"
