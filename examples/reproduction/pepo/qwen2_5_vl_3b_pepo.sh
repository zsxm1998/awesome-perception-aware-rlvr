#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "Qwen/Qwen2.5-VL-3B-Instruct"
pr_default EXPERIMENT_NAME "qwen2_5_vl_3b_pepo"
ALGO_ARGS=(
    "algorithm.visual_sensitivity_metric=hidden_state_similarity"
    "algorithm.visual_sensitivity_hidden_metric=cosine"
    "algorithm.visual_token=auto"
    "algorithm.advantage_scaling_method=pepo"
    "algorithm.advantage_scaling_schedule=linear"
    "algorithm.pepo_gate_alpha=0.05"
    "algorithm.pepo_gate_temperature=1.8"
)

launch_pepo_geometry3k "$@"
