#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "pepo"
ALGO_ARGS=(
    "algorithm.visual_sensitivity_metric=hidden_state_similarity"
    "algorithm.visual_sensitivity_hidden_metric=cosine"
    "algorithm.visual_token=auto"
    "algorithm.advantage_scaling_method=pepo"
    "algorithm.advantage_scaling_schedule=linear"
    "algorithm.pepo_gate_alpha=0.05"
    "algorithm.pepo_gate_temperature=1.8"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
