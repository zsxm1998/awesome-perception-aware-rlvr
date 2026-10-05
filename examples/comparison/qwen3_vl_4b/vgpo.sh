#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "vgpo"
ALGO_ARGS=(
    "algorithm.visual_sensitivity_metric=hidden_state_similarity"
    "algorithm.visual_sensitivity_hidden_metric=cosine"
    "algorithm.visual_sensitivity_hidden_layers=last"
    "algorithm.visual_sensitivity_hidden_pooling=prototype"
    "algorithm.visual_sensitivity_reference=old"
    "algorithm.visual_token=auto"
    "algorithm.advantage_scaling_method=vgpo"
    "algorithm.vgpo_compensation_strength=0.3"
    "algorithm.vgpo_gate_tail_ratio=0.5"
    "algorithm.vgpo_gate_top_ratio=0.2"
)
EXTRA_ARGS=()

launch_grpo_comparison "$@"
