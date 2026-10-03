#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "OpenGVLab/InternVL3-2B"
pr_default EXPERIMENT_NAME "internvl3_2b_grpo_grit"
ALGO_ARGS=()
EXTRA_ARGS=(
    # as GRIT's InternVL training: images at their resolution cut into at most 2 tiles of 448 px (plus a
    # thumbnail), prompts up to 1,500 tokens
    "worker.actor.model.max_dynamic_patch=2"
    "data.max_pixels=12845056"
    "data.max_prompt_length=1500"
)

launch_grit "$@"
