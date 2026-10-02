#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default MODEL_PATH "OpenGVLab/InternVL3-2B-Instruct"
pr_default EXPERIMENT_NAME "internvl3_2b_grpo"
ALGO_ARGS=()
EXTRA_ARGS=(
    "trainer.val_freq=25"
    "trainer.save_freq=25"
)

launch_pepo_geometry3k "$@"
