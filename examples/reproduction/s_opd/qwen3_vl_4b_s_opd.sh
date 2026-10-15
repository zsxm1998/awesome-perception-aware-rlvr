#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# Table 1 (4B student, teacher Qwen3-VL-8B-Instruct): S-OPD.
pr_default MODEL_PATH "Qwen/Qwen3-VL-4B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_4b_s_opd"
s_opd_terms 0.02 0.02
# micro-batches of at most 2 x 8,192 tokens for the update, 4 x 8,192 for the log-probs of the 4B student
EXTRA_ARGS=(
    "worker.actor.micro_batch_size_per_device_for_update=2"
    "worker.actor.micro_batch_size_per_device_for_experience=4"
)

launch_s_opd virl39k 8b "$@"
