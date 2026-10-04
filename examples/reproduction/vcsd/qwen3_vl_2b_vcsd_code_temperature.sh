#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

# VCSD on Qwen3-VL-2B-Instruct with the temperature as the released code applies it: the teacher's two views,
# the support set, the target and the student stay at T = 1 and T_KD = 2 only multiplies the loss by 4.
pr_default MODEL_PATH "Qwen/Qwen3-VL-2B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_2b_vcsd_code_temperature"
ALGO_ARGS=(
    "algorithm.distill_temperature_scope=loss_scale_only"
)
EXTRA_ARGS=()

launch_vcsd "$@"
