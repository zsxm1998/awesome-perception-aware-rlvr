#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../opd_qwen3_vl_2b/common.sh"

# Bridge to the OPD comparison: OPD from sampled tokens (../opd_qwen3_vl_2b/opd_sampled.sh) with the 4B
# student of this table and the same Qwen3-VL-8B-Instruct teacher, on the OPD recipe
# (../opd_qwen3_vl_2b/common.sh), logged in this table's project.
pr_default MODEL_PATH "Qwen/Qwen3-VL-4B-Instruct"
pr_default EXPERIMENT_NAME "opd_sampled"
ALGO_ARGS=(
    "algorithm.adv_estimator=teacher_log_ratio"
    "algorithm.teacher_log_ratio_clip=10"
)
EXTRA_ARGS=("trainer.project_name=Comparison-Qwen3-VL-4B")

launch_opd_comparison "$@"
