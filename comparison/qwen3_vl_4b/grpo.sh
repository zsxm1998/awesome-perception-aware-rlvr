#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/common.sh"

pr_default EXPERIMENT_NAME "grpo"
ALGO_ARGS=()
EXTRA_ARGS=()

launch_grpo_comparison "$@"
