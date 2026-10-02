#!/usr/bin/env bash
# Shared launcher for every training script under examples/reproduction/ and examples/comparison/.
#
# A leaf script sets EXPERIMENT_NAME and ALGO_ARGS, its method-level common.sh
# sets MODEL_PATH and METHOD_COMMON_ARGS, then calls `launch_training "$@"`.
# Precedence (highest first): command-line overrides passed to the leaf script >
# environment variables > leaf defaults > method common defaults.
#
# Environment variables understood by every script:
#   MODEL_PATH       HF id or local path of the policy model
#   DATA_ROOT        root of the prepared data (default: <repo>/data, see scripts/prepare_data.sh)
#   LOGGER           trainer loggers, e.g. '["console","wandb"]' (default: '["console","file"]')
#   N_GPUS_PER_NODE  GPUs per node (default: the paper setting of each method)
#   NNODES           number of nodes (default: 1)
#   EXPERIMENT_NAME  run name; checkpoints go to <repo>/checkpoints/<project>/<experiment>
#   DRY_RUN=1        only build and validate the configuration (no GPU, no Ray)
set -euo pipefail

LAUNCHER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LAUNCHER_DIR/.." && pwd)

pr_default() {
    local var_name=$1
    local default_value=$2
    if [[ -z "${!var_name-}" ]]; then
        printf -v "$var_name" '%s' "$default_value"
    fi
}

pr_default DATA_ROOT "$ROOT_DIR/data"
pr_default LOGGER '["console","file"]'
pr_default NNODES "1"

# Fail early with an actionable message when a prepared data file is missing.
require_data() {
    local path
    for path in "$@"; do
        path="${path%@*}"
        if [[ "$path" == /* && ! -e "$path" ]]; then
            echo "[launcher] missing data: $path" >&2
            echo "[launcher] run 'bash scripts/prepare_data.sh <dataset>' first (see data/README.md)." >&2
            exit 1
        fi
    done
}

launch_training() {
    local model_path="${MODEL_PATH:?MODEL_PATH must be set before launch_training}"
    local experiment_name="${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set before launch_training}"
    local -a method_common_args=()
    local -a algo_args=()
    local -a extra_args=()
    if declare -p METHOD_COMMON_ARGS >/dev/null 2>&1; then
        method_common_args=("${METHOD_COMMON_ARGS[@]}")
    fi
    if declare -p ALGO_ARGS >/dev/null 2>&1; then
        algo_args=("${ALGO_ARGS[@]}")
    fi
    if declare -p EXTRA_ARGS >/dev/null 2>&1; then
        extra_args=("${EXTRA_ARGS[@]}")
    fi

    local arg
    for arg in "${method_common_args[@]}" "${algo_args[@]}" "${extra_args[@]}"; do
        case "$arg" in
            data.train_files=*|data.val_files=*) require_data "${arg#*=}" ;;
        esac
    done

    local -a runtime_args=(
        "trainer.logger=$LOGGER"
        "trainer.nnodes=$NNODES"
    )
    if [[ -n "${N_GPUS_PER_NODE-}" ]]; then
        runtime_args+=("trainer.n_gpus_per_node=$N_GPUS_PER_NODE")
    fi

    cd "$ROOT_DIR"
    python3 -m verl.trainer.main \
        "config=$ROOT_DIR/examples/config.yaml" \
        "worker.actor.model.model_path=$model_path" \
        "trainer.experiment_name=$experiment_name" \
        "${method_common_args[@]}" \
        "${algo_args[@]}" \
        "${extra_args[@]}" \
        "${runtime_args[@]}" \
        "$@"
}
