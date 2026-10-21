#!/usr/bin/env bash
# DeepEyes (examples/reproduction/deepeyes/qwen3_vl_8b_grpo_deepeyes.sh) with the task served over HTTP by
# examples/agent_env/deepeyes_http.py instead of the in-process environment; the reward is DeepEyes' reward
# function, as there. Write the task rows and start the server first:
#   python examples/agent_env/deepeyes_http.py tasks --data-dir data/deepeyes --out data/agent_env/deepeyes_http
#   python examples/agent_env/deepeyes_http.py serve --data-dir data/deepeyes --port 8001 &
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$THIS_DIR/../reproduction/deepeyes/common.sh"

pr_default MODEL_PATH "Qwen/Qwen3-VL-8B-Instruct"
pr_default EXPERIMENT_NAME "qwen3_vl_8b_grpo_deepeyes_http"
pr_default AGENT_ENV_URL "http://127.0.0.1:8001"
ALGO_ARGS=("${DEEPEYES_AGENT_ARGS[@]}")
EXTRA_ARGS=(
    "data.train_files=$DATA_ROOT/agent_env/deepeyes_http/train.parquet"
    "data.val_files=$DATA_ROOT/agent_env/deepeyes_http/val.parquet"
    "worker.rollout.agent_environment=remote_http"
    "worker.rollout.agent_environment_kwargs.url=$AGENT_ENV_URL"
)

launch_deepeyes "$@"
