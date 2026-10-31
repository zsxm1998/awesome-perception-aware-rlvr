#!/usr/bin/env bash
# GiGPO (step-level advantages, docs/agent_environments.md#step-level-advantages-gigpo) of Qwen3-VL-4B on the toy
# visual-search environment over HTTP: the GRPO script's settings with algorithm.adv_estimator=gigpo. The zoom-in
# crops rarely repeat pixel for pixel, so the turns are grouped by the environment's names of the views
# (state_key: zoom level and grid cell). The invalid-action penalty is 1% of the success reward, as in verl-agent's
# scripts (0.1 for rewards of 10, 0.01 for rewards of 1). Start the server and write the tasks first:
#   python examples/agent_env/toy_search.py tasks --out data/agent_env/toy_search
#   python examples/agent_env/toy_search.py serve --port 8000 &
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_vl_4b_gigpo_toy_search}"
exec bash "$THIS_DIR/qwen3_vl_4b_toy_search.sh" \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.95 \
    algorithm.step_state_key=environment \
    algorithm.group_advantage_norm=mean \
    algorithm.invalid_action_penalty=0.01 \
    "$@"
