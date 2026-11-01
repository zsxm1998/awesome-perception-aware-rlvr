# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The episode reward that an agent environment reports when the episode ends (its finalize).

The environment scores the answer itself (it holds the ground truth), so this function only passes its
reward on and logs its components and metrics under ``env/``: ``reward/env/<name>`` in training,
``val/env/<name>_reward`` in validation (the environment's names never meet the trainer's own, such as
``overall``). An episode may report only the names that apply to it; a name is averaged over the episodes
that reported it, and ``reward_count/env/<name>`` counts them when some did not. A trajectory without an
environment outcome, or with a null reward (an environment that does not score its episodes), fails: that is
an integration error, not a zero reward. Use it with ``worker.rollout.agent_environment`` set to an
environment that reports rewards (e.g. ``remote_http``).
"""

from __future__ import annotations

from typing import Any


REWARD_NAME = "environment"
REWARD_TYPE = "batch"

# Trajectory statistics of the agent loop, logged next to the environment's numbers.
_LOOP_STATISTICS = (
    "tool_call_attempts",
    "tool_call_errors",
    "malformed_tool_calls",
    "unrun_tool_calls",
    "trajectory_retries",
)
ENVIRONMENT_PREFIX = "env/"


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        outcome = reward_input.get("env_outcome")
        if outcome is None:
            raise ValueError(
                "the environment reward function needs the environment's episode outcome; set "
                "worker.rollout.agent_environment to an environment that reports one (e.g. remote_http)"
            )
        if outcome["reward"] is None:
            raise ValueError(
                "the environment returned no episode reward (finalize gave reward: null, i.e. it does not score "
                "its episodes); use a reward function that scores the answer"
            )
        score = {"overall": float(outcome["reward"])}
        for name in _LOOP_STATISTICS:
            if name in reward_input:
                score[name] = float(reward_input[name])
        score["answered"] = float(reward_input.get("agent_status") == "answered")
        score["truncated"] = float(bool(reward_input.get("truncated", False)))
        repeated = set(outcome["components"]) & set(outcome["metrics"])
        if repeated:
            raise ValueError(
                f"the environment reported {sorted(repeated)} both as components and as metrics; give each name once"
            )
        for group in ("components", "metrics"):
            for name, value in outcome[group].items():
                score[ENVIRONMENT_PREFIX + name] = float(value)
        scores.append(score)
    return scores
