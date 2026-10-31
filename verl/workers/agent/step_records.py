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
"""What step-level advantage estimators (GiGPO) need to know about each turn of a trajectory.

A record holds facts only; which state a turn acts in and what it earned are worked out on the driver
(``verl/trainer/step_advantage.py``), so the rollout does not depend on the estimator's options:

    {"rollout_index": the trajectory's index among its prompt's rollouts,
     "initial": {"state_key": the environment's name of the initial state or None,
                 "observation": hash of the prompt tokens and source image pixels},
     "turns": [{"span": [start, end),              # the turn's model tokens, in response coordinates
                "valid": bool,                    # the turn's action was well formed (see ``turn_is_valid``)
                "env_calls": [{"state_key": str or None, "reward": float or None}, ...],  # calls the environment ran
                "observation": hash or None,      # the observation committed after the turn (tokens, image pixels
                                                  # as the model receives them)
                "observation_text": str or None}  # the tool result's text, when that observation has no image
               , ...]}
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional, Sequence

from .protocol import AgentStatus, AgentTrajectory


def turn_is_valid(trajectory: AgentTrajectory, index: int) -> bool:
    """A turn's action is valid when it is well formed: a tool call that went to the environment (whatever the
    environment made of it) or the answer that ended the episode. Malformed calls, malformed answers, a wrong turn
    end, a turn cut by a length limit, a turn without an action and a call after the tool-call limit are invalid,
    as verl-agent's ``is_action_valid`` judges the format only."""
    step = trajectory.steps[index]
    if step.tool_call is not None:
        return True
    return index == len(trajectory.steps) - 1 and trajectory.status == AgentStatus.ANSWERED


def _image_digest(image: Any, cache: dict[int, bytes]) -> bytes:
    image = getattr(image, "image", image)  # ProcessedImageInput
    digest = cache.get(id(image))
    if digest is None:
        hasher = hashlib.sha1(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode("ascii"))
        hasher.update(image.tobytes())
        digest = hasher.digest()
        cache[id(image)] = digest
    return digest


def _digest(token_ids: Sequence[int], images: Sequence[Any], cache: dict[int, bytes]) -> str:
    hasher = hashlib.sha1(json.dumps(list(token_ids)).encode("ascii"))
    for image in images:
        hasher.update(_image_digest(image, cache))
    return hasher.hexdigest()


def _result_text(result: Any) -> str:
    """The text of a tool result as the environment wrote it (without the chat template around it)."""
    if result.parts is not None:
        return "".join(part.get("text", "") for part in result.parts if part.get("type") == "text")
    if result.observation_text is not None:
        return result.observation_text
    error = result.content.get("error") if isinstance(result.content, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return json.dumps(result.content, sort_keys=True, ensure_ascii=False, default=str)


def step_records(
    trajectory: AgentTrajectory,
    spans: Sequence[tuple[int, int]],
    model_images: Sequence[Any],
    rollout_index: int,
    image_cache: Optional[dict[int, bytes]] = None,
) -> dict[str, Any]:
    """The step record of a trajectory whose turns' model tokens are at ``spans`` (``turn_action_spans``).
    ``model_images`` are the trajectory's images as the model receives them (the materialized source and observation
    images, in order), so that observations the model cannot tell apart hash alike. ``rollout_index`` orders the
    trajectories of a prompt independently of how the batch is split across workers. ``image_cache`` keeps pixel
    hashes by image object across the trajectories of a batch."""
    if len(spans) != len(trajectory.steps):
        raise ValueError("one span per turn is required")
    source_count = len(trajectory.source_images)
    if len(model_images) != source_count + len(trajectory.observation_images):
        raise ValueError("model_images must hold the trajectory's source and observation images")
    cache = {} if image_cache is None else image_cache
    turns = []
    observation_images = list(model_images[source_count:])
    cursor = 0
    for index, (step, span) in enumerate(zip(trajectory.steps, spans)):
        env_calls = []
        if step.tool_call is not None:  # one call per turn; a malformed call does not reach the environment
            env_calls.append({"state_key": step.state_key, "reward": step.env_reward})
        observation = observation_text = None
        if step.observation_committed:
            images = observation_images[cursor : cursor + step.observation_image_count]
            cursor += step.observation_image_count
            observation = _digest(step.observation_token_ids, images, cache)
            if not images and step.tool_result is not None:
                observation_text = _result_text(step.tool_result)
        turns.append(
            {
                "span": [int(span[0]), int(span[1])],
                "valid": turn_is_valid(trajectory, index),
                "env_calls": env_calls,
                "observation": observation,
                "observation_text": observation_text,
            }
        )
    if cursor != len(observation_images):
        raise RuntimeError("the steps' observation image counts do not add up to the trajectory's images")
    return {
        "rollout_index": int(rollout_index),
        "initial": {
            "state_key": trajectory.initial_state_key,
            "observation": _digest(trajectory.prompt_ids, model_images[:source_count], cache),
        },
        "turns": turns,
    }
