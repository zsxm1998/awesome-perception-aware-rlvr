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
"""The parts of an agentic rollout batch that do not depend on how the model is served.

Both rollout paths, the batched one inside each training process (``vllm_rollout_spmd``) and the server mode
(``rollout_worker``), read the batch's rows the same way (``agent_rollout_rows``), give each trajectory the same
identity and seeds (``agent_rollout_tasks``) and turn the finished trajectories into the same training batch
(``agent_rollout_output``); only the generation in between differs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ...protocol import DataProto
from ...utils.dataset import process_image
from .protocol import (
    AgentImageConfig,
    AgentLoopConfig,
    AgentTrajectory,
    derive_environment_seed,
    environment_report,
    turn_env_reward,
)
from .step_records import step_records
from .tool_call import NativeToolCallCodec
from .trajectory import (
    MaterializedAgentTrajectory,
    collate_materialized_agent_trajectories,
    turn_action_spans,
)


# A retried trajectory keeps its group identity but moves its generation seeds, so that it does not resample the
# rejected trajectory.
RETRY_SEED_STRIDE = 1_000_003


def retry_base_seed(base_seed: int, retry_index: int) -> int:
    return base_seed + retry_index * RETRY_SEED_STRIDE


def _row_environment_config(value: Any) -> dict[str, Any]:
    """One row's agent-environment configuration: a JSON object, stored as a string (parquet) or a mapping."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"a row's environment configuration is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("a row's environment configuration must be a JSON object")
    return value


@dataclass
class AgentRolloutRows:
    """The rows of a rollout batch: each row's group uid and the configuration its environment's reset gets."""

    environment_name: str
    uids: Sequence[Any]
    env_configs: list[dict[str, Any]]


def agent_rollout_rows(prompts: DataProto, config: Any) -> AgentRolloutRows:
    """Pop the agentic inputs from ``prompts`` (``config`` is the RolloutConfig) and build each row's environment
    configuration. DeepEyes' task is the row itself (its chat messages and source images, one image object per
    source image shared by the row's rollouts); other environments get the row's configuration column."""
    environment_name = config.resolved_agent_environment
    messages_batch = prompts.non_tensor_batch.pop("raw_prompt", None)
    raw_multi_modal_batch = prompts.non_tensor_batch.pop("multi_modal_data", None)
    uids = prompts.non_tensor_batch.pop("uid", None)
    env_names = prompts.non_tensor_batch.pop("env_name", None)  # "" = no tool for that row
    row_env_configs = prompts.non_tensor_batch.pop(config.agent_environment_config_key, None)
    prompts.non_tensor_batch.pop("raw_prompt_ids", None)
    batch_size = len(prompts)
    if environment_name == "deepeyes":
        if messages_batch is None or raw_multi_modal_batch is None or uids is None:
            raise ValueError("agentic rollout requires aligned raw_prompt, multi_modal_data, and uid fields")
        if not (len(messages_batch) == len(raw_multi_modal_batch) == len(uids) == batch_size):
            raise ValueError("agentic rollout inputs do not align by sample")
        env_configs = []
        for sample_position, multi_modal_data in enumerate(raw_multi_modal_batch):
            if not isinstance(multi_modal_data, dict):
                raise TypeError("DeepEyes currently requires image-based multi_modal_data")
            if multi_modal_data.get("videos"):
                raise ValueError("DeepEyes image_zoom_in_tool does not support videos")
            images = multi_modal_data.get("images")
            if not images:
                raise ValueError("DeepEyes rollout requires at least one source image")
            env_configs.append(
                {
                    "messages": messages_batch[sample_position],
                    "images": [process_image(image, None, None) for image in images],
                    "tools_enabled": env_names is None or bool(env_names[sample_position]),
                }
            )
    else:
        if uids is None or row_env_configs is None:
            raise ValueError(
                f"agent environment {environment_name!r} requires the uid field and the data column "
                f"{config.agent_environment_config_key!r} (worker.rollout.agent_environment_config_key)"
            )
        if not (len(uids) == len(row_env_configs) == batch_size):
            raise ValueError("agentic rollout inputs do not align by sample")
        env_configs = [_row_environment_config(value) for value in row_env_configs]
    return AgentRolloutRows(environment_name=environment_name, uids=uids, env_configs=env_configs)


@dataclass(frozen=True)
class AgentRolloutTask:
    """One trajectory of a rollout batch: its group (``uid``, the row's ``sample_position``), its place in the
    group (``rollout_index``) and its seeds."""

    uid: str
    sample_position: int
    rollout_index: int
    env_config: Mapping[str, Any]
    env_seed: int  # the same for the rollouts of a row and their retries
    sample_index: int  # the row's identity in the generation seeds
    lora_request: Optional[Any] = None


def agent_rollout_tasks(
    rows: AgentRolloutRows, *, rollout_n: int, base_seed: int, lora_requests: Optional[Sequence[Any]] = None
) -> list[AgentRolloutTask]:
    """The trajectories of a rollout batch, row by row and ``rollout_n`` per row, in the order of the batch that
    ``DataProto.repeat(rollout_n, interleave=True)`` gives."""
    tasks = []
    for sample_position, uid in enumerate(rows.uids):
        sample_identity = int.from_bytes(
            hashlib.blake2b(str(uid).encode("utf-8"), digest_size=8).digest(), byteorder="big", signed=False
        )
        for rollout_index in range(rollout_n):
            tasks.append(
                AgentRolloutTask(
                    uid=str(uid),
                    sample_position=sample_position,
                    rollout_index=rollout_index,
                    env_config=rows.env_configs[sample_position],
                    env_seed=derive_environment_seed(base_seed, sample_identity),
                    sample_index=sample_identity,
                    lora_request=None if lora_requests is None else lora_requests[sample_position],
                )
            )
    return tasks


def agent_image_config(config: Any, meta_info: Mapping[str, Any]) -> AgentImageConfig:
    return AgentImageConfig(
        min_pixels=meta_info.get("min_pixels"),
        max_pixels=meta_info.get("max_pixels"),
        limit_images=config.limit_images,
        observation_min_pixels=config.agent_observation_min_pixels,
    )


def agent_loop_config(config: Any) -> AgentLoopConfig:
    return AgentLoopConfig(
        max_tool_calls=config.agent_max_tool_calls,
        max_response_tokens=config.response_length,
        max_tokens_per_turn=config.agent_max_tokens_per_turn,
        max_tool_calls_per_turn=config.agent_max_tool_calls_per_turn,
    )


def agent_max_model_len(config: Any) -> int:
    return config.max_model_len or config.prompt_length + config.response_length


def agent_environment_kwargs(config: Any, bbox_format: str) -> dict[str, Any]:
    """The options of the environment's constructor; DeepEyes also gets the tool's coordinate convention and the
    prompt style."""
    kwargs = dict(config.agent_environment_kwargs)
    if config.resolved_agent_environment == "deepeyes":
        kwargs.update(bbox_format=bbox_format, prompt_style=config.agent_prompt_style)
    return kwargs


def agent_rollout_output(
    tasks: Sequence[AgentRolloutTask],
    trajectories: Sequence[AgentTrajectory],
    materialized: Sequence[MaterializedAgentTrajectory],
    retry_counts: Sequence[int],
    *,
    config: Any,
    tokenizer: Any,
    pad_token_id: int,
    meta_info: Mapping[str, Any],
) -> DataProto:
    """Collate the finished trajectories of ``tasks`` (in the same order) into the training batch, with the reward
    inputs, diagnostics, environment records and, for step-level advantages, the step records."""
    output = collate_materialized_agent_trajectories(
        materialized,
        pad_token_id=pad_token_id,
        padded_prompt_width=config.prompt_length,
        padded_response_width=config.response_length,
        uids=[task.uid for task in tasks],
        meta_info=meta_info,
    )

    # where each turn's model tokens are, for the environment's step rewards and the step records
    needs_spans = config.agent_step_records or any(
        turn_env_reward(step.calls) is not None for trajectory in trajectories for step in trajectory.steps
    )
    turn_spans = (
        [
            turn_action_spans(trajectory.steps, sample.response_mask)
            for trajectory, sample in zip(trajectories, materialized)
        ]
        if needs_spans
        else None
    )
    reward_inputs = []
    diagnostics = []
    codec = NativeToolCallCodec()
    for trajectory_id, trajectory in zip(
        output.non_tensor_batch["agent_trajectory_id"],
        trajectories,
    ):
        response = tokenizer.decode(
            trajectory.response_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if trajectory.steps and trajectory.steps[0].reasoning_prefilled:
            response = f"<think>{response}"
        metrics = trajectory.metrics
        tool_calls_inside_reasoning = sum(
            codec.count_tool_calls_inside_reasoning(
                step.model_text,
                reasoning_open=step.reasoning_prefilled,
            )
            for step in trajectory.steps
        )
        reward_input = {
            "trajectory_id": str(trajectory_id),
            "response": response,
            "final_answer": trajectory.final_answer,
            "status": trajectory.status.value,
            "tool_call_successes": metrics.tool_call_successes,
            "trajectory_retries": retry_counts[len(reward_inputs)],
        }
        if trajectory.env_outcome is not None:  # what the environment reported at the end of the episode
            reward_input["env_outcome"] = {
                "reward": trajectory.env_outcome.reward,
                "components": dict(trajectory.env_outcome.components),
                "metrics": dict(trajectory.env_outcome.metrics),
            }
        step_rewards = [turn_env_reward(step.calls) for step in trajectory.steps]
        if any(reward is not None for reward in step_rewards):
            # the rewards the environment gave for a turn's calls, on the last model token of the turn
            spans = turn_spans[len(reward_inputs)]
            reward_input["env_step_rewards"] = [
                [spans[turn][1] - 1, reward] for turn, reward in enumerate(step_rewards) if reward is not None
            ]
        reward_inputs.append(reward_input)
        diagnostics.append(
            {
                "trajectory_id": str(trajectory_id),
                "status": trajectory.status.value,
                "tool_call_attempts": metrics.tool_call_attempts,
                "tool_call_successes": metrics.tool_call_successes,
                "tool_call_errors": metrics.tool_call_errors,
                "malformed_tool_calls": metrics.malformed_tool_calls,
                "unrun_tool_calls": metrics.unrun_tool_calls,
                "tool_internal_errors": metrics.tool_internal_errors,
                "image_idx_errors": metrics.image_idx_errors,
                "invalid_final_answers": metrics.invalid_final_answers,
                "turn_end_errors": metrics.turn_end_errors,
                "tool_calls_inside_reasoning": tool_calls_inside_reasoning,
                "unclosed_answer": response.count("<answer>") > response.count("</answer>"),
                "truncated": metrics.truncated,
                "no_action": trajectory.status.value == "no_action",
                "trajectory_retries": retry_counts[len(diagnostics)],
            }
        )
    output.non_tensor_batch["agent_reward_input"] = np.array(
        reward_inputs,
        dtype=object,
    )
    output.non_tensor_batch["agent_diagnostics"] = np.array(
        diagnostics,
        dtype=object,
    )
    if config.resolved_agent_environment != "deepeyes":
        # what the environment reported per trajectory (training generation records keep it)
        output.non_tensor_batch["agent_environment"] = np.array(
            [
                environment_report(trajectory.steps, trajectory.env_outcome, trajectory.initial_state_key)
                for trajectory in trajectories
            ],
            dtype=object,
        )
    if config.agent_step_records:  # step-level advantages (algorithm.adv_estimator=gigpo)
        image_cache: dict[int, bytes] = {}
        output.non_tensor_batch["agent_step_records"] = np.array(
            [
                step_records(trajectory, spans, sample.multi_modal_data["images"], task.rollout_index, image_cache)
                for task, trajectory, sample, spans in zip(tasks, trajectories, materialized, turn_spans)
            ],
            dtype=object,
        )
    return output
