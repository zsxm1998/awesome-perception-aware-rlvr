# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib.util
import inspect
import math
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Mapping, NotRequired, Sequence, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reasoning import canonicalize_response_for_prefilled_think, decode_prompt_from_batch
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str
    response_ids: NotRequired[list[int]]  # the model's response tokens (those in response_mask)
    num_images: NotRequired[int]
    grounding_consistency: NotRequired[float]
    grounding_consistency_raw: NotRequired[float]
    perception_score: NotRequired[float]  # claim probes (verl/trainer/claim_probes.py), only for probed responses
    data_source: NotRequired[str]
    question: NotRequired[str]
    tool_call_successes: NotRequired[int]
    tool_call_attempts: NotRequired[int]
    tool_call_errors: NotRequired[int]
    malformed_tool_calls: NotRequired[int]
    unrun_tool_calls: NotRequired[int]
    tool_internal_errors: NotRequired[int]
    image_idx_errors: NotRequired[int]
    invalid_final_answers: NotRequired[int]
    turn_end_errors: NotRequired[int]
    tool_calls_inside_reasoning: NotRequired[int]
    unclosed_answer: NotRequired[bool]
    no_action: NotRequired[bool]
    truncated: NotRequired[bool]
    trajectory_retries: NotRequired[int]
    final_answer: NotRequired[str | None]
    agent_status: NotRequired[str]
    # What the agent environment reported at the end of the episode: {"reward", "components", "metrics"}.
    env_outcome: NotRequired[dict[str, Any]]


class RewardScore(TypedDict):
    overall: float
    format: NotRequired[float]
    accuracy: NotRequired[float]
    grounding_consistency: NotRequired[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


def _get_num_images(data: DataProto, idx: int) -> int | None:
    multi_modal_batch = data.non_tensor_batch.get("multi_modal_data")
    if multi_modal_batch is None:
        return None
    sample_multi_modal = multi_modal_batch[idx]
    if not isinstance(sample_multi_modal, dict):
        return None
    images = sample_multi_modal.get("images")
    if images is None:
        return None
    return len(images)


_PASSTHROUGH_STRING_FIELDS = ("data_source", "question")


def _add_dataset_fields(reward_input: RewardInput, data: DataProto, idx: int) -> None:
    """Forward optional per-row dataset columns (e.g. ``data_source`` for rewards that route by
    sub-dataset, ``question`` for LLM-judged rewards) when the training data provides them."""
    for key in _PASSTHROUGH_STRING_FIELDS:
        values = data.non_tensor_batch.get(key)
        if values is not None and values[idx] is not None:
            reward_input[key] = str(values[idx])


def _add_response_ids(reward_input: RewardInput, data: DataProto, idx: int) -> None:
    response_mask = data.batch["response_mask"][idx].to(torch.bool)
    reward_input["response_ids"] = data.batch["responses"][idx][response_mask].tolist()


def _get_optional_scalar(data: DataProto, idx: int, key: str) -> float | None:
    values = data.non_tensor_batch.get(key)
    if values is None:
        return None
    return float(values[idx])


def _add_perception_score(reward_input: RewardInput, data: DataProto, idx: int) -> None:
    """The claim probes' perception score, for the responses that were probed (the others hold NaN)."""
    values = data.non_tensor_batch.get("perception_score")
    if values is not None and math.isfinite(float(values[idx])):
        reward_input["perception_score"] = float(values[idx])


def _response_payload(
    data: DataProto,
    idx: int,
    *,
    tokenizer: PreTrainedTokenizer,
    skip_special_tokens: bool,
) -> tuple[str, int, int]:
    response_ids = data.batch["responses"][idx]
    response_mask = data.batch["response_mask"][idx].to(torch.bool)
    action_positions = torch.nonzero(response_mask, as_tuple=False).flatten()
    if action_positions.numel() == 0:
        raise ValueError("reward computation requires at least one model action token")

    agent_inputs = data.non_tensor_batch.get("agent_reward_input")
    if agent_inputs is not None:
        if len(agent_inputs) != len(data):
            raise ValueError("agent_reward_input does not align with the reward batch")
        agent_input = agent_inputs[idx]
        if not isinstance(agent_input, dict):
            raise TypeError("each agent_reward_input must be a dictionary")
        response = agent_input.get("response")
        if not isinstance(response, str):
            raise TypeError("agent_reward_input.response must be a string")
        return response, int(action_positions.numel()), int(action_positions[-1].item())

    valid_response_ids = response_ids[response_mask]
    response = tokenizer.decode(
        valid_response_ids,
        skip_special_tokens=skip_special_tokens,
    )
    prompt = decode_prompt_from_batch(tokenizer, data.batch, idx)
    response = canonicalize_response_for_prefilled_think(prompt, response)
    return response, int(action_positions.numel()), int(action_positions[-1].item())


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validated_env_outcome(outcome: Any) -> dict[str, Any]:
    if not isinstance(outcome, dict) or set(outcome) != {"reward", "components", "metrics"}:
        raise ValueError("agent_reward_input.env_outcome must hold exactly reward, components and metrics")
    reward = outcome["reward"]
    if reward is not None and not _finite_number(reward):
        raise TypeError("env_outcome.reward must be a finite number or None")
    for key in ("components", "metrics"):
        values = outcome[key]
        if not isinstance(values, dict) or not all(_finite_number(value) for value in values.values()):
            raise TypeError(f"env_outcome.{key} must map names to finite numbers")
    return {"reward": reward, "components": dict(outcome["components"]), "metrics": dict(outcome["metrics"])}


def _attach_agent_reward_fields(
    reward_input: RewardInput,
    data: DataProto,
    idx: int,
) -> None:
    agent_inputs = data.non_tensor_batch.get("agent_reward_input")
    if agent_inputs is None:
        return
    agent_input = agent_inputs[idx]
    tool_call_successes = agent_input.get("tool_call_successes")
    if isinstance(tool_call_successes, bool) or not isinstance(
        tool_call_successes,
        int,
    ):
        raise TypeError("agent_reward_input.tool_call_successes must be an integer")
    if tool_call_successes < 0:
        raise ValueError("agent tool_call_successes cannot be negative")
    reward_input["tool_call_successes"] = tool_call_successes
    if "final_answer" not in agent_input:
        raise ValueError("agent_reward_input.final_answer is required")
    final_answer = agent_input["final_answer"]
    if final_answer is not None and not isinstance(final_answer, str):
        raise TypeError("agent_reward_input.final_answer must be a string or None")
    if isinstance(final_answer, str) and not final_answer.strip():
        raise ValueError("agent_reward_input.final_answer cannot be empty")
    reward_input["final_answer"] = final_answer
    agent_status = agent_input.get("status")
    if not isinstance(agent_status, str):
        raise TypeError("agent_reward_input.status must be a string")
    if (agent_status == "answered") != (final_answer is not None):
        raise ValueError("agent_reward_input.status and final_answer must agree on answered state")
    reward_input["agent_status"] = agent_status
    if "env_outcome" in agent_input:
        reward_input["env_outcome"] = _validated_env_outcome(agent_input["env_outcome"])

    diagnostics = data.non_tensor_batch.get("agent_diagnostics")
    if diagnostics is None or len(diagnostics) != len(data):
        raise ValueError("agent reward batches require aligned agent_diagnostics")
    diagnostic = diagnostics[idx]
    if not isinstance(diagnostic, dict):
        raise TypeError("each agent_diagnostics item must be a dictionary")
    if diagnostic.get("trajectory_id") != agent_input.get("trajectory_id"):
        raise ValueError("agent reward input and diagnostics trajectory IDs do not match")
    for key in (
        "tool_call_attempts",
        "tool_call_errors",
        "malformed_tool_calls",
        "unrun_tool_calls",
        "tool_internal_errors",
        "image_idx_errors",
        "invalid_final_answers",
        "turn_end_errors",
        "tool_calls_inside_reasoning",
        "trajectory_retries",
    ):
        value = diagnostic.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"agent diagnostic {key} must be a non-negative integer")
        reward_input[key] = value
    for key in ("unclosed_answer", "no_action", "truncated"):
        value = diagnostic.get(key)
        if not isinstance(value, bool):
            raise TypeError(f"agent diagnostic {key} must be boolean")
        reward_input[key] = value


STEP_REWARDS_KEY = "step_rewards"


def _add_env_step_rewards(score: dict[str, Any], reward_tensor: torch.Tensor, data: DataProto, idx: int) -> None:
    """Add the rewards an environment gave for single tool calls (``EnvStep.reward``) to the sample's return: each
    on the last action token of its turn, their sum to ``overall`` and, as ``step_rewards``, to the logs. The
    reward function's score is the reward of the last turn; all estimators, logs and online filtering see the sum.
    Samples without step rewards are left as they are."""
    agent_inputs = data.non_tensor_batch.get("agent_reward_input")
    step_rewards = None if agent_inputs is None else agent_inputs[idx].get("env_step_rewards")
    if step_rewards is None:
        return
    if STEP_REWARDS_KEY in score:
        raise ValueError(
            f"the reward function returned {STEP_REWARDS_KEY!r}, which the trainer logs the step rewards as"
        )
    response_mask = data.batch["response_mask"][idx]
    total = 0.0
    for position, reward in step_rewards:
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < response_mask.numel():
            raise ValueError(f"env_step_rewards position {position!r} is outside the response")
        if not response_mask[position]:
            raise ValueError(f"env_step_rewards position {position} is not a model token")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError(f"env_step_rewards reward {reward!r} must be a finite number")
        reward_tensor[idx, position] += reward
        total += reward
    score["overall"] = score["overall"] + total
    score[STEP_REWARDS_KEY] = total


def collect_reward_metrics(scores: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Per-metric lists aligned with the samples. Samples may report different metrics (e.g. an environment's
    components that apply to some tasks only): a sample without a metric has None at its position."""
    keys: dict[str, None] = {}
    for score in scores:
        keys.update(dict.fromkeys(score))
    return defaultdict(list, {key: [score.get(key) for score in scores] for key in keys})


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction
    add_response_ids: bool = False

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        scores = []
        for i in range(len(data)):
            response_str, cur_response_length, reward_position = _response_payload(
                data,
                i,
                tokenizer=self.tokenizer,
                skip_special_tokens=self.config.skip_special_tokens,
            )
            reward_input: RewardInput = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            _attach_agent_reward_fields(reward_input, data, i)
            _add_dataset_fields(reward_input, data, i)
            if self.add_response_ids:
                _add_response_ids(reward_input, data, i)
            num_images = _get_num_images(data, i)
            if num_images is not None:
                reward_input["num_images"] = num_images
            grounding_consistency = _get_optional_scalar(data, i, "grounding_consistency")
            if grounding_consistency is not None:
                reward_input["grounding_consistency"] = grounding_consistency
            grounding_consistency_raw = _get_optional_scalar(data, i, "grounding_consistency_raw")
            if grounding_consistency_raw is not None:
                reward_input["grounding_consistency_raw"] = grounding_consistency_raw
            _add_perception_score(reward_input, data, i)
            score = self.reward_fn(reward_input)
            reward_tensor[i, reward_position] = score["overall"]
            _add_env_step_rewards(score, reward_tensor, data, i)
            scores.append(score)

        return reward_tensor, collect_reward_metrics(scores)


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction
    add_response_ids: bool = False

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        reward_positions = []
        for i in range(len(data)):
            response_str, cur_response_length, reward_position = _response_payload(
                data,
                i,
                tokenizer=self.tokenizer,
                skip_special_tokens=self.config.skip_special_tokens,
            )
            reward_input: RewardInput = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            _attach_agent_reward_fields(reward_input, data, i)
            _add_dataset_fields(reward_input, data, i)
            if self.add_response_ids:
                _add_response_ids(reward_input, data, i)
            num_images = _get_num_images(data, i)
            if num_images is not None:
                reward_input["num_images"] = num_images
            grounding_consistency = _get_optional_scalar(data, i, "grounding_consistency")
            if grounding_consistency is not None:
                reward_input["grounding_consistency"] = grounding_consistency
            grounding_consistency_raw = _get_optional_scalar(data, i, "grounding_consistency_raw")
            if grounding_consistency_raw is not None:
                reward_input["grounding_consistency_raw"] = grounding_consistency_raw
            _add_perception_score(reward_input, data, i)
            reward_inputs.append(reward_input)
            reward_positions.append(reward_position)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i, score in enumerate(scores):
            reward_tensor[i, reward_positions[i]] = score["overall"]
            _add_env_step_rewards(score, reward_tensor, data, i)

        return reward_tensor, collect_reward_metrics(scores)


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        parameters = inspect.signature(reward_fn).parameters.values()
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            unknown = sorted(set(config.reward_function_kwargs) - {parameter.name for parameter in parameters})
            if unknown:  # fail at start-up rather than at the first reward
                raise TypeError(
                    f"`{config.reward_function_name}` of {config.reward_function} takes no argument {unknown} "
                    "(worker.reward.reward_function_kwargs); e.g. perception_weight needs math.py:compute_score."
                )
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        # the response token ids, only for reward functions that ask for them (a list per response is costly)
        self.add_response_ids = bool(getattr(module, "REWARD_INPUT_RESPONSE_IDS", False))
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
