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

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional, Sequence

from PIL import Image

from ...utils.dataset import ProcessedImageInput, process_image
from .protocol import (
    TOO_MANY_TOOL_CALLS_IN_TURN,
    TOOL_CALL_LIMIT_REACHED,
    AgentCall,
    AgentEnvironment,
    AgentImageConfig,
    AgentLoopConfig,
    AgentMetrics,
    AgentSeedContext,
    AgentStatus,
    AgentStep,
    AgentTrajectory,
    GenerationBackend,
    GenerationRequest,
    ObservationEncoder,
    ObservationEncodingRequest,
    ParsedToolCall,
    ToolResult,
    checked_step_reward,
)
from .tool_call import NativeToolCallCodec, ToolCallCodec
from .tools.base import tool_error


def _too_many_calls_error(name: str, per_turn: int) -> ToolResult:
    plural = "" if per_turn == 1 else "s"
    return tool_error(name, TOO_MANY_TOOL_CALLS_IN_TURN, f"not run: at most {per_turn} tool call{plural} run per turn")


async def run_tool_calls(
    environment: AgentEnvironment,
    parsed_calls: Sequence[ParsedToolCall],
    metrics: AgentMetrics,
    config: AgentLoopConfig,
    *,
    prepare_result: Callable[[ToolResult], ToolResult] = lambda result: result,
    record_result: Callable[[AgentMetrics, ToolResult], None] = lambda metrics, result: None,
) -> tuple[list[AgentCall], bool]:
    """Run a turn's tool calls one after another, in the order written; return them as AgentCalls and whether the
    environment ended the episode.

    The first ``max_tool_calls_per_turn`` calls within the trajectory's remaining calls run (each counts as one
    call, a malformed one too, which gets an error result and never reaches the environment); every other call gets
    an error result and does not count. Calls after one that ends the episode do not run. ``prepare_result`` turns
    an environment result into what the model is shown; ``record_result`` counts each result in ``metrics``."""
    per_turn = config.max_tool_calls_per_turn
    remaining = config.max_tool_calls - metrics.tool_call_attempts
    calls: list[AgentCall] = []
    for index, parsed in enumerate(parsed_calls):
        call = AgentCall(tool_call=parsed.call, tool_call_error=parsed.error)
        calls.append(call)
        name = "unknown" if parsed.call is None else parsed.call.name
        if index >= per_turn:
            metrics.unrun_tool_calls += 1
            call.tool_result = _too_many_calls_error(name, per_turn)
        elif index >= remaining:
            metrics.unrun_tool_calls += 1
            metrics.tool_call_limit_reached = True
            call.tool_result = tool_error(
                name,
                TOOL_CALL_LIMIT_REACHED,
                f"not run: the {config.max_tool_calls} tool calls of the response are used up",
            )
        else:
            metrics.tool_call_attempts += 1
            if parsed.error is not None:
                metrics.malformed_tool_calls += 1
                call.tool_result = tool_error("unknown", "malformed_tool_call", parsed.error)
            else:
                env_step = await environment.step(parsed.call)
                call.executed = True
                call.tool_result = prepare_result(env_step.result)
                call.env_reward = checked_step_reward(env_step.reward)
                call.state_key = env_step.state_key
                call.cost = env_step.cost
                call.env_info = dict(env_step.info)
                call.env_done = bool(env_step.done)
        record_result(metrics, call.tool_result)
        if call.env_done:
            # No model turn would read the observation, so it is not committed. The calls after this one do not
            # run; those beyond the per-turn limit keep that error, so the turn reads as having too many.
            for later_index, later in enumerate(parsed_calls[index + 1 :], start=index + 1):
                metrics.unrun_tool_calls += 1
                calls.append(AgentCall(tool_call=later.call, tool_call_error=later.error))
                if later_index >= per_turn:
                    later_name = "unknown" if later.call is None else later.call.name
                    calls[-1].tool_result = _too_many_calls_error(later_name, per_turn)
                    record_result(metrics, calls[-1].tool_result)
            return calls, True
    return calls, False


class AgentLoop:
    """Backend-agnostic token-in/token-out agent loop.

    Model-produced IDs are appended verbatim with response mask 1. Well-formed
    tool calls go to ``environment``; only its observation is encoded by
    ``observation_encoder`` and receives response mask 0.
    """

    _IMAGE_IDX_ERROR_CODES = {
        "missing_image_idx",
        "invalid_image_idx_type",
        "image_idx_out_of_range",
    }

    def __init__(
        self,
        backend: GenerationBackend,
        observation_encoder: ObservationEncoder,
        environment: AgentEnvironment,
        config: Optional[AgentLoopConfig] = None,
        tool_call_codec: Optional[ToolCallCodec] = None,
        seed_context: Optional[AgentSeedContext] = None,
    ):
        self.backend = backend
        self.observation_encoder = observation_encoder
        self.environment = environment
        self.config = config or AgentLoopConfig()
        self.tool_call_codec = tool_call_codec or NativeToolCallCodec()
        if seed_context is not None and not isinstance(seed_context, AgentSeedContext):
            raise TypeError("seed_context must be an AgentSeedContext")
        self.seed_context = seed_context
        image_config = getattr(observation_encoder, "image_config", AgentImageConfig())
        if not isinstance(image_config, AgentImageConfig):
            raise TypeError("observation encoder image_config must be an AgentImageConfig")
        self.image_config = image_config
        self.assistant_termination_token_ids = tuple(
            getattr(observation_encoder, "assistant_termination_token_ids", ())
        )
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in self.assistant_termination_token_ids
        ):
            raise TypeError("assistant termination token IDs must be integers")
        encoder_max_tool_calls = getattr(observation_encoder, "max_tool_calls", None)
        if encoder_max_tool_calls is not None and encoder_max_tool_calls != self.config.max_tool_calls:
            raise ValueError(
                "observation encoder and agent loop disagree on max_tool_calls: "
                f"{encoder_max_tool_calls} != {self.config.max_tool_calls}"
            )

    async def run(
        self,
        prompt_ids: Sequence[int],
        source_images: Sequence[Image.Image],
        *,
        effective_prompt_tokens: Optional[int] = None,
        max_model_len: Optional[int] = None,
    ) -> AgentTrajectory:
        if effective_prompt_tokens is not None and effective_prompt_tokens < len(prompt_ids):
            raise ValueError("effective prompt token count cannot be smaller than raw prompt length")
        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be positive when provided")
        prompt_ids = list(prompt_ids)
        source_images = list(source_images)
        response_ids: list[int] = []
        response_mask: list[int] = []
        observation_images: list[Image.Image] = []
        steps: list[AgentStep] = []
        metrics = AgentMetrics()
        final_answer: Optional[str] = None
        status = AgentStatus.NO_ACTION
        effective_response_tokens = 0

        while True:
            remaining_tokens = self.config.max_response_tokens - effective_response_tokens
            if remaining_tokens <= 0:
                metrics.truncated = True
                metrics.trajectory_length_limited = True
                status = AgentStatus.LENGTH_LIMIT
                break

            tools_allowed = metrics.tool_call_attempts < self.config.max_tool_calls
            reasoning_prefilled = bool(getattr(self.observation_encoder, "reasoning_prefilled", False))
            context_remaining: Optional[int] = None
            if effective_prompt_tokens is not None and max_model_len is not None:
                context_remaining = max_model_len - effective_prompt_tokens - effective_response_tokens
                if context_remaining <= 0:
                    metrics.truncated = True
                    metrics.context_window_limited = True
                    status = AgentStatus.LENGTH_LIMIT
                    break
            max_new_tokens = min(
                remaining_tokens,
                self.config.max_tokens_per_turn,
                *(() if context_remaining is None else (context_remaining,)),
            )
            output = await self.backend.generate(
                GenerationRequest(
                    prompt_token_ids=[*prompt_ids, *response_ids],
                    images=[*source_images, *observation_images],
                    max_new_tokens=max_new_tokens,
                    seed=(None if self.seed_context is None else self.seed_context.for_turn(len(steps))),
                )
            )
            action_token_ids = list(output.token_ids)
            if len(action_token_ids) > max_new_tokens:
                raise ValueError(
                    "generation backend returned more tokens than requested: "
                    f"{len(action_token_ids)} > {max_new_tokens}"
                )

            response_ids.extend(action_token_ids)
            response_mask.extend([1] * len(action_token_ids))
            effective_response_tokens += len(action_token_ids)
            metrics.action_tokens += len(action_token_ids)
            step = AgentStep(
                model_text=output.text,
                model_token_ids=action_token_ids,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
                backend_text=output.backend_text,
                reasoning_prefilled=reasoning_prefilled,
                is_finalization_turn=not tools_allowed,
            )
            steps.append(step)

            # A length-stopped output may end with syntactically complete tags
            # only by accident. Never execute or accept a partial generation.
            if output.finish_reason == "length":
                metrics.truncated = True
                if len(action_token_ids) == max_new_tokens:
                    if max_new_tokens == self.config.max_tokens_per_turn:
                        metrics.turn_length_limited = True
                    if max_new_tokens == remaining_tokens:
                        metrics.trajectory_length_limited = True
                    if context_remaining is not None and max_new_tokens == context_remaining:
                        metrics.context_window_limited = True
                else:
                    # vLLM uses the same finish reason for max_tokens and
                    # max_model_len. Since the backend returned fewer tokens
                    # than requested, the model context window bound first.
                    metrics.context_window_limited = True
                status = AgentStatus.LENGTH_LIMIT
                break

            if (
                self.assistant_termination_token_ids
                and tuple(action_token_ids[-len(self.assistant_termination_token_ids) :])
                != self.assistant_termination_token_ids
            ):
                tail_width = len(self.assistant_termination_token_ids)
                step.expected_termination_token_ids = list(self.assistant_termination_token_ids)
                step.actual_termination_tail_ids = action_token_ids[-tail_width:]
                step.turn_end_error = (
                    "model action does not end with the native assistant "
                    "termination token sequence: "
                    f"expected={step.expected_termination_token_ids}, "
                    f"actual_tail={step.actual_termination_tail_ids}"
                )
                metrics.turn_end_errors += 1
                status = AgentStatus.INVALID_TURN_END
                break

            parsed = self.tool_call_codec.parse(
                output.text,
                reasoning_open=reasoning_prefilled,
            )
            if parsed.final_answer is not None:
                final_answer = parsed.final_answer
                status = AgentStatus.ANSWERED
                break

            if parsed.final_answer_error is not None:
                step.final_answer_error = parsed.final_answer_error
                metrics.invalid_final_answers += 1
                status = AgentStatus.INVALID_FINAL_ANSWER
                break

            if not parsed.has_tool_attempt:
                if effective_response_tokens >= self.config.max_response_tokens:
                    metrics.truncated = True
                    metrics.trajectory_length_limited = True
                    status = AgentStatus.LENGTH_LIMIT
                else:
                    status = AgentStatus.NO_ACTION
                break

            if not tools_allowed:
                metrics.tool_call_limit_reached = True
                status = AgentStatus.TOOL_CALL_LIMIT
                break

            step.calls, done = await run_tool_calls(
                self.environment,
                parsed.tool_calls,
                metrics,
                self.config,
                prepare_result=self._resize_observation_images,
                record_result=self._record_tool_result,
            )
            if done:
                status = AgentStatus.ENVIRONMENT_DONE
                break
            results = tuple(call.tool_result for call in step.calls)

            remaining_tokens = self.config.max_response_tokens - effective_response_tokens
            if remaining_tokens <= 0:
                metrics.truncated = True
                metrics.trajectory_length_limited = True
                status = AgentStatus.LENGTH_LIMIT
                break

            encoded = await self.observation_encoder.encode(ObservationEncodingRequest(results=results))
            observation_token_ids = list(encoded.token_ids)
            observation_effective_tokens = (
                len(observation_token_ids) if encoded.effective_token_count is None else encoded.effective_token_count
            )
            if observation_effective_tokens < len(observation_token_ids):
                raise ValueError("effective observation token count cannot be smaller than its raw token count")
            if encoded.visual_token_count is not None and (
                encoded.visual_token_count < 0 or encoded.visual_token_count > observation_effective_tokens
            ):
                raise ValueError("visual_token_count must be within the encoded observation token range")
            if observation_effective_tokens > remaining_tokens:
                # Never append a partial multimodal observation because doing so
                # could separate an image placeholder from its processor data.
                metrics.truncated = True
                metrics.trajectory_length_limited = True
                status = AgentStatus.LENGTH_LIMIT
                break
            if (
                effective_prompt_tokens is not None
                and max_model_len is not None
                and effective_prompt_tokens + effective_response_tokens + observation_effective_tokens >= max_model_len
            ):
                # An observation is committed only if a following model turn
                # can consume it and still generate at least one token.
                metrics.truncated = True
                metrics.context_window_limited = True
                status = AgentStatus.LENGTH_LIMIT
                break

            step.observation_token_ids = observation_token_ids
            step.observation_committed = True
            step.observation_image_count = len(encoded.images)
            response_ids.extend(observation_token_ids)
            response_mask.extend([0] * len(observation_token_ids))
            effective_response_tokens += observation_effective_tokens
            observation_images.extend(encoded.images)
            metrics.tool_call_successes += sum(result.success for result in results)
            metrics.raw_observation_tokens += len(observation_token_ids)
            metrics.observation_tokens += observation_effective_tokens
            if metrics.visual_tokens is not None:
                if encoded.visual_token_count is None:
                    metrics.visual_tokens = None
                else:
                    metrics.visual_tokens += encoded.visual_token_count
            metrics.visual_observations += len(encoded.images)

        if len(response_ids) != len(response_mask):
            raise RuntimeError("response_ids and response_mask must stay aligned")

        return AgentTrajectory(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            source_images=source_images,
            observation_images=observation_images,
            steps=steps,
            status=status,
            metrics=metrics,
            final_answer=final_answer,
            effective_response_tokens=effective_response_tokens,
            response_token_budget=self.config.max_response_tokens,
            image_config=self.image_config,
            assistant_termination_token_ids=self.assistant_termination_token_ids,
        )

    def _resize_observation_images(self, result: ToolResult) -> ToolResult:
        """With observation_min_pixels, resize tool images once to [observation_min_pixels, max_pixels]; the
        wrapped result is not resized again by the rollout, the observation encoder or the trainer."""
        min_pixels = self.image_config.observation_min_pixels
        if min_pixels is None or not result.images:
            return result
        return replace(
            result,
            images=[
                ProcessedImageInput(process_image(image, min_pixels, self.image_config.max_pixels))
                for image in result.images
            ],
        )

    def _record_tool_result(self, metrics: AgentMetrics, result: ToolResult) -> None:
        if result.success:
            metrics.tool_execution_successes += 1
        else:
            metrics.tool_call_errors += 1
            if result.error_code == "tool_internal_error":
                metrics.tool_internal_errors += 1
            if result.error_code in self._IMAGE_IDX_ERROR_CODES:
                metrics.image_idx_errors += 1
