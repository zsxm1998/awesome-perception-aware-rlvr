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
from typing import Optional, Sequence

from PIL import Image

from ...utils.dataset import ProcessedImageInput, process_image
from .protocol import (
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
    ToolResult,
)
from .tool_call import NativeToolCallCodec, ToolCallCodec
from .tools.base import ToolRegistry, tool_error


class AgentLoop:
    """Backend-agnostic token-in/token-out agent loop.

    Model-produced IDs are appended verbatim with response mask 1. Only the
    environment observation is encoded by ``observation_encoder`` and receives
    response mask 0.
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
        tool_registry: ToolRegistry,
        config: Optional[AgentLoopConfig] = None,
        tool_call_codec: Optional[ToolCallCodec] = None,
        seed_context: Optional[AgentSeedContext] = None,
    ):
        self.backend = backend
        self.observation_encoder = observation_encoder
        self.tool_registry = tool_registry
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

            metrics.tool_call_attempts += 1
            if parsed.tool_call_error is not None:
                step.tool_call_error = parsed.tool_call_error
                result = tool_error(
                    "unknown",
                    "malformed_tool_call",
                    parsed.tool_call_error,
                )
            else:
                assert parsed.tool_call is not None
                step.tool_call = parsed.tool_call
                result = self.tool_registry.execute(parsed.tool_call, source_images)
                result = self._resize_observation_images(result)
            step.tool_result = result
            self._record_tool_result(metrics, result)

            remaining_tokens = self.config.max_response_tokens - effective_response_tokens
            if remaining_tokens <= 0:
                metrics.truncated = True
                metrics.trajectory_length_limited = True
                status = AgentStatus.LENGTH_LIMIT
                break

            encoded = await self.observation_encoder.encode(ObservationEncodingRequest(result=result))
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
            response_ids.extend(observation_token_ids)
            response_mask.extend([0] * len(observation_token_ids))
            effective_response_tokens += observation_effective_tokens
            observation_images.extend(encoded.images)
            if result.success:
                metrics.tool_call_successes += 1
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
