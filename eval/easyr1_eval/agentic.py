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
from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from verl.workers.agent.backends import (
    VLLMAgentBatchScheduler,
    VLLMAgentImageCache,
    VLLMAgentRequestError,
    VLLMAgentSchedulerError,
)
from verl.workers.agent.inference import run_deepeyes_inference
from verl.workers.agent.protocol import (
    AgentImageConfig,
    AgentLoopConfig,
    AgentStatus,
    AgentTrajectory,
)
from verl.workers.agent.tool_call import NativeToolCallCodec

from .backends import VLLMBackend, image_size_records, process_eval_image
from .schemas import EvalSample, GenerationConfig, GenerationOutput


if TYPE_CHECKING:
    from vllm import SamplingParams


AgentRunner = Callable[..., Awaitable[AgentTrajectory]]
_AGENT_PROFILES: dict[str, AgentRunner] = {
    "deepeyes": run_deepeyes_inference,
}
_ANSWER_OPEN_PATTERN = re.compile(r"<answer>", flags=re.IGNORECASE)
_ANSWER_CLOSE_PATTERN = re.compile(r"</answer>", flags=re.IGNORECASE)
_TOOL_CALL_BLOCK_PATTERN = re.compile(
    r"<tool_call>.*?(?:</tool_call>|$)",
    flags=re.DOTALL | re.IGNORECASE,
)


class AgenticConfigurationError(ValueError):
    """An evaluation configuration that cannot represent the requested rollout."""


class AgenticSampleError(ValueError):
    """A malformed or unreadable individual evaluation sample."""


class AgenticVLLMBackend:
    """Run independent agent trajectories over one shared batched vLLM engine."""

    def __init__(
        self,
        model: str,
        *,
        agent_profile: str,
        agent_config: AgentLoopConfig,
        agent_max_images_per_prompt: int,
        agent_max_batch_images: int,
        agent_tool_image_mode: str = "original",
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        dtype: str = "bfloat16",
        perturbation: Any = None,
        perturbation_seed: int = 0,
        output_dir: Any = None,
        save_perturbation_samples: int = 0,
        force_vllm_feature_wrapper: bool = False,
        agent_bbox_format: str = "norm1000",
        chat_template: str | None = None,
        plain_think_tokens: str = "auto",
    ):
        try:
            self.agent_runner = _AGENT_PROFILES[agent_profile]
        except KeyError as exc:
            raise ValueError(f"unknown agent profile: {agent_profile}") from exc
        if not isinstance(agent_config, AgentLoopConfig):
            raise TypeError("agent_config must be an AgentLoopConfig")
        if agent_max_images_per_prompt <= 0:
            raise ValueError("agent_max_images_per_prompt must be positive")
        if agent_max_batch_images <= 0:
            raise ValueError("agent_max_batch_images must be positive")
        if agent_tool_image_mode not in {"original", "fixed_gray", "text_skipped"}:
            raise ValueError("agent_tool_image_mode must be 'original', 'fixed_gray', or 'text_skipped'")
        if getattr(perturbation, "enabled", False) or force_vllm_feature_wrapper:
            raise ValueError("agentic evaluation does not yet support pixel or feature perturbations")
        if agent_bbox_format not in {"norm1000", "pixel"}:
            raise ValueError("agent_bbox_format must be 'norm1000' or 'pixel'")

        self.agent_profile = agent_profile
        self.agent_config = agent_config
        self.agent_max_images_per_prompt = agent_max_images_per_prompt
        self.agent_max_batch_images = agent_max_batch_images
        self.agent_tool_image_mode = agent_tool_image_mode
        # Coordinate convention of the zoom-in tool's bbox_2d: 0-1000 of the source image, or
        # absolute pixels of the frame a Qwen2-VL / Qwen2.5-VL model sees.
        self.agent_bbox_format = agent_bbox_format
        self.base = VLLMBackend(
            model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            limit_images=agent_max_images_per_prompt,
            dtype=dtype,
            perturbation=perturbation,
            perturbation_seed=perturbation_seed,
            output_dir=output_dir,
            save_perturbation_samples=save_perturbation_samples,
            force_vllm_feature_wrapper=False,
            chat_template=chat_template,
            plain_think_tokens=plain_think_tokens,
        )
        self.processor = self.base.processor
        self.tokenizer = self.processor.tokenizer
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        resolved_max_model_len = max_model_len
        if resolved_max_model_len is None:
            model_config = getattr(
                getattr(self.base.llm, "llm_engine", None),
                "model_config",
                None,
            )
            resolved_max_model_len = getattr(model_config, "max_model_len", None)
        self.max_model_len = (
            int(resolved_max_model_len)
            if isinstance(resolved_max_model_len, int) and not isinstance(resolved_max_model_len, bool)
            else None
        )
        if self.max_model_len is None:
            raise RuntimeError("unable to resolve max_model_len from the vLLM engine")

    def generate(
        self,
        samples: list[EvalSample],
        config: GenerationConfig,
    ) -> list[list[GenerationOutput]]:
        return asyncio.run(self._generate_async(samples, config))

    async def _generate_async(
        self,
        samples: list[EvalSample],
        config: GenerationConfig,
    ) -> list[list[GenerationOutput]]:
        if config.max_new_tokens != self.agent_config.max_response_tokens:
            raise ValueError(
                "agentic generation and loop disagree on response budget: "
                f"{config.max_new_tokens} != "
                f"{self.agent_config.max_response_tokens}"
            )
        image_cache = VLLMAgentImageCache(
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            **({} if config.top_k is None else {"top_k": config.top_k}),
            n=1,
            max_tokens=self.agent_config.max_tokens_per_turn,
            seed=config.seed,
        )
        image_config = AgentImageConfig(
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            limit_images=self.agent_max_images_per_prompt,
        )

        source_images_by_sample: list[tuple[list[Any] | None, BaseException | None]] = []
        image_sizes_by_sample: list[list[dict] | None] = []
        for sample in samples:
            try:
                source_images = [
                    process_eval_image(image, min_pixels=None, max_pixels=None) for image in sample.images
                ]
            except Exception as exc:  # noqa: BLE001
                source_images_by_sample.append((None, exc))
                image_sizes_by_sample.append(None)
            else:
                source_images_by_sample.append((source_images, None))
                # The frame the policy reads and writes coordinates in (data-side resize, then the
                # processor's own resize): needed to read pixel boxes of the final answer.
                try:
                    sizes = image_size_records(
                        sample, [image_cache.prepare(image) for image in source_images], self.processor
                    )
                except Exception:  # noqa: BLE001 - diagnostics only; the trajectory reports real errors
                    sizes = None
                image_sizes_by_sample.append(sizes)

        for sample, (source_images, source_image_error) in zip(
            samples,
            source_images_by_sample,
        ):
            if source_image_error is not None:
                continue
            assert source_images is not None
            required_images = len(source_images)
            if self.agent_tool_image_mode != "text_skipped":
                required_images += self.agent_config.max_tool_calls
            if required_images > self.agent_max_images_per_prompt:
                raise AgenticConfigurationError(
                    "DeepEyes rollout image capacity is too small for "
                    f"{sample.benchmark}:{sample.sample_id}: "
                    f"limit_images={self.agent_max_images_per_prompt}, "
                    f"required at least {required_images}"
                )

        scheduler = VLLMAgentBatchScheduler(
            self.base.llm,
            use_tqdm=False,
            max_batch_size=max(1, len(samples) * config.num_samples),
            max_batch_images=self.agent_max_batch_images,
        )
        identities: list[tuple[int, int]] = []
        tasks = []
        for sample_position, sample in enumerate(samples):
            source_images, source_image_error = source_images_by_sample[sample_position]
            for rollout_index in range(config.num_samples):
                identities.append((sample_position, rollout_index))
                tasks.append(
                    asyncio.create_task(
                        self._run_trajectory(
                            sample,
                            source_images=source_images,
                            source_image_error=source_image_error,
                            sampling_params=sampling_params,
                            image_config=image_config,
                            sample_index=_stable_sample_index(sample),
                            rollout_index=rollout_index,
                            base_seed=config.seed,
                            scheduler=scheduler,
                            image_cache=image_cache,
                        )
                    )
                )

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await scheduler.close()
        scheduler_failures = [result for result in results if isinstance(result, VLLMAgentSchedulerError)]
        if scheduler_failures:
            signatures = sorted({str(error) for error in scheduler_failures})
            raise RuntimeError("shared agentic vLLM scheduler failed: " + "; ".join(signatures[:3]))
        engine_failures = [result for result in results if isinstance(result, VLLMAgentRequestError)]
        if engine_failures and len(engine_failures) == len(results):
            signatures = sorted({str(error) for error in engine_failures})
            raise RuntimeError(
                "every trajectory in the agentic batch failed inside vLLM: " + "; ".join(signatures[:3])
            )
        grouped: list[list[GenerationOutput | None]] = [[None] * config.num_samples for _ in samples]
        for (sample_position, rollout_index), result in zip(identities, results):
            if isinstance(result, BaseException):
                output = _failed_trajectory_output(result)
            else:
                output = trajectory_to_generation_output(
                    result,
                    tokenizer=self.tokenizer,
                )
                if image_sizes_by_sample[sample_position]:
                    output.diagnostics["image_sizes"] = image_sizes_by_sample[sample_position]
            grouped[sample_position][rollout_index] = output

        return [[output for output in sample_outputs if output is not None] for sample_outputs in grouped]

    async def _run_trajectory(
        self,
        sample: EvalSample,
        *,
        source_images: list[Any] | None,
        source_image_error: BaseException | None,
        sampling_params: SamplingParams,
        image_config: AgentImageConfig,
        sample_index: int,
        rollout_index: int,
        base_seed: int,
        scheduler: VLLMAgentBatchScheduler,
        image_cache: VLLMAgentImageCache,
    ) -> AgentTrajectory:
        if not sample.messages:
            raise AgenticSampleError("agentic evaluation requires chat-mode messages for every sample")
        if source_image_error is not None:
            raise AgenticSampleError(
                "failed to load source images for agentic evaluation: "
                f"{type(source_image_error).__name__}: {source_image_error}"
            ) from source_image_error
        assert source_images is not None
        return await self.agent_runner(
            inference_engine=self.base.llm,
            sampling_params=sampling_params,
            processor=self.processor,
            messages=sample.messages,
            source_images=source_images,
            sample_index=sample_index,
            rollout_index=rollout_index,
            image_config=image_config,
            config=self.agent_config,
            use_tqdm=False,
            base_seed=base_seed,
            scheduler=scheduler,
            image_cache=image_cache,
            max_model_len=self.max_model_len,
            tool_image_mode=self.agent_tool_image_mode,
            bbox_format=self.agent_bbox_format,
        )


def trajectory_to_generation_output(
    trajectory: AgentTrajectory,
    *,
    tokenizer: Any,
) -> GenerationOutput:
    """Expose one score-facing final reply plus compact trajectory diagnostics."""

    action_text = "".join(step.model_text for step in trajectory.steps)
    final_step = trajectory.steps[-1] if trajectory.steps else None
    scoring_text = _score_facing_response(
        trajectory,
        tokenizer=tokenizer,
    )
    finish_reason = final_step.finish_reason if final_step is not None else None
    if trajectory.status == AgentStatus.LENGTH_LIMIT:
        finish_reason = "length"

    turn_end_failures = []
    for turn_index, step in enumerate(trajectory.steps):
        if step.turn_end_error is None:
            continue
        turn_end_failures.append(
            {
                "turn_index": turn_index,
                "stop_reason": step.stop_reason,
                "stop_reason_text": _decode_stop_reason(tokenizer, step.stop_reason),
                "expected_token_ids": list(step.expected_termination_token_ids),
                "expected_text": _decode_token_ids(
                    tokenizer,
                    step.expected_termination_token_ids,
                ),
                "actual_tail_token_ids": list(step.actual_termination_tail_ids),
                "actual_tail_text": _decode_token_ids(
                    tokenizer,
                    step.actual_termination_tail_ids,
                ),
            }
        )

    codec = NativeToolCallCodec()
    format_replayed_inside_reasoning = any(
        codec.contains_answer_inside_reasoning(
            step.model_text,
            reasoning_open=step.reasoning_prefilled,
        )
        for step in trajectory.steps
    )
    tool_calls_inside_reasoning = sum(
        codec.count_tool_calls_inside_reasoning(
            step.model_text,
            reasoning_open=step.reasoning_prefilled,
        )
        for step in trajectory.steps
    )
    tool_errors = [
        {
            "turn_index": turn_index,
            "error_code": step.tool_result.error_code or "unknown",
            "message": _tool_error_message(step.tool_result)[:300],
        }
        for turn_index, step in enumerate(trajectory.steps)
        if step.tool_result is not None and not step.tool_result.success
    ]
    committed_tool_regions = []
    for turn_index, step in enumerate(trajectory.steps):
        if (
            step.tool_call is None
            or step.tool_call.name != "image_zoom_in_tool"
            or step.tool_result is None
            or not step.tool_result.success
            or not step.observation_committed
        ):
            continue
        metadata = step.tool_result.metadata
        bbox = metadata.get("bbox_2d")
        # Source-image 0-1000 box of the crop, identical for norm1000 and pixel runs; bbox_2d is the
        # model's own argument (absolute frame pixels in pixel mode).
        bbox_norm1000 = metadata.get("bbox_norm1000", bbox)
        image_idx = metadata.get("image_idx")
        if (
            not isinstance(bbox_norm1000, (list, tuple))
            or len(bbox_norm1000) != 4
            or isinstance(image_idx, bool)
            or not isinstance(image_idx, int)
        ):
            raise RuntimeError("committed image_zoom_in_tool result is missing canonical region metadata")
        region = {
            "turn_index": turn_index,
            "image_idx": image_idx,
            "bbox_norm1000": list(bbox_norm1000),
            "bbox_2d": list(bbox) if isinstance(bbox, (list, tuple)) else bbox,
            "bbox_format": metadata.get("bbox_format", "norm1000"),
        }
        label = metadata.get("label")
        if isinstance(label, str) and label:
            region["label"] = label
        committed_tool_regions.append(region)
    metrics = trajectory.metrics
    agent_diagnostics = {
        "status": trajectory.status.value,
        "final_answer": trajectory.final_answer,
        "num_steps": len(trajectory.steps),
        "tool_call_attempts": metrics.tool_call_attempts,
        "tool_execution_successes": metrics.tool_execution_successes,
        "tool_call_successes": metrics.tool_call_successes,
        "tool_call_errors": metrics.tool_call_errors,
        "tool_internal_errors": metrics.tool_internal_errors,
        "image_idx_errors": metrics.image_idx_errors,
        "invalid_final_answers": metrics.invalid_final_answers,
        "turn_end_errors": metrics.turn_end_errors,
        "action_tokens": metrics.action_tokens,
        "raw_observation_tokens": metrics.raw_observation_tokens,
        "observation_tokens": metrics.observation_tokens,
        "visual_tokens": metrics.visual_tokens,
        "visual_observations": metrics.visual_observations,
        "raw_response_tokens": len(trajectory.response_ids),
        "effective_response_tokens": trajectory.effective_response_tokens,
        "raw_prompt_tokens": len(trajectory.prompt_ids),
        "effective_prompt_tokens": trajectory.effective_prompt_tokens,
        "prompt_visual_tokens": trajectory.prompt_visual_tokens,
        "raw_total_tokens": len(trajectory.input_ids),
        "effective_total_tokens": (
            None
            if trajectory.effective_prompt_tokens is None or trajectory.effective_response_tokens is None
            else trajectory.effective_prompt_tokens + trajectory.effective_response_tokens
        ),
        "response_token_budget": trajectory.response_token_budget,
        "truncated": metrics.truncated,
        "turn_length_limited": metrics.turn_length_limited,
        "trajectory_length_limited": metrics.trajectory_length_limited,
        "context_window_limited": metrics.context_window_limited,
        "tool_call_limit_reached": metrics.tool_call_limit_reached,
        "no_action": trajectory.status == AgentStatus.NO_ACTION,
        "opening_answer_unclosed": (
            len(_ANSWER_OPEN_PATTERN.findall(action_text)) > len(_ANSWER_CLOSE_PATTERN.findall(action_text))
        ),
        "empty_answer": any(step.final_answer_error == "final answer must not be empty" for step in trajectory.steps),
        "invalid_answer": trajectory.status == AgentStatus.INVALID_FINAL_ANSWER,
        "format_replayed_inside_reasoning": format_replayed_inside_reasoning,
        # DeepEyes executes complete tool calls regardless of reasoning-tag
        # balance. Preserve that behavior while exposing format drift.
        "tool_calls_inside_reasoning": tool_calls_inside_reasoning,
        "turn_end_failures": turn_end_failures,
        # This is a compact, score-independent record of the visual evidence
        # actions actually observed by the model. It is intentionally kept
        # outside score-facing ``responses``.
        "committed_tool_regions": committed_tool_regions,
        # Why failed tool calls failed (invalid bbox, image_idx, ...).
        "tool_errors": tool_errors,
    }
    return GenerationOutput(
        text=scoring_text,
        finish_reason=finish_reason,
        stop_reason=(None if final_step is None else final_step.stop_reason),
        token_count=(0 if final_step is None else len(final_step.model_token_ids)),
        truncated=metrics.truncated,
        diagnostics={"agent": agent_diagnostics},
    )


def _tool_error_message(result: Any) -> str:
    content = result.content if isinstance(result.content, dict) else {}
    error = content.get("error") if isinstance(content.get("error"), dict) else {}
    return str(error.get("message") or "")


def _score_facing_response(
    trajectory: AgentTrajectory,
    *,
    tokenizer: Any,
) -> str:
    """Build the standalone final assistant reply consumed by existing scorers.

    Earlier assistant actions are environment interaction history, not answer
    predictions for the benchmark. Tool-call payloads therefore stay out of
    this text; evidence-grounding scorers consume their separately persisted
    structured regions when the benchmark contract calls for it.
    """

    if not trajectory.steps:
        return ""
    step = trajectory.steps[-1]
    token_ids = list(step.model_token_ids)
    termination = tuple(trajectory.assistant_termination_token_ids)
    if termination and len(token_ids) >= len(termination) and tuple(token_ids[-len(termination) :]) == termination:
        del token_ids[-len(termination) :]
    else:
        stop_token_ids = _stop_token_ids(step.stop_reason)
        special_ids = {
            int(token_id)
            for token_id in (getattr(tokenizer, "all_special_ids", None) or [])
            if isinstance(token_id, int) and not isinstance(token_id, bool)
        }
        if (
            stop_token_ids
            and all(token_id in special_ids for token_id in stop_token_ids)
            and len(token_ids) >= len(stop_token_ids)
            and tuple(token_ids[-len(stop_token_ids) :]) == stop_token_ids
        ):
            del token_ids[-len(stop_token_ids) :]
    text = _decode_token_ids(tokenizer, token_ids)
    if step.reasoning_prefilled:
        text = f"<think>{text}"
    return _TOOL_CALL_BLOCK_PATTERN.sub("", text).strip()


def _stop_token_ids(stop_reason: Any) -> tuple[int, ...]:
    if isinstance(stop_reason, int) and not isinstance(stop_reason, bool):
        return (stop_reason,)
    if isinstance(stop_reason, (list, tuple)) and all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in stop_reason
    ):
        return tuple(stop_reason)
    return ()


def _failed_trajectory_output(exc: BaseException) -> GenerationOutput:
    return GenerationOutput(
        text="",
        finish_reason="error",
        token_count=0,
        truncated=False,
        diagnostics={
            "agent": {
                "status": "sample_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        },
    )


def _stable_sample_index(sample: EvalSample) -> int:
    payload = f"{sample.benchmark}\0{sample.sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _decode_stop_reason(tokenizer: Any, stop_reason: Any) -> str | None:
    if isinstance(stop_reason, bool) or stop_reason is None:
        return None if stop_reason is None else str(stop_reason)
    if isinstance(stop_reason, int):
        return _decode_token_ids(tokenizer, [stop_reason])
    if isinstance(stop_reason, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in stop_reason
    ):
        return _decode_token_ids(tokenizer, list(stop_reason))
    return str(stop_reason)


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
