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

import asyncio
import hashlib
import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from packaging import version
from tensordict import TensorDict
from transformers import AutoConfig, PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...models.transformers.qwen3_5 import register_qwen3_5
from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.py_functional import get_package_version
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from ..agent.coordinates import resolve_bbox_format
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    return_video_metadata: bool = False,
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(
                process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_metadata=return_video_metadata,
                )
            )

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


def _check_rollout_dependency_versions(model_path: str, trust_remote_code: bool) -> None:
    register_qwen3_5()
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if model_config.model_type != "qwen3_5":
        return

    min_vllm_version = version.parse("0.17.0")
    current_vllm_version = get_package_version("vllm")
    if current_vllm_version < min_vllm_version:
        raise RuntimeError(
            "Qwen3.5 rollout requires vLLM >= 0.17.0. "
            f"Current vLLM is {current_vllm_version}; please upgrade vLLM or use a non-Qwen3.5 model."
        )


def _set_sampling_param(sampling_params: SamplingParams, key: str, value: Any) -> None:
    attr = getattr(type(sampling_params), key, None)
    if isinstance(attr, property) and attr.fset is None:
        if key != "eos_token_id":
            raise AttributeError(f"SamplingParams.{key} is read-only.")

        if value is None:
            eos_token_ids = []
        elif isinstance(value, int):
            eos_token_ids = [value]
        elif isinstance(value, (list, tuple, set)):
            eos_token_ids = list(value)
        else:
            raise TypeError(f"Unsupported eos_token_id type: {type(value).__name__}")

        primary_eos_token_id = eos_token_ids[0] if eos_token_ids else None
        extra_eos_token_ids = set(eos_token_ids[1:])

        sampling_params._eos_token_id = primary_eos_token_id
        all_stop_token_ids = set(sampling_params.stop_token_ids or [])
        if primary_eos_token_id is not None:
            all_stop_token_ids.add(primary_eos_token_id)
        all_stop_token_ids.update(extra_eos_token_ids)
        sampling_params._all_stop_token_ids = all_stop_token_ids
        if extra_eos_token_ids:
            sampling_params.stop_token_ids = list(set(sampling_params.stop_token_ids or []) | extra_eos_token_ids)
        return

    setattr(sampling_params, key, value)
    if key in {"seed", "temperature"}:
        sampling_params.__dict__.pop("sampling_type", None)


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        **kwargs,
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.agent_bbox_format = resolve_bbox_format(config.agent_bbox_format, model_path)
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_token_id = tokenizer.pad_token_id
        self.return_video_metadata = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        _check_rollout_dependency_versions(model_path, trust_remote_code=config.trust_remote_code)

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["mm_processor_cache_gb"] = config.mm_processor_cache_gb
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        if not kwargs:
            yield
            return

        original_sampling_params = self.sampling_params
        updated_sampling_params = original_sampling_params.clone()
        has_updates = False
        for key, value in kwargs.items():
            if not hasattr(updated_sampling_params, key):
                continue
            _set_sampling_param(updated_sampling_params, key, value)
            has_updates = True

        if has_updates:
            self.sampling_params = updated_sampling_params

        try:
            yield
        finally:
            self.sampling_params = original_sampling_params

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if self.config.interaction_mode == "agentic":
            return self._generate_agent_sequences(prompts)

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=self.use_tqdm,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    @torch.no_grad()
    def _generate_agent_sequences(self, prompts: DataProto) -> DataProto:
        from ..agent.backends import (
            VLLMAgentBatchScheduler,
            VLLMAgentImageCache,
        )
        from ..agent.inference import run_deepeyes_inference
        from ..agent.protocol import AgentImageConfig, AgentLoopConfig
        from ..agent.tool_call import NativeToolCallCodec
        from ..agent.trajectory import (
            AgentTrajectoryMaterializer,
            AgentTrajectoryRejected,
            collate_materialized_agent_trajectories,
        )

        if self.processor is None:
            raise RuntimeError("agentic rollout requires a multimodal processor")
        messages_batch = prompts.non_tensor_batch.pop("raw_prompt", None)
        raw_multi_modal_batch = prompts.non_tensor_batch.pop(
            "multi_modal_data",
            None,
        )
        uids = prompts.non_tensor_batch.pop("uid", None)
        prompts.non_tensor_batch.pop("raw_prompt_ids", None)
        if messages_batch is None or raw_multi_modal_batch is None or uids is None:
            raise ValueError("agentic rollout requires aligned raw_prompt, multi_modal_data, and uid fields")
        batch_size = len(prompts)
        if not (len(messages_batch) == len(raw_multi_modal_batch) == len(uids) == batch_size):
            raise ValueError("agentic rollout inputs do not align by sample")

        image_config = AgentImageConfig(
            min_pixels=prompts.meta_info.get("min_pixels"),
            max_pixels=prompts.meta_info.get("max_pixels"),
            limit_images=self.config.limit_images,
        )
        loop_config = AgentLoopConfig(
            max_tool_calls=self.config.agent_max_tool_calls,
            max_response_tokens=self.config.response_length,
            max_tokens_per_turn=self.config.agent_max_tokens_per_turn,
        )
        max_model_len = self.config.max_model_len or self.config.prompt_length + self.config.response_length
        materializer = AgentTrajectoryMaterializer(
            self.processor,
            max_prompt_tokens=self.config.prompt_length,
            max_model_len=max_model_len,
        )

        source_images_batch = []
        for multi_modal_data in raw_multi_modal_batch:
            if not isinstance(multi_modal_data, dict):
                raise TypeError("DeepEyes currently requires image-based multi_modal_data")
            if multi_modal_data.get("videos"):
                raise ValueError("DeepEyes image_zoom_in_tool does not support videos")
            images = multi_modal_data.get("images")
            if not images:
                raise ValueError("DeepEyes rollout requires at least one source image")
            source_images_batch.append([process_image(image, None, None) for image in images])

        lora_requests = self._agent_lora_requests(batch_size)
        with self.update_sampling_params(**prompts.meta_info):
            base_sampling_params = self.sampling_params.clone()
            rollout_n = int(base_sampling_params.n)
            if rollout_n <= 0:
                raise ValueError("agentic rollout requires sampling n > 0")
            base_sampling_params.n = 1
            if not isinstance(base_sampling_params.seed, int):
                raise ValueError("agentic rollout requires an integer sampling seed")
            scheduler = VLLMAgentBatchScheduler(
                self.inference_engine,
                use_tqdm=self.use_tqdm,
                max_batch_images=(self.config.agent_max_batch_images or None),
            )
            image_cache = VLLMAgentImageCache(
                min_pixels=image_config.min_pixels,
                max_pixels=image_config.max_pixels,
            )

            async def run_batch():
                try:
                    identities = []
                    requests = []
                    for sample_position, (
                        messages,
                        source_images,
                        uid,
                    ) in enumerate(zip(messages_batch, source_images_batch, uids)):
                        sample_identity = int.from_bytes(
                            hashlib.blake2b(
                                str(uid).encode("utf-8"),
                                digest_size=8,
                            ).digest(),
                            byteorder="big",
                            signed=False,
                        )
                        for rollout_index in range(rollout_n):
                            identities.append(
                                (
                                    str(uid),
                                    sample_position,
                                    rollout_index,
                                )
                            )
                            requests.append(
                                {
                                    "messages": messages,
                                    "source_images": source_images,
                                    "sample_index": sample_identity,
                                    "rollout_index": rollout_index,
                                    "lora_request": (
                                        None if lora_requests is None else lora_requests[sample_position]
                                    ),
                                }
                            )

                    async def generate_one(request: dict[str, Any], retry_index: int = 0):
                        # A retry retains group identity but changes the deterministic
                        # seed namespace, avoiding an identical invalid resample.
                        return await run_deepeyes_inference(
                            inference_engine=self.inference_engine,
                            sampling_params=base_sampling_params,
                            processor=self.processor,
                            messages=request["messages"],
                            source_images=request["source_images"],
                            sample_index=request["sample_index"],
                            rollout_index=request["rollout_index"],
                            image_config=image_config,
                            config=loop_config,
                            lora_request=request["lora_request"],
                            use_tqdm=False,
                            base_seed=(int(base_sampling_params.seed) + retry_index * 1_000_003),
                            scheduler=scheduler,
                            image_cache=image_cache,
                            max_model_len=max_model_len,
                            bbox_format=self.agent_bbox_format,
                        )

                    trajectories = list(await asyncio.gather(*(generate_one(request) for request in requests)))
                    materialized = []
                    retry_counts = []
                    for index, (request, trajectory) in enumerate(zip(requests, trajectories)):
                        for retry_index in range(self.config.agent_max_trajectory_retries + 1):
                            try:
                                materialized.append(materializer.materialize(trajectory))
                                retry_counts.append(retry_index)
                                break
                            except AgentTrajectoryRejected as error:
                                if not error.recoverable or retry_index >= self.config.agent_max_trajectory_retries:
                                    raise
                                trajectory = await generate_one(
                                    request,
                                    retry_index + 1,
                                )
                                trajectories[index] = trajectory
                    return identities, trajectories, materialized, retry_counts
                finally:
                    await scheduler.close()

            identities, trajectories, materialized, retry_counts = asyncio.run(run_batch())
        repeated_uids = [uid for uid, _, _ in identities]
        output = collate_materialized_agent_trajectories(
            materialized,
            pad_token_id=self.pad_token_id,
            padded_prompt_width=self.config.prompt_length,
            padded_response_width=self.config.response_length,
            uids=repeated_uids,
            meta_info=prompts.meta_info,
        )

        reward_inputs = []
        diagnostics = []
        codec = NativeToolCallCodec()
        for trajectory_id, trajectory in zip(
            output.non_tensor_batch["agent_trajectory_id"],
            trajectories,
        ):
            response = self.tokenizer.decode(
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
            reward_inputs.append(
                {
                    "trajectory_id": str(trajectory_id),
                    "response": response,
                    "final_answer": trajectory.final_answer,
                    "status": trajectory.status.value,
                    "tool_call_successes": metrics.tool_call_successes,
                    "trajectory_retries": retry_counts[len(reward_inputs)],
                }
            )
            diagnostics.append(
                {
                    "trajectory_id": str(trajectory_id),
                    "status": trajectory.status.value,
                    "tool_call_attempts": metrics.tool_call_attempts,
                    "tool_call_successes": metrics.tool_call_successes,
                    "tool_call_errors": metrics.tool_call_errors,
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
        return output

    def _agent_lora_requests(
        self,
        batch_size: int,
    ) -> Optional[list[LoRARequest]]:
        if not self.lora_kwargs:
            return None
        lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
        if not lora_int_ids:
            return None
        lora_int_id = lora_int_ids[0]
        return [
            LoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="/simon-stub-path",
            )
        ] * batch_size
