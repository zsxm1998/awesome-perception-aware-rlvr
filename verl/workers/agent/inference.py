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
from typing import Any, Mapping, Optional

from vllm import SamplingParams

from .backends import (
    VLLMAgentBatchScheduler,
    VLLMAgentImageCache,
    VLLMGenerationBackend,
)
from .loop import AgentLoop
from .protocol import (
    AgentEnvironment,
    AgentImageConfig,
    AgentLoopConfig,
    AgentSeedContext,
    AgentTrajectory,
)
from .trajectory import measure_expanded_observation


async def run_agent_inference(
    environment: AgentEnvironment,
    env_config: Mapping[str, Any],
    inference_engine: Any,
    sampling_params: SamplingParams,
    processor: Any,
    *,
    sample_index: int,
    rollout_index: int,
    image_config: AgentImageConfig,
    env_seed: int,
    config: Optional[AgentLoopConfig] = None,
    lora_request: Optional[Any] = None,
    use_tqdm: bool = False,
    base_seed: Optional[int] = None,
    scheduler: VLLMAgentBatchScheduler,
    image_cache: Optional[VLLMAgentImageCache] = None,
    max_model_len: int,
) -> AgentTrajectory:
    """Run one inference-only agent trajectory in ``environment`` and close it.

    The environment's ``reset`` gives the task (``env_config`` is its row-level configuration, ``env_seed``
    its group-level seed); its chat adapter renders the prompt and observations. One immutable image
    configuration is carried from rollout through trajectory materialization. One call produces one
    trajectory; group sampling schedules each member with explicit sample and rollout identities.
    """

    agent_config = config or AgentLoopConfig()
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int):
        raise TypeError("max_model_len must be an integer")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    if not isinstance(image_config, AgentImageConfig):
        raise TypeError("image_config must be an AgentImageConfig")
    resolved_base_seed = base_seed
    if resolved_base_seed is None and isinstance(getattr(sampling_params, "seed", None), int):
        resolved_base_seed = int(sampling_params.seed)
    if resolved_base_seed is None:
        raise ValueError("agent inference requires an explicit deterministic base seed")
    try:
        reset = await environment.reset(env_config, env_seed)
        source_images = list(reset.images)
        if not reset.tool_schemas:  # no tool offered: a single turn (e.g. DeepEyes' ThinkLite rows)
            agent_config = replace(agent_config, max_tool_calls=0)
        required_images = len(source_images) + agent_config.max_tool_calls * reset.max_images_per_step
        if image_config.limit_images is not None and image_config.limit_images < required_images:
            raise ValueError(
                "agent rollout image capacity is too small: "
                f"limit_images={image_config.limit_images}, required at least {required_images}"
            )
        resolved_image_cache = image_cache or VLLMAgentImageCache(
            min_pixels=image_config.min_pixels,
            max_pixels=image_config.max_pixels,
        )
        backend = VLLMGenerationBackend(
            inference_engine=inference_engine,
            sampling_params=sampling_params,
            tokenizer=processor.tokenizer,
            lora_request=lora_request,
            use_tqdm=use_tqdm,
            min_pixels=image_config.min_pixels,
            max_pixels=image_config.max_pixels,
            scheduler=scheduler,
            image_cache=resolved_image_cache,
            forbidden_token_ids=(
                (int(processor.image_token_id),) if isinstance(getattr(processor, "image_token_id", None), int) else ()
            ),
        )
        chat_adapter = environment.chat_adapter(
            processor,
            reset,
            max_tool_calls=agent_config.max_tool_calls,
            image_config=image_config,
            image_preprocessor=resolved_image_cache.prepare,
        )
        prompt = chat_adapter.encode_initial_prompt(reset.messages)
        effective_prompt_tokens, prompt_visual_tokens = measure_expanded_observation(
            processor,
            prompt.token_ids,
            source_images,
            min_pixels=image_config.min_pixels,
            max_pixels=image_config.max_pixels,
            image_preprocessor=resolved_image_cache.prepare,
        )
        loop = AgentLoop(
            backend=backend,
            observation_encoder=chat_adapter,
            environment=environment,
            config=agent_config,
            seed_context=AgentSeedContext(
                base_seed=resolved_base_seed,
                sample_index=sample_index,
                rollout_index=rollout_index,
            ),
        )
        trajectory = await loop.run(
            prompt_ids=prompt.token_ids,
            source_images=source_images,
            effective_prompt_tokens=effective_prompt_tokens,
            max_model_len=max_model_len,
        )
        trajectory.effective_prompt_tokens = effective_prompt_tokens
        trajectory.prompt_visual_tokens = prompt_visual_tokens
        trajectory.env_outcome = await environment.finalize(trajectory.final_answer, trajectory.status)
        return trajectory
    finally:
        await environment.close()
