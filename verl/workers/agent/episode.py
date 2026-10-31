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
"""One episode of an agent environment, token by token: the part of a rollout that does not depend on where the
model's actions come from (a vLLM engine in rollouts and evaluation, a scripted policy in replays)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

from .loop import AgentLoop
from .protocol import (
    AgentEnvironment,
    AgentImageConfig,
    AgentLoopConfig,
    AgentSeedContext,
    AgentTrajectory,
    EnvReset,
    GenerationBackend,
)
from .trajectory import measure_expanded_observation


@dataclass
class EpisodeRun:
    """What one episode produced: the task, the encoder that rendered it and the trajectory."""

    reset: EnvReset
    chat_adapter: Any
    trajectory: AgentTrajectory


async def run_environment_episode(
    environment: AgentEnvironment,
    env_config: Mapping[str, Any],
    processor: Any,
    *,
    env_seed: int,
    config: AgentLoopConfig,
    image_config: AgentImageConfig,
    image_preprocessor: Callable[[Any], Any],
    make_backend: Callable[[], GenerationBackend],
    seed_context: Optional[AgentSeedContext],
    max_model_len: int,
) -> EpisodeRun:
    """Reset ``environment``, run the agent loop with the backend of ``make_backend``, finalize and close.

    The environment's ``reset`` gives the task (``env_config`` is its row-level configuration, ``env_seed`` its
    group-level seed) and its chat adapter renders the prompt and observations.
    """

    agent_config = config
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
        backend = make_backend()
        chat_adapter = environment.chat_adapter(
            processor,
            reset,
            max_tool_calls=agent_config.max_tool_calls,
            image_config=image_config,
            image_preprocessor=image_preprocessor,
        )
        prompt = chat_adapter.encode_initial_prompt(reset.messages)
        effective_prompt_tokens, prompt_visual_tokens = measure_expanded_observation(
            processor,
            prompt.token_ids,
            source_images,
            min_pixels=image_config.min_pixels,
            max_pixels=image_config.max_pixels,
            image_preprocessor=image_preprocessor,
        )
        loop = AgentLoop(
            backend=backend,
            observation_encoder=chat_adapter,
            environment=environment,
            config=agent_config,
            seed_context=seed_context,
        )
        trajectory = await loop.run(
            prompt_ids=prompt.token_ids,
            source_images=source_images,
            effective_prompt_tokens=effective_prompt_tokens,
            max_model_len=max_model_len,
        )
        trajectory.effective_prompt_tokens = effective_prompt_tokens
        trajectory.prompt_visual_tokens = prompt_visual_tokens
        trajectory.initial_state_key = reset.state_key
        trajectory.env_outcome = await environment.finalize(trajectory.final_answer, trajectory.status)
        return EpisodeRun(reset=reset, chat_adapter=chat_adapter, trajectory=trajectory)
    finally:
        await environment.close()
