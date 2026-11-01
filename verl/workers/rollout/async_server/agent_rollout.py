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
"""Agentic rollouts in the server mode: CPU workers run the agent loops against the rollout servers.

As verl v0.9.1's ``AgentLoopManager`` / ``AgentLoopWorker`` (``verl/experimental/agent_loop/agent_loop.py``): the
driver splits a batch among the workers, each worker runs all its trajectories as coroutines, and the driver puts the
results back in the batch's order. Here a worker gets whole groups (the rollouts of a row share their environment
context, image objects and multimodal uuids, as in the batch mode), a trajectory that materialization rejects is
generated again at once in its own coroutine, and the driver builds the training batch with the batch mode's code
(``rollout_batch.agent_rollout_output``).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from vllm import SamplingParams

from ....protocol import DataProto
from ...agent.backends import VLLMAgentImageCache
from ...agent.backends.async_server import AsyncVLLMServerBackend
from ...agent.environments import EnvironmentContext, environment_factory
from ...agent.episode import run_environment_episode
from ...agent.protocol import AgentSeedContext
from ...agent.rollout_batch import (
    AgentRolloutTask,
    agent_environment_kwargs,
    agent_image_config,
    agent_loop_config,
    agent_max_model_len,
    agent_rollout_output,
    agent_rollout_rows,
    agent_rollout_tasks,
    retry_base_seed,
)
from ...agent.trajectory import AgentTrajectoryMaterializer, AgentTrajectoryRejected
from .tasks import finish, gather_or_cancel, run_all


class AgentRolloutWorker:
    """Runs agent trajectories on the rollout servers (a CPU Ray actor). One environment context and one image
    cache per call: the resources a batch-mode rank holds for its share of a batch."""

    def __init__(self, config: Any, processor: Any, servers: Sequence[Any], router: Any, bbox_format: str):
        self.config = config
        self.processor = processor
        self.servers = list(servers)
        self.router = router
        self.bbox_format = bbox_format
        image_token_id = getattr(processor, "image_token_id", None)
        self.forbidden_token_ids = (int(image_token_id),) if isinstance(image_token_id, int) else ()

    async def run(
        self, tasks: Sequence[AgentRolloutTask], sampling_params: SamplingParams, meta_info: dict[str, Any]
    ) -> list[tuple[Any, Any, int]]:
        """Each task's trajectory, its materialization and how many times it was generated again, in task order."""
        config = self.config
        image_config = agent_image_config(config, meta_info)
        loop_config = agent_loop_config(config)
        max_model_len = agent_max_model_len(config)
        materializer = AgentTrajectoryMaterializer(
            self.processor, max_prompt_tokens=config.prompt_length, max_model_len=max_model_len
        )
        image_cache = VLLMAgentImageCache(min_pixels=image_config.min_pixels, max_pixels=image_config.max_pixels)
        context = EnvironmentContext(
            processor=self.processor,
            image_preprocessor=image_cache.prepare,
            max_tool_calls=config.agent_max_tool_calls,
            max_tool_calls_per_turn=config.agent_max_tool_calls_per_turn,
        )
        make_environment = environment_factory(
            config.resolved_agent_environment, context, agent_environment_kwargs(config, self.bbox_format)
        )
        base_seed = int(sampling_params.seed)

        async def run_task(task: AgentRolloutTask):
            for retry_index in range(config.agent_max_trajectory_retries + 1):
                episode = await run_environment_episode(
                    make_environment(),
                    task.env_config,
                    self.processor,
                    env_seed=task.env_seed,
                    config=loop_config,
                    image_config=image_config,
                    image_preprocessor=image_cache.prepare,
                    make_backend=lambda retry_index=retry_index: AsyncVLLMServerBackend(
                        servers=self.servers,
                        router=self.router,
                        sticky_key=(task.uid, task.rollout_index, retry_index),
                        group_key=task.uid,
                        sampling_params=sampling_params,
                        tokenizer=self.processor.tokenizer,
                        image_cache=image_cache,
                        forbidden_token_ids=self.forbidden_token_ids,
                    ),
                    # a retry keeps the group's identity but moves the generation seeds, as in the batch mode
                    seed_context=AgentSeedContext(
                        base_seed=retry_base_seed(base_seed, retry_index),
                        sample_index=task.sample_index,
                        rollout_index=task.rollout_index,
                    ),
                    max_model_len=max_model_len,
                )
                try:
                    return episode.trajectory, materializer.materialize(episode.trajectory), retry_index
                except AgentTrajectoryRejected as error:
                    if not error.recoverable or retry_index >= config.agent_max_trajectory_retries:
                        raise

        # one trajectory failing, or the manager cancelling this call, stops the others; every trajectory has ended
        # (its server requests too) and the environments are closed before the error or cancellation goes on
        try:
            results = await run_all(run_task(task) for task in tasks)
        except BaseException:
            await finish(context.aclose(), quiet=True)
            raise
        await finish(context.aclose())
        return results


def split_by_group(tasks: Sequence[AgentRolloutTask], parts: int) -> list[list[AgentRolloutTask]]:
    """Contiguous runs of whole groups, as even in size as groups allow (the rollouts of a row stay together)."""
    groups: list[list[AgentRolloutTask]] = []
    for task in tasks:
        if groups and groups[-1][0].sample_position == task.sample_position:
            groups[-1].append(task)
        else:
            groups.append([task])
    parts = max(1, min(parts, len(groups)))
    chunks, start = [], 0
    for part in range(parts):
        end = start + (len(groups) - start) // (parts - part)
        chunks.append([task for group in groups[start:end] for task in group])
        start = end
    return chunks


class AgentRolloutManager:
    """The driver's side of the server mode: ``generate_sequences`` takes the batch the batch mode's workers take
    and returns the batch they return."""

    def __init__(
        self,
        *,
        config: Any,
        tokenizer: Any,
        base_sampling_params: SamplingParams,
        generation_meta_info: dict[str, Any],
        workers: Sequence[Any],
        router: Any,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.base_sampling_params = base_sampling_params
        self.generation_meta_info = dict(generation_meta_info)
        self.workers = list(workers)
        self.router = router

    def sampling_params(self, meta_info: dict[str, Any]) -> tuple[SamplingParams, int]:
        """The per-trajectory sampling parameters of a batch (``n`` is the rollouts per row) and ``n``."""
        from ..vllm_rollout_spmd import sampling_params_with

        sampling_params = sampling_params_with(self.base_sampling_params, meta_info).clone()
        rollout_n = int(sampling_params.n)
        if rollout_n <= 0:
            raise ValueError("agentic rollout requires sampling n > 0")
        sampling_params.n = 1
        if self.config.resolved_record_rollout_log_probs:
            sampling_params.logprobs = 0  # the sampled token's only
        if not isinstance(sampling_params.seed, int):
            raise ValueError("agentic rollout requires an integer sampling seed")
        return sampling_params, rollout_n

    def generate_sequences(self, prompts: DataProto, *, ray_api: Optional[Any] = None) -> DataProto:
        """``ray_api`` is the ``ray`` module (default) or a stand-in with ``get``, ``wait`` and ``cancel``."""
        if ray_api is None:
            import ray as ray_api
        # the batch mode's workers get a copy of the batch and add the generation meta info to it
        prompts = DataProto(
            batch=prompts.batch,
            non_tensor_batch=dict(prompts.non_tensor_batch),
            meta_info={**prompts.meta_info, **self.generation_meta_info},
        )
        rows = agent_rollout_rows(prompts, self.config)
        sampling_params, rollout_n = self.sampling_params(prompts.meta_info)
        tasks = agent_rollout_tasks(rows, rollout_n=rollout_n, base_seed=int(sampling_params.seed))
        ray_api.get(self.router.reset.remote())
        chunks = split_by_group(tasks, len(self.workers))
        # when a worker fails, the others are cancelled and close their environments before the error is raised
        results = gather_or_cancel(
            [
                worker.run.remote(chunk, sampling_params, dict(prompts.meta_info))
                for worker, chunk in zip(self.workers, chunks)
            ],
            ray_api,
        )
        trajectories, materialized, retry_counts = [], [], []
        for chunk_results in results:
            for trajectory, sample, retries in chunk_results:
                trajectories.append(trajectory)
                materialized.append(sample)
                retry_counts.append(retries)
        if len(trajectories) != len(tasks):
            raise RuntimeError(f"the rollout workers returned {len(trajectories)} trajectories for {len(tasks)}")
        return agent_rollout_output(
            tasks,
            trajectories,
            materialized,
            retry_counts,
            config=self.config,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
            meta_info=prompts.meta_info,
        )
