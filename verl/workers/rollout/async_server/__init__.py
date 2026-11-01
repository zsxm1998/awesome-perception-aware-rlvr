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
"""The server mode of agentic rollouts (worker.rollout.agent_engine_mode=server), after verl v0.9.1's colocated
("hybrid") agent loop: a vLLM server per training GPU, a router, and CPU workers that run the agent loops.

``launch_server_rollout`` starts them once the training workers exist; the trainer then calls the returned manager's
``generate_sequences`` where the batch mode calls the workers', between the workers' ``prepare_rollout_engine``
(weight sync, server awake) and ``release_rollout_engine`` (server asleep).
"""

import os
import zlib
from typing import Any


def launch_server_rollout(config: Any, tokenizer: Any, processor: Any, worker_group: Any) -> Any:
    """Start a rollout server on the GPU of each training worker, the router and the agent workers; ``config`` is
    the PPOConfig. Returns the AgentRolloutManager."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from transformers import AutoConfig

    from ...agent.coordinates import resolve_bbox_format
    from ..vllm_rollout_spmd import rollout_engine_kwargs, rollout_sampling_params
    from .agent_rollout import AgentRolloutManager, AgentRolloutWorker
    from .router import ReplicaRouter
    from .server import AsyncVLLMServer

    rollout_config = config.worker.rollout
    model_path = config.worker.actor.model.model_path
    if processor is None:
        raise ValueError("agentic rollout requires a multimodal processor")
    model_type = AutoConfig.from_pretrained(model_path, trust_remote_code=rollout_config.trust_remote_code).model_type
    if model_type == "qwen3_5":
        raise NotImplementedError(
            "worker.rollout.agent_engine_mode=server does not support Qwen3.5 yet (its vLLM model is registered in "
            "the training process only); use the batch mode"
        )

    device_infos = worker_group.rollout_device_info()
    job_id = ray.get_runtime_context().get_job_id()
    trajectories_per_batch = config.data.rollout_batch_size * rollout_config.n
    max_concurrency = rollout_config.server_actor_max_concurrency or max(1024, trajectories_per_batch + 64)
    # each server scans for its engine's ports from its own base, so that servers starting at once on a node do
    # not take the same free port; the bases differ between jobs too
    port_base = 20000 + zlib.crc32(job_id.encode()) % 10000
    server_class = ray.remote(AsyncVLLMServer)
    servers = [
        server_class.options(
            name=f"easyr1_rollout_server_{job_id}_{rank}",
            num_cpus=1,
            num_gpus=0,
            max_concurrency=max_concurrency,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            # the server asks Ray for no GPU and sees its training process's GPU instead; NCCL_CUMEM_ENABLE=0 as verl
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
                }
            },
        ).remote(rank, gpu_id, port_base + 64 * rank)
        for rank, (node_id, gpu_id) in enumerate(device_infos)
    ]
    engine_kwargs = rollout_engine_kwargs(rollout_config, model_path, tokenizer.name_or_path, processor)
    ray.get([server.start.remote(engine_kwargs, rollout_config.server_sleep_level) for server in servers])
    worker_group.attach_rollout_servers(servers, job_id)

    router = (
        ray.remote(ReplicaRouter)
        .options(num_cpus=0)
        .remote(
            len(servers),
            group_affinity=rollout_config.agent_group_affinity,
            affinity_slack=(rollout_config.max_num_seqs or 1024) // 4,
            seed=rollout_config.seed,
        )
    )
    bbox_format = resolve_bbox_format(rollout_config.agent_bbox_format, model_path)
    worker_class = ray.remote(AgentRolloutWorker)
    workers = [
        worker_class.options(num_cpus=1).remote(rollout_config, processor, servers, router, bbox_format)
        for _ in range(rollout_config.agent_num_workers or len(servers))
    ]
    (generation_meta_info, *_) = worker_group.generation_meta_info()
    return AgentRolloutManager(
        config=rollout_config,
        tokenizer=tokenizer,
        base_sampling_params=rollout_sampling_params(rollout_config, processor, tokenizer, model_path),
        generation_meta_info=generation_meta_info,
        workers=workers,
        router=router,
    )
