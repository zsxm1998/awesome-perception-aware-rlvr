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
"""A rollout server: vLLM's AsyncLLM on the GPU of one training process (worker.rollout.agent_engine_mode=server).

As verl v0.9.1's ``vLLMHttpServer`` (``verl/workers/rollout/vllm_rollout/vllm_async_server.py``), without its HTTP
front end: a Ray actor that asks Ray for no GPU, sees the GPU of its training process through CUDA_VISIBLE_DEVICES
and holds an AsyncLLM, whose engine core and GPU worker run in their own processes. Requests are token in, token
out; every turn is scheduled by vLLM as it arrives (continuous batching). Between rollouts the engine sleeps and
the training process gets the GPU memory; before a rollout it wakes up and receives the new weights over CUDA IPC
(``EasyR1ColocateWorkerExtension``). Training is synchronous: no request is in flight while the weights change.
"""

import os
from typing import Any, Optional

from vllm import SamplingParams


def extension_class_path() -> str:
    from . import worker_extension

    return f"{worker_extension.__name__}.EasyR1ColocateWorkerExtension"


class AsyncVLLMServer:
    """One replica (one GPU, tensor_parallel_size=1). Create it as a Ray actor with ``num_gpus=0`` on the node of
    its training process, ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` and that process's GPU id, then call
    ``start``."""

    def __init__(self, replica_rank: int, cuda_visible_devices: str, port_base: int):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        # vLLM's engine picks the port of its worker's process group by scanning from VLLM_PORT (else any free
        # port); servers starting together on one node could otherwise pick the same free port
        os.environ["VLLM_PORT"] = str(port_base)
        self.replica_rank = replica_rank
        self.engine = None
        self.sleep_level = 2
        self.asleep = False
        self.inflight = 0
        self.requests = 0

    async def start(self, engine_kwargs: dict[str, Any], sleep_level: int) -> None:
        """Start the engine on dummy weights and put it to sleep; the first weight sync comes before the first
        rollout (as verl's ``sleep_replicas`` after launch)."""
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            **{**engine_kwargs, "distributed_executor_backend": "mp", "worker_extension_cls": extension_class_path()}
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)
        self.sleep_level = sleep_level
        await self.sleep()

    async def generate(
        self, prompt: dict[str, Any], sampling_params: SamplingParams, request_id: str
    ) -> dict[str, Any]:
        """Generate one turn. Returns the token ids, the text vLLM decoded (None without detokenization), the
        finish and stop reasons, the sampled tokens' log-probabilities when ``sampling_params.logprobs`` asks for
        them, and the prompt tokens served from the prefix cache."""
        if self.engine is None or self.asleep:
            raise RuntimeError(f"rollout server {self.replica_rank} got a request while asleep")
        self.inflight += 1
        self.requests += 1
        try:
            final = None
            async for output in self.engine.generate(prompt, sampling_params, request_id):
                final = output
        finally:
            self.inflight -= 1
        if final is None or len(final.outputs) != 1:
            raise RuntimeError(f"rollout server {self.replica_rank} returned no output for request {request_id}")
        completion = final.outputs[0]
        log_probs = None
        if sampling_params.logprobs is not None:
            from ...agent.backends.vllm import sampled_log_probs

            log_probs = sampled_log_probs(completion.token_ids, completion.logprobs)
        return {
            "token_ids": list(completion.token_ids),
            "text": completion.text if sampling_params.detokenize else None,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "log_probs": log_probs,
            "num_cached_tokens": final.num_cached_tokens,
        }

    async def sleep(self) -> None:
        """Release the GPU memory for training. The multimodal processor cache is emptied on both sides first:
        sleep empties the engine's side only, after which the front end would send just the hash of an image the
        engine no longer holds ("Expected a cached item for mm_hash")."""
        if self.inflight:
            raise RuntimeError(f"rollout server {self.replica_rank} has {self.inflight} requests in flight")
        await self.engine.reset_mm_cache()
        await self.engine.sleep(level=self.sleep_level)
        await self.engine.reset_encoder_cache()
        self.asleep = True

    async def wake_up(self, tags: list[str]) -> None:
        await self.engine.wake_up(tags=tags)
        if "kv_cache" in tags:
            self.asleep = False

    async def reset_caches(self) -> None:
        """Drop what was computed with the previous weights: prefix (KV) cache, multimodal and encoder caches."""
        await self.engine.reset_prefix_cache()
        await self.engine.reset_mm_cache()
        await self.engine.reset_encoder_cache()

    async def update_weights(self, zmq_handle: str, use_shm: bool = False) -> dict[str, Any]:
        """Receive the training process's weights (the call returns once the sender has sent them all)."""
        (result,) = await self.engine.collective_rpc(
            "update_weights_from_ipc", kwargs={"zmq_handle": zmq_handle, "use_shm": use_shm}
        )
        return result

    def stats(self) -> dict[str, Optional[int]]:
        return {"inflight": self.inflight, "requests": self.requests}
