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
"""The training process's side of a rollout server on its GPU (worker.rollout.agent_engine_mode=server).

Each step follows verl v0.9.1's synchronous colocated update (``ActorRolloutRefWorker.update_weights`` with the
"naive" backend, ``verl/workers/engine_workers.py``): wake the server's weights, stream the FSDP model's full
weights to it over CUDA IPC in buckets, drop its caches, release the training memory, wait for every rank (verl
#7851: a peer's KV cache must not come back while another rank still holds its weights), then wake the KV cache.
After the rollout the server sleeps and training gets the memory back.
"""

from typing import Any, Optional

import ray
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from ...utils.model_utils import print_gpu_memory_usage
from ..rollout.async_server.weight_transfer import BucketedWeightSender
from .base import BaseShardingManager
from .fsdp_weights import full_weight_iterator, rename_weight_keys


class FSDPAsyncServerShardingManager(BaseShardingManager):
    def __init__(self, module: Any, use_param_offload: bool, bucket_size_mb: int):
        self.module = module
        self.use_param_offload = use_param_offload
        self.bucket_size_mb = bucket_size_mb
        self.server: Optional[Any] = None
        self.zmq_handle: Optional[str] = None
        self.loaded = False
        self.synced_once = False
        # the server's memory is not this process's, so the actor's memory metrics need no correction
        self.freed_bytes = 0

    def attach(self, server: Any, zmq_handle: str) -> None:
        self.server = server
        self.zmq_handle = zmq_handle

    def load_vllm_and_sync_weights(self) -> None:
        if self.server is None:
            raise RuntimeError("the rollout server of this rank is not attached (attach_rollout_servers)")
        if self.loaded:
            raise RuntimeError("the rollout server is already awake")
        self.loaded = True
        torch.cuda.empty_cache()
        ray.get(self.server.wake_up.remote(["weights"]))
        received = self.server.update_weights.remote(self.zmq_handle)
        if self.use_param_offload:
            load_fsdp_model(self.module)
        actor_weights = rename_weight_keys(get_model_state_dict(self.module), self.module._fsdp_wrapped_module)

        def check_receiver() -> None:  # the server's error, when its receiving call failed
            ready, _ = ray.wait([received], timeout=0)
            if ready:
                ray.get(ready[0])

        try:
            BucketedWeightSender(self.zmq_handle, bucket_size_mb=self.bucket_size_mb).send_weights(
                full_weight_iterator(actor_weights), check_receiver=check_receiver
            )
        except BaseException:
            # the receiving side's own error says what went wrong
            ready, _ = ray.wait([received], timeout=60)
            if ready:
                ray.get(ready[0])
            raise
        result = ray.get(received)
        del actor_weights
        if not self.synced_once:
            if result["missing"] is None:
                raise RuntimeError(
                    "the rollout model's load_weights does not report the parameters it loaded, so the weight sync "
                    "cannot check that none was left at the engine's dummy initialization; the server mode needs a "
                    "model whose vLLM loader reports them"
                )
            if result["missing"]:
                raise RuntimeError(
                    f"the weight sync left {len(result['missing'])} parameters of the rollout model at the engine's "
                    f"dummy initialization, e.g. {result['missing'][:5]}"
                )
            self.synced_once = True
        ray.get(self.server.reset_caches.remote())
        if self.use_param_offload:
            offload_fsdp_model(self.module)
        torch.cuda.empty_cache()
        dist.barrier()
        ray.get(self.server.wake_up.remote(["kv_cache"]))
        print_gpu_memory_usage("After the rollout server woke up")

    def offload_vllm(self) -> None:
        if not self.loaded:
            raise RuntimeError("the rollout server is not awake")
        self.loaded = False
        ray.get(self.server.sleep.remote())
        self.module.train()
        torch.cuda.empty_cache()
