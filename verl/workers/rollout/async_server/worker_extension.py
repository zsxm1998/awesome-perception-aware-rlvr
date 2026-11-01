# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""The methods the vLLM worker process of a rollout server gets (``worker_extension_cls``), called through
``AsyncLLM.collective_rpc``.

Ported from verl v0.9.1 (``vLLMColocateWorkerExtension.update_weights_from_ipc``) without its LoRA, quantization
and MTP branches, which the server mode does not support.
"""

import torch


class EasyR1ColocateWorkerExtension:
    """Mixed into vLLM's GPU worker: ``self.model_runner`` and ``self.local_rank`` are the worker's."""

    def update_weights_from_ipc(self, zmq_handle: str, use_shm: bool = False) -> dict:
        """Receive the training process's weights over ``zmq_handle`` and load them into the model.

        Returns the names of the model's parameters that no bucket loaded (``missing``; a parameter left out keeps
        the dummy weights the engine started with; None when the model's loader does not report what it loaded, so
        that nothing can be checked) and how many tensors arrived (``received``)."""
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        from .weight_transfer import BucketedWeightReceiver

        device = getattr(self, "device", None) or torch.device(f"cuda:{self.local_rank}")
        model = self.model_runner.model
        named_buffers = dict(model.named_buffers())
        loaded: set[str] = set()
        received = 0
        reports_names = True

        def on_bucket_received(weights: list[tuple[str, torch.Tensor]], is_last: bool) -> None:
            nonlocal received, reports_names
            received += len(weights)
            params = [(name, tensor) for name, tensor in weights if name not in named_buffers]
            buffers = [(name, tensor) for name, tensor in weights if name in named_buffers]
            if params:
                names = model.load_weights(params)
                if names is None:
                    reports_names = False
                else:
                    loaded.update(names)
            for name, tensor in buffers:
                target = named_buffers[name]
                if target.shape != tensor.shape:
                    raise ValueError(f"buffer {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded.add(name)

        BucketedWeightReceiver(zmq_handle, device=device, use_shm=use_shm).receive_weights(on_bucket_received)
        # some post-load transforms are not idempotent: once per sync, after every bucket (as verl)
        process_weights_after_loading(model, self.model_runner.model_config, device)
        missing = sorted(name for name, _ in model.named_parameters() if name not in loaded) if reports_names else None
        return {"missing": missing, "received": received}
