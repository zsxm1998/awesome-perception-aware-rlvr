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

import gc
from collections import defaultdict
from functools import partial
from typing import Callable, Union

import torch
import torch.distributed.fsdp._traversal_utils as _traversal_utils
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.optim import Optimizer
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name


def get_init_fn(model: nn.Module, device: Union[str, torch.device]) -> Callable[[nn.Module], None]:
    param_occurrence = defaultdict(int)
    for _, param in model.named_parameters(remove_duplicate=False):
        param_occurrence[param] += 1

    duplicated_params = {param for param in param_occurrence.keys() if param_occurrence[param] > 1}
    materialized_params = {}

    def init_fn(module: nn.Module):
        for name, param in module.named_parameters(recurse=False):
            if param in duplicated_params:
                module._parameters[name] = materialized_params.setdefault(
                    param, nn.Parameter(torch.empty_like(param.data, device=device), requires_grad=param.requires_grad)
                )
            else:
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param.data, device=device), requires_grad=param.requires_grad
                )

    return init_fn


def get_fsdp_wrap_policy(model: PreTrainedModel, is_lora_model=False):
    """Get FSDP wrap policy for the model.

    Args:
        module: The module to get wrap policy for
        is_lora_model: Whether to enable lambda policy for LoRA modules
    """
    transformer_cls_to_wrap = set()
    # remote-code models may list layer classes of several backbones (InternVL3: LlamaDecoderLayer and
    # Qwen2DecoderLayer); wrap those the model has
    for module in model._no_split_modules:
        transformer_cls = get_module_class_from_name(model, module)
        if transformer_cls is not None:
            transformer_cls_to_wrap.add(transformer_cls)
    if not transformer_cls_to_wrap:
        raise Exception(f"Cannot find any of {model._no_split_modules} in pretrained model.")

    policies = []

    # Add lambda policy for LoRA modules if is_lora_model is True
    if is_lora_model:

        def lambda_policy_fn(module):
            # If there are no child modules (leaf node), and there is a weight, and the weight requires gradient (usually LoRA A/B matrices), then wrap
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    # Add transformer auto wrap policy
    transformer_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap)
    policies.append(transformer_policy)

    # if there are multiple policies, use _or_policy to combine them
    if len(policies) > 0:
        auto_wrap_policy = partial(_or_policy, policies=policies)

    return auto_wrap_policy


@torch.no_grad()
def offload_fsdp_model(model: FSDP, empty_cache: bool = True):
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model offloading to CPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to("cpu", non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data
        assert id(flat_param._local_shard) != id(flat_param.data)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model(model: FSDP, empty_cache: bool = True):
    # lazy init FSDP model
    _lazy_init(model, model)
    assert model._is_root, "Only support root model loading to GPU"
    for handle in model._all_handles:
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        handle.flat_param_to("cuda", non_blocking=True)
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        gc.collect()


def local_flat_param_shards(module: FSDP) -> list[torch.Tensor]:
    """The local shards of the flat parameters of an FSDP module, in a fixed order (for saving and restoring)."""
    _lazy_init(module, module)
    return [handle.flat_param._local_shard for handle in module._all_handles]


def _paired_local_shards(target: FSDP, source: FSDP) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The local shards of the flat parameters of two FSDP modules built with the same wrapping, paired by the
    names of the parameters they hold (fail closed on any difference)."""
    _lazy_init(target, target)
    _lazy_init(source, source)
    target_handles, source_handles = target._all_handles, source._all_handles
    if len(target_handles) != len(source_handles):
        raise RuntimeError(f"FSDP modules hold {len(target_handles)} and {len(source_handles)} flat parameters.")
    pairs = []
    for target_handle, source_handle in zip(target_handles, source_handles):
        target_param, source_param = target_handle.flat_param, source_handle.flat_param
        if tuple(target_param._fqns) != tuple(source_param._fqns):
            raise RuntimeError(f"FSDP flat parameters hold different parameters: {target_param._fqns[:3]} ...")
        if target_param._local_shard.shape != source_param._local_shard.shape:
            raise RuntimeError(
                f"FSDP local shards differ in shape: {target_param._local_shard.shape} vs "
                f"{source_param._local_shard.shape} ({target_param._fqns[:3]} ...)"
            )
        pairs.append((target_param._local_shard, source_param._local_shard))
    return pairs


@torch.no_grad()
def copy_fsdp_params_(target: FSDP, source: FSDP) -> None:
    """target <- source on the local shards of the parameters, and on the (unsharded) buffers, which keep the
    source's values in the target's dtype: a bf16 actor holds bf16-rounded rotary frequencies, and a copy that
    recomputed them in fp32 would not be the same model."""
    for target_shard, source_shard in _paired_local_shards(target, source):
        target_shard.copy_(source_shard.to(device=target_shard.device, dtype=target_shard.dtype))

    target_buffers, source_buffers = dict(target.named_buffers()), dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise RuntimeError("FSDP modules hold different buffers.")
    for name, buffer in target_buffers.items():
        buffer.copy_(source_buffers[name].to(device=buffer.device, dtype=buffer.dtype))


@torch.no_grad()
def ema_update_fsdp_params_(ema: FSDP, source: FSDP, rate: float) -> None:
    """ema <- (1 - rate) * ema + rate * source on the local shards, in the precision of the EMA (parameters only,
    not buffers). The two modules must be built with the same wrapping and sharding."""
    for ema_shard, source_shard in _paired_local_shards(ema, source):
        ema_shard.mul_(1.0 - rate).add_(source_shard.to(device=ema_shard.device, dtype=ema_shard.dtype), alpha=rate)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True):
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cuda", non_blocking=True)

    if empty_cache:
        gc.collect()


@torch.no_grad()
def offload_fsdp_submodule(module: FSDP, empty_cache: bool = True):
    for handle in _traversal_utils._get_fsdp_handles(module):
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )
        handle.flat_param_to("cpu", non_blocking=True)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_submodule(module: FSDP, empty_cache: bool = True):
    for handle in _traversal_utils._get_fsdp_handles(module):
        if handle._offload_params:
            continue

        flat_param = handle.flat_param
        handle.flat_param_to("cuda", non_blocking=True)
        flat_param._local_shard = flat_param.data

    if empty_cache:
        gc.collect()
