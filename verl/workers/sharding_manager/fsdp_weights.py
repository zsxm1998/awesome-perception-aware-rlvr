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
"""The FSDP model's weights as the rollout engine loads them: HF checkpoint names, full tensors one at a time."""

import re
from typing import Iterable, Union

import torch
from torch.distributed._tensor import DTensor
from transformers import PreTrainedModel


def rename_weight_keys(
    actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel
) -> dict[str, Union[torch.Tensor, DTensor]]:
    """The checkpoint names of the state dict's keys: https://github.com/huggingface/transformers/pull/38385."""
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return actor_weights

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in actor_weights.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break

        original_weights[key] = value

    return original_weights


def full_weight_iterator(actor_weights: dict[str, Union[torch.Tensor, DTensor]]) -> Iterable[tuple[str, torch.Tensor]]:
    """Each weight as a full tensor, gathered when it is reached (a collective: every rank iterates in step)."""
    for name, tensor in actor_weights.items():
        yield name, tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
