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
"""Padding-free layout helpers: flash-attn's ``bert_padding`` functions, or the same in plain torch where
flash-attn is not installed (CPU environments, such as the unit tests)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def _torch_index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Rows ``indices`` of the first axis."""
    return tensor[indices]


def _torch_unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor, unused_mask=None):
    """``(rows of the attended positions, their flat indices, cu_seqlens, max_seqlen)`` of a padded batch, as
    flash-attn's ``unpad_input`` (its later versions also return the per-sequence counts, which callers ignore)."""
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return (
        _torch_index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen,
    )


def _torch_pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """The padded ``(batch, seqlen, ...)`` layout of rows at flat ``indices``, zeros elsewhere."""
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    index_first_axis, pad_input, unpad_input = _torch_index_first_axis, _torch_pad_input, _torch_unpad_input

__all__ = ["index_first_axis", "pad_input", "unpad_input"]
