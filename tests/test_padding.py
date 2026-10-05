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

import importlib.util

import pytest
import torch

from verl.utils import padding


MASK = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 1]])


def test_torch_unpad_and_pad_round_trip():
    hidden = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
    rows, indices, cu_seqlens, max_seqlen = padding._torch_unpad_input(hidden, MASK)
    assert indices.tolist() == [1, 2, 3, 4, 5, 11]
    assert cu_seqlens.tolist() == [0, 3, 5, 6] and max_seqlen == 3
    assert torch.equal(rows, hidden.reshape(-1, 2)[indices])
    assert torch.equal(padding._torch_index_first_axis(hidden.reshape(-1, 2), indices), rows)
    padded = padding._torch_pad_input(rows, indices, batch=3, seqlen=4)
    assert torch.equal(padded, hidden * MASK[..., None])


@pytest.mark.skipif(importlib.util.find_spec("flash_attn") is None, reason="flash-attn not installed")
def test_torch_fallbacks_match_flash_attn():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    hidden = torch.randn(3, 4, 5)
    rows, indices, cu_seqlens, max_seqlen, *_ = unpad_input(hidden, MASK)
    torch_rows, torch_indices, torch_cu_seqlens, torch_max_seqlen = padding._torch_unpad_input(hidden, MASK)
    assert torch.equal(rows, torch_rows) and torch.equal(indices, torch_indices)
    assert torch.equal(cu_seqlens, torch_cu_seqlens) and max_seqlen == torch_max_seqlen
    assert torch.equal(pad_input(rows, indices, 3, 4), padding._torch_pad_input(rows, indices, 3, 4))
    flat = hidden.reshape(-1, 5)
    assert torch.equal(index_first_axis(flat, indices), padding._torch_index_first_axis(flat, indices))
