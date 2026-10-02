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

import torch

from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.dp_actor import _pad_and_slice_mm_feature_indices


def test_mm_feature_indices_padding_does_not_select_feature_zero(monkeypatch):
    sp_size = 4
    sp_rank = 3

    def fake_slice_input_tensor(x, dim, padding):
        assert dim == -1
        assert not padding
        part_size = x.size(-1) // sp_size
        return x[..., sp_rank * part_size : (sp_rank + 1) * part_size].contiguous()

    monkeypatch.setattr(dp_actor_module, "slice_input_tensor", fake_slice_input_tensor)

    feature_indices = torch.tensor([[0, 1, 2]])
    sliced = _pad_and_slice_mm_feature_indices(feature_indices, pad_size=1)

    assert sliced.tolist() == [[-1]]


def test_mm_feature_indices_slice_keeps_real_feature_indices(monkeypatch):
    sp_size = 4
    sp_rank = 0

    def fake_slice_input_tensor(x, dim, padding):
        assert dim == -1
        assert not padding
        part_size = x.size(-1) // sp_size
        return x[..., sp_rank * part_size : (sp_rank + 1) * part_size].contiguous()

    monkeypatch.setattr(dp_actor_module, "slice_input_tensor", fake_slice_input_tensor)

    feature_indices = torch.tensor([[0, 1, 2, -1]])
    sliced = _pad_and_slice_mm_feature_indices(feature_indices, pad_size=0)

    assert sliced.tolist() == [[0]]
