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
"""``logits_to_keep`` in the patched Qwen-VL forwards: 0 keeps the old full-sequence path."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from verl.models.transformers.qwen2_vl import qwen2_vl_model_forward
from verl.models.transformers.qwen3_vl import qwen3_vl_model_forward


class _StubBackbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(50, hidden_size)
        self.seen_kwargs = {}

    def forward(self, input_ids, **kwargs):
        self.seen_kwargs = kwargs
        return BaseModelOutput(last_hidden_state=self.embed(input_ids))


def _stub_model(hidden_size: int = 8, vocab_size: int = 50):
    torch.manual_seed(0)
    model = SimpleNamespace(model=_StubBackbone(hidden_size), lm_head=nn.Linear(hidden_size, vocab_size, bias=False))
    return model


@pytest.mark.parametrize("forward", [qwen3_vl_model_forward, qwen2_vl_model_forward])
def test_logits_to_keep_matches_full_logits(forward):
    model = _stub_model()
    input_ids = torch.randint(0, 50, (1, 12))
    full = forward(model, input_ids=input_ids, position_ids=None).logits
    assert full.shape == (1, 12, 50)
    # the argument is consumed by the head and never reaches the backbone
    assert "logits_to_keep" not in model.model.seen_kwargs

    default = forward(model, input_ids=input_ids, logits_to_keep=0).logits
    assert torch.equal(default, full)

    last = forward(model, input_ids=input_ids, logits_to_keep=3).logits
    assert torch.equal(last, full[:, -3:])

    keep_idx = torch.tensor([0, 4, 5, 11])
    picked = forward(model, input_ids=input_ids, logits_to_keep=keep_idx).logits
    assert picked.shape == (1, 4, 50)
    torch.testing.assert_close(picked, full[:, keep_idx], rtol=0, atol=1e-6)
    assert "logits_to_keep" not in model.model.seen_kwargs
