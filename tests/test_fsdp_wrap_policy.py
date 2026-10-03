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
"""FSDP wrap policy for models whose _no_split_modules name layers of several backbones."""

import pytest
from torch import nn

from verl.utils.fsdp_utils import get_fsdp_wrap_policy


class Qwen2DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _InternVLLike(nn.Module):
    _no_split_modules = ["InternVisionModel", "LlamaDecoderLayer", "Qwen2DecoderLayer"]  # InternVL3 remote code

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Qwen2DecoderLayer(), Qwen2DecoderLayer()])


def test_wraps_the_layer_classes_the_model_has():
    policy = get_fsdp_wrap_policy(_InternVLLike())
    (transformer_policy,) = policy.keywords["policies"]
    assert transformer_policy.keywords["transformer_layer_cls"] == {Qwen2DecoderLayer}


def test_raises_when_no_listed_class_is_present():
    class _Model(nn.Module):
        _no_split_modules = ["LlamaDecoderLayer"]

    with pytest.raises(Exception, match="Cannot find any of"):
        get_fsdp_wrap_policy(_Model())
