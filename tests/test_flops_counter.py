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

from types import SimpleNamespace

import pytest

from verl.utils import flops_counter as flops_counter_module
from verl.utils.flops_counter import FlopsCounter


def _qwen3_5_config():
    text_config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_key_heads=5,
        linear_num_value_heads=7,
        linear_conv_kernel_dim=3,
    )
    return SimpleNamespace(model_type="qwen3_5", text_config=text_config)


def _internvl_chat_config():
    llm_config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    vision_config = SimpleNamespace(
        image_size=448,
        patch_size=14,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
    )
    return SimpleNamespace(
        model_type="internvl_chat",
        llm_config=llm_config,
        vision_config=vision_config,
        downsample_ratio=0.5,
    )


def test_qwen3_5_flops_counter_is_supported(capsys):
    counter = FlopsCounter(_qwen3_5_config())

    assert counter._estimate_flops == counter._estimate_qwen3_5_flops
    assert "MFU will always be zero" not in capsys.readouterr().out


def test_qwen3_5_flops_counter_estimates_mixed_attention(monkeypatch):
    monkeypatch.setattr(flops_counter_module, "get_device_flops", lambda: 100.0)
    counter = FlopsCounter(_qwen3_5_config())

    estimated_flops, promised_flops = counter.estimate_flops(batch_seqlens=[3, 5], delta_time=2.0)

    assert estimated_flops == pytest.approx(4.73976e-7)
    assert promised_flops == 100.0


def test_internvl_chat_flops_counter_is_supported(capsys):
    counter = FlopsCounter(_internvl_chat_config())

    assert counter._estimate_flops == counter._estimate_internvl_chat_flops
    assert "MFU will always be zero" not in capsys.readouterr().out


def test_internvl_chat_flops_counter_estimates_llm_flops(monkeypatch):
    monkeypatch.setattr(flops_counter_module, "get_device_flops", lambda: 100.0)
    counter = FlopsCounter(_internvl_chat_config())

    estimated_flops, promised_flops = counter.estimate_flops(batch_seqlens=[3, 5], delta_time=2.0)

    assert estimated_flops == pytest.approx(2.15424e-7)
    assert promised_flops == 100.0


def test_unknown_flops_counter_still_warns(capsys):
    config = SimpleNamespace(model_type="unknown")

    counter = FlopsCounter(config)

    assert counter._estimate_flops == counter._estimate_unknown_flops
    assert "MFU will always be zero" in capsys.readouterr().out
