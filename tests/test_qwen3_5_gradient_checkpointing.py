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

import pytest
import torch
from transformers.modeling_layers import GradientCheckpointingLayer

from verl.models.transformers import qwen3_5 as qwen3_5_module
from verl.models.transformers.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
    torch_chunk_gated_delta_rule,
)


def _tiny_text_config(num_hidden_layers: int = 2) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"] * num_hidden_layers,
        partial_rotary_factor=0.25,
        rope_scaling={"mrope_section": [1, 0, 0]},
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return config


def _tiny_linear_text_config(num_hidden_layers: int = 2) -> Qwen3_5TextConfig:
    """Linear-attention layers only, which run on the CPU with the torch implementation."""
    config = _tiny_text_config(num_hidden_layers).to_dict()
    config.update(
        layer_types=["linear_attention"] * num_hidden_layers,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    config.pop("model_type", None)
    config = Qwen3_5TextConfig(**config)
    config._attn_implementation = "eager"
    return config


def _tiny_model_config(num_hidden_layers: int = 2, text_config: Qwen3_5TextConfig | None = None) -> Qwen3_5Config:
    text_config = _tiny_text_config(num_hidden_layers) if text_config is None else text_config
    config = Qwen3_5Config(
        text_config=text_config.to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return config


def _use_torch_linear_attention(model: torch.nn.Module) -> None:
    # the fast path (flash-linear-attention, causal-conv1d) needs a GPU
    for module in model.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.causal_conv1d_fn = None
            module.chunk_gated_delta_rule = torch_chunk_gated_delta_rule


def _packed_position_ids(lengths: list[int]) -> torch.Tensor:
    positions = torch.cat([torch.arange(length) for length in lengths])
    return positions.view(1, 1, -1).expand(4, 1, -1)


def _checkpoint_counter():
    calls = {"count": 0}

    def checkpoint_func(function, *args, **kwargs):
        calls["count"] += 1
        return function(*args, **kwargs)

    return calls, checkpoint_func


def test_qwen3_5_decoder_layers_enable_gradient_checkpointing():
    model = Qwen3_5TextModel(_tiny_text_config())

    assert all(isinstance(layer, Qwen3_5DecoderLayer) for layer in model.layers)
    assert all(hasattr(layer, "gradient_checkpointing") for layer in model.layers)
    assert not model.is_gradient_checkpointing

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    assert model.is_gradient_checkpointing
    assert all(layer.gradient_checkpointing for layer in model.layers)


def test_qwen3_5_text_forward_uses_checkpointing_in_training():
    model = Qwen3_5TextModel(_tiny_text_config())
    model.train()
    calls, checkpoint_func = _checkpoint_counter()
    model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=checkpoint_func)

    outputs = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)

    assert outputs.last_hidden_state.shape == (1, 3, model.config.hidden_size)
    assert calls["count"] == model.config.num_hidden_layers


def test_qwen3_5_text_backward_with_real_gradient_checkpointing():
    model = Qwen3_5TextModel(_tiny_text_config(num_hidden_layers=1))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    outputs = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    outputs.last_hidden_state.float().sum().backward()

    assert model.embed_tokens.weight.grad is not None


def test_qwen3_5_checkpointing_is_skipped_when_disabled_or_eval():
    input_ids = torch.tensor([[1, 2, 3]])

    disabled_model = Qwen3_5TextModel(_tiny_text_config())
    disabled_model.train()
    disabled_calls, disabled_checkpoint_func = _checkpoint_counter()
    disabled_model._set_gradient_checkpointing(enable=False, gradient_checkpointing_func=disabled_checkpoint_func)
    disabled_model(input_ids=input_ids, use_cache=False)
    assert disabled_calls["count"] == 0

    eval_model = Qwen3_5TextModel(_tiny_text_config())
    eval_model.eval()
    eval_calls, eval_checkpoint_func = _checkpoint_counter()
    eval_model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=eval_checkpoint_func)
    eval_model(input_ids=input_ids, use_cache=False)
    assert eval_calls["count"] == 0


def test_qwen3_5_packed_segments_restart_at_text_position_zero():
    position_ids = torch.tensor([3, 4, 0, 1, 2, 0]).view(1, 1, 6).expand(4, 1, 6)

    assert Qwen3_5Model._packed_segments(position_ids) == ((0, 2), (2, 5), (5, 6))


def test_qwen3_5_linear_attention_restarts_at_every_packed_sample():
    torch.manual_seed(0)
    config = _tiny_linear_text_config()
    layer = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    _use_torch_linear_attention(layer)
    hidden_states = torch.randn(1, 7, config.hidden_size)
    segments = ((0, 3), (3, 7))

    packed = layer(hidden_states, segments=segments)
    separate = torch.cat([layer(hidden_states[:, start:end]) for start, end in segments], dim=1)

    torch.testing.assert_close(packed, separate)
    # without the restart, the second sample would continue the recurrence of the first
    assert not torch.allclose(packed[:, 3:], layer(hidden_states)[:, 3:])


def test_qwen3_5_packed_forward_calls_each_layer_once():
    torch.manual_seed(0)
    model = Qwen3_5Model(_tiny_model_config(text_config=_tiny_linear_text_config()))
    _use_torch_linear_attention(model)
    # packed inputs need flash attention for the full-attention layers; this model has none
    model.language_model.config._attn_implementation = "flash_attention_2"
    model.train()
    calls, checkpoint_func = _checkpoint_counter()
    model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=checkpoint_func)
    lengths = [3, 2, 4]
    input_ids = torch.randint(0, 32, (1, sum(lengths)))
    position_ids = _packed_position_ids(lengths)

    packed = model(input_ids=input_ids, position_ids=position_ids).last_hidden_state

    # one call per layer (the decoder layers and the vision blocks of the dummy image) whatever the number of
    # samples, so that every FSDP rank issues the same collectives
    assert calls["count"] == sum(isinstance(module, GradientCheckpointingLayer) for module in model.modules())
    separate, start = [], 0
    for length in lengths:
        end = start + length
        separate.append(
            model(input_ids=input_ids[:, start:end], position_ids=position_ids[:, :, start:end]).last_hidden_state
        )
        start = end
    torch.testing.assert_close(packed, torch.cat(separate, dim=1))


def test_qwen3_5_packed_inputs_require_flash_attention():
    model = Qwen3_5Model(_tiny_model_config())

    with pytest.raises(ValueError, match="flash_attention_2"):
        model(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), position_ids=_packed_position_ids([3, 2]))


def test_qwen3_5_rejects_ulysses_sequence_parallelism(monkeypatch):
    model = Qwen3_5Model(_tiny_model_config(num_hidden_layers=1))
    monkeypatch.setattr(qwen3_5_module, "get_ulysses_sequence_parallel_world_size", lambda: 2)

    with pytest.raises(ValueError, match="Ulysses"):
        model(input_ids=torch.tensor([[1, 2, 3]]))


def test_qwen3_5_empty_image_feature_indices_do_not_require_local_image_tokens():
    model = Qwen3_5Model(_tiny_model_config(num_hidden_layers=1))
    input_ids = torch.tensor([[1, 2, 3]])
    pixel_values = torch.zeros((4, 12), dtype=torch.float32)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    image_token_feature_indices = torch.full_like(input_ids, -1)

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        image_token_feature_indices=image_token_feature_indices,
    )

    assert outputs.last_hidden_state.shape == (1, 3, model.config.text_config.hidden_size)
