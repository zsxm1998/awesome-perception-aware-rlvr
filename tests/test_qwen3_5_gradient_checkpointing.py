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

from verl.models.transformers.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
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


def _tiny_model_config(num_hidden_layers: int = 2) -> Qwen3_5Config:
    config = Qwen3_5Config(
        text_config=_tiny_text_config(num_hidden_layers).to_dict(),
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


def test_qwen3_5_packed_segments_checkpoint_each_segment_layer():
    model = Qwen3_5Model(_tiny_model_config())
    model.train()
    calls, checkpoint_func = _checkpoint_counter()
    model._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=checkpoint_func)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    inputs_embeds = model.get_input_embeddings()(input_ids)
    position_ids = torch.tensor([[[0, 1, 2, 0, 1]], [[0, 1, 2, 0, 1]], [[0, 1, 2, 0, 1]]])

    outputs = model._forward_packed_segments(inputs_embeds=inputs_embeds, position_ids=position_ids)

    assert outputs.last_hidden_state.shape == (1, 5, model.config.text_config.hidden_size)
    assert calls["count"] == 2 * model.config.text_config.num_hidden_layers


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
