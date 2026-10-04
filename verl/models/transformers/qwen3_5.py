# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
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

from __future__ import annotations

import itertools
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDynamicCache,
    Qwen3NextMLP,
    Qwen3NextRMSNorm,
    Qwen3NextRMSNormGated,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    chunk_gated_delta_rule,
    create_causal_mask,
    fused_recurrent_gated_delta_rule,
    is_fast_path_available,
    logger,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionConfig,
    Qwen3VLVisionModel,
)

from ...utils.ulysses import get_ulysses_sequence_parallel_world_size


try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - compatibility with older transformers.
    AutoModelForImageTextToText = None


class Qwen3_5TextConfig(Qwen3NextConfig):
    model_type = "qwen3_5_text"

    def __init__(self, rope_parameters: Optional[dict] = None, **kwargs):
        kwargs.setdefault("vocab_size", 248320)
        kwargs.setdefault("hidden_size", 4096)
        kwargs.setdefault("intermediate_size", 12288)
        kwargs.setdefault("num_hidden_layers", 32)
        kwargs.setdefault("num_attention_heads", 16)
        kwargs.setdefault("num_key_value_heads", 4)
        kwargs.setdefault("num_experts", 0)
        kwargs.setdefault("mlp_only_layers", [])
        if kwargs.get("layer_types") is None:
            full_attention_interval = kwargs.pop("full_attention_interval", 4)
            kwargs["layer_types"] = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(kwargs["num_hidden_layers"])
            ]

        if rope_parameters is None:
            rope_parameters = {}
        else:
            rope_parameters = dict(rope_parameters)

        rope_scaling = dict(kwargs.get("rope_scaling") or {})
        rope_parameters.setdefault("rope_type", rope_scaling.get("rope_type", "default"))
        rope_parameters.setdefault("rope_theta", kwargs.get("rope_theta", 10000.0))
        rope_parameters.setdefault("partial_rotary_factor", kwargs.get("partial_rotary_factor", 0.25))
        rope_parameters.setdefault("mrope_section", rope_scaling.pop("mrope_section", [11, 11, 10]))

        kwargs.setdefault("rope_theta", rope_parameters["rope_theta"])
        kwargs.setdefault("partial_rotary_factor", rope_parameters["partial_rotary_factor"])
        rope_scaling.setdefault("rope_type", rope_parameters["rope_type"])
        kwargs["rope_scaling"] = rope_scaling

        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)
        self.rope_parameters = rope_parameters
        self.rope_scaling = dict(self.rope_scaling or {"rope_type": rope_parameters["rope_type"]})
        self.rope_scaling["mrope_section"] = rope_parameters["mrope_section"]


class Qwen3_5VisionConfig(Qwen3VLVisionConfig):
    model_type = "qwen3_5_vision"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: Optional[list[int]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(
            depth=depth,
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            in_channels=in_channels,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            out_hidden_size=out_hidden_size,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=[] if deepstack_visual_indexes is None else deepstack_visual_indexes,
            initializer_range=initializer_range,
            **kwargs,
        )


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"text_config": Qwen3_5TextConfig, "vision_config": Qwen3_5VisionConfig}

    def __init__(
        self,
        text_config: Optional[Union[dict, Qwen3_5TextConfig]] = None,
        vision_config: Optional[Union[dict, Qwen3_5VisionConfig]] = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.pop("model_type", None)
            text_config = Qwen3_5TextConfig(**text_config)

        if vision_config is None:
            vision_config = {}
        if isinstance(vision_config, dict):
            vision_config = dict(vision_config)
            vision_config.pop("model_type", None)
            vision_config = Qwen3_5VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vocab_size = text_config.vocab_size
        self.tie_word_embeddings = text_config.tie_word_embeddings


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log())
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The Qwen3.5 fast path (flash-linear-attention and causal-conv1d) is not available, so the much "
                "slower torch implementation is used. Add it to this environment with "
                "`QWEN35_FASTPATH_ONLY=1 ENV_NAME=<env> bash scripts/install_env.sh`, and start training from a shell "
                "where the environment was activated with `conda activate <env>`."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Qwen3NextDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        segments: Optional[tuple[tuple[int, int], ...]] = None,
    ):
        if segments is not None and len(segments) > 1:
            # padding-free packed samples (batch size 1): the recurrence must restart at every sample, so each
            # sample runs separately here, inside the layer, whose parameters are already gathered
            if cache_params is not None or attention_mask is not None:
                raise ValueError("packed Qwen3.5 linear attention takes no cache and no attention mask.")
            outputs = [self.forward(hidden_states[:, start:end]) for start, end in segments]
            return torch.cat(outputs, dim=1)

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

        self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        linear_attn_segments: Optional[tuple[tuple[int, int], ...]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                segments=linear_attn_segments,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        return residual + hidden_states


class Qwen3_5TextModel(PreTrainedModel):
    config_class = Qwen3_5TextConfig
    base_model_prefix = "language_model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _no_split_modules = ["Qwen3_5DecoderLayer"]

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_linear_attn_mask(
        self, attention_mask: Optional[torch.Tensor], cache_position: torch.LongTensor
    ) -> Optional[torch.Tensor]:
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3NextDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def _build_mm_token_type_ids(input_ids: torch.Tensor, image_token_id: int, video_token_id: int) -> torch.Tensor:
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    mm_token_type_ids[input_ids == image_token_id] = 1
    mm_token_type_ids[input_ids == video_token_id] = 2
    return mm_token_type_ids


def _split_video_grid_thw(video_grid_thw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if video_grid_thw is None:
        return None

    video_grid_thw = video_grid_thw.clone()
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1
    return video_grid_thw


def _get_vision_position_ids(
    start_position: int,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item(),
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )
    position_temporal = torch.arange(llm_grid_t, device=device)
    position_width = torch.arange(llm_grid_w, device=device) + start_position
    position_height = torch.arange(llm_grid_h, device=device) + start_position

    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _compute_qwen3_5_rope_index(
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    spatial_merge_size: int,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_grid_thw = _split_video_grid_thw(video_grid_thw)
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    mrope_position_deltas = []
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask[batch_idx].bool()
            current_input_ids = current_input_ids[valid_mask]
            input_token_type = input_token_type[valid_mask]

        groups = []
        for modality_type, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda item: item[1]):
            group = list(group)
            groups.append((modality_type, group[0][0], group[-1][0] + 1))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in groups:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                if grid_iters[modality_type] is None:
                    raise ValueError("Qwen3.5 multimodal token ids were found but the matching grid_thw is missing.")
                grid_thw = next(grid_iters[modality_type])
                llm_pos_ids_list.append(
                    _get_vision_position_ids(current_pos, grid_thw, spatial_merge_size, input_ids.device)
                )
                current_pos += max(grid_thw[1].item(), grid_thw[2].item()) // spatial_merge_size

        if llm_pos_ids_list:
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        else:
            llm_positions = torch.empty(3, 0, dtype=input_ids.dtype, device=input_ids.device)

        if llm_positions.shape[-1] != current_input_ids.numel():
            raise ValueError(
                "Qwen3.5 M-RoPE position length does not match input length: "
                f"positions={llm_positions.shape[-1]}, input={current_input_ids.numel()}."
            )

        if valid_mask is not None:
            position_ids[:, batch_idx, valid_mask] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - current_input_ids.numel())

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Gets the 3D M-RoPE ids for Qwen3.5 before padding-free sharding."""
    squeeze_batch = input_ids.ndim == 1
    if squeeze_batch:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

    if image_grid_thw is None and video_grid_thw is None:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
    else:
        spatial_merge_size = getattr(processor.image_processor, "merge_size", None)
        if spatial_merge_size is None:
            spatial_merge_size = getattr(processor.image_processor, "spatial_merge_size", 2)
        image_token_id = getattr(processor, "image_token_id", 248056)
        video_token_id = getattr(processor, "video_token_id", 248057)
        mm_token_type_ids = _build_mm_token_type_ids(input_ids, image_token_id, video_token_id)
        position_ids, _ = _compute_qwen3_5_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            spatial_merge_size=spatial_merge_size,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    return position_ids[:, 0] if squeeze_batch else position_ids


class Qwen3_5Model(PreTrainedModel):
    config_class = Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3VLVisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(image_embeds, split_sizes), deepstack_image_embeds

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                f"features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, "
                f"features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def _merge_multimodal_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.FloatTensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_token_feature_indices: Optional[torch.LongTensor] = None,
        video_token_feature_indices: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.FloatTensor, Optional[torch.Tensor]]:
        if pixel_values is not None:
            image_embeds, _ = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if image_token_feature_indices is not None:
                image_indices = image_token_feature_indices.to(inputs_embeds.device)
                image_indices = image_indices[image_indices >= 0]
                image_embeds = image_embeds.index_select(0, image_indices)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, _ = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if video_token_feature_indices is not None:
                video_indices = video_token_feature_indices.to(inputs_embeds.device)
                video_indices = video_indices[video_indices >= 0]
                video_embeds = video_embeds.index_select(0, video_indices)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if pixel_values is None and pixel_values_videos is None:
            config = self.config.vision_config
            patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
            dummy_pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            dummy_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
            dummy_image_embeds, dummy_deepstack_image_embeds = self.visual(dummy_pixel_values, grid_thw=dummy_grid_thw)
            inputs_embeds = inputs_embeds + 0.0 * dummy_image_embeds.mean()
            for emb in dummy_deepstack_image_embeds or []:
                inputs_embeds = inputs_embeds + 0.0 * emb.mean()

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        return inputs_embeds, attention_mask

    @staticmethod
    def _is_packed_position_ids(position_ids: Optional[torch.LongTensor]) -> bool:
        if position_ids is None:
            return False
        if position_ids.ndim == 2 and position_ids.shape[0] == 1:
            return bool((torch.diff(position_ids[0]) < 0).any())
        if position_ids.ndim == 3 and position_ids.shape[0] == 4 and position_ids.shape[1] == 1:
            return bool((torch.diff(position_ids[0, 0]) < 0).any())
        return False

    @staticmethod
    def _packed_segments(position_ids: torch.LongTensor) -> tuple[tuple[int, int], ...]:
        """(start, end) of every sample in a padding-free packed sequence, where the text position restarts at 0."""
        token_positions = position_ids[0] if position_ids.ndim == 2 else position_ids[0, 0]
        starts = (token_positions == 0).nonzero(as_tuple=False).flatten()
        if starts.numel() == 0 or starts[0].item() != 0:
            starts = torch.cat([starts.new_zeros(1), starts])
        ends = torch.cat([starts[1:], starts.new_tensor([token_positions.numel()])])
        return tuple(zip(starts.tolist(), ends.tolist()))

    def compute_3d_position_ids(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
    ) -> Optional[torch.Tensor]:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if input_ids is not None and mm_token_type_ids is None and has_multimodal:
            mm_token_type_ids = _build_mm_token_type_ids(
                input_ids, self.config.image_token_id, self.config.video_token_id
            )

        if input_ids is not None and mm_token_type_ids is not None and has_multimodal:
            position_ids, rope_deltas = _compute_qwen3_5_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=mm_token_type_ids,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
            return position_ids

        if self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            return position_ids + delta.to(device=inputs_embeds.device)

        return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        image_token_feature_indices: Optional[torch.LongTensor] = None,
        video_token_feature_indices: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        inputs_embeds, attention_mask = self._merge_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_token_feature_indices=image_token_feature_indices,
            video_token_feature_indices=video_token_feature_indices,
        )

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        if get_ulysses_sequence_parallel_world_size() > 1:
            # the linear-attention recurrence would only see this rank's slice of each sequence
            raise ValueError("Qwen3.5 does not support Ulysses sequence parallelism (worker.actor.ulysses_size=1).")

        if (
            attention_mask is None
            and self._is_packed_position_ids(position_ids)
            and past_key_values is None
            and cache_position is None
            and not use_cache
        ):
            # Padding-free packed samples run through the language model in one call, so that every rank calls each
            # FSDP unit once whatever its number of samples: flash attention keeps the full-attention layers within
            # each sample (from the restarting position ids) and the linear-attention layers run sample by sample.
            if self.language_model.config._attn_implementation != "flash_attention_2":
                raise ValueError("packed Qwen3.5 inputs require attn_implementation='flash_attention_2'.")
            kwargs["linear_attn_segments"] = self._packed_segments(position_ids)

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen3_5ForConditionalGeneration(PreTrainedModel):
    config_class = Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3VLVisionBlock"]
    _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        image_token_feature_indices: Optional[torch.LongTensor] = None,
        video_token_feature_indices: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Qwen3VLCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            image_token_feature_indices=image_token_feature_indices,
            video_token_feature_indices=video_token_feature_indices,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


_QWEN3_5_REGISTERED = False


def register_qwen3_5() -> None:
    global _QWEN3_5_REGISTERED
    if _QWEN3_5_REGISTERED:
        return

    try:
        AutoConfig.register(Qwen3_5Config.model_type, Qwen3_5Config)
    except ValueError:
        pass

    try:
        AutoModelForCausalLM.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration)
    except ValueError:
        pass

    if AutoModelForImageTextToText is not None:
        try:
            AutoModelForImageTextToText.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration)
        except ValueError:
            pass

    _QWEN3_5_REGISTERED = True
