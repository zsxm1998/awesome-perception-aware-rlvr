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

from typing import TYPE_CHECKING, List, Tuple

import torch


if TYPE_CHECKING:
    from transformers.models.llama.configuration_llama import LlamaConfig


def get_device_flops(unit: str = "T") -> float:
    def unit_convert(number: float, level: str):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number

        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1

        return number

    device_name = torch.cuda.get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type
    if "H100" in device_name or "H800" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12

    flops_unit = unit_convert(flops, unit)
    return flops_unit


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)
    """

    def __init__(self, config: "LlamaConfig"):
        _ESTIMATE_FUNC = {
            "llama": self._estimate_llama_flops,
            "qwen2": self._estimate_llama_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_llama_flops,
            "qwen2_5_vl": self._estimate_llama_flops,
            "qwen3": self._estimate_llama_flops,
            "qwen3_vl": self._estimate_llama_flops,
            "qwen3_5": self._estimate_qwen3_5_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl_moe": self._estimate_qwen2_moe_flops,
            "internvl_chat": self._estimate_internvl_chat_flops,
        }

        if config.model_type not in _ESTIMATE_FUNC:
            print(f"Only support {_ESTIMATE_FUNC.keys()}, but got {config.model_type}. MFU will always be zero.")

        self.config = getattr(config, "text_config", config)
        self._estimate_flops = _ESTIMATE_FUNC.get(config.model_type, self._estimate_unknown_flops)

    def _estimate_unknown_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        return 0

    def _estimate_llama_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        return self._estimate_llama_flops_from_config(self.config, tokens_sum, batch_seqlens, delta_time)

    def _estimate_llama_flops_from_config(
        self, config: "LlamaConfig", tokens_sum: int, batch_seqlens: List[int], delta_time: float
    ) -> float:
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen

        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_internvl_chat_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        # The current MFU interface only receives sequence lengths, not per-batch
        # pixel tile counts, so count the InternVL language model FLOPs here.
        llm_config = getattr(self.config, "llm_config", None)
        if llm_config is None:
            return self._estimate_unknown_flops(tokens_sum, batch_seqlens, delta_time)
        return self._estimate_llama_flops_from_config(llm_config, tokens_sum, batch_seqlens, delta_time)

    def _estimate_qwen3_5_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        layer_types = list(getattr(config, "layer_types", None) or [])
        if not layer_types:
            full_attention_interval = getattr(config, "full_attention_interval", 4)
            layer_types = [
                "linear_attention" if bool((layer_idx + 1) % full_attention_interval) else "full_attention"
                for layer_idx in range(num_hidden_layers)
            ]

        layer_types = layer_types[:num_hidden_layers]
        full_attention_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
        linear_attention_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
        full_attention_layers += num_hidden_layers - full_attention_layers - linear_attention_layers

        # Qwen3.5 uses the Qwen3-Next gated attention projection:
        # q_proj produces query and gate, so full-attention layers have 2 * q_size q-proj output.
        mlp_N = hidden_size * intermediate_size * 3
        full_attn_linear_N = hidden_size * (q_size * 3 + k_size + v_size)

        linear_head_k_dim = getattr(config, "linear_key_head_dim", head_dim)
        linear_head_v_dim = getattr(config, "linear_value_head_dim", head_dim)
        linear_num_key_heads = getattr(config, "linear_num_key_heads", num_key_value_heads)
        linear_num_value_heads = getattr(config, "linear_num_value_heads", num_attention_heads)
        linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

        linear_key_dim = linear_head_k_dim * linear_num_key_heads
        linear_value_dim = linear_head_v_dim * linear_num_value_heads
        linear_conv_dim = linear_key_dim * 2 + linear_value_dim
        linear_attn_linear_N = (
            hidden_size * (linear_key_dim * 2 + linear_value_dim * 3 + linear_num_value_heads * 2)
            + linear_conv_dim * linear_conv_kernel_dim
        )

        emd_and_lm_head_N = vocab_size * hidden_size * 2
        dense_N = (
            mlp_N * num_hidden_layers
            + full_attn_linear_N * full_attention_layers
            + linear_attn_linear_N * linear_attention_layers
            + emd_and_lm_head_N
        )
        dense_N_flops = 6 * dense_N * tokens_sum

        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen

        full_attn_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * full_attention_layers
        linear_attn_state_flops = (
            12 * tokens_sum * linear_attention_layers * linear_num_value_heads * linear_head_k_dim * linear_head_v_dim
        )

        flops_all_token = dense_N_flops + full_attn_flops + linear_attn_state_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_moe_flops(self, tokens_sum: int, batch_seqlens: List[int], delta_time: float) -> float:
        config = self.config
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        num_attention_heads = config.num_attention_heads
        moe_intermediate_size = config.moe_intermediate_size
        moe_topk = config.num_experts_per_tok
        num_experts = config.num_experts

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # gate + moe export
        moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen

        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens: List[int], delta_time: float) -> Tuple[float, float]:
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        estimated_flops = self._estimate_flops(tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
