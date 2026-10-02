# Copyright 2024 The Fairseq Authors and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Based on https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/modeling_flash_attention_utils.py
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

import inspect
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import _flash_attention_forward, fa_peft_integration_check
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from ...utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_deterministic_enabled = os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


@dataclass(frozen=True)
class _AttentionInterventionState:
    attention_mask: torch.Tensor
    visual_spans: tuple[tuple[tuple[int, int], ...], ...]
    saliency_std_multiplier: float


_ACTIVE_ATTENTION_INTERVENTION: ContextVar[Optional[_AttentionInterventionState]] = ContextVar(
    "active_attention_intervention",
    default=None,
)
_CROSS_MODAL_QUERY_CHUNK_SIZE = 64
# The official CFPO call path explicitly passes ``attention > 1e-8`` as its
# validity mask. Its standalone GMM helper's 1e-6 fallback is not used there.
_CROSS_MODAL_VALID_ATTENTION_EPS = 1e-8


def _resolve_visual_spans(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_start_token_id: int,
    vision_end_token_id: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Model-level visual corruption requires 2D input_ids and a matching attention_mask.")

    all_spans: list[tuple[tuple[int, int], ...]] = []
    valid_mask = attention_mask.to(torch.bool)
    for row_idx in range(input_ids.size(0)):
        row_ids = input_ids[row_idx]
        row_valid = valid_mask[row_idx]
        starts = torch.where((row_ids == vision_start_token_id) & row_valid)[0].tolist()
        ends = torch.where((row_ids == vision_end_token_id) & row_valid)[0].tolist()
        if len(starts) != len(ends):
            raise ValueError(
                "Model-level visual corruption found unbalanced vision_start/vision_end tokens "
                f"in sample {row_idx}: {len(starts)} start token(s), {len(ends)} end token(s)."
            )

        row_spans: list[tuple[int, int]] = []
        for start, end in zip(starts, ends):
            visual_start = int(start) + 1
            visual_end = int(end)
            if visual_start >= visual_end:
                raise ValueError(
                    f"Model-level visual corruption found an empty or reversed visual span in sample {row_idx}."
                )
            row_spans.append((visual_start, visual_end))
        all_spans.append(tuple(row_spans))
    return tuple(all_spans)


@contextmanager
def use_model_level_visual_corruption(
    *,
    name: str,
    kwargs: Optional[dict[str, Any]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_start_token_id: int,
    vision_end_token_id: int,
) -> Iterator[None]:
    if name != "cross_modal_attention_value_mean":
        raise ValueError(f"Unsupported model-level visual corruption: {name!r}.")

    corruption_kwargs = kwargs or {}
    state = _AttentionInterventionState(
        attention_mask=attention_mask.to(torch.bool),
        visual_spans=_resolve_visual_spans(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
        ),
        saliency_std_multiplier=float(corruption_kwargs.get("saliency_std_multiplier", 2.0)),
    )
    token = _ACTIVE_ATTENTION_INTERVENTION.set(state)
    try:
        yield
    finally:
        _ACTIVE_ATTENTION_INTERVENTION.reset(token)


def _repeat_key_value_heads(states: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_key_value_heads = states.size(2)
    if num_key_value_heads == num_query_heads:
        return states
    if num_query_heads % num_key_value_heads != 0:
        raise ValueError(
            f"Query heads ({num_query_heads}) must be divisible by key/value heads ({num_key_value_heads})."
        )
    return states.repeat_interleave(num_query_heads // num_key_value_heads, dim=2)


def _attention_probabilities(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    valid_key_mask: torch.Tensor,
    scaling: float,
    is_causal: bool,
    sliding_window: Optional[int],
    softcap: Optional[float],
) -> torch.Tensor:
    logits = torch.einsum("qhd,khd->hqk", query, key) * scaling
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    if query.dtype == torch.float16:
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)

    key_positions = torch.arange(key.size(0), device=key.device)
    allowed_keys = valid_key_mask.unsqueeze(0).expand(query_positions.numel(), -1)
    if is_causal:
        allowed_keys = allowed_keys & (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
    if sliding_window is not None:
        window_start = query_positions.unsqueeze(1) - int(sliding_window) + 1
        allowed_keys = allowed_keys & (key_positions.unsqueeze(0) >= window_start)

    logits = logits.masked_fill(~allowed_keys.unsqueeze(0), torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)


def _merge_moments(
    count: int,
    mean: Optional[torch.Tensor],
    squared_deviation_sum: Optional[torch.Tensor],
    values: torch.Tensor,
) -> tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    values = values[values > _CROSS_MODAL_VALID_ATTENTION_EPS].float()
    chunk_count = values.numel()
    if chunk_count == 0:
        return count, mean, squared_deviation_sum

    chunk_mean = values.mean()
    chunk_squared_deviation_sum = torch.sum((values - chunk_mean) ** 2)
    if count == 0:
        return chunk_count, chunk_mean, chunk_squared_deviation_sum

    total_count = count + chunk_count
    delta = chunk_mean - mean
    merged_mean = mean + delta * (chunk_count / total_count)
    merged_squared_deviation_sum = (
        squared_deviation_sum + chunk_squared_deviation_sum + delta.square() * (count * chunk_count / total_count)
    )
    return total_count, merged_mean, merged_squared_deviation_sum


def compute_cross_modal_attention_value_mean_correction(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    visual_spans: tuple[tuple[tuple[int, int], ...], ...],
    saliency_std_multiplier: float,
    scaling: float,
    is_causal: bool = True,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    query_chunk_size: int = _CROSS_MODAL_QUERY_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute the exact CF visual-value correction without materializing a full attention matrix."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Attention intervention expects query/key/value tensors with shape [batch, seq, heads, dim].")
    if query.size(0) != attention_mask.size(0) or query.size(1) != attention_mask.size(1):
        raise ValueError("Attention intervention mask must match the query batch and sequence dimensions.")
    if len(visual_spans) != query.size(0):
        raise ValueError("Attention intervention visual spans must have one entry per batch sample.")
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, but got {query_chunk_size}.")

    num_query_heads = query.size(2)
    key = _repeat_key_value_heads(key, num_query_heads)
    value = _repeat_key_value_heads(value, num_query_heads)
    correction = torch.zeros_like(query)
    valid_mask = attention_mask.to(device=query.device, dtype=torch.bool)

    for row_idx, row_spans in enumerate(visual_spans):
        if not row_spans:
            continue

        visual_positions = torch.cat(
            [torch.arange(start, end, device=query.device, dtype=torch.long) for start, end in row_spans]
        )
        visual_values = value[row_idx].index_select(0, visual_positions)
        visual_values_by_head = visual_values.permute(1, 0, 2).contiguous()
        mean_visual_value = (visual_values_by_head.sum(dim=(1, 2)) / (visual_positions.numel() * value.size(-1))).view(
            num_query_heads, 1, 1
        )
        row_query = query[row_idx]
        row_key = key[row_idx]
        row_value = value[row_idx]
        row_valid_mask = valid_mask[row_idx]

        for visual_start, visual_end in row_spans:
            query_positions = torch.arange(visual_end + 1, query.size(1), device=query.device)
            query_positions = query_positions[row_valid_mask[query_positions]]
            if query_positions.numel() == 0:
                continue

            count = 0
            mean = None
            squared_deviation_sum = None
            for query_chunk in query_positions.split(query_chunk_size):
                probabilities = _attention_probabilities(
                    row_query.index_select(0, query_chunk),
                    row_key,
                    query_positions=query_chunk,
                    valid_key_mask=row_valid_mask,
                    scaling=scaling,
                    is_causal=is_causal,
                    sliding_window=sliding_window,
                    softcap=softcap,
                )
                count, mean, squared_deviation_sum = _merge_moments(
                    count,
                    mean,
                    squared_deviation_sum,
                    probabilities[:, :, visual_start:visual_end],
                )

            if count == 0:
                continue
            if count == 1:
                standard_deviation = mean.new_tensor(float("nan"))
            else:
                standard_deviation = torch.sqrt(squared_deviation_sum / (count - 1))
            # CFPO computes mean/std on the model-dtype attention probabilities.
            # Keep FP32 Welford accumulation, then reproduce that final BF16/FP16
            # rounding point before selecting salient entries.
            threshold = mean.to(row_query.dtype) + saliency_std_multiplier * standard_deviation.to(row_query.dtype)
            visual_value_delta = mean_visual_value - row_value[visual_start:visual_end].permute(1, 0, 2)

            for query_chunk in query_positions.split(query_chunk_size):
                probabilities = _attention_probabilities(
                    row_query.index_select(0, query_chunk),
                    row_key,
                    query_positions=query_chunk,
                    valid_key_mask=row_valid_mask,
                    scaling=scaling,
                    is_causal=is_causal,
                    sliding_window=sliding_window,
                    softcap=softcap,
                )
                cross_modal_probabilities = probabilities[:, :, visual_start:visual_end]
                salient_probabilities = cross_modal_probabilities * (cross_modal_probabilities > threshold).to(
                    cross_modal_probabilities.dtype
                )
                chunk_correction = torch.einsum(
                    "hqi,hid->qhd",
                    salient_probabilities,
                    visual_value_delta,
                )
                correction[row_idx].index_add_(0, query_chunk, chunk_correction)

    return correction


def prepare_fa2_from_position_ids(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
):
    assert position_ids.ndim == 2  # (batch_size, seq_length)
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}
    position_ids = position_ids.view(-1)
    cu_seqlens = torch.cat(
        (
            (position_ids == 0).nonzero().view(-1).to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length for qwen2vl mrope
    return (query, key, value, (cu_seqlens, cu_seqlens), (max_length, max_length))


def _custom_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    deterministic: Optional[bool] = None,
    **kwargs,
):
    """
    Patches flash attention forward to handle 3D position ids in mrope. (3, batch_size, seq_length)
    """
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if _flash_supports_deterministic:
        flash_kwargs["deterministic"] = deterministic if deterministic is not None else _flash_deterministic_enabled

    if kwargs.get("softcap") is not None:
        flash_kwargs["softcap"] = kwargs.pop("softcap")

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype=torch.bfloat16
    )

    sp_size = get_ulysses_sequence_parallel_world_size()
    if sp_size > 1 and position_ids is not None:
        # qkv: (batch_size, seq_length / sp_size, num_head, head_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)
        position_ids_lst = [torch.empty_like(position_ids) for _ in range(sp_size)]
        position_ids = dist.all_gather(position_ids_lst, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.cat(position_ids_lst, dim=-1)  # (batch_size, seq_length)

    if position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
        batch_size = query_states.size(0)
        q, k, v, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids
        )
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=kwargs.pop("dropout", 0.0),
            softmax_scale=kwargs.pop("softmax_scale", None),
            causal=is_causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            position_ids=position_ids,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            deterministic=deterministic,
            **kwargs,
        )

    if sp_size > 1 and position_ids is not None:
        # output: (batch_size, seq_length / sp_size, num_head, head_size)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    return attn_output


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    q_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # FA2 uses the kwargs value if explicitly passed, otherwise it uses the module attribute
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)

    attn_output = _custom_flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=q_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_flash_use_top_left_mask,
        **kwargs,
    )

    intervention = _ACTIVE_ATTENTION_INTERVENTION.get()
    if intervention is not None and is_causal:
        if get_ulysses_sequence_parallel_world_size() > 1:
            raise ValueError("Model-level visual corruption currently requires worker.actor.ulysses_size=1.")
        correction = compute_cross_modal_attention_value_mean_correction(
            query=query,
            key=key,
            value=value,
            attention_mask=intervention.attention_mask,
            visual_spans=intervention.visual_spans,
            saliency_std_multiplier=intervention.saliency_std_multiplier,
            scaling=float(scaling if scaling is not None else query.size(-1) ** -0.5),
            is_causal=is_causal,
            sliding_window=sliding_window,
            softcap=softcap,
        )
        attn_output = attn_output + correction.to(attn_output.dtype)

    return attn_output, None
