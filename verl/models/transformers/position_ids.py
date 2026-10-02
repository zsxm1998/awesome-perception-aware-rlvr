# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from typing import Any, Mapping, Optional

import torch


def is_qwen_vl_processor(processor: Any) -> bool:
    """Return whether ``processor`` uses EasyR1's Qwen-VL M-RoPE layout."""

    if processor is None:
        return False
    if getattr(processor, "_easy_r1_model_type", None) == "qwen3_5":
        return True
    processor_name = processor.__class__.__name__
    if any(family in processor_name for family in ("Qwen2VLProcessor", "Qwen2_5_VLProcessor", "Qwen3VLProcessor")):
        return True
    image_processor = getattr(processor, "image_processor", None)
    return image_processor is not None and "Qwen2VLImageProcessor" in image_processor.__class__.__name__


def _get_qwen_rope_index(processor: Any):
    if getattr(processor, "_easy_r1_model_type", None) == "qwen3_5":
        from .qwen3_5 import get_rope_index
    elif "Qwen3VLProcessor" in processor.__class__.__name__:
        from .qwen3_vl import get_rope_index
    else:
        from .qwen2_vl import get_rope_index
    return get_rope_index


def build_multimodal_position_ids(
    processor: Optional[Any],
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Build the canonical EasyR1 position IDs for one unbatched sample.

    Qwen-VL processors receive one text channel plus their native three-channel
    M-RoPE. Other processors use the existing scalar cumulative positions.
    """

    if input_ids.ndim != 1 or attention_mask.ndim != 1:
        raise ValueError("position IDs must be built from one unbatched 1D sample")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have identical shapes")

    if not is_qwen_vl_processor(processor):
        return torch.clip(attention_mask.long().cumsum(dim=0) - 1, min=0, max=None).to(
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

    model_inputs = dict(multi_modal_inputs or {})
    get_rope_index = _get_qwen_rope_index(processor)
    vision_position_ids = get_rope_index(
        processor,
        input_ids=input_ids,
        image_grid_thw=model_inputs.get("image_grid_thw"),
        video_grid_thw=model_inputs.get("video_grid_thw"),
        second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
        attention_mask=attention_mask,
    )
    text_position_ids = attention_mask.long().cumsum(dim=0) - 1
    text_position_ids.masked_fill_(attention_mask == 0, 0)
    text_position_ids = text_position_ids.to(
        device=input_ids.device,
        dtype=input_ids.dtype,
    ).unsqueeze(0)
    return torch.cat(
        (
            text_position_ids,
            vision_position_ids.to(device=input_ids.device, dtype=input_ids.dtype),
        ),
        dim=0,
    )
