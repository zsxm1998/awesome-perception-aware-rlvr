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

"""Materialize raw agent trajectories into model-ready multimodal samples.

The agent loop deliberately keeps vLLM's generated token IDs untouched. Vision
processors, however, expand one raw image placeholder into many model-visible
tokens. This module performs only that deterministic placeholder expansion,
then computes the model-native position IDs and the processor tensors required
by actor/reference forwards. Text and model-generated token IDs are never
decoded and re-tokenized.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch

from ...models.transformers.internvl import (
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    is_internvl_processor,
)
from ...models.transformers.position_ids import build_multimodal_position_ids
from ...protocol import DataProto
from ...utils.dataset import ProcessedImageInput, process_image
from .protocol import AgentTrajectory


class AgentTrajectoryRejected(ValueError):
    """One invalid trajectory, optionally safe to regenerate independently."""

    def __init__(self, message: str, *, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable


@dataclass(frozen=True)
class ImageTokenExpansion:
    """One image's raw placeholder and model-visible replacement."""

    raw_placeholder_ids: tuple[int, ...]
    expanded_placeholder_ids: tuple[int, ...]
    visual_start: int
    visual_end: int

    @property
    def visual_token_count(self) -> int:
        return self.visual_end - self.visual_start


@dataclass(frozen=True)
class PreparedAgentImageLayout:
    """Processed image sizes and placeholder expansion without pixel tensors."""

    images: tuple[ProcessedImageInput, ...]
    expansions: tuple[ImageTokenExpansion, ...]
    multi_modal_layout: dict[str, Any]

    @property
    def visual_token_counts(self) -> tuple[int, ...]:
        return tuple(expansion.visual_token_count for expansion in self.expansions)


@dataclass(frozen=True)
class MaterializedAgentTrajectory:
    """An unpadded, actor-ready representation of one agent trajectory."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    position_ids: torch.Tensor
    visual_token_mask: tuple[int, ...]
    multi_modal_data: dict[str, Any]
    multi_modal_layout: dict[str, Any]
    visual_token_counts: tuple[int, ...]
    # All spans use absolute offsets into this sample's unpadded ``input_ids``.
    action_spans: tuple[tuple[int, int], ...]
    observation_spans: tuple[tuple[int, int], ...]
    image_spans: tuple[tuple[int, int], ...]
    source_image_count: int
    prompt_token_budget: int
    response_token_budget: int
    trajectory: AgentTrajectory

    @property
    def input_ids(self) -> tuple[int, ...]:
        return (*self.prompt_ids, *self.response_ids)

    @property
    def effective_response_tokens(self) -> int:
        return len(self.response_ids)

    @property
    def visual_tokens(self) -> int:
        return sum(self.visual_token_counts)


def _encode_literal(tokenizer: Any, text: str) -> tuple[int, ...]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError(f"tokenizer produced no token IDs for multimodal literal {text!r}")
    return tuple(int(token_id) for token_id in token_ids)


def prepare_agent_image_layout(
    processor: Any,
    images: Sequence[Any],
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    image_preprocessor: Optional[Any] = None,
) -> PreparedAgentImageLayout:
    """Derive visual placeholder lengths without constructing pixel tensors."""

    if processor is None or not hasattr(processor, "image_processor"):
        raise TypeError("agent image layout requires a multimodal processor")
    if not hasattr(processor, "tokenizer"):
        raise TypeError("multimodal processor must expose its tokenizer")

    if image_preprocessor is None:
        processed = [process_image(image, min_pixels, max_pixels) for image in images]
    else:
        processed = [image_preprocessor(image) for image in images]
    processed_images = tuple(ProcessedImageInput(image) for image in processed)
    if not processed_images:
        return PreparedAgentImageLayout(
            images=(),
            expansions=(),
            multi_modal_layout={"kind": "empty"},
        )

    expansions = []
    if is_internvl_processor(processor):
        raw_placeholder_ids = _encode_literal(processor.tokenizer, "<image>")
        start_ids = _encode_literal(processor.tokenizer, IMG_START_TOKEN)
        end_ids = _encode_literal(processor.tokenizer, IMG_END_TOKEN)
        context_id = int(processor.image_token_id)
        for wrapped_image in processed_images:
            num_patches = int(processor.get_num_image_patches(wrapped_image.image))
            num_context_tokens = num_patches * int(processor.num_image_token)
            expanded_ids = (
                *start_ids,
                *([context_id] * num_context_tokens),
                *end_ids,
            )
            expansions.append(
                ImageTokenExpansion(
                    raw_placeholder_ids=raw_placeholder_ids,
                    expanded_placeholder_ids=tuple(expanded_ids),
                    visual_start=len(start_ids),
                    visual_end=len(start_ids) + num_context_tokens,
                )
            )
        return PreparedAgentImageLayout(
            images=processed_images,
            expansions=tuple(expansions),
            multi_modal_layout={
                "kind": "internvl",
                "pixel_values_batch": sum(
                    int(processor.get_num_image_patches(image.image)) for image in processed_images
                ),
            },
        )

    image_processor = processor.image_processor
    image_token_id = getattr(processor, "image_token_id", None)
    merge_size = getattr(image_processor, "merge_size", None)
    get_num_patches = getattr(image_processor, "get_number_of_image_patches", None)
    if not isinstance(image_token_id, int) or not isinstance(merge_size, int) or not callable(get_num_patches):
        raise TypeError(
            "unsupported multimodal processor layout: expected Qwen-style "
            "get_number_of_image_patches or an InternVLProcessorAdapter"
        )

    image_processor_module = importlib.import_module(type(image_processor).__module__)
    smart_resize = getattr(image_processor_module, "smart_resize", None)
    patch_size = int(getattr(image_processor, "patch_size", 1))
    temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 2))
    factor = patch_size * merge_size
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        min_layout_pixels = int(size["shortest_edge"])
        max_layout_pixels = int(size["longest_edge"])
    elif size is not None:
        min_layout_pixels = int(size.shortest_edge)
        max_layout_pixels = int(size.longest_edge)
    else:
        min_layout_pixels = max_layout_pixels = None

    image_grid_thw = []
    for wrapped_image in processed_images:
        width, height = wrapped_image.image.size
        num_patches = int(
            get_num_patches(
                height=height,
                width=width,
                images_kwargs={},
            )
        )
        num_image_tokens = num_patches // (merge_size**2)
        if num_image_tokens <= 0 or num_image_tokens * (merge_size**2) != num_patches:
            raise RuntimeError("Qwen image layout produced a non-integral visual token count")
        if callable(smart_resize):
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=min_layout_pixels,
                max_pixels=max_layout_pixels,
            )
            grid_row = (
                1,
                int(resized_height) // patch_size,
                int(resized_width) // patch_size,
            )
        else:
            # Minimal/custom processors may expose only the vLLM patch-count
            # hook. Recover the unique factor pair closest to the source aspect
            # ratio without constructing pixel tensors.
            target_ratio = height / width
            factor_pairs = [
                (candidate, num_patches // candidate)
                for candidate in range(1, int(math.sqrt(num_patches)) + 1)
                if num_patches % candidate == 0
            ]
            grid_h, grid_w = min(
                (pair for height_width in factor_pairs for pair in (height_width, height_width[::-1])),
                key=lambda pair: abs((pair[0] / pair[1]) - target_ratio),
            )
            grid_row = (1, grid_h, grid_w)
        if grid_row[1] * grid_row[2] != num_patches:
            raise RuntimeError("Qwen lightweight layout disagrees with the processor patch count")
        if temporal_patch_size <= 0:
            raise RuntimeError("Qwen temporal_patch_size must be positive")
        image_grid_thw.append(grid_row)
        expansions.append(
            ImageTokenExpansion(
                raw_placeholder_ids=(image_token_id,),
                expanded_placeholder_ids=tuple([image_token_id] * num_image_tokens),
                visual_start=0,
                visual_end=num_image_tokens,
            )
        )
    return PreparedAgentImageLayout(
        images=processed_images,
        expansions=tuple(expansions),
        multi_modal_layout={
            "kind": "qwen",
            "image_grid_thw": tuple(image_grid_thw),
        },
    )


def validate_agent_multi_modal_layout(
    layout: dict[str, Any],
    multi_modal_inputs: dict[str, Any],
) -> None:
    """Validate worker-recomputed processor tensors against a compact layout."""

    kind = layout.get("kind")
    if kind == "empty":
        if multi_modal_inputs:
            raise RuntimeError("empty agent layout unexpectedly produced multimodal tensors")
        return
    if kind == "qwen":
        actual_grid = multi_modal_inputs.get("image_grid_thw")
        if actual_grid is None:
            raise RuntimeError("Qwen worker inputs are missing image_grid_thw")
        actual_rows = tuple(tuple(int(value) for value in row) for row in torch.as_tensor(actual_grid).tolist())
        expected_rows = tuple(tuple(int(value) for value in row) for row in layout["image_grid_thw"])
        if actual_rows != expected_rows:
            raise RuntimeError(
                "worker-recomputed Qwen image layout does not match agent token/M-RoPE layout: "
                f"{actual_rows} != {expected_rows}"
            )
        return
    if kind == "internvl":
        pixel_values = multi_modal_inputs.get("pixel_values")
        if pixel_values is None or not hasattr(pixel_values, "shape"):
            raise RuntimeError("InternVL worker inputs are missing pixel_values")
        actual_batch = int(pixel_values.shape[0])
        expected_batch = int(layout["pixel_values_batch"])
        if actual_batch != expected_batch:
            raise RuntimeError(
                "worker-recomputed InternVL patch layout does not match agent token layout: "
                f"{actual_batch} != {expected_batch}"
            )
        return
    raise ValueError(f"unsupported agent multimodal layout kind: {kind!r}")


def measure_expanded_observation(
    processor: Any,
    token_ids: Sequence[int],
    images: Sequence[Any],
    *,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    image_preprocessor: Optional[Any] = None,
) -> tuple[int, int]:
    """Return model-visible observation length and visual feature-token count."""

    if not images:
        return len(token_ids), 0
    prepared = prepare_agent_image_layout(
        processor,
        images,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_preprocessor=image_preprocessor,
    )
    expanded_ids, _, _ = _expand_placeholders(token_ids, prepared.expansions)
    return len(expanded_ids), sum(prepared.visual_token_counts)


def _expand_placeholders(
    token_ids: Sequence[int],
    expansions: Sequence[ImageTokenExpansion],
    *,
    token_mask: Optional[Sequence[int]] = None,
    require_mask_value: Optional[int] = None,
) -> tuple[list[int], Optional[list[int]], list[tuple[int, int]]]:
    token_ids = list(token_ids)
    if token_mask is not None:
        token_mask = list(token_mask)
        if len(token_ids) != len(token_mask):
            raise ValueError("token IDs and token mask must have equal lengths before image expansion")

    expanded_ids: list[int] = []
    expanded_mask: Optional[list[int]] = [] if token_mask is not None else None
    visual_spans: list[tuple[int, int]] = []
    cursor = 0
    for expansion in expansions:
        placeholder = list(expansion.raw_placeholder_ids)
        match_index = -1
        for index in range(cursor, len(token_ids) - len(placeholder) + 1):
            if token_ids[index : index + len(placeholder)] != placeholder:
                continue
            if token_mask is not None and require_mask_value is not None:
                candidate_mask = token_mask[index : index + len(placeholder)]
                # InternVL's raw "<image>" marker is ordinary text. A model may
                # legitimately quote it, so only environment-owned occurrences
                # are candidates for multimodal expansion.
                if any(mask_value != require_mask_value for mask_value in candidate_mask):
                    continue
            match_index = index
            break
        if match_index < 0:
            raise AgentTrajectoryRejected("trajectory image count does not match raw image placeholders")

        expanded_ids.extend(token_ids[cursor:match_index])
        if expanded_mask is not None:
            expanded_mask.extend(token_mask[cursor:match_index])
            placeholder_mask = token_mask[match_index : match_index + len(placeholder)]
            if len(set(placeholder_mask)) != 1:
                raise ValueError("one image placeholder crosses response-mask ownership boundaries")
            mask_value = placeholder_mask[0]
        else:
            mask_value = 0

        replacement_start = len(expanded_ids)
        expanded_ids.extend(expansion.expanded_placeholder_ids)
        visual_spans.append(
            (
                replacement_start + expansion.visual_start,
                replacement_start + expansion.visual_end,
            )
        )
        if expanded_mask is not None:
            expanded_mask.extend([mask_value] * len(expansion.expanded_placeholder_ids))
        cursor = match_index + len(placeholder)

    expanded_ids.extend(token_ids[cursor:])
    if expanded_mask is not None:
        expanded_mask.extend(token_mask[cursor:])

    if expansions:
        placeholder = list(expansions[0].raw_placeholder_ids)
        for index in range(cursor, len(token_ids) - len(placeholder) + 1):
            if token_ids[index : index + len(placeholder)] != placeholder:
                continue
            if token_mask is not None and require_mask_value is not None:
                candidate_mask = token_mask[index : index + len(placeholder)]
                if any(mask_value != require_mask_value for mask_value in candidate_mask):
                    continue
            raise AgentTrajectoryRejected("raw trajectory contains more image placeholders than trajectory images")

    return expanded_ids, expanded_mask, visual_spans


def _contains_subsequence(
    token_ids: Sequence[int],
    pattern: Sequence[int],
    *,
    token_mask: Optional[Sequence[int]] = None,
    mask_value: Optional[int] = None,
) -> bool:
    if not pattern:
        return False
    token_ids = list(token_ids)
    pattern = list(pattern)
    if token_mask is not None:
        token_mask = list(token_mask)
        if len(token_ids) != len(token_mask):
            raise ValueError("token IDs and token mask must align")
    for index in range(len(token_ids) - len(pattern) + 1):
        if token_ids[index : index + len(pattern)] != pattern:
            continue
        if token_mask is not None and mask_value is not None:
            if any(value != mask_value for value in token_mask[index : index + len(pattern)]):
                continue
        return True
    return False


def _contiguous_spans(mask: Sequence[int], value: int) -> tuple[tuple[int, int], ...]:
    spans = []
    start = None
    for index, item in enumerate([*mask, 1 - value]):
        if item == value and start is None:
            start = index
        elif item != value and start is not None:
            spans.append((start, index))
            start = None
    return tuple(spans)


class AgentTrajectoryMaterializer:
    """Convert a raw token-in/token-out trajectory to actor/ref model inputs."""

    def __init__(
        self,
        processor: Any,
        *,
        max_prompt_tokens: int,
        max_model_len: Optional[int] = None,
    ):
        if isinstance(max_prompt_tokens, bool) or not isinstance(max_prompt_tokens, int) or max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be a positive integer")
        if max_model_len is not None and max_model_len <= 0:
            raise ValueError("max_model_len must be positive when provided")
        self.processor = processor
        self.max_prompt_tokens = max_prompt_tokens
        self.max_model_len = max_model_len

    def materialize(self, trajectory: AgentTrajectory) -> MaterializedAgentTrajectory:
        if trajectory.response_token_budget is None or trajectory.response_token_budget <= 0:
            raise ValueError("agent trajectory must carry one positive response_token_budget")
        response_token_budget = trajectory.response_token_budget
        all_images = trajectory.all_images
        image_config = trajectory.image_config
        if image_config.limit_images is not None and len(all_images) > image_config.limit_images:
            raise AgentTrajectoryRejected(
                f"trajectory uses {len(all_images)} images but rollout limit_images={image_config.limit_images}"
            )

        prepared = prepare_agent_image_layout(
            self.processor,
            all_images,
            min_pixels=image_config.min_pixels,
            max_pixels=image_config.max_pixels,
        )
        source_count = len(trajectory.source_images)
        prompt_ids, _, prompt_visual_spans = _expand_placeholders(
            trajectory.prompt_ids,
            prepared.expansions[:source_count],
        )
        observation_expansions = prepared.expansions[source_count:]
        image_token_id = getattr(self.processor, "image_token_id", None)
        if isinstance(image_token_id, int) and any(
            token_id == image_token_id and mask_value == 1
            for token_id, mask_value in zip(trajectory.response_ids, trajectory.response_mask)
        ):
            raise AgentTrajectoryRejected(
                "model generated a reserved visual context token",
                recoverable=True,
            )
        if (
            not observation_expansions
            and prepared.expansions
            and _contains_subsequence(
                trajectory.response_ids,
                prepared.expansions[0].raw_placeholder_ids,
                token_mask=trajectory.response_mask,
                mask_value=0,
            )
        ):
            raise AgentTrajectoryRejected("environment image placeholders have no matching observation image")
        response_ids, response_mask, response_visual_spans = _expand_placeholders(
            trajectory.response_ids,
            observation_expansions,
            token_mask=trajectory.response_mask,
            require_mask_value=0,
        )
        assert response_mask is not None

        raw_action_ids = [
            token_id
            for token_id, mask_value in zip(trajectory.response_ids, trajectory.response_mask)
            if mask_value == 1
        ]
        expanded_action_ids = [
            token_id for token_id, mask_value in zip(response_ids, response_mask) if mask_value == 1
        ]
        if raw_action_ids != expanded_action_ids:
            raise RuntimeError("multimodal expansion modified model-generated action token IDs")
        if len(response_ids) > response_token_budget:
            raise AgentTrajectoryRejected(
                f"expanded agent response exceeds its trajectory budget: {len(response_ids)} > {response_token_budget}"
            )
        if trajectory.effective_response_tokens is not None and trajectory.effective_response_tokens != len(
            response_ids
        ):
            raise RuntimeError(
                "rollout and training disagree on expanded response length: "
                f"{trajectory.effective_response_tokens} != {len(response_ids)}"
            )
        observation_visual_tokens = sum(prepared.visual_token_counts[source_count:])
        if (
            trajectory.metrics.visual_tokens is not None
            and trajectory.observation_images
            and trajectory.metrics.visual_tokens != observation_visual_tokens
        ):
            raise RuntimeError(
                "rollout and training disagree on observation visual token count: "
                f"{trajectory.metrics.visual_tokens} != {observation_visual_tokens}"
            )

        if len(prompt_ids) > self.max_prompt_tokens:
            raise AgentTrajectoryRejected(
                f"expanded agent prompt has {len(prompt_ids)} tokens but max_prompt_tokens={self.max_prompt_tokens}"
            )

        input_ids = torch.tensor([*prompt_ids, *response_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        if self.max_model_len is not None and input_ids.numel() > self.max_model_len:
            raise AgentTrajectoryRejected(
                f"expanded agent input has {input_ids.numel()} tokens but max_model_len={self.max_model_len}"
            )

        position_ids = build_multimodal_position_ids(
            self.processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=(
                {
                    "image_grid_thw": torch.tensor(
                        prepared.multi_modal_layout["image_grid_thw"],
                        dtype=torch.long,
                    )
                }
                if prepared.multi_modal_layout.get("kind") == "qwen"
                else {}
            ),
        )

        unpadded_prompt_len = len(prompt_ids)
        absolute_image_spans = [
            *prompt_visual_spans,
            *((unpadded_prompt_len + start, unpadded_prompt_len + end) for start, end in response_visual_spans),
        ]
        visual_token_mask = [0] * input_ids.numel()
        for start, end in absolute_image_spans:
            visual_token_mask[start:end] = [1] * (end - start)

        return MaterializedAgentTrajectory(
            prompt_ids=tuple(prompt_ids),
            response_ids=tuple(response_ids),
            response_mask=tuple(response_mask),
            position_ids=position_ids,
            visual_token_mask=tuple(visual_token_mask),
            multi_modal_data={"images": list(prepared.images)},
            multi_modal_layout=prepared.multi_modal_layout,
            visual_token_counts=prepared.visual_token_counts,
            action_spans=tuple(
                (unpadded_prompt_len + start, unpadded_prompt_len + end)
                for start, end in _contiguous_spans(response_mask, 1)
            ),
            observation_spans=tuple(
                (unpadded_prompt_len + start, unpadded_prompt_len + end)
                for start, end in _contiguous_spans(response_mask, 0)
            ),
            image_spans=tuple(absolute_image_spans),
            source_image_count=source_count,
            prompt_token_budget=self.max_prompt_tokens,
            response_token_budget=response_token_budget,
            trajectory=trajectory,
        )


def _pad_1d(values: Sequence[int], width: int, pad_value: int, *, left: bool) -> torch.Tensor:
    if len(values) > width:
        raise ValueError(f"sequence length {len(values)} exceeds configured padded width {width}")
    padding = [pad_value] * (width - len(values))
    return torch.tensor([*padding, *values] if left else [*values, *padding], dtype=torch.long)


def _span_to_padded_input(
    span: tuple[int, int],
    *,
    unpadded_prompt_len: int,
    padded_prompt_width: int,
) -> tuple[int, int]:
    start, end = span
    if not (0 <= start < end):
        raise ValueError(f"invalid unpadded input span {span}")
    if end <= unpadded_prompt_len:
        offset = padded_prompt_width - unpadded_prompt_len
    elif start >= unpadded_prompt_len:
        offset = padded_prompt_width - unpadded_prompt_len
    else:
        raise ValueError(f"token span crosses the prompt/response boundary: {span}")
    return start + offset, end + offset


def collate_materialized_agent_trajectories(
    samples: Sequence[MaterializedAgentTrajectory],
    *,
    pad_token_id: int,
    padded_prompt_width: int,
    padded_response_width: int,
    uids: Sequence[str],
    meta_info: Optional[dict[str, Any]] = None,
) -> DataProto:
    """Pad materialized samples using EasyR1's prompt-left/response-right layout."""

    samples = list(samples)
    if not samples:
        raise ValueError("cannot collate an empty agent trajectory batch")
    if padded_prompt_width <= 0 or padded_response_width <= 0:
        raise ValueError("agent prompt and response padded widths must be positive")
    if any(sample.prompt_token_budget != padded_prompt_width for sample in samples):
        raise ValueError("materializer and collator disagree on the global prompt padded width")
    if any(sample.response_token_budget != padded_response_width for sample in samples):
        raise ValueError("agent loop and collator disagree on the global response padded width")
    if len(uids) != len(samples):
        raise ValueError("uids must align one-to-one with materialized trajectories")

    position_rank = samples[0].position_ids.ndim
    if position_rank not in (1, 2):
        raise ValueError("agent position IDs must be 1D or channel-first 2D")
    if any(sample.position_ids.ndim != position_rank for sample in samples):
        raise ValueError("one agent batch cannot mix M-RoPE and scalar position ID layouts")
    if position_rank == 2:
        position_channels = samples[0].position_ids.size(0)
        if any(sample.position_ids.size(0) != position_channels for sample in samples):
            raise ValueError("M-RoPE position channel counts differ within one agent batch")

    prompts = []
    responses = []
    input_ids = []
    attention_masks = []
    response_masks = []
    position_ids = []
    visual_token_masks = []
    boundaries = []
    for sample in samples:
        unpadded_prompt_len = len(sample.prompt_ids)
        unpadded_response_len = len(sample.response_ids)
        padded_prompt = _pad_1d(
            sample.prompt_ids,
            padded_prompt_width,
            pad_token_id,
            left=True,
        )
        padded_response = _pad_1d(
            sample.response_ids,
            padded_response_width,
            pad_token_id,
            left=False,
        )
        prompts.append(padded_prompt)
        responses.append(padded_response)
        input_ids.append(torch.cat((padded_prompt, padded_response)))

        prompt_attention = _pad_1d(
            [1] * unpadded_prompt_len,
            padded_prompt_width,
            0,
            left=True,
        )
        response_attention = _pad_1d(
            [1] * unpadded_response_len,
            padded_response_width,
            0,
            left=False,
        )
        attention_masks.append(torch.cat((prompt_attention, response_attention)))
        response_masks.append(
            _pad_1d(
                sample.response_mask,
                padded_response_width,
                0,
                left=False,
            )
        )
        padded_visual_token_mask = torch.cat(
            (
                _pad_1d(
                    sample.visual_token_mask[:unpadded_prompt_len],
                    padded_prompt_width,
                    0,
                    left=True,
                ),
                _pad_1d(
                    sample.visual_token_mask[unpadded_prompt_len:],
                    padded_response_width,
                    0,
                    left=False,
                ),
            )
        )
        visual_token_masks.append(padded_visual_token_mask)

        sample_prompt_positions = sample.position_ids[..., :unpadded_prompt_len]
        sample_response_positions = sample.position_ids[..., unpadded_prompt_len:]
        if position_rank == 1:
            padded_positions = torch.cat(
                (
                    torch.nn.functional.pad(
                        sample_prompt_positions,
                        (padded_prompt_width - unpadded_prompt_len, 0),
                        value=0,
                    ),
                    torch.nn.functional.pad(
                        sample_response_positions,
                        (0, padded_response_width - unpadded_response_len),
                        value=0,
                    ),
                )
            )
        else:
            padded_positions = torch.cat(
                (
                    torch.nn.functional.pad(
                        sample_prompt_positions,
                        (padded_prompt_width - unpadded_prompt_len, 0),
                        value=0,
                    ),
                    torch.nn.functional.pad(
                        sample_response_positions,
                        (0, padded_response_width - unpadded_response_len),
                        value=0,
                    ),
                ),
                dim=-1,
            )
        position_ids.append(padded_positions)
        padded_action_spans = tuple(
            _span_to_padded_input(
                span,
                unpadded_prompt_len=unpadded_prompt_len,
                padded_prompt_width=padded_prompt_width,
            )
            for span in sample.action_spans
        )
        padded_observation_spans = tuple(
            _span_to_padded_input(
                span,
                unpadded_prompt_len=unpadded_prompt_len,
                padded_prompt_width=padded_prompt_width,
            )
            for span in sample.observation_spans
        )
        padded_image_spans = tuple(
            _span_to_padded_input(
                span,
                unpadded_prompt_len=unpadded_prompt_len,
                padded_prompt_width=padded_prompt_width,
            )
            for span in sample.image_spans
        )
        if any(not bool(torch.all(padded_visual_token_mask[start:end] == 1)) for start, end in padded_image_spans):
            raise RuntimeError("collated image spans do not align with visual_token_mask")
        boundaries.append(
            {
                "coordinate_space": "padded_input_ids",
                "action_spans": padded_action_spans,
                "observation_spans": padded_observation_spans,
                "image_spans": padded_image_spans,
                "source_image_count": sample.source_image_count,
            }
        )

    tensors = {
        "prompts": torch.stack(prompts),
        "responses": torch.stack(responses),
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "response_mask": torch.stack(response_masks),
        "position_ids": torch.stack(position_ids),
        "visual_token_mask": torch.stack(visual_token_masks),
    }
    non_tensors = {
        "multi_modal_data": np.array([sample.multi_modal_data for sample in samples], dtype=object),
        "multi_modal_layout": np.array([sample.multi_modal_layout for sample in samples], dtype=object),
        "agent_token_boundaries": np.array(boundaries, dtype=object),
        "uid": np.array(list(uids), dtype=object),
        # Group UIDs intentionally repeat for GRPO. Worker multimodal caches
        # need a per-trajectory identity because different rollouts may commit
        # different crop observations.
        "agent_trajectory_id": np.array(
            [f"{uid}:{index}" for index, uid in enumerate(uids)],
            dtype=object,
        ),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)
