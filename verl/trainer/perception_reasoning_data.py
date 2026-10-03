# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from ..models.transformers.position_ids import build_multimodal_position_ids
from ..protocol import DataProto
from ..utils.dataset import ProcessedImageInput, process_image
from ..utils.perturbations.pixel import (
    compute_noise_schedule,
    pixelate_image,
    random_patch_blackening,
    vp_diffusion_noise,
)
from ..utils.perturbations.pixel_values import PIXEL_VALUES_NOISE_KEY
from .config import MODEL_LEVEL_VISUAL_CORRUPTIONS, AlgorithmConfig
from .visual_sensitivity import is_full_vocab_sensitivity_metric


REGION_PAT = re.compile(r"<region\b(?P<attrs>[^>]*)>(?P<body>.*?)</region>", re.DOTALL)
REGION_MARKER_PAT = re.compile(r"<region\b", re.IGNORECASE)
THINK_PAT = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)
ATTR_PAT = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


@dataclass
class ParsedResponseRegion:
    region_id: int
    name: str
    normalized_name: str
    image_idx: int
    boxes: list[list[int]]


def normalize_region_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _is_valid_bbox(box: list[int]) -> bool:
    x1, y1, x2, y2 = box
    return all(0 <= coord <= 1000 for coord in box) and x1 < x2 and y1 < y2


def parse_bbox_list_literal(text: str, *, exact: bool = False) -> list[list[int]]:
    candidate = text.strip()
    if not candidate:
        return []

    decoder = json.JSONDecoder()
    if exact:
        try:
            parsed, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            return []
        if candidate[end:].strip():
            return []
    else:
        start = 0
        parsed = None
        while True:
            start = candidate.find("[", start)
            if start == -1:
                return []
            try:
                parsed, _ = decoder.raw_decode(candidate[start:])
                break
            except json.JSONDecodeError:
                start += 1
        if parsed is None:
            return []

    if not isinstance(parsed, list):
        return []

    validated_boxes: list[list[int]] = []
    for item in parsed:
        if not isinstance(item, list) or len(item) != 4:
            return []
        coords: list[int] = []
        for coord in item:
            if isinstance(coord, bool):
                return []
            if isinstance(coord, int):
                coords.append(coord)
                continue
            if isinstance(coord, float) and coord.is_integer():
                coords.append(int(coord))
                continue
            return []
        validated_boxes.append(coords)
    return validated_boxes


def _extract_json_candidates(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    start = 0
    while True:
        next_starts = [idx for token in ("{", "[") if (idx := text.find(token, start)) != -1]
        if not next_starts:
            return candidates
        start = min(next_starts)
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start += 1
            continue
        candidates.append(parsed)
        start += max(end, 1)


def _coerce_grit_image_idx(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _coerce_grit_bbox_list(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        return []
    boxes: list[list[int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 4:
            continue
        coords: list[int] = []
        valid_box = True
        for coord in item:
            if isinstance(coord, bool):
                valid_box = False
                break
            if isinstance(coord, int):
                coords.append(coord)
                continue
            if isinstance(coord, float) and math.isfinite(coord) and coord.is_integer():
                coords.append(int(coord))
                continue
            valid_box = False
            break
        if not valid_box:
            continue
        if not _is_valid_bbox(coords):
            continue
        boxes.append(coords)
    return boxes


def _coerce_grit_candidate(candidate: Any, *, num_images: int | None = None) -> list[ParsedResponseRegion]:
    candidate_items = [candidate] if isinstance(candidate, dict) else candidate
    if not isinstance(candidate_items, list):
        return []
    parsed_groups: list[ParsedResponseRegion] = []
    next_region_id = 0
    for item in candidate_items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        image_idx = _coerce_grit_image_idx(item.get("image_idx", 0))
        if image_idx is None:
            continue
        if num_images is not None and image_idx >= num_images:
            continue
        boxes = _coerce_grit_bbox_list(item.get("bbox_list"))
        if not boxes:
            continue
        name = label.strip()
        parsed_groups.append(
            ParsedResponseRegion(
                region_id=next_region_id,
                name=name,
                normalized_name=normalize_region_name(name),
                image_idx=image_idx,
                boxes=boxes,
            )
        )
        next_region_id += 1
    return parsed_groups


def parse_json_grounding_regions(text: str, num_images: int | None = None) -> list[ParsedResponseRegion]:
    groups: list[ParsedResponseRegion] = []
    for candidate in _extract_json_candidates(text):
        groups.extend(_coerce_grit_candidate(candidate, num_images=num_images))
    return groups


def _parse_tag_attrs(attr_text: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in ATTR_PAT.finditer(attr_text)}


def _parse_xml_response_regions(response: str, num_images: int | None = None) -> list[ParsedResponseRegion]:
    regions_by_key: dict[tuple[int, str], ParsedResponseRegion] = {}
    ordered_keys: list[tuple[int, str]] = []
    seen_ids: set[int] = set()

    for match in REGION_PAT.finditer(response):
        attrs = _parse_tag_attrs(match.group("attrs"))
        if not {"name", "image_idx", "id"} <= attrs.keys():
            continue

        try:
            region_id = int(attrs["id"])
            image_idx = int(attrs["image_idx"])
        except ValueError:
            continue
        if region_id in seen_ids or region_id < 0 or image_idx < 0:
            continue
        if num_images is not None and image_idx >= num_images:
            continue

        boxes = [box for box in parse_bbox_list_literal(match.group("body"), exact=True) if _is_valid_bbox(box)]
        if not boxes:
            continue

        name = attrs["name"].strip()
        if not name:
            continue
        normalized_name = normalize_region_name(name)
        region_key = (image_idx, normalized_name)
        seen_ids.add(region_id)

        if region_key in regions_by_key:
            regions_by_key[region_key].boxes.extend(boxes)
            continue

        regions_by_key[region_key] = ParsedResponseRegion(
            region_id=region_id,
            name=name,
            normalized_name=normalized_name,
            image_idx=image_idx,
            boxes=boxes,
        )
        ordered_keys.append(region_key)

    return [regions_by_key[key] for key in ordered_keys]


def _extract_think_text(response: str) -> str | None:
    match = THINK_PAT.search(response)
    return match.group("think") if match is not None else None


def parse_response_regions(response: str, num_images: int | None = None) -> list[ParsedResponseRegion]:
    regions = _parse_xml_response_regions(response, num_images=num_images)
    if regions:
        return regions
    if REGION_MARKER_PAT.search(response):
        return []
    think_text = _extract_think_text(response)
    if think_text is None:
        return []
    return parse_json_grounding_regions(think_text, num_images=num_images)


def resolve_visual_token_ids(tokenizer: Any, visual_token: str | None = "auto") -> set[int]:
    token_ids: set[int] = set()
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    def add_token_string(token: str) -> None:
        if not callable(convert_tokens_to_ids):
            return
        try:
            token_id = convert_tokens_to_ids(token)
        except Exception:
            return
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            token_ids.add(token_id)

    if visual_token is not None and visual_token != "auto":
        add_token_string(visual_token)
        return token_ids

    for attr_name in ("image_token_id", "video_token_id", "img_context_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if isinstance(token_id, int) and token_id >= 0:
            token_ids.add(token_id)

    for token in ("<|image_pad|>", "<|video_pad|>", "<IMG_CONTEXT>"):
        add_token_string(token)

    return token_ids


def uses_hidden_state_visual_sensitivity(config: AlgorithmConfig) -> bool:
    return config.visual_sensitivity_metric == "hidden_state_similarity"


def uses_model_level_visual_corruption(config: AlgorithmConfig) -> bool:
    return config.corrupt_image in MODEL_LEVEL_VISUAL_CORRUPTIONS


def needs_hidden_state_visual_sensitivity(config: AlgorithmConfig) -> bool:
    return uses_hidden_state_visual_sensitivity(config) and (
        config.top_perception_quantile < 1.0
        or config.advantage_scaling_method is not None
        or config.response_advantage_scaling_method is not None
        or (config.tor_use_token_weighting and config.top_perception_quantile < 1.0)
    )


def needs_decremental_auxiliary(config: AlgorithmConfig) -> bool:
    return config.corrupt_image is not None and (
        config.visual_sensitivity_loss_coef != 0.0
        or config.decremental_entropy_coef != 0.0
        or (
            not uses_hidden_state_visual_sensitivity(config)
            and (
                config.top_perception_quantile < 1.0
                or config.advantage_scaling_method is not None
                or config.response_advantage_scaling_method is not None
                or config.tor_use_token_weighting
            )
        )
    )


def uses_full_vocab_visual_sensitivity(config: AlgorithmConfig) -> bool:
    return is_full_vocab_sensitivity_metric(config.visual_sensitivity_metric)


def needs_full_vocab_visual_sensitivity(config: AlgorithmConfig) -> bool:
    return uses_full_vocab_visual_sensitivity(config) and (
        config.top_perception_quantile < 1.0
        or config.advantage_scaling_method is not None
        or config.response_advantage_scaling_method is not None
    )


def uses_incremental_dvrp_mode(config: AlgorithmConfig) -> bool:
    return (
        config.incremental_image_transform is not None
        or config.incremental_image_kwargs is not None
        or config.visual_robustness_loss_coef != 0.0
        or config.incremental_entropy_coef != 0.0
    )


def needs_incremental_auxiliary(config: AlgorithmConfig) -> bool:
    return config.incremental_image_transform is not None and (
        config.visual_robustness_loss_coef != 0.0 or config.incremental_entropy_coef != 0.0
    )


def needs_auxiliary_log_probs(config: AlgorithmConfig) -> bool:
    needs_decremental_log_probs = config.corrupt_image is not None and (
        config.visual_sensitivity_loss_coef != 0.0
        or config.decremental_entropy_coef != 0.0
        or (
            not uses_hidden_state_visual_sensitivity(config)
            and (
                config.advantage_scaling_method is not None
                or config.response_advantage_scaling_method is not None
                or (config.top_perception_quantile < 1.0 and not uses_full_vocab_visual_sensitivity(config))
            )
        )
        or (
            config.tor_use_token_weighting
            and config.top_perception_quantile < 1.0
            and not uses_full_vocab_visual_sensitivity(config)
            and not uses_hidden_state_visual_sensitivity(config)
        )
    )
    return needs_decremental_log_probs or needs_incremental_auxiliary(config)


def needs_media_corruption_builder(config: AlgorithmConfig) -> bool:
    needs_decremental_media = not uses_model_level_visual_corruption(config) and (
        needs_auxiliary_log_probs(config) or needs_decremental_auxiliary(config)
    )
    return needs_decremental_media or needs_incremental_auxiliary(config)


def needs_decremental_entropy(config: AlgorithmConfig) -> bool:
    return needs_decremental_auxiliary(config) and config.decremental_entropy_coef != 0.0


def needs_incremental_entropy(config: AlgorithmConfig) -> bool:
    return needs_incremental_auxiliary(config) and config.incremental_entropy_coef != 0.0


def needs_dvrp_auxiliary_views(config: AlgorithmConfig) -> bool:
    return uses_incremental_dvrp_mode(config) and (
        needs_decremental_auxiliary(config) or needs_incremental_auxiliary(config)
    )


def build_perception_reasoning_loss_config(
    config: AlgorithmConfig,
    decremental_stats: dict[str, float] | None = None,
    incremental_stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    loss_config: dict[str, Any] = {
        "log_entropy": config.log_entropy,
        "top_entropy_quantile": config.top_entropy_quantile,
        "corrupt_image": config.corrupt_image,
        "corrupt_image_position": config.corrupt_image_position,
        "visual_sensitivity_loss_coef": config.visual_sensitivity_loss_coef,
        "decremental_entropy_coef": config.decremental_entropy_coef,
        "invariant_entropy_coef": config.invariant_entropy_coef,
        "entropy_loss_type": config.entropy_loss_type,
        "top_perception_quantile": config.top_perception_quantile,
        "visual_sensitivity_reference": config.visual_sensitivity_reference,
        "visual_sensitivity_metric": config.visual_sensitivity_metric,
        "visual_sensitivity_boxcox_alpha": config.visual_sensitivity_boxcox_alpha,
        "visual_sensitivity_log_metrics": config.visual_sensitivity_log_metrics,
        "visual_sensitivity_hidden_metric": config.visual_sensitivity_hidden_metric,
        "visual_token": config.visual_token,
        "visual_sensitivity_jsd_weight": config.visual_sensitivity_jsd_weight,
        "visual_sensitivity_entropy_gate": config.visual_sensitivity_entropy_gate,
        "entropy_thr_granularity": config.entropy_thr_granularity,
        "entropy_top_p": config.entropy_top_p,
        "advantage_scaling_method": config.advantage_scaling_method,
        "response_advantage_scaling_method": config.response_advantage_scaling_method,
        "vppo_response_scaling_min": config.vppo_response_scaling_min,
        "cgpo_response_scaling_coef": config.cgpo_response_scaling_coef,
        "pgpo_token_scaling_threshold": config.pgpo_token_scaling_threshold,
        "pgpo_token_scaling_boost": config.pgpo_token_scaling_boost,
        "pgpo_threshold_mode": config.pgpo_threshold_mode,
        "pgpo_low_weight_floor": config.pgpo_low_weight_floor,
        "pgpo_mass_normalization": config.pgpo_mass_normalization,
        "advantage_scaling_schedule": config.advantage_scaling_schedule,
        "pepo_gate_alpha": config.pepo_gate_alpha,
        "pepo_gate_temperature": config.pepo_gate_temperature,
        "tor_use_token_weighting": config.tor_use_token_weighting,
        "tor_rsn_weight": config.tor_rsn_weight,
        "tor_prcp_weight": config.tor_prcp_weight,
        "normalize_pg_loss_by_selected_tokens": config.normalize_pg_loss_by_selected_tokens,
        "perception_thr_granularity": config.perception_thr_granularity,
        "include_region_tokens_in_perception_mask": config.include_region_tokens_in_perception_mask,
        "incremental_image_transform": config.incremental_image_transform,
        "visual_robustness_loss_coef": config.visual_robustness_loss_coef,
        "incremental_entropy_coef": config.incremental_entropy_coef,
    }
    for stats in (decremental_stats, incremental_stats):
        if stats:
            loss_config.update(stats)
    return loss_config


def _stable_prompt_seed(corruption_group_key: str, salt: str = "corrupt_prompt_v1") -> int:
    payload = f"{salt}:{corruption_group_key}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _sample_bytes_for_digest(data: bytes) -> bytes:
    if len(data) <= 8192:
        return data
    return data[:4096] + data[-4096:]


def _digest_bytes(data: bytes) -> str:
    return hashlib.blake2b(_sample_bytes_for_digest(data), digest_size=16).hexdigest()


def _canonicalize_media_value(value: Any) -> Any:
    if isinstance(value, ProcessedImageInput):
        value = value.image
    if isinstance(value, Image.Image):
        image_bytes = value.convert("RGB").tobytes()
        return {
            "__kind__": "pil_image",
            "mode": value.mode,
            "size": list(value.size),
            "digest": _digest_bytes(image_bytes),
        }
    if isinstance(value, np.ndarray):
        return {
            "__kind__": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "digest": _digest_bytes(value.tobytes()),
        }
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__kind__": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "digest": _digest_bytes(tensor.numpy().tobytes()),
        }
    if isinstance(value, (bytes, bytearray)):
        return {"__kind__": "bytes", "digest": _digest_bytes(bytes(value))}
    return value


def _canonicalize_prompt_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_prompt_payload(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_canonicalize_prompt_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize_prompt_payload(item) for item in value]
    if isinstance(value, (ProcessedImageInput, Image.Image, np.ndarray, torch.Tensor, bytes, bytearray)):
        return _canonicalize_media_value(value)
    return value


def _compute_corruption_group_key(
    raw_prompt: list[dict[str, Any]], multi_modal_data: dict[str, Any] | None = None
) -> str:
    canonical_prompt = _canonicalize_prompt_payload(
        {
            "raw_prompt": raw_prompt,
            "multi_modal_data": multi_modal_data,
        }
    )
    serialized = json.dumps(canonical_prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _group_regions_by_image(
    regions: list[ParsedResponseRegion],
    num_images: int,
) -> list[list[list[list[int]]]]:
    grouped: list[list[list[list[int]]]] = [[] for _ in range(num_images)]
    for region in regions:
        grouped[region.image_idx].append(region.boxes)
    return grouped


def build_region_token_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> torch.Tensor:
    region_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    ent_pat = re.compile(r"<region\b[^>]*?>.*?</region>", re.DOTALL)
    is_fast_tokenizer = getattr(tokenizer, "is_fast", False)

    for row_idx in range(responses.size(0)):
        valid_length = int(response_mask[row_idx].sum().item())
        if valid_length <= 0:
            continue

        valid_ids = responses[row_idx, :valid_length].tolist()
        completion_text = tokenizer.decode(
            valid_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        matches = list(ent_pat.finditer(completion_text))
        if not matches:
            continue

        spans = [(match.start(), match.end()) for match in matches]
        enc = tokenizer(completion_text, add_special_tokens=False)
        if "input_ids" not in enc or enc["input_ids"] != valid_ids:
            continue

        if is_fast_tokenizer:
            enc = tokenizer(completion_text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc.get("offset_mapping") or []
        else:
            offsets = []
            cursor = 0
            for token_id in valid_ids:
                token_text = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                end = cursor + len(token_text)
                offsets.append((cursor, end))
                cursor = end

        if not offsets:
            continue

        for token_idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            for span_start, span_end in spans:
                if start < span_end and end > span_start:
                    region_mask[row_idx, token_idx] = True
                    break

    return region_mask & response_mask.to(torch.bool)


def to_abs_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    abs_x1 = int(round(x1 / 1000.0 * width))
    abs_y1 = int(round(y1 / 1000.0 * height))
    abs_x2 = int(round(x2 / 1000.0 * width))
    abs_y2 = int(round(y2 / 1000.0 * height))
    abs_x1 = max(0, min(width, abs_x1))
    abs_x2 = max(0, min(width, abs_x2))
    abs_y1 = max(0, min(height, abs_y1))
    abs_y2 = max(0, min(height, abs_y2))
    return abs_x1, abs_y1, abs_x2, abs_y2


def get_fill_val(arr: np.ndarray, fill_type: str, all_bboxes: list[list[list[int]]]):
    if fill_type == "black":
        return 0
    if fill_type == "mean":
        return arr.mean(axis=(0, 1)).astype(arr.dtype)
    if fill_type == "keep_mean":
        mask = np.zeros(arr.shape[:2], dtype=bool)
        for region_boxes in all_bboxes:
            for x1, y1, x2, y2 in region_boxes:
                abs_x1, abs_y1, abs_x2, abs_y2 = to_abs_bbox(x1, y1, x2, y2, arr.shape[1], arr.shape[0])
                if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                    continue
                mask[abs_y1:abs_y2, abs_x1:abs_x2] = True
        if np.any(mask):
            return np.round(arr[mask].mean(axis=0)).astype(arr.dtype)
        return 0
    if fill_type == "local_mean":
        return "local_mean"
    raise ValueError(f"Unsupported fill_type={fill_type!r}")


def cgpo_flat(
    images: list[Image.Image],
    response: str,
    fill_type: str = "mean",
) -> list[Image.Image]:
    regions = parse_response_regions(response, num_images=len(images))
    regions_by_image = _group_regions_by_image(regions, len(images))
    transformed_images: list[Image.Image] = []

    for image, image_regions in zip(images, regions_by_image):
        if not image_regions:
            transformed_images.append(image.copy())
            continue

        arr = np.array(image)
        height, width = arr.shape[:2]
        fill_val = get_fill_val(arr, fill_type, image_regions)
        if isinstance(fill_val, str) and fill_val == "local_mean":
            orig_arr = arr.copy()

        for region_boxes in image_regions:
            for x1, y1, x2, y2 in region_boxes:
                abs_x1, abs_y1, abs_x2, abs_y2 = to_abs_bbox(x1, y1, x2, y2, width, height)
                if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                    continue
                if isinstance(fill_val, str) and fill_val == "local_mean":
                    arr[abs_y1:abs_y2, abs_x1:abs_x2] = np.round(
                        orig_arr[abs_y1:abs_y2, abs_x1:abs_x2].mean(axis=(0, 1))
                    ).astype(arr.dtype)
                else:
                    arr[abs_y1:abs_y2, abs_x1:abs_x2] = fill_val

        transformed_images.append(Image.fromarray(arr))

    return transformed_images


def cgpo_hierarchical(
    images: list[Image.Image],
    response: str,
    fill_type: str = "mean",
) -> list[Image.Image]:
    regions = parse_response_regions(response, num_images=len(images))
    regions_by_image = _group_regions_by_image(regions, len(images))
    transformed_images: list[Image.Image] = []

    for image, image_regions in zip(images, regions_by_image):
        if not image_regions:
            transformed_images.append(image.copy())
            continue

        orig_arr = np.array(image)
        arr = orig_arr.copy()
        height, width = arr.shape[:2]
        nodes = []
        for region_boxes in image_regions:
            for x1, y1, x2, y2 in region_boxes:
                abs_x1, abs_y1, abs_x2, abs_y2 = to_abs_bbox(x1, y1, x2, y2, width, height)
                if abs_x2 <= abs_x1 or abs_y2 <= abs_y1:
                    continue
                area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)
                nodes.append(
                    {
                        "ax1": abs_x1,
                        "ay1": abs_y1,
                        "ax2": abs_x2,
                        "ay2": abs_y2,
                        "area": area,
                        "parent": -1,
                        "children": [],
                        "depth": -1,
                        "max_depth": -1,
                        "alpha": 1.0,
                    }
                )

        def _contains(parent, child):
            return (
                parent["ax1"] <= child["ax1"]
                and parent["ay1"] <= child["ay1"]
                and parent["ax2"] >= child["ax2"]
                and parent["ay2"] >= child["ay2"]
                and parent["area"] > child["area"]
            )

        for child_idx, child in enumerate(nodes):
            best_parent = -1
            best_area = None
            for parent_idx, parent in enumerate(nodes):
                if child_idx == parent_idx:
                    continue
                if _contains(parent, child) and (best_area is None or parent["area"] < best_area):
                    best_parent = parent_idx
                    best_area = parent["area"]
            child["parent"] = best_parent
            if best_parent != -1:
                nodes[best_parent]["children"].append(child_idx)

        def _calc_depth(node_idx: int) -> int:
            parent_idx = nodes[node_idx]["parent"]
            if parent_idx == -1:
                return 0
            if nodes[parent_idx]["depth"] == -1:
                nodes[parent_idx]["depth"] = _calc_depth(parent_idx)
            return nodes[parent_idx]["depth"] + 1

        for node_idx in range(len(nodes)):
            if nodes[node_idx]["depth"] == -1:
                nodes[node_idx]["depth"] = _calc_depth(node_idx)

        def _dfs_max(node_idx: int) -> int:
            if nodes[node_idx]["max_depth"] != -1:
                return nodes[node_idx]["max_depth"]
            max_depth = nodes[node_idx]["depth"]
            for child_idx in nodes[node_idx]["children"]:
                max_depth = max(max_depth, _dfs_max(child_idx))
            nodes[node_idx]["max_depth"] = max_depth
            return max_depth

        for node_idx, node in enumerate(nodes):
            if node["parent"] == -1:
                _dfs_max(node_idx)

        for node in nodes:
            node["alpha"] = (node["depth"] + 1) / (node["max_depth"] + 1)

        fill_val = get_fill_val(orig_arr, fill_type, image_regions)
        use_local_mean = isinstance(fill_val, str) and fill_val == "local_mean"
        nodes.sort(key=lambda node: node["depth"])
        for node in nodes:
            abs_x1, abs_y1, abs_x2, abs_y2 = node["ax1"], node["ay1"], node["ax2"], node["ay2"]
            patch_orig = orig_arr[abs_y1:abs_y2, abs_x1:abs_x2]
            if patch_orig.size == 0:
                continue

            if use_local_mean:
                local_fill_val = np.round(patch_orig.mean(axis=(0, 1))).astype(orig_arr.dtype)
            else:
                local_fill_val = fill_val

            if node["alpha"] == 1.0:
                arr[abs_y1:abs_y2, abs_x1:abs_x2] = local_fill_val
            else:
                mixed = (1.0 - node["alpha"]) * patch_orig.astype(np.float32) + node["alpha"] * np.asarray(
                    local_fill_val, dtype=np.float32
                )
                arr[abs_y1:abs_y2, abs_x1:abs_x2] = np.asarray(np.round(mixed), dtype=orig_arr.dtype)

        transformed_images.append(Image.fromarray(arr))

    return transformed_images


@dataclass
class CachedAuxiliaryMedia:
    multi_modal_data: dict[str, Any]
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class AuxiliaryBatchBuildResult:
    batch: DataProto
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class PerceptionReasoningCorruptionBuilder:
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    processor: ProcessorMixin | None
    image_patch_size: int
    min_pixels: int | None = None
    max_pixels: int | None = None
    video_fps: float = 2.0
    _prompt_cache: dict[tuple[str, str], CachedAuxiliaryMedia] = field(default_factory=dict)
    _prompt_cache_step: int | None = field(default=None, init=False)

    def build_batch(self, batch: DataProto, config: AlgorithmConfig, global_step: int) -> AuxiliaryBatchBuildResult:
        transform_kwargs = dict(config.corrupt_image_kwargs or {})
        return self._build_auxiliary_batch(
            batch=batch,
            transform_name=config.corrupt_image,
            transform_kwargs=transform_kwargs or None,
            transform_position=config.corrupt_image_position,
            global_step=global_step,
        )

    def build_dvrp_batches(
        self,
        batch: DataProto,
        config: AlgorithmConfig,
        global_step: int,
        total_training_steps: int,
    ) -> tuple[AuxiliaryBatchBuildResult, AuxiliaryBatchBuildResult]:
        decremental_result = self._build_auxiliary_batch(
            batch=batch,
            transform_name=config.corrupt_image,
            transform_kwargs=config.corrupt_image_kwargs,
            transform_position=config.corrupt_image_position,
            global_step=global_step,
        )
        incremental_result = self.build_incremental_batch(
            batch=batch,
            config=config,
            global_step=global_step,
            total_training_steps=total_training_steps,
        )
        return decremental_result, incremental_result

    def build_incremental_batch(
        self,
        batch: DataProto,
        config: AlgorithmConfig,
        global_step: int,
        total_training_steps: int,
    ) -> AuxiliaryBatchBuildResult:
        incremental_kwargs = dict(config.incremental_image_kwargs or {})
        incremental_kwargs.setdefault("noise_t_init", config.noise_t_init)
        incremental_kwargs.setdefault("noise_gamma", config.noise_gamma)
        incremental_kwargs.setdefault("noise_t_max", config.noise_t_max)
        return self._build_auxiliary_batch(
            batch=batch,
            transform_name=config.incremental_image_transform,
            transform_kwargs=incremental_kwargs,
            transform_position=config.corrupt_image_position,
            global_step=global_step,
            total_training_steps=total_training_steps,
        )

    def _build_auxiliary_batch(
        self,
        batch: DataProto,
        transform_name: str | None,
        transform_kwargs: dict[str, Any] | None,
        transform_position: str,
        global_step: int,
        total_training_steps: int | None = None,
    ) -> AuxiliaryBatchBuildResult:
        if transform_name is None:
            raise ValueError("Auxiliary batch reconstruction requires a configured image transform.")
        if "raw_prompt" not in batch.non_tensor_batch:
            raise KeyError("Corrupted batch reconstruction requires 'raw_prompt' in non_tensor_batch.")
        if self._prompt_cache_step != global_step:
            self._prompt_cache.clear()
            self._prompt_cache_step = global_step

        batch_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        if "response_mask" in batch.batch.keys():
            batch_keys.append("response_mask")
        non_tensor_keys = [key for key in ["uid", "raw_prompt"] if key in batch.non_tensor_batch]
        corrupted_batch = batch.select(
            batch_keys=batch_keys,
            non_tensor_batch_keys=non_tensor_keys,
            meta_info_keys=None,
            deepcopy=True,
        )

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        multi_modal_cache_id_list = []
        multi_modal_data_list = []
        collected_stats: dict[str, list[float]] = {}
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        batch_multi_modal_data = batch.non_tensor_batch.get("multi_modal_data")
        batch_uids = batch.non_tensor_batch.get("uid")

        for sample_idx in range(len(batch)):
            multi_modal_data = None if batch_multi_modal_data is None else deepcopy(batch_multi_modal_data[sample_idx])
            sample_uid = None if batch_uids is None else batch_uids[sample_idx]
            sample_tensors, sample_multi_modal_data, sample_cache_id, sample_stats = self._build_sample(
                raw_prompt=raw_prompts[sample_idx],
                multi_modal_data=multi_modal_data,
                full_input_ids=batch.batch["input_ids"][sample_idx],
                attention_mask=batch.batch["attention_mask"][sample_idx],
                position_ids=batch.batch["position_ids"][sample_idx],
                responses=batch.batch["responses"][sample_idx],
                response_mask=batch.batch.get("response_mask", None),
                sample_idx=sample_idx,
                transform_name=transform_name,
                transform_kwargs=transform_kwargs,
                transform_position=transform_position,
                global_step=global_step,
                total_training_steps=total_training_steps,
                sample_uid=sample_uid,
            )
            input_ids_list.append(sample_tensors["input_ids"])
            attention_mask_list.append(sample_tensors["attention_mask"])
            position_ids_list.append(sample_tensors["position_ids"])
            multi_modal_cache_id_list.append(sample_cache_id)
            multi_modal_data_list.append(sample_multi_modal_data)
            for key, value in sample_stats.items():
                collected_stats.setdefault(key, []).append(value)

        corrupted_batch.batch["input_ids"] = torch.stack(input_ids_list, dim=0)
        corrupted_batch.batch["attention_mask"] = torch.stack(attention_mask_list, dim=0)
        corrupted_batch.batch["position_ids"] = torch.stack(position_ids_list, dim=0)
        corrupted_batch.non_tensor_batch["multi_modal_cache_id"] = np.array(multi_modal_cache_id_list, dtype=object)
        corrupted_batch.non_tensor_batch["multi_modal_data"] = np.array(multi_modal_data_list, dtype=object)
        stats = {key: float(np.mean(values)) for key, values in collected_stats.items() if values}
        return AuxiliaryBatchBuildResult(batch=corrupted_batch, stats=stats)

    def _build_sample(
        self,
        raw_prompt: list[dict[str, Any]],
        multi_modal_data: dict[str, Any] | None,
        full_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        responses: torch.Tensor,
        response_mask: torch.Tensor | None,
        sample_idx: int,
        transform_name: str,
        transform_kwargs: dict[str, Any] | None,
        transform_position: str,
        global_step: int,
        total_training_steps: int | None,
        sample_uid: Any | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None, str, dict[str, float]]:
        if response_mask is None:
            response_attention_mask = attention_mask[-responses.size(0) :]
        else:
            response_attention_mask = response_mask[sample_idx]

        if transform_name == "no_image":
            sample_tensors = self._rebuild_no_image_sample(
                raw_prompt=raw_prompt,
                full_input_ids=full_input_ids,
                responses=responses,
                response_attention_mask=response_attention_mask,
            )
            sample_cache_id = self._build_multi_modal_cache_id(
                sample_uid=sample_uid,
                sample_idx=sample_idx,
                transform_position=transform_position,
            )
            return sample_tensors, None, sample_cache_id, {}

        if transform_name == "mask_visual_attention":
            sample_tensors, sample_stats = self._build_visual_attention_masked_sample(
                full_input_ids=full_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                responses=responses,
            )
            sample_cache_id = self._build_multi_modal_cache_id(
                sample_uid=sample_uid,
                sample_idx=sample_idx,
                transform_position=transform_position,
            )
            return sample_tensors, None, sample_cache_id, sample_stats

        corruption_group_key = self._resolve_corruption_group_key(raw_prompt, multi_modal_data)
        response_text = self._decode_valid_tokens(responses, response_attention_mask)
        transformed_multi_modal_data, transform_stats = self._transform_multi_modal_data(
            multi_modal_data=multi_modal_data,
            response_text=response_text,
            transform_name=transform_name,
            transform_kwargs=transform_kwargs,
            transform_position=transform_position,
            corruption_group_key=corruption_group_key,
            global_step=global_step,
            total_training_steps=total_training_steps,
            sample_idx=sample_idx,
        )
        sample_tensors = {
            "input_ids": full_input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "position_ids": position_ids.clone(),
        }
        sample_cache_id = self._build_multi_modal_cache_id(
            sample_uid=sample_uid,
            sample_idx=sample_idx,
            transform_position=transform_position,
        )
        return sample_tensors, transformed_multi_modal_data, sample_cache_id, transform_stats

    def _resolve_corruption_group_key(
        self, raw_prompt: list[dict[str, Any]], multi_modal_data: dict[str, Any] | None
    ) -> str:
        return _compute_corruption_group_key(raw_prompt, multi_modal_data)

    def _rebuild_no_image_sample(
        self,
        raw_prompt: list[dict[str, Any]],
        full_input_ids: torch.Tensor,
        responses: torch.Tensor,
        response_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prompt_messages = self._drop_visuals_from_messages(raw_prompt)
        prompt_text = self._apply_chat_template(prompt_messages)
        prompt_inputs = self.tokenizer([prompt_text], add_special_tokens=False, return_tensors="pt")
        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_attention_mask = prompt_inputs["attention_mask"][0]
        prompt_width = full_input_ids.size(0) - responses.size(0)
        prompt_ids, prompt_attention_mask = self._left_pad_prompt(prompt_ids, prompt_attention_mask, prompt_width)

        input_ids = torch.cat([prompt_ids, responses], dim=-1)
        attention = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        position_ids = self._compute_position_ids(
            input_ids=input_ids,
            attention_mask=attention,
            image_grid_thw=None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "position_ids": position_ids,
        }

    def _build_visual_attention_masked_sample(
        self,
        full_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        responses: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        prompt_width = full_input_ids.size(0) - responses.size(0)
        visual_token_ids = self._visual_payload_token_ids()
        visual_token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        if visual_token_ids:
            prompt_ids = full_input_ids[:prompt_width]
            visual_ids = torch.tensor(sorted(visual_token_ids), dtype=prompt_ids.dtype, device=prompt_ids.device)
            visual_token_mask[:prompt_width] = torch.isin(prompt_ids, visual_ids)

        masked_attention = attention_mask.clone()
        masked_attention[visual_token_mask] = 0
        stats = {
            "algo/vision/masked_visual_tokens": float(visual_token_mask.sum().item()),
        }
        return {
            "input_ids": full_input_ids.clone(),
            "attention_mask": masked_attention,
            "position_ids": position_ids.clone(),
        }, stats

    def _visual_payload_token_ids(self) -> set[int]:
        return resolve_visual_token_ids(self.tokenizer, "auto")

    def _drop_visuals_from_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stripped_messages = deepcopy(messages)
        for message in stripped_messages:
            content = message.get("content")
            if isinstance(content, list):
                text_items = [item for item in content if item.get("type") not in {"image", "video"}]
                if not text_items:
                    text_items = [{"type": "text", "text": ""}]
                message["content"] = text_items
        return stripped_messages

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        if self.processor is not None:
            return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _left_pad_prompt(
        self, prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor, prompt_width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_width = prompt_ids.size(-1)
        if current_width > prompt_width:
            raise ValueError(f"Corrupted prompt length {current_width} exceeds existing prompt width {prompt_width}.")
        pad_width = prompt_width - current_width
        if pad_width == 0:
            return prompt_ids, prompt_attention_mask
        pad_token_id = self.tokenizer.pad_token_id
        padded_prompt_ids = torch.full((prompt_width,), pad_token_id, dtype=prompt_ids.dtype)
        padded_prompt_attention = torch.zeros((prompt_width,), dtype=prompt_attention_mask.dtype)
        padded_prompt_ids[-current_width:] = prompt_ids
        padded_prompt_attention[-current_width:] = prompt_attention_mask
        return padded_prompt_ids, padded_prompt_attention

    def _transform_multi_modal_data(
        self,
        multi_modal_data: dict[str, Any] | None,
        response_text: str,
        transform_name: str,
        transform_kwargs: dict[str, Any] | None,
        transform_position: str,
        corruption_group_key: str,
        global_step: int,
        total_training_steps: int | None,
        sample_idx: int = 0,
    ) -> tuple[dict[str, Any] | None, dict[str, float]]:
        if multi_modal_data is None:
            return None, {}

        transformed = deepcopy(multi_modal_data)
        raw_images = list(transformed.get("images", []) or [])
        if len(raw_images) == 0:
            return transformed, {}

        if transform_name == "pixelation":
            pixelation_kwargs = dict(transform_kwargs or {})
            pixelation_kwargs.setdefault("ratio", 0.1)
            cache_descriptor = self._cache_descriptor(transform_name, pixelation_kwargs, global_step)
            if transform_position == "prompt":
                cache_key = (corruption_group_key, cache_descriptor)
                if cache_key in self._prompt_cache:
                    cache_entry = self._prompt_cache[cache_key]
                    return deepcopy(cache_entry.multi_modal_data), dict(cache_entry.stats)

            transformed["images"] = self._mark_processed_images(
                [pixelate_image(self._load_image(image), **pixelation_kwargs) for image in raw_images]
            )
            if transform_position == "prompt":
                self._prompt_cache[cache_key] = CachedAuxiliaryMedia(
                    multi_modal_data=deepcopy(transformed),
                    stats={},
                )
            return transformed, {}

        if transform_name == "random_patch":
            random_patch_kwargs = dict(transform_kwargs or {})
            random_patch_kwargs.setdefault("patch_size", self.image_patch_size)
            # mask_before_resize (PAPO's code): mask the image at its original resolution, then resize it like the
            # clean view, so both views keep the same image tokens. By default the patches are masked after
            # resizing, aligned with the vision encoder's patches.
            mask_before_resize = bool(random_patch_kwargs.pop("mask_before_resize", False))
            cache_descriptor = self._cache_descriptor(transform_name, random_patch_kwargs, global_step)
            if mask_before_resize:
                cache_descriptor += ":mask_before_resize"

            def mask(image: Any, seed: int | None) -> Image.Image:
                if not mask_before_resize:
                    return random_patch_blackening(self._load_image(image), seed=seed, **random_patch_kwargs)
                masked = random_patch_blackening(process_image(image, None, None), seed=seed, **random_patch_kwargs)
                return masked if isinstance(image, ProcessedImageInput) else self._load_image(masked)

            if transform_position == "prompt":
                cache_key = (corruption_group_key, cache_descriptor)
                if cache_key in self._prompt_cache:
                    cache_entry = self._prompt_cache[cache_key]
                    return deepcopy(cache_entry.multi_modal_data), dict(cache_entry.stats)

                seed_base = _stable_prompt_seed(cache_descriptor + ":" + corruption_group_key)
                transformed["images"] = self._mark_processed_images(
                    [mask(image, seed_base + image_idx) for image_idx, image in enumerate(raw_images)]
                )
                self._prompt_cache[cache_key] = CachedAuxiliaryMedia(multi_modal_data=deepcopy(transformed), stats={})
                return transformed, {}

            transformed["images"] = self._mark_processed_images([mask(image, None) for image in raw_images])
            return transformed, {}

        if transform_name == "gaussian_noise":
            # As VEPO's code, the noise goes on the image processor's normalized pixel_values, without clipping:
            # the view keeps the clean images and carries the noise, which the worker adds after the processor
            # (see add_pixel_values_noise). One noise draw per prompt, or per response for position=response.
            gaussian_kwargs = dict(transform_kwargs or {})
            cache_descriptor = self._cache_descriptor(transform_name, gaussian_kwargs, global_step)
            if transform_position == "prompt":
                cache_key = (corruption_group_key, cache_descriptor)
                if cache_key in self._prompt_cache:
                    cache_entry = self._prompt_cache[cache_key]
                    return deepcopy(cache_entry.multi_modal_data), dict(cache_entry.stats)

                seed = _stable_prompt_seed(cache_descriptor + ":" + corruption_group_key, salt="pixel_values_noise_v1")
            else:
                seed = _stable_prompt_seed(
                    f"{cache_descriptor}:{corruption_group_key}:{sample_idx}:{response_text}",
                    salt="pixel_values_noise_response_v1",
                )
            transformed["images"] = self._mark_processed_images([self._load_image(image) for image in raw_images])
            transformed[PIXEL_VALUES_NOISE_KEY] = {"std": float(gaussian_kwargs.get("std", 2.0)), "seed": seed}
            if transform_position == "prompt":
                self._prompt_cache[cache_key] = CachedAuxiliaryMedia(multi_modal_data=deepcopy(transformed), stats={})
            return transformed, {}

        if transform_name == "vp_diffusion":
            if total_training_steps is None:
                raise ValueError("vp_diffusion requires total_training_steps to build the noise schedule.")
            cache_descriptor = self._cache_descriptor(transform_name, transform_kwargs or {}, global_step)
            noise_t, noise_beta = compute_noise_schedule(
                global_step=global_step,
                total_training_steps=total_training_steps,
                noise_t_init=float(transform_kwargs.get("noise_t_init", 500.0) if transform_kwargs else 500.0),
                noise_gamma=float(transform_kwargs.get("noise_gamma", 10.0) if transform_kwargs else 10.0),
                noise_t_max=float(transform_kwargs.get("noise_t_max", 1000.0) if transform_kwargs else 1000.0),
            )
            if transform_position == "prompt":
                cache_key = (corruption_group_key, cache_descriptor)
                if cache_key in self._prompt_cache:
                    cache_entry = self._prompt_cache[cache_key]
                    return deepcopy(cache_entry.multi_modal_data), dict(cache_entry.stats)

                seed_base = _stable_prompt_seed(cache_descriptor + ":" + corruption_group_key)
                transformed["images"] = self._mark_processed_images(
                    [
                        vp_diffusion_noise(self._load_image(image), beta=noise_beta, seed=seed_base + image_idx)
                        for image_idx, image in enumerate(raw_images)
                    ]
                )
                stats = {"noise_t": noise_t, "noise_beta": noise_beta}
                self._prompt_cache[cache_key] = CachedAuxiliaryMedia(
                    multi_modal_data=deepcopy(transformed), stats=stats
                )
                return transformed, stats

            seed_base = _stable_prompt_seed(
                cache_descriptor + ":" + corruption_group_key + ":" + response_text,
                salt="corrupt_response_v1",
            )
            transformed["images"] = self._mark_processed_images(
                [
                    vp_diffusion_noise(self._load_image(image), beta=noise_beta, seed=seed_base + image_idx)
                    for image_idx, image in enumerate(raw_images)
                ]
            )
            stats = {"noise_t": noise_t, "noise_beta": noise_beta}
            return transformed, stats

        loaded_images = [self._load_image(image) for image in raw_images]
        if transform_name in ("cgpo_flat", "cgpo_hierarchical"):
            if transform_position != "response":
                raise ValueError(f"{transform_name} only supports corrupt_image_position='response'.")
            cgpo_kwargs = dict(transform_kwargs or {})
            cgpo_fn = cgpo_flat if transform_name == "cgpo_flat" else cgpo_hierarchical
            transformed["images"] = self._mark_processed_images(cgpo_fn(loaded_images, response_text, **cgpo_kwargs))
            return transformed, {}

        raise NotImplementedError(f"Unsupported image transform mode: {transform_name!r}")

    def _cache_descriptor(self, transform_name: str, transform_kwargs: dict[str, Any], global_step: int) -> str:
        serializable = json.dumps(transform_kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{transform_name}:step:{global_step}:kwargs:{serializable}"

    def _load_image(self, image: Any) -> Image.Image:
        return process_image(image, self.min_pixels, self.max_pixels)

    def _mark_processed_images(self, images: list[Image.Image]) -> list[ProcessedImageInput]:
        return [ProcessedImageInput(image=image) for image in images]

    def _build_multi_modal_cache_id(self, sample_uid: Any | None, sample_idx: int, transform_position: str) -> str:
        if transform_position == "prompt":
            if sample_uid is not None:
                return str(sample_uid)
            return f"prompt:{sample_idx}"
        if sample_uid is not None:
            return f"{sample_uid}:response:{sample_idx}"
        return f"response:{sample_idx}"

    def _decode_valid_tokens(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        valid_ids = token_ids[attention_mask.to(torch.bool)].tolist()
        return self.tokenizer.decode(valid_ids, skip_special_tokens=True)

    def _compute_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        second_per_grid_ts: torch.Tensor | None,
    ) -> torch.Tensor:
        return build_multimodal_position_ids(
            self.processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs={
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "second_per_grid_ts": second_per_grid_ts,
            },
        )
