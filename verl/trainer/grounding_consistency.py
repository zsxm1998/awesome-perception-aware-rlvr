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

import hashlib
import inspect
import time
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils.dataset import process_image
from .perception_reasoning_data import (
    ParsedResponseRegion,
    _is_valid_bbox,
    parse_bbox_list_literal,
    parse_response_regions,
)


def parse_bbox_string(text: str) -> list[tuple[int, int, int, int]]:
    # fail closed on invalid detector output: an out-of-range (e.g. pixel-coordinate)
    # pseudo-GT box is a silently wrong anchor, worse than no anchor for the region
    return [tuple(box) for box in parse_bbox_list_literal(text, exact=False) if _is_valid_bbox(box)]


def _pairwise_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _max_match_iou_sum(
    gt_boxes: list[tuple[int, int, int, int]],
    pred_boxes: list[tuple[int, int, int, int]],
) -> float:
    if not gt_boxes or not pred_boxes:
        return 0.0

    iou_matrix = np.array(
        [[_pairwise_iou(gt, pred) for pred in pred_boxes] for gt in gt_boxes],
        dtype=np.float64,
    )
    row_idx, col_idx = linear_sum_assignment(iou_matrix, maximize=True)
    return float(iou_matrix[row_idx, col_idx].sum())


def compute_detection_reward(
    gt_boxes: list[tuple[int, int, int, int]],
    pred_boxes: list[tuple[int, int, int, int]],
) -> float:
    if not gt_boxes and not pred_boxes:
        return 1.0
    if not gt_boxes or not pred_boxes:
        return 0.0

    matched_iou_sum = _max_match_iou_sum(gt_boxes, pred_boxes)
    precision = matched_iou_sum / len(pred_boxes) if pred_boxes else 0.0
    recall = matched_iou_sum / len(gt_boxes) if gt_boxes else 0.0
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def compute_group_eligibility_mask(uids: Sequence[Any], accuracy_values: Sequence[float]) -> list[bool]:
    """Mark samples whose group has at least one fully correct answer.

    GCR is added to the training reward only when accuracy == 1.0 (the gating convention of
    the grounded-reasoning reward functions), so detection for a group with no correct answer
    is wasted work: every score in it is zeroed downstream. Group granularity is required \u2014
    within an eligible group all regions must be detected, including those mentioned only by
    incorrect responses, because they enter the shared group-level denominator (`group`
    aggregation; with `response` they are detected but unused).
    """
    if len(uids) != len(accuracy_values):
        raise ValueError(f"uids and accuracy_values must align, but got {len(uids)} and {len(accuracy_values)}.")
    eligible_uids = {uid for uid, accuracy in zip(uids, accuracy_values) if float(accuracy) == 1.0}
    return [uid in eligible_uids for uid in uids]


_GROUNDING_CONSISTENCY_DETECTORS = {"self", "grounding-dino"}
GROUNDING_CONSISTENCY_AGGREGATIONS = {"group", "response"}
"""How the per-region match scores of a response become its reward:
- `group`: the frequency-weighted share of the group's detectable regions that the response grounds consistently,
  sum_{r in response} w_r * match_r / sum_{r in group} w_r, w_r = fraction of the group's responses that name r;
  regions the detector does not find are left out. Naming more of the group's regions raises the score.
- `response`: the mean match score over the response's own regions, a region the detector does not find scoring 0
  (the CGPO paper: the response's predicted vs. re-detected regions; the released code averages over its entities)."""
_GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
_GROUNDING_DINO_BOX_THRESHOLD = 0.4
_GROUNDING_DINO_TEXT_THRESHOLD = 0.3
_SELF_DETECTION_MAX_TOKENS = 512
"""cap of a self-detection answer. With a trained CGPO policy (Qwen3-VL-4B, comparison setting, 3,934 detections)
the answers have a median of 21 tokens and 99.5% end within 512; the rest repeat or enumerate boxes up to the
rollout's response length without closing the list (no box parsed) or list dozens of boxes. One such answer holds
up the whole detection batch."""


def _normalize_grounding_dino_query(region_name: str) -> str:
    query = region_name.strip()
    if not query:
        return query
    if not _contains_chinese(query):
        query = query.lower()
    if not query.endswith("."):
        query = f"{query}."
    return query


def _normalize_abs_bbox_to_1000(box: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if width <= 0 or height <= 0:
        return None
    if hasattr(box, "detach"):
        box = box.detach().cpu().tolist()
    elif hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(coord) for coord in box]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    normalized = (
        int(round(x1 / float(width) * 1000.0)),
        int(round(y1 / float(height) * 1000.0)),
        int(round(x2 / float(width) * 1000.0)),
        int(round(y2 / float(height) * 1000.0)),
    )
    return (
        max(0, min(1000, normalized[0])),
        max(0, min(1000, normalized[1])),
        max(0, min(1000, normalized[2])),
        max(0, min(1000, normalized[3])),
    )


def _post_process_grounding_dino_object_detection(
    processor: Any,
    outputs: Any,
    input_ids: Any,
    target_sizes: list[tuple[int, int]],
) -> Any:
    post_process = processor.post_process_grounded_object_detection
    kwargs = {
        "outputs": outputs,
        "input_ids": input_ids,
        "text_threshold": _GROUNDING_DINO_TEXT_THRESHOLD,
        "target_sizes": target_sizes,
    }
    parameters = inspect.signature(post_process).parameters
    threshold_param = "threshold"
    if "box_threshold" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        threshold_param = "box_threshold"
    kwargs[threshold_param] = _GROUNDING_DINO_BOX_THRESHOLD
    return post_process(**kwargs)


def _resolve_grounding_dino_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _get_grounding_dino_model_dtype(model: Any) -> torch.dtype | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        for parameter in parameters():
            if torch.is_floating_point(parameter):
                return parameter.dtype
    except TypeError:
        return None
    return None


def _grounding_dino_autocast_context(device: torch.device, model: Any) -> Any:
    dtype = _get_grounding_dino_model_dtype(model)
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _is_cuda_out_of_memory_error(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _run_grounding_dino_detection_requests(
    indexed_requests: list[tuple[int, str, Any]],
    processor: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> list[tuple[int, list[tuple[int, int, int, int]]]]:
    box_lists: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    pending: list[tuple[tuple[int, str], str, Any]] = []
    pending_indices: dict[tuple[int, str], list[int]] = {}

    for request_idx, region_name, image in indexed_requests:
        query = _normalize_grounding_dino_query(region_name)
        if not query:
            box_lists.append((request_idx, []))
            continue
        cache_key = (id(image), query)
        if cache_key in pending_indices:
            pending_indices[cache_key].append(request_idx)
            continue
        pending_indices[cache_key] = [request_idx]
        pending.append((cache_key, query, image))

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        try:
            images = [process_image(image, min_pixels, max_pixels) for _, _, image in chunk]
            queries = [query for _, query, _ in chunk]
            inputs = processor(images=images, text=queries, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode(), _grounding_dino_autocast_context(device, model):
                outputs = model(**inputs)
            results = _post_process_grounding_dino_object_detection(
                processor=processor,
                outputs=outputs,
                input_ids=inputs.input_ids,
                target_sizes=[image.size[::-1] for image in images],
            )
        except Exception as exc:
            if _is_cuda_out_of_memory_error(exc):
                raise
            for cache_key, _query, _original_image in chunk:
                for request_idx in pending_indices[cache_key]:
                    box_lists.append((request_idx, []))
            continue
        for (cache_key, _query, _original_image), image, result in zip(chunk, images, results):
            boxes = result.get("boxes", []) if result else []
            normalized_boxes = []
            for box in boxes:
                normalized_box = _normalize_abs_bbox_to_1000(box, image.width, image.height)
                if normalized_box is not None:
                    normalized_boxes.append(normalized_box)
            for request_idx in pending_indices[cache_key]:
                box_lists.append((request_idx, list(normalized_boxes)))

    return box_lists


@dataclass
class GroundingConsistencyRewardResult:
    reward_tensor: torch.Tensor
    weighted_scores: list[float]
    raw_scores: list[float]
    metrics: dict[str, float]


class GroundingConsistencyRewardScorer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int,
        min_pixels: Optional[int],
        max_pixels: Optional[int],
        video_fps: float,
        reward_weight: float = 0.1,
        prompt_template: str = (
            "检测图像中的{region}，只输出归一化到0-1000范围的二维边界框列表，例如[[x1, y1, x2, y2]]。"
            "如果图中不存在该目标，则输出[]。"
        ),
        prompt_template_en: str = (
            "Detect the {region} in the image. Output only a 2D list of boxes normalized to the 0-1000 range, "
            "for example [[x1, y1, x2, y2]]. If the object is absent, output []."
        ),
        detector: str = "self",
        grounding_dino_device: str = "worker",
        grounding_dino_batch_size: int = 4,
        aggregation: str = "group",
    ):
        if detector not in _GROUNDING_CONSISTENCY_DETECTORS:
            raise ValueError(f"Unknown grounding consistency detector: {detector!r}.")
        if aggregation not in GROUNDING_CONSISTENCY_AGGREGATIONS:
            raise ValueError(f"Unknown grounding consistency aggregation: {aggregation!r}.")
        if grounding_dino_batch_size <= 0:
            raise ValueError(f"grounding_dino_batch_size must be positive, but got {grounding_dino_batch_size}.")
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.video_fps = video_fps
        self.reward_weight = reward_weight
        self.prompt_template = prompt_template
        self.prompt_template_en = prompt_template_en
        self.detector = detector
        self.aggregation = aggregation
        self.grounding_dino_device = grounding_dino_device
        self.grounding_dino_batch_size = grounding_dino_batch_size
        self._grounding_dino_processor: Any | None = None
        self._grounding_dino_model: Any | None = None
        self._grounding_dino_device: torch.device | None = None
        # pseudo-GT cache, valid only within one policy version (see _sync_pseudo_gt_cache)
        self._pseudo_gt_cache: dict[Any, list[tuple[int, int, int, int]]] = {}
        self._pseudo_gt_cache_token: Any = None
        self._image_key_by_id: dict[int, tuple[Any, Any]] = {}
        self._uncacheable_image_count = 0

    def _sync_pseudo_gt_cache(self, cache_token: Any) -> None:
        """Detection results depend on the current policy weights, so cached pseudo-GT is only
        valid while the weights are unchanged. Callers pass a token identifying the policy
        version (e.g. the train step); a token change — or no token — clears the cache, leaving
        pure within-call request deduplication."""
        if cache_token is None or cache_token != self._pseudo_gt_cache_token:
            self._pseudo_gt_cache = {}
            self._image_key_by_id = {}
            self._pseudo_gt_cache_token = cache_token

    def _image_content_key(self, image: Any) -> Any:
        cached_entry = self._image_key_by_id.get(id(image))
        if cached_entry is not None:
            cached_ref, cached_key = cached_entry
            if cached_ref() is image:
                return cached_key
        key = self._compute_image_content_key(image)
        try:
            # weak reference: images are not kept alive by the cache, and if one dies and its
            # id() is reused, the dead reference fails the identity check above and the key
            # is recomputed instead of being served stale
            reference = weakref.ref(image)
        except TypeError:
            # non-weakrefable objects are never memoized: recomputing the key on the next
            # lookup is cheaper than holding a strong reference for the cache lifetime
            return key
        self._image_key_by_id[id(image)] = (reference, key)
        return key

    def _compute_image_content_key(self, image: Any) -> Any:
        try:
            payload = image.tobytes()
            dims = getattr(image, "size", None)
            if not isinstance(dims, (tuple, list)):
                dims = getattr(image, "shape", dims)
            dims_key = tuple(dims) if isinstance(dims, (tuple, list)) else (dims,)
            dtype_key = str(getattr(image, "dtype", ""))
            return (getattr(image, "mode", ""), dtype_key, dims_key, hashlib.md5(payload).hexdigest())
        except Exception:
            # content cannot be hashed: a unique key stays correct (no dedup, no collisions —
            # unlike an id()-based key, which could alias a new object after garbage collection)
            self._uncacheable_image_count += 1
            return ("uncacheable", self._uncacheable_image_count)

    def score_batch(
        self,
        batch: DataProto,
        rollout_worker_group: Any,
        rollout_config: dict[str, Any],
        eligible_sample_mask: Optional[Sequence[bool]] = None,
        cache_token: Any = None,
    ) -> GroundingConsistencyRewardResult:
        start_time = time.perf_counter()
        reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        zero_metrics = {
            "algo/gcr/response_region_count": 0.0,
            "algo/gcr/group_unique_region_count": 0.0,
            "algo/gcr/scored_sample_fraction": 0.0,
            "algo/gcr/reward_raw": 0.0,
            "algo/gcr/detection_request_count": 0.0,
            "algo/gcr/detection_time_s": 0.0,
        }
        if eligible_sample_mask is not None and len(eligible_sample_mask) != len(batch):
            raise ValueError(
                f"eligible_sample_mask length {len(eligible_sample_mask)} does not match batch size {len(batch)}."
            )

        if "multi_modal_data" not in batch.non_tensor_batch:
            zero_metrics["algo/gcr/detection_time_s"] = time.perf_counter() - start_time
            zero_scores = [0.0 for _ in range(len(batch))]
            return GroundingConsistencyRewardResult(
                reward_tensor=reward_tensor,
                weighted_scores=list(zero_scores),
                raw_scores=list(zero_scores),
                metrics=zero_metrics,
            )

        response_lengths = torch.sum(batch.batch["response_mask"], dim=-1)
        response_texts = []
        for response_ids, response_length in zip(batch.batch["responses"], response_lengths):
            valid_ids = response_ids[: int(response_length.item())]
            response_texts.append(
                self.tokenizer.decode(
                    valid_ids.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        sample_regions: list[list[ParsedResponseRegion]] = []
        region_counts: list[int] = []
        for response_text, multi_modal_data in zip(response_texts, batch.non_tensor_batch["multi_modal_data"]):
            if not multi_modal_data or "images" not in multi_modal_data:
                sample_regions.append([])
                region_counts.append(0)
                continue
            images = multi_modal_data.get("images", []) or []
            regions = parse_response_regions(response_text, num_images=len(images))
            sample_regions.append(regions)
            region_counts.append(len(regions))

        uids = batch.non_tensor_batch.get("uid")
        if uids is None:
            uids = np.array([f"sample-{idx}" for idx in range(len(batch))], dtype=object)

        grouped_indices: OrderedDict[Any, list[int]] = OrderedDict()
        for sample_idx, uid in enumerate(uids):
            grouped_indices.setdefault(uid, []).append(sample_idx)

        eligible_uids = set(grouped_indices.keys())
        if eligible_sample_mask is not None:
            eligible_uids = {
                uid
                for uid, indices in grouped_indices.items()
                if any(eligible_sample_mask[sample_idx] for sample_idx in indices)
            }

        self._sync_pseudo_gt_cache(cache_token)

        infer_requests = []
        request_consumers: list[list[tuple[Any, tuple[int, str]]]] = []
        pending_request_by_key: dict[Any, int] = {}
        group_states: dict[Any, dict[str, Any]] = {}
        for uid, indices in grouped_indices.items():
            if uid not in eligible_uids:
                continue
            unique_regions: OrderedDict[tuple[int, str], ParsedResponseRegion] = OrderedDict()
            region_occurrence_count: dict[tuple[int, str], int] = {}
            sample_region_map: dict[int, dict[tuple[int, str], ParsedResponseRegion]] = {}
            for sample_idx in indices:
                regions = sample_regions[sample_idx]
                sample_map: dict[tuple[int, str], ParsedResponseRegion] = {}
                for region in regions:
                    region_key = (region.image_idx, region.normalized_name)
                    if region_key not in sample_map:
                        sample_map[region_key] = region
                    if region_key not in unique_regions:
                        unique_regions[region_key] = region
                sample_region_map[sample_idx] = sample_map
                for region_key in sample_map:
                    region_occurrence_count[region_key] = region_occurrence_count.get(region_key, 0) + 1

            if not unique_regions:
                continue

            group_states[uid] = {
                "indices": indices,
                "sample_region_map": sample_region_map,
                "weights": {
                    region_key: count / max(len(indices), 1) for region_key, count in region_occurrence_count.items()
                },
                "pseudo_gt_boxes": {},
            }
            for region_key, region in unique_regions.items():
                source_idx = indices[0]
                source_images = batch.non_tensor_batch["multi_modal_data"][source_idx].get("images", []) or []
                if region.image_idx >= len(source_images):
                    continue
                image = source_images[region.image_idx]
                # dedup by (image content, raw region name): the raw name determines the
                # detection prompt, so equal keys imply identical requests
                dedup_key = (self._image_content_key(image), region.name)
                if dedup_key in self._pseudo_gt_cache:
                    cached_boxes = self._pseudo_gt_cache[dedup_key]
                    if cached_boxes:
                        group_states[uid]["pseudo_gt_boxes"][region_key] = list(cached_boxes)
                    continue
                request_idx = pending_request_by_key.get(dedup_key)
                if request_idx is None:
                    request_idx = len(infer_requests)
                    pending_request_by_key[dedup_key] = request_idx
                    infer_requests.append(
                        {
                            "region_name": region.name,
                            "image": image,
                        }
                    )
                    request_consumers.append([])
                request_consumers[request_idx].append((uid, region_key))

        if infer_requests:
            infer_box_lists = self._detect_pseudo_gt_boxes(
                infer_requests=infer_requests,
                rollout_worker_group=rollout_worker_group,
                rollout_config=rollout_config,
            )
            for dedup_key, request_idx in pending_request_by_key.items():
                gt_boxes = infer_box_lists[request_idx]
                # empty results are cached too: a region the detector cannot find is left out (`group`)
                # or scores 0 (`response`)
                self._pseudo_gt_cache[dedup_key] = gt_boxes
                if gt_boxes:
                    for uid, region_key in request_consumers[request_idx]:
                        group_states[uid]["pseudo_gt_boxes"][region_key] = list(gt_boxes)

        raw_scores = [0.0 for _ in range(len(batch))]
        weighted_scores = [0.0 for _ in range(len(batch))]
        scored_raw_values: list[float] = []
        scored_samples = 0
        unique_region_counts = []
        for uid, state in group_states.items():
            valid_region_keys = [
                region_key for region_key in state["weights"] if region_key in state["pseudo_gt_boxes"]
            ]
            unique_region_counts.append(len(valid_region_keys))
            denominator = sum(state["weights"][region_key] for region_key in valid_region_keys)
            if self.aggregation == "group" and denominator <= 0.0:
                continue

            for sample_idx in state["indices"]:
                sample_map = state["sample_region_map"].get(sample_idx, {})
                if self.aggregation == "response":
                    # compute_detection_reward([], boxes) = 0: nothing re-detected to match; no regions -> 0
                    region_scores = [
                        compute_detection_reward(
                            state["pseudo_gt_boxes"].get(region_key, []), [tuple(box) for box in region.boxes]
                        )
                        for region_key, region in sample_map.items()
                    ]
                    raw_score = float(np.mean(region_scores)) if region_scores else 0.0
                else:
                    numerator = 0.0
                    for region_key, region in sample_map.items():
                        if region_key not in state["pseudo_gt_boxes"]:
                            continue
                        pred_boxes = [tuple(box) for box in region.boxes]
                        numerator += state["weights"][region_key] * compute_detection_reward(
                            state["pseudo_gt_boxes"][region_key],
                            pred_boxes,
                        )
                    raw_score = numerator / denominator
                weighted_score = raw_score * self.reward_weight
                raw_scores[sample_idx] = raw_score
                weighted_scores[sample_idx] = weighted_score
                scored_raw_values.append(raw_score)
                response_length = int(response_lengths[sample_idx].item())
                if response_length > 0:
                    reward_tensor[sample_idx, response_length - 1] = weighted_score
                scored_samples += 1
        metrics = {
            "algo/gcr/response_region_count": float(np.mean(region_counts)) if region_counts else 0.0,
            "algo/gcr/group_unique_region_count": (
                float(np.mean(unique_region_counts)) if unique_region_counts else 0.0
            ),
            "algo/gcr/scored_sample_fraction": (float(scored_samples) / float(len(batch)) if len(batch) > 0 else 0.0),
            "algo/gcr/reward_raw": float(np.mean(scored_raw_values)) if scored_raw_values else 0.0,
            "algo/gcr/detection_request_count": float(len(infer_requests)),
            "algo/gcr/detection_time_s": time.perf_counter() - start_time,
        }
        return GroundingConsistencyRewardResult(
            reward_tensor=reward_tensor,
            weighted_scores=weighted_scores,
            raw_scores=raw_scores,
            metrics=metrics,
        )

    def _detect_pseudo_gt_boxes(
        self,
        infer_requests: list[dict[str, Any]],
        rollout_worker_group: Any,
        rollout_config: dict[str, Any],
    ) -> list[list[tuple[int, int, int, int]]]:
        if self.detector == "self":
            return self._detect_with_self_rollout(infer_requests, rollout_worker_group, rollout_config)
        if self.detector == "grounding-dino":
            if self.grounding_dino_device == "worker":
                return self._detect_with_grounding_dino_workers(infer_requests, rollout_worker_group)
            return self._detect_with_grounding_dino(infer_requests)
        raise ValueError(f"Unknown grounding consistency detector: {self.detector!r}.")

    def _detect_with_self_rollout(
        self,
        infer_requests: list[dict[str, Any]],
        rollout_worker_group: Any,
        rollout_config: dict[str, Any],
    ) -> list[list[tuple[int, int, int, int]]]:
        infer_batch = self._build_prompt_batch(infer_requests, rollout_config)
        infer_batch, pad_size = pad_dataproto_to_divisor(infer_batch, rollout_worker_group.world_size)
        infer_output = rollout_worker_group.generate_from_raw_prompts(infer_batch)
        infer_output = unpad_dataproto(infer_output, pad_size)

        infer_response_texts = []
        for response_ids, response_length in zip(
            infer_output.batch["responses"], infer_output.batch["response_lengths"]
        ):
            valid_ids = response_ids[: int(response_length.item())]
            infer_response_texts.append(
                self.tokenizer.decode(
                    valid_ids.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        return [parse_bbox_string(detection_text) for detection_text in infer_response_texts]

    def _detect_with_grounding_dino(
        self,
        infer_requests: list[dict[str, Any]],
    ) -> list[list[tuple[int, int, int, int]]]:
        processor, model, device = self._get_grounding_dino()
        try:
            indexed_results = _run_grounding_dino_detection_requests(
                indexed_requests=[
                    (request_idx, str(request["region_name"]), request["image"])
                    for request_idx, request in enumerate(infer_requests)
                ],
                processor=processor,
                model=model,
                device=device,
                batch_size=self.grounding_dino_batch_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        finally:
            model = None
            self.release_grounding_dino()
        box_lists: list[list[tuple[int, int, int, int]]] = [[] for _ in range(len(infer_requests))]
        for request_idx, boxes in indexed_results:
            box_lists[int(request_idx)] = boxes
        return box_lists

    def _detect_with_grounding_dino_workers(
        self,
        infer_requests: list[dict[str, Any]],
        rollout_worker_group: Any,
    ) -> list[list[tuple[int, int, int, int]]]:
        if not hasattr(rollout_worker_group, "detect_grounding_dino"):
            raise ValueError(
                "grounding_dino_device='worker' requires rollout workers with `detect_grounding_dino` support."
            )

        world_size = int(rollout_worker_group.world_size)
        request_shards: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
        for request_idx, request in enumerate(infer_requests):
            worker_idx = request_idx % world_size
            request_shards[worker_idx].append(
                {
                    "request_idx": request_idx,
                    "region_name": request["region_name"],
                    "image": request["image"],
                }
            )

        worker_outputs = rollout_worker_group.detect_grounding_dino(
            request_shards,
            [self.grounding_dino_batch_size] * world_size,
            [self.min_pixels] * world_size,
            [self.max_pixels] * world_size,
        )
        box_lists: list[list[tuple[int, int, int, int]]] = [[] for _ in range(len(infer_requests))]
        for worker_output in worker_outputs:
            for request_idx, boxes in worker_output:
                box_lists[int(request_idx)] = [tuple(box) for box in boxes]
        return box_lists

    def _get_grounding_dino(self) -> tuple[Any, Any, torch.device]:
        if self._grounding_dino_processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "algorithm.grounding_consistency_detector=grounding-dino requires "
                    "`transformers` with Grounding DINO support installed."
                ) from exc
            self._grounding_dino_processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_MODEL_ID)

        if self._grounding_dino_model is None:
            try:
                from transformers import AutoModelForZeroShotObjectDetection
            except ImportError as exc:
                raise ImportError(
                    "algorithm.grounding_consistency_detector=grounding-dino requires "
                    "`transformers` with Grounding DINO support installed."
                ) from exc
            device = self._resolve_grounding_dino_device()
            torch_dtype = _resolve_grounding_dino_torch_dtype(device)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                _GROUNDING_DINO_MODEL_ID,
                torch_dtype=torch_dtype,
            ).to(device)
            model.eval()
            self._grounding_dino_model = model
            self._grounding_dino_device = device
            print(
                "Grounding DINO detector loaded "
                f"model={_GROUNDING_DINO_MODEL_ID} device={device} dtype={torch_dtype} "
                f"batch_size={self.grounding_dino_batch_size}",
                flush=True,
            )
        assert self._grounding_dino_device is not None
        return self._grounding_dino_processor, self._grounding_dino_model, self._grounding_dino_device

    def release_grounding_dino(self) -> None:
        model = self._grounding_dino_model
        self._grounding_dino_model = None
        self._grounding_dino_device = None
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_grounding_dino_device(self) -> torch.device:
        if self.grounding_dino_device == "worker":
            raise ValueError("grounding_dino_device='worker' is only valid for distributed worker-side detection.")
        if self.grounding_dino_device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.grounding_dino_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"grounding_dino_device={self.grounding_dino_device!r} requires CUDA, but CUDA is unavailable."
            )
        return device

    def _build_prompt_batch(self, requests: list[dict[str, Any]], rollout_config: dict[str, Any]) -> DataProto:
        """The detection requests as raw prompt ids and images; the rollout worker processes each image (once per
        call) and builds the vLLM inputs, as for the policy's own rollout."""
        meta_info = dict(rollout_config)
        meta_info.update(
            {
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": _SELF_DETECTION_MAX_TOKENS,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "video_fps": self.video_fps,
            }
        )
        return DataProto.from_dict(
            # one tensor, so that the rollout's tensor-parallel all-gather has a batch to gather
            tensors={"request_index": torch.arange(len(requests))},
            non_tensors={
                "raw_prompt_ids": np.array(
                    [self._raw_prompt_ids(request["region_name"]) for request in requests] + [None], dtype=object
                )[:-1],
                "multi_modal_data": np.array([{"images": [request["image"]]} for request in requests], dtype=object),
            },
            meta_info=meta_info,
        )

    def _raw_prompt_ids(self, region_name: str) -> list[int]:
        """Token ids of the detection prompt (one image placeholder, as the rollout's raw prompts)."""
        prompt_text = self._build_prompt(region_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "\n" + prompt_text},
                ],
            }
        ]

        if self.processor is None:
            raise ValueError("Grounding consistency reward requires a multimodal processor.")
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if hasattr(self.processor, "get_raw_prompt_ids"):
            raw_prompt_ids = self.processor.get_raw_prompt_ids(prompt)
        else:
            raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
        return list(raw_prompt_ids)

    def _build_prompt(self, region_name: str) -> str:
        template = self.prompt_template if _contains_chinese(region_name) else self.prompt_template_en
        return template.format(region=region_name)
