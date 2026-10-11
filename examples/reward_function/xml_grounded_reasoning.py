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

import ast
import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


REWARD_NAME = "xml_grounded_reasoning"
REWARD_TYPE = "batch"

_BASIC_PATTERN = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>(?P<think>.*?)</think>", re.DOTALL)
_REGION_PATTERN = re.compile(r"<region\b(?P<attrs>[^>]*)>(?P<body>.*?)</region>", re.DOTALL)
_ATTR_PATTERN = re.compile(r'([A-Za-z_][\w-]*)="([^"]*)"')
_GROUNDING_TAG_PATTERN = re.compile(r"<region\b[^>]*>.*?</region>", re.DOTALL)
_GROUNDING_LABEL_PATTERN = re.compile(
    r"\b(?:regions?|bboxes?|boxes|objects?|evidence|grounding)\b",
    re.IGNORECASE,
)
_STANDALONE_FILLER_PATTERN = re.compile(r"^[\s\d\-*().,;:!?，。；：！？、\[\]（）()]*$")
_QUOTE_CHARS_PATTERN = re.compile(r"['\"‘’“”`()]")
_SEGMENT_BOUNDARIES = ".!?。！？;\n\r"
_WHITESPACE_PATTERN = re.compile(r"\s+")
_ANSWER_DECISION_PATTERN = re.compile(
    r"\b(?:the\s+)?(?:correct|final)\s+(?:answer|choice|option|statement)\s+is\b|"
    r"\bthe\s+only\s+correct\s+(?:answer|choice|option|statement)\s+is\b|"
    r"\b(?:option|choice)\s*\(?[A-Z]\)?\s+is\s+(?:the\s+)?correct\b|"
    r"\b(?:therefore|thus|hence|so)[^.\n]{0,200}\b(?:answer|choice|option|statement)\b",
    re.IGNORECASE,
)


def _normalize_region_name(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def _parse_tag_attrs(attr_text: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in _ATTR_PATTERN.finditer(attr_text)}


def _parse_bbox_list_literal(text: str) -> list[list[int]] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = ast.literal_eval(stripped)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None
    if not isinstance(value, list) or not value:
        return None
    boxes: list[list[int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 4:
            return None
        if any(not isinstance(coord, int) or isinstance(coord, bool) for coord in item):
            return None
        x1, y1, x2, y2 = item
        if not all(0 <= coord <= 1000 for coord in item):
            return None
        if not (x1 < x2 and y1 < y2):
            return None
        boxes.append(item)
    return boxes


def _collect_tag_names_in_segment(segment: str) -> set[str]:
    names: set[str] = set()
    for m in _REGION_PATTERN.finditer(segment):
        name = _parse_tag_attrs(m.group("attrs")).get("name", "").strip()
        if name:
            names.add(name)
    return names


def _has_standalone_grounding_tag(text: str) -> bool:
    for match in _GROUNDING_TAG_PATTERN.finditer(text):
        segment_start = max(text.rfind(boundary, 0, match.start()) for boundary in _SEGMENT_BOUNDARIES) + 1
        next_boundaries = [idx for boundary in _SEGMENT_BOUNDARIES if (idx := text.find(boundary, match.end())) != -1]
        segment_end = min(next_boundaries) + 1 if next_boundaries else len(text)
        segment = text[segment_start:segment_end]
        without_tags = _GROUNDING_TAG_PATTERN.sub(" ", segment)
        without_labels = _GROUNDING_LABEL_PATTERN.sub(" ", without_tags)
        if _STANDALONE_FILLER_PATTERN.fullmatch(without_labels):
            return True
        without_names = without_labels
        for name in _collect_tag_names_in_segment(segment):
            without_names = re.sub(re.escape(name), " ", without_names, flags=re.IGNORECASE)
        if _STANDALONE_FILLER_PATTERN.fullmatch(without_names):
            return True
        clean_text = _QUOTE_CHARS_PATTERN.sub(" ", without_labels)
        for name in _collect_tag_names_in_segment(segment):
            words = name.strip().split()
            if words:
                clean_text = re.sub(r"\s+".join(re.escape(w) for w in words), " ", clean_text, flags=re.IGNORECASE)
        if _STANDALONE_FILLER_PATTERN.fullmatch(clean_text):
            return True
    return False


def _has_post_answer_grounding(think_text: str) -> bool:
    first_region = _REGION_PATTERN.search(think_text)
    if first_region is None:
        return False
    return _ANSWER_DECISION_PATTERN.search(think_text[: first_region.start()]) is not None


def _non_space_length(text: str) -> int:
    return len(_WHITESPACE_PATTERN.sub("", text))


def first_region_position(think_text: str) -> float | None:
    """Share of the reasoning prose before the first region, None without regions. The prose is the reasoning without
    its region tags, counted in non-whitespace characters, so neither whitespace nor the regions themselves (e.g.
    more regions appended after the first) move the position."""
    first_region = _REGION_PATTERN.search(think_text)
    if first_region is None:
        return None
    prose_length = _non_space_length(_REGION_PATTERN.sub("", think_text))
    return _non_space_length(think_text[: first_region.start()]) / prose_length if prose_length else 0.0


def _has_late_grounding(think_text: str, max_first_region_fraction: float | None) -> bool:
    """Whether the first region comes after `max_first_region_fraction` of the reasoning prose: grounding appended
    once the reasoning is done rather than used in it."""
    if max_first_region_fraction is None:
        return False
    position = first_region_position(think_text)
    return position is not None and position > max_first_region_fraction


def validate_reward_function_kwargs(late_grounding_fraction: float | None = None, **_: Any) -> None:
    """Checks of `worker.reward.reward_function_kwargs` that the reward manager runs when it starts."""
    if late_grounding_fraction is not None and not 0.0 < late_grounding_fraction < 1.0:
        raise ValueError(f"late_grounding_fraction must be in (0, 1), but got {late_grounding_fraction!r}.")


def format_reward(response: str, num_images: int | None = None, late_grounding_fraction: float | None = None) -> float:
    # Format reward ladder:
    # 0.0: The response misses the basic `<think>...</think>...\boxed{...}` structure.
    # 0.5: The basic structure exists, but there is no valid in-think grounding, a
    #      region appears outside `<think>`, or any <region> tag is isolated as its
    #      own sentence/line instead of being embedded in normal reasoning text, or
    #      grounding is appended only after the answer decision has already been made, or (with
    #      `late_grounding_fraction`) the first region starts after that fraction of the reasoning.
    # 0.6: At least one embedded region is inside `<think>`, but region attributes, ids,
    #      image indices, or names are malformed.
    # 0.7: Region metadata is valid, but one or more bounding-box lists are malformed.
    # 0.8: Bounding boxes are valid, but the same semantic region is defined more than
    #      once for the same image.
    # 1.0: All grounding tags are valid and naturally embedded in the reasoning text.
    if re.fullmatch(_BASIC_PATTERN, response) is None:
        return 0.0
    reward = 0.5
    max_reward = 1.0

    think_match = _THINK_PATTERN.search(response)
    if think_match is None:
        return min(reward, max_reward)
    think_start, think_end = think_match.start(), think_match.end()
    think_text = think_match.group("think")
    if _has_standalone_grounding_tag(think_text):
        max_reward = 0.5
    if _has_post_answer_grounding(think_text):
        max_reward = min(max_reward, 0.5)
    if _has_late_grounding(think_text, late_grounding_fraction):
        max_reward = min(max_reward, 0.5)

    regions = list(_REGION_PATTERN.finditer(response))
    if not regions:
        return min(reward, max_reward)
    if not all(think_start <= match.start() and match.end() <= think_end for match in regions):
        return min(reward, max_reward)
    reward = 0.6

    region_defs: dict[int, dict[str, Any]] = {}
    next_expected_id = 0
    for match in regions:
        attrs = _parse_tag_attrs(match.group("attrs"))
        if set(attrs) != {"name", "image_idx", "id"}:
            return min(reward, max_reward)
        try:
            region_id = int(attrs["id"])
            image_idx = int(attrs["image_idx"])
        except ValueError:
            return min(reward, max_reward)
        if region_id != next_expected_id or image_idx < 0:
            return min(reward, max_reward)
        if num_images is not None and image_idx >= num_images:
            return min(reward, max_reward)
        normalized_name = _normalize_region_name(attrs["name"])
        if not normalized_name:
            return min(reward, max_reward)
        if region_id in region_defs:
            return min(reward, max_reward)
        region_defs[region_id] = {
            "match": match,
            "name": attrs["name"],
            "normalized_name": normalized_name,
            "image_idx": image_idx,
        }
        next_expected_id += 1
    reward = 0.7

    for match in regions:
        boxes = _parse_bbox_list_literal(match.group("body"))
        if boxes is None:
            return min(reward, max_reward)
    reward = 0.8

    seen_region_keys: set[tuple[int, str]] = set()
    for region in region_defs.values():
        region_key = (region["image_idx"], region["normalized_name"])
        if region_key in seen_region_keys:
            return min(reward, max_reward)
        seen_region_keys.add(region_key)

    return min(1.0, max_reward)


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]], format_weight: float = 0.5, late_grounding_fraction: float | None = None
) -> list[dict[str, float]]:
    """`late_grounding_fraction` (e.g. 0.6): the first region must come within that fraction of the reasoning prose
    (`first_region_position`), otherwise the format reward is capped at 0.5 like grounding appended after the answer;
    None (default) does not check."""
    validate_reward_function_kwargs(late_grounding_fraction=late_grounding_fraction)
    scores = []
    for reward_input in reward_inputs:
        raw_response = reward_input["response"]
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", raw_response)
        format_score = format_reward(
            response, num_images=reward_input.get("num_images"), late_grounding_fraction=late_grounding_fraction
        )
        raw_think_match = _THINK_PATTERN.search(raw_response)
        if raw_think_match is not None:
            raw_think = raw_think_match.group("think")
            if _has_standalone_grounding_tag(raw_think) or _has_post_answer_grounding(raw_think):
                format_score = min(format_score, 0.5)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        grounding_consistency_weighted = float(reward_input.get("grounding_consistency", 0.0))
        grounding_consistency_raw = float(reward_input.get("grounding_consistency_raw", 0.0))
        if accuracy_score != 1.0:
            grounding_consistency_weighted = 0.0
            grounding_consistency_raw = 0.0
        score: dict[str, float] = {
            "overall": (1 - format_weight) * accuracy_score
            + format_weight * format_score
            + grounding_consistency_weighted,
            "format": format_score,
            "accuracy": accuracy_score,
        }
        if "grounding_consistency" in reward_input:
            score["grounding_consistency"] = grounding_consistency_raw
        scores.append(score)
    return scores
