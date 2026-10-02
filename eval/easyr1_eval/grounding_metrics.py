# Copyright 2026 the Awesome-Perception-Aware-RLVR authors.
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

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxSetMetrics:
    """Detection-style statistics of one predicted box set against one GT box set.

    Boxes are matched one-to-one (maximum total IoU); ``threshold_match_count`` counts matched
    pairs with IoU >= the threshold, which gives the usual box precision/recall/F1@0.5.
    ``grounding_iou`` is the IoU between the union of the predicted boxes and the union of the
    GT boxes (the GRIT "grounding IoU").
    """

    pred_count: int
    gt_count: int
    matched_iou_sum: float
    matched_count: int
    threshold_match_count: int
    grounding_iou: float


def score_box_sets(pred_boxes: list[Box], gt_boxes: list[Box], *, iou_threshold: float = 0.5) -> BoxSetMetrics:
    pred_boxes = validate_normalized_boxes(pred_boxes)
    gt_boxes = validate_normalized_boxes(gt_boxes)

    matched_ious: list[float] = []
    if pred_boxes and gt_boxes:
        matrix = np.asarray([[box_iou(pred, gt) for gt in gt_boxes] for pred in pred_boxes], dtype=np.float64)
        pred_indices, gt_indices = linear_sum_assignment(-matrix)
        matched_ious = [float(matrix[pred_idx, gt_idx]) for pred_idx, gt_idx in zip(pred_indices, gt_indices)]

    return BoxSetMetrics(
        pred_count=len(pred_boxes),
        gt_count=len(gt_boxes),
        matched_iou_sum=sum(matched_ious),
        matched_count=len(matched_ious),
        threshold_match_count=sum(iou >= iou_threshold for iou in matched_ious),
        grounding_iou=union_iou(pred_boxes, gt_boxes),
    )


def validate_normalized_boxes(boxes: list[Box]) -> list[Box]:
    valid = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        if 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0:
            valid.append((x1, y1, x2, y2))
    return valid


def union_iou(first: list[Box], second: list[Box]) -> float:
    first = validate_normalized_boxes(first)
    second = validate_normalized_boxes(second)
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0

    intersections = []
    for a in first:
        for b in second:
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            if x1 < x2 and y1 < y2:
                intersections.append((x1, y1, x2, y2))
    intersection_area = _union_area(intersections)
    union_area = _union_area(first) + _union_area(second) - intersection_area
    return intersection_area / union_area if union_area > 0.0 else 0.0


def intersection_over_target(region: Box, target: Box) -> float:
    """Fraction of one target box covered by a possibly larger crop region."""

    region_values = validate_normalized_boxes([region])
    target_values = validate_normalized_boxes([target])
    if not region_values or not target_values:
        return 0.0
    region = region_values[0]
    target = target_values[0]
    x1, y1 = max(region[0], target[0]), max(region[1], target[1])
    x2, y2 = min(region[2], target[2]), min(region[3], target[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    return intersection / target_area if target_area > 0.0 else 0.0


def mean_target_coverage_by_regions(
    regions: list[Box],
    targets: list[Box],
) -> float | None:
    """Mean GT-area coverage by the geometric union of crop regions.

    Empty-GT samples return ``None`` because the diagnostic describes how much
    annotated evidence was observed, rather than whether an empty prediction
    set was correct.
    """

    regions = validate_normalized_boxes(regions)
    targets = validate_normalized_boxes(targets)
    if not targets:
        return None
    coverage = []
    for target in targets:
        intersections = []
        for region in regions:
            x1, y1 = max(region[0], target[0]), max(region[1], target[1])
            x2, y2 = min(region[2], target[2]), min(region[3], target[3])
            if x1 < x2 and y1 < y2:
                intersections.append((x1, y1, x2, y2))
        target_area = (target[2] - target[0]) * (target[3] - target[1])
        coverage.append(_union_area(intersections) / target_area if target_area > 0.0 else 0.0)
    return sum(coverage) / len(coverage)


def _union_area(boxes: list[Box]) -> float:
    if not boxes:
        return 0.0
    xs = sorted({box[0] for box in boxes} | {box[2] for box in boxes})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if left >= right:
            continue
        intervals = sorted((box[1], box[3]) for box in boxes if box[0] < right and box[2] > left)
        covered_y = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    covered_y += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            covered_y += end - start
        area += (right - left) * covered_y
    return area


def box_iou(first: Box, second: Box) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    denominator = first_area + second_area - intersection
    return intersection / denominator if denominator > 0.0 else 0.0
