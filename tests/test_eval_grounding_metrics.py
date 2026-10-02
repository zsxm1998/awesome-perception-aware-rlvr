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
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.grounding_metrics import (  # noqa: E402
    box_iou,
    intersection_over_target,
    mean_target_coverage_by_regions,
    score_box_sets,
    union_iou,
)


def test_one_to_one_matching_counts_threshold_matches():
    result = score_box_sets(
        [(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0), (0.0, 0.5, 0.5, 1.0)],
        [(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)],
    )

    assert (result.pred_count, result.gt_count) == (3, 2)
    assert result.matched_count == 2
    assert result.matched_iou_sum == pytest.approx(2.0)
    assert result.threshold_match_count == 2


def test_box_iou_is_shared_by_grounding_scorers():
    assert box_iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.5, 1.0)) == pytest.approx(0.5)


def test_union_iou_of_empty_box_sets():
    both_empty = score_box_sets([], [])
    false_positive = score_box_sets([(0.0, 0.0, 1.0, 1.0)], [])
    missed = score_box_sets([], [(0.0, 0.0, 1.0, 1.0)])

    assert both_empty.grounding_iou == 1.0
    assert false_positive.grounding_iou == 0.0
    assert missed.grounding_iou == 0.0


def test_union_iou_uses_geometric_unions_without_double_counting_overlap():
    assert union_iou(
        [(0.0, 0.0, 0.75, 1.0), (0.25, 0.0, 1.0, 1.0)],
        [(0.0, 0.0, 1.0, 1.0)],
    ) == pytest.approx(1.0)


def test_tool_crop_iogt_does_not_penalize_a_larger_context_crop():
    assert intersection_over_target(
        (0.0, 0.0, 1.0, 1.0),
        (0.25, 0.25, 0.5, 0.5),
    ) == pytest.approx(1.0)


def test_tool_crop_union_coverage_avoids_double_counting_and_marks_empty_gt_na():
    coverage = mean_target_coverage_by_regions(
        [(0.0, 0.0, 0.6, 1.0), (0.4, 0.0, 0.8, 1.0)],
        [(0.0, 0.0, 1.0, 1.0)],
    )

    assert coverage == pytest.approx(0.8)
    assert (
        mean_target_coverage_by_regions(
            [(0.0, 0.0, 1.0, 1.0)],
            [],
        )
        is None
    )
