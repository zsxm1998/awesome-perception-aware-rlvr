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
import http.client
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.schemas import BenchmarkSpec  # noqa: E402
from easyr1_eval.scorers import (  # noqa: E402
    JudgeConfig,
    JudgeResult,
    _format_mmvet_judge_prompt,
    _load_judge_cache,
    _parse_mmvet_score,
    _request_text_judge,
    _sample_score_and_correct,
    _threshold_prf,
    extract_final_response_text,
    extract_normalized_boxes,
    generation_diagnostics,
    judge_config_from_args,
    resolve_judge_max_tokens,
    score_answer_bbox,
    score_grounding_iou,
    score_mme,
    score_mmvet,
    score_pope,
    score_refcoco,
    score_seed_bench,
)


def _spec(key, scorer, primary="score", group="Test", metadata=None):
    return BenchmarkSpec(
        key=key,
        label=key,
        group=group,
        loader="dummy",
        scorer=scorer,
        primary_metric=primary,
        metadata=metadata or {},
    )


def _native_agentic_row(
    *,
    sample_id="agent",
    target="cat",
    response="<answer>cat</answer>",
    gt_boxes=None,
    tool_boxes=None,
):
    return {
        "sample_id": sample_id,
        "target": target,
        "responses": [response],
        "extra_info": {"bboxs_normalized": ([[0.0, 0.0, 0.5, 0.5]] if gt_boxes is None else gt_boxes)},
        "image_refs": ["x.jpg"],
        "eval_metadata": {
            "interaction_mode": "agentic",
            "agent_output_contract": "native",
        },
        "agent_diagnostics": [
            {
                "status": "answered",
                "committed_tool_regions": [
                    {
                        "turn_index": index,
                        "image_idx": 0,
                        "bbox_2d": box,
                    }
                    for index, box in enumerate([[0, 0, 500, 500]] if tool_boxes is None else tool_boxes)
                ],
            }
        ],
    }


def test_extract_normalized_boxes_accepts_region_1000_coordinates():
    boxes = extract_normalized_boxes('<region name="cat">[100, 200, 500, 800]</region>')
    assert boxes == [(0.1, 0.2, 0.5, 0.8)]


@pytest.mark.parametrize(
    ("matches", "pred_count", "gt_count", "expected"),
    [
        (0, 0, 0, (1.0, 1.0, 1.0)),
        (0, 0, 1, (0.0, 0.0, 0.0)),
        (0, 1, 0, (0.0, 0.0, 0.0)),
        (1, 2, 1, (0.5, 1.0, 2.0 / 3.0)),
    ],
)
def test_threshold_prf_empty_and_nonempty_conventions(matches, pred_count, gt_count, expected):
    assert _threshold_prf(matches, pred_count, gt_count) == pytest.approx(expected)


def test_relaxed_answer_match_ignores_terminal_period_after_number(tmp_path):
    rows = [
        {
            "sample_id": "numeric-ending",
            "target": "The magnetic force is stronger in Pair 2.",
            "responses": ["<answer>The magnetic force is stronger in Pair 2</answer>"],
            "extra_info": {"bboxs_normalized": []},
        }
    ]

    result = score_answer_bbox(_spec("grit", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(100.0)


def test_answer_bbox_perturbation_score_is_answer_correctness():
    score, correct = _sample_score_and_correct(
        _spec("grit", "answer_bbox", "answer_accuracy"),
        {
            "target": "2",
            "responses": ["<bbox>[[0, 0, 500, 500]]</bbox><answer>two</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
        },
    )

    assert score == pytest.approx(1.0)
    assert correct is True


@pytest.mark.parametrize("width", [0, -1, float("nan")])
def test_answer_bbox_rejects_invalid_pixel_dimensions(tmp_path, width):
    rows = [
        {
            "sample_id": "bad-dimensions",
            "target": "cat",
            "responses": ["<answer>cat</answer>"],
            "extra_info": {
                "bboxs": [[0, 0, 10, 10]],
                "bbox_format": "pixel_xyxy",
                "width": width,
                "height": 100,
            },
        }
    ]

    with pytest.raises(ValueError, match="invalid image dimensions"):
        score_answer_bbox(_spec("grit", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)


def test_answer_bbox_reports_answer_accuracy_and_grit_iou(tmp_path):
    rows = [
        {
            # correct answer, exact box
            "sample_id": "hit",
            "target": "cat",
            "responses": ["<think>the cat [0, 0, 500, 500]</think><answer>cat</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5]]},
        },
        {
            # wrong answer, half-overlapping box: grounding is scored independently of the answer
            "sample_id": "wrong-answer",
            "target": "dog",
            "responses": ["[0, 0, 500, 1000] <answer>cat</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
        },
        {
            # correct answer, no predicted box -> GRIT IoU 0
            "sample_id": "no-box",
            "target": "2",
            "responses": ["<answer>two</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
        },
        {
            # empty GT: excluded from the GRIT IoU, still counts for the answer accuracy
            "sample_id": "empty-gt",
            "target": "0",
            "responses": ["[0, 0, 100, 100] <answer>zero</answer>"],
            "extra_info": {"bboxs_normalized": []},
        },
    ]

    result = score_answer_bbox(_spec("grit_vsr", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(75.0)
    assert result.details["answer/relaxed_accuracy"] == pytest.approx(0.75)
    assert result.details["grounding/samples_with_gt"] == 3
    assert result.details["grounding/grit_iou"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.details["grounding/missing_prediction_rate"] == pytest.approx(0.25)
    assert result.details["grounding/box_format"] == "norm1000"
    per_sample = [
        json.loads(line) for line in (tmp_path / "grit_vsr_per_sample_answer_bbox.jsonl").read_text().splitlines()
    ]
    assert [item["grit_iou"] for item in per_sample] == [pytest.approx(1.0), pytest.approx(0.5), 0.0, None]


def test_answer_bbox_box_precision_recall_at_half_iou(tmp_path):
    rows = [
        {
            "sample_id": "extra-box",
            "target": "cat",
            "responses": ["[0, 0, 500, 500] [500, 500, 1000, 1000] <answer>cat</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5]]},
        },
        {
            "sample_id": "partial",
            "target": "cat",
            "responses": ["[0, 0, 500, 500] <answer>cat</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]},
        },
    ]

    result = score_answer_bbox(_spec("grit_gqa", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)

    # 2 matches at IoU >= 0.5 out of 3 predicted and 3 GT boxes
    assert result.details["grounding/box_precision_at_0_5"] == pytest.approx(2 / 3)
    assert result.details["grounding/box_recall_at_0_5"] == pytest.approx(2 / 3)
    assert result.details["grounding/pred_box_count"] == 3
    assert result.details["grounding/gt_box_count"] == 3


def test_native_agentic_answer_bbox_scores_committed_tool_regions(tmp_path):
    spec = _spec("grit_gqa", "answer_bbox", "answer_accuracy")
    row = _native_agentic_row(
        response="<bbox>[[600, 600, 1000, 1000]]</bbox><answer>cat</answer>",
    )

    result = score_answer_bbox(spec, [row], None, tmp_path)

    assert result.raw_score == pytest.approx(100.0)
    assert result.details["grounding/box_source"] == "committed_tool_regions"
    # the committed crop [0, 0, 500, 500] matches the GT exactly; the box in the text is ignored
    assert result.details["grounding/grit_iou"] == pytest.approx(1.0)
    assert result.details["agentic/tool_crop_iogt_hit_rate_at_0_5"] == pytest.approx(1.0)
    assert result.details["agentic/tool_gt_union_coverage"] == pytest.approx(1.0)
    per_sample = json.loads((tmp_path / "grit_gqa_per_sample_answer_bbox.jsonl").read_text())
    assert per_sample["grounding_box_source"] == "committed_tool_regions"


def test_answer_bbox_mcq_spec_uses_letter_matching(tmp_path):
    spec = BenchmarkSpec(
        key="mcq_grounding",
        label="mcq_grounding",
        group="Test",
        loader="dummy",
        scorer="answer_bbox",
        primary_metric="answer_accuracy",
        metadata={"answer_style": "mcq_letter"},
    )
    rows = [
        {
            # correct letter
            "sample_id": "hit",
            "target": "B",
            "responses": ["<bbox>[[0, 0, 500, 500]]</bbox><answer>B</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5]], "options": ["cat", "dog", "cow", "fox"]},
        },
        {
            # option-text answer resolves to the wrong letter
            "sample_id": "text-wrong",
            "target": "B",
            "responses": ["<bbox>[[0, 0, 500, 500]]</bbox><answer>cat</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5]], "options": ["cat", "dog", "cow", "fox"]},
        },
    ]

    result = score_answer_bbox(spec, rows, None, tmp_path)

    assert result.raw_score == pytest.approx(50.0)
    assert result.details["answer/relaxed_accuracy"] == pytest.approx(0.5)


def test_bbox_parse_failure_ignores_non_box_numeric_lists(tmp_path):
    rows = [
        {
            "sample_id": "three-values",
            "target": "yes",
            "responses": ["<answer>[1, 2, 3]</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
        },
        {
            "sample_id": "invalid-box",
            "target": "yes",
            "responses": ["<answer>[0, 0, 0, 0]</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
        },
    ]

    result = score_answer_bbox(_spec("grit", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)

    assert result.details["grounding/bbox_parse_failure_rate"] == pytest.approx(0.5)


def test_grounding_iou_is_grit_iou_primary_and_reports_acc_at_half(tmp_path):
    rows = [
        {
            "sample_id": "hit",
            "responses": ["<answer>[0, 0, 1000, 1000]</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
        },
        {
            "sample_id": "miss",
            "responses": ["<answer>[0, 0, 100, 100]</answer>"],
            "extra_info": {"bboxs_normalized": [[0.5, 0.5, 1, 1]]},
        },
    ]

    result = score_grounding_iou(_spec("ovd", "grounding_iou", "grit_iou"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(50.0)
    assert result.details["grounding/grit_iou"] == pytest.approx(0.5)
    assert result.details["grounding/acc_at_0_5_iou"] == pytest.approx(0.5)


def test_grounding_iou_uses_every_predicted_box_like_grit(tmp_path):
    anchor = '{"label": "red bread", "bbox_list": [[500, 0, 1000, 500]]}'
    rows = [
        {
            # GRIT takes the union of every box the model wrote, including cited reference objects
            "sample_id": "anchor-cited",
            "responses": [f"<think>The red bread {anchor}.</think><answer>[0, 500, 500, 1000]</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0.5, 0.5, 1]]},
        },
    ]

    result = score_grounding_iou(_spec("ovd", "grounding_iou", "grit_iou"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(50.0)
    assert result.details["grounding/acc_at_0_5_iou"] == pytest.approx(1.0)


def test_grounding_iou_rejects_multi_target_samples(tmp_path):
    rows = [
        {
            "sample_id": "two-targets",
            "responses": ["<answer>[0, 0, 500, 500]</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]},
        }
    ]

    with pytest.raises(ValueError, match="requires exactly one GT box"):
        score_grounding_iou(_spec("ovd", "grounding_iou", "grit_iou"), rows, None, tmp_path)


@pytest.mark.parametrize(
    ("scorer", "target", "response", "metadata", "extra_info"),
    [
        ("seed_bench", "A", "A", {}, {}),
        ("hallusionbench", "yes", "yes", {}, {}),
        ("gqa", "cat", "cat", {}, {}),
        ("pope", "yes", "yes", {}, {}),
        ("mme", "yes", "yes", {}, {}),
        ("refcoco", "", "[0, 0, 1000, 1000]", {}, {"bbox": [0, 0, 1, 1]}),
    ],
)
def test_perturbation_scoring_rejects_truncated_responses(
    scorer,
    target,
    response,
    metadata,
    extra_info,
):
    score, correct = _sample_score_and_correct(
        _spec(scorer, scorer, "score"),
        {
            "sample_id": "truncated",
            "target": target,
            "responses": [response],
            "response_metadata": [{"finish_reason": "length", "truncated": True}],
            "metadata": metadata,
            "extra_info": extra_info,
        },
    )

    assert score == pytest.approx(0.0)
    assert correct is False


def test_answer_bbox_truncated_response_without_final_answer_is_wrong(tmp_path):
    rows = [
        {
            "sample_id": "truncated-empty-gt",
            "target": "0",
            "responses": ["zero"],
            "response_metadata": [{"finish_reason": "length", "truncated": True}],
            "extra_info": {"bboxs_normalized": []},
        }
    ]

    result = score_answer_bbox(_spec("grit", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(0.0)
    assert result.details["answer/relaxed_accuracy"] == pytest.approx(0.0)


def test_refcoco_acc_at_half_iou_prefers_final_answer_with_fallback(tmp_path):
    rows = [
        {
            "sample_id": "ok",
            "responses": ["<think>[500, 500, 600, 600]</think><answer>[0, 0, 1000, 1000]</answer>"],
            "extra_info": {"bbox": [0, 0, 1, 1]},
        },
        {
            "sample_id": "fallback",
            "responses": ["<think>[0, 0, 1000, 1000]</think><answer>the object is visible</answer>"],
            "extra_info": {"bbox": [0, 0, 1, 1]},
        },
        {
            "sample_id": "bad",
            "responses": ["[0, 0, 100, 100]"],
            "extra_info": {"bbox": [0.5, 0.5, 1, 1]},
        },
    ]
    result = score_refcoco(_spec("refcoco", "refcoco", "acc_at_0_5_iou"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(2 / 3 * 100)
    assert result.details["final_answer_boxes"] == 1
    assert result.details["unwrapped_response_boxes"] == 1
    assert result.details["fallback_to_full_response"] == 1


def test_refcoco_truncated_without_final_answer_does_not_use_full_response_box(tmp_path):
    rows = [
        {
            "sample_id": "truncated",
            "responses": ["<think>candidate box [0, 0, 1000, 1000]"],
            "response_metadata": [{"finish_reason": "length", "truncated": True}],
            "extra_info": {"bbox": [0, 0, 1, 1]},
        }
    ]

    result = score_refcoco(_spec("refcoco", "refcoco", "acc_at_0_5_iou"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(0.0)
    assert result.details["truncated_without_final_answer"] == 1


def test_generation_diagnostics_counts_truncated_responses():
    diagnostics = generation_diagnostics(
        [
            {
                "sample_id": "1",
                "responses": ["a", "b"],
                "response_metadata": [
                    {"finish_reason": "stop", "truncated": False},
                    {"finish_reason": "length", "truncated": True},
                ],
            },
            {"sample_id": "2", "responses": ["c"]},
        ]
    )

    assert diagnostics["generation/response_count"] == 3
    assert diagnostics["generation/truncated_count"] == 1
    assert diagnostics["generation/truncated_rate"] == pytest.approx(1 / 3)
    assert diagnostics["generation/samples_with_truncation"] == 1
    assert diagnostics["generation/missing_metadata_count"] == 1


def test_pope_macro_f1(tmp_path):
    rows = [
        {
            "sample_id": "1",
            "target": "yes",
            "responses": [r"<think>x</think>\boxed{yes}"],
            "metadata": {"category": "random"},
        },
        {"sample_id": "2", "target": "no", "responses": ["<answer>yes</answer>"], "metadata": {"category": "random"}},
        {"sample_id": "3", "target": "yes", "responses": ["no"], "metadata": {"category": "popular"}},
    ]
    result = score_pope(_spec("pope", "pope", "macro_f1"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx((2 / 3 + 0) / 2 * 100)
    assert result.details["accuracy"] == pytest.approx(1 / 3)
    assert result.details["accuracy_by_category"]["random"] == pytest.approx(1 / 2)
    assert result.details["accuracy_by_category"]["popular"] == pytest.approx(0.0)


def test_pope_truncated_without_final_answer_is_other(tmp_path):
    rows = [
        {
            "sample_id": "1",
            "target": "yes",
            "responses": ["<think>yes appears in unfinished reasoning"],
            "response_metadata": [{"finish_reason": "length", "truncated": True}],
            "metadata": {"category": "random"},
        }
    ]

    result = score_pope(_spec("pope", "pope", "macro_f1"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(0.0)
    assert result.details["accuracy"] == pytest.approx(0.0)
    assert result.details["truncated_without_final_answer"] == 1


def test_seed_option_accuracy(tmp_path):
    rows = [
        {"sample_id": "1", "target": "B", "responses": ["The answer is B."]},
        {"sample_id": "2", "target": "A", "responses": ["C"]},
    ]
    result = score_seed_bench(_spec("seed", "seed_bench", "seed_all"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("scorer_name", "score_fn"),
    [
        ("seed_bench", score_seed_bench),
    ],
)
def test_option_scorers_reject_unparseable_gold_and_prediction(tmp_path, scorer_name, score_fn):
    row = {
        "sample_id": "invalid-option",
        "target": "unknown",
        "responses": ["unknown"],
    }

    result = score_fn(_spec(scorer_name, scorer_name, "accuracy"), [row], None, tmp_path)
    perturbation_score, perturbation_correct = _sample_score_and_correct(
        _spec(scorer_name, scorer_name, "accuracy"),
        row,
    )

    assert result.raw_score == pytest.approx(0.0)
    assert perturbation_score == pytest.approx(0.0)
    assert perturbation_correct is False


def test_seed_truncated_without_final_answer_is_wrong(tmp_path):
    rows = [
        {
            "sample_id": "1",
            "target": "A",
            "responses": ["<think>A. option text appears before final answer"],
            "response_metadata": [{"finish_reason": "length", "truncated": True}],
        }
    ]

    result = score_seed_bench(_spec("seed", "seed_bench", "seed_all"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(0.0)
    assert result.details["truncated_without_final_answer"] == 1


def test_mme_total_score_and_normalization(tmp_path):
    rows = [
        {
            "sample_id": "1a",
            "target": "yes",
            "responses": [r"<think>locate object</think>\boxed{yes}"],
            "metadata": {"category": "existence", "question_id": "1"},
        },
        {
            "sample_id": "1b",
            "target": "no",
            "responses": ["<answer>no</answer>"],
            "metadata": {"category": "existence", "question_id": "1"},
        },
        {
            "sample_id": "2a",
            "target": "yes",
            "responses": ["no"],
            "metadata": {"category": "count", "question_id": "2"},
        },
        {
            "sample_id": "2b",
            "target": "no",
            "responses": ["no"],
            "metadata": {"category": "count", "question_id": "2"},
        },
    ]
    result = score_mme(_spec("mme", "mme", "total"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(250.0)
    assert result.normalized_score_0_100 == pytest.approx(250.0 / 2800.0 * 100.0)


def test_mme_truncated_without_final_answer_is_other(tmp_path):
    rows = [
        {
            "sample_id": "1a",
            "target": "yes",
            "responses": ["<think>yes appears in unfinished reasoning"],
            "response_metadata": [{"finish_reason": "length", "truncated": True}],
            "metadata": {"category": "existence", "question_id": "1"},
        }
    ]

    result = score_mme(_spec("mme", "mme", "total"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(0.0)
    assert result.details["truncated_without_final_answer"] == 1


def test_extract_final_response_text_prefers_answer_wrappers():
    assert extract_final_response_text(r"<think>yes appears here</think>\boxed{no}") == "no"
    assert extract_final_response_text(r"<think>x</think>\boxed{\text{hot dogs}}") == "hot dogs"
    assert extract_final_response_text(r"<think>x</think>Final Answer: \boxed{-1} and \boxed{-5}") == "-1 and -5"
    assert extract_final_response_text("<answer>red cube</answer>") == "red cube"
    assert extract_final_response_text("After reasoning, final answer: B.") == "B"


def test_mmvet_judges_full_prediction_and_records_raw_response(monkeypatch, tmp_path):
    captured = {}

    def fake_judge(question, target, prediction, judge_config):
        captured["prediction"] = prediction
        return JudgeResult(
            score=1.0,
            raw_content="1.0",
            parse_method="first_token_float",
            attempts=1,
            raw_attempts=[{"attempt": 1, "temperature": 0.0, "content": "1.0"}],
            model="deepseek-v4-flash",
        )

    monkeypatch.setattr("easyr1_eval.scorers._call_text_judge", fake_judge)
    rows = [
        {
            "sample_id": "1",
            "prompt": "What color is the object?",
            "target": "red",
            "responses": [r"<think>long grounded trace</think>\boxed{red}"],
        }
    ]
    judge_config = JudgeConfig(
        provider="deepseek", model="deepseek-v4-flash", base_url="http://example.test", api_key="x"
    )

    result = score_mmvet(_spec("mmvet", "mmvet", "overall"), rows, judge_config, tmp_path)

    assert captured["prediction"] == r"<think>long grounded trace</think>\boxed{red}"
    assert result.raw_score == pytest.approx(100.0)
    cache_rows = [
        json.loads(line) for line in (tmp_path / "mmvet_judge_cache.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert cache_rows[0]["judge_raw_content"] == "1.0"
    assert cache_rows[0]["judge_parse_method"] == "first_token_float"
    judgment_rows = [
        json.loads(line) for line in (tmp_path / "mmvet_judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert judgment_rows[0]["judge_raw_content"] == "1.0"


def test_mmvet_judge_requests_are_bounded_and_cache_writes_are_single_threaded(monkeypatch, tmp_path):
    import easyr1_eval.scorers as scorer_module

    barrier = threading.Barrier(3)
    lock = threading.Lock()
    active = 0
    max_active = 0
    write_threads = []
    real_write_jsonl = scorer_module.write_jsonl

    def fake_judge(question, target, prediction, judge_config):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return JudgeResult(
            score=float(prediction),
            raw_content=prediction,
            parse_method="first_token_float",
            attempts=1,
            raw_attempts=[{"attempt": 1, "temperature": 0.0, "content": prediction}],
            model=judge_config.model,
        )

    def tracking_write_jsonl(path, rows):
        write_threads.append(threading.current_thread().name)
        return real_write_jsonl(path, rows)

    monkeypatch.setattr("easyr1_eval.scorers._call_text_judge", fake_judge)
    monkeypatch.setattr("easyr1_eval.scorers.write_jsonl", tracking_write_jsonl)
    rows = [
        {
            "sample_id": str(index),
            "prompt": f"Question {index}",
            "target": "answer",
            "responses": [str(index % 2)],
        }
        for index in range(6)
    ]
    judge_config = JudgeConfig(
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="http://example.test",
        api_key="x",
        concurrency=3,
    )

    result = score_mmvet(_spec("mmvet", "mmvet", "overall"), rows, judge_config, tmp_path)

    assert max_active == 3
    assert result.raw_score == pytest.approx(50.0)
    judgments = [
        json.loads(line) for line in (tmp_path / "mmvet_judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["sample_id"] for item in judgments] == [str(index) for index in range(6)]
    assert set(write_threads) == {threading.current_thread().name}


def test_mmvet_judge_prompt_and_score_parser_follow_official_numeric_protocol():
    prompt = _format_mmvet_judge_prompt("Question?", "a<AND>b", "model answer")

    assert "Question | Ground truth | Prediction | Correctness" in prompt
    assert "gpt_query_prompt | Ground truth | Prediction | Correctness" not in prompt
    assert "Question? | a <AND> b | model answer |" in prompt
    assert _parse_mmvet_score("0.8") == (0.8, "first_token_float")
    assert _parse_mmvet_score("Score: 0.8") == (0.8, "embedded_float")


def test_mmvet_text_judge_retries_and_keeps_raw_attempts(monkeypatch):
    from easyr1_eval.scorers import _call_text_judge

    calls = []

    def fake_request(prompt, judge_config, *, temperature):
        calls.append((prompt, temperature))
        return ("not sure", "judge") if len(calls) == 1 else ("0.8", "judge")

    monkeypatch.setattr("easyr1_eval.scorers._request_text_judge", fake_request)
    judge_config = JudgeConfig(
        provider="deepseek", model="deepseek-v4-flash", base_url="http://example.test", api_key="x"
    )

    result = _call_text_judge("Q", "A", "P", judge_config)

    assert result.score == pytest.approx(0.8)
    assert result.attempts == 2
    assert result.raw_content == "0.8"
    assert result.raw_attempts[0]["content"] == "not sure"
    assert "Predict the correctness of the answer (digit):" in calls[1][0]


def test_mmvet_judge_cache_skips_failed_parse_rows(tmp_path):
    path = tmp_path / "cache.jsonl"
    rows = [
        {"cache_key": "failed", "judge_parse_method": "failed", "score": 0.0},
        {"cache_key": "ok", "judge_parse_method": "first_token_float", "score": 1.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    cache = _load_judge_cache(path)

    assert "failed" not in cache
    assert cache["ok"]["score"] == 1.0


def test_mmvet_unparseable_judge_result_fails_without_metric(monkeypatch, tmp_path):
    def fake_judge(question, target, prediction, judge_config):
        return JudgeResult(
            score=0.0,
            raw_content="",
            parse_method="failed",
            attempts=5,
            raw_attempts=[{"attempt": 1, "temperature": 0.0, "content": ""}],
            model="deepseek-v4-flash",
        )

    monkeypatch.setattr("easyr1_eval.scorers._call_text_judge", fake_judge)
    rows = [{"sample_id": "bad", "prompt": "Q", "target": "A", "responses": ["P"]}]
    judge_config = JudgeConfig(
        provider="deepseek", model="deepseek-v4-flash", base_url="http://example.test", api_key="x"
    )

    with pytest.raises(RuntimeError, match="no parseable score"):
        score_mmvet(_spec("mmvet", "mmvet", "overall"), rows, judge_config, tmp_path)

    failures = [
        json.loads(line) for line in (tmp_path / "mmvet_judge_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert failures[0]["sample_id"] == "bad"


def test_deepseek_judge_request_disables_thinking_by_default(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"model":"deepseek-v4-flash","choices":[{"message":{"content":"1.0"}}]}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("easyr1_eval.scorers.urllib.request.urlopen", fake_urlopen)
    judge_config = JudgeConfig(
        provider="deepseek", model="deepseek-v4-flash", base_url="http://example.test", api_key="x"
    )

    content, model = _request_text_judge("prompt", judge_config, temperature=0.0)

    assert content == "1.0"
    assert model == "deepseek-v4-flash"
    assert captured["payload"]["thinking"] == {"type": "disabled"}


def test_text_judge_retries_incomplete_chunked_response(monkeypatch):
    attempts = 0

    class FakeResponse:
        def __init__(self, body=None, error=None):
            self.body = body
            self.error = error

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            if self.error is not None:
                raise self.error
            return self.body

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return FakeResponse(error=http.client.IncompleteRead(b"partial"))
        return FakeResponse(body=b'{"model":"deepseek-v4-flash","choices":[{"message":{"content":"1|synonym"}}]}')

    monkeypatch.setattr("easyr1_eval.scorers.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("easyr1_eval.scorers.time.sleep", lambda _: None)
    judge_config = JudgeConfig(
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="http://example.test",
        api_key="x",
        request_retries=2,
    )

    content, model = _request_text_judge("prompt", judge_config, temperature=0.0)

    assert attempts == 2
    assert content == "1|synonym"
    assert model == "deepseek-v4-flash"


def test_judge_max_tokens_defaults_follow_thinking_mode():
    assert resolve_judge_max_tokens(None, "disabled") == 32
    assert resolve_judge_max_tokens(None, "enabled") == 1024
    assert resolve_judge_max_tokens(256, "enabled") == 256


def test_judge_config_uses_resolved_max_tokens_for_deepseek():
    args = SimpleNamespace(
        judge_provider="deepseek",
        judge_model="deepseek-v4-flash",
        judge_base_url="http://example.test",
        judge_api_key="x",
        judge_max_tokens=None,
        judge_request_retries=5,
        judge_thinking="enabled",
    )

    config = judge_config_from_args(args)

    assert config is not None
    assert config.max_tokens == 1024
    assert config.concurrency == 4
    assert config.thinking == "enabled"


@pytest.mark.parametrize(
    "response,letter",
    [
        ("<think>x</think><answer>B</answer>", "B"),
        ("<answer>The answer is C.</answer>", "C"),
        ("<answer>\\boxed{D}</answer>", "D"),
        ("<answer>Option (A) is right", "A"),
        ("no tags, so B", "B"),
        ("<answer>42</answer>", None),
    ],
)
def test_pepo_logicvista_letter(response, letter):
    from easyr1_eval.scorers import pepo_logicvista_letter

    assert pepo_logicvista_letter(response) == letter


def test_pepo_protocol_reads_logicvista_letters(tmp_path):
    from easyr1_eval.scorers import score_boxed_exact_match

    spec = BenchmarkSpec(
        key="logicvista",
        label="LogicVista",
        group="Reasoning",
        loader="sharegpt",
        scorer="boxed_exact_match",
        primary_metric="mean_acc_at_k",
    )
    rows = [
        {"sample_id": "0", "target": "B", "responses": ["<answer>The answer is B</answer>"]},
        {"sample_id": "1", "target": "A, C", "responses": ["\\boxed{A, C}"]},
    ]
    default = score_boxed_exact_match(spec, rows, None, tmp_path)
    pepo_rows = [{**row, "eval_metadata": {"answer_protocol": "pepo"}} for row in rows]
    pepo = score_boxed_exact_match(spec, pepo_rows, None, tmp_path)
    assert default.raw_score == 50.0 and pepo.raw_score == 100.0  # multi-letter answers keep the default rule
    assert pepo.details["pepo_letter_items"] == 1


def test_pepo_protocol_scores_agree_across_scorer_comparison_and_diagnostics(tmp_path):
    """The run comparison and the perturbation diagnostics must read a row as the scorer does."""
    from easyr1_eval import compare
    from easyr1_eval.scorers import _sample_score_and_correct, score_boxed_exact_match

    spec = BenchmarkSpec(
        key="logicvista",
        label="LogicVista",
        group="Reasoning",
        loader="sharegpt",
        scorer="boxed_exact_match",
        primary_metric="mean_acc_at_k",
    )
    row = {
        "benchmark": "logicvista",
        "sample_id": "0",
        "target": "B",
        "responses": ["<answer>The correct option is B.</answer>"],
        "eval_metadata": {"answer_protocol": "pepo"},
    }
    assert score_boxed_exact_match(spec, [row], None, tmp_path).raw_score == 100.0
    assert compare._extract_boxed_exact_match(spec, [row], tmp_path).payload["value"].tolist() == [1.0]
    assert _sample_score_and_correct(spec, row) == (1.0, True)
    default_row = {**row, "eval_metadata": {}}
    assert score_boxed_exact_match(spec, [default_row], None, tmp_path).raw_score == 0.0
    assert _sample_score_and_correct(spec, default_row) == (0.0, False)
