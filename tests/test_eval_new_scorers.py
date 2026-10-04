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
import json
import re
import sys
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import pytest
from mathruler.grader import extract_boxed_content, grade_answer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.schemas import BenchmarkSpec  # noqa: E402
from easyr1_eval.scorers import (  # noqa: E402
    SCORERS,
    _sample_score_and_correct,
    boxed_answer,
    boxed_exact_match,
    extract_choice_letter,
    extract_final_response_text,
    judge_config_from_args,
    judge_missing_reason,
    score_answer_bbox,
    score_boxed_exact_match,
    score_gqa,
    score_hallusionbench,
    score_mcq,
    skipped_result,
)


def _spec(key, scorer, primary="score", metadata=None, group="Test"):
    return BenchmarkSpec(
        key=key, label=key, group=group, loader="dummy", scorer=scorer, primary_metric=primary, metadata=metadata or {}
    )


def papo_eval_reference(predict: str, ground_truth: str) -> float:
    """Verbatim copy of PAPO-Eval papo_eval/eval_utils.py::compute_accuracy_boxed_math."""
    predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


PAPO_CASES = [
    (r"<think>2+2</think> \boxed{4}", "4"),
    (r"\boxed{48}", "48"),
    (r"\boxed{47}", "48"),
    (r"\boxed{5\sqrt{3}}", r"5 \sqrt { 3 }"),
    (r"\boxed{\frac{1}{2}}", "0.5"),
    (r"first \boxed{3} then \boxed{7}", "7"),
    (r"first \boxed{3} then \boxed{7}", "3"),
    (r"\boxed{1 / 2}", "1/2"),
    (r"\boxed{B}", "B"),
    (r"\boxed{B. 60}", "B"),
    (r"\boxed{\text{B}}", "B"),
    (r"\boxed{yes}", "Yes"),
    (r"\boxed{x^{2}}", "x^2"),
    (r"unbalanced \boxed{4", "4"),
    ("no final answer", "4"),
    ("", "4"),
]


@pytest.mark.parametrize(("response", "target"), PAPO_CASES)
def test_boxed_exact_match_reproduces_papo_eval(response, target):
    assert boxed_exact_match(response, target) == papo_eval_reference(response, target)


def test_boxed_answer_falls_back_to_answer_tag_only_without_any_box():
    assert boxed_answer("<answer> 48 </answer>") == ("48", "answer_tag")
    assert boxed_exact_match("<think>x</think><answer>48</answer>", "48") == 1.0
    # A response that has a \boxed{} is always scored from the box (PAPO-Eval behaviour).
    assert boxed_answer(r"\boxed{47} <answer>48</answer>") == ("47", "boxed")
    assert boxed_exact_match(r"\boxed{47} <answer>48</answer>", "48") == 0.0
    assert boxed_answer("nothing") == ("None", "none")


def test_score_boxed_exact_match_is_mean_acc_at_k(tmp_path):
    rows = [
        {"sample_id": "a", "target": "4", "responses": [r"\boxed{4}", r"\boxed{5}", r"\boxed{4}", "no box"]},
        {"sample_id": "b", "target": "B", "responses": [r"\boxed{B}"] * 4},
        {
            "sample_id": "c",
            "target": "1",
            "responses": ["<answer>1</answer>", r"\boxed{2}", r"\boxed{2}", r"\boxed{2}"],
        },
    ]
    result = score_boxed_exact_match(_spec("geo3k", "boxed_exact_match", "mean_acc_at_k"), rows, None, tmp_path)

    # PAPO-Eval run_eval.py: mean over the k samples of each question, then mean over questions.
    expected = mean([mean([1, 0, 1, 0]), 1.0, mean([1, 0, 0, 0])]) * 100
    assert result.raw_score == pytest.approx(expected)
    assert result.normalized_score_0_100 == pytest.approx(expected)
    assert result.details["k"] == 4
    assert result.details["pass_at_k"] == pytest.approx(1.0)
    assert result.details["answer_source_counts"] == {"boxed": 10, "answer_tag": 1, "none": 1}
    per_sample = [json.loads(line) for line in (tmp_path / "geo3k_per_sample.jsonl").read_text().splitlines()]
    assert per_sample[0]["answers"] == ["4", "5", "4", "None"]
    assert per_sample[0]["scores"] == [1.0, 0.0, 1.0, 0.0]

    score, correct = _sample_score_and_correct(_spec("geo3k", "boxed_exact_match"), rows[0])
    assert score == pytest.approx(0.5)
    assert correct is True


def _hb_row(category, subcategory, set_id, figure_id, question_id, target, response):
    return {
        "sample_id": f"{category}_{subcategory}_{set_id}_{figure_id}_{question_id}",
        "target": target,
        "responses": [response],
        "metadata": {
            "category": category,
            "subcategory": subcategory,
            "set_id": set_id,
            "figure_id": figure_id,
            "question_id": question_id,
        },
    }


def test_hallusionbench_question_figure_and_pair_accuracy(tmp_path):
    rows = [
        # figure 0 of set 0: both questions right -> figure correct
        _hb_row("VD", "illusion", "0", "0", "0", "yes", r"\boxed{yes}"),
        _hb_row("VD", "illusion", "0", "0", "1", "no", "No."),
        # figure 1 of set 0: one wrong -> figure wrong; question pair 0 (fig0+fig1) wrong
        _hb_row("VD", "illusion", "0", "1", "0", "no", "yes"),
        _hb_row("VD", "illusion", "0", "1", "1", "yes", "<answer>yes</answer>"),
        # VS without figure (figure_id 0) is excluded from the figure accuracy
        _hb_row("VS", "chart", "1", "0", "0", "no", "no"),
    ]
    result = score_hallusionbench(_spec("hallusionbench", "hallusionbench", "question_accuracy"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(80.0)
    assert result.details["question_accuracy"] == pytest.approx(0.8)
    assert result.details["figure_count"] == 2
    assert result.details["figure_accuracy"] == pytest.approx(0.5)
    # pairs: VD_illusion_0_q0 (wrong), VD_illusion_0_q1 (right), VS_chart_1_q0 (right)
    assert result.details["question_pair_count"] == 3
    assert result.details["question_pair_accuracy"] == pytest.approx(2 / 3)
    assert result.details["accuracy_by_category"] == {"VD": pytest.approx(0.75), "VS": pytest.approx(1.0)}
    # the paper's names for the same accuracies and their mean (VCSD's HallusionBench score)
    assert (result.details["aAcc"], result.details["fAcc"], result.details["qAcc"]) == pytest.approx((0.8, 0.5, 2 / 3))
    assert result.details["aqf_mean"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_hallusionbench_truncated_response_is_wrong(tmp_path):
    row = _hb_row("VD", "x", "0", "1", "0", "yes", "yes")
    row["response_metadata"] = [{"finish_reason": "length", "truncated": True}]
    result = score_hallusionbench(_spec("hallusionbench", "hallusionbench"), [row], None, tmp_path)
    assert result.raw_score == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("B", "B"),
        ("(c)", "C"),
        ("D.", "D"),
        (r"<think>zoom</think>\boxed{A}", "A"),
        ("<answer>C. green</answer>", "C"),
        ("The answer is (B).", "B"),
        ("<answer>blue</answer>", "B"),  # unique option text
        ("<answer>A cat</answer>", None),  # article, not a letter
        ("<answer>E</answer>", None),  # out of range
        ('<answer>C {"label": "cup", "bbox_list": [[1, 2, 3, 4]]}</answer>', "C"),
        ("", None),
    ],
)
def test_extract_choice_letter(response, expected):
    assert extract_choice_letter(response, ["red", "blue", "green", "white"]) == expected


def test_mcq_scorer_micro_and_category_mean(tmp_path):
    rows = [
        {
            "sample_id": "1",
            "target": "A",
            "responses": ["A"],
            "extra_info": {"options": ["x", "y"]},
            "metadata": {"category": "single"},
        },
        {
            "sample_id": "2",
            "target": "B",
            "responses": ["A"],
            "extra_info": {"options": ["x", "y"]},
            "metadata": {"category": "single"},
        },
        {
            "sample_id": "3",
            "target": "B",
            "responses": ["<answer>y</answer>"],
            "extra_info": {"options": ["x", "y"]},
            "metadata": {"category": "single"},
        },
        {
            "sample_id": "4",
            "target": "A",
            "responses": ["\\boxed{A}"],
            "extra_info": {"options": ["x", "y"]},
            "metadata": {"category": "cross"},
        },
    ]
    micro = score_mcq(_spec("vstar", "mcq", "overall_accuracy", {"aggregate": "micro"}), rows, None, tmp_path)
    macro = score_mcq(
        _spec("hrbench_4k", "mcq", "overall_accuracy", {"aggregate": "category_mean"}), rows, None, tmp_path
    )

    assert micro.raw_score == pytest.approx(75.0)
    assert macro.raw_score == pytest.approx((2 / 3 + 1.0) / 2 * 100)
    assert macro.details["accuracy_by_category"] == {"cross": 1.0, "single": pytest.approx(2 / 3)}
    assert macro.details["micro_accuracy"] == pytest.approx(0.75)


def test_gqa_without_judge_is_exact_match_only(tmp_path):
    rows = [
        {"sample_id": "1", "target": "sofa", "responses": [r"\boxed{sofa}"], "metadata": {"question": "What is it?"}},
        {"sample_id": "2", "target": "sofa", "responses": [r"\boxed{couch}"], "metadata": {"question": "What is it?"}},
    ]
    result = score_gqa(_spec("gqa", "gqa", "accuracy"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(50.0)
    assert result.details["scoring"] == "exact_match"
    assert "judge_rescued" not in result.details


def test_answer_bbox_reports_grit_grounding_iou(tmp_path):
    rows = [
        {
            "sample_id": "a",
            "target": "cat",
            "responses": ["[[0, 0, 500, 500]] <answer>cat</answer>"],
            "extra_info": {"bboxs_normalized": [[0.0, 0.0, 0.5, 0.5]]},
        },
        {
            "sample_id": "empty",
            "target": "0",
            "responses": ["<answer>0</answer>"],
            "extra_info": {"bboxs_normalized": []},
        },
    ]
    result = score_answer_bbox(_spec("grit_tallyqa", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)

    # GRIT protocol: samples without GT boxes are left out of the grounding IoU.
    assert result.details["grounding/grit_iou"] == pytest.approx(1.0)
    assert result.raw_score == pytest.approx(100.0)


def test_judge_provider_defaults_to_none_and_reads_openai_environment(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    base = dict(
        judge_model=None,
        judge_base_url=None,
        judge_api_key=None,
        judge_max_tokens=None,
        judge_request_retries=5,
        judge_thinking="disabled",
    )

    assert judge_config_from_args(SimpleNamespace(judge_provider="none", **base)) is None
    assert judge_config_from_args(SimpleNamespace(**base)) is None
    # Provider chosen but no credentials: treated as "no judge" (benchmark gets skipped).
    assert judge_config_from_args(SimpleNamespace(judge_provider="openai", **base)) is None
    assert "OPENAI_API_KEY" in judge_missing_reason(SimpleNamespace(judge_provider="openai"))

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    config = judge_config_from_args(SimpleNamespace(judge_provider="openai", **base))
    assert (config.provider, config.model, config.base_url, config.api_key) == (
        "openai",
        "gpt-4o-mini",
        "http://localhost:8000/v1",
        "k",
    )
    assert config.thinking is None

    monkeypatch.setenv("DEEPSEEK_API_KEY", "d")
    deepseek = judge_config_from_args(SimpleNamespace(judge_provider="deepseek", **base))
    assert deepseek.provider == "deepseek" and deepseek.api_key == "d"


def test_skipped_result_and_scorer_registry():
    result = skipped_result(_spec("mm_vet", "mmvet"), "no judge")
    assert result.status == "skipped"
    assert result.normalized_score_0_100 is None
    assert result.details["skip_reason"] == "no judge"
    assert set(SCORERS) == {
        "boxed_exact_match",
        "pope",
        "hallusionbench",
        "mme",
        "gqa",
        "mmvet",
        "seed_bench",
        "mcq",
        "mmmu",
        "zoombench",
        "answer_bbox",
        "grounding_iou",
        "refcoco",
        "cfpo_match",
    }


def test_unclosed_trailing_answer_tag_is_the_final_answer():
    assert extract_final_response_text("<think>reasoning</think>\n<answer>\nno") == "no"
    assert extract_final_response_text("<answer>yes</answer> then <answer>no") == "yes"
    assert extract_final_response_text(r"<answer>\boxed{4}") == "4"
    assert extract_final_response_text("plain text without wrapper") == "plain text without wrapper"
