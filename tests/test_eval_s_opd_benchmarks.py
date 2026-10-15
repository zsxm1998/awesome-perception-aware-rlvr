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
"""Benchmarks of the S-OPD table that the harness did not have: VisualPuzzles and the ZeroBench sub-questions.

All rows are synthetic (ZeroBench asks not to publish its answers)."""

import dataclasses
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.loaders import MCQ_LETTER_SUFFIX, VISUALPUZZLES_IMAGE_OPTIONS, load_samples  # noqa: E402
from easyr1_eval.registry import load_benchmark_specs  # noqa: E402
from easyr1_eval.scorers import (  # noqa: E402
    SCORERS,
    _sample_score_and_correct,
    score_mcq,
    score_zerobench,
    zerobench_answer,
    zerobench_contains,
    zerobench_correct,
)

from eval.prepare import lmms_lab, visualpuzzles, zerobench  # noqa: E402
from eval.prepare.common import PrepareContext, PrepareError  # noqa: E402
from eval.prepare.sources import SOURCES  # noqa: E402


def _registry(key):
    return next(spec for spec in load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml") if spec.key == key)


def _png(color=(255, 0, 0), size=(4, 4)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _quiet_ctx(data_root):
    return PrepareContext(data_root=data_root, keep_raw=True, log=lambda message: None)


def _with_options(source, **options):
    return dataclasses.replace(source, options={**source.options, **options})


def _prediction_rows(samples, responses):
    return [
        {
            "sample_id": sample.sample_id,
            "target": sample.target,
            "responses": response if isinstance(response, list) else [response],
            "extra_info": sample.extra_info,
            "metadata": sample.metadata,
        }
        for sample, response in zip(samples, responses)
    ]


def test_registry_entries():
    for key, loader, scorer in (
        ("visualpuzzles", "visualpuzzles", "mcq"),
        ("zerobench_sub", "zerobench", "zerobench"),
    ):
        spec = _registry(key)
        assert (spec.group, spec.loader, spec.scorer, spec.primary_metric) == (
            "Reasoning",
            loader,
            scorer,
            "overall_accuracy",
        )
        assert (spec.temperature, spec.num_samples, spec.max_new_tokens) == (0.0, 1, 2048)
        assert spec.scorer in SCORERS and SOURCES[key].target == key
    assert _registry("visualpuzzles").metadata["aggregate"] == "micro"
    assert not _registry("visualpuzzles").optional and _registry("zerobench_sub").optional  # gated: not in `all`
    assert SOURCES["visualpuzzles"].options["expected_rows"] == 1168
    assert SOURCES["zerobench_sub"].options["expected_rows"] == 334


# ---------------------------------------------------------------------------
# VisualPuzzles
# ---------------------------------------------------------------------------


def _visualpuzzles_row(row_id, question, options, answer, category, color):
    return {
        "id": row_id,
        "image": {"bytes": _png(color), "path": None},
        "question": question,
        "options": options,
        "answer": answer,
        "category": category,
        "difficulty": "easy",
    }


def test_visualpuzzles_prepare_and_loader(tmp_path, monkeypatch):
    rows = [
        _visualpuzzles_row(3, "How many cubes are hidden?  ", ["29", "68", "82", "75"], "D", "spatial", (1, 0, 0)),
        _visualpuzzles_row(7, "Which figure continues the pattern?", None, "B", "inductive", (2, 0, 0)),
    ]
    raw = tmp_path / "raw.parquet"
    pd.DataFrame(rows).to_parquet(raw)
    monkeypatch.setattr(lmms_lab, "hf_download", lambda ctx, repos, filename: raw)
    data_root = tmp_path / "data"

    info = lmms_lab.prepare_renamed(_quiet_ctx(data_root), _with_options(visualpuzzles.SOURCES[0], expected_rows=2))

    assert info == {"repo_id": "neulab/VisualPuzzles", "files": ["train.parquet"], "rows": 2}
    text_options, image_options = load_samples(_registry("visualpuzzles"), data_root)
    assert text_options.prompt == f"How many cubes are hidden?\n(A) 29\n(B) 68\n(C) 82\n(D) 75\n{MCQ_LETTER_SUFFIX}"
    image_question = "Which figure continues the pattern?"
    assert image_options.prompt == f"{image_question}\n{VISUALPUZZLES_IMAGE_OPTIONS}\n{MCQ_LETTER_SUFFIX}"
    assert (text_options.sample_id, text_options.target, text_options.extra_info) == (
        "3",
        "D",
        {"options": ["29", "68", "82", "75"]},
    )
    assert (image_options.target, image_options.extra_info) == ("B", {"options": []})
    assert [sample.images[0]["bytes"] for sample in (text_options, image_options)] == [
        _png((1, 0, 0)),
        _png((2, 0, 0)),
    ]
    assert text_options.metadata["category"] == "spatial" and not text_options.metadata["options_in_image"]
    assert image_options.metadata["options_in_image"] and image_options.metadata["difficulty"] == "easy"


def test_visualpuzzles_scoring_reports_the_reasoning_categories(tmp_path):
    rows = [
        {
            "sample_id": "1",
            "target": "D",
            "responses": ["<think>so 75</think> \\boxed{75}"],  # an option text maps to its letter
            "extra_info": {"options": ["29", "68", "82", "75"]},
            "metadata": {"category": "spatial"},
        },
        {
            "sample_id": "2",
            "target": "B",
            "responses": ["Let me think.\nAnswer: B"],  # the last line the official prompt asks for
            "extra_info": {"options": []},
            "metadata": {"category": "inductive"},
        },
        {
            "sample_id": "3",
            "target": "C",
            "responses": ["\\boxed{(A)}"],
            "extra_info": {"options": []},
            "metadata": {"category": "inductive"},
        },
        {
            "sample_id": "4",
            "target": "A",
            "responses": ["The pattern is unclear."],  # no letter: wrong (the official parser picks a random one)
            "extra_info": {"options": []},
            "metadata": {"category": "deductive"},
        },
    ]

    result = score_mcq(_registry("visualpuzzles"), rows, None, tmp_path)

    assert result.primary_metric == "overall_accuracy"
    assert result.raw_score == pytest.approx(50.0)
    assert result.details["accuracy_by_category"] == {"deductive": 0.0, "inductive": 0.5, "spatial": 1.0}
    assert result.details["unparsed_response_rate"] == pytest.approx(0.25)
    per_sample = [json.loads(line) for line in (tmp_path / "visualpuzzles_per_sample.jsonl").read_text().splitlines()]
    assert [item["letters"] for item in per_sample] == [["D"], ["B"], ["A"], [None]]
    assert _sample_score_and_correct(_registry("visualpuzzles"), rows[1]) == (1.0, True)


# ---------------------------------------------------------------------------
# ZeroBench sub-questions
# ---------------------------------------------------------------------------


def _zerobench_row(question_id, question, answer, colors):
    return {
        "question_id": question_id,
        "question_text": question,
        "question_images_decoded": [{"bytes": _png(color), "path": None} for color in colors],
        "question_answer": answer,
        "question_images": [f"{question_id}_{index}.png" for index in range(len(colors))],
        "image_attribution": "",
    }


def test_zerobench_prepare_and_loader_keep_every_image_in_order(tmp_path, monkeypatch):
    rows = [
        _zerobench_row("1_1", "  How many apples are on the table?\n", "7", [(1, 1, 1)]),
        _zerobench_row("12_3", "Which of the two images is brighter?", "Second", [(2, 2, 2), (3, 3, 3)]),
    ]
    raw = tmp_path / "raw.parquet"
    pd.DataFrame(rows).to_parquet(raw)
    requested = []

    def fake_download(ctx, repos, filename):
        requested.append((repos, filename))
        return raw

    monkeypatch.setattr(lmms_lab, "hf_download", fake_download)
    data_root = tmp_path / "data"

    info = zerobench.prepare_zerobench(_quiet_ctx(data_root), _with_options(zerobench.SOURCES[0], expected_rows=2))

    assert requested == [(["jonathan-roberts1/zerobench"], "data/zerobench_subquestions-00000-of-00001.parquet")]
    assert info == {"repo_id": "jonathan-roberts1/zerobench", "files": ["zerobench_subquestions.parquet"], "rows": 2}
    one, two = load_samples(_registry("zerobench_sub"), data_root)
    assert (one.sample_id, one.prompt, one.target) == ("1_1", "How many apples are on the table?", "7")
    assert [image["bytes"] for image in two.images] == [_png((2, 2, 2)), _png((3, 3, 3))]
    assert (two.metadata["main_question"], two.metadata["image_count"]) == ("12", 2)
    assert one.extra_info == {}


def test_zerobench_prepare_explains_the_gated_dataset(tmp_path, monkeypatch):
    def gated(ctx, repos, filename):
        raise PrepareError(f"could not download {filename} from {repos}: 403 Cannot access gated repo for url ...")

    monkeypatch.setattr(lmms_lab, "hf_download", gated)
    with pytest.raises(PrepareError, match=r"gated dataset\. Accept its terms .*hf auth login"):
        zerobench.prepare_zerobench(_quiet_ctx(tmp_path), zerobench.SOURCES[0])

    def offline(ctx, repos, filename):
        raise PrepareError("could not download: connection reset")

    monkeypatch.setattr(lmms_lab, "hf_download", offline)
    with pytest.raises(PrepareError, match=r"^could not download: connection reset$"):
        zerobench.prepare_zerobench(_quiet_ctx(tmp_path), zerobench.SOURCES[0])


def official_zerobench_correct(response, ground_truth):
    """The official example evaluation (jonathan-roberts1/zerobench README.md), for contrast."""
    formatted_response = response.strip().lower()
    try:
        parsed_answer = re.findall(r"\{(.*?)\}", formatted_response)[-1]
    except IndexError:
        parsed_answer = ""
    return parsed_answer[: len(ground_truth)].lower() == ground_truth.strip().lower() and len(parsed_answer) == len(
        ground_truth.strip()
    )


def _ours(response, ground_truth):
    return zerobench_correct(zerobench_answer(response)[0], ground_truth)


@pytest.mark.parametrize(
    ("response", "ground_truth", "expected"),
    [
        ("<think>count them</think> \\boxed{7}", "7", True),
        ("\\boxed{Second}", "second", True),  # case is ignored
        ("\\boxed{7}", "7 ", True),  # the reference is stripped
        ("<think>maybe \\boxed{6}</think> \\boxed{7}", "7", True),  # the last box
        ("\\boxed{7 kg}", "7", False),  # exact match, no containment
        ("\\boxed{7}", "7 kg", False),
        ("\\boxed{(2, 3)}", "(2, 3)", True),
        ("\\boxed{2, 3}", "(2, 3)", False),
        ("\\boxed{}", "7", False),
    ],
)
def test_zerobench_answers_match_the_official_exact_match(response, ground_truth, expected):
    assert _ours(response, ground_truth) is expected
    assert official_zerobench_correct(response, ground_truth) is expected  # the official regex reads \boxed{x} too


@pytest.mark.parametrize(
    ("response", "ground_truth", "ours", "official"),
    [
        # the official code reads the last "{...}" of the response, ours the last \boxed{} (else <answer>)
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", True, False),  # nested braces: the official reads "2"
        ("\\boxed{ 7 }", "7", True, False),  # mathruler strips the box content
        ("<answer>7</answer>", "7", True, False),  # an <answer> block counts as the final answer
        ("{7}", "7", False, True),  # plain braces are not a box
        ("\\boxed{7} so the set is {1, 2}", "7", True, False),  # braces after the box
        ("7", "7", False, False),  # no answer at all
        ("\\boxed{1 / 2}", "1 / 2", True, True),  # whitespace around "/" is removed on both sides
        ("\\boxed{1/2}", "1 / 2", True, False),
    ],
)
def test_zerobench_answer_reading_differs_from_the_official_braces(response, ground_truth, ours, official):
    assert _ours(response, ground_truth) is ours
    assert official_zerobench_correct(response, ground_truth) is official


def test_zerobench_answer_sources_and_the_unclosed_box():
    assert zerobench_answer("\\boxed{Seven}") == ("seven", "boxed")
    assert zerobench_answer("<answer> Seven </answer>") == ("seven", "answer_tag")
    assert zerobench_answer("The answer is seven.") == ("", "none")
    assert zerobench_answer("\\boxed{None}") == ("none", "boxed")
    assert zerobench_answer("\\boxed{12} and then \\boxed{None") == ("", "none")  # mathruler's "None" sentinel
    assert not zerobench_correct("", "none")
    assert zerobench_contains("7 kg", "7") and not zerobench_correct("7 kg", "7")


def test_score_zerobench_reports_exact_match_and_containment(tmp_path):
    rows = [
        {
            "sample_id": "1_1",
            "target": "7",
            "responses": ["\\boxed{7}", "\\boxed{7 apples}"],
            "metadata": {"main_question": "1"},
        },
        {"sample_id": "1_2", "target": "Blue", "responses": ["<answer>blue</answer>", "no idea"]},
        {"sample_id": "2_1", "target": "3", "responses": ["\\boxed{4}", "\\boxed{4}"]},
    ]

    result = score_zerobench(_registry("zerobench_sub"), rows, None, tmp_path)

    assert result.primary_metric == "overall_accuracy"
    assert result.raw_score == pytest.approx((0.5 + 0.5 + 0.0) / 3 * 100)
    assert result.details["accuracy"] == pytest.approx(1 / 3)
    assert result.details["vlmevalkit_containment_accuracy"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert result.details["k"] == 2
    assert result.details["answer_source_counts"] == {"boxed": 4, "answer_tag": 1, "none": 1}
    per_sample = [json.loads(line) for line in (tmp_path / "zerobench_sub_per_sample.jsonl").read_text().splitlines()]
    assert per_sample[0]["answers"] == ["7", "7 apples"] and per_sample[0]["scores"] == [1.0, 0.0]
    assert per_sample[0]["main_question"] == "1" and per_sample[1]["sources"] == ["answer_tag", "none"]
    assert _sample_score_and_correct(_registry("zerobench_sub"), rows[0]) == (0.5, True)
