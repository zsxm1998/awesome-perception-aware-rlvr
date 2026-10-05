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
"""Benchmarks of the OPD papers: MMStar, BLINK, AI2D, MMMU (val), ZoomBench, HallusionBench's
aqf_mean, and the va_opd / vgs / vcsd / vision_opd / opd suites."""

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

from easyr1_eval.loaders import MCQ_LETTER_SUFFIX, MMMU_OPEN_SUFFIX, load_samples, mmstar_options  # noqa: E402
from easyr1_eval.registry import load_benchmark_specs  # noqa: E402
from easyr1_eval.scorers import (  # noqa: E402
    ZOOMBENCH_CONFLICT,
    _sample_score_and_correct,
    mmmu_eval_open,
    mmmu_open_answer,
    score_hallusionbench,
    score_mmmu,
    score_zoombench,
    zoombench_choice,
    zoombench_count,
    zoombench_response,
)
from easyr1_eval.suites import load_suites  # noqa: E402

from eval.prepare import blink, cli, lmms_lab, mmstar, zoombench  # noqa: E402
from eval.prepare.common import PrepareContext  # noqa: E402


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


# ---------------------------------------------------------------------------
# MMStar
# ---------------------------------------------------------------------------


def test_mmstar_options_are_read_from_both_question_layouts():
    inline = "What is shown?\nOptions: A: A cat, sitting, B: A dog., C: 3, D: none"
    assert mmstar_options(inline) == ["A cat, sitting", "A dog.", "3", "none"]
    choices = "Hint: Please answer the question.\nQuestion: Which?\nChoices:\n(A) 40.7\n(B) 74\n(C) 70.4"
    assert mmstar_options(choices) == ["40.7", "74", "70.4"]
    assert mmstar_options("No options here") == []


def test_mmstar_prepare_and_loader(tmp_path, monkeypatch):
    rows = [
        {
            "index": 0,
            "question": "What is shown?\nOptions: A: a cat, B: a dog, C: a cow, D: a fox",
            "answer": "B",
            "category": "coarse perception",
            "l2_category": "image scene and topic",
            "image": _png((1, 2, 3)),
        },
        {
            "index": 1,
            "question": "Question: What is 2+2?\nChoices:\n(A) 3\n(B) 4",
            "answer": "B",
            "category": "math",
            "l2_category": "numeric commonsense and calculation",
            "image": _png((4, 5, 6)),
        },
    ]
    raw = tmp_path / "raw.parquet"
    pd.DataFrame(rows).to_parquet(raw)
    monkeypatch.setattr(lmms_lab, "hf_download", lambda ctx, repos, filename: raw)
    data_root = tmp_path / "data"

    info = lmms_lab.prepare_renamed(_quiet_ctx(data_root), _with_options(mmstar.SOURCES[0], expected_rows=2))

    assert info == {"repo_id": "Lin-Chen/MMStar", "files": ["val.parquet"], "rows": 2}
    samples = load_samples(_registry("mmstar"), data_root)
    assert [sample.sample_id for sample in samples] == ["0", "1"]
    assert samples[0].prompt == f"{rows[0]['question']}\n{MCQ_LETTER_SUFFIX}"
    assert samples[0].extra_info["options"] == ["a cat", "a dog", "a cow", "a fox"]
    assert samples[1].extra_info["options"] == ["3", "4"]
    assert samples[0].target == "B" and samples[0].images == [rows[0]["image"]]
    assert samples[1].metadata["category"] == "math"


# ---------------------------------------------------------------------------
# BLINK
# ---------------------------------------------------------------------------


def _blink_row(subtask, index, images, choices, answer):
    letters = "".join(f"\n({chr(ord('A') + position)}) {choice}" for position, choice in enumerate(choices))
    row = {
        "idx": f"val_{subtask}_{index}",
        "question": f"Question {index}?",
        "sub_task": subtask.replace("_", " "),
        "choices": choices,
        "answer": answer,
        "prompt": f"Question {index}?\nSelect from the following choices.{letters}\n",
        "explanation": "",
    }
    for position in range(4):
        row[f"image_{position + 1}"] = (
            {"bytes": _png((position, index, 0)), "path": None} if position < images else None
        )
    return row


def test_blink_prepare_and_loader_keep_subtasks_images_and_letters(tmp_path, monkeypatch):
    raw = {
        "Counting": [_blink_row("Counting", 1, 1, ["0", "3", "2", "1"], "(D)")],
        "Jigsaw": [_blink_row("Jigsaw", 1, 3, ["the second image", "the third image"], "B")],
    }

    def fake_download(ctx, repos, filename):
        subtask = filename.split("/")[0]
        path = tmp_path / f"{subtask}.raw.parquet"
        pd.DataFrame(raw[subtask]).to_parquet(path)
        return path

    monkeypatch.setattr(lmms_lab, "hf_download", fake_download)
    source = dataclasses.replace(
        _with_options(
            blink.SOURCES[0],
            files={f"{name}/val-00000-of-00001.parquet": f"{name}.parquet" for name in raw},
            expected_rows=2,
        ),
        outputs=tuple(f"blink/{name}.parquet" for name in raw),
    )
    data_root = tmp_path / "data"

    assert lmms_lab.prepare_renamed(_quiet_ctx(data_root), source)["rows"] == 2
    counting, jigsaw = load_samples(_registry("blink"), data_root)
    assert (counting.sample_id, counting.target, len(counting.images)) == ("val_Counting_1", "D", 1)
    assert (jigsaw.target, len(jigsaw.images)) == ("B", 3)
    assert jigsaw.images[2]["bytes"] == _png((2, 1, 0))  # dataset order
    assert jigsaw.extra_info["options"] == ["the second image", "the third image"]
    assert jigsaw.metadata["category"] == "Jigsaw"
    assert counting.prompt.endswith("(D) 1\n" + MCQ_LETTER_SUFFIX)


# ---------------------------------------------------------------------------
# AI2D
# ---------------------------------------------------------------------------


def test_ai2d_prepare_and_loader_map_the_answer_index_to_a_letter(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    for shard in range(2):
        pd.DataFrame(
            [
                {
                    "question": f"which label is the {shard}?",
                    "options": ["c", "D", "b", "a"],
                    "answer": str(shard + 1),
                    "image": {"bytes": _png((shard, 0, 0)), "path": None},
                }
            ]
        ).to_parquet(raw / "data" / f"test-0000{shard}-of-00002.parquet")
    monkeypatch.setattr(lmms_lab, "hf_snapshot", lambda ctx, repos, patterns: raw)
    data_root = tmp_path / "data"

    info = lmms_lab.prepare_glob(_quiet_ctx(data_root), _with_options(cli.SOURCES["ai2d"], expected_rows=2))

    assert info["rows"] == 2
    first, second = load_samples(_registry("ai2d"), data_root)
    assert (first.sample_id, first.target, second.sample_id, second.target) == ("ai2d:0", "B", "ai2d:1", "C")
    assert first.prompt == f"which label is the 0?\n(A) c\n(B) D\n(C) b\n(D) a\n{MCQ_LETTER_SUFFIX}"
    assert first.extra_info["options"] == ["c", "D", "b", "a"]


# ---------------------------------------------------------------------------
# MMMU (val)
# ---------------------------------------------------------------------------


def _mmmu_row(sample_id, question, options, answer, question_type, images):
    row = {
        "id": sample_id,
        "question": question,
        "options": options,
        "explanation": "",
        "img_type": "['Diagrams']",
        "answer": answer,
        "topic_difficulty": "Easy",
        "question_type": question_type,
        "subfield": "Algebra",
    }
    for number in range(1, 8):
        row[f"image_{number}"] = {"bytes": _png((number, 0, 0)), "path": None} if number <= images else None
    return row


def test_mmmu_loader_uses_the_referenced_images_and_parses_list_answers(tmp_path):
    rows = [
        # images in placeholder order (image 1, 2, then 3 from the options); image 4 is not referenced
        _mmmu_row(
            "validation_Art_Theory_3",
            "Compare <image 2> with <image 1>.",
            "['a cat', '<image 3>']",
            "B",
            "multiple-choice",
            4,
        ),
        _mmmu_row("validation_Math_15", "Solve <image 1>.", "[]", "['24/7', '3.429']", "open", 1),
        _mmmu_row("validation_Physics_2", "Compute <image 1>.", "[]", "1.06", "open", 1),
    ]
    (tmp_path / "mmmu_val").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "mmmu_val" / "validation.parquet")

    choice, listed, number = load_samples(_registry("mmmu_val"), tmp_path)

    assert [image["bytes"] for image in choice.images] == [_png((n, 0, 0)) for n in (1, 2, 3)]
    assert choice.prompt == f"Compare <image 2> with <image 1>.\n(A) a cat\n(B) <image 3>\n{MCQ_LETTER_SUFFIX}"
    assert (choice.target, choice.extra_info) == (
        "B",
        {"options": ["a cat", "<image 3>"], "question_type": "multiple-choice"},
    )
    assert choice.metadata["category"] == "Art_Theory"
    assert listed.target == ["24/7", "3.429"] and listed.prompt == f"Solve <image 1>.\n{MMMU_OPEN_SUFFIX}"
    assert number.target == "1.06" and number.extra_info["options"] == []


@pytest.mark.parametrize(
    ("response", "target", "expected"),
    [
        # outcomes of the official MMMU eval_utils.py (parse_open_response + eval_open) on the same inputs
        ("So the answer is 3.43.", ["24/7", "3.429"], True),
        ("The estimatedRTT is 253.75 msec", "253.75", True),
        ("1,249", "1249", True),
        ("Therefore the result is 6.333", "6.333", True),
        ("The process shown is transformation.", "Transformation", True),
        ("C", "C", True),
        ("It could be cats", "C", False),  # single letters only match as a separate word
        ("x = 0.3", "0.3", True),
        ("The value is 0.29", "0.3", False),
        ("The city is Tampa, Florida", ["Tampa", "Florida"], True),
        ("2e6", "2000000", True),
        ("The answer is 0.015625", "1/64", False),
        (r"<think>maybe 7</think> \boxed{1.06}", "1.06", True),  # the final answer is parsed
    ],
)
def test_mmmu_open_answers_follow_the_official_matching(response, target, expected):
    assert mmmu_eval_open(target, mmmu_open_answer(response)) is expected


def test_score_mmmu_reports_question_types_disciplines_and_subjects(tmp_path):
    rows = [
        {
            "sample_id": "validation_Math_1",
            "target": "B",
            "responses": [r"\boxed{B}", "I am not sure."],  # unparsed choices are wrong, not a random letter
            "extra_info": {"options": ["1", "2"], "question_type": "multiple-choice"},
            "metadata": {"category": "Math"},
        },
        {
            "sample_id": "validation_Art_1",
            "target": "A",
            "responses": ["<answer>(C)</answer>"],
            "extra_info": {"options": ["red", "blue", "green"], "question_type": "multiple-choice"},
            "metadata": {"category": "Art"},
        },
        {
            "sample_id": "validation_Physics_1",
            "target": "1.06",
            "responses": ["The deflection is 1.06 in."],
            "extra_info": {"options": [], "question_type": "open"},
            "metadata": {"category": "Physics"},
        },
    ]

    result = score_mmmu(_registry("mmmu_val"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx((0.5 + 0.0 + 1.0) / 3 * 100)
    assert result.details["accuracy_by_question_type"] == {"multiple-choice": 0.25, "open": 1.0}
    assert result.details["accuracy_by_discipline"] == {"Art and Design": 0.0, "Science": 0.75}
    assert result.details["accuracy_by_category"] == {"Art": 0.0, "Math": 0.5, "Physics": 1.0}
    assert result.details["unparsed_choice_rate"] == pytest.approx(1 / 3)
    per_sample = [json.loads(line) for line in (tmp_path / "mmmu_val_per_sample.jsonl").read_text().splitlines()]
    assert per_sample[0]["answers"] == ["B", None] and per_sample[2]["answers"][0] == ["1.06 in", "1.06"]
    assert _sample_score_and_correct(_registry("mmmu_val"), rows[2]) == (1.0, True)


# ---------------------------------------------------------------------------
# ZoomBench
# ---------------------------------------------------------------------------


def _zoombench_record(record_id, query, response, question_type, image, crop):
    return {
        "id": record_id,
        "query": query,
        "response": response,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "question_type": question_type,
        "image": {"bytes": image, "path": None},
        "crop_image": {"bytes": crop, "path": None},
    }


def test_zoombench_prepare_and_loader(tmp_path, monkeypatch):
    full = _png((9, 9, 9), (16, 16))
    records = [
        _zoombench_record(
            "a1",
            "What color is the cup?\nA. red\nB. tan\nC. gray\nD. beige\nAnswer with the option's letter from the given choices.",
            "D",
            "mcq",
            full,
            _png((1, 1, 1)),
        ),
        _zoombench_record("b2", "Is there a handle?\n\nA. No\nB. Yes\n\n", "B", "mcq", full, _png((2, 2, 2))),
        _zoombench_record(
            "c3",
            "How many bolts are there?\n Please answer using Arabic numerals. ",
            "4",
            "blank",
            _png((3, 3, 3)),
            _png((4, 4, 4)),
        ),
    ]
    raw = tmp_path / "test.parquet"
    pd.DataFrame(records).to_parquet(raw)
    monkeypatch.setattr(zoombench, "hf_download", lambda ctx, repo, filename: raw)
    monkeypatch.setattr(zoombench, "EXPECTED_ROWS", 3)
    data_root = tmp_path / "data"

    info = zoombench.prepare_zoombench(_quiet_ctx(data_root), zoombench.SOURCES[0])

    assert (info["rows"], info["images"], info["crops"]) == (3, 2, 3)  # the shared full image is stored once
    annotations = [
        json.loads(line) for line in (data_root / "zoombench" / "annotations.jsonl").read_text().splitlines()
    ]
    assert all((data_root / "zoombench" / row["crop_image"]).is_file() for row in annotations)
    four, two, count = load_samples(_registry("zoombench"), data_root)
    assert four.prompt == records[0]["query"] and count.prompt.endswith("Please answer using Arabic numerals.")
    assert Path(four.images[0]).read_bytes() == full  # the full image, not the crop
    assert four.extra_info == {"options": ["red", "tan", "gray", "beige"], "question_type": "mcq"}
    assert two.extra_info["options"] == ["No", "Yes"]
    assert (count.target, count.extra_info, count.metadata["category"]) == (
        "4",
        {"options": [], "question_type": "blank"},
        "blank",
    )
    rows = _prediction_rows([four, two, count], ["The cup is beige.", "Yes", ["Answer: 4", "five"]])
    result = score_zoombench(_registry("zoombench"), rows, None, tmp_path)
    assert result.raw_score == pytest.approx((1.0 + 1.0 + 0.5) / 3 * 100)


FOUR = ["brown", "tan", "gray", "beige"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        # final-answer wrappers, in priority order
        ("<answer>D</answer>", "D"),
        ("<answer>D</answer> \\boxed{B}", "D"),  # <answer> before \boxed{}
        ("<think>B or C?</think>\n\\boxed{D}", "D"),
        ("\\boxed{\\text{D}}", "D"),
        ("Answer: D", "D"),  # the first-capital-letter reading of the official fallback gives A
        ("Final Answer: D. beige", "D"),
        ("The correct answer is (D).", "D"),
        ("The answer is:\n(D) beige.", "D"),
        ("The answer is not A. Answer: D", "D"),  # an answer line without a letter is skipped
        ("Answer: D\n\nOption A (brown) is too dark.", "D"),
        ("I choose option D.", "D"),
        ("<answer>Option D</answer>", "D"),
        # a response that is (or starts with) a letter
        ("D", "D"),
        ("(D)", "D"),
        ("D.", "D"),
        ("D)", "D"),
        ("D: beige", "D"),
        ("D\n\nThe umbrella is beige.", "D"),
        ("**D. beige**", "D"),  # markdown emphasis is removed
        ("`D`", "D"),
        ("A is correct.", "A"),
        # option texts
        ("beige", "D"),
        ("Beige.", "D"),
        ("The umbrella looks beige.", "D"),
        ("<answer>the beige one</answer>", "D"),
        ("A beige umbrella.", "D"),  # "A" is the article here
        # no choice: no letter is taken from inside the text
        ("According to the image, it is hard to say.", None),
        ("A red umbrella is behind the person.", None),
        ("Brown or beige, I cannot tell.", None),
        ("<answer>E</answer>", None),
        ("", None),
        # two different final letters
        ("A or D", ZOOMBENCH_CONFLICT),
        ("<answer>B, D</answer>", ZOOMBENCH_CONFLICT),
        ("Answer: (A) and (C)", ZOOMBENCH_CONFLICT),
    ],
)
def test_zoombench_choice_reads_the_final_letter(response, expected):
    assert zoombench_choice(response, FOUR) == expected


def first_letter_reference(text):
    """The official fallback (Vision-OPD eval/judge_qwenlm.py, extract_first_option) for contrast."""
    match = re.search(r"\(([A-Z])\)", text) or re.search(r"([A-Z])[\.\)\s]", text) or re.search(r"([A-Z])", text)
    return match.group(1) if match else ""


def test_zoombench_choice_never_takes_the_first_capital_letter():
    for response in ("Answer: D", "The answer is D", "Correct option: D"):
        assert first_letter_reference(response) != "D"
        assert zoombench_choice(response, FOUR) == "D"
    assert first_letter_reference("According to the image, it is hard to say.") == "A"
    assert zoombench_choice("According to the image, it is hard to say.", FOUR) is None


@pytest.mark.parametrize(
    ("options", "response", "expected"),
    [
        (["No", "Yes"], "Yes", "B"),
        (["No", "Yes"], "No.", "A"),
        (["Yes", "No"], "No, there is no handle.", "B"),
        (["no", "yes"], "<answer>Yes</answer>", "B"),
        (["No", "Yes"], "B", "B"),
        (["No", "Yes"], "C", None),  # not an option of this question
        (["red", "dark red"], "dark red", "B"),  # the longer mentioned option wins
        (["C-shaped", "V-shaped", "O-shaped", "U-shaped"], "C-shaped", "A"),  # not the letter C
        (["头巾", "盾牌", "mask", "hat"], "答案是盾牌", "B"),
        (["m_", "m-", "m.", "m/"], "m.", "C"),
    ],
)
def test_zoombench_choice_maps_option_texts(options, response, expected):
    assert zoombench_choice(response, options) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("4", 4),
        ("4.", 4),
        ("4.0", 4),
        ("Four.", 4),
        ("**4**", 4),
        ("<answer>4</answer>", 4),
        ("\\boxed{4}", 4),
        ("Answer: 4", 4),
        ("The answer is four.", 4),
        ("Answer: There are twenty-one bolts", 21),
        ("There are fourteen.", 14),
        ("There are 1,200 tiles.", 1200),
        ("There are 4 red chairs.", 4),
        ("4\n\nThere are four windows; the one on the left is open.", 4),
        ("<think>maybe 3</think>There are 5.", 5),
        ("The one on the left has three panes.", 3),  # "the one" is not a count
        ("I count 4 windows: 2 on the left and 2 on the right.", 2),  # several numbers: the last one
        ("There are 2.5 cups.", 2.5),
        ("I cannot count them.", None),
    ],
)
def test_zoombench_count_reads_numbers_and_number_words(response, expected):
    assert zoombench_count(response) == expected


def test_zoombench_response_applies_mathruler_first_and_rejects_truncation():
    choice = {"target": "B", "extra_info": {"options": FOUR, "question_type": "mcq"}}
    counting = {"target": "3", "extra_info": {"options": [], "question_type": "blank"}}
    assert zoombench_response(choice, 0, "b") == (True, None, "mathruler")  # grade_answer("B", "b")
    assert zoombench_response(choice, 0, "Answer: B") == (True, "B", "rule")
    assert zoombench_response(choice, 0, "Answer: A or B") == (False, ZOOMBENCH_CONFLICT, "conflict")
    assert zoombench_response(counting, 0, "3 seconds") == (True, 3.0, "mathruler")  # units are dropped
    assert zoombench_response(counting, 0, "three") == (True, 3.0, "rule")
    truncated = {**counting, "response_metadata": [{"finish_reason": "length", "truncated": True}]}
    assert zoombench_response(truncated, 0, "Let me count: 1, 2, 3") == (False, None, "truncated")


def test_score_zoombench_reports_choice_and_counting_accuracy(tmp_path):
    rows = [
        {
            "sample_id": "a",
            "target": "D",
            "responses": ["Answer: D", "beige"],
            "extra_info": {"options": FOUR, "question_type": "mcq"},
        },
        {
            "sample_id": "b",
            "target": "B",
            "responses": ["No", "I cannot tell."],
            "extra_info": {"options": ["No", "Yes"], "question_type": "mcq"},
        },
        {
            "sample_id": "c",
            "target": "4",
            "responses": ["There are four.", "4"],
            "extra_info": {"options": [], "question_type": "blank"},
        },
    ]

    result = score_zoombench(_registry("zoombench"), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(200 / 3)
    assert (result.details["mcq_accuracy"], result.details["counting_accuracy"]) == (0.5, 1.0)
    assert (result.details["mcq_count"], result.details["counting_count"]) == (2, 1)
    assert result.details["answer_source_counts"] == {
        "mathruler": 1,
        "rule": 4,
        "conflict": 0,
        "unparsed": 1,
        "truncated": 0,
    }
    per_sample = [json.loads(line) for line in (tmp_path / "zoombench_per_sample.jsonl").read_text().splitlines()]
    assert per_sample[0]["predictions"] == ["D", "D"] and per_sample[1]["predictions"] == ["A", None]
    assert _sample_score_and_correct(_registry("zoombench"), rows[0]) == (1.0, True)


# ---------------------------------------------------------------------------
# HallusionBench (aAcc, fAcc, qAcc and their mean)
# ---------------------------------------------------------------------------


def _hallusion_row(set_id, figure_id, question_id, correct, category="VD"):
    return {
        "sample_id": f"{category}_math_{set_id}_{figure_id}_{question_id}",
        "target": "yes",
        "responses": ["yes" if correct else "no"],
        "metadata": {
            "category": category,
            "subcategory": "math",
            "set_id": set_id,
            "figure_id": figure_id,
            "question_id": question_id,
        },
    }


def test_hallusionbench_aqf_mean_averages_question_figure_and_pair_accuracy(tmp_path):
    rows = [
        _hallusion_row("0", "0", "0", True),
        _hallusion_row("0", "0", "1", True),
        _hallusion_row("0", "1", "0", True),
        _hallusion_row("0", "1", "1", False),
        _hallusion_row("1", "1", "0", False),
        _hallusion_row("1", "1", "1", True),
    ]

    result = score_hallusionbench(_registry("hallusionbench"), rows, None, tmp_path)

    # aAcc 4/6; figures 0_0 right, 0_1 and 1_1 wrong -> 1/3; pairs 0_q0 and 1_q1 right of 4 -> 1/2
    assert result.primary_metric == "question_accuracy"
    assert result.raw_score == pytest.approx(400 / 6)
    assert (result.details["aAcc"], result.details["fAcc"], result.details["qAcc"]) == pytest.approx(
        (4 / 6, 1 / 3, 0.5)
    )
    assert result.details["aqf_mean"] == pytest.approx((4 / 6 + 1 / 3 + 0.5) / 3)

    only_text = score_hallusionbench(
        _registry("hallusionbench"), [_hallusion_row("2", "0", "0", True, category="VS")], None, tmp_path
    )
    assert only_text.details["fAcc"] is None and only_text.details["aqf_mean"] is None


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def test_opd_suites_list_the_paper_benchmarks_and_can_be_prepared():
    specs = load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml")
    suites = load_suites(ROOT / "eval/config/suites.yaml", specs)
    expected = {
        "va_opd": ("wemath", "mathvista", "mathverse", "hallusionbench", "ai2d", "mmmu_val", "mmstar"),
        "vgs": ("mathvision", "mathverse_v", "logicvista", "mmmu_pro"),
        "vcsd": ("blink", "mmstar", "vstar", "mathvista", "hrbench_4k", "hrbench_8k", "hallusionbench"),
        "vision_opd": ("vstar", "zoombench", "hrbench_4k", "hrbench_8k", "mme_realworld_lite", "mmstar", "pope"),
    }
    # the defaults are the prompt and image size of each reproduction's training scripts
    defaults = {
        "va_opd": {"format_prompt": "examples/format_prompt/math.jinja", "min_pixels": 262144, "max_pixels": 4194304},
        "vgs": {
            "format_prompt": "none",
            "system_prompt": "examples/system_prompt/vgs.txt",
            "min_pixels": 262144,
            "max_pixels": 4194304,
        },
        "vcsd": {
            "format_prompt": "none",
            "system_prompt": "none",
            "chat_template": "examples/chat_template/qwen_no_thinking.jinja",
            "plain_think_tokens": "false",
            "min_pixels": 262144,
            "max_pixels": 4194304,
        },
        "vision_opd": {
            "format_prompt": "none",
            "system_prompt": "none",
            "chat_template": "examples/chat_template/qwen_no_thinking.jinja",
            "min_pixels": 65536,
            "max_pixels": 16777216,
        },
    }
    for name, benchmarks in expected.items():
        assert suites[name].benchmarks == benchmarks
        assert suites[name].notes and suites[name].defaults == defaults[name]
        for key in ("format_prompt", "system_prompt", "chat_template"):
            value = defaults[name].get(key, "none")
            assert value == "none" or (ROOT / value).is_file()
        assert cli.resolve_targets([name]) == list(benchmarks)
    assert not suites["opd"].defaults
    assert suites["opd"].benchmarks == suites["comparison"].benchmarks
    assert {"mmstar", "blink", "ai2d", "mmmu_val", "zoombench"} <= set(suites["all"].benchmarks)
