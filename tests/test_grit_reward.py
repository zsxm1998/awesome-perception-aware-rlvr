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

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "grit.py"
_SPEC = importlib.util.spec_from_file_location("grit_reward", _MODULE_PATH)
grit = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(grit)


def _response(think: str, rethink: str = "The visual evidence supports the answer.", answer: str = "cat") -> str:
    return f"<think>{think}</think><rethink>{rethink}</rethink><answer>{answer}</answer>"


def test_grit_format_reward_matches_paper_structure_and_bbox_components():
    response = _response('The visible target {"label": "cat", "bbox_list": [[10, 20, 110, 220]]} is relevant.')

    assert grit.format_reward(response) == {
        "format": 1.5,
        "format_structure": 1.0,
        "format_bbox": 0.5,
        "format_raw_bbox": 0.25,
        "format_parseable_bbox": 0.25,
    }


def test_grit_format_reward_keeps_raw_bbox_reward_for_non_json_quadruplet():
    response = _response("bbox_2d: [10, 20, 110, 220]")

    assert grit.format_reward(response) == {
        "format": 1.25,
        "format_structure": 1.0,
        "format_bbox": 0.25,
        "format_raw_bbox": 0.25,
        "format_parseable_bbox": 0.0,
    }


def test_grit_format_reward_requires_bbox_before_rethink():
    response = "<think>No boxes yet.</think><rethink>[10, 20, 110, 220]</rethink><answer>cat</answer>"

    assert grit.format_reward(response)["format_bbox"] == 0.0


def test_grit_format_reward_rejects_malformed_bbox_group():
    response = _response('[{"label": "cat", "bbox_list": [[10, 20, 10, 220]]}]')

    assert grit.format_reward(response)["format_parseable_bbox"] == 0.0
    assert grit.format_reward(response)["format_raw_bbox"] == 0.25


def test_grit_bbox_parser_keeps_valid_groups_when_siblings_are_invalid():
    groups = grit.parse_grit_bbox_groups(
        (
            '[{"label": "cat", "bbox_list": [[10, 20, 110, 220]]}, '
            '{"label": "", "bbox_list": [[30, 40, 130, 240]]}, '
            '{"label": "dog", "bbox_list": [[30, 40, 1300, 240]]}, '
            '{"label": "bird", "bbox_list": [[50, 60, 150, 260]]}]'
        )
    )

    assert groups == [
        {"label": "cat", "image_idx": 0, "bbox_list": [[10, 20, 110, 220]]},
        {"label": "bird", "image_idx": 0, "bbox_list": [[50, 60, 150, 260]]},
    ]


def test_grit_bbox_parser_keeps_valid_boxes_when_siblings_are_invalid():
    groups = grit.parse_grit_bbox_groups(
        '{"label": "cat", "bbox_list": [[10, 20, 110, 220], [5, 5, 5, 5], [30, 40, 1300, 240]]}'
    )

    assert groups == [{"label": "cat", "image_idx": 0, "bbox_list": [[10, 20, 110, 220]]}]


def test_grit_format_reward_rejects_non_integer_image_idx_types():
    string_idx_response = _response('[{"label": "cat", "image_idx": "0", "bbox_list": [[10, 20, 110, 220]]}]')
    float_idx_response = _response('[{"label": "cat", "image_idx": 0.0, "bbox_list": [[10, 20, 110, 220]]}]')

    assert grit.format_reward(string_idx_response)["format_parseable_bbox"] == 0.0
    assert grit.format_reward(float_idx_response)["format_parseable_bbox"] == 0.0


def test_grit_format_reward_accepts_valid_multi_image_idx():
    response = _response('[{"label": "cat", "image_idx": 1, "bbox_list": [[10, 20, 110, 220]]}]')

    assert grit.format_reward(response, num_images=2)["format_parseable_bbox"] == 0.25
    assert grit.format_reward(response, num_images=1)["format_parseable_bbox"] == 0.0


def test_grit_answer_reward_uses_rule_accuracy_and_bleu1():
    scores = grit.compute_score(
        [
            {
                "response": _response('[{"label": "cat", "bbox_list": [[10, 20, 110, 220]]}]', answer="cat"),
                "ground_truth": "cat",
                "response_length": 1,
            }
        ]
    )

    assert scores[0] == pytest.approx(
        {
            "overall": 2.6,
            "format": 1.5,
            "format_structure": 1.0,
            "format_bbox": 0.5,
            "format_raw_bbox": 0.25,
            "format_parseable_bbox": 0.25,
            "answer": 1.1,
            "accuracy": 1.0,
            "bleu1": 1.0,
        }
    )


def test_grit_answer_reward_gives_bleu_partial_credit_without_rule_accuracy():
    reward = grit.answer_reward("<answer>red cat</answer>", "cat")

    assert reward["accuracy"] == 0.0
    assert reward["bleu1"] == pytest.approx(0.5)
    assert reward["answer"] == pytest.approx(0.05)


def test_grit_bleu1_reward_matches_grit_official_punctuation_cleaning():
    assert grit.bleu1_reward("cat.", "cat") == pytest.approx(1.0)


def test_grit_answer_reward_is_zero_when_answer_tag_is_missing():
    response = (
        '<think>[{"label": "cat", "bbox_list": [[10, 20, 110, 220]]}] The answer is cat.</think><rethink>cat</rethink>'
    )

    assert grit.extract_answer(response) is None
    assert grit.answer_reward(response, "cat") == {"answer": 0.0, "accuracy": 0.0, "bleu1": 0.0}


@pytest.mark.parametrize(
    ("prediction", "ground_truth", "expected"),
    [
        ("There are 9 dogs.", "9", True),
        ("There are 5 dogs.", "9", False),
        ("nine", "9", True),
        ("No, it does not.", "No", True),
        ("The answer is yes", "No", False),
        ("hot dog", "hot dog", True),
        ("a cat", "dog", False),
    ],
)
def test_official_answer_matching(prediction, ground_truth, expected):
    assert grit._answers_match(prediction, ground_truth) is expected


def test_official_reward_terms():
    response = (
        "<think>I count the people [10, 20, 30, 40] and [50, 60, 70, 80] in the photo.</think>"
        "<rethink>Two grounded people.</rethink><answer>2</answer>"
    )
    (score,) = grit.compute_score_official([{"response": response, "ground_truth": "2"}])
    assert score["accuracy"] == 1.0
    assert score["grounded_format"] == 1.5  # >= 1 box before <rethink> plus the count bonus
    assert score["format"] == pytest.approx(1.0)
    assert score["overall"] == pytest.approx(
        score["accuracy"] + 0.1 * score["bleu1"] + score["format"] + score["repetition"] + score["grounded_format"]
    )

    (missing,) = grit.compute_score_official([{"response": "<think>no answer</think>", "ground_truth": "2"}])
    assert missing["accuracy"] == 0.0
    assert missing["grounded_format"] == 0.0


# UCSB-AI/GRIT@e6d8835 grpo-gr/rewards.py, repetitive_reward and think_and_rethink_format_reward (prints removed)
def _official_repetitive_reward(completions, completion_ids):
    import torch

    ngram_size = 8
    max_reward = 0.5
    rewards = []
    pad_token_id = 151643
    for ids, completion in zip(completion_ids, completions):
        if completion == "":
            rewards.append(max_reward)
            continue
        if len(completion.split()) < ngram_size:
            rewards.append(max_reward)
            continue
        repeat_count = 0
        total = 0
        tokens = completion.split()
        for i in range(len(tokens) - ngram_size):
            ng1 = tuple(tokens[i : i + ngram_size])
            ng2 = tuple(tokens[i + ngram_size : i + ngram_size + ngram_size])
            total += 1
            if ng1 == ng2:
                repeat_count += 1
        if total == 0:
            reward = 1.0
        else:
            reward = 1.0 - (repeat_count / total)
        ids_list = torch.tensor(ids).tolist()
        if pad_token_id is not None:
            if pad_token_id in ids_list:
                ids_list = ids_list[: ids_list.index(pad_token_id)]
        if len(ids_list) < 2 * ngram_size:
            rewards.append(max_reward)
            continue
        repeat_count = 0
        total_pairs = 0
        for i in range(len(ids_list) - 2 * ngram_size + 1):
            ng1 = tuple(ids_list[i : i + ngram_size])
            ng2 = tuple(ids_list[i + ngram_size : i + 2 * ngram_size])
            total_pairs += 1
            if ng1 == ng2:
                repeat_count += 1
        if total_pairs == 0:
            score = 1.0
        else:
            score = 1.0 - (repeat_count / total_pairs)
        rewards.append((score - (1 - reward)) * max_reward)
    return rewards


def _official_think_and_rethink_format_reward(completions):
    import re

    rewards = []
    max_reward = 0.5
    keys = ["<think>", "</think>", "<rethink>", "</rethink>"]
    for response in completions:
        reward = 0.0
        response_original = response
        for key in keys:
            if key in response:
                reward += 1.0
                response = response.split(key)[-1]
        if reward == len(keys):
            try:
                response_list = response_original.split("<think>")[-1].split("</think>")[0].strip()
                assert isinstance(response_list, str) and len(response_list) > 1
                response_list = re.sub(r"[^a-zA-Z0-9\s]", " ", response_list)
                response_list = response_list.split(" ")
                assert len(response_list) > 1
                reward += 1.0
            except Exception:
                pass
        rewards.append(reward / (len(keys) + 1) * max_reward)
    return rewards


_LOOP = "the cat sits on the mat and looks " * 4
OFFICIAL_CASES = [
    ("", []),
    ("short answer", [1, 2]),
    (_response('A cat {"bbox_2d": [1, 2, 3, 4]} sits.'), list(range(40))),
    (_response(_LOOP), [5, 6, 7, 8, 9, 10, 11, 12] * 6),  # repeated words and tokens: negative
    (_response(_LOOP), list(range(60)) + [151643, 7, 7]),  # ids cut at the pad id
    ("<think>x</think><answer>1</answer>", list(range(30))),  # no rethink
    ("<rethink>r</rethink><think>a b</think>", list(range(30))),  # tags out of order
    ("<think>ab</think><rethink>r</rethink>", list(range(30))),  # one-word think
    ("<think>a b</think> <think>c</think><rethink>r</rethink><answer>2</answer>", list(range(30))),
]


@pytest.mark.parametrize("response,ids", OFFICIAL_CASES)
def test_repetition_and_structure_terms_match_grits_code(response, ids):
    assert grit._repetition_reward(response, ids) == pytest.approx(_official_repetitive_reward([response], [ids])[0])
    assert grit._think_rethink_structure(response) == pytest.approx(
        _official_think_and_rethink_format_reward([response])[0]
    )


def test_repetition_can_be_negative_and_uses_the_reward_input_ids():
    response = _response(_LOOP)
    ids = [5, 6, 7, 8, 9, 10, 11, 12] * 6
    assert grit._repetition_reward(response, ids) < 0
    (score,) = grit.compute_score_official([{"response": response, "ground_truth": "cat", "response_ids": ids}])
    assert score["repetition"] == pytest.approx(grit._repetition_reward(response, ids))


def test_eval_grit_rule_matches_the_training_reward():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from easyr1_eval.scorers import grit_regex_boxes, grit_rule_answer_correct

    cases = [
        ("<think>x</think><answer>yes</answer>", "yes"),
        ("<answer>No, it is not.</answer>", "no"),
        ("<answer>There are three dogs.</answer>", "3"),
        ("<answer>2</answer>", "3"),
        ("<answer>hot dog", "hot dog"),
        ("<answer>a big red hot dog on a plate today</answer>", "hot dog"),
        ("no answer tag", "yes"),
        ("<answer>cows</answer>", "cow"),
    ]
    for response, target in cases:
        answer = grit._official_answer_text(response)
        expected = answer is not None and grit._answers_match(answer, target)
        assert grit_rule_answer_correct(response, target) == expected, (response, target)

    row = {"eval_metadata": {"box_format": "norm1000"}}
    boxes = grit_regex_boxes('{"bbox_2d": (100, 200, 300, 400)} and [500, 600, 700, 800]', row)
    assert boxes == [(0.1, 0.2, 0.3, 0.4), (0.5, 0.6, 0.7, 0.8)]
