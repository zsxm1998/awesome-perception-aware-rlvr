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
"""examples/reward_function/r1v.py gives the same rewards as PEPO's reward plugin."""

import importlib.util
import re
import unicodedata
from pathlib import Path

import pytest
from mathruler.grader import grade_answer


SPEC = importlib.util.spec_from_file_location(
    "r1v_reward", Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "r1v.py"
)
r1v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r1v)

# xzxxntxdy/PEPO@2b1e788 src/pepo/rewards/plugin.py, Format and QA_Accuracy without the ms-swift base class
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def official_format(content):
    text = "" if content is None else str(content)
    return 1.0 if len(THINK_PATTERN.findall(text)) == 1 and len(ANSWER_PATTERN.findall(text)) == 1 else 0.0


def official_normalize_answer(ans):
    if ans is None:
        return ""
    text = unicodedata.normalize("NFKC", str(ans))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(".") or text.endswith("。"):
        text = text[:-1].strip()
    return text.replace("π", "\\pi")


def official_accuracy(pred, gt):
    try:
        match = ANSWER_PATTERN.search("" if pred is None else str(pred))
        if not match:
            return 0.0
        pred_answer = official_normalize_answer(match.group(1).strip())
        ground_truth = official_normalize_answer(gt)
        return 1.0 if grade_answer(ground_truth, pred_answer) else 0.0
    except Exception:
        return 0.0


CASES = [
    ("<think>x</think><answer>15</answer>", "15"),
    ("Sure. <think>x</think>\n<answer>15</answer> Done.", "15"),  # text outside the tags is allowed
    ("<think>x</think><answer>1</answer><answer>15</answer>", "15"),  # two answers: wrong format
    ("<think>a</think><think>b</think><answer>15</answer>", "15"),
    ("<think>x</think><answer>\n15\n</answer>", "15"),  # newline inside the answer
    ("<think>x</think><answer>4π</answer>", "4\\pi"),
    ("<think>x</think><answer>Yes.</answer>", "yes"),
    ("<think>x</think><answer>６.45</answer>", "6.45"),  # full-width digit
    ("<think>x</think><answer>\\frac{1}{2}</answer>", "0.5"),
    ("<think>x</think>15", "15"),  # no answer tag
    ("no tags at all, 15", "15"),
    ("<think>x</think><answer>16</answer>", "15"),
]


@pytest.mark.parametrize("response,ground_truth", CASES)
def test_matches_the_official_plugin(response, ground_truth):
    score = r1v.compute_score({"response": response, "ground_truth": ground_truth})
    assert score["format"] == official_format(response)
    assert score["accuracy"] == official_accuracy(response, ground_truth)
    assert score["overall"] == 0.5 * score["accuracy"] + 0.5 * score["format"]


def test_cases_cover_both_outcomes():
    scores = [r1v.compute_score({"response": r, "ground_truth": g}) for r, g in CASES]
    assert {s["format"] for s in scores} == {0.0, 1.0} and {s["accuracy"] for s in scores} == {0.0, 1.0}
    assert r1v.compute_score({"response": CASES[5][0], "ground_truth": CASES[5][1]})["accuracy"] == 1.0
