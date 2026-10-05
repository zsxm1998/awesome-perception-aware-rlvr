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
"""The correctness reward of the VGS reproduction (examples/reward_function/vgs.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
_MODULE_PATH = ROOT / "examples" / "reward_function" / "vgs.py"
_SPEC = importlib.util.spec_from_file_location("vgs_reward", _MODULE_PATH)
vgs = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(vgs)

from easyr1_eval.scorers import boxed_row_answers  # noqa: E402


CASES = [
    ("<reason>...</reason> \\boxed{C}", "C", 1.0),
    ("<reason>...</reason> C", "C", 1.0),  # the paper's samples often answer right after </reason>
    ("<reason>...</reason> (D) Increased signals", "D", 1.0),
    ("<reason>...</reason> \\boxed{B. slide}", "B", 1.0),
    ("<reason>...</reason> \\boxed{\\text{C}}", "C", 1.0),
    ("<reason>...</reason> \\boxed{C. Grade A}", "C", 1.0),
    ("<reason>...</reason> \\boxed{C}", "D", 0.0),
    ("<reason>...</reason> A and C", "A", 0.0),  # several options
    ("<reason>...</reason> The answer is C", "C", 0.0),  # no option letter at the start
    ("C", "C", 0.0),  # neither \boxed{} nor </reason>
    ("<reason>...</reason> \\boxed{31}", "31.0", 1.0),
    ("<reason>...</reason> 2018", "2018", 1.0),
    ("<reason>...</reason> \\boxed{0.5}", "1/2", 1.0),
    ("<reason>...</reason> \\boxed{2} then \\boxed{3}", "3", 1.0),  # the last \boxed{}
    ("<reason>Work</reason>42", "42", 1.0),
    ("\\boxed{C: the red object}", "C", 1.0),
]


@pytest.mark.parametrize("response, ground_truth, expected", CASES)
def test_accuracy_reward(response, ground_truth, expected):
    assert vgs.accuracy_reward(response, ground_truth) == expected


@pytest.mark.parametrize("response, ground_truth, expected", CASES)
def test_the_evaluation_reads_answers_as_the_reward(response, ground_truth, expected):
    """--answer-protocol vgs scores the \\boxed{} benchmarks with the reward's reading."""
    row = {"target": ground_truth, "responses": [response], "eval_metadata": {"answer_protocol": "vgs"}}
    assert boxed_row_answers(row, "mathvision")[0][1] == expected


def test_the_default_reading_is_unchanged():
    # PAPO-Eval's reading needs a \boxed{} and compares the whole answer
    for response in ("<reason>Work</reason>C", "\\boxed{C: the red object}"):
        assert boxed_row_answers({"target": "C", "responses": [response]}, "mathvision")[0][1] == 0.0


def test_compute_score_has_no_format_term():
    scores = vgs.compute_score([{"response": "<reason>x</reason> \\boxed{A}", "ground_truth": "A"}])
    assert scores == [{"overall": 1.0, "accuracy": 1.0}]
