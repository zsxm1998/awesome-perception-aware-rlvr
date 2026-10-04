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
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "vgs.py"
_SPEC = importlib.util.spec_from_file_location("vgs_reward", _MODULE_PATH)
vgs = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(vgs)


@pytest.mark.parametrize(
    "response, ground_truth, expected",
    [
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
    ],
)
def test_accuracy_reward(response, ground_truth, expected):
    assert vgs.accuracy_reward(response, ground_truth) == expected


def test_compute_score_has_no_format_term():
    scores = vgs.compute_score([{"response": "<reason>x</reason> \\boxed{A}", "ground_truth": "A"}])
    assert scores == [{"overall": 1.0, "accuracy": 1.0}]
