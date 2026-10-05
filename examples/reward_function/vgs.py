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
"""Correctness reward of the VGS reproduction (examples/reproduction/vgs): 1 if the final answer matches, else 0.

The answer is read as verl/utils/vgs_answer.py describes (the last \\boxed{}, else the text after </reason>; option
letters for multiple-choice answers, mathruler otherwise), the same reading as the evaluation's
``--answer-protocol vgs``. The paper only mentions a correctness reward, so there is no format term.
"""

from typing import Any

from verl.utils.vgs_answer import answer_correct


# Metadata
REWARD_NAME = "vgs"
REWARD_TYPE = "batch"


def accuracy_reward(response: str, ground_truth: str) -> float:
    return 1.0 if answer_correct(response, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        accuracy = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
        scores.append({"overall": accuracy, "accuracy": accuracy})
    return scores
