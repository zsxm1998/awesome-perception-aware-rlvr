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

The VGS prompt asks for the reasoning in <reason></reason> and the final answer in \\boxed{}; the paper's sample
outputs often give the answer right after </reason> instead. The answer is the last \\boxed{} of the response, or
the text after the last </reason> when there is no \\boxed{}. Multiple-choice answers (a single letter in the
data) compare option letters; other answers are compared with mathruler (numbers, expressions, strings). The
paper only mentions a correctness reward, so there is no format term.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "vgs"
REWARD_TYPE = "batch"

_LETTER = re.compile(r"^\s*(?:\(\s*([A-Z])\s*\)|([A-Z]))(?:\s*$|[.):\s])")
# the rest of an answer that only lists further options: `and C`, `, C`, `or (C)`
_MORE_LETTERS = re.compile(r"^(?:\s*(?:and|or|,|&|/)\s*\(?[A-Z]\)?)+\s*\.?\s*$")


def extract_answer(response: str) -> str:
    boxed = extract_boxed_content(response)
    if boxed != "None":
        return boxed.strip()
    if "</reason>" in response:
        return response.rsplit("</reason>", 1)[1].strip()
    return ""


def option_letter(answer: str) -> str | None:
    """The option letter an answer starts with: `C`, `(C)`, `C.`, `C)`, `C: text`; None when there is none or
    when the answer lists several options (`A and C`)."""
    answer = answer.replace("\\text{", "").replace("}", "").strip()
    match = _LETTER.match(answer)
    if match is None:
        return None
    if _MORE_LETTERS.match(answer[match.end() :]):
        return None
    return match.group(1) or match.group(2)


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_answer(response)
    if not answer:
        return 0.0
    ground_truth = ground_truth.strip()
    if re.fullmatch(r"[A-Z]", ground_truth):
        return 1.0 if option_letter(answer) == ground_truth else 0.0
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        accuracy = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
        scores.append({"overall": accuracy, "accuracy": accuracy})
    return scores
