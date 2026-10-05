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
"""The answer reading of the VGS reproduction, shared by its training reward
(examples/reward_function/vgs.py) and the evaluation (``--answer-protocol vgs``).

The VGS prompt asks for the reasoning in <reason></reason> and the final answer in \\boxed{}; the paper's sample
outputs often give the answer right after </reason> instead. The answer is the last \\boxed{} of the response, or
the text after the last </reason> when there is no \\boxed{}. Multiple-choice answers (a single letter in the
data) compare option letters; other answers are compared with mathruler (numbers, expressions, strings).
"""

import re
from typing import Optional

from mathruler.grader import extract_boxed_content, grade_answer


_LETTER = re.compile(r"^\s*(?:\(\s*([A-Z])\s*\)|([A-Z]))(?:\s*$|[.):\s])")
# the rest of an answer that only lists further options: `and C`, `, C`, `or (C)`
_MORE_LETTERS = re.compile(r"^(?:\s*(?:and|or|,|&|/)\s*\(?[A-Z]\)?)+\s*\.?\s*$")


def extract_answer(response: str) -> tuple[str, str]:
    """(answer, source): the last \\boxed{} ("boxed"), else the text after the last </reason> ("after_reason"),
    else ("", "none")."""
    boxed = extract_boxed_content(response)
    if boxed != "None":
        return boxed.strip(), "boxed"
    if "</reason>" in response:
        return response.rsplit("</reason>", 1)[1].strip(), "after_reason"
    return "", "none"


def option_letter(answer: str) -> Optional[str]:
    """The option letter an answer starts with: `C`, `(C)`, `C.`, `C)`, `C: text`; None when there is none or
    when the answer lists several options (`A and C`)."""
    answer = answer.replace("\\text{", "").replace("}", "").strip()
    match = _LETTER.match(answer)
    if match is None:
        return None
    if _MORE_LETTERS.match(answer[match.end() :]):
        return None
    return match.group(1) or match.group(2)


def answer_matches(answer: str, ground_truth: str) -> bool:
    """Whether an extracted answer matches the reference: option letters for a single-letter reference, mathruler
    otherwise."""
    if not answer:
        return False
    ground_truth = ground_truth.strip()
    if re.fullmatch(r"[A-Z]", ground_truth):
        return option_letter(answer) == ground_truth
    return bool(grade_answer(answer, ground_truth))


def answer_correct(response: str, ground_truth: str) -> bool:
    return answer_matches(extract_answer(response)[0], ground_truth)
