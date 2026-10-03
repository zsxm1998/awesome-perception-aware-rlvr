# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""PEPO's reward (xzxxntxdy/PEPO, src/pepo/rewards/plugin.py), used by examples/reproduction/pepo.

format: exactly one <think>...</think> and one <answer>...</answer> pair (text outside them is allowed);
accuracy: the <answer> content and the ground truth, both NFKC-normalized, case-folded, with collapsed spaces,
no trailing period and π written as \\pi, compared with mathruler's grade_answer.
"""

import re
import unicodedata
from typing import Any

from mathruler.grader import grade_answer


# Metadata
REWARD_NAME = "r1v"
REWARD_TYPE = "sequential"

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def format_reward(response: str) -> float:
    return 1.0 if len(THINK_PATTERN.findall(response)) == 1 and len(ANSWER_PATTERN.findall(response)) == 1 else 0.0


def normalize_answer(answer: str) -> str:
    text = unicodedata.normalize("NFKC", str(answer)).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(".") or text.endswith("\u3002"):
        text = text[:-1].strip()
    return text.replace("\u03c0", "\\pi")


def accuracy_reward(response: str, ground_truth: str) -> float:
    match = ANSWER_PATTERN.search(response)
    if not match:
        return 0.0
    try:  # argument order as in PEPO's plugin
        return 1.0 if grade_answer(normalize_answer(ground_truth), normalize_answer(match.group(1).strip())) else 0.0
    except Exception:
        return 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
