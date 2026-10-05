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

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]], format_weight: float = 0.1, perception_weight: float = 0.0
) -> list[dict[str, float]]:
    """(1 - format_weight) * accuracy + format_weight * format. With `perception_weight` > 0 (VAPO, with
    algorithm.claim_probe_count > 0), a response that carries the claim probes' `perception_score` gets
    (1 - format_weight - perception_weight) * accuracy + format_weight * format
    + perception_weight * 1[accuracy = 1] * perception_score, as the released VAPO reward."""
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        score = {
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        }
        if perception_weight > 0.0:
            perception = reward_input.get("perception_score")
            if perception is not None:
                score["overall"] = (
                    (1 - format_weight - perception_weight) * accuracy_score
                    + format_weight * format_score
                    + perception_weight * (perception if accuracy_score == 1.0 else 0.0)
                )
            score["perception"] = 0.0 if perception is None else perception  # 0 for responses not probed
            score["perception_scored"] = float(perception is not None)
        scores.append(score)

    return scores


def compute_score_wo_format(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": accuracy_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
