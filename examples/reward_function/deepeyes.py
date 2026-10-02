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

"""Controlled DeepEyes reward used by the EasyR1 reproduction.

Accuracy deliberately reuses the existing Qwen3-VL comparison scorer. Format
and conditional tool rewards follow the original DeepEyes release, while tool
eligibility comes from committed trajectory state rather than decoded visual
tokens.
"""

from __future__ import annotations

import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


REWARD_NAME = "deepeyes"
REWARD_TYPE = "batch"

_VISION_OPEN = "<|vision_start|><|image_pad|>"
_VISION_CLOSE = "<|image_pad|><|vision_end|>"
_MAX_ANSWER_CHARS = 1000


def _validated_final_answer(final_answer: Any) -> str | None:
    if final_answer is None:
        return None
    if not isinstance(final_answer, str):
        raise TypeError("final_answer must be a string or None")
    final_answer = final_answer.strip()
    return final_answer or None


def format_reward(response: str, final_answer: str | None) -> float:
    is_format_error = response.count("<think>") != response.count("</think>")
    is_format_error |= response.count(_VISION_OPEN) != response.count(_VISION_CLOSE)

    after_reasoning = response.split("</think>")[-1].strip()
    is_format_error |= after_reasoning.count("<answer>") != after_reasoning.count("</answer>")

    if final_answer is None or len(final_answer) >= _MAX_ANSWER_CHARS:
        is_format_error = True
    return -1.0 if is_format_error else 0.0


def accuracy_reward(final_answer: str | None, ground_truth: str) -> float:
    if final_answer is None:
        return 0.0
    try:
        return 1.0 if grade_answer(final_answer, str(ground_truth).strip()) else 0.0
    except Exception:
        return 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    *,
    accuracy_weight: float = 0.8,
    format_weight: float = 0.2,
    tool_weight: float = 1.2,
) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        # Agent trajectories must carry the parser-validated answer explicitly.
        # ``None`` is meaningful (the trajectory did not reach a valid answer),
        # whereas a missing key is an integration error and must fail closed.
        final_answer = _validated_final_answer(reward_input["final_answer"])
        accuracy = accuracy_reward(final_answer, reward_input["ground_truth"])
        format_score = format_reward(response, final_answer)
        committed_crops = reward_input.get("tool_call_successes", 0)
        if isinstance(committed_crops, bool) or not isinstance(committed_crops, int):
            raise TypeError("tool_call_successes must be an integer")
        if committed_crops < 0:
            raise ValueError("tool_call_successes cannot be negative")
        tool_score = 1.0 if accuracy == 1.0 and committed_crops > 0 else 0.0
        scores.append(
            {
                "overall": accuracy_weight * accuracy + format_weight * format_score + tool_weight * tool_score,
                "accuracy": accuracy,
                "format": format_score,
                "tool": tool_score,
                "tool_call_successes": float(committed_crops),
                "tool_call_attempts": float(reward_input.get("tool_call_attempts", 0)),
                "tool_call_errors": float(reward_input.get("tool_call_errors", 0)),
                "tool_internal_errors": float(reward_input.get("tool_internal_errors", 0)),
                "image_idx_errors": float(reward_input.get("image_idx_errors", 0)),
                "invalid_final_answers": float(reward_input.get("invalid_final_answers", 0)),
                "turn_end_errors": float(reward_input.get("turn_end_errors", 0)),
                "tool_calls_inside_reasoning": float(reward_input.get("tool_calls_inside_reasoning", 0)),
                "unclosed_answer": float(bool(reward_input.get("unclosed_answer", False))),
                "no_action": float(bool(reward_input.get("no_action", False))),
                "truncated": float(bool(reward_input.get("truncated", False))),
                "trajectory_retries": float(reward_input.get("trajectory_retries", 0)),
            }
        )
    return scores


# ---------------------------------------------------------------------------
# Official-recipe reward used by examples/reproduction/deepeyes (DeepEyes-Datasets-47k).
#
# Released DeepEyes code (verl/utils/reward_score/vl_agent.py) routes by data source:
#   vstar / chart:     0.8 * acc + 0.2 * format(0 / -1) + 1.2 * tool   (tool only if correct)
#   thinklite_eureka:  1.2 * acc + 0.4 * format(0 / -1)                (no tool bonus)
# where acc is decided by a Qwen2.5-72B-Instruct judge. Set DEEPEYES_JUDGE_BASE_URL
# (OpenAI-compatible, e.g. a vLLM server), DEEPEYES_JUDGE_MODEL and optionally
# DEEPEYES_JUDGE_API_KEY to use a judge; otherwise a rule-based matcher is used.
# ---------------------------------------------------------------------------


_JUDGE_PROMPT = """You are an expert evaluator. Given a question, a reference answer and a model's prediction, decide whether the prediction is consistent with the reference answer. Answer with "Judgement: 1" if it is consistent and "Judgement: 0" otherwise.

Question: {question}
Reference answer: {reference}
Prediction: {prediction}

Judgement:"""
_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans(dict.fromkeys(string.punctuation, " "))


def _words(text: str) -> list[str]:
    return [word for word in str(text).lower().translate(_PUNCT_TABLE).split() if word not in _ARTICLES]


def _rule_match(prediction: str, reference: str) -> bool:
    prediction, reference = prediction.strip(), str(reference).strip()
    if not prediction or not reference:
        return False
    boxed = extract_boxed_content(prediction)
    if boxed and boxed != "None":
        prediction = boxed
    letter = re.fullmatch(r"\(?([A-Ha-h])\)?[.:)]?(\s.*)?", prediction)
    if re.fullmatch(r"[A-Ha-h]", reference):  # multiple-choice reference (chart data)
        return letter is not None and letter.group(1).upper() == reference.upper()
    pred_words, ref_words = _words(prediction), _words(reference)
    if not pred_words or not ref_words:
        return False
    if pred_words == ref_words:
        return True
    if ref_words[0] in {"yes", "no"}:  # sentence answers such as "No, the car is not ..."
        decision = next((word for word in pred_words if word in {"yes", "no"}), None)
        return decision == ref_words[0]
    if " ".join(ref_words) in " ".join(pred_words):
        return True
    # short predictions ("brown") against sentence references ("The puppy is brown.")
    if len(pred_words) <= 4 and all(word in ref_words for word in pred_words):
        return True
    try:
        return bool(grade_answer(prediction, reference))
    except Exception:
        return False


def _judge_client():
    base_url = os.environ.get("DEEPEYES_JUDGE_BASE_URL")
    if not base_url:
        return None
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=os.environ.get("DEEPEYES_JUDGE_API_KEY", "EMPTY"))


def _judge_match(client, question: str, prediction: str, reference: str) -> bool:
    model = os.environ.get("DEEPEYES_JUDGE_MODEL", "judge")
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": _JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction),
                    }
                ],
                temperature=0.3,
                max_tokens=16,
            )
            text = completion.choices[0].message.content or ""
            return "1" in text.split("Judgement:")[-1][:8]
        except Exception:
            continue
    return _rule_match(prediction, reference)


def _math_format_reward(response: str, final_answer: str | None) -> float:
    is_format_error = response.count("<think>") != response.count("</think>")
    if final_answer is None or len(final_answer) >= _MAX_ANSWER_CHARS:
        is_format_error = True
    return -1.0 if is_format_error else 0.0


def compute_score_official(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    client = _judge_client()
    finals = [_validated_final_answer(reward_input["final_answer"]) for reward_input in reward_inputs]

    def decide(index: int) -> float:
        reward_input, final_answer = reward_inputs[index], finals[index]
        if final_answer is None:
            return 0.0
        reference = str(reward_input["ground_truth"])
        is_math = reward_input.get("data_source", "") == "thinklite_eureka"
        if is_math or client is None:
            if is_math:
                boxed = extract_boxed_content(final_answer)
                candidate = boxed if boxed and boxed != "None" else final_answer
                try:
                    if grade_answer(candidate, reference):
                        return 1.0
                except Exception:
                    pass
                if client is None:
                    return 0.0
            else:
                return 1.0 if _rule_match(final_answer, reference) else 0.0
        return 1.0 if _judge_match(client, reward_input.get("question", ""), final_answer, reference) else 0.0

    workers = int(os.environ.get("DEEPEYES_JUDGE_WORKERS", "32")) if client is not None else 1
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        accuracies = list(pool.map(decide, range(len(reward_inputs))))

    base_scores = compute_score(reward_inputs)
    scores = []
    for reward_input, final_answer, accuracy, base in zip(reward_inputs, finals, accuracies, base_scores):
        response = reward_input["response"]
        if reward_input.get("data_source", "") == "thinklite_eureka":
            format_score = _math_format_reward(response, final_answer)
            overall = 1.2 * accuracy + 0.4 * format_score
            tool_score = 0.0
        else:
            format_score = format_reward(response, final_answer)
            tool_score = 1.0 if accuracy == 1.0 and int(reward_input.get("tool_call_successes", 0)) > 0 else 0.0
            overall = 0.8 * accuracy + 0.2 * format_score + 1.2 * tool_score
        scores.append({**base, "overall": overall, "accuracy": accuracy, "format": format_score, "tool": tool_score})
    return scores


_ANSWER_SPAN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def compute_score_text_only(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Same reward without tools, for the "RL with text-only CoT" baseline of the paper."""
    adapted = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        after_reasoning = response.split("</think>")[-1]
        matches = _ANSWER_SPAN.findall(after_reasoning)
        final_answer = matches[-1].strip() if matches else None
        adapted.append({**reward_input, "final_answer": final_answer or None, "tool_call_successes": 0})
    return compute_score_official(adapted)
