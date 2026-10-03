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

import re
import warnings
from typing import Any

from mathruler.grader import grade_answer
from nltk.translate.bleu_score import sentence_bleu

from verl.trainer.perception_reasoning_data import parse_json_grounding_regions


REWARD_NAME = "grit"
REWARD_TYPE = "batch"

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_RETHINK_OPEN = "<rethink>"
_RETHINK_CLOSE = "</rethink>"
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"
_RAW_BBOX_PATTERN = re.compile(r"\b\d+,\s*\d+,\s*\d+,\s*\d+\b")
_ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
_NON_ALNUM_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")


def _ordered_once(response: str, tokens: tuple[str, ...]) -> bool:
    cursor = -1
    for token in tokens:
        if response.count(token) != 1:
            return False
        cursor = response.find(token, cursor + 1)
        if cursor == -1:
            return False
    return True


def _pre_rethink_text(response: str) -> str:
    rethink_idx = response.find(_RETHINK_OPEN)
    return response[:rethink_idx] if rethink_idx != -1 else response


def parse_grit_bbox_groups(text: str, num_images: int | None = None) -> list[dict[str, Any]]:
    return [
        {"label": region.name, "image_idx": region.image_idx, "bbox_list": region.boxes}
        for region in parse_json_grounding_regions(text, num_images=num_images)
    ]


def format_reward(response: str, num_images: int | None = None) -> dict[str, float]:
    think_score = 0.5 if _ordered_once(response, (_THINK_OPEN, _THINK_CLOSE)) else 0.0
    rethink_score = 0.5 if _ordered_once(response, (_RETHINK_OPEN, _RETHINK_CLOSE)) else 0.0

    pre_rethink = _pre_rethink_text(response)
    raw_bbox_score = 0.25 if _RAW_BBOX_PATTERN.search(pre_rethink) is not None else 0.0
    parseable_bbox_score = 0.25 if parse_grit_bbox_groups(pre_rethink, num_images=num_images) else 0.0
    bbox_score = raw_bbox_score + parseable_bbox_score

    format_score = think_score + rethink_score + bbox_score
    return {
        "format": format_score,
        "format_structure": think_score + rethink_score,
        "format_bbox": bbox_score,
        "format_raw_bbox": raw_bbox_score,
        "format_parseable_bbox": parseable_bbox_score,
    }


def extract_answer(response: str) -> str | None:
    match = _ANSWER_PATTERN.search(response)
    return match.group("answer").strip() if match is not None else None


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_answer(response)
    if answer is None:
        return 0.0
    try:
        return 1.0 if grade_answer(answer, str(ground_truth).strip()) else 0.0
    except Exception:
        return 0.0


def bleu1_reward(prediction: str, ground_truth: str) -> float:
    cleaned_prediction = _NON_ALNUM_PATTERN.sub(" ", prediction)
    cleaned_ground_truth = _NON_ALNUM_PATTERN.sub(" ", str(ground_truth))
    pred_tokens = cleaned_prediction.lower().split()
    gt_tokens = cleaned_ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"nltk\.translate\.bleu_score")
        return sentence_bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0))


def answer_reward(response: str, ground_truth: str) -> dict[str, float]:
    answer = extract_answer(response)
    if answer is None:
        return {
            "answer": 0.0,
            "accuracy": 0.0,
            "bleu1": 0.0,
        }
    accuracy_score = accuracy_reward(response, ground_truth)
    bleu_score = bleu1_reward(answer, ground_truth)
    return {
        "answer": accuracy_score + 0.1 * bleu_score,
        "accuracy": accuracy_score,
        "bleu1": bleu_score,
    }


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        format_scores = format_reward(response, num_images=reward_input.get("num_images"))
        answer_scores = answer_reward(response, ground_truth)
        scores.append(
            {
                "overall": format_scores["format"] + answer_scores["answer"],
                **format_scores,
                **answer_scores,
            }
        )
    return scores


# ---------------------------------------------------------------------------
# Official-recipe reward used by examples/reproduction/grit.
#
# Mirrors the reward set of the released GRIT code (grpo-gr/rewards.py, "think_rethink"
# setting), all terms summed with weight 1:
#   answer (binary)  + 0.1 * BLEU-1 + answer format (<= 0.5) + repetition (<= 0.5)
#   + grounded format (0.5 if >= 1 box before <rethink>, +1.0 if #boxes == count answer)
#   + think/rethink structure (<= 0.5).
# The official answer term is a GPT-4o judgement; here it is replaced by a rule-based
# normalized match so that training needs no API access.
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": "0",
    "none": "0",
    "no": "no",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}


def _official_answer_text(response: str) -> str | None:
    if _ANSWER_OPEN not in response:
        return None
    answer = response.split(_ANSWER_OPEN, 1)[1]
    return answer.split(_ANSWER_CLOSE, 1)[0].strip()


def _normalize_words(text: str) -> list[str]:
    words = _NON_ALNUM_PATTERN.sub(" ", str(text).lower()).split()
    return [_NUMBER_WORDS.get(word, word) for word in words if word not in _ARTICLES]


def _answers_match(prediction: str, ground_truth: str) -> bool:
    pred_words = _normalize_words(prediction)
    gt_words = _normalize_words(ground_truth)
    if not pred_words or not gt_words:
        return False
    if pred_words == gt_words:
        return True
    if len(gt_words) == 1 and gt_words[0] in {"yes", "no"}:
        # yes/no answers: the first yes/no word of the prediction decides
        decision = next((word for word in pred_words if word in {"yes", "no"}), None)
        return decision == gt_words[0]
    if len(gt_words) == 1 and gt_words[0].isdigit():
        # counting answers ("There are 9 dogs."): the first number of the prediction decides
        number = next((word for word in pred_words if word.isdigit()), None)
        return number is not None and int(number) == int(gt_words[0])
    # short open-ended answers ("hot dog"): accept the GT phrase inside a short prediction
    gt_phrase, pred_phrase = " ".join(gt_words), " ".join(pred_words)
    if len(pred_words) <= len(gt_words) + 3 and f" {gt_phrase} " in f" {pred_phrase} ":
        return True
    try:
        return bool(grade_answer(prediction.strip(), str(ground_truth).strip()))
    except Exception:
        return False


_GRIT_PAD_TOKEN_ID = 151643  # GRIT's repetitive_reward cuts the completion ids at this (Qwen) pad id


def _repetition_reward(
    completion: str, completion_ids: list[int] | None, ngram_size: int = 8, max_reward: float = 0.5
) -> float:
    """GRIT's repetitive_reward (grpo-gr/rewards.py): the share of 8-grams repeated right after themselves, in
    words and in token ids; (1 - token share - word share) * 0.5, so it can be negative. Without token ids
    only the word share counts."""
    if completion == "" or len(completion.split()) < ngram_size:
        return max_reward
    tokens = completion.split()
    repeat_count, total = 0, 0
    for i in range(len(tokens) - ngram_size):
        total += 1
        if tuple(tokens[i : i + ngram_size]) == tuple(tokens[i + ngram_size : i + 2 * ngram_size]):
            repeat_count += 1
    word_score = 1.0 if total == 0 else 1.0 - repeat_count / total
    if completion_ids is None:
        return word_score * max_reward

    ids = list(completion_ids)
    if _GRIT_PAD_TOKEN_ID in ids:
        ids = ids[: ids.index(_GRIT_PAD_TOKEN_ID)]
    if len(ids) < 2 * ngram_size:
        return max_reward
    repeat_count, total = 0, 0
    for i in range(len(ids) - 2 * ngram_size + 1):
        total += 1
        if tuple(ids[i : i + ngram_size]) == tuple(ids[i + ngram_size : i + 2 * ngram_size]):
            repeat_count += 1
    token_score = 1.0 if total == 0 else 1.0 - repeat_count / total
    return (token_score - (1.0 - word_score)) * max_reward


def _think_rethink_structure(response: str, max_reward: float = 0.5) -> float:
    """GRIT's think_and_rethink_format_reward: one point per tag of <think>, </think>, <rethink>, </rethink>
    found in order (each searched after the last occurrence of the previous one; a missing tag is skipped),
    one more when all four are present and the think part has at least two words; scaled to 0.5."""
    tags = (_THINK_OPEN, _THINK_CLOSE, _RETHINK_OPEN, _RETHINK_CLOSE)
    reward, remaining = 0.0, response
    for tag in tags:
        if tag in remaining:
            reward += 1.0
            remaining = remaining.split(tag)[-1]
    if reward == len(tags):
        think = response.split(_THINK_OPEN)[-1].split(_THINK_CLOSE)[0].strip()
        if len(think) > 1 and len(_NON_ALNUM_PATTERN.sub(" ", think).split(" ")) > 1:
            reward += 1.0
    return reward / (len(tags) + 1) * max_reward


def compute_score_official(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = str(reward_input["ground_truth"]).strip()
        answer = _official_answer_text(response)

        answer_correct = 1.0 if answer is not None and _answers_match(answer, ground_truth) else 0.0
        bleu = bleu1_reward(answer, ground_truth) if answer else 0.0
        answer_format = (0.25 if answer is not None else 0.0) + (0.25 if response.count(_ANSWER_OPEN) == 1 else 0.0)
        repetition = _repetition_reward(response, reward_input.get("response_ids"))

        grounded = 0.0
        if _RETHINK_OPEN in response:
            boxes = _RAW_BBOX_PATTERN.findall(_pre_rethink_text(response))
            if boxes:
                grounded += 0.5
            if ground_truth.isdigit() and len(boxes) == int(ground_truth):
                grounded += 1.0
        structure = _think_rethink_structure(response)

        overall = answer_correct + 0.1 * bleu + answer_format + repetition + grounded + structure
        scores.append(
            {
                "overall": overall,
                "accuracy": answer_correct,
                "bleu1": bleu,
                "format": answer_format + structure,
                "grounded_format": grounded,
                "repetition": repetition,
            }
        )
    return scores
