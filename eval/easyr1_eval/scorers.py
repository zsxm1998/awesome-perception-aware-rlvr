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
from __future__ import annotations

import http.client
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .grounding_metrics import (
    Box,
    BoxSetMetrics,
    box_iou,
    intersection_over_target,
    mean_target_coverage_by_regions,
    score_box_sets,
    validate_normalized_boxes,
)
from .json_utils import read_jsonl, write_json, write_jsonl
from .schemas import (
    AGENT_OUTPUT_CONTRACT_NATIVE,
    SKIPPED_STATUS,
    BenchmarkSpec,
    MetricResult,
)


DEFAULT_DEEPSEEK_JUDGE_MODEL = "deepseek-v4-flash"
SCORER_VERSION = 22
TOOL_CROP_IOGT_THRESHOLD = 0.5
MM_VET_JUDGE_VERSION = "official_full_prediction_v1"
DEFAULT_JUDGE_MAX_TOKENS_DISABLED_THINKING = 32
DEFAULT_JUDGE_MAX_TOKENS_ENABLED_THINKING = 1024

# GQA is scored with the official normalized exact match by default. Exact match
# under-credits semantically correct answers (synonyms, morphology), so when an LLM judge
# is configured (--judge-provider) the score becomes an EM -> LLM-judge cascade: exact
# matches are accepted as-is and exact-match-wrong answers are judged for semantic
# equivalence (LAVE-style; the judge never sees the image and trusts the reference).
# The plain exact-match accuracy is always kept in the details. The judge protocol is
# versioned: changing the prompt requires bumping GQA_JUDGE_VERSION. Verdicts are cached
# globally, keyed by (question, reference, normalized answer, judge model), so identical
# answers get identical verdicts across runs and re-scoring is free.
GQA_JUDGE_VERSION = "gqa_semantic_v2"
GQA_JUDGE_CACHE_PATH = Path(__file__).resolve().parents[1] / "results" / "_judge_cache" / f"{GQA_JUDGE_VERSION}.jsonl"
GQA_PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."
GQA_JUDGE_PARSE_RETRIES = 3
_GQA_VERDICT_RE = re.compile(r"([01])\s*\|\s*([a-z\-]+)")

GQA_JUDGE_PROMPT = """You are grading answers for a visual question answering benchmark (GQA). The image is not shown; assume the reference answer is correct for the question. Decide whether the model's answer is semantically equivalent to the reference answer in the context of the question.

Count as CORRECT (verdict 1):
- A synonym or paraphrase of the reference ("couch" vs "sofa", "man" vs "guy").
- The reference plus extra non-contradicting modifiers ("black cat" when reference is "cat").
- The same number written as digits or words ("2" vs "two").
- A morphological variant (singular/plural, "walking" vs "walk").
- A strictly more specific term within the reference category ("terrier" when reference is "dog").

Count as WRONG (verdict 0):
- A different object, attribute, color, direction, material, or yes/no value.
- A more general term than the reference ("animal" when reference is "cat").
- An answer that drops a distinguishing modifier the reference has ("brown" when reference is "light brown").
- An answer that repeats only part of a multi-word reference ("picture" when reference is "picture frame").
- Person terms of a different age or gender category ("woman" when reference is "girl", "boy" when reference is "man").
- Hedged or multiple alternative answers ("cat or dog").
- Empty, evasive, or unrelated text.

Question: {question}
Reference answer: {gold}
Model answer: {answer}

Reply with exactly one line: the verdict (1 or 0), then "|", then a category. Categories for verdict 1: synonym, modifier, number, morph, specific. For verdict 0 use "-".
Example replies: 1|synonym
0|-"""


PredictionRows = list[dict[str, Any]]
Scorer = Callable[[BenchmarkSpec, PredictionRows, "JudgeConfig | None", Path], MetricResult]


@dataclass(frozen=True)
class JudgeConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 32
    concurrency: int = 4
    request_retries: int = 5
    request_backoff_seconds: float = 2.0
    thinking: str | None = "disabled"


@dataclass(frozen=True)
class JudgeResult:
    score: float
    raw_content: str
    parse_method: str
    attempts: int
    raw_attempts: list[dict[str, Any]]
    model: str | None = None


@dataclass(frozen=True)
class AnswerBBoxSample:
    answer_correct: bool
    boxes: BoxSetMetrics
    parse_failed: bool
    missing_prediction: bool


@dataclass(frozen=True)
class MMVetJudgeTask:
    index: int
    sample_id: str
    question: str
    target: str
    full_prediction: str
    prediction: str
    cache_key: str
    capability: Any


MM_VET_PROMPT = """Compare the ground truth and prediction from AI models, to give a correctness score for the prediction. <AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Just complete the last space of the correctness score.
Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is x in the equation? | -1 <AND> -5 | x = 3 | 0.0
What is x in the equation? | -1 <AND> -5 | x = -1 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -5 or 5 | 0.5
What is x in the equation? | -1 <AND> -5 | x = -1 or x = -5 | 1.0
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you explain this meme? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
"""


def score_predictions(
    spec: BenchmarkSpec,
    prediction_path: Path,
    output_json: Path,
    *,
    judge_config: JudgeConfig | None = None,
    metric_metadata: dict[str, Any] | None = None,
) -> MetricResult:
    rows = read_jsonl(prediction_path)
    try:
        scorer = SCORERS[spec.scorer]
    except KeyError as exc:
        raise KeyError(f"unknown scorer for {spec.key}: {spec.scorer}") from exc
    result = scorer(spec, rows, judge_config, output_json.parent)
    result.details.update(generation_diagnostics(rows))
    if metric_metadata:
        result.metadata.update(metric_metadata)
    result.metadata.update(_metadata_from_existing_metric(output_json))
    result.metadata.update(_metadata_from_predictions(rows))
    _write_perturbation_sample_diagnostics(spec, rows, result, output_json.parent)
    write_json(output_json, {"result": result.__dict__})
    return result


def skipped_result(spec: BenchmarkSpec, reason: str) -> MetricResult:
    """Placeholder metric for a benchmark that could not be scored in this run (e.g. no judge)."""
    return MetricResult(
        benchmark=spec.key,
        group=spec.group,
        primary_metric=spec.primary_metric,
        raw_score=None,
        normalized_score_0_100=None,
        num_examples=0,
        status=SKIPPED_STATUS,
        details={"skip_reason": reason},
    )


def _metadata_from_existing_metric(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result = data.get("result")
    if not isinstance(result, dict):
        return {}
    metadata = result.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _metadata_from_predictions(rows: PredictionRows) -> dict[str, Any]:
    for row in rows:
        metadata = row.get("eval_metadata")
        if isinstance(metadata, dict):
            return dict(metadata)
    return {}


def is_native_agentic_row(row: dict[str, Any]) -> bool:
    metadata = row.get("eval_metadata")
    if not isinstance(metadata, dict):
        return False
    if metadata.get("interaction_mode") != "agentic":
        return False
    output_contract = metadata.get("agent_output_contract")
    if output_contract is None:
        raise ValueError(
            "agentic prediction is missing agent_output_contract metadata; "
            "rerun inference with the current agent runner before scoring"
        )
    return output_contract == AGENT_OUTPUT_CONTRACT_NATIVE


def committed_tool_boxes(
    row: dict[str, Any],
    *,
    response_index: int = 0,
) -> list[Box]:
    diagnostics = row.get("agent_diagnostics")
    if not isinstance(diagnostics, list) or response_index >= len(diagnostics):
        raise ValueError(
            f"{row.get('sample_id')}: native agentic prediction is missing aligned agent diagnostics; rerun inference"
        )
    agent = diagnostics[response_index]
    if not isinstance(agent, dict):
        raise ValueError(f"{row.get('sample_id')}: agent diagnostics entry {response_index} must be an object")
    if agent.get("status") == "sample_error":
        return []
    regions = agent.get("committed_tool_regions")
    if not isinstance(regions, list):
        raise ValueError(
            f"{row.get('sample_id')}: native agentic prediction predates "
            "committed_tool_regions diagnostics; rerun inference"
        )

    num_images = len(row.get("image_refs") or []) or 1
    if num_images != 1:
        raise ValueError(
            f"{row.get('sample_id')}: tool-region grounding currently requires "
            f"a single-image benchmark row, got {num_images}"
        )

    boxes: list[Box] = []
    for region_index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f"{row.get('sample_id')}: committed tool region {region_index} must be an object")
        image_idx = region.get("image_idx")
        # bbox_norm1000: the crop on the source image in 0-1000, the same for norm1000 and pixel
        # runs; predictions written before it existed only have bbox_2d, which was 0-1000 then.
        bbox = region.get("bbox_norm1000", region.get("bbox_2d"))
        if image_idx != 0:
            raise ValueError(
                f"{row.get('sample_id')}: committed tool region {region_index} "
                f"references image_idx={image_idx} in a single-image benchmark"
            )
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in (bbox or [])
            )
        ):
            raise ValueError(
                f"{row.get('sample_id')}: committed tool region {region_index} has an invalid box: {bbox}"
            )
        normalized = tuple(float(value) / 1000.0 for value in bbox)
        validated = validate_normalized_boxes([normalized])
        if len(validated) != 1:
            raise ValueError(
                f"{row.get('sample_id')}: committed tool region {region_index} "
                f"is outside normalized 0-1000 xyxy bounds: {bbox}"
            )
        boxes.append(validated[0])
    return boxes


def _tool_evidence_sample_record(
    row: dict[str, Any],
    tool_boxes: list[Box],
) -> dict[str, Any]:
    gt_boxes = _ground_truth_boxes(row.get("extra_info") or {})
    hit_count = sum(
        max(
            (intersection_over_target(tool_box, gt_box) for gt_box in gt_boxes),
            default=0.0,
        )
        >= TOOL_CROP_IOGT_THRESHOLD
        for tool_box in tool_boxes
    )
    return {
        "tool_crop_count": len(tool_boxes),
        "tool_crop_iogt_hit_count_at_0_5": hit_count,
        "tool_crop_iogt_hit_rate_at_0_5": (hit_count / len(tool_boxes) if tool_boxes else None),
        "tool_gt_union_coverage": mean_target_coverage_by_regions(
            tool_boxes,
            gt_boxes,
        ),
        "tool_gt_count": len(gt_boxes),
    }


def _aggregate_tool_evidence_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    crop_count = sum(int(record["tool_crop_count"]) for record in records)
    hit_count = sum(int(record["tool_crop_iogt_hit_count_at_0_5"]) for record in records)
    coverages = [
        float(record["tool_gt_union_coverage"]) for record in records if record["tool_gt_union_coverage"] is not None
    ]
    return {
        "agentic/tool_crop_count": crop_count,
        "agentic/tool_crop_iogt_hit_count_at_0_5": hit_count,
        "agentic/tool_crop_iogt_hit_rate_at_0_5": (hit_count / crop_count if crop_count else None),
        "agentic/tool_gt_union_coverage": (sum(coverages) / len(coverages) if coverages else None),
        "agentic/tool_gt_union_coverage_sample_count": len(coverages),
        "agentic/tool_empty_gt_sample_count": len(records) - len(coverages),
    }


# ---------------------------------------------------------------------------
# Reasoning: PAPO-Eval rule-based exact match on the last \boxed{} (mean acc@k)
# ---------------------------------------------------------------------------

_PAPO_TAG_SPACE_RE = re.compile(r"\s*(<|>|/)\s*")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def boxed_answer(response: str) -> tuple[str, str]:
    """PAPO-Eval answer extraction. Returns ``(answer, source)``.

    Mirrors ``papo_eval/eval_utils.py``: whitespace around ``<``, ``>`` and ``/`` is removed
    and the content of the last ``\\boxed{}`` is taken with ``mathruler.grader.
    extract_boxed_content`` (``"None"`` when the box is missing or unbalanced). Responses
    that contain no ``\\boxed{`` at all fall back to the last ``<answer>...</answer>`` block,
    so models trained with an ``<answer>``-tag format prompt can be scored too; this never
    changes the score of a response that has a box.
    """
    from mathruler.grader import extract_boxed_content

    normalized = _PAPO_TAG_SPACE_RE.sub(r"\1", str(response))
    if "\\boxed{" in normalized:
        return extract_boxed_content(normalized), "boxed"
    tagged = _ANSWER_TAG_RE.findall(normalized)
    if tagged:
        return tagged[-1].strip(), "answer_tag"
    return "None", "none"


@lru_cache(maxsize=65536)
def _grade_answer_cached(answer: str, ground_truth: str) -> bool:
    from mathruler.grader import grade_answer

    return bool(grade_answer(answer, ground_truth))


def boxed_exact_match(response: str, ground_truth: Any) -> float:
    """1.0 when the extracted answer matches ``ground_truth`` under ``mathruler.grade_answer``."""
    answer, _ = boxed_answer(response)
    return 1.0 if _grade_answer_cached(answer, str(ground_truth).strip()) else 0.0


def boxed_row_answers(row: dict[str, Any], benchmark: str | None = None) -> list[tuple[str, float, str]]:
    """(answer, score, source) per response of a boxed_exact_match row, the one reading that the scorer, the run
    comparison and the perturbation diagnostics share. ``source`` is boxed, answer_tag, none, or pepo_letter for
    LogicVista items read with --answer-protocol pepo. ``benchmark`` defaults to the row's own."""
    target = str(row.get("target")).strip()
    benchmark = benchmark if benchmark is not None else str(row.get("benchmark") or "")
    pepo_letter = (
        benchmark == "logicvista" and _row_answer_protocol(row) == "pepo" and re.fullmatch(r"[A-Za-z]", target)
    )
    answers = []
    for response in row.get("responses") or []:
        if pepo_letter:
            letter = pepo_logicvista_letter(str(response))
            answers.append((letter or "", 1.0 if letter == target.upper() else 0.0, "pepo_letter"))
            continue
        answer, source = boxed_answer(str(response))
        answers.append((answer, 1.0 if _grade_answer_cached(answer, target) else 0.0, source))
    return answers


def boxed_row_scores(row: dict[str, Any], benchmark: str | None = None) -> list[float]:
    return [score for _, score, _ in boxed_row_answers(row, benchmark)]


def pepo_logicvista_letter(text: str) -> str | None:
    """PEPO's LogicVista reading (evaluate_logicvista.py: extract_answer, normalize_prediction_to_letter): the
    <answer> span (or the text after <answer>, or the whole response), unwrapped from \\boxed{}, then its last
    standalone letter."""
    span = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if span and span.group(1).strip():
        answer = span.group(1).strip()
    else:
        opening = re.search(r"<\s*answer\s*>", text, flags=re.IGNORECASE)
        answer = text[opening.end() :].strip() if opening else text.strip()
    boxed = re.search(r"\\boxed\s*\{(.*)\}\s*$", answer, flags=re.DOTALL)
    if boxed:
        answer = boxed.group(1).strip()
    if answer and answer[-1] in (".", "\u3002"):
        answer = answer[:-1].strip()
    answer = answer.replace("\u03c0", "\\pi")  # as extract_answer
    letters = re.findall(r"\b([A-Za-z])\b", answer)
    if letters:
        return letters[-1].upper()
    first = re.search(r"[A-Za-z]", answer)
    return first.group(0).upper() if first else None


def _row_answer_protocol(row: dict[str, Any]) -> str:
    metadata = row.get("eval_metadata")
    return str(metadata.get("answer_protocol") or "default") if isinstance(metadata, dict) else "default"


def score_boxed_exact_match(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    per_sample = []
    source_counts = {"boxed": 0, "answer_tag": 0, "none": 0}
    k_values = set()
    pepo_letter_rows = 0
    for row in rows:
        target = str(row.get("target")).strip()
        # --answer-protocol pepo: LogicVista items with a single-letter answer are read as PEPO reads them
        read = boxed_row_answers(row, spec.key)
        answers = [answer for answer, _, _ in read]
        scores = [score for _, score, _ in read]
        sources = [source for _, _, source in read]
        pepo_letter_rows += int("pepo_letter" in sources)
        for source in sources:
            if source in source_counts:
                source_counts[source] += 1
        k_values.add(len(scores))
        per_sample.append(
            {
                "sample_id": row.get("sample_id"),
                "target": target,
                "answers": answers,
                "scores": scores,
                "mean_accuracy": sum(scores) / len(scores) if scores else 0.0,
            }
        )
    count = len(per_sample)
    mean_acc = sum(item["mean_accuracy"] for item in per_sample) / count if count else 0.0
    pass_at_k = sum(any(score == 1.0 for score in item["scores"]) for item in per_sample) / count if count else 0.0
    raw = mean_acc * 100.0
    write_jsonl(output_dir / f"{spec.key}_per_sample.jsonl", per_sample)
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        raw,
        count,
        details={
            "scoring": "papo_eval_boxed_exact_match",
            "k": max(k_values) if k_values else 0,
            "mean_acc_at_k": mean_acc,
            "pass_at_k": pass_at_k,
            "answer_source_counts": source_counts,
            **({"pepo_letter_items": pepo_letter_rows} if pepo_letter_rows else {}),
        },
    )


# ---------------------------------------------------------------------------
# HallusionBench (yes/no; aAcc / fAcc / qAcc as in the official evaluation)
# ---------------------------------------------------------------------------


def hallusionbench_row(row: dict[str, Any]) -> bool:
    response = first_response(row)
    if is_truncated_without_final_answer(row, 0, response):
        return False
    return _parse_yes_no(response) == _parse_yes_no(str(row.get("target")))


def score_hallusionbench(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    figures: dict[str, list[bool]] = {}
    questions: dict[str, list[bool]] = {}
    categories: dict[str, list[bool]] = {}
    correct_flags = []
    per_sample = []
    for row in rows:
        meta = row.get("metadata") or {}
        correct = hallusionbench_row(row)
        correct_flags.append(correct)
        per_sample.append({"sample_id": row.get("sample_id"), "answer_correct": correct})
        category = str(meta.get("category"))
        categories.setdefault(category, []).append(correct)
        base = [category, str(meta.get("subcategory")), str(meta.get("set_id"))]
        if not (category == "VS" and str(meta.get("figure_id")) == "0"):
            figures.setdefault("_".join(base + [str(meta.get("figure_id"))]), []).append(correct)
        questions.setdefault("_".join(base + [str(meta.get("question_id"))]), []).append(correct)
    accuracy = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    raw = accuracy * 100.0
    write_jsonl(output_dir / f"{spec.key}_per_sample.jsonl", per_sample)
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        raw,
        len(rows),
        details={
            "question_accuracy": accuracy,
            "figure_accuracy": sum(all(items) for items in figures.values()) / len(figures) if figures else None,
            "question_pair_accuracy": sum(all(items) for items in questions.values()) / len(questions)
            if questions
            else None,
            "accuracy_by_category": {key: sum(items) / len(items) for key, items in sorted(categories.items())},
            "figure_count": len(figures),
            "question_pair_count": len(questions),
        },
    )


# ---------------------------------------------------------------------------
# Multiple choice (V*, HR-Bench): rule-based option-letter extraction
# ---------------------------------------------------------------------------


_EVIDENCE_OBJECT_RE = re.compile(r'\{\s*"label"\s*:[^{}]*\}')
_REGION_TAG_RE = re.compile(r"<region\b[^>]*>.*?</region>", flags=re.DOTALL | re.IGNORECASE)


def _norm_option_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(text).lower())).strip()


def extract_choice_letter(
    response: str, options: list[str] | None = None, *, num_options: int | None = None
) -> str | None:
    """Option letter chosen by ``response``, or None when it cannot be determined.

    The final answer is taken from ``\\boxed{}`` / ``<answer>`` / "answer is" wrappers first.
    Accepted forms: ``C``, ``C.``, ``(C)``, ``C) text``, ``C. text``, ``Option C``; a bare
    letter followed by text (``A cat ...``) only counts when the text matches that option.
    Otherwise the answer text is matched against the option texts (exact, then unique
    containment).
    """
    if options is not None and num_options is None:
        num_options = len(options)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: num_options or 26]
    text = extract_final_response_text(str(response)).strip()
    # Grounded answers may carry evidence objects / region tags next to the letter.
    text = _EVIDENCE_OBJECT_RE.sub(" ", text)
    text = _REGION_TAG_RE.sub(" ", text).strip()
    text = re.sub(
        r"^\s*(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*", "", text, flags=re.IGNORECASE
    )
    match = re.fullmatch(r"\(?([A-Za-z])\)?[.:)]?", text)
    if match and match.group(1).upper() in letters:
        return match.group(1).upper()
    match = re.match(r"^\(([A-Za-z])\)\s*(.*)$", text, flags=re.DOTALL) or re.match(
        r"^([A-Z])[.:)]\s*(.*)$", text, flags=re.DOTALL
    )
    if match and match.group(1).upper() in letters:
        return match.group(1).upper()
    normalized = _norm_option_text(text)
    match = re.match(r"^([A-Z])\s+(.+)$", text, flags=re.DOTALL)
    if match and match.group(1) in letters and options:
        index = letters.index(match.group(1))
        if index < len(options) and _norm_option_text(match.group(2)) == _norm_option_text(options[index]):
            return match.group(1)
    if options and normalized:
        option_norms = [_norm_option_text(option) for option in options]
        exact = [index for index, option in enumerate(option_norms) if option and option == normalized]
        if len(exact) == 1:
            return letters[exact[0]]
        contained = [
            index
            for index, option in enumerate(option_norms)
            if option and re.search(rf"(?<![a-z0-9]){re.escape(option)}(?![a-z0-9])", normalized)
        ]
        if len(contained) == 1:
            return letters[contained[0]]
    return None


def mcq_response_correct(row: dict[str, Any], response_index: int, response: str) -> bool:
    if is_truncated_without_final_answer(row, response_index, response):
        return False
    options = (row.get("extra_info") or {}).get("options")
    letter = extract_choice_letter(response, list(options) if options else None)
    return letter is not None and letter == str(row.get("target")).strip().upper()


def mcq_letter_correct(response: str, row: dict[str, Any]) -> bool:
    """Letter-based answer correctness for answer+bbox benchmarks with ``answer_style: mcq_letter``."""
    options = (row.get("extra_info") or {}).get("options")
    letter = extract_choice_letter(response, list(options) if options else None)
    return letter is not None and letter == str(row.get("target")).strip().upper()


def mcq_row_score(row: dict[str, Any]) -> float:
    scores = [
        float(mcq_response_correct(row, index, str(response)))
        for index, response in enumerate(row.get("responses") or [])
    ]
    return sum(scores) / len(scores) if scores else 0.0


def mcq_aggregate(scores: list[float], categories: list[str], mode: str) -> float:
    if not scores:
        return 0.0
    if mode == "category_mean":
        by_category: dict[str, list[float]] = {}
        for score, category in zip(scores, categories):
            by_category.setdefault(category, []).append(score)
        return sum(sum(items) / len(items) for items in by_category.values()) / len(by_category)
    return sum(scores) / len(scores)


def _mean_by_group(scores: list[float], groups: list[str]) -> dict[str, float]:
    by_group: dict[str, list[float]] = {}
    for score, group in zip(scores, groups):
        by_group.setdefault(group, []).append(score)
    return {group: sum(items) / len(items) for group, items in sorted(by_group.items())}


def score_mcq(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    mode = str(spec.metadata.get("aggregate", "micro"))
    if mode not in {"micro", "category_mean"}:
        raise ValueError(f"{spec.key}: unsupported mcq aggregate {mode!r}")
    scores, categories, per_sample = [], [], []
    unparsed = 0
    response_count = 0
    for row in rows:
        options = (row.get("extra_info") or {}).get("options")
        letters = []
        for response in row.get("responses") or []:
            letter = extract_choice_letter(str(response), list(options) if options else None)
            letters.append(letter)
            unparsed += int(letter is None)
            response_count += 1
        score = mcq_row_score(row)
        category = str((row.get("metadata") or {}).get("category") or "all")
        scores.append(score)
        categories.append(category)
        per_sample.append(
            {
                "sample_id": row.get("sample_id"),
                "target": row.get("target"),
                "letters": letters,
                "mean_accuracy": score,
                "category": category,
            }
        )
    accuracy_by_category = _mean_by_group(scores, categories)
    tasks = [str((row.get("metadata") or {}).get("task") or "") for row in rows]
    raw = mcq_aggregate(scores, categories, mode) * 100.0
    write_jsonl(output_dir / f"{spec.key}_per_sample.jsonl", per_sample)
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        raw,
        len(rows),
        details={
            "aggregate": mode,
            "micro_accuracy": sum(scores) / len(scores) if scores else 0.0,
            **({"accuracy_by_task": _mean_by_group(scores, tasks)} if any(tasks) else {}),
            "accuracy_by_category": accuracy_by_category,
            "unparsed_response_rate": unparsed / response_count if response_count else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# CFPO real-world benchmarks (C-VQA-Real, MARS-Bench, TextVQA)
# ---------------------------------------------------------------------------


def cfpo_extract_answer(response: str) -> str:
    """Answer extraction of CFPO's ``Counterfactual-Eval/inference_eval.py`` (non-PAPO models).

    The content of the last ``\\boxed{}`` (``mathruler.grader.extract_boxed_content``); when the
    response has no box, the whole response is graded.
    """
    from mathruler.grader import extract_boxed_content

    answer = extract_boxed_content(str(response))
    return str(response).strip() if answer == "None" else answer


def cfpo_correct(answer: str, target: Any) -> bool:
    """CFPO grading: ``mathruler.grader.grade_answer`` against the reference (any reference of a
    list counts), retrying the part before a ``:`` and mapping ``No.`` to ``No`` like CFPO."""
    if isinstance(target, list):
        for reference in target:
            correct = _grade_answer_cached(answer, str(reference))
            if not correct and ":" in answer:
                correct = _grade_answer_cached(answer.split(":")[0].strip(), str(reference))
            if correct:
                return True
        return False
    correct = _grade_answer_cached(answer, str(target))
    if not correct:
        if ":" in answer:
            correct = _grade_answer_cached(answer.split(":")[0].strip(), str(target))
        elif answer == "No.":
            correct = _grade_answer_cached("No", str(target))
    return correct


_VQA_PUNCTUATION_RE = re.compile(r"(?<!\d)[.,](?!\d)|[;/\\\\\[\]\"'{}()<>@`?!*_=+^&$#%|~-]")


def vqa_normalize(text: str) -> str:
    """Light version of the VQA answer normalization (case, punctuation, articles, number words)."""
    text = _VQA_PUNCTUATION_RE.sub(" ", str(text).lower().replace("\n", " "))
    tokens = [_NUMBER_WORDS.get(token, token) for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def vqa_accuracy(answer: str, references: list[Any]) -> float:
    """Standard VQA accuracy ``min(#human answers equal to the prediction / 3, 1)``."""
    normalized = vqa_normalize(answer)
    matches = sum(vqa_normalize(str(reference)) == normalized for reference in references)
    return min(matches / 3.0, 1.0)


def cfpo_row_scores(row: dict[str, Any]) -> list[float]:
    target = row.get("target")
    return [float(cfpo_correct(cfpo_extract_answer(str(response)), target)) for response in row.get("responses") or []]


def score_cfpo_match(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    """Mean accuracy@k with CFPO's grading, plus CFPO's breakdowns (cf/ncf, question type)."""
    per_sample = []
    groups: dict[str, list[float]] = {}
    vqa_scores: list[float] = []
    for row in rows:
        target = row.get("target")
        answers = [cfpo_extract_answer(str(response)) for response in row.get("responses") or []]
        scores = [float(cfpo_correct(answer, target)) for answer in answers]
        mean = sum(scores) / len(scores) if scores else 0.0
        metadata = row.get("metadata") or {}
        cf = "cf" if metadata.get("is_cf") else "ncf"
        question_type = str(metadata.get("type") or "unknown")
        for key in (cf, question_type, f"{question_type}_{cf}"):
            groups.setdefault(key, []).append(mean)
        record = {"sample_id": row.get("sample_id"), "answers": answers, "scores": scores, "mean_accuracy": mean}
        if isinstance(target, list):
            vqa = [vqa_accuracy(answer, target) for answer in answers]
            record["vqa_accuracy"] = sum(vqa) / len(vqa) if vqa else 0.0
            vqa_scores.append(record["vqa_accuracy"])
        per_sample.append(record)
    count = len(per_sample)
    mean_acc = sum(item["mean_accuracy"] for item in per_sample) / count if count else 0.0
    details: dict[str, Any] = {
        "scoring": "cfpo_counterfactual_eval",
        "k": max((len(item["scores"]) for item in per_sample), default=0),
        "mean_acc_at_k": mean_acc,
        "accuracy_by_group": {key: sum(values) / len(values) for key, values in sorted(groups.items())},
    }
    if vqa_scores:
        details["vqa_accuracy"] = sum(vqa_scores) / len(vqa_scores)
    raw = mean_acc * 100.0
    write_jsonl(output_dir / f"{spec.key}_per_sample.jsonl", per_sample)
    return MetricResult(spec.key, spec.group, spec.primary_metric, raw, raw, count, details=details)


# ---------------------------------------------------------------------------
# Grounding (GRIT protocol): answer accuracy + GRIT grounding IoU
# ---------------------------------------------------------------------------

BOX_FORMATS = ("norm1000", "pixel")


def row_box_format(row: dict[str, Any]) -> str:
    """Coordinate convention of the predicted boxes of a prediction row (runner --box-format)."""
    metadata = row.get("eval_metadata")
    value = str(metadata.get("box_format") or "norm1000") if isinstance(metadata, dict) else "norm1000"
    return value if value in BOX_FORMATS else "norm1000"


def model_input_size(row: dict[str, Any], image_idx: int = 0) -> tuple[float, float] | None:
    """(width, height) of the image the model saw, recorded by the backend at inference time."""
    sizes = row.get("image_sizes")
    if not isinstance(sizes, list) or image_idx >= len(sizes) or not isinstance(sizes[image_idx], dict):
        return None
    value = sizes[image_idx].get("model_input")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    width, height = (float(item) for item in value)
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        return None
    return width, height


def predicted_boxes(text: str, row: dict[str, Any]) -> list[Box]:
    """Boxes written in ``text``, normalized to [0, 1] according to the row's box format.

    ``norm1000``: ``[x1, y1, x2, y2]`` in 0-1000 (values <= 1 are taken as already normalized).
    ``pixel``: absolute pixels of the image the model saw (Qwen2-VL / Qwen2.5-VL convention); they
    are divided by the recorded model-input size, which maps them back onto the original image.
    Pixel rows without a recorded size fall back to the 0-1000 reading.
    """
    if row_box_format(row) == "pixel":
        size = model_input_size(row)
        if size is not None:
            width, height = size
            boxes = []
            for x1, y1, x2, y2 in extract_raw_boxes(text):
                box = (
                    min(max(x1 / width, 0.0), 1.0),
                    min(max(y1 / height, 0.0), 1.0),
                    min(max(x2 / width, 0.0), 1.0),
                    min(max(y2 / height, 0.0), 1.0),
                )
                if box[0] < box[2] and box[1] < box[3]:
                    boxes.append(box)
            return boxes
    return extract_normalized_boxes(text)


# GRIT's training-reward answer rule (examples/reward_function/grit.py: _official_answer_text and
# _answers_match; tests/test_eval_scorers.py checks that the two agree). The reward module is not imported here
# because it pulls in the training code.
_GRIT_NUMBER_WORDS = {
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


def _grit_words(text: str) -> list[str]:
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower()).split()
    return [_GRIT_NUMBER_WORDS.get(word, word) for word in words if word not in {"a", "an", "the"}]


def grit_rule_answer_correct(response: str, target: str) -> bool:
    """The text after <answer> (up to </answer>) matched as GRIT's training reward does."""
    if "<answer>" not in response:
        return False
    prediction = response.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    pred_words, gt_words = _grit_words(prediction), _grit_words(target)
    if not pred_words or not gt_words:
        return False
    if pred_words == gt_words:
        return True
    if len(gt_words) == 1 and gt_words[0] in {"yes", "no"}:
        return next((word for word in pred_words if word in {"yes", "no"}), None) == gt_words[0]
    if len(gt_words) == 1 and gt_words[0].isdigit():
        number = next((word for word in pred_words if word.isdigit()), None)
        return number is not None and int(number) == int(gt_words[0])
    gt_phrase, pred_phrase = " ".join(gt_words), " ".join(pred_words)
    if len(pred_words) <= len(gt_words) + 3 and f" {gt_phrase} " in f" {pred_phrase} ":
        return True
    try:
        return _grade_answer_cached(prediction, str(target).strip())
    except Exception:
        return False


_GRIT_BOX_PATTERN = re.compile(r"\b\d+,\s*\d+,\s*\d+,\s*\d+\b")


def grit_regex_boxes(text: str, row: dict[str, Any]) -> list[Box]:
    """Boxes found by GRIT's own pattern (extract_eval_results.py: any four comma-separated integers, with or
    without brackets), mapped onto the original image like ``predicted_boxes``."""
    size = model_input_size(row) if row_box_format(row) == "pixel" else None
    boxes = []
    for match in _GRIT_BOX_PATTERN.findall(text):
        x1, y1, x2, y2 = (float(value) for value in match.split(","))
        if size is not None:
            box = (x1 / size[0], y1 / size[1], x2 / size[0], y2 / size[1])
        else:
            box = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
        box = tuple(min(max(value, 0.0), 1.0) for value in box)
        if box[0] < box[2] and box[1] < box[3]:
            boxes.append(box)
    return boxes


def _grit_regex_iou(rows: PredictionRows) -> float | None:
    """GRIT grounding IoU with the boxes GRIT's pattern finds (one-shot rows only)."""
    ious = []
    for row in rows:
        if is_native_agentic_row(row):
            continue
        gt_boxes = _ground_truth_boxes(row.get("extra_info") or {})
        if gt_boxes:
            ious.append(
                score_box_sets(
                    validate_normalized_boxes(grit_regex_boxes(first_response(row), row)), gt_boxes
                ).grounding_iou
            )
    return sum(ious) / len(ious) if ious else None


def _answer_correctness(spec: BenchmarkSpec, row: dict[str, Any], response: str) -> tuple[bool, bool]:
    """(exact match, relaxed match) of the final answer; a truncated response without one is wrong."""
    if is_truncated_without_final_answer(row, 0, response):
        return False, False
    target = str(row.get("target"))
    if spec.metadata.get("answer_style") == "mcq_letter":
        correct = mcq_letter_correct(response, row)
        return correct, correct
    return _normalized_exact(response, target), _relaxed_answer_match(response, target)


def score_answer_bbox(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    """GRIT-style answer + grounding evaluation.

    Primary: answer accuracy (relaxed exact match: articles, number words and punctuation are
    normalized). Details: the GRIT grounding IoU (``grounding/grit_iou``: IoU between the union of
    all predicted boxes and the union of the GT boxes, averaged over samples with GT boxes; a
    sample without predicted boxes scores 0), box precision/recall/F1@0.5 and box counts. For
    native agentic runs the predicted boxes are the zoom-in regions the agent committed.
    """
    per_sample = []
    box_samples = []
    tool_records = []
    exact_answer_correct = 0
    relaxed_answer_correct = 0
    grit_rule_correct = 0
    for row in rows:
        response = first_response(row)
        exact_correct, relaxed_correct = _answer_correctness(spec, row, response)
        exact_answer_correct += int(exact_correct)
        relaxed_answer_correct += int(relaxed_correct)
        grit_correct = not is_truncated_without_final_answer(row, 0, response) and grit_rule_answer_correct(
            response, str(row.get("target"))
        )
        grit_rule_correct += int(grit_correct)
        record: dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "answer_correct": exact_correct,
            "answer_relaxed_correct": relaxed_correct,
            "answer_grit_rule_correct": grit_correct,
        }
        if is_native_agentic_row(row):
            tool_boxes = committed_tool_boxes(row)
            sample = _score_answer_bbox_sample(row, "", relaxed_correct, pred_boxes=tool_boxes)
            tool_record = _tool_evidence_sample_record(row, tool_boxes)
            tool_records.append(tool_record)
            record.update({"grounding_box_source": "committed_tool_regions", **tool_record})
        else:
            sample = _score_answer_bbox_sample(row, response, relaxed_correct)
        box_samples.append(sample)
        per_sample.append({**record, **_box_sample_record(sample)})

    details = _aggregate_box_samples(box_samples, rows)
    details["answer/exact_match_accuracy"] = exact_answer_correct / len(rows) if rows else 0.0
    details["answer/relaxed_accuracy"] = relaxed_answer_correct / len(rows) if rows else 0.0
    # GRIT's training-reward rule on the <answer> text (GRIT reports a GPT-4o judgement)
    details["answer/grit_rule_accuracy"] = grit_rule_correct / len(rows) if rows else 0.0
    details["grounding/grit_iou_grit_pattern"] = _grit_regex_iou(rows)
    if tool_records:
        details.update(_aggregate_tool_evidence_records(tool_records))
        details["grounding/box_source"] = "committed_tool_regions"
    raw = details["answer/relaxed_accuracy"] * 100.0
    write_jsonl(output_dir / f"{spec.key}_per_sample_answer_bbox.jsonl", per_sample)
    return MetricResult(spec.key, spec.group, spec.primary_metric, raw, raw, len(rows), details=details)


def score_grounding_iou(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    """Single-target grounding (OVDEval): primary = GRIT grounding IoU, plus Acc@0.5 IoU."""
    per_sample = []
    samples = []
    for row in rows:
        response = first_response(row)
        sample = _score_answer_bbox_sample(row, response, answer_correct=True)
        if sample.boxes.gt_count != 1:
            sample_id = row.get("sample_id")
            raise ValueError(
                f"{spec.key}:{sample_id} requires exactly one GT box for single-target grounding evaluation; "
                f"got {sample.boxes.gt_count}"
            )
        samples.append(sample)
        per_sample.append({"sample_id": row.get("sample_id"), **_box_sample_record(sample)})
    details = _aggregate_box_samples(samples, rows)
    details["grounding/acc_at_0_5_iou"] = (
        sum(sample.boxes.threshold_match_count > 0 for sample in samples) / len(samples) if samples else 0.0
    )
    details["grounding/grit_iou_grit_pattern"] = _grit_regex_iou(rows)
    raw = (details["grounding/grit_iou"] or 0.0) * 100.0
    write_jsonl(output_dir / f"{spec.key}_per_sample_grounding.jsonl", per_sample)
    return MetricResult(spec.key, spec.group, spec.primary_metric, raw, raw, len(rows), details=details)


def _score_answer_bbox_sample(
    row: dict[str, Any],
    response: str,
    answer_correct: bool,
    *,
    pred_boxes: list[Box] | None = None,
) -> AnswerBBoxSample:
    """Box statistics of one row; as in GRIT, every box written in the response is a prediction."""
    gt_boxes = _ground_truth_boxes(row.get("extra_info") or {})
    parsed_boxes = validate_normalized_boxes(predicted_boxes(response, row) if pred_boxes is None else pred_boxes)
    return AnswerBBoxSample(
        answer_correct=answer_correct,
        boxes=score_box_sets(parsed_boxes, gt_boxes),
        parse_failed=_contains_bbox_syntax(response) and not parsed_boxes,
        missing_prediction=bool(gt_boxes) and not parsed_boxes,
    )


def _ground_truth_boxes(extra_info: dict[str, Any]) -> list[Box]:
    normalized = extra_info.get("bboxs_normalized")
    if isinstance(normalized, list):
        return validate_normalized_boxes([tuple(box) for box in normalized if isinstance(box, (list, tuple))])

    values = extra_info.get("bboxs") or []
    bbox_format = str(extra_info.get("bbox_format") or "")
    if not values:
        return []
    if "pixel" in bbox_format:
        try:
            width = float(extra_info["width"])
            height = float(extra_info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("pixel bbox ground truth requires numeric width and height") from exc
        if not math.isfinite(width) or width <= 0.0 or not math.isfinite(height) or height <= 0.0:
            raise ValueError(f"invalid image dimensions for pixel bbox ground truth: width={width}, height={height}")
    boxes = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        x1, y1, x2, y2 = (float(item) for item in value)
        if "pixel" in bbox_format:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
        elif "1000" in bbox_format or max(x1, y1, x2, y2) > 1.0:
            x1, y1, x2, y2 = (item / 1000.0 for item in (x1, y1, x2, y2))
        boxes.append((x1, y1, x2, y2))
    return validate_normalized_boxes(boxes)


def _contains_bbox_syntax(response: str) -> bool:
    if re.search(r"<(?:region|bbox)\b", response, flags=re.IGNORECASE):
        return True
    return any(
        len(re.findall(r"-?\d+(?:\.\d+)?", match.group(1))) == 4
        for match in re.finditer(r"\[\s*\[?([0-9.,\s-]+)\]?\s*\]", response)
    )


def _box_sample_record(sample: AnswerBBoxSample) -> dict[str, Any]:
    metrics = sample.boxes
    precision, recall, f1 = _threshold_prf(metrics.threshold_match_count, metrics.pred_count, metrics.gt_count)
    return {
        "grit_iou": metrics.grounding_iou if metrics.gt_count else None,
        "box_precision_at_0_5": precision,
        "box_recall_at_0_5": recall,
        "box_f1_at_0_5": f1,
        "matched_mean_iou": metrics.matched_iou_sum / metrics.matched_count if metrics.matched_count else 0.0,
        "pred_count": metrics.pred_count,
        "gt_count": metrics.gt_count,
        "parse_failed": sample.parse_failed,
        "missing_prediction": sample.missing_prediction,
    }


def _aggregate_box_samples(samples: list[AnswerBBoxSample], rows: PredictionRows | None = None) -> dict[str, Any]:
    count = len(samples)
    with_gt = [sample for sample in samples if sample.boxes.gt_count]
    pred_count = sum(sample.boxes.pred_count for sample in samples)
    gt_count = sum(sample.boxes.gt_count for sample in samples)
    threshold_matches = sum(sample.boxes.threshold_match_count for sample in samples)
    matched_iou_sum = sum(sample.boxes.matched_iou_sum for sample in samples)
    matched_count = sum(sample.boxes.matched_count for sample in samples)
    precision, recall, f1 = _threshold_prf(threshold_matches, pred_count, gt_count)
    details: dict[str, Any] = {
        # GRIT grounding IoU (UCSB-AI/GRIT extract_eval_results.py): union of predicted boxes vs
        # union of GT boxes per sample, samples without GT boxes skipped, no prediction -> 0.
        "grounding/grit_iou": sum(sample.boxes.grounding_iou for sample in with_gt) / len(with_gt)
        if with_gt
        else None,
        "grounding/samples_with_gt": len(with_gt),
        "grounding/box_precision_at_0_5": precision,
        "grounding/box_recall_at_0_5": recall,
        "grounding/box_f1_at_0_5": f1,
        "grounding/matched_mean_iou": matched_iou_sum / matched_count if matched_count else 0.0,
        "grounding/pred_box_count": pred_count,
        "grounding/gt_box_count": gt_count,
        "grounding/bbox_parse_failure_rate": sum(sample.parse_failed for sample in samples) / count if count else 0.0,
        "grounding/missing_prediction_rate": sum(sample.missing_prediction for sample in samples) / count
        if count
        else 0.0,
    }
    if rows is not None:
        formats = sorted({row_box_format(row) for row in rows})
        details["grounding/box_format"] = ",".join(formats)
        if "pixel" in formats:
            details["grounding/pixel_rows_without_image_size"] = sum(
                1 for row in rows if row_box_format(row) == "pixel" and model_input_size(row) is None
            )
    return details


def _threshold_prf(threshold_matches: int, pred_count: int, gt_count: int) -> tuple[float, float, float]:
    precision = threshold_matches / pred_count if pred_count else float(gt_count == 0)
    recall = threshold_matches / gt_count if gt_count else float(pred_count == 0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def refcoco_row_best_iou(row: dict[str, Any]) -> tuple[float, str]:
    gt = _refcoco_gt_box(row.get("extra_info") or {})
    response = first_response(row)
    if is_truncated_without_final_answer(row, 0, response):
        return 0.0, "truncated_without_final_answer"
    answer_text = extract_final_response_text(response)
    pred_boxes = predicted_boxes(answer_text, row)
    source = "final_answer" if answer_text != response.strip() else "unwrapped_response"
    if not pred_boxes:
        pred_boxes = predicted_boxes(response, row)
        source = "full_response"
    best_iou = max((box_iou(gt, pred) for pred in pred_boxes), default=0.0)
    return best_iou, source


def score_refcoco(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    correct = 0
    source_counts = {
        "final_answer": 0,
        "unwrapped_response": 0,
        "full_response": 0,
        "truncated_without_final_answer": 0,
    }
    scored = []
    for row in rows:
        best_iou, source = refcoco_row_best_iou(row)
        source_counts[source] += 1
        hit = best_iou >= 0.5
        correct += int(hit)
        scored.append({"sample_id": row["sample_id"], "best_iou": best_iou, "correct": hit, "box_source": source})
    raw = correct / len(rows) * 100.0 if rows else 0.0
    write_jsonl(output_dir / f"{spec.key}_per_sample_refcoco.jsonl", scored)
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        raw,
        len(rows),
        details={
            "correct": correct,
            "final_answer_boxes": source_counts["final_answer"],
            "unwrapped_response_boxes": source_counts["unwrapped_response"],
            "fallback_to_full_response": source_counts["full_response"],
            "truncated_without_final_answer": source_counts["truncated_without_final_answer"],
            "box_extraction": "final_answer_with_full_response_fallback",
        },
    )


def pope_row(row: dict[str, Any]) -> tuple[str, str, str, bool]:
    category = str((row.get("metadata") or {}).get("category") or "all")
    response = first_response(row)
    truncated = is_truncated_without_final_answer(row, 0, response)
    pred = "other" if truncated else _parse_yes_no(response)
    return pred, _parse_yes_no(str(row.get("target"))), category, truncated


def score_pope(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    by_category: dict[str, list[tuple[str, str]]] = {}
    truncated_without_final_answer = 0
    for row in rows:
        pred, gold, category, truncated = pope_row(row)
        truncated_without_final_answer += int(truncated)
        by_category.setdefault(category, []).append((pred, gold))
    f1s = {category: _binary_f1(items) for category, items in sorted(by_category.items())}
    accuracies = {
        category: sum(pred == gold for pred, gold in items) / len(items)
        for category, items in sorted(by_category.items())
    }
    accuracy = sum(pred == gold for items in by_category.values() for pred, gold in items) / len(rows) if rows else 0.0
    raw = sum(f1s.values()) / len(f1s) * 100.0 if f1s else 0.0
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        raw,
        len(rows),
        details={
            "accuracy": accuracy,
            "accuracy_by_category": accuracies,
            "f1_by_category": f1s,
            "truncated_without_final_answer": truncated_without_final_answer,
        },
    )


def gqa_judge_unit_key(
    question: str,
    gold: str,
    answer: str,
    *,
    provider: str = "deepseek",
    model: str = DEFAULT_DEEPSEEK_JUDGE_MODEL,
) -> str:
    """Cross-run cache key: identical (question, reference, normalized answer) units share one verdict."""
    return json.dumps(
        {
            "question": question,
            "gold": _normalize_text(gold),
            "answer": _normalize_text(answer),
            "provider": provider,
            "model": model,
            "judge_version": GQA_JUDGE_VERSION,
        },
        sort_keys=True,
    )


def gqa_row_scores(row: dict[str, Any]) -> tuple[bool, bool, str, str, str]:
    """(exact, truncated, question, gold, answer) for responses[0] under the official EM rule."""
    response = first_response(row)
    gold = str(row.get("target"))
    truncated = is_truncated_without_final_answer(row, 0, response)
    answer = "" if truncated else extract_final_response_text(response)
    exact = (not truncated) and _normalize_text(answer) == _normalize_text(gold)
    question = str((row.get("metadata") or {}).get("question") or "").strip()
    if not question:
        question = str(row.get("prompt", "")).removesuffix(GQA_PROMPT_SUFFIX).strip()
    return exact, truncated, question, gold, answer


def _judge_gqa_units(
    units: dict[str, tuple[str, str, str]],
    judge_config: JudgeConfig,
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    """Judge uncached units concurrently; verdicts are appended to the shared cache.

    Unparseable judge output is retried with fresh requests and eventually raises:
    silently treating it as not-rescued would make scores depend on parse luck.
    """
    lock = threading.Lock()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def judge_one(item: tuple[str, tuple[str, str, str]]) -> tuple[str, dict[str, Any]]:
        key, (question, gold, answer) = item
        prompt = GQA_JUDGE_PROMPT.format(question=question, gold=gold, answer=answer)
        last_content = ""
        for _ in range(GQA_JUDGE_PARSE_RETRIES):
            content, served_model = _request_text_judge(prompt, judge_config, temperature=judge_config.temperature)
            match = _GQA_VERDICT_RE.search(content)
            if match:
                verdict_row = {
                    "cache_key": key,
                    "question": question,
                    "gold": gold,
                    "answer": answer,
                    "verdict": int(match.group(1)),
                    "category": match.group(2),
                    "judge_raw_content": content[:200],
                    "judge_parse_method": "regex",
                    "judge_model": served_model or judge_config.model,
                    "judge_version": GQA_JUDGE_VERSION,
                }
                with lock:
                    _append_jsonl(cache_path, verdict_row)
                return key, verdict_row
            last_content = content
        raise RuntimeError(
            f"gqa judge returned unparseable verdicts after {GQA_JUDGE_PARSE_RETRIES} attempts: {last_content[:200]!r}"
        )

    fresh: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=judge_config.concurrency, thread_name_prefix="gqa-judge") as pool:
        for key, verdict_row in pool.map(judge_one, units.items()):
            fresh[key] = verdict_row
    return fresh


def score_gqa(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    """Official normalized exact match; EM -> LLM-judge cascade only when a judge is configured."""
    del output_dir  # verdicts live in the shared cross-run cache, not the run directory
    units: list[tuple[bool, str | None]] = []
    judgeable: dict[str, tuple[str, str, str]] = {}
    truncated_without_final_answer = 0
    for row in rows:
        exact, truncated, question, gold, answer = gqa_row_scores(row)
        truncated_without_final_answer += int(truncated)
        key = None
        if judge_config is not None and not exact and not truncated and answer.strip():
            key = gqa_judge_unit_key(question, gold, answer, provider=judge_config.provider, model=judge_config.model)
            judgeable.setdefault(key, (question, gold, answer))
        units.append((exact, key))

    pending: dict[str, tuple[str, str, str]] = {}
    cache: dict[str, dict[str, Any]] = {}
    if judge_config is not None:
        cache = _load_judge_cache(GQA_JUDGE_CACHE_PATH)
        pending = {key: unit for key, unit in judgeable.items() if key not in cache}
        if pending:
            cache.update(_judge_gqa_units(pending, judge_config, GQA_JUDGE_CACHE_PATH))

    scores: list[bool] = []
    rescued = 0
    categories: dict[str, int] = {}
    for exact, key in units:
        correct = exact
        if key is not None:
            verdict = cache.get(key)
            if verdict is not None and int(verdict.get("verdict", 0)) == 1:
                correct = True
                rescued += 1
                category = str(verdict.get("category", "-"))
                categories[category] = categories.get(category, 0) + 1
        scores.append(correct)
    raw = sum(scores) / len(scores) * 100.0 if scores else 0.0
    exact_correct = sum(1 for exact, _ in units if exact)
    details: dict[str, Any] = {
        "scoring": "exact_match" if judge_config is None else "exact_match_then_llm_judge",
        "correct": sum(scores),
        "exact_match_accuracy": exact_correct / len(units) * 100.0 if units else 0.0,
        "exact_match_correct": exact_correct,
        "truncated_without_final_answer": truncated_without_final_answer,
    }
    if judge_config is not None:
        details.update(
            {
                "judge_rescued": rescued,
                "judge_rescue_categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
                "judge_units": len(judgeable),
                "judge_newly_judged": len(pending),
                "judge_version": GQA_JUDGE_VERSION,
                "judge_provider": judge_config.provider,
                "judge_model": judge_config.model,
            }
        )
    return MetricResult(spec.key, spec.group, spec.primary_metric, raw, raw, len(rows), details=details)


def seed_bench_correct(row: dict[str, Any]) -> bool:
    response = first_response(row)
    if is_truncated_without_final_answer(row, 0, response):
        return False
    options = (row.get("extra_info") or {}).get("options")
    letter = extract_choice_letter(response, list(options) if options else None, num_options=4)
    gold = _parse_option(str(row.get("target")))
    return bool(gold) and letter == gold


def score_seed_bench(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    """SEED-Bench accuracy over all questions (seed_all), with image/video and per-dimension splits."""
    truncated_without_final_answer = 0
    scores: list[float] = []
    data_types: list[str] = []
    dimensions: list[str] = []
    for row in rows:
        truncated_without_final_answer += int(is_truncated_without_final_answer(row, 0, first_response(row)))
        scores.append(float(seed_bench_correct(row)))
        metadata = row.get("metadata") or {}
        data_types.append(str(metadata.get("data_type") or "image"))
        dimensions.append(str(metadata.get("question_type_id") or ""))
    by_type = _mean_by_group(scores, data_types)
    raw = sum(scores) / len(scores) * 100.0 if scores else 0.0
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        raw,
        len(rows),
        details={
            "correct": int(sum(scores)),
            "seed_image_accuracy": by_type.get("image"),
            "seed_video_accuracy": by_type.get("video"),
            "accuracy_by_question_type": _mean_by_group(scores, dimensions),
            "truncated_without_final_answer": truncated_without_final_answer,
        },
    )


def mme_question_contribution(scores: list[float]) -> float:
    acc = sum(scores) / len(scores) * 100.0
    acc_plus = 100.0 if len(scores) >= 2 and sum(scores) == len(scores) else 0.0
    return acc + acc_plus


def mme_row(row: dict[str, Any]) -> tuple[str, str, bool, bool]:
    metadata = row.get("metadata") or {}
    response = first_response(row)
    truncated = is_truncated_without_final_answer(row, 0, response)
    pred = "other" if truncated else _parse_yes_no(response)
    correct = pred == _parse_yes_no(str(row.get("target")))
    return str(metadata.get("category")), str(metadata.get("question_id")), correct, truncated


def score_mme(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    category_to_questions: dict[str, dict[str, list[float]]] = {}
    truncated_without_final_answer = 0
    for row in rows:
        category, question_id, correct, truncated = mme_row(row)
        truncated_without_final_answer += int(truncated)
        category_to_questions.setdefault(category, {}).setdefault(question_id, []).append(float(correct))
    category_scores = {}
    for category, question_scores in category_to_questions.items():
        total = sum(mme_question_contribution(scores) for scores in question_scores.values())
        category_scores[category] = total / len(question_scores) if question_scores else 0.0
    raw = sum(category_scores.values())
    normalized = raw / 2800.0 * 100.0
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        normalized,
        len(rows),
        details={"category_scores": category_scores, "truncated_without_final_answer": truncated_without_final_answer},
    )


def score_mmvet(
    spec: BenchmarkSpec, rows: PredictionRows, judge_config: JudgeConfig | None, output_dir: Path
) -> MetricResult:
    if judge_config is None:
        raise ValueError("MM-Vet needs an LLM judge: pass --judge-provider openai|deepseek (see eval/README.md)")
    if judge_config.concurrency < 1:
        raise ValueError("--judge-concurrency must be >= 1")

    cache_path = output_dir / f"{spec.key}_judge_cache.jsonl"
    cache = _load_judge_cache(cache_path)
    judged: list[dict[str, Any] | None] = [None] * len(rows)
    pending: list[MMVetJudgeTask] = []
    for index, row in enumerate(rows):
        question = str((row.get("metadata") or {}).get("question") or row.get("prompt") or "")
        target = str(row.get("target") or "")
        full_prediction = first_response(row)
        prediction = full_prediction.strip()
        sample_id = str(row["sample_id"])
        task = MMVetJudgeTask(
            index=index,
            sample_id=sample_id,
            question=question,
            target=target,
            full_prediction=full_prediction,
            prediction=prediction,
            cache_key=_judge_key(sample_id, prediction, target, judge_config),
            capability=(row.get("metadata") or {}).get("capability"),
        )
        cached = cache.get(task.cache_key)
        if cached is None:
            pending.append(task)
            continue
        judged[index] = _mmvet_judgment(
            task,
            score=float(cached["score"]),
            raw_content=str(cached.get("judge_raw_content") or ""),
            parse_method=str(cached.get("judge_parse_method") or ""),
            attempts=int(cached.get("judge_attempts") or 0),
        )

    if pending:
        _score_mmvet_cache_misses(
            spec,
            pending,
            judge_config,
            output_dir,
            cache_path,
            cache,
            judged,
        )

    if any(item is None for item in judged):
        raise RuntimeError(f"{spec.key} finished judging with missing results")
    completed_judgments = [item for item in judged if item is not None]
    raw = (
        sum(item["score"] for item in completed_judgments) / len(completed_judgments) * 100.0
        if completed_judgments
        else 0.0
    )
    write_jsonl(output_dir / f"{spec.key}_judgments.jsonl", completed_judgments)
    return MetricResult(
        spec.key,
        spec.group,
        spec.primary_metric,
        raw,
        raw,
        len(rows),
        details={
            "judge_provider": judge_config.provider,
            "judge_model": judge_config.model,
            "judge_concurrency": judge_config.concurrency,
        },
    )


def _score_mmvet_cache_misses(
    spec: BenchmarkSpec,
    tasks: list[MMVetJudgeTask],
    judge_config: JudgeConfig,
    output_dir: Path,
    cache_path: Path,
    cache: dict[str, dict[str, Any]],
    judged: list[dict[str, Any] | None],
) -> None:
    task_iter = iter(tasks)
    first_error: RuntimeError | None = None
    first_cause: Exception | None = None
    max_workers = min(judge_config.concurrency, len(tasks))

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"{spec.key}-judge") as executor:
        in_flight: dict[Future, MMVetJudgeTask] = {}

        def fill_available_slots() -> None:
            while len(in_flight) < max_workers:
                try:
                    task = next(task_iter)
                except StopIteration:
                    return
                future = executor.submit(
                    _call_text_judge,
                    task.question,
                    task.target,
                    task.prediction,
                    judge_config,
                )
                in_flight[future] = task

        fill_available_slots()
        while in_flight:
            done, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: in_flight[item].index):
                task = in_flight.pop(future)
                if future.cancelled():
                    continue
                try:
                    judge_result = future.result()
                except Exception as exc:  # noqa: BLE001
                    if first_error is None:
                        first_error = RuntimeError(f"MM-Vet judge request failed for sample {task.sample_id}: {exc}")
                        first_cause = exc
                    continue

                if judge_result.parse_method == "failed":
                    failure_row = {
                        "sample_id": task.sample_id,
                        "prediction": task.prediction,
                        "target": task.target,
                        "judge_raw_content": judge_result.raw_content,
                        "judge_raw_attempts": judge_result.raw_attempts,
                        "judge_model": judge_result.model or judge_config.model,
                        "judge_version": MM_VET_JUDGE_VERSION,
                    }
                    _append_jsonl(output_dir / f"{spec.key}_judge_failures.jsonl", failure_row)
                    if first_error is None:
                        first_error = RuntimeError(
                            f"MM-Vet judge returned no parseable score for sample {task.sample_id}"
                        )
                    continue

                cache[task.cache_key] = {
                    "cache_key": task.cache_key,
                    "sample_id": task.sample_id,
                    "score": judge_result.score,
                    "prediction": task.prediction,
                    "full_prediction": task.full_prediction,
                    "target": task.target,
                    "judge_raw_content": judge_result.raw_content,
                    "judge_raw_attempts": judge_result.raw_attempts,
                    "judge_parse_method": judge_result.parse_method,
                    "judge_attempts": judge_result.attempts,
                    "judge_model": judge_result.model or judge_config.model,
                    "judge_version": MM_VET_JUDGE_VERSION,
                }
                write_jsonl(cache_path, cache.values())
                judged[task.index] = _mmvet_judgment(
                    task,
                    score=judge_result.score,
                    raw_content=judge_result.raw_content,
                    parse_method=judge_result.parse_method,
                    attempts=judge_result.attempts,
                )

            if first_error is None:
                fill_available_slots()
            else:
                for future in in_flight:
                    future.cancel()

    if first_error is not None:
        raise first_error from first_cause


def _mmvet_judgment(
    task: MMVetJudgeTask,
    *,
    score: float,
    raw_content: str,
    parse_method: str,
    attempts: int,
) -> dict[str, Any]:
    return {
        "sample_id": task.sample_id,
        "score": score,
        "prediction_for_judge": task.prediction,
        "judge_raw_content": raw_content,
        "judge_parse_method": parse_method,
        "judge_attempts": attempts,
        "capability": task.capability,
    }


def first_response(row: dict[str, Any]) -> str:
    responses = row.get("responses") or []
    return str(responses[0]) if responses else ""


def is_truncated_without_final_answer(row: dict[str, Any], response_index: int, response: str) -> bool:
    metadata = _response_metadata(row, response_index)
    return _is_truncated_metadata(metadata) and not has_final_answer(response)


def has_final_answer(text: str) -> bool:
    return extract_final_response_text(text) != text.strip()


def _response_metadata(row: dict[str, Any], response_index: int) -> dict[str, Any] | None:
    metadata_items = row.get("response_metadata")
    if not isinstance(metadata_items, list) or response_index >= len(metadata_items):
        return None
    metadata = metadata_items[response_index]
    return metadata if isinstance(metadata, dict) else None


def _is_truncated_metadata(metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    finish_reason = str(metadata.get("finish_reason") or "").lower()
    return bool(metadata.get("truncated")) or finish_reason == "length"


def generation_diagnostics(rows: PredictionRows) -> dict[str, Any]:
    response_count = 0
    truncated_count = 0
    missing_metadata_count = 0
    samples_with_truncation = 0
    samples_with_missing_metadata = 0
    finish_reasons: dict[str, int] = {}
    for row in rows:
        responses = row.get("responses") or []
        metadata_items = row.get("response_metadata")
        if not isinstance(metadata_items, list):
            metadata_items = []
        sample_truncated = False
        sample_missing_metadata = False
        for index, _response in enumerate(responses):
            response_count += 1
            metadata = (
                metadata_items[index]
                if index < len(metadata_items) and isinstance(metadata_items[index], dict)
                else None
            )
            if metadata is None:
                missing_metadata_count += 1
                sample_missing_metadata = True
                continue
            finish_reason = str(metadata.get("finish_reason") or "unknown")
            finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
            truncated_value = metadata.get("truncated")
            is_truncated = bool(truncated_value) or finish_reason.lower() == "length"
            if truncated_value is None:
                missing_metadata_count += 1
                sample_missing_metadata = True
            if is_truncated:
                truncated_count += 1
                sample_truncated = True
        samples_with_truncation += int(sample_truncated)
        samples_with_missing_metadata += int(sample_missing_metadata)
    sample_count = len(rows)
    diagnostics = {
        "generation/response_count": response_count,
        "generation/truncated_count": truncated_count,
        "generation/truncated_rate": truncated_count / response_count if response_count else 0.0,
        "generation/samples_with_truncation": samples_with_truncation,
        "generation/sample_truncated_rate": samples_with_truncation / sample_count if sample_count else 0.0,
        "generation/missing_metadata_count": missing_metadata_count,
        "generation/samples_with_missing_metadata": samples_with_missing_metadata,
        "generation/finish_reasons": finish_reasons,
    }
    agent_items = [item for row in rows for item in (row.get("agent_diagnostics") or []) if isinstance(item, dict)]
    if not agent_items:
        return diagnostics

    statuses: dict[str, int] = {}
    termination_classes: dict[str, int] = {}
    tool_error_codes: dict[str, int] = {}
    for item in agent_items:
        status = str(item.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        for error in item.get("tool_errors") or []:
            if isinstance(error, dict):
                code = str(error.get("error_code") or "unknown")
                tool_error_codes[code] = tool_error_codes.get(code, 0) + 1
        for failure in item.get("turn_end_failures") or []:
            if not isinstance(failure, dict):
                continue
            key = f"stop={failure.get('stop_reason')!r}|tail={failure.get('actual_tail_text')!r}"
            termination_classes[key] = termination_classes.get(key, 0) + 1

    count = len(agent_items)
    summed_fields = (
        "tool_call_attempts",
        "tool_execution_successes",
        "tool_call_successes",
        "tool_call_errors",
        "tool_calls_inside_reasoning",
        "tool_internal_errors",
        "image_idx_errors",
        "turn_end_errors",
        "action_tokens",
        "observation_tokens",
        "visual_observations",
    )
    flag_fields = (
        "no_action",
        "opening_answer_unclosed",
        "empty_answer",
        "invalid_answer",
        "format_replayed_inside_reasoning",
        "turn_length_limited",
        "trajectory_length_limited",
        "context_window_limited",
        "tool_call_limit_reached",
    )
    diagnostics.update(
        {
            "agent/trajectory_count": count,
            "agent/statuses": statuses,
            "agent/turn_end_classes": termination_classes,
            "agent/tool_error_codes": tool_error_codes,
        }
    )
    for field in summed_fields:
        diagnostics[f"agent/{field}_total"] = sum(int(item.get(field) or 0) for item in agent_items)
    for field in flag_fields:
        flagged = sum(bool(item.get(field)) for item in agent_items)
        diagnostics[f"agent/{field}_count"] = flagged
        diagnostics[f"agent/{field}_rate"] = flagged / count
    return diagnostics


def _write_perturbation_sample_diagnostics(
    spec: BenchmarkSpec,
    rows: PredictionRows,
    result: MetricResult,
    output_dir: Path,
) -> None:
    if not any(row.get("perturbation_diagnostics") for row in rows):
        return
    records = []
    for row in rows:
        score, correct = _sample_score_and_correct(spec, row)
        diagnostics = row.get("perturbation_diagnostics") or []
        records.append(
            {
                "benchmark": spec.key,
                "sample_id": row.get("sample_id"),
                "score": score,
                "correct": correct,
                "perturbation_diagnostics": diagnostics,
                "perturbation_type": result.metadata.get("perturbation_type", ""),
                "perturbation_seed": result.metadata.get("perturbation_seed", ""),
            }
        )
    write_jsonl(output_dir / f"{spec.key}_perturbation_samples.jsonl", records)


def _sample_score_and_correct(spec: BenchmarkSpec, row: dict[str, Any]) -> tuple[float | None, bool | None]:
    """Per-sample primary score used by the perturbation diagnostics file."""
    response = first_response(row)
    if spec.scorer in {
        "answer_bbox",
        "refcoco",
        "gqa",
        "seed_bench",
        "pope",
        "mme",
        "hallusionbench",
    } and is_truncated_without_final_answer(row, 0, response):
        return 0.0, False
    if spec.scorer == "answer_bbox":
        _, relaxed = _answer_correctness(spec, row, response)
        return float(relaxed), relaxed
    if spec.scorer == "grounding_iou":
        sample = _score_answer_bbox_sample(row, response, answer_correct=True)
        return sample.boxes.grounding_iou, sample.boxes.grounding_iou >= 0.5
    if spec.scorer == "refcoco":
        best_iou, _ = refcoco_row_best_iou(row)
        return best_iou, best_iou >= 0.5
    if spec.scorer == "gqa":
        correct = _normalized_exact(response, str(row.get("target")))
        return float(correct), correct
    if spec.scorer == "seed_bench":
        correct = seed_bench_correct(row)
        return float(correct), correct
    if spec.scorer == "cfpo_match":
        values = cfpo_row_scores(row)
        score = sum(values) / len(values) if values else 0.0
        return score, score >= 0.5
    if spec.scorer in {"pope", "mme", "hallusionbench"}:
        correct = _parse_yes_no(response) == _parse_yes_no(str(row.get("target")))
        return float(correct), correct
    if spec.scorer == "boxed_exact_match":
        values = boxed_row_scores(row, spec.key)
        score = sum(values) / len(values) if values else 0.0
        return score, score >= 0.5
    if spec.scorer == "mcq":
        score = mcq_row_score(row)
        return score, score >= 0.5
    return None, None


def extract_raw_boxes(text: str) -> list[tuple[float, float, float, float]]:
    """Every ``[x1, y1, x2, y2]`` (or ``[[...]]``) written in ``text``, unscaled."""
    boxes = []
    for match in re.finditer(r"\[\s*\[?([0-9.,\s-]+)\]?\s*\]", text):
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
        if len(numbers) == 4 and numbers[0] < numbers[2] and numbers[1] < numbers[3]:
            boxes.append((numbers[0], numbers[1], numbers[2], numbers[3]))
    return boxes


def extract_normalized_boxes(text: str) -> list[tuple[float, float, float, float]]:
    boxes = []
    for match in re.finditer(r"\[\s*\[?([0-9.,\s-]+)\]?\s*\]", text):
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
        if len(numbers) != 4:
            continue
        x1, y1, x2, y2 = numbers
        if max(numbers) > 1.0:
            x1, y1, x2, y2 = [value / 1000.0 for value in (x1, y1, x2, y2)]
        if x1 < x2 and y1 < y2:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _refcoco_gt_box(extra_info: dict[str, Any]) -> tuple[float, float, float, float]:
    box = extra_info.get("bbox")
    if box is None and extra_info.get("bbox_1000") is not None:
        box = [float(value) / 1000.0 for value in extra_info["bbox_1000"]]
    if box is None:
        raise ValueError("RefCOCO row is missing extra_info.bbox")
    values = [float(value) for value in box]
    if len(values) != 4:
        raise ValueError(f"invalid RefCOCO bbox: {box}")
    return tuple(values)  # type: ignore[return-value]


def _parse_yes_no(text: str) -> str:
    normalized = extract_final_response_text(text).lower().strip().replace(".", "")
    if normalized.startswith("yes") or normalized == "y":
        return "yes"
    if normalized.startswith("no") or normalized == "n":
        return "no"
    first = re.search(r"\b(yes|no)\b", normalized)
    return first.group(1) if first else "other"


def _binary_f1(items: list[tuple[str, str]]) -> float:
    tp = sum(1 for pred, gold in items if pred == "yes" and gold == "yes")
    fp = sum(1 for pred, gold in items if pred == "yes" and gold == "no")
    fn = sum(1 for pred, gold in items if pred == "no" and gold == "yes")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalized_exact(pred: str, gold: str) -> bool:
    return _normalize_text(extract_final_response_text(pred)) == _normalize_text(gold)


def _relaxed_answer_match(pred: str, gold: str) -> bool:
    return _normalize_relaxed_answer(extract_final_response_text(pred)) == _normalize_relaxed_answer(gold)


def _normalize_relaxed_answer(text: str) -> str:
    tokens = _normalize_text(_clean_answer_text(text)).split()
    normalized_tokens = []
    for token in tokens:
        if token in {"a", "an", "the"}:
            continue
        normalized_tokens.append(_NUMBER_WORDS.get(token, token))
    return " ".join(normalized_tokens)


_NUMBER_WORDS = {
    "zero": "0",
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


def _parse_option(text: str) -> str:
    text = extract_final_response_text(text).strip()
    match = re.search(r"\b([A-H])\b", text.upper())
    if match:
        return match.group(1)
    return ""


def _response_scores_with_truncation(
    row: dict[str, Any],
    score_response: Callable[[str], float],
) -> tuple[list[float], int]:
    scores = []
    truncated_count = 0
    for index, response in enumerate(row.get("responses") or []):
        if is_truncated_without_final_answer(row, index, str(response)):
            scores.append(0.0)
            truncated_count += 1
        else:
            scores.append(score_response(str(response)))
    return scores, truncated_count


def extract_final_response_text(text: str) -> str:
    tagged_answers: list[tuple[int, str]] = []
    boxed_answers = _extract_boxed_answers(text)
    final_marker_position = _last_final_answer_marker_position(text)
    if final_marker_position >= 0:
        final_boxes = [(position, answer) for position, answer in boxed_answers if position >= final_marker_position]
        if len(final_boxes) > 1:
            tagged_answers.append((final_boxes[-1][0], " and ".join(answer for _, answer in final_boxes)))
        else:
            tagged_answers.extend(boxed_answers)
    else:
        tagged_answers.extend(boxed_answers)
    for match in re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE):
        tagged_answers.append((match.start(), match.group(1)))
    if tagged_answers:
        return _clean_answer_text(max(tagged_answers, key=lambda item: item[0])[1])
    # A final "<answer> ..." that the model never closed (it stopped right after the answer).
    opening = list(re.finditer(r"<answer>", text, flags=re.IGNORECASE))
    if opening:
        trailing = text[opening[-1].end() :].strip()
        if trailing and "</answer>" not in trailing.lower() and "<answer>" not in trailing.lower():
            return _clean_answer_text(trailing)
    # "answer is: <newline> (D) ..." -> "(D) ..." (the colon used to be captured as the answer)
    marker = re.findall(
        r"(?:answer is|answer:|final answer is|final answer:)\s*:?\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if marker:
        return _clean_answer_text(marker[-1])
    return text.strip()


def _extract_boxed_answers(text: str) -> list[tuple[int, str]]:
    answers: list[tuple[int, str]] = []
    prefix = r"\boxed{"
    start = 0
    while True:
        position = text.find(prefix, start)
        if position < 0:
            break
        content, end = _balanced_brace_content(text, position + len(prefix))
        if content is not None:
            answers.append((position, _clean_answer_text(content)))
            start = end
        else:
            start = position + len(prefix)
    return answers


def _balanced_brace_content(text: str, start: int) -> tuple[str | None, int]:
    depth = 1
    chars = []
    index = start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
            chars.append(char)
        else:
            chars.append(char)
        index += 1
    return None, len(text)


def _last_final_answer_marker_position(text: str) -> int:
    matches = list(
        re.finditer(
            r"(?:answer is|answer:|final answer is|final answer:)",
            text,
            flags=re.IGNORECASE,
        )
    )
    return matches[-1].end() if matches else -1


def _clean_answer_text(text: str) -> str:
    cleaned = _unwrap_latex_text_command(text.strip().strip("\"'`"))
    cleaned = re.sub(r"\\\s+", " ", cleaned)
    cleaned = cleaned.replace(r"\{", "{").replace(r"\}", "}")
    return re.sub(r"[\s。．.!！,，;；:：]+$", "", cleaned).strip()


def _unwrap_latex_text_command(text: str) -> str:
    wrappers = "text|mathrm|textrm|textbf|textit|operatorname"
    previous = None
    current = text.strip()
    while previous != current:
        previous = current
        match = re.fullmatch(rf"\\(?:{wrappers})\{{(.*)\}}", current, flags=re.DOTALL)
        if match:
            current = match.group(1).strip()
    return current


def _to_float(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group(0))
    return value if math.isfinite(value) else None


def _load_judge_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache = {}
    for row in read_jsonl(path):
        if row.get("judge_parse_method") == "failed":
            continue
        key = row.get("cache_key") or _legacy_cache_key(row)
        if key:
            cache[str(key)] = row
    return cache


def _legacy_cache_key(row: dict[str, Any]) -> str:
    return f"{row.get('sample_id')}::{row.get('prediction')}::{row.get('target')}"


def _judge_key(sample_id: str, prediction: str, target: str, judge_config: JudgeConfig) -> str:
    return json.dumps(
        {
            "sample_id": sample_id,
            "prediction": prediction,
            "target": target,
            "provider": judge_config.provider,
            "model": judge_config.model,
            "thinking": judge_config.thinking,
            "judge_version": MM_VET_JUDGE_VERSION,
        },
        sort_keys=True,
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_mmvet_judge_prompt(question: str, target: str, prediction: str, *, retry: bool = False) -> str:
    prompt = f"{MM_VET_PROMPT}\n" + " | ".join(
        [
            question,
            target.replace("<AND>", " <AND> ").replace("<OR>", " <OR> "),
            prediction,
            "",
        ]
    )
    if retry:
        prompt += "\nPredict the correctness of the answer (digit): "
    return prompt


def _call_text_judge(question: str, target: str, prediction: str, judge_config: JudgeConfig) -> JudgeResult:
    raw_attempts: list[dict[str, Any]] = []
    model_used: str | None = None
    last_content = ""
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        temperature = judge_config.temperature if attempt == 1 else judge_config.temperature + 0.5 * (attempt - 1)
        prompt = _format_mmvet_judge_prompt(question, target, prediction, retry=attempt > 1)
        content, model_used = _request_text_judge(prompt, judge_config, temperature=temperature)
        last_content = content
        raw_attempts.append({"attempt": attempt, "temperature": temperature, "content": content})
        parsed = _parse_mmvet_score(content)
        if parsed is not None:
            score, parse_method = parsed
            return JudgeResult(
                score=score,
                raw_content=content,
                parse_method=parse_method,
                attempts=attempt,
                raw_attempts=raw_attempts,
                model=model_used,
            )
    return JudgeResult(
        score=0.0,
        raw_content=last_content,
        parse_method="failed",
        attempts=max_attempts,
        raw_attempts=raw_attempts,
        model=model_used,
    )


def _request_text_judge(prompt: str, judge_config: JudgeConfig, *, temperature: float) -> tuple[str, str | None]:
    payload = {
        "model": judge_config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": judge_config.max_tokens,
    }
    if judge_config.provider == "deepseek" and judge_config.thinking:
        payload["thinking"] = {"type": judge_config.thinking}
    retries = max(1, judge_config.request_retries)
    transient_http_codes = {408, 409, 425, 429, 500, 502, 503, 504}
    last_error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            judge_config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {judge_config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            return content, data.get("model")
        except urllib.error.HTTPError as exc:
            body = _read_http_error_body(exc)
            last_error = f"HTTP {exc.code}: {body}".strip()
            if exc.code not in transient_http_codes or attempt == retries:
                raise RuntimeError(f"judge request failed after {attempt} attempt(s): {last_error}") from exc
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
            if attempt == retries:
                raise RuntimeError(f"judge request failed after {attempt} attempt(s): {last_error}") from exc
        except http.client.IncompleteRead as exc:
            # A proxy or the remote judge may close a chunked response before
            # its terminating chunk arrives. The partial body cannot be parsed
            # safely, but issuing the same deterministic judge request again is
            # equivalent to the existing retries for transient transport errors.
            last_error = f"IncompleteRead: {exc}"
            if attempt == retries:
                raise RuntimeError(f"judge request failed after {attempt} attempt(s): {last_error}") from exc
        sleep_seconds = min(judge_config.request_backoff_seconds * (2 ** (attempt - 1)), 30.0)
        time.sleep(sleep_seconds)
    raise RuntimeError(f"judge request failed: {last_error}")


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:  # noqa: BLE001
        return str(exc)


def _parse_mmvet_score(content: str) -> tuple[float, str] | None:
    stripped = content.strip()
    first_token = re.match(r"^([01](?:\.\d+)?)\b", stripped)
    if first_token:
        score = float(first_token.group(1))
        if 0.0 <= score <= 1.0:
            return score, "first_token_float"
    embedded = re.search(r"\b([01](?:\.\d+)?)\b", stripped)
    if embedded:
        score = float(embedded.group(1))
        if 0.0 <= score <= 1.0:
            return score, "embedded_float"
    return None


JUDGE_PROVIDERS = ("none", "openai", "deepseek")


def judge_config_from_args(args) -> JudgeConfig | None:
    """LLM judge settings, or None when no judge is configured.

    ``--judge-provider`` defaults to ``none``: benchmarks that need a judge (MM-Vet) are then
    skipped and GQA uses exact match only. ``openai`` targets any OpenAI-compatible endpoint
    (``--judge-base-url``/``--judge-api-key``/``--judge-model`` or ``OPENAI_BASE_URL``/
    ``OPENAI_API_KEY``/``OPENAI_MODEL``); ``deepseek`` uses ``DEEPSEEK_API_KEY`` (and
    optionally ``DEEPSEEK_BASE_URL``). A provider without credentials counts as "no judge".
    """
    provider = (getattr(args, "judge_provider", None) or "none").lower()
    if provider == "none":
        return None
    if provider not in JUDGE_PROVIDERS:
        raise ValueError(f"unknown judge provider {provider!r}; expected one of {JUDGE_PROVIDERS}")
    model = getattr(args, "judge_model", None)
    base_url = getattr(args, "judge_base_url", None)
    api_key = getattr(args, "judge_api_key", None)
    if provider == "deepseek":
        model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_JUDGE_MODEL
        base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    else:
        model = model or os.environ.get("OPENAI_MODEL")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not (model and base_url and api_key):
        return None
    concurrency = int(getattr(args, "judge_concurrency", 4))
    if concurrency < 1:
        raise ValueError("--judge-concurrency must be >= 1")
    return JudgeConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=resolve_judge_max_tokens(
            getattr(args, "judge_max_tokens", None), getattr(args, "judge_thinking", None)
        ),
        concurrency=concurrency,
        request_retries=getattr(args, "judge_request_retries", 5),
        thinking=getattr(args, "judge_thinking", "disabled") if provider == "deepseek" else None,
    )


def judge_missing_reason(args) -> str:
    provider = (getattr(args, "judge_provider", None) or "none").lower()
    if provider == "none":
        return "no LLM judge configured (pass --judge-provider openai|deepseek)"
    if provider == "deepseek":
        return "--judge-provider deepseek needs DEEPSEEK_API_KEY (or --judge-api-key)"
    return "--judge-provider openai needs OPENAI_API_KEY and OPENAI_MODEL (or --judge-api-key/--judge-model)"


def resolve_judge_max_tokens(value: int | None, thinking: str | None) -> int:
    if value is not None:
        return value
    if thinking == "enabled":
        return DEFAULT_JUDGE_MAX_TOKENS_ENABLED_THINKING
    return DEFAULT_JUDGE_MAX_TOKENS_DISABLED_THINKING


SCORERS: dict[str, Scorer] = {
    "boxed_exact_match": score_boxed_exact_match,
    "cfpo_match": score_cfpo_match,
    "pope": score_pope,
    "hallusionbench": score_hallusionbench,
    "mme": score_mme,
    "gqa": score_gqa,
    "mmvet": score_mmvet,
    "seed_bench": score_seed_bench,
    "mcq": score_mcq,
    "answer_bbox": score_answer_bbox,
    "grounding_iou": score_grounding_iou,
    "refcoco": score_refcoco,
}
