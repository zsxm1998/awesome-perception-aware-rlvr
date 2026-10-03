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
import threading
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
# where acc is decided by a Qwen2.5-72B-Instruct judge; ThinkLite rows (compute_score_math) take
# the last \boxed{} after </think> (with nested braces matched, where DeepEyes' regex stops at the
# first "}"), check it with math_verify and ask the judge (with its math prompt) only when that
# fails, and their format is wrong unless the think tags balance and there is exactly one boxed
# answer. Set DEEPEYES_JUDGE_BASE_URL (OpenAI-compatible, e.g. a vLLM server), DEEPEYES_JUDGE_MODEL
# and optionally DEEPEYES_JUDGE_API_KEY to use a judge; otherwise a rule-based matcher
# (_rule_match) takes the judge's place, also for ThinkLite answers to word references that
# math_verify rejects (about half of the ThinkLite references are words, such as "brick"), while
# ThinkLite references with a digit fall back to mathruler's equivalence check.
# ---------------------------------------------------------------------------


# DeepEyes' judge prompt (verl/utils/reward_score/vl_agent.py: get_chat_template, get_gpt4_score_ICE, get_prompt)
_JUDGE_INSTRUCTION = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""  # noqa: E501
_JUDGE_EXAMPLES = [
    ("Is the countertop tan or blue?", "The countertop is tan.", "tan", 1),
    ("On which side of the picture is the barrier?", "The barrier is on the left side of the picture.", "left", 1),
    ("Is the kite brown and large?", "Yes, the kite is brown and large.", "Yes", 1),
    ("Are the spots on a giraffe?", "No, the spots are on a banana.", "no", 1),
    ("Who is wearing pants?", "The boy is wearing pants.", "The person in the picture is wearing pants.", 1),
    ("Is the man phone both blue and closed?", "Yes, the man phone is both blue and closed.", "No.", 0),
    (
        "What color is the towel in the center of the picture?",
        "The towel in the center of the picture is blue.",
        "The towel in the center of the picture is pink.",
        0,
    ),
]


def _judge_prompt(question: str, reference: str, prediction: str) -> str:
    prompt = _JUDGE_INSTRUCTION
    for example_question, example_reference, example_answer, judgement in _JUDGE_EXAMPLES:
        prompt += (
            f"\n[Question]: {example_question}\n[Standard Answer]: {example_reference}\n"
            f"[Model_answer] : {example_answer}\nJudgement: {judgement}\n\n\n"
        )
    return (
        prompt + f"\n[Question]: {question}\n[Standard Answer]: {reference}\n[Model_answer] : {prediction}\nJudgement:"
    )


def _parse_judgement(text: str) -> bool:
    text = text.strip()
    if "Judgement:" in text:
        text = text.split("Judgement:")[-1].strip()
        return "1" in text
    return text == "1"


# Words that carry no answer content; together with the question's own words they are ignored when the
# no-judge rule compares a prediction with a sentence reference.
_RULE_STOP_WORDS = set(
    "a an the is are was were be been being it its this that these those of to in on at by for with from and or as "
    "there here which what who whom whose where when how why do does did has have had can could would should will "
    "shall may might must appear appears appeared seem seems seemed look looks looked located placed positioned side "
    "image picture photo shown visible one".split()
)


def _rule_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _rule_keywords(text: str, question: str) -> list[str]:
    """Words of ``text`` that are neither stop words nor in the question; in "A or B" questions A and B stay."""
    options = re.search(r"\b(\w+)\s+or\s+(\w+)\b", question.lower())
    question_words = set(_rule_words(question)) - ({options.group(1), options.group(2)} if options else set())
    return [word for word in _rule_words(text) if word not in _RULE_STOP_WORDS and word not in question_words]


def _rule_match(prediction: str, reference: str, question: str = "") -> bool:
    """No-judge accuracy. Multiple-choice references compare the option letter, yes/no references the first
    yes/no word; otherwise the prediction's keywords (words outside the question and the stop words) must be
    non-empty, all appear in the reference, and include every keyword of the reference, so naming an object of
    the question is not enough. mathruler's equivalence check is the last resort."""
    prediction, reference = prediction.strip(), str(reference).strip()
    if not prediction or not reference:
        return False
    boxed = extract_boxed_content(prediction)
    if boxed and boxed != "None":
        prediction = boxed
    letter = re.fullmatch(r"\(?([A-Ha-h])\)?[.:)]?(\s.*)?", prediction)
    if re.fullmatch(r"[A-Ha-h]", reference):  # multiple-choice reference (chart data)
        return letter is not None and letter.group(1).upper() == reference.upper()
    reference_words = _rule_words(reference)
    if reference_words and reference_words[0] in {"yes", "no"}:  # sentence answers such as "No, the car is ..."
        return next((word for word in _rule_words(prediction) if word in {"yes", "no"}), None) == reference_words[0]
    prediction_keys = set(_rule_keywords(prediction, question))
    reference_keys = set(_rule_keywords(reference, question))
    if (
        prediction_keys
        and reference_keys
        and prediction_keys <= set(reference_words)
        and reference_keys <= prediction_keys
    ):
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
    """DeepEyes' judge call: its few-shot prompt, system "You are a helpful assistant.", temperature 0.3."""
    model = os.environ.get("DEEPEYES_JUDGE_MODEL", "judge")
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": _judge_prompt(question, reference, prediction)},
                ],
                temperature=0.3,
            )
            return _parse_judgement(completion.choices[0].message.content or "")
        except Exception:
            continue
    return _rule_match(prediction, reference, question)


# DeepEyes' math judge prompt (verl/utils/reward_score/vl_agent.py: MATH_VERIFY_PROMPT)
_MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. \n\
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""  # noqa: E501
_BOXED_OPEN = "\\boxed{"


def _boxed_answers(text: str) -> list[str]:
    """The non-empty contents of every closed \\boxed{...} in ``text``, with nested braces matched. DeepEyes' regex
    ``\\boxed{([^}]+)}`` stops at the first "}", which cuts answers such as \\frac{1}{2} to \\frac{1."""
    answers, start = [], text.find(_BOXED_OPEN)
    while start != -1:
        depth, index = 1, start + len(_BOXED_OPEN)
        while index < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        content = text[start + len(_BOXED_OPEN) : index - 1]
        if depth == 0 and content:
            answers.append(content)
        start = text.find(_BOXED_OPEN, index if depth == 0 else start + len(_BOXED_OPEN))
    return answers


def _math_answer(response: str) -> tuple[str | None, bool]:
    """DeepEyes' compute_score_math parsing: the last \\boxed{...} after the last </think> (with nested braces
    matched, see _boxed_answers); returns it and whether the format is wrong (unbalanced think tags, or not exactly
    one boxed answer)."""
    is_format_error = response.count("<think>") != response.count("</think>")
    answers = _boxed_answers(response.split("</think>")[-1].strip())
    if len(answers) != 1:
        is_format_error = True
    return (answers[-1] if answers else None), is_format_error


def _math_verify(reference: str, answer: str) -> bool:
    """math_verify's parse/verify, as DeepEyes' rule_math_verify. Their timeouts use SIGALRM, which only the main
    thread can set; elsewhere they run without a timeout."""
    from math_verify import parse, verify

    main_thread = threading.current_thread() is threading.main_thread()
    parse_kwargs = {} if main_thread else {"parsing_timeout": None}
    return bool(
        verify(
            parse(reference, **parse_kwargs),
            parse(answer, **parse_kwargs),
            **({} if main_thread else {"timeout_seconds": None}),
        )
    )


def _math_rule_fallback(answer: str, reference: str, question: str) -> bool:
    """Stands in for the math judge without one, after math_verify rejected the answer. About half of the ThinkLite
    references are words ("brick"), which math_verify cannot check: they go to the keyword rule. References with a
    digit go to mathruler's equivalence check, which keeps signs, decimal points and the order of the terms that the
    keyword rule ignores (it would take 2 for -2 and 2/1 for 1/2)."""
    if re.search(r"\d", reference):
        try:
            return bool(grade_answer(answer, reference))
        except Exception:
            return False
    return _rule_match(answer, reference, question)


def _judge_math(client, question: str, reference: str, answer: str) -> bool:
    """DeepEyes' generative_verify: MATH_VERIFY_PROMPT as the only (user) message, temperature 0."""
    model = os.environ.get("DEEPEYES_JUDGE_MODEL", "judge")
    prompt = _MATH_VERIFY_PROMPT.format(query=question, gold_ans=reference, pred_ans=answer)
    for _ in range(8):
        try:
            completion = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0
            )
            reply = (completion.choices[0].message.content or "").strip()
            break
        except Exception:
            continue
    else:
        return False
    judgement = reply.split("## Equivalence Judgement")[-1].lower()
    return "true" in judgement and "false" not in judgement


def compute_score_official(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    client = _judge_client()
    finals = [_validated_final_answer(reward_input["final_answer"]) for reward_input in reward_inputs]
    # ThinkLite: the boxed answer of the response, whether or not it is inside <answer> tags (official
    # compute_score_math); math_verify runs here, in the calling thread, because of its signal-based timeouts
    math_answers = {
        index: _math_answer(reward_input["response"])
        for index, reward_input in enumerate(reward_inputs)
        if reward_input.get("data_source", "") == "thinklite_eureka"
    }
    math_rule = {
        index: answer is not None and _math_verify(str(reward_inputs[index]["ground_truth"]), answer)
        for index, (answer, _) in math_answers.items()
    }

    def decide(index: int) -> float:
        reward_input = reward_inputs[index]
        reference = str(reward_input["ground_truth"])
        if index in math_answers:
            answer = math_answers[index][0]
            if math_rule[index]:
                return 1.0
            if answer is None:
                return 0.0
            if client is None:
                return 1.0 if _math_rule_fallback(answer, reference, reward_input.get("question", "")) else 0.0
            return 1.0 if _judge_math(client, reward_input.get("question", ""), reference, answer) else 0.0
        final_answer = finals[index]
        if final_answer is None:
            return 0.0
        if client is None:
            return 1.0 if _rule_match(final_answer, reference, reward_input.get("question", "")) else 0.0
        return 1.0 if _judge_match(client, reward_input.get("question", ""), final_answer, reference) else 0.0

    workers = int(os.environ.get("DEEPEYES_JUDGE_WORKERS", "32")) if client is not None else 1
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        accuracies = list(pool.map(decide, range(len(reward_inputs))))

    base_scores = compute_score(reward_inputs)
    scores = []
    for index, (reward_input, final_answer, accuracy, base) in enumerate(
        zip(reward_inputs, finals, accuracies, base_scores)
    ):
        response = reward_input["response"]
        if index in math_answers:
            format_score = -1.0 if math_answers[index][1] else 0.0
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
