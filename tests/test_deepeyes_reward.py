# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
import re
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.workers.reward.config import RewardConfig
from verl.workers.reward.function import AutoRewardManager


_REWARD_PATH = Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "deepeyes.py"
_SPEC = importlib.util.spec_from_file_location("deepeyes_reward", _REWARD_PATH)
deepeyes_reward = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(deepeyes_reward)


def _response(answer: str = "42") -> str:
    return f"<think>Reason.</think><answer>{answer}</answer>"


def test_deepeyes_reward_components_match_the_controlled_formula():
    correct_with_tool, correct_without_tool, wrong_with_tool = deepeyes_reward.compute_score(
        [
            {
                "response": _response(),
                "final_answer": "42",
                "ground_truth": "42",
                "tool_call_successes": 1,
            },
            {
                "response": _response(),
                "final_answer": "42",
                "ground_truth": "42",
                "tool_call_successes": 0,
            },
            {
                "response": _response("24"),
                "final_answer": "24",
                "ground_truth": "42",
                "tool_call_successes": 2,
            },
        ]
    )

    assert correct_with_tool["overall"] == pytest.approx(2.0)
    assert correct_with_tool["tool"] == 1.0
    assert correct_without_tool["overall"] == pytest.approx(0.8)
    assert correct_without_tool["tool"] == 0.0
    assert wrong_with_tool["overall"] == pytest.approx(0.0)
    assert wrong_with_tool["tool"] == 0.0


def test_deepeyes_format_reward_matches_original_binary_penalty_and_length_boundary():
    balanced_visual = "<think>Inspect.<|vision_start|><|image_pad|><|vision_end|></think><answer>42</answer>"
    assert deepeyes_reward.format_reward(balanced_visual, "42") == 0.0
    # A closed reasoning block alone is still invalid when the parser did not
    # validate a final answer; hold this semantic independently of tag balance.
    assert deepeyes_reward.format_reward("<think>Done.</think>", None) == -1.0
    assert deepeyes_reward.format_reward("<think>Done.</think>", "42") == 0.0
    assert deepeyes_reward.format_reward("<think>Inspect.<answer>42</answer>", None) == -1.0
    assert (
        deepeyes_reward.format_reward(
            "<think>Done.</think><answer>" + ("x" * 1000) + "</answer>",
            "x" * 1000,
        )
        == -1.0
    )


class _UnusedTokenizer:
    def decode(self, *args, **kwargs):
        raise AssertionError("agent reward must use its compact canonical response")


def test_agent_reward_uses_compact_summary_and_places_score_on_last_action_token():
    config = RewardConfig(
        reward_function=f"{_REWARD_PATH}:compute_score",
        skip_special_tokens=False,
    )
    config.post_init()
    manager = AutoRewardManager(config, _UnusedTokenizer())
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[10, 20, 30, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0, 1, 0]], dtype=torch.long),
        },
        non_tensors={
            "ground_truth": np.array(["42"], dtype=object),
            "agent_reward_input": np.array(
                [
                    {
                        "trajectory_id": "sample:0",
                        "response": _response(),
                        "final_answer": "42",
                        "status": "answered",
                        "tool_call_successes": 1,
                        "trajectory_retries": 2,
                    }
                ],
                dtype=object,
            ),
            "agent_diagnostics": np.array(
                [
                    {
                        "trajectory_id": "sample:0",
                        "tool_call_attempts": 1,
                        "tool_call_errors": 0,
                        "tool_internal_errors": 0,
                        "image_idx_errors": 0,
                        "invalid_final_answers": 0,
                        "turn_end_errors": 0,
                        "tool_calls_inside_reasoning": 0,
                        "unclosed_answer": False,
                        "no_action": False,
                        "truncated": False,
                        "trajectory_retries": 2,
                    }
                ],
                dtype=object,
            ),
        },
    )

    rewards, metrics = manager.compute_reward(batch)

    assert rewards.tolist() == [[0.0, 0.0, 2.0, 0.0]]
    assert metrics["accuracy"] == [1.0]
    assert metrics["format"] == [0.0]
    assert metrics["tool"] == [1.0]
    assert metrics["trajectory_retries"] == [2.0]


@pytest.mark.parametrize(
    ("agent_input_update", "match"),
    [
        ({"final_answer": None}, "must agree on answered state"),
        ({"status": "length_limit"}, "must agree on answered state"),
    ],
)
def test_agent_reward_rejects_inconsistent_status_and_final_answer(agent_input_update, match):
    config = RewardConfig(reward_function=f"{_REWARD_PATH}:compute_score")
    config.post_init()
    manager = AutoRewardManager(config, _UnusedTokenizer())
    agent_input = {
        "trajectory_id": "sample:0",
        "response": _response(),
        "final_answer": "42",
        "status": "answered",
        "tool_call_successes": 0,
    }
    agent_input.update(agent_input_update)
    diagnostics = {
        "trajectory_id": "sample:0",
        "tool_call_attempts": 0,
        "tool_call_errors": 0,
        "tool_internal_errors": 0,
        "image_idx_errors": 0,
        "invalid_final_answers": 0,
        "turn_end_errors": 0,
        "tool_calls_inside_reasoning": 0,
        "trajectory_retries": 0,
        "unclosed_answer": False,
        "no_action": False,
        "truncated": False,
    }
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[10]], dtype=torch.long),
            "response_mask": torch.tensor([[1]], dtype=torch.long),
        },
        non_tensors={
            "ground_truth": np.array(["42"], dtype=object),
            "agent_reward_input": np.array([agent_input], dtype=object),
            "agent_diagnostics": np.array([diagnostics], dtype=object),
        },
    )

    with pytest.raises(ValueError, match=match):
        manager.compute_reward(batch)


def test_deepeyes_reward_requires_explicit_final_answer_key():
    with pytest.raises(KeyError, match="final_answer"):
        deepeyes_reward.compute_score([{"response": _response(), "ground_truth": "42", "tool_call_successes": 0}])


def test_deepeyes_reward_uses_agent_validated_final_answer_not_answer_tags_in_reasoning():
    score = deepeyes_reward.compute_score(
        [
            {
                "response": "<think>Quoted <answer>42</answer> while reasoning.",
                "final_answer": None,
                "ground_truth": "42",
                "tool_call_successes": 1,
            },
            {
                "response": "<think>Quoted <answer>42</answer>.</think><answer>42</answer>",
                "final_answer": "42",
                "ground_truth": "42",
                "tool_call_successes": 1,
            },
        ]
    )

    assert score[0]["accuracy"] == 0.0
    assert score[0]["format"] == -1.0
    assert score[0]["tool"] == 0.0
    assert score[0]["overall"] == pytest.approx(-0.2)
    assert score[1]["overall"] == pytest.approx(2.0)


def _official_input(answer, ground_truth, source, crops=1, question=""):
    return {
        "response": f"<think>looking</think><answer>{answer}</answer>",
        "final_answer": answer,
        "ground_truth": ground_truth,
        "data_source": source,
        "question": question,
        "tool_call_successes": crops,
    }


def test_official_reward_routes_by_data_source(monkeypatch):
    monkeypatch.delenv("DEEPEYES_JUDGE_BASE_URL", raising=False)
    scores = deepeyes_reward.compute_score_official(
        [
            _official_input("No", "No, the car is not on the left side of the person.", "vstar"),
            _official_input("Yes", "No, the car is not on the left side of the person.", "vstar"),
            _official_input(
                "brown", "The color of the puppy is brown.", "vstar", 0, "What is the color of the puppy?"
            ),
            _official_input("D. Fe2-Se", "D", "chart"),
            _official_input("The answer is \\boxed{-4}", "-4", "thinklite_eureka", crops=3),
        ]
    )
    assert [score["accuracy"] for score in scores] == [1.0, 0.0, 1.0, 1.0, 1.0]
    assert scores[0]["overall"] == pytest.approx(0.8 + 1.2)  # correct + committed crop
    assert scores[2]["overall"] == pytest.approx(0.8)  # correct without a crop: no tool bonus
    assert scores[4]["tool"] == 0.0 and scores[4]["overall"] == pytest.approx(1.2)  # math route


def test_official_reward_zeroes_answers_of_1000_characters(monkeypatch):
    """vl_agent.py: an answer of 1,000 characters or more gets no accuracy (hence no tool reward) and a format error."""
    monkeypatch.delenv("DEEPEYES_JUDGE_BASE_URL", raising=False)
    question, reference = "What color is the chair?", "The chair is white."
    long_answer = "white " * 200
    short, long = deepeyes_reward.compute_score_official(
        [
            _official_input("white", reference, "vstar", 1, question),
            _official_input(long_answer, reference, "vstar", 1, question),
        ]
    )
    assert short["overall"] == pytest.approx(2.0)
    assert (long["accuracy"], long["tool"], long["format"]) == (0.0, 0.0, -1.0)
    assert long["overall"] == pytest.approx(-0.2)


def test_text_only_reward_extracts_answer_span():
    (score,) = deepeyes_reward.compute_score_text_only(
        [
            {
                "response": "<think>the sign is red</think> <answer>red</answer>",
                "ground_truth": "The sign is red.",
                "data_source": "vstar",
                "question": "What color is the sign?",
            }
        ]
    )
    assert score["accuracy"] == 1.0 and score["tool"] == 0.0


@pytest.mark.parametrize(
    "prediction,reference,question,expected",
    [
        # naming an object of the question is not an answer
        ("chair", "The color of the chair is white.", "What is the color of the chair?", False),
        ("white", "The color of the chair is white.", "What is the color of the chair?", True),
        ("It is white.", "The color of the chair is white.", "What is the color of the chair?", True),
        ("white and red", "The color of the chair is white.", "What is the color of the chair?", False),
        # "A or B" questions: the options count as answer words
        ("black", "The cat is white.", "Is the cat black or white?", False),
        ("white", "The cat is white.", "Is the cat black or white?", True),
        ("The cat is white.", "The cat is white.", "Is the cat black or white?", True),
        # yes/no and multiple-choice references
        ("No, it is not.", "No, the car is not red.", "Is the car red?", True),
        ("Yes", "No, the car is not red.", "Is the car red?", False),
        ("(B) 42", "B", "Which option?", True),
        ("C.\nBecause the curve rises.\nSo C.", "C", "Which option?", True),  # explanation on further lines
        ("C", "B", "Which option?", False),
        (
            "left",
            "The barrier is on the left side of the picture.",
            "On which side of the picture is the barrier?",
            True,
        ),
        # options after an article or a preposition, lists, "how <adjective>" questions
        (
            "right",
            "The pizza is on the right side of the bowl.",
            "Is the pizza on the left or on the right side of the bowl?",
            True,
        ),
        (
            "left",
            "The pizza is on the right side of the bowl.",
            "Is the pizza on the left or on the right side of the bowl?",
            False,
        ),
        ("bed", "The bed is made of wood.", "What type of furniture is made of wood, the table or the bed?", True),
        ("table", "The bed is made of wood.", "What type of furniture is made of wood, the table or the bed?", False),
        ("small", "The boat is small.", "Which size is the boat, small or large?", True),
        ("boat", "The boat is small.", "Which size is the boat, small or large?", False),
        ("black", "The jacket is black.", "Is the jacket black, white or red?", True),
        ("red", "The jacket is black.", "Is the jacket black, white or red?", False),
        ("tall", "The trees are tall.", "How tall are the green trees?", True),
        ("trees", "The trees are tall.", "How tall are the green trees?", False),
        ("white", "I believe the telephone is white.", "Do you believe the telephone is white or green?", True),
    ],
)
def test_rule_match_needs_the_reference_keywords(prediction, reference, question, expected):
    assert deepeyes_reward._rule_match(prediction, reference, question) is expected


def test_judge_call_follows_deepeyes():
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            reply = "Judgement: 1" if "[Model_answer] : white\nJudgement:" in kwargs["messages"][1]["content"] else "0"
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": reply})()})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    assert deepeyes_reward._judge_match(client, "What color is the chair?", "white", "The chair is white.")
    assert not deepeyes_reward._judge_match(client, "What color is the chair?", "black", "The chair is white.")
    messages = calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert calls[0]["temperature"] == 0.3 and messages[1]["content"].count("Judgement: ") == 7
    assert deepeyes_reward._parse_judgement(" 1 ") and not deepeyes_reward._parse_judgement("yes")


def _official_compute_score_math(predict_str, ground_truth, judge=lambda ground_truth, model_answer: False):
    """DeepEyes verl/utils/reward_score/vl_agent.py compute_score_math and rule_math_verify, verbatim apart from the
    debug print; generative_verify is replaced by ``judge``."""
    from math_verify import parse, verify

    def rule_math_verify(ground_truth, model_answer):
        gold = parse(ground_truth)
        answer = parse(model_answer)
        return verify(gold, answer)

    is_format_error = False
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split("</think>")[-1].strip()
    answer_pattern = r"\\boxed{([^}]+)}"
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if judge(ground_truth, model_answer) else 0.0

    format_reward = -1.0 if is_format_error else 0.0
    return 1.2 * acc_reward + 0.4 * format_reward


_MATH_CASES = [
    ("<think>a</think>The answer is \\boxed{-4}", "-4"),
    ("<think>a</think><answer>\\boxed{10}</answer>", "10"),
    ("<think>a</think><answer>10</answer>", "10"),  # no boxed answer
    ("<think>a</think>\\boxed{3} or \\boxed{4}", "4"),  # two boxed answers: the last one, wrong format
    ("<think>a</think>\\boxed{5}", "4"),
    ("<think>a \\boxed{4}</think>so it is four", "4"),  # boxed only inside the reasoning
    ("<think>a</think><think>b \\boxed{4}", "4"),  # unbalanced think tags
    ("\\boxed{0.5}", "\\frac{1}{2}"),
    ("<think>a</think>\\boxed{B}", "B"),
    ("<think>a</think>\\boxed{x = 3}", "3"),
    ("<think>a</think>\\boxed{12\\%}", "12"),
    ("<think>a</think>\\boxed{brick}", "brick"),  # word answers: math_verify rejects them
    ("<think>a</think>\\boxed{stone}", "brick"),
]


def _math_input(response, ground_truth):
    return {
        "response": response,
        "final_answer": None,
        "ground_truth": ground_truth,
        "data_source": "thinklite_eureka",
        "question": "Find x.",
        "tool_call_successes": 0,
    }


def test_thinklite_reward_matches_the_official_compute_score_math(monkeypatch):
    monkeypatch.delenv("DEEPEYES_JUDGE_BASE_URL", raising=False)
    scores = deepeyes_reward.compute_score_official([_math_input(*case) for case in _MATH_CASES])

    def no_judge(ground_truth, model_answer):  # the rule that stands in for the judge
        return deepeyes_reward._math_rule_fallback(model_answer, ground_truth, "Find x.")

    expected = [_official_compute_score_math(*case, judge=no_judge) for case in _MATH_CASES]
    assert [score["overall"] for score in scores] == pytest.approx(expected)
    assert {score["accuracy"] for score in scores} == {0.0, 1.0} and {score["format"] for score in scores} == {
        0.0,
        -1.0,
    }
    assert all(score["tool"] == 0.0 for score in scores)


def test_thinklite_rule_fallback_keeps_signs_and_order_of_numbers(monkeypatch):
    """Without a judge, word references use the keyword rule; numeric ones must not (it ignores -, . and order)."""
    monkeypatch.delenv("DEEPEYES_JUDGE_BASE_URL", raising=False)
    cases = [
        ("<think>a</think>\\boxed{2}", "-2"),
        ("<think>a</think>\\boxed{2/1}", "1/2"),
        ("<think>a</think>\\boxed{5.12}", "12.5"),
        ("<think>a</think>\\boxed{-2}", "-2"),
        ("<think>a</think>\\boxed{brick}", "brick"),
        ("<think>a</think>\\boxed{the underground lake}", "underground lake"),
        ("<think>a</think>\\boxed{stone}", "brick"),
        # math references without a digit
        ("<think>a</think>\\boxed{-\\pi}", "\\pi"),
        ("<think>a</think>\\boxed{y-x}", "x-y"),
        ("<think>a</think>\\boxed{y/x}", "x/y"),
        ("<think>a</think>\\boxed{\\frac{x}{y}}", "x/y"),
        ("<think>a</think>\\boxed{red-tailed hawk}", "red-tailed hawk"),  # hyphenated words stay words
    ]
    scores = deepeyes_reward.compute_score_official([_math_input(*case) for case in cases])
    assert [score["accuracy"] for score in scores] == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_thinklite_answers_with_nested_braces_are_read_whole(monkeypatch):
    """DeepEyes' regex stops at the first "}" (\\frac{1}{2} -> \\frac{1); the braces are matched here."""
    monkeypatch.delenv("DEEPEYES_JUDGE_BASE_URL", raising=False)
    cases = [
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("<think>a</think>\\boxed{\\frac{1}{2}}", "0.5"),
        ("<think>a</think>\\boxed{\\frac{1}{2}} or \\boxed{\\frac{1}{3}}", "\\frac{1}{3}"),  # two answers
        ("<think>a</think>\\boxed{\\frac{2}{3}}", "\\frac{1}{2}"),
    ]
    scores = deepeyes_reward.compute_score_official([_math_input(*case) for case in cases])
    assert [score["overall"] for score in scores] == pytest.approx([1.2, 1.2, 0.8, 0.0])
    assert [_official_compute_score_math(*case) for case in cases] == pytest.approx([0.0, 0.0, -0.4, 0.0])

    assert deepeyes_reward._boxed_answers("\\boxed{} \\boxed{{3}} \\boxed{\\sqrt{2}} \\boxed{4") == [
        "{3}",
        "\\sqrt{2}",
    ]


def test_thinklite_reward_asks_the_math_judge_only_when_math_verify_fails(monkeypatch):
    calls = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            reply = "## Equivalence Judgement\nTRUE" if kwargs["messages"][0]["content"].endswith("\n6") else "FALSE"
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": reply})()})()]})()

    client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    monkeypatch.setattr(deepeyes_reward, "_judge_client", lambda: client)
    cases = [("<think>a</think>\\boxed{4}", "4"), ("<think>a</think>\\boxed{6}", "4"), ("\\boxed{5}", "4"), ("x", "4")]
    scores = deepeyes_reward.compute_score_official([_math_input(*case) for case in cases])

    def judge(ground_truth, model_answer):
        return model_answer == "6"

    assert [score["overall"] for score in scores] == pytest.approx(
        [_official_compute_score_math(*case, judge=judge) for case in cases]
    )
    assert len(calls) == 2  # math_verify accepted the first answer, the last has no boxed answer
    (message,) = calls[0]["messages"]
    assert message["role"] == "user" and calls[0]["temperature"] == 0.0
    assert message["content"].startswith("# CONTEXT #\nI am a teacher")
    assert "**Question**:\nFind x.\n\n**Reference Answer**\n4\n\n## Student Final Answer\n" in message["content"]


def test_judge_failures_are_reported(monkeypatch):
    """A judge that never answers: the answer judge falls back to the rule, the math judge counts as wrong, and the
    score says so (judge_failed) instead of hiding it."""

    class _Completions:
        def create(self, **kwargs):
            raise TimeoutError("judge down")

    client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
    monkeypatch.setattr(deepeyes_reward, "_judge_client", lambda: client)
    rule, math_wrong, math_rule = deepeyes_reward.compute_score_official(
        [
            _official_input("white", "The chair is white.", "vstar", 1, "What color is the chair?"),
            _math_input("<think>a</think>\\boxed{7}", "6"),
            _math_input("<think>a</think>\\boxed{6}", "6"),
        ]
    )
    assert (rule["accuracy"], rule["judge_failed"]) == (1.0, 1.0)
    assert (math_wrong["accuracy"], math_wrong["judge_failed"]) == (0.0, 1.0)
    assert (math_rule["accuracy"], math_rule["judge_failed"]) == (1.0, 0.0)  # math_verify decided, no judge call


def test_judge_client_waits_a_bounded_time(monkeypatch):
    for proxy in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(proxy, raising=False)  # a SOCKS proxy would need the optional socksio package
    monkeypatch.setenv("DEEPEYES_JUDGE_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("DEEPEYES_JUDGE_TIMEOUT", "30")
    client = deepeyes_reward._judge_client()
    assert client.max_retries == 0 and client.timeout == 30.0


def test_math_verify_runs_outside_the_main_thread():
    results = []
    thread = threading.Thread(target=lambda: results.append(deepeyes_reward._math_verify("\\frac{1}{2}", "0.5")))
    thread.start()
    thread.join()
    assert results == [True]
