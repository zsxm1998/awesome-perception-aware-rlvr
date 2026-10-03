# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import importlib.util
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
        ("C", "B", "Which option?", False),
        (
            "left",
            "The barrier is on the left side of the picture.",
            "On which side of the picture is the barrier?",
            True,
        ),
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
