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
"""Step-level advantages (GiGPO): the estimator against a reference written from the paper and verl-agent's
code, its edge cases, the step records of a trajectory and the environment's step rewards in the return."""

import math
from collections import defaultdict

import numpy as np
import pytest
import torch
from PIL import Image

from verl.protocol import DataProto
from verl.trainer.core_algos import AdvantageEstimator, compute_advantage_return
from verl.trainer.step_advantage import compute_gigpo_advantage
from verl.workers.agent.protocol import AgentMetrics, AgentStatus, AgentStep, AgentTrajectory, ToolCall, ToolResult
from verl.workers.agent.step_records import step_records
from verl.workers.agent.trajectory import turn_action_spans


# ----------------------------------------------------------------------------- reference
def _reference(records, rewards, index, *, gamma, omega, use_std, steps_stats, source, penalty, eps=1e-6):
    """GiGPO per turn, in float64, written from the paper (Eq. 2-8) and verl-agent's choices: episode statistics
    over every step (or one score per trajectory), unbiased std with eps outside, a step group of one gets 0, an
    episode group of one keeps its score, the penalty taken from the turn's own scores after discounting."""

    def norm(value, pool):
        if len(pool) == 1:
            return 0.0, 1.0
        mean = sum(pool) / len(pool)
        std = math.sqrt(sum((x - mean) ** 2 for x in pool) / (len(pool) - 1))
        return mean, std

    turns = {}  # (row, turn) -> (R_episode, R_step, invalid, anchor)
    for row, record in enumerate(records):
        turn_rewards = [rewards[row][s:e].sum().item() for s, e in (t["span"] for t in record["turns"])]
        episode = sum(rewards[row].tolist())
        returns, running = [], 0.0
        for reward in reversed(turn_rewards):
            running = reward + gamma * running
            returns.append(running)
        returns.reverse()
        anchor = None
        for turn, info in enumerate(record["turns"]):
            if turn == 0:
                initial = record["initial"]
                anchor = (
                    ("obs", initial["observation"])
                    if source == "observation"
                    else ("env", initial["state_key"] if initial["state_key"] is not None else "<initial>")
                )
            else:
                previous = record["turns"][turn - 1]
                if source == "observation":
                    anchor = ("obs", previous["observation"])
                elif previous["env_calls"]:
                    key = previous["env_calls"][-1]["state_key"]
                    anchor = ("env", key) if key is not None else ("none", row, turn)
            invalid = 0.0 if info["valid"] else 1.0
            turns[row, turn] = (episode, returns[turn], invalid, anchor)

    advantage = {}
    by_group = defaultdict(list)
    for member in turns:
        by_group[index[member[0]]].append(member)
    for members in by_group.values():
        if steps_stats:
            pool = [turns[m][0] - penalty * turns[m][2] for m in members]
        else:
            pool = [turns[m][0] for m in sorted({(m[0], 0) for m in members})]
        mean, std = norm(None, pool) if len(pool) > 1 else (0.0, 1.0)
        by_anchor = defaultdict(list)
        for member in members:
            by_anchor[turns[member][3]].append(member)
        step = {}
        for group in by_anchor.values():
            values = [turns[m][1] - penalty * turns[m][2] for m in group]
            if len(group) == 1:
                step[group[0]] = 0.0
                continue
            step_mean, step_std = norm(None, values)
            for m, value in zip(group, values):
                step[m] = (value - step_mean) / (step_std + eps) if use_std else value - step_mean
        for m in members:
            score = turns[m][0] - penalty * turns[m][2]
            episode_advantage = (score - mean) / (std + eps) if use_std else score - mean
            advantage[m] = episode_advantage + omega * step[m]
    return advantage


def _random_batch(seed, *, groups=3, n=4):
    rng = np.random.default_rng(seed)
    records, index = [], []
    width = 0
    for group in range(groups):
        for _ in range(n):
            turns, position = [], 0
            count = int(rng.integers(1, 6))
            for turn in range(count):
                length = int(rng.integers(1, 4))
                last = turn == count - 1
                called = not last and rng.random() < 0.8
                turns.append(
                    {
                        "span": [position, position + length],
                        "valid": bool(rng.random() > 0.25),
                        "env_calls": [{"state_key": f"s{rng.integers(0, 3)}", "reward": None}] if called else [],
                        "observation": None if last else f"o{rng.integers(0, 3)}",
                        "observation_text": None,
                    }
                )
                position += length + int(rng.integers(0, 3))  # an empty observation joins two turns' tokens
            width = max(width, position)
            records.append(
                {
                    "rollout_index": len(records) % n,
                    "initial": {"state_key": None, "observation": f"prompt{group}"},
                    "turns": turns,
                }
            )
            index.append(f"g{group}")
    rewards = torch.zeros(len(records), width + 2)
    mask = torch.zeros_like(rewards)
    for row, record in enumerate(records):
        for turn, info in enumerate(record["turns"]):
            start, end = info["span"]
            mask[row, start:end] = 1
            if end > start and turn < len(record["turns"]) - 1 and rng.random() < 0.3:
                rewards[row, end - 1] += float(rng.integers(-1, 3))  # an environment's step reward
        start, end = record["turns"][-1]["span"]
        rewards[row, end - 1] += float(rng.integers(0, 2))  # the reward function's score
    return records, np.array(index, dtype=object), rewards, mask


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("use_std", [False, True])
@pytest.mark.parametrize("steps_stats", [True, False])
@pytest.mark.parametrize("source", ["observation", "environment"])
def test_gigpo_matches_a_reference_written_from_the_paper_and_the_official_code(seed, use_std, steps_stats, source):
    records, index, rewards, mask = _random_batch(seed)
    gamma, omega, penalty = 0.9, 0.7, 0.1
    advantages, returns = compute_gigpo_advantage(
        rewards,
        mask,
        index,
        records,
        gamma=gamma,
        step_advantage_weight=omega,
        group_advantage_norm="mean_std" if use_std else "mean",
        episode_group_stats="steps" if steps_stats else "trajectories",
        step_state_key=source,
        invalid_action_penalty=penalty,
    )
    expected = _reference(
        records,
        rewards,
        index,
        gamma=gamma,
        omega=omega,
        use_std=use_std,
        steps_stats=steps_stats,
        source=source,
        penalty=penalty,
    )
    for (row, turn), value in expected.items():
        start, end = records[row]["turns"][turn]["span"]
        assert torch.allclose(
            advantages[row, start:end], torch.tensor(value, dtype=torch.float32), rtol=1e-4, atol=1e-4
        )
    assert torch.equal(advantages, returns)
    assert bool((advantages[mask == 0] == 0).all())  # observation tokens


def test_gigpo_without_step_advantages_and_with_trajectory_statistics_is_grpo():
    records, index, rewards, mask = _random_batch(3)
    grpo, _ = compute_advantage_return(
        AdvantageEstimator.GRPO, token_level_rewards=rewards, response_mask=mask, index=index, gamma=1.0, lam=1.0
    )
    gigpo, _ = compute_advantage_return(
        AdvantageEstimator.GIGPO,
        token_level_rewards=rewards,
        response_mask=mask,
        index=index,
        gamma=1.0,
        lam=1.0,
        step_records=records,
        step_advantage_weight=0.0,
        group_advantage_norm="mean_std",
        episode_group_stats="trajectories",
    )
    assert torch.equal(grpo, gigpo)


def test_gigpo_follows_its_rows_when_the_batch_is_reordered():
    records, index, rewards, mask = _random_batch(5)
    options = dict(gamma=0.95, invalid_action_penalty=0.1, group_advantage_norm="mean_std")
    advantages, _ = compute_gigpo_advantage(rewards, mask, index, records, **options)
    order = np.random.default_rng(0).permutation(len(records))
    permuted, _ = compute_gigpo_advantage(
        rewards[order], mask[order], index[order], [records[i] for i in order], **options
    )
    assert torch.allclose(permuted, advantages[order], atol=1e-5)


def _two_turns(first_observation, initial="prompt", text=None, state_key=None, called=True, rollout_index=0):
    return {
        "rollout_index": rollout_index,
        "initial": {"state_key": None, "observation": initial},
        "turns": [
            {
                "span": [0, 1],
                "valid": True,
                "env_calls": [{"state_key": state_key, "reward": None}] if called else [],
                "observation": first_observation,
                "observation_text": text,
            },
            {"span": [3, 4], "valid": True, "env_calls": [], "observation": None, "observation_text": None},
        ],
    }


def _second_turn_step_advantages(records, rewards_last, **options):
    rewards = torch.zeros(len(records), 5)
    mask = torch.zeros(len(records), 5)
    mask[:, [0, 3]] = 1
    rewards[:, 3] = torch.tensor(rewards_last, dtype=torch.float32)
    index = np.array(["g"] * len(records), dtype=object)
    with_steps, _ = compute_gigpo_advantage(rewards, mask, index, records, **options)
    episode_only, _ = compute_gigpo_advantage(rewards, mask, index, records, step_advantage_weight=0.0, **options)
    return (with_steps - episode_only)[:, 3].tolist()


def test_gigpo_groups_turns_by_the_last_observation_or_by_the_environments_state():
    records = [_two_turns("a", state_key="x"), _two_turns("a", state_key="y"), _two_turns("b", state_key="x")]
    # observation (default): the first two acted on the same observation
    step = _second_turn_step_advantages(records, [1.0, 0.0, 1.0])
    assert step == pytest.approx([0.5, -0.5, 0.0])
    # environment: the first and the last are in the same state
    step = _second_turn_step_advantages(records, [1.0, 0.0, 0.0], step_state_key="environment")
    assert step == pytest.approx([0.5, 0.0, -0.5])


def test_gigpo_environment_states_unnamed_carried_and_initial():
    # a call answered without a state key: the next turn is compared with no other
    records = [_two_turns("a", state_key=None), _two_turns("a", state_key=None)]
    assert _second_turn_step_advantages(records, [1.0, 0.0], step_state_key="environment") == [0.0, 0.0]
    # a malformed call (no environment call) keeps the state the turn was in: here the initial one, so the second
    # turn of the first trajectory joins the first turns of both
    malformed = _two_turns("a", called=False)
    rewards = torch.zeros(2, 5)
    mask = torch.zeros(2, 5)
    mask[:, [0, 3]] = 1
    rewards[0, 3], rewards[1, 3] = 1.0, 0.0
    index = np.array(["g", "g"], dtype=object)
    records = [malformed, _two_turns("b", state_key="x")]
    options = dict(step_state_key="environment", gamma=0.5)
    full, _ = compute_gigpo_advantage(rewards, mask, index, records, **options)
    episode_only, _ = compute_gigpo_advantage(rewards, mask, index, records, step_advantage_weight=0.0, **options)
    step = full - episode_only
    # initial state group: returns 0.5 (row 0 turn 0), 1.0 (row 0 turn 1), 0.0 (row 1 turn 0); mean 0.5
    assert [step[0, 0].item(), step[0, 3].item(), step[1, 0].item()] == pytest.approx([0.0, 0.5, -0.5])
    assert step[1, 3].item() == 0.0  # alone in state x
    # the environment names the initial state: a later turn in it joins the first turns
    named = _two_turns("a", state_key="start")
    named["initial"]["state_key"] = "start"
    other = _two_turns("b", state_key="elsewhere")
    other["initial"]["state_key"] = "start"
    rewards[0, 3], rewards[1, 3] = 1.0, 1.0
    full, _ = compute_gigpo_advantage(rewards, mask, index, [named, other], **options)
    episode_only, _ = compute_gigpo_advantage(
        rewards, mask, index, [named, other], step_advantage_weight=0.0, **options
    )
    step = full - episode_only
    assert [step[0, 0].item(), step[0, 3].item(), step[1, 0].item()] == pytest.approx([-1 / 6, 1 / 3, -1 / 6])


def test_gigpo_similarity_groups_similar_text_observations_and_matches_images_exactly():
    texts = ["zoomed into the left half", "zoomed into the left halves", "an error: the box is empty"]
    records = [_two_turns(f"h{i}", text=text, rollout_index=i) for i, text in enumerate(texts)]
    step = _second_turn_step_advantages(records, [1.0, 0.0, 1.0], step_similarity_threshold=0.9)
    assert step == pytest.approx([0.5, -0.5, 0.0])
    assert _second_turn_step_advantages(records, [1.0, 0.0, 1.0]) == [0.0, 0.0, 0.0]  # exact: all differ
    images = [_two_turns("same-image"), _two_turns("same-image"), _two_turns("other-image")]  # no text
    step = _second_turn_step_advantages(images, [1.0, 0.0, 1.0], step_similarity_threshold=0.9)
    assert step == pytest.approx([0.5, -0.5, 0.0])


def test_gigpo_similarity_groups_do_not_depend_on_how_the_batch_was_split():
    # greedy grouping depends on which turn comes first: "aaab" joins "aaaa" (ratio 0.75) but "aabb" does not (0.5)
    texts = ["aaaa", "aaab", "aabb", "zzzz"]
    records = [_two_turns(f"h{i}", text=text, rollout_index=i) for i, text in enumerate(texts)]
    step = _second_turn_step_advantages(records, [1.0, 0.0, 0.0, 0.0], step_similarity_threshold=0.7)
    assert step == pytest.approx([0.5, -0.5, 0.0, 0.0])
    # the same trajectories in another order in the batch (another split, length balancing): the same groups
    order = [3, 1, 2, 0]
    moved = _second_turn_step_advantages(
        [records[i] for i in order], [[1.0, 0.0, 0.0, 0.0][i] for i in order], step_similarity_threshold=0.7
    )
    assert moved == pytest.approx([step[i] for i in order])


def test_gigpo_episode_group_of_one_keeps_its_score_and_step_group_of_one_gets_zero():
    record = {
        "rollout_index": 0,
        "initial": {"state_key": None, "observation": "p"},
        "turns": [{"span": [0, 2], "valid": True, "env_calls": [], "observation": None, "observation_text": None}],
    }
    rewards = torch.tensor([[0.0, 3.0]])
    advantages, _ = compute_gigpo_advantage(rewards, torch.ones(1, 2), np.array(["g"], dtype=object), [record])
    assert advantages.tolist() == [[3.0, 3.0]]


def test_gigpo_rejects_rewards_outside_the_turns():
    records, index, rewards, mask = _random_batch(0)
    rewards[0, -1] = 1.0  # past every span
    with pytest.raises(ValueError, match="outside its turns"):
        compute_gigpo_advantage(rewards, mask, index, records)


def test_gigpo_reports_step_group_statistics():
    records = [_two_turns("a"), _two_turns("a"), _two_turns("b")]
    rewards = torch.zeros(3, 5)
    mask = torch.zeros(3, 5)
    mask[:, [0, 3]] = 1
    metrics = {}
    compute_gigpo_advantage(rewards, mask, np.array(["g"] * 3, dtype=object), records, metrics=metrics)
    # groups: the three first turns (one prompt), {second turns after "a"}, {after "b"}
    assert metrics["gigpo/step_group_size_mean"] == pytest.approx(2.0)
    assert metrics["gigpo/singleton_step_group_fraction"] == pytest.approx(1 / 3)
    assert metrics["gigpo/turns_in_shared_groups"] == pytest.approx(5 / 6)
    assert metrics["gigpo/turns_per_trajectory"] == pytest.approx(2.0)


# ----------------------------------------------------------------------------- step records
def _trajectory(observations, *, status=AgentStatus.ANSWERED, source_image=None, malformed_first=False):
    """Turns that call a tool and see ``observations`` (token ids, images), then answer."""
    response_ids, response_mask, steps, observation_images = [], [], [], []
    for index, (tokens, images) in enumerate(observations):
        action = [100 + index, 101]
        response_ids += action
        response_mask += [1] * len(action)
        result = ToolResult(
            tool_name="zoom",
            success=True,
            content={},
            images=list(images),
            parts=[{"type": "text", "text": f"view {tokens}"}],
        )
        step = AgentStep(
            model_text="call",
            model_token_ids=action,
            tool_call=None if (malformed_first and index == 0) else ToolCall("zoom", {}, ""),
            tool_call_error="bad json" if (malformed_first and index == 0) else None,
            tool_result=result,
            observation_token_ids=list(tokens),
            observation_committed=True,
            observation_image_count=len(images),
            state_key=None if (malformed_first and index == 0) else f"s{index}",
            env_reward=None if index else 0.25,
        )
        response_ids += list(tokens)
        response_mask += [0] * len(tokens)
        observation_images += list(images)
        steps.append(step)
    steps.append(AgentStep(model_text="<answer>1</answer>", model_token_ids=[7, 8, 9]))
    response_ids += [7, 8, 9]
    response_mask += [1, 1, 1]
    return AgentTrajectory(
        prompt_ids=[1, 2, 3],
        response_ids=response_ids,
        response_mask=response_mask,
        source_images=[source_image or Image.new("RGB", (4, 4))],
        observation_images=observation_images,
        steps=steps,
        status=status,
        metrics=AgentMetrics(),
        final_answer="1" if status == AgentStatus.ANSWERED else None,
        initial_state_key="start",
    )


def test_step_records_hash_the_images_the_model_receives():
    from verl.utils.dataset import process_image

    small = _trajectory([([5, 6], [Image.new("RGB", (512, 512), color=(9, 9, 9))])])
    large = _trajectory([([5, 6], [Image.new("RGB", (1024, 1024), color=(9, 9, 9))])])
    records = []
    for index, trajectory in enumerate((small, large)):
        model_images = [process_image(image, None, 256 * 256) for image in trajectory.all_images]  # as materialized
        spans = turn_action_spans(trajectory.steps, trajectory.response_mask)
        records.append(step_records(trajectory, spans, model_images, index))
    # different files, the same pixels for the model: the same observation
    assert records[0]["turns"][0]["observation"] == records[1]["turns"][0]["observation"]
    with pytest.raises(ValueError, match="model_images"):
        step_records(small, turn_action_spans(small.steps, small.response_mask), small.source_images, 0)


def test_turn_spans_count_model_tokens_even_when_an_observation_is_empty():
    trajectory = _trajectory([([], []), ([5, 6], [])])  # the first observation is empty
    spans = turn_action_spans(trajectory.steps, trajectory.response_mask)
    assert spans == [(0, 2), (2, 4), (6, 9)]  # the first two turns are one run of the mask, not one turn
    with pytest.raises(RuntimeError, match="more action tokens"):
        turn_action_spans(trajectory.steps[:-1], trajectory.response_mask)


def test_step_records_hold_each_turns_facts():
    image = Image.new("RGB", (4, 4), color=(1, 2, 3))
    first = _trajectory([([5, 6], [image]), ([5, 6], [])], malformed_first=True)
    spans = turn_action_spans(first.steps, first.response_mask)
    record = step_records(first, spans, first.all_images, 0)
    assert record["initial"]["state_key"] == "start"
    assert [turn["span"] for turn in record["turns"]] == [list(span) for span in spans]
    assert [turn["valid"] for turn in record["turns"]] == [False, True, True]  # malformed call, call, answer
    assert [turn["env_calls"] for turn in record["turns"]] == [[], [{"state_key": "s1", "reward": None}], []]
    # same tokens, different images: different observations; text only without images
    assert record["turns"][0]["observation"] != record["turns"][1]["observation"]
    assert [turn["observation_text"] for turn in record["turns"]] == [None, "view [5, 6]", None]
    # another trajectory of the same prompt and an equal image sees the same observations
    second = _trajectory([([5, 6], [Image.new("RGB", (4, 4), color=(1, 2, 3))]), ([5, 6], [])])
    other = step_records(second, turn_action_spans(second.steps, second.response_mask), second.all_images, 1)
    assert other["initial"]["observation"] == record["initial"]["observation"]
    assert other["turns"][0]["observation"] == record["turns"][0]["observation"]
    # the last turn is valid only when it answered
    cut = _trajectory([([5], [])], status=AgentStatus.LENGTH_LIMIT)
    assert (
        step_records(cut, turn_action_spans(cut.steps, cut.response_mask), cut.all_images, 0)["turns"][-1]["valid"]
        is False
    )


# ----------------------------------------------------------------------------- step rewards in the return
def _reward_batch(step_rewards):
    responses = torch.zeros(1, 6, dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0, 0, 1, 1]])
    agent_input = {"trajectory_id": "t", "final_answer": "1", "status": "answered", "tool_call_successes": 1}
    if step_rewards is not None:
        agent_input["env_step_rewards"] = step_rewards
    return DataProto.from_dict(
        tensors={"responses": responses, "response_mask": response_mask},
        non_tensors={"agent_reward_input": np.array([agent_input], dtype=object)},
    )


def test_environment_step_rewards_add_to_the_return_on_their_turns():
    from verl.workers.reward.function import _add_env_step_rewards

    data = _reward_batch([[1, 0.25], [5, -0.5]])
    reward_tensor = torch.zeros(1, 6)
    reward_tensor[0, 5] = 1.0  # the reward function's score on the last model token
    score = {"overall": 1.0, "accuracy": 1.0}
    _add_env_step_rewards(score, reward_tensor, data, 0)
    assert reward_tensor[0].tolist() == [0.0, 0.25, 0.0, 0.0, 0.0, 0.5]
    assert score == {"overall": 0.75, "accuracy": 1.0, "step_rewards": -0.25}

    untouched = torch.zeros(1, 6)
    score = {"overall": 1.0}
    _add_env_step_rewards(score, untouched, _reward_batch(None), 0)
    assert score == {"overall": 1.0} and not untouched.any()
    with pytest.raises(ValueError, match="not a model token"):
        _add_env_step_rewards({"overall": 1.0}, torch.zeros(1, 6), _reward_batch([[2, 1.0]]), 0)
    with pytest.raises(ValueError, match="step_rewards"):
        _add_env_step_rewards({"overall": 1.0, "step_rewards": 0.0}, torch.zeros(1, 6), _reward_batch([[1, 1.0]]), 0)


def test_sft_export_counts_step_rewards_in_the_return():
    from verl.workers.agent.protocol import EnvOutcome
    from verl.workers.agent.sft_export import keep_episode

    class Run:
        trajectory = _trajectory([([5], [])])

    Run.trajectory.env_outcome = EnvOutcome(reward=1.0)  # its first call earned 0.25
    assert keep_episode(Run, min_reward=1.25) is None
    assert keep_episode(Run, min_reward=1.3) == "reward below --min-reward"


# ----------------------------------------------------------------------------- configuration
def _gigpo_config(**algorithm):
    from verl.trainer.config import PPOConfig

    config = PPOConfig()
    config.worker.rollout.interaction_mode = "agentic"
    config.worker.rollout.agent_environment = "remote_http"
    config.worker.rollout.agent_environment_kwargs = {"url": "http://env.test"}
    config.worker.rollout.tensor_parallel_size = 1
    config.worker.rollout.n = 4
    config.data.format_prompt = None
    config.algorithm.adv_estimator = "gigpo"
    for key, value in algorithm.items():
        setattr(config.algorithm, key, value)
    return config


def test_config_turns_on_step_records_and_rejects_what_gigpo_does_not_support():
    config = _gigpo_config(gamma=0.95, step_similarity_threshold=0.9)
    config.algorithm.post_init()
    config.post_init()
    assert config.worker.rollout.agent_step_records

    with pytest.raises(ValueError, match="loss_avg_mode=token"):
        bad = _gigpo_config()
        bad.worker.actor.loss_avg_mode = "seq"
        bad.post_init()
    with pytest.raises(ValueError, match="worker.rollout.n > 1"):
        bad = _gigpo_config()
        bad.worker.rollout.n = 1
        bad.post_init()
    with pytest.raises(ValueError, match="interaction_mode=agentic"):
        bad = _gigpo_config()
        bad.worker.rollout.interaction_mode = "one_shot"
        bad.worker.rollout.agent_environment = None
        bad.post_init()
    with pytest.raises(ValueError, match="gamma"):
        _gigpo_config(gamma=0.0).algorithm.post_init()
    with pytest.raises(ValueError, match="step_state_key=observation only"):
        _gigpo_config(step_state_key="environment", step_similarity_threshold=0.9).algorithm.post_init()
    with pytest.raises(ValueError, match="group_advantage_norm"):
        _gigpo_config(group_advantage_norm="std").algorithm.post_init()

    from verl.trainer.config import AlgorithmConfig

    with pytest.raises(ValueError, match="only apply to adv_estimator=gigpo"):
        AlgorithmConfig(adv_estimator="grpo", invalid_action_penalty=0.1).post_init()
    grpo = _gigpo_config()
    grpo.algorithm.adv_estimator = "grpo"
    grpo.post_init()
    assert not grpo.worker.rollout.agent_step_records


def test_trainer_computes_gigpo_from_the_rollouts_step_records():
    from verl.trainer.ray_trainer import compute_advantage

    records, index, rewards, mask = _random_batch(1)
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards, "response_mask": mask},
        non_tensors={"uid": index, "agent_step_records": np.array(records, dtype=object)},
    )
    options = {"step_advantage_weight": 1.0, "group_advantage_norm": "mean", "episode_group_stats": "steps"}
    data = compute_advantage(data, AdvantageEstimator.GIGPO, gamma=0.95, step_advantage_options=options)
    expected, _ = compute_gigpo_advantage(rewards, mask, index, records, gamma=0.95, **options)
    assert torch.equal(data.batch["advantages"], expected)
    assert "gigpo/step_group_size_mean" in data.meta_info["advantage_metrics"]

    data.non_tensor_batch.pop("agent_step_records")
    with pytest.raises(ValueError, match="agent_step_records"):
        compute_advantage(data, AdvantageEstimator.GIGPO, step_advantage_options=options)
