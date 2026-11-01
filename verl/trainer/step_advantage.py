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
"""Step-level advantages of multi-turn agent trajectories: GiGPO (Feng et al., arXiv 2505.10978).

Each turn of a trajectory (one model reply) is a step. Its advantage is A = A_E + w * A_S: A_E normalizes the
episode's return within the group of rollouts of a prompt (as GRPO), A_S normalizes the turn's discounted return
within the turns of the group that acted in the same state (the anchor state). The arithmetic follows the official
implementation, verl-agent ``gigpo/core_gigpo.py`` (commit 20bd331): unbiased std with eps outside it, a step group
of one gets A_S = 0, an episode group of one keeps its score (mean 0, std 1), the invalid-action penalty is taken
from the turn's own episode score and step return after discounting.

A trajectory is one row here (the prompt and every turn, as the model sees them at inference), not one row per
step with a window of past steps as in verl-agent; the advantage of a turn goes to its model tokens
(``agent_step_records``, verl/workers/agent/step_records.py), observation tokens get 0.

A turn's reward is the sum of ``token_level_rewards`` over its model tokens: the environment's rewards for its tool
calls and, on the last turn, the reward function's score (and with a KL penalty in the reward, the penalty of its
tokens). The episode's return is their sum, the same score GRPO normalizes.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Hashable, Optional, Sequence

import numpy as np
import torch


STATE_KEY_SOURCES = ("observation", "environment")
NORMALIZATIONS = ("mean", "mean_std")
EPISODE_GROUP_STATS = ("steps", "trajectories")


def _anchors(record: dict[str, Any], row: int, source: str) -> list[tuple[Hashable, Optional[str]]]:
    """The state each turn acts in, as (key, text): ``text`` is set for text-only observations, which the similarity
    grouping may match by text. With ``source="environment"`` the keys are the environment's names of the states."""
    turns = record["turns"]
    anchors: list[tuple[Hashable, Optional[str]]] = []
    initial = record["initial"]
    if source == "environment":
        key = initial.get("state_key")
        anchors.append((("initial",) if key is None else ("state", key), None))
    else:
        anchors.append((("observation", initial["observation"]), None))
    for index in range(1, len(turns)):
        previous = turns[index - 1]
        if source == "environment":
            if previous["env_calls"]:
                key = previous["env_calls"][-1]["state_key"]
                # an environment that does not name the state after a call: the turn is not compared with others
                anchors.append(((("unnamed", row, index) if key is None else ("state", key)), None))
            else:  # a malformed call does not reach the environment and leaves its state as it was
                anchors.append(anchors[-1])
        else:
            if previous["observation"] is None:
                raise ValueError("a turn after one without a committed observation")
            anchors.append((("observation", previous["observation"]), previous.get("observation_text")))
    return anchors


def _step_groups(
    members: Sequence[tuple[int, int]],
    anchors: dict[tuple[int, int], tuple[Hashable, Optional[str]]],
    similarity_threshold: Optional[float],
) -> list[list[tuple[int, int]]]:
    """The turns of a prompt's group that act in the same state. With a similarity threshold, a text observation
    joins the first group whose first member's text matches with ``SequenceMatcher.ratio() >= threshold`` (the
    official ``enable_similarity``); other states (images, the initial prompt, environment keys) match exactly."""
    exact: dict[Hashable, list[tuple[int, int]]] = {}
    similar: list[tuple[str, list[tuple[int, int]]]] = []
    groups: list[list[tuple[int, int]]] = []
    for member in members:
        key, text = anchors[member]
        if similarity_threshold is not None and text is not None:
            for representative, group in similar:
                if SequenceMatcher(None, text, representative).ratio() >= similarity_threshold:
                    group.append(member)
                    break
            else:
                similar.append((text, [member]))
                groups.append(similar[-1][1])
            continue
        if key not in exact:
            exact[key] = []
            groups.append(exact[key])
        exact[key].append(member)
    return groups


def _normalize(values: list[torch.Tensor], pool: list[torch.Tensor], use_std: bool, eps: float) -> list[torch.Tensor]:
    mean = torch.mean(torch.tensor(pool))
    std = torch.std(torch.tensor([pool]))
    return [(value - mean) / (std + eps) if use_std else value - mean for value in values]


def compute_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    step_records: Sequence[dict[str, Any]],
    gamma: float = 1.0,
    step_advantage_weight: float = 1.0,
    group_advantage_norm: str = "mean",
    episode_group_stats: str = "steps",
    step_state_key: str = "observation",
    step_similarity_threshold: Optional[float] = None,
    invalid_action_penalty: float = 0.0,
    eps: float = 1e-6,
    metrics: Optional[dict[str, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GiGPO advantages of a batch of trajectories (one per row; ``index`` the group of each row). ``metrics``, when
    given, receives the step-group statistics. The similarity grouping, whose groups depend on the order of the turns,
    takes them in the order of their rollout index among the prompt's rollouts (``rollout_index`` of the records), so
    that it does not change with the batch order (length balancing) or how the batch was split across workers."""
    if step_state_key not in STATE_KEY_SOURCES:
        raise ValueError(f"step_state_key must be one of {STATE_KEY_SOURCES}")
    if group_advantage_norm not in NORMALIZATIONS:
        raise ValueError(f"group_advantage_norm must be one of {NORMALIZATIONS}")
    if episode_group_stats not in EPISODE_GROUP_STATS:
        raise ValueError(f"episode_group_stats must be one of {EPISODE_GROUP_STATS}")
    batch_size, response_length = token_level_rewards.shape
    if len(step_records) != batch_size or len(index) != batch_size:
        raise ValueError("agent_step_records and uid must have one entry per row")
    use_std = group_advantage_norm == "mean_std"
    scores = token_level_rewards.sum(dim=-1)  # the episode's return, as GRPO takes it

    # turns: (row, turn) -> span, reward, validity; rewards outside the turns would be lost, so there are none
    spans: dict[tuple[int, int], tuple[int, int]] = {}
    turn_rewards: dict[tuple[int, int], float] = {}
    valid: dict[tuple[int, int], bool] = {}
    anchors: dict[tuple[int, int], tuple[Hashable, Optional[str]]] = {}
    for row, record in enumerate(step_records):
        turns = record["turns"]
        if not turns:
            raise ValueError(f"row {row} has no turns")
        covered = torch.zeros(response_length, dtype=torch.bool)
        for turn, (anchor, info) in enumerate(zip(_anchors(record, row, step_state_key), turns)):
            start, end = info["span"]
            spans[row, turn] = (start, end)
            covered[start:end] = True
            turn_rewards[row, turn] = float(token_level_rewards[row, start:end].sum())
            valid[row, turn] = bool(info["valid"])
            anchors[row, turn] = anchor
        if bool((token_level_rewards[row][~covered] != 0).any()):
            raise ValueError(f"row {row} has rewards outside its turns' model tokens")

    # discounted step returns, accumulated as the official code does (float32 numpy)
    step_returns: dict[tuple[int, int], torch.Tensor] = {}
    for row, record in enumerate(step_records):
        rewards = np.asarray([turn_rewards[row, turn] for turn in range(len(record["turns"]))], dtype=np.float32)
        returns = np.zeros_like(rewards)
        running = 0
        for turn in reversed(range(len(rewards))):
            running = rewards[turn] + gamma * running
            returns[turn] = running
        for turn, value in enumerate(torch.tensor(returns, dtype=torch.float32)):
            step_returns[row, turn] = value
    penalty = {member: invalid_action_penalty * (0.0 if ok else 1.0) for member, ok in valid.items()}
    episode_scores = {member: scores[member[0]] - penalty[member] for member in valid}
    step_scores = {member: step_returns[member] - penalty[member] for member in valid}

    groups: dict[Hashable, list[int]] = defaultdict(list)
    for row in range(batch_size):
        groups[index[row]].append(row)
    episode_advantage: dict[tuple[int, int], torch.Tensor] = {}
    step_advantage: dict[tuple[int, int], torch.Tensor] = {}
    group_sizes: list[int] = []
    for rows in groups.values():
        members = [(row, turn) for row in rows for turn in range(len(step_records[row]["turns"]))]
        # episode level: every turn's score (official) or one score per trajectory (the paper's Eq. 3)
        if episode_group_stats == "steps":
            pool = [episode_scores[member] for member in members]
        else:
            pool = [scores[row] for row in rows]
        if len(pool) == 1:  # the official mean 0, std 1: the score itself
            values = [episode_scores[member] for member in members]
            normalized = [(value - 0.0) / (1.0 + eps) if use_std else value - 0.0 for value in values]
        else:
            normalized = _normalize([episode_scores[member] for member in members], pool, use_std, eps)
        episode_advantage.update(zip(members, normalized))
        # step level: the turns of the group acting in the same state
        ordered = sorted(members, key=lambda member: (int(step_records[member[0]]["rollout_index"]), member[1]))
        for group in _step_groups(ordered, anchors, step_similarity_threshold):
            group = sorted(group)  # pooled in batch order, as the official code does
            group_sizes.append(len(group))
            if len(group) == 1:
                step_advantage[group[0]] = torch.tensor(0.0)
                continue
            pool = [step_scores[member] for member in group]
            step_advantage.update(zip(group, _normalize(pool, pool, use_std, eps)))

    advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
    for member, (start, end) in spans.items():
        advantages[member[0], start:end] = episode_advantage[member] + step_advantage_weight * step_advantage[member]
    advantages = advantages * response_mask
    if metrics is not None:
        sizes = np.asarray(group_sizes, dtype=np.float64)
        members_in_groups = sizes.sum()
        metrics.update(
            {
                "gigpo/step_group_size_mean": float(sizes.mean()),
                "gigpo/singleton_step_group_fraction": float((sizes == 1).mean()),
                "gigpo/turns_in_shared_groups": float(sizes[sizes > 1].sum() / members_in_groups),
                "gigpo/turns_per_trajectory": float(members_in_groups / batch_size),
                "gigpo/invalid_turn_fraction": float(np.mean([not ok for ok in valid.values()])),
                "gigpo/episode_advantage_abs": float(
                    np.mean([abs(float(value)) for value in episode_advantage.values()])
                ),
                "gigpo/step_advantage_abs": float(np.mean([abs(float(value)) for value in step_advantage.values()])),
            }
        )
    return advantages, advantages
