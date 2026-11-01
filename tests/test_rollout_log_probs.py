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
"""worker.rollout.record_rollout_log_probs: the inference engine's log-probabilities of the model tokens, and how far
they are from the trainer's (rollout_corr/*, verl's definitions)."""

import asyncio
import math

import pytest
import torch

from verl.trainer.rollout_corr import compute_offpolicy_metrics
from verl.workers.agent.loop import AgentLoop
from verl.workers.agent.protocol import AgentLoopConfig, EncodedObservation, GenerationOutput
from verl.workers.rollout.config import RolloutConfig


def test_identical_log_probs_give_no_gap():
    old = torch.tensor([[-0.5, -1.0, -2.0, 0.0], [-0.1, -0.2, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

    metrics = compute_offpolicy_metrics(old, old.clone(), mask)

    for name in ("kl", "k3_kl", "log_ppl_diff", "log_ppl_abs_diff", "chi2_token", "chi2_seq"):
        assert metrics[f"rollout_corr/{name}"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rollout_corr/ppl_ratio"] == pytest.approx(1.0)
    assert metrics["rollout_corr/training_ppl"] == pytest.approx(metrics["rollout_corr/rollout_ppl"])


def test_the_gap_follows_verls_definitions_on_model_tokens_only():
    old = torch.tensor([[-1.0, -2.0, 5.0]])
    rollout = torch.tensor([[-0.5, -2.5, -9.0]])
    mask = torch.tensor([[1, 1, 0]])  # the third token is an observation token

    metrics = compute_offpolicy_metrics(old, rollout, mask)

    log_ratio = [-0.5, 0.5]  # old - rollout on the model tokens
    assert metrics["rollout_corr/kl"] == pytest.approx(0.0, abs=1e-6)  # mean(rollout - old)
    assert metrics["rollout_corr/k3_kl"] == pytest.approx(sum(math.exp(r) - r - 1 for r in log_ratio) / 2, rel=1e-5)
    assert metrics["rollout_corr/chi2_token"] == pytest.approx(
        sum(math.exp(2 * r) for r in log_ratio) / 2 - 1, rel=1e-5
    )
    assert metrics["rollout_corr/chi2_seq"] == pytest.approx(math.exp(2 * sum(log_ratio)) - 1, abs=1e-6)
    assert metrics["rollout_corr/training_ppl"] == pytest.approx(math.exp(1.5), rel=1e-5)


class _Backend:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def generate(self, request):
        return self.outputs.pop(0)


class _Encoder:
    max_tool_calls = None

    async def encode(self, request):
        return EncodedObservation(token_ids=[7, 7], effective_token_count=2, visual_token_count=0)


def test_the_loop_keeps_each_turns_log_probs():
    answer = GenerationOutput(token_ids=[3, 4], text="<answer>A</answer>", finish_reason="stop", log_probs=[-0.5, -1])
    trajectory = asyncio.run(AgentLoop(_Backend([answer]), _Encoder(), None).run(prompt_ids=[1], source_images=[]))
    assert trajectory.steps[0].model_log_probs == [-0.5, -1.0]

    unrecorded = GenerationOutput(token_ids=[3, 4], text="<answer>A</answer>", finish_reason="stop")
    trajectory = asyncio.run(AgentLoop(_Backend([unrecorded]), _Encoder(), None).run(prompt_ids=[1], source_images=[]))
    assert trajectory.steps[0].model_log_probs is None

    short = GenerationOutput(token_ids=[3, 4], text="<answer>A</answer>", finish_reason="stop", log_probs=[-0.5])
    with pytest.raises(ValueError, match="1 log-probs for 2 tokens"):
        asyncio.run(
            AgentLoop(_Backend([short]), _Encoder(), None, AgentLoopConfig()).run(prompt_ids=[1], source_images=[])
        )


def test_recording_needs_an_agentic_rollout():
    config = RolloutConfig(interaction_mode="one_shot", record_rollout_log_probs=True)
    with pytest.raises(ValueError, match="record_rollout_log_probs requires"):
        config.post_init()
    assert not RolloutConfig(interaction_mode="agentic").resolved_record_rollout_log_probs
