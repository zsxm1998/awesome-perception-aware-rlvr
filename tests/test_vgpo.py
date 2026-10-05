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
"""VGPO: last-layer prototype scores, the advantage factors per prompt group, and the config checks."""

import math

import pytest
import torch
from torch import nn

from verl.trainer.config import PPOConfig
from verl.trainer.perception_reasoning_loss import compute_vgpo_advantage_factors
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor


def _reference_factors(cosine, mask, uids, beta=0.3, tail=0.5, top=0.2, offset_mode="official", raw=False):
    """Eqs. 4-12 written per response with Python loops, with the released code's epsilons, rounding and clamps."""
    rows, length = cosine.shape
    valid = [cosine[r, c].item() for r in range(rows) for c in range(length) if mask[r, c]]
    offset = min(0.0, min(valid)) if offset_mode == "official" else -1.0
    intra = [[1.0] * length for _ in range(rows)]
    scores = []
    for r in range(rows):
        n = int(mask[r].sum())
        rho = [cosine[r, t].item() - offset for t in range(n)]
        start = int(n * (1.0 - tail))
        gate = [False] * n
        if start < n:
            k = max(1, int((n - start) * top))
            threshold = sorted(rho[start:], reverse=True)[k - 1]
            gate = [t >= start and rho[t] >= threshold for t in range(n)]
        w = [rho[t] * (1.0 + (beta * (t / (n - 1 + 1e-8)) / (1.0 + 1e-8) if gate[t] else 0.0)) for t in range(n)]
        lo, hi = min(w), max(w)
        w_hat = [(x - lo) / (hi - lo + 1e-8) for x in w]
        mean = sum(w_hat) / n
        for t in range(n):
            intra[r][t] = min(max(1.0 + w_hat[t] - mean, 0.1), 2.0)
        scores.append(sum(w) if raw else sum(w_hat))
    inter = [1.0] * rows
    for uid in set(uids):
        members = [r for r in range(rows) if uids[r] == uid]
        lo, hi = min(scores[r] for r in members), max(scores[r] for r in members)
        normalized = {r: (scores[r] - lo) / (hi - lo + 1e-8) for r in members}
        mean = sum(normalized.values()) / len(members)
        for r in members:
            inter[r] = min(max(1.0 + normalized[r] - mean, 0.9), 2.0)
    return torch.tensor([[inter[r] * intra[r][t] for t in range(length)] for r in range(rows)])


def _batch(seed, rows=12, length=40, low=-0.2):
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(2, length + 1, (rows,), generator=generator)
    mask = torch.arange(length)[None] < lengths[:, None]
    cosine = torch.rand(rows, length, generator=generator) * 0.6 + low
    uids = [f"p{row % 3}" for row in range(rows)]  # groups are not adjacent
    return cosine, mask, uids


@pytest.mark.parametrize("offset_mode", ["official", "paper"])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("low", [-0.2, 0.1])
def test_factors_match_the_reference(offset_mode, raw, low):
    cosine, mask, uids = _batch(0, low=low)
    cosine[0, :10] = 0.25  # ties in a tail
    factors, metrics = compute_vgpo_advantage_factors(
        cosine,
        mask,
        uids,
        score_offset=offset_mode,
        trajectory_score="raw" if raw else "compensated",
    )
    expected = _reference_factors(cosine, mask, uids, offset_mode=offset_mode, raw=raw)
    # every position, padding included (there the token factor is 1, so the factor is the response's f_inter)
    assert torch.allclose(factors, expected, atol=1e-5)
    assert metrics["vgpo/score_offset"] == (min(0.0, cosine[mask].min().item()) if offset_mode == "official" else -1.0)


def test_factors_do_not_depend_on_the_batch_order():
    cosine, mask, uids = _batch(1)
    factors, _ = compute_vgpo_advantage_factors(cosine, mask, uids)
    order = torch.randperm(len(uids), generator=torch.Generator().manual_seed(3))
    shuffled, _ = compute_vgpo_advantage_factors(cosine[order], mask[order], [uids[i] for i in order])
    assert torch.equal(shuffled, factors[order])


def test_the_offset_only_matters_through_the_compensation():
    cosine, mask, uids = _batch(2)
    official, _ = compute_vgpo_advantage_factors(cosine, mask, uids, compensation_strength=0.0)
    paper, _ = compute_vgpo_advantage_factors(cosine, mask, uids, compensation_strength=0.0, score_offset="paper")
    assert torch.allclose(official, paper, atol=1e-5)
    official, _ = compute_vgpo_advantage_factors(cosine, mask, uids)
    paper, _ = compute_vgpo_advantage_factors(cosine, mask, uids, score_offset="paper")
    assert not torch.allclose(official, paper, atol=1e-5)


def test_gate_and_factor_ranges():
    cosine, mask, uids = _batch(4, rows=30, length=200)
    factors, metrics = compute_vgpo_advantage_factors(cosine, mask, uids)
    # about half of each response is in the tail and a fifth of the tail passes the gate
    assert 0.07 < metrics["vgpo/gate_pass_rate"] < 0.12
    assert metrics["vgpo/response_factor_min"] >= 0.9 - 1e-6
    assert 0.1 - 1e-6 <= metrics["vgpo/token_factor_min"] and metrics["vgpo/token_factor_max"] <= 2.0 + 1e-6


class _ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.norm = nn.LayerNorm(4)
        self.layer = nn.Linear(4, 4)

    def forward(self, x):
        return self.language_model.norm(self.layer(x))


def _actor(module=None):
    config = ActorConfig()
    config.padding_free = False
    return DataParallelPPOActor(config=config, actor_module=module or nn.Linear(1, 1))


def test_final_norm_capture_records_the_last_hidden_state():
    module = _ToyLanguageModel()
    actor = _actor(module)
    x = torch.randn(2, 3, 4)
    with actor._capture_final_hidden_state(True) as captured:
        output = module(x)
    assert len(captured) == 1 and torch.equal(captured[0], output)
    module(x)
    assert len(captured) == 1  # the hook is removed on exit
    with actor._capture_final_hidden_state(False) as captured:
        assert captured is None
    with pytest.raises(ValueError, match="final norm"):
        _actor(nn.Linear(1, 1))._final_norm_module()


def test_prototype_pooling_is_the_cosine_to_the_mean_visual_state():
    actor = _actor()
    input_ids = torch.tensor([[7, 9, 9, 1, 2]])
    hidden = torch.randn(1, 5, 6)
    scores = actor._compute_hidden_visual_scores(
        hidden_states=(hidden,),
        input_ids=input_ids,
        response_length=2,
        response_mask=torch.tensor([[1, 1]]),
        visual_token_ids=[9],
        metric="cosine",
        pooling="prototype",
    )
    prototype = hidden[0, 1:3].mean(0)
    expected = torch.stack([torch.cosine_similarity(hidden[0, t], prototype, dim=0) for t in (3, 4)])
    assert torch.allclose(scores[0], expected, atol=1e-6)
    pairwise = actor._compute_hidden_visual_scores(
        hidden_states=(hidden,),
        input_ids=input_ids,
        response_length=2,
        response_mask=torch.tensor([[1, 1]]),
        visual_token_ids=[9],
        metric="cosine",
    )
    reference = torch.stack(
        [
            torch.stack([torch.cosine_similarity(hidden[0, t], hidden[0, v], dim=0) for v in (1, 2)]).mean()
            for t in (3, 4)
        ]
    )
    assert torch.allclose(pairwise[0], reference, atol=1e-6)  # the default (PEPO) is unchanged


VGPO = {
    "visual_sensitivity_metric": "hidden_state_similarity",
    "visual_sensitivity_hidden_layers": "last",
    "visual_sensitivity_hidden_pooling": "prototype",
    "visual_sensitivity_reference": "old",
    "advantage_scaling_method": "vgpo",
}


def _config(**algorithm):
    config = PPOConfig()
    for key, value in algorithm.items():
        setattr(config.algorithm, key, value)
    config.deep_post_init()
    return config


def test_config_checks():
    _config(**VGPO)
    for change, match in [
        ({"visual_sensitivity_reference": "current"}, "visual_sensitivity_reference=old"),
        ({"top_perception_quantile": 0.4}, "top_perception_quantile"),
        ({"vgpo_gate_tail_ratio": 0.0}, "vgpo_gate_tail_ratio"),
        ({"vgpo_score_offset": "zero"}, "vgpo_score_offset"),
        ({"visual_sensitivity_hidden_metric": "l2"}, "prototype requires"),
    ]:
        with pytest.raises(ValueError, match=match):
            _config(**{**VGPO, **change})
    with pytest.raises(ValueError, match="only apply to"):
        _config(visual_sensitivity_hidden_layers="last")


def test_loss_multiplies_the_advantages_by_the_driver_factors():
    from verl.trainer.perception_reasoning_loss import compute_perception_reasoning_policy_loss

    actor_config = ActorConfig()
    log_prob = torch.zeros(2, 3, requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    factors = torch.tensor([[0.5, 1.0, 2.0], [1.5, 1.0, 0.9]])
    mask = torch.ones(2, 3)
    loss_config = {"advantage_scaling_method": "vgpo"}
    loss, _ = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=torch.zeros(2, 3),
        advantages=advantages,
        response_mask=mask,
        entropy=None,
        decremental_old_log_probs=None,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        advantage_scaling_factors=factors,
    )
    loss.backward()
    # ratio 1: d(-A' * ratio)/d log_prob = -A' / number of tokens
    assert torch.allclose(log_prob.grad, -(advantages * factors) / 6)
    with pytest.raises(ValueError, match="advantage_scaling_factors"):
        compute_perception_reasoning_policy_loss(
            actor_config=actor_config,
            loss_config=loss_config,
            log_prob=log_prob,
            old_log_prob=torch.zeros(2, 3),
            advantages=advantages,
            response_mask=mask,
            entropy=None,
            decremental_old_log_probs=None,
            decremental_entropies=None,
            incremental_old_log_probs=None,
            incremental_entropies=None,
        )
    assert math.isfinite(loss.item())
