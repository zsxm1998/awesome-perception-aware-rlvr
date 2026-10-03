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
"""Top-quantile token masks over the whole rollout batch, micro-batch masks and the top-p entropy."""

import math
from types import SimpleNamespace

import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.config import AlgorithmConfig
from verl.trainer.perception_reasoning_data import build_perception_reasoning_loss_config
from verl.trainer.perception_reasoning_loss import (
    build_batch_token_masks,
    compute_perception_reasoning_policy_loss,
    needs_current_policy_entropy,
)
from verl.utils.torch_functional import entropy_from_logits, top_p_entropy_from_logits
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor


def _tor_config(**overrides) -> AlgorithmConfig:
    kwargs = dict(
        corrupt_image="no_image",
        top_entropy_quantile=0.3,
        entropy_thr_granularity="batch",
        top_perception_quantile=0.3,
        perception_thr_granularity="batch",
        visual_sensitivity_reference="old",
        visual_sensitivity_metric="sampled_abs_log_ratio",
        tor_use_token_weighting=True,
        tor_rsn_weight=1.0,
        tor_prcp_weight=0.5,
    )
    kwargs.update(overrides)
    config = AlgorithmConfig(**kwargs)
    config.post_init()
    return config


def _batch(seed: int = 0, batch_size: int = 8, length: int = 12) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(3, length + 1, (batch_size,), generator=generator)
    response_mask = (torch.arange(length)[None, :] < lengths[:, None]).float()
    old_log_probs = -torch.rand(batch_size, length, generator=generator) * 3
    return {
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "decremental_old_log_probs": old_log_probs + torch.randn(batch_size, length, generator=generator),
        "old_entropies": torch.rand(batch_size, length, generator=generator) * 2,
        "advantages": torch.randn(batch_size, 1, generator=generator).expand(batch_size, length) * response_mask,
    }


def _expected_top_mask(values: torch.Tensor, response_mask: torch.Tensor, quantile: float) -> torch.Tensor:
    valid = response_mask.bool()
    flat = values[valid]
    keep = math.ceil(flat.numel() * quantile)
    threshold = torch.sort(flat, descending=True).values[keep - 1]
    return (values >= threshold) & valid


def test_batch_masks_are_taken_over_the_whole_rollout_batch():
    batch = _batch()
    loss_config = build_perception_reasoning_loss_config(_tor_config())
    masks, metrics = build_batch_token_masks(loss_config, batch)

    expected_entropy = _expected_top_mask(batch["old_entropies"], batch["response_mask"], 0.3)
    scores = (batch["old_log_probs"] - batch["decremental_old_log_probs"]).abs()
    expected_perception = _expected_top_mask(scores, batch["response_mask"], 0.3)
    assert torch.equal(masks["batch_entropy_mask"], expected_entropy)
    assert torch.equal(masks["batch_perception_mask"], expected_perception)
    assert metrics["algo/token_selection/entropy_fraction"] == pytest.approx(0.3, abs=0.05)


def test_tokens_tied_at_the_threshold_are_drawn_at_random():
    """With top-p entropies most values are exactly 0; the tied tokens must not come in runs of responses."""
    generator = torch.Generator().manual_seed(0)
    rows, length = 64, 256
    positive = torch.rand(rows, length, generator=generator) < 0.2
    entropies = torch.where(positive, torch.rand(rows, length, generator=generator) + 0.01, torch.zeros(rows, length))
    batch = {"response_mask": torch.ones(rows, length), "old_entropies": entropies}
    loss_config = build_perception_reasoning_loss_config(_tor_config(top_perception_quantile=1.0))
    keep = math.ceil(rows * length * 0.3)

    in_order, _ = build_batch_token_masks(loss_config, batch)
    drawn, metrics = build_batch_token_masks(loss_config, batch, tie_break_seed=7)
    for masks in (in_order, drawn):
        mask = masks["batch_entropy_mask"]
        assert int(mask.sum()) == keep and bool(mask[positive].all())  # exactly 30%, every positive entropy
    assert metrics["algo/token_selection/entropy_threshold"] == 0.0

    tied_per_response = (drawn["batch_entropy_mask"] & ~positive).sum(dim=1).float()
    expected = (keep - int(positive.sum())) * (~positive).sum(dim=1).float() / int((~positive).sum())
    assert int(((in_order["batch_entropy_mask"] & ~positive).sum(dim=1) > 0).sum()) < rows // 2  # sort order: runs
    assert bool((tied_per_response > 0).all())
    # binomial-sized spread around each response's share of the tied tokens (about 26 per response here)
    assert float((tied_per_response - expected).abs().max()) < 5 * float(expected.mean()) ** 0.5

    again, _ = build_batch_token_masks(loss_config, batch, tie_break_seed=7)
    other, _ = build_batch_token_masks(loss_config, batch, tie_break_seed=8)
    assert torch.equal(again["batch_entropy_mask"], drawn["batch_entropy_mask"])
    assert not torch.equal(other["batch_entropy_mask"], drawn["batch_entropy_mask"])


def test_the_tie_break_seed_does_not_change_masks_without_ties():
    batch = _batch(seed=2)
    loss_config = build_perception_reasoning_loss_config(_tor_config())
    masks, _ = build_batch_token_masks(loss_config, batch)
    seeded, _ = build_batch_token_masks(loss_config, batch, tie_break_seed=3)
    for key in ("batch_entropy_mask", "batch_perception_mask"):
        assert torch.equal(masks[key], seeded[key])


def test_batch_masks_do_not_depend_on_the_micro_batch_split():
    """The loss uses the driver masks as they are, whichever micro-batch a response lands in."""
    batch = _batch(seed=1)
    loss_config = build_perception_reasoning_loss_config(_tor_config())
    masks, _ = build_batch_token_masks(loss_config, batch)
    actor_config = ActorConfig(loss_avg_mode="token")

    def token_losses(rows: slice) -> torch.Tensor:
        log_prob = batch["old_log_probs"][rows].clone().requires_grad_(True)
        loss, _ = compute_perception_reasoning_policy_loss(
            actor_config=actor_config,
            loss_config=loss_config,
            log_prob=log_prob,
            old_log_prob=batch["old_log_probs"][rows],
            advantages=batch["advantages"][rows],
            response_mask=batch["response_mask"][rows],
            entropy=None,
            decremental_old_log_probs=batch["decremental_old_log_probs"][rows],
            decremental_entropies=None,
            incremental_old_log_probs=None,
            incremental_entropies=None,
            batch_entropy_mask=masks["batch_entropy_mask"][rows],
            batch_perception_mask=masks["batch_perception_mask"][rows],
        )
        loss.backward()
        return log_prob.grad * batch["response_mask"][rows].sum()

    whole = token_losses(slice(0, 8))
    halves = torch.cat([token_losses(slice(0, 3)), token_losses(slice(3, 8))])
    # per-token weights (and hence the selected tokens) agree; only the per-call token-mean scale differs
    assert torch.equal(whole != 0, halves != 0)


def test_micro_batch_granularity_keeps_selecting_within_the_given_tensor():
    config = _tor_config(entropy_thr_granularity="micro_batch", perception_thr_granularity="micro_batch")
    loss_config = build_perception_reasoning_loss_config(config)
    assert needs_current_policy_entropy(loss_config)
    batch = _batch(seed=2)
    log_prob = batch["old_log_probs"].clone().requires_grad_(True)
    entropy = batch["old_entropies"]
    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=ActorConfig(loss_avg_mode="token"),
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=batch["old_log_probs"],
        advantages=batch["advantages"],
        response_mask=batch["response_mask"],
        entropy=entropy,
        decremental_old_log_probs=batch["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )
    expected = _expected_top_mask(entropy, batch["response_mask"], 0.3)
    assert metrics["algo/token_selection/entropy_fraction"] == pytest.approx(
        float(expected.sum() / batch["response_mask"].sum())
    )
    assert not needs_current_policy_entropy(build_perception_reasoning_loss_config(_tor_config()))


def test_top_p_entropy():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    kept = torch.tensor([0.5, 0.3]) / 0.8  # tokens ranked above have mass 0 and 0.5 (< 0.8); the third has 0.8
    expected = -(kept * kept.log()).sum()
    assert top_p_entropy_from_logits(logits[None], 0.8)[0] == pytest.approx(float(expected), rel=1e-5)
    assert top_p_entropy_from_logits(logits[None], 1.0)[0] == pytest.approx(
        float(entropy_from_logits(logits[None])[0])
    )
    assert top_p_entropy_from_logits(logits[None], 0.1)[0] == pytest.approx(0.0, abs=1e-6)  # only the top token


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"entropy_top_p": 0.95, "entropy_thr_granularity": "micro_batch"}, "entropy_top_p"),
        ({"visual_sensitivity_reference": "current"}, "visual_sensitivity_reference=old"),
        ({"entropy_thr_granularity": "global"}, "entropy_thr_granularity"),
    ],
)
def test_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tor_config(**overrides)


class _TableActor(torch.nn.Module):
    """Logits looked up from a fixed random table by the input token."""

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.randn(vocab_size, vocab_size, generator=torch.Generator().manual_seed(0))
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        return SimpleNamespace(logits=self.table[input_ids])


@pytest.mark.parametrize("top_p", [1.0, 0.9])
def test_compute_log_prob_returns_the_rollout_policy_entropy(top_p):
    """The entropy behind batch-level entropy masks comes from compute_log_prob, without gradient."""
    config = ActorConfig(padding_free=False, use_torch_compile=False, micro_batch_size_per_device_for_experience=2)
    actor = DataParallelPPOActor(config=config, actor_module=_TableActor())
    actor.log_probs_from_logits = lambda logits, labels: (  # the default kernel needs CUDA tensors
        torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    )
    input_ids = torch.randint(0, 16, (3, 7), generator=torch.Generator().manual_seed(1))
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(7).expand(3, -1),
            "responses": input_ids[:, -4:],
        },
        meta_info={"temperature": 1.0},
    )
    _, entropy = actor.compute_log_prob(data, return_entropy=True, entropy_top_p=top_p)

    logits = actor.actor_module.table[input_ids[:, -5:-1]].detach()
    expected = top_p_entropy_from_logits(logits, top_p) if top_p < 1 else entropy_from_logits(logits)
    torch.testing.assert_close(entropy, expected)
    assert not entropy.requires_grad
