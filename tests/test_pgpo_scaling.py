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
"""PGPO token weights: the paper's form and the authors' development code (Yzk1114/EasyR1)."""

import pytest
import torch

from verl.trainer.config import AlgorithmConfig
from verl.trainer.perception_reasoning_loss import _compute_pgpo_token_scaling


def _development_code_weights(k3_scores, mask, threshold, gating_beta):
    """Yzk1114/EasyR1@6259f02 verl/workers/actor/dp_actor.py (vig, vig_entropy off), from the clamped k3 score on;
    returns the factor the advantages are multiplied by."""
    epsilon = 1e-8
    mask_bool = mask.bool()
    scores = torch.log1p(k3_scores)
    scores = scores * mask
    inf_mask = ~mask_bool
    min_scores = scores.masked_fill(inf_mask, float("inf")).min(dim=1, keepdim=True).values
    max_scores = scores.masked_fill(inf_mask, float("-inf")).max(dim=1, keepdim=True).values
    range_val = max_scores - min_scores
    range_val = torch.where(range_val > epsilon, range_val, torch.ones_like(range_val))
    weighted_vig_reward = (scores - min_scores) / range_val
    norm_importance = torch.clamp(weighted_vig_reward, 0.0, 1.0)
    norm_importance = norm_importance * mask

    valid_lengths = mask.sum(dim=1).long()
    quantile_indices = (valid_lengths.float() * threshold).long()
    max_indices = (valid_lengths - 1).clamp(min=0)
    quantile_indices = torch.min(quantile_indices.clamp(min=0), max_indices)
    values_sorted, _ = torch.sort(norm_importance.masked_fill(inf_mask, float("inf")), dim=1)
    row_thresholds = values_sorted.gather(1, quantile_indices.unsqueeze(1))

    weights = torch.ones_like(norm_importance)
    is_low = (norm_importance < row_thresholds) & mask_bool
    is_high = (~is_low) & mask_bool
    low_vals = norm_importance / (row_thresholds + epsilon)
    low_vals = torch.clamp(low_vals, min=0.1)
    weights = torch.where(is_low, low_vals, weights)
    if is_high.any():
        relative_pos = (norm_importance - row_thresholds) / (1.0 - row_thresholds + epsilon)
        high_vals = 1.0 + gating_beta * relative_pos
        weights = torch.where(is_high, high_vals, weights)
    return weights * mask


def _batch(seed=0, batch_size=6, length=20):
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, length + 1, (batch_size,), generator=generator)
    mask = (torch.arange(length)[None, :] < lengths[:, None]).float()
    vig_scores = torch.randn(batch_size, length, generator=generator) * 2  # log p(clean) - log p(blind)
    k3 = -vig_scores.clamp(-20.0, 20.0)
    k3 = torch.clamp(torch.exp(k3) - k3 - 1, min=0.0, max=10.0)
    k3[0, : int(lengths[0])] = 0.5  # a response with one score everywhere
    return k3, mask


@pytest.mark.parametrize("threshold", [0.2, 0.4, 0.7])
def test_development_code_variant_matches_the_snapshot(threshold):
    k3, mask = _batch()
    scaling, metrics = _compute_pgpo_token_scaling(
        per_token_sensitivity=k3,
        response_mask=mask,
        threshold=threshold,
        boost=2.0,
        threshold_mode="quantile",
        low_weight_floor=0.1,
        mass_normalization=False,
    )
    torch.testing.assert_close(scaling, _development_code_weights(k3, mask, threshold, 2.0), rtol=1e-5, atol=1e-6)
    assert "algo/pgpo/threshold_mean" in metrics


def test_paper_variant_is_unchanged():
    """The defaults: absolute threshold, no floor, weights of each response rescaled to sum to its length."""
    k3, mask = _batch(seed=1)
    scaling, _ = _compute_pgpo_token_scaling(per_token_sensitivity=k3, response_mask=mask, threshold=0.4, boost=2.0)
    normalized = torch.log1p(k3)
    valid = mask.bool()
    low = normalized.masked_fill(~valid, float("inf")).min(-1, keepdim=True).values
    high = normalized.masked_fill(~valid, float("-inf")).max(-1, keepdim=True).values
    normalized = ((normalized - low) / (high - low).clamp(min=1e-8)).clamp(0, 1) * mask
    raw = torch.where(normalized < 0.4, normalized / 0.4, 1 + 2.0 * (normalized - 0.4) / 0.6) * mask
    expected = raw * mask.sum(-1, keepdim=True) / raw.sum(-1, keepdim=True)
    expected = torch.where(raw.sum(-1, keepdim=True) > 1e-8, expected, mask)
    torch.testing.assert_close(scaling, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(scaling.sum(-1), mask.sum(-1))


def test_switches_are_validated():
    with pytest.raises(ValueError, match="pgpo_threshold_mode"):
        AlgorithmConfig(pgpo_threshold_mode="median").post_init()
    with pytest.raises(ValueError, match="pgpo_low_weight_floor"):
        AlgorithmConfig(pgpo_low_weight_floor=-0.1).post_init()
