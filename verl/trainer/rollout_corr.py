# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""How far the inference engine's log-probabilities are from the trainer's (worker.rollout.record_rollout_log_probs).

``compute_offpolicy_metrics`` is verl's (v0.9.1, ``verl/trainer/ppo/rollout_corr_helper.py``), logged under the same
``rollout_corr/`` prefix. They are diagnostics only: the loss does not use the rollout log-probabilities.
"""

from typing import Any

import torch

from ..utils.torch_functional import masked_mean


SAFETY_BOUND = 20.0  # bound of log importance ratios before exponentiation, as in verl


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor, rollout_log_prob: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Any]:
    """The gap between the training policy (``old_log_prob``, recomputed by the trainer) and the rollout policy
    (``rollout_log_prob``, from the inference engine) on the model tokens (``response_mask``); shapes
    (batch_size, response_length).

    - kl: E[log pi_rollout - log pi_training], k3_kl: E[r - log r - 1] with r = pi_training / pi_rollout;
    - training_ppl / rollout_ppl: batch means of the per-sequence perplexities (and their logs);
    - log_ppl_diff: per-sequence mean log-prob of the rollout minus that of the training policy (mean, mean of the
      absolute values, max, min); ppl_ratio: mean of exp(log_ppl_diff);
    - chi2_token: E[rho^2] - 1 over tokens, chi2_seq: E[(prod rho_t)^2] - 1 over sequences, rho = pi_training /
      pi_rollout.
    """
    response_mask = response_mask.to(old_log_prob.dtype)
    if not bool(response_mask.any()):
        raise ValueError("the off-policy metrics need at least one model token")
    old_log_prob = old_log_prob.float()
    rollout_log_prob = rollout_log_prob.float()
    response_mask = response_mask.float()
    metrics = {}

    mean_log_prob_training = masked_mean(old_log_prob, response_mask, dim=-1)
    metrics["training_ppl"] = torch.exp(-mean_log_prob_training).mean().item()
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().item()

    metrics["kl"] = masked_mean(rollout_log_prob - old_log_prob, response_mask).item()
    log_ratio = old_log_prob - rollout_log_prob
    metrics["k3_kl"] = masked_mean(torch.exp(log_ratio) - log_ratio - 1, response_mask).item()

    mean_log_prob_rollout = masked_mean(rollout_log_prob, response_mask, dim=-1)
    metrics["rollout_ppl"] = torch.exp(-mean_log_prob_rollout).mean().item()
    metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().item()

    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    metrics["log_ppl_diff"] = log_ppl_diff.mean().item()
    metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().item()
    metrics["log_ppl_diff_max"] = log_ppl_diff.max().item()
    metrics["log_ppl_diff_min"] = log_ppl_diff.min().item()
    metrics["ppl_ratio"] = torch.exp(log_ppl_diff).mean().item()

    rho_token = torch.exp(torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND))
    metrics["chi2_token"] = (masked_mean(rho_token.square(), response_mask) - 1.0).item()
    log_ratio_sum = (log_ratio * response_mask).sum(dim=-1)
    rho_squared_seq = torch.exp(2.0 * torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND))
    metrics["chi2_seq"] = (rho_squared_seq.mean() - 1.0).item()
    return {f"rollout_corr/{name}": value for name, value in metrics.items()}
