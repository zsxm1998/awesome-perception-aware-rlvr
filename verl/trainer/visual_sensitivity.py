# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Visual sensitivity signal computation.

This module is the single home for *computing* visual sensitivity signals — how much a
response token's prediction depends on the visual input, measured by comparing the policy
on the normal view against an auxiliary (corrupted) view. Signal *consumption* (losses,
perception masks, advantage scaling) lives in ``perception_reasoning_loss.py``.

Metric families
---------------
- Sampled metrics (``SAMPLED_SENSITIVITY_METRICS``) compare the two views only at the
  sampled tokens, using the normal/corrupted log-probs. They need one extra forward for
  the corrupted-view log-probs but no logits storage.
    * ``sampled_low_var_kl``: the k3 estimator ``r - log r - 1`` with
      ``r = q/p`` (``p`` = normal-view prob, ``q`` = corrupted-view prob). Non-negative,
      ratio-based, heavy-tailed; hard-clamped to ``[-10, 10]``.
    * ``sampled_boxcox``: the Box-Cox power gap ``(p^α - q^α) / α`` (PowerOPD-style).
      Natively bounded to ``[-1/α, 1/α]`` and sign-consistent with the log-ratio:
      positive iff the normal view makes the token more likely. ``α → 0`` recovers the
      sampled log-ratio ``log p - log q``.
- Full-vocab metrics (``FULL_VOCAB_SENSITIVITY_METRICS``) compare the whole output
  distributions of the two views (JSD/KL/Hellinger/entropy gap); they require response
  logits from both views. ``vepo`` is VEPO's method-specific recipe: JSD and the absolute
  entropy gap fused with a soft-or (weighted by ``visual_sensitivity_jsd_weight``);
  the other full-vocab names are generic mathematical measures.
- ``hidden_state_similarity`` derives scores from hidden states in a single forward.

When a full-vocab or hidden-state metric is configured but a *sampled* signal is still
required (the PAPO sensitivity loss, or driver-side fallbacks), the sampled signal
defaults to ``sampled_low_var_kl`` — see ``resolve_sampled_sensitivity_metric``.

Metric-key naming convention
----------------------------
Signal layer:

- ``algo/sensitivity/<estimator>``: masked mean of a per-token signal under its
  estimator's true name — the key tail always equals the ``visual_sensitivity_metric``
  string that selects it (``sampled_low_var_kl``, ``sampled_boxcox``,
  ``full_vocab_jsd``, ...). Emitted where the signal is computed: sampled estimators at
  loss time, full-vocab components in the actor's score pass. Only the active estimator
  is computed by default; extra sampled estimators can be logged for diagnosis via
  ``visual_sensitivity_log_metrics`` (detached, never trained on).
- ``algo/sensitivity/vepo_fused``: VEPO's fused signal (post fusion and entropy gate),
  the final signal under ``visual_sensitivity_metric='vepo'``.
- ``algo/sensitivity/active``: consumption-stage key — masked mean of the per-token
  signal actually used for token credit assignment (perception masks / advantage
  scaling) in the policy loss, whichever family produced it. Exactly one per step
  whenever such a signal is consumed. It is separate from the computation-stage keys
  because the two stages have different batch semantics (micro-batch means across PPO
  epochs vs one pass over the rollout batch).
- ``algo/sensitivity/diag_topq_overlap_{active}_vs_{extra}`` and
  ``algo/sensitivity/diag_response_rank_corr_{active}_vs_{extra}``: comparison
  diagnostics between the active signal and each ``visual_sensitivity_log_metrics``
  entry, auto-named after the pair actually compared. Overlap is |A∩B|/k of the two
  top-quantile token sets at ``top_perception_quantile`` (emitted only when that
  quantile is < 1; the random-selection baseline equals the quantile itself); rank
  correlation is Spearman over response-mean scores — the invariant deciding whether
  quantile masks and monotone response scalings would change under a signal swap.

Consumption-side namespaces (emitted from ``perception_reasoning_loss.py``):

- ``algo/token_selection/*``: top-quantile token masks — thresholds and selected
  fractions (``entropy_*``, ``perception_*``, ``combined_fraction``).
- ``algo/advantage_scaling/*``: advantage-modulation mechanics — ``factor_*``,
  ``schedule_progress``, ``response_signal_mean``.
- ``algo/{vppo,cgpo,pgpo,pepo,tor,dvrp}/*``: method-internal dynamic quantities only;
  config constants are not echoed as metrics.

"""

import math

import torch
import torch.nn.functional as F


SAMPLED_SENSITIVITY_METRICS = ("sampled_low_var_kl", "sampled_boxcox")
FULL_VOCAB_SENSITIVITY_METRICS = (
    "full_vocab_jsd",
    "full_vocab_kl",
    "full_vocab_hellinger",
    "full_vocab_abs_entropy_gap",
    "vepo",
)
HIDDEN_STATE_SENSITIVITY_METRIC = "hidden_state_similarity"
VISUAL_SENSITIVITY_METRIC_CHOICES = frozenset(
    SAMPLED_SENSITIVITY_METRICS + FULL_VOCAB_SENSITIVITY_METRICS + (HIDDEN_STATE_SENSITIVITY_METRIC,)
)


def is_sampled_sensitivity_metric(metric: str) -> bool:
    return metric in SAMPLED_SENSITIVITY_METRICS


def is_full_vocab_sensitivity_metric(metric: str) -> bool:
    return metric in FULL_VOCAB_SENSITIVITY_METRICS


def resolve_sampled_sensitivity_metric(metric: str) -> str:
    """Effective sampled metric backing the loss/fallback paths.

    Full-vocab and hidden-state configurations historically keep the sampled low-var KL
    as the signal wherever a sampled signal is needed (e.g. the PAPO sensitivity loss);
    only sampled metrics change it.
    """
    return metric if is_sampled_sensitivity_metric(metric) else "sampled_low_var_kl"


# ---------------------------------------------------------------------------
# Sampled (token-level) estimators
# ---------------------------------------------------------------------------


def compute_sampled_low_var_kl(
    corrupted_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    reference_mode: str,
) -> torch.Tensor:
    """k3 estimator of KL(reference || corrupted) at the sampled tokens (KL_prcp in PAPO)."""
    log_diff = torch.clamp(corrupted_log_probs - reference_log_probs, min=-20.0, max=20.0)
    kl_prcp = torch.exp(log_diff) - log_diff - 1.0
    kl_prcp_min = -10.0 if reference_mode == "current" else 0.0
    return torch.clamp(kl_prcp, min=kl_prcp_min, max=10.0).contiguous()


def compute_sampled_boxcox(
    corrupted_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Box-Cox power gap ``(p^α - q^α) / α`` at the sampled tokens.

    ``p``/``q`` are the reference/corrupted-view token probabilities. The score is
    natively bounded to ``[-1/α, 1/α]`` and shares the log-ratio's sign (positive iff
    the reference view makes the token more likely); ``α → 0`` recovers
    ``log p - log q``. The 1/α denominator is kept (not folded into a coefficient or the
    learning rate) so scores stay in nats-like units comparable across α sweeps and with
    the sampled-KL baseline.
    """
    if alpha <= 0.0:
        raise ValueError(f"sampled_boxcox requires alpha > 0, but got {alpha}.")
    if reference_log_probs.dtype in (torch.float16, torch.bfloat16):
        reference_log_probs = reference_log_probs.float()
    if corrupted_log_probs.dtype in (torch.float16, torch.bfloat16):
        corrupted_log_probs = corrupted_log_probs.float()
    reference_powers = torch.exp(alpha * reference_log_probs)
    corrupted_powers = torch.exp(alpha * corrupted_log_probs)
    return ((reference_powers - corrupted_powers) / alpha).contiguous()


def compute_sampled_sensitivity_scores(
    metric: str,
    corrupted_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    reference_mode: str = "current",
    boxcox_alpha: float = 1.0,
) -> torch.Tensor:
    """Dispatch a sampled sensitivity metric to its estimator."""
    if metric == "sampled_low_var_kl":
        return compute_sampled_low_var_kl(
            corrupted_log_probs=corrupted_log_probs,
            reference_log_probs=reference_log_probs,
            reference_mode=reference_mode,
        )
    if metric == "sampled_boxcox":
        return compute_sampled_boxcox(
            corrupted_log_probs=corrupted_log_probs,
            reference_log_probs=reference_log_probs,
            alpha=boxcox_alpha,
        )
    raise ValueError(f"Unsupported sampled visual_sensitivity_metric: {metric}")


def compute_response_mean_scores(
    per_token_scores: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked per-response mean of token scores, plus a validity flag per response."""
    num_valid_tokens = response_mask.sum(dim=1)
    valid_scores_mask = num_valid_tokens > 0
    response_scores = (per_token_scores * response_mask).sum(dim=1) / num_valid_tokens.clamp(min=1.0)
    return response_scores, valid_scores_mask


def compute_spearman_rank_correlation(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Spearman rank correlation between two 1-D score vectors over valid entries.

    Returns NaN when fewer than two valid entries exist or either vector is constant
    (ranks would be arbitrary there, not meaningless-but-plausible numbers).
    """
    valid_mask = valid_mask.to(torch.bool)
    values_a = values_a[valid_mask].detach().float()
    values_b = values_b[valid_mask].detach().float()
    if values_a.numel() < 2 or values_a.max() == values_a.min() or values_b.max() == values_b.min():
        return torch.tensor(float("nan"), device=values_a.device)
    ranks_a = _rank_values(values_a)
    ranks_b = _rank_values(values_b)
    ranks_a = ranks_a - ranks_a.mean()
    ranks_b = ranks_b - ranks_b.mean()
    return (ranks_a * ranks_b).sum() / (ranks_a.norm() * ranks_b.norm())


def _rank_values(values: torch.Tensor) -> torch.Tensor:
    """Fractional ranks: ties receive the average of their ordinal ranks (standard Spearman)."""
    order = torch.argsort(values)
    ordinal_ranks = torch.empty_like(values)
    ordinal_ranks[order] = torch.arange(values.numel(), dtype=values.dtype, device=values.device)
    unique_values, inverse_indices = torch.unique(values, return_inverse=True)
    rank_sums = torch.zeros(unique_values.numel(), dtype=values.dtype, device=values.device)
    rank_sums.index_add_(0, inverse_indices, ordinal_ranks)
    counts = torch.zeros(unique_values.numel(), dtype=values.dtype, device=values.device)
    counts.index_add_(0, inverse_indices, torch.ones_like(values))
    return (rank_sums / counts)[inverse_indices]


# ---------------------------------------------------------------------------
# Full-vocab (distribution-level) metrics
# ---------------------------------------------------------------------------


def _required_full_vocab_component_names(metric: str, entropy_gate: str = "none") -> set[str]:
    if metric == "full_vocab_jsd":
        component_names = {"jsd"}
    elif metric == "full_vocab_kl":
        component_names = {"kl"}
    elif metric == "full_vocab_hellinger":
        component_names = {"hellinger"}
    elif metric == "full_vocab_abs_entropy_gap":
        component_names = {"abs_entropy_gap"}
    elif metric == "vepo":
        component_names = {"jsd", "abs_entropy_gap"}
    else:
        raise ValueError(f"Unsupported visual_sensitivity_metric: {metric}")

    if entropy_gate == "normal_entropy":
        component_names.add("entropy")
    elif entropy_gate != "none":
        raise ValueError(f"Unsupported visual_sensitivity_entropy_gate: {entropy_gate}")
    return component_names


def compute_full_vocab_visual_sensitivity_scores(
    logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    response_mask: torch.Tensor,
    metric: str,
    jsd_weight: float = 0.5,
    entropy_gate: str = "none",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    component_names = _required_full_vocab_component_names(metric=metric, entropy_gate=entropy_gate)
    components = compute_full_vocab_visual_sensitivity_components(
        logits=logits,
        corrupted_logits=corrupted_logits,
        component_names=component_names,
        eps=eps,
    )
    return combine_full_vocab_visual_sensitivity_scores(
        components=components,
        response_mask=response_mask,
        metric=metric,
        jsd_weight=jsd_weight,
        entropy_gate=entropy_gate,
    )


def compute_full_vocab_visual_sensitivity_components(
    logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    component_names: set[str] | None = None,
    eps: float = 1e-8,
    chunk_size: int = 32,
) -> dict[str, torch.Tensor]:
    if logits.shape != corrupted_logits.shape:
        raise ValueError(
            "Full-vocab visual sensitivity requires normal and corrupted logits to have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(corrupted_logits.shape)}."
        )
    if component_names is None:
        component_names = {
            "jsd",
            "kl",
            "hellinger",
            "entropy",
            "corrupted_entropy",
            "abs_entropy_gap",
        }
    component_names = set(component_names)
    if "abs_entropy_gap" in component_names:
        component_names.update({"entropy", "corrupted_entropy"})

    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_corrupted_logits = corrupted_logits.reshape(-1, vocab_size)
    if flat_logits.size(0) == 0:
        empty = logits.new_zeros(original_shape, dtype=torch.float32)
        return dict.fromkeys(component_names, empty)

    component_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in component_names}
    log_eps = math.log(eps)

    for logits_chunk, corrupted_logits_chunk in zip(
        flat_logits.split(chunk_size, dim=0),
        flat_corrupted_logits.split(chunk_size, dim=0),
        strict=True,
    ):
        log_probs = F.log_softmax(logits_chunk.float(), dim=-1)
        corrupted_log_probs = F.log_softmax(corrupted_logits_chunk.float(), dim=-1)
        safe_log_probs = torch.clamp(log_probs, min=log_eps)
        safe_corrupted_log_probs = torch.clamp(corrupted_log_probs, min=log_eps)
        probs = torch.exp(log_probs)
        corrupted_probs = torch.exp(corrupted_log_probs)

        if "entropy" in component_names:
            component_chunks["entropy"].append(-(probs * safe_log_probs).sum(dim=-1))
        if "corrupted_entropy" in component_names:
            component_chunks["corrupted_entropy"].append(-(corrupted_probs * safe_corrupted_log_probs).sum(dim=-1))
        if "jsd" in component_names:
            mean_log_probs = torch.log(0.5 * (probs + corrupted_probs) + eps)
            jsd_chunk = 0.5 * (probs * (safe_log_probs - mean_log_probs)).sum(dim=-1)
            jsd_chunk = jsd_chunk + 0.5 * (corrupted_probs * (safe_corrupted_log_probs - mean_log_probs)).sum(dim=-1)
            component_chunks["jsd"].append(torch.clamp(jsd_chunk, min=0.0))
        if "kl" in component_names:
            kl_chunk = (probs * (safe_log_probs - safe_corrupted_log_probs)).sum(dim=-1)
            component_chunks["kl"].append(torch.clamp(kl_chunk, min=0.0))
        if "hellinger" in component_names:
            hellinger_chunk = torch.sqrt(0.5 * (torch.sqrt(probs) - torch.sqrt(corrupted_probs)).pow(2).sum(dim=-1))
            component_chunks["hellinger"].append(hellinger_chunk)

    components = {
        name: torch.cat(chunks, dim=0).reshape(original_shape)
        for name, chunks in component_chunks.items()
        if name != "abs_entropy_gap"
    }
    if "abs_entropy_gap" in component_names:
        components["abs_entropy_gap"] = torch.abs(components["corrupted_entropy"] - components["entropy"])
    return components


def combine_full_vocab_visual_sensitivity_scores(
    components: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    metric: str,
    jsd_weight: float = 0.5,
    entropy_gate: str = "none",
) -> tuple[torch.Tensor, dict[str, float]]:
    components = {key: value.float() for key, value in components.items()}
    valid_mask = response_mask.to(torch.bool)
    response_dtype = next(iter(components.values())).dtype
    response_mask = response_mask.to(dtype=response_dtype)
    if metric == "full_vocab_jsd":
        scores = components["jsd"]
    elif metric == "full_vocab_kl":
        scores = components["kl"]
    elif metric == "full_vocab_hellinger":
        scores = components["hellinger"]
    elif metric == "full_vocab_abs_entropy_gap":
        scores = components["abs_entropy_gap"]
    elif metric == "vepo":
        # VEPO recipe: soft-or fusion of JSD and the absolute entropy gap.
        jsd_norm = minmax_normalize_by_response(components["jsd"], valid_mask)
        gap_norm = minmax_normalize_by_response(components["abs_entropy_gap"], valid_mask)
        scores = 1.0 - (1.0 - jsd_norm).pow(jsd_weight) * (1.0 - gap_norm).pow(1.0 - jsd_weight)
    else:
        raise ValueError(f"Unsupported visual_sensitivity_metric: {metric}")

    if entropy_gate == "normal_entropy":
        scores = scores * minmax_normalize_by_response(components["entropy"], valid_mask)
    elif entropy_gate != "none":
        raise ValueError(f"Unsupported visual_sensitivity_entropy_gate: {entropy_gate}")

    scores = scores * response_mask
    total_valid = response_mask.sum().clamp(min=1)
    metrics = {}
    for name, values in components.items():
        metrics[f"algo/sensitivity/full_vocab_{name}"] = to_float((values * response_mask).sum() / total_valid)
    if metric == "vepo":
        metrics["algo/sensitivity/vepo_fused"] = to_float((scores * response_mask).sum() / total_valid)
    return scores, metrics


# ---------------------------------------------------------------------------
# Shared small helpers
# ---------------------------------------------------------------------------


def minmax_normalize_by_response(values: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = values.float()
    masked_for_min = values.masked_fill(~valid_mask, float("inf"))
    min_values = masked_for_min.min(dim=-1, keepdim=True).values
    masked_for_max = values.masked_fill(~valid_mask, float("-inf"))
    max_values = masked_for_max.max(dim=-1, keepdim=True).values
    normalized = (values - min_values) / (max_values - min_values).clamp(min=eps)
    return normalized.masked_fill(~valid_mask, 0.0).clamp(0.0, 1.0)


def masked_vector_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_values = values[valid_mask]
    if valid_values.numel() == 0:
        return torch.tensor(float("nan"), device=values.device)
    return valid_values.mean()


def masked_vector_min(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_values = values[valid_mask]
    if valid_values.numel() == 0:
        return torch.tensor(float("nan"), device=values.device)
    return valid_values.min()


def masked_vector_max(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid_values = values[valid_mask]
    if valid_values.numel() == 0:
        return torch.tensor(float("nan"), device=values.device)
    return valid_values.max()


def to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(value)
