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

from typing import Any, Optional

import torch

from ..protocol import DataProto
from ..utils import torch_functional as VF
from ..workers.actor.config import ActorConfig
from .core_algos import average_loss, compute_policy_loss
from .visual_sensitivity import (
    compute_response_mean_scores,
    compute_sampled_low_var_kl,
    compute_sampled_sensitivity_scores,
    compute_spearman_rank_correlation,
    masked_vector_max,
    masked_vector_mean,
    masked_vector_min,
    minmax_normalize_by_response,
    resolve_sampled_sensitivity_metric,
    to_float,
)


def has_perception_reasoning(loss_config: dict[str, Any] | None) -> bool:
    if loss_config is None:
        return False
    return any(
        [
            loss_config.get("visual_sensitivity_loss_coef", 0.0) != 0.0,
            loss_config.get("decremental_entropy_coef", 0.0) != 0.0,
            loss_config.get("invariant_entropy_coef", 0.0) != 0.0,
            loss_config.get("top_entropy_quantile", 1.0) < 1.0,
            loss_config.get("top_perception_quantile", 1.0) < 1.0,
            _get_advantage_scaling_method(loss_config) is not None,
            loss_config.get("response_advantage_scaling_method") is not None,
            loss_config.get("tor_use_token_weighting", False),
            loss_config.get("visual_robustness_loss_coef", 0.0) != 0.0,
            loss_config.get("incremental_entropy_coef", 0.0) != 0.0,
        ]
    )


def uses_incremental_dvrp_mode(
    loss_config: dict[str, Any] | None,
    *,
    incremental_old_log_probs: torch.Tensor | None = None,
    incremental_entropies: torch.Tensor | None = None,
) -> bool:
    if not loss_config:
        return False
    return (
        loss_config.get("incremental_image_transform") is not None
        or loss_config.get("visual_robustness_loss_coef", 0.0) != 0.0
        or loss_config.get("incremental_entropy_coef", 0.0) != 0.0
        or incremental_old_log_probs is not None
        or incremental_entropies is not None
    )


def current_policy_entropy_requires_grad(loss_config: dict[str, Any] | None) -> bool:
    if not loss_config:
        return False
    return (
        loss_config.get("invariant_entropy_coef", 0.0) != 0.0
        and loss_config.get("entropy_loss_type", "sampled") == "full"
    )


def needs_current_policy_entropy(loss_config: dict[str, Any] | None) -> bool:
    if not loss_config:
        return False
    return (
        loss_config.get("log_entropy", False)
        or (
            loss_config.get("top_entropy_quantile", 1.0) < 1.0
            and loss_config.get("entropy_thr_granularity", "micro_batch") != "batch"
        )
        or _get_advantage_scaling_method(loss_config) == "pepo"
        or current_policy_entropy_requires_grad(loss_config)
    )


def uses_batch_entropy_mask(loss_config: dict[str, Any] | None) -> bool:
    """Whether the top-entropy mask is taken over the whole rollout batch (from the rollout policy's entropy)."""
    return bool(
        loss_config
        and loss_config.get("top_entropy_quantile", 1.0) < 1.0
        and loss_config.get("entropy_thr_granularity", "micro_batch") == "batch"
    )


def uses_batch_perception_mask(loss_config: dict[str, Any] | None) -> bool:
    """Whether the top-perception mask is taken over the whole rollout batch."""
    return bool(
        loss_config
        and loss_config.get("top_perception_quantile", 1.0) < 1.0
        and loss_config.get("perception_thr_granularity", "micro_batch") == "batch"
    )


def _mask_granularity(granularity: str) -> str:
    """``_compute_top_quantile_mask`` granularity for a configured one: ``micro_batch`` selects within the tensor it
    is given; ``batch`` masks are built on the driver over the whole rollout batch (``build_batch_token_masks``)."""
    return "response" if granularity == "response" else "batch"


def build_batch_token_masks(
    loss_config: dict[str, Any] | None,
    data: DataProto | dict[str, torch.Tensor] | Any,
    tie_break_seed: Optional[int] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Top-quantile token masks over every response token of the rollout batch (``*_thr_granularity=batch``).

    Runs on the driver before the update, so the thresholds do not depend on how the batch is split into
    micro-batches. The entropy comes from the rollout (old) policy (``old_entropies``); perception scores are the
    precomputed ``per_token_sensitivity_scores`` or the sampled metric between ``old_log_probs`` and
    ``decremental_old_log_probs``. Exactly the top fraction is kept; with ``tie_break_seed`` the tokens tied at the
    threshold (e.g. the many zero entropies of a top-p truncated distribution) are drawn uniformly at random
    instead of in sort order, which would take them in runs of consecutive responses.
    """
    masks: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}
    batch = data.batch if isinstance(data, DataProto) else data
    generator = None if tie_break_seed is None else torch.Generator().manual_seed(int(tie_break_seed))
    if uses_batch_entropy_mask(loss_config):
        if "old_entropies" not in batch:
            raise ValueError("entropy_thr_granularity=batch requires old_entropies from the rollout policy.")
        entropy_mask, threshold, fraction = _compute_top_quantile_mask(
            values=batch["old_entropies"].float(),
            response_mask=batch["response_mask"],
            quantile=loss_config["top_entropy_quantile"],
            granularity="batch",
            generator=generator,
        )
        masks["batch_entropy_mask"] = entropy_mask
        metrics["algo/token_selection/entropy_threshold"] = to_float(threshold)
        metrics["algo/token_selection/entropy_fraction"] = to_float(fraction)
    if uses_batch_perception_mask(loss_config):
        if "per_token_sensitivity_scores" in batch:
            scores = batch["per_token_sensitivity_scores"]
        elif "old_log_probs" in batch and "decremental_old_log_probs" in batch:
            scores = compute_sampled_sensitivity_scores(
                metric=resolve_sampled_sensitivity_metric(
                    str(loss_config.get("visual_sensitivity_metric", "sampled_low_var_kl"))
                ),
                corrupted_log_probs=batch["decremental_old_log_probs"],
                reference_log_probs=batch["old_log_probs"],
                reference_mode="old",
                boxcox_alpha=loss_config.get("visual_sensitivity_boxcox_alpha", 1.0),
            )
        else:
            raise ValueError(
                "perception_thr_granularity=batch requires per_token_sensitivity_scores or "
                "old_log_probs and decremental_old_log_probs before the update."
            )
        perception_mask, threshold, _ = _compute_top_quantile_mask(
            values=scores.float(),
            response_mask=batch["response_mask"],
            quantile=loss_config["top_perception_quantile"],
            granularity="batch",
            generator=generator,
        )
        masks["batch_perception_mask"] = perception_mask
        metrics["algo/token_selection/perception_threshold"] = to_float(threshold)
    return masks, metrics


def build_sensitivity_advantage_shaping_context(
    loss_config: dict[str, Any] | None,
    data: DataProto | dict[str, torch.Tensor] | Any,
) -> dict[str, float] | None:
    scaling_method = _get_advantage_scaling_method(loss_config)
    if not loss_config or scaling_method is None:
        return None
    if scaling_method in {"pgpo", "pepo"}:
        return None
    batch = data.batch if isinstance(data, DataProto) else data
    if "per_token_sensitivity_scores" in batch:
        per_token_sensitivity = batch["per_token_sensitivity_scores"]
    elif "old_log_probs" in batch and "decremental_old_log_probs" in batch:
        per_token_sensitivity = compute_sampled_sensitivity_scores(
            metric=resolve_sampled_sensitivity_metric(
                str(loss_config.get("visual_sensitivity_metric", "sampled_low_var_kl"))
            ),
            corrupted_log_probs=batch["decremental_old_log_probs"],
            reference_log_probs=batch["old_log_probs"],
            reference_mode="old",
            boxcox_alpha=loss_config.get("visual_sensitivity_boxcox_alpha", 1.0),
        )
    else:
        return None

    response_mask = batch["response_mask"].to(bool)
    sensitivity_scores, valid_scores_mask = compute_response_mean_scores(
        per_token_scores=per_token_sensitivity,
        response_mask=response_mask,
    )
    if scaling_method == "vppo":
        return _compute_vppo_advantage_shaping_context_from_scores(
            sensitivity_scores=sensitivity_scores,
            valid_scores_mask=valid_scores_mask,
        )
    if scaling_method == "cgpo":
        return _compute_cgpo_advantage_shaping_context_from_scores(
            sensitivity_scores=sensitivity_scores,
            valid_scores_mask=valid_scores_mask,
        )
    if (
        loss_config.get("advantage_scaling_method") is None
        and loss_config.get("response_advantage_scaling_method") is not None
    ):
        raise ValueError(f"Unknown response_advantage_scaling_method: {scaling_method}")
    raise ValueError(f"Unknown advantage_scaling_method: {scaling_method}")


def compute_perception_reasoning_policy_loss(
    actor_config: ActorConfig,
    loss_config: dict[str, Any],
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    entropy: torch.Tensor | None,
    decremental_old_log_probs: torch.Tensor | None,
    decremental_entropies: torch.Tensor | None,
    incremental_old_log_probs: torch.Tensor | None,
    incremental_entropies: torch.Tensor | None,
    region_token_mask: torch.Tensor | None = None,
    per_token_sensitivity_scores: torch.Tensor | None = None,
    advantage_shaping_context: dict[str, float] | None = None,
    batch_entropy_mask: torch.Tensor | None = None,
    batch_perception_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Unified policy loss for PAPO / VPPO / ToR / DVRP / VEPO / PGPO / PEPO / CFPO / CGPO.

    Pipeline:
      1. Build control signals (entropy_mask, perception_mask, etc.).
      2. Compute token-shaped advantage scaling_factors and modulate advantages
         (VPPO expands response factors to all tokens; CGPO scales perception tokens;
         PGPO computes token-level perception-grounded weights).
      3. Call ``compute_policy_loss(loss_avg_mode="none")`` to obtain token-wise PG terms.
      4. Apply token selection / reweighting on the PG terms — ToR weighted sum, or
         the union of entropy_mask and perception_mask.
      5. Add auxiliary terms: visual sensitivity (PAPO/VPPO; sampled KL by default, or
         the Box-Cox score-function form under `visual_sensitivity_metric='sampled_boxcox'`),
         invariant / decremental / incremental entropy (PAPO/DVRP), visual robustness KL (DVRP).

    Returns the scalar policy loss and a metrics dict. ``actor/pg_loss`` stays the
    raw PG loss (unmasked, comparable across algorithms); ``algo/loss/pg_loss_effective``
    is the mask/weight-applied PG loss that contributes to ``algo/loss/policy_loss``
    together with the auxiliary losses.
    """
    metrics: dict[str, float] = {}

    if entropy is not None and loss_config.get("log_entropy", False):
        metrics["actor/policy_entropy"] = to_float(VF.masked_mean(entropy, response_mask))

    # Step 1: Compute perception/reasoning control signals.
    entropy_mask = None
    if uses_batch_entropy_mask(loss_config):
        if batch_entropy_mask is None:
            raise ValueError("entropy_thr_granularity=batch requires the batch_entropy_mask built before the update.")
        entropy_mask = batch_entropy_mask.to(torch.bool) & response_mask.to(torch.bool)
    elif loss_config.get("top_entropy_quantile", 1.0) < 1.0:
        if entropy is None:
            raise ValueError("top_entropy_quantile requires current-policy entropy, but entropy was not computed.")
        entropy_mask, entropy_threshold, entropy_fraction = _compute_top_quantile_mask(
            values=entropy,
            response_mask=response_mask,
            quantile=loss_config["top_entropy_quantile"],
            granularity=_mask_granularity(loss_config.get("entropy_thr_granularity", "micro_batch")),
        )
        metrics["algo/token_selection/entropy_threshold"] = to_float(entropy_threshold)
        metrics["algo/token_selection/entropy_fraction"] = to_float(entropy_fraction)

    use_dvrp = uses_incremental_dvrp_mode(
        loss_config,
        incremental_old_log_probs=incremental_old_log_probs,
        incremental_entropies=incremental_entropies,
    )
    per_token_sampled_sensitivity = None
    extra_sampled_signals: dict[str, torch.Tensor] = {}
    sampled_metric = resolve_sampled_sensitivity_metric(
        str(loss_config.get("visual_sensitivity_metric", "sampled_low_var_kl"))
    )
    if decremental_old_log_probs is not None:
        requested_log_metrics = tuple(loss_config.get("visual_sensitivity_log_metrics") or ())
        sensitivity_reference_mode = loss_config.get("visual_sensitivity_reference", "current")
        sensitivity_reference = log_prob if sensitivity_reference_mode == "current" else old_log_prob
        boxcox_alpha = loss_config.get("visual_sensitivity_boxcox_alpha", 1.0)
        # Need-gated: the sampled signal exists only when something consumes or requests
        # it — the sensitivity loss, the perception fallback (no precomputed scores), or
        # an explicit visual_sensitivity_log_metrics entry.
        need_sampled_signal = (
            loss_config.get("visual_sensitivity_loss_coef", 0.0) != 0.0
            or per_token_sensitivity_scores is None
            or sampled_metric in requested_log_metrics
        )
        if need_sampled_signal:
            if sampled_metric == "sampled_boxcox":
                # Detached on purpose: the Box-Cox sensitivity loss re-attaches gradients
                # through its score-function form below, and mask/scaling consumers treat
                # sensitivity scores as data.
                per_token_sampled_sensitivity = compute_sampled_sensitivity_scores(
                    metric=sampled_metric,
                    corrupted_log_probs=decremental_old_log_probs,
                    reference_log_probs=sensitivity_reference.detach(),
                    reference_mode=sensitivity_reference_mode,
                    boxcox_alpha=boxcox_alpha,
                )
            elif sampled_metric == "sampled_abs_log_ratio":
                per_token_sampled_sensitivity = compute_sampled_sensitivity_scores(
                    metric=sampled_metric,
                    corrupted_log_probs=decremental_old_log_probs,
                    reference_log_probs=sensitivity_reference,
                    reference_mode=sensitivity_reference_mode,
                )
            else:
                per_token_sampled_sensitivity = compute_sampled_low_var_kl(
                    decremental_old_log_probs,
                    sensitivity_reference,
                    reference_mode=sensitivity_reference_mode,
                )
            metrics[f"algo/sensitivity/{sampled_metric}"] = to_float(
                VF.masked_mean(per_token_sampled_sensitivity, response_mask)
            )
        # Opt-in diagnostics: extra sampled estimators logged detached, never trained on.
        for extra_metric in requested_log_metrics:
            if extra_metric in extra_sampled_signals:
                continue
            if extra_metric == sampled_metric:
                extra_sampled_signals[extra_metric] = per_token_sampled_sensitivity
                continue
            extra_scores = compute_sampled_sensitivity_scores(
                metric=extra_metric,
                corrupted_log_probs=decremental_old_log_probs,
                reference_log_probs=sensitivity_reference.detach(),
                reference_mode=sensitivity_reference_mode,
                boxcox_alpha=boxcox_alpha,
            )
            metrics[f"algo/sensitivity/{extra_metric}"] = to_float(VF.masked_mean(extra_scores, response_mask))
            extra_sampled_signals[extra_metric] = extra_scores
    perception_scores = (
        per_token_sensitivity_scores if per_token_sensitivity_scores is not None else per_token_sampled_sensitivity
    )
    if perception_scores is not None:
        # Consumption-stage key: the signal actually used for token credit assignment.
        metrics["algo/sensitivity/active"] = to_float(VF.masked_mean(perception_scores, response_mask))
        comparison_signals = {
            name: scores for name, scores in extra_sampled_signals.items() if scores is not perception_scores
        }
        if comparison_signals:
            active_signal_name = (
                str(loss_config.get("visual_sensitivity_metric", "sampled_low_var_kl"))
                if per_token_sensitivity_scores is not None
                else sampled_metric
            )
            metrics.update(
                _compute_sensitivity_comparison_diagnostics(
                    active_name=active_signal_name,
                    active_scores=perception_scores,
                    extra_signals=comparison_signals,
                    response_mask=response_mask,
                    quantile=loss_config.get("top_perception_quantile", 1.0),
                    granularity=_mask_granularity(loss_config.get("perception_thr_granularity", "micro_batch")),
                )
            )

    perception_mask = None
    if uses_batch_perception_mask(loss_config):
        if batch_perception_mask is None:
            raise ValueError(
                "perception_thr_granularity=batch requires the batch_perception_mask built before the update."
            )
        perception_mask = batch_perception_mask.to(torch.bool) & response_mask.to(torch.bool)
    elif loss_config.get("top_perception_quantile", 1.0) < 1.0:
        if perception_scores is None:
            raise ValueError(
                "top_perception_quantile requires either decremental_old_log_probs or "
                "per_token_sensitivity_scores to be present."
            )
        perception_mask, perception_threshold, perception_fraction = _compute_top_quantile_mask(
            values=perception_scores,
            response_mask=response_mask,
            quantile=loss_config["top_perception_quantile"],
            granularity=_mask_granularity(loss_config.get("perception_thr_granularity", "micro_batch")),
        )
        del perception_fraction
        metrics["algo/token_selection/perception_threshold"] = to_float(perception_threshold)

    if (
        region_token_mask is not None
        and loss_config.get("include_region_tokens_in_perception_mask", False)
        and perception_mask is not None
    ):
        perception_mask = perception_mask | region_token_mask.to(torch.bool)

    if perception_mask is not None:
        valid_response_tokens = response_mask.to(torch.bool).sum().clamp(min=1).float()
        metrics["algo/token_selection/perception_fraction"] = to_float(
            perception_mask.sum().float() / valid_response_tokens
        )

    # Step 2: Compute and apply advantage modulation.
    scaling_method = _get_advantage_scaling_method(loss_config)
    scaling_factors = None
    if scaling_method is not None:
        if perception_scores is None:
            raise ValueError(
                "advantage_scaling_method requires either decremental_old_log_probs or per_token_sensitivity_scores."
            )
        scaling_factors, scaling_metrics = _compute_advantage_scaling(
            scaling_method=scaling_method,
            per_token_sensitivity=perception_scores,
            response_mask=response_mask,
            loss_config=loss_config,
            entropy=entropy,
            shaping_context=advantage_shaping_context,
            perception_mask=perception_mask,
        )
        scaling_factors, schedule_metrics = _apply_advantage_scaling_schedule(
            scaling_factors=scaling_factors,
            response_mask=response_mask,
            schedule=loss_config.get("advantage_scaling_schedule", "none"),
            progress=loss_config.get("advantage_scaling_schedule_progress", 1.0),
        )
        metrics.update(scaling_metrics)
        metrics.update(schedule_metrics)
        if scaling_method == "pepo" or schedule_metrics:
            metrics.update(_compute_token_scaling_factor_metrics(scaling_factors, response_mask))

    modulated_advantages = _apply_advantage_scaling(
        advantages=advantages,
        scaling_factors=scaling_factors,
    )

    # Step 3: Compute vanilla token-wise PG terms from the modulated advantages.
    pg_losses, pg_metrics = compute_policy_loss(
        old_log_probs=old_log_prob,
        log_probs=log_prob,
        advantages=modulated_advantages,
        response_mask=response_mask,
        clip_ratio_low=actor_config.clip_ratio_low,
        clip_ratio_high=actor_config.clip_ratio_high,
        clip_ratio_dual=actor_config.clip_ratio_dual,
        tau_positive=actor_config.tau_positive,
        tau_negative=actor_config.tau_negative,
        loss_type=actor_config.loss_type,
        loss_avg_mode="none",
    )
    metrics.update({f"actor/{key}": value for key, value in pg_metrics.items()})
    metrics["actor/pg_loss"] = to_float(average_loss(pg_losses, response_mask, mode=actor_config.loss_avg_mode))

    # Step 4: Apply token selection / token reweighting on the token-wise PG terms.
    weighted_pg_losses = pg_losses
    selected_token_mask = None
    if loss_config.get("tor_use_token_weighting", False):
        token_multiplier = _compute_tor_weight(
            entropy_mask=entropy_mask,
            perception_mask=perception_mask,
            tor_rsn_weight=loss_config.get("tor_rsn_weight", 1.0),
            tor_prcp_weight=loss_config.get("tor_prcp_weight", 0.5),
        )
        weighted_pg_losses = weighted_pg_losses * token_multiplier
        selected_token_mask = token_multiplier > 0
        metrics["algo/tor/effective_token_weight_mean"] = to_float(VF.masked_mean(token_multiplier, response_mask))
    elif entropy_mask is not None or perception_mask is not None:
        selected_token_mask = _combine_masks(entropy_mask, perception_mask)
        weighted_pg_losses = weighted_pg_losses * selected_token_mask.float()

    if selected_token_mask is not None:
        valid_response_tokens = response_mask.to(torch.bool).sum().clamp(min=1).float()
        metrics["algo/token_selection/combined_fraction"] = to_float(
            selected_token_mask.sum().float() / valid_response_tokens
        )

    pg_loss_mask = response_mask
    if selected_token_mask is not None and loss_config.get("normalize_pg_loss_by_selected_tokens", False):
        pg_loss_mask = selected_token_mask & response_mask.to(torch.bool)

    pg_loss = average_loss(weighted_pg_losses, pg_loss_mask, mode=actor_config.loss_avg_mode)
    metrics["algo/loss/pg_loss_effective"] = to_float(pg_loss)
    policy_loss = pg_loss

    # Step 5: Add auxiliary perception-reasoning losses.
    if loss_config.get("visual_sensitivity_loss_coef", 0.0) != 0.0:
        if per_token_sampled_sensitivity is None:
            raise ValueError("visual_sensitivity_loss_coef requires decremental_old_log_probs to be present.")
        if sampled_metric == "sampled_boxcox":
            # Score-function form: the bounded, detached Box-Cox score multiplies
            # log_prob so the per-token gradient is exactly -coef * score. A naive
            # pathwise swap would drop the corrupted branch from the gradient
            # (the detached q^alpha term differentiates to zero).
            per_token_sensitivity_objective = per_token_sampled_sensitivity.detach() * log_prob
        else:
            per_token_sensitivity_objective = per_token_sampled_sensitivity
        sensitivity_loss = average_loss(
            -loss_config["visual_sensitivity_loss_coef"] * per_token_sensitivity_objective,
            response_mask,
            mode=actor_config.loss_avg_mode,
        )
        policy_loss = policy_loss + sensitivity_loss

    invariant_entropy_coef = loss_config.get("invariant_entropy_coef", 0.0)
    if invariant_entropy_coef != 0.0:
        entropy_term = _compute_entropy_term(
            log_prob=log_prob,
            entropy=entropy,
            entropy_loss_type=loss_config["entropy_loss_type"],
        )
        invariant_entropy_loss = average_loss(
            invariant_entropy_coef * entropy_term,
            response_mask,
            mode=actor_config.loss_avg_mode,
        )
        policy_loss = policy_loss + invariant_entropy_loss

    decremental_entropy_coef = loss_config.get("decremental_entropy_coef", 0.0)
    if decremental_entropy_coef != 0.0:
        decremental_entropy_term = _compute_entropy_term(
            log_prob=decremental_old_log_probs,
            entropy=decremental_entropies,
            entropy_loss_type=loss_config["entropy_loss_type"],
        )
        decremental_entropy_loss = average_loss(
            decremental_entropy_coef * decremental_entropy_term,
            response_mask,
            mode=actor_config.loss_avg_mode,
        )
        policy_loss = policy_loss + decremental_entropy_loss
        if decremental_entropies is not None:
            metrics["algo/entropy/decremental"] = to_float(VF.masked_mean(decremental_entropies, response_mask))

    if use_dvrp and incremental_old_log_probs is not None:
        # The DVRP robustness term is a KL by definition; it does not follow
        # visual_sensitivity_metric.
        per_token_robustness_kl = compute_sampled_low_var_kl(
            incremental_old_log_probs,
            log_prob,
            reference_mode="current",
        )
        metrics["algo/dvrp/robustness_kl"] = to_float(VF.masked_mean(per_token_robustness_kl, response_mask))
        if loss_config.get("visual_robustness_loss_coef", 0.0) != 0.0:
            robustness_loss = average_loss(
                loss_config["visual_robustness_loss_coef"] * per_token_robustness_kl,
                response_mask,
                mode=actor_config.loss_avg_mode,
            )
            policy_loss = policy_loss + robustness_loss

    incremental_entropy_coef = loss_config.get("incremental_entropy_coef", 0.0)
    if use_dvrp and incremental_entropy_coef != 0.0:
        incremental_entropy_term = _compute_entropy_term(
            log_prob=incremental_old_log_probs,
            entropy=incremental_entropies,
            entropy_loss_type=loss_config["entropy_loss_type"],
        )
        incremental_entropy_loss = average_loss(
            incremental_entropy_coef * incremental_entropy_term,
            response_mask,
            mode=actor_config.loss_avg_mode,
        )
        policy_loss = policy_loss + incremental_entropy_loss
        if incremental_entropies is not None:
            metrics["algo/entropy/incremental"] = to_float(VF.masked_mean(incremental_entropies, response_mask))

    if use_dvrp:
        for stats_key in ("noise_t", "noise_beta"):
            if stats_key in loss_config:
                metrics[f"algo/dvrp/{stats_key}"] = float(loss_config[stats_key])

    metrics["algo/loss/policy_loss"] = to_float(policy_loss)
    return policy_loss, metrics


def _get_advantage_scaling_method(loss_config: dict[str, Any] | None) -> str | None:
    if not loss_config:
        return None
    return loss_config.get("advantage_scaling_method") or loss_config.get("response_advantage_scaling_method")


@torch.no_grad()
def _compute_sensitivity_comparison_diagnostics(
    active_name: str,
    active_scores: torch.Tensor,
    extra_signals: dict[str, torch.Tensor],
    response_mask: torch.Tensor,
    quantile: float,
    granularity: str,
) -> dict[str, float]:
    """Compare the active sensitivity signal against each diagnostic signal.

    Emits, per pair, the top-quantile token-set overlap |A∩B|/k (only when a
    perception quantile is configured; the random-selection baseline equals the
    quantile itself) and the Spearman rank correlation of response-mean scores —
    the invariant that decides whether quantile masks and monotone response
    scalings (CGPO/VPPO) would change under a signal swap.
    """
    diagnostics: dict[str, float] = {}
    active_scores = active_scores.detach()
    active_response_means, valid_scores_mask = compute_response_mean_scores(active_scores, response_mask)

    active_topq_mask = None
    if quantile < 1.0:
        active_topq_mask, _, _ = _compute_top_quantile_mask(
            values=active_scores,
            response_mask=response_mask,
            quantile=quantile,
            granularity=granularity,
        )

    for extra_name, extra_scores in extra_signals.items():
        pair_name = f"{active_name}_vs_{extra_name}"
        extra_scores = extra_scores.detach()
        if active_topq_mask is not None:
            extra_topq_mask, _, _ = _compute_top_quantile_mask(
                values=extra_scores,
                response_mask=response_mask,
                quantile=quantile,
                granularity=granularity,
            )
            selected_tokens = active_topq_mask.sum().float().clamp(min=1.0)
            diagnostics[f"algo/sensitivity/diag_topq_overlap_{pair_name}"] = to_float(
                (active_topq_mask & extra_topq_mask).sum().float() / selected_tokens
            )
        extra_response_means, _ = compute_response_mean_scores(extra_scores, response_mask)
        diagnostics[f"algo/sensitivity/diag_response_rank_corr_{pair_name}"] = to_float(
            compute_spearman_rank_correlation(active_response_means, extra_response_means, valid_scores_mask)
        )
    return diagnostics


def _compute_top_quantile_mask(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    quantile: float,
    granularity: str,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = response_mask.to(torch.bool)
    if granularity == "batch":
        valid_values = values[valid_mask]
        topk = int(torch.ceil(valid_values.new_tensor(valid_values.numel() * quantile)).item())
        if topk > 0:
            valid_positions = valid_mask.nonzero(as_tuple=False)
            if generator is None:
                sorted_vals, sorted_order = torch.sort(valid_values.float(), descending=True)
            else:  # a stable sort of a random permutation orders tied values uniformly at random
                permutation = torch.randperm(valid_values.numel(), generator=generator).to(valid_values.device)
                sorted_vals, order = torch.sort(valid_values[permutation].float(), descending=True, stable=True)
                sorted_order = permutation[order]
            selected_positions = valid_positions[sorted_order[:topk]]
            mask = torch.zeros_like(valid_mask)
            mask[selected_positions[:, 0], selected_positions[:, 1]] = True
            threshold = sorted_vals[topk - 1]
        else:
            threshold = torch.tensor(float("nan"), device=values.device)
            mask = torch.zeros_like(valid_mask)
    elif granularity == "response":
        values_for_sort = values.masked_fill(~valid_mask, float("-inf"))
        num_valid_tokens = valid_mask.sum(dim=1)
        topk = torch.ceil(num_valid_tokens.float() * quantile).to(torch.long)
        sorted_vals, sorted_indices = torch.sort(values_for_sort, dim=1, descending=True)
        range_tensor = torch.arange(values_for_sort.size(1), device=values.device).expand_as(values_for_sort)
        rank_mask = range_tensor < topk.unsqueeze(1)
        mask = torch.zeros_like(valid_mask)
        mask.scatter_(1, sorted_indices, rank_mask)
        mask = mask & valid_mask
        valid_thresholds = []
        for row_idx, keep_count in enumerate(topk.tolist()):
            if keep_count > 0:
                valid_thresholds.append(sorted_vals[row_idx, keep_count - 1])
        threshold = (
            torch.stack(valid_thresholds).mean()
            if valid_thresholds
            else torch.tensor(float("nan"), device=values.device)
        )
    else:
        raise ValueError(f"Unsupported threshold granularity: {granularity}")

    fraction = mask.sum().float() / valid_mask.sum().clamp(min=1).float()
    return mask, threshold, fraction


def _compute_advantage_scaling(
    scaling_method: str,
    per_token_sensitivity: torch.Tensor,
    response_mask: torch.Tensor,
    loss_config: dict[str, Any],
    entropy: torch.Tensor | None = None,
    shaping_context: dict[str, float] | None = None,
    perception_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if scaling_method == "vppo":
        response_scaling, metrics = _compute_vppo_response_scaling(
            per_token_sensitivity=per_token_sensitivity,
            response_mask=response_mask,
            scaling_min=loss_config.get("vppo_response_scaling_min", 0.9),
            shaping_context=shaping_context,
        )
        return response_scaling.unsqueeze(-1).expand_as(per_token_sensitivity), metrics
    if scaling_method == "cgpo":
        response_scaling, metrics = _compute_cgpo_response_scaling(
            per_token_sensitivity=per_token_sensitivity,
            response_mask=response_mask,
            scaling_coef=loss_config.get("cgpo_response_scaling_coef", 0.1),
            shaping_context=shaping_context,
        )
        if perception_mask is None:
            return torch.ones_like(per_token_sensitivity), metrics
        token_scaling = torch.where(
            perception_mask,
            response_scaling.unsqueeze(-1).expand_as(per_token_sensitivity),
            torch.ones_like(per_token_sensitivity),
        )
        return token_scaling, metrics
    if scaling_method == "pgpo":
        return _compute_pgpo_token_scaling(
            per_token_sensitivity=per_token_sensitivity,
            response_mask=response_mask,
            threshold=loss_config.get("pgpo_token_scaling_threshold", 0.4),
            boost=loss_config.get("pgpo_token_scaling_boost", 2.0),
            threshold_mode=loss_config.get("pgpo_threshold_mode", "absolute"),
            low_weight_floor=loss_config.get("pgpo_low_weight_floor", 0.0),
            mass_normalization=loss_config.get("pgpo_mass_normalization", True),
        )
    if scaling_method == "pepo":
        if entropy is None:
            raise ValueError("PEPO advantage scaling requires current-policy entropy.")
        return _compute_pepo_token_scaling(
            visual_scores=per_token_sensitivity,
            entropy=entropy,
            response_mask=response_mask,
            gate_alpha=loss_config.get("pepo_gate_alpha", 0.05),
            gate_temperature=loss_config.get("pepo_gate_temperature", 1.8),
        )
    raise ValueError(f"Unknown advantage_scaling_method: {scaling_method}")


def _apply_advantage_scaling(
    advantages: torch.Tensor,
    scaling_factors: torch.Tensor | None,
) -> torch.Tensor:
    if scaling_factors is None:
        return advantages
    return advantages * scaling_factors


def _apply_advantage_scaling_schedule(
    scaling_factors: torch.Tensor,
    response_mask: torch.Tensor,
    schedule: str,
    progress: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if schedule == "none":
        return scaling_factors, {}
    if schedule != "linear":
        raise ValueError(f"Unknown advantage_scaling_schedule: {schedule}")
    progress = max(0.0, min(float(progress), 1.0))
    valid_mask = response_mask.to(torch.bool)
    scheduled = 1.0 + progress * (scaling_factors.float() - 1.0)
    scheduled = torch.where(valid_mask, scheduled, torch.zeros_like(scheduled))
    return scheduled.to(dtype=scaling_factors.dtype), {"algo/advantage_scaling/schedule_progress": progress}


def _compute_token_scaling_factor_metrics(
    scaling_factors: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    valid_mask = response_mask.to(torch.bool)
    return {
        "algo/advantage_scaling/factor_mean": to_float(VF.masked_mean(scaling_factors, response_mask)),
        "algo/advantage_scaling/factor_min": to_float(masked_vector_min(scaling_factors, valid_mask)),
        "algo/advantage_scaling/factor_max": to_float(masked_vector_max(scaling_factors, valid_mask)),
    }


def _compute_pgpo_token_scaling(
    per_token_sensitivity: torch.Tensor,
    response_mask: torch.Tensor,
    threshold: float,
    boost: float,
    threshold_mode: str = "absolute",
    low_weight_floor: float = 0.0,
    mass_normalization: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PGPO token weights. The defaults follow the paper (Eq. 6-7); `threshold_mode="quantile"`,
    `low_weight_floor=0.1` and `mass_normalization=False` follow the authors' development code."""
    valid_mask = response_mask.to(torch.bool)
    scores = torch.log1p(torch.clamp(per_token_sensitivity.float(), min=0.0))
    normalized_scores = minmax_normalize_by_response(scores, valid_mask, eps=eps)

    metrics = {}
    if threshold_mode == "quantile":
        # the ascending normalized score at index floor(length * threshold), within [0, length - 1]
        lengths = valid_mask.sum(dim=-1, keepdim=True)
        index = torch.minimum((lengths.float() * threshold).long(), (lengths - 1).clamp(min=0))
        sorted_scores = normalized_scores.masked_fill(~valid_mask, float("inf")).sort(dim=-1).values
        threshold_tensor = torch.where(lengths > 0, sorted_scores.gather(-1, index), 0.0)
        metrics["algo/pgpo/threshold_mean"] = to_float(threshold_tensor[lengths > 0].mean())
    elif threshold_mode == "absolute":
        threshold_tensor = scores.new_tensor(threshold)
    else:
        raise ValueError(f"Unknown pgpo_threshold_mode: {threshold_mode}")
    below = normalized_scores / threshold_tensor.clamp(min=eps)
    if low_weight_floor > 0.0:
        below = below.clamp(min=low_weight_floor)
    above = 1.0 + boost * (normalized_scores - threshold_tensor) / (1.0 - threshold_tensor).clamp(min=eps)
    raw_scaling = torch.where(normalized_scores < threshold_tensor, below, above)
    raw_scaling = raw_scaling.masked_fill(~valid_mask, 0.0)

    if mass_normalization:
        token_counts = valid_mask.sum(dim=-1, keepdim=True).to(raw_scaling.dtype)
        raw_sums = raw_scaling.sum(dim=-1, keepdim=True)
        mass_normalized = raw_scaling * token_counts / raw_sums.clamp(min=eps)
        fallback = valid_mask.to(raw_scaling.dtype)
        scaling = torch.where(raw_sums > eps, mass_normalized, fallback)
        scaling = scaling.masked_fill(~valid_mask, 0.0)
    else:
        scaling = raw_scaling

    metrics = {
        **metrics,
        "algo/pgpo/token_importance_mean": to_float(VF.masked_mean(normalized_scores, response_mask)),
        "algo/advantage_scaling/factor_mean": to_float(VF.masked_mean(scaling, response_mask)),
        "algo/advantage_scaling/factor_min": to_float(masked_vector_min(scaling, valid_mask)),
        "algo/advantage_scaling/factor_max": to_float(masked_vector_max(scaling, valid_mask)),
    }
    return scaling.to(dtype=per_token_sensitivity.dtype), metrics


def _compute_pepo_token_scaling(
    visual_scores: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    gate_alpha: float,
    gate_temperature: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    if visual_scores.shape != response_mask.shape or entropy.shape != response_mask.shape:
        raise ValueError(
            "PEPO token scaling expects visual_scores, entropy, and response_mask to have the same shape."
        )
    if gate_temperature <= 0.0:
        raise ValueError(f"pepo_gate_temperature must be positive, but got {gate_temperature}.")

    valid_mask = response_mask.to(torch.bool)
    token_counts = valid_mask.sum(dim=-1, keepdim=True).to(dtype=torch.float32)
    if (token_counts == 0).any():
        raise ValueError("PEPO token scaling requires every response to contain at least one valid token.")

    visual_scores = visual_scores.detach().float()
    entropy = entropy.detach().float()
    visual_norm = minmax_normalize_by_response(visual_scores, valid_mask, eps=eps)
    entropy_norm = minmax_normalize_by_response(entropy, valid_mask, eps=eps)
    mixed = visual_norm + entropy_norm
    mixed_mean = (mixed * valid_mask.float()).sum(dim=-1, keepdim=True) / token_counts.clamp(min=1.0)
    gate = 1.0 + float(gate_alpha) * torch.tanh(mixed - mixed_mean)

    logits = visual_scores * gate / float(gate_temperature)
    logits = logits.masked_fill(~valid_mask, float("-inf"))
    scaling = torch.softmax(logits, dim=-1) * token_counts
    scaling = scaling.masked_fill(~valid_mask, 0.0)

    metrics = _compute_token_scaling_factor_metrics(scaling, response_mask)
    return scaling.to(dtype=visual_scores.dtype), metrics


def _compute_vppo_response_scaling(
    per_token_sensitivity: torch.Tensor,
    response_mask: torch.Tensor,
    scaling_min: float,
    shaping_context: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if shaping_context is None:
        raise ValueError(
            "VPPO response scaling requires a precomputed advantage_shaping_context. "
            "Precompute it with build_sensitivity_advantage_shaping_context on the driver side."
        )

    sensitivity_scores, valid_scores_mask = compute_response_mean_scores(
        per_token_scores=per_token_sensitivity,
        response_mask=response_mask,
    )
    scaling_factors = torch.ones_like(sensitivity_scores)

    global_min_score = torch.tensor(
        shaping_context["global_min_score"] if shaping_context is not None else float("nan"),
        device=sensitivity_scores.device,
        dtype=sensitivity_scores.dtype,
    )
    global_max_score = torch.tensor(
        shaping_context["global_max_score"] if shaping_context is not None else float("nan"),
        device=sensitivity_scores.device,
        dtype=sensitivity_scores.dtype,
    )

    dynamic_tas_beta_max = torch.tensor(float("nan"), device=sensitivity_scores.device, dtype=sensitivity_scores.dtype)
    if valid_scores_mask.any() and not torch.isnan(global_min_score) and not torch.isnan(global_max_score):
        valid_scores = sensitivity_scores[valid_scores_mask]
        if (global_max_score - global_min_score) > 1e-6:
            normalized_scores = (valid_scores - global_min_score) / (global_max_score - global_min_score)
            normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)
            mu_norm = normalized_scores.mean()
            dynamic_tas_beta_max = scaling_min + (1.0 - scaling_min) / (mu_norm + 1e-8)
            mapped_scores = scaling_min + normalized_scores * (dynamic_tas_beta_max - scaling_min)
            scaling_factors[valid_scores_mask] = mapped_scores

    metrics = {
        "algo/vppo/dynamic_tas_beta_max": to_float(dynamic_tas_beta_max),
        "algo/advantage_scaling/response_signal_mean": to_float(
            masked_vector_mean(sensitivity_scores, valid_scores_mask)
        ),
        "algo/vppo/batch_signal_min": to_float(global_min_score),
        "algo/vppo/batch_signal_max": to_float(global_max_score),
        "algo/advantage_scaling/factor_mean": to_float(masked_vector_mean(scaling_factors, valid_scores_mask)),
        "algo/advantage_scaling/factor_min": to_float(masked_vector_min(scaling_factors, valid_scores_mask)),
        "algo/advantage_scaling/factor_max": to_float(masked_vector_max(scaling_factors, valid_scores_mask)),
    }
    return scaling_factors, metrics


def _compute_vppo_advantage_shaping_context_from_scores(
    sensitivity_scores: torch.Tensor,
    valid_scores_mask: torch.Tensor,
) -> dict[str, float] | None:
    valid_scores = sensitivity_scores[valid_scores_mask]
    if valid_scores.numel() <= 1:
        return None
    return {
        "global_min_score": torch.quantile(valid_scores, 0.0).item(),
        "global_max_score": torch.quantile(valid_scores, 1.0).item(),
    }


def _compute_cgpo_advantage_shaping_context_from_scores(
    sensitivity_scores: torch.Tensor,
    valid_scores_mask: torch.Tensor,
) -> dict[str, float] | None:
    valid_scores = sensitivity_scores[valid_scores_mask]
    if valid_scores.numel() <= 1:
        return None
    global_median_score = valid_scores.median()
    global_mad_score = (valid_scores - global_median_score).abs().median()
    return {
        "global_median_score": global_median_score.item(),
        "global_mad_score": global_mad_score.item(),
    }


def _compute_cgpo_response_scaling(
    per_token_sensitivity: torch.Tensor,
    response_mask: torch.Tensor,
    scaling_coef: float,
    shaping_context: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if shaping_context is None:
        raise ValueError(
            "CGPO response scaling requires a precomputed advantage_shaping_context. "
            "Precompute it with build_sensitivity_advantage_shaping_context on the driver side."
        )

    sensitivity_scores, valid_scores_mask = compute_response_mean_scores(
        per_token_scores=per_token_sensitivity,
        response_mask=response_mask,
    )
    scaling_factors = torch.ones_like(sensitivity_scores)

    global_median_score = torch.tensor(
        shaping_context["global_median_score"] if shaping_context is not None else float("nan"),
        device=sensitivity_scores.device,
        dtype=sensitivity_scores.dtype,
    )
    global_mad_score = torch.tensor(
        shaping_context["global_mad_score"] if shaping_context is not None else float("nan"),
        device=sensitivity_scores.device,
        dtype=sensitivity_scores.dtype,
    )

    if valid_scores_mask.any() and not torch.isnan(global_median_score) and not torch.isnan(global_mad_score):
        valid_scores = sensitivity_scores[valid_scores_mask]
        normalized_scores = (valid_scores - global_median_score) / (global_mad_score + 1e-8)
        mapped_scores = 1.0 + scaling_coef * torch.tanh(normalized_scores)
        scaling_factors[valid_scores_mask] = mapped_scores

    metrics = {
        "algo/advantage_scaling/response_signal_mean": to_float(
            masked_vector_mean(sensitivity_scores, valid_scores_mask)
        ),
        "algo/cgpo/batch_signal_median": to_float(global_median_score),
        "algo/cgpo/batch_signal_mad": to_float(global_mad_score),
        "algo/advantage_scaling/factor_mean": to_float(masked_vector_mean(scaling_factors, valid_scores_mask)),
        "algo/advantage_scaling/factor_min": to_float(masked_vector_min(scaling_factors, valid_scores_mask)),
        "algo/advantage_scaling/factor_max": to_float(masked_vector_max(scaling_factors, valid_scores_mask)),
    }
    return scaling_factors, metrics


def _compute_entropy_term(
    log_prob: torch.Tensor | None,
    entropy: torch.Tensor | None,
    entropy_loss_type: str,
) -> torch.Tensor:
    if entropy_loss_type == "sampled":
        if log_prob is None:
            raise ValueError("Sampled entropy loss requires log_prob to be available.")
        return -log_prob
    if entropy_loss_type == "full":
        if entropy is None:
            raise ValueError("Full entropy loss requires entropy to be available.")
        return entropy
    raise ValueError(f"Unknown entropy_loss_type: {entropy_loss_type}")


def _compute_tor_weight(
    entropy_mask: torch.Tensor | None,
    perception_mask: torch.Tensor | None,
    tor_rsn_weight: float,
    tor_prcp_weight: float,
) -> torch.Tensor:
    if entropy_mask is None and perception_mask is None:
        raise ValueError("tor_use_token_weighting requires top_entropy_quantile<1 or top_perception_quantile<1.")
    reference = entropy_mask if entropy_mask is not None else perception_mask
    combined_weight = torch.zeros_like(reference, dtype=torch.float32)
    if entropy_mask is not None:
        combined_weight = combined_weight + entropy_mask.float() * tor_rsn_weight
    if perception_mask is not None:
        if entropy_mask is not None:
            perception_mask = perception_mask & (~entropy_mask)
        combined_weight = combined_weight + perception_mask.float() * tor_prcp_weight
    return combined_weight


def _combine_masks(entropy_mask: torch.Tensor | None, perception_mask: torch.Tensor | None) -> torch.Tensor:
    if entropy_mask is not None and perception_mask is not None:
        return entropy_mask | perception_mask
    if entropy_mask is not None:
        return entropy_mask
    if perception_mask is not None:
        return perception_mask
    raise ValueError("At least one mask must be provided.")
