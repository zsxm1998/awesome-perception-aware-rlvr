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
"""Distribution-level on-policy distillation: per-token divergences between the student's and a target's
next-token distributions, computed on response rows in fp32 chunks.

Portions of the chunked divergence are adapted from TRL (Apache-2.0),
trl/trainer/distillation_trainer.py@14c8d70 L105-291 (`_chunk`, `_chunked_divergence_loss`). Modified: the inputs
are logit rows (the LM head runs inside the model forward with `logits_to_keep`), the support is the tokenizer's
ids instead of the whole LM head, the target can be built from several teacher (and student) views, the sampled
token's full-width log-prob and the entropy are returned, and the loss is aggregated by the caller.

Each response position is one row. A target distribution q and the student distribution p are compared on the
support S = ids [0, vocab_size) (the LM head of Qwen models has padding rows beyond the tokenizer, whose logits are
non-zero; they are sliced away, never masked with -inf):
  reverse_kl   KL(p || q)
  forward_kl   KL(q || p)
  jsd          beta * KL(q || M) + (1 - beta) * KL(p || M), M = (1 - beta) p + beta q (GKD's generalized JSD)
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


DIVERGENCES = ("reverse_kl", "forward_kl", "jsd")
SUPPORTS = ("full", "student_top_k")
TARGETS = ("teacher", "contrast_sharpened", "visual_gain")
TEMPERATURE_SCOPES = ("all", "loss_scale_only")


@dataclass(frozen=True)
class DistillationSpec:
    """Hyperparameters of the per-row divergence (hashable, so it passes through activation checkpointing)."""

    vocab_size: int
    """size of the support: ids [0, vocab_size), the tokenizer's ids"""
    divergence: str = "reverse_kl"
    jsd_beta: float = 0.5
    support: str = "full"
    top_k: int = 100
    temperature: float = 1.0
    """distillation temperature T; the loss is multiplied by T^2 when T != 1"""
    temperature_scope: str = "all"
    """`all`: the distributions are softmax(z / T); `loss_scale_only`: they stay at T = 1 and only T^2 is applied"""
    target: str = "teacher"
    # contrast_sharpened (VCSD): score = anchor * log q + alpha * (log q - log q_ctrl) on the support
    # {v: q(v) >= beta * max q}, with the score of the keep ids (the end-of-sequence tokens) at anchor * log q
    contrast_alpha: float = 1.0
    contrast_support_beta: float = 0.1
    contrast_anchor_coef: float = 1.0
    contrast_keep_ids: tuple[int, ...] = ()
    # visual_gain (VGS): loss = loss_scale * (KL(p || q) + steering_coef * KL(p || q*)
    #                                         + text_prior_coef * gate * KL(p_text || q_text))
    steering_coef: float = 2.0
    text_prior_coef: float = 0.01
    loss_scale: float = 1.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DistillationSpec":
        names = cls.__dataclass_fields__.keys()
        values = {name: config[name] for name in names if name in config}
        if "contrast_keep_ids" in values:
            values["contrast_keep_ids"] = tuple(int(index) for index in values["contrast_keep_ids"])
        return cls(**values)


def _divergence(student_log_probs: torch.Tensor, target_log_probs: torch.Tensor, divergence: str, beta: float):
    """Per-row divergence between two log-distributions over the same classes, (rows, classes) -> (rows,)."""
    # F.kl_div(input, target, log_target=True) = exp(target) * (target - input), i.e. KL(target || input)
    if divergence == "reverse_kl":
        per_class = F.kl_div(target_log_probs, student_log_probs, reduction="none", log_target=True)
    elif divergence == "forward_kl":
        per_class = F.kl_div(student_log_probs, target_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), target_log_probs + torch.log(beta_t)]), dim=0
        )
        per_class = beta_t * F.kl_div(mixture, target_log_probs, reduction="none", log_target=True) + (
            1 - beta_t
        ) * F.kl_div(mixture, student_log_probs, reduction="none", log_target=True)
    return per_class.sum(-1)


def _with_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
    """Append log(1 - sum p) of the classes left out (Vision-OPD's tail bucket), clamped away from log(0)."""
    covered = torch.logsumexp(log_probs, dim=-1, keepdim=True).clamp(max=-1e-7)
    return torch.cat([log_probs, torch.log(-torch.expm1(covered))], dim=-1)


def _support_log_probs(logits: torch.Tensor, spec: DistillationSpec, temperature: float) -> torch.Tensor:
    """fp32 log-probs over the support (ids [0, vocab_size)) of logit rows at a temperature."""
    support = logits.float()[:, : spec.vocab_size]
    if temperature != 1.0:
        support = support / temperature
    return F.log_softmax(support, dim=-1)


def _kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Per-row KL(p || q) of two log-distributions; terms with p = 0 count 0 (finite log_q required)."""
    return F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(-1)


def _contrast_sharpened_target(
    teacher_log_probs: torch.Tensor, contrast_log_probs: torch.Tensor, spec: DistillationSpec
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VCSD's target (Eq. 7-9, 17): q*(v) proportional to q(v)^anchor * (q(v) / q_ctrl(v))^alpha on the support
    S = {v: q(v) >= beta * max q}; the end-of-sequence tokens keep anchor * log q (no contrast). Outside S the
    log-probability is clamped to -1e4 (exp gives exactly 0, and the KL stays finite)."""
    score = spec.contrast_anchor_coef * teacher_log_probs + spec.contrast_alpha * (
        teacher_log_probs - contrast_log_probs
    )
    if spec.contrast_keep_ids:
        keep_ids = torch.tensor(spec.contrast_keep_ids, device=score.device)
        score[:, keep_ids] = spec.contrast_anchor_coef * teacher_log_probs[:, keep_ids]
    support = torch.ones_like(score, dtype=torch.bool)
    if spec.contrast_support_beta > 0.0:
        threshold = teacher_log_probs.max(dim=-1, keepdim=True).values + math.log(spec.contrast_support_beta)
        support = teacher_log_probs >= threshold
        score = score.masked_fill(~support, float("-inf"))
    target = F.log_softmax(score, dim=-1).clamp_min(-1e4)
    stats = {
        "target_kl_to_teacher": _kl(target, teacher_log_probs),
        "target_argmax_change": (target.argmax(-1) != teacher_log_probs.argmax(-1)).float(),
        "target_support_size": support.sum(-1).float(),
    }
    return target, stats


def stat_names(spec: DistillationSpec) -> tuple[str, ...]:
    """Names of the per-row statistics `_distill_rows` returns for a spec, in order."""
    names = ("entropy", "student_oov_mass", "teacher_oov_mass")
    if spec.target == "contrast_sharpened":
        names += ("target_kl_to_teacher", "target_argmax_change", "target_support_size")
    elif spec.target == "visual_gain":
        names += ("standard_kl", "visual_kl", "text_prior_kl", "text_prior_gate")
    return names


def _distill_rows(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_contrast_logits: Optional[torch.Tensor],
    student_contrast_logits: Optional[torch.Tensor],
    row_gate: Optional[torch.Tensor],
    spec: DistillationSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One chunk of rows.

    Args:
        student_logits: (rows, V_lm_head), with gradient
        labels: (rows,), the sampled tokens
        teacher_logits: (rows, V_lm_head), without gradient
        teacher_contrast_logits / student_contrast_logits: the same rows under a contrast view, for targets that
            need them (None otherwise); the student's may carry gradient (visual_gain's text-prior term)
        row_gate: (rows,) 0/1 gate of visual_gain's text-prior term (None otherwise)

    Returns:
        loss: (rows,) with gradient, already multiplied by T^2 when T != 1
        sampled_log_probs: (rows,) with gradient, log p(label) over the whole LM head at T = 1 (the rollout
            temperature), the same quantity as the actor's old log-probs
        stats: (rows, len(stat_names(spec))) without gradient: student entropy (whole LM head, T = 1), student and
            teacher probability mass outside the support, then the target's own statistics
    """
    student = student_logits.float()
    student_lse = torch.logsumexp(student, dim=-1)
    sampled_log_probs = student.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - student_lse

    with torch.no_grad():
        student_detached = student.detach()
        full_log_probs = student_detached - student_lse.detach().unsqueeze(-1)
        entropy = -(full_log_probs.exp() * full_log_probs).sum(-1)
        # 1 - exp(lse_support - lse_all), clamped at 0 against rounding
        student_oov = (
            -torch.expm1(torch.logsumexp(student_detached[:, : spec.vocab_size], dim=-1) - student_lse.detach())
        ).clamp(min=0.0)
        teacher = teacher_logits.float()
        teacher_oov = (
            -torch.expm1(torch.logsumexp(teacher[:, : spec.vocab_size], dim=-1) - torch.logsumexp(teacher, dim=-1))
        ).clamp(min=0.0)
        del teacher, full_log_probs

    temperature = spec.temperature if spec.temperature_scope == "all" else 1.0
    with torch.no_grad():
        teacher_log_probs = _support_log_probs(teacher_logits, spec, temperature)
        target_log_probs, target_stats = teacher_log_probs, {}
        if spec.target == "contrast_sharpened":
            contrast_log_probs = _support_log_probs(teacher_contrast_logits, spec, temperature)
            target_log_probs, target_stats = _contrast_sharpened_target(teacher_log_probs, contrast_log_probs, spec)

    if spec.target == "visual_gain":
        # VGS (Eq. 10-17): q*(v) proportional to p_text(v) * q(v) / q_text(v) with the student's text-only
        # distribution detached; the text-prior term pulls p_text toward q_text on the gated rows
        student_log_probs = _support_log_probs(student_logits, spec, temperature)
        student_text_log_probs = _support_log_probs(student_contrast_logits, spec, temperature)
        with torch.no_grad():
            teacher_text_log_probs = _support_log_probs(teacher_contrast_logits, spec, temperature)
            steered_log_probs = F.log_softmax(
                student_text_log_probs.detach() + teacher_log_probs - teacher_text_log_probs, dim=-1
            )
        standard = _kl(student_log_probs, teacher_log_probs)
        visual = _kl(student_log_probs, steered_log_probs)
        text_prior = _kl(student_text_log_probs, teacher_text_log_probs)
        gate = row_gate.to(text_prior.dtype)
        loss = spec.loss_scale * (standard + spec.steering_coef * visual + spec.text_prior_coef * gate * text_prior)
        target_stats = {
            "standard_kl": standard.detach(),
            "visual_kl": visual.detach(),
            "text_prior_kl": text_prior.detach(),
            "text_prior_gate": gate,
        }
        stats = torch.stack([entropy, student_oov, teacher_oov, *target_stats.values()], dim=-1)
        return loss, sampled_log_probs, stats

    if spec.support == "student_top_k":
        # Vision-OPD: the classes are the student's top-k ids plus one bucket for the rest; both sides are
        # normalized over the whole support first, so the bucket holds the exact remaining mass
        student_support = student[:, : spec.vocab_size]
        if temperature != 1.0:
            student_support = student_support / temperature
        with torch.no_grad():
            top_k_ids = student_support.detach().topk(spec.top_k, dim=-1).indices
            target_log_probs = _with_tail_bucket(target_log_probs.gather(-1, top_k_ids))
        student_log_probs = _with_tail_bucket(
            student_support.gather(-1, top_k_ids) - torch.logsumexp(student_support, dim=-1, keepdim=True)
        )
    else:
        student_log_probs = _support_log_probs(student_logits, spec, temperature)

    loss = _divergence(student_log_probs, target_log_probs, spec.divergence, spec.jsd_beta)
    if spec.temperature != 1.0:
        loss = loss * spec.temperature**2

    stats = torch.stack([entropy, student_oov, teacher_oov, *target_stats.values()], dim=-1)
    return loss, sampled_log_probs, stats


def chunked_distillation(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    spec: DistillationSpec,
    chunk_size: int = 256,
    teacher_contrast_logits: Optional[torch.Tensor] = None,
    student_contrast_logits: Optional[torch.Tensor] = None,
    row_gate: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run `_distill_rows` over chunks of rows under non-reentrant activation checkpointing: the forward keeps
    only the chunk inputs, and the [chunk, V] fp32 intermediates are recomputed in the backward (TRL's scheme).

    Returns the concatenated (loss, sampled_log_probs, stats), see `_distill_rows`.
    """
    for name, other in (
        ("teacher", teacher_logits),
        ("teacher contrast", teacher_contrast_logits),
        ("student contrast", student_contrast_logits),
    ):
        if other is not None and other.shape != student_logits.shape:
            raise RuntimeError(
                f"student logits {tuple(student_logits.shape)} and {name} logits {tuple(other.shape)} "
                "do not cover the same response rows."
            )

    def _slice(tensor: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
        return None if tensor is None else tensor[start:end]

    use_checkpoint = torch.is_grad_enabled() and (
        student_logits.requires_grad or (student_contrast_logits is not None and student_contrast_logits.requires_grad)
    )
    outputs = []
    for start in range(0, student_logits.size(0), chunk_size):
        end = start + chunk_size
        chunk_inputs = (
            student_logits[start:end],
            labels[start:end],
            teacher_logits[start:end],
            _slice(teacher_contrast_logits, start, end),
            _slice(student_contrast_logits, start, end),
            _slice(row_gate, start, end),
            spec,
        )
        if use_checkpoint:
            outputs.append(checkpoint(_distill_rows, *chunk_inputs, use_reentrant=False))
        else:
            outputs.append(_distill_rows(*chunk_inputs))
    loss, sampled_log_probs, stats = (torch.cat(values, dim=0) for values in zip(*outputs))
    return loss, sampled_log_probs, stats


def resolve_end_token_ids(model_path: str, tokenizer: Any) -> tuple[int, ...]:
    """The end-of-sequence ids of a model: its generation config's `eos_token_id` (Qwen3-VL: <|im_end|> 151645 and
    <|endoftext|> 151643; Qwen3.5: 248046 and 248044), or the tokenizer's when there is no generation config."""
    from transformers import GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(model_path).eos_token_id
    except OSError:
        eos = None
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    return tuple(sorted({int(index) for index in eos}))


def black_pixel_values(processor: Any) -> list[float]:
    """The normalized value of a black pixel per channel, (0 - mean) / std, as the image processor maps it."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not getattr(image_processor, "do_normalize", False):
        raise ValueError("the black-image contrast view needs an image processor that normalizes the pixels.")
    mean, std = image_processor.image_mean, image_processor.image_std
    return [(0.0 - float(m)) / float(s) for m, s in zip(mean, std)]


def build_distillation_config(
    algorithm: Any,
    vocab_size: int,
    end_token_ids: tuple[int, ...] = (),
    black_pixel_values: Optional[list[float]] = None,
) -> Optional[dict[str, Any]]:
    """The settings the actor needs for the distillation loss (`meta_info["distillation_config"]`), or None when
    `algorithm.distill_loss_coef` is 0. `vocab_size` is the length of the student's tokenizer; `end_token_ids`
    (the end-of-sequence ids) resolve `vcsd_keep_token_ids=auto`; `black_pixel_values` are the processor's
    normalized values of a black pixel per channel, for `distill_contrast_view=black`."""
    if algorithm.distill_loss_coef == 0.0:
        return None
    views = []
    if algorithm.teacher_view == "data_image":
        views.append("data_image")
    if algorithm.distill_contrast_view == "no_image":
        views.append("no_image")
    keep_ids = algorithm.vcsd_keep_token_ids
    if keep_ids == "auto":
        keep_ids = end_token_ids
    keep_ids = tuple(int(index) for index in keep_ids)
    outside = [index for index in keep_ids if not 0 <= index < int(vocab_size)]
    if outside:
        raise ValueError(
            f"vcsd_keep_token_ids {outside} are not ids of the tokenizer (0 <= id < {vocab_size}); ids of the LM "
            "head's padding rows would have no effect."
        )
    if algorithm.distill_contrast_view == "black" and black_pixel_values is None:
        raise ValueError("distill_contrast_view=black needs the image processor's normalization.")
    config = {
        "vocab_size": int(vocab_size),
        "divergence": algorithm.distill_divergence,
        "jsd_beta": float(algorithm.distill_jsd_beta),
        "support": algorithm.distill_support,
        "top_k": int(algorithm.distill_top_k),
        "temperature": float(algorithm.distill_temperature),
        "temperature_scope": algorithm.distill_temperature_scope,
        "target": algorithm.distill_target,
        "loss_coef": float(algorithm.distill_loss_coef),
        "policy_loss_coef": float(algorithm.policy_loss_coef),
        "chunk_size": int(algorithm.distill_chunk_size),
        "is_clip": algorithm.distill_is_clip,
        "teacher_view": algorithm.teacher_view,
        "contrast_view": algorithm.distill_contrast_view,
        # rebuilt inputs the driver attaches to the batch as `distill_view_<name>_*` (see `view_keys`)
        "views": views,
        "black_pixel_values": black_pixel_values,
        "contrast_alpha": float(algorithm.vcsd_alpha),
        "contrast_support_beta": float(algorithm.vcsd_support_beta),
        "contrast_anchor_coef": float(algorithm.vcsd_anchor_coef),
        "contrast_keep_ids": [int(index) for index in keep_ids],
        "steering_coef": float(algorithm.vgs_steering_coef),
        "text_prior_coef": float(algorithm.vgs_text_prior_coef),
        "loss_scale": float(algorithm.vgs_loss_scale),
        "text_prior_quantile": float(algorithm.vgs_vds_quantile),
        "text_prior_scope": algorithm.vgs_vds_scope,
    }
    validate_distillation_spec(DistillationSpec.from_config(config))
    return config


def compute_grouped_token_weights(
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    softmax_temperature: float,
    high_fraction: float,
    high_weight: float,
    generator: torch.Generator,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-token weights c_t of VA-OPD (Eq. 3-6), from per-token scores a_t (the teacher's visual advantage).

    Within the K responses of a prompt (same `index`), the mean score of each response is z-normalized with the
    population std (ddof=0, which reproduces the paper's Fig. 3 example) and weighted by softmax(z / tau). Within a
    response of T tokens, the top max(1, ceil(p_v T)) tokens by score (ties broken by independent random numbers
    from `generator`) share `high_weight` of its weight and the rest the remainder; a response of one token gives
    it all. The weights of a prompt sum to 1, so sum_t c_t KL_t is the paper's per-prompt loss.

    Returns (weights (bs, R), metrics).
    """
    mask = response_mask.bool()
    scores = scores.float().masked_fill(~mask, 0.0)
    lengths = mask.sum(-1)
    means = scores.sum(-1) / lengths.clamp(min=1)

    response_weights = torch.zeros(scores.size(0), dtype=torch.float64)
    groups: dict[Any, list[int]] = {}
    for row, key in enumerate(index):
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        group_means = means[rows].double()
        z = (group_means - group_means.mean()) / (group_means.std(correction=0) + eps)
        response_weights[rows] = torch.softmax(z / softmax_temperature, dim=0)

    weights = torch.zeros_like(scores, dtype=torch.float64)
    high_score_mass, total_score_mass, high_zero, high_count = 0.0, 0.0, 0, 0
    for row in range(scores.size(0)):
        length = int(lengths[row])
        if length == 0:
            continue
        positions = mask[row].nonzero(as_tuple=True)[0]
        row_scores = scores[row, positions]
        tie_break = torch.rand(length, generator=generator)
        # sort by score descending, then by the random key: lexsort with the primary key last
        order = np.lexsort((tie_break.numpy(), -row_scores.double().numpy()))
        n_high = max(1, math.ceil(high_fraction * length))
        high = positions[torch.as_tensor(order[:n_high])]
        low = positions[torch.as_tensor(order[n_high:])]
        if low.numel() == 0:
            weights[row, high] = response_weights[row] / n_high
        else:
            weights[row, high] = response_weights[row] * high_weight / n_high
            weights[row, low] = response_weights[row] * (1.0 - high_weight) / low.numel()
        high_score_mass += float(scores[row, high].sum())
        total_score_mass += float(row_scores.sum())
        high_zero += int((scores[row, high] == 0).sum())
        high_count += n_high

    valid_rows = lengths > 0
    metrics = {}
    if valid_rows.any():
        metrics = {
            "distill/high_group_score_share": high_score_mass / total_score_mass if total_score_mass > 0 else 0.0,
            "distill/high_group_zero_score_frac": high_zero / max(high_count, 1),
            "distill/response_weight_max": float(response_weights[valid_rows].max()),
            "distill/response_weight_min": float(response_weights[valid_rows].min()),
        }
    return weights.float(), metrics


def token_weights_for_token_mean(weights: torch.Tensor, response_mask: torch.Tensor, index: Any) -> torch.Tensor:
    """Rescale per-prompt weights c_t (summing to 1 per prompt) for the actor's token-mean loss over the whole batch:
    mean_t(w_t KL_t) over all N response tokens equals sum_t c_t KL_t / B for B prompts when w_t = c_t N / B."""
    num_prompts = len(set(index))
    return weights * response_mask.sum().float() / num_prompts


def view_keys(name: str) -> dict[str, str]:
    """Batch keys of a rebuilt input view of the distillation (same responses, other prompt)."""
    prefix = f"distill_view_{name}_"
    return {
        field: prefix + field
        for field in (
            "input_ids",
            "attention_mask",
            "position_ids",
            "multi_modal_data",
            "multi_modal_inputs",
            "multi_modal_cache_id",
        )
    }


def validate_distillation_spec(spec: DistillationSpec) -> None:
    if spec.divergence not in DIVERGENCES:
        raise ValueError(f"distill_divergence must be one of {list(DIVERGENCES)}, but got {spec.divergence!r}.")
    if spec.divergence == "jsd" and not 0.0 < spec.jsd_beta < 1.0:
        # beta = 0 / 1 are not the limits of the JSD (it vanishes at both ends); choose forward_kl / reverse_kl
        raise ValueError(f"distill_jsd_beta must be in (0, 1), but got {spec.jsd_beta}.")
    if spec.support not in SUPPORTS:
        raise ValueError(f"distill_support must be one of {list(SUPPORTS)}, but got {spec.support!r}.")
    if spec.support == "student_top_k" and spec.top_k < 1:
        raise ValueError(f"distill_top_k must be positive, but got {spec.top_k}.")
    if spec.temperature <= 0.0:
        raise ValueError(f"distill_temperature must be positive, but got {spec.temperature}.")
    if spec.temperature_scope not in TEMPERATURE_SCOPES:
        raise ValueError(
            f"distill_temperature_scope must be one of {list(TEMPERATURE_SCOPES)}, but got {spec.temperature_scope!r}."
        )
    if spec.target not in TARGETS:
        raise ValueError(f"distill_target must be one of {list(TARGETS)}, but got {spec.target!r}.")


def masked_stat_means(stats: torch.Tensor, response_mask: torch.Tensor, names: tuple[str, ...]) -> dict[str, float]:
    """Means over the response tokens of per-token statistics (bs, R, n) -> {name: float}."""
    mask = response_mask.bool()
    if not mask.any():
        return {}
    values = stats[mask].float().mean(dim=0)
    return {name: value.item() for name, value in zip(names, values)}


class DistillationMetricSums:
    """The distillation metrics of one update, summed over its micro-batches and reduced over the ranks as the loss
    is, so that they do not depend on how the update is split: ``distill/loss`` is the averaged distillation term
    (before its coefficient, with the actor's loss_avg_mode), the other values are means over the response tokens
    of the update."""

    def __init__(self, names: tuple[str, ...], with_is_weights: bool):
        self.columns = [index for index, name in enumerate(names) if name != "entropy"]
        self.names = [f"distill/{names[index]}" for index in self.columns]
        self.with_is_weights = with_is_weights
        self.sums: Optional[torch.Tensor] = None  # loss, statistics..., importance weights, tokens

    def add(
        self,
        loss: torch.Tensor,
        loss_weight: torch.Tensor,
        stats: torch.Tensor,
        is_weights: Optional[torch.Tensor],
        response_mask: torch.Tensor,
    ) -> None:
        """``loss`` averaged over a micro-batch and its weight in the update (rows in seq mode, tokens in token
        mode); per-token ``stats`` (bs, R, n) and importance weights (bs, R)."""
        mask = response_mask.bool()
        values = [
            (loss.detach().float() * loss_weight).reshape(1),
            stats.detach()[mask][:, self.columns].float().sum(dim=0),
            (is_weights.detach()[mask].float().sum() if is_weights is not None else loss.new_zeros(())).reshape(1),
            mask.sum().float().reshape(1),
        ]
        sums = torch.cat([value.to(loss.device) for value in values])
        self.sums = sums if self.sums is None else self.sums + sums

    def reduce(self, loss_denominator: torch.Tensor) -> dict[str, float]:
        """Sum over the ranks (one all-reduce, on every rank) and divide: the loss by the update's total (responses
        in seq mode, tokens in token mode, as the loss is scaled), the rest by its response tokens."""
        sums = self.sums.clone()
        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        tokens = sums[-1].clamp(min=1.0)
        metrics = {"distill/loss": (sums[0] / loss_denominator.float().clamp(min=1.0)).item()}
        metrics.update({name: (value / tokens).item() for name, value in zip(self.names, sums[1:-2])})
        if self.with_is_weights:
            metrics["distill/is_weight_mean"] = (sums[-2] / tokens).item()
        return metrics


def importance_weights(
    log_probs: torch.Tensor, old_log_probs: torch.Tensor, clip: Optional[float]
) -> Optional[torch.Tensor]:
    """Detached truncated importance weight min(pi / pi_old, clip) of a direct loss (Vision-OPD's and TRL SDPO's
    `distillation_is_clip`); it reweights states only and never adds a score-function term. None when off."""
    if clip is None:
        return None
    return torch.exp(torch.clamp(log_probs.detach() - old_log_probs, -20.0, 20.0)).clamp(max=clip)
