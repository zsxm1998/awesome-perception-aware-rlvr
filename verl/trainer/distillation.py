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

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


DIVERGENCES = ("reverse_kl", "forward_kl", "jsd")
SUPPORTS = ("full", "student_top_k")
TARGETS = ("teacher",)
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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DistillationSpec":
        names = cls.__dataclass_fields__.keys()
        return cls(**{name: config[name] for name in names if name in config})


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


def _target_log_probs(
    teacher_logits: torch.Tensor,
    teacher_contrast_logits: Optional[torch.Tensor],
    student_contrast_logits: Optional[torch.Tensor],
    spec: DistillationSpec,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The target distribution over the support (no gradient) and per-row statistics of its construction."""
    teacher_log_probs = _support_log_probs(teacher_logits, spec, temperature)
    return teacher_log_probs, {}


def stat_names(spec: DistillationSpec) -> tuple[str, ...]:
    """Names of the per-row statistics `_distill_rows` returns for a spec, in order."""
    return ("entropy", "student_oov_mass", "teacher_oov_mass")


def _distill_rows(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_contrast_logits: Optional[torch.Tensor],
    student_contrast_logits: Optional[torch.Tensor],
    spec: DistillationSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One chunk of rows.

    Args:
        student_logits: (rows, V_lm_head), with gradient
        labels: (rows,), the sampled tokens
        teacher_logits: (rows, V_lm_head), without gradient
        teacher_contrast_logits / student_contrast_logits: the same rows under a contrast view, for targets that
            need them (None otherwise)

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
        student_oov = 1.0 - torch.exp(
            torch.logsumexp(student_detached[:, : spec.vocab_size], dim=-1) - student_lse.detach()
        )
        teacher = teacher_logits.float()
        teacher_oov = 1.0 - torch.exp(
            torch.logsumexp(teacher[:, : spec.vocab_size], dim=-1) - torch.logsumexp(teacher, dim=-1)
        )
        del teacher, full_log_probs

    temperature = spec.temperature if spec.temperature_scope == "all" else 1.0
    with torch.no_grad():
        target_log_probs, target_stats = _target_log_probs(
            teacher_logits, teacher_contrast_logits, student_contrast_logits, spec, temperature
        )

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
            spec,
        )
        if use_checkpoint:
            outputs.append(checkpoint(_distill_rows, *chunk_inputs, use_reentrant=False))
        else:
            outputs.append(_distill_rows(*chunk_inputs))
    loss, sampled_log_probs, stats = (torch.cat(values, dim=0) for values in zip(*outputs))
    return loss, sampled_log_probs, stats


def build_distillation_config(algorithm: Any, vocab_size: int) -> Optional[dict[str, Any]]:
    """The settings the actor needs for the distillation loss (`meta_info["distillation_config"]`), or None when
    `algorithm.distill_loss_coef` is 0. `vocab_size` is the length of the student's tokenizer."""
    if algorithm.distill_loss_coef == 0.0:
        return None
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
        # rebuilt inputs the driver attaches to the batch as `distill_view_<name>_*` (see `view_keys`)
        "views": ["data_image"] if algorithm.teacher_view == "data_image" else [],
    }
    validate_distillation_spec(DistillationSpec.from_config(config))
    return config


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


def importance_weights(
    log_probs: torch.Tensor, old_log_probs: torch.Tensor, clip: Optional[float]
) -> Optional[torch.Tensor]:
    """Detached truncated importance weight min(pi / pi_old, clip) of a direct loss (Vision-OPD's and TRL SDPO's
    `distillation_is_clip`); it reweights states only and never adds a score-function term. None when off."""
    if clip is None:
        return None
    return torch.exp(torch.clamp(log_probs.detach() - old_log_probs, -20.0, 20.0)).clamp(max=clip)


def reference_divergence(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, spec: DistillationSpec
) -> torch.Tensor:
    """Direct, unchunked definition of the per-row divergence; the tests compare the chunked path to it."""
    temperature = spec.temperature if spec.temperature_scope == "all" else 1.0
    student = student_logits.float()[:, : spec.vocab_size] / temperature
    teacher = teacher_logits.float()[:, : spec.vocab_size] / temperature
    p = torch.softmax(student, dim=-1)
    q = torch.softmax(teacher, dim=-1)
    if spec.support == "student_top_k":
        top_k_ids = student.detach().topk(spec.top_k, dim=-1).indices
        p_top, q_top = p.gather(-1, top_k_ids), q.gather(-1, top_k_ids)
        p = torch.cat([p_top, (1 - p_top.sum(-1, keepdim=True)).clamp(min=1e-30)], dim=-1)
        q = torch.cat([q_top, (1 - q_top.sum(-1, keepdim=True)).clamp(min=1e-30)], dim=-1)
    if spec.divergence == "reverse_kl":
        value = (p * (p.log() - q.log())).sum(-1)
    elif spec.divergence == "forward_kl":
        value = (q * (q.log() - p.log())).sum(-1)
    else:
        beta = spec.jsd_beta
        m = (1 - beta) * p + beta * q
        value = beta * (q * (q.log() - m.log())).sum(-1) + (1 - beta) * (p * (p.log() - m.log())).sum(-1)
    return value * spec.temperature**2 if spec.temperature != 1.0 else value
