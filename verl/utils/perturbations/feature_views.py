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
from __future__ import annotations

import math
from typing import Any

import torch

from .seeding import derive_seed


FEATURE_DIVERGENCE = "feature_divergence"
FEATURE_CONSISTENCY = "feature_consistency"


def perturb_visual_features(
    main_features: torch.Tensor,
    deepstack_features: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    *,
    image_grid_thw: torch.Tensor,
    merge_size: int,
    perturbation_type: str,
    params: dict[str, Any],
    global_seed: int,
    sample_id: str,
    include_token_indices: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | tuple[torch.Tensor, ...] | None, list[dict[str, Any]]]:
    if perturbation_type not in {FEATURE_DIVERGENCE, FEATURE_CONSISTENCY}:
        return main_features, deepstack_features, []
    if image_grid_thw is None:
        raise ValueError("Feature perturbation requires image_grid_thw.")
    if merge_size < 1:
        raise ValueError(f"invalid spatial_merge_size: {merge_size}")

    split_sizes = _token_split_sizes(image_grid_thw, merge_size)
    levels: list[tuple[str, torch.Tensor]] = [("main", main_features)]
    if deepstack_features:
        levels.extend((f"deepstack_{index}", tensor) for index, tensor in enumerate(deepstack_features))
    for level_name, tensor in levels:
        if int(tensor.shape[0]) != sum(split_sizes):
            raise ValueError(
                f"{level_name} token count {int(tensor.shape[0])} does not match image_grid_thw total {sum(split_sizes)}."
            )

    token_masks = [
        _shared_token_mask(
            token_count=count,
            device=main_features.device,
            global_seed=global_seed,
            sample_id=sample_id,
            image_idx=image_idx,
            token_p=float(params.get("token_p", 1.0)) if perturbation_type == FEATURE_DIVERGENCE else 1.0,
        )
        for image_idx, count in enumerate(split_sizes)
    ]

    diagnostics: list[dict[str, Any]] = []
    perturbed_levels = []
    for level_name, level_tensor in levels:
        chunks = list(torch.split(level_tensor, split_sizes, dim=0))
        updated_chunks = []
        for image_idx, chunk in enumerate(chunks):
            mask, mask_seed = token_masks[image_idx]
            updated, diag = _perturb_level(
                chunk,
                mask=mask,
                mask_seed=mask_seed,
                perturbation_type=perturbation_type,
                params=params,
                global_seed=global_seed,
                sample_id=sample_id,
                image_idx=image_idx,
                level=level_name,
                include_token_indices=include_token_indices,
            )
            updated_chunks.append(updated)
            diagnostics.append(diag)
        perturbed_levels.append(torch.cat(updated_chunks, dim=0))

    main_out = perturbed_levels[0]
    if deepstack_features is None:
        deepstack_out = None
    else:
        deepstack_items = perturbed_levels[1:]
        deepstack_out = tuple(deepstack_items) if isinstance(deepstack_features, tuple) else deepstack_items
    return main_out, deepstack_out, diagnostics


def _perturb_level(
    features: torch.Tensor,
    *,
    mask: torch.Tensor,
    mask_seed: int,
    perturbation_type: str,
    params: dict[str, Any],
    global_seed: int,
    sample_id: str,
    image_idx: int,
    level: str,
    include_token_indices: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x = features.float()
    centered = x - x.mean(dim=0, keepdim=True)
    basis_info = _semantic_basis(centered, params, require_tail=perturbation_type == FEATURE_CONSISTENCY)
    out = x.clone()
    if perturbation_type == FEATURE_DIVERGENCE:
        subspace_seed = derive_seed(global_seed, sample_id, image_idx, "subspace", level)
        semantic_basis = basis_info["semantic_basis"]
        deletion_basis = _nested_random_basis(
            semantic_basis,
            rho=float(params.get("rho", 1.0)),
            seed=subspace_seed,
            device=features.device,
        )
        if mask.any():
            projection = centered[mask] @ deletion_basis
            out[mask] = out[mask] - projection @ deletion_basis.T
        seed_key = "subspace_seed"
        seed_value = subspace_seed
    elif perturbation_type == FEATURE_CONSISTENCY:
        noise_seed = derive_seed(global_seed, sample_id, image_idx, "tail_noise", level)
        tail_basis = basis_info["tail_basis"]
        tail_std = basis_info["tail_std"].to(device=features.device)
        epsilon = _normal(
            (features.shape[0], tail_basis.shape[1]),
            seed=noise_seed,
            device=features.device,
            dtype=torch.float32,
        )
        noise = epsilon * tail_std.unsqueeze(0) * float(params.get("noise_lambda", 1.0))
        out = out + noise @ tail_basis.T
        seed_key = "tail_noise_seed"
        seed_value = noise_seed
    else:
        raise ValueError(f"Unsupported feature perturbation: {perturbation_type}")

    out = out.to(dtype=features.dtype)
    delta = out.float() - features.float()
    token_delta_norm = delta.norm(dim=1)
    token_base_norm = features.float().norm(dim=1).clamp_min(1e-12)
    selected = mask if perturbation_type == FEATURE_DIVERGENCE else torch.ones_like(mask, dtype=torch.bool)
    if selected.any():
        selected_delta = delta[selected]
        selected_base = centered[selected].norm().clamp_min(1e-12)
        deletion_energy_ratio = float((selected_delta.norm() ** 2 / (selected_base**2)).detach().cpu())
        mean_relative_change = float((token_delta_norm[selected] / token_base_norm[selected]).mean().detach().cpu())
    else:
        deletion_energy_ratio = 0.0
        mean_relative_change = 0.0
    diag: dict[str, Any] = {
        "type": perturbation_type,
        "sample_id": sample_id,
        "image_idx": image_idx,
        "level": level,
        "token_mask_seed": mask_seed,
        "token_mask_count": int(mask.sum().item())
        if perturbation_type == FEATURE_DIVERGENCE
        else int(features.shape[0]),
        "token_count": int(features.shape[0]),
        seed_key: seed_value,
        "rank": basis_info["rank"],
        "k_effective": basis_info["k_effective"],
        "tail_dim": basis_info["tail_dim"],
        "rogue_component_count": len(basis_info["rogue_indices"]),
        "rogue_indices": basis_info["rogue_indices"],
        "deletion_energy_ratio": deletion_energy_ratio,
        "mean_relative_change": mean_relative_change,
        "l2_change": float(delta.norm().detach().cpu()),
    }
    if basis_info.get("k_clamped_from") is not None:
        diag["k_clamped_from"] = basis_info["k_clamped_from"]
    if include_token_indices:
        diag["token_indices"] = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    return out, diag


def _semantic_basis(centered: torch.Tensor, params: dict[str, Any], *, require_tail: bool) -> dict[str, Any]:
    token_count, dim = centered.shape
    rank_cap = min(max(token_count - 1, 0), dim)
    if rank_cap < 1:
        raise ValueError("Feature perturbation requires at least two tokens and one non-zero rank component.")
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    singular_values = singular_values[:rank_cap]
    vh = vh[:rank_cap]
    rank = int(rank_cap)
    rogue_indices = set(range(min(int(params.get("n_skip", 0)), rank)))
    if bool(params.get("auto_rogue", True)):
        threshold = float(params.get("rogue_corr_threshold", 0.95))
        norms = centered.norm(dim=1)
        for index in range(rank):
            coords = centered @ vh[index].T
            corr = _pearson_abs(coords.abs(), norms)
            if corr >= threshold:
                rogue_indices.add(index)
    candidates = [index for index in range(rank) if index not in rogue_indices]
    if not candidates:
        raise ValueError("Semantic subspace is empty after n_skip/auto_rogue filtering.")

    if params.get("ev_threshold") is not None and params.get("k") is not None:
        raise ValueError("Specify only one of k and ev_threshold for feature perturbation.")
    if params.get("ev_threshold") is not None:
        energy = singular_values[candidates].square()
        total = energy.sum()
        if float(total.item()) <= 0.0:
            raise ValueError("Semantic subspace has zero energy.")
        cutoff = torch.searchsorted(torch.cumsum(energy, dim=0), float(params["ev_threshold"]) * total).item() + 1
        k_effective = int(min(max(cutoff, 1), len(candidates)))
        k_clamped_from = None
    else:
        requested_k = int(params.get("k", 64))
        k_effective = min(max(requested_k, 1), len(candidates))
        k_clamped_from = requested_k if requested_k != k_effective else None
    semantic_indices = candidates[:k_effective]
    if not semantic_indices:
        raise ValueError("Semantic subspace is empty.")

    explicit_m_tail = params.get("m_tail") is not None
    if explicit_m_tail:
        tail_start = int(params["m_tail"])
        if tail_start >= len(candidates):
            raise ValueError(f"m_tail={tail_start} exceeds available non-rogue component count {len(candidates)}.")
    else:
        tail_start = k_effective
    tail_indices = candidates[tail_start:]
    if not explicit_m_tail and not tail_indices:
        tail_indices = candidates[k_effective:]
    if require_tail and not tail_indices:
        raise ValueError("Tail subspace is empty; lower k/m_tail or increase token count.")

    semantic_basis = vh[semantic_indices].T.contiguous()
    tail_basis = vh[tail_indices].T.contiguous() if tail_indices else vh[:0].T.contiguous()
    tail_std = (
        singular_values[tail_indices] / math.sqrt(max(token_count - 1, 1)) if tail_indices else singular_values[:0]
    )
    return {
        "rank": rank,
        "k_effective": k_effective,
        "tail_dim": int(len(tail_indices)),
        "semantic_basis": semantic_basis,
        "tail_basis": tail_basis,
        "tail_std": tail_std,
        "rogue_indices": sorted(rogue_indices),
        "k_clamped_from": k_clamped_from,
    }


def _shared_token_mask(
    *,
    token_count: int,
    device,
    global_seed: int,
    sample_id: str,
    image_idx: int,
    token_p: float,
) -> tuple[torch.Tensor, int]:
    seed = derive_seed(global_seed, sample_id, image_idx, "token_mask")
    values = _uniform((token_count,), seed=seed, device=device, dtype=torch.float32)
    return values < token_p, seed


def _nested_random_basis(semantic_basis: torch.Tensor, *, rho: float, seed: int, device) -> torch.Tensor:
    k = int(semantic_basis.shape[1])
    columns = max(1, min(k, math.ceil(k * rho)))
    q, _ = torch.linalg.qr(_normal((k, k), seed=seed, device=device, dtype=torch.float32))
    return semantic_basis @ q[:, :columns]


def _uniform(shape, *, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.rand(shape, generator=generator, dtype=dtype).to(device=device)


def _normal(shape, *, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=dtype).to(device=device)


def _pearson_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((a @ b / denom).abs().detach().cpu())


def _token_split_sizes(image_grid_thw: torch.Tensor, merge_size: int) -> list[int]:
    if not torch.is_tensor(image_grid_thw):
        image_grid_thw = torch.as_tensor(image_grid_thw)
    grid = image_grid_thw.detach().cpu().long()
    return [int(item.prod().item() // (merge_size**2)) for item in grid]
