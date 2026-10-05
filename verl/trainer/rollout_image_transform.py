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
"""Rollouts generated from transformed images (NoisyRollout).

For every prompt, the first half of the ``worker.rollout.n`` responses is sampled from the clean images and the
second half from a transformed copy of them. All responses of a prompt share its uid (one advantage group), and
the transformed half is trained on the clean images: after generation its ``multi_modal_data`` is set back to the
prompt's clean images, so the old, reference and current log-probs are all conditioned on them. This is what the
released NoisyRollout code does, where the noised images only reach vLLM.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

import numpy as np

from ..protocol import DataProto
from ..utils.dataset import ProcessedImageInput, process_image
from ..utils.perturbations.pixel import compute_noise_schedule, vp_diffusion_noise


ROLLOUT_IMAGE_TRANSFORMED_KEY = "rollout_image_transformed"
"""bool column of the training batch: the response was sampled from the transformed images (metrics only)."""

ROLLOUT_IMAGE_TRANSFORMS = ("vp_diffusion",)

_VP_DIFFUSION_REQUIRED_KWARGS = ("noise_t_init", "noise_gamma", "noise_t_mid")
_VP_DIFFUSION_DEFAULT_KWARGS = {"noise_t_max": 1000.0, "pixel_rounding": "floor"}


def normalize_rollout_image_transform_kwargs(transform: str, kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Check the kwargs of ``algorithm.rollout_image_transform`` and fill the defaults."""
    if transform not in ROLLOUT_IMAGE_TRANSFORMS:
        raise ValueError(
            f"rollout_image_transform must be one of {list(ROLLOUT_IMAGE_TRANSFORMS)}, got {transform!r}."
        )
    kwargs = dict(kwargs or {})
    allowed = set(_VP_DIFFUSION_REQUIRED_KWARGS) | set(_VP_DIFFUSION_DEFAULT_KWARGS)
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"Unknown rollout_image_transform_kwargs {unknown}; the accepted keys are {sorted(allowed)}.")
    missing = [key for key in _VP_DIFFUSION_REQUIRED_KWARGS if key not in kwargs]
    if missing:
        raise ValueError(
            f"rollout_image_transform=vp_diffusion requires rollout_image_transform_kwargs {missing} "
            "(the annealing alpha_0, lambda and gamma / t_max of NoisyRollout's Eq. 3)."
        )
    normalized = {**_VP_DIFFUSION_DEFAULT_KWARGS, **kwargs}
    for key in ("noise_t_init", "noise_gamma", "noise_t_mid", "noise_t_max"):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"rollout_image_transform_kwargs.{key} must be a finite number, got {value!r}.")
        normalized[key] = float(value)
    if not 0.0 < normalized["noise_t_mid"] < 1.0:
        raise ValueError(
            f"noise_t_mid is a fraction of the training steps in (0, 1), got {normalized['noise_t_mid']}."
        )
    if not 0.0 < normalized["noise_t_max"] <= 1000.0:
        raise ValueError(f"noise_t_max must be in (0, 1000], got {normalized['noise_t_max']}.")
    if normalized["noise_t_init"] < 0.0:
        raise ValueError(f"noise_t_init must be non-negative, got {normalized['noise_t_init']}.")
    if normalized["pixel_rounding"] not in ("floor", "round"):
        raise ValueError(f"pixel_rounding must be 'floor' or 'round', got {normalized['pixel_rounding']!r}.")
    return normalized


def _image_seed(data_seed: int, global_step: int, generation_round: int, row: int, image_idx: int) -> int:
    payload = f"rollout_image_transform_v1:{data_seed}:{global_step}:{generation_round}:{row}:{image_idx}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little", signed=False)


def _transform_multi_modal_data(
    multi_modal_data: Any,
    *,
    noise_beta: float,
    rounding: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    seeds: list[int],
) -> Any:
    if multi_modal_data is None:
        return None
    if multi_modal_data.get("videos"):
        raise ValueError("algorithm.rollout_image_transform supports image inputs only.")
    images = multi_modal_data.get("images")
    if not images:
        return multi_modal_data
    transformed = dict(multi_modal_data)
    # resize like the rollout worker does for the clean images, then noise; the worker does not resize them again
    transformed["images"] = [
        ProcessedImageInput(
            image=vp_diffusion_noise(
                process_image(image, min_pixels, max_pixels), beta=noise_beta, seed=seed, rounding=rounding
            )
        )
        for image, seed in zip(images, seeds)
    ]
    return transformed


def build_rollout_image_views(
    gen_batch: DataProto,
    *,
    transform: str,
    transform_kwargs: dict[str, Any],
    rollout_n: int,
    global_step: int,
    total_training_steps: int,
    generation_round: int,
    data_seed: int,
) -> tuple[DataProto, dict[str, float]]:
    """The generation batch [clean_0, transformed_0, clean_1, transformed_1, ...] with ``meta_info["n"]`` set to
    half of ``rollout_n``, so that the rollout returns the responses in the order of the repeated prompts."""
    if transform != "vp_diffusion":
        raise ValueError(f"Unsupported rollout_image_transform {transform!r}.")
    if rollout_n % 2 != 0:
        raise ValueError(f"algorithm.rollout_image_transform requires an even worker.rollout.n, got {rollout_n}.")
    noise_t, noise_beta = compute_noise_schedule(
        global_step=global_step,
        total_training_steps=total_training_steps,
        noise_t_init=transform_kwargs["noise_t_init"],
        noise_gamma=transform_kwargs["noise_gamma"],
        noise_t_max=transform_kwargs["noise_t_max"],
        noise_t_mid=transform_kwargs["noise_t_mid"],
    )
    clean_multi_modal_data = gen_batch.non_tensor_batch.get("multi_modal_data")
    if clean_multi_modal_data is None:
        raise ValueError("algorithm.rollout_image_transform requires image inputs (multi_modal_data).")
    transformed_multi_modal_data = np.empty(len(clean_multi_modal_data), dtype=object)
    for row, multi_modal_data in enumerate(clean_multi_modal_data):
        num_images = len((multi_modal_data or {}).get("images") or [])
        transformed_multi_modal_data[row] = _transform_multi_modal_data(
            multi_modal_data,
            noise_beta=noise_beta,
            rounding=transform_kwargs["pixel_rounding"],
            min_pixels=gen_batch.meta_info.get("min_pixels"),
            max_pixels=gen_batch.meta_info.get("max_pixels"),
            seeds=[
                _image_seed(data_seed, global_step, generation_round, row, image_idx)
                for image_idx in range(num_images)
            ],
        )

    transformed_batch = DataProto(
        batch=gen_batch.batch,
        non_tensor_batch={**gen_batch.non_tensor_batch, "multi_modal_data": transformed_multi_modal_data},
        meta_info=gen_batch.meta_info,
    )
    num_prompts = len(gen_batch)
    combined = DataProto.concat([gen_batch, transformed_batch])
    interleaved = combined[np.arange(2 * num_prompts).reshape(2, num_prompts).T.reshape(-1)]
    interleaved.meta_info = {**gen_batch.meta_info, "n": rollout_n // 2}
    stats = {"rollout_transform/noise_t": float(int(noise_t)), "rollout_transform/noise_std": math.sqrt(noise_beta)}
    return interleaved, stats


def restore_clean_images(gen_output: DataProto, clean_multi_modal_data: np.ndarray, rollout_n: int) -> None:
    """Point every response at its prompt's clean images and mark the ones sampled from the transformed images.

    ``gen_output`` holds, per prompt, ``rollout_n / 2`` responses from the clean images and then ``rollout_n / 2``
    from the transformed ones, the order of ``build_rollout_image_views``.
    """
    half = rollout_n // 2
    num_prompts = len(clean_multi_modal_data)
    if len(gen_output) != num_prompts * rollout_n:
        raise RuntimeError(f"Expected {num_prompts * rollout_n} responses, got {len(gen_output)}.")
    gen_output.non_tensor_batch["multi_modal_data"] = np.repeat(clean_multi_modal_data, rollout_n)
    gen_output.non_tensor_batch[ROLLOUT_IMAGE_TRANSFORMED_KEY] = np.tile(
        np.array([False] * half + [True] * half, dtype=bool), num_prompts
    )
    gen_output.meta_info.pop("n", None)


def compute_rollout_image_transform_metrics(batch: DataProto) -> dict[str, float]:
    """Accuracy of the responses sampled from the clean and from the transformed images (NoisyRollout Fig. 3)."""
    transformed = batch.non_tensor_batch.get(ROLLOUT_IMAGE_TRANSFORMED_KEY)
    details = batch.non_tensor_batch.get("reward_details")
    if transformed is None or details is None:
        return {}
    metrics = {}
    for name, selected in (("clean", ~transformed.astype(bool)), ("transformed", transformed.astype(bool))):
        values = [details[idx].get("accuracy") for idx in np.flatnonzero(selected)]
        values = [float(value) for value in values if value is not None]
        if values:
            metrics[f"rollout_transform/accuracy_{name}"] = float(np.mean(values))
    return metrics
