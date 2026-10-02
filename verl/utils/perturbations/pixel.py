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
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VPDiffusionParams:
    noise_t: float
    noise_beta: float

    def __iter__(self):
        yield self.noise_t
        yield self.noise_beta


def compute_noise_schedule(
    global_step: int,
    total_training_steps: int,
    noise_t_init: float,
    noise_gamma: float,
    noise_t_max: float,
) -> VPDiffusionParams:
    if total_training_steps <= 0:
        progress = 0.0
    else:
        progress = min(max(float(global_step), 0.0), float(total_training_steps)) / float(max(total_training_steps, 1))
    noise_t = noise_t_init * (1.0 / (1.0 + math.exp(-noise_gamma * (0.5 - progress))))
    noise_beta = min(max(noise_t / noise_t_max, 0.0), 1.0)
    return VPDiffusionParams(noise_t=noise_t, noise_beta=noise_beta)


def vp_diffusion_params_from_fixed_t(noise_t: float, noise_t_max: float) -> VPDiffusionParams:
    noise_t = min(max(float(noise_t), 0.0), float(noise_t_max))
    noise_beta = min(max(noise_t / float(noise_t_max), 0.0), 1.0)
    return VPDiffusionParams(noise_t=noise_t, noise_beta=noise_beta)


def random_patch_blackening(
    pil_img: Image.Image, patch_size: int = 14, black_prob: float = 0.6, seed: int | None = None
):
    rng = np.random.default_rng(seed)
    img = np.array(pil_img)
    height, width = img.shape[:2]

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            if rng.random() < black_prob:
                y_end = min(y + patch_size, height)
                x_end = min(x + patch_size, width)
                img[y:y_end, x:x_end, ...] = 0

    return Image.fromarray(img)


def gaussian_noise_image(pil_img: Image.Image, std: float = 0.5, seed: int | None = None) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    noisy = arr + rng.standard_normal(size=arr.shape, dtype=np.float32) * std
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray(np.asarray(np.round(noisy * 255.0), dtype=np.uint8))


def pixelate_image(pil_img: Image.Image, ratio: float = 0.1) -> Image.Image:
    """Remove fine visual details by bilinear downsampling and nearest-neighbor upsampling."""
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.0 < float(ratio) <= 1.0:
        raise ValueError(f"ratio must be a number in (0, 1], but got {ratio!r}.")

    width, height = pil_img.size
    downsampled_size = (
        max(1, int(width * float(ratio))),
        max(1, int(height * float(ratio))),
    )
    downsampled = pil_img.resize(downsampled_size, resample=Image.Resampling.BILINEAR)
    return downsampled.resize((width, height), resample=Image.Resampling.NEAREST)


def vp_diffusion_noise(
    pil_img: Image.Image,
    beta: float,
    seed: int | None = None,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    noise = rng.standard_normal(size=arr.shape, dtype=np.float32)
    noisy = math.sqrt(max(1.0 - beta, 0.0)) * arr + math.sqrt(beta) * noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray(np.asarray(np.round(noisy * 255.0), dtype=np.uint8))


def vp_diffusion_noise_fixed_t(
    pil_img: Image.Image,
    noise_t: float,
    noise_t_max: float = 1000.0,
    seed: int | None = None,
) -> tuple[Image.Image, VPDiffusionParams]:
    params = vp_diffusion_params_from_fixed_t(noise_t, noise_t_max)
    return vp_diffusion_noise(pil_img, beta=params.noise_beta, seed=seed), params
