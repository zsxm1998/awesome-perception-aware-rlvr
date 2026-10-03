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
"""Noise on the image processor's normalized pixel_values (training-side `gaussian_noise`, as VEPO's code).

The std is in normalized units: for Qwen2.5-VL (image_std about 0.27) 2.0 is about 0.54 in [0, 1] pixel units, for
Qwen3-VL (image_std 0.5) 1.08 is.
"""

from __future__ import annotations

from typing import Any

import torch


# key of multi_modal_data that carries {"std": float, "seed": int} from the auxiliary view to the worker
PIXEL_VALUES_NOISE_KEY = "pixel_values_noise"


def add_pixel_values_noise(multi_modal_inputs: dict[str, Any], std: float, seed: int) -> dict[str, Any]:
    """Return a copy of ``multi_modal_inputs`` whose image pixel_values have N(0, std^2) noise added, unclipped.

    The noise is drawn from ``seed`` alone, so every call for the same view adds the same noise; the input tensors
    are left untouched.
    """
    pixel_values = multi_modal_inputs.get("pixel_values")
    if pixel_values is None:
        raise ValueError(f"{PIXEL_VALUES_NOISE_KEY} needs image pixel_values from the image processor.")
    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(pixel_values.shape, generator=generator, dtype=torch.float32)
    noisy = (pixel_values.float() + noise * float(std)).to(pixel_values.dtype)
    return {**multi_modal_inputs, "pixel_values": noisy}
