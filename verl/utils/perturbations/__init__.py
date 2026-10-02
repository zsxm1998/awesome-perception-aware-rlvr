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
"""Shared image and feature perturbation helpers."""

from .feature_views import perturb_visual_features
from .pixel import (
    compute_noise_schedule,
    gaussian_noise_image,
    pixelate_image,
    random_patch_blackening,
    vp_diffusion_noise,
    vp_diffusion_noise_fixed_t,
)
from .seeding import derive_seed


__all__ = [
    "compute_noise_schedule",
    "derive_seed",
    "gaussian_noise_image",
    "perturb_visual_features",
    "pixelate_image",
    "random_patch_blackening",
    "vp_diffusion_noise",
    "vp_diffusion_noise_fixed_t",
]
