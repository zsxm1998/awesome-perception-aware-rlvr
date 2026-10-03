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
"""Auxiliary image views built by the corruption builder: random patches masked before or after resizing."""

import numpy as np
import pytest
from PIL import Image

from verl.trainer.config import AlgorithmConfig
from verl.trainer.perception_reasoning_data import PerceptionReasoningCorruptionBuilder, _stable_prompt_seed
from verl.utils.dataset import process_image
from verl.utils.perturbations.pixel import random_patch_blackening


MIN_PIXELS = 112 * 112  # a 40x30 image is upscaled about 3.2x


def _builder() -> PerceptionReasoningCorruptionBuilder:
    return PerceptionReasoningCorruptionBuilder(
        tokenizer=None, processor=None, image_patch_size=14, min_pixels=MIN_PIXELS, max_pixels=1024 * 1024
    )


def _image(width=40, height=30) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(1, 255, size=(height, width, 3), dtype=np.uint8))


def _random_patch(builder, image, kwargs, position="prompt"):
    transformed, _ = builder._transform_multi_modal_data(
        multi_modal_data={"images": [image]},
        response_text="answer",
        transform_name="random_patch",
        transform_kwargs=kwargs,
        transform_position=position,
        corruption_group_key="group",
        global_step=3,
        total_training_steps=None,
    )
    return np.asarray(transformed["images"][0].image)


def _prompt_seed(builder, kwargs) -> int:
    return _stable_prompt_seed(builder._cache_descriptor("random_patch", kwargs, 3) + ":group")


def test_mask_after_resize_is_unchanged():
    builder, image = _builder(), _image()
    kwargs = {"patch_size": 14, "black_prob": 0.6}
    expected = random_patch_blackening(
        process_image(image, MIN_PIXELS, 1024 * 1024), seed=_prompt_seed(builder, kwargs), **kwargs
    )
    for flag in ({}, {"mask_before_resize": False}):
        np.testing.assert_array_equal(_random_patch(_builder(), image, {**kwargs, **flag}), np.asarray(expected))


def test_mask_before_resize_masks_the_original_image():
    """PAPO's code masks the image at its original resolution, then resizes it like the clean image."""
    builder, image = _builder(), _image()
    kwargs = {"patch_size": 14, "black_prob": 0.6}
    masked = _random_patch(builder, image, {**kwargs, "mask_before_resize": True})

    seed = _stable_prompt_seed(builder._cache_descriptor("random_patch", kwargs, 3) + ":mask_before_resize:group")
    original = process_image(image, None, None)
    expected = process_image(random_patch_blackening(original, seed=seed, **kwargs), MIN_PIXELS, 1024 * 1024)
    np.testing.assert_array_equal(masked, np.asarray(expected))
    clean = process_image(image, MIN_PIXELS, 1024 * 1024)
    assert expected.size == clean.size and clean.size != image.size  # same image tokens as the clean view
    assert not np.array_equal(masked, _random_patch(_builder(), image, kwargs))


def test_mask_before_resize_is_validated():
    with pytest.raises(ValueError, match="mask_before_resize"):
        AlgorithmConfig(corrupt_image="random_patch", corrupt_image_kwargs={"mask_before_resize": "yes"}).post_init()
    with pytest.raises(ValueError, match="unsupported corrupt_image_kwargs"):
        AlgorithmConfig(corrupt_image="random_patch", corrupt_image_kwargs={"patch": 14}).post_init()
    AlgorithmConfig(
        corrupt_image="random_patch", corrupt_image_kwargs={"patch_size": 14, "mask_before_resize": True}
    ).post_init()
