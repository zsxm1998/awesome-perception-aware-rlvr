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
"""Auxiliary image views: random patches masked before or after resizing; noise on normalized pixel_values."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from verl.protocol import DataProto
from verl.trainer.config import AlgorithmConfig
from verl.trainer.perception_reasoning_data import PerceptionReasoningCorruptionBuilder, _stable_prompt_seed
from verl.utils.dataset import process_image
from verl.utils.perturbations.pixel import random_patch_blackening
from verl.utils.perturbations.pixel_values import PIXEL_VALUES_NOISE_KEY, add_pixel_values_noise
from verl.workers.fsdp_workers import FSDPWorker


MIN_PIXELS = 112 * 112  # a 40x30 image is upscaled about 3.2x


def _builder() -> PerceptionReasoningCorruptionBuilder:
    return PerceptionReasoningCorruptionBuilder(
        tokenizer=None, processor=None, image_patch_size=14, min_pixels=MIN_PIXELS, max_pixels=1024 * 1024
    )


def _image(width=40, height=30) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(1, 255, size=(height, width, 3), dtype=np.uint8))


def _transform(builder, image, name, kwargs, position="prompt", group="group", sample_idx=0, response="answer"):
    transformed, _ = builder._transform_multi_modal_data(
        multi_modal_data={"images": [image]},
        response_text=response,
        transform_name=name,
        transform_kwargs=kwargs,
        transform_position=position,
        corruption_group_key=group,
        global_step=3,
        total_training_steps=None,
        sample_idx=sample_idx,
    )
    return transformed


def _random_patch(builder, image, kwargs, position="prompt"):
    return np.asarray(_transform(builder, image, "random_patch", kwargs, position)["images"][0].image)


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


def test_gaussian_noise_view_keeps_the_clean_image_and_carries_the_noise():
    builder, image = _builder(), _image()
    view = _transform(builder, image, "gaussian_noise", {"std": 2.0})
    clean = process_image(image, MIN_PIXELS, 1024 * 1024)
    np.testing.assert_array_equal(np.asarray(view["images"][0].image), np.asarray(clean))
    assert view[PIXEL_VALUES_NOISE_KEY]["std"] == 2.0

    def seed(**kwargs):
        return _transform(_builder(), image, "gaussian_noise", {"std": 2.0}, **kwargs)[PIXEL_VALUES_NOISE_KEY]["seed"]

    # position=prompt: one draw per prompt; position=response: one per response, even for equal texts
    assert seed(sample_idx=0) == seed(sample_idx=1) != seed(group="other")
    assert seed(position="response", sample_idx=0) != seed(position="response", sample_idx=1)
    assert seed(position="response") == seed(position="response")


def test_add_pixel_values_noise():
    pixel_values = torch.randn(4096, 1176)
    inputs = {"pixel_values": pixel_values, "image_grid_thw": torch.tensor([[1, 64, 64]])}
    before = pixel_values.clone()
    noisy = add_pixel_values_noise(inputs, std=2.0, seed=7)

    assert torch.equal(pixel_values, before) and noisy["image_grid_thw"] is inputs["image_grid_thw"]
    noise = noisy["pixel_values"] - pixel_values
    assert noise.std().item() == pytest.approx(2.0, rel=0.01) and noise.abs().max() > 6  # not clipped
    assert torch.equal(noisy["pixel_values"], add_pixel_values_noise(inputs, std=2.0, seed=7)["pixel_values"])
    assert not torch.equal(noisy["pixel_values"], add_pixel_values_noise(inputs, std=2.0, seed=8)["pixel_values"])


class _ImageProcessor:
    def __call__(self, images, return_tensors):
        pixel_values = torch.cat(
            [torch.from_numpy(np.asarray(image, dtype=np.float32)).reshape(-1, 3) for image in images]
        )
        return {"pixel_values": pixel_values / 255.0, "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images))}


def test_worker_adds_the_noise_to_the_auxiliary_view_only():
    """The clean view's features (and their cache) stay untouched; rows sharing a cache id share the noise."""
    image = _image(4, 4)
    worker = object.__new__(FSDPWorker)
    worker._cache = {}
    worker.processor = SimpleNamespace(image_processor=_ImageProcessor())
    meta_info = {"min_pixels": None, "max_pixels": None, "video_fps": 2.0}
    clean = DataProto.from_dict(
        non_tensors={
            "multi_modal_data": np.array([{"images": [image]}] * 3, dtype=object),
            "uid": np.array(["u", "u", "v"], dtype=object),
        },
        meta_info=meta_info,
    )
    noise = {"std": 2.0, "seed": 11}
    aux = DataProto.from_dict(
        non_tensors={
            "multi_modal_data": np.array(
                [{"images": [image], PIXEL_VALUES_NOISE_KEY: {**noise, "seed": seed}} for seed in (11, 11, 12)],
                dtype=object,
            ),
            "multi_modal_cache_id": np.array(["u", "u", "v"], dtype=object),
        },
        meta_info=meta_info,
    )
    worker._process_multi_modal_inputs(clean)
    worker._process_multi_modal_inputs(aux, cache_namespace="decremental", cache_key_field="multi_modal_cache_id")

    expected_clean = _ImageProcessor()([image], "pt")["pixel_values"]
    clean_inputs = clean.non_tensor_batch["multi_modal_inputs"]
    aux_inputs = aux.non_tensor_batch["multi_modal_inputs"]
    for row in range(3):
        assert torch.equal(clean_inputs[row]["pixel_values"], expected_clean)
    assert torch.equal(worker._cache["multi_modal_data:multi_modal_inputs"][0]["pixel_values"], expected_clean)
    expected_aux = add_pixel_values_noise({"pixel_values": expected_clean}, **noise)["pixel_values"]
    assert torch.equal(aux_inputs[0]["pixel_values"], expected_aux)
    assert torch.equal(aux_inputs[1]["pixel_values"], expected_aux)
    assert not torch.equal(aux_inputs[2]["pixel_values"], expected_aux)


def test_gaussian_noise_kwargs_are_validated():
    with pytest.raises(ValueError, match="std"):
        AlgorithmConfig(corrupt_image="gaussian_noise", corrupt_image_kwargs={"std": 0}).post_init()
    with pytest.raises(ValueError, match="unsupported corrupt_image_kwargs"):
        AlgorithmConfig(corrupt_image="gaussian_noise", corrupt_image_kwargs={"sigma": 1.0}).post_init()
