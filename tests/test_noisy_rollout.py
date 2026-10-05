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
"""NoisyRollout: the annealed noise step, the uint8 rounding, and rollouts generated from noised images."""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from verl.protocol import DataProto
from verl.trainer import ray_trainer as ray_trainer_module
from verl.trainer.config import PPOConfig
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.trainer.rollout_image_transform import (
    ROLLOUT_IMAGE_TRANSFORMED_KEY,
    compute_rollout_image_transform_metrics,
    normalize_rollout_image_transform_kwargs,
)
from verl.utils.dataset import ProcessedImageInput, process_image
from verl.utils.perturbations import compute_noise_schedule, vp_diffusion_noise


def _released_noise_step(alpha0, lam, gamma_steps, total_steps, step):
    # NoisyRollout's Eq. 3 with gamma in steps, truncated to an int step
    return int(alpha0 * (1.0 - 1.0 / (1.0 + math.exp(-lam * (step / total_steps - gamma_steps / total_steps)))))


@pytest.mark.parametrize(
    "alpha0, lam, gamma_steps, total_steps",
    [(500, 30, 40, 60), (450, 60, 40, 120), (450, 30, 35, 40), (450, 60, 67, 202), (500, 30, 135, 202)],
)
def test_noise_schedule_with_midpoint_matches_eq3_every_step(alpha0, lam, gamma_steps, total_steps):
    for step in range(1, total_steps + 1):
        noise_t, _ = compute_noise_schedule(
            step, total_steps, alpha0, lam, noise_t_max=1000.0, noise_t_mid=gamma_steps / total_steps
        )
        assert int(noise_t) == _released_noise_step(alpha0, lam, gamma_steps, total_steps, step)


def test_noise_schedule_anchor_steps():
    def step_of(alpha0, lam, gamma_steps, total_steps, step):
        noise_t, _ = compute_noise_schedule(
            step, total_steps, alpha0, lam, 1000.0, noise_t_mid=gamma_steps / total_steps
        )
        return int(noise_t)

    assert [step_of(500, 30, 40, 60, t) for t in (1, 27, 40, 48, 53, 60)] == [499, 499, 250, 8, 0, 0]
    assert [step_of(450, 60, 40, 120, t) for t in (1, 40, 53, 120)] == [449, 225, 0, 0]


def test_noise_schedule_without_midpoint_is_unchanged():
    for step in range(0, 61):
        noise_t, noise_beta = compute_noise_schedule(step, 60, 500.0, 10.0, 1000.0)
        progress = min(max(step, 0), 60) / 60
        assert noise_t == 500.0 * (1.0 / (1.0 + math.exp(-10.0 * (0.5 - progress))))
        betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
        assert noise_beta == 1.0 - torch.cumprod(1 - betas, dim=0).tolist()[int(noise_t)]


def test_vp_diffusion_noise_rounding():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, size=(12, 10, 3), dtype=np.uint8))
    beta = 0.2
    arr = np.asarray(image, dtype=np.float32) / 255.0
    noise = np.random.default_rng(5).standard_normal(size=arr.shape, dtype=np.float32)
    expected = np.clip(math.sqrt(1.0 - beta) * arr + math.sqrt(beta) * noise, 0.0, 1.0) * 255.0

    rounded = np.asarray(vp_diffusion_noise(image, beta, seed=5))
    assert np.array_equal(rounded, np.asarray(vp_diffusion_noise(image, beta, seed=5, rounding="round")))
    assert np.array_equal(rounded, np.round(expected).astype(np.uint8))
    floored = np.asarray(vp_diffusion_noise(image, beta, seed=5, rounding="floor"))
    assert np.array_equal(floored, np.floor(expected).astype(np.uint8))
    assert (floored != rounded).any()
    with pytest.raises(ValueError, match="rounding"):
        vp_diffusion_noise(image, beta, rounding="ceil")


def test_kwargs_are_checked_and_filled():
    kwargs = {"noise_t_init": 450, "noise_gamma": 60, "noise_t_mid": 0.5}
    assert normalize_rollout_image_transform_kwargs("vp_diffusion", kwargs) == {
        "noise_t_init": 450.0,
        "noise_gamma": 60.0,
        "noise_t_mid": 0.5,
        "noise_t_max": 1000.0,
        "pixel_rounding": "floor",
    }
    with pytest.raises(ValueError, match="requires"):
        normalize_rollout_image_transform_kwargs("vp_diffusion", {"noise_t_init": 450})
    with pytest.raises(ValueError, match="Unknown"):
        normalize_rollout_image_transform_kwargs("vp_diffusion", {**kwargs, "noise_t": 3})
    with pytest.raises(ValueError, match="noise_t_mid"):
        normalize_rollout_image_transform_kwargs("vp_diffusion", {**kwargs, "noise_t_mid": 40})
    with pytest.raises(ValueError, match="pixel_rounding"):
        normalize_rollout_image_transform_kwargs("vp_diffusion", {**kwargs, "pixel_rounding": "ceil"})
    with pytest.raises(ValueError, match="one of"):
        normalize_rollout_image_transform_kwargs("rotate", kwargs)


KWARGS = {"noise_t_init": 450, "noise_gamma": 60, "noise_t_mid": 0.5}
N = 4  # responses per prompt: 2 clean + 2 noised
NUM_PROMPTS = 3


def _config(**algorithm) -> PPOConfig:
    config = PPOConfig()
    config.data.rollout_batch_size = NUM_PROMPTS
    config.data.min_pixels = 16 * 16
    config.data.max_pixels = 40 * 40
    config.worker.rollout.n = N
    config.worker.actor.global_batch_size = NUM_PROMPTS
    config.worker.actor.micro_batch_size_per_device_for_update = 1
    config.worker.actor.micro_batch_size_per_device_for_experience = 1
    config.trainer.n_gpus_per_node = 1
    for key, value in algorithm.items():
        setattr(config.algorithm, key, value)
    config.deep_post_init()
    return config


def test_config_checks():
    config = _config(rollout_image_transform="vp_diffusion", rollout_image_transform_kwargs=dict(KWARGS))
    assert config.algorithm.rollout_image_transform_kwargs["pixel_rounding"] == "floor"
    for change, match in [
        (lambda c: setattr(c.worker.rollout, "n", 3), "even"),
        (lambda c: setattr(c.worker.rollout, "interaction_mode", "agentic"), "one_shot"),
        (lambda c: setattr(c.algorithm, "adv_estimator", "remax"), "remax"),
    ]:
        config = PPOConfig()
        config.worker.rollout.n = 4
        config.algorithm.rollout_image_transform = "vp_diffusion"
        config.algorithm.rollout_image_transform_kwargs = dict(KWARGS)
        change(config)
        with pytest.raises(ValueError, match=match):
            config.deep_post_init()
    config = PPOConfig()
    config.algorithm.rollout_image_transform_kwargs = dict(KWARGS)
    with pytest.raises(ValueError, match="requires algorithm.rollout_image_transform"):
        config.deep_post_init()


def _images(seed):
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, size=(60, 50, 3), dtype=np.uint8))]


class _Rollout:
    """Samples meta_info["n"] (else rollout.n) responses per prompt; each response records its prompt row."""

    def __init__(self, n):
        self.n = n
        self.calls = []

    def generate_sequences(self, gen_batch):
        self.calls.append(gen_batch)
        n = gen_batch.meta_info.get("n", self.n)
        rows = torch.arange(len(gen_batch)).repeat_interleave(n)
        multi_modal_data = np.repeat(gen_batch.non_tensor_batch["multi_modal_data"], n)
        return DataProto(
            batch=DataProto.from_dict({"responses": rows[:, None]}).batch,
            non_tensor_batch={"multi_modal_data": multi_modal_data},
            meta_info=dict(gen_batch.meta_info),
        )


def _trainer(monkeypatch, config):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.global_step = 5
    trainer.training_steps = 10
    trainer.grounding_consistency_scorer = None
    clean = [{"images": _images(seed)} for seed in range(NUM_PROMPTS)]

    def batches():
        while True:
            yield {
                "input_ids": torch.arange(NUM_PROMPTS)[:, None],
                "attention_mask": torch.ones(NUM_PROMPTS, 1, dtype=torch.long),
                "position_ids": torch.zeros(NUM_PROMPTS, 1, dtype=torch.long),
                "prompt_id": torch.arange(NUM_PROMPTS),
                "raw_prompt_ids": np.array([[idx] for idx in range(NUM_PROMPTS)] + [None], dtype=object)[:-1],
                "multi_modal_data": np.array(clean + [None], dtype=object)[:-1],
            }

    trainer.data_iterator = batches()
    trainer.actor_rollout_ref_wg = _Rollout(config.worker.rollout.n)
    monkeypatch.setattr(ray_trainer_module.ray, "get", lambda value: value)
    return trainer, clean


def test_noised_half_is_generated_from_noise_and_trained_on_clean_images(monkeypatch):
    config = _config(rollout_image_transform="vp_diffusion", rollout_image_transform_kwargs=dict(KWARGS))
    trainer, clean = _trainer(monkeypatch, config)
    metrics = {}
    batch = trainer._make_batch_data(metrics)

    (gen_batch,) = trainer.actor_rollout_ref_wg.calls
    assert gen_batch.meta_info["n"] == N // 2 and len(gen_batch) == 2 * NUM_PROMPTS
    assert gen_batch.batch["input_ids"][:, 0].tolist() == [0, 0, 1, 1, 2, 2]  # [clean_0, noised_0, clean_1, ...]
    expected_t, expected_beta = compute_noise_schedule(5, 10, 450.0, 60.0, 1000.0, noise_t_mid=0.5)
    for prompt in range(NUM_PROMPTS):
        assert gen_batch.non_tensor_batch["multi_modal_data"][2 * prompt] is clean[prompt]
        (noised,) = gen_batch.non_tensor_batch["multi_modal_data"][2 * prompt + 1]["images"]
        resized = process_image(clean[prompt]["images"][0], config.data.min_pixels, config.data.max_pixels)
        assert isinstance(noised, ProcessedImageInput) and noised.image.size == resized.size
        assert not np.array_equal(np.asarray(noised.image), np.asarray(resized))

    # per prompt: N / 2 responses from the clean images, then N / 2 from the noised ones, all in one uid group
    assert batch.batch["responses"][:, 0].tolist() == [
        row for prompt in range(NUM_PROMPTS) for row in [2 * prompt] * 2 + [2 * prompt + 1] * 2
    ]
    assert batch.batch["prompt_id"].tolist() == [prompt for prompt in range(NUM_PROMPTS) for _ in range(N)]
    uids = batch.non_tensor_batch["uid"].reshape(NUM_PROMPTS, N)
    assert all(len(set(row)) == 1 for row in uids) and len(set(uids[:, 0])) == NUM_PROMPTS
    for idx, multi_modal_data in enumerate(batch.non_tensor_batch["multi_modal_data"]):
        assert multi_modal_data is clean[idx // N]  # the actor recomputes pixels from these
    assert batch.non_tensor_batch[ROLLOUT_IMAGE_TRANSFORMED_KEY].tolist() == [False, False, True, True] * NUM_PROMPTS
    assert "n" not in batch.meta_info
    assert metrics["rollout_transform/noise_t"] == int(expected_t)
    assert metrics["rollout_transform/noise_std"] == pytest.approx(math.sqrt(expected_beta))


def test_noise_is_seeded_by_step_and_data_seed(monkeypatch):
    def noised_pixels(step, data_seed):
        config = _config(rollout_image_transform="vp_diffusion", rollout_image_transform_kwargs=dict(KWARGS))
        config.data.seed = data_seed
        trainer, _ = _trainer(monkeypatch, config)
        trainer.global_step = step
        trainer._make_batch_data({})
        return np.asarray(trainer.actor_rollout_ref_wg.calls[0].non_tensor_batch["multi_modal_data"][1]["images"][0])

    assert np.array_equal(noised_pixels(5, 1), noised_pixels(5, 1))
    assert not np.array_equal(noised_pixels(5, 1), noised_pixels(6, 1))
    assert not np.array_equal(noised_pixels(5, 1), noised_pixels(5, 2))


def test_disabled_generation_is_unchanged(monkeypatch):
    trainer, clean = _trainer(monkeypatch, _config())
    batch = trainer._make_batch_data({})
    (gen_batch,) = trainer.actor_rollout_ref_wg.calls
    assert "n" not in gen_batch.meta_info and len(gen_batch) == NUM_PROMPTS
    assert list(gen_batch.non_tensor_batch["multi_modal_data"]) == clean
    assert ROLLOUT_IMAGE_TRANSFORMED_KEY not in batch.non_tensor_batch


def test_accuracy_metrics_by_view():
    batch = DataProto.from_dict(
        tensors={"responses": torch.zeros(4, 1)},
        non_tensors={
            ROLLOUT_IMAGE_TRANSFORMED_KEY: np.array([False, True, False, True]),
            "reward_details": np.array([{"accuracy": 1.0}, {"accuracy": 0.0}, {"accuracy": 1.0}, {"accuracy": 1.0}]),
        },
    )
    assert compute_rollout_image_transform_metrics(batch) == {
        "rollout_transform/accuracy_clean": 1.0,
        "rollout_transform/accuracy_transformed": 0.5,
    }
