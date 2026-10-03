# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from verl.models.transformers import flash_attention_utils as flash_attention_module
from verl.models.transformers.flash_attention_utils import (
    compute_cross_modal_attention_value_mean_correction,
    prepare_fa2_from_position_ids,
    use_model_level_visual_corruption,
)
from verl.protocol import DataProto
from verl.trainer import grounding_consistency as grounding_consistency_module
from verl.trainer import perception_reasoning_loss as pr_loss_module
from verl.trainer import visual_sensitivity as vs_module
from verl.trainer.config import AlgorithmConfig
from verl.trainer.core_algos import average_loss, compute_policy_loss
from verl.trainer.grounding_consistency import (
    GroundingConsistencyRewardScorer,
    _max_match_iou_sum,
    _normalize_abs_bbox_to_1000,
    _normalize_grounding_dino_query,
    _post_process_grounding_dino_object_detection,
    _resolve_grounding_dino_torch_dtype,
    _run_grounding_dino_detection_requests,
    compute_detection_reward,
    compute_group_eligibility_mask,
    parse_bbox_string,
)
from verl.trainer.perception_reasoning_data import (
    PerceptionReasoningCorruptionBuilder,
    build_perception_reasoning_loss_config,
    cgpo_flat,
    cgpo_hierarchical,
    needs_auxiliary_log_probs,
    needs_decremental_auxiliary,
    needs_full_vocab_visual_sensitivity,
    needs_hidden_state_visual_sensitivity,
    parse_response_regions,
    resolve_visual_token_ids,
    uses_model_level_visual_corruption,
)
from verl.trainer.perception_reasoning_loss import (
    build_sensitivity_advantage_shaping_context,
    compute_perception_reasoning_policy_loss,
    current_policy_entropy_requires_grad,
    has_perception_reasoning,
    needs_current_policy_entropy,
)
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.trainer.visual_sensitivity import (
    compute_full_vocab_visual_sensitivity_scores,
)
from verl.utils import torch_functional as VF
from verl.utils.dataset import ProcessedImageInput, process_image
from verl.utils.perturbations.pixel import (
    compute_noise_schedule,
    gaussian_noise_image,
    pixelate_image,
    vp_diffusion_noise,
    vp_diffusion_noise_fixed_t,
)
from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import FSDPWorker


requires_flash_attn = pytest.mark.skipif(
    importlib.util.find_spec("flash_attn") is None, reason="exercises the flash-attn code path"
)


class _DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=False):
        del add_special_tokens
        prompt = text if isinstance(text, str) else text[0]
        length = max(len(prompt.split()), 1)
        input_ids = torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(idx, idx + 1) for idx in range(length)]
        return encoded

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        del tokenize
        chunks = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                chunks.append(content)
                continue
            for item in content:
                if item.get("type") == "text":
                    chunks.append(item.get("text", ""))
                elif item.get("type") == "image":
                    chunks.append("<image>")
        if add_generation_prompt:
            chunks.append("<assistant>")
        return " ".join(chunks)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, max(len(text.split()), 1) + 1))

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del skip_special_tokens
        del clean_up_tokenization_spaces
        return " ".join(str(token_id) for token_id in token_ids if token_id != self.pad_token_id)


class _DummyProcessor:
    def __init__(self):
        self.image_processor = type("DummyImageProcessor", (), {"patch_size": 2})()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return _DummyTokenizer().apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize
        )

    def __call__(self, images=None, videos=None, text=None, add_special_tokens=False, return_tensors="pt"):
        del images, videos, add_special_tokens, return_tensors
        prompt = text[0]
        length = max(len(prompt.split()), 1)
        input_ids = torch.arange(1, length + 1, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyQwenTokenizer(_DummyTokenizer):
    _SPECIAL_TOKEN_IDS = {
        "<|image_pad|>": 151655,
        "<|video_pad|>": 151656,
        "<|vision_start|>": 151652,
    }

    def convert_tokens_to_ids(self, token):
        return self._SPECIAL_TOKEN_IDS[token]


class _DummyQwenProcessor:
    def __init__(self):
        self.image_processor = type("Qwen2VLImageProcessor", (), {"merge_size": 2})()
        self.tokenizer = _DummyQwenTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
        )


class _CharTokenizer:
    pad_token_id = 0
    is_fast = True

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        del tokenize
        chunks = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                chunks.append(content)
                continue
            for item in content:
                if item.get("type") == "image":
                    chunks.append("<image>")
                elif item.get("type") == "text":
                    chunks.append(item.get("text", ""))
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "".join(chunks)

    def __call__(self, text, add_special_tokens=False, return_tensors=None, return_offsets_mapping=False):
        del add_special_tokens
        if isinstance(text, list):
            assert len(text) == 1
            text = text[0]
        ids = [ord(char) + 1 for char in text]
        payload = {"input_ids": ids}
        if return_tensors == "pt":
            payload["input_ids"] = torch.tensor([ids], dtype=torch.long)
            payload["attention_mask"] = torch.ones_like(payload["input_ids"])
        if return_offsets_mapping:
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(ids))]
        return payload

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) + 1 for char in text]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id - 1) for token_id in token_ids if token_id != self.pad_token_id)


class _CharProcessor:
    def __init__(self):
        self.image_processor = type("DummyImageProcessor", (), {"patch_size": 2})()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return _CharTokenizer().apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize
        )

    def __call__(self, images=None, videos=None, text=None, add_special_tokens=False, return_tensors="pt"):
        del images, videos, add_special_tokens
        return _CharTokenizer()(text, return_tensors=return_tensors)


def _make_raw_prompt():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image"},
            ],
        }
    ]


def _make_batch(
    image_a,
    image_b=None,
    *,
    uid_a="sample-a",
    uid_b="sample-b",
    response_a=(21, 22),
    response_b=(23, 24),
    cache_id_a=None,
    cache_id_b=None,
):
    if image_b is None:
        image_b = image_a
    raw_prompt = _make_raw_prompt()
    non_tensors = {
        "raw_prompt": np.array([raw_prompt, raw_prompt], dtype=object),
        "uid": np.array([uid_a, uid_b], dtype=object),
        "multi_modal_data": np.array([{"images": [image_a]}, {"images": [image_b]}], dtype=object),
    }
    if cache_id_a is not None or cache_id_b is not None:
        non_tensors["multi_modal_cache_id"] = np.array(
            [
                uid_a if cache_id_a is None else cache_id_a,
                uid_b if cache_id_b is None else cache_id_b,
            ],
            dtype=object,
        )
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[11, 12, 13, 14, 21, 22], [11, 12, 13, 14, 23, 24]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=torch.long),
            "responses": torch.tensor([list(response_a), list(response_b)], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
        },
        non_tensors=non_tensors,
    )


def _make_actor_config() -> ActorConfig:
    config = ActorConfig(
        global_batch_size=2,
        micro_batch_size_per_device_for_update=2,
        micro_batch_size_per_device_for_experience=2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        clip_ratio_dual=3.0,
        loss_avg_mode="token",
        loss_type="default",
        ppo_epochs=1,
        padding_free=False,
        dynamic_batching=False,
    )
    config.global_batch_size_per_device = config.global_batch_size
    return config


def _make_common_tensors():
    return {
        "log_prob": torch.tensor(
            [[-0.9, -0.4, -0.7, -0.2], [-0.8, -0.3, -0.5, -0.6]],
            dtype=torch.float32,
        ),
        "old_log_prob": torch.tensor(
            [[-1.0, -0.5, -0.6, -0.3], [-0.7, -0.1, -0.4, -0.7]],
            dtype=torch.float32,
        ),
        "advantages": torch.tensor(
            [[0.6, -0.3, 0.2, 0.9], [0.4, -0.5, 0.8, -0.2]],
            dtype=torch.float32,
        ),
        "response_mask": torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        ),
        "entropy": torch.tensor(
            [[0.5, 0.4, 0.3, 0.2], [0.7, 0.6, 0.5, 0.4]],
            dtype=torch.float32,
        ),
        "decremental_old_log_probs": torch.tensor(
            [[-0.6, -0.1, -0.4, -0.1], [-0.4, 0.1, -0.2, -0.3]],
            dtype=torch.float32,
        ),
        "decremental_entropies": torch.tensor(
            [[0.9, 0.7, 0.5, 0.2], [0.8, 0.6, 0.4, 0.3]],
            dtype=torch.float32,
        ),
        "incremental_old_log_probs": torch.tensor(
            [[-1.2, -0.7, -1.0, -0.4], [-1.1, -0.6, -0.8, -0.9]],
            dtype=torch.float32,
        ),
        "incremental_entropies": torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.5, 0.4, 0.3, 0.2]],
            dtype=torch.float32,
        ),
    }


def test_average_loss_none_mode_returns_unreduced_tensor():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

    torch.testing.assert_close(average_loss(values, mask, mode="none"), values)


def test_compute_policy_loss_none_mode_returns_tokenwise_pg_terms():
    log_probs = torch.tensor([[-0.2, -0.4], [-0.1, -0.3]], dtype=torch.float32)
    old_log_probs = torch.tensor([[-0.3, -0.5], [-0.2, -0.4]], dtype=torch.float32)
    advantages = torch.tensor([[0.5, -0.2], [0.3, 0.1]], dtype=torch.float32)
    response_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    token_pg_loss, token_metrics = compute_policy_loss(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        clip_ratio_dual=3.0,
        tau_positive=1.0,
        tau_negative=1.05,
        loss_type="default",
        loss_avg_mode="none",
    )
    reduced_pg_loss, reduced_metrics = compute_policy_loss(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.3,
        clip_ratio_dual=3.0,
        tau_positive=1.0,
        tau_negative=1.05,
        loss_type="default",
        loss_avg_mode="token",
    )

    assert token_pg_loss.shape == advantages.shape
    torch.testing.assert_close(
        average_loss(token_pg_loss, response_mask, mode="token"),
        reduced_pg_loss,
    )
    assert token_metrics["ppo_kl"] == pytest.approx(reduced_metrics["ppo_kl"])
    assert token_metrics["entropy_loss"] == pytest.approx(reduced_metrics["entropy_loss"])


def test_algorithm_config_parses_perception_kwargs():
    config = AlgorithmConfig(
        corrupt_image="gaussian_noise",
        corrupt_image_kwargs='{"std": 2.0}',
        top_entropy_quantile=0.5,
        top_perception_quantile=0.2,
        visual_sensitivity_metric="vepo",
        visual_sensitivity_jsd_weight=0.7,
        visual_sensitivity_entropy_gate="normal_entropy",
        normalize_pg_loss_by_selected_tokens=True,
    )
    config.post_init()

    assert config.corrupt_image_kwargs == {"std": 2.0}
    assert config.top_entropy_quantile == 0.5
    loss_config = build_perception_reasoning_loss_config(config)
    assert loss_config["normalize_pg_loss_by_selected_tokens"] is True
    assert loss_config["visual_sensitivity_metric"] == "vepo"
    assert loss_config["visual_sensitivity_jsd_weight"] == 0.7
    assert needs_full_vocab_visual_sensitivity(config)
    assert not needs_auxiliary_log_probs(config)


def test_algorithm_config_parses_model_level_visual_corruption():
    config = AlgorithmConfig(
        corrupt_image="cross_modal_attention_value_mean",
        corrupt_image_kwargs='{"saliency_std_multiplier": 2.0}',
        visual_sensitivity_loss_coef=0.02,
        visual_sensitivity_reference="current",
        visual_sensitivity_metric="sampled_low_var_kl",
    )
    config.post_init()

    assert config.corrupt_image_kwargs == {"saliency_std_multiplier": 2.0}
    assert uses_model_level_visual_corruption(config)
    assert needs_decremental_auxiliary(config)
    assert needs_auxiliary_log_probs(config)


@pytest.mark.parametrize("position", ["prompt", "response"])
def test_algorithm_config_parses_pixelation_for_both_corruption_positions(position):
    config = AlgorithmConfig(
        corrupt_image="pixelation",
        corrupt_image_kwargs='{"ratio": 0.1}',
        corrupt_image_position=position,
    )
    config.post_init()

    assert config.corrupt_image_kwargs == {"ratio": 0.1}
    assert config.corrupt_image_position == position
    assert not uses_model_level_visual_corruption(config)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"unknown": 1}, "unsupported corrupt_image_kwargs"),
        ({"ratio": True}, "number in"),
        ({"ratio": 0.0}, "number in"),
        ({"ratio": 1.01}, "number in"),
        ({"ratio": "0.1"}, "number in"),
    ],
)
def test_algorithm_config_rejects_invalid_pixelation_kwargs(kwargs, error):
    config = AlgorithmConfig(corrupt_image="pixelation", corrupt_image_kwargs=kwargs)

    with pytest.raises(ValueError, match=error):
        config.post_init()


@pytest.mark.parametrize(
    ("kwargs", "position", "metric", "error"),
    [
        ({"unknown": 1}, "prompt", "sampled_low_var_kl", "unsupported corrupt_image_kwargs"),
        ({"saliency_std_multiplier": -1}, "prompt", "sampled_low_var_kl", "non-negative number"),
        (None, "response", "sampled_low_var_kl", "only supports corrupt_image_position='prompt'"),
        (None, "prompt", "full_vocab_kl", "does not currently support full-vocab"),
    ],
)
def test_algorithm_config_rejects_invalid_model_level_visual_corruption(kwargs, position, metric, error):
    config = AlgorithmConfig(
        corrupt_image="cross_modal_attention_value_mean",
        corrupt_image_kwargs=kwargs,
        corrupt_image_position=position,
        visual_sensitivity_metric=metric,
    )
    with pytest.raises(ValueError, match=error):
        config.post_init()


def test_model_level_visual_corruption_accepts_hidden_metric_with_sampled_loss_fallback():
    config = AlgorithmConfig(
        corrupt_image="cross_modal_attention_value_mean",
        visual_sensitivity_metric="hidden_state_similarity",
        visual_sensitivity_loss_coef=0.02,
    )
    config.post_init()

    assert needs_auxiliary_log_probs(config)
    assert vs_module.resolve_sampled_sensitivity_metric(config.visual_sensitivity_metric) == "sampled_low_var_kl"


def test_trainer_routes_model_level_visual_corruption_without_media_builder():
    algorithm = AlgorithmConfig(
        corrupt_image="cross_modal_attention_value_mean",
        corrupt_image_kwargs={"saliency_std_multiplier": 2.0},
        visual_sensitivity_loss_coef=0.02,
    )
    algorithm.post_init()
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(algorithm=algorithm)
    trainer.perception_reasoning_corruption_builder = None
    trainer.actor_rollout_ref_wg = SimpleNamespace()
    trainer._latest_vision_metrics = {}
    trainer.global_step = 1
    trainer.training_steps = 1
    captured = {}

    def compute_aux_log_probs(auxiliary_batch):
        captured["meta_info"] = dict(auxiliary_batch.meta_info)
        return DataProto.from_dict(
            tensors={"decremental_old_log_probs": torch.zeros((2, 2))},
        )

    trainer.actor_rollout_ref_wg.compute_aux_log_probs = compute_aux_log_probs
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0).expand(2, -1),
            "responses": torch.ones((2, 2), dtype=torch.long),
        },
        meta_info={"temperature": 1.0},
    )

    output = trainer._compute_auxiliary_old_log_probs(batch)

    assert output.batch["decremental_old_log_probs"].shape == (2, 2)
    assert captured["meta_info"]["model_level_visual_corruption"] == {
        "name": "cross_modal_attention_value_mean",
        "kwargs": {"saliency_std_multiplier": 2.0},
    }
    assert captured["meta_info"]["aux_log_probs_output_key"] == "decremental_old_log_probs"
    assert "model_level_visual_corruption" not in batch.meta_info


def test_algorithm_config_parses_full_vocab_kl_and_hellinger_metrics():
    for metric in ("full_vocab_kl", "full_vocab_hellinger"):
        config = AlgorithmConfig(
            corrupt_image="gaussian_noise",
            corrupt_image_kwargs={"std": 0.5},
            top_perception_quantile=0.2,
            visual_sensitivity_metric=metric,
        )
        config.post_init()

        loss_config = build_perception_reasoning_loss_config(config)
        assert loss_config["visual_sensitivity_metric"] == metric
        assert needs_full_vocab_visual_sensitivity(config)


def test_algorithm_config_validates_grounding_consistency_detector():
    config = AlgorithmConfig()
    config.post_init()
    assert config.grounding_consistency_detector == "self"
    assert config.grounding_dino_device == "worker"
    assert config.grounding_dino_batch_size == 4

    config = AlgorithmConfig(grounding_consistency_detector="grounding-dino")
    config.post_init()
    assert config.grounding_consistency_detector == "grounding-dino"

    config = AlgorithmConfig(grounding_consistency_detector="grounding-dnio")
    with pytest.raises(ValueError, match="grounding_consistency_detector"):
        config.post_init()

    config = AlgorithmConfig(grounding_dino_batch_size=0)
    with pytest.raises(ValueError, match="grounding_dino_batch_size"):
        config.post_init()


def test_grounding_consistency_scorer_default_device_matches_config_default():
    scorer = GroundingConsistencyRewardScorer(
        tokenizer=_CharTokenizer(),
        processor=_CharProcessor(),
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
    )

    assert scorer.grounding_dino_device == AlgorithmConfig.grounding_dino_device


def test_algorithm_config_parses_pgpo_advantage_scaling():
    config = AlgorithmConfig(
        corrupt_image="mask_visual_attention",
        advantage_scaling_method="pgpo",
        pgpo_token_scaling_threshold=0.35,
        pgpo_token_scaling_boost=1.5,
    )
    config.post_init()

    loss_config = build_perception_reasoning_loss_config(config)

    assert loss_config["advantage_scaling_method"] == "pgpo"
    assert loss_config["pgpo_token_scaling_threshold"] == 0.35
    assert loss_config["pgpo_token_scaling_boost"] == 1.5
    assert needs_decremental_auxiliary(config)
    assert needs_auxiliary_log_probs(config)


def test_algorithm_config_parses_pepo_without_decremental_auxiliary():
    config = AlgorithmConfig(
        visual_sensitivity_metric="hidden_state_similarity",
        visual_sensitivity_hidden_metric="cosine",
        visual_token="auto",
        advantage_scaling_method="pepo",
        advantage_scaling_schedule="linear",
        pepo_gate_alpha=0.05,
        pepo_gate_temperature=1.8,
    )
    config.post_init()

    loss_config = build_perception_reasoning_loss_config(config)

    assert loss_config["advantage_scaling_method"] == "pepo"
    assert loss_config["visual_sensitivity_metric"] == "hidden_state_similarity"
    assert loss_config["visual_sensitivity_hidden_metric"] == "cosine"
    assert loss_config["advantage_scaling_schedule"] == "linear"
    assert loss_config["pepo_gate_alpha"] == 0.05
    assert loss_config["pepo_gate_temperature"] == 1.8
    assert needs_hidden_state_visual_sensitivity(config)
    assert not needs_decremental_auxiliary(config)
    assert not needs_auxiliary_log_probs(config)


def test_pepo_requires_hidden_state_visual_sensitivity_metric():
    config = AlgorithmConfig(advantage_scaling_method="pepo")
    with pytest.raises(ValueError, match="requires visual_sensitivity_metric='hidden_state_similarity'"):
        config.post_init()


def test_hidden_state_visual_sensitivity_rejects_response_level_scaling():
    config = AlgorithmConfig(
        visual_sensitivity_metric="hidden_state_similarity",
        advantage_scaling_method="vppo",
    )
    with pytest.raises(ValueError, match="not 'vppo' or 'cgpo'"):
        config.post_init()


def test_resolve_visual_token_ids_auto_and_explicit_fail_closed():
    tokenizer = _DummyQwenTokenizer()

    assert resolve_visual_token_ids(tokenizer, "auto") == {151655, 151656}
    assert resolve_visual_token_ids(tokenizer, "<|image_pad|>") == {151655}
    assert resolve_visual_token_ids(_DummyTokenizer(), "<missing_image>") == set()


def test_algorithm_config_keeps_legacy_response_advantage_scaling_alias():
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        response_advantage_scaling_method="vppo",
    )
    config.post_init()

    assert config.advantage_scaling_method == "vppo"
    assert build_perception_reasoning_loss_config(config)["advantage_scaling_method"] == "vppo"


def test_algorithm_config_requires_corruption_for_perception_losses():
    config = AlgorithmConfig(visual_sensitivity_loss_coef=0.1)
    with pytest.raises(ValueError):
        config.post_init()


def test_algorithm_config_parses_vepo_metric_without_fusion_knob():
    config = AlgorithmConfig(
        corrupt_image="gaussian_noise",
        top_perception_quantile=0.2,
        visual_sensitivity_metric="vepo",
        visual_sensitivity_jsd_weight=0.7,
    )
    config.post_init()
    assert needs_full_vocab_visual_sensitivity(config)
    assert build_perception_reasoning_loss_config(config)["visual_sensitivity_metric"] == "vepo"


def test_gaussian_noise_image_is_seeded_and_changes_pixels():
    image = Image.fromarray(np.full((4, 4, 3), 127, dtype=np.uint8))
    noisy_a = np.asarray(gaussian_noise_image(image, std=0.5, seed=7))
    noisy_b = np.asarray(gaussian_noise_image(image, std=0.5, seed=7))

    np.testing.assert_array_equal(noisy_a, noisy_b)
    assert not np.array_equal(noisy_a, np.asarray(image))


def test_pixelate_image_uses_paper_resize_order_and_preserves_input(monkeypatch):
    pixels = np.arange(23 * 37 * 3, dtype=np.uint8).reshape(23, 37, 3)
    image = Image.fromarray(pixels)
    original_resize = Image.Image.resize
    resize_calls = []

    def recording_resize(self, size, resample=None, box=None, reducing_gap=None):
        resize_calls.append((size, resample))
        return original_resize(self, size, resample=resample, box=box, reducing_gap=reducing_gap)

    monkeypatch.setattr(Image.Image, "resize", recording_resize)

    pixelated = pixelate_image(image)

    assert resize_calls == [
        ((3, 2), Image.Resampling.BILINEAR),
        ((37, 23), Image.Resampling.NEAREST),
    ]
    assert pixelated.size == image.size
    assert pixelated.mode == image.mode
    np.testing.assert_array_equal(np.asarray(image), pixels)
    assert not np.array_equal(np.asarray(pixelated), pixels)


def test_pixelate_image_clamps_small_downsampled_dimensions_and_ratio_one_is_identity():
    image = Image.fromarray(np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3))

    pixelated = pixelate_image(image, ratio=0.1)
    identity = pixelate_image(image, ratio=1.0)

    assert pixelated.size == image.size
    assert np.unique(np.asarray(pixelated).reshape(-1, 3), axis=0).shape[0] == 1
    np.testing.assert_array_equal(np.asarray(identity), np.asarray(image))


@pytest.mark.parametrize("ratio", [False, 0.0, -0.1, 1.1, "0.1"])
def test_pixelate_image_rejects_invalid_ratio(ratio):
    image = Image.new("RGB", (4, 4))

    with pytest.raises(ValueError, match="ratio must be a number in"):
        pixelate_image(image, ratio=ratio)


def test_vp_diffusion_shared_fixed_t_matches_beta_path_bitwise():
    image = Image.fromarray(np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3))

    fixed_image, params = vp_diffusion_noise_fixed_t(image, noise_t=500.0, noise_t_max=1000.0, seed=11)
    direct_image = vp_diffusion_noise(image, beta=params.noise_beta, seed=11)

    np.testing.assert_array_equal(np.asarray(fixed_image), np.asarray(direct_image))
    assert params.noise_t == 500.0
    # NoisyRollout's schedule (verl/utils/image_aug.py), which DVRP follows
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1 - betas, dim=0)
    assert params.noise_beta == pytest.approx(1 - alphas_prod[500].item(), abs=1e-7)


def test_vp_diffusion_noise_follows_the_noisyrollout_step():
    """x_t = sqrt(alpha_bar_t) x + sqrt(1 - alpha_bar_t) eps: at t=250 the noise std is about 0.147."""
    image = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
    noisy, params = vp_diffusion_noise_fixed_t(image, noise_t=250.0, noise_t_max=1000.0, seed=3)
    pixels = np.asarray(noisy, dtype=np.float64) / 255.0
    assert pixels.std() == pytest.approx(math.sqrt(params.noise_beta), rel=0.02)
    assert params.noise_beta**0.5 == pytest.approx(0.147, abs=1e-3)
    assert pixels.mean() == pytest.approx(math.sqrt(1 - params.noise_beta) * 128 / 255, abs=2e-3)
    with pytest.raises(ValueError, match="noise_t_max"):
        vp_diffusion_noise_fixed_t(image, noise_t=250.0, noise_t_max=2000.0)


def test_shared_noise_schedule_keeps_legacy_tuple_unpacking():
    noise_t, noise_beta = compute_noise_schedule(
        global_step=3,
        total_training_steps=10,
        noise_t_init=500.0,
        noise_gamma=10.0,
        noise_t_max=1000.0,
    )

    assert isinstance(noise_t, float)
    assert isinstance(noise_beta, float)


def test_builder_prompt_random_patch_shares_corruption_across_repeated_responses():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        corrupt_image_position="prompt",
        corrupt_image_kwargs={"patch_size": 2, "black_prob": 0.5},
    )

    result = builder.build_batch(batch, config, global_step=3)
    first_pixels = np.asarray(result.batch.non_tensor_batch["multi_modal_data"][0]["images"][0])
    second_pixels = np.asarray(result.batch.non_tensor_batch["multi_modal_data"][1]["images"][0])

    assert isinstance(result.batch.non_tensor_batch["multi_modal_data"][0]["images"][0], ProcessedImageInput)
    assert (
        result.batch.non_tensor_batch["multi_modal_cache_id"][0]
        == result.batch.non_tensor_batch["multi_modal_cache_id"][1]
    )
    assert np.array_equal(first_pixels, second_pixels)
    assert len(builder._prompt_cache) == 1


@pytest.mark.parametrize("position", ["prompt", "response"])
def test_builder_pixelation_transforms_all_images_without_changing_sequence_tensors(position):
    image_a = Image.fromarray(np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3))
    image_b = Image.fromarray(np.flip(np.asarray(image_a), axis=1).copy())
    batch = _make_batch(image_a)[:1]
    batch.non_tensor_batch["multi_modal_data"][0] = {"images": [image_a, image_b]}
    original_tensors = {key: batch.batch[key].clone() for key in ("input_ids", "attention_mask", "position_ids")}
    original_pixels = [np.asarray(image).copy() for image in (image_a, image_b)]
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="pixelation",
        corrupt_image_position=position,
        corrupt_image_kwargs={"ratio": 0.25},
    )

    result = builder.build_batch(batch, config, global_step=3)
    transformed_images = result.batch.non_tensor_batch["multi_modal_data"][0]["images"]

    assert len(transformed_images) == 2
    assert all(isinstance(image, ProcessedImageInput) for image in transformed_images)
    for transformed, original, original_array in zip(transformed_images, (image_a, image_b), original_pixels):
        assert transformed.image.size == original.size
        assert transformed.image.mode == original.mode
        assert not np.array_equal(np.asarray(transformed), original_array)
        np.testing.assert_array_equal(np.asarray(original), original_array)
    for key, original in original_tensors.items():
        torch.testing.assert_close(result.batch.batch[key], original)
    assert len(builder._prompt_cache) == (1 if position == "prompt" else 0)


def test_builder_prompt_pixelation_shares_cached_view_across_repeated_rollouts():
    image = Image.fromarray(np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3))
    batch = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="pixelation",
        corrupt_image_position="prompt",
    )

    result = builder.build_batch(batch, config, global_step=3)
    first_pixels = np.asarray(result.batch.non_tensor_batch["multi_modal_data"][0]["images"][0])
    second_pixels = np.asarray(result.batch.non_tensor_batch["multi_modal_data"][1]["images"][0])

    np.testing.assert_array_equal(first_pixels, second_pixels)
    assert (
        result.batch.non_tensor_batch["multi_modal_cache_id"][0]
        == result.batch.non_tensor_batch["multi_modal_cache_id"][1]
    )
    assert len(builder._prompt_cache) == 1


def test_builder_prompt_cache_discards_entries_from_previous_step():
    image_a = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    image_b = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    batch = _make_batch(image_a, image_b, uid_a="rollout-a", uid_b="rollout-b")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        corrupt_image_position="prompt",
        corrupt_image_kwargs={"patch_size": 2, "black_prob": 0.5},
    )

    builder.build_batch(batch, config, global_step=3)
    step_three_keys = set(builder._prompt_cache)
    assert len(step_three_keys) == 2
    assert all(":step:3:" in descriptor for _, descriptor in step_three_keys)

    builder.build_batch(batch[:1], config, global_step=4)
    step_four_keys = set(builder._prompt_cache)

    assert builder._prompt_cache_step == 4
    assert len(step_four_keys) == 1
    assert all(":step:4:" in descriptor for _, descriptor in step_four_keys)
    assert step_three_keys.isdisjoint(step_four_keys)


def test_builder_prompt_cache_keeps_all_branches_from_current_dvrp_step():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        corrupt_image_position="prompt",
        corrupt_image_kwargs={"patch_size": 2, "black_prob": 0.5},
        incremental_image_transform="vp_diffusion",
    )

    builder.build_dvrp_batches(
        batch=batch,
        config=config,
        global_step=3,
        total_training_steps=10,
    )
    descriptors = {descriptor for _, descriptor in builder._prompt_cache}

    assert builder._prompt_cache_step == 3
    assert len(descriptors) == 2
    assert any(descriptor.startswith("random_patch:step:3:") for descriptor in descriptors)
    assert any(descriptor.startswith("vp_diffusion:step:3:") for descriptor in descriptors)


def test_processed_auxiliary_images_skip_second_resize():
    image = Image.fromarray(np.full((277, 67, 3), 255, dtype=np.uint8))
    processed = process_image(image, min_pixels=200704, max_pixels=1003520)
    reprocessed = process_image(processed, min_pixels=200704, max_pixels=1003520)
    protected = process_image(ProcessedImageInput(processed), min_pixels=200704, max_pixels=1003520)

    assert processed.size == (220, 910)
    assert reprocessed.size == (220, 911)
    assert protected.size == processed.size


def test_process_image_accepts_path_only_image_dict(tmp_path):
    image = Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8))
    image_path = tmp_path / "sample.png"
    image.save(image_path)

    processed = process_image({"path": str(image_path), "bytes": None}, min_pixels=None, max_pixels=None)

    assert processed.size == image.size
    assert processed.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(processed), np.asarray(image))


def test_builder_response_mode_assigns_unique_aux_cache_ids_per_rollout():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        corrupt_image_position="response",
        corrupt_image_kwargs={"patch_size": 2, "black_prob": 0.5},
    )

    result = builder.build_batch(batch, config, global_step=3)

    assert result.batch.non_tensor_batch["uid"][0] == result.batch.non_tensor_batch["uid"][1]
    assert (
        result.batch.non_tensor_batch["multi_modal_cache_id"][0]
        != result.batch.non_tensor_batch["multi_modal_cache_id"][1]
    )


def test_dvrp_response_mode_splits_cache_ids_for_both_aux_branches():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        corrupt_image_position="response",
        corrupt_image_kwargs={"patch_size": 2, "black_prob": 0.5},
        incremental_image_transform="vp_diffusion",
    )

    decremental_result, incremental_result = builder.build_dvrp_batches(
        batch=batch,
        config=config,
        global_step=3,
        total_training_steps=10,
    )

    assert (
        decremental_result.batch.non_tensor_batch["multi_modal_cache_id"][0]
        != decremental_result.batch.non_tensor_batch["multi_modal_cache_id"][1]
    )
    assert (
        incremental_result.batch.non_tensor_batch["multi_modal_cache_id"][0]
        != incremental_result.batch.non_tensor_batch["multi_modal_cache_id"][1]
    )
    assert "noise_t" in incremental_result.stats
    assert "noise_beta" in incremental_result.stats


def test_build_incremental_batch_matches_dvrp_incremental_branch():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        corrupt_image_position="prompt",
        corrupt_image_kwargs={"patch_size": 2, "black_prob": 0.5},
        incremental_image_transform="vp_diffusion",
    )

    _, dvrp_incremental = builder.build_dvrp_batches(
        batch=batch,
        config=config,
        global_step=3,
        total_training_steps=10,
    )
    incremental_only = builder.build_incremental_batch(
        batch=batch,
        config=config,
        global_step=3,
        total_training_steps=10,
    )

    torch.testing.assert_close(
        incremental_only.batch.batch["input_ids"],
        dvrp_incremental.batch.batch["input_ids"],
    )
    assert (
        incremental_only.batch.non_tensor_batch["multi_modal_cache_id"].tolist()
        == dvrp_incremental.batch.non_tensor_batch["multi_modal_cache_id"].tolist()
    )
    assert incremental_only.stats == dvrp_incremental.stats


def test_vp_diffusion_prompt_mode_still_shares_cache_ids():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        corrupt_image_position="prompt",
        corrupt_image_kwargs={"patch_size": 2, "black_prob": 0.5},
        incremental_image_transform="vp_diffusion",
    )

    _, incremental_result = builder.build_dvrp_batches(
        batch=batch,
        config=config,
        global_step=3,
        total_training_steps=10,
    )

    assert (
        incremental_result.batch.non_tensor_batch["multi_modal_cache_id"][0]
        == incremental_result.batch.non_tensor_batch["multi_modal_cache_id"][1]
    )


def test_builder_no_image_rebuilds_prompt_and_drops_modal_data():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image)
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(corrupt_image="no_image")

    result = builder.build_batch(batch[:1], config, global_step=0)

    assert result.batch.batch["input_ids"].shape == (1, 6)
    assert result.batch.batch["position_ids"].shape == (1, 6)
    assert torch.equal(result.batch.batch["input_ids"][0, -2:], result.batch.batch["responses"][0])
    assert result.batch.non_tensor_batch["multi_modal_data"][0] is None


def test_builder_mask_visual_attention_keeps_positions_and_masks_visual_payload():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image)
    batch.batch["input_ids"] = torch.tensor(
        [
            [11, 151652, 151655, 14, 21, 22],
            [11, 151652, 151655, 14, 23, 24],
        ],
        dtype=torch.long,
    )
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyQwenTokenizer(),
        processor=_DummyQwenProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(corrupt_image="mask_visual_attention")

    result = builder.build_batch(batch, config, global_step=0)

    torch.testing.assert_close(result.batch.batch["input_ids"], batch.batch["input_ids"])
    torch.testing.assert_close(result.batch.batch["position_ids"], batch.batch["position_ids"])
    assert result.batch.batch["attention_mask"].tolist() == [[1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1]]
    assert result.batch.non_tensor_batch["multi_modal_data"][0] is None
    assert result.stats["algo/vision/masked_visual_tokens"] == 1.0


def test_corruption_group_key_distinguishes_same_text_different_images():
    black_image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    white_image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )

    key_black = builder._resolve_corruption_group_key(_make_raw_prompt(), {"images": [black_image]})
    key_white = builder._resolve_corruption_group_key(_make_raw_prompt(), {"images": [white_image]})

    assert key_black != key_white


class _RecordingImageProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, images=None, videos=None, return_tensors="pt"):
        assert videos is None
        assert return_tensors == "pt"
        values = [int(np.asarray(image).sum()) for image in images]
        self.calls.append(values)
        return {
            "pixel_values": torch.tensor([[values[0]]], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _NextTokenActorModule(nn.Module):
    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        del attention_mask, position_ids, kwargs
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            fill_value=-100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_tokens = torch.roll(input_ids, shifts=-1, dims=1)
        logits.scatter_(2, next_tokens.unsqueeze(-1), 0.0)
        return SimpleNamespace(logits=logits)


class _RecordingVisualActorModule(_NextTokenActorModule):
    def __init__(self, model_type):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            vision_start_token_id=10,
            vision_end_token_id=11,
        )
        self.seen_input_shape = None
        self.seen_attention_mask = None

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        self.seen_input_shape = tuple(input_ids.shape)
        self.seen_attention_mask = attention_mask
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


def _cpu_log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch_dim = logits.shape[:-1]
    vocab_dim = logits.shape[-1]
    losses = -F.cross_entropy(logits.reshape(-1, vocab_dim).float(), labels.reshape(-1), reduction="none")
    return losses.view(*batch_dim)


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen3_vl"])
def test_model_level_visual_corruption_uses_sample_isolated_padded_forward(model_type):
    actor_config = _make_actor_config()
    actor_config.padding_free = True
    actor_config.use_torch_compile = False
    actor_module = _RecordingVisualActorModule(model_type)
    actor = DataParallelPPOActor(config=actor_config, actor_module=actor_module)
    actor.rank = 1
    actor.log_probs_from_logits = _cpu_log_probs_from_logits
    input_ids = torch.tensor(
        [
            [0, 0, 1, 10, 20, 21, 11, 30, 31, 32],
            [0, 2, 10, 22, 23, 11, 33, 34, 35, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    text_position_ids = torch.tensor(
        [
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    micro_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": text_position_ids.unsqueeze(1).expand(-1, 4, -1),
        "responses": input_ids[:, -2:],
    }
    corruption = {
        "name": "cross_modal_attention_value_mean",
        "kwargs": {"saliency_std_multiplier": 2.0},
    }

    batched_log_probs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        model_level_visual_corruption=corruption,
    )
    batched_input_shape = actor_module.seen_input_shape
    batched_attention_mask = actor_module.seen_attention_mask.clone()
    single_log_probs = torch.cat(
        [
            actor._forward_micro_batch(
                {key: value[row_idx : row_idx + 1] for key, value in micro_batch.items()},
                temperature=1.0,
                model_level_visual_corruption=corruption,
            )
            for row_idx in range(input_ids.size(0))
        ],
        dim=0,
    )

    assert batched_log_probs.shape == (2, 2)
    assert batched_input_shape == input_ids.shape
    assert torch.equal(batched_attention_mask, attention_mask)
    torch.testing.assert_close(batched_log_probs, single_log_probs, atol=0.0, rtol=0.0)


def test_model_level_visual_corruption_caps_dynamic_micro_batch_size():
    actor_config = _make_actor_config()
    actor_config.dynamic_batching = True
    actor_config.micro_batch_size_per_device_for_experience = 2
    actor = DataParallelPPOActor(config=actor_config, actor_module=_NextTokenActorModule())
    actor.rank = 1
    seen_batch_sizes = []

    def record_forward(micro_batch, **kwargs):
        seen_batch_sizes.append(micro_batch["input_ids"].size(0))
        return torch.zeros(
            (micro_batch["input_ids"].size(0), micro_batch["responses"].size(1)),
            dtype=torch.float32,
        )

    actor._forward_micro_batch = record_forward
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(24, dtype=torch.long).view(6, 4),
            "attention_mask": torch.tensor([[1, 0, 0, 0]] * 6, dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long).unsqueeze(0).expand(6, -1),
            "responses": torch.arange(6, dtype=torch.long).unsqueeze(-1),
        },
        meta_info={
            "temperature": 1.0,
            "model_level_visual_corruption": {
                "name": "cross_modal_attention_value_mean",
                "kwargs": {"saliency_std_multiplier": 2.0},
            },
        },
    )

    output = actor.compute_log_prob(data)

    assert output.shape == (6, 1)
    assert seen_batch_sizes == [2, 2, 2]


def test_regular_forward_keeps_effective_token_dynamic_batching():
    actor_config = _make_actor_config()
    actor_config.dynamic_batching = True
    actor_config.micro_batch_size_per_device_for_experience = 2
    actor = DataParallelPPOActor(config=actor_config, actor_module=_NextTokenActorModule())
    actor.rank = 1
    seen_batch_sizes = []

    def record_forward(micro_batch, **kwargs):
        seen_batch_sizes.append(micro_batch["input_ids"].size(0))
        return torch.zeros(
            (micro_batch["input_ids"].size(0), micro_batch["responses"].size(1)),
            dtype=torch.float32,
        )

    actor._forward_micro_batch = record_forward
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(24, dtype=torch.long).view(6, 4),
            "attention_mask": torch.tensor([[1, 0, 0, 0]] * 6, dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long).unsqueeze(0).expand(6, -1),
            "responses": torch.arange(6, dtype=torch.long).unsqueeze(-1),
        },
        meta_info={"temperature": 1.0},
    )

    output = actor.compute_log_prob(data)

    assert output.shape == (6, 1)
    assert seen_batch_sizes == [6]


@requires_flash_attn
def test_padded_qwen_text_positions_use_masked_flash_path(monkeypatch):
    monkeypatch.setattr(
        flash_attention_module,
        "fa_peft_integration_check",
        lambda query, key, value, target_dtype: (query, key, value),
    )
    called = {}

    def masked_flash_path(query, key, value, attention_mask, query_length, **kwargs):
        called["attention_mask"] = attention_mask
        called["query_length"] = query_length
        return torch.zeros_like(query)

    def fail_packed_path(*args, **kwargs):
        raise AssertionError("padded Qwen text positions must not enter the packed-position path")

    monkeypatch.setattr(flash_attention_module, "_flash_attention_forward", masked_flash_path)
    monkeypatch.setattr(flash_attention_module, "prepare_fa2_from_position_ids", fail_packed_path)
    query = torch.randn(2, 10, 2, 4)
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    text_position_ids = torch.tensor(
        [
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
            [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )

    output = flash_attention_module._custom_flash_attention_forward(
        query,
        query,
        query,
        attention_mask,
        query_length=query.size(1),
        position_ids=text_position_ids,
    )

    assert output.shape == query.shape
    assert called["query_length"] == query.size(1)
    assert torch.equal(called["attention_mask"], attention_mask)


@pytest.mark.parametrize("model_type", ["qwen2_vl", "qwen3_vl_moe"])
def test_model_level_visual_corruption_rejects_unsupported_model_types(model_type):
    actor_config = _make_actor_config()
    actor_module = _RecordingVisualActorModule(model_type)
    actor = DataParallelPPOActor(config=actor_config, actor_module=actor_module)

    with pytest.raises(ValueError, match="currently supports qwen2_5_vl and qwen3_vl"):
        actor._visual_corruption_context(
            {"name": "cross_modal_attention_value_mean", "kwargs": {}},
            input_ids=torch.tensor([[10, 20, 11]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )


def test_model_level_visual_corruption_rejects_ulysses_sequence_parallelism():
    actor_config = _make_actor_config()
    actor_config.ulysses_size = 2
    actor_module = _RecordingVisualActorModule("qwen2_5_vl")
    actor = DataParallelPPOActor(config=actor_config, actor_module=actor_module)

    with pytest.raises(ValueError, match="requires worker.actor.ulysses_size=1"):
        actor._visual_corruption_context(
            {"name": "cross_modal_attention_value_mean", "kwargs": {}},
            input_ids=torch.tensor([[10, 20, 11]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )


def _make_fsdp_worker_for_mm_cache_tests():
    worker = object.__new__(FSDPWorker)
    worker._cache = {}
    worker.processor = SimpleNamespace(image_processor=_RecordingImageProcessor())
    return worker


def test_fsdp_worker_keeps_uid_dedup_for_original_images():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    data = _make_batch(image, image, uid_a="rollout", uid_b="rollout")
    data.meta_info.update({"min_pixels": None, "max_pixels": None, "video_fps": 2.0})
    worker = _make_fsdp_worker_for_mm_cache_tests()

    worker._process_multi_modal_inputs(data)

    assert len(worker.processor.image_processor.calls) == 1
    first = data.non_tensor_batch["multi_modal_inputs"][0]["pixel_values"].item()
    second = data.non_tensor_batch["multi_modal_inputs"][1]["pixel_values"].item()
    assert first == second


def test_fsdp_worker_uses_aux_cache_ids_to_keep_response_corruptions_distinct():
    black = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    white = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    data = _make_batch(
        black,
        white,
        uid_a="rollout",
        uid_b="rollout",
        cache_id_a="rollout:response:0",
        cache_id_b="rollout:response:1",
    )
    # An explicit corrupted-view cache identity must outrank the generic
    # agent-trajectory identity.
    data.non_tensor_batch["agent_trajectory_id"] = np.array(
        ["trajectory", "trajectory"],
        dtype=object,
    )
    data.meta_info.update({"min_pixels": None, "max_pixels": None, "video_fps": 2.0})
    worker = _make_fsdp_worker_for_mm_cache_tests()

    worker._process_multi_modal_inputs(data, cache_key_field="multi_modal_cache_id")

    assert len(worker.processor.image_processor.calls) == 2
    first = data.non_tensor_batch["multi_modal_inputs"][0]["pixel_values"].item()
    second = data.non_tensor_batch["multi_modal_inputs"][1]["pixel_values"].item()
    assert first != second


def test_fsdp_worker_requires_explicit_multimodal_cache_key():
    image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    data = _make_batch(image, image)
    data.meta_info.update({"min_pixels": None, "max_pixels": None, "video_fps": 2.0})
    worker = _make_fsdp_worker_for_mm_cache_tests()

    with pytest.raises(KeyError, match="multi_modal_cache_id"):
        worker._process_multi_modal_inputs(
            data,
            cache_key_field="multi_modal_cache_id",
        )


def test_no_image_padding_free_preserves_response_alignment():
    if not hasattr(dp_actor_module, "unpad_input"):
        pytest.skip("flash-attn padding-free ops are unavailable")

    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    batch = _make_batch(image, image)
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyTokenizer(),
        processor=_DummyProcessor(),
        image_patch_size=2,
    )
    config = AlgorithmConfig(corrupt_image="no_image")

    rebuilt = builder.build_batch(batch[:1], config, global_step=0).batch
    rebuilt.meta_info["temperature"] = 1.0

    actor_config_dense = _make_actor_config()
    actor_config_dense.use_torch_compile = False
    actor_config_dense.padding_free = False
    actor_config_dense.micro_batch_size_per_device_for_experience = 1
    actor_dense = DataParallelPPOActor(
        config=actor_config_dense,
        actor_module=_NextTokenActorModule(),
    )
    actor_dense.rank = 1
    actor_dense.log_probs_from_logits = _cpu_log_probs_from_logits

    actor_config_pf = _make_actor_config()
    actor_config_pf.use_torch_compile = False
    actor_config_pf.padding_free = True
    actor_config_pf.micro_batch_size_per_device_for_experience = 1
    actor_pf = DataParallelPPOActor(
        config=actor_config_pf,
        actor_module=_NextTokenActorModule(),
    )
    actor_pf.rank = 1
    actor_pf.log_probs_from_logits = _cpu_log_probs_from_logits

    dense_log_probs = actor_dense.compute_log_prob(rebuilt)
    padding_free_log_probs = actor_pf.compute_log_prob(rebuilt)

    assert torch.equal(
        rebuilt.batch["input_ids"][0, -rebuilt.batch["responses"].size(-1) :],
        rebuilt.batch["responses"][0],
    )
    assert torch.allclose(dense_log_probs, padding_free_log_probs, atol=1e-6, rtol=1e-6)
    assert torch.all(dense_log_probs > -1e-4)


def test_no_image_qwen_position_ids_restart_at_zero_for_packed_fa2():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    raw_prompt = _make_raw_prompt()
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor(
                [[11, 12, 13, 14, 15, 16, 17, 18, 21, 22], [31, 32, 33, 34, 35, 36, 37, 38, 23, 24]],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones((2, 10), dtype=torch.long),
            "position_ids": torch.arange(10, dtype=torch.long).unsqueeze(0).expand(2, -1),
            "responses": torch.tensor([[21, 22], [23, 24]], dtype=torch.long),
            "response_mask": torch.ones((2, 2), dtype=torch.long),
        },
        non_tensors={
            "raw_prompt": np.array([raw_prompt, raw_prompt], dtype=object),
            "uid": np.array(["sample-a", "sample-b"], dtype=object),
            "multi_modal_data": np.array([{"images": [image]}, {"images": [image]}], dtype=object),
        },
    )
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=_DummyQwenTokenizer(),
        processor=_DummyQwenProcessor(),
        image_patch_size=2,
    )

    rebuilt = builder.build_batch(batch, AlgorithmConfig(corrupt_image="no_image"), global_step=0).batch

    text_position_ids = rebuilt.batch["position_ids"][:, 0, :]
    attention_mask = rebuilt.batch["attention_mask"].to(torch.bool)
    packed_text_positions = torch.cat(
        [text_position_ids[row][attention_mask[row]] for row in range(text_position_ids.size(0))],
        dim=0,
    ).unsqueeze(0)

    assert text_position_ids[0][attention_mask[0]][0].item() == 0
    assert text_position_ids[1][attention_mask[1]][0].item() == 0

    total_nnz = packed_text_positions.size(-1)
    q = torch.zeros((1, total_nnz, 1, 2), dtype=torch.float32)
    _, _, _, (cu_seqlens_q, _), (max_seqlen_q, _) = prepare_fa2_from_position_ids(
        q,
        q,
        q,
        packed_text_positions,
    )

    sample_lengths = attention_mask.sum(dim=-1).tolist()
    assert cu_seqlens_q.tolist() == [0, sample_lengths[0], sum(sample_lengths)]
    assert max_seqlen_q == sample_lengths[0]


def _dense_cross_modal_value_mean_reference(
    query,
    key,
    value,
    attention_mask,
    visual_spans,
    saliency_std_multiplier,
    scaling,
):
    if key.size(2) != query.size(2):
        repeat = query.size(2) // key.size(2)
        key = key.repeat_interleave(repeat, dim=2)
        value = value.repeat_interleave(repeat, dim=2)

    correction = torch.zeros_like(query)
    sequence_positions = torch.arange(query.size(1), device=query.device)
    for row_idx, spans in enumerate(visual_spans):
        if not spans:
            continue
        logits = torch.einsum("qhd,khd->hqk", query[row_idx], key[row_idx]) * scaling
        if query.dtype == torch.float16:
            logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
        allowed = attention_mask[row_idx].to(torch.bool).unsqueeze(0).expand(query.size(1), -1)
        allowed = allowed & (sequence_positions.unsqueeze(0) <= sequence_positions.unsqueeze(1))
        probabilities = torch.softmax(
            logits.masked_fill(~allowed.unsqueeze(0), torch.finfo(logits.dtype).min),
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)

        visual_positions = torch.cat([torch.arange(start, end, device=query.device) for start, end in spans])
        visual_values_by_head = value[row_idx, visual_positions].permute(1, 0, 2).contiguous()
        mean_visual_value = (visual_values_by_head.sum(dim=(1, 2)) / (visual_positions.numel() * value.size(-1))).view(
            query.size(2), 1, 1
        )
        for visual_start, visual_end in spans:
            query_positions = torch.arange(visual_end + 1, query.size(1), device=query.device)
            query_positions = query_positions[attention_mask[row_idx, query_positions].to(torch.bool)]
            cross_modal_probabilities = probabilities[:, query_positions, visual_start:visual_end]
            valid_probabilities = cross_modal_probabilities[cross_modal_probabilities > 1e-8]
            threshold = valid_probabilities.mean() + saliency_std_multiplier * valid_probabilities.std()
            salient_probabilities = cross_modal_probabilities * (cross_modal_probabilities > threshold).to(
                cross_modal_probabilities.dtype
            )
            visual_value_delta = mean_visual_value - value[row_idx, visual_start:visual_end].permute(1, 0, 2)
            correction[row_idx, query_positions] += torch.einsum(
                "hqi,hid->qhd",
                salient_probabilities,
                visual_value_delta,
            )
    return correction


def test_cross_modal_value_mean_chunked_correction_matches_dense_formula():
    torch.manual_seed(17)
    query = torch.randn(2, 9, 4, 3)
    key = torch.randn(2, 9, 2, 3)
    value = torch.randn(2, 9, 2, 3)
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.long,
    )
    visual_spans = (((1, 3),), ((2, 5),))
    scaling = query.size(-1) ** -0.5

    expected = _dense_cross_modal_value_mean_reference(
        query,
        key,
        value,
        attention_mask,
        visual_spans,
        saliency_std_multiplier=0.5,
        scaling=scaling,
    )
    actual = compute_cross_modal_attention_value_mean_correction(
        query,
        key,
        value,
        attention_mask=attention_mask,
        visual_spans=visual_spans,
        saliency_std_multiplier=0.5,
        scaling=scaling,
        query_chunk_size=2,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(actual[0, :4]) == 0


def test_cross_modal_value_mean_chunked_correction_matches_official_bfloat16_reductions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    query = (torch.randn(1, 129, 8, 32, device=device) * 0.2).to(torch.bfloat16)
    key = torch.randn(1, 129, 8, 32, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, 129, 8, 32, device=device, dtype=torch.bfloat16)
    attention_mask = torch.ones((1, 129), device=device, dtype=torch.long)
    visual_spans = (((1, 65),),)
    scaling = query.size(-1) ** -0.5

    expected = _dense_cross_modal_value_mean_reference(
        query,
        key,
        value,
        attention_mask,
        visual_spans,
        saliency_std_multiplier=2.0,
        scaling=scaling,
    )
    actual = compute_cross_modal_attention_value_mean_correction(
        query,
        key,
        value,
        attention_mask=attention_mask,
        visual_spans=visual_spans,
        saliency_std_multiplier=2.0,
        scaling=scaling,
        query_chunk_size=17,
    )

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_cross_modal_value_mean_empty_visual_span_is_identity():
    torch.manual_seed(19)
    query = torch.randn(1, 8, 4, 3)
    key = torch.randn(1, 8, 2, 3)
    value = torch.randn(1, 8, 2, 3)

    correction = compute_cross_modal_attention_value_mean_correction(
        query,
        key,
        value,
        attention_mask=torch.ones((1, 8), dtype=torch.long),
        visual_spans=((),),
        saliency_std_multiplier=2.0,
        scaling=query.size(-1) ** -0.5,
    )

    assert torch.count_nonzero(correction) == 0


def test_model_level_visual_corruption_rejects_unbalanced_visual_boundaries():
    with pytest.raises(ValueError, match="unbalanced vision_start/vision_end"):
        with use_model_level_visual_corruption(
            name="cross_modal_attention_value_mean",
            kwargs={},
            input_ids=torch.tensor([[10, 20, 21, 30]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            vision_start_token_id=10,
            vision_end_token_id=11,
        ):
            pass


def test_cross_modal_value_mean_correction_is_sample_isolated():
    torch.manual_seed(23)
    query = torch.randn(2, 8, 4, 3)
    key = torch.randn(2, 8, 2, 3)
    value = torch.randn(2, 8, 2, 3)
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    visual_spans = (((1, 3),), ((1, 3),))
    kwargs = {
        "attention_mask": attention_mask,
        "visual_spans": visual_spans,
        "saliency_std_multiplier": 1.0,
        "scaling": query.size(-1) ** -0.5,
        "query_chunk_size": 2,
    }

    original = compute_cross_modal_attention_value_mean_correction(query, key, value, **kwargs)
    changed_query = query.clone()
    changed_key = key.clone()
    changed_value = value.clone()
    changed_query[1] += 100.0
    changed_key[1] -= 50.0
    changed_value[1] *= -7.0
    changed = compute_cross_modal_attention_value_mean_correction(
        changed_query,
        changed_key,
        changed_value,
        **kwargs,
    )

    torch.testing.assert_close(original[0], changed[0], atol=0.0, rtol=0.0)


@requires_flash_attn
def test_model_level_visual_corruption_does_not_touch_noncausal_vision_attention(monkeypatch):
    monkeypatch.setattr(
        flash_attention_module,
        "_custom_flash_attention_forward",
        lambda query, key, value, attention_mask, query_length, **kwargs: torch.zeros_like(query),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("noncausal vision attention must not receive the language intervention")

    monkeypatch.setattr(
        flash_attention_module,
        "compute_cross_modal_attention_value_mean_correction",
        fail_if_called,
    )
    input_ids = torch.tensor([[10, 20, 21, 11, 30]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    query = torch.randn(2, 3, 4, 5)

    with use_model_level_visual_corruption(
        name="cross_modal_attention_value_mean",
        kwargs={"saliency_std_multiplier": 2.0},
        input_ids=input_ids,
        attention_mask=attention_mask,
        vision_start_token_id=10,
        vision_end_token_id=11,
    ):
        output, _ = flash_attention_module.flash_attention_forward(
            SimpleNamespace(is_causal=False),
            query,
            query,
            query,
            attention_mask=None,
            is_causal=False,
        )

    assert output.shape == (2, 4, 3, 5)


def test_cgpo_multi_image_transforms_only_modify_referenced_images():
    image_a = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    image_b = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
    response = (
        '<think>inspect <region name="lesion" image_idx="1" id="0">[[0, 0, 1000, 1000]]</region> '
        "then check the details</think>\\boxed{A}"
    )

    flat_images = cgpo_flat([image_a, image_b], response, fill_type="black")
    hierarchical_images = cgpo_hierarchical([image_a, image_b], response, fill_type="black")

    assert np.array_equal(np.asarray(flat_images[0]), np.asarray(image_a))
    assert np.array_equal(np.asarray(hierarchical_images[0]), np.asarray(image_a))
    assert not np.array_equal(np.asarray(flat_images[1]), np.asarray(image_b))
    assert not np.array_equal(np.asarray(hierarchical_images[1]), np.asarray(image_b))


def test_parse_response_regions_falls_back_to_json_grounding_inside_think():
    response = (
        '<think>inspect {"label": "lesion", "image_idx": 1, "bbox_list": [[0, 0, 1000, 1000]]} '
        "then check the details</think><answer>A</answer>"
    )

    regions = parse_response_regions(response, num_images=2)

    assert [(region.name, region.image_idx, region.boxes) for region in regions] == [
        ("lesion", 1, [[0, 0, 1000, 1000]])
    ]


def test_parse_response_regions_keeps_malformed_xml_fail_closed_without_json_fallback():
    response = (
        '<think><region name="cat" image_idx="0" id="0">not a box</region> '
        '{"label": "cat", "bbox_list": [[10, 20, 110, 220]]}</think><answer>A</answer>'
    )

    assert parse_response_regions(response, num_images=1) == []


def test_cgpo_multi_image_transforms_support_json_grounding():
    image_a = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    image_b = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
    response = (
        '<think>inspect {"label": "lesion", "image_idx": 1, "bbox_list": [[0, 0, 1000, 1000]]} '
        "then check the details</think><answer>A</answer>"
    )

    flat_images = cgpo_flat([image_a, image_b], response, fill_type="black")
    hierarchical_images = cgpo_hierarchical([image_a, image_b], response, fill_type="black")

    assert np.array_equal(np.asarray(flat_images[0]), np.asarray(image_a))
    assert np.array_equal(np.asarray(hierarchical_images[0]), np.asarray(image_a))
    assert not np.array_equal(np.asarray(flat_images[1]), np.asarray(image_b))
    assert not np.array_equal(np.asarray(hierarchical_images[1]), np.asarray(image_b))


def test_has_perception_reasoning_detects_enabled_algorithms():
    assert not has_perception_reasoning(None)
    assert not has_perception_reasoning({})
    assert has_perception_reasoning({"visual_sensitivity_loss_coef": 0.1})
    assert has_perception_reasoning({"top_entropy_quantile": 0.8})
    assert has_perception_reasoning({"tor_use_token_weighting": True})
    assert not has_perception_reasoning({"log_entropy": True})


def test_current_policy_entropy_helpers_only_require_grad_for_full_entropy_loss():
    assert not needs_current_policy_entropy(None)
    assert not current_policy_entropy_requires_grad(None)

    assert needs_current_policy_entropy({"log_entropy": True})
    assert not current_policy_entropy_requires_grad({"log_entropy": True})

    assert needs_current_policy_entropy({"top_entropy_quantile": 0.8})
    assert not current_policy_entropy_requires_grad({"top_entropy_quantile": 0.8})

    sampled_config = {"invariant_entropy_coef": 0.05, "entropy_loss_type": "sampled"}
    assert not needs_current_policy_entropy(sampled_config)
    assert not current_policy_entropy_requires_grad(sampled_config)

    full_config = {"invariant_entropy_coef": 0.05, "entropy_loss_type": "full"}
    assert needs_current_policy_entropy(full_config)
    assert current_policy_entropy_requires_grad(full_config)

    pepo_config = {"advantage_scaling_method": "pepo"}
    assert needs_current_policy_entropy(pepo_config)
    assert not current_policy_entropy_requires_grad(pepo_config)


def test_hidden_state_visual_similarity_scores_and_fail_closed():
    actor = DataParallelPPOActor(
        config=_make_actor_config(),
        actor_module=nn.Linear(1, 1),
    )
    input_ids = torch.tensor([[1, 151655, 2, 3, 4], [1, 2, 151655, 5, 6]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.long)
    layer_0 = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.2, 0.1], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.1, 0.2], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    layer_1 = layer_0 * 2.0

    scores = actor._compute_hidden_visual_scores(
        hidden_states=(layer_0, layer_1),
        input_ids=input_ids,
        response_length=2,
        response_mask=response_mask,
        visual_token_ids=[151655],
        metric="cosine",
    )

    assert scores.shape == response_mask.shape
    assert not scores.requires_grad
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] == 0.0

    with pytest.raises(ValueError, match="could not find visual tokens"):
        actor._compute_hidden_visual_scores(
            hidden_states=(layer_0,),
            input_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
            response_length=2,
            response_mask=torch.tensor([[1, 1]], dtype=torch.long),
            visual_token_ids=[151655],
            metric="cosine",
        )


def test_hidden_state_l1_l2_similarity_matches_official_sum_distance():
    response_hidden = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float32)
    visual_hidden = torch.tensor([[0.0, 4.0], [6.0, 8.0]], dtype=torch.float32)

    l1_scores = DataParallelPPOActor._hidden_similarity_scores(
        response_hidden=response_hidden,
        visual_hidden=visual_hidden,
        metric="l1",
    )
    l2_scores = DataParallelPPOActor._hidden_similarity_scores(
        response_hidden=response_hidden,
        visual_hidden=visual_hidden,
        metric="l2",
    )

    torch.testing.assert_close(l1_scores, torch.tensor([-9.0, -5.0]))
    torch.testing.assert_close(l2_scores, torch.tensor([-7.0, -4.0]))


def test_full_vocab_visual_sensitivity_scores_match_jsd_and_entropy_gap():
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    corrupted_logits = torch.tensor(
        [
            [[0.0, 3.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    scores, metrics = compute_full_vocab_visual_sensitivity_scores(
        logits=logits,
        corrupted_logits=corrupted_logits,
        response_mask=response_mask,
        metric="vepo",
        jsd_weight=0.7,
        entropy_gate="none",
    )

    assert scores.shape == response_mask.shape
    assert scores[0, 2].item() == 0.0
    assert scores[0, 0].item() > scores[0, 1].item()
    assert metrics["algo/sensitivity/full_vocab_jsd"] > 0.0
    assert "algo/sensitivity/full_vocab_kl" not in metrics
    assert "algo/sensitivity/full_vocab_hellinger" not in metrics
    assert metrics["algo/sensitivity/vepo_fused"] > 0.0


def test_full_vocab_visual_sensitivity_scores_support_kl_and_hellinger():
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    corrupted_logits = torch.tensor(
        [
            [[0.0, 2.0, -1.0], [0.0, 2.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    expected_kl = torch.tensor([[1.4592, 0.0]], dtype=torch.float32) * response_mask
    expected_hellinger = torch.tensor([[0.5807, 0.0]], dtype=torch.float32) * response_mask

    kl_scores, kl_metrics = compute_full_vocab_visual_sensitivity_scores(
        logits=logits,
        corrupted_logits=corrupted_logits,
        response_mask=response_mask,
        metric="full_vocab_kl",
    )
    hellinger_scores, hellinger_metrics = compute_full_vocab_visual_sensitivity_scores(
        logits=logits,
        corrupted_logits=corrupted_logits,
        response_mask=response_mask,
        metric="full_vocab_hellinger",
    )

    torch.testing.assert_close(kl_scores, expected_kl, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hellinger_scores, expected_hellinger, rtol=1e-4, atol=1e-4)
    assert "algo/sensitivity/full_vocab_hellinger" not in kl_metrics
    assert "algo/sensitivity/full_vocab_kl" not in hellinger_metrics
    assert kl_metrics["algo/sensitivity/full_vocab_kl"] == pytest.approx(
        expected_kl.mean().item(),
        abs=1e-4,
    )
    assert hellinger_metrics["algo/sensitivity/full_vocab_hellinger"] == pytest.approx(
        expected_hellinger.mean().item(),
        abs=1e-4,
    )


def test_full_vocab_kl_handles_negative_infinity_logits():
    logits = torch.tensor([[[0.0, float("-inf")]]], dtype=torch.float32)
    corrupted_logits = torch.tensor([[[float("-inf"), 0.0]]], dtype=torch.float32)
    response_mask = torch.ones((1, 1), dtype=torch.float32)

    scores, metrics = compute_full_vocab_visual_sensitivity_scores(
        logits=logits,
        corrupted_logits=corrupted_logits,
        response_mask=response_mask,
        metric="full_vocab_kl",
    )

    assert torch.isfinite(scores).all()
    assert metrics["algo/sensitivity/full_vocab_kl"] == pytest.approx(-math.log(1e-8))


def test_full_vocab_visual_sensitivity_components_support_sparse_response_scatter():
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 3.0, -1.0], [0.0, 0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    corrupted_logits = torch.tensor(
        [
            [[0.0, 3.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, -1.0], [0.0, 0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.long)

    dense_scores, _ = compute_full_vocab_visual_sensitivity_scores(
        logits=logits,
        corrupted_logits=corrupted_logits,
        response_mask=response_mask,
        metric="vepo",
        jsd_weight=0.7,
        entropy_gate="normal_entropy",
    )

    rows, cols = response_mask.nonzero(as_tuple=True)
    sparse_components = vs_module.compute_full_vocab_visual_sensitivity_components(
        logits=logits[rows, cols],
        corrupted_logits=corrupted_logits[rows, cols],
        component_names={"jsd", "abs_entropy_gap", "entropy"},
    )
    components = {}
    for name, sparse_values in sparse_components.items():
        dense_values = sparse_values.new_zeros(response_mask.shape)
        dense_values[rows, cols] = sparse_values
        components[name] = dense_values

    sparse_scores, _ = vs_module.combine_full_vocab_visual_sensitivity_scores(
        components=components,
        response_mask=response_mask,
        metric="vepo",
        jsd_weight=0.7,
        entropy_gate="normal_entropy",
    )

    assert sparse_scores.dtype.is_floating_point
    assert torch.allclose(sparse_scores, dense_scores)


def test_fused_full_vocab_visual_sensitivity_reuses_normal_logits_for_old_log_probs():
    logits = torch.tensor(
        [
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    corrupted_logits = torch.tensor(
        [
            [[0.0, 3.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    responses = torch.tensor([[0, 1, 2]], dtype=torch.long)
    response_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)

    def log_probs_from_logits(logits, labels):
        return -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)

    actor.log_probs_from_logits = log_probs_from_logits
    actor.entropy_from_logits = lambda logits: (
        -(F.softmax(logits.float(), dim=-1) * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    )

    old_log_probs = actor._log_probs_from_response_logits(logits, responses)
    decremental_old_log_probs = actor._log_probs_from_response_logits(corrupted_logits, responses)
    decremental_entropies = actor._entropy_from_response_logits(corrupted_logits)
    scores, metrics = actor._full_vocab_visual_sensitivity_from_logits_outputs(
        logits_output=logits,
        auxiliary_logits_output=corrupted_logits,
        response_mask=response_mask,
        metric="vepo",
        jsd_weight=0.7,
        entropy_gate="none",
    )
    expected_scores, expected_metrics = compute_full_vocab_visual_sensitivity_scores(
        logits=logits,
        corrupted_logits=corrupted_logits,
        response_mask=response_mask,
        metric="vepo",
        jsd_weight=0.7,
        entropy_gate="none",
    )

    torch.testing.assert_close(old_log_probs, log_probs_from_logits(logits, responses))
    torch.testing.assert_close(decremental_old_log_probs, log_probs_from_logits(corrupted_logits, responses))
    torch.testing.assert_close(decremental_entropies, actor.entropy_from_logits(corrupted_logits))
    torch.testing.assert_close(scores, expected_scores)
    assert metrics == expected_metrics


def test_sampled_boxcox_bounds_sign_and_log_ratio_limit():
    log_p = torch.log(torch.tensor([[0.9, 0.5, 0.01, 1e-6]], dtype=torch.float64))
    log_q = torch.log(torch.tensor([[0.1, 0.5, 0.5, 0.9]], dtype=torch.float64))

    for alpha in (0.5, 1.0, 2.0):
        scores = vs_module.compute_sampled_boxcox(corrupted_log_probs=log_q, reference_log_probs=log_p, alpha=alpha)
        assert scores.abs().max().item() <= 1.0 / alpha + 1e-9
        assert torch.equal(torch.sign(scores), torch.sign(log_p - log_q))

    small_alpha_scores = vs_module.compute_sampled_boxcox(
        corrupted_log_probs=log_q, reference_log_probs=log_p, alpha=1e-8
    )
    torch.testing.assert_close(small_alpha_scores, log_p - log_q, atol=1e-5, rtol=1e-5)

    with pytest.raises(ValueError, match="alpha > 0"):
        vs_module.compute_sampled_boxcox(corrupted_log_probs=log_q, reference_log_probs=log_p, alpha=0.0)


def test_sampled_sensitivity_dispatcher_and_metric_resolution():
    log_p = torch.tensor([[-0.5, -1.0]])
    log_q = torch.tensor([[-1.5, -0.2]])

    kl = vs_module.compute_sampled_sensitivity_scores(
        metric="sampled_low_var_kl", corrupted_log_probs=log_q, reference_log_probs=log_p, reference_mode="old"
    )
    torch.testing.assert_close(kl, vs_module.compute_sampled_low_var_kl(log_q, log_p, reference_mode="old"))

    boxcox = vs_module.compute_sampled_sensitivity_scores(
        metric="sampled_boxcox", corrupted_log_probs=log_q, reference_log_probs=log_p, boxcox_alpha=2.0
    )
    torch.testing.assert_close(boxcox, vs_module.compute_sampled_boxcox(log_q, log_p, alpha=2.0))

    with pytest.raises(ValueError, match="Unsupported sampled"):
        vs_module.compute_sampled_sensitivity_scores(
            metric="full_vocab_jsd", corrupted_log_probs=log_q, reference_log_probs=log_p
        )

    assert vs_module.resolve_sampled_sensitivity_metric("sampled_boxcox") == "sampled_boxcox"
    assert vs_module.resolve_sampled_sensitivity_metric("full_vocab_jsd") == "sampled_low_var_kl"
    assert vs_module.resolve_sampled_sensitivity_metric("hidden_state_similarity") == "sampled_low_var_kl"


def test_algorithm_config_parses_sampled_boxcox_and_validates_alpha():
    config = AlgorithmConfig(
        corrupt_image="random_patch",
        visual_sensitivity_loss_coef=0.01,
        visual_sensitivity_metric="sampled_boxcox",
        visual_sensitivity_boxcox_alpha=0.5,
    )
    config.post_init()

    loss_config = build_perception_reasoning_loss_config(config)
    assert loss_config["visual_sensitivity_metric"] == "sampled_boxcox"
    assert loss_config["visual_sensitivity_boxcox_alpha"] == 0.5
    assert needs_decremental_auxiliary(config)
    assert needs_auxiliary_log_probs(config)
    assert not needs_full_vocab_visual_sensitivity(config)

    with pytest.raises(ValueError, match="visual_sensitivity_boxcox_alpha"):
        bad_config = AlgorithmConfig(visual_sensitivity_boxcox_alpha=0.0)
        bad_config.post_init()


def test_config_metric_choices_match_visual_sensitivity_module():
    for metric in sorted(vs_module.VISUAL_SENSITIVITY_METRIC_CHOICES):
        config = AlgorithmConfig(
            visual_sensitivity_metric=metric,
        )
        config.post_init()

    with pytest.raises(ValueError):
        unknown = AlgorithmConfig(visual_sensitivity_metric="sampled_unknown")
        unknown.post_init()


def test_boxcox_sensitivity_loss_uses_score_function_gradient():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    response_mask = tensors["response_mask"]
    coef, alpha = 0.05, 0.5

    log_prob = tensors["log_prob"].clone().requires_grad_(True)
    loss_config = {
        "visual_sensitivity_loss_coef": coef,
        "visual_sensitivity_metric": "sampled_boxcox",
        "visual_sensitivity_boxcox_alpha": alpha,
        "visual_sensitivity_reference": "current",
        "entropy_loss_type": "sampled",
    }
    policy_loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=tensors["old_log_prob"],
        advantages=torch.zeros_like(tensors["advantages"]),
        response_mask=response_mask,
        entropy=None,
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )
    policy_loss.backward()

    expected_scores = vs_module.compute_sampled_boxcox(
        corrupted_log_probs=tensors["decremental_old_log_probs"],
        reference_log_probs=tensors["log_prob"],
        alpha=alpha,
    )
    expected_grad = -coef * expected_scores * response_mask / response_mask.sum()
    torch.testing.assert_close(log_prob.grad, expected_grad)
    # The Box-Cox score is bounded, so per-token gradients are bounded by coef / alpha.
    assert log_prob.grad.abs().max().item() <= coef / alpha / response_mask.sum().item() + 1e-8
    assert "algo/sensitivity/sampled_boxcox" in metrics
    assert "algo/sensitivity/active" in metrics
    assert metrics["algo/sensitivity/active"] == metrics["algo/sensitivity/sampled_boxcox"]
    # The diagnostic KL is opt-in and off by default.
    assert "algo/sensitivity/sampled_low_var_kl" not in metrics

    # The default KL metric keeps its historical pathwise loss and metric keys.
    kl_log_prob = tensors["log_prob"].clone().requires_grad_(True)
    kl_policy_loss, kl_metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config={"visual_sensitivity_loss_coef": coef, "entropy_loss_type": "sampled"},
        log_prob=kl_log_prob,
        old_log_prob=tensors["old_log_prob"],
        advantages=torch.zeros_like(tensors["advantages"]),
        response_mask=response_mask,
        entropy=None,
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )
    expected_kl = vs_module.compute_sampled_low_var_kl(
        tensors["decremental_old_log_probs"], tensors["log_prob"], reference_mode="current"
    )
    expected_kl_loss = average_loss(-coef * expected_kl, response_mask, mode=actor_config.loss_avg_mode)
    torch.testing.assert_close(kl_policy_loss, expected_kl_loss)
    assert "algo/sensitivity/sampled_boxcox" not in kl_metrics
    assert "algo/sensitivity/sampled_low_var_kl" in kl_metrics
    assert kl_metrics["algo/sensitivity/active"] == kl_metrics["algo/sensitivity/sampled_low_var_kl"]


def test_visual_sensitivity_log_metrics_opt_in_diagnostics():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    response_mask = tensors["response_mask"]
    loss_config = {
        "visual_sensitivity_loss_coef": 0.05,
        "visual_sensitivity_metric": "sampled_boxcox",
        "visual_sensitivity_boxcox_alpha": 0.5,
        "visual_sensitivity_log_metrics": ("sampled_low_var_kl",),
        "entropy_loss_type": "sampled",
    }
    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=torch.zeros_like(tensors["advantages"]),
        response_mask=response_mask,
        entropy=None,
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )
    assert "algo/sensitivity/sampled_boxcox" in metrics
    expected_kl = vs_module.compute_sampled_low_var_kl(
        tensors["decremental_old_log_probs"], tensors["log_prob"], reference_mode="current"
    )
    assert metrics["algo/sensitivity/sampled_low_var_kl"] == pytest.approx(
        VF.masked_mean(expected_kl, response_mask).item()
    )

    config = AlgorithmConfig(
        corrupt_image="random_patch",
        visual_sensitivity_loss_coef=0.01,
        visual_sensitivity_metric="sampled_boxcox",
        visual_sensitivity_log_metrics="sampled_low_var_kl",
    )
    config.post_init()
    assert config.visual_sensitivity_log_metrics == ("sampled_low_var_kl",)
    assert build_perception_reasoning_loss_config(config)["visual_sensitivity_log_metrics"] == ("sampled_low_var_kl",)

    with pytest.raises(ValueError, match="visual_sensitivity_log_metrics"):
        bad_config = AlgorithmConfig(visual_sensitivity_log_metrics=["full_vocab_jsd"])
        bad_config.post_init()


def test_spearman_rank_correlation_basic_cases():
    values = torch.tensor([0.1, 0.5, 0.9, 1.2])
    aligned = torch.tensor([1.0, 2.0, 3.0, 4.0])
    valid = torch.ones(4, dtype=torch.bool)

    assert vs_module.compute_spearman_rank_correlation(values, aligned, valid).item() == pytest.approx(1.0)
    assert vs_module.compute_spearman_rank_correlation(values, -aligned, valid).item() == pytest.approx(-1.0)

    single_valid = torch.tensor([True, False, False, False])
    assert math.isnan(vs_module.compute_spearman_rank_correlation(values, aligned, single_valid).item())
    constant = torch.full((4,), 0.7)
    assert math.isnan(vs_module.compute_spearman_rank_correlation(values, constant, valid).item())

    # Ties get average ranks: ranks([1, 2, 2, 3]) = [0, 1.5, 1.5, 3] -> corr = 3 / sqrt(10).
    tied = torch.tensor([1.0, 2.0, 2.0, 3.0])
    expected_tied_corr = 3.0 / math.sqrt(10.0)
    assert vs_module.compute_spearman_rank_correlation(tied, aligned, valid).item() == pytest.approx(
        expected_tied_corr
    )
    # Average-rank handling makes the result invariant to how tied entries are ordered.
    permutation = torch.tensor([2, 0, 3, 1])
    assert vs_module.compute_spearman_rank_correlation(
        tied[permutation], aligned[permutation], valid
    ).item() == pytest.approx(expected_tied_corr)


def test_log_metrics_diagnostics_emit_overlap_and_rank_corr():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    response_mask = tensors["response_mask"]
    base_kwargs = dict(
        actor_config=actor_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=torch.zeros_like(tensors["advantages"]),
        response_mask=response_mask,
        entropy=None,
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )

    loss_config = {
        "visual_sensitivity_metric": "sampled_boxcox",
        "visual_sensitivity_boxcox_alpha": 0.5,
        "visual_sensitivity_log_metrics": ("sampled_low_var_kl",),
        "visual_sensitivity_loss_coef": 0.05,
        "top_perception_quantile": 0.5,
        "perception_thr_granularity": "micro_batch",
        "entropy_loss_type": "sampled",
    }
    _, metrics = compute_perception_reasoning_policy_loss(loss_config=loss_config, **base_kwargs)

    boxcox_scores = vs_module.compute_sampled_boxcox(
        corrupted_log_probs=tensors["decremental_old_log_probs"],
        reference_log_probs=tensors["log_prob"],
        alpha=0.5,
    )
    kl_scores = vs_module.compute_sampled_low_var_kl(
        tensors["decremental_old_log_probs"], tensors["log_prob"], reference_mode="current"
    )
    boxcox_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=boxcox_scores, response_mask=response_mask, quantile=0.5, granularity="batch"
    )
    kl_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=kl_scores, response_mask=response_mask, quantile=0.5, granularity="batch"
    )
    expected_overlap = (boxcox_mask & kl_mask).sum().item() / boxcox_mask.sum().item()
    overlap_key = "algo/sensitivity/diag_topq_overlap_sampled_boxcox_vs_sampled_low_var_kl"
    assert metrics[overlap_key] == pytest.approx(expected_overlap)

    boxcox_means, valid_scores_mask = vs_module.compute_response_mean_scores(boxcox_scores, response_mask)
    kl_means, _ = vs_module.compute_response_mean_scores(kl_scores, response_mask)
    expected_corr = vs_module.compute_spearman_rank_correlation(boxcox_means, kl_means, valid_scores_mask)
    corr_key = "algo/sensitivity/diag_response_rank_corr_sampled_boxcox_vs_sampled_low_var_kl"
    assert metrics[corr_key] == pytest.approx(expected_corr.item())

    # Without a perception quantile the overlap is undefined; rank correlation stays.
    no_quantile_config = dict(loss_config)
    no_quantile_config.pop("top_perception_quantile")
    _, no_quantile_metrics = compute_perception_reasoning_policy_loss(loss_config=no_quantile_config, **base_kwargs)
    assert overlap_key not in no_quantile_metrics
    assert corr_key in no_quantile_metrics

    # Comparing the active metric with itself is skipped entirely.
    self_config = dict(loss_config)
    self_config["visual_sensitivity_log_metrics"] = ("sampled_boxcox",)
    _, self_metrics = compute_perception_reasoning_policy_loss(loss_config=self_config, **base_kwargs)
    assert not any(key.startswith("algo/sensitivity/diag_") for key in self_metrics)


def test_log_metrics_diagnostics_use_consumed_signal_name():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    response_mask = tensors["response_mask"]
    precomputed_scores = torch.tensor(
        [[0.9, 0.1, 0.5, 0.0], [0.2, 0.8, 0.3, 0.6]],
        dtype=torch.float32,
    )
    loss_config = {
        "visual_sensitivity_metric": "full_vocab_hellinger",
        "visual_sensitivity_log_metrics": ("sampled_boxcox",),
        "visual_sensitivity_boxcox_alpha": 1.0,
        "top_perception_quantile": 0.5,
        "perception_thr_granularity": "micro_batch",
        "entropy_loss_type": "sampled",
    }
    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=torch.zeros_like(tensors["advantages"]),
        response_mask=response_mask,
        entropy=None,
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        per_token_sensitivity_scores=precomputed_scores,
    )

    assert metrics["algo/sensitivity/active"] == pytest.approx(
        VF.masked_mean(precomputed_scores, response_mask).item()
    )
    assert "algo/sensitivity/diag_topq_overlap_full_vocab_hellinger_vs_sampled_boxcox" in metrics
    assert "algo/sensitivity/diag_response_rank_corr_full_vocab_hellinger_vs_sampled_boxcox" in metrics
    assert "algo/sensitivity/sampled_boxcox" in metrics
    # The sampled KL was neither consumed nor requested here, so its key must not leak.
    assert "algo/sensitivity/sampled_low_var_kl" not in metrics


def test_fused_full_vocab_flag_accounts_for_log_metrics():
    from types import SimpleNamespace

    def make_fake_trainer(**algo_kwargs):
        algorithm = AlgorithmConfig(
            corrupt_image="random_patch",
            visual_sensitivity_metric="full_vocab_hellinger",
            top_perception_quantile=0.5,
            **algo_kwargs,
        )
        algorithm.post_init()
        return SimpleNamespace(config=SimpleNamespace(algorithm=algorithm))

    with_diagnostics = make_fake_trainer(visual_sensitivity_log_metrics="sampled_boxcox")
    assert RayPPOTrainer._full_vocab_fused_needs_decremental_log_probs(with_diagnostics)

    without_diagnostics = make_fake_trainer()
    assert not RayPPOTrainer._full_vocab_fused_needs_decremental_log_probs(without_diagnostics)


def test_boxcox_shaping_context_matches_manual_median_mad():
    tensors = _make_common_tensors()
    response_mask = tensors["response_mask"].to(torch.bool)
    loss_config = {
        "advantage_scaling_method": "cgpo",
        "visual_sensitivity_metric": "sampled_boxcox",
        "visual_sensitivity_boxcox_alpha": 2.0,
    }
    batch = {
        "old_log_probs": tensors["old_log_prob"],
        "decremental_old_log_probs": tensors["decremental_old_log_probs"],
        "response_mask": tensors["response_mask"],
    }
    context = build_sensitivity_advantage_shaping_context(loss_config, batch)

    scores = vs_module.compute_sampled_boxcox(
        corrupted_log_probs=tensors["decremental_old_log_probs"],
        reference_log_probs=tensors["old_log_prob"],
        alpha=2.0,
    )
    response_scores, _ = vs_module.compute_response_mean_scores(scores, response_mask)
    expected_median = response_scores.median()
    expected_mad = (response_scores - expected_median).abs().median()
    assert context["global_median_score"] == pytest.approx(expected_median.item())
    assert context["global_mad_score"] == pytest.approx(expected_mad.item())


def test_batch_top_quantile_mask_uses_exact_topk_for_ties():
    values = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    mask, threshold, fraction = pr_loss_module._compute_top_quantile_mask(
        values=values,
        response_mask=response_mask,
        quantile=0.3,
        granularity="batch",
    )

    assert mask.sum().item() == 3
    assert mask[0, 3]
    assert mask[1, 2]
    assert not mask[1, 3]
    assert threshold.item() == pytest.approx(0.0)
    assert fraction.item() == pytest.approx(3.0 / 7.0)


def test_response_top_quantile_mask_keeps_exact_topk_per_response():
    values = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    mask, _, fraction = pr_loss_module._compute_top_quantile_mask(
        values=values,
        response_mask=response_mask,
        quantile=0.5,
        granularity="response",
    )

    torch.testing.assert_close(mask.sum(dim=1), torch.tensor([2, 2]))
    assert mask[0, 2]
    assert mask[1, 1]
    assert not mask[0, 3]
    assert fraction.item() == pytest.approx(4.0 / 7.0)


def test_entropy_from_logits_matches_naive_full_vocab_entropy():
    logits = torch.randn(2, 3, 11, dtype=torch.float32)

    expected_logps = torch.log_softmax(logits, dim=-1)
    expected = -(torch.exp(expected_logps) * expected_logps).sum(dim=-1)
    actual = VF.entropy_from_logits(logits)

    assert actual.shape == logits.shape[:-1]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_entropy_from_logits_preserves_multidimensional_shapes():
    logits = torch.randn(2, 4, 3, 7, dtype=torch.float32)

    entropies = VF.entropy_from_logits(logits)

    assert entropies.shape == (2, 4, 3)


def test_log_entropy_only_uses_base_pg_path(monkeypatch):
    actor_config = _make_actor_config()
    actor_config.use_torch_compile = False
    actor_module = nn.Linear(1, 1)

    actor = DataParallelPPOActor(
        config=actor_config,
        actor_module=actor_module,
        actor_optimizer=torch.optim.SGD(actor_module.parameters(), lr=0.1),
    )
    actor.rank = 1

    def fake_forward(_model_inputs, temperature, return_entropy=False, entropy_requires_grad=True):
        del temperature
        assert return_entropy is True
        assert entropy_requires_grad is False
        log_probs = torch.tensor([[-0.5, -0.4], [-0.3, -0.2]], dtype=torch.float32, requires_grad=True)
        entropy = torch.tensor([[0.7, 0.6], [0.5, 0.4]], dtype=torch.float32)
        return (log_probs, entropy) if return_entropy else log_probs

    def fake_compute_policy_loss(**kwargs):
        del kwargs
        pg_loss = torch.tensor(0.25, dtype=torch.float32, requires_grad=True)
        return pg_loss, {"clipfrac": 0.0}

    def fake_pr_loss(**kwargs):
        raise AssertionError("perception reasoning loss helper should not be used when only log_entropy is enabled")

    monkeypatch.setattr(actor, "_forward_micro_batch", fake_forward)
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    monkeypatch.setattr(dp_actor_module, "compute_policy_loss", fake_compute_policy_loss)
    monkeypatch.setattr(dp_actor_module, "compute_perception_reasoning_policy_loss", fake_pr_loss)
    monkeypatch.setattr(dp_actor_module.dist, "all_reduce", lambda tensor, op=None: None)

    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[3, 4], [3, 4]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "old_log_probs": torch.tensor([[-0.6, -0.5], [-0.4, -0.3]], dtype=torch.float32),
            "advantages": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        },
        non_tensors={"multi_modal_inputs": np.array([None, None], dtype=object)},
        meta_info={
            "temperature": 1.0,
            "perception_reasoning_config": {"log_entropy": True, "top_entropy_quantile": 1.0},
        },
    )

    metrics = actor.update_policy(batch)

    assert "actor/policy_entropy" in metrics
    assert metrics["actor/policy_entropy"][0] == pytest.approx(0.55)


def test_top_entropy_quantile_pr_path_uses_entropy_without_grad(monkeypatch):
    actor_config = _make_actor_config()
    actor_config.use_torch_compile = False
    actor_module = nn.Linear(1, 1)

    actor = DataParallelPPOActor(
        config=actor_config,
        actor_module=actor_module,
        actor_optimizer=torch.optim.SGD(actor_module.parameters(), lr=0.1),
    )
    actor.rank = 1

    def fake_forward(_model_inputs, temperature, return_entropy=False, entropy_requires_grad=True):
        del temperature
        assert return_entropy is True
        assert entropy_requires_grad is False
        log_probs = torch.tensor([[-0.5, -0.4], [-0.3, -0.2]], dtype=torch.float32, requires_grad=True)
        entropy = torch.tensor([[0.7, 0.6], [0.5, 0.4]], dtype=torch.float32)
        return (log_probs, entropy) if return_entropy else log_probs

    def fake_pr_loss(**kwargs):
        del kwargs
        pg_loss = torch.tensor(0.25, dtype=torch.float32, requires_grad=True)
        return pg_loss, {"actor/policy_entropy": 0.55}

    monkeypatch.setattr(actor, "_forward_micro_batch", fake_forward)
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    monkeypatch.setattr(dp_actor_module, "compute_perception_reasoning_policy_loss", fake_pr_loss)
    monkeypatch.setattr(dp_actor_module.dist, "all_reduce", lambda tensor, op=None: None)

    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[3, 4], [3, 4]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "old_log_probs": torch.tensor([[-0.6, -0.5], [-0.4, -0.3]], dtype=torch.float32),
            "advantages": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        },
        non_tensors={"multi_modal_inputs": np.array([None, None], dtype=object)},
        meta_info={
            "temperature": 1.0,
            "perception_reasoning_config": {"top_entropy_quantile": 0.8, "entropy_loss_type": "sampled"},
        },
    )

    metrics = actor.update_policy(batch)

    assert "actor/policy_entropy" in metrics


def test_update_policy_hidden_state_similarity_passes_scores_to_pr_loss(monkeypatch):
    actor_config = _make_actor_config()
    actor_config.use_torch_compile = False
    actor_module = nn.Linear(1, 1)

    actor = DataParallelPPOActor(
        config=actor_config,
        actor_module=actor_module,
        actor_optimizer=torch.optim.SGD(actor_module.parameters(), lr=0.1),
    )
    actor.rank = 1
    hidden_scores = torch.tensor([[0.2, 0.8], [0.5, 0.1]], dtype=torch.float32)

    def fake_forward(
        _model_inputs,
        temperature,
        return_entropy=False,
        entropy_requires_grad=True,
        return_hidden_visual_scores=False,
        visual_token_ids=None,
        hidden_visual_metric="cosine",
    ):
        del temperature
        assert return_entropy is True
        assert entropy_requires_grad is False
        assert return_hidden_visual_scores is True
        assert visual_token_ids == [151655]
        assert hidden_visual_metric == "cosine"
        log_probs = torch.tensor([[-0.5, -0.4], [-0.3, -0.2]], dtype=torch.float32, requires_grad=True)
        entropy = torch.tensor([[0.7, 0.6], [0.5, 0.4]], dtype=torch.float32)
        return log_probs, entropy, hidden_scores

    captured = {}

    def fake_pr_loss(**kwargs):
        captured["per_token_sensitivity_scores"] = kwargs["per_token_sensitivity_scores"]
        captured["entropy"] = kwargs["entropy"]
        pg_loss = torch.tensor(0.25, dtype=torch.float32, requires_grad=True)
        return pg_loss, {"algo/sensitivity/active": 0.4}

    monkeypatch.setattr(actor, "_forward_micro_batch", fake_forward)
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    monkeypatch.setattr(dp_actor_module, "compute_perception_reasoning_policy_loss", fake_pr_loss)
    monkeypatch.setattr(dp_actor_module.dist, "all_reduce", lambda tensor, op=None: None)

    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 151655, 3, 4], [1, 151655, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[3, 4], [3, 4]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "old_log_probs": torch.tensor([[-0.6, -0.5], [-0.4, -0.3]], dtype=torch.float32),
            "advantages": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        },
        non_tensors={"multi_modal_inputs": np.array([None, None], dtype=object)},
        meta_info={
            "temperature": 1.0,
            "perception_reasoning_config": {
                "visual_sensitivity_metric": "hidden_state_similarity",
                "visual_sensitivity_hidden_metric": "cosine",
                "advantage_scaling_method": "pepo",
                "visual_token_ids": [151655],
            },
        },
    )

    actor.update_policy(batch)

    torch.testing.assert_close(captured["per_token_sensitivity_scores"], hidden_scores)
    assert captured["entropy"] is not None


def test_pr_loss_uses_single_compute_policy_loss_call_in_vppo_mode(monkeypatch):
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "top_perception_quantile": 0.5,
        "perception_thr_granularity": "response",
        "response_advantage_scaling_method": "vppo",
        "vppo_response_scaling_min": 0.85,
    }
    batch = DataProto.from_dict(
        tensors={
            "response_mask": tensors["response_mask"],
            "old_log_probs": tensors["old_log_prob"],
            "decremental_old_log_probs": tensors["decremental_old_log_probs"],
        }
    )
    shaping_context = build_sensitivity_advantage_shaping_context(loss_config=loss_config, data=batch)
    call_count = 0
    captured = {}
    fake_pg_losses = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
        dtype=torch.float32,
    )

    def fake_compute_policy_loss(**kwargs):
        nonlocal call_count
        call_count += 1
        captured["advantages"] = kwargs["advantages"].clone()
        captured["loss_avg_mode"] = kwargs["loss_avg_mode"]
        return fake_pg_losses.clone(), {"ppo_kl": 0.1, "entropy_loss": 0.2}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        advantage_shaping_context=shaping_context,
    )

    assert call_count == 1
    assert captured["loss_avg_mode"] == "none"
    per_token_sensitivity_kl = vs_module.compute_sampled_low_var_kl(
        tensors["decremental_old_log_probs"],
        tensors["log_prob"],
        reference_mode="current",
    )
    scaling_factors, _ = pr_loss_module._compute_vppo_response_scaling(
        per_token_sensitivity=per_token_sensitivity_kl,
        response_mask=tensors["response_mask"],
        scaling_min=loss_config["vppo_response_scaling_min"],
        shaping_context=shaping_context,
    )
    expected_advantages = tensors["advantages"] * scaling_factors.unsqueeze(-1)
    torch.testing.assert_close(captured["advantages"], expected_advantages)

    perception_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=per_token_sensitivity_kl,
        response_mask=tensors["response_mask"],
        quantile=loss_config["top_perception_quantile"],
        granularity=loss_config["perception_thr_granularity"],
    )
    expected_raw_pg_loss = average_loss(fake_pg_losses, tensors["response_mask"], mode=actor_config.loss_avg_mode)
    expected_weighted_pg_loss = average_loss(
        fake_pg_losses * perception_mask.float(),
        tensors["response_mask"],
        mode=actor_config.loss_avg_mode,
    )
    assert loss.item() == pytest.approx(expected_weighted_pg_loss.item())
    assert metrics["actor/pg_loss"] == pytest.approx(expected_raw_pg_loss.item())
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected_weighted_pg_loss.item())


def test_pr_loss_prefers_precomputed_sensitivity_scores_for_perception_mask(monkeypatch):
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    response_mask = tensors["response_mask"]
    precomputed_scores = torch.tensor(
        [[0.1, 0.9, 0.2, 0.3], [0.8, 0.1, 0.4, 0.2]],
        dtype=torch.float32,
    )
    loss_config = {
        "top_perception_quantile": 0.25,
        "perception_thr_granularity": "response",
        "normalize_pg_loss_by_selected_tokens": True,
    }
    fake_pg_losses = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
        dtype=torch.float32,
    )

    def fake_compute_policy_loss(**kwargs):
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=response_mask,
        entropy=tensors["entropy"],
        decremental_old_log_probs=None,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        per_token_sensitivity_scores=precomputed_scores,
    )

    perception_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=precomputed_scores,
        response_mask=response_mask,
        quantile=loss_config["top_perception_quantile"],
        granularity=loss_config["perception_thr_granularity"],
    )
    expected = average_loss(
        fake_pg_losses * perception_mask.float(),
        perception_mask & response_mask.to(torch.bool),
        mode=actor_config.loss_avg_mode,
    )
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["algo/sensitivity/active"] == pytest.approx(
        VF.masked_mean(precomputed_scores, response_mask).item()
    )


def test_private_vanilla_pg_helper_is_removed():
    assert not hasattr(pr_loss_module, "_compute_vanilla_pg_losses")


def test_vppo_advantage_shaping_context_and_loss_metrics():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "top_perception_quantile": 0.5,
        "perception_thr_granularity": "response",
        "response_advantage_scaling_method": "vppo",
        "vppo_response_scaling_min": 0.85,
    }
    batch = DataProto.from_dict(
        tensors={
            "response_mask": tensors["response_mask"],
            "old_log_probs": tensors["old_log_prob"],
            "decremental_old_log_probs": tensors["decremental_old_log_probs"],
        }
    )
    shaping_context = build_sensitivity_advantage_shaping_context(loss_config=loss_config, data=batch)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        advantage_shaping_context=shaping_context,
    )

    assert loss.ndim == 0
    assert "algo/token_selection/perception_threshold" in metrics
    assert "algo/token_selection/combined_fraction" in metrics
    assert "algo/vppo/dynamic_tas_beta_max" in metrics
    assert not math.isnan(metrics["algo/vppo/batch_signal_min"])
    assert "algo/advantage_scaling/factor_min" in metrics


def test_vppo_requires_precomputed_advantage_shaping_context():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "top_perception_quantile": 0.5,
        "perception_thr_granularity": "response",
        "response_advantage_scaling_method": "vppo",
        "vppo_response_scaling_min": 0.85,
    }

    with pytest.raises(ValueError, match="requires a precomputed advantage_shaping_context"):
        compute_perception_reasoning_policy_loss(
            actor_config=actor_config,
            loss_config=loss_config,
            log_prob=tensors["log_prob"],
            old_log_prob=tensors["old_log_prob"],
            advantages=tensors["advantages"],
            response_mask=tensors["response_mask"],
            entropy=tensors["entropy"],
            decremental_old_log_probs=tensors["decremental_old_log_probs"],
            decremental_entropies=None,
            incremental_old_log_probs=None,
            incremental_entropies=None,
            advantage_shaping_context=None,
        )


def test_pgpo_token_scaling_is_mass_conserving_and_handles_flat_scores():
    per_token_sensitivity = torch.tensor(
        [[0.0, 1.0, 4.0, 0.0], [2.0, 2.0, 2.0, 0.0]],
        dtype=torch.float32,
    )
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.float32)

    scaling, metrics = pr_loss_module._compute_pgpo_token_scaling(
        per_token_sensitivity=per_token_sensitivity,
        response_mask=response_mask,
        threshold=0.4,  # tau in the PGPO paper
        boost=2.0,  # beta in the PGPO paper
    )

    assert scaling.shape == per_token_sensitivity.shape
    torch.testing.assert_close((scaling * response_mask).sum(dim=-1), response_mask.sum(dim=-1))
    torch.testing.assert_close(scaling[1], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert metrics["algo/advantage_scaling/factor_mean"] == pytest.approx(1.0)


def test_pgpo_scales_token_advantages_before_single_pg_call(monkeypatch):
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "advantage_scaling_method": "pgpo",
        "pgpo_token_scaling_threshold": 0.4,
        "pgpo_token_scaling_boost": 2.0,
    }
    captured = {}
    fake_pg_losses = torch.ones_like(tensors["advantages"])

    def fake_compute_policy_loss(**kwargs):
        captured["advantages"] = kwargs["advantages"].clone()
        captured["loss_avg_mode"] = kwargs["loss_avg_mode"]
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )

    per_token_sensitivity_kl = vs_module.compute_sampled_low_var_kl(
        tensors["decremental_old_log_probs"],
        tensors["log_prob"],
        reference_mode="current",
    )
    scaling_factors, _ = pr_loss_module._compute_pgpo_token_scaling(
        per_token_sensitivity=per_token_sensitivity_kl,
        response_mask=tensors["response_mask"],
        threshold=loss_config["pgpo_token_scaling_threshold"],
        boost=loss_config["pgpo_token_scaling_boost"],
    )
    torch.testing.assert_close(captured["advantages"], tensors["advantages"] * scaling_factors)
    assert captured["loss_avg_mode"] == "none"
    assert "algo/pgpo/token_importance_mean" in metrics


def test_pepo_token_scaling_is_mass_conserving_and_detached():
    visual_scores = torch.tensor(
        [[0.1, 0.8, 0.2, 0.0], [0.5, 0.5, 0.5, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    entropy = torch.tensor(
        [[0.2, 0.1, 0.9, 0.0], [0.3, 0.7, 0.5, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.float32)

    scaling, metrics = pr_loss_module._compute_pepo_token_scaling(
        visual_scores=visual_scores,
        entropy=entropy,
        response_mask=response_mask,
        gate_alpha=0.05,
        gate_temperature=1.8,
    )

    assert scaling.shape == visual_scores.shape
    assert not scaling.requires_grad
    torch.testing.assert_close((scaling * response_mask).sum(dim=-1), response_mask.sum(dim=-1))
    assert metrics["algo/advantage_scaling/factor_mean"] == pytest.approx(1.0)


def test_advantage_scaling_schedule_interpolates_before_pg_loss(monkeypatch):
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    visual_scores = torch.tensor(
        [[0.1, 0.8, 0.2, 0.0], [0.5, 0.5, 0.5, 0.2]],
        dtype=torch.float32,
    )
    loss_config = {
        "visual_sensitivity_metric": "hidden_state_similarity",
        "advantage_scaling_method": "pepo",
        "advantage_scaling_schedule": "linear",
        "advantage_scaling_schedule_progress": 0.5,
        "pepo_gate_alpha": 0.05,
        "pepo_gate_temperature": 1.8,
    }
    captured = {}
    fake_pg_losses = torch.ones_like(tensors["advantages"])

    def fake_compute_policy_loss(**kwargs):
        captured["advantages"] = kwargs["advantages"].clone()
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=None,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        per_token_sensitivity_scores=visual_scores,
    )

    raw_scaling, _ = pr_loss_module._compute_pepo_token_scaling(
        visual_scores=visual_scores,
        entropy=tensors["entropy"],
        response_mask=tensors["response_mask"],
        gate_alpha=loss_config["pepo_gate_alpha"],
        gate_temperature=loss_config["pepo_gate_temperature"],
    )
    scheduled_scaling, _ = pr_loss_module._apply_advantage_scaling_schedule(
        scaling_factors=raw_scaling,
        response_mask=tensors["response_mask"],
        schedule="linear",
        progress=0.5,
    )
    torch.testing.assert_close(captured["advantages"], tensors["advantages"] * scheduled_scaling)
    assert metrics["algo/advantage_scaling/schedule_progress"] == pytest.approx(0.5)


def test_cgpo_scales_advantages_before_single_pg_call(monkeypatch):
    actor_config = _make_actor_config()
    log_prob = torch.zeros((2, 3), dtype=torch.float32)
    old_log_prob = torch.zeros((2, 3), dtype=torch.float32)
    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    response_mask = torch.ones((2, 3), dtype=torch.float32)
    entropy = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]], dtype=torch.float32)
    decremental_old_log_probs = torch.tensor([[0.1, 0.2, 1.0], [0.1, 0.5, 0.2]], dtype=torch.float32)
    loss_config = {
        "top_entropy_quantile": 0.33,
        "entropy_thr_granularity": "response",
        "top_perception_quantile": 0.33,
        "perception_thr_granularity": "response",
        "response_advantage_scaling_method": "cgpo",
        "cgpo_response_scaling_coef": 0.5,
    }
    shaping_context = {"global_median_score": 0.0, "global_mad_score": 1.0}
    captured = {}
    fake_pg_losses = torch.tensor([[1.2, 2.4, 3.6], [0.8, 1.6, 2.4]], dtype=torch.float32)

    def fake_compute_policy_loss(**kwargs):
        captured["advantages"] = kwargs["advantages"].clone()
        captured["loss_avg_mode"] = kwargs["loss_avg_mode"]
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        entropy=entropy,
        decremental_old_log_probs=decremental_old_log_probs,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        advantage_shaping_context=shaping_context,
    )

    per_token_sensitivity_kl = vs_module.compute_sampled_low_var_kl(
        decremental_old_log_probs,
        old_log_prob,
        reference_mode="old",
    )
    entropy_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=entropy,
        response_mask=response_mask,
        quantile=loss_config["top_entropy_quantile"],
        granularity=loss_config["entropy_thr_granularity"],
    )
    perception_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=per_token_sensitivity_kl,
        response_mask=response_mask,
        quantile=loss_config["top_perception_quantile"],
        granularity=loss_config["perception_thr_granularity"],
    )
    scaling_factors, _ = pr_loss_module._compute_cgpo_response_scaling(
        per_token_sensitivity=per_token_sensitivity_kl,
        response_mask=response_mask,
        scaling_coef=loss_config["cgpo_response_scaling_coef"],
        shaping_context=shaping_context,
    )
    expected_advantages = torch.where(
        perception_mask,
        advantages * scaling_factors.unsqueeze(-1).expand_as(advantages),
        advantages,
    )

    torch.testing.assert_close(captured["advantages"], expected_advantages)
    assert captured["loss_avg_mode"] == "none"

    union_mask = (entropy_mask | perception_mask).float()
    expected_raw_pg_loss = average_loss(fake_pg_losses, response_mask, mode=actor_config.loss_avg_mode)
    expected_weighted_pg_loss = average_loss(
        fake_pg_losses * union_mask, response_mask, mode=actor_config.loss_avg_mode
    )
    assert metrics["actor/pg_loss"] == pytest.approx(expected_raw_pg_loss.item())
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected_weighted_pg_loss.item())


def test_cgpo_requires_precomputed_advantage_shaping_context():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "top_entropy_quantile": 0.5,
        "entropy_thr_granularity": "response",
        "top_perception_quantile": 0.5,
        "perception_thr_granularity": "response",
        "response_advantage_scaling_method": "cgpo",
        "cgpo_response_scaling_coef": 0.1,
    }

    with pytest.raises(ValueError, match="requires a precomputed advantage_shaping_context"):
        compute_perception_reasoning_policy_loss(
            actor_config=actor_config,
            loss_config=loss_config,
            log_prob=tensors["log_prob"],
            old_log_prob=tensors["old_log_prob"],
            advantages=tensors["advantages"],
            response_mask=tensors["response_mask"],
            entropy=tensors["entropy"],
            decremental_old_log_probs=tensors["decremental_old_log_probs"],
            decremental_entropies=None,
            incremental_old_log_probs=None,
            incremental_entropies=None,
            advantage_shaping_context=None,
        )


def test_dvrp_loss_emits_vision_metrics():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "visual_sensitivity_loss_coef": 0.01,
        "visual_sensitivity_reference": "current",
        "decremental_entropy_coef": 0.05,
        "incremental_entropy_coef": 0.05,
        "visual_robustness_loss_coef": 0.01,
        "incremental_image_transform": "vp_diffusion",
        "entropy_loss_type": "sampled",
        "noise_t": 50.0,
        "noise_beta": 0.05,
    }

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=tensors["decremental_entropies"],
        incremental_old_log_probs=tensors["incremental_old_log_probs"],
        incremental_entropies=tensors["incremental_entropies"],
    )

    assert loss.ndim == 0
    assert "algo/sensitivity/sampled_low_var_kl" in metrics
    assert "algo/dvrp/robustness_kl" in metrics
    assert "algo/dvrp/noise_t" in metrics
    assert "algo/dvrp/noise_beta" in metrics


def test_auxiliary_terms_add_on_top_of_single_pg_loss(monkeypatch):
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "visual_sensitivity_loss_coef": 0.01,
        "visual_sensitivity_reference": "current",
        "decremental_entropy_coef": 0.05,
        "incremental_entropy_coef": 0.05,
        "visual_robustness_loss_coef": 0.01,
        "incremental_image_transform": "vp_diffusion",
        "entropy_loss_type": "sampled",
    }
    fake_pg_losses = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
        dtype=torch.float32,
    )

    def fake_compute_policy_loss(**kwargs):
        del kwargs
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=tensors["decremental_entropies"],
        incremental_old_log_probs=tensors["incremental_old_log_probs"],
        incremental_entropies=tensors["incremental_entropies"],
    )

    expected_pg_loss = average_loss(fake_pg_losses, tensors["response_mask"], mode=actor_config.loss_avg_mode)
    expected_sensitivity_kl = vs_module.compute_sampled_low_var_kl(
        tensors["decremental_old_log_probs"],
        tensors["log_prob"],
        reference_mode="current",
    )
    expected_sensitivity_loss = average_loss(
        -loss_config["visual_sensitivity_loss_coef"] * expected_sensitivity_kl,
        tensors["response_mask"],
        mode=actor_config.loss_avg_mode,
    )
    expected_decremental_entropy_loss = average_loss(
        loss_config["decremental_entropy_coef"]
        * pr_loss_module._compute_entropy_term(
            log_prob=tensors["decremental_old_log_probs"],
            entropy=tensors["decremental_entropies"],
            entropy_loss_type=loss_config["entropy_loss_type"],
        ),
        tensors["response_mask"],
        mode=actor_config.loss_avg_mode,
    )
    expected_robustness_kl = vs_module.compute_sampled_low_var_kl(
        tensors["incremental_old_log_probs"],
        tensors["log_prob"],
        reference_mode="current",
    )
    expected_robustness_loss = average_loss(
        loss_config["visual_robustness_loss_coef"] * expected_robustness_kl,
        tensors["response_mask"],
        mode=actor_config.loss_avg_mode,
    )
    expected_incremental_entropy_loss = average_loss(
        loss_config["incremental_entropy_coef"]
        * pr_loss_module._compute_entropy_term(
            log_prob=tensors["incremental_old_log_probs"],
            entropy=tensors["incremental_entropies"],
            entropy_loss_type=loss_config["entropy_loss_type"],
        ),
        tensors["response_mask"],
        mode=actor_config.loss_avg_mode,
    )
    expected_policy_loss = (
        expected_pg_loss.item()
        + expected_sensitivity_loss.item()
        + expected_decremental_entropy_loss.item()
        + expected_robustness_loss.item()
        + expected_incremental_entropy_loss.item()
    )
    assert metrics["actor/pg_loss"] == pytest.approx(expected_pg_loss.item())
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected_pg_loss.item())
    assert loss.item() == pytest.approx(expected_policy_loss)
    assert metrics["algo/loss/policy_loss"] == pytest.approx(expected_policy_loss)


def test_cgpo_advantage_shaping_context_and_region_mask_metrics():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "top_entropy_quantile": 0.5,
        "entropy_thr_granularity": "micro_batch",
        "top_perception_quantile": 0.5,
        "perception_thr_granularity": "micro_batch",
        "response_advantage_scaling_method": "cgpo",
        "cgpo_response_scaling_coef": 0.1,
        "include_region_tokens_in_perception_mask": True,
    }
    region_token_mask = torch.tensor(
        [[False, True, False, False], [False, False, True, False]],
        dtype=torch.bool,
    )
    batch = DataProto.from_dict(
        tensors={
            "response_mask": tensors["response_mask"],
            "old_log_probs": tensors["old_log_prob"],
            "decremental_old_log_probs": tensors["decremental_old_log_probs"],
        }
    )
    shaping_context = build_sensitivity_advantage_shaping_context(loss_config=loss_config, data=batch)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=tensors["decremental_old_log_probs"],
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        region_token_mask=region_token_mask,
        advantage_shaping_context=shaping_context,
    )

    assert loss.ndim == 0
    assert "algo/cgpo/batch_signal_median" in metrics
    assert "algo/cgpo/batch_signal_mad" in metrics
    assert "algo/token_selection/combined_fraction" in metrics


def test_tor_overlap_tokens_keep_reasoning_weight(monkeypatch):
    actor_config = _make_actor_config()
    log_prob = torch.zeros((2, 3), dtype=torch.float32)
    old_log_prob = torch.zeros((2, 3), dtype=torch.float32)
    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    response_mask = torch.ones((2, 3), dtype=torch.float32)
    entropy = torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]], dtype=torch.float32)
    decremental_old_log_probs = torch.tensor([[1.0, 0.1, 0.2], [0.1, 0.2, 1.0]], dtype=torch.float32)
    loss_config = {
        "top_entropy_quantile": 0.33,
        "entropy_thr_granularity": "response",
        "top_perception_quantile": 0.33,
        "perception_thr_granularity": "response",
        "tor_use_token_weighting": True,
        "tor_rsn_weight": 2.0,
        "tor_prcp_weight": 0.5,
    }
    captured = {}
    fake_pg_losses = torch.tensor([[1.5, 2.5, 3.5], [0.5, 1.0, 1.5]], dtype=torch.float32)

    def fake_compute_policy_loss(**kwargs):
        captured["advantages"] = kwargs["advantages"].clone()
        captured["loss_avg_mode"] = kwargs["loss_avg_mode"]
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        entropy=entropy,
        decremental_old_log_probs=decremental_old_log_probs,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )

    per_token_sensitivity_kl = vs_module.compute_sampled_low_var_kl(
        decremental_old_log_probs,
        log_prob,
        reference_mode="current",
    )
    entropy_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=entropy,
        response_mask=response_mask,
        quantile=loss_config["top_entropy_quantile"],
        granularity=loss_config["entropy_thr_granularity"],
    )
    perception_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=per_token_sensitivity_kl,
        response_mask=response_mask,
        quantile=loss_config["top_perception_quantile"],
        granularity=loss_config["perception_thr_granularity"],
    )
    expected_multiplier = pr_loss_module._compute_tor_weight(
        entropy_mask=entropy_mask,
        perception_mask=perception_mask,
        tor_rsn_weight=loss_config["tor_rsn_weight"],
        tor_prcp_weight=loss_config["tor_prcp_weight"],
    )

    torch.testing.assert_close(captured["advantages"], advantages)
    assert captured["loss_avg_mode"] == "none"
    expected_raw_pg_loss = average_loss(fake_pg_losses, response_mask, mode=actor_config.loss_avg_mode)
    expected_weighted_pg_loss = average_loss(
        fake_pg_losses * expected_multiplier, response_mask, mode=actor_config.loss_avg_mode
    )
    assert metrics["actor/pg_loss"] == pytest.approx(expected_raw_pg_loss.item())
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected_weighted_pg_loss.item())
    assert metrics["algo/tor/effective_token_weight_mean"] == pytest.approx(
        average_loss(expected_multiplier, response_mask, mode=actor_config.loss_avg_mode).item()
    )
    assert expected_multiplier[0, 0].item() == pytest.approx(loss_config["tor_rsn_weight"])


def test_vppo_with_entropy_quantile_applies_union_mask(monkeypatch):
    actor_config = _make_actor_config()
    log_prob = torch.zeros((2, 4), dtype=torch.float32)
    old_log_prob = torch.zeros((2, 4), dtype=torch.float32)
    advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    # entropy top-1 (per row, response granularity, quantile=0.25) → token (0,2) and (1,1)
    entropy = torch.tensor([[0.1, 0.2, 0.9, 0.3], [0.1, 0.8, 0.2, 0.3]], dtype=torch.float32)
    # Driven by decremental - log_prob: picks token (0,0) and (1,3)
    decremental_old_log_probs = torch.tensor(
        [[0.9, 0.0, 0.1, 0.2], [0.0, 0.1, 0.2, 0.9]],
        dtype=torch.float32,
    )
    loss_config = {
        "top_entropy_quantile": 0.25,
        "entropy_thr_granularity": "response",
        "top_perception_quantile": 0.25,
        "perception_thr_granularity": "response",
        "response_advantage_scaling_method": "vppo",
        "vppo_response_scaling_min": 0.85,
    }
    shaping_context = {"global_min_score": 0.0, "global_max_score": 1.0}
    fake_pg_losses = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
        dtype=torch.float32,
    )

    def fake_compute_policy_loss(**kwargs):
        del kwargs
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    _, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        entropy=entropy,
        decremental_old_log_probs=decremental_old_log_probs,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        advantage_shaping_context=shaping_context,
    )

    per_token_sensitivity_kl = vs_module.compute_sampled_low_var_kl(
        decremental_old_log_probs,
        log_prob,
        reference_mode="current",
    )
    entropy_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=entropy,
        response_mask=response_mask,
        quantile=loss_config["top_entropy_quantile"],
        granularity=loss_config["entropy_thr_granularity"],
    )
    perception_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=per_token_sensitivity_kl,
        response_mask=response_mask,
        quantile=loss_config["top_perception_quantile"],
        granularity=loss_config["perception_thr_granularity"],
    )
    union_mask = (entropy_mask | perception_mask).float()
    # Fails under the pre-fix behavior where VPPO step-4 only applied perception_mask.
    assert not torch.equal(union_mask, perception_mask.float())
    expected_weighted_pg_loss = average_loss(
        fake_pg_losses * union_mask, response_mask, mode=actor_config.loss_avg_mode
    )
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected_weighted_pg_loss.item())
    assert metrics["algo/token_selection/combined_fraction"] == pytest.approx(
        union_mask.sum().item() / response_mask.sum().item()
    )


def test_selected_pg_denominator_disabled_uses_response_tokens(monkeypatch):
    actor_config = _make_actor_config()
    log_prob = torch.zeros((2, 4), dtype=torch.float32)
    old_log_prob = torch.zeros((2, 4), dtype=torch.float32)
    advantages = torch.ones((2, 4), dtype=torch.float32)
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    entropy = torch.tensor([[0.1, 0.9, 0.2, 0.3], [0.8, 0.1, 0.4, 0.2]], dtype=torch.float32)
    fake_pg_losses = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    loss_config = {
        "top_entropy_quantile": 0.25,
        "entropy_thr_granularity": "response",
        "normalize_pg_loss_by_selected_tokens": False,
    }

    def fake_compute_policy_loss(**kwargs):
        del kwargs
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        entropy=entropy,
        decremental_old_log_probs=None,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )

    entropy_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=entropy,
        response_mask=response_mask,
        quantile=loss_config["top_entropy_quantile"],
        granularity=loss_config["entropy_thr_granularity"],
    )
    expected = average_loss(fake_pg_losses * entropy_mask.float(), response_mask, mode=actor_config.loss_avg_mode)
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected.item())


def test_selected_pg_denominator_enabled_uses_selected_tokens(monkeypatch):
    actor_config = _make_actor_config()
    log_prob = torch.zeros((2, 4), dtype=torch.float32)
    old_log_prob = torch.zeros((2, 4), dtype=torch.float32)
    advantages = torch.ones((2, 4), dtype=torch.float32)
    response_mask = torch.ones((2, 4), dtype=torch.float32)
    entropy = torch.tensor([[0.1, 0.9, 0.2, 0.3], [0.8, 0.1, 0.4, 0.2]], dtype=torch.float32)
    fake_pg_losses = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    loss_config = {
        "top_entropy_quantile": 0.25,
        "entropy_thr_granularity": "response",
        "normalize_pg_loss_by_selected_tokens": True,
    }

    def fake_compute_policy_loss(**kwargs):
        del kwargs
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config=loss_config,
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        entropy=entropy,
        decremental_old_log_probs=None,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )

    entropy_mask, _, _ = pr_loss_module._compute_top_quantile_mask(
        values=entropy,
        response_mask=response_mask,
        quantile=loss_config["top_entropy_quantile"],
        granularity=loss_config["entropy_thr_granularity"],
    )
    selected_mask = entropy_mask & response_mask.to(torch.bool)
    expected = average_loss(fake_pg_losses * entropy_mask.float(), selected_mask, mode=actor_config.loss_avg_mode)
    response_denominator_loss = average_loss(
        fake_pg_losses * entropy_mask.float(), response_mask, mode=actor_config.loss_avg_mode
    )
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected.item())
    assert loss.item() != pytest.approx(response_denominator_loss.item())


def test_selected_pg_denominator_enabled_falls_back_without_selection(monkeypatch):
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    fake_pg_losses = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
        dtype=torch.float32,
    )

    def fake_compute_policy_loss(**kwargs):
        del kwargs
        return fake_pg_losses.clone(), {"ppo_kl": 0.0, "entropy_loss": 0.0}

    monkeypatch.setattr(pr_loss_module, "compute_policy_loss", fake_compute_policy_loss)

    loss, metrics = compute_perception_reasoning_policy_loss(
        actor_config=actor_config,
        loss_config={"normalize_pg_loss_by_selected_tokens": True},
        log_prob=tensors["log_prob"],
        old_log_prob=tensors["old_log_prob"],
        advantages=tensors["advantages"],
        response_mask=tensors["response_mask"],
        entropy=tensors["entropy"],
        decremental_old_log_probs=None,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
    )

    expected = average_loss(fake_pg_losses, tensors["response_mask"], mode=actor_config.loss_avg_mode)
    assert loss.item() == pytest.approx(expected.item())
    assert metrics["algo/loss/pg_loss_effective"] == pytest.approx(expected.item())


def test_tor_without_any_quantile_raises():
    actor_config = _make_actor_config()
    tensors = _make_common_tensors()
    loss_config = {
        "tor_use_token_weighting": True,
        "tor_rsn_weight": 1.0,
        "tor_prcp_weight": 0.5,
    }
    with pytest.raises(ValueError, match="tor_use_token_weighting"):
        compute_perception_reasoning_policy_loss(
            actor_config=actor_config,
            loss_config=loss_config,
            log_prob=tensors["log_prob"],
            old_log_prob=tensors["old_log_prob"],
            advantages=tensors["advantages"],
            response_mask=tensors["response_mask"],
            entropy=tensors["entropy"],
            decremental_old_log_probs=None,
            decremental_entropies=None,
            incremental_old_log_probs=None,
            incremental_entropies=None,
        )


def test_build_sensitivity_shaping_context_rejects_unknown_method():
    tensors = _make_common_tensors()
    batch = DataProto.from_dict(
        tensors={
            "response_mask": tensors["response_mask"],
            "old_log_probs": tensors["old_log_prob"],
            "decremental_old_log_probs": tensors["decremental_old_log_probs"],
        }
    )
    with pytest.raises(ValueError, match="Unknown response_advantage_scaling_method"):
        build_sensitivity_advantage_shaping_context(
            loss_config={"response_advantage_scaling_method": "bogus"},
            data=batch,
        )


@pytest.mark.parametrize(
    ("response_text_0", "response_text_1"),
    [
        (
            '<think>focus on <region name="cat" image_idx="0" id="0">[[10, 10, 20, 20]]</region> '
            'and <region name="dog" image_idx="1" id="1">[[30, 30, 40, 40]]</region></think>\\boxed{A}',
            '<think>focus on <region name="cat" image_idx="0" id="0">[[10, 10, 20, 20]]</region> '
            "then reason about the cat</think>\\boxed{A}",
        ),
        (
            '<think>focus on {"label": "cat", "image_idx": 0, "bbox_list": [[10, 10, 20, 20]]} '
            'and {"label": "dog", "image_idx": 1, "bbox_list": [[30, 30, 40, 40]]}</think><answer>A</answer>',
            '<think>focus on {"label": "cat", "image_idx": 0, "bbox_list": [[10, 10, 20, 20]]} '
            "then reason about the cat</think><answer>A</answer>",
        ),
    ],
)
def test_grounding_consistency_reward_uses_weighted_rollout_group_scoring(response_text_0, response_text_1):
    tokenizer = _CharTokenizer()
    processor = _CharProcessor()
    scorer = GroundingConsistencyRewardScorer(
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
        reward_weight=0.5,
    )

    response_ids_0 = tokenizer.encode(response_text_0)
    response_ids_1 = tokenizer.encode(response_text_1)
    max_response_len = max(len(response_ids_0), len(response_ids_1))
    response_tensor = torch.tensor(
        [
            response_ids_0 + [tokenizer.pad_token_id] * (max_response_len - len(response_ids_0)),
            response_ids_1 + [tokenizer.pad_token_id] * (max_response_len - len(response_ids_1)),
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1] * len(response_ids_0) + [0] * (max_response_len - len(response_ids_0)),
            [1] * len(response_ids_1) + [0] * (max_response_len - len(response_ids_1)),
        ],
        dtype=torch.long,
    )
    batch = DataProto.from_dict(
        tensors={
            "responses": response_tensor,
            "response_mask": response_mask,
        },
        non_tensors={
            "multi_modal_data": np.array(
                [
                    {
                        "images": [
                            Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)),
                            Image.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8)),
                        ]
                    },
                    {
                        "images": [
                            Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)),
                            Image.fromarray(np.full((4, 4, 3), 64, dtype=np.uint8)),
                        ]
                    },
                ],
                dtype=object,
            ),
            "uid": np.array(["prompt-0", "prompt-0"], dtype=object),
        },
    )

    class _FakeRolloutWG:
        world_size = 1

        def generate_sequences(self, prompts):
            assert all(len(item["images"]) == 1 for item in prompts.non_tensor_batch["multi_modal_data"])
            outputs = [
                "[[10, 10, 20, 20]]",
                "[[30, 30, 40, 40]]",
            ]
            max_len = max(len(tokenizer.encode(text)) for text in outputs)
            response_rows = []
            response_masks = []
            input_rows = []
            attention_rows = []
            position_rows = []
            for text in outputs:
                ids = tokenizer.encode(text)
                pad = [tokenizer.pad_token_id] * (max_len - len(ids))
                response_rows.append(ids + pad)
                response_masks.append([1] * len(ids) + [0] * len(pad))
                input_rows.append(ids + pad)
                attention_rows.append([1] * len(ids) + [0] * len(pad))
                position_rows.append(list(range(max_len)))
            return DataProto.from_dict(
                tensors={
                    "responses": torch.tensor(response_rows, dtype=torch.long),
                    "response_mask": torch.tensor(response_masks, dtype=torch.long),
                    "input_ids": torch.tensor(input_rows, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
                    "position_ids": torch.tensor(position_rows, dtype=torch.long),
                }
            )

    result = scorer.score_batch(batch, _FakeRolloutWG(), {"n": 1, "temperature": 1.0, "top_p": 0.99})

    expected_raw_scores = [1.0, 2.0 / 3.0]
    expected_weighted_scores = [score * 0.5 for score in expected_raw_scores]
    assert result.weighted_scores == pytest.approx(expected_weighted_scores)
    assert torch.isclose(
        result.reward_tensor.sum(),
        torch.tensor(sum(expected_weighted_scores), dtype=torch.float32),
    )
    assert result.metrics["algo/gcr/response_region_count"] == pytest.approx(1.5)
    assert result.metrics["algo/gcr/group_unique_region_count"] == pytest.approx(2.0)
    assert result.metrics["algo/gcr/scored_sample_fraction"] == pytest.approx(1.0)
    assert compute_detection_reward(
        parse_bbox_string("[[10, 10, 20, 20]]"),
        parse_bbox_string("[[10, 10, 20, 20]]"),
    ) == pytest.approx(1.0)


def _make_single_image_grounding_batch(tokenizer, response_text):
    response_ids = tokenizer.encode(response_text)
    response_tensor = torch.tensor([response_ids], dtype=torch.long)
    response_mask = torch.tensor([[1] * len(response_ids)], dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "responses": response_tensor,
            "response_mask": response_mask,
        },
        non_tensors={
            "multi_modal_data": np.array(
                [{"images": [Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))]}],
                dtype=object,
            ),
            "uid": np.array(["prompt-0"], dtype=object),
        },
    )


def test_grounding_dino_helpers_normalize_query_and_boxes():
    assert _normalize_grounding_dino_query("Cat") == "cat."
    assert _normalize_grounding_dino_query("cat.") == "cat."
    assert _normalize_grounding_dino_query("猫") == "猫."
    assert _normalize_grounding_dino_query("  ") == ""
    assert _resolve_grounding_dino_torch_dtype(torch.device("cpu")) == torch.float32
    assert _resolve_grounding_dino_torch_dtype(torch.device("cuda:0")) == torch.float16

    assert _normalize_abs_bbox_to_1000(torch.tensor([10.0, 20.0, 30.0, 60.0]), width=100, height=200) == (
        100,
        100,
        300,
        300,
    )
    assert _normalize_abs_bbox_to_1000([-5, -10, 110, 220], width=100, height=200) == (0, 0, 1000, 1000)
    assert _normalize_abs_bbox_to_1000([10, 10, 10, 20], width=100, height=200) is None


def test_grounding_dino_post_process_supports_transformers_threshold_aliases():
    class _ThresholdProcessor:
        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids=None,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=None,
        ):
            assert threshold == 0.4
            assert text_threshold == 0.3
            assert target_sizes == [(4, 4)]
            return [{"boxes": []}]

    class _BoxThresholdProcessor:
        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids=None,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=None,
        ):
            assert box_threshold == 0.4
            assert text_threshold == 0.3
            assert target_sizes == [(4, 4)]
            return [{"boxes": []}]

    class _KwargsProcessor:
        def post_process_grounded_object_detection(self, **kwargs):
            assert kwargs["box_threshold"] == 0.4
            assert kwargs["text_threshold"] == 0.3
            assert kwargs["target_sizes"] == [(4, 4)]
            return [{"boxes": []}]

    assert _post_process_grounding_dino_object_detection(
        _ThresholdProcessor(),
        outputs=object(),
        input_ids=object(),
        target_sizes=[(4, 4)],
    ) == [{"boxes": []}]
    assert _post_process_grounding_dino_object_detection(
        _BoxThresholdProcessor(),
        outputs=object(),
        input_ids=object(),
        target_sizes=[(4, 4)],
    ) == [{"boxes": []}]
    assert _post_process_grounding_dino_object_detection(
        _KwargsProcessor(),
        outputs=object(),
        input_ids=object(),
        target_sizes=[(4, 4)],
    ) == [{"boxes": []}]


def test_grounding_dino_detection_fail_closed_on_non_oom_batch_error():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))

    class _FailingProcessor:
        def __call__(self, images=None, text=None, return_tensors=None, padding=None):
            raise ValueError("bad image")

    result = _run_grounding_dino_detection_requests(
        indexed_requests=[(0, "cat", image), (1, "cat", image), (2, "dog", image)],
        processor=_FailingProcessor(),
        model=object(),
        device=torch.device("cpu"),
        batch_size=2,
        min_pixels=None,
        max_pixels=None,
    )

    assert result == [(0, []), (1, []), (2, [])]


def test_grounding_dino_detection_reraises_cuda_oom():
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))

    class _CudaOomProcessor:
        def __call__(self, images=None, text=None, return_tensors=None, padding=None):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _run_grounding_dino_detection_requests(
            indexed_requests=[(0, "cat", image)],
            processor=_CudaOomProcessor(),
            model=object(),
            device=torch.device("cpu"),
            batch_size=1,
            min_pixels=None,
            max_pixels=None,
        )


def test_grounding_dino_loader_uses_float16_on_cuda(monkeypatch):
    import transformers

    tokenizer = _CharTokenizer()
    processor = _CharProcessor()
    scorer = GroundingConsistencyRewardScorer(
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
        detector="grounding-dino",
        grounding_dino_device="auto",
    )
    calls = {}

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id):
            calls["processor_model_id"] = model_id
            return object()

    class _FakeGroundingDinoModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, torch_dtype=None):
            calls["model_id"] = model_id
            calls["torch_dtype"] = torch_dtype
            return _FakeGroundingDinoModel()

    monkeypatch.setattr(transformers, "AutoProcessor", _FakeAutoProcessor)
    monkeypatch.setattr(transformers, "AutoModelForZeroShotObjectDetection", _FakeAutoModel)
    monkeypatch.setattr(scorer, "_resolve_grounding_dino_device", lambda: torch.device("cuda:0"))

    _processor, _model, device = scorer._get_grounding_dino()

    assert device == torch.device("cuda:0")
    assert calls["torch_dtype"] == torch.float16
    assert calls["device"] == torch.device("cuda:0")
    assert calls["eval"] is True


def test_grounding_dino_detection_batches_and_deduplicates_requests(monkeypatch):
    tokenizer = _CharTokenizer()
    processor = _CharProcessor()
    scorer = GroundingConsistencyRewardScorer(
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
        detector="grounding-dino",
        grounding_dino_device="auto",
        grounding_dino_batch_size=2,
    )
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))

    class _FakeInputs(dict):
        def __init__(self, batch_size):
            input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
            super().__init__(input_ids=input_ids)
            self.input_ids = input_ids

        def to(self, device):
            return self

    class _FakeProcessor:
        def __init__(self):
            self.calls = []

        def __call__(self, images=None, text=None, return_tensors=None, padding=None):
            self.calls.append((len(images), list(text), padding))
            return _FakeInputs(len(images))

        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids=None,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=None,
        ):
            return [
                {"boxes": [torch.tensor([0.0, 0.0, float(width), float(height)])]} for height, width in target_sizes
            ]

    class _FakeModel:
        def __call__(self, **kwargs):
            return object()

    fake_processor = _FakeProcessor()
    release_calls = []
    monkeypatch.setattr(scorer, "_get_grounding_dino", lambda: (fake_processor, _FakeModel(), torch.device("cpu")))
    monkeypatch.setattr(scorer, "release_grounding_dino", lambda: release_calls.append("release"))

    box_lists = scorer._detect_with_grounding_dino(
        [
            {"region_name": "cat", "image": image},
            {"region_name": "cat", "image": image},
            {"region_name": "dog", "image": image},
        ]
    )

    assert fake_processor.calls == [(2, ["cat.", "dog."], True)]
    assert box_lists == [
        [(0, 0, 1000, 1000)],
        [(0, 0, 1000, 1000)],
        [(0, 0, 1000, 1000)],
    ]
    assert box_lists[0] is not box_lists[1]
    assert release_calls == ["release"]


def test_grounding_dino_worker_detection_shards_requests_across_worker_group():
    tokenizer = _CharTokenizer()
    processor = _CharProcessor()
    scorer = GroundingConsistencyRewardScorer(
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
        detector="grounding-dino",
        grounding_dino_batch_size=3,
    )
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    requests = [{"region_name": f"obj-{idx}", "image": image} for idx in range(5)]

    class _FakeWorkerGroup:
        world_size = 2

        def __init__(self):
            self.request_shards = None
            self.batch_sizes = None
            self.min_pixels = None
            self.max_pixels = None

        def detect_grounding_dino(self, request_shards, batch_sizes, min_pixels, max_pixels):
            self.request_shards = request_shards
            self.batch_sizes = batch_sizes
            self.min_pixels = min_pixels
            self.max_pixels = max_pixels
            outputs = []
            for shard in request_shards:
                shard_outputs = []
                for request in shard:
                    request_idx = request["request_idx"]
                    shard_outputs.append((request_idx, [(request_idx, 0, request_idx + 1, 1)]))
                outputs.append(shard_outputs)
            return outputs

    worker_group = _FakeWorkerGroup()
    box_lists = scorer._detect_with_grounding_dino_workers(requests, worker_group)

    assert [[request["request_idx"] for request in shard] for shard in worker_group.request_shards] == [
        [0, 2, 4],
        [1, 3],
    ]
    assert worker_group.batch_sizes == [3, 3]
    assert worker_group.min_pixels == [None, None]
    assert worker_group.max_pixels == [None, None]
    assert box_lists == [
        [(0, 0, 1, 1)],
        [(1, 0, 2, 1)],
        [(2, 0, 3, 1)],
        [(3, 0, 4, 1)],
        [(4, 0, 5, 1)],
    ]


def test_grounding_dino_worker_detection_releases_detector(monkeypatch):
    worker = object.__new__(FSDPWorker)
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    release_calls = []
    run_calls = []

    def fake_run(indexed_requests, processor, model, device, batch_size, min_pixels, max_pixels):
        run_calls.append(
            {
                "indexed_requests": indexed_requests,
                "processor": processor,
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
        return [(7, [(0, 0, 1000, 1000)])]

    monkeypatch.setattr(grounding_consistency_module, "_run_grounding_dino_detection_requests", fake_run)
    monkeypatch.setattr(worker, "_get_grounding_dino_detector", lambda: ("processor", "model", torch.device("cpu")))
    monkeypatch.setattr(worker, "release_grounding_dino", lambda: release_calls.append("release"))

    outputs = FSDPWorker.detect_grounding_dino(
        worker,
        requests=[{"request_idx": 7, "region_name": "cat", "image": image}],
        batch_size=3,
        min_pixels=16,
        max_pixels=64,
    )

    assert outputs == [(7, [(0, 0, 1000, 1000)])]
    assert release_calls == ["release"]
    assert run_calls == [
        {
            "indexed_requests": [(7, "cat", image)],
            "processor": "processor",
            "model": "model",
            "device": torch.device("cpu"),
            "batch_size": 3,
            "min_pixels": 16,
            "max_pixels": 64,
        }
    ]


def test_grounding_consistency_reward_grounding_dino_uses_external_detector(monkeypatch):
    tokenizer = _CharTokenizer()
    processor = _CharProcessor()
    scorer = GroundingConsistencyRewardScorer(
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
        reward_weight=0.5,
        detector="grounding-dino",
        grounding_dino_device="auto",
    )
    batch = _make_single_image_grounding_batch(
        tokenizer,
        '<think>focus on {"label": "cat", "bbox_list": [[10, 10, 20, 20]]}</think><answer>A</answer>',
    )

    def fake_grounding_dino(requests):
        assert [request["region_name"] for request in requests] == ["cat"]
        return [[(10, 10, 20, 20)]]

    monkeypatch.setattr(scorer, "_detect_with_grounding_dino", fake_grounding_dino)

    class _FailingRolloutWG:
        world_size = 1

        def generate_sequences(self, prompts):
            raise AssertionError("grounding-dino detector must not call the rollout worker")

    result = scorer.score_batch(batch, _FailingRolloutWG(), {"n": 1})

    assert result.weighted_scores == pytest.approx([0.5])
    assert torch.isclose(result.reward_tensor.sum(), torch.tensor(0.5, dtype=torch.float32))
    assert result.metrics["algo/gcr/response_region_count"] == pytest.approx(1.0)
    assert result.metrics["algo/gcr/group_unique_region_count"] == pytest.approx(1.0)
    assert result.metrics["algo/gcr/scored_sample_fraction"] == pytest.approx(1.0)


def test_grounding_consistency_reward_grounding_dino_empty_detection_is_fail_closed(monkeypatch):
    tokenizer = _CharTokenizer()
    processor = _CharProcessor()
    scorer = GroundingConsistencyRewardScorer(
        tokenizer=tokenizer,
        processor=processor,
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
        reward_weight=0.5,
        detector="grounding-dino",
        grounding_dino_device="auto",
    )
    batch = _make_single_image_grounding_batch(
        tokenizer,
        '<think>focus on {"label": "cat", "bbox_list": [[10, 10, 20, 20]]}</think><answer>A</answer>',
    )

    def fake_empty_grounding_dino(requests):
        return [[] for _ in requests]

    monkeypatch.setattr(scorer, "_detect_with_grounding_dino", fake_empty_grounding_dino)

    class _FailingRolloutWG:
        world_size = 1

        def generate_sequences(self, prompts):
            raise AssertionError("grounding-dino detector must not call the rollout worker")

    result = scorer.score_batch(batch, _FailingRolloutWG(), {"n": 1})

    assert result.weighted_scores == pytest.approx([0.0])
    assert torch.isclose(result.reward_tensor.sum(), torch.tensor(0.0, dtype=torch.float32))
    assert result.metrics["algo/gcr/group_unique_region_count"] == pytest.approx(0.0)
    assert result.metrics["algo/gcr/scored_sample_fraction"] == pytest.approx(0.0)


def test_max_match_iou_sum_empty():
    assert _max_match_iou_sum([], []) == 0.0
    assert _max_match_iou_sum([(0, 0, 10, 10)], []) == 0.0
    assert _max_match_iou_sum([], [(0, 0, 10, 10)]) == 0.0


def test_max_match_iou_sum_picks_optimal_permutation():
    # swapped pred order — optimal matching must pair (0,0,10,10)↔(0,0,10,10)
    # and (100,...)↔(100,...), scoring 2.0 rather than the off-diagonal 0.0
    gt = [(0, 0, 10, 10), (100, 100, 110, 110)]
    pred = [(100, 100, 110, 110), (0, 0, 10, 10)]
    assert _max_match_iou_sum(gt, pred) == pytest.approx(2.0)


def test_max_match_iou_sum_rectangular():
    gt = [(0, 0, 10, 10)]
    pred = [(0, 0, 10, 10), (100, 100, 110, 110), (200, 200, 210, 210)]
    assert _max_match_iou_sum(gt, pred) == pytest.approx(1.0)

    gt = [(0, 0, 10, 10), (50, 50, 60, 60), (100, 100, 110, 110)]
    pred = [(50, 50, 60, 60)]
    assert _max_match_iou_sum(gt, pred) == pytest.approx(1.0)


def test_max_match_iou_sum_large_input_finishes_quickly():
    # Regression guard: the previous bitmask-DP implementation was exponential
    # in min(|gt|, |pred|) and would effectively hang beyond ~30 boxes.
    import time

    boxes = [(i * 5, 0, i * 5 + 3, 3) for i in range(200)]
    start = time.perf_counter()
    score = _max_match_iou_sum(boxes, boxes)
    elapsed = time.perf_counter() - start
    assert score == pytest.approx(200.0)
    assert elapsed < 2.0, f"_max_match_iou_sum too slow on 200x200 input: {elapsed:.3f}s"


def _make_grounding_batch(tokenizer, response_texts, uids=None, num_images=1):
    encoded = [tokenizer.encode(text) for text in response_texts]
    max_len = max(len(ids) for ids in encoded)
    response_rows = []
    mask_rows = []
    for ids in encoded:
        pad = [tokenizer.pad_token_id] * (max_len - len(ids))
        response_rows.append(ids + pad)
        mask_rows.append([1] * len(ids) + [0] * len(pad))
    if uids is None:
        uids = ["prompt-0"] * len(response_texts)
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor(response_rows, dtype=torch.long),
            "response_mask": torch.tensor(mask_rows, dtype=torch.long),
        },
        non_tensors={
            "multi_modal_data": np.array(
                [
                    {"images": [Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8)) for _ in range(num_images)]}
                    for _ in response_texts
                ],
                dtype=object,
            ),
            "uid": np.array(uids, dtype=object),
        },
    )


def test_parse_xml_response_regions_filters_invalid_boxes():
    response = (
        "<think>"
        '<region name="cat" image_idx="0" id="0">[[1100, 1100, 1500, 1500]]</region>'
        '<region name="dog" image_idx="0" id="1">[[300, 300, 100, 100]]</region>'
        '<region name="bird" image_idx="0" id="2">[[10, 10, 20, 20], [30, 30, 20, 40]]</region>'
        "</think><answer>A</answer>"
    )
    regions = parse_response_regions(response, num_images=1)

    # out-of-range and inverted boxes are dropped; regions keep only their valid boxes
    assert [(region.name, region.boxes) for region in regions] == [("bird", [[10, 10, 20, 20]])]


def test_parse_bbox_string_filters_invalid_detector_boxes():
    # self-detector pseudo-GT: out-of-range and inverted boxes must not become anchors
    text = "[[10, 10, 20, 20], [1100, 1100, 1500, 1500], [300, 300, 100, 100]]"
    assert parse_bbox_string(text) == [(10, 10, 20, 20)]

    # all boxes invalid -> empty pseudo-GT, region is excluded from GCR scoring
    assert parse_bbox_string("[[0, 0, 1400, 1050]]") == []


class _CountingFakeRolloutWG:
    world_size = 1

    def __init__(self, tokenizer, detection_text="[[10, 10, 20, 20]]"):
        self.tokenizer = tokenizer
        self.detection_text = detection_text
        self.request_counts = []

    def generate_sequences(self, prompts):
        num_prompts = len(prompts.non_tensor_batch["multi_modal_data"])
        self.request_counts.append(num_prompts)
        ids = self.tokenizer.encode(self.detection_text)
        rows = torch.tensor([ids] * num_prompts, dtype=torch.long)
        ones = torch.ones_like(rows)
        return DataProto.from_dict(
            tensors={
                "responses": rows,
                "response_mask": ones,
                "input_ids": rows,
                "attention_mask": ones,
                "position_ids": torch.arange(rows.shape[1]).unsqueeze(0).repeat(num_prompts, 1),
            }
        )


def _make_gcr_scorer(tokenizer):
    return GroundingConsistencyRewardScorer(
        tokenizer=tokenizer,
        processor=_CharProcessor(),
        max_prompt_length=256,
        min_pixels=None,
        max_pixels=None,
        video_fps=2.0,
        reward_weight=1.0,
    )


_GCR_TEST_ROLLOUT_CONFIG = {"n": 1, "temperature": 1.0, "top_p": 0.99}
_GCR_TEST_RESPONSE = (
    '<think>{"label": "cat", "image_idx": 0, "bbox_list": [[10, 10, 20, 20]]}</think><answer>A</answer>'
)


def test_compute_group_eligibility_mask_marks_groups_with_a_correct_answer():
    assert compute_group_eligibility_mask(["a", "a", "b", "b"], [0.0, 1.0, 0.0, 0.0]) == [
        True,
        True,
        False,
        False,
    ]
    with pytest.raises(ValueError, match="align"):
        compute_group_eligibility_mask(["a"], [1.0, 0.0])


def test_grounding_consistency_scorer_eligibility_mask_skips_ineligible_groups():
    tokenizer = _CharTokenizer()
    scorer = _make_gcr_scorer(tokenizer)
    batch = _make_grounding_batch(tokenizer, [_GCR_TEST_RESPONSE, _GCR_TEST_RESPONSE], uids=["g0", "g0"])
    wg = _CountingFakeRolloutWG(tokenizer)

    # a group without any correct answer is skipped entirely: no detection, zero scores
    result = scorer.score_batch(batch, wg, _GCR_TEST_ROLLOUT_CONFIG, eligible_sample_mask=[False, False])
    assert wg.request_counts == []
    assert result.weighted_scores == [0.0, 0.0]
    assert result.metrics["algo/gcr/detection_request_count"] == 0.0

    # one correct answer makes the whole group eligible; every sample in it is scored,
    # and the scores match the unmasked computation exactly
    masked = scorer.score_batch(batch, wg, _GCR_TEST_ROLLOUT_CONFIG, eligible_sample_mask=[True, False])
    unmasked = scorer.score_batch(batch, wg, _GCR_TEST_ROLLOUT_CONFIG)
    assert masked.weighted_scores == pytest.approx(unmasked.weighted_scores)
    assert masked.weighted_scores[1] > 0.0
    assert wg.request_counts == [1, 1]

    with pytest.raises(ValueError, match="eligible_sample_mask"):
        scorer.score_batch(batch, wg, _GCR_TEST_ROLLOUT_CONFIG, eligible_sample_mask=[True])


def test_grounding_consistency_scorer_deduplicates_identical_requests_across_groups():
    tokenizer = _CharTokenizer()
    scorer = _make_gcr_scorer(tokenizer)
    # two groups whose images are distinct PIL objects with identical content and the same
    # region name: content-based dedup must collapse them into one detection request
    batch = _make_grounding_batch(tokenizer, [_GCR_TEST_RESPONSE, _GCR_TEST_RESPONSE], uids=["g0", "g1"])
    wg = _CountingFakeRolloutWG(tokenizer)

    result = scorer.score_batch(batch, wg, _GCR_TEST_ROLLOUT_CONFIG)

    assert wg.request_counts == [1]
    assert result.weighted_scores == pytest.approx([1.0, 1.0])
    assert result.metrics["algo/gcr/detection_request_count"] == 1.0


def test_grounding_consistency_scorer_pseudo_gt_cache_scoped_by_policy_token():
    tokenizer = _CharTokenizer()
    scorer = _make_gcr_scorer(tokenizer)
    wg = _CountingFakeRolloutWG(tokenizer)

    first = scorer.score_batch(
        _make_grounding_batch(tokenizer, [_GCR_TEST_RESPONSE]),
        wg,
        _GCR_TEST_ROLLOUT_CONFIG,
        cache_token="train-step-1",
    )
    assert wg.request_counts == [1]

    # same token, fresh but content-identical batch: served from cache, no generation at all
    second = scorer.score_batch(
        _make_grounding_batch(tokenizer, [_GCR_TEST_RESPONSE]),
        wg,
        _GCR_TEST_ROLLOUT_CONFIG,
        cache_token="train-step-1",
    )
    assert wg.request_counts == [1]
    assert second.weighted_scores == pytest.approx(first.weighted_scores)

    # new token means new policy weights: the cache is dropped and detection reruns
    scorer.score_batch(
        _make_grounding_batch(tokenizer, [_GCR_TEST_RESPONSE]),
        wg,
        _GCR_TEST_ROLLOUT_CONFIG,
        cache_token="train-step-2",
    )
    assert wg.request_counts == [1, 1]

    # no token disables cross-call reuse entirely
    scorer.score_batch(
        _make_grounding_batch(tokenizer, [_GCR_TEST_RESPONSE]),
        wg,
        _GCR_TEST_ROLLOUT_CONFIG,
    )
    assert wg.request_counts == [1, 1, 1]


def test_gcr_image_content_key_does_not_retain_images():
    import gc
    import weakref

    scorer = _make_gcr_scorer(_CharTokenizer())
    image = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))
    twin = Image.fromarray(np.full((4, 4, 3), 255, dtype=np.uint8))

    key = scorer._image_content_key(image)
    assert scorer._image_content_key(image) == key  # stable per object via the id map
    assert scorer._image_content_key(twin) == key  # content-identical objects share a key

    watcher = weakref.ref(image)
    del image
    gc.collect()
    assert watcher() is None  # the key cache must not keep images alive


def test_gcr_image_content_key_handles_non_pil_inputs():
    scorer = _make_gcr_scorer(_CharTokenizer())

    array = np.full((4, 4, 3), 128, dtype=np.uint8)
    array_key = scorer._image_content_key(array)
    assert scorer._image_content_key(array) == array_key
    assert scorer._image_content_key(np.full((4, 4, 3), 128, dtype=np.uint8)) == array_key
    # same bytes but different shape must not collide
    assert scorer._image_content_key(np.full((48,), 128, dtype=np.uint8)) != array_key

    class _NoBytes:
        pass

    first = _NoBytes()
    second = _NoBytes()
    first_key = scorer._image_content_key(first)
    assert scorer._image_content_key(first) == first_key  # stable while the object lives
    assert scorer._image_content_key(second) != first_key  # never dedups, never collides


def test_gcr_image_content_key_never_stores_strong_references():
    scorer = _make_gcr_scorer(_CharTokenizer())

    class _SlottedImage:  # __slots__ without __weakref__: not weakrefable, but hashable content
        __slots__ = ("payload",)

        def __init__(self, payload):
            self.payload = payload

        def tobytes(self):
            return self.payload

    # key is recomputed per lookup, stays deterministic, and the object is never memoized
    slotted = _SlottedImage(b"\x80" * 48)
    slotted_key = scorer._image_content_key(slotted)
    assert scorer._image_content_key(_SlottedImage(b"\x80" * 48)) == slotted_key
    assert id(slotted) not in scorer._image_key_by_id

    # non-weakrefable and non-hashable content (bytes): unique key per lookup, never retained
    blob = b"\x80" * 48
    assert scorer._image_content_key(blob) != scorer._image_content_key(blob)
    assert id(blob) not in scorer._image_key_by_id


def test_gcr_image_content_key_includes_dtype():
    scorer = _make_gcr_scorer(_CharTokenizer())

    # identical bytes and shape but different dtype must not collide
    uint8_key = scorer._image_content_key(np.zeros((4,), dtype=np.uint8))
    int8_key = scorer._image_content_key(np.zeros((4,), dtype=np.int8))
    assert uint8_key != int8_key


def test_abs_log_ratio_is_symmetric():
    reference = torch.tensor([[-1.0, -2.0, -0.5]])
    corrupted = torch.tensor([[-2.0, -1.0, -0.5]])
    scores = vs_module.compute_sampled_sensitivity_scores(
        "sampled_abs_log_ratio", corrupted, reference, reference_mode="old"
    )
    assert torch.allclose(scores, torch.tensor([[1.0, 1.0, 0.0]]))
