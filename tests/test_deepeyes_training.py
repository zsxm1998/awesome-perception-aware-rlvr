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

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from vllm import SamplingParams

from verl.models.transformers import position_ids as position_ids_module
from verl.models.transformers.internvl import InternVLProcessorAdapter
from verl.protocol import DataProto
from verl.trainer.config import PPOConfig
from verl.trainer.core_algos import compute_policy_loss
from verl.trainer.metrics import compute_data_metrics
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.agent import (
    AgentImageConfig,
    AgentLoop,
    AgentLoopConfig,
    AgentSeedContext,
    AgentStatus,
    AgentTrajectoryMaterializer,
    AgentTrajectoryRejected,
    EncodedObservation,
    GenerationOutput,
    MaterializedAgentTrajectory,
    collate_materialized_agent_trajectories,
    derive_agent_seed,
)
from verl.workers.agent.backends import VLLMAgentBatchScheduler
from verl.workers.agent.inference import run_deepeyes_inference
from verl.workers.agent.protocol import AgentMetrics, AgentStep, AgentTrajectory
from verl.workers.agent.tools import ImageZoomInTool, ToolRegistry
from verl.workers.agent.trajectory import measure_expanded_observation
from verl.workers.fsdp_workers import FSDPWorker
from verl.workers.rollout.config import RolloutConfig
from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(str(int(token_id)) for token_id in token_ids)


class _FakeQwenImageProcessor:
    merge_size = 1

    def __init__(self):
        self.calls = []

    def __call__(self, images, return_tensors="pt"):
        assert return_tensors == "pt"
        assert len(images) >= 1
        self.calls.extend(images)
        return {
            "pixel_values": torch.ones((4 * len(images), 3, 1, 1), dtype=torch.float32),
            "image_grid_thw": torch.tensor(
                [[1, 2, 2]] * len(images),
                dtype=torch.long,
            ),
        }

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        assert height > 0 and width > 0
        assert images_kwargs == {}
        return 4


class _FakeQwen3VLProcessor:
    image_token_id = 41
    video_token_id = 43
    vision_start_token_id = 40

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeQwenImageProcessor()


def test_shared_multimodal_position_helper_forwards_all_qwen_layout_metadata(monkeypatch):
    captured = {}

    def fake_get_rope_index(processor, **kwargs):
        captured.update(kwargs)
        return torch.ones((3, kwargs["input_ids"].numel()), dtype=torch.long)

    monkeypatch.setattr(
        position_ids_module,
        "_get_qwen_rope_index",
        lambda processor: fake_get_rope_index,
    )
    input_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    attention_mask = torch.tensor([0, 1, 1], dtype=torch.long)
    second_per_grid_ts = torch.tensor([0.5])
    result = position_ids_module.build_multimodal_position_ids(
        _FakeQwen3VLProcessor(),
        input_ids=input_ids,
        attention_mask=attention_mask,
        multi_modal_inputs={"second_per_grid_ts": second_per_grid_ts},
    )

    assert result.shape == (4, 3)
    assert result[0].tolist() == [0, 0, 1]
    assert captured["second_per_grid_ts"] is second_per_grid_ts


def _make_trajectory(num_crops: int) -> AgentTrajectory:
    prompt_ids = [1, 40, 41, 42, 2]
    response_ids = []
    response_mask = []
    observation_images = []
    steps = []
    for crop_index in range(num_crops):
        action_id = 10 + crop_index
        observation_ids = [50, 40, 41, 42, 51]
        response_ids.append(action_id)
        response_mask.append(1)
        response_ids.extend(observation_ids)
        response_mask.extend([0] * len(observation_ids))
        observation_images.append(Image.new("RGB", (8, 8), color=(crop_index, 0, 0)))
        steps.append(
            AgentStep(
                model_text=f"tool-{crop_index}",
                model_token_ids=[action_id],
                observation_token_ids=observation_ids,
                observation_committed=True,
            )
        )

    response_ids.append(30)
    response_mask.append(1)
    steps.append(AgentStep(model_text="<answer>x</answer>", model_token_ids=[30]))
    return AgentTrajectory(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        source_images=[Image.new("RGB", (8, 8), color="white")],
        observation_images=observation_images,
        steps=steps,
        status=AgentStatus.ANSWERED,
        metrics=AgentMetrics(visual_tokens=4 * num_crops),
        final_answer="x",
        response_token_budget=256,
        image_config=AgentImageConfig(limit_images=7),
    )


@pytest.mark.parametrize("num_crops", [0, 1, 6])
def test_qwen_agent_trajectory_materialization_preserves_actions_and_recomputes_mrope(num_crops):
    processor = _FakeQwen3VLProcessor()
    trajectory = _make_trajectory(num_crops)
    materialized = AgentTrajectoryMaterializer(
        processor,
        max_prompt_tokens=64,
        max_model_len=512,
    ).materialize(trajectory)

    expected_images = 1 + num_crops
    raw_action_ids = [
        token_id for token_id, mask_value in zip(trajectory.response_ids, trajectory.response_mask) if mask_value
    ]
    expanded_action_ids = [
        token_id for token_id, mask_value in zip(materialized.response_ids, materialized.response_mask) if mask_value
    ]

    assert isinstance(materialized, MaterializedAgentTrajectory)
    assert expanded_action_ids == raw_action_ids
    assert materialized.visual_token_counts == (4,) * expected_images
    assert sum(materialized.visual_token_mask) == 4 * expected_images
    assert materialized.input_ids.count(processor.image_token_id) == 4 * expected_images
    assert materialized.multi_modal_layout["kind"] == "qwen"
    assert materialized.multi_modal_layout["image_grid_thw"] == ((1, 2, 2),) * expected_images
    assert materialized.position_ids.shape == (4, len(materialized.input_ids))
    assert materialized.position_ids[0].tolist() == list(range(len(materialized.input_ids)))
    assert all(
        materialized.response_mask[index] == 0
        for start, end in materialized.image_spans[1:]
        for index in range(start - len(materialized.prompt_ids), end - len(materialized.prompt_ids))
    )
    if num_crops == 0:
        assert materialized.prompt_ids == (1, 40, 41, 41, 41, 41, 42, 2)
        assert materialized.response_ids == (30,)


def test_qwen_observation_layout_does_not_materialize_pixel_tensors():
    processor = _FakeQwen3VLProcessor()

    effective_tokens, visual_tokens = measure_expanded_observation(
        processor,
        [10, processor.image_token_id, 11],
        [Image.new("RGB", (8, 8))],
    )

    assert effective_tokens == 6
    assert visual_tokens == 4
    assert processor.image_processor.calls == []


class _FakeInternVLTokenizer:
    pad_token_id = 0

    _TOKENS = {
        "<image>": [60],
        "<img>": [61],
        "</img>": [62],
    }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text not in self._TOKENS:
            raise AssertionError(f"unexpected literal {text!r}")
        return list(self._TOKENS[text])


class _FakeInternVLImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        assert return_tensors == "pt"
        assert len(images) == 1
        return {
            "pixel_values": torch.ones((2, 3, 1, 1), dtype=torch.float32),
            "image_flags": torch.ones((2, 1), dtype=torch.long),
        }


def _fake_internvl_processor():
    processor = object.__new__(InternVLProcessorAdapter)
    processor.tokenizer = _FakeInternVLTokenizer()
    processor.image_processor = _FakeInternVLImageProcessor()
    processor.image_token_id = 63
    processor.num_image_token = 2
    processor.get_num_image_patches = lambda image: 2
    return processor


def test_internvl_materialization_expands_dynamic_tiles_without_mrope():
    trajectory = AgentTrajectory(
        prompt_ids=[1, 60, 2],
        response_ids=[10, 70, 60, 71, 20],
        response_mask=[1, 0, 0, 0, 1],
        source_images=[Image.new("RGB", (8, 8), color="white")],
        observation_images=[Image.new("RGB", (8, 8), color="black")],
        steps=[],
        status=AgentStatus.ANSWERED,
        metrics=AgentMetrics(visual_tokens=4),
        final_answer="x",
        response_token_budget=32,
    )

    materialized = AgentTrajectoryMaterializer(
        _fake_internvl_processor(),
        max_prompt_tokens=32,
    ).materialize(trajectory)

    assert materialized.visual_token_counts == (4, 4)
    assert materialized.prompt_ids == (1, 61, 63, 63, 63, 63, 62, 2)
    assert materialized.response_ids == (10, 70, 61, 63, 63, 63, 63, 62, 71, 20)
    assert materialized.response_mask == (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    assert materialized.position_ids.ndim == 1
    assert materialized.position_ids.tolist() == list(range(len(materialized.input_ids)))
    assert materialized.multi_modal_layout == {
        "kind": "internvl",
        "pixel_values_batch": 4,
    }


def test_internvl_observation_layout_does_not_materialize_pixel_tensors():
    processor = _fake_internvl_processor()

    effective_tokens, visual_tokens = measure_expanded_observation(
        processor,
        [70, 60, 71],
        [Image.new("RGB", (8, 8))],
    )

    assert effective_tokens == 8
    assert visual_tokens == 4


def test_internvl_action_may_quote_image_literal_before_environment_observation():
    trajectory = AgentTrajectory(
        prompt_ids=[1, 60, 2],
        response_ids=[60, 10, 70, 60, 71, 20],
        response_mask=[1, 1, 0, 0, 0, 1],
        source_images=[Image.new("RGB", (8, 8), color="white")],
        observation_images=[Image.new("RGB", (8, 8), color="black")],
        steps=[],
        status=AgentStatus.ANSWERED,
        metrics=AgentMetrics(visual_tokens=4),
        final_answer="x",
        response_token_budget=32,
    )

    materialized = AgentTrajectoryMaterializer(
        _fake_internvl_processor(),
        max_prompt_tokens=32,
    ).materialize(trajectory)

    assert materialized.response_ids[0] == 60
    assert [
        token_id
        for token_id, mask_value in zip(
            materialized.response_ids,
            materialized.response_mask,
        )
        if mask_value == 1
    ] == [60, 10, 20]
    assert materialized.response_ids.count(63) == 4


def test_materializer_rejects_model_generated_reserved_visual_token():
    trajectory = _make_trajectory(0)
    trajectory.response_ids = [41]
    trajectory.response_mask = [1]

    with pytest.raises(
        AgentTrajectoryRejected,
        match="reserved visual context token",
    ) as error:
        AgentTrajectoryMaterializer(
            _FakeQwen3VLProcessor(),
            max_prompt_tokens=64,
        ).materialize(trajectory)
    assert error.value.recoverable is True


def test_materializer_rejects_expanded_response_and_image_capacity_overflow():
    trajectory = _make_trajectory(1)
    processor = _FakeQwen3VLProcessor()

    trajectory.response_token_budget = len(trajectory.response_ids)
    with pytest.raises(
        AgentTrajectoryRejected,
        match="expanded agent response exceeds",
    ) as error:
        AgentTrajectoryMaterializer(
            processor,
            max_prompt_tokens=64,
        ).materialize(trajectory)
    assert error.value.recoverable is False

    trajectory.response_token_budget = 256
    trajectory.image_config = AgentImageConfig(limit_images=1)
    with pytest.raises(AgentTrajectoryRejected, match="limit_images=1") as error:
        AgentTrajectoryMaterializer(
            processor,
            max_prompt_tokens=64,
        ).materialize(trajectory)
    assert error.value.recoverable is False


def test_materializer_requires_agent_loop_response_budget():
    trajectory = _make_trajectory(0)
    trajectory.response_token_budget = None

    with pytest.raises(ValueError, match="positive response_token_budget"):
        AgentTrajectoryMaterializer(
            _FakeQwen3VLProcessor(),
            max_prompt_tokens=64,
        ).materialize(trajectory)


def test_agent_image_config_normalizes_unbounded_limit_and_materializer_rejects_prompt_overflow():
    assert AgentImageConfig(limit_images=0).limit_images is None
    with pytest.raises(ValueError, match="non-negative"):
        AgentImageConfig(limit_images=-1)

    with pytest.raises(AgentTrajectoryRejected, match="expanded agent prompt"):
        AgentTrajectoryMaterializer(
            _FakeQwen3VLProcessor(),
            max_prompt_tokens=7,
        ).materialize(_make_trajectory(0))


def test_collator_rejects_local_or_mismatched_padding_widths():
    sample = AgentTrajectoryMaterializer(
        _FakeQwen3VLProcessor(),
        max_prompt_tokens=64,
    ).materialize(_make_trajectory(0))

    with pytest.raises(ValueError, match="global prompt padded width"):
        collate_materialized_agent_trajectories(
            [sample],
            pad_token_id=0,
            padded_prompt_width=63,
            padded_response_width=256,
            uids=["sample"],
        )
    with pytest.raises(ValueError, match="global response padded width"):
        collate_materialized_agent_trajectories(
            [sample],
            pad_token_id=0,
            padded_prompt_width=64,
            padded_response_width=255,
            uids=["sample"],
        )


def test_collator_requires_explicit_group_uids():
    sample = AgentTrajectoryMaterializer(
        _FakeQwen3VLProcessor(),
        max_prompt_tokens=64,
    ).materialize(_make_trajectory(0))

    with pytest.raises(TypeError, match="uids"):
        collate_materialized_agent_trajectories(
            [sample],
            pad_token_id=0,
            padded_prompt_width=64,
            padded_response_width=256,
        )

    group = collate_materialized_agent_trajectories(
        [sample] * 8,
        pad_token_id=0,
        padded_prompt_width=64,
        padded_response_width=256,
        uids=["prompt-3"] * 8,
    )
    assert group.non_tensor_batch["uid"].tolist() == ["prompt-3"] * 8


@pytest.mark.parametrize("uids", [[], ["sample-0", "sample-1"]])
def test_collator_rejects_uid_count_mismatch(uids):
    sample = AgentTrajectoryMaterializer(
        _FakeQwen3VLProcessor(),
        max_prompt_tokens=64,
    ).materialize(_make_trajectory(0))

    with pytest.raises(ValueError, match="uids must align one-to-one"):
        collate_materialized_agent_trajectories(
            [sample],
            pad_token_id=0,
            padded_prompt_width=64,
            padded_response_width=256,
            uids=uids,
        )


def test_materialized_batch_aligns_dataproto_masks_positions_and_compact_vision_layout():
    processor = _FakeQwen3VLProcessor()
    materializer = AgentTrajectoryMaterializer(processor, max_prompt_tokens=64)
    samples = [materializer.materialize(_make_trajectory(0)), materializer.materialize(_make_trajectory(1))]
    batch = collate_materialized_agent_trajectories(
        samples,
        pad_token_id=processor.tokenizer.pad_token_id,
        padded_prompt_width=64,
        padded_response_width=256,
        uids=["zero-crop", "one-crop"],
        meta_info={"temperature": 1.0},
    )

    assert batch.batch["prompts"].shape == (2, 64)
    assert batch.batch["responses"].shape == (2, 256)
    assert batch.batch["input_ids"].shape == (2, 320)
    assert batch.batch["position_ids"].shape == (2, 4, 320)
    assert batch.batch["response_mask"].shape == (2, 256)
    assert batch.batch["visual_token_mask"].shape == (2, 320)
    assert batch.batch["response_mask"][0].sum().item() == 1
    assert batch.batch["response_mask"][1].sum().item() == 2
    assert batch.non_tensor_batch["uid"].tolist() == ["zero-crop", "one-crop"]
    assert "multi_modal_inputs" not in batch.non_tensor_batch
    assert "agent_trajectory" not in batch.non_tensor_batch
    assert batch.non_tensor_batch["multi_modal_layout"][1]["image_grid_thw"] == ((1, 2, 2), (1, 2, 2))
    assert len(set(batch.non_tensor_batch["agent_trajectory_id"])) == 2
    boundaries = batch.non_tensor_batch["agent_token_boundaries"][1]
    assert boundaries["coordinate_space"] == "padded_input_ids"
    assert boundaries["image_spans"] == ((58, 62), (67, 71))
    assert boundaries["action_spans"] == ((64, 65), (73, 74))
    assert boundaries["observation_spans"] == ((65, 73),)
    for start, end in boundaries["image_spans"]:
        assert torch.all(batch.batch["visual_token_mask"][1, start:end] == 1)


class _NextTokenMultimodalModule(nn.Module):
    def __init__(self, vocab_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.seen_position_shape = None
        self.seen_image_grids = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        image_grid_thw=None,
        pixel_values=None,
        **kwargs,
    ):
        del attention_mask, kwargs
        assert image_grid_thw is not None
        assert pixel_values is not None
        assert int((input_ids == 41).sum()) == int(image_grid_thw.prod(dim=-1).sum())
        self.seen_position_shape = tuple(position_ids.shape)
        self.seen_image_grids = image_grid_thw.clone()
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            dtype=torch.float32,
        )
        next_tokens = torch.roll(input_ids, shifts=-1, dims=1)
        logits.scatter_(2, next_tokens.unsqueeze(-1), 0.0)
        return SimpleNamespace(logits=logits)


def _cpu_log_probs(logits, labels):
    vocab_size = logits.size(-1)
    return -F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)


def _actor_for_recompute(module, *, padding_free=False, dynamic_batching=False, batch_size=1):
    config = ActorConfig(
        global_batch_size=batch_size,
        micro_batch_size_per_device_for_update=batch_size,
        micro_batch_size_per_device_for_experience=batch_size,
        padding_free=padding_free,
        dynamic_batching=dynamic_batching,
        use_torch_compile=False,
    )
    config.global_batch_size_per_device = batch_size
    actor = DataParallelPPOActor(config=config, actor_module=module)
    actor.rank = 1
    actor.log_probs_from_logits = _cpu_log_probs
    return actor


def _recompute_worker_multimodal_inputs(batch, processor):
    worker = object.__new__(FSDPWorker)
    worker._cache = {}
    worker.processor = processor
    batch.meta_info.update({"min_pixels": None, "max_pixels": None, "video_fps": 2.0})
    FSDPWorker._process_multi_modal_inputs(worker, batch)
    return batch


@pytest.mark.parametrize("num_crops", [0, 1, 6])
def test_materialized_trajectory_recomputes_actor_and_reference_log_probs_with_identical_inputs(num_crops):
    processor = _FakeQwen3VLProcessor()
    sample = AgentTrajectoryMaterializer(processor, max_prompt_tokens=64).materialize(_make_trajectory(num_crops))
    batch = collate_materialized_agent_trajectories(
        [sample],
        pad_token_id=0,
        padded_prompt_width=64,
        padded_response_width=256,
        uids=["sample"],
        meta_info={"temperature": 1.0},
    )
    _recompute_worker_multimodal_inputs(batch, processor)
    actor_module = _NextTokenMultimodalModule()
    ref_module = _NextTokenMultimodalModule()

    actor_log_probs = _actor_for_recompute(actor_module).compute_log_prob(batch)
    ref_log_probs = _actor_for_recompute(ref_module).compute_log_prob(batch)

    assert actor_log_probs.shape == batch.batch["responses"].shape
    torch.testing.assert_close(actor_log_probs, ref_log_probs)
    assert actor_module.seen_position_shape == (4, 1, batch.batch["input_ids"].size(1))
    torch.testing.assert_close(actor_module.seen_image_grids, ref_module.seen_image_grids)
    if num_crops:
        assert int((batch.batch["response_mask"] == 0).sum()) > 0


def test_environment_observation_tokens_have_zero_policy_gradient():
    processor = _FakeQwen3VLProcessor()
    sample = AgentTrajectoryMaterializer(processor, max_prompt_tokens=64).materialize(_make_trajectory(1))
    batch = collate_materialized_agent_trajectories(
        [sample],
        pad_token_id=0,
        padded_prompt_width=64,
        padded_response_width=256,
        uids=["sample"],
        meta_info={"temperature": 1.0},
    )
    _recompute_worker_multimodal_inputs(batch, processor)
    response_mask = batch.batch["response_mask"].float()
    log_probs = torch.zeros_like(response_mask, requires_grad=True)
    loss, _ = compute_policy_loss(
        old_log_probs=torch.zeros_like(response_mask),
        log_probs=log_probs,
        advantages=torch.ones_like(response_mask),
        response_mask=response_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_dual=3.0,
        tau_positive=1.0,
        tau_negative=1.0,
        loss_type="default",
        loss_avg_mode="token",
    )
    loss.backward()

    assert torch.all(log_probs.grad[response_mask == 0] == 0)
    assert torch.all(log_probs.grad[response_mask == 1] != 0)


def test_materialized_mrope_survives_padding_free_dynamic_batching():
    if not hasattr(dp_actor_module, "unpad_input"):
        pytest.skip("flash-attn padding-free ops are unavailable")

    processor = _FakeQwen3VLProcessor()
    materializer = AgentTrajectoryMaterializer(processor, max_prompt_tokens=64)
    samples = [materializer.materialize(_make_trajectory(0)), materializer.materialize(_make_trajectory(1))]
    batch = collate_materialized_agent_trajectories(
        samples,
        pad_token_id=0,
        padded_prompt_width=64,
        padded_response_width=256,
        uids=["zero-crop", "one-crop"],
        meta_info={"temperature": 1.0},
    )
    _recompute_worker_multimodal_inputs(batch, processor)

    dense = _actor_for_recompute(
        _NextTokenMultimodalModule(),
        batch_size=2,
    ).compute_log_prob(batch)
    padding_free = _actor_for_recompute(
        _NextTokenMultimodalModule(),
        padding_free=True,
        dynamic_batching=True,
        batch_size=2,
    ).compute_log_prob(batch)

    torch.testing.assert_close(dense, padding_free)


class _ScriptedBackend:
    def __init__(self, output):
        self.output = output
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return self.output


class _ExpandedObservationEncoder:
    async def encode(self, request):
        return EncodedObservation(
            token_ids=[50, 41],
            images=request.result.images,
            effective_token_count=5,
            visual_token_count=4,
        )


def test_agent_loop_applies_expanded_observation_budget_atomically():
    tool_call = (
        "<tool_call>"
        + json.dumps(
            {
                "name": "image_zoom_in_tool",
                "arguments": {"bbox_2d": [0, 0, 1000, 1000]},
            }
        )
        + "</tool_call>"
    )
    loop = AgentLoop(
        _ScriptedBackend(GenerationOutput([10, 11], tool_call)),
        _ExpandedObservationEncoder(),
        ToolRegistry([ImageZoomInTool()]),
        config=AgentLoopConfig(max_response_tokens=6, max_tokens_per_turn=6),
    )

    trajectory = asyncio.run(
        loop.run(
            prompt_ids=[1],
            source_images=[Image.new("RGB", (100, 100))],
        )
    )

    assert trajectory.status == AgentStatus.LENGTH_LIMIT
    assert trajectory.response_ids == [10, 11]
    assert trajectory.response_mask == [1, 1]
    assert trajectory.effective_response_tokens == 2
    assert trajectory.metrics.raw_observation_tokens == 0
    assert trajectory.metrics.observation_tokens == 0
    assert trajectory.metrics.tool_execution_successes == 1
    assert trajectory.metrics.tool_call_successes == 0
    assert trajectory.steps[0].observation_committed is False


def test_fsdp_worker_requires_provenance_for_precomputed_multimodal_inputs():
    worker = object.__new__(FSDPWorker)
    worker._cache = {}
    precomputed = np.array([{"pixel_values": torch.tensor([[1.0]])}], dtype=object)
    data = DataProto.from_dict(
        non_tensors={
            "uid": np.array(["sample"], dtype=object),
            "multi_modal_data": np.array([{"images": []}], dtype=object),
            "multi_modal_inputs": precomputed,
        }
    )

    with pytest.raises(ValueError, match="processor_atomic"):
        FSDPWorker._process_multi_modal_inputs(worker, data)

    data.non_tensor_batch["multi_modal_inputs_provenance"] = np.array(
        ["processor_atomic"],
        dtype=object,
    )
    FSDPWorker._process_multi_modal_inputs(worker, data)

    assert data.non_tensor_batch["multi_modal_inputs"] is precomputed


def test_agent_rollout_pop_preserves_driver_raw_prompt():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(worker=SimpleNamespace(rollout=SimpleNamespace(interaction_mode="agentic")))
    raw_prompt = np.empty(1, dtype=object)
    raw_prompt[0] = [{"role": "user", "content": "inspect"}]
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]]),
        },
        non_tensors={
            "raw_prompt_ids": np.array([[1, 2]], dtype=object),
            "multi_modal_data": np.array([{"images": []}], dtype=object),
            "raw_prompt": raw_prompt,
            "uid": np.array(["prompt-0"], dtype=object),
        },
    )

    rollout_batch = trainer._pop_rollout_inputs(batch)

    assert batch.non_tensor_batch["raw_prompt"] is raw_prompt
    assert "uid" not in batch.non_tensor_batch
    assert rollout_batch.non_tensor_batch["raw_prompt"] is raw_prompt
    assert rollout_batch.non_tensor_batch["uid"].tolist() == ["prompt-0"]


def test_group_rollout_and_turn_seeds_are_stable_and_distinct():
    first = [
        derive_agent_seed(7, sample_index=3, rollout_index=rollout, turn_index=turn)
        for rollout in range(8)
        for turn in range(3)
    ]
    second = [
        derive_agent_seed(7, sample_index=3, rollout_index=rollout, turn_index=turn)
        for rollout in range(8)
        for turn in range(3)
    ]

    assert first == second
    assert len(set(first)) == 24


def _make_agentic_materialization_test_case(monkeypatch, inference_calls):
    async def fake_run_deepeyes_inference(**kwargs):
        inference_calls.append((kwargs["rollout_index"], kwargs["base_seed"]))
        trajectory = _make_trajectory(kwargs["rollout_index"])
        trajectory._test_rollout_index = kwargs["rollout_index"]
        trajectory._test_base_seed = kwargs["base_seed"]
        return trajectory

    monkeypatch.setattr(
        "verl.workers.agent.inference.run_deepeyes_inference",
        fake_run_deepeyes_inference,
    )
    config = RolloutConfig(
        interaction_mode="agentic",
        n=2,
        limit_images=7,
        agent_max_tool_calls=6,
        agent_max_tokens_per_turn=128,
    )
    config.prompt_length = 64
    config.response_length = 256
    config.max_model_len = 512
    rollout = object.__new__(vLLMRollout)
    rollout.agent_bbox_format = "norm1000"
    rollout.config = config
    rollout.processor = _FakeQwen3VLProcessor()
    rollout.tokenizer = rollout.processor.tokenizer
    rollout.pad_token_id = 0
    rollout.inference_engine = object()
    rollout.sampling_params = SamplingParams(n=2, seed=7)
    rollout.use_tqdm = False
    rollout.lora_kwargs = {}
    prompts = DataProto.from_dict(
        tensors={"input_ids": torch.ones((1, 64), dtype=torch.long)},
        non_tensors={
            "raw_prompt_ids": np.array([[1, 41, 2]], dtype=object),
            "raw_prompt": np.array(
                [
                    [
                        {"role": "system", "content": "{{ max_tool_calls }}"},
                        {
                            "role": "user",
                            "content": [{"type": "image"}],
                        },
                    ]
                ],
                dtype=object,
            ),
            "multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (8, 8))]}],
                dtype=object,
            ),
            "uid": np.array(["prompt-3"], dtype=object),
        },
        meta_info={
            "min_pixels": None,
            "max_pixels": None,
            "video_fps": 2.0,
        },
    )

    return rollout, prompts


def test_vllm_rollout_materializes_agent_groups_without_shipping_full_trajectories(
    monkeypatch,
):
    inference_calls = []
    rollout, prompts = _make_agentic_materialization_test_case(
        monkeypatch,
        inference_calls,
    )
    original_materialize = AgentTrajectoryMaterializer.materialize

    def reject_one_initial_trajectory(self, trajectory):
        if trajectory._test_rollout_index == 0 and trajectory._test_base_seed == 7:
            raise AgentTrajectoryRejected(
                "model generated a reserved visual context token",
                recoverable=True,
            )
        return original_materialize(self, trajectory)

    monkeypatch.setattr(
        AgentTrajectoryMaterializer,
        "materialize",
        reject_one_initial_trajectory,
    )

    output = rollout.generate_sequences(prompts)

    assert output.batch["responses"].shape == (2, 256)
    assert output.non_tensor_batch["uid"].tolist() == [
        "prompt-3",
        "prompt-3",
    ]
    assert len(set(output.non_tensor_batch["agent_trajectory_id"])) == 2
    assert "agent_trajectory" not in output.non_tensor_batch
    assert "multi_modal_inputs" not in output.non_tensor_batch
    assert [item["tool_call_successes"] for item in output.non_tensor_batch["agent_reward_input"]] == [0, 0]
    assert [item["tool_calls_inside_reasoning"] for item in output.non_tensor_batch["agent_diagnostics"]] == [0, 0]
    assert [item["unclosed_answer"] for item in output.non_tensor_batch["agent_diagnostics"]] == [False, False]
    assert inference_calls == [(0, 7), (1, 7), (0, 1_000_010)]
    assert [item["trajectory_retries"] for item in output.non_tensor_batch["agent_diagnostics"]] == [1, 0]
    assert [item["final_answer"] for item in output.non_tensor_batch["agent_reward_input"]] == ["x", "x"]


def test_vllm_rollout_does_not_retry_nonrecoverable_agent_materialization(
    monkeypatch,
):
    inference_calls = []
    rollout, prompts = _make_agentic_materialization_test_case(
        monkeypatch,
        inference_calls,
    )

    def reject_nonrecoverable(self, trajectory):
        raise AgentTrajectoryRejected("expanded agent prompt overflow")

    monkeypatch.setattr(
        AgentTrajectoryMaterializer,
        "materialize",
        reject_nonrecoverable,
    )

    with pytest.raises(AgentTrajectoryRejected, match="expanded agent prompt overflow"):
        rollout.generate_sequences(prompts)
    assert inference_calls == [(0, 7), (1, 7)]


def test_vllm_rollout_raises_after_exact_agent_retry_budget(monkeypatch):
    inference_calls = []
    rollout, prompts = _make_agentic_materialization_test_case(
        monkeypatch,
        inference_calls,
    )

    def reject_recoverable(self, trajectory):
        raise AgentTrajectoryRejected("reserved visual context token", recoverable=True)

    monkeypatch.setattr(
        AgentTrajectoryMaterializer,
        "materialize",
        reject_recoverable,
    )

    with pytest.raises(AgentTrajectoryRejected, match="reserved visual context token"):
        rollout.generate_sequences(prompts)
    assert inference_calls == [
        (0, 7),
        (1, 7),
        (0, 1_000_010),
        (0, 2_000_013),
    ]


def test_agent_rejected_trajectory_only_marks_reserved_visual_tokens_recoverable():
    recoverable = AgentTrajectoryRejected("reserved", recoverable=True)
    fail_fast = AgentTrajectoryRejected("layout mismatch")

    assert recoverable.recoverable is True
    assert fail_fast.recoverable is False


def test_data_metrics_use_interleaved_agent_response_mask():
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[10, 11, 12, 13]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "response_mask": torch.tensor([[1, 0, 1, 0]], dtype=torch.long),
            "token_level_scores": torch.tensor([[1.0, 0.0, 2.0, 0.0]]),
            "token_level_rewards": torch.tensor([[1.0, 0.0, 2.0, 0.0]]),
            "advantages": torch.tensor([[3.0, 99.0, 5.0, 99.0]]),
            "returns": torch.tensor([[7.0, 99.0, 11.0, 99.0]]),
        }
    )

    metrics = compute_data_metrics(batch)

    assert metrics["critic/advantages/mean"] == pytest.approx(4.0)
    assert metrics["critic/returns/mean"] == pytest.approx(9.0)


def test_agent_reward_rpc_input_excludes_images_layouts_and_processor_tensors():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(worker=SimpleNamespace(rollout=SimpleNamespace(interaction_mode="agentic")))
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2]]),
            "response_mask": torch.tensor([[1, 0]]),
            "pixel_values": torch.ones((1, 2)),
        },
        non_tensors={
            "ground_truth": np.array(["x"], dtype=object),
            "agent_reward_input": np.array([{"response": "x"}], dtype=object),
            "agent_diagnostics": np.array([{"status": "answered"}], dtype=object),
            "multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (8, 8))]}],
                dtype=object,
            ),
            "multi_modal_layout": np.array([{"kind": "qwen"}], dtype=object),
        },
    )

    reward_batch = trainer._reward_rpc_input(batch)

    assert set(reward_batch.batch.keys()) == {"responses", "response_mask"}
    assert set(reward_batch.non_tensor_batch) == {
        "ground_truth",
        "agent_reward_input",
        "agent_diagnostics",
    }
    assert "multi_modal_data" in batch.non_tensor_batch
    assert "pixel_values" in batch.batch


def test_training_config_requires_the_shared_deepeyes_prompt_contract():
    config = PPOConfig()
    config.worker.rollout.interaction_mode = "agentic"
    config.data.system_prompt = str(
        Path(__file__).resolve().parents[1] / "examples" / "system_prompt" / "deepeyes.txt"
    )
    config.data.format_prompt = None

    config.post_init()

    config.data.format_prompt = "examples/system_prompt/xml_GR.txt"
    with pytest.raises(ValueError, match="cannot use data.format_prompt"):
        config.post_init()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("corrupt_image", "cgpo_flat"),
        ("use_grounding_consistency_reward", True),
        ("advantage_scaling_method", "cgpo"),
    ],
)
def test_agentic_training_rejects_one_shot_grounding_interventions(field, value):
    config = PPOConfig()
    config.worker.rollout.interaction_mode = "agentic"
    config.data.system_prompt = str(
        Path(__file__).resolve().parents[1] / "examples" / "system_prompt" / "deepeyes.txt"
    )
    config.data.format_prompt = None
    setattr(config.algorithm, field, value)

    with pytest.raises(
        ValueError,
        match=r"does not support grounded-reasoning interventions.*algorithm\.",
    ):
        config.post_init()


def test_agent_rollout_config_rejects_inconsistent_turn_budget():
    config = RolloutConfig(
        interaction_mode="agentic",
        agent_max_tokens_per_turn=10240,
    )
    config.response_length = 2048

    with pytest.raises(ValueError, match="cannot exceed"):
        config.post_init()


def test_agent_loop_uses_explicit_group_identity_for_generation_seed():
    seeds = []
    for rollout_index in range(8):
        backend = _ScriptedBackend(
            GenerationOutput([10], "<answer>x</answer>"),
        )
        trajectory = asyncio.run(
            AgentLoop(
                backend,
                _ExpandedObservationEncoder(),
                ToolRegistry([ImageZoomInTool()]),
                seed_context=AgentSeedContext(
                    base_seed=7,
                    sample_index=3,
                    rollout_index=rollout_index,
                ),
            ).run(
                prompt_ids=[1],
                source_images=[Image.new("RGB", (8, 8))],
            )
        )
        assert trajectory.status == AgentStatus.ANSWERED
        seeds.append(backend.requests[0].seed)

    assert len(set(seeds)) == 8


def test_deepeyes_inference_rejects_insufficient_vllm_image_capacity_before_rollout():
    with pytest.raises(ValueError, match="required at least 7"):
        asyncio.run(
            run_deepeyes_inference(
                inference_engine=object(),
                sampling_params=SamplingParams(n=1),
                processor=_FakeQwen3VLProcessor(),
                messages=[],
                source_images=[Image.new("RGB", (8, 8))],
                sample_index=0,
                rollout_index=0,
                image_config=AgentImageConfig(limit_images=6),
                base_seed=1,
                scheduler=VLLMAgentBatchScheduler(object()),
                max_model_len=128,
            )
        )
