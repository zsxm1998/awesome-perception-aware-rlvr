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
"""Teacher-weighted visual-sensitivity loss (S-OPD's teacher-gated perception contrast): configuration, the loss
against its definition, independence of the micro-batch and rank split, and the trainer's teacher pass."""

import numpy as np
import pytest
import torch
from torch import nn

from verl.protocol import DataProto
from verl.trainer.config import PPOConfig
from verl.trainer.core_algos import average_loss
from verl.trainer.perception_reasoning_loss import compute_perception_reasoning_policy_loss
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.trainer.visual_sensitivity import compute_sampled_low_var_kl
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor


COEF = 0.02


def _s_opd_config(**algorithm) -> PPOConfig:
    config = PPOConfig()
    config.worker.teacher.source = "model"
    config.worker.teacher.model.model_path = "teacher"
    options = dict(
        adv_estimator="teacher_log_ratio",
        disable_kl=True,
        corrupt_image="random_patch",
        corrupt_image_kwargs={"black_prob": 0.6},
        visual_sensitivity_loss_coef=COEF,
        visual_sensitivity_loss_weighting="teacher",
        incremental_image_transform="rgb_gaussian_noise",
        incremental_image_kwargs={"std": 0.2},
        visual_robustness_loss_coef=COEF,
    )
    options.update(algorithm)
    for key, value in options.items():
        setattr(config.algorithm, key, value)
    return config


def test_s_opd_config():
    _s_opd_config().deep_post_init()
    # the weighting alone is an objective that uses the teacher
    _s_opd_config(adv_estimator="grpo").deep_post_init()


@pytest.mark.parametrize(
    "algorithm, message",
    [
        (dict(visual_sensitivity_loss_weighting="student"), "visual_sensitivity_loss_weighting"),
        (dict(visual_sensitivity_loss_coef=0.0), "set visual_sensitivity_loss_coef"),
        (dict(corrupt_image="mask_visual_attention", corrupt_image_kwargs=None), "image corruption"),
        (
            dict(corrupt_image="cross_modal_attention_value_mean", corrupt_image_kwargs=None),
            "image corruption",
        ),
        (dict(visual_sensitivity_metric="full_vocab_kl"), "sampled visual_sensitivity_metric"),
    ],
)
def test_invalid_weighting_configs_fail(algorithm, message):
    with pytest.raises(ValueError, match=message):
        _s_opd_config(**algorithm).deep_post_init()


def test_weighting_requires_a_teacher():
    config = _s_opd_config(adv_estimator="grpo")
    config.worker.teacher.source = "none"
    config.worker.teacher.model.model_path = None
    with pytest.raises(ValueError, match="visual_sensitivity_loss_weighting=teacher requires a teacher"):
        config.deep_post_init()


@pytest.mark.parametrize(
    "algorithm, message",
    [
        (dict(incremental_image_kwargs={"sigma": 0.2}), "unsupported incremental_image_kwargs"),
        (dict(incremental_image_kwargs={"std": 0.0}), "std must be a positive number"),
        (dict(incremental_image_kwargs={"std": True}), "std must be a positive number"),
        (dict(incremental_image_kwargs={"noise_before_resize": 1}), "noise_before_resize"),
        (dict(noise_t_init=300.0), "noise_t_init set the vp_diffusion schedule"),
        (dict(incremental_image_transform="gaussian_noise"), "incremental_image_transform must be one of"),
        (dict(incremental_image_transform=None), "vp_diffusion or rgb_gaussian_noise"),
    ],
)
def test_invalid_rgb_noise_configs_fail(algorithm, message):
    with pytest.raises(ValueError, match=message):
        _s_opd_config(**algorithm).deep_post_init()


# ---------------------------------------------------------------------------------------------------------------
# The loss against its definition
# ---------------------------------------------------------------------------------------------------------------

LENGTHS = [1, 7, 2, 6, 0, 3, 8, 4]  # one empty response
RESPONSE_LENGTH = 8


def _tensors(lengths=LENGTHS):
    generator = torch.Generator().manual_seed(0)
    rows = len(lengths)
    mask = (torch.arange(RESPONSE_LENGTH)[None, :] < torch.tensor(lengths)[:, None]).long()
    decremental = -torch.rand(rows, RESPONSE_LENGTH, generator=generator) * 3.0
    teacher_gap = torch.randn(rows, RESPONSE_LENGTH, generator=generator)
    weights = teacher_gap.clamp(min=0.0) * mask
    weights[2] = 0.0  # a response without weighted tokens
    return mask, decremental, weights


def _reference_loss(log_prob, decremental, weights, mask, mode):
    """S-OPD: -coef * sum(w * k3) / #(w > 0) over the batch (its code), or per response with at least one token in
    the denominator, then over the responses (its Eq. 5)."""
    k3 = compute_sampled_low_var_kl(decremental, log_prob, reference_mode="current") * mask
    gated = ((weights > 0) & mask.bool()).float()
    if mode == "token":
        return -COEF * (weights * k3).sum() / gated.sum()
    per_response = (weights * k3).sum(-1) / gated.sum(-1).clamp(min=1)
    return -COEF * per_response[mask.sum(-1) > 0].mean()


def _loss(log_prob, decremental, weights, mask, mode, normalizer=None, weighting="teacher"):
    loss, _ = compute_perception_reasoning_policy_loss(
        actor_config=ActorConfig(loss_avg_mode=mode, use_torch_compile=False),
        loss_config={
            "visual_sensitivity_loss_coef": COEF,
            "visual_sensitivity_loss_weighting": weighting,
            "visual_sensitivity_reference": "current",
            "visual_sensitivity_metric": "sampled_low_var_kl",
        },
        log_prob=log_prob,
        old_log_prob=log_prob.detach(),
        advantages=torch.zeros_like(log_prob),
        response_mask=mask,
        entropy=None,
        decremental_old_log_probs=decremental,
        decremental_entropies=None,
        incremental_old_log_probs=None,
        incremental_entropies=None,
        visual_sensitivity_loss_weights=weights if weighting == "teacher" else None,
        visual_sensitivity_loss_normalizer=normalizer,
    )
    return loss


def test_token_mode_averages_over_the_weighted_tokens():
    mask, decremental, weights = _tensors()
    log_prob = torch.zeros(mask.shape)
    normalizer = mask.sum() / ((weights > 0) & mask.bool()).sum()
    # the actor's token-mean rescaling multiplies a micro-batch's loss by its tokens over the mini-batch's tokens;
    # with the whole mini-batch in one micro-batch that is 1
    torch.testing.assert_close(
        _loss(log_prob, decremental, weights, mask, "token", normalizer),
        _reference_loss(log_prob, decremental, weights, mask, "token"),
    )


def test_seq_mode_averages_within_each_response():
    mask, decremental, weights = _tensors()
    log_prob = torch.zeros(mask.shape)
    # average_loss(seq) takes the mean over all rows; the actor rescales it to the rows with a token
    rows_with_tokens = (mask.sum(-1) > 0).sum()
    torch.testing.assert_close(
        _loss(log_prob, decremental, weights, mask, "seq") * len(LENGTHS) / rows_with_tokens,
        _reference_loss(log_prob, decremental, weights, mask, "seq"),
    )


def test_without_weighting_the_loss_is_the_unweighted_mean():
    mask, decremental, weights = _tensors()
    log_prob = torch.zeros(mask.shape)
    k3 = compute_sampled_low_var_kl(decremental, log_prob, reference_mode="current")
    for mode in ("token", "seq"):
        expected = average_loss(-COEF * k3, mask, mode=mode)
        assert torch.equal(_loss(log_prob, decremental, weights, mask, mode, weighting="none"), expected)


def test_weighting_without_weights_fails():
    mask, decremental, _ = _tensors()
    with pytest.raises(ValueError, match="visual_sensitivity_loss_weights"):
        _loss(torch.zeros(mask.shape), decremental, None, mask, "seq")


# ---------------------------------------------------------------------------------------------------------------
# Independence of the micro-batch and rank split (dp_actor's all-reduced counts)
# ---------------------------------------------------------------------------------------------------------------


def _update_data(rows: list[int]) -> DataProto:
    mask, decremental, weights = _tensors()
    prompt = torch.ones(len(rows), 2, dtype=torch.long)
    response_mask = mask[rows]
    attention_mask = torch.cat([prompt, response_mask], dim=-1)
    # the first input id holds the row, so that the fake forward pass can look up the row's parameters
    input_ids = torch.cat([torch.tensor(rows)[:, None], prompt[:, :1], torch.zeros_like(response_mask)], dim=-1)
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.cumsum(attention_mask, dim=-1),
            "responses": torch.zeros_like(response_mask),
            "response_mask": response_mask,
            "old_log_probs": torch.zeros(len(rows), RESPONSE_LENGTH),
            "advantages": torch.zeros(len(rows), RESPONSE_LENGTH),
            "decremental_old_log_probs": decremental[rows],
            "visual_sensitivity_loss_weights": weights[rows],
        },
        non_tensors={"multi_modal_inputs": np.array([None] * len(rows), dtype=object)},
        meta_info={
            "temperature": 1.0,
            "perception_reasoning_config": {
                "visual_sensitivity_loss_coef": COEF,
                "visual_sensitivity_loss_weighting": "teacher",
                "visual_sensitivity_reference": "current",
                "visual_sensitivity_metric": "sampled_low_var_kl",
            },
        },
    )


def _update(monkeypatch, mode, rows, world_size, micro, dynamic, totals=None):
    """One update of one rank: the gradient and the values the rank passed to each all_reduce, which returns
    ``totals[i]`` for the i-th call (the rank's own value if None)."""
    config = ActorConfig(
        global_batch_size=len(rows),
        micro_batch_size_per_device_for_update=micro,
        micro_batch_size_per_device_for_experience=micro,
        loss_avg_mode=mode,
        padding_free=False,
        dynamic_batching=dynamic,
        use_torch_compile=False,
    )
    config.global_batch_size_per_device = len(rows)
    actor = DataParallelPPOActor(config=config, actor_module=nn.Linear(1, 1), actor_optimizer=None)
    actor.world_size = world_size
    params = nn.Parameter(torch.zeros(len(LENGTHS), RESPONSE_LENGTH))  # the log-prob of every token
    reduced = []

    def all_reduce(tensor, op=None):
        reduced.append(tensor.clone())
        if totals is not None:
            tensor.fill_(totals[len(reduced) - 1])

    monkeypatch.setattr(
        actor, "_forward_micro_batch", lambda model_inputs, **kwargs: params[model_inputs["input_ids"][:, 0]]
    )
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    actor.update_policy(_update_data(rows))
    return params.grad, reduced


def _expected_grad(mode):
    mask, decremental, weights = _tensors()
    params = nn.Parameter(torch.zeros(mask.shape))
    _reference_loss(params, decremental, weights, mask, mode).backward()
    return params.grad


@pytest.mark.parametrize("mode", ["seq", "token"])
@pytest.mark.parametrize(
    "world, micro, dynamic",
    [
        ([list(range(8))], 8, False),
        ([list(range(8))], 4, False),
        ([list(range(8))], 1, False),
        ([list(range(8))], 2, True),
        ([[0, 2, 3, 5], [1, 4, 6, 7]], 2, False),
        ([[2, 4], [0, 1, 3, 5, 6, 7]], 2, False),  # a rank without weighted tokens
    ],
)
def test_weighted_loss_does_not_depend_on_the_split(monkeypatch, mode, world, micro, dynamic):
    reduced = [_update(monkeypatch, mode, rows, len(world), micro, dynamic)[1] for rows in world]
    totals = [sum(values) for values in zip(*reduced)]
    grads = [_update(monkeypatch, mode, rows, len(world), micro, dynamic, totals)[0] for rows in world]
    torch.testing.assert_close(sum(grads) / len(world), _expected_grad(mode))


# ---------------------------------------------------------------------------------------------------------------
# The trainer: the teacher scores the view the student was scored on
# ---------------------------------------------------------------------------------------------------------------


class _FakeWorkerGroup:
    def __init__(self, teacher_view_log_probs):
        self.teacher_view_log_probs = teacher_view_log_probs
        self.aux_inputs, self.teacher_inputs = [], []

    def compute_aux_log_probs(self, data):
        self.aux_inputs.append(data)
        key = data.meta_info["aux_log_probs_output_key"]
        return DataProto.from_dict(tensors={key: torch.zeros(len(data), RESPONSE_LENGTH)})

    def compute_teacher_log_probs(self, data):
        self.teacher_inputs.append(data)
        key = data.meta_info["teacher_log_probs_output_key"]
        return DataProto.from_dict(tensors={key: self.teacher_view_log_probs})


class _ViewResult:
    def __init__(self, batch):
        self.batch, self.stats = batch, {}


class _FakeBuilder:
    """A fresh view object per call, as random_patch at the response position (drawn without a seed)."""

    def __init__(self):
        self.views = []

    def _view(self, batch):
        view = DataProto.from_dict(tensors={"input_ids": batch.batch["input_ids"].clone()}, meta_info={})
        self.views.append(view)
        return _ViewResult(view)

    def build_batch(self, batch, config, global_step):
        return self._view(batch)

    def build_dvrp_batches(self, batch, config, global_step, total_training_steps):
        return self._view(batch), self._view(batch)


def _trainer(config, teacher_view_log_probs):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.actor_rollout_ref_wg = _FakeWorkerGroup(teacher_view_log_probs)
    trainer.perception_reasoning_corruption_builder = _FakeBuilder()
    trainer.global_step, trainer.training_steps = 1, 10
    trainer._latest_vision_metrics, trainer._sensitivity_view = {}, None
    return trainer


@pytest.mark.parametrize("dvrp", [True, False])
def test_teacher_scores_the_students_view(dvrp):
    no_dvrp = dict(incremental_image_transform=None, incremental_image_kwargs=None, visual_robustness_loss_coef=0.0)
    config = _s_opd_config(**({} if dvrp else no_dvrp))
    config.deep_post_init()
    mask, _, _ = _tensors()
    teacher = torch.randn(mask.shape)
    teacher_view = teacher - torch.randn(mask.shape)
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.zeros(len(LENGTHS), 3), "response_mask": mask, "teacher_log_probs": teacher}
    )
    trainer = _trainer(config, teacher_view)
    RayPPOTrainer._compute_auxiliary_old_log_probs(trainer, batch)
    student_view = trainer.perception_reasoning_corruption_builder.views[0]
    assert trainer.actor_rollout_ref_wg.aux_inputs[0] is student_view

    metrics = {}
    RayPPOTrainer._attach_visual_sensitivity_loss_weights(trainer, batch, metrics)
    assert trainer.actor_rollout_ref_wg.teacher_inputs == [student_view]
    assert trainer._sensitivity_view is None
    expected = (teacher - teacher_view).clamp(min=0) * mask
    torch.testing.assert_close(batch.batch["visual_sensitivity_loss_weights"], expected)
    gated = (expected > 0).sum()
    assert metrics["algo/sensitivity/teacher_gate_fraction"] == pytest.approx((gated / mask.sum()).item())
    assert metrics["algo/sensitivity/teacher_gate_mean"] == pytest.approx((expected.sum() / gated).item())


def test_without_weighting_the_view_is_not_kept():
    config = _s_opd_config(visual_sensitivity_loss_weighting="none", adv_estimator="teacher_log_ratio")
    config.deep_post_init()
    mask, _, _ = _tensors()
    batch = DataProto.from_dict(tensors={"input_ids": torch.zeros(len(LENGTHS), 3), "response_mask": mask})
    trainer = _trainer(config, None)
    RayPPOTrainer._compute_auxiliary_old_log_probs(trainer, batch)
    assert trainer._sensitivity_view is None
    with pytest.raises(RuntimeError, match="no corrupt_image view"):
        RayPPOTrainer._attach_visual_sensitivity_loss_weights(trainer, batch, {})
