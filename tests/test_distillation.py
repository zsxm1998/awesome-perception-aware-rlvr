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
"""On-policy distillation: the teacher role, the sampled-token objective and their configuration."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.config import PPOConfig
from verl.trainer.core_algos import (
    AdvantageEstimator,
    compute_advantage_return,
    compute_teacher_log_ratio_metrics,
)
from verl.trainer.ray_trainer import compute_advantage
from verl.utils import tokenizer as tokenizer_module


def _teacher_config(**algorithm) -> PPOConfig:
    config = PPOConfig()
    config.worker.teacher.source = "model"
    config.worker.teacher.model.model_path = "teacher"
    config.algorithm.adv_estimator = "teacher_log_ratio"
    config.algorithm.disable_kl = True
    for key, value in algorithm.items():
        setattr(config.algorithm, key, value)
    return config


def test_teacher_is_off_by_default():
    config = PPOConfig()
    config.deep_post_init()
    assert not config.worker.teacher.enabled
    assert not config.worker.rollout.ban_ids_beyond_tokenizer


def test_teacher_config_bans_ids_beyond_the_tokenizer():
    config = _teacher_config()
    config.deep_post_init()
    assert config.worker.teacher.enabled
    assert config.worker.rollout.ban_ids_beyond_tokenizer


def test_teacher_log_ratio_requires_a_teacher():
    config = PPOConfig()
    config.algorithm.adv_estimator = "teacher_log_ratio"
    with pytest.raises(ValueError, match="requires a teacher"):
        config.deep_post_init()


def test_teacher_without_an_objective_fails():
    config = _teacher_config(adv_estimator="grpo")
    with pytest.raises(ValueError, match="no objective uses the teacher"):
        config.deep_post_init()


@pytest.mark.parametrize("field, value", [("temperature", 0.7), ("top_p", 0.99), ("top_k", 20)])
def test_teacher_requires_untruncated_sampling(field, value):
    config = _teacher_config()
    setattr(config.worker.rollout, field, value)
    with pytest.raises(ValueError, match="own distribution"):
        config.deep_post_init()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda teacher: setattr(teacher, "source", "unknown"), "worker.teacher.source"),
        (lambda teacher: setattr(teacher.model, "model_path", None), "requires worker.teacher.model.model_path"),
        (lambda teacher: setattr(teacher.offload, "offload_optimizer", True), "no optimizer"),
    ],
)
def test_invalid_teacher_configs_fail(mutate, message):
    config = _teacher_config()
    mutate(config.worker.teacher)
    with pytest.raises(ValueError, match=message):
        config.deep_post_init()


def test_teacher_model_path_without_source_fails():
    config = PPOConfig()
    config.worker.teacher.model.model_path = "teacher"
    with pytest.raises(ValueError, match="source=none"):
        config.deep_post_init()


def _log_probs():
    teacher = torch.tensor([[-1.0, -2.0, -30.0, -0.5], [-0.1, -0.2, -0.3, -0.4]])
    old = torch.tensor([[-1.5, -1.0, -0.5, -9.0], [-0.1, -0.2, -0.3, -0.4]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    return teacher, old, mask


def test_teacher_log_ratio_advantage():
    teacher, old, mask = _log_probs()
    advantages, returns = compute_advantage_return(
        AdvantageEstimator.TEACHER_LOG_RATIO,
        teacher_log_probs=teacher,
        old_log_probs=old,
        response_mask=mask,
        log_ratio_clip=10.0,
    )
    expected = torch.tensor([[0.5, -1.0, -10.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)

    unclamped, _ = compute_advantage_return(
        AdvantageEstimator.TEACHER_LOG_RATIO,
        teacher_log_probs=teacher,
        old_log_probs=old,
        response_mask=mask,
        log_ratio_clip=None,
    )
    assert unclamped[0, 2].item() == pytest.approx(-29.5)
    # padded positions are zero even though their log-probs differ (8.5 at [0, 3])
    assert unclamped[0, 3].item() == 0.0


def test_teacher_log_ratio_metrics_are_taken_before_clamping():
    teacher, old, mask = _log_probs()
    metrics = compute_teacher_log_ratio_metrics(teacher, old, mask, log_ratio_clip=10.0)
    assert metrics["teacher/log_ratio_min"] == pytest.approx(-29.5)
    assert metrics["teacher/log_ratio_max"] == pytest.approx(0.5)
    assert metrics["teacher/log_ratio_mean"] == pytest.approx((0.5 - 1.0 - 29.5) / 5)
    assert metrics["teacher/log_ratio_clip_frac"] == pytest.approx(1 / 5)


def test_compute_advantage_passes_the_teacher_inputs():
    teacher, old, mask = _log_probs()
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.zeros_like(teacher),
            "response_mask": mask,
            "teacher_log_probs": teacher,
            "old_log_probs": old,
        },
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
    )
    data = compute_advantage(data, AdvantageEstimator.TEACHER_LOG_RATIO, teacher_log_ratio_clip=10.0)
    assert data.batch["advantages"][0, :3].tolist() == pytest.approx([0.5, -1.0, -10.0])


def test_ids_beyond_the_tokenizer_are_banned(monkeypatch):
    rollout_module = pytest.importorskip("verl.workers.rollout.vllm_rollout_spmd")
    text_config = SimpleNamespace(vocab_size=13)
    monkeypatch.setattr(
        rollout_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(get_text_config=lambda: text_config),
    )
    bias = rollout_module._get_ids_beyond_tokenizer_bias("model", [None] * 10, trust_remote_code=True)
    assert bias == {10: -100, 11: -100, 12: -100}


class _FakeConfig(SimpleNamespace):
    def get_text_config(self):
        return self.text_config


def _fake_model(vocab=None, vocab_size=8, mrope=(1, 1, 2), patch=16, mean=(0.5,), model_type="qwen3_vl"):
    vocab = vocab or {f"t{i}": i for i in range(6)}
    config = _FakeConfig(
        model_type=model_type,
        text_config=SimpleNamespace(vocab_size=vocab_size, rope_scaling={"mrope_section": list(mrope)}),
        vision_config=SimpleNamespace(patch_size=patch, spatial_merge_size=2, temporal_patch_size=2),
        image_token_id=5,
        video_token_id=4,
        vision_start_token_id=3,
        vision_end_token_id=2,
    )
    tokenizer = type("Tok", (), {"get_vocab": lambda self: dict(vocab), "__len__": lambda self: len(vocab)})()
    image_processor = SimpleNamespace(
        patch_size=patch, merge_size=2, temporal_patch_size=2, image_mean=list(mean), image_std=[0.5]
    )
    return config, tokenizer, SimpleNamespace(image_processor=image_processor)


def _patch_models(monkeypatch, models):
    for auto, index in (("AutoConfig", 0), ("AutoTokenizer", 1), ("AutoProcessor", 2)):
        monkeypatch.setattr(
            getattr(tokenizer_module, auto),
            "from_pretrained",
            lambda path, *args, index=index, **kwargs: models[path][index],
        )


def test_teacher_compatibility_accepts_a_larger_model_of_the_same_family(monkeypatch):
    _patch_models(monkeypatch, {"student": _fake_model(), "teacher": _fake_model(model_type="qwen3_vl_moe")})
    tokenizer_module.check_teacher_compatibility("student", "teacher")


def test_teacher_compatibility_ignores_plain_think_tokens(monkeypatch):
    vocab = {f"t{i}": i for i in range(6)}
    student = _fake_model(vocab={**vocab, "<think>": 6, "</think>": 7})
    _patch_models(monkeypatch, {"student": student, "teacher": _fake_model(vocab=vocab)})
    tokenizer_module.check_teacher_compatibility("student", "teacher")


@pytest.mark.parametrize(
    "teacher, message",
    [
        (dict(vocab_size=9), "vocab_size"),
        (dict(vocab={"t0": 0, "t1": 1, "x": 2, "t3": 3, "t4": 4, "t5": 5}), "tokenizer vocabularies differ"),
        (dict(mrope=(2, 1, 1)), "mrope_section"),
        (dict(patch=14), "patch_size"),
        (dict(mean=(0.48,)), "image_mean"),
        (dict(model_type="qwen2_5_vl"), "model_type"),
    ],
)
def test_teacher_compatibility_rejects_mismatches(monkeypatch, teacher, message):
    _patch_models(monkeypatch, {"student": _fake_model(), "teacher": _fake_model(**teacher)})
    with pytest.raises(ValueError, match=message):
        tokenizer_module.check_teacher_compatibility("student", "teacher")


# ---------------------------------------------------------------------------------------------------------------
# distribution-level distillation: the chunked kernel


from verl.trainer.distillation import (  # noqa: E402
    DistillationSpec,
    build_distillation_config,
    chunked_distillation,
    reference_divergence,
)


VOCAB, LM_HEAD = 37, 40  # the LM head has padding rows beyond the tokenizer, as Qwen's


def _logits(rows=9, seed=0, dtype=torch.float32, scale=3.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, LM_HEAD, generator=generator) * scale).to(dtype)


SPECS = [
    DistillationSpec(vocab_size=VOCAB, divergence="reverse_kl"),
    DistillationSpec(vocab_size=VOCAB, divergence="forward_kl"),
    DistillationSpec(vocab_size=VOCAB, divergence="jsd", jsd_beta=0.3),
    DistillationSpec(vocab_size=VOCAB, divergence="forward_kl", temperature=2.0),
    DistillationSpec(vocab_size=VOCAB, divergence="forward_kl", temperature=2.0, temperature_scope="loss_scale_only"),
    DistillationSpec(vocab_size=VOCAB, divergence="jsd", jsd_beta=0.5, support="student_top_k", top_k=5),
]


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("chunk_size", [1, 4, 100])
def test_chunked_divergence_matches_the_definition(spec, chunk_size):
    student = _logits(seed=1).requires_grad_(True)
    teacher = _logits(seed=2)
    labels = torch.randint(0, VOCAB, (student.size(0),), generator=torch.Generator().manual_seed(3))
    loss, sampled, _ = chunked_distillation(student, labels, teacher, spec, chunk_size=chunk_size)

    reference_student = student.detach().clone().requires_grad_(True)
    expected = reference_divergence(reference_student, teacher, spec)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    weights = torch.linspace(0.5, 1.5, student.size(0))
    (loss * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(student.grad, reference_student.grad, rtol=1e-4, atol=1e-6)

    full_log_probs = torch.log_softmax(student.detach(), dim=-1)
    torch.testing.assert_close(sampled.detach(), full_log_probs.gather(-1, labels[:, None]).squeeze(-1))


def test_chunked_divergence_in_bf16():
    spec = DistillationSpec(vocab_size=VOCAB)
    student, teacher = _logits(seed=1, dtype=torch.bfloat16), _logits(seed=2, dtype=torch.bfloat16)
    labels = torch.zeros(student.size(0), dtype=torch.long)
    loss, _, _ = chunked_distillation(student, labels, teacher, spec, chunk_size=4)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, reference_divergence(student, teacher, spec), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("spec", SPECS)
def test_divergence_vanishes_when_the_teacher_is_the_student(spec):
    student = _logits(seed=1)
    loss, _, _ = chunked_distillation(student, torch.zeros(9, dtype=torch.long), student.clone(), spec)
    assert loss.abs().max().item() < 1e-5


def test_padding_rows_of_the_lm_head_are_left_out():
    spec = DistillationSpec(vocab_size=VOCAB)
    student, teacher = _logits(seed=1), _logits(seed=2)
    shifted = teacher.clone()
    shifted[:, VOCAB:] += 50.0  # mass on ids the tokenizer cannot produce
    loss, _, stats = chunked_distillation(student, torch.zeros(9, dtype=torch.long), teacher, spec)
    shifted_loss, _, shifted_stats = chunked_distillation(student, torch.zeros(9, dtype=torch.long), shifted, spec)
    torch.testing.assert_close(loss, shifted_loss)
    expected_oov = 1 - torch.softmax(shifted, -1)[:, :VOCAB].sum(-1)
    torch.testing.assert_close(shifted_stats[:, 2], expected_oov)
    assert stats[:, 2].max() < shifted_stats[:, 2].min()
    entropy = -(torch.softmax(student, -1) * torch.log_softmax(student, -1)).sum(-1)
    torch.testing.assert_close(stats[:, 0], entropy)


def test_reverse_kl_gradient_is_that_of_the_exact_kl():
    """The full-vocabulary reverse KL is differentiated exactly (no sampled-token estimator)."""
    spec = DistillationSpec(vocab_size=LM_HEAD)
    student = _logits(rows=1, seed=4).requires_grad_(True)
    teacher = _logits(rows=1, seed=5)
    loss, _, _ = chunked_distillation(student, torch.zeros(1, dtype=torch.long), teacher, spec)
    loss.sum().backward()
    p = torch.softmax(student.detach(), -1)
    log_ratio = torch.log_softmax(student.detach(), -1) - torch.log_softmax(teacher, -1)
    expected = p * (log_ratio - (p * log_ratio).sum(-1, keepdim=True))  # d KL(p||q) / d logits
    torch.testing.assert_close(student.grad, expected, rtol=1e-4, atol=1e-6)


def test_build_distillation_config_is_off_by_default():
    assert build_distillation_config(PPOConfig().algorithm, vocab_size=VOCAB) is None
    config = PPOConfig().algorithm
    config.distill_loss_coef = 1.0
    built = build_distillation_config(config, vocab_size=VOCAB)
    assert built["vocab_size"] == VOCAB and built["divergence"] == "reverse_kl" and built["loss_coef"] == 1.0


def _distill_config(**algorithm) -> PPOConfig:
    config = PPOConfig()
    config.worker.teacher.source = "model"
    config.worker.teacher.model.model_path = "teacher"
    config.algorithm.disable_kl = True
    config.algorithm.distill_loss_coef = 1.0
    config.algorithm.policy_loss_coef = 0.0
    for key, value in algorithm.items():
        setattr(config.algorithm, key, value)
    return config


def test_distillation_config_is_valid():
    config = _distill_config()
    config.deep_post_init()
    assert config.worker.rollout.ban_ids_beyond_tokenizer


@pytest.mark.parametrize(
    "algorithm, message",
    [
        (dict(distill_divergence="tv"), "distill_divergence"),
        (dict(distill_divergence="jsd", distill_jsd_beta=1.0), "distill_jsd_beta"),
        (dict(distill_support="top_p"), "distill_support"),
        (dict(distill_temperature_scope="student"), "distill_temperature_scope"),
        (dict(distill_is_clip=0.0), "distill_is_clip"),
        (dict(invariant_entropy_coef=0.01), "perception-aware"),
        (dict(top_perception_quantile=0.5, corrupt_image="no_image"), "perception-aware"),
    ],
)
def test_invalid_distillation_configs_fail(algorithm, message):
    config = _distill_config(**algorithm)
    with pytest.raises(ValueError, match=message):
        config.deep_post_init()


def test_policy_loss_coef_needs_distillation():
    config = PPOConfig()
    config.algorithm.policy_loss_coef = 0.5
    with pytest.raises(ValueError, match="policy_loss_coef only"):
        config.deep_post_init()


def test_distillation_requires_a_teacher():
    config = PPOConfig()
    config.algorithm.distill_loss_coef = 1.0
    with pytest.raises(ValueError, match="requires a teacher"):
        config.deep_post_init()


# ---------------------------------------------------------------------------------------------------------------
# distribution-level distillation: the actor


from torch import nn  # noqa: E402

from verl.workers.actor.config import ActorConfig  # noqa: E402
from verl.workers.actor.dp_actor import DataParallelPPOActor  # noqa: E402


class _StubLM(nn.Module):
    """A position-wise language model: logits depend on the token and its position, so misaligned rows show."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.tokens = nn.Embedding(VOCAB, 8)
        self.positions = nn.Embedding(64, 8)
        self.head = nn.Linear(8, LM_HEAD)
        self.config = SimpleNamespace(model_type="stub")

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, logits_to_keep=0, **kwargs):
        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        hidden = self.tokens(input_ids) + self.positions(position_ids.clamp(min=0))
        if not isinstance(logits_to_keep, int):
            hidden = hidden[:, logits_to_keep]
        return SimpleNamespace(logits=self.head(hidden))


def _micro_batch():
    """Three samples: left-padded prompts, right-padded responses (one of them empty)."""
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 0, 1]])
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    generator = torch.Generator().manual_seed(7)
    prompts = torch.randint(1, VOCAB, (3, 3), generator=generator) * prompt_mask
    responses = torch.randint(1, VOCAB, (3, 4), generator=generator) * response_mask
    attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
    return {
        "input_ids": torch.cat([prompts, responses], dim=-1),
        "attention_mask": attention_mask,
        "position_ids": (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0),
        "responses": responses,
        "response_mask": response_mask,
    }


def _actor(teacher_seed: int = 1) -> DataParallelPPOActor:
    config = ActorConfig(padding_free=True, use_torch_compile=False)
    actor = DataParallelPPOActor(config=config, actor_module=_StubLM(seed=0))
    actor.teacher_module = _StubLM(seed=teacher_seed)
    return actor


def test_distillation_rows_are_the_response_positions():
    actor = _actor()
    micro_batch = _micro_batch()
    distill_config = {"vocab_size": VOCAB, "chunk_size": 2}
    log_probs, loss, stats = actor._forward_micro_batch_distill(micro_batch, distill_config)
    # the padded forward of the same position-wise model (the old log-probs' path without unpadding)
    with torch.no_grad():
        logits = actor.actor_module(micro_batch["input_ids"], position_ids=micro_batch["position_ids"]).logits
    expected_log_probs = torch.log_softmax(logits[:, -5:-1], dim=-1).gather(-1, micro_batch["responses"][..., None])
    expected_log_probs = expected_log_probs.squeeze(-1)
    mask = micro_batch["response_mask"].bool()
    torch.testing.assert_close(log_probs[mask], expected_log_probs[mask])
    assert (loss[~mask] == 0).all() and (loss[mask] > 0).all()
    assert stats.shape == (3, 4, 3)

    rows = actor._prepare_response_rows(micro_batch, micro_batch["response_mask"], 4)
    labels = actor._rows_to_response(rows["labels"].float(), rows, 4)
    torch.testing.assert_close(labels, (micro_batch["responses"] * micro_batch["response_mask"]).float())


def test_distillation_with_the_student_as_teacher_is_zero():
    actor = _actor(teacher_seed=0)
    _, loss, _ = actor._forward_micro_batch_distill(_micro_batch(), {"vocab_size": VOCAB, "chunk_size": 2})
    assert loss.abs().max().item() < 1e-6


def test_a_rank_without_response_rows_still_runs_backward():
    actor = _actor()
    micro_batch = _micro_batch()
    micro_batch["response_mask"] = torch.zeros_like(micro_batch["response_mask"])
    log_probs, loss, _ = actor._forward_micro_batch_distill(micro_batch, {"vocab_size": VOCAB, "chunk_size": 2})
    (loss.sum() + log_probs.sum()).backward()
    assert loss.abs().sum() == 0
    assert actor.actor_module.head.weight.grad is not None


# ---------------------------------------------------------------------------------------------------------------
# distribution-level distillation: averaging across micro-batches and ranks (as tests/test_loss_averaging.py)


LENGTHS = [1, 7, 2, 6, 5, 3, 8, 4]
RESPONSE_LENGTH = 8


def _update_data(rows: list[int], lengths: list[int]) -> DataProto:
    response_mask = (torch.arange(RESPONSE_LENGTH)[None, :] < torch.tensor(lengths)[rows, None]).long()
    prompt = torch.ones(len(rows), 2, dtype=torch.long)
    attention_mask = torch.cat([prompt, response_mask], dim=-1)
    input_ids = torch.cat([torch.tensor(rows)[:, None], prompt[:, :1], torch.zeros_like(response_mask)], dim=-1)
    targets = torch.tensor([float(row + 1) * (-1) ** row for row in rows])[:, None] * response_mask
    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.cumsum(attention_mask, dim=-1),
        "responses": torch.zeros_like(response_mask),
        "response_mask": response_mask,
        "old_log_probs": torch.zeros(len(rows), RESPONSE_LENGTH),
        "advantages": torch.zeros(len(rows), RESPONSE_LENGTH),
        "distill_token_weights": targets,
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={"multi_modal_inputs": np.array([None] * len(rows), dtype=object)},
        meta_info={
            "temperature": 1.0,
            "distillation_config": {
                "vocab_size": VOCAB,
                "chunk_size": 4,
                "loss_coef": 1.0,
                "policy_loss_coef": 0.0,
                "is_clip": None,
            },
        },
    )


def _distill_update(monkeypatch, mode, rows, world_size, micro, dynamic, total=None):
    config = ActorConfig(
        global_batch_size=len(rows),
        micro_batch_size_per_device_for_update=micro,
        micro_batch_size_per_device_for_experience=micro,
        loss_avg_mode=mode,
        padding_free=True,
        dynamic_batching=dynamic,
        use_torch_compile=False,
    )
    config.global_batch_size_per_device = len(rows)
    actor = DataParallelPPOActor(config=config, actor_module=nn.Linear(1, 1))
    actor.world_size = world_size
    params = nn.Parameter(torch.zeros(len(LENGTHS), RESPONSE_LENGTH))  # the per-token divergence of every token
    reduced = []

    def all_reduce(tensor, op=None):
        reduced.append(tensor.clone())
        if total is not None:
            tensor.fill_(total)

    def forward(model_inputs, distill_config):
        values = params[model_inputs["input_ids"][:, 0]]
        # with the token weights, loss = -target * param per token: the policy loss of
        # tests/test_loss_averaging.py at its initial point
        return values, -values, torch.zeros(*values.shape, 3)

    monkeypatch.setattr(actor, "_forward_micro_batch_distill", forward)
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    actor.update_policy(_update_data(rows, LENGTHS))
    return params.grad, reduced[0]


def _expected_distill_grad(mode: str) -> torch.Tensor:
    data = _update_data(list(range(len(LENGTHS))), LENGTHS)
    mask = data.batch["response_mask"].float()
    tokens = mask.sum(-1, keepdim=True)
    weights = mask / tokens.clamp(min=1) / (tokens > 0).sum() if mode == "seq" else mask / mask.sum()
    return -data.batch["distill_token_weights"] * weights


@pytest.mark.parametrize("mode", ["seq", "token"])
@pytest.mark.parametrize(
    "world, micro, dynamic",
    [
        ([list(range(8))], 8, False),
        ([list(range(8))], 1, False),
        ([list(range(8))], 2, True),
        ([[0, 2, 3, 5], [1, 4, 6, 7]], 2, False),
    ],
)
def test_distillation_loss_does_not_depend_on_the_split(monkeypatch, mode, world, micro, dynamic):
    total = sum(_distill_update(monkeypatch, mode, rows, len(world), micro, dynamic)[1] for rows in world)
    grads = [_distill_update(monkeypatch, mode, rows, len(world), micro, dynamic, total)[0] for rows in world]
    torch.testing.assert_close(sum(grads) / len(world), _expected_distill_grad(mode))


# ---------------------------------------------------------------------------------------------------------------
# the EMA teacher


from verl.utils import fsdp_utils  # noqa: E402


def _fake_fsdp(shards: list[torch.Tensor], names: list[tuple[str, ...]], buffers: dict | None = None):
    handles = [
        SimpleNamespace(flat_param=SimpleNamespace(_local_shard=shard, _fqns=fqns))
        for shard, fqns in zip(shards, names)
    ]
    buffers = buffers or {}
    return SimpleNamespace(_all_handles=handles, named_buffers=lambda: iter(buffers.items()))


def test_ema_update_pairs_shards_by_parameter_names(monkeypatch):
    monkeypatch.setattr(fsdp_utils, "_lazy_init", lambda *args: None)
    names = [("a.weight", "a.bias"), ("b.weight",)]
    inv_freq = torch.tensor([1.0, 0.1234567])
    actor = _fake_fsdp(
        [torch.full((4,), 2.0, dtype=torch.bfloat16), torch.full((2,), -1.0)],
        names,
        buffers={"rotary.inv_freq": inv_freq.to(torch.bfloat16)},
    )
    ema_inv_freq = inv_freq.clone()
    ema = _fake_fsdp([torch.zeros(4), torch.ones(2)], names, buffers={"rotary.inv_freq": ema_inv_freq})
    fsdp_utils.copy_fsdp_params_(ema, actor)
    assert ema._all_handles[0].flat_param._local_shard.dtype == torch.float32
    assert ema._all_handles[0].flat_param._local_shard.tolist() == [2.0] * 4
    # the copy holds the actor's bf16-rounded buffer values, in its own dtype
    assert ema_inv_freq.dtype == torch.float32
    torch.testing.assert_close(ema_inv_freq, inv_freq.to(torch.bfloat16).float(), rtol=0, atol=0)

    actor._all_handles[0].flat_param._local_shard.fill_(4.0)
    fsdp_utils.ema_update_fsdp_params_(ema, actor, rate=0.05)
    torch.testing.assert_close(ema._all_handles[0].flat_param._local_shard, torch.full((4,), 0.95 * 2.0 + 0.05 * 4.0))
    torch.testing.assert_close(ema._all_handles[1].flat_param._local_shard, torch.full((2,), -1.0))


def test_ema_update_refuses_different_layouts(monkeypatch):
    monkeypatch.setattr(fsdp_utils, "_lazy_init", lambda *args: None)
    actor = _fake_fsdp([torch.zeros(4)], [("a.weight",)])
    with pytest.raises(RuntimeError, match="different parameters"):
        fsdp_utils.ema_update_fsdp_params_(_fake_fsdp([torch.zeros(4)], [("b.weight",)]), actor, rate=0.1)
    with pytest.raises(RuntimeError, match="shape"):
        fsdp_utils.ema_update_fsdp_params_(_fake_fsdp([torch.zeros(3)], [("a.weight",)]), actor, rate=0.1)
    with pytest.raises(RuntimeError, match="flat parameters"):
        fsdp_utils.ema_update_fsdp_params_(_fake_fsdp([], []), actor, rate=0.1)


def _ema_config(**teacher) -> PPOConfig:
    config = _distill_config()
    config.worker.teacher.source = "ema"
    config.worker.teacher.model.model_path = None
    for key, value in teacher.items():
        setattr(config.worker.teacher, key, value)
    return config


def test_ema_teacher_config():
    config = _ema_config()
    config.deep_post_init()
    assert config.worker.teacher.ema_rate == 0.05


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda config: setattr(config.worker.teacher.model, "model_path", "teacher"), "copies the actor"),
        (lambda config: setattr(config.worker.teacher.fsdp, "torch_dtype", "bf16"), "fp32"),
        (lambda config: setattr(config.worker.teacher, "ema_rate", 0.0), "ema_rate"),
        (lambda config: setattr(config.worker.actor.model.lora, "rank", 8), "LoRA"),
    ],
)
def test_invalid_ema_teacher_configs_fail(mutate, message):
    config = _ema_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        config.deep_post_init()


# ---------------------------------------------------------------------------------------------------------------
# rebuilt input views


def _cached_qwen3_vl_processor_path():
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    for repo in ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"):
        found = try_to_load_from_cache(repo, "preprocessor_config.json")
        if isinstance(found, str):
            return str(Path(found).parent)
        modelscope = Path.home() / ".cache" / "modelscope" / "hub" / "models" / repo
        if (modelscope / "preprocessor_config.json").exists():
            return str(modelscope)
    return None


def test_replaced_image_view_rebuilds_the_prompt_and_keeps_the_responses():
    import io

    from PIL import Image

    from verl.trainer.perception_reasoning_data import PerceptionReasoningCorruptionBuilder
    from verl.utils.dataset import ProcessedImageInput
    from verl.utils.tokenizer import get_processor

    path = _cached_qwen3_vl_processor_path()
    if path is None:
        pytest.skip("no Qwen3-VL processor in the local Hugging Face or ModelScope cache")
    processor = get_processor(path, use_fast=True)
    builder = PerceptionReasoningCorruptionBuilder(
        tokenizer=processor.tokenizer, processor=processor, image_patch_size=16, min_pixels=32 * 32, max_pixels=None
    )
    raw_prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is red?"}]}]
    responses = torch.tensor([[11, 12, 13, 0], [21, 22, 0, 0], [31, 32, 33, 34]])
    response_mask = (responses != 0).long()

    def png(size, color):
        buffer = io.BytesIO()
        Image.new("RGB", size, color).save(buffer, format="PNG")
        return {"bytes": buffer.getvalue()}  # as images are stored in the parquet files

    images = [png((64, 64), "red")] * 3
    crops = [png((128, 96), "blue"), png((128, 96), "blue"), png((32, 32), "blue")]
    batch = DataProto.from_dict(
        tensors={"responses": responses, "response_mask": response_mask},
        non_tensors={
            "uid": np.array(["a", "a", "b"], dtype=object),
            "raw_prompt": np.array([raw_prompt] * 3, dtype=object),
            "images": np.array([[image] for image in images], dtype=object),
            "teacher_images": np.array([[crop] for crop in crops], dtype=object),
        },
    )
    student = builder.build_replaced_image_batch(batch, "images").batch
    teacher = builder.build_replaced_image_batch(batch, "teacher_images").batch

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    teacher_ids, teacher_attention = teacher.batch["input_ids"], teacher.batch["attention_mask"]
    # each prompt holds the image tokens of its own crop; all prompts are left-padded to the widest one
    merge = processor.image_processor.merge_size**2
    for row, crop in enumerate(crops):
        grid = processor.image_processor(images=[builder._load_image(crop)])["image_grid_thw"]
        assert (teacher_ids[row] == image_token_id).sum() == grid.prod() // merge
    assert teacher_ids.size(1) > student.batch["input_ids"].size(1)
    assert torch.equal(teacher_ids[:, -4:], responses)
    assert torch.equal(teacher_attention[:, -4:], response_mask)
    assert teacher_attention[2, 0] == 0 and teacher_attention[0, 0] == 1
    assert teacher.batch["position_ids"].shape[0] == 3 and teacher.batch["position_ids"].shape[-1] == teacher_ids.size(
        1
    )
    assert list(teacher.non_tensor_batch["multi_modal_cache_id"]) == ["a", "a", "b"]
    assert all(
        isinstance(data["images"][0], ProcessedImageInput) for data in teacher.non_tensor_batch["multi_modal_data"]
    )

    # the student's and the teacher's views yield the same response rows (the check of the distillation forward)
    actor = _actor()
    main_rows = actor._prepare_response_rows(dict(student.batch), response_mask, 4)
    view_rows = actor._prepare_response_rows(dict(teacher.batch), response_mask, 4)
    assert torch.equal(main_rows["labels"], view_rows["labels"])
    assert main_rows["labels"].tolist() == [11, 12, 13, 21, 22, 31, 32, 33, 34]


def test_replaced_image_view_needs_one_image_per_placeholder():
    from verl.trainer.perception_reasoning_data import PerceptionReasoningCorruptionBuilder

    builder = PerceptionReasoningCorruptionBuilder(tokenizer=None, processor=object(), image_patch_size=16)
    raw_prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "?"}]}]
    batch = DataProto.from_dict(
        tensors={"responses": torch.ones(1, 2, dtype=torch.long), "response_mask": torch.ones(1, 2, dtype=torch.long)},
        non_tensors={
            "raw_prompt": np.array([raw_prompt], dtype=object),
            "teacher_images": np.array([[]], dtype=object),
        },
    )
    with pytest.raises(ValueError, match="1 image placeholders but 0 images"):
        builder.build_replaced_image_batch(batch, "teacher_images")
    with pytest.raises(KeyError, match="crops"):
        builder.build_replaced_image_batch(batch, "crops")


@pytest.mark.parametrize(
    "algorithm, data, message",
    [
        (dict(teacher_view="data_image"), {}, "requires data.teacher_image_key"),
        (dict(teacher_view="crop"), {"teacher_image_key": "teacher_images"}, "teacher_view"),
        (dict(), {"teacher_image_key": "teacher_images"}, "only read with algorithm.teacher_view=data_image"),
    ],
)
def test_invalid_teacher_view_configs_fail(algorithm, data, message):
    config = _distill_config(**algorithm)
    for key, value in data.items():
        setattr(config.data, key, value)
    with pytest.raises(ValueError, match=message):
        config.deep_post_init()


def test_teacher_view_config_needs_no_sequence_parallelism():
    config = _distill_config(teacher_view="data_image")
    config.data.teacher_image_key = "teacher_images"
    config.worker.actor.ulysses_size = 2
    with pytest.raises(ValueError, match="ulysses_size=1"):
        config.deep_post_init()
    config.worker.actor.ulysses_size = 1
    config.deep_post_init()
    assert build_distillation_config(config.algorithm, VOCAB)["views"] == ["data_image"]
