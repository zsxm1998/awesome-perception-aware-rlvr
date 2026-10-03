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
"""Online filtering: which groups a criterion keeps, and what each fallback does when rounds come up short."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer import ray_trainer as ray_trainer_module
from verl.trainer.config import PPOConfig
from verl.trainer.ray_trainer import RayPPOTrainer


N = 2  # responses per prompt
ROLLOUT_BATCH_SIZE = 2  # prompts per step
MIXED, SOLVED, FAILED = [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]


def _config(criterion="mean_range", fallback="error", max_try=20) -> PPOConfig:
    config = PPOConfig()
    config.data.rollout_batch_size = ROLLOUT_BATCH_SIZE
    config.worker.rollout.n = N
    config.algorithm.online_filtering = True
    config.algorithm.filter_key = "accuracy"
    config.algorithm.filter_criterion = criterion
    config.algorithm.online_filtering_fallback = fallback
    config.trainer.max_try_make_batch = max_try
    return config


class _Reward:
    """Hands out the scripted accuracy of each round, one list of group scores per round."""

    def __init__(self, rounds):
        self.rounds = iter(rounds)
        self.compute_reward = SimpleNamespace(remote=self._compute)

    def _compute(self, batch):
        scores = [score for group in next(self.rounds) for score in group]
        assert len(scores) == len(batch)
        return torch.tensor(scores)[:, None], {"accuracy": scores}


def _trainer(monkeypatch, config: PPOConfig, rounds) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    prompt_ids = itertools.count()
    trainer.config = config
    trainer.global_step = 1
    trainer.grounding_consistency_scorer = None
    trainer.reward_fn = _Reward(rounds)
    trainer.data_iterator = (
        {"prompt_id": torch.tensor([next(prompt_ids) for _ in range(ROLLOUT_BATCH_SIZE)])} for _ in itertools.count()
    )
    trainer.actor_rollout_ref_wg = SimpleNamespace(
        generate_sequences=lambda gen_batch: DataProto.from_dict({"responses": torch.zeros(len(gen_batch) * N, 1)})
    )
    trainer._pop_rollout_inputs = lambda batch, meta_info_keys: batch.select(batch_keys=["prompt_id"])
    trainer._attach_grounding_consistency_reward_inputs = lambda *args, **kwargs: None
    monkeypatch.setattr(ray_trainer_module.ray, "get", lambda value: value)
    return trainer


def _make(monkeypatch, rounds, **config_kwargs):
    metrics = {}
    batch = _trainer(monkeypatch, _config(**config_kwargs), rounds)._make_batch_data(metrics)
    return batch.batch["prompt_id"][::N].tolist(), metrics


def test_criteria():
    groups = {"mixed": MIXED, "solved": SOLVED, "failed": FAILED, "flat": [0.5, 0.5], "partial": [0.2, 0.9]}
    uids = [uid for uid, scores in groups.items() for _ in scores]
    scores = [score for group in groups.values() for score in group]

    def kept(criterion):
        trainer = SimpleNamespace(config=_config(criterion))
        return {uids[idx] for idx in RayPPOTrainer._select_filtered_sample_idxs(trainer, uids, scores)}

    assert kept("mean_range") == {"mixed", "flat", "partial"}
    assert kept("std") == {"mixed", "partial"}


def test_criteria_agree_on_binary_scores():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 2, size=8 * 64).astype(float).tolist()
    uids = [str(idx // 8) for idx in range(len(scores))]
    selected = [
        RayPPOTrainer._select_filtered_sample_idxs(SimpleNamespace(config=_config(criterion)), uids, scores)
        for criterion in ("mean_range", "std")
    ]
    assert selected[0] == selected[1]


def test_rounds_accumulate_kept_groups(monkeypatch):
    prompt_ids, metrics = _make(monkeypatch, [[MIXED, SOLVED], [FAILED, MIXED]])
    assert prompt_ids == [0, 3]
    assert "reward/filter_first_round_fallback" not in metrics and "reward/accuracy" in metrics


def test_error_fallback(monkeypatch):
    with pytest.raises(RuntimeError, match="No sample is kept"):
        _make(monkeypatch, [[SOLVED, FAILED]])
    with pytest.raises(RuntimeError, match="Generated too many"):
        _make(monkeypatch, [[MIXED, SOLVED]], max_try=1)


def test_keep_round_fallback(monkeypatch):
    """A round that keeps no group is kept whole (PAPO's code); running out of rounds is still an error."""
    prompt_ids, _ = _make(monkeypatch, [[SOLVED, FAILED], [MIXED, MIXED]], fallback="keep_round")
    assert prompt_ids == [0, 1]
    prompt_ids, _ = _make(monkeypatch, [[MIXED, SOLVED], [FAILED, SOLVED]], fallback="keep_round")
    assert prompt_ids == [0, 2]  # the second round is kept whole, then cut to ROLLOUT_BATCH_SIZE prompts
    with pytest.raises(RuntimeError, match="Generated too many"):
        _make(monkeypatch, [[MIXED, SOLVED], [MIXED, FAILED]], fallback="keep_round", max_try=1)


def test_first_round_fallback(monkeypatch):
    """A round that keeps no group is skipped; after the last round the first one is used unfiltered (ms-swift)."""
    rounds = [[MIXED, SOLVED], [SOLVED, FAILED], [FAILED, SOLVED]]
    prompt_ids, metrics = _make(monkeypatch, rounds, criterion="std", fallback="first_round", max_try=3)
    assert prompt_ids == [0, 1] and metrics["reward/filter_first_round_fallback"] == 1.0

    rounds = [[SOLVED, FAILED], [MIXED, SOLVED], [SOLVED, MIXED]]
    prompt_ids, metrics = _make(monkeypatch, rounds, criterion="std", fallback="first_round", max_try=3)
    assert prompt_ids == [2, 5] and metrics["reward/filter_first_round_fallback"] == 0.0


def test_first_round_needs_a_round_limit():
    config = _config(fallback="first_round", max_try=-1)
    config.trainer.n_gpus_per_node = 1
    config.worker.actor.global_batch_size = ROLLOUT_BATCH_SIZE
    config.worker.actor.micro_batch_size_per_device_for_update = 1
    config.worker.actor.micro_batch_size_per_device_for_experience = 1
    with pytest.raises(ValueError, match="max_try_make_batch"):
        config.post_init()
    with pytest.raises(ValueError, match="online_filtering_fallback"):
        _config(fallback="retry").algorithm.post_init()


def test_first_round_needs_whole_rounds():
    """With rounds smaller than the rollout batch, the first round alone cannot fill a training batch."""
    config = _config(fallback="first_round", max_try=3)
    config.trainer.n_gpus_per_node = 1
    config.worker.actor.global_batch_size = ROLLOUT_BATCH_SIZE
    config.worker.actor.micro_batch_size_per_device_for_update = 1
    config.worker.actor.micro_batch_size_per_device_for_experience = 1
    config.data.mini_rollout_batch_size = ROLLOUT_BATCH_SIZE // 2
    with pytest.raises(ValueError, match="mini_rollout_batch_size"):
        config.post_init()
    config.data.mini_rollout_batch_size = ROLLOUT_BATCH_SIZE
    config.post_init()
    config.algorithm.online_filtering_fallback = "keep_round"  # accumulates rounds; partial rounds are fine
    config.data.mini_rollout_batch_size = ROLLOUT_BATCH_SIZE // 2
    config.post_init()
