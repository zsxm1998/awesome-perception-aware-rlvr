# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import json
import os
import shutil
import uuid

import pytest
from transformers import GenerationConfig, PretrainedConfig

from verl.utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from verl.utils.checkpoint.fsdp_checkpoint_manager import _save_generation_config


class _ModelWithoutGenerationConfig:
    generation_config = None
    config = PretrainedConfig(eos_token_id=2, pad_token_id=0)


class _ModelWithGenerationConfig:
    generation_config = GenerationConfig(max_new_tokens=7)
    config = PretrainedConfig(eos_token_id=2, pad_token_id=0)


@pytest.fixture
def save_checkpoint_path():
    ckpt_dir = os.path.join("checkpoints", str(uuid.uuid4()))
    os.makedirs(ckpt_dir, exist_ok=True)
    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_find_latest_ckpt(save_checkpoint_path):
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as f:
        json.dump({"last_global_step": 10}, f, ensure_ascii=False, indent=2)

    assert find_latest_ckpt(save_checkpoint_path)[0] is None
    os.makedirs(os.path.join(save_checkpoint_path, "global_step_10"), exist_ok=True)
    assert find_latest_ckpt(save_checkpoint_path)[0] == os.path.join(save_checkpoint_path, "global_step_10")


def test_remove_obsolete_ckpt(save_checkpoint_path):
    for step in range(5, 30, 5):
        os.makedirs(os.path.join(save_checkpoint_path, f"global_step_{step}"), exist_ok=True)

    remove_obsolete_ckpt(save_checkpoint_path, global_step=30, best_global_step=10, save_limit=3)
    for step in range(5, 30, 5):
        is_exist = step in [10, 25]
        assert os.path.exists(os.path.join(save_checkpoint_path, f"global_step_{step}")) == is_exist


def test_save_generation_config_falls_back_to_model_config(tmp_path):
    _save_generation_config(_ModelWithoutGenerationConfig(), str(tmp_path))

    generation_config = GenerationConfig.from_pretrained(tmp_path)
    assert generation_config.eos_token_id == 2
    assert generation_config.pad_token_id == 0


def test_save_generation_config_uses_existing_generation_config(tmp_path):
    _save_generation_config(_ModelWithGenerationConfig(), str(tmp_path))

    generation_config = GenerationConfig.from_pretrained(tmp_path)
    assert generation_config.max_new_tokens == 7


class _FakeActorWorkerGroup:
    def __init__(self):
        self.fail = False

    def save_checkpoint(self, path, save_model_only=False):
        os.makedirs(path, exist_ok=True)
        if self.fail:
            raise RuntimeError("node lost while saving")
        with open(os.path.join(path, "model_world_size_1_rank_0.pt"), "wb") as f:
            f.write(b"x")


def _trainer(path, save_limit):
    from types import SimpleNamespace

    from verl.trainer.ray_trainer import RayPPOTrainer

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(save_checkpoint_path=str(path), save_limit=save_limit, save_model_only=False)
    )
    trainer.val_reward_score = None
    trainer.best_val_reward_score = -1.0
    trainer.best_global_step = None
    trainer.use_critic = False
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {})
    trainer.actor_rollout_ref_wg = _FakeActorWorkerGroup()
    return trainer


@pytest.mark.parametrize("save_limit", [1, 2])
def test_failed_save_keeps_the_checkpoint_to_resume_from(tmp_path, save_limit):
    """Old checkpoints are removed after the new one is saved, also when the best step is older."""
    trainer = _trainer(tmp_path, save_limit=save_limit)
    for step, score in ((5, 0.5), (10, 0.4), (15, 0.3)):
        trainer.global_step, trainer.val_reward_score = step, score
        trainer._save_checkpoint()
    assert sorted(os.listdir(tmp_path)) == [CHECKPOINT_TRACKER, "global_step_15", "global_step_5"]

    trainer.global_step, trainer.val_reward_score = 20, 0.2
    trainer.actor_rollout_ref_wg.fail = True
    with pytest.raises(RuntimeError):
        trainer._save_checkpoint()

    path, tracker = find_latest_ckpt(str(tmp_path))
    assert path == os.path.join(str(tmp_path), "global_step_15")
    assert tracker["best_global_step"] == 5


def test_save_limit_one_keeps_the_latest_and_the_best(tmp_path):
    trainer = _trainer(tmp_path, save_limit=1)
    for step, score in ((5, 0.3), (10, 0.5), (15, 0.4), (20, 0.6)):
        trainer.global_step, trainer.val_reward_score = step, score
        trainer._save_checkpoint()
        expected = {10: ["global_step_10"], 15: ["global_step_10", "global_step_15"]}.get(
            step, [f"global_step_{step}"]
        )
        assert sorted(os.listdir(tmp_path)) == [CHECKPOINT_TRACKER, *expected]


def test_no_best_step_without_validation(tmp_path):
    trainer = _trainer(tmp_path, save_limit=2)
    for step in (10, 20, 30):
        trainer.global_step = step
        trainer._save_checkpoint()

    _, tracker = find_latest_ckpt(str(tmp_path))
    assert tracker["best_global_step"] is None and tracker["last_global_step"] == 30
    assert sorted(os.listdir(tmp_path)) == [CHECKPOINT_TRACKER, "global_step_20", "global_step_30"]


def test_finalized_runs_are_not_resumed(tmp_path):
    os.makedirs(tmp_path / "global_step_10")
    tracker = {"last_global_step": 10, "best_global_step": 10, "finalized": {"keep": "last", "steps": [10]}}
    (tmp_path / CHECKPOINT_TRACKER).write_text(json.dumps(tracker))
    with pytest.raises(RuntimeError, match="finalized"):
        find_latest_ckpt(str(tmp_path))
