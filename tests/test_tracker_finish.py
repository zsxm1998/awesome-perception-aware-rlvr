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
"""The experiment loggers are closed at the end of training, once, while the trainer still runs."""

from types import SimpleNamespace

import pytest

from verl.trainer import ray_trainer as ray_trainer_module
from verl.trainer.config import PPOConfig
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.utils.logger import logger as logger_module


class _CountingLogger(logger_module.Logger):
    finished = 0

    def __init__(self, config):
        pass

    def log(self, data, step):
        pass

    def finish(self):
        type(self).finished += 1


@pytest.fixture
def counting(monkeypatch):
    _CountingLogger.finished = 0
    monkeypatch.setitem(logger_module.LOGGERS, "counting", _CountingLogger)
    return _CountingLogger


def test_finish_closes_each_logger_once(counting):
    tracker = logger_module.Tracker(loggers=["counting"], config={})
    tracker.finish()
    tracker.finish()
    tracker.__del__()
    assert counting.finished == 1

    unfinished = logger_module.Tracker(loggers=["counting"], config={})
    unfinished.__del__()  # a run that ended without finish(), e.g. by an exception
    assert counting.finished == 2


@pytest.mark.parametrize("val_only", [False, True])
def test_fit_finishes_the_loggers_before_returning(monkeypatch, counting, val_only):
    config = PPOConfig()
    config.trainer.logger = ["counting"]
    config.trainer.val_only = val_only
    config.trainer.val_before_train = val_only
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.training_steps = 0
    trainer.train_dataloader = []
    trainer.val_reward_fn = SimpleNamespace() if val_only else None
    trainer._load_checkpoint = lambda: None
    trainer._save_checkpoint = lambda: None
    trainer._validate = lambda: {"val/reward_score": 0.0}
    monkeypatch.setattr(ray_trainer_module, "tqdm", lambda *args, **kwargs: SimpleNamespace(update=lambda n=1: None))
    trainer.fit()
    assert counting.finished == 1
