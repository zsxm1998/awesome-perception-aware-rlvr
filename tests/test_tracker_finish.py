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


class _FailingLogger(_CountingLogger):
    def finish(self):
        raise RuntimeError("upload failed")


def _swanlab_logger(monkeypatch, finish, timeout=0.2):
    """A SwanlabLogger without swanlab.init, whose swanlab.finish() is `finish`."""
    monkeypatch.setattr(logger_module, "swanlab", SimpleNamespace(finish=finish), raising=False)
    swanlab_logger = object.__new__(logger_module.SwanlabLogger)
    swanlab_logger.finish_timeout = timeout
    return swanlab_logger


def _returns_within(function, seconds):
    """Whether `function` returns within `seconds` (a finish without a bound would block forever)."""
    import threading

    thread = threading.Thread(target=function, daemon=True)
    thread.start()
    thread.join(seconds)
    return not thread.is_alive()


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_swanlab_finish_stops_waiting_for_a_stalled_upload(monkeypatch, capsys):
    import threading

    release = threading.Event()  # like SwanLab's finish when its upload thread stopped during the run
    swanlab_logger = _swanlab_logger(monkeypatch, release.wait)
    try:
        assert _returns_within(swanlab_logger.finish, 10.0)
        assert "did not return within 0.2 s" in capsys.readouterr().err
    finally:
        release.set()


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_swanlab_finish_errors_reach_the_caller(monkeypatch):
    def fail():
        raise RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        _swanlab_logger(monkeypatch, fail).finish()


def test_a_failing_logger_does_not_skip_the_others(monkeypatch, counting, capsys):
    monkeypatch.setitem(logger_module.LOGGERS, "failing", _FailingLogger)
    tracker = logger_module.Tracker(loggers=["failing", "counting"], config={})
    tracker.finish()
    assert counting.finished == 1
    assert "_FailingLogger.finish() failed: RuntimeError('upload failed')" in capsys.readouterr().err


def test_other_loggers_close_in_the_calling_thread(monkeypatch):
    """W&B's console redirect restores signal handlers when it closes, which works only in the main thread."""
    import threading

    threads = []

    class _ThreadLogger(_CountingLogger):
        def finish(self):
            threads.append(threading.current_thread())

    monkeypatch.setitem(logger_module.LOGGERS, "thread", _ThreadLogger)
    logger_module.Tracker(loggers=["thread"], config={}).finish()
    assert threads == [threading.current_thread()]


def test_no_finish_while_the_interpreter_shuts_down(counting):
    tracker = logger_module.Tracker(loggers=["counting"], config={})
    tracker.__del__(_is_finalizing=lambda: True)
    assert counting.finished == 0
    tracker.__del__(_is_finalizing=lambda: False)
    assert counting.finished == 1


def test_an_unfinished_tracker_does_not_hold_the_exit():
    """An unfinished tracker with a stalled SwanLab, collected at interpreter shutdown: the process still exits."""
    import subprocess
    import sys
    from pathlib import Path

    script = """
import threading
from types import SimpleNamespace
from verl.utils.logger import logger as L
L.swanlab = SimpleNamespace(finish=threading.Event().wait)  # never returns
swanlab_logger = object.__new__(L.SwanlabLogger)
swanlab_logger.finish_timeout = 0.1
tracker = object.__new__(L.Tracker)
tracker.loggers, tracker._finished = [swanlab_logger], False
"""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("value", ["abc", "", "nan", "inf", "1e300", "0", "-1"])
def test_finish_timeout_is_checked_when_the_logger_starts(monkeypatch, value):
    monkeypatch.setenv("LOGGER_FINISH_TIMEOUT", value)
    with pytest.raises(ValueError, match="LOGGER_FINISH_TIMEOUT"):
        logger_module.logger_finish_timeout()


def test_finish_timeout_default_and_override(monkeypatch):
    monkeypatch.delenv("LOGGER_FINISH_TIMEOUT", raising=False)
    assert logger_module.logger_finish_timeout() == 300.0
    monkeypatch.setenv("LOGGER_FINISH_TIMEOUT", "0.5")
    assert logger_module.logger_finish_timeout() == 0.5
