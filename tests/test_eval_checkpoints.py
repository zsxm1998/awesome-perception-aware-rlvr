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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval import oneclick  # noqa: E402
from easyr1_eval.checkpoints import resolve_eval_targets  # noqa: E402


def _hf_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"x")
    return path


def _actor(step_dir: Path, *, merged: bool = False) -> Path:
    actor = step_dir / "actor"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "huggingface" / "config.json").write_text("{}", encoding="utf-8")
    (actor / "model_world_size_2_rank_0.pt").write_bytes(b"x")
    (actor / "model_world_size_2_rank_1.pt").write_bytes(b"x")
    if merged:
        (actor / "huggingface" / "model.safetensors").write_bytes(b"x")
    return actor


def test_hf_id_and_merged_directory(tmp_path):
    (target,) = resolve_eval_targets("Qwen/Qwen2.5-VL-3B-Instruct")
    assert (target.model, target.run_name, target.step, target.needs_merge) == (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        None,
        False,
    )
    merged = _hf_dir(tmp_path / "my_model")
    (target,) = resolve_eval_targets(str(merged))
    assert (target.model, target.run_name, target.label) == (str(merged), "my_model", "my_model")
    with pytest.raises(FileNotFoundError):
        resolve_eval_targets("./checkpoints/does/not/exist")


def test_actor_step_and_run_root_resolution(tmp_path):
    run = tmp_path / "checkpoints" / "papo" / "qwen_grpo_papo"
    actor_10 = _actor(run / "global_step_10", merged=True)
    actor_200 = _actor(run / "global_step_200")
    _actor(run / "global_step_50")

    (target,) = resolve_eval_targets(str(actor_200))
    assert target.needs_merge
    assert (target.run_name, target.step, target.label) == (
        "qwen_grpo_papo",
        "global_step_200",
        "qwen_grpo_papo/global_step_200",
    )
    assert target.model == str(actor_200 / "huggingface")

    (target,) = resolve_eval_targets(str(run / "global_step_10"))
    assert not target.needs_merge and target.actor_dir == actor_10

    (target,) = resolve_eval_targets(str(actor_10 / "huggingface"))
    assert (target.run_name, target.step) == ("qwen_grpo_papo", "global_step_10")

    (latest,) = resolve_eval_targets(str(run))
    assert latest.step == "global_step_200"
    every = resolve_eval_targets(str(run), all_steps=True)
    assert [item.step for item in every] == ["global_step_10", "global_step_50", "global_step_200"]
    renamed = resolve_eval_targets(str(run), run_name="papo_3b")
    assert renamed[0].label == "papo_3b/global_step_200"


def test_split_args_separates_wrapper_options():
    model, own, passthrough = oneclick.split_args(
        ["ckpt", "--suite", "papo", "--all-steps", "--run-name=x", "--limit", "8", "--results-root", "/r"]
    )
    assert model == "ckpt"
    assert own == {"--all-steps": True, "--run-name": "x", "--results-root": "/r"}
    assert passthrough == ["--suite", "papo", "--limit", "8"]
    with pytest.raises(SystemExit):
        oneclick.split_args(["--suite", "papo"])


def test_oneclick_merges_and_launches_one_runner_per_step(tmp_path, monkeypatch):
    run = tmp_path / "ckpts" / "exp"
    _actor(run / "global_step_1")
    _actor(run / "global_step_2", merged=True)
    merged, commands = [], []
    monkeypatch.setattr(oneclick, "merge_actor_checkpoint", lambda target: merged.append(target.step))
    monkeypatch.setattr(
        oneclick.subprocess,
        "run",
        lambda command, check=False: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert oneclick.main([str(run), "--all-steps", "--results-root", str(tmp_path / "res"), "--suite", "papo"]) == 0

    assert merged == ["global_step_1"]
    assert len(commands) == 2
    first = commands[0]
    assert first[first.index("--model") + 1] == str(run / "global_step_1" / "actor" / "huggingface")
    assert first[first.index("--output-dir") + 1] == str(tmp_path / "res" / "exp" / "global_step_1")
    assert first[first.index("--run-name") + 1] == "exp/global_step_1"
    assert first[-2:] == ["--suite", "papo"]


def test_oneclick_reports_runner_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(oneclick.subprocess, "run", lambda command, check=False: SimpleNamespace(returncode=1))
    assert oneclick.main(["Qwen/Qwen2.5-VL-3B-Instruct", "--results-root", str(tmp_path)]) == 1
