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
"""scripts/finalize_run.py: keep the asked steps (default: the last and the best) as Hugging Face weights, delete the
rest; or one step."""

import io
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

import finalize_run  # noqa: E402
from easyr1_eval.checkpoints import resolve_eval_targets  # noqa: E402

from verl.utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt  # noqa: E402


def _write_weights(hf_dir: Path) -> None:
    save_file({"lm_head.weight": torch.zeros(2, 2)}, str(hf_dir / "model.safetensors"))


def _run(root: Path, steps=(5, 10, 15), last=15, best=10, merged=True) -> Path:
    """A run directory as the trainer leaves it: FSDP states, dataloader.pt and actor/huggingface/."""
    run = root / "checkpoints" / "comparison" / "grpo"
    for step in steps:
        actor = run / f"global_step_{step}" / "actor"
        (actor / "huggingface").mkdir(parents=True)
        for name in ("model", "optim", "extra_state"):
            (actor / f"{name}_world_size_1_rank_0.pt").write_bytes(b"x" * 100)
        (actor.parent / "dataloader.pt").write_bytes(b"x")
        (actor / "huggingface" / "config.json").write_text("{}", encoding="utf-8")
        (actor / "huggingface" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        if merged:
            _write_weights(actor / "huggingface")
    tracker = {"best_global_step": best, "best_val_reward_score": 0.7, "last_global_step": last}
    (run / CHECKPOINT_TRACKER).write_text(json.dumps(tracker), encoding="utf-8")
    (run / "completions.jsonl").write_text("{}\n", encoding="utf-8")
    return run


def _steps(run: Path) -> list[str]:
    return sorted(path.name for path in run.glob("global_step_*"))


def _assert_finalized_step(step_dir: Path) -> None:
    actor = step_dir / "actor"
    assert (actor / "config.json").is_file() and (actor / "model.safetensors").is_file()
    assert (actor / "tokenizer_config.json").is_file()
    assert not list(actor.glob("*.pt")) and not (actor / "huggingface").exists()
    assert not (step_dir / "dataloader.pt").exists()


def test_keep_last_and_best_by_default(tmp_path):
    run = _run(tmp_path)
    assert finalize_run.main([str(run), "--yes"]) == 0

    assert _steps(run) == ["global_step_10", "global_step_15"]
    _assert_finalized_step(run / "global_step_10")
    _assert_finalized_step(run / "global_step_15")
    assert (run / "completions.jsonl").is_file()
    tracker = json.loads((run / CHECKPOINT_TRACKER).read_text())
    assert tracker["finalized"] == {"keep": "last,best", "steps": [10, 15]} and tracker["best_global_step"] == 10
    with pytest.raises(RuntimeError, match="finalized"):
        find_latest_ckpt(str(run))
    (target,) = resolve_eval_targets(str(run))
    assert (target.model, target.label) == (str(run / "global_step_15" / "actor"), "grpo/global_step_15")

    assert finalize_run.main([str(run), "--yes"]) == 0  # running it again changes nothing
    assert _steps(run) == ["global_step_10", "global_step_15"]


@pytest.mark.parametrize(
    "keep,kept",
    [
        ("last", ["global_step_15"]),
        ("best", ["global_step_10"]),
        ("both", ["global_step_10", "global_step_15"]),
        ("best,last", ["global_step_10", "global_step_15"]),
        ("all", ["global_step_10", "global_step_15", "global_step_5"]),
        ("last,5", ["global_step_15", "global_step_5"]),
    ],
)
def test_keep_choices(tmp_path, keep, kept):
    run = _run(tmp_path)
    assert finalize_run.main([str(run), "--keep", keep, "--yes"]) == 0
    assert _steps(run) == kept
    for name in kept:
        _assert_finalized_step(run / name)


def test_keep_is_an_unordered_set():
    assert (
        finalize_run.parse_keep("best,last") == finalize_run.parse_keep("last,best") == finalize_run.parse_keep("both")
    )
    assert finalize_run.parse_keep(" Last , 100,50 ").text == "last,50,100"
    assert finalize_run.parse_keep("all,last").text == "all,last"


@pytest.mark.parametrize("keep", ["first", "last,", "last;best", "-5"])
def test_rejects_unknown_keep(tmp_path, keep):
    with pytest.raises(SystemExit):
        finalize_run.parse_args([str(tmp_path), "--keep", keep])


def test_an_absent_step_is_an_error(tmp_path):
    run = _run(tmp_path)
    assert finalize_run.main([str(run), "--keep", "last,7", "--yes"]) == 1
    assert _steps(run) == ["global_step_10", "global_step_15", "global_step_5"]


def test_best_gone_after_keeping_only_the_last(tmp_path):
    run = _run(tmp_path)
    assert finalize_run.main([str(run), "--keep", "last", "--yes"]) == 0
    assert _steps(run) == ["global_step_15"]

    plan = finalize_run.make_plan(run, None)  # the default asks for the best step too: a note, nothing deleted
    assert plan.keep == {15: ["last"]} and not plan.delete
    assert "global_step_10 no longer exists (finalized earlier with --keep last)" in plan.notes[0]
    assert finalize_run.main([str(run), "--yes"]) == 0
    assert finalize_run.main([str(run), "--keep", "best", "--yes"]) == 1  # the best step alone: an error
    assert _steps(run) == ["global_step_15"]


def test_narrowing_an_earlier_finalization_says_what_it_deletes(tmp_path):
    run = _run(tmp_path)
    assert finalize_run.main([str(run), "--yes"]) == 0
    plan = finalize_run.make_plan(run, "last")
    assert [path.name for path in plan.delete] == ["global_step_10"]
    assert any("kept by the earlier finalization: global_step_10" in note for note in plan.notes)


def test_best_equal_to_last_is_kept_once(tmp_path):
    run = _run(tmp_path, best=15)
    plan = finalize_run.make_plan(run, "both")
    assert plan.keep == {15: ["last", "best"]}
    assert [path.name for path in plan.delete] == ["global_step_5", "global_step_10"]


def test_without_validation(tmp_path):
    run = _run(tmp_path, best=None)
    assert finalize_run.main([str(run), "--keep", "best", "--yes"]) == 1
    assert _steps(run) == ["global_step_10", "global_step_15", "global_step_5"]
    plan = finalize_run.make_plan(run, None)
    assert plan.keep == {15: ["last"]} and "no best step" in plan.notes[0]


def test_dry_run_and_declined_confirmation_change_nothing(tmp_path, monkeypatch):
    run = _run(tmp_path)
    before = sorted(str(path) for path in run.rglob("*"))

    assert finalize_run.main([str(run), "--dry-run"]) == 0
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))  # not a terminal: --yes is required
    assert finalize_run.main([str(run)]) == 1
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert finalize_run.main([str(run)]) == 1

    assert sorted(str(path) for path in run.rglob("*")) == before


def test_refuses_a_step_newer_than_the_tracker(tmp_path):
    """A global_step_* newer than the tracker means a save is in progress, or was interrupted."""
    run = _run(tmp_path)
    other = _run(tmp_path / "other")
    (run / "global_step_20" / "actor").mkdir(parents=True)

    assert finalize_run.main([str(other), str(run), "--yes"]) == 1
    assert len(_steps(run)) == 4 and len(_steps(other)) == 3  # no run is touched


def test_merges_fsdp_shards_first(tmp_path, monkeypatch):
    run = _run(tmp_path, merged=False)
    merged = []

    def fake_merger(actor_dir):
        assert (actor_dir / "model_world_size_1_rank_0.pt").is_file()
        _write_weights(actor_dir / "huggingface")
        merged.append(actor_dir.parent.name)

    monkeypatch.setattr(finalize_run, "run_merger", fake_merger)
    assert finalize_run.main([str(run), "--yes"]) == 0
    assert merged == ["global_step_10", "global_step_15"]
    _assert_finalized_step(run / "global_step_10")
    _assert_finalized_step(run / "global_step_15")


def test_failed_merge_deletes_nothing(tmp_path, monkeypatch):
    run = _run(tmp_path, merged=False)

    def failing_merger(actor_dir):
        raise finalize_run.FinalizeError("model_merger.py failed")

    monkeypatch.setattr(finalize_run, "run_merger", failing_merger)
    assert finalize_run.main([str(run), "--yes"]) == 1
    assert len(_steps(run)) == 3
    assert (run / "global_step_15" / "actor" / "optim_world_size_1_rank_0.pt").is_file()
    assert "finalized" not in json.loads((run / CHECKPOINT_TRACKER).read_text())


@pytest.mark.parametrize("suffix", ["", "/actor"])
def test_single_step_leaves_the_rest_alone(tmp_path, suffix):
    run = _run(tmp_path)
    tracker = (run / CHECKPOINT_TRACKER).read_text()
    assert finalize_run.main([f"{run}/global_step_10{suffix}", "--yes"]) == 0

    _assert_finalized_step(run / "global_step_10")
    assert _steps(run) == ["global_step_10", "global_step_15", "global_step_5"]
    assert (run / "global_step_15" / "actor" / "optim_world_size_1_rank_0.pt").is_file()
    assert (run / CHECKPOINT_TRACKER).read_text() == tracker
    assert finalize_run.main([f"{run}/global_step_10", "--keep", "best", "--yes"]) == 1  # --keep is for runs


def test_project_directories_and_runs_without_checkpoints(tmp_path):
    run = _run(tmp_path)
    (run.parent / "crashed_before_saving").mkdir()
    (run.parent / "crashed_before_saving" / "experiment_log.jsonl").write_text("{}\n")

    assert finalize_run.main([str(run.parent), "--yes"]) == 1  # the project directory holds several runs
    assert finalize_run.main([str(run.parent / "crashed_before_saving"), "--yes"]) == 0  # nothing to do
    assert finalize_run.main([*map(str, run.parent.iterdir()), "--yes"]) == 0  # shell glob over the project
    assert _steps(run) == ["global_step_10", "global_step_15"]
