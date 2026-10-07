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
"""examples/comparison/opd_qwen3_vl_2b/common.sh: which teacher the OPD comparison passes to the trainer, and when
the GRPO run's best step counts as a usable teacher."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "examples" / "comparison" / "opd_qwen3_vl_2b" / "common.sh"
VCSD_ARGS = ("worker.teacher.source=ema", "worker.teacher.model.model_path=null")


def _bash(script: str, *args: str, **env: str) -> subprocess.CompletedProcess:
    environ = {k: v for k, v in os.environ.items() if k not in {"TEACHER_PATH", "GRPO_TEACHER_RUN", "DRY_RUN"}}
    environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{environ.get('PATH', '')}"  # python3 of this env
    environ.update(env)
    return subprocess.run(
        ["bash", "-c", script, "bash", *args], env=environ, capture_output=True, text=True, cwd=ROOT, check=False
    )


def _launch(algo_args=(), cli_args=(), **env: str) -> subprocess.CompletedProcess:
    """Run launch_opd_comparison with launch_training replaced by a print of the final override list."""
    script = (
        f'source "{COMMON}"\n'
        "launch_training() {\n"
        '    printf "%s\\n" "${METHOD_COMMON_ARGS[@]}" "${ALGO_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"\n'
        "}\n"
        "EXPERIMENT_NAME=test\n"
        f"ALGO_ARGS=({' '.join(repr(a) for a in algo_args)})\n"
        "EXTRA_ARGS=()\n"
        'launch_opd_comparison "$@"\n'
    )
    return _bash(script, *cli_args, **env)


def _effective(stdout: str) -> dict[str, str]:
    """The teacher settings the trainer ends up with: as in the config merge, the last override wins."""
    settings = {}
    for line in stdout.splitlines():
        key, _, value = line.partition("=")
        if key.startswith("worker.teacher."):
            settings[key] = value
    return settings


def _teacher_path(run: Path, dry_run: bool = False) -> subprocess.CompletedProcess:
    env = {"GRPO_TEACHER_RUN": str(run)}
    if dry_run:
        env["DRY_RUN"] = "1"
    return _bash(f'source "{COMMON}"\ngrpo_teacher_path', **env)


def _grpo_run(tmp_path: Path, best: int = 175) -> Path:
    run = tmp_path / "grpo"
    (run / f"global_step_{best}" / "actor").mkdir(parents=True)
    (run / "checkpoint_tracker.json").write_text(json.dumps({"last_global_step": 202, "best_global_step": best}))
    return run


def _write_config(hf_dir: Path) -> None:
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (hf_dir / "tokenizer_config.json").write_text("{}")


def _write_weights(hf_dir: Path) -> None:
    _write_config(hf_dir)
    save_file({"lm_head.weight": torch.zeros(2, 2)}, str(hf_dir / "model.safetensors"))


def test_command_line_teacher_skips_the_grpo_run(tmp_path):
    result = _launch(
        cli_args=("worker.teacher.model.model_path=Qwen/Qwen3-VL-8B-Instruct",),
        GRPO_TEACHER_RUN=str(tmp_path / "missing"),
    )
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout) == {
        "worker.teacher.source": "model",
        "worker.teacher.model.model_path": "Qwen/Qwen3-VL-8B-Instruct",
    }


def test_teacher_path_variable_skips_the_grpo_run(tmp_path):
    result = _launch(TEACHER_PATH="/teachers/a", GRPO_TEACHER_RUN=str(tmp_path / "missing"))
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout)["worker.teacher.model.model_path"] == "/teachers/a"


def test_command_line_teacher_overrides_teacher_path_variable(tmp_path):
    result = _launch(cli_args=("worker.teacher.model.model_path=/teachers/b",), TEACHER_PATH="/teachers/a")
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout)["worker.teacher.model.model_path"] == "/teachers/b"


def test_default_teacher_is_the_grpo_best_step(tmp_path):
    run = _grpo_run(tmp_path)
    _write_weights(run / "global_step_175" / "actor")
    result = _launch(GRPO_TEACHER_RUN=str(run))
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout) == {
        "worker.teacher.source": "model",
        "worker.teacher.model.model_path": str(run / "global_step_175" / "actor"),
    }


def test_self_distillation_resolves_no_teacher(tmp_path):
    result = _launch(algo_args=VCSD_ARGS, GRPO_TEACHER_RUN=str(tmp_path / "missing"))
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout) == {"worker.teacher.source": "ema", "worker.teacher.model.model_path": "null"}


def test_command_line_source_decides_by_precedence(tmp_path):
    # ema on the command line overrides a method's external teacher: nothing to resolve
    result = _launch(cli_args=("worker.teacher.source=ema",), GRPO_TEACHER_RUN=str(tmp_path / "missing"))
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout) == {"worker.teacher.source": "ema"}
    # model on the command line overrides the method's ema, and the default teacher fills the method's null path
    result = _launch(algo_args=VCSD_ARGS, cli_args=("worker.teacher.source=model",), TEACHER_PATH="/teachers/a")
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout) == {
        "worker.teacher.source": "model",
        "worker.teacher.model.model_path": "/teachers/a",
    }
    # a later ema wins over an earlier model
    result = _launch(
        cli_args=("worker.teacher.source=model", "worker.teacher.source=ema"),
        GRPO_TEACHER_RUN=str(tmp_path / "missing"),
    )
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout) == {"worker.teacher.source": "ema"}


def test_missing_grpo_run_fails_before_training(tmp_path):
    result = _launch(GRPO_TEACHER_RUN=str(tmp_path / "missing"))
    assert result.returncode != 0
    assert "no GRPO teacher" in result.stderr and "worker." not in result.stdout


def test_finalized_and_merged_weights_are_found(tmp_path):
    run = _grpo_run(tmp_path)
    actor = run / "global_step_175" / "actor"
    _write_weights(actor / "huggingface")  # merged, not finalized
    result = _teacher_path(run)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(actor / "huggingface")
    _write_weights(actor)  # finalized: actor/ comes first
    result = _teacher_path(run)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(actor)


def test_unmerged_checkpoint_is_not_a_teacher(tmp_path):
    run = _grpo_run(tmp_path)
    actor = run / "global_step_175" / "actor"
    _write_config(actor / "huggingface")  # what the trainer saves next to the FSDP shards
    (actor / "model_world_size_4_rank_0.pt").write_bytes(b"shard")
    result = _teacher_path(run)
    assert result.returncode != 0
    assert "finalize_run.py" in result.stderr
    # the configuration check does not load the teacher
    result = _teacher_path(run, dry_run=True)
    assert result.returncode == 0, result.stderr


def test_config_without_weights_does_not_hide_merged_weights(tmp_path):
    run = _grpo_run(tmp_path)
    actor = run / "global_step_175" / "actor"
    _write_config(actor)  # a stray config.json in actor/
    _write_weights(actor / "huggingface")
    result = _teacher_path(run)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(actor / "huggingface")


def test_sharded_weights_need_every_shard(tmp_path):
    run = _grpo_run(tmp_path)
    hf_dir = run / "global_step_175" / "actor" / "huggingface"
    _write_config(hf_dir)
    save_file({"a": torch.zeros(1)}, str(hf_dir / "model-00001-of-00002.safetensors"))
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    result = _teacher_path(run)
    assert result.returncode != 0
    save_file({"b": torch.zeros(1)}, str(hf_dir / "model-00002-of-00002.safetensors"))
    result = _teacher_path(run)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(hf_dir)
