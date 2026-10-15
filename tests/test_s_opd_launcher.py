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
"""examples/reproduction/s_opd/common.sh: the teacher each script trains with, and the S-OPD terms it adds."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "examples" / "reproduction" / "s_opd" / "common.sh"


def _launch(terms: str = "", data: str = "virl39k", teacher: str = "8b", cli_args=(), **env: str):
    """Run launch_s_opd with launch_training replaced by a print of the final override list."""
    environ = {k: v for k, v in os.environ.items() if k not in {"TEACHER_PATH", "GRPO_TEACHER_RUN", "DRY_RUN"}}
    environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{environ.get('PATH', '')}"  # python3 of this env
    environ.update(env)
    script = (
        f'source "{COMMON}"\n'
        "launch_training() {\n"
        '    printf "%s\\n" "${METHOD_COMMON_ARGS[@]}" "${ALGO_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"\n'
        "}\n"
        "ALGO_ARGS=()\n"
        f"{terms}\n"
        "EXTRA_ARGS=()\n"
        f'launch_s_opd {data} {teacher} "$@"\n'
    )
    return subprocess.run(
        ["bash", "-c", script, "bash", *cli_args], env=environ, capture_output=True, text=True, cwd=ROOT, check=False
    )


def _effective(stdout: str) -> dict[str, str]:
    """The settings the trainer ends up with: as in the config merge, the last override wins."""
    return dict(line.partition("=")[::2] for line in stdout.splitlines())


def _grpo_run(tmp_path: Path, last: int = 202, weights: bool = True) -> Path:
    run = tmp_path / "grpo"
    actor = run / f"global_step_{last}" / "actor"
    actor.mkdir(parents=True)
    (run / "checkpoint_tracker.json").write_text(json.dumps({"last_global_step": last, "best_global_step": 150}))
    if weights:
        (actor / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
        save_file({"lm_head.weight": torch.zeros(2, 2)}, str(actor / "model.safetensors"))
    return run


def test_default_teacher_is_the_8b_instruct_model():
    result = _launch()
    assert result.returncode == 0, result.stderr
    settings = _effective(result.stdout)
    assert settings["worker.teacher.model.model_path"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert settings["algorithm.adv_estimator"] == "teacher_log_ratio"
    assert settings["data.rollout_batch_size"] == settings["worker.actor.global_batch_size"] == "192"
    assert "algorithm.visual_sensitivity_loss_coef" not in settings  # OPD only


def test_grpo_teacher_is_the_last_step_of_the_teacher_run(tmp_path):
    run = _grpo_run(tmp_path)
    result = _launch(teacher="grpo", GRPO_TEACHER_RUN=str(run))
    assert result.returncode == 0, result.stderr
    assert _effective(result.stdout)["worker.teacher.model.model_path"] == str(run / "global_step_202" / "actor")


def test_grpo_teacher_without_weights_fails_before_training(tmp_path):
    for run in (tmp_path / "missing", _grpo_run(tmp_path, weights=False)):
        result = _launch(teacher="grpo", GRPO_TEACHER_RUN=str(run))
        assert result.returncode != 0 and "worker.teacher" not in result.stdout
        assert "TEACHER_PATH" in result.stderr


def test_grpo_teacher_check_is_skipped_by_dry_run_and_explicit_teachers(tmp_path):
    missing = str(tmp_path / "missing")
    assert _launch(teacher="grpo", GRPO_TEACHER_RUN=missing, DRY_RUN="1").returncode == 0
    result = _launch(teacher="grpo", GRPO_TEACHER_RUN=missing, TEACHER_PATH="/teachers/a")
    assert result.returncode == 0 and _effective(result.stdout)["worker.teacher.model.model_path"] == "/teachers/a"
    result = _launch(teacher="grpo", GRPO_TEACHER_RUN=missing, cli_args=("worker.teacher.model.model_path=/t/b",))
    assert result.returncode == 0 and _effective(result.stdout)["worker.teacher.model.model_path"] == "/t/b"


@pytest.mark.parametrize("source", ["ema", "none"])
@pytest.mark.parametrize("where", ["cli", "script"])
def test_a_teacher_without_a_model_gets_no_external_path(tmp_path, source, where):
    """worker.teacher.source=ema|none (on the command line or in the script) neither looks for the GRPO run nor
    passes a teacher path, also under DRY_RUN and with TEACHER_PATH set."""
    terms = f"ALGO_ARGS=(worker.teacher.source={source})" if where == "script" else ""
    cli = (f"worker.teacher.source={source}",) if where == "cli" else ()
    for env in ({}, {"DRY_RUN": "1"}, {"TEACHER_PATH": "/teachers/a"}):
        result = _launch(terms=terms, teacher="grpo", cli_args=cli, GRPO_TEACHER_RUN=str(tmp_path / "missing"), **env)
        assert result.returncode == 0, result.stderr
        settings = _effective(result.stdout)
        assert settings["worker.teacher.source"] == source
        assert "worker.teacher.model.model_path" not in settings


def test_a_teacher_path_in_the_script_skips_the_default(tmp_path):
    result = _launch(
        terms="ALGO_ARGS=(worker.teacher.model.model_path=/teachers/s)",
        teacher="grpo",
        GRPO_TEACHER_RUN=str(tmp_path / "missing"),
    )
    assert result.returncode == 0, result.stderr
    settings = _effective(result.stdout)
    assert settings["worker.teacher.source"] == "model"
    assert settings["worker.teacher.model.model_path"] == "/teachers/s"


def test_an_unknown_default_teacher_fails():
    result = _launch(teacher="4b")
    assert result.returncode != 0 and "8b or grpo" in result.stderr


@pytest.mark.parametrize(
    "tpc, pa",
    [("0.02", "0.02"), ("0.005", "0"), ("0", "0.02")],
)
def test_s_opd_terms(tpc, pa):
    result = _launch(terms=f"s_opd_terms {tpc} {pa}", data="geo3k")
    assert result.returncode == 0, result.stderr
    settings = _effective(result.stdout)
    assert settings["algorithm.corrupt_image"] == "random_patch"
    assert json.loads(settings["algorithm.corrupt_image_kwargs"]) == {"black_prob": 0.6, "mask_before_resize": True}
    assert settings.get("algorithm.visual_sensitivity_loss_coef") == (tpc if tpc != "0" else None)
    assert settings.get("algorithm.visual_sensitivity_loss_weighting") == ("teacher" if tpc != "0" else None)
    assert settings.get("algorithm.visual_robustness_loss_coef") == (pa if pa != "0" else None)
    noise = settings.get("algorithm.incremental_image_kwargs")
    assert (json.loads(noise) if noise else None) == ({"std": 0.2, "noise_before_resize": True} if pa != "0" else None)
    assert settings["data.max_prompt_length"] == "1024" and settings["trainer.total_epochs"] == "20"
