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

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval import runner  # noqa: E402
from easyr1_eval.training_record import find_training_record, load_training_record  # noqa: E402


MATH = ROOT / "examples/format_prompt/math.jinja"
MATH_PERCEPTION = ROOT / "examples/format_prompt/math_perception.jinja"
NO_THINKING = ROOT / "examples/chat_template/qwen_no_thinking.jinja"


def _checkpoint(tmp_path: Path, data: dict, model: dict | None = None, rollout: dict | None = None) -> Path:
    """A run root with experiment_config.json and the actor folder of global_step_10."""
    run = tmp_path / "run"
    actor = run / "global_step_10" / "actor"
    (actor / "huggingface").mkdir(parents=True)
    config = {"data": data, "worker": {"actor": {"model": model or {}}, "rollout": rollout or {}}}
    (run / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
    return actor


def _parse(monkeypatch, model, *argv):
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py", "--model", str(model), *argv])
    return runner.parse_args()


def test_the_record_is_found_from_the_checkpoint_folders_only(tmp_path):
    actor = _checkpoint(tmp_path, {"format_prompt": None})
    record = tmp_path / "run" / "experiment_config.json"

    assert find_training_record(str(actor)) == record
    assert find_training_record(str(actor / "huggingface")) == record
    assert find_training_record(str(actor.parent)) == record
    assert find_training_record("Qwen/Qwen3-VL-2B-Instruct") is None
    assert find_training_record(str(tmp_path / "run")) is None  # a run root is not a checkpoint
    (tmp_path / "plain").mkdir()
    assert find_training_record(str(tmp_path / "plain")) is None


def test_the_record_resolves_prompt_files_and_normalizes_values(tmp_path):
    other_clone = "/somewhere/else/awesome/examples/format_prompt/math.jinja"
    actor = _checkpoint(
        tmp_path,
        {
            "format_prompt": other_clone,
            "system_prompt": None,
            "min_pixels": 262144,
            "max_pixels": 4194304,
            "override_chat_template": str(NO_THINKING),
            "system_prompt_key": "row_system_prompt",
        },
        model={"plain_think_tokens": False},
        rollout={"interaction_mode": "one_shot", "agent_prompt_style": "native"},
    )

    record = load_training_record(str(actor))

    assert record.values == {
        "format_prompt": str(MATH),  # the same file in this repository
        "system_prompt": "none",
        "plain_think_tokens": "false",
        "min_pixels": 262144,
        "max_pixels": 4194304,
        "interaction_mode": "one_shot",
        "agent_prompt_style": "native",
    }
    assert record.chat_template == str(NO_THINKING)
    assert "row_system_prompt" in record.notes[0]


def test_a_missing_prompt_file_fails_closed(tmp_path, monkeypatch):
    actor = _checkpoint(tmp_path, {"format_prompt": "/gone/prompts/custom.jinja"})

    with pytest.raises(FileNotFoundError, match="--format-prompt"):
        load_training_record(str(actor))
    with pytest.raises(SystemExit):
        _parse(monkeypatch, actor)
    # a flag replaces the missing file
    args = _parse(monkeypatch, actor, "--format-prompt", "none")
    assert args.format_prompt is None and args.prompt_sources["format_prompt"] == "flag"
    args = _parse(monkeypatch, actor, "--format-prompt", str(MATH))
    assert Path(args.format_prompt) == MATH


def test_the_vgs_reward_sets_its_answer_protocol(tmp_path, monkeypatch):
    actor = _checkpoint(tmp_path, {"format_prompt": None})
    config_path = tmp_path / "run" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["worker"]["reward"] = {"reward_function": "/elsewhere/examples/reward_function/vgs.py:compute_score"}
    config_path.write_text(json.dumps(config), encoding="utf-8")

    args = _parse(monkeypatch, actor, "--suite", "papo")
    assert args.answer_protocol == "vgs" and args.prompt_sources["answer_protocol"] == "checkpoint"
    assert _parse(monkeypatch, actor, "--answer-protocol", "default").answer_protocol == "default"

    config["worker"]["reward"] = {"reward_function": "/x/examples/reward_function/math.py:compute_score"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert _parse(monkeypatch, actor).answer_protocol == "default"
    assert _parse(monkeypatch, "Qwen/Qwen3-VL-2B-Instruct", "--suite", "vgs").answer_protocol == "vgs"


def test_the_training_record_replaces_the_suite_defaults(tmp_path, monkeypatch):
    # e.g. VA-OPD of the OPD comparison (comparison prompt) evaluated on the VA-OPD paper's benchmarks
    actor = _checkpoint(
        tmp_path,
        {"format_prompt": str(MATH_PERCEPTION), "system_prompt": None, "min_pixels": 200704, "max_pixels": 1003520},
    )

    args = _parse(monkeypatch, actor, "--suite", "va_opd")

    assert Path(args.format_prompt) == MATH_PERCEPTION  # the suite's default is math.jinja
    assert (args.min_pixels, args.max_pixels) == (200704, 1003520)
    assert args.prompt_sources["format_prompt"] == "checkpoint"
    assert args.suite_defaults_applied == {}
    assert args.training_record == str(tmp_path / "run" / "experiment_config.json")


def test_a_checkpoint_keeps_its_own_chat_template(tmp_path, monkeypatch):
    # trained with the stock template: the vcsd suite's template must not replace it
    actor = _checkpoint(tmp_path, {"format_prompt": None, "system_prompt": None}, model={"plain_think_tokens": "auto"})

    args = _parse(monkeypatch, actor, "--suite", "vcsd")

    assert args.chat_template is None and args.prompt_sources["chat_template"] == "checkpoint"
    assert args.plain_think_tokens == "auto"


def test_a_flag_wins_over_the_record_with_a_warning(tmp_path, monkeypatch, capsys):
    actor = _checkpoint(tmp_path, {"format_prompt": str(MATH_PERCEPTION), "override_chat_template": str(NO_THINKING)})

    args = _parse(monkeypatch, actor, "--format-prompt", str(MATH), "--chat-template", "none")

    assert Path(args.format_prompt) == MATH and args.prompt_sources["format_prompt"] == "flag"
    output = capsys.readouterr().out
    assert "--format-prompt" in output and "differs from the training setting" in output
    assert "--chat-template" in output

    capsys.readouterr()
    _parse(monkeypatch, actor, "--format-prompt", str(MATH_PERCEPTION))  # the same file: no warning
    assert "differs" not in capsys.readouterr().out


def test_suite_defaults_set_the_template_of_models_without_a_record(monkeypatch):
    args = _parse(monkeypatch, "Qwen/Qwen3-VL-2B-Instruct", "--suite", "vcsd")
    assert Path(args.chat_template) == NO_THINKING
    assert args.plain_think_tokens == "false"
    assert args.format_prompt is None and args.system_prompt is None
    assert args.training_record is None

    args = _parse(monkeypatch, "Qwen/Qwen3.5-4B", "--suite", "vision_opd")
    assert Path(args.chat_template) == NO_THINKING
    assert (args.min_pixels, args.max_pixels) == (65536, 16777216)

    args = _parse(monkeypatch, "Qwen/Qwen3.5-4B", "--suite", "vision_opd", "--chat-template", "none")
    assert args.chat_template is None

    args = _parse(monkeypatch, "Qwen/Qwen3-VL-2B-Instruct")
    assert args.plain_think_tokens == "auto" and args.agent_prompt_style == "native"
    assert args.prompt_sources == {}


def test_the_vgs_suite_does_not_read_other_checkpoints_with_the_vgs_reward(tmp_path, monkeypatch, capsys):
    # a PEPO-style run (<answer> tags, another reward) evaluated on the VGS paper's benchmarks
    actor = _checkpoint(tmp_path, {"format_prompt": str(ROOT / "examples/format_prompt/pepo.jinja")})
    config_path = tmp_path / "run" / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["worker"]["reward"] = {"reward_function": "/x/examples/reward_function/r1v.py:compute_score"}
    config_path.write_text(json.dumps(config), encoding="utf-8")

    args = _parse(monkeypatch, actor, "--suite", "vgs")

    assert args.answer_protocol == "default" and "answer_protocol" not in args.prompt_sources
    assert "--answer-protocol vgs" in capsys.readouterr().out
    assert Path(args.format_prompt) == ROOT / "examples/format_prompt/pepo.jinja"
    # its <answer> answers are read as before
    from easyr1_eval.scorers import boxed_row_answers

    row = {"target": "42", "responses": ["<think>Work</think><answer>42</answer>"], "eval_metadata": {}}
    assert boxed_row_answers(row, "mathvision")[0][1] == 1.0
