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
"""How a checkpoint trained in this repository was prompted, from its run's ``experiment_config.json``.

The trainer writes the full configuration to ``<run>/experiment_config.json`` (the
``trainer.save_checkpoint_path``) and saves the processor with each checkpoint, so a checkpoint
``<run>/global_step_N/actor`` (or its ``huggingface/`` folder) carries its chat template and tokenizer,
and the run root records the rest of the prompt: the format prompt, the system prompt, the image size,
the ``<think>`` tokenization and the interaction mode. The evaluation uses these settings unless a flag
overrides them, so that a checkpoint is evaluated with the prompt it was trained with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checkpoints import step_number
from .paths import PROJECT_ROOT


RECORD_FILENAME = "experiment_config.json"
_NONE = {"", "none", "null"}


@dataclass(frozen=True)
class TrainingRecord:
    path: Path  # the run's experiment_config.json
    values: dict[str, Any]  # runner option -> training value (prompt files resolved, "none" for no file)
    chat_template: str | None = None  # data.override_chat_template, saved with the checkpoint's processor
    notes: list[str] = field(default_factory=list)  # settings of training the evaluation cannot reproduce


def find_training_record(model: str) -> Path | None:
    """``<run>/experiment_config.json`` of a checkpoint ``<run>/global_step_N[/actor[/huggingface]]``, or None
    for a model id or any other directory."""
    path = Path(model).expanduser()
    if not path.is_dir():
        return None
    path = path.resolve()
    if path.name == "huggingface":
        path = path.parent
    if path.name == "actor":
        path = path.parent
    if step_number(path) is None:
        return None
    record = path.parent / RECORD_FILENAME
    return record if record.is_file() else None


def _resolve_prompt_file(value: Any, key: str, record: Path) -> str:
    """A prompt file named in the record: as recorded when it exists, else the file of the same path under
    this repository's ``examples/`` (a run trained from another clone), else an error."""
    if value is None or str(value).strip().lower() in _NONE:
        return "none"
    recorded = Path(str(value)).expanduser()
    if recorded.is_file():
        return str(recorded)
    text = recorded.as_posix()
    index = text.rfind("examples/")
    if index >= 0:
        candidate = PROJECT_ROOT / text[index:]
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"{record} trained with {key}={value}, which does not exist; pass the file with "
        f"--{key.split('.')[-1].replace('_', '-')} (or 'none')"
    )


def load_training_record(model: str) -> TrainingRecord | None:
    record = find_training_record(model)
    if record is None:
        return None
    from verl.utils.plain_think import normalize_plain_think_tokens

    config = json.loads(record.read_text(encoding="utf-8"))
    data = config.get("data") or {}
    model_config = ((config.get("worker") or {}).get("actor") or {}).get("model") or {}
    rollout = (config.get("worker") or {}).get("rollout") or {}
    values: dict[str, Any] = {
        "format_prompt": _resolve_prompt_file(data.get("format_prompt"), "data.format_prompt", record),
        "system_prompt": _resolve_prompt_file(data.get("system_prompt"), "data.system_prompt", record),
        "plain_think_tokens": normalize_plain_think_tokens(model_config.get("plain_think_tokens", "auto")),
    }
    for key in ("min_pixels", "max_pixels"):
        if data.get(key) is not None:
            values[key] = int(data[key])
    for key in ("interaction_mode", "agent_prompt_style"):
        if rollout.get(key) is not None:
            values[key] = str(rollout[key])
    notes = []
    if data.get("system_prompt_key") and values.get("agent_prompt_style") != "official":
        # (DeepEyes' official prompts come from the data in training and are rebuilt by the agentic evaluation)
        notes.append(
            f"training read per-row system prompts from the column data.system_prompt_key="
            f"{data['system_prompt_key']} (rows without one use data.system_prompt); the benchmarks have no such "
            "column"
        )
    template = data.get("override_chat_template") or model_config.get("override_chat_template")
    return TrainingRecord(path=record, values=values, chat_template=str(template) if template else None, notes=notes)


def same_setting(key: str, first: Any, second: Any) -> bool:
    """Whether two values of an option give the same prompt (prompt files compare by content)."""
    if key in {"format_prompt", "system_prompt", "chat_template"}:

        def text(value: Any) -> str | None:
            if value is None or str(value).strip().lower() in _NONE:
                return None
            path = Path(str(value)).expanduser()
            return path.read_text(encoding="utf-8") if path.is_file() else str(value)

        return text(first) == text(second)
    if key in {"min_pixels", "max_pixels"}:
        return int(first) == int(second)
    return str(first).strip().lower() == str(second).strip().lower()
