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
"""Resolve "what to evaluate" from a model id or an EasyR1 checkpoint path.

Accepted inputs (see ``resolve_eval_targets``):

- a Hugging Face model id (``Qwen/Qwen2.5-VL-3B-Instruct``);
- a merged Hugging Face directory (``config.json`` + weights);
- an actor directory with FSDP shards (``.../global_step_N/actor``), or with merged weights directly
  in it after ``scripts/finalize_run.py``;
- a step directory (``.../global_step_N``);
- a run checkpoint root containing ``global_step_*`` directories (latest step, or every
  step with ``all_steps=True``).

FSDP shards are merged with ``scripts/model_merger.py`` into ``<actor>/huggingface``
(the shards are kept).
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .paths import PROJECT_ROOT


STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")
SHARD_RE = re.compile(r"^model_world_size_\d+_rank_0\.pt$")
WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
MODEL_MERGER = PROJECT_ROOT / "scripts" / "model_merger.py"


@dataclass(frozen=True)
class EvalTarget:
    model: str  # what the backend loads: HF id or a merged HF directory
    run_name: str
    step: str | None = None  # "global_step_N" when known
    actor_dir: Path | None = None  # set when FSDP shards must be merged first

    @property
    def label(self) -> str:
        return f"{self.run_name}/{self.step}" if self.step else self.run_name

    @property
    def needs_merge(self) -> bool:
        return self.actor_dir is not None and not has_hf_weights(self.actor_dir / "huggingface")


def has_hf_weights(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return any(any(path.glob(pattern)) for pattern in WEIGHT_PATTERNS)


def has_fsdp_shards(path: Path) -> bool:
    return path.is_dir() and any(SHARD_RE.match(child.name) for child in path.iterdir())


def step_number(path: Path) -> int | None:
    match = STEP_DIR_RE.match(path.name)
    return int(match.group(1)) if match else None


def list_step_dirs(root: Path) -> list[Path]:
    steps = [child for child in root.iterdir() if child.is_dir() and step_number(child) is not None]
    return sorted(steps, key=lambda child: step_number(child) or 0)


def _actor_target(actor_dir: Path, run_name: str | None) -> EvalTarget:
    step_dir = actor_dir.parent
    step = step_dir.name if step_number(step_dir) is not None else None
    name = run_name or (step_dir.parent.name if step else actor_dir.name)
    if has_hf_weights(actor_dir):  # finalized by scripts/finalize_run.py
        return EvalTarget(model=str(actor_dir), run_name=name, step=step)
    hf_dir = actor_dir / "huggingface"
    if not has_hf_weights(hf_dir) and not has_fsdp_shards(actor_dir):
        raise FileNotFoundError(f"{actor_dir} has neither merged weights in huggingface/ nor FSDP shards")
    return EvalTarget(model=str(hf_dir), run_name=name, step=step, actor_dir=actor_dir)


def _step_target(step_dir: Path, run_name: str | None) -> EvalTarget:
    actor_dir = step_dir / "actor"
    if actor_dir.is_dir():
        return _actor_target(actor_dir, run_name)
    if has_hf_weights(step_dir / "huggingface"):
        return EvalTarget(
            model=str(step_dir / "huggingface"), run_name=run_name or step_dir.parent.name, step=step_dir.name
        )
    raise FileNotFoundError(f"{step_dir} has no actor/ checkpoint")


def resolve_eval_targets(value: str, *, all_steps: bool = False, run_name: str | None = None) -> list[EvalTarget]:
    path = Path(value).expanduser()
    if not path.exists():
        if value.startswith((".", "/", "~")) or value.count("/") > 1:
            raise FileNotFoundError(f"model path does not exist: {value}")
        # Hugging Face model id.
        return [EvalTarget(model=value, run_name=run_name or value.rstrip("/").split("/")[-1])]
    path = path.resolve()
    if has_hf_weights(path):
        if path.name == "actor" and step_number(path.parent) is not None:
            return [_actor_target(path, run_name)]
        if path.name == "huggingface" and path.parent.name == "actor" and step_number(path.parent.parent) is not None:
            return [
                EvalTarget(
                    model=str(path), run_name=run_name or path.parent.parent.parent.name, step=path.parent.parent.name
                )
            ]
        return [EvalTarget(model=str(path), run_name=run_name or path.name)]
    if path.name == "huggingface" and has_fsdp_shards(path.parent):
        return [_actor_target(path.parent, run_name)]
    if has_fsdp_shards(path) or (path.name == "actor" and (path / "huggingface").is_dir()):
        return [_actor_target(path, run_name)]
    if step_number(path) is not None:
        return [_step_target(path, run_name)]
    steps = list_step_dirs(path)
    if steps:
        selected = steps if all_steps else steps[-1:]
        return [_step_target(step, run_name or path.name) for step in selected]
    raise FileNotFoundError(
        f"{path} is not a Hugging Face model directory, an actor/step checkpoint, or a run root with global_step_* folders"
    )


def merge_actor_checkpoint(target: EvalTarget, *, python: str | None = None) -> None:
    """Merge FSDP shards into ``<actor>/huggingface`` with scripts/model_merger.py (shards are kept)."""
    if not target.needs_merge:
        return
    assert target.actor_dir is not None
    command = [python or sys.executable, str(MODEL_MERGER), "--local_dir", str(target.actor_dir)]
    print(f"[merge] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    if not has_hf_weights(target.actor_dir / "huggingface"):
        raise RuntimeError(f"model merger finished but {target.actor_dir / 'huggingface'} has no weights")
