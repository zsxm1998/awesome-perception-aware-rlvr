#!/usr/bin/env python3
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
"""Finalize training runs: keep the chosen steps as Hugging Face weights and free the disk space.

    python3 scripts/finalize_run.py checkpoints/<project>/<experiment> [more runs] [--keep last,best]
    python3 scripts/finalize_run.py checkpoints/<project>/<experiment>/global_step_N   # one step only
    (both take --dry-run and --yes)

A run directory is the trainer's ``save_checkpoint_path`` (``checkpoints/<project>/<experiment>`` by
default) and holds ``checkpoint_tracker.json`` and ``global_step_*/``. For each run:

1. the kept step(s) are finalized as below;
2. ``checkpoint_tracker.json`` is marked as finalized, so that training refuses to resume the run;
3. every other ``global_step_*`` directory is deleted. Logs and generation records are kept.

Finalizing a step:

1. ``scripts/model_merger.py`` merges the FSDP shards into ``actor/huggingface`` (skipped when merged
   weights already exist), and the merged weights are validated;
2. ``actor/*.pt`` (model shards, optimizer and extra states) and ``global_step_N/dataloader.pt`` are
   deleted;
3. the contents of ``actor/huggingface`` are moved into ``actor/``, so that ``actor/`` itself is a
   Hugging Face model directory.

A step passed directly (``global_step_N`` or ``global_step_N/actor``) is finalized alone; the other
steps and the tracker are left as they are (an intermediate step, or a run without a tracker).

Nothing changes before the plan is printed and confirmed (``--yes`` skips the question,
``--dry-run`` only prints the plan), and nothing is deleted before the merged weights are validated.
An interrupted run continues where it stopped when started again. Run it only after training has
finished. A 4B run shrinks from 34 GiB per saved step to about 9 GiB.

``--keep`` (run directories) is a comma-separated set, in any order; the default is ``last,best``:
  last  the final step: every method is compared after the same training budget, and the choice does
        not look at validation data.
  best  the step with the highest validation reward (``val/reward_score``) among the saved steps. The
        validation set is often also an evaluated benchmark (MMK12 test in the controlled comparison),
        so results on it are optimistically biased.
  all   every saved step, e.g. to study the training dynamics; the trainer only leaves the steps that
        ``trainer.save_limit`` allows (-1: all of them).
  N     global_step_N, e.g. ``--keep last,50,100``.
  both  the same as ``last,best``.
Keeping only one step takes ``--keep last`` or ``--keep best``. When the best step is unknown (validation
never ran) or gone (an earlier ``--keep last``), asking for it alone is an error; with other steps it is a
note.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from safetensors import safe_open


TRACKER = "checkpoint_tracker.json"  # verl.utils.checkpoint.CHECKPOINT_TRACKER
STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")
KEEP_WORDS = ("all", "last", "best")
DEFAULT_KEEP = "last,best"
RECENT_SECONDS = 30 * 60
MODEL_RANK_ZERO_RE = re.compile(r"model_world_size_\d+_rank_0\.pt")
MODEL_INDEX = "model.safetensors.index.json"
SINGLE_MODEL = "model.safetensors"
TOTAL_STEPS = 8
MISSING_FILE = "missing file:"


# ---------------------------------------------------------------- one step


class FinalizeError(RuntimeError):
    """A safe-to-report checkpoint finalization error."""


@dataclass(frozen=True)
class CheckpointValidation:
    ok: bool
    detail: str
    weight_files: tuple[Path, ...] = ()


def log(message: str) -> None:
    print(f"[finalize] {message}", flush=True)


def step_log(step: int, message: str) -> None:
    log(f"[step {step}/{TOTAL_STEPS}] {message}")


def existing_path(path: Path) -> bool:
    """Return True for normal paths and broken symlinks."""
    return os.path.lexists(path)


def locate_unique(relative_name: str, roots: Sequence[Path]) -> Path:
    relative_path = PurePosixPath(relative_name)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in ("", ".", "..") for part in relative_path.parts)
    ):
        raise FinalizeError(f"unsafe path in the weight index: {relative_name!r}")

    matches = [
        root.joinpath(*relative_path.parts) for root in roots if existing_path(root.joinpath(*relative_path.parts))
    ]
    if not matches:
        raise FinalizeError(f"{MISSING_FILE} {relative_name}")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise FinalizeError(f"the same file exists in several places, refusing to pick one: {locations}")
    return matches[0]


def locate_optional(relative_name: str, roots: Sequence[Path]) -> Path | None:
    try:
        return locate_unique(relative_name, roots)
    except FinalizeError as error:
        if str(error).startswith(MISSING_FILE):
            return None
        raise


def validate_regular_nonempty_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FinalizeError(f"not a regular file: {path}")
    if path.stat().st_size <= 0:
        raise FinalizeError(f"empty file: {path}")


def validate_safetensors(path: Path, expected_keys: Iterable[str] | None = None) -> None:
    validate_regular_nonempty_file(path)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
    except Exception as error:
        raise FinalizeError(f"cannot read the safetensors file {path}: {error}") from error

    if not actual_keys:
        raise FinalizeError(f"the safetensors file has no tensors: {path}")
    if expected_keys is not None:
        missing_keys = set(expected_keys) - actual_keys
        if missing_keys:
            examples = ", ".join(sorted(missing_keys)[:3])
            raise FinalizeError(
                f"the weight index does not match the shard: {path} lacks {len(missing_keys)} tensors "
                f"(e.g. {examples})"
            )


def inspect_checkpoint(roots: Sequence[Path]) -> CheckpointValidation:
    """Validate a checkpoint whose top-level files may be split across roots."""
    roots = tuple(root for root in roots if root.is_dir() and not root.is_symlink())
    if not roots:
        return CheckpointValidation(False, "the checkpoint directory does not exist")

    try:
        config_path = locate_unique("config.json", roots)
        validate_regular_nonempty_file(config_path)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FinalizeError(f"cannot parse config.json: {config_path}: {error}") from error
        if not isinstance(config, dict):
            raise FinalizeError(f"config.json is not a JSON object: {config_path}")

        index_path = locate_optional(MODEL_INDEX, roots)
        single_model_path = locate_optional(SINGLE_MODEL, roots)
        if index_path is not None and single_model_path is not None:
            raise FinalizeError(f"both {MODEL_INDEX} and {SINGLE_MODEL} exist, the weight layout is ambiguous")

        if index_path is None:
            if single_model_path is None:
                raise FinalizeError(f"neither {SINGLE_MODEL} nor {MODEL_INDEX} found")
            validate_safetensors(single_model_path)
            return CheckpointValidation(True, f"valid single-file weights: {single_model_path}", (single_model_path,))

        validate_regular_nonempty_file(index_path)
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FinalizeError(f"cannot parse the weight index: {index_path}: {error}") from error

        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise FinalizeError(f"the weight index has no non-empty weight_map: {index_path}")

        keys_by_shard: dict[str, list[str]] = defaultdict(list)
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise FinalizeError(f"malformed weight_map in the weight index: {index_path}")
            keys_by_shard[shard_name].append(tensor_name)

        shard_paths = []
        for shard_name, expected_keys in sorted(keys_by_shard.items()):
            shard_path = locate_unique(shard_name, roots)
            validate_safetensors(shard_path, expected_keys)
            shard_paths.append(shard_path)

        return CheckpointValidation(
            True,
            f"valid sharded weights: {len(shard_paths)} shards, index {index_path}",
            tuple(shard_paths),
        )
    except FinalizeError as error:
        return CheckpointValidation(False, str(error))


def actor_pt_files(actor_dir: Path) -> list[Path]:
    files = []
    for path in actor_dir.iterdir():
        if path.name.endswith(".pt"):
            if path.is_dir() and not path.is_symlink():
                raise FinalizeError(f"refusing to delete a directory named *.pt recursively: {path}")
            files.append(path)
    return sorted(files)


def raw_model_rank_zero_exists(actor_dir: Path) -> bool:
    return any(MODEL_RANK_ZERO_RE.fullmatch(path.name) for path in actor_dir.iterdir() if path.is_file())


def ensure_no_move_collisions(actor_dir: Path, hf_dir: Path) -> None:
    if not hf_dir.exists():
        return
    if hf_dir.is_symlink() or not hf_dir.is_dir():
        raise FinalizeError(f"huggingface is not a regular directory: {hf_dir}")

    collisions = [actor_dir / source.name for source in hf_dir.iterdir() if existing_path(actor_dir / source.name)]
    if collisions:
        examples = ", ".join(str(path) for path in collisions[:3])
        raise FinalizeError(f"files in huggingface/ collide with paths in actor/ (nothing was deleted): {examples}")


def delete_cleanup_files(actor_dir: Path) -> None:
    pt_files = actor_pt_files(actor_dir)
    dataloader_path = actor_dir.parent / "dataloader.pt"
    dataloader_exists = existing_path(dataloader_path)

    if pt_files:
        log(f"{len(pt_files)} .pt files to delete in actor/")
    else:
        log("no .pt files in actor/")

    for path in pt_files:
        try:
            path.unlink()
            log(f"deleted {path}")
        except FileNotFoundError:
            log(f"already gone: {path}")
        except OSError as error:
            raise FinalizeError(f"cannot delete {path}: {error}") from error

    if dataloader_exists:
        if dataloader_path.is_dir() and not dataloader_path.is_symlink():
            raise FinalizeError(f"dataloader.pt is a directory, refusing to delete it recursively: {dataloader_path}")
        try:
            dataloader_path.unlink()
            log(f"deleted {dataloader_path}")
        except FileNotFoundError:
            log(f"already gone: {dataloader_path}")
        except OSError as error:
            raise FinalizeError(f"cannot delete {dataloader_path}: {error}") from error
    else:
        log("no dataloader.pt next to actor/")

    remaining = actor_pt_files(actor_dir)
    if remaining or existing_path(dataloader_path):
        raise FinalizeError("cleanup check failed: .pt files that should be deleted remain")
    log("cleanup check passed: the training-state .pt files are deleted")


def move_huggingface_contents(actor_dir: Path, hf_dir: Path) -> None:
    if not hf_dir.exists():
        log("huggingface/ does not exist: already moved")
        return
    if hf_dir.is_symlink() or not hf_dir.is_dir():
        raise FinalizeError(f"huggingface is not a regular directory: {hf_dir}")

    # Put the index last so a directly inspected actor directory never advertises
    # a sharded checkpoint before all of its shards have arrived.
    sources = sorted(
        hf_dir.iterdir(),
        key=lambda path: (path.name == MODEL_INDEX, path.name),
    )
    if not sources:
        log("huggingface/ is empty: nothing to move")
        return

    log(f"moving {len(sources)} items into {actor_dir}")
    for source in sources:
        destination = actor_dir / source.name
        if existing_path(destination):
            raise FinalizeError(f"refusing to overwrite {destination} (source: {source})")
        try:
            source.rename(destination)
            log(f"moved {source} -> {destination}")
        except OSError as error:
            raise FinalizeError(f"cannot move {source} -> {destination}: {error}") from error
    log(f"moved {len(sources)} items")


def resolve_actor_dir(path: Path) -> Path:
    try:
        actor_dir = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalizeError(
            f"the checkpoint directory does not exist or is not accessible: {path}: {error}"
        ) from error
    if not actor_dir.is_dir():
        raise FinalizeError(f"not a directory: {actor_dir}")
    if actor_dir.name != "actor":
        raise FinalizeError(
            f"only directories named actor are accepted, to avoid deleting wrong files; got {actor_dir}"
        )
    return actor_dir


def run_merger(actor_dir: Path) -> None:
    merger = Path(__file__).resolve().with_name("model_merger.py")
    if not merger.is_file():
        raise FinalizeError(f"model merger not found: {merger}")

    log(f"merging {actor_dir} with {merger}")
    try:
        subprocess.run(
            [sys.executable, str(merger), "--local_dir", str(actor_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise FinalizeError(
            f"model_merger.py failed (exit code {error.returncode}); nothing was deleted or moved, fix the problem "
            "and run this script again"
        ) from error
    log("model_merger.py finished")


def finalize_actor_checkpoint(checkpoint_dir: Path) -> Path:
    """Merge, clean up and flatten one actor directory (the eight steps); return it. Raises FinalizeError."""
    step_log(1, f"checking {checkpoint_dir}")
    actor_dir = resolve_actor_dir(checkpoint_dir)
    hf_dir = actor_dir / "huggingface"
    log(f"actor directory: {actor_dir}")

    step_log(2, "inspecting the merged weights")
    actor_checkpoint = inspect_checkpoint((actor_dir,))
    hf_checkpoint = inspect_checkpoint((hf_dir,))
    combined_checkpoint = inspect_checkpoint((actor_dir, hf_dir))
    log(f"in actor/: {'complete' if actor_checkpoint.ok else 'incomplete'}; {actor_checkpoint.detail}")
    log(f"in actor/huggingface/: {'complete' if hf_checkpoint.ok else 'incomplete'}; {hf_checkpoint.detail}")
    log(
        f"across both (interrupted move): {'complete' if combined_checkpoint.ok else 'incomplete'}; "
        f"{combined_checkpoint.detail}"
    )

    if actor_checkpoint.ok:
        step_log(3, "actor/ already holds merged weights, skipping model_merger.py")
    elif hf_checkpoint.ok:
        step_log(3, "actor/huggingface/ already holds merged weights, skipping model_merger.py")
    elif combined_checkpoint.ok:
        step_log(3, "the merged weights are complete across actor/ and huggingface/, skipping model_merger.py")
    else:
        if not raw_model_rank_zero_exists(actor_dir):
            raise FinalizeError(
                "found neither complete merged weights nor the rank-0 model shard that model_merger.py needs; "
                f"validation error: {combined_checkpoint.detail}"
            )
        step_log(3, "no merged weights yet, running model_merger.py")
        run_merger(actor_dir)
        hf_checkpoint = inspect_checkpoint((hf_dir,))
        if not hf_checkpoint.ok:
            raise FinalizeError(
                "model_merger.py finished but its output is not valid; nothing was deleted or moved. "
                f"Reason: {hf_checkpoint.detail}"
            )
        log(f"the merged weights are valid: {hf_checkpoint.detail}")

    # Check every destination before deleting training-state files. A collision
    # is therefore non-destructive and can be resolved manually.
    step_log(4, "checks before deleting anything")
    ensure_no_move_collisions(actor_dir, hf_dir)
    log("no file in huggingface/ would overwrite a file in actor/")

    # Revalidate immediately before the destructive step, including the
    # partially-moved layout used when resuming an interrupted run.
    safe_checkpoint = inspect_checkpoint((actor_dir, hf_dir))
    if not safe_checkpoint.ok:
        raise FinalizeError(f"the merged weights are not valid, nothing was deleted. Reason: {safe_checkpoint.detail}")
    log(f"the merged weights are revalidated: {safe_checkpoint.detail}")

    step_log(5, "deleting actor/*.pt and dataloader.pt")
    delete_cleanup_files(actor_dir)

    step_log(6, "moving the contents of huggingface/ into actor/")
    move_huggingface_contents(actor_dir, hf_dir)

    step_log(7, "validating the Hugging Face checkpoint in actor/")
    final_checkpoint = inspect_checkpoint((actor_dir,))
    if not final_checkpoint.ok:
        raise FinalizeError(
            "actor/ is not a valid checkpoint after the move; huggingface/ is kept so that you can retry. "
            f"Reason: {final_checkpoint.detail}"
        )
    log(f"the final checkpoint is valid: {final_checkpoint.detail}")

    step_log(8, "removing the empty huggingface/ directory")
    if hf_dir.exists():
        try:
            hf_dir.rmdir()
            log(f"removed {hf_dir}")
        except OSError as error:
            raise FinalizeError(
                f"the merged weights are safely in actor/, but huggingface/ cannot be removed: {error}"
            ) from error
    else:
        log("huggingface/ is already gone")

    log(f"[done] {actor_dir} is a merged Hugging Face checkpoint")
    return actor_dir


# ---------------------------------------------------------------- runs and steps


class PlanError(RuntimeError):
    """The target cannot be finalized as asked; nothing has been changed."""


@dataclass(frozen=True)
class KeepSpec:
    """The steps ``--keep`` asks for: ``words`` from KEEP_WORDS and explicit step numbers."""

    words: frozenset[str]
    steps: frozenset[int]

    @property
    def text(self) -> str:
        return ",".join(
            [word for word in KEEP_WORDS if word in self.words] + [str(step) for step in sorted(self.steps)]
        )


def parse_keep(text: str) -> KeepSpec:
    """``last,best`` / ``best,last`` / ``both`` / ``all`` / ``last,50,100`` -> KeepSpec (order does not matter)."""
    words: set[str] = set()
    steps: set[int] = set()
    for item in text.split(","):
        item = item.strip().lower()
        if item == "both":
            words |= {"last", "best"}
        elif item in KEEP_WORDS:
            words.add(item)
        elif item.isdigit():
            steps.add(int(item))
        else:
            raise argparse.ArgumentTypeError(
                f"--keep takes a comma-separated list of last, best, all, both and step numbers, got {text!r}"
            )
    return KeepSpec(frozenset(words), frozenset(steps))


@dataclass
class Plan:
    target: Path  # a run directory, or the step directory that was passed
    run_dir: Path
    keep: dict[int, list[str]]  # step -> its reasons: "last", "best", "saved" (--keep all), "given"
    delete: list[Path] = field(default_factory=list)
    tracker: dict | None = None  # run directories only: marked as finalized
    keep_policy: str | None = None
    delete_bytes: int = 0
    state_bytes: dict[int, int] = field(default_factory=dict)  # *.pt and dataloader.pt of each kept step
    notes: list[str] = field(default_factory=list)


def gigabytes(num_bytes: int) -> str:
    return f"{num_bytes / 2**30:.1f} GiB"


def tree_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def step_dirs(run_dir: Path) -> dict[int, Path]:
    steps = {}
    for child in run_dir.iterdir():
        match = STEP_DIR_RE.match(child.name)
        if match is None:
            continue
        if child.is_symlink() or not child.is_dir():
            raise PlanError(f"{child} is not a regular directory")
        steps[int(match.group(1))] = child
    return dict(sorted(steps.items()))


def training_state_bytes(step_dir: Path) -> int:
    paths = [step_dir / "dataloader.pt", *(step_dir / "actor").glob("*.pt")]
    return sum(path.lstat().st_size for path in paths if path.is_file() and not path.is_symlink())


def recent_activity_note(run_dir: Path) -> list[str]:
    files = [child for child in run_dir.iterdir() if child.is_file() and not child.is_symlink()]
    if not files:
        return []
    newest = max(files, key=lambda child: child.stat().st_mtime)
    age = time.time() - newest.stat().st_mtime
    if age >= RECENT_SECONDS:
        return []
    return [f"{newest.name} was written {int(age // 60)} min ago: make sure that training has finished"]


def plan_step(step_dir: Path, keep: KeepSpec | None) -> Plan:
    """A single step (``global_step_N`` or ``global_step_N/actor``): other steps and the tracker are left alone."""
    if keep is not None:
        raise PlanError(f"--keep applies to run directories, not to the single step {step_dir}")
    if not (step_dir / "actor").is_dir():
        raise PlanError(f"{step_dir} has no actor/ directory")
    step = int(STEP_DIR_RE.match(step_dir.name).group(1))
    plan = Plan(target=step_dir, run_dir=step_dir.parent, keep={step: ["given"]})
    plan.state_bytes = {step: training_state_bytes(step_dir)}
    plan.notes = ["other steps and checkpoint_tracker.json are left as they are", *recent_activity_note(plan.run_dir)]
    return plan


def plan_run(run_dir: Path, keep: KeepSpec | None) -> Plan:
    """A run directory: keep the asked steps (default: the last and the best), delete the others, mark the tracker."""
    keep = keep or parse_keep(DEFAULT_KEEP)
    tracker_path = run_dir / TRACKER
    steps = step_dirs(run_dir)
    if not tracker_path.is_file():
        if steps:
            raise PlanError(
                f"{run_dir} has global_step_* directories but no {TRACKER}; pass a single step "
                f"({run_dir}/global_step_N) to finalize only that step"
            )
        if any((child / TRACKER).is_file() for child in run_dir.iterdir() if child.is_dir()):
            raise PlanError(f"{run_dir} holds several runs; pass the runs themselves, e.g. {run_dir}/*")
        return Plan(target=run_dir, run_dir=run_dir, keep={}, notes=["no checkpoints: nothing to do"])

    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    last = tracker.get("last_global_step")
    best = tracker.get("best_global_step")
    if not isinstance(last, int):
        raise PlanError(f"{tracker_path} has no last_global_step")

    newer = [path.name for step, path in steps.items() if step > last]
    if newer:
        raise PlanError(
            f"{', '.join(newer)} in {run_dir} is newer than the last complete checkpoint (global_step_{last}): "
            "training is still running, or it stopped while saving. If it is not running, delete the newer "
            "directory and run this script again."
        )

    wanted: dict[int, list[str]] = {}
    notes = []
    if "all" in keep.words:
        for step in steps:
            wanted.setdefault(step, []).append("saved")
    if "last" in keep.words:
        if last not in steps:
            raise PlanError(f"global_step_{last} (last step) does not exist in {run_dir}")
        wanted.setdefault(last, []).append("last")
    if "best" in keep.words:
        others = bool(wanted) or bool(keep.steps)
        if not isinstance(best, int):
            problem = f"{tracker_path} records no best step (validation never ran)"
        elif best not in steps:
            earlier = (tracker.get("finalized") or {}).get("keep")
            problem = f"the best step global_step_{best} no longer exists" + (
                f" (finalized earlier with --keep {earlier})" if earlier else ""
            )
        else:
            problem = None
            wanted.setdefault(best, []).append("best")
        if problem and not others:
            raise PlanError(f"{problem}; keep another step (e.g. --keep last)")
        if problem:
            notes.append(f"{problem}: not kept")
    for step in sorted(keep.steps):
        if step not in steps:
            raise PlanError(f"global_step_{step} does not exist in {run_dir}")
        wanted.setdefault(step, []).append("given")
    for step, reasons in wanted.items():
        if not (steps[step] / "actor").is_dir():
            raise PlanError(f"{steps[step]} has no actor/ directory")
    earlier_steps = (tracker.get("finalized") or {}).get("steps") or []
    dropped = [step for step in earlier_steps if step in steps and step not in wanted]
    if dropped:
        notes.append(
            "deletes step(s) kept by the earlier finalization: " + ", ".join(f"global_step_{step}" for step in dropped)
        )

    plan = Plan(
        target=run_dir,
        run_dir=run_dir,
        keep=dict(sorted(wanted.items())),
        delete=[path for step, path in steps.items() if step not in wanted],
        tracker=tracker,
        keep_policy=keep.text,
        notes=notes,
    )
    plan.delete_bytes = sum(tree_bytes(path) for path in plan.delete)
    plan.state_bytes = {step: training_state_bytes(steps[step]) for step in plan.keep}
    plan.notes += recent_activity_note(run_dir)
    return plan


def make_plan(path: Path, keep: KeepSpec | str | None) -> Plan:
    if isinstance(keep, str):
        keep = parse_keep(keep)
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise PlanError(f"{path} is not a directory")
    if path.name == "actor" and STEP_DIR_RE.match(path.parent.name):
        return plan_step(path.parent, keep)
    if STEP_DIR_RE.match(path.name):
        return plan_step(path, keep)
    return plan_run(path, keep)


def describe(plan: Plan) -> str:
    lines = [str(plan.target)]
    tracker = plan.tracker
    if tracker is not None:
        best = tracker.get("best_global_step")
        best_text = (
            f"best step {best} (validation reward {tracker.get('best_val_reward_score')})"
            if isinstance(best, int)
            else "no best step"
        )
        lines.append(f"  tracker: last step {tracker['last_global_step']}, {best_text}")
        if tracker.get("finalized"):
            lines.append(f"  already finalized: {tracker['finalized']}")
    for step, reasons in plan.keep.items():
        state_bytes = plan.state_bytes.get(step, 0)
        state = f", delete its training state ({gigabytes(state_bytes)})" if state_bytes else ""
        label = "" if reasons == ["given"] else f" ({'+'.join(reasons)})"
        lines.append(f"  keep:    global_step_{step}{label} as Hugging Face weights in actor/{state}")
    for path in plan.delete:
        lines.append(f"  delete:  {path.name}")
    if plan.delete:
        lines.append(f"           ({gigabytes(plan.delete_bytes)} in {len(plan.delete)} step directories)")
    for note in plan.notes:
        lines.append(f"  note:    {note}")
    return "\n".join(lines)


def write_tracker(run_dir: Path, tracker: dict) -> None:
    path = run_dir / TRACKER
    tmp = path.with_name(f"{TRACKER}.tmp")
    tmp.write_text(json.dumps(tracker, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def execute(plan: Plan) -> None:
    for step in plan.keep:
        finalize_actor_checkpoint(plan.run_dir / f"global_step_{step}" / "actor")
    if plan.tracker is None:
        return
    # Mark the run before deleting the other steps: from here on, it cannot be resumed.
    tracker = dict(plan.tracker)
    tracker["finalized"] = {"keep": plan.keep_policy, "steps": list(plan.keep)}
    write_tracker(plan.run_dir, tracker)
    for path in plan.delete:
        if path.is_symlink() or not STEP_DIR_RE.match(path.name) or path.parent != plan.run_dir:
            raise FinalizeError(f"refusing to delete {path}")
        shutil.rmtree(path)
        log(f"deleted {path}")


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the last and the best step (or the steps of --keep) of finished runs as Hugging Face weights in "
            "global_step_N/actor and delete the optimizer states and the other steps; or finalize single steps."
        ),
        epilog="See the docstring of this script for the choices of --keep.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="run directories (checkpoints/<project>/<experiment>), or single steps (.../global_step_N[/actor])",
    )
    parser.add_argument(
        "--keep",
        type=parse_keep,
        default=None,
        metavar="last,best",
        help=(
            "for run directories, a comma-separated set (any order) of: last (the final step), best (the highest "
            "validation reward), all (every saved step), step numbers; default last,best; both = last,best"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and change nothing")
    parser.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plans, failed = [], False
    for path in args.paths:
        try:
            plans.append(make_plan(path, args.keep))
        except (PlanError, OSError, json.JSONDecodeError) as error:
            print(f"[finalize] [error] {error}", file=sys.stderr, flush=True)
            failed = True
    for plan in plans:
        print(describe(plan), flush=True)
    if failed:
        print("[finalize] nothing was changed; fix the errors above, or leave those paths out", file=sys.stderr)
        return 1
    plans = [plan for plan in plans if plan.keep]
    if not plans:
        return 0
    total = sum(plan.delete_bytes + sum(plan.state_bytes.values()) for plan in plans)
    print(
        f"\ndeletes {gigabytes(total)} and writes the merged weights of every kept step (about the size of the "
        "model); finalized steps can no longer be resumed",
        flush=True,
    )
    if args.dry_run:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("[finalize] pass --yes to proceed without a terminal", file=sys.stderr)
            return 1
        if not confirm("Proceed?"):
            print("[finalize] nothing was changed")
            return 1
    for plan in plans:
        try:
            execute(plan)
        except (FinalizeError, OSError) as error:
            print(f"[finalize] [failed] {plan.target}: {error}", file=sys.stderr, flush=True)
            return 1
        log(f"[done] {plan.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
