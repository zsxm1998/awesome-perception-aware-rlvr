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
"""One-click evaluation: ``bash scripts/eval.sh <model_or_ckpt> [options] [runner options]``.

<model_or_ckpt> is a Hugging Face id, a merged HF directory, ``.../global_step_N/actor``,
``.../global_step_N`` or a run checkpoint root (latest step; ``--all-steps`` for every
step). FSDP shards are merged into ``<actor>/huggingface`` first. Results go to
``<results-root>/<run_name>/<step>`` (or ``<results-root>/<run_name>`` for plain models).

Wrapper options:
  --all-steps            evaluate every global_step_* of a run root (default: latest only)
  --run-name NAME        name used for the results folder and the summary rows
  --results-root DIR     default eval/results
  --no-merge             fail instead of merging FSDP shards
Everything else (e.g. --suite papo, --benchmarks geo3k,pope, --limit 32, --gpus 0,1)
is passed to eval/run_all_benchmarks.py unchanged; see ``--help-runner``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .checkpoints import merge_actor_checkpoint, resolve_eval_targets
from .paths import DEFAULT_RESULTS_DIR, EVAL_ROOT


RUNNER = EVAL_ROOT / "run_all_benchmarks.py"
VALUE_OPTIONS = {"--run-name", "--results-root"}
FLAG_OPTIONS = {"--all-steps", "--no-merge"}


def split_args(argv: list[str]) -> tuple[str, dict[str, str | bool], list[str]]:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        raise SystemExit(0)
    if argv[0] == "--help-runner":
        subprocess.run([sys.executable, str(RUNNER), "--help"], check=False)
        raise SystemExit(0)
    model = argv[0]
    if model.startswith("-"):
        raise SystemExit("usage: bash scripts/eval.sh <model_or_ckpt> [options]; the model must come first")
    own: dict[str, str | bool] = {}
    passthrough: list[str] = []
    index = 1
    while index < len(argv):
        item = argv[index]
        name, _, inline = item.partition("=")
        if name in FLAG_OPTIONS:
            own[name] = True
        elif name in VALUE_OPTIONS:
            if inline:
                own[name] = inline
            else:
                if index + 1 >= len(argv):
                    raise SystemExit(f"{name} needs a value")
                own[name] = argv[index + 1]
                index += 1
        else:
            passthrough.append(item)
        index += 1
    return model, own, passthrough


def _has_option(args: list[str], name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in args)


def main(argv: list[str] | None = None) -> int:
    model, own, passthrough = split_args(list(sys.argv[1:] if argv is None else argv))
    targets = resolve_eval_targets(
        model,
        all_steps=bool(own.get("--all-steps")),
        run_name=str(own["--run-name"]) if own.get("--run-name") else None,
    )
    results_root = Path(str(own.get("--results-root") or DEFAULT_RESULTS_DIR)).expanduser().resolve()
    user_output_dir = _has_option(passthrough, "--output-dir")
    if user_output_dir and len(targets) > 1:
        raise SystemExit("--output-dir cannot be combined with several checkpoint steps; use --results-root")
    dry_run = _has_option(passthrough, "--dry-run")

    print(f"[eval] {len(targets)} target(s):", flush=True)
    for target in targets:
        print(
            f"  - {target.label}: {target.model}" + ("  (needs FSDP merge)" if target.needs_merge else ""), flush=True
        )

    failed: list[str] = []
    for target in targets:
        if target.needs_merge and not dry_run:
            if own.get("--no-merge"):
                raise SystemExit(f"{target.actor_dir} is not merged and --no-merge was given")
            merge_actor_checkpoint(target)
        command = [sys.executable, str(RUNNER), "--model", target.model]
        if not user_output_dir:
            output_dir = (
                results_root / target.run_name / target.step if target.step else results_root / target.run_name
            )
            command += ["--output-dir", str(output_dir)]
        if not _has_option(passthrough, "--run-name"):
            command += ["--run-name", target.label]
        command += passthrough
        print(f"[eval] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failed.append(f"{target.label} (exit code {completed.returncode})")
    if failed:
        print(f"[eval] failed: {', '.join(failed)}", file=sys.stderr, flush=True)
        return 1
    return 0
