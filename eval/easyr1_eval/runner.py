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
"""One-click benchmark runner: inference on a GPU worker pool, then serial scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .backends import build_backend
from .json_utils import read_json, write_json, write_jsonl
from .loaders import apply_box_format_to_prompts, load_samples, missing_data_files, shard_samples
from .paths import (
    DEFAULT_CONFIG,
    DEFAULT_FORMAT_PROMPT,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SUITES,
    PROJECT_ROOT,
    default_data_root,
)
from .perturbation_summary import write_perturbation_aggregate_summary
from .perturbations import add_perturbation_args, expand_eval_runs, perturbation_metadata
from .prompting import apply_prompt_config, prompt_config_from_args
from .registry import load_benchmark_specs, select_benchmarks
from .schemas import (
    AGENT_OUTPUT_CONTRACT_NATIVE,
    SKIPPED_STATUS,
    BenchmarkSpec,
    EvalSample,
    GenerationConfig,
    GenerationOutput,
)
from .scorers import JUDGE_PROVIDERS, SCORER_VERSION, resolve_judge_max_tokens
from .state import fingerprint, is_complete, mark_complete, mark_failed
from .suites import SUITE_DEFAULT_KEYS, load_suites, merged_suite_defaults, suite_benchmarks
from .summary import update_global_summary_csv, write_summary_csv


DEFAULT_MIN_PIXELS = 200704
DEFAULT_MAX_PIXELS = 1003520
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_BATCH_IMAGES = 64
AGENTIC_RUNNER_VERSION = 6
DEFAULT_GPU_MEMORY_UTILIZATION = 0.7
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GLOBAL_SUMMARY = DEFAULT_RESULTS_DIR / "summary.csv"
DEFAULT_INTERACTION_MODE = "one_shot"
DEFAULT_AGENT_PROFILE = "deepeyes"


class BenchmarkFailures(dict):
    """benchmark key -> error message for benchmarks that could not be evaluated."""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.output_dir = Path(args.output_dir).resolve()

    specs = selected_specs(args)
    print(f"[info] box format: {args.box_format} ({args.box_format_reason})", flush=True)
    if getattr(args, "interaction_mode", "one_shot") == "agentic":
        print(f"[info] agent system prompt: {args.system_prompt}", flush=True)
    if getattr(args, "chat_template", None):
        print(f"[info] chat template: {args.chat_template}", flush=True)
    if getattr(args, "plain_think_tokens", "auto") != "auto":
        print(f"[info] plain_think_tokens: {args.plain_think_tokens}", flush=True)
    if args.dry_run:
        print_dry_run(specs, args)
        return
    if not (args.score_only or args.summary_only):
        specs = check_benchmark_data(specs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = expand_eval_runs(args)
    failures = BenchmarkFailures()
    for run_args in runs:
        run_args.output_dir = Path(run_args.output_dir).resolve()
        run_args.output_dir.mkdir(parents=True, exist_ok=True)
        run_failures = run_all(specs, run_args)
        failures.update(
            {
                f"{run_args.output_dir.name}/{key}" if len(runs) > 1 else key: error
                for key, error in (run_failures or {}).items()
            }
        )
    aggregate_path = write_perturbation_aggregate_summary([Path(run.output_dir) for run in runs], args.output_dir)
    if aggregate_path is not None:
        print(f"[done] perturbation aggregate: {aggregate_path}", flush=True)
    if failures:
        print(f"[error] {len(failures)} benchmark(s) failed:", file=sys.stderr, flush=True)
        for key, error in failures.items():
            print(f"  - {key}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)


def selected_specs(args: argparse.Namespace) -> list[BenchmarkSpec]:
    """Benchmarks of --suite (union with --benchmarks) minus --skip-benchmarks."""
    specs = load_benchmark_specs(Path(args.config))
    include: list[str] = []
    suite_names = _split_csv(getattr(args, "suite", None)) or []
    if suite_names:
        include.extend(suite_benchmarks(load_suites(Path(args.suites_config), specs), suite_names))
    include.extend(_split_csv(args.benchmarks) or [])
    return select_benchmarks(
        specs,
        include=list(dict.fromkeys(include)) or None,
        skip=_split_csv(args.skip_benchmarks),
    )


def check_benchmark_data(specs: list[BenchmarkSpec], args: argparse.Namespace) -> list[BenchmarkSpec]:
    """Fail fast (before any GPU work) when a selected benchmark has not been prepared.

    Judge-only benchmarks are not checked when no judge is configured (they are skipped anyway).
    """
    from .scorers import judge_config_from_args

    judge_available = judge_config_from_args(args) is not None
    checked = [spec for spec in specs if judge_available or not spec.requires_judge]
    missing = {spec.key: missing_data_files(spec, args.data_root) for spec in checked}
    missing = {key: paths for key, paths in missing.items() if paths}
    if not missing:
        return specs
    lines = [f"  - {key}: {', '.join(str(path) for path in paths[:2])}" for key, paths in missing.items()]
    hint = f"bash scripts/prepare_eval_data.sh {' '.join(missing)}"
    if getattr(args, "data_root_explicit", False) or os.environ.get("EVAL_DATA_ROOT") or os.environ.get("DATA_ROOT"):
        hint += f" --data-root {args.data_root}"
    message = (
        f"benchmark data not found under {args.data_root} for:\n" + "\n".join(lines) + f"\nPrepare it with:\n  {hint}"
    )
    if getattr(args, "skip_missing_data", False):
        print(f"[warn] {message}\n[warn] --skip-missing-data: continuing without {', '.join(missing)}", flush=True)
        return [spec for spec in specs if spec.key not in missing]
    raise SystemExit(f"[error] {message}\n(or pass --skip-missing-data to evaluate the other benchmarks)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged HF/vLLM-loadable checkpoint path or model id.")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR / "latest"))
    parser.add_argument(
        "--run-name",
        help="Name of this run in the summaries (default: the output directory name).",
    )
    parser.add_argument("--global-summary", default=str(DEFAULT_GLOBAL_SUMMARY))
    parser.add_argument(
        "--data-root",
        default=None,
        help="Evaluation data root (default: $EVAL_DATA_ROOT, else $DATA_ROOT/eval, else <repo>/data/eval).",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--suites-config", default=str(DEFAULT_SUITES))
    parser.add_argument(
        "--suite",
        help="Comma-separated suite names from eval/config/suites.yaml (e.g. papo, vppo, deepeyes). "
        "Combined with --benchmarks as a union; --skip-benchmarks is applied last.",
    )
    parser.add_argument(
        "--skip-missing-data",
        action="store_true",
        help="Skip benchmarks whose data has not been prepared instead of aborting.",
    )
    parser.add_argument("--backend", choices=["vllm", "transformers", "dummy"], default="vllm")
    parser.add_argument(
        "--interaction-mode",
        choices=["one_shot", "agentic"],
        default=None,
        help="one_shot (default) or a multi-turn agent profile (agentic).",
    )
    parser.add_argument(
        "--agent-profile",
        choices=["deepeyes"],
        default=None,
        help="Agent implementation selected when --interaction-mode agentic (default deepeyes).",
    )
    parser.add_argument("--agent-max-tool-calls", type=int, default=6)
    parser.add_argument("--agent-max-response-tokens", type=int, default=20480)
    parser.add_argument("--agent-max-tokens-per-turn", type=int, default=10240)
    parser.add_argument("--agent-max-images-per-prompt", type=int, default=16)
    parser.add_argument(
        "--agent-tool-image-mode",
        choices=["original", "fixed_gray", "text_skipped"],
        default="original",
        help=(
            "Observation returned by a successful agent tool call: preserve the "
            "original crop, replace every pixel by a fixed neutral gray value, "
            "or omit the image and return the fixed TextCall sentinel."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark keys to run/score. Summaries always cover every "
        "benchmark already scored in the output dir, not just this selection.",
    )
    parser.add_argument("--skip-benchmarks", help="Comma-separated benchmark keys to skip.")
    parser.add_argument(
        "--gpus", help="Comma-separated visible GPU ids. Defaults to CUDA_VISIBLE_DEVICES or nvidia-smi."
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size per worker.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batch-images", type=int, default=DEFAULT_MAX_BATCH_IMAGES)
    parser.add_argument(
        "--limit", type=int, help="Only evaluate the first N samples of every benchmark (smoke tests)."
    )
    parser.add_argument("--temperature", type=float, help="Override temperature for all benchmarks.")
    parser.add_argument("--num-samples", type=int, help="Override sample count for all benchmarks.")
    parser.add_argument("--max-new-tokens", type=int, help="Override max generated tokens for all benchmarks.")
    parser.add_argument(
        "--top-p", type=float, default=None, help="Override top_p for all benchmarks (registry default 1.0)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sample only from the k most likely tokens (default: no limit, as in training). "
        "PAPO-Eval's LLaMA-Factory default is 50.",
    )
    parser.add_argument(
        "--format-prompt",
        help="Jinja template rendered with {{ content }} (same semantics as data.format_prompt in training). "
        "Default for one-shot runs: examples/format_prompt/math_perception.jinja; 'none' disables it.",
    )
    parser.add_argument("--system-prompt", help="Path to a system prompt text file ('none' for no system prompt).")
    parser.add_argument(
        "--box-format",
        choices=["auto", "norm1000", "pixel"],
        default="auto",
        help="Coordinate convention of predicted boxes. norm1000: [x1, y1, x2, y2] in 0-1000; pixel: absolute "
        "pixels of the resized image the model saw (Qwen2-VL / Qwen2.5-VL), mapped back with the recorded "
        "resize. auto (default): pixel for Qwen2-VL / Qwen2.5-VL models unless the system or format prompt "
        "asks for 0-1000 coordinates, norm1000 otherwise. Agentic runs resolve auto from the model alone "
        "(as agent_bbox_format=auto in training) and use it for the zoom-in tool's bbox_2d.",
    )
    parser.add_argument(
        "--chat-template",
        help="Jinja chat template that replaces the processor's (data.override_chat_template in training). "
        "Agentic runs of Qwen2-VL / Qwen2.5-VL default to examples/chat_template/qwen2_5_vl_tool_call.jinja, "
        "because the stock template ignores tool definitions; 'none' keeps the model's template.",
    )
    parser.add_argument(
        "--plain-think-tokens",
        choices=["auto", "true", "false"],
        default="auto",
        help="Tokenize <think>/</think> as plain text (worker.actor.model.plain_think_tokens in training). auto "
        "(default) applies to models whose tokenizer has them as added tokens that the chat template never uses, "
        "i.e. Qwen3-VL Instruct; evaluate a checkpoint with the setting it was trained with.",
    )
    parser.add_argument(
        "--grounding-instruction",
        choices=["append", "none"],
        default=None,
        help="append (default): the GRIT sets and OVDEval ask for 0-1000 boxes after the question; none: the "
        "question only, for models trained to answer it as is (the grit suite's default).",
    )
    parser.add_argument(
        "--answer-protocol",
        choices=["default", "pepo"],
        default=None,
        help="pepo (the pepo_geometry suite's default): as PEPO's evaluation scripts, MathVerse asks for the "
        "option letter and LogicVista reads the last standalone letter of the answer.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["raw", "chat"],
        default="chat",
        help="raw prepends system text into the prompt; chat uses model chat templates and a system role.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument(
        "--min-pixels", type=int, default=None, help=f"default: the suite's, else {DEFAULT_MIN_PIXELS} (as training)"
    )
    parser.add_argument(
        "--max-pixels", type=int, default=None, help=f"default: the suite's, else {DEFAULT_MAX_PIXELS} (as training)"
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only rebuild summary CSVs from all existing metrics/*.json files (missing benchmarks are skipped).",
    )
    parser.add_argument(
        "--judge-provider",
        choices=list(JUDGE_PROVIDERS),
        default=os.environ.get("EVAL_JUDGE_PROVIDER", "none"),
        help="LLM judge for MM-Vet (required) and the optional GQA judge cascade. 'none' (default) skips "
        "judge-only benchmarks; 'openai' uses OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL; 'deepseek' uses DEEPSEEK_API_KEY.",
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key")
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent external judge requests within one benchmark.",
    )
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument("--judge-request-retries", type=int, default=5)
    parser.add_argument("--judge-thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--dry-run", action="store_true")
    add_perturbation_args(parser)
    args = parser.parse_args(argv)
    args.data_root_explicit = args.data_root is not None
    if args.data_root is None:
        args.data_root = str(default_data_root())
    try:
        apply_prompt_defaults(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.box_format_requested = args.box_format
    args.chat_template_explicit = args.chat_template is not None
    if args.chat_template is not None and args.chat_template.strip().lower() in {"", "none", "null"}:
        args.chat_template = None
    elif args.chat_template is not None and not Path(args.chat_template).is_file():
        parser.error(f"--chat-template file not found: {args.chat_template}")
    if args.interaction_mode == "agentic":
        args.box_format, args.box_format_reason = resolve_agentic_box_format(args)
        select_agentic_system_prompt(args)
        apply_agentic_chat_template_default(args)
    else:
        args.box_format, args.box_format_reason = resolve_box_format(args)
    if args.interaction_mode == "agentic":
        if args.backend != "vllm":
            parser.error("--interaction-mode agentic currently requires --backend vllm")
        if args.prompt_mode != "chat":
            parser.error("--interaction-mode agentic requires --prompt-mode chat")
        if args.max_new_tokens is not None:
            parser.error(
                "agentic mode does not use --max-new-tokens; use "
                "--agent-max-response-tokens and "
                "--agent-max-tokens-per-turn"
            )
        if not args.system_prompt:
            parser.error(
                "native DeepEyes agentic mode requires --system-prompt examples/system_prompt/deepeyes.txt "
                "(0-1000 boxes) or examples/system_prompt/deepeyes_pixel.txt (absolute pixel boxes, "
                "Qwen2-VL / Qwen2.5-VL)"
            )
        if args.format_prompt:
            parser.error(
                "native DeepEyes agentic mode does not use --format-prompt; "
                "its method contract must come from --system-prompt"
            )
        try:
            agent_config = agent_loop_config_from_args(args)
            from verl.workers.agent.chat import render_deepeyes_system_prompt

            system_prompt_template = Path(args.system_prompt).read_text(encoding="utf-8")
            render_deepeyes_system_prompt(
                system_prompt_template,
                max_tool_calls=agent_config.max_tool_calls,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        from verl.workers.agent.coordinates import check_chat_template_supports_tools, check_prompt_matches_bbox_format

        try:
            check_prompt_matches_bbox_format(system_prompt_template, args.box_format)
        except ValueError as exc:
            parser.error(
                f"{exc}. In eval: pass --box-format {'norm1000' if args.box_format == 'pixel' else 'pixel'} "
                "or the matching --system-prompt (examples/system_prompt/deepeyes.txt for 0-1000, "
                "examples/system_prompt/deepeyes_pixel.txt for absolute pixels)"
            )
        try:
            check_chat_template_supports_tools(args.model, args.chat_template)
        except ValueError as exc:
            parser.error(f"{exc}. In eval: pass --chat-template {PROJECT_ROOT / AGENT_TOOL_CHAT_TEMPLATE}")
        if args.agent_max_images_per_prompt <= 0:
            parser.error("--agent-max-images-per-prompt must be positive")
    elif args.agent_tool_image_mode != "original":
        parser.error("--agent-tool-image-mode is only valid with --interaction-mode agentic")
    return args


def apply_prompt_defaults(args: argparse.Namespace) -> None:
    """Fill prompt/interaction flags that were not given: suite defaults first, then built-ins.

    Suite defaults (eval/config/suites.yaml) hold paths relative to the repository root.
    ``none`` disables the format prompt / system prompt.
    """
    suite_defaults: dict[str, Any] = {}
    suite_names = _split_csv(getattr(args, "suite", None)) or []
    if suite_names:
        specs = load_benchmark_specs(Path(args.config))
        suites = load_suites(Path(args.suites_config), specs)
        for name in suite_names:
            if name not in suites:
                raise KeyError(f"unknown suite: {name} (available: {', '.join(sorted(suites))})")
        explicit = {key for key in SUITE_DEFAULT_KEYS if getattr(args, key, None) is not None}
        suite_defaults = merged_suite_defaults(suites, suite_names, ignore=explicit)
    applied = {}
    for key, value in suite_defaults.items():
        if getattr(args, key, None) is None:
            if key in {"format_prompt", "system_prompt"} and str(value).lower() != "none":
                value = str(PROJECT_ROOT / str(value))
            setattr(args, key, value)
            applied[key] = value
    args.suite_defaults_applied = applied
    if getattr(args, "grounding_instruction", None) is None:
        args.grounding_instruction = "append"
    if getattr(args, "answer_protocol", None) is None:
        args.answer_protocol = "default"
    if getattr(args, "min_pixels", None) is None:
        args.min_pixels = DEFAULT_MIN_PIXELS
    if getattr(args, "max_pixels", None) is None:
        args.max_pixels = DEFAULT_MAX_PIXELS
    if args.interaction_mode is None:
        args.interaction_mode = DEFAULT_INTERACTION_MODE
    if args.agent_profile is None:
        args.agent_profile = DEFAULT_AGENT_PROFILE
    if args.format_prompt is None and args.interaction_mode == "one_shot":
        args.format_prompt = str(DEFAULT_FORMAT_PROMPT)
    for key in ("format_prompt", "system_prompt"):
        value = getattr(args, key, None)
        if value is not None and str(value).strip().lower() in {"", "none", "null"}:
            setattr(args, key, None)
        elif value is not None and not Path(value).is_file():
            raise FileNotFoundError(f"--{key.replace('_', '-')} file not found: {value}")


PIXEL_BOX_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl"}
_PIXEL_BOX_MODEL_NAME_RE = re.compile(r"qwen2(?:[._-]?5)?[-_]?vl", flags=re.IGNORECASE)
_NORM1000_PROMPT_RE = re.compile(r"0\s*[-\u2013~]\s*1000")


def detect_model_type(model: str) -> str | None:
    """``model_type`` from a local checkpoint or the Hugging Face cache (no network access)."""
    path = Path(model).expanduser()
    config_path: Path | None = path / "config.json" if path.is_dir() else None
    if config_path is None or not config_path.is_file():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model, "config.json")
            config_path = Path(cached) if isinstance(cached, str) else None
        except Exception:  # noqa: BLE001 - not an HF id / cache unavailable
            config_path = None
    if config_path is None or not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError):
        return None


AGENT_TOOL_CHAT_TEMPLATE = "examples/chat_template/qwen2_5_vl_tool_call.jinja"


def resolve_agentic_box_format(args: argparse.Namespace) -> tuple[str, str]:
    """(bbox format of the zoom-in tool, reason) for agentic runs.

    Uses the training-side rule (``verl.workers.agent.coordinates.resolve_bbox_format``): ``auto``
    is ``pixel`` for Qwen2-VL / Qwen2.5-VL checkpoints and ``norm1000`` otherwise. The system
    prompt is chosen to match (see ``select_agentic_system_prompt``) and checked against it.
    """
    from verl.workers.agent.coordinates import resolve_bbox_format

    requested = getattr(args, "box_format", "auto") or "auto"
    resolved = resolve_bbox_format(requested, str(args.model))
    if requested != "auto":
        return resolved, "--box-format"
    model_type = detect_model_type(str(args.model))
    return resolved, f"agentic auto, model_type={model_type}" if model_type else "agentic auto, model name"


def _pixel_prompt_variant(path: str) -> Path | None:
    candidate = Path(path).with_name(f"{Path(path).stem}_pixel{Path(path).suffix}")
    return candidate if candidate.is_file() else None


def select_agentic_system_prompt(args: argparse.Namespace) -> None:
    """Swap a suite-default agent system prompt for its ``*_pixel`` variant in pixel mode.

    ``--suite deepeyes`` defaults to ``deepeyes.txt`` (0-1000 boxes); a Qwen2-VL / Qwen2.5-VL run
    resolves to pixel boxes and gets ``deepeyes_pixel.txt``. An explicit --system-prompt is kept
    (and later checked against the box format).
    """
    applied = getattr(args, "suite_defaults_applied", None) or {}
    if args.box_format != "pixel" or "system_prompt" not in applied or not args.system_prompt:
        return
    variant = _pixel_prompt_variant(args.system_prompt)
    if variant is not None:
        args.system_prompt = str(variant)
        applied["system_prompt"] = str(variant)


def apply_agentic_chat_template_default(args: argparse.Namespace) -> None:
    """Default the tool-capable chat template for agentic Qwen2-VL / Qwen2.5-VL runs."""
    from verl.workers.agent.coordinates import PIXEL_COORDINATE_MODEL_TYPES

    if getattr(args, "chat_template_explicit", False):
        return
    if args.box_format == "pixel" or detect_model_type(str(args.model)) in PIXEL_COORDINATE_MODEL_TYPES:
        args.chat_template = str(PROJECT_ROOT / AGENT_TOOL_CHAT_TEMPLATE)


def resolve_box_format(args: argparse.Namespace) -> tuple[str, str]:
    """(box format, reason) for --box-format auto.

    Qwen2-VL / Qwen2.5-VL emit absolute pixel coordinates of their resized input, Qwen3-VL and
    most other models 0-1000 coordinates; a prompt that explicitly asks for 0-1000 coordinates
    (e.g. the CGPO, GRIT-style and DeepEyes prompts) overrides the model default.
    """
    requested = getattr(args, "box_format", "auto") or "auto"
    if requested != "auto":
        return requested, "--box-format"
    for key in ("system_prompt", "format_prompt"):
        value = getattr(args, key, None)
        if value and Path(value).is_file() and _NORM1000_PROMPT_RE.search(Path(value).read_text(encoding="utf-8")):
            return "norm1000", f"--{key.replace('_', '-')} asks for 0-1000 coordinates"
    model_type = detect_model_type(str(args.model))
    if model_type:
        return ("pixel" if model_type in PIXEL_BOX_MODEL_TYPES else "norm1000"), f"model_type={model_type}"
    if _PIXEL_BOX_MODEL_NAME_RE.search(str(args.model)):
        return "pixel", "model name"
    return "norm1000", "default"


def run_all(specs: list[BenchmarkSpec], args: argparse.Namespace) -> dict[str, str]:
    gpus = parse_gpu_list(args.gpus)
    gpu_groups = group_gpus(gpus, args.tp)
    if args.output_dir.name == "latest":
        args.output_dir = args.output_dir.parent / datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = getattr(args, "run_name", None) or args.output_dir.name

    if args.summary_only:
        write_run_summaries(collect_run_results(args, []), args, run_id)
        return {}

    from .scorers import judge_config_from_args, judge_missing_reason

    judge_config = judge_config_from_args(args)
    failures: dict[str, str] = {}
    skipped_results = []
    if judge_config is None:
        reason = judge_missing_reason(args)
        for spec in specs:
            if spec.requires_judge:
                print(f"[warn] skipping {spec.key}: it needs an LLM judge and {reason}", flush=True)
                skipped_results.append(record_skipped_benchmark(spec, args, reason))
        specs = [spec for spec in specs if not spec.requires_judge]
    if not specs:
        write_run_summaries(collect_run_results(args, skipped_results), args, run_id)
        return failures

    score_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark-score")
    score_futures: dict[str, Future] = {}
    ready: set[str] = set()
    next_score_index = 0

    def score_task(spec: BenchmarkSpec):
        # A failing scorer (bad data, judge outage, ...) must not cancel the scoring of the
        # remaining benchmarks: log it, keep going, and report a non-zero exit at the end.
        try:
            return score_benchmark(spec, args, judge_config)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            failures[spec.key] = f"scoring failed: {type(exc).__name__}: {exc}"
            print(f"[error] scoring {spec.key} failed ({type(exc).__name__}: {exc}); continuing", flush=True)
            return None

    def benchmark_ready(spec: BenchmarkSpec) -> None:
        nonlocal next_score_index
        if spec.key in ready:
            raise RuntimeError(f"benchmark became ready more than once: {spec.key}")
        ready.add(spec.key)
        while next_score_index < len(specs) and specs[next_score_index].key in ready:
            next_spec = specs[next_score_index]
            if next_spec.key not in failures:
                score_futures[next_spec.key] = score_executor.submit(score_task, next_spec)
            next_score_index += 1

    try:
        if args.score_only:
            for spec in specs:
                benchmark_ready(spec)
        else:
            run_inference(specs, args, gpu_groups, on_benchmark_ready=benchmark_ready, failures=failures)
        missing = [spec.key for spec in specs if spec.key not in score_futures and spec.key not in failures]
        if missing:
            raise RuntimeError(f"benchmarks never became ready for scoring: {', '.join(missing)}")
        results = [score_futures[spec.key].result() for spec in specs if spec.key in score_futures]
    finally:
        score_executor.shutdown(wait=True, cancel_futures=True)

    results = [result for result in results if result is not None] + skipped_results
    write_run_summaries(collect_run_results(args, results), args, run_id)
    return failures


def record_skipped_benchmark(spec: BenchmarkSpec, args: argparse.Namespace, reason: str):
    """Write a 'skipped' metric file unless the run already holds a real score for it."""
    from .schemas import MetricResult
    from .scorers import skipped_result

    metric_path = metrics_dir(args.output_dir) / f"{spec.key}.json"
    if metric_path.exists():
        try:
            existing = MetricResult(**read_json(metric_path)["result"])
        except Exception:  # noqa: BLE001 - unreadable file: overwrite with the skip marker
            existing = None
        if existing is not None and existing.status != SKIPPED_STATUS:
            return existing
    result = skipped_result(spec, reason)
    write_json(metric_path, {"result": result.__dict__})
    return result


def score_benchmark(spec: BenchmarkSpec, args: argparse.Namespace, judge_config):
    from .schemas import MetricResult
    from .scorers import score_predictions

    prediction_path = merged_prediction_path(args.output_dir, spec)
    metric_path = metrics_dir(args.output_dir) / f"{spec.key}.json"
    fp = task_fingerprint(args, spec, phase="score")
    if args.resume and is_complete(args.output_dir, f"score:{spec.key}", fp, [metric_path]):
        result_data = read_json(metric_path)["result"]
        return MetricResult(**result_data)
    try:
        result = score_predictions(
            spec,
            prediction_path,
            metric_path,
            judge_config=judge_config,
            metric_metadata=metric_metadata_for(spec, args),
        )
    except Exception as exc:
        mark_failed(args.output_dir, f"score:{spec.key}", fp, f"{type(exc).__name__}: {exc}")
        raise
    mark_complete(args.output_dir, f"score:{spec.key}", fp, [metric_path])
    return result


def write_run_summaries(results, args: argparse.Namespace, run_id: str) -> None:
    run_metadata = {
        "run_id": run_id,
        "output_dir": str(args.output_dir),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "backend": args.backend,
        "temperature": args.temperature if args.temperature is not None else "per_benchmark",
        "num_samples": args.num_samples if args.num_samples is not None else "per_benchmark",
        **({"top_k": args.top_k} if getattr(args, "top_k", None) is not None else {}),
        **(
            {"grounding_instruction": args.grounding_instruction}
            if getattr(args, "grounding_instruction", "append") != "append"
            else {}
        ),
        **(
            {"answer_protocol": args.answer_protocol}
            if getattr(args, "answer_protocol", "default") != "default"
            else {}
        ),
        "batch_size": args.batch_size,
        "max_batch_images": args.max_batch_images,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "format_prompt": args.format_prompt or "",
        "system_prompt": args.system_prompt or "",
        "chat_template": getattr(args, "chat_template", None) or "",
        "plain_think_tokens": getattr(args, "plain_think_tokens", "auto"),
        "prompt_mode": args.prompt_mode,
        "box_format": getattr(args, "box_format", "norm1000"),
        "interaction_mode": getattr(args, "interaction_mode", "one_shot"),
        "judge_provider": args.judge_provider,
        "judge_model": _resolved_judge_model(args),
        "judge_concurrency": args.judge_concurrency,
        "judge_max_tokens": resolve_judge_max_tokens(args.judge_max_tokens, args.judge_thinking),
        "judge_request_retries": args.judge_request_retries,
        "judge_thinking": args.judge_thinking,
        "perturbation_vllm_force_feature_wrapper": bool(
            getattr(args, "perturbation_vllm_force_feature_wrapper", False)
        ),
    }
    if getattr(args, "interaction_mode", "one_shot") == "agentic":
        run_metadata.update(_agent_fingerprint_fields(args))
    run_metadata.update(perturbation_metadata(args.perturbation, args.perturbation_seed))
    write_summary_csv(args.output_dir / "summary.csv", results, run_metadata=run_metadata)
    update_global_summary_csv(Path(args.global_summary), results, run_metadata=run_metadata)
    print(format_results_table(results, title=f"Results for {run_id} ({args.model})"), flush=True)
    print(f"[done] summary: {args.output_dir / 'summary.csv'}", flush=True)
    print(f"[done] global summary: {Path(args.global_summary)}", flush=True)


def format_results_table(results, *, title: str = "Results") -> str:
    """Plain-text table of the primary scores (benchmarks, group averages, overall)."""
    lines = [title, f"  {'benchmark':<20} {'group':<11} {'metric':<20} {'score':>8} {'n':>7}  status"]
    ok = []
    for result in results:
        score = result.normalized_score_0_100
        text = f"{score:8.2f}" if isinstance(score, (int, float)) and result.status == "ok" else f"{'-':>8}"
        lines.append(
            f"  {result.benchmark:<20} {result.group:<11} {result.primary_metric:<20} {text} {result.num_examples:>7}  {result.status}"
        )
        if result.status == "ok" and isinstance(score, (int, float)):
            ok.append(result)
    groups: dict[str, list[float]] = {}
    for result in ok:
        groups.setdefault(result.group, []).append(float(result.normalized_score_0_100))
    for group, scores in groups.items():
        lines.append(f"  {group + ' avg':<20} {'':<11} {'':<20} {sum(scores) / len(scores):8.2f}")
    if ok:
        overall = sum(float(result.normalized_score_0_100) for result in ok) / len(ok)
        lines.append(f"  {'overall avg':<20} {'':<11} {'':<20} {overall:8.2f}")
    return "\n".join(lines)


def metric_metadata_for(spec: BenchmarkSpec, args: argparse.Namespace) -> dict[str, Any]:
    metadata = {"max_model_len": args.max_model_len, "box_format": getattr(args, "box_format", "norm1000")}
    if getattr(args, "answer_protocol", "default") != "default":
        metadata["answer_protocol"] = args.answer_protocol
    if getattr(args, "chat_template", None):
        metadata["chat_template"] = args.chat_template
    if getattr(args, "plain_think_tokens", "auto") != "auto":
        metadata["plain_think_tokens"] = args.plain_think_tokens
    if getattr(args, "interaction_mode", "one_shot") == "agentic":
        metadata.update(_agent_fingerprint_fields(args))
    metadata.update(perturbation_metadata(args.perturbation, args.perturbation_seed))
    return metadata


def collect_run_results(args: argparse.Namespace, fresh_results: list) -> list:
    """All results known for this run: fresh scores merged with on-disk metric jsons.

    Summaries are always built from this union in benchmark-config order, so a
    partial invocation (--benchmarks gqa) updates that benchmark's metric json
    without collapsing summary.csv / the global summary row to the subset that
    happened to run. --benchmarks therefore only selects what gets (re)scored,
    never what the summaries cover.
    """
    from .schemas import MetricResult

    fresh = {result.benchmark: result for result in fresh_results}
    results = []
    for spec in load_benchmark_specs(Path(args.config)):
        if spec.key in fresh:
            results.append(fresh[spec.key])
            continue
        metric_path = metrics_dir(args.output_dir) / f"{spec.key}.json"
        if metric_path.exists():
            results.append(MetricResult(**read_json(metric_path)["result"]))
    return results


def run_inference(
    specs: list[BenchmarkSpec],
    args: argparse.Namespace,
    gpu_groups: list[str],
    *,
    on_benchmark_ready: Callable[[BenchmarkSpec], None] | None = None,
    failures: dict[str, str] | None = None,
) -> None:
    failures = failures if failures is not None else {}
    if args.backend == "dummy":
        for spec in specs:
            try:
                infer_benchmark_shard(spec, args, shard_index=0, num_shards=1)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                failures[spec.key] = f"inference failed: {type(exc).__name__}: {exc}"
            else:
                merge_prediction_shards(args.output_dir, spec, 1)
            if on_benchmark_ready is not None:
                on_benchmark_ready(spec)
        return

    if not gpu_groups:
        raise RuntimeError("No GPUs detected. Pass --gpus or use --backend dummy for tests.")

    if args.backend == "vllm":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    tasks = []
    num_shards = len(gpu_groups)
    remaining_shards = {spec.key: 0 for spec in specs}
    ready: set[str] = set()
    specs_by_key = {spec.key: spec for spec in specs}
    for spec in specs:
        for shard_index in range(num_shards):
            task_output = shard_prediction_path(args.output_dir, spec, shard_index)
            fp = task_fingerprint(args, spec, phase="infer", shard_index=shard_index, num_shards=num_shards)
            if args.resume and is_complete(args.output_dir, f"infer:{spec.key}:shard{shard_index}", fp, [task_output]):
                continue
            tasks.append({"spec": spec, "shard_index": shard_index, "num_shards": num_shards, "fingerprint": fp})
            remaining_shards[spec.key] += 1

    def finish_benchmark_if_ready(spec: BenchmarkSpec) -> None:
        if remaining_shards[spec.key] != 0 or spec.key in ready:
            return
        if spec.key not in failures:
            merge_prediction_shards(args.output_dir, spec, num_shards)
        ready.add(spec.key)
        if on_benchmark_ready is not None:
            on_benchmark_ready(spec)

    for spec in specs:
        finish_benchmark_if_ready(spec)

    if tasks:
        mp_context = mp.get_context("spawn")
        task_queue: mp.Queue = mp_context.Queue()
        result_queue: mp.Queue = mp_context.Queue()
        processes = []
        for worker_index, gpu_group in enumerate(gpu_groups):
            process = mp_context.Process(
                target=worker_main,
                args=(task_queue, result_queue, args, gpu_group, worker_index),
                daemon=False,
            )
            process.start()
            processes.append(process)
        for item in tasks:
            task_queue.put(item)
        for _ in processes:
            task_queue.put(None)

        pending = len(tasks)
        completed = False
        try:
            while pending:
                try:
                    result = result_queue.get(timeout=10)
                except queue.Empty:
                    failed = [process.exitcode for process in processes if process.exitcode not in {None, 0}]
                    if failed:
                        raise RuntimeError(f"inference worker exited unexpectedly: {failed}")
                    continue
                pending -= 1
                spec = specs_by_key[result["key"]]
                state_key = f"infer:{spec.key}:shard{result['shard_index']}"
                if result["status"] != "ok":
                    # Keep the other benchmarks running; this one is reported at the end.
                    mark_failed(args.output_dir, state_key, result["fingerprint"], result["error"])
                    print(f"[error] {state_key} failed: {result['error']}", flush=True)
                    failures.setdefault(spec.key, f"inference failed: {result['error']}")
                else:
                    mark_complete(args.output_dir, state_key, result["fingerprint"], [Path(result["output"])])
                remaining_shards[spec.key] -= 1
                finish_benchmark_if_ready(spec)
            completed = True
        finally:
            cleanup_worker_processes(processes, terminate=not completed)

    missing = [spec.key for spec in specs if spec.key not in ready]
    if missing:
        raise RuntimeError(f"inference completed without ready benchmarks: {', '.join(missing)}")


def cleanup_worker_processes(processes: list[mp.Process], *, terminate: bool) -> None:
    if terminate:
        for process in processes:
            _signal_process_group(process.pid, signal.SIGTERM)
            if process.is_alive():
                process.terminate()
        join_timeout = 10
    else:
        join_timeout = 30

    for process in processes:
        process.join(timeout=join_timeout)

    for process in processes:
        if process.is_alive():
            _signal_process_group(process.pid, signal.SIGKILL)
            process.kill()
            process.join(timeout=5)
        else:
            _signal_process_group(process.pid, signal.SIGTERM)


def _signal_process_group(pid: int | None, sig: int) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def worker_main(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    args: argparse.Namespace,
    gpu_group: str,
    worker_index: int,
) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
    # Ports are laid out per worker below a base that concurrent runner
    # invocations on the same node must override (EASYR1_WORKER_PORT_BASE),
    # otherwise their worker-0 engines all race for 29500/29501 (EADDRINUSE).
    port_base = int(os.environ.get("EASYR1_WORKER_PORT_BASE", "29500"))
    os.environ["VLLM_PORT"] = str(port_base + worker_index * 10)
    os.environ["MASTER_PORT"] = str(port_base + 1 + worker_index * 10)
    if args.backend == "vllm":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    backend = build_eval_backend(args)
    while True:
        item = task_queue.get()
        if item is None:
            return
        spec: BenchmarkSpec = item["spec"]
        try:
            output = infer_benchmark_shard(
                spec,
                args,
                shard_index=item["shard_index"],
                num_shards=item["num_shards"],
                backend=backend,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            result_queue.put(
                {
                    "key": spec.key,
                    "shard_index": item["shard_index"],
                    "fingerprint": item["fingerprint"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        result_queue.put(
            {
                "key": spec.key,
                "shard_index": item["shard_index"],
                "fingerprint": item["fingerprint"],
                "status": "ok",
                "output": str(output),
            }
        )


def infer_benchmark_shard(
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    *,
    shard_index: int,
    num_shards: int,
    backend: Any | None = None,
) -> Path:
    # --limit keeps the first N samples of the benchmark (in total), which are then sharded.
    samples = shard_samples(load_samples(spec, args.data_root, limit=args.limit), shard_index, num_shards)
    samples = apply_interaction_prompt_contract(samples, args)
    samples = apply_grounding_instruction(samples, args)
    samples = apply_answer_protocol(samples, spec, args)
    samples = apply_box_format_to_prompts(samples, getattr(args, "box_format", "norm1000"))
    samples = apply_prompt_config(samples, prompt_config_from_args(args))
    output = shard_prediction_path(args.output_dir, spec, shard_index)
    if backend is None:
        backend = build_eval_backend(args)
    rows = []
    agent_trajectory_count = 0
    agent_success_count = 0
    eval_metadata = metric_metadata_for(spec, args)
    for batch in iter_sample_batches(samples, args.batch_size, args.max_batch_images):
        responses = backend.generate(batch, generation_config_for(spec, args))
        for sample, sample_responses in zip(batch, responses):
            for response in sample_responses:
                if not isinstance(response, GenerationOutput):
                    continue
                diagnostics = response.diagnostics
                agent = diagnostics.get("agent") if isinstance(diagnostics, dict) else None
                if not isinstance(agent, dict):
                    continue
                agent_trajectory_count += 1
                if agent.get("status") != "sample_error":
                    agent_success_count += 1
            rows.append(prediction_row(sample, sample_responses, eval_metadata=eval_metadata))
    if (
        getattr(args, "interaction_mode", "one_shot") == "agentic"
        and agent_trajectory_count > 0
        and agent_success_count == 0
    ):
        raise RuntimeError(
            f"{spec.key}: every agent trajectory in shard {shard_index} failed before producing a scoreable result"
        )
    write_jsonl(output, rows)
    return output


def iter_sample_batches(samples: list[EvalSample], max_samples: int, max_images: int | None):
    if max_samples < 1:
        raise ValueError("--batch-size must be >= 1")
    if max_images is not None and max_images < 1:
        raise ValueError("--max-batch-images must be >= 1")
    batch: list[EvalSample] = []
    image_count = 0
    for sample in samples:
        sample_images = len(sample.images)
        image_limit_reached = max_images is not None and batch and image_count + sample_images > max_images
        if len(batch) >= max_samples or image_limit_reached:
            yield batch
            batch = []
            image_count = 0
        batch.append(sample)
        image_count += sample_images
    if batch:
        yield batch


def apply_interaction_prompt_contract(
    samples: list[EvalSample],
    args: argparse.Namespace,
) -> list[EvalSample]:
    """Select a method-native output contract before rendering chat messages."""

    if (
        getattr(args, "interaction_mode", "one_shot") != "agentic"
        or getattr(
            args,
            "agent_output_contract",
            AGENT_OUTPUT_CONTRACT_NATIVE,
        )
        != AGENT_OUTPUT_CONTRACT_NATIVE
    ):
        return samples
    return [
        (replace(sample, prompt=sample.native_agentic_prompt) if sample.native_agentic_prompt is not None else sample)
        for sample in samples
    ]


PEPO_MATHVERSE_INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."


def apply_answer_protocol(
    samples: list[EvalSample], spec: BenchmarkSpec, args: argparse.Namespace
) -> list[EvalSample]:
    """--answer-protocol pepo: MathVerse questions ask for the option letter, as PEPO's evaluate_mathverse.py."""
    if getattr(args, "answer_protocol", "default") != "pepo" or spec.key != "mathverse":
        return samples
    return [replace(sample, prompt=sample.prompt + PEPO_MATHVERSE_INSTRUCTION) for sample in samples]


def apply_grounding_instruction(samples: list[EvalSample], args: argparse.Namespace) -> list[EvalSample]:
    """With --grounding-instruction none, ask the bare question where a loader appended a box instruction."""
    if getattr(args, "grounding_instruction", "append") != "none":
        return samples
    return [
        replace(sample, prompt=sample.question_only_prompt) if sample.question_only_prompt is not None else sample
        for sample in samples
    ]


def prediction_row(
    sample: EvalSample,
    responses: list[str | GenerationOutput | dict[str, Any]],
    *,
    eval_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .json_utils import to_jsonable

    response_texts, response_metadata = normalize_generation_outputs(responses)
    perturbation_diagnostics = []
    for response in responses:
        diagnostics = getattr(response, "diagnostics", None) if isinstance(response, GenerationOutput) else None
        if isinstance(diagnostics, dict) and diagnostics.get("perturbation"):
            # Perturbations are deterministic per sample/image/seed; keep one compact sample-level record.
            perturbation_diagnostics.append(diagnostics["perturbation"])
            break
    image_sizes = None
    for response in responses:
        diagnostics = getattr(response, "diagnostics", None) if isinstance(response, GenerationOutput) else None
        if isinstance(diagnostics, dict) and diagnostics.get("image_sizes"):
            # Original / processed / model-input sizes; needed to read absolute-pixel boxes.
            image_sizes = diagnostics["image_sizes"]
            break
    row = {
        "benchmark": sample.benchmark,
        "sample_id": sample.sample_id,
        "prompt": sample.prompt,
        "messages": to_jsonable(sample.messages),
        "target": to_jsonable(sample.target),
        "responses": response_texts,
        "response_metadata": response_metadata,
        "extra_info": compact_extra_info_for_prediction(to_jsonable(sample.extra_info)),
        "metadata": to_jsonable(sample.metadata),
        "eval_metadata": to_jsonable(eval_metadata or {}),
        "perturbation_diagnostics": to_jsonable(perturbation_diagnostics),
        "image_refs": [str(image) if isinstance(image, str) else "<embedded>" for image in sample.images],
    }
    if image_sizes is not None:
        row["image_sizes"] = to_jsonable(image_sizes)
    agent_diagnostics = [
        response.diagnostics["agent"]
        for response in responses
        if isinstance(response, GenerationOutput)
        and isinstance(response.diagnostics, dict)
        and isinstance(response.diagnostics.get("agent"), dict)
    ]
    if agent_diagnostics:
        row["agent_diagnostics"] = to_jsonable(agent_diagnostics)
    return row


def normalize_generation_outputs(
    responses: list[str | GenerationOutput | dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    response_texts = []
    response_metadata = []
    for response in responses:
        if isinstance(response, GenerationOutput):
            response_texts.append(response.text)
            response_metadata.append(
                {
                    "finish_reason": response.finish_reason,
                    "stop_reason": response.stop_reason,
                    "token_count": response.token_count,
                    "truncated": response.truncated,
                }
            )
        elif isinstance(response, dict):
            response_texts.append(str(response.get("text", "")))
            response_metadata.append(
                {
                    "finish_reason": response.get("finish_reason"),
                    "stop_reason": response.get("stop_reason"),
                    "token_count": response.get("token_count"),
                    "truncated": response.get("truncated"),
                }
            )
        else:
            response_texts.append(str(response))
            response_metadata.append(
                {"finish_reason": None, "stop_reason": None, "token_count": None, "truncated": None}
            )
    return response_texts, response_metadata


def compact_extra_info_for_prediction(value: Any) -> Any:
    if isinstance(value, dict):
        compacted = {}
        for key, item in value.items():
            if key in {"patches", "rle", "segmentation"}:
                continue
            compacted[key] = compact_extra_info_for_prediction(item)
        return compacted
    if isinstance(value, list):
        return [compact_extra_info_for_prediction(item) for item in value]
    return value


def generation_config_for(spec: BenchmarkSpec, args: argparse.Namespace) -> GenerationConfig:
    temperature = (
        args.temperature
        if args.temperature is not None
        else (spec.temperature if spec.temperature is not None else 0.0)
    )
    num_samples = (
        args.num_samples if args.num_samples is not None else (spec.num_samples if spec.num_samples is not None else 1)
    )
    if getattr(args, "interaction_mode", "one_shot") == "agentic":
        max_new_tokens = agent_loop_config_from_args(args).max_response_tokens
    else:
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else spec.max_new_tokens
    return GenerationConfig(
        temperature=temperature,
        top_p=args.top_p if args.top_p is not None else spec.top_p,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        seed=args.seed,
        top_k=getattr(args, "top_k", None),
    )


def agent_loop_config_from_args(args: argparse.Namespace):
    from verl.workers.agent.protocol import AgentLoopConfig

    return AgentLoopConfig(
        max_tool_calls=int(getattr(args, "agent_max_tool_calls", 6)),
        max_response_tokens=int(getattr(args, "agent_max_response_tokens", 20480)),
        max_tokens_per_turn=int(getattr(args, "agent_max_tokens_per_turn", 10240)),
    )


def build_eval_backend(args: argparse.Namespace):
    return build_backend(
        args.backend,
        args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        perturbation=args.perturbation,
        perturbation_seed=args.perturbation_seed,
        output_dir=args.output_dir,
        save_perturbation_samples=args.perturbation_save_samples,
        force_vllm_feature_wrapper=bool(getattr(args, "perturbation_vllm_force_feature_wrapper", False)),
        interaction_mode=getattr(args, "interaction_mode", "one_shot"),
        agent_profile=getattr(args, "agent_profile", "deepeyes"),
        agent_config=agent_loop_config_from_args(args),
        agent_max_images_per_prompt=int(getattr(args, "agent_max_images_per_prompt", 16)),
        agent_max_batch_images=int(args.max_batch_images),
        agent_tool_image_mode=getattr(
            args,
            "agent_tool_image_mode",
            "original",
        ),
        agent_bbox_format=getattr(args, "box_format", "norm1000"),
        chat_template=getattr(args, "chat_template", None),
        plain_think_tokens=getattr(args, "plain_think_tokens", "auto"),
    )


def merge_prediction_shards(out_dir: Path, spec: BenchmarkSpec, num_shards: int) -> None:
    output = merged_prediction_path(out_dir, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for shard_index in range(num_shards):
            shard_path = shard_prediction_path(out_dir, spec, shard_index)
            if not shard_path.exists():
                raise FileNotFoundError(f"missing shard output: {shard_path}")
            with shard_path.open("r", encoding="utf-8") as f:
                shutil.copyfileobj(f, out)


def task_fingerprint(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    *,
    phase: str,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> str:
    return fingerprint(
        {
            "phase": phase,
            "model": args.model,
            "backend": args.backend,
            **_agent_fingerprint_fields(args),
            "spec": spec.__dict__,
            # top_k only when set, so runs without it keep their fingerprints
            "generation": {
                key: value
                for key, value in generation_config_for(spec, args).__dict__.items()
                if key != "top_k" or value is not None
            },
            "limit": args.limit,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_model_len": args.max_model_len,
            "format_prompt": _file_fingerprint(getattr(args, "format_prompt", None)),
            "system_prompt": _file_fingerprint(getattr(args, "system_prompt", None)),
            # only present when set, so runs without a template override keep their fingerprints
            **(
                {"chat_template": _file_fingerprint(args.chat_template)}
                if getattr(args, "chat_template", None)
                else {}
            ),
            **(
                {"plain_think_tokens": args.plain_think_tokens}
                if getattr(args, "plain_think_tokens", "auto") != "auto"
                else {}
            ),
            "prompt_mode": args.prompt_mode,
            # only when set, so runs with the default keep their fingerprints
            **(
                {"grounding_instruction": args.grounding_instruction}
                if getattr(args, "grounding_instruction", "append") != "append"
                else {}
            ),
            **(
                {"answer_protocol": args.answer_protocol}
                if getattr(args, "answer_protocol", "default") != "default"
                else {}
            ),
            "box_format": getattr(args, "box_format", "norm1000"),
            "shard_index": shard_index,
            "num_shards": num_shards,
            # The judge only affects scoring; changing it must not invalidate inference.
            **(_judge_fingerprint_fields(args) if phase == "score" else {}),
            "scorer_version": SCORER_VERSION if phase == "score" else None,
            "perturbation": args.perturbation.__dict__,
            "perturbation_seed": args.perturbation_seed,
            "perturbation_vllm_force_feature_wrapper": bool(
                getattr(args, "perturbation_vllm_force_feature_wrapper", False)
            ),
        }
    )


def predictions_dir(out_dir: Path) -> Path:
    return out_dir / "predictions"


def metrics_dir(out_dir: Path) -> Path:
    return out_dir / "metrics"


def shard_prediction_path(out_dir: Path, spec: BenchmarkSpec, shard_index: int) -> Path:
    return predictions_dir(out_dir) / spec.key / f"shard{shard_index:02d}.jsonl"


def merged_prediction_path(out_dir: Path, spec: BenchmarkSpec) -> Path:
    return predictions_dir(out_dir) / spec.key / "predictions.jsonl"


def parse_gpu_list(value: str | None) -> list[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_value:
        return [item.strip() for item in env_value.split(",") if item.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def group_gpus(gpus: list[str], tp: int) -> list[str]:
    if tp < 1:
        raise ValueError("--tp must be >= 1")
    if not gpus:
        return []
    if len(gpus) % tp != 0:
        raise ValueError(f"GPU count ({len(gpus)}) must be divisible by --tp ({tp})")
    return [",".join(gpus[index : index + tp]) for index in range(0, len(gpus), tp)]


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def print_dry_run(specs: list[BenchmarkSpec], args: argparse.Namespace) -> None:
    from .scorers import judge_config_from_args

    print(f"model: {args.model}")
    print(f"data_root: {args.data_root}")
    print(f"output_dir: {args.output_dir}")
    if getattr(args, "suite", None):
        print(f"suite: {args.suite}")
    if getattr(args, "suite_defaults_applied", None):
        print(f"suite_defaults_applied: {args.suite_defaults_applied}")
    judge = judge_config_from_args(args)
    print(f"judge: {judge.provider + '/' + judge.model if judge else 'none'}")
    print(f"global_summary: {args.global_summary}")
    print(f"prompt_mode: {args.prompt_mode}")
    print(f"box_format: {getattr(args, 'box_format', 'norm1000')} ({getattr(args, 'box_format_reason', '')})")
    print(f"chat_template: {getattr(args, 'chat_template', None) or 'model default'}")
    print(f"interaction_mode: {getattr(args, 'interaction_mode', 'one_shot')}")
    if getattr(args, "interaction_mode", "one_shot") == "agentic":
        print(f"agent_profile: {args.agent_profile}")
        print(f"agent_output_contract: {getattr(args, 'agent_output_contract', AGENT_OUTPUT_CONTRACT_NATIVE)}")
        print(f"agent_config: {agent_loop_config_from_args(args)}")
        print(f"agent_max_images_per_prompt: {args.agent_max_images_per_prompt}")
        print(f"agent_tool_image_mode: {args.agent_tool_image_mode}")
    for run_args in expand_eval_runs(args):
        print(
            f"perturbation: {run_args.perturbation.summary()} "
            f"seed={run_args.perturbation_seed} output_dir={run_args.output_dir}"
        )
    if args.format_prompt:
        print(f"format_prompt: {args.format_prompt}")
    if args.system_prompt:
        print(f"system_prompt: {args.system_prompt}")
    print(f"min_pixels: {args.min_pixels}")
    print(f"max_pixels: {args.max_pixels}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_batch_images: {args.max_batch_images}")
    print(f"max_model_len: {args.max_model_len}")
    print(f"gpu_memory_utilization: {args.gpu_memory_utilization}")
    for spec in specs:
        gen = generation_config_for(spec, args)
        notes = []
        if missing_data_files(spec, args.data_root):
            notes.append("DATA MISSING")
        if spec.requires_judge and judge is None:
            notes.append("skipped: needs judge")
        print(
            f"- {spec.key}: group={spec.group} loader={spec.loader} scorer={spec.scorer} "
            f"temp={gen.temperature} top_p={gen.top_p} top_k={gen.top_k} n={gen.num_samples} "
            f"max_new_tokens={gen.max_new_tokens}" + (f"  [{'; '.join(notes)}]" if notes else "")
        )


def _agent_fingerprint_fields(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "interaction_mode", "one_shot") != "agentic":
        return {}
    config = agent_loop_config_from_args(args)
    return {
        "interaction_mode": "agentic",
        "agent_profile": getattr(args, "agent_profile", "deepeyes"),
        "agent_runner_version": AGENTIC_RUNNER_VERSION,
        "agent_output_contract": getattr(
            args,
            "agent_output_contract",
            AGENT_OUTPUT_CONTRACT_NATIVE,
        ),
        "agent_config": config.__dict__,
        "agent_max_tool_calls": config.max_tool_calls,
        "agent_max_response_tokens": config.max_response_tokens,
        "agent_max_tokens_per_turn": config.max_tokens_per_turn,
        "agent_max_images_per_prompt": int(getattr(args, "agent_max_images_per_prompt", 16)),
        "agent_tool_image_mode": getattr(
            args,
            "agent_tool_image_mode",
            "original",
        ),
    }


def _file_fingerprint(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return {"name": p.name, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def _resolved_judge_model(args: argparse.Namespace) -> str:
    from .scorers import judge_config_from_args

    config = judge_config_from_args(args)
    return config.model if config is not None else ""


def _judge_fingerprint_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "judge_provider": args.judge_provider,
        "judge_model": _resolved_judge_model(args),
        "judge_max_tokens": resolve_judge_max_tokens(args.judge_max_tokens, args.judge_thinking),
        "judge_thinking": args.judge_thinking,
    }
