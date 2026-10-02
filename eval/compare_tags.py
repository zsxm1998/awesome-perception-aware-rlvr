#!/usr/bin/env python
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
"""Paired significance comparison between two eval runs.

Usage:
    python eval/compare_tags.py RUN_A RUN_B [options]

RUN_A / RUN_B are result directories or tag names under eval/results/.
For every benchmark evaluated in both runs (discovered from the benchmark
config plus each run's predictions), the tool recomputes per-sample primary
scores with the current scorer code and reports the paired-bootstrap delta,
95% CI, p-value and Benjamini-Hochberg q-value; group averages and the
overall average are tested the same way by pooling per-benchmark replicates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT))

from easyr1_eval.compare import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    ComparisonReport,
    compare_runs,
    report_to_dict,
)
from easyr1_eval.registry import load_benchmark_specs, select_benchmarks  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "eval/config/benchmarks.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "eval/results"


def resolve_run_dir(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir():
        return candidate.resolve()
    tagged = DEFAULT_RESULTS_ROOT / value
    if tagged.is_dir():
        return tagged.resolve()
    raise SystemExit(f"cannot resolve run '{value}': neither a directory nor a tag under {DEFAULT_RESULTS_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", help="baseline run: results directory or tag name under eval/results/")
    parser.add_argument("run_b", help="candidate run: results directory or tag name under eval/results/")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="benchmark config YAML")
    parser.add_argument("--benchmarks", help="comma-separated benchmark keys to compare (default: all common)")
    parser.add_argument("--skip", help="comma-separated benchmark keys to skip")
    parser.add_argument("--num-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05, help="CI level is 1-alpha; sig marker threshold")
    parser.add_argument("--json", dest="json_path", help="also write the full report as JSON to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress per-benchmark progress messages")
    return parser.parse_args()


def _format_p(value: float, floor: float) -> str:
    if value <= floor:
        return f"<{floor:.4g}"
    return f"{value:.4f}" if value >= 0.0001 else f"{value:.1e}"


def format_report(report: ComparisonReport, alpha: float) -> str:
    name_a = Path(report.run_a).name
    name_b = Path(report.run_b).name
    p_floor = 2.0 / (report.n_bootstrap + 1)
    lines = [
        f"A = {name_a}  ({report.run_a})",
        f"B = {name_b}  ({report.run_b})",
        f"paired bootstrap: {report.n_bootstrap} replicates, seed={report.seed}, delta = B - A",
        "",
    ]
    header = (
        f"{'group':<12}{'benchmark':<22}{'n':>7}{'A':>8}{'B':>8}{'delta':>8}{'95% CI':>19}{'p':>10}{'q(BH)':>10}  sig"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for item in report.benchmarks:
        significant = item.q_value is not None and item.q_value < alpha
        lines.append(
            f"{item.group:<12}{item.key:<22}{item.n_units:>7}{item.score_a:>8.2f}{item.score_b:>8.2f}"
            f"{item.delta:>+8.2f}  [{item.ci_low:>+6.2f},{item.ci_high:>+6.2f}]"
            f"{_format_p(item.p_value, p_floor):>10}{_format_p(item.q_value, p_floor):>10}"
            f"  {'*' if significant else ''}"
        )
    lines.append("-" * len(header))
    aggregates = list(report.groups) + ([report.overall] if report.overall else [])
    for item in aggregates:
        significant = item.p_value < alpha
        lines.append(
            f"{'':<12}{item.name:<22}{len(item.members):>7}{item.score_a:>8.2f}{item.score_b:>8.2f}"
            f"{item.delta:>+8.2f}  [{item.ci_low:>+6.2f},{item.ci_high:>+6.2f}]"
            f"{_format_p(item.p_value, p_floor):>10}{'':>10}  {'*' if significant else ''}"
        )
    lines.append("")
    lines.append(f"n = paired bootstrap units; sig = q<{alpha} for benchmarks, p<{alpha} for averages.")
    lines.append("Averages weight member benchmarks equally, matching summary.csv group/overall averages.")

    mcnemar_lines = [
        f"  {item.key}: A-only-correct={item.mcnemar['a_only_correct']}, "
        f"B-only-correct={item.mcnemar['b_only_correct']}, z={item.mcnemar['z']:+.2f}, "
        f"p={item.mcnemar['p']:.2e}"
        for item in report.benchmarks
        if item.mcnemar is not None
    ]
    if mcnemar_lines:
        lines.append("")
        lines.append("McNemar (binary-accuracy benchmarks):")
        lines.extend(mcnemar_lines)

    warning_lines = [f"  {item.key}: {message}" for item in report.benchmarks for message in item.warnings]
    if warning_lines:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(warning_lines)
    if report.skipped:
        lines.append("")
        lines.append("Skipped benchmarks:")
        lines.extend(f"  {message}" for message in report.skipped)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dir_a = resolve_run_dir(args.run_a)
    dir_b = resolve_run_dir(args.run_b)
    if dir_a == dir_b:
        raise SystemExit("run_a and run_b resolve to the same directory")

    specs = select_benchmarks(
        load_benchmark_specs(Path(args.config)),
        include=args.benchmarks.split(",") if args.benchmarks else None,
        skip=args.skip.split(",") if args.skip else None,
    )
    progress = None if args.quiet else lambda message: print(message, file=sys.stderr, flush=True)
    report = compare_runs(
        dir_a,
        dir_b,
        specs,
        n_bootstrap=args.num_bootstrap,
        seed=args.seed,
        alpha=args.alpha,
        progress=progress,
    )
    print(format_report(report, args.alpha))
    if args.json_path:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report_to_dict(report), f, ensure_ascii=False, indent=2)
        print(f"\nJSON report written to {output_path}")


if __name__ == "__main__":
    main()
