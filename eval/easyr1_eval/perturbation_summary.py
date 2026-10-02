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
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


FIELDNAMES = [
    "model",
    "backend",
    "row_type",
    "group",
    "benchmark",
    "perturbation_type",
    "perturbation_summary",
    "perturbation_params",
    "seeds",
    "num_runs",
    "score_mean",
    "score_std",
    "score_p10",
    "score_p50",
    "score_p90",
    "delta_vs_clean_mean",
    "delta_baseline_summary",
    "delta_missing_reason",
]


def write_perturbation_aggregate_summary(run_dirs: list[Path], output_dir: Path) -> Path | None:
    rows = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            continue
        with summary_path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    benchmark_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("row_type") in {"benchmark", "group_average", "overall"}
    ]
    if not benchmark_rows:
        return None

    has_perturbed_run = any(
        str(row.get("perturbation_type", "none") or "none").strip().lower() not in {"", "none", "clean"}
        for row in benchmark_rows
    )
    if not has_perturbed_run:
        return None

    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in benchmark_rows:
        key = (
            row.get("model", ""),
            row.get("backend", ""),
            row.get("row_type", ""),
            row.get("group", ""),
            row.get("benchmark", ""),
            row.get("perturbation_type", "none") or "none",
            row.get("perturbation_summary", "clean") or "clean",
            _canonical_params(row.get("perturbation_params", "")),
        )
        grouped.setdefault(key, []).append(row)

    clean_by_target = {}
    for key, items in grouped.items():
        model, backend, row_type, group, benchmark, perturb_type, perturb_summary, _params = key
        if perturb_type in {"", "none"} or perturb_summary == "clean":
            clean_by_target[(model, backend, row_type, group, benchmark)] = _score_stats(items)["mean"]

    output_rows = []
    for key, items in sorted(grouped.items()):
        model, backend, row_type, group, benchmark, perturb_type, perturb_summary, params = key
        stats = _score_stats(items)
        clean_key = (model, backend, row_type, group, benchmark)
        clean_score = clean_by_target.get(clean_key)
        if clean_score is None:
            delta = ""
            baseline = ""
            reason = "missing_same_backend_clean"
        else:
            delta = stats["mean"] - clean_score
            baseline = "clean"
            reason = ""
        output_rows.append(
            {
                "model": model,
                "backend": backend,
                "row_type": row_type,
                "group": group,
                "benchmark": benchmark,
                "perturbation_type": perturb_type,
                "perturbation_summary": perturb_summary,
                "perturbation_params": params,
                "seeds": ",".join(
                    sorted(
                        {str(row.get("perturbation_seed", "")) for row in items if row.get("perturbation_seed", "")}
                    )
                ),
                "num_runs": len(items),
                "score_mean": stats["mean"],
                "score_std": stats["std"],
                "score_p10": stats["p10"],
                "score_p50": stats["p50"],
                "score_p90": stats["p90"],
                "delta_vs_clean_mean": delta,
                "delta_baseline_summary": baseline,
                "delta_missing_reason": reason,
            }
        )

    path = output_dir / "perturbation_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)
    return path


def _score_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    scores = sorted(float(row["normalized_score_0_100"]) for row in rows)
    return {
        "mean": mean(scores),
        "std": pstdev(scores) if len(scores) > 1 else 0.0,
        "p10": _percentile(scores, 10),
        "p50": _percentile(scores, 50),
        "p90": _percentile(scores, 90),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _canonical_params(value: str) -> str:
    if not value:
        return "{}"
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return str(value)
    try:
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)
