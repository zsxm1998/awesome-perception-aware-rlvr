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
"""Paired significance comparison between two eval result directories.

Per-sample primary scores are recomputed from each run's stored
``predictions/<key>/predictions.jsonl`` with the same scorer helpers used at
eval time, so both runs are always judged by the current scoring code. The
only exception is MM-Vet, whose per-sample judge scores are read from the
persisted ``<key>_judgments.jsonl`` instead of re-querying the judge.

Statistics:
- Per benchmark: paired bootstrap over evaluation units (resampled within
  strata for benchmarks whose metric is not a plain per-sample mean, i.e.
  POPE macro-F1 and MME), percentile CI and add-one two-sided p-value.
  Binary-mean benchmarks additionally get a McNemar test.
- Per group / overall: the per-replicate benchmark deltas are averaged with
  equal benchmark weights, matching how summary.py builds "<Group> Avg" and
  "Overall Avg" from normalized scores.
- Benchmark-level p-values receive Benjamini-Hochberg q-values to account
  for reading many benchmarks at once.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import scorers as S
from .json_utils import read_jsonl
from .schemas import BenchmarkSpec


DEFAULT_BOOTSTRAP_SAMPLES = 10000
_BOOTSTRAP_CHUNK = 256

# aggregate(payload, idx_by_stratum) -> (B,) normalized 0-100 scores, where each
# idx matrix holds global unit positions resampled within one stratum.
Aggregate = Callable[[dict[str, np.ndarray], dict[int, np.ndarray]], np.ndarray]


@dataclass
class BenchmarkSamples:
    unit_ids: list[str]
    payload: dict[str, np.ndarray]
    aggregate: Aggregate
    strata: np.ndarray  # (n,) int stratum id per unit; mean-kind benchmarks use a single stratum
    binary: bool = False  # payload["value"] is 0/1 per unit -> McNemar applies

    def __post_init__(self) -> None:
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("duplicate unit ids in benchmark samples")

    @property
    def size(self) -> int:
        return len(self.unit_ids)

    def observed_score(self) -> float:
        identity = {stratum: np.flatnonzero(self.strata == stratum)[None, :] for stratum in np.unique(self.strata)}
        return float(self.aggregate(self.payload, identity)[0])

    def subset(self, positions: np.ndarray) -> "BenchmarkSamples":
        return BenchmarkSamples(
            unit_ids=[self.unit_ids[index] for index in positions],
            payload={key: value[positions] for key, value in self.payload.items()},
            aggregate=self.aggregate,
            strata=self.strata[positions],
            binary=self.binary,
        )


def _mean_aggregate(payload: dict[str, np.ndarray], idx_by_stratum: dict[int, np.ndarray]) -> np.ndarray:
    (idx,) = idx_by_stratum.values()
    return payload["value"][idx].mean(axis=1) * 100.0


def mean_samples(unit_ids: list[str], values: list[float]) -> BenchmarkSamples:
    array = np.asarray(values, dtype=np.float64)
    return BenchmarkSamples(
        unit_ids=unit_ids,
        payload={"value": array},
        aggregate=_mean_aggregate,
        strata=np.zeros(len(unit_ids), dtype=np.int64),
        binary=bool(np.isin(array, (0.0, 1.0)).all()),
    )


def _pope_aggregate(payload: dict[str, np.ndarray], idx_by_stratum: dict[int, np.ndarray]) -> np.ndarray:
    f1_by_stratum = []
    for _, idx in sorted(idx_by_stratum.items()):
        tp = payload["tp"][idx].sum(axis=1)
        fp = payload["fp"][idx].sum(axis=1)
        fn = payload["fn"][idx].sum(axis=1)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        denominator = precision + recall
        f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
        f1_by_stratum.append(f1)
    return np.mean(f1_by_stratum, axis=0) * 100.0


def _mme_aggregate(
    payload: dict[str, np.ndarray], idx_by_stratum: dict[int, np.ndarray], *, max_score: float
) -> np.ndarray:
    category_means = [payload["value"][idx].mean(axis=1) for _, idx in sorted(idx_by_stratum.items())]
    return np.sum(category_means, axis=0) / max_score * 100.0


def _cvbench_aggregate(
    payload: dict[str, np.ndarray], idx_by_stratum: dict[int, np.ndarray], *, sources: list[str]
) -> np.ndarray:
    """scorers.cvbench_accuracies on bootstrap draws: the mean of the 2D accuracy (mean of the ADE20K and COCO
    accuracies) and the 3D accuracy; ``sources[stratum]`` names the source of a stratum."""
    by_source = {sources[stratum]: payload["value"][idx].mean(axis=1) for stratum, idx in idx_by_stratum.items()}
    parts = []
    for group in (S.CVBENCH_SOURCES_2D, S.CVBENCH_SOURCES_3D):
        present = [by_source[source] for source in group if source in by_source]
        if present:
            parts.append(np.mean(present, axis=0))
    return np.mean(parts, axis=0) * 100.0


def _category_mean_aggregate(payload: dict[str, np.ndarray], idx_by_stratum: dict[int, np.ndarray]) -> np.ndarray:
    category_means = [payload["value"][idx].mean(axis=1) for _, idx in sorted(idx_by_stratum.items())]
    return np.mean(category_means, axis=0) * 100.0


def _strata_ids(names: list[str]) -> np.ndarray:
    order = {name: index for index, name in enumerate(sorted(set(names)))}
    return np.asarray([order[name] for name in names], dtype=np.int64)


# ---------------------------------------------------------------------------
# Per-scorer extraction of bootstrap units, reusing the scorer code itself.
# Keyed by BenchmarkSpec.scorer, so benchmarks can be added to or removed from
# benchmarks.yaml freely; a new entry is only needed for a brand-new scorer.
# ---------------------------------------------------------------------------


def _row_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Sample ids, disambiguated by occurrence index when a benchmark reuses ids.

    Predictions are written in dataset order, so the k-th occurrence of an id
    refers to the same underlying sample in both runs.
    """
    seen: dict[str, int] = {}
    ids = []
    for row in rows:
        sample_id = str(row["sample_id"])
        count = seen.get(sample_id, 0)
        seen[sample_id] = count + 1
        ids.append(sample_id if count == 0 else f"{sample_id}#dup{count}")
    return ids


def _gqa_uses_judge(results_dir: Path, key: str) -> bool:
    path = results_dir / "metrics" / f"{key}.json"
    if not path.exists():
        return False
    try:
        details = json.loads(path.read_text(encoding="utf-8"))["result"]["details"]
    except (KeyError, TypeError, ValueError):
        return False
    return details.get("scoring") == "exact_match_then_llm_judge"


def _extract_gqa(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    """Exact match, or the EM -> LLM-judge cascade when the run was scored with a judge.

    Cascade verdicts are read from the shared cache only: comparisons never issue judge
    requests, so a missing verdict is a hard error (mirroring the mm_vet extractor).
    """
    use_judge = _gqa_uses_judge(results_dir, spec.key)
    judge = {}
    if use_judge:
        metric = json.loads((results_dir / "metrics" / f"{spec.key}.json").read_text(encoding="utf-8"))["result"][
            "details"
        ]
        judge = {
            "provider": metric.get("judge_provider", "deepseek"),
            "model": metric.get("judge_model", S.DEFAULT_DEEPSEEK_JUDGE_MODEL),
        }
    cache = S._load_judge_cache(S.GQA_JUDGE_CACHE_PATH) if use_judge else {}
    values = []
    missing = 0
    missing_example = ""
    for row in rows:
        exact, truncated, question, gold, answer = S.gqa_row_scores(row)
        correct = exact
        if use_judge and not exact and not truncated and answer.strip():
            verdict = cache.get(S.gqa_judge_unit_key(question, gold, answer, **judge))
            if verdict is None:
                missing += 1
                missing_example = missing_example or f"{row.get('sample_id')}: {answer[:40]!r}"
            elif int(verdict.get("verdict", 0)) == 1:
                correct = True
        values.append(float(correct))
    if missing:
        raise ValueError(
            f"{spec.key}: {missing} exact-match-wrong answers have no cached judge verdict "
            f"(e.g. {missing_example}); re-score this run with the same judge first; "
            f"judge requests are not issued during comparisons"
        )
    return mean_samples(_row_ids(rows), values)


def _extract_seed_bench(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    return mean_samples(_row_ids(rows), [float(S.seed_bench_correct(row)) for row in rows])


def _extract_cfpo_match(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    values = []
    for row in rows:
        scores = S.cfpo_row_scores(row)
        values.append(sum(scores) / len(scores) if scores else 0.0)
    return mean_samples(_row_ids(rows), values)


def _extract_boxed_exact_match(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    values = []
    for row in rows:
        scores = S.boxed_row_scores(row, spec.key)
        values.append(sum(scores) / len(scores) if scores else 0.0)
    return mean_samples(_row_ids(rows), values)


def _extract_hallusionbench(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    return mean_samples(_row_ids(rows), [float(S.hallusionbench_row(row)) for row in rows])


def _extract_mcq(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    values = [S.mcq_row_score(row) for row in rows]
    mode = spec.metadata.get("aggregate", "micro")
    if mode not in {"category_mean", "cvbench"}:
        return mean_samples(_row_ids(rows), values)
    categories = [str((row.get("metadata") or {}).get("category") or "all") for row in rows]
    array = np.asarray(values, dtype=np.float64)
    if mode == "cvbench":
        S.cvbench_accuracies(values, categories)  # rejects unknown sources, as the scorer does
        aggregate = partial(_cvbench_aggregate, sources=sorted(set(categories)))
    else:
        aggregate = _category_mean_aggregate
    return BenchmarkSamples(
        unit_ids=_row_ids(rows),
        payload={"value": array},
        aggregate=aggregate,
        strata=_strata_ids(categories),
        binary=bool(np.isin(array, (0.0, 1.0)).all()),
    )


def _extract_mmmu(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    return mean_samples(_row_ids(rows), [S.mmmu_row_score(row) for row in rows])


def _extract_zoombench(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    return mean_samples(_row_ids(rows), [S.zoombench_row_score(row) for row in rows])


def _extract_refcoco(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    values = []
    for row in rows:
        best_iou, _ = S.refcoco_row_best_iou(row)
        values.append(float(best_iou >= 0.5))
    return mean_samples(_row_ids(rows), values)


def _extract_answer_bbox(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    """Primary of the GRIT sets: answer accuracy (relaxed exact match)."""
    values = []
    for row in rows:
        _, relaxed_correct = S._answer_correctness(spec, row, S.first_response(row))
        values.append(float(relaxed_correct))
    return mean_samples(_row_ids(rows), values)


def _extract_grounding_iou(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    """GRIT grounding IoU per sample (single-target benchmarks always have a GT box)."""
    values = []
    for row in rows:
        sample = S._score_answer_bbox_sample(row, S.first_response(row), answer_correct=True)
        values.append(sample.boxes.grounding_iou)
    return mean_samples(_row_ids(rows), values)


def _extract_mmvet(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    judgments_path = results_dir / "metrics" / f"{spec.key}_judgments.jsonl"
    if not judgments_path.exists():
        raise FileNotFoundError(
            f"{judgments_path} not found; run the eval scorer for {spec.key} first "
            "(judge scores cannot be recomputed offline)"
        )
    score_by_id = {str(item["sample_id"]): float(item["score"]) for item in read_jsonl(judgments_path)}
    ids = _row_ids(rows)
    missing = [sample_id for sample_id in ids if sample_id not in score_by_id]
    if missing:
        raise ValueError(f"{spec.key}: {len(missing)} predictions missing judge scores (e.g. {missing[0]})")
    return mean_samples(ids, [score_by_id[sample_id] for sample_id in ids])


def _extract_pope(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    tp, fp, fn, categories = [], [], [], []
    for row in rows:
        pred, gold, category, _ = S.pope_row(row)
        tp.append(float(pred == "yes" and gold == "yes"))
        fp.append(float(pred == "yes" and gold == "no"))
        fn.append(float(pred == "no" and gold == "yes"))
        categories.append(category)
    return BenchmarkSamples(
        unit_ids=_row_ids(rows),
        payload={
            "tp": np.asarray(tp, dtype=np.float64),
            "fp": np.asarray(fp, dtype=np.float64),
            "fn": np.asarray(fn, dtype=np.float64),
        },
        aggregate=_pope_aggregate,
        strata=_strata_ids(categories),
    )


def _extract_mme(spec: BenchmarkSpec, rows: list[dict[str, Any]], results_dir: Path) -> BenchmarkSamples:
    question_scores: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        category, question_id, correct, _ = S.mme_row(row)
        question_scores.setdefault((category, question_id), []).append(float(correct))
    keys = sorted(question_scores)
    return BenchmarkSamples(
        unit_ids=[f"{category}|{question_id}" for category, question_id in keys],
        payload={
            "value": np.asarray([S.mme_question_contribution(question_scores[key]) for key in keys], dtype=np.float64)
        },
        aggregate=partial(_mme_aggregate, max_score=S.mme_max_score(spec)),
        strata=_strata_ids([category for category, _ in keys]),
    )


SCORER_EXTRACTORS: dict[str, Callable[[BenchmarkSpec, list[dict[str, Any]], Path], BenchmarkSamples]] = {
    "boxed_exact_match": _extract_boxed_exact_match,
    "cfpo_match": _extract_cfpo_match,
    "hallusionbench": _extract_hallusionbench,
    "mcq": _extract_mcq,
    "mmmu": _extract_mmmu,
    "zoombench": _extract_zoombench,
    "gqa": _extract_gqa,
    "seed_bench": _extract_seed_bench,
    "refcoco": _extract_refcoco,
    "answer_bbox": _extract_answer_bbox,
    "grounding_iou": _extract_grounding_iou,
    "mmvet": _extract_mmvet,
    "pope": _extract_pope,
    "mme": _extract_mme,
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def align_samples(a: BenchmarkSamples, b: BenchmarkSamples) -> tuple[BenchmarkSamples, BenchmarkSamples, int, int]:
    """Restrict both sides to their common unit ids (order taken from side A)."""
    ids_b = set(b.unit_ids)
    common = [unit_id for unit_id in a.unit_ids if unit_id in ids_b]
    dropped_a = a.size - len(common)
    dropped_b = b.size - len(common)
    if not common:
        raise ValueError("no common evaluation units between the two runs")
    if dropped_a == 0 and a.unit_ids == b.unit_ids:
        return a, b, 0, 0
    position_a = {unit_id: index for index, unit_id in enumerate(a.unit_ids)}
    position_b = {unit_id: index for index, unit_id in enumerate(b.unit_ids)}
    idx_a = np.asarray([position_a[unit_id] for unit_id in common], dtype=np.int64)
    idx_b = np.asarray([position_b[unit_id] for unit_id in common], dtype=np.int64)
    return a.subset(idx_a), b.subset(idx_b), dropped_a, dropped_b


def paired_bootstrap_deltas(
    a: BenchmarkSamples,
    b: BenchmarkSamples,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-replicate normalized-score deltas (B - A) with shared resampling."""
    if a.unit_ids != b.unit_ids:
        raise ValueError("samples must be aligned before bootstrapping")
    if not np.array_equal(a.strata, b.strata):
        raise ValueError("stratum assignment differs between the two runs")
    positions = {int(stratum): np.flatnonzero(a.strata == stratum) for stratum in np.unique(a.strata)}
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    done = 0
    while done < n_bootstrap:
        chunk = min(_BOOTSTRAP_CHUNK, n_bootstrap - done)
        idx_by_stratum = {
            stratum: pos[rng.integers(0, pos.size, size=(chunk, pos.size))] for stratum, pos in positions.items()
        }
        deltas[done : done + chunk] = a.aggregate(b.payload, idx_by_stratum) - a.aggregate(a.payload, idx_by_stratum)
        done += chunk
    return deltas


def bootstrap_summary(deltas: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """(ci_low, ci_high, two-sided add-one p) from delta replicates."""
    ci_low, ci_high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    n = deltas.size
    p_low = (np.count_nonzero(deltas <= 0) + 1) / (n + 1)
    p_high = (np.count_nonzero(deltas >= 0) + 1) / (n + 1)
    return float(ci_low), float(ci_high), min(1.0, 2.0 * min(p_low, p_high))


def mcnemar_test(a_values: np.ndarray, b_values: np.ndarray) -> dict[str, float]:
    only_a = int(np.count_nonzero((a_values == 1.0) & (b_values == 0.0)))
    only_b = int(np.count_nonzero((a_values == 0.0) & (b_values == 1.0)))
    discordant = only_a + only_b
    if discordant == 0:
        return {"a_only_correct": 0, "b_only_correct": 0, "z": 0.0, "p": 1.0}
    z = (only_b - only_a) / math.sqrt(discordant)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"a_only_correct": only_a, "b_only_correct": only_b, "z": z, "p": p}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda index: p_values[index])
    q_values = [0.0] * n
    running_min = 1.0
    for rank_from_end, index in enumerate(reversed(order)):
        rank = n - rank_from_end
        running_min = min(running_min, p_values[index] * n / rank)
        q_values[index] = running_min
    return q_values


# ---------------------------------------------------------------------------
# Run-level comparison
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkComparison:
    key: str
    group: str
    n_units: int
    dropped_a: int
    dropped_b: int
    score_a: float
    score_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float | None = None
    mcnemar: dict[str, float] | None = None
    stored_score_a: float | None = None
    stored_score_b: float | None = None
    warnings: list[str] = field(default_factory=list)
    delta_replicates: np.ndarray | None = None


@dataclass
class AggregateComparison:
    name: str
    members: list[str]
    score_a: float
    score_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass
class ComparisonReport:
    run_a: str
    run_b: str
    n_bootstrap: int
    seed: int
    benchmarks: list[BenchmarkComparison]
    groups: list[AggregateComparison]
    overall: AggregateComparison | None
    skipped: list[str]


def _stored_normalized_score(results_dir: Path, key: str) -> float | None:
    path = results_dir / "metrics" / f"{key}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return float(json.load(f)["result"]["normalized_score_0_100"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _predictions_path(results_dir: Path, key: str) -> Path:
    return results_dir / "predictions" / key / "predictions.jsonl"


def _scored_rows(results_dir: Path, key: str) -> list[dict[str, Any]]:
    """The predictions of a benchmark, read with the answer protocol its stored score used (--answer-protocol of
    the scoring run, which can differ from the one recorded with the predictions)."""
    rows = read_jsonl(_predictions_path(results_dir, key))
    path = results_dir / "metrics" / f"{key}.json"
    if not path.exists():
        return rows
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))["result"].get("metadata") or {}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return rows
    return S.set_answer_protocol(rows, str(metadata.get("answer_protocol") or "default"))


def usable_specs(
    specs: list[BenchmarkSpec],
    dir_a: Path,
    dir_b: Path,
) -> tuple[list[BenchmarkSpec], list[str]]:
    """Benchmarks evaluated in both runs and supported by a known scorer."""
    usable: list[BenchmarkSpec] = []
    skipped: list[str] = []
    for spec in specs:
        missing = [str(d) for d in (dir_a, dir_b) if not _predictions_path(d, spec.key).exists()]
        if missing:
            skipped.append(f"{spec.key}: no predictions in {', '.join(missing)}")
            continue
        if spec.scorer not in SCORER_EXTRACTORS:
            skipped.append(f"{spec.key}: scorer '{spec.scorer}' has no comparison extractor")
            continue
        usable.append(spec)
    for results_dir in (dir_a, dir_b):
        known = {spec.key for spec in specs}
        predictions_root = results_dir / "predictions"
        if predictions_root.is_dir():
            for child in sorted(predictions_root.iterdir()):
                if child.is_dir() and child.name not in known and (child / "predictions.jsonl").exists():
                    skipped.append(f"{child.name}: present in {results_dir} but not in the benchmark config")
    return usable, sorted(set(skipped))


def compare_benchmark(
    spec: BenchmarkSpec,
    dir_a: Path,
    dir_b: Path,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> BenchmarkComparison:
    extractor = SCORER_EXTRACTORS[spec.scorer]
    samples_a = extractor(spec, _scored_rows(dir_a, spec.key), dir_a)
    samples_b = extractor(spec, _scored_rows(dir_b, spec.key), dir_b)
    warnings: list[str] = []

    for side, results_dir, samples in (("A", dir_a, samples_a), ("B", dir_b, samples_b)):
        stored = _stored_normalized_score(results_dir, spec.key)
        recomputed = samples.observed_score()
        if stored is not None and abs(stored - recomputed) > 0.05:
            warnings.append(
                f"side {side}: recomputed score {recomputed:.2f} differs from stored metrics json "
                f"{stored:.2f} (scoring code may have changed since that run was scored)"
            )

    aligned_a, aligned_b, dropped_a, dropped_b = align_samples(samples_a, samples_b)
    if dropped_a or dropped_b:
        warnings.append(f"non-overlapping units dropped: {dropped_a} from A, {dropped_b} from B")

    deltas = paired_bootstrap_deltas(aligned_a, aligned_b, n_bootstrap=n_bootstrap, rng=rng)
    score_a = aligned_a.observed_score()
    score_b = aligned_b.observed_score()
    ci_low, ci_high, p_value = bootstrap_summary(deltas, alpha=alpha)
    mcnemar = (
        mcnemar_test(aligned_a.payload["value"], aligned_b.payload["value"])
        if aligned_a.binary and aligned_b.binary
        else None
    )
    return BenchmarkComparison(
        key=spec.key,
        group=spec.group,
        n_units=aligned_a.size,
        dropped_a=dropped_a,
        dropped_b=dropped_b,
        score_a=score_a,
        score_b=score_b,
        delta=score_b - score_a,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        mcnemar=mcnemar,
        stored_score_a=_stored_normalized_score(dir_a, spec.key),
        stored_score_b=_stored_normalized_score(dir_b, spec.key),
        warnings=warnings,
        delta_replicates=deltas,
    )


def _aggregate_comparison(name: str, members: list[BenchmarkComparison], alpha: float) -> AggregateComparison:
    delta_matrix = np.stack([item.delta_replicates for item in members])
    pooled = delta_matrix.mean(axis=0)
    ci_low, ci_high, p_value = bootstrap_summary(pooled, alpha=alpha)
    return AggregateComparison(
        name=name,
        members=[item.key for item in members],
        score_a=float(np.mean([item.score_a for item in members])),
        score_b=float(np.mean([item.score_b for item in members])),
        delta=float(np.mean([item.delta for item in members])),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
    )


def compare_runs(
    dir_a: Path,
    dir_b: Path,
    specs: list[BenchmarkSpec],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> ComparisonReport:
    selected, skipped = usable_specs(specs, dir_a, dir_b)
    rng = np.random.default_rng(seed)
    comparisons: list[BenchmarkComparison] = []
    for spec in selected:
        if progress is not None:
            progress(f"comparing {spec.key} ...")
        comparisons.append(compare_benchmark(spec, dir_a, dir_b, n_bootstrap=n_bootstrap, rng=rng, alpha=alpha))

    q_values = benjamini_hochberg([item.p_value for item in comparisons])
    for item, q_value in zip(comparisons, q_values):
        item.q_value = q_value

    groups: list[AggregateComparison] = []
    seen_groups: list[str] = []
    for item in comparisons:
        if item.group not in seen_groups:
            seen_groups.append(item.group)
    for group in seen_groups:
        members = [item for item in comparisons if item.group == group]
        groups.append(_aggregate_comparison(f"{group} Avg", members, alpha))
    overall = _aggregate_comparison("Overall Avg", comparisons, alpha) if comparisons else None

    return ComparisonReport(
        run_a=str(dir_a),
        run_b=str(dir_b),
        n_bootstrap=n_bootstrap,
        seed=seed,
        benchmarks=comparisons,
        groups=groups,
        overall=overall,
        skipped=skipped,
    )


def report_to_dict(report: ComparisonReport) -> dict[str, Any]:
    def benchmark_dict(item: BenchmarkComparison) -> dict[str, Any]:
        data = {
            "key": item.key,
            "group": item.group,
            "n_units": item.n_units,
            "dropped_a": item.dropped_a,
            "dropped_b": item.dropped_b,
            "score_a": item.score_a,
            "score_b": item.score_b,
            "delta": item.delta,
            "ci_low": item.ci_low,
            "ci_high": item.ci_high,
            "p_value": item.p_value,
            "q_value": item.q_value,
            "stored_score_a": item.stored_score_a,
            "stored_score_b": item.stored_score_b,
            "warnings": item.warnings,
        }
        if item.mcnemar is not None:
            data["mcnemar"] = item.mcnemar
        return data

    def aggregate_dict(item: AggregateComparison) -> dict[str, Any]:
        return {
            "name": item.name,
            "members": item.members,
            "score_a": item.score_a,
            "score_b": item.score_b,
            "delta": item.delta,
            "ci_low": item.ci_low,
            "ci_high": item.ci_high,
            "p_value": item.p_value,
        }

    return {
        "run_a": report.run_a,
        "run_b": report.run_b,
        "n_bootstrap": report.n_bootstrap,
        "seed": report.seed,
        "benchmarks": [benchmark_dict(item) for item in report.benchmarks],
        "groups": [aggregate_dict(item) for item in report.groups],
        "overall": aggregate_dict(report.overall) if report.overall else None,
        "skipped": report.skipped,
    }
