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

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.compare import (  # noqa: E402
    SCORER_EXTRACTORS,
    align_samples,
    benjamini_hochberg,
    bootstrap_summary,
    compare_runs,
    mcnemar_test,
    mean_samples,
    paired_bootstrap_deltas,
    report_to_dict,
)
from easyr1_eval.schemas import BenchmarkSpec  # noqa: E402
from easyr1_eval.scorers import (  # noqa: E402
    score_answer_bbox,
    score_boxed_exact_match,
    score_gqa,
    score_grounding_iou,
    score_hallusionbench,
    score_mcq,
    score_mme,
    score_pope,
    score_refcoco,
    score_seed_bench,
)


def _spec(key, scorer, primary="score", group="Test", metadata=None):
    return BenchmarkSpec(
        key=key,
        label=key,
        group=group,
        loader="dummy",
        scorer=scorer,
        primary_metric=primary,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Extractor parity: the comparison tool must reproduce each scorer's
# normalized primary score exactly when given the full sample set.
# ---------------------------------------------------------------------------


GQA_ROWS = [
    # Every row must be decidable by exact match alone: the compare extractor is
    # cache-only for the judge cascade, and synthetic questions have no cached
    # verdicts. Wrong-but-empty answers skip the judge lookup by design.
    {"sample_id": "g1", "target": "no", "responses": ["<answer>no</answer>"]},
    {"sample_id": "g2", "target": "yes", "responses": ["<answer></answer>"]},
    {"sample_id": "g3", "target": "left", "responses": ["<answer>left</answer>"]},
]

SEED_ROWS = [
    {"sample_id": "s1", "target": "A", "responses": ["<answer>A</answer>"]},
    {"sample_id": "s2", "target": "B", "responses": ["<answer>C</answer>"]},
]

BOXED_ROWS = [
    {"sample_id": "b1", "target": "3", "responses": [r"\boxed{3}", r"\boxed{4}"]},
    {"sample_id": "b2", "target": "7.5", "responses": [r"\boxed{7.5}", "<answer>7.5</answer>"]},
    {"sample_id": "b3", "target": "B", "responses": ["no box", r"\boxed{C}"]},
]

HALLUSION_ROWS = [
    {
        "sample_id": "h1",
        "target": "yes",
        "responses": ["yes"],
        "metadata": {"category": "VD", "subcategory": "a", "set_id": "0", "figure_id": "1", "question_id": "0"},
    },
    {
        "sample_id": "h2",
        "target": "no",
        "responses": ["yes"],
        "metadata": {"category": "VD", "subcategory": "a", "set_id": "0", "figure_id": "1", "question_id": "1"},
    },
    {
        "sample_id": "h3",
        "target": "no",
        "responses": [r"\boxed{no}"],
        "metadata": {"category": "VS", "subcategory": "b", "set_id": "1", "figure_id": "0", "question_id": "0"},
    },
]

MCQ_ROWS = [
    {
        "sample_id": "c1",
        "target": "A",
        "responses": ["A"],
        "extra_info": {"options": ["x", "y"]},
        "metadata": {"category": "single"},
    },
    {
        "sample_id": "c2",
        "target": "B",
        "responses": ["A"],
        "extra_info": {"options": ["x", "y"]},
        "metadata": {"category": "single"},
    },
    {
        "sample_id": "c3",
        "target": "B",
        "responses": ["<answer>y</answer>"],
        "extra_info": {"options": ["x", "y"]},
        "metadata": {"category": "cross"},
    },
]

REFCOCO_ROWS = [
    {
        "sample_id": "r1",
        "target": "box",
        "responses": ["<answer>[100, 100, 500, 500]</answer>"],
        "extra_info": {"bbox": [0.1, 0.1, 0.5, 0.5]},
    },
    {
        "sample_id": "r2",
        "target": "box",
        "responses": ["<answer>[0, 0, 100, 100]</answer>"],
        "extra_info": {"bbox": [0.5, 0.5, 1.0, 1.0]},
    },
]

POPE_ROWS = [
    {"sample_id": "p1", "target": "yes", "responses": ["Yes, it is."], "metadata": {"category": "random"}},
    {"sample_id": "p2", "target": "no", "responses": ["yes"], "metadata": {"category": "random"}},
    {"sample_id": "p3", "target": "no", "responses": ["no"], "metadata": {"category": "popular"}},
    {"sample_id": "p4", "target": "yes", "responses": ["maybe"], "metadata": {"category": "popular"}},
    {"sample_id": "p5", "target": "yes", "responses": ["yes"], "metadata": {"category": "popular"}},
]

MME_ROWS = [
    {"sample_id": "q1a", "target": "yes", "responses": ["yes"], "metadata": {"category": "art", "question_id": "q1"}},
    {"sample_id": "q1b", "target": "no", "responses": ["no"], "metadata": {"category": "art", "question_id": "q1"}},
    {"sample_id": "q2a", "target": "yes", "responses": ["yes"], "metadata": {"category": "art", "question_id": "q2"}},
    {"sample_id": "q2b", "target": "no", "responses": ["yes"], "metadata": {"category": "art", "question_id": "q2"}},
    {"sample_id": "q3a", "target": "yes", "responses": ["no"], "metadata": {"category": "ocr", "question_id": "q3"}},
    {"sample_id": "q3b", "target": "no", "responses": ["yes"], "metadata": {"category": "ocr", "question_id": "q3"}},
]

ANSWER_BBOX_ROWS = [
    {
        "sample_id": "two-boxes",
        "target": "2",
        "responses": ["<bbox>[[0, 0, 500, 500], [500, 500, 1000, 1000]]</bbox><answer>two</answer>"],
        "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]},
    },
    {
        "sample_id": "wrong-answer",
        "target": "yes",
        "responses": ['<region name="object">[[0, 0, 1000, 1000]]</region><answer>no</answer>'],
        "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
    },
    {
        "sample_id": "true-negative",
        "target": "0",
        "responses": ["<answer>zero</answer>"],
        "extra_info": {"bboxs_normalized": []},
    },
]

GROUNDING_IOU_ROWS = [
    {
        "sample_id": "gi1",
        "target": "x",
        "responses": ["<bbox>[[0, 0, 500, 500]]</bbox><answer>x</answer>"],
        "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
    },
    {
        "sample_id": "gi2",
        "target": "x",
        "responses": ["<bbox>[[0, 0, 1000, 1000]]</bbox><answer>x</answer>"],
        "extra_info": {"bboxs_normalized": [[0, 0, 1, 1]]},
    },
]


@pytest.mark.parametrize(
    ("scorer", "score_fn", "rows", "metadata"),
    [
        ("gqa", score_gqa, GQA_ROWS, {}),
        ("seed_bench", score_seed_bench, SEED_ROWS, {}),
        ("boxed_exact_match", score_boxed_exact_match, BOXED_ROWS, {}),
        ("hallusionbench", score_hallusionbench, HALLUSION_ROWS, {}),
        ("mcq", score_mcq, MCQ_ROWS, {"aggregate": "micro"}),
        ("mcq", score_mcq, MCQ_ROWS, {"aggregate": "category_mean"}),
        ("refcoco", score_refcoco, REFCOCO_ROWS, {}),
        ("pope", score_pope, POPE_ROWS, {}),
        ("mme", score_mme, MME_ROWS, {}),
        ("answer_bbox", score_answer_bbox, ANSWER_BBOX_ROWS, {}),
        ("grounding_iou", score_grounding_iou, GROUNDING_IOU_ROWS, {}),
    ],
)
def test_extractor_matches_scorer_normalized_score(tmp_path, scorer, score_fn, rows, metadata):
    spec = _spec("bench", scorer, metadata=metadata)
    result = score_fn(spec, rows, None, tmp_path)
    samples = SCORER_EXTRACTORS[scorer](spec, rows, tmp_path)
    assert samples.observed_score() == pytest.approx(result.normalized_score_0_100, abs=1e-9)


def test_agentic_answer_bbox_extractor_matches_scorer(tmp_path):
    spec = _spec("grit_gqa", "answer_bbox", primary="answer_accuracy")
    rows = [
        {
            "sample_id": "one",
            "target": "cat",
            "responses": ["<bbox>[[600, 600, 1000, 1000]]</bbox><answer>cat</answer>"],
            "extra_info": {"bboxs_normalized": [[0, 0, 0.5, 0.5]]},
            "image_refs": ["x.jpg"],
            "eval_metadata": {
                "interaction_mode": "agentic",
                "agent_output_contract": "native",
            },
            "agent_diagnostics": [
                {
                    "status": "answered",
                    "committed_tool_regions": [
                        {
                            "turn_index": 0,
                            "image_idx": 0,
                            "bbox_2d": [0, 0, 500, 500],
                        }
                    ],
                }
            ],
        }
    ]

    result = score_answer_bbox(spec, rows, None, tmp_path)
    samples = SCORER_EXTRACTORS["answer_bbox"](spec, rows, tmp_path)

    assert result.normalized_score_0_100 == pytest.approx(100.0)
    assert samples.observed_score() == pytest.approx(100.0)


def test_gqa_extractor_raises_on_missing_cached_verdict(tmp_path):
    """Comparisons are cache-only: an exact-match-wrong answer without a cached
    judge verdict must be a hard error, never a silent judge request."""
    rows = [
        {
            "sample_id": "g1",
            "prompt": "synthetic compare-test question never sent to the judge?",
            "target": "yes",
            "responses": ["<answer>no</answer>"],
        }
    ]
    # Runs scored without a judge are compared on exact match.
    assert SCORER_EXTRACTORS["gqa"](_spec("bench", "gqa"), rows, tmp_path).observed_score() == pytest.approx(0.0)

    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "bench.json").write_text(
        json.dumps(
            {
                "result": {
                    "details": {
                        "scoring": "exact_match_then_llm_judge",
                        "judge_provider": "openai",
                        "judge_model": "synthetic-judge-never-cached",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no cached judge verdict"):
        SCORER_EXTRACTORS["gqa"](_spec("bench", "gqa"), rows, tmp_path)


def test_mmvet_extractor_reads_persisted_judgments(tmp_path):
    rows = [
        {"sample_id": "v1_0", "target": "x", "responses": ["a"]},
        {"sample_id": "v1_1", "target": "y", "responses": ["b"]},
    ]
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    judgments = [{"sample_id": "v1_0", "score": 1.0}, {"sample_id": "v1_1", "score": 0.4}]
    with (metrics_dir / "mm_vet_judgments.jsonl").open("w", encoding="utf-8") as f:
        for item in judgments:
            f.write(json.dumps(item) + "\n")

    samples = SCORER_EXTRACTORS["mmvet"](_spec("mm_vet", "mmvet"), rows, tmp_path)

    assert samples.observed_score() == pytest.approx(70.0)


def test_mmvet_extractor_requires_judgments_file(tmp_path):
    (tmp_path / "metrics").mkdir()
    with pytest.raises(FileNotFoundError):
        SCORER_EXTRACTORS["mmvet"](_spec("mm_vet", "mmvet"), [{"sample_id": "a", "responses": ["x"]}], tmp_path)


# ---------------------------------------------------------------------------
# Statistical machinery
# ---------------------------------------------------------------------------


def test_mean_samples_flags_binary_values():
    assert mean_samples(["a", "b"], [1.0, 0.0]).binary
    assert not mean_samples(["a", "b"], [0.5, 0.0]).binary


def test_align_samples_intersects_and_reports_drops():
    a = mean_samples(["u1", "u2", "u3"], [1.0, 0.0, 1.0])
    b = mean_samples(["u2", "u3", "u4"], [1.0, 1.0, 0.0])
    aligned_a, aligned_b, dropped_a, dropped_b = align_samples(a, b)
    assert aligned_a.unit_ids == ["u2", "u3"]
    assert aligned_b.unit_ids == ["u2", "u3"]
    assert (dropped_a, dropped_b) == (1, 1)
    assert aligned_a.payload["value"].tolist() == [0.0, 1.0]
    assert aligned_b.payload["value"].tolist() == [1.0, 1.0]


def test_paired_bootstrap_is_deterministic_and_detects_constant_shift():
    ids = [f"u{i}" for i in range(50)]
    base = [float(i % 2) for i in range(50)]
    a = mean_samples(ids, base)
    b = mean_samples(ids, [value + 0.2 for value in base])
    deltas_1 = paired_bootstrap_deltas(a, b, n_bootstrap=200, rng=np.random.default_rng(7))
    deltas_2 = paired_bootstrap_deltas(a, b, n_bootstrap=200, rng=np.random.default_rng(7))
    assert np.array_equal(deltas_1, deltas_2)
    # constant per-unit shift -> every replicate sees exactly +20 points
    assert deltas_1 == pytest.approx(np.full(200, 20.0))
    ci_low, ci_high, p_value = bootstrap_summary(deltas_1)
    assert (ci_low, ci_high) == pytest.approx((20.0, 20.0))
    assert p_value == pytest.approx(2.0 / 201.0)


def test_paired_bootstrap_identical_runs_gives_null_delta():
    ids = [f"u{i}" for i in range(30)]
    values = [float(i % 3 == 0) for i in range(30)]
    a = mean_samples(ids, values)
    b = mean_samples(ids, list(values))
    deltas = paired_bootstrap_deltas(a, b, n_bootstrap=100, rng=np.random.default_rng(0))
    assert deltas == pytest.approx(np.zeros(100))
    _, _, p_value = bootstrap_summary(deltas)
    assert p_value == 1.0


def test_bootstrap_summary_two_sided_p():
    deltas = np.asarray([1.0] * 90 + [-1.0] * 10, dtype=np.float64)
    _, _, p_value = bootstrap_summary(deltas)
    assert p_value == pytest.approx(2.0 * 11.0 / 101.0)


def test_mcnemar_counts_and_two_sided_p():
    a = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0])
    b = np.asarray([1.0, 0.0, 1.0, 1.0, 1.0])
    result = mcnemar_test(a, b)
    assert result["a_only_correct"] == 1
    assert result["b_only_correct"] == 2
    assert result["z"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert 0.0 < result["p"] < 1.0
    assert mcnemar_test(a, a)["p"] == 1.0


def test_benjamini_hochberg_known_example():
    q_values = benjamini_hochberg([0.005, 0.01, 0.03, 0.04])
    assert q_values == pytest.approx([0.02, 0.02, 0.04, 0.04])
    assert benjamini_hochberg([]) == []


# ---------------------------------------------------------------------------
# End-to-end over fabricated result directories
# ---------------------------------------------------------------------------


def _write_predictions(results_dir: Path, key: str, rows: list[dict]) -> None:
    bench_dir = results_dir / "predictions" / key
    bench_dir.mkdir(parents=True)
    with (bench_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _binary_rows(correct_flags: list[bool]) -> list[dict]:
    # seed_bench-style rows: plumbing tests need a judge-free scorer so the
    # synthetic data stays hermetic (gqa's extractor demands cached verdicts).
    return [
        {"sample_id": f"s{i}", "target": "A", "responses": ["<answer>A</answer>" if flag else "<answer>B</answer>"]}
        for i, flag in enumerate(correct_flags)
    ]


def test_compare_runs_end_to_end(tmp_path):
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    specs = [
        _spec("bench_g1", "seed_bench", group="Grounding"),
        _spec("bench_g2", "seed_bench", group="Grounding"),
        _spec("bench_r1", "seed_bench", group="Reasoning"),
        _spec("bench_missing", "seed_bench", group="Reasoning"),
    ]
    n = 40
    _write_predictions(dir_a, "bench_g1", _binary_rows([i % 4 == 0 for i in range(n)]))
    _write_predictions(dir_b, "bench_g1", _binary_rows([i % 2 == 0 for i in range(n)]))
    _write_predictions(dir_a, "bench_g2", _binary_rows([i % 2 == 0 for i in range(n)]))
    _write_predictions(dir_b, "bench_g2", _binary_rows([i % 2 == 0 for i in range(n)]))
    _write_predictions(dir_a, "bench_r1", _binary_rows([True] * n))
    _write_predictions(dir_b, "bench_r1", _binary_rows([False] * n))
    _write_predictions(dir_a, "bench_missing", _binary_rows([True] * n))  # absent from run B

    report = compare_runs(dir_a, dir_b, specs, n_bootstrap=400, seed=3)

    assert [item.key for item in report.benchmarks] == ["bench_g1", "bench_g2", "bench_r1"]
    assert any("bench_missing" in message for message in report.skipped)

    by_key = {item.key: item for item in report.benchmarks}
    assert by_key["bench_g1"].delta == pytest.approx(25.0)
    assert by_key["bench_g2"].delta == pytest.approx(0.0)
    assert by_key["bench_r1"].delta == pytest.approx(-100.0)
    assert all(item.q_value is not None for item in report.benchmarks)
    assert by_key["bench_g2"].p_value == 1.0
    assert by_key["bench_r1"].p_value < 0.01
    assert by_key["bench_g1"].mcnemar is not None

    groups = {item.name: item for item in report.groups}
    assert set(groups) == {"Grounding Avg", "Reasoning Avg"}
    assert groups["Grounding Avg"].delta == pytest.approx((25.0 + 0.0) / 2)
    assert groups["Reasoning Avg"].delta == pytest.approx(-100.0)
    assert report.overall.delta == pytest.approx((25.0 + 0.0 - 100.0) / 3)
    assert report.overall.members == ["bench_g1", "bench_g2", "bench_r1"]

    payload = report_to_dict(report)
    assert payload["overall"]["delta"] == pytest.approx(report.overall.delta)
    assert len(payload["benchmarks"]) == 3
    json.dumps(payload)  # must be JSON-serializable


def test_compare_runs_warns_on_stored_score_drift(tmp_path):
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    rows = _binary_rows([True, False, True, False])
    _write_predictions(dir_a, "bench", rows)
    _write_predictions(dir_b, "bench", rows)
    metrics_dir = dir_a / "metrics"
    metrics_dir.mkdir()
    with (metrics_dir / "bench.json").open("w", encoding="utf-8") as f:
        json.dump({"result": {"normalized_score_0_100": 99.0}}, f)

    report = compare_runs(dir_a, dir_b, [_spec("bench", "seed_bench")], n_bootstrap=50, seed=0)

    (item,) = report.benchmarks
    assert item.stored_score_a == pytest.approx(99.0)
    assert any("differs from stored" in message for message in item.warnings)
