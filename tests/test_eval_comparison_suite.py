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
"""The grouped comparison suite: suite groups, their summary means, CV-Bench and MME perception."""

import csv
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.loaders import load_samples  # noqa: E402
from easyr1_eval.registry import load_benchmark_specs  # noqa: E402
from easyr1_eval.runner import all_suite_groups, format_results_table  # noqa: E402
from easyr1_eval.schemas import MetricResult  # noqa: E402
from easyr1_eval.scorers import cvbench_accuracies, mcq_aggregate, score_mcq, score_mme  # noqa: E402
from easyr1_eval.suites import load_suites, suite_groupings  # noqa: E402
from easyr1_eval.summary import suite_group_scores, update_global_summary_csv, write_summary_csv  # noqa: E402

from eval.prepare import cli, lmms_lab  # noqa: E402
from eval.prepare.common import PrepareContext, is_prepared  # noqa: E402


MATH = ["geo3k", "mathvista", "wemath", "mmk12", "mathverse", "mathvision", "dynamath"]
VISION = ["mathverse_v", "mmmu_pro", "logicvista", "clevr_count", "ai2d", "mme_cognition"]
PERCEPTION = ["pope", "hallusionbench", "mmstar", "blink", "mme_perception", "cvbench"]
MME_PERCEPTION = ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork"]
MME_PERCEPTION += ["OCR"]
MME_COGNITION = ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]


def _specs():
    return load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml")


def _spec(key):
    return next(spec for spec in _specs() if spec.key == key)


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(0, 128, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_suites(tmp_path, text):
    path = tmp_path / "suites.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _result(key, score, status="ok", n=10):
    return MetricResult(key, "Test", "acc", score, score, n, status=status)


def test_comparison_suite_has_three_groups_and_opd_shares_them():
    suites = load_suites(ROOT / "eval/config/suites.yaml", _specs())
    comparison = suites["comparison"]
    assert comparison.benchmarks == tuple(MATH + VISION + PERCEPTION)
    assert comparison.groups == (
        ("Math reasoning", tuple(MATH)),
        ("Vision-dependent reasoning", tuple(VISION)),
        ("Perception and hallucination", tuple(PERCEPTION)),
    )
    assert suites["opd"].benchmarks == comparison.benchmarks
    assert suites["opd"].groups == comparison.groups
    assert "vstar" not in comparison.benchmarks
    # opd repeats the groups of comparison, so they are reported once; papo has none
    assert suite_groupings(suites, list(suites)) == {"comparison": comparison.groups}
    assert suites["papo"].groups == ()


@pytest.mark.parametrize(
    "groups, message",
    [
        ("{g: [geo3k, pope]}", "not in the suite"),
        ("{g: [geo3k, mathvista], h: [mathvista]}", "is in groups"),
        ("{g: [geo3k]}", "without a group"),
        ("[geo3k, mathvista]", "mapping"),
        ("{g: []}", "must list benchmarks"),
    ],
)
def test_suite_groups_must_partition_the_suite(tmp_path, groups, message):
    path = _write_suites(tmp_path, f"suites:\n  s:\n    benchmarks: [geo3k, mathvista]\n    groups: {groups}\n")
    with pytest.raises(ValueError, match=message):
        load_suites(path, _specs())


def test_a_suite_that_adds_benchmarks_does_not_inherit_groups(tmp_path):
    path = _write_suites(
        tmp_path,
        "suites:\n  s:\n    benchmarks: [geo3k, mathvista]\n    groups: {g: [geo3k], h: [mathvista]}\n"
        "  alias:\n    include: [s]\n  wider:\n    include: [s]\n    benchmarks: [pope]\n",
    )
    suites = load_suites(path, _specs())
    assert suites["alias"].groups == suites["s"].groups
    assert suites["wider"].groups == ()


GROUPS = {"suite": (("A", ("a1", "a2")), ("B", ("b1",)))}


def test_group_means_weight_groups_equally_and_keep_the_benchmark_mean():
    results = [_result("a1", 40), _result("a2", 60), _result("b1", 80), _result("other", 0)]
    scores = {score.name: score for score in suite_group_scores(results, GROUPS)}
    assert scores["A"].value == pytest.approx(50.0)
    assert scores["B"].value == pytest.approx(80.0)
    assert scores["Group mean"].value == pytest.approx(65.0)
    assert scores["Benchmark mean"].value == pytest.approx(60.0)
    assert scores["Group mean"].column == "suite: Group mean"
    assert scores["Benchmark mean"].num_examples == 30


def test_incomplete_groups_have_no_mean_and_absent_suites_are_left_out():
    results = [_result("a1", 40), _result("a2", 0, status="failed"), _result("b1", 80)]
    scores = {score.name: score for score in suite_group_scores(results, GROUPS)}
    assert scores["A"].value is None and scores["A"].missing == ("a2",)
    assert scores["B"].value == pytest.approx(80.0)
    assert scores["Group mean"].value is None and scores["Group mean"].missing == ("a2",)
    assert scores["Benchmark mean"].value is None
    assert suite_group_scores([_result("other", 10)], GROUPS) == []
    assert suite_group_scores(results, None) == []


def test_summaries_report_the_group_means(tmp_path):
    results = [_result("a1", 40), _result("a2", 60), _result("b1", 80)]
    per_run = tmp_path / "run" / "summary.csv"
    write_summary_csv(per_run, results, run_metadata={"run_id": "r", "model": "m"}, suite_groups=GROUPS)
    with per_run.open(newline="", encoding="utf-8") as f:
        rows = {row["benchmark"]: row for row in csv.DictReader(f)}
    assert rows["suite: A"]["row_type"] == "suite_group"
    assert rows["suite: Group mean"]["normalized_score_0_100"] == "65.0"
    assert rows["suite: Benchmark mean"]["row_type"] == "suite_benchmark_mean"
    assert rows["overall_average"]["normalized_score_0_100"] == "60.0"

    global_path = tmp_path / "summary.csv"
    update_global_summary_csv(global_path, results, run_metadata={"run_id": "r", "model": "m"}, suite_groups=GROUPS)
    with global_path.open(newline="", encoding="utf-8") as f:
        header_groups, header, values = list(csv.reader(f))
    row = dict(zip(header, values))
    assert row["suite: A"] == "50.00" and row["suite: Group mean"] == "65.00"
    assert row["suite: Benchmark mean"] == "60.00" and row["Overall Avg"] == "60.00"
    assert header_groups[header.index("suite: A")] == "Suite suite"
    assert header.index("suite: Benchmark mean") < header.index("Overall Avg")

    partial = [_result("a1", 40), _result("b1", 80)]
    write_summary_csv(per_run, partial, run_metadata={"run_id": "r", "model": "m"}, suite_groups=GROUPS)
    with per_run.open(newline="", encoding="utf-8") as f:
        rows = {row["benchmark"]: row for row in csv.DictReader(f)}
    assert rows["suite: A"]["status"] == "incomplete"
    assert rows["suite: A"]["normalized_score_0_100"] == ""
    assert json.loads(rows["suite: A"]["details_json"]) == {"missing": ["a2"]}
    table = format_results_table(partial, suite_groups=GROUPS)
    assert "suite: Group mean" in table and "missing: a2" in table


def _cvbench_rows():
    image = {"bytes": _png_bytes(), "path": "x.png"}
    row = {"type": "2D", "image": image, "question": "How many cups?", "filename": "f", "source_dataset": "s"}
    return [
        {**row, "idx": 0, "task": "Count", "source": "ADE20K", "choices": ["1", "2"], "answer": "(B)",
         "prompt": "How many cups? Select from the following choices.\n(A) 1\n(B) 2"},
        {**row, "idx": 1, "task": "Relation", "source": "COCO", "choices": ["left", "right"], "answer": "(A)",
         "prompt": "Where is the cup?\n(A) left\n(B) right"},
        {**row, "idx": 2, "type": "3D", "task": "Depth", "source": "Omni3D", "choices": ["lamp", "chair"],
         "answer": "(A)", "prompt": "Which is closer?\n(A) lamp\n(B) chair"},
    ]  # fmt: skip


def test_cvbench_loader_keeps_the_official_prompt_and_scores_by_source(tmp_path):
    (tmp_path / "cvbench").mkdir()
    rows = _cvbench_rows()
    pd.DataFrame(rows[:2]).to_parquet(tmp_path / "cvbench" / "test_2d.parquet")
    pd.DataFrame(rows[2:]).to_parquet(tmp_path / "cvbench" / "test_3d.parquet")
    samples = load_samples(_spec("cvbench"), tmp_path)

    assert [sample.sample_id for sample in samples] == ["0", "1", "2"]
    assert samples[0].prompt.startswith("How many cups? Select from the following choices.\n(A) 1\n(B) 2\n")
    assert [sample.target for sample in samples] == ["B", "A", "A"]
    assert samples[2].metadata["category"] == "Omni3D" and samples[2].metadata["task"] == "3D Depth"
    assert samples[0].extra_info["options"] == ["1", "2"]

    predictions = [
        {"sample_id": s.sample_id, "target": s.target, "responses": [r], "extra_info": s.extra_info,
         "metadata": s.metadata}
        for s, r in zip(samples, [r"\boxed{B}", r"\boxed{B}", r"\boxed{A}"])
    ]  # fmt: skip
    result = score_mcq(_spec("cvbench"), predictions, None, tmp_path)
    # 2D = (ADE20K 1.0 + COCO 0.0) / 2, 3D = Omni3D 1.0, overall = (0.5 + 1.0) / 2
    assert result.normalized_score_0_100 == pytest.approx(75.0)
    assert result.details["accuracy_2d"] == pytest.approx(0.5)
    assert result.details["accuracy_3d"] == pytest.approx(1.0)


def test_cvbench_combination_matches_the_dataset_card():
    scores = [1, 1, 0, 1, 0, 1]
    sources = ["ADE20K", "COCO", "COCO", "Omni3D", "Omni3D", "Omni3D"]
    # 2D = (1 + 0.5) / 2 = 0.75, 3D = 2/3
    assert mcq_aggregate(scores, sources, "cvbench") == pytest.approx((0.75 + 2 / 3) / 2)
    assert cvbench_accuracies([1, 0], ["ADE20K", "ADE20K"]) == {"accuracy_2d": 0.5, "overall": 0.5}
    with pytest.raises(ValueError, match="unknown sources"):
        cvbench_accuracies([1], ["ScanNet"])


def _mme_parquet(path, categories):
    rows = [
        {"question_id": f"{c}/{i}.jpg", "question": f"Is it {c}? Please answer yes or no.", "answer": a,
         "category": c, "image": {"bytes": _png_bytes(), "path": None}}
        for c in categories for i, a in ((0, "Yes"), (0, "No"))
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def test_mme_perception_and_cognition_split_the_subtasks_of_mme(tmp_path):
    _mme_parquet(tmp_path / "mme" / "test.parquet", MME_PERCEPTION + MME_COGNITION)
    full = load_samples(_spec("mme"), tmp_path)
    perception = load_samples(_spec("mme_perception"), tmp_path)
    cognition = load_samples(_spec("mme_cognition"), tmp_path)
    assert len(full) == 28 and len(perception) == 20 and len(cognition) == 8
    assert {sample.metadata["category"] for sample in perception} == set(MME_PERCEPTION)
    assert {sample.metadata["category"] for sample in cognition} == set(MME_COGNITION)

    _mme_parquet(tmp_path / "partial" / "mme" / "test.parquet", MME_PERCEPTION[:-1])
    with pytest.raises(ValueError, match="OCR"):
        load_samples(_spec("mme_perception"), tmp_path / "partial")
    assert len(load_samples(_spec("mme_perception"), tmp_path / "partial", limit=4)) == 4


def test_mme_perception_and_cognition_are_scored_out_of_2000_and_800(tmp_path):
    def rows(categories):
        return [
            {"sample_id": f"{c}:{q}", "target": t, "responses": [t], "metadata": {"category": c, "question_id": "1"}}
            for c in categories for q, t in (("a", "Yes"), ("b", "No"))
        ]  # fmt: skip

    perception = score_mme(_spec("mme_perception"), rows(MME_PERCEPTION), None, tmp_path)
    assert perception.raw_score == pytest.approx(2000.0)
    assert perception.normalized_score_0_100 == pytest.approx(100.0)
    full = score_mme(_spec("mme"), rows(MME_PERCEPTION), None, tmp_path)
    assert full.normalized_score_0_100 == pytest.approx(2000.0 / 2800.0 * 100.0)
    with pytest.raises(ValueError, match="outside"):
        score_mme(_spec("mme_perception"), rows(["code_reasoning"]), None, tmp_path)
    cognition = score_mme(_spec("mme_cognition"), rows(MME_COGNITION), None, tmp_path)
    assert cognition.raw_score == pytest.approx(800.0)
    assert cognition.normalized_score_0_100 == pytest.approx(100.0)
    # one of the two questions of each image wrong: acc 50 + acc+ 0 per subtask
    half = [dict(row, responses=["Yes"]) for row in rows(MME_COGNITION)]
    assert score_mme(_spec("mme_cognition"), half, None, tmp_path).normalized_score_0_100 == pytest.approx(25.0)
    with pytest.raises(ValueError, match="outside"):
        score_mme(_spec("mme_cognition"), rows(["OCR"]), None, tmp_path)


def test_mme_perception_and_cognition_prepare_mme_once(tmp_path, monkeypatch):
    calls = []
    mme = cli.SOURCES["mme"]

    def fake_prepare(ctx, source):
        calls.append(source.key)
        _mme_parquet(ctx.data_root / "mme" / "test.parquet", MME_PERCEPTION)
        return {"rows": 20}

    fake_mme = type(mme)(**{**mme.__dict__, "prepare": fake_prepare})
    monkeypatch.setitem(cli.SOURCES, "mme", fake_mme)
    for key in ("mme_perception", "mme_cognition"):
        shared = cli.SOURCES[key]
        assert shared.prepare is lmms_lab.prepare_shared and shared.options["shares"] is mme
        monkeypatch.setitem(cli.SOURCES, key, type(shared)(**{**shared.__dict__, "options": {"shares": fake_mme}}))
    ctx = PrepareContext(data_root=tmp_path, log=lambda message: None)

    assert cli.prepare_one(ctx, "mme_perception")[0] == "prepared"
    assert is_prepared(tmp_path / "mme", "mme", fake_mme.output_paths(tmp_path))
    assert cli.prepare_one(ctx, "mme_cognition")[0] == "prepared"
    assert cli.prepare_one(ctx, "mme")[0] == "skipped"
    assert calls == ["mme"]

    # --force refreshes the shared data once per invocation, whichever of the three benchmarks asks first
    forced = PrepareContext(data_root=tmp_path, force=True, log=lambda message: None)
    assert [cli.prepare_one(forced, key)[0] for key in ("mme_cognition", "mme_perception", "mme")] == [
        "prepared",
        "prepared",
        "skipped",
    ]
    assert calls == ["mme", "mme"]
    forced = PrepareContext(data_root=tmp_path, force=True, log=lambda message: None)
    for key in ("mme", "mme_perception"):
        cli.prepare_one(forced, key)
    assert calls == ["mme", "mme", "mme"]
    assert cli.SOURCES["cvbench"].outputs == ("cvbench/test_2d.parquet", "cvbench/test_3d.parquet")


def test_global_summary_keeps_suite_means_after_new_benchmark_groups(tmp_path):
    path = tmp_path / "summary.csv"
    first = [_result("a1", 40), _result("a2", 60), _result("b1", 80)]
    update_global_summary_csv(path, first, run_metadata={"run_id": "r1", "model": "m"}, suite_groups=GROUPS)
    later = [MetricResult("hr", "HighRes", "acc", 10, 10, 5)]
    update_global_summary_csv(path, later, run_metadata={"run_id": "r2", "model": "m"}, suite_groups=GROUPS)
    with path.open(newline="", encoding="utf-8") as f:
        header_groups, header, *rows = list(csv.reader(f))
    assert header.index("hr") < header.index("suite: A") < header.index("Overall Avg")
    assert dict(zip(header, rows[0]))["suite: Group mean"] == "65.00"
    assert dict(zip(header, rows[1]))["suite: Group mean"] == ""


def test_a_custom_registry_without_suite_skips_the_suite_means(tmp_path):
    import argparse

    import yaml

    registry = yaml.safe_load((ROOT / "eval/config/benchmarks.yaml").read_text(encoding="utf-8"))
    registry["benchmarks"] = [entry for entry in registry["benchmarks"] if entry["key"] == "pope"]
    config = tmp_path / "benchmarks.yaml"
    config.write_text(yaml.safe_dump(registry), encoding="utf-8")
    suites_config = str(ROOT / "eval/config/suites.yaml")

    args = argparse.Namespace(config=str(config), suites_config=suites_config, suite=None)
    assert all_suite_groups(args) == {}
    with pytest.raises(ValueError, match="unknown benchmark"):
        all_suite_groups(argparse.Namespace(config=str(config), suites_config=suites_config, suite="comparison"))
    default = argparse.Namespace(
        config=str(ROOT / "eval/config/benchmarks.yaml"), suites_config=suites_config, suite=None
    )
    assert list(all_suite_groups(default)) == ["comparison"]
