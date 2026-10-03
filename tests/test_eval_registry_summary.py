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
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.registry import load_benchmark_specs, select_benchmarks  # noqa: E402
from easyr1_eval.schemas import MetricResult  # noqa: E402
from easyr1_eval.scorers import DEFAULT_DEEPSEEK_JUDGE_MODEL  # noqa: E402
from easyr1_eval.summary import (  # noqa: E402
    GLOBAL_META_FIELDNAMES,
    update_global_summary_csv,
    write_summary_csv,
)


REASONING = [
    "geo3k",
    "mathvista",
    "wemath",
    "mmk12",
    "mathverse",
    "mathverse_v",
    "logicvista",
    "clevr_count",
    "mmmu_pro",
    "dynamath",
    "mathvision",
]


def _specs():
    return load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml")


def test_registry_groups_and_keys():
    by_group = {}
    for spec in _specs():
        by_group.setdefault(spec.group, []).append(spec.key)

    assert by_group == {
        "Reasoning": REASONING,
        "Perception": [
            "pope",
            "hallusionbench",
            "mme",
            "gqa",
            "mm_vet",
            "seed_bench",
            "cvqa_real",
            "mars_bench",
            "textvqa",
        ],
        "HighRes": ["vstar", "hrbench_4k", "hrbench_8k", "mme_realworld_lite"],
        "Grounding": [
            "grit_vsr",
            "grit_tallyqa",
            "tallyqa_relabeled",
            "grit_gqa",
            "ovdeval_position",
            "refcoco_val",
            "refcoco_plus_val",
            "refcocog_val",
        ],
    }


def test_registry_paths_are_relative_to_the_data_root():
    specs = _specs()
    for spec in specs:
        paths = list(spec.paths or []) + ([spec.path] if isinstance(spec.path, str) else list(spec.path or []))
        for value in paths + ([spec.image_root] if spec.image_root else []):
            assert not Path(value).is_absolute(), (spec.key, value)
            assert value.split("/")[0] in {spec.key, "refcoco"}, (spec.key, value)


def test_reasoning_benchmarks_use_papo_eval_protocol():
    by_key = {spec.key: spec for spec in _specs()}
    for key in REASONING:
        spec = by_key[key]
        assert (spec.loader, spec.scorer, spec.primary_metric) == ("sharegpt", "boxed_exact_match", "mean_acc_at_k")
        assert (spec.num_samples, spec.temperature, spec.top_p, spec.max_new_tokens) == (8, 1.0, 1.0, 2048)


def test_registry_metric_contracts():
    by_key = {spec.key: spec for spec in _specs()}
    for key in ["grit_vsr", "grit_tallyqa", "grit_gqa"]:
        assert by_key[key].scorer == "answer_bbox"
        assert by_key[key].primary_metric == "answer_accuracy"
    assert by_key["grit_tallyqa"].path == "grit_tallyqa/tallyqa_val.jsonl"
    assert (by_key["ovdeval_position"].scorer, by_key["ovdeval_position"].primary_metric) == (
        "grounding_iou",
        "grit_iou",
    )
    for key in ["refcoco_val", "refcoco_plus_val", "refcocog_val"]:
        assert by_key[key].primary_metric == "acc_at_0_5_iou"
    assert by_key["hallusionbench"].primary_metric == "question_accuracy"
    assert by_key["vstar"].metadata["aggregate"] == "micro"
    assert by_key["hrbench_4k"].metadata["aggregate"] == "category_mean"
    assert (by_key["mme_realworld_lite"].scorer, by_key["mme_realworld_lite"].metadata["aggregate"]) == (
        "mcq",
        "micro",
    )
    for key in ["cvqa_real", "mars_bench", "textvqa"]:
        spec = by_key[key]
        assert (spec.loader, spec.scorer, spec.num_samples, spec.temperature) == ("cfpo_json", "cfpo_match", 8, 1.0)
    assert by_key["mm_vet"].requires_judge
    assert not by_key["gqa"].requires_judge
    assert {key for key, spec in by_key.items() if spec.optional} == {
        "seed_bench",
        "refcoco_val",
        "refcoco_plus_val",
        "refcocog_val",
    }


def test_every_registered_benchmark_has_a_prepare_recipe_and_known_scorer():
    sys.path.insert(0, str(ROOT))
    from easyr1_eval.scorers import SCORERS

    from eval.prepare.sources import SOURCES

    for spec in _specs():
        assert spec.key in SOURCES, spec.key
        assert spec.scorer in SCORERS, spec.scorer
        target = SOURCES[spec.key].target
        for path in spec.data_paths(Path("/root")):
            assert path.relative_to("/root").parts[0] == target, (spec.key, path)
        for output in SOURCES[spec.key].outputs:
            assert output.split("/")[0] == target


def test_select_benchmarks_include_and_skip():
    selected = select_benchmarks(_specs(), include=["geo3k", "gqa"], skip=["gqa"])

    assert [spec.key for spec in selected] == ["geo3k"]


def test_summary_csv_adds_group_and_overall_rows(tmp_path):
    results = [
        MetricResult(
            "a",
            "Grounding",
            "acc",
            50,
            50,
            10,
            details={
                "generation/truncated_rate": 0.25,
                "generation/truncated_count": 1,
                "generation/response_count": 4,
            },
            metadata={"max_model_len": 16384},
        ),
        MetricResult("b", "Grounding", "acc", 100, 100, 10, metadata={"max_model_len": 16384}),
        MetricResult("c", "Reasoning", "acc", 25, 25, 10, metadata={"max_model_len": 32768}),
    ]
    path = tmp_path / "summary.csv"
    write_summary_csv(
        path, results, run_metadata={"run_id": "run", "model": "m", "backend": "dummy", "max_model_len": 32768}
    )

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lookup = {row["benchmark"]: row for row in rows}
    assert lookup["a"]["truncated_rate"] == "0.25"
    assert lookup["a"]["truncated_count"] == "1"
    assert lookup["a"]["response_count"] == "4"
    assert lookup["a"]["max_model_len"] == "16384"
    assert lookup["Grounding_average"]["normalized_score_0_100"] == "75.0"
    assert lookup["Grounding_average"]["max_model_len"] == "16384"
    assert lookup["overall_average"]["max_model_len"] == "mixed"
    assert lookup["overall_average"]["normalized_score_0_100"] == str((50 + 100 + 25) / 3)


def test_summary_csv_records_the_run_options_that_were_set(tmp_path):
    import csv

    result = MetricResult("geo3k", "Reasoning", "mean_acc_at_k", 50.0, 50.0, 2)
    path = tmp_path / "summary.csv"
    write_summary_csv(path, [result], run_metadata={"run_id": "r", "top_k": 50, "answer_protocol": "pepo"})
    (row, *_) = list(csv.DictReader(path.open()))
    assert (row["top_k"], row["answer_protocol"], row["grounding_instruction"]) == ("50", "pepo", "")


def test_summary_csv_preserves_flat_agent_configuration(tmp_path):
    path = tmp_path / "summary.csv"
    global_path = tmp_path / "global.csv"
    metadata = {
        "run_id": "agent-run",
        "model": "model",
        "interaction_mode": "agentic",
        "agent_profile": "deepeyes",
        "agent_output_contract": "native",
        "agent_runner_version": 2,
        "agent_max_tool_calls": 6,
        "agent_max_response_tokens": 20480,
        "agent_max_tokens_per_turn": 10240,
        "agent_max_images_per_prompt": 16,
        "agent_tool_image_mode": "fixed_gray",
    }

    write_summary_csv(
        path,
        [MetricResult("a", "Grounding", "acc", 50, 50, 1)],
        run_metadata=metadata,
    )

    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["interaction_mode"] == "agentic"
    assert row["agent_profile"] == "deepeyes"
    assert row["agent_output_contract"] == "native"
    assert row["agent_runner_version"] == "2"
    assert row["agent_max_tool_calls"] == "6"
    assert row["agent_max_response_tokens"] == "20480"
    assert row["agent_max_tokens_per_turn"] == "10240"
    assert row["agent_max_images_per_prompt"] == "16"
    assert row["agent_tool_image_mode"] == "fixed_gray"

    update_global_summary_csv(
        global_path,
        [MetricResult("a", "Grounding", "acc", 50, 50, 1)],
        run_metadata=metadata,
    )
    with global_path.open(newline="", encoding="utf-8") as f:
        global_rows = list(csv.reader(f))
    global_row = dict(zip(global_rows[1], global_rows[2]))
    assert global_row["interaction_mode"] == "agentic"
    assert global_row["agent_profile"] == "deepeyes"
    assert global_row["agent_output_contract"] == "native"
    assert global_row["agent_runner_version"] == "2"
    assert global_row["agent_max_tool_calls"] == "6"
    assert global_row["agent_tool_image_mode"] == "fixed_gray"


def test_primary_na_skipped_and_auxiliary_columns_do_not_enter_averages(
    tmp_path,
):
    per_run = tmp_path / "per_run.csv"
    global_path = tmp_path / "global.csv"
    results = [
        MetricResult("grit_tallyqa", "Grounding", "answer_accuracy", None, None, 10, status="primary_na"),
        MetricResult(
            "grit_vsr",
            "Grounding",
            "answer_accuracy",
            50,
            50,
            10,
            details={"grounding/grit_iou": 0.75},
        ),
        MetricResult("grit_gqa", "Grounding", "answer_accuracy", 40, 40, 10),
        MetricResult("gqa", "Perception", "accuracy", 60, 60, 10),
        MetricResult("mm_vet", "Perception", "judge_score", None, None, 0, status="skipped"),
    ]
    metadata = {
        "run_id": "agent",
        "output_dir": "/tmp/agent",
        "interaction_mode": "agentic",
        "agent_output_contract": "native",
    }

    write_summary_csv(per_run, results, run_metadata=metadata)
    update_global_summary_csv(global_path, results, run_metadata=metadata)

    with per_run.open(newline="", encoding="utf-8") as f:
        per_run_rows = {row["benchmark"]: row for row in csv.DictReader(f)}
    assert per_run_rows["grit_tallyqa"]["raw_score"] == "n/a"
    assert per_run_rows["grit_tallyqa"]["normalized_score_0_100"] == "n/a"
    assert per_run_rows["mm_vet"]["status"] == "skipped"
    assert per_run_rows["Grounding_average"]["normalized_score_0_100"] == "45.0"
    assert per_run_rows["Perception_average"]["normalized_score_0_100"] == "60.0"
    assert per_run_rows["overall_average"]["normalized_score_0_100"] == "50.0"

    with global_path.open(newline="", encoding="utf-8") as f:
        global_rows = list(csv.reader(f))
    header = global_rows[1]
    row = dict(zip(header, global_rows[2]))
    assert header.index("grit_tallyqa_giou") == header.index("grit_tallyqa") + 1
    assert header.index("grit_vsr_giou") == header.index("grit_vsr") + 1
    assert "mm_vet" not in header
    assert row["grit_tallyqa"] == "n/a"
    assert row["grit_tallyqa_giou"] == ""
    assert row["grit_vsr"] == "50.00"
    assert row["grit_vsr_giou"] == "75.00"
    assert row["Grounding Avg"] == "45.00"
    assert row["Overall Avg"] == "50.00"


def test_global_summary_replaces_same_run(tmp_path):
    path = tmp_path / "summary.csv"
    update_global_summary_csv(
        path,
        [
            MetricResult("a", "Grounding", "acc", 10, 10, 1),
            MetricResult("b", "Reasoning", "acc", 40, 40, 1),
        ],
        run_metadata={"run_id": "run1", "output_dir": "/tmp/run1", "model": "m"},
    )
    update_global_summary_csv(
        path,
        [
            MetricResult("a", "Grounding", "acc", 20, 20, 1),
            MetricResult("b", "Reasoning", "acc", 60, 60, 1),
        ],
        run_metadata={"run_id": "run1", "output_dir": "/tmp/run1", "model": "m", "format_prompt": "/tmp/format.jinja"},
    )
    update_global_summary_csv(
        path,
        [MetricResult("a", "Grounding", "acc", 30, 30, 1)],
        run_metadata={"run_id": "run2", "output_dir": "/tmp/run2", "model": "m"},
    )

    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0][: len(GLOBAL_META_FIELDNAMES)] == [
        "Meta",
        *([""] * (len(GLOBAL_META_FIELDNAMES) - 1)),
    ]
    assert rows[0][len(GLOBAL_META_FIELDNAMES) :] == [
        "Grounding",
        "",
        "Reasoning",
        "",
        "Overall",
    ]
    assert rows[1] == [
        *GLOBAL_META_FIELDNAMES,
        "a",
        "Grounding Avg",
        "b",
        "Reasoning Avg",
        "Overall Avg",
    ]
    assert rows[2][0] == "run1"
    assert rows[2][5] == "/tmp/format.jinja"
    assert rows[2][len(GLOBAL_META_FIELDNAMES) :] == [
        "20.00",
        "20.00",
        "60.00",
        "60.00",
        "40.00",
    ]
    assert rows[3][0] == "run2"
    assert rows[3][len(GLOBAL_META_FIELDNAMES) :] == [
        "30.00",
        "30.00",
        "",
        "",
        "30.00",
    ]


def test_global_summary_inserts_new_benchmarks_before_averages_and_moves_existing_values(tmp_path):
    path = tmp_path / "summary.csv"
    update_global_summary_csv(
        path,
        [
            MetricResult("ground_old", "Grounding", "acc", 10, 10, 1),
            MetricResult("reason_old", "Reasoning", "acc", 40, 40, 1),
        ],
        run_metadata={"run_id": "old", "output_dir": "/tmp/old", "model": "m"},
    )
    update_global_summary_csv(
        path,
        [
            MetricResult("ground_old", "Grounding", "acc", 20, 20, 1),
            MetricResult("ground_new", "Grounding", "acc", 30, 30, 1),
            MetricResult("reason_old", "Reasoning", "acc", 60, 60, 1),
            MetricResult("reason_new", "Reasoning", "acc", 80, 80, 1),
        ],
        run_metadata={"run_id": "new", "output_dir": "/tmp/new", "model": "m"},
    )

    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0][: len(GLOBAL_META_FIELDNAMES)] == [
        "Meta",
        *([""] * (len(GLOBAL_META_FIELDNAMES) - 1)),
    ]
    assert rows[0][len(GLOBAL_META_FIELDNAMES) :] == [
        "Grounding",
        "",
        "",
        "Reasoning",
        "",
        "",
        "Overall",
    ]
    assert rows[1] == [
        *GLOBAL_META_FIELDNAMES,
        "ground_old",
        "ground_new",
        "Grounding Avg",
        "reason_old",
        "reason_new",
        "Reasoning Avg",
        "Overall Avg",
    ]
    assert rows[2][len(GLOBAL_META_FIELDNAMES) :] == [
        "10.00",
        "",
        "10.00",
        "40.00",
        "",
        "40.00",
        "25.00",
    ]
    assert rows[3][len(GLOBAL_META_FIELDNAMES) :] == [
        "20.00",
        "30.00",
        "25.00",
        "60.00",
        "80.00",
        "70.00",
        "47.50",
    ]


def test_default_deepseek_judge_model_uses_v4_flash():
    assert DEFAULT_DEEPSEEK_JUDGE_MODEL == "deepseek-v4-flash"
