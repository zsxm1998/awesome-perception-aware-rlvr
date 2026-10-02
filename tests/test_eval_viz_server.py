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
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import viz_server  # noqa: E402


def _group(sample_id, row_index=None, **metadata):
    if row_index is not None:
        metadata["row_index"] = row_index
    return {"sample_id": sample_id, "metadata": metadata}


def test_display_sort_restores_source_order_for_new_benchmarks():
    grit = [
        _group("grit_gqa:4", 4),
        _group("grit_gqa:0", 0),
        _group("grit_gqa:2", 2),
    ]
    assert [item["sample_id"] for item in viz_server._sort_groups_for_display("grit_gqa", grit)] == [
        "grit_gqa:0",
        "grit_gqa:2",
        "grit_gqa:4",
    ]

    sharegpt = [
        _group("dynamath:10", 10, source_file="test.json"),
        _group("dynamath:2", 2, source_file="test.json"),
    ]
    assert [item["sample_id"] for item in viz_server._sort_groups_for_display("dynamath", sharegpt)] == [
        "dynamath:2",
        "dynamath:10",
    ]

    no_index = [_group("v1_10"), _group("v1_2"), _group("v1_1")]
    assert [item["sample_id"] for item in viz_server._sort_groups_for_display("logicvista", no_index)] == [
        "v1_1",
        "v1_2",
        "v1_10",
    ]


def test_ground_truth_regions_support_pixel_normalized_and_1000_schemas():
    pixel_regions = viz_server._ground_truth_regions(
        {
            "bboxs": [[100, 50, 300, 150], [0, 0, 40, 20]],
            "bbox_names": ["cat", "tail"],
            "width": 400,
            "height": 200,
        }
    )
    assert pixel_regions == [
        {"label": "cat", "bbox_list": [[250.0, 250.0, 750.0, 750.0]], "image_idx": 0},
        {"label": "tail", "bbox_list": [[0.0, 0.0, 100.0, 100.0]], "image_idx": 0},
    ]

    normalized_regions = viz_server._ground_truth_regions(
        {"bboxs_normalized": [[0.1, 0.2, 0.4, 0.6], [0.5, 0.5, 0.9, 0.9]]}
    )
    assert normalized_regions[0]["bbox_list"] == [[100.0, 200.0, 400.0, 600.0]]
    assert normalized_regions[1]["label"] == "GT 2"

    refcoco_regions = viz_server._ground_truth_regions({"bbox_1000": [10, 20, 300, 400]})
    assert refcoco_regions == [{"label": "GT", "bbox_list": [[10.0, 20.0, 300.0, 400.0]], "image_idx": 0}]
    assert viz_server._ground_truth_regions({"bboxs_normalized": []}) == []
    assert viz_server._ground_truth_regions({"bboxs": [[1, 2, 3, 4]], "width": "invalid", "height": 10}) == []


def test_load_per_sample_metrics_reads_exact_and_suffixed_files(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "dynamath_per_sample.jsonl").write_text(
        json.dumps({"sample_id": "variant1:1", "mean_accuracy": 0.5, "scores": [1, 0]}) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "dynamath_per_sample_extra.jsonl").write_text(
        json.dumps({"sample_id": "variant1:1", "subject": "geometry"}) + "\n",
        encoding="utf-8",
    )

    metrics = viz_server._load_per_sample_metrics(tmp_path, "dynamath")

    assert metrics["variant1:1"] == {
        "sample_id": "variant1:1",
        "mean_accuracy": 0.5,
        "scores": [1, 0],
        "subject": "geometry",
    }


def test_samples_api_sorts_rows_and_exposes_multi_box_gt(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    prediction_dir = model_dir / "predictions" / "grit_gqa"
    metrics_dir = model_dir / "metrics"
    prediction_dir.mkdir(parents=True)
    metrics_dir.mkdir()

    records = []
    for row_index in (4, 0):
        records.append(
            {
                "sample_id": f"grit_gqa:{row_index}",
                "prompt": f"question {row_index}",
                "target": "yes",
                "responses": ["<answer>yes</answer>"],
                "response_metadata": [{}],
                "image_refs": [f"/tmp/image_{row_index}.jpg"],
                "extra_info": {
                    "bboxs_normalized": [[0.1, 0.2, 0.3, 0.4]],
                    "bbox_names": ["object"],
                },
                "metadata": {"row_index": row_index},
            }
        )
    (prediction_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (metrics_dir / "grit_gqa_per_sample_answer_bbox.jsonl").write_text(
        json.dumps({"sample_id": "grit_gqa:0", "answer_correct": True, "grit_iou": 0.8}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        viz_server,
        "_benchmark_specs",
        {"grit_gqa": SimpleNamespace(label="GRIT-GQA", group="Grounding")},
    )
    monkeypatch.setitem(viz_server._cfg, "results_dir", str(tmp_path))
    response = viz_server.app.test_client().get(
        "/api/samples",
        query_string={"model_dir": str(model_dir), "benchmark": "grit_gqa", "size": 20},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert [sample["sample_id"] for sample in payload["samples"]] == [
        "grit_gqa:0",
        "grit_gqa:4",
    ]
    first = payload["samples"][0]
    assert first["benchmark"] == "grit_gqa"
    assert first["metrics"]["grit_iou"] == 0.8
    assert first["gt_regions"] == [
        {
            "label": "object",
            "bbox_list": [[100.0, 200.0, 300.0, 400.0]],
            "image_idx": 0,
        }
    ]
    assert first["image_urls"][0].startswith("/api/image?p=")


def test_benchmarks_api_follows_registry_order(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    for key in ("dynamath", "grit_gqa"):
        directory = model_dir / "predictions" / key
        directory.mkdir(parents=True)
        (directory / "predictions.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        viz_server,
        "_benchmark_specs",
        {
            "grit_gqa": SimpleNamespace(label="GRIT-GQA", group="Grounding"),
            "dynamath": SimpleNamespace(label="DynaMath", group="Reasoning"),
        },
    )
    monkeypatch.setitem(viz_server._cfg, "results_dir", str(tmp_path))
    response = viz_server.app.test_client().get("/api/benchmarks", query_string={"model_dir": str(model_dir)})

    assert response.status_code == 200
    assert [item["key"] for item in response.get_json()] == ["grit_gqa", "dynamath"]


def test_benchmarks_api_skips_primary_na_without_hiding_other_scores(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "model"
    for key in ("grit_tallyqa", "grit_gqa"):
        directory = model_dir / "predictions" / key
        directory.mkdir(parents=True)
        (directory / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (model_dir / "summary.csv").write_text(
        "benchmark,row_type,raw_score,status\ngrit_tallyqa,benchmark,n/a,primary_na\ngrit_gqa,benchmark,75.0,ok\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        viz_server,
        "_benchmark_specs",
        {
            "grit_tallyqa": SimpleNamespace(
                label="GRIT-TallyQA",
                group="Grounding",
            ),
            "grit_gqa": SimpleNamespace(
                label="GRIT-GQA",
                group="Grounding",
            ),
        },
    )
    monkeypatch.setitem(viz_server._cfg, "results_dir", str(tmp_path))

    response = viz_server.app.test_client().get(
        "/api/benchmarks",
        query_string={"model_dir": str(model_dir)},
    )
    by_key = {item["key"]: item for item in response.get_json()}

    assert "score" not in by_key["grit_tallyqa"]
    assert by_key["grit_gqa"]["score"] == 75.0


def _b64(path):
    import base64

    return base64.urlsafe_b64encode(str(path).encode()).decode()


def test_image_api_only_serves_results_and_data_roots(tmp_path, monkeypatch):
    from PIL import Image

    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    outside = tmp_path / "private"
    for directory in (data_root / "geo3k", results_dir, outside):
        directory.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(data_root / "geo3k" / "a.png")
    Image.new("RGB", (4, 4)).save(outside / "secret.png")
    monkeypatch.setitem(viz_server._cfg, "data_root", str(data_root))
    monkeypatch.setitem(viz_server._cfg, "results_dir", str(results_dir))
    monkeypatch.setattr(viz_server, "_benchmark_specs", {})
    client = viz_server.app.test_client()

    assert client.get("/api/image", query_string={"p": _b64(data_root / "geo3k" / "a.png")}).status_code == 200
    assert client.get("/api/image", query_string={"p": _b64(outside / "secret.png")}).status_code == 403
    assert client.get("/api/image", query_string={"p": _b64("/etc/passwd")}).status_code == 403
    traversal = f"{data_root}/geo3k/../../private/secret.png"
    assert client.get("/api/image", query_string={"p": _b64(traversal)}).status_code == 403
    assert client.get("/api/image", query_string={"p": _b64(data_root / "missing.png")}).status_code == 404


def test_result_apis_reject_directories_outside_results(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    run_dir = results_dir / "my_run" / "global_step_10"
    (run_dir / "predictions" / "pope").mkdir(parents=True)
    (run_dir / "predictions" / "pope" / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    (outside / "predictions" / "pope").mkdir(parents=True)
    (outside / "predictions" / "pope" / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(viz_server._cfg, "results_dir", str(results_dir))
    monkeypatch.setattr(viz_server, "_benchmark_specs", {})
    client = viz_server.app.test_client()

    models = client.get("/api/models").get_json()
    assert [item["name"] for item in models] == ["my_run/global_step_10"]
    assert client.get("/api/benchmarks", query_string={"model_dir": str(run_dir)}).status_code == 200
    assert client.get("/api/benchmarks", query_string={"model_dir": str(outside)}).status_code == 403
    assert client.get("/api/samples", query_string={"model_dir": str(outside), "benchmark": "pope"}).status_code == 403
    assert (
        client.get(
            "/api/samples",
            query_string={"model_dir": str(run_dir), "benchmark": "pope", "compare_dir": str(outside)},
        ).status_code
        == 403
    )


def test_viz_server_defaults_are_local_and_repo_relative():
    assert viz_server.DEFAULT_HOST == "127.0.0.1"
    from easyr1_eval.paths import default_data_root

    assert viz_server.DEFAULT_DATA_ROOT == default_data_root()
