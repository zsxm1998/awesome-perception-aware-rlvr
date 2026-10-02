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
"""MME-RealWorld-Lite, CFPO real-world benchmarks (C-VQA-Real, MARS-Bench, TextVQA), SEED-Bench."""

import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from mathruler.grader import extract_boxed_content, grade_answer
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.loaders import (  # noqa: E402
    _mme_realworld_option_texts,
    load_cfpo_json,
    load_samples,
    load_seed_bench,
)
from easyr1_eval.registry import load_benchmark_specs  # noqa: E402
from easyr1_eval.scorers import (  # noqa: E402
    _sample_score_and_correct,
    cfpo_correct,
    cfpo_extract_answer,
    extract_choice_letter,
    score_cfpo_match,
    score_mcq,
    score_seed_bench,
    vqa_accuracy,
)

from eval.prepare import cfpo, cli, mme_realworld, seed_bench  # noqa: E402
from eval.prepare.common import PrepareContext, PrepareError, RemoteZip, extract_remote_zip_members  # noqa: E402


def _registry(key):
    return next(spec for spec in load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml") if spec.key == key)


def _png(color=(255, 0, 0), size=(4, 4)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _quiet_ctx(data_root, **kwargs):
    return PrepareContext(data_root=data_root, keep_raw=True, log=lambda message: None, **kwargs)


# ---------------------------------------------------------------------------
# MME-RealWorld-Lite
# ---------------------------------------------------------------------------


def _mme_rw_record(index, category, answer, color):
    return {
        "bytes": base64.b64encode(_png(color)).decode(),
        "path": "dummy/path",
        "index": index,
        "question": f"What is shown {index}?",
        "multi-choice options": [
            "(A) red",
            "(B) green",
            "(C) blue",
            "(D) white",
            "(E) The image does not feature the object.",
        ],
        "answer": answer,
        "category": category,
        "l2-category": "diagram",
    }


def test_mme_realworld_prepare_loader_and_scorer(tmp_path, monkeypatch):
    records = [
        _mme_rw_record(2, "Perception/OCR with Complex Context", "A", (255, 0, 0)),
        _mme_rw_record(1, "Reasoning/Autonomous_Driving", "B", (255, 0, 0)),
        _mme_rw_record(3, "Perception/Remote Sensing", "C", (0, 0, 255)),
    ]
    shards = []
    for index in range(4):  # one record per shard; the last shard is empty
        path = tmp_path / f"shard{index}.parquet"
        pd.DataFrame(records[index : index + 1], columns=list(records[0])).to_parquet(path)
        shards.append(path)
    monkeypatch.setattr(mme_realworld, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(mme_realworld, "hf_download", lambda ctx, repo, name: shards[mme_realworld.SHARDS.index(name)])
    data_root = tmp_path / "data"
    source = mme_realworld.SOURCES[0]

    info = mme_realworld.prepare_mme_realworld_lite(_quiet_ctx(data_root), source)

    assert info["rows"] == 3 and info["images"] == 2  # identical images are stored once
    samples = load_samples(_registry("mme_realworld_lite"), data_root)
    assert [sample.sample_id for sample in samples] == ["1", "2", "3"]
    first = samples[1]
    assert first.prompt.startswith("What is shown 2? The choices are listed below:\n(A) red\n")
    assert first.prompt.endswith("Respond with only the letter (A, B, C, D, or E) of the correct option.")
    assert first.extra_info["options"][0] == "red"
    assert (first.metadata["task"], first.metadata["category"]) == ("Perception", "Perception/OCR")
    assert samples[0].metadata["category"] == "Reasoning/AD"

    rows = [
        {
            "sample_id": s.sample_id,
            "target": s.target,
            "responses": [response],
            "extra_info": s.extra_info,
            "metadata": s.metadata,
        }
        for s, response in zip(samples, [r"\boxed{B}", "<answer>(A) red</answer>", "The best answer is D"])
    ]
    result = score_mcq(_registry("mme_realworld_lite"), rows, None, tmp_path)
    assert result.raw_score == pytest.approx(200 / 3)
    assert result.details["accuracy_by_task"] == {"Perception": pytest.approx(0.5), "Reasoning": pytest.approx(1.0)}
    assert result.details["accuracy_by_category"] == {"Perception/OCR": 1.0, "Perception/RS": 0.0, "Reasoning/AD": 1.0}


def test_answer_is_colon_newline_is_parsed():
    options = ["red", "green", "blue", "white", "The image does not feature the object."]
    response = "Looking at the scene...\n\nTherefore, the correct answer is:\n(D) white."
    assert extract_choice_letter(response, options) == "D"
    assert extract_choice_letter("The answer is: B", options) == "B"


def test_mme_realworld_duplicate_option_letter_keeps_letter_positions():
    # two Autonomous_Driving rows print "(D)" twice; the letter inside the parentheses decides
    options = [
        "(A) fast",
        "(B) right",
        "(C) straight",
        "(D) not moving",
        "(D) None of the above",
        "(E) All the above answers are wrong.",
    ]
    texts = _mme_realworld_option_texts(options)
    assert texts == ["fast", "right", "straight", "not moving", "All the above answers are wrong."]
    assert extract_choice_letter("All the above answers are wrong.", texts) == "E"
    assert extract_choice_letter("F", texts) is None


# ---------------------------------------------------------------------------
# CFPO benchmarks
# ---------------------------------------------------------------------------


def cfpo_reference(raw_output, gt):
    """Verbatim scoring of CFPO Counterfactual-Eval/inference_eval.py (non-PAPO models)."""
    answer = extract_boxed_content(raw_output)
    if answer == "None":
        answer = raw_output.strip()
    if isinstance(gt, list):
        for gt_single in gt:
            is_correct = grade_answer(answer, gt_single)
            if not is_correct and isinstance(answer, str) and ":" in answer:
                is_correct = grade_answer(answer.split(":")[0].strip(), gt_single)
            if is_correct:
                break
    else:
        is_correct = grade_answer(answer, gt)
        if not is_correct and isinstance(answer, str):
            if ":" in answer:
                is_correct = grade_answer(answer.split(":")[0].strip(), gt)
            elif answer == "No.":
                is_correct = grade_answer("No", gt)
    return is_correct


CFPO_CASES = [
    (r"<think>count</think>\boxed{1}", "1"),
    (r"\boxed{2}", "1"),
    ("yes", "yes"),  # no box: the whole response is graded
    ("No.", "no"),
    (r"\boxed{B: the left one}", "B"),
    ("I think there are 3 plates", "3"),
    (r"\boxed{Dakota}", ["nous les gosses", "dakota", "dakota digital"]),
    (r"\boxed{nikon: camera}", ["nikon", "canon"]),
    (r"\boxed{sony}", ["nikon", "canon"]),
]


@pytest.mark.parametrize(("response", "target"), CFPO_CASES)
def test_cfpo_grading_matches_counterfactual_eval(response, target):
    assert cfpo_correct(cfpo_extract_answer(response), target) == bool(cfpo_reference(response, target))


def test_vqa_accuracy_is_min_matches_over_three():
    answers = ["dakota", "dakota", "dakota digital", "nous les gosses"] + ["other"] * 6
    assert vqa_accuracy("Dakota", answers) == pytest.approx(2 / 3)
    assert vqa_accuracy("dakota digital.", answers) == pytest.approx(1 / 3)
    assert vqa_accuracy("x", ["x"] * 5) == 1.0


def test_cfpo_loader_and_scorer(tmp_path):
    (tmp_path / "textvqa" / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "textvqa" / "images" / "Textvqa_validation_a.png")
    rows = [
        {
            "images": ["Textvqa_validation_a.png"],
            "problem": "<image>what brand?",
            "answer": ["dakota", "dakota", "dakota", "nikon"],
            "id": "Textvqa_1",
            "type": "Textvqa",
            "is_cf": False,
        },
    ]
    (tmp_path / "textvqa" / "test.json").write_text(json.dumps(rows), encoding="utf-8")
    (sample,) = load_cfpo_json(_registry("textvqa"), tmp_path)
    assert (sample.prompt, sample.sample_id) == ("what brand?", "Textvqa_1")
    assert sample.target == ["dakota", "dakota", "dakota", "nikon"]
    assert sample.metadata["type"] == "Textvqa" and sample.metadata["is_cf"] is False

    prediction_rows = [
        {
            "sample_id": "t1",
            "target": sample.target,
            "responses": [r"\boxed{Dakota}", r"\boxed{Nikon}"],
            "metadata": sample.metadata,
        },
        {
            "sample_id": "c1",
            "target": "2",
            "responses": [r"\boxed{2}", "two"],
            "metadata": {"type": "direct", "is_cf": True},
        },
        {
            "sample_id": "c2",
            "target": "yes",
            "responses": ["Yes", r"\boxed{no}"],
            "metadata": {"type": "boolean", "is_cf": False},
        },
    ]
    result = score_cfpo_match(_registry("textvqa"), prediction_rows, None, tmp_path)
    # t1: 2/2, c1: 1/2 ("two" is not graded equal to 2 by mathruler), c2: 1/2
    assert result.raw_score == pytest.approx((1.0 + 0.5 + 0.5) / 3 * 100)
    groups = result.details["accuracy_by_group"]
    assert groups["cf"] == pytest.approx(0.5) and groups["ncf"] == pytest.approx(0.75)
    assert groups["direct_cf"] == pytest.approx(0.5) and groups["Textvqa"] == pytest.approx(1.0)
    assert result.details["vqa_accuracy"] == pytest.approx((1.0 + 1 / 3) / 2)
    score, correct = _sample_score_and_correct(_registry("cvqa_real"), prediction_rows[1])
    assert (score, correct) == (0.5, True)


def test_cfpo_prepare_downloads_coco_images_and_extracts_archives(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    annotations = {
        "C-VQA-Real_test.json": [
            {
                "images": ["COCO_val2014_000000000001.jpg"],
                "problem": "<image>How many?",
                "answer": "1",
                "id": "c0",
                "type": "direct",
                "is_cf": False,
            }
        ],
        "Textvqa_test.json": [
            {
                "images": ["Textvqa_validation_a.png"],
                "problem": "<image>what?",
                "answer": ["a"] * 10,
                "id": "t0",
                "type": "Textvqa",
                "is_cf": False,
            }
        ],
    }
    for name, rows in annotations.items():
        (raw / name).write_text(json.dumps(rows))
    with zipfile.ZipFile(raw / "Textvqa_images.zip", "w") as zf:
        zf.writestr("Textvqa_validation_a.png", _png())
        zf.writestr("Textvqa_validation_unused.png", _png())

    downloads = []

    def fake_http(ctx, items, *, desc, timeout=60.0):
        for item in items:
            downloads.append(item.urls[0])
            item.dest.parent.mkdir(parents=True, exist_ok=True)
            item.dest.write_bytes(_png())
        return {}

    monkeypatch.setattr(cfpo, "hf_download", lambda ctx, repo, name: raw / name)
    monkeypatch.setattr(cfpo, "http_download_many", fake_http)
    for key in ("cvqa_real", "textvqa"):
        monkeypatch.setitem(cfpo.SPLITS, key, (*cfpo.SPLITS[key][:2], 1, cfpo.SPLITS[key][3]))
    data_root = tmp_path / "data"
    sources = {source.key: source for source in cfpo.SOURCES}

    assert cfpo.prepare_cfpo(_quiet_ctx(data_root), sources["cvqa_real"])["images"] == 1
    assert downloads == ["http://images.cocodataset.org/val2014/COCO_val2014_000000000001.jpg"]
    assert cfpo.prepare_cfpo(_quiet_ctx(data_root), sources["textvqa"])["images"] == 1
    assert (data_root / "textvqa" / "images" / "Textvqa_validation_a.png").is_file()
    assert not (data_root / "textvqa" / "images" / "Textvqa_validation_unused.png").exists()
    assert load_samples(_registry("textvqa"), data_root)[0].target == ["a"] * 10


class _BytesRemoteZip(RemoteZip):
    """RemoteZip served from an in-memory archive (range semantics of an HTTP server)."""

    payload = b""

    def __init__(self, repo_id, filename, *, retries=5, timeout=120.0):  # noqa: D401 - test double
        self.source_url = f"memory://{repo_id}/{filename}"
        self.retries = retries
        self.timeout = timeout
        self._url = None
        self._resolved_at = 0.0
        import threading

        self._lock = threading.Lock()
        self.size = len(self.payload)

    def _request(self, url, start, end):
        return self.payload[start : end + 1], url, 206


def test_remote_zip_reads_stored_and_deflated_members(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(
            "COCO_val2014_images/COCO_val2014_000000000001.jpg", _png((1, 2, 3)), compress_type=zipfile.ZIP_DEFLATED
        )
        zf.writestr(
            "COCO_val2014_images/COCO_val2014_000000000002.jpg", _png((4, 5, 6)), compress_type=zipfile.ZIP_STORED
        )
        zf.writestr("COCO_val2014_images/", b"")
    _BytesRemoteZip.payload = buffer.getvalue()
    archive = _BytesRemoteZip("repo", "x.zip")
    entries = {entry.name: entry for entry in archive.entries()}
    assert len(entries) == 2
    assert archive.read(entries["COCO_val2014_images/COCO_val2014_000000000002.jpg"]) == _png((4, 5, 6))

    import eval.prepare.common as common

    monkeypatch.setattr(common, "RemoteZip", _BytesRemoteZip)
    wanted = {"COCO_val2014_000000000001.jpg": tmp_path / "out" / "COCO_val2014_000000000001.jpg"}
    assert extract_remote_zip_members(_quiet_ctx(tmp_path), "repo", "x.zip", wanted, desc="test") == 1
    assert (tmp_path / "out" / "COCO_val2014_000000000001.jpg").read_bytes() == _png((1, 2, 3))
    # already extracted files are skipped (resumable)
    assert extract_remote_zip_members(_quiet_ctx(tmp_path), "repo", "x.zip", wanted, desc="test") == 0
    with pytest.raises(PrepareError, match="not found"):
        extract_remote_zip_members(
            _quiet_ctx(tmp_path), "repo", "x.zip", {"missing.jpg": tmp_path / "m.jpg"}, desc="test"
        )


# ---------------------------------------------------------------------------
# SEED-Bench
# ---------------------------------------------------------------------------


def _seed_shard(path, rows):
    pd.DataFrame(rows).to_parquet(path)


def _seed_row(question_id, data_type, frames, answer="A", question_type_id=1):
    return {
        "answer": answer,
        "choice_a": "cat",
        "choice_b": "dog",
        "choice_c": "cow",
        "choice_d": "fox",
        "data_id": f"d{question_id}",
        "data_type": data_type,
        "question": f"Which animal {question_id}?",
        "question_id": str(question_id),
        "question_type_id": question_type_id,
        "image": [{"bytes": _png((index, 0, 0)), "path": None} for index in range(frames)],
        "segment": None,
    }


def test_seed_bench_prepare_is_resumable_and_loader_scorer_split_image_video(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    shard_names = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
    _seed_shard(raw / "a.parquet", [_seed_row(1, "image", 1), _seed_row(2, "image", 1, answer="B")])
    _seed_shard(raw / "b.parquet", [_seed_row(3, "video", 8, answer="C", question_type_id=10)])
    files = {shard_names[0]: raw / "a.parquet", shard_names[1]: raw / "b.parquet"}
    calls = []

    def fake_download(ctx, repo, name):
        calls.append(name)
        if len(calls) == 2 and name == shard_names[1] and not getattr(fake_download, "failed", False):
            fake_download.failed = True
            raise PrepareError("network down")
        return files[name]

    monkeypatch.setattr(seed_bench, "_list_shards", lambda: ("lmms-lab/SEED-Bench", shard_names))
    monkeypatch.setattr(seed_bench, "hf_download", fake_download)
    monkeypatch.setattr(seed_bench, "EXPECTED_ROWS", 3)
    data_root = tmp_path / "data"
    source = seed_bench.SOURCES[0]

    with pytest.raises(PrepareError, match="network down"):
        seed_bench.prepare_seed_bench(_quiet_ctx(data_root), source)
    info = seed_bench.prepare_seed_bench(_quiet_ctx(data_root), source)

    assert calls == [shard_names[0], shard_names[1], shard_names[1]]  # the converted shard is not fetched again
    assert info["rows"] == 3 and info["images"] == 8  # frame 0 of the video equals the image of q1/q2
    samples = load_seed_bench(_registry("seed_bench"), data_root)
    assert [len(sample.images) for sample in samples] == [1, 1, 8]
    assert samples[2].metadata["data_type"] == "video"
    assert samples[0].extra_info["options"] == ["cat", "dog", "cow", "fox"]

    rows = [
        {
            "sample_id": s.sample_id,
            "target": s.target,
            "responses": [response],
            "extra_info": s.extra_info,
            "metadata": s.metadata,
        }
        for s, response in zip(samples, [r"\boxed{A}", "<answer>dog</answer>", "A cat is visible"])
    ]
    result = score_seed_bench(_registry("seed_bench"), rows, None, tmp_path)
    assert result.raw_score == pytest.approx(200 / 3)
    assert result.details["seed_image_accuracy"] == pytest.approx(1.0)
    assert result.details["seed_video_accuracy"] == pytest.approx(0.0)
    assert result.details["accuracy_by_question_type"] == {"1": 1.0, "10": 0.0}


def test_prepare_refuses_when_disk_space_is_short(tmp_path, monkeypatch):
    called = []
    source = cli.SOURCES["seed_bench"]
    monkeypatch.setattr(cli, "free_disk_gb", lambda path: 1.0)
    monkeypatch.setitem(
        cli.SOURCES, "seed_bench", type(source)(**{**source.__dict__, "prepare": lambda ctx, s: called.append(1)})
    )
    with pytest.raises(PrepareError, match="needs about 30 GB"):
        cli.prepare_one(_quiet_ctx(tmp_path), "seed_bench")
    assert not called


def test_new_benchmarks_are_in_their_suites():
    sys.path.insert(0, str(ROOT / "eval"))
    from easyr1_eval.suites import load_suites

    suites = load_suites(ROOT / "eval/config/suites.yaml", load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml"))
    assert "mme_realworld_lite" in suites["deepeyes"].benchmarks
    assert {"cvqa_real", "mars_bench", "textvqa", "pope"} <= set(suites["cfpo"].benchmarks)
    assert "seed_bench" not in suites["all"].benchmarks
