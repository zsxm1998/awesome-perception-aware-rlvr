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
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.loaders import (  # noqa: E402
    load_grit_jsonl,
    load_hallusionbench,
    load_hrbench,
    load_refcoco,
    load_samples,
    load_sharegpt,
    load_vstar,
    missing_data_files,
    sharegpt_image_relpath,
    strip_sharegpt_question,
)
from easyr1_eval.prompting import PromptConfig, apply_prompt_to_sample  # noqa: E402
from easyr1_eval.schemas import BenchmarkSpec  # noqa: E402


PAPO_SUFFIX = (
    "\n\nYou first think through the reasoning process as an internal monologue, enclosed within "
    "<think> </think> tags. Then, provide your final answer enclosed within \\boxed{}."
)
FORMAT_PROMPT = (ROOT / "examples/format_prompt/math_perception.jinja").read_text(encoding="utf-8")


def _spec(key, loader, scorer, path, image_root=None, **kwargs):
    return BenchmarkSpec(
        key=key,
        label=key,
        group="Reasoning",
        loader=loader,
        scorer=scorer,
        primary_metric="score",
        path=str(path),
        image_root=str(image_root) if image_root is not None else None,
        **kwargs,
    )


def _png_bytes(size=(8, 8)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_strip_sharegpt_question_removes_papo_suffix_and_leading_placeholder():
    assert strip_sharegpt_question("<image>What is x?" + PAPO_SUFFIX, 1) == "What is x?"
    # VPPO-Eval / single-newline variant
    assert (
        strip_sharegpt_question(
            "<image>What is x?\nYou first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}.",
            1,
        )
        == "What is x?"
    )
    # questions without the suffix are kept verbatim
    assert strip_sharegpt_question("<image>\nHow many?", 1) == "How many?"
    # interleaved placeholders keep their positions
    assert strip_sharegpt_question("Compare <image> and <image>." + PAPO_SUFFIX, 2) == "Compare <image> and <image>."
    assert strip_sharegpt_question("<image>A <image> B", 2) == "<image>A <image> B"


def test_sharegpt_image_relpath_maps_llama_factory_layout():
    assert sharegpt_image_relpath("./data/images/geo/geo_0_0.png") == "images/geo/geo_0_0.png"
    assert sharegpt_image_relpath("images/x.png") == "images/x.png"


def test_sharegpt_loader_reads_papo_eval_parquet_and_vppo_json(tmp_path):
    image_dir = tmp_path / "geo3k" / "images" / "hiyouga_geometry3k"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_dir / "hiyouga_geometry3k_0_0.png")
    rows = [
        {
            "messages": [
                {"role": "user", "content": "<image>Find $AC$." + PAPO_SUFFIX},
                {"role": "assistant", "content": "48"},
            ],
            "images": ["./data/images/hiyouga_geometry3k/hiyouga_geometry3k_0_0.png"],
            "id": "hiyouga_geometry3k_0",
        }
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "geo3k" / "test.parquet")
    spec = _spec("geo3k", "sharegpt", "boxed_exact_match", "geo3k/test.parquet", "geo3k")

    (sample,) = load_samples(spec, tmp_path)

    assert sample.sample_id == "hiyouga_geometry3k_0"
    assert sample.prompt == "Find $AC$."
    assert sample.target == "48"
    assert sample.images == [str(image_dir / "hiyouga_geometry3k_0_0.png")]
    assert sample.metadata["question"] == "Find $AC$."

    # Applying the training format prompt reproduces the PAPO-Eval prompt text exactly.
    rendered = apply_prompt_to_sample(sample, PromptConfig(format_prompt=FORMAT_PROMPT, prompt_mode="chat"))
    assert rendered.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": rows[0]["messages"][0]["content"][len("<image>") :]},
            ],
        }
    ]

    # VPPO-Eval json rows have no id
    (tmp_path / "dynamath" / "images" / "D").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "dynamath" / "images" / "D" / "a.png")
    (tmp_path / "dynamath" / "test.json").write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"content": "<image>Range?" + PAPO_SUFFIX, "role": "user"},
                        {"content": "12", "role": "assistant"},
                    ],
                    "images": ["./data/images/D/a.png"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (vppo,) = load_sharegpt(
        _spec("dynamath", "sharegpt", "boxed_exact_match", "dynamath/test.json", "dynamath"), tmp_path
    )
    assert vppo.sample_id == "dynamath:0"
    assert (vppo.prompt, vppo.target) == ("Range?", "12")


def test_sharegpt_loader_reports_missing_images(tmp_path):
    rows = [
        {
            "messages": [{"role": "user", "content": "<image>Q"}, {"role": "assistant", "content": "1"}],
            "images": ["./data/images/x/y.png"],
        }
    ]
    (tmp_path / "b").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "b" / "test.parquet")
    with pytest.raises(FileNotFoundError, match="missing benchmark image"):
        load_sharegpt(_spec("b", "sharegpt", "boxed_exact_match", "b/test.parquet", "b"), tmp_path)


def test_hallusionbench_loader_maps_binary_answers_and_ids(tmp_path):
    rows = []
    for figure_id, gt in (("0", "1"), ("1", "0")):
        rows.append(
            {
                "category": "VD",
                "subcategory": "illusion",
                "visual_input": "1",
                "set_id": "3",
                "figure_id": figure_id,
                "sample_note": "",
                "question_id": "0",
                "question": "Is the left line longer?",
                "gt_answer_details": "",
                "gt_answer": gt,
                "filename": "x.png",
                "image": {"bytes": _png_bytes(), "path": None},
            }
        )
    (tmp_path / "hallusionbench").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "hallusionbench" / "image.parquet")
    spec = _spec("hallusionbench", "hallusionbench", "hallusionbench", "hallusionbench/image.parquet")

    samples = load_hallusionbench(spec, tmp_path)

    assert [sample.target for sample in samples] == ["yes", "no"]
    assert samples[0].sample_id == "VD_illusion_3_0_0"
    assert samples[0].prompt == "Is the left line longer? Please answer yes or no."
    assert samples[1].metadata["figure_id"] == "1"
    assert len(samples[0].images) == 1


def test_vstar_loader_parses_official_prompt(tmp_path):
    (tmp_path / "vstar" / "direct_attributes").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "vstar" / "direct_attributes" / "sa_1.jpg")
    row = {
        "image": "direct_attributes/sa_1.jpg",
        "text": "What is the color of the cup?\n(A) red\n(B) blue\n(C) green\n(D) white\nAnswer with the option's letter from the given choices directly.",
        "category": "direct_attributes",
        "question_id": "7",
        "label": "B",
    }
    (tmp_path / "vstar" / "test_questions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    (sample,) = load_vstar(_spec("vstar", "vstar", "mcq", "vstar/test_questions.jsonl", "vstar"), tmp_path)

    assert sample.target == "B"
    assert sample.extra_info["options"] == ["red", "blue", "green", "white"]
    assert sample.prompt == (
        "What is the color of the cup?\n(A) red\n(B) blue\n(C) green\n(D) white\n"
        "Answer with the option's letter from the given choices."
    )
    assert sample.metadata["category"] == "direct_attributes"
    assert sample.sample_id == "direct_attributes:7"


def test_hrbench_loader_builds_mcq_prompt(tmp_path):
    (tmp_path / "hrbench_4k" / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "hrbench_4k" / "images" / "abc.png")
    row = {
        "index": 5,
        "question": "What is written on the sign?",
        "answer": "C",
        "category": "single",
        "A": "stop",
        "B": "go",
        "C": "exit",
        "D": "open",
        "cycle_category": "c0",
        "image": "images/abc.png",
    }
    (tmp_path / "hrbench_4k" / "annotations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    (sample,) = load_hrbench(
        _spec("hrbench_4k", "hrbench", "mcq", "hrbench_4k/annotations.jsonl", "hrbench_4k"), tmp_path
    )

    assert sample.sample_id == "5"
    assert sample.target == "C"
    assert sample.extra_info["options"] == ["stop", "go", "exit", "open"]
    assert "(C) exit" in sample.prompt
    assert sample.metadata["category"] == "single"


def test_refcoco_loader_keeps_normalized_box(tmp_path):
    (tmp_path / "refcoco" / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "refcoco" / "images" / "COCO_train2014_000000000001.jpg")
    row = {
        "sample_id": "1:0",
        "image": "images/COCO_train2014_000000000001.jpg",
        "expression": "left dog",
        "bbox": [0.1, 0.2, 0.5, 0.6],
    }
    (tmp_path / "refcoco" / "refcoco_val.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    (sample,) = load_refcoco(
        _spec("refcoco_val", "refcoco", "refcoco", "refcoco/refcoco_val.jsonl", "refcoco"), tmp_path
    )

    assert sample.extra_info["bbox"] == [0.1, 0.2, 0.5, 0.6]
    assert '"left dog"' in sample.prompt
    assert "[x1, y1, x2, y2]" in sample.prompt
    assert "normalized to 0-1000" in sample.prompt


def test_missing_data_files_reports_unprepared_benchmarks(tmp_path):
    spec = _spec("mme", "mme", "mme", "mme/*.parquet")
    assert missing_data_files(spec, tmp_path) == [tmp_path / "mme/*.parquet"]
    (tmp_path / "mme").mkdir()
    (tmp_path / "mme" / "test-0.parquet").write_bytes(b"x")
    assert missing_data_files(spec, tmp_path) == []
    with_images = _spec("geo3k", "sharegpt", "boxed_exact_match", "geo3k/test.parquet", "geo3k")
    assert set(missing_data_files(with_images, tmp_path)) == {tmp_path / "geo3k/test.parquet", tmp_path / "geo3k"}


def test_grit_loader_normalizes_pixel_boxes(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 100)).save(image_root / "sample.jpg")
    data_path = tmp_path / "tallyqa_val.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "question": "How many birds?",
                "answer": 1,
                "image": "sample.jpg",
                "width": 100,
                "height": 100,
                "bboxs": [[10, 20, 30, 40]],
                "dataset": "tallyqa",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        key="grit_tallyqa",
        label="GRIT-TallyQA",
        group="Grounding",
        loader="grit_jsonl",
        scorer="answer_bbox",
        primary_metric="answer_accuracy",
        path=str(data_path),
        image_root=str(image_root),
    )

    sample = load_grit_jsonl(spec, tmp_path)[0]

    assert sample.extra_info["bboxs_normalized"] == [[0.1, 0.2, 0.3, 0.4]]
    assert sample.extra_info["bboxs"] == [[10, 20, 30, 40]]
    assert sample.native_agentic_prompt == "How many birds?"
    assert "ground every image region needed" in sample.prompt
    assert sample.target == "1"
    assert sample.metadata["question"] == "How many birds?"


@pytest.mark.parametrize("width", [0, -1, "nan"])
def test_grit_loader_rejects_invalid_image_dimensions(tmp_path, width):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 100)).save(image_root / "sample.jpg")
    data_path = tmp_path / "tallyqa_val.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "question": "How many birds?",
                "answer": 1,
                "image": "sample.jpg",
                "width": width,
                "height": 100,
                "bboxs": [[10, 20, 30, 40]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        key="grit_tallyqa",
        label="GRIT-TallyQA",
        group="Grounding",
        loader="grit_jsonl",
        scorer="answer_bbox",
        primary_metric="answer_accuracy",
        path=str(data_path),
        image_root=str(image_root),
    )

    with pytest.raises(ValueError, match="invalid image dimensions"):
        load_grit_jsonl(spec, tmp_path)


def test_ovdeval_prompt_requires_normalized_1000_coordinates(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (100, 100)).save(image_root / "sample.jpg")
    data_path = tmp_path / "ovd_position_val.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "question": "Locate the object.",
                "answer": [10, 20, 30, 40],
                "image": "sample.jpg",
                "width": 100,
                "height": 100,
                "bboxs": [[10, 20, 30, 40]],
                "dataset": "ovd_position",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = BenchmarkSpec(
        key="ovdeval_position",
        label="OVDEval-position",
        group="Grounding",
        loader="grit_jsonl",
        scorer="grounding_iou",
        primary_metric="grounding_iou",
        path=str(data_path),
        image_root=str(image_root),
    )

    sample = load_grit_jsonl(spec, tmp_path)[0]

    assert "normalized to 0-1000" in sample.prompt
    assert sample.native_agentic_prompt is None
