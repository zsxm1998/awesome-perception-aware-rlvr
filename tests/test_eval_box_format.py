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

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import viz_server  # noqa: E402
from easyr1_eval import runner  # noqa: E402
from easyr1_eval.backends import image_size_records, model_input_image_size, qwen_smart_resize  # noqa: E402
from easyr1_eval.loaders import apply_box_format_to_prompts, refcoco_prompt  # noqa: E402
from easyr1_eval.schemas import BenchmarkSpec, EvalSample, GenerationOutput  # noqa: E402
from easyr1_eval.scorers import (  # noqa: E402
    predicted_boxes,
    refcoco_row_best_iou,
    score_answer_bbox,
    score_grounding_iou,
)


QWEN25_PROCESSOR = SimpleNamespace(
    image_processor=SimpleNamespace(patch_size=14, merge_size=2, min_pixels=3136, max_pixels=12845056)
)


def _spec(key, scorer, primary):
    return BenchmarkSpec(key=key, label=key, group="Grounding", loader="dummy", scorer=scorer, primary_metric=primary)


def _pixel_row(response, model_input, **extra):
    return {
        "sample_id": "s",
        "responses": [response],
        "eval_metadata": {"box_format": "pixel"},
        "image_sizes": [{"original": [1600, 1200], "processed": [1148, 861], "model_input": model_input}],
        **extra,
    }


@pytest.mark.parametrize(("height", "width"), [(1200, 1600), (861, 1148), (500, 375), (30, 4000), (4000, 3000)])
def test_qwen_smart_resize_matches_transformers(height, width):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    expected = smart_resize(height, width, factor=28, min_pixels=3136, max_pixels=1003520)
    assert qwen_smart_resize(height, width, factor=28, min_pixels=3136, max_pixels=1003520) == expected


def test_model_input_size_uses_the_processor_resize():
    # 1148x861 -> multiples of 28 (patch 14 x merge 2)
    assert model_input_image_size(1148, 861, QWEN25_PROCESSOR) == (1148, 868)
    # processors without Qwen-style patch merging do not resize further
    assert model_input_image_size(640, 480, SimpleNamespace(image_processor=SimpleNamespace())) == (640, 480)
    # size dict (newer transformers) instead of min/max attributes; Qwen3-VL uses 16 x 2 = 32
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            patch_size=16, merge_size=2, size={"shortest_edge": 65536, "longest_edge": 16777216}
        )
    )
    height, width = smart_resize(30, 1000, factor=32, min_pixels=65536, max_pixels=16777216)
    assert model_input_image_size(1000, 30, processor) == (width, height)


def test_image_size_records_keep_original_processed_and_model_input_sizes(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (1600, 1200)).save(path)
    sample = EvalSample(benchmark="b", sample_id="s", prompt="p", target="t", images=[str(path)])
    processed = Image.new("RGB", (1148, 861))

    (record,) = image_size_records(sample, [processed], QWEN25_PROCESSOR)

    assert record == {"original": [1600, 1200], "processed": [1148, 861], "model_input": [1148, 868]}


def test_pixel_boxes_are_mapped_back_with_the_model_input_size():
    row = _pixel_row("<answer>[112, 86.8, 574, 434]</answer>", [1120, 868])
    (box,) = predicted_boxes(row["responses"][0], row)
    assert box == pytest.approx((0.1, 0.1, 0.5125, 0.5))
    # boxes slightly outside the image are clipped instead of dropped
    (clipped,) = predicted_boxes("[0, 0, 1130, 870]", row)
    assert clipped == pytest.approx((0.0, 0.0, 1.0, 1.0))
    # norm1000 rows (and pixel rows without recorded sizes) keep the 0-1000 reading
    assert predicted_boxes("[100, 100, 500, 500]", {"eval_metadata": {"box_format": "norm1000"}}) == [
        (0.1, 0.1, 0.5, 0.5)
    ]
    assert predicted_boxes("[100, 100, 500, 500]", {"eval_metadata": {"box_format": "pixel"}}) == [
        (0.1, 0.1, 0.5, 0.5)
    ]


def test_grit_and_ovdeval_scoring_in_pixel_space(tmp_path):
    gt = {"bboxs": [[160, 120, 800, 600]], "bbox_format": "pixel_xyxy", "width": 1600, "height": 1200}
    # model saw the image at 800x600: the same object is at [80, 60, 400, 300]
    rows = [{**_pixel_row("[80, 60, 400, 300] <answer>cat</answer>", [800, 600]), "target": "cat", "extra_info": gt}]

    grit = score_answer_bbox(_spec("grit_vsr", "answer_bbox", "answer_accuracy"), rows, None, tmp_path)
    ovd = score_grounding_iou(_spec("ovd", "grounding_iou", "grit_iou"), rows, None, tmp_path)

    assert grit.raw_score == pytest.approx(100.0)
    assert grit.details["grounding/grit_iou"] == pytest.approx(1.0)
    assert grit.details["grounding/box_format"] == "pixel"
    assert grit.details["grounding/pixel_rows_without_image_size"] == 0
    assert ovd.raw_score == pytest.approx(100.0)


def test_refcoco_pixel_boxes():
    row = _pixel_row(r"\boxed{[80, 60, 400, 300]}", [800, 600], extra_info={"bbox": [0.1, 0.1, 0.5, 0.5]})
    best_iou, source = refcoco_row_best_iou(row)
    assert best_iou == pytest.approx(1.0)
    assert source == "final_answer"


def test_pixel_mode_rewrites_the_coordinate_instruction():
    samples = [
        EvalSample(benchmark="refcoco_val", sample_id="1", prompt=refcoco_prompt("left dog"), target="left dog"),
        EvalSample(benchmark="pope", sample_id="2", prompt="Is there a dog? Please answer yes or no.", target="yes"),
    ]
    pixel = apply_box_format_to_prompts(samples, "pixel")
    assert pixel[0].prompt.endswith("Return its bounding box as [x1, y1, x2, y2] using absolute pixel coordinates.")
    assert "0-1000" not in pixel[0].prompt
    assert pixel[1] is samples[1]
    assert apply_box_format_to_prompts(samples, "norm1000") is samples


def _args(tmp_path, model, *, system_prompt=None, format_prompt=None, box_format="auto"):
    return SimpleNamespace(
        model=str(model), system_prompt=system_prompt, format_prompt=format_prompt, box_format=box_format
    )


def test_resolve_box_format(tmp_path):
    qwen25 = tmp_path / "global_step_10" / "actor" / "huggingface"
    qwen25.mkdir(parents=True)
    (qwen25 / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    qwen3 = tmp_path / "qwen3"
    qwen3.mkdir()
    (qwen3 / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    norm_prompt = tmp_path / "grounded.jinja"
    norm_prompt.write_text("{{ content }} Use integer boxes normalized to the 0-1000 range.")
    plain_prompt = tmp_path / "plain.jinja"
    plain_prompt.write_text("{{ content }} Answer in \\boxed{}.")

    assert runner.resolve_box_format(_args(tmp_path, qwen25))[0] == "pixel"
    assert runner.resolve_box_format(_args(tmp_path, qwen25, format_prompt=str(plain_prompt)))[0] == "pixel"
    assert runner.resolve_box_format(_args(tmp_path, qwen3))[0] == "norm1000"
    # a prompt that asks for 0-1000 coordinates wins over the model default
    assert runner.resolve_box_format(_args(tmp_path, qwen25, format_prompt=str(norm_prompt))) == (
        "norm1000",
        "--format-prompt asks for 0-1000 coordinates",
    )
    # name heuristic when no config can be found, explicit values are kept
    assert runner.resolve_box_format(_args(tmp_path, "some-org/Qwen2.5-VL-3B-Instruct-not-cached"))[0] == "pixel"
    assert runner.resolve_box_format(_args(tmp_path, "some-org/unknown-model"))[0] == "norm1000"
    assert runner.resolve_box_format(_args(tmp_path, qwen25, box_format="norm1000")) == ("norm1000", "--box-format")
    # the repository's grounding prompts all ask for 0-1000 boxes
    for prompt in (
        "examples/format_prompt/xml_grounded_reasoning.jinja",
        "examples/system_prompt/grit_GR.txt",
        "examples/system_prompt/deepeyes.txt",
    ):
        key = "format_prompt" if "format_prompt" in prompt else "system_prompt"
        assert runner.resolve_box_format(_args(tmp_path, qwen25, **{key: str(ROOT / prompt)}))[0] == "norm1000"


def test_parse_args_records_box_format(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py", "--model", "m", "--box-format", "pixel"])
    args = runner.expand_eval_runs(runner.parse_args())[0]
    assert (args.box_format, args.box_format_requested) == ("pixel", "pixel")
    spec = _spec("grit_vsr", "answer_bbox", "answer_accuracy")
    assert runner.metric_metadata_for(spec, args)["box_format"] == "pixel"
    pixel_fp = runner.task_fingerprint(args, spec, phase="infer")
    args.box_format = "norm1000"
    assert runner.task_fingerprint(args, spec, phase="infer") != pixel_fp


def test_prediction_row_keeps_image_sizes():
    sample = EvalSample(benchmark="b", sample_id="s", prompt="p", target="t", images=["x.jpg"])
    sizes = [{"original": [1600, 1200], "processed": [1148, 861], "model_input": [1148, 868]}]
    row = runner.prediction_row(
        sample, [GenerationOutput(text="a", diagnostics={"perturbation": [], "image_sizes": sizes})]
    )
    assert row["image_sizes"] == sizes
    assert "image_sizes" not in runner.prediction_row(sample, [GenerationOutput(text="a")])


def test_viz_scales_pixel_boxes_to_the_display_canvas():
    groundings = [{"label": "dog", "bbox_list": [[80, 60, 400, 300]], "image_idx": 0}]
    record = {"eval_metadata": {"box_format": "pixel"}, "image_sizes": [{"model_input": [800, 600]}]}
    (scaled,) = viz_server._to_canvas_groundings(groundings, record)
    assert scaled["bbox_list"] == [pytest.approx([100.0, 100.0, 500.0, 500.0])]
    assert viz_server._to_canvas_groundings(groundings, {"eval_metadata": {"box_format": "norm1000"}}) is groundings
