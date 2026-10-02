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
"""Agentic (DeepEyes) evaluation with Qwen2-VL / Qwen2.5-VL absolute pixel coordinates."""

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval import runner  # noqa: E402
from easyr1_eval.agentic import AgenticVLLMBackend, trajectory_to_generation_output  # noqa: E402
from easyr1_eval.schemas import BenchmarkSpec, EvalSample, GenerationConfig  # noqa: E402
from easyr1_eval.scorers import committed_tool_boxes, score_answer_bbox  # noqa: E402

from verl.workers.agent.coordinates import model_input_image_size  # noqa: E402
from verl.workers.agent.protocol import (  # noqa: E402
    AgentLoopConfig,
    AgentMetrics,
    AgentStatus,
    AgentStep,
    AgentTrajectory,
    ToolCall,
)
from verl.workers.agent.tools.image_zoom import ImageZoomInTool  # noqa: E402


DEEPEYES = ROOT / "examples/system_prompt/deepeyes.txt"
DEEPEYES_PIXEL = ROOT / "examples/system_prompt/deepeyes_pixel.txt"
TOOL_TEMPLATE = ROOT / "examples/chat_template/qwen2_5_vl_tool_call.jinja"


def _model_dir(tmp_path, model_type):
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return str(path)


def _parse(*argv):
    return runner.parse_args(["--output-dir", "/tmp/unused", *argv])


def _fingerprint(args, spec):
    (run_args,) = runner.expand_eval_runs(args)
    return runner.task_fingerprint(run_args, spec, phase="inference")


def test_suite_deepeyes_picks_pixel_prompt_and_tool_template_for_qwen2_5_vl(tmp_path):
    qwen25 = _model_dir(tmp_path, "qwen2_5_vl")
    args = _parse("--model", qwen25, "--suite", "deepeyes")
    assert (args.box_format, args.box_format_reason) == ("pixel", "agentic auto, model_type=qwen2_5_vl")
    assert Path(args.system_prompt) == DEEPEYES_PIXEL
    assert Path(args.chat_template) == TOOL_TEMPLATE

    # norm1000 keeps the 0-1000 prompt; the stock Qwen2.5-VL template still cannot render tools
    args = _parse("--model", qwen25, "--suite", "deepeyes", "--box-format", "norm1000")
    assert (args.box_format, Path(args.system_prompt)) == ("norm1000", DEEPEYES)
    assert Path(args.chat_template) == TOOL_TEMPLATE

    # Qwen3-VL: 0-1000 boxes, stock template
    qwen3 = _model_dir(tmp_path, "qwen3_vl")
    args = _parse("--model", qwen3, "--suite", "deepeyes")
    assert (args.box_format, Path(args.system_prompt), args.chat_template) == ("norm1000", DEEPEYES, None)


def test_prompt_and_template_must_match_the_box_format(tmp_path, capsys):
    qwen25 = _model_dir(tmp_path, "qwen2_5_vl")
    qwen3 = _model_dir(tmp_path, "qwen3_vl")
    # an explicit 0-1000 prompt is not swapped and conflicts with pixel boxes
    with pytest.raises(SystemExit):
        _parse("--model", qwen25, "--suite", "deepeyes", "--system-prompt", str(DEEPEYES))
    assert "--box-format norm1000" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _parse("--model", qwen3, "--suite", "deepeyes", "--system-prompt", str(DEEPEYES_PIXEL))
    assert "absolute pixel" in capsys.readouterr().err
    # the stock Qwen2.5-VL template ignores tools=
    with pytest.raises(SystemExit):
        _parse("--model", qwen25, "--suite", "deepeyes", "--chat-template", "none")
    assert "--chat-template" in capsys.readouterr().err

    args = _parse("--model", qwen25, "--interaction-mode", "agentic", "--system-prompt", str(DEEPEYES_PIXEL))
    assert args.box_format == "pixel" and Path(args.chat_template) == TOOL_TEMPLATE


def test_chat_template_is_fingerprinted_and_recorded_only_when_set(tmp_path):
    qwen25 = _model_dir(tmp_path, "qwen2_5_vl")
    spec = BenchmarkSpec(key="vstar", label="V*", group="HighRes", loader="vstar", scorer="mcq", primary_metric="acc")
    template_args = _parse("--model", qwen25, "--suite", "deepeyes")
    plain = _parse("--model", qwen25, "--suite", "deepeyes", "--chat-template", str(TOOL_TEMPLATE))
    assert _fingerprint(template_args, spec) == _fingerprint(plain, spec)
    one_shot = _parse("--model", qwen25)
    assert one_shot.chat_template is None
    other = _parse("--model", qwen25, "--chat-template", str(TOOL_TEMPLATE))
    assert _fingerprint(one_shot, spec) != _fingerprint(other, spec)


class _Tokenizer:
    all_special_ids = [0]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids if token_id)


def _zoom_step(tool, arguments, source):
    result = tool.execute(arguments, [source])
    assert result.success, result.content
    return AgentStep(
        model_text="<tool_call>zoom</tool_call>",
        model_token_ids=[1],
        tool_call=ToolCall(name="image_zoom_in_tool", arguments=arguments, raw_text="zoom"),
        tool_result=result,
        observation_committed=True,
    )


def _trajectory(steps, source):
    final = AgentStep(model_text="<answer>cat</answer>", model_token_ids=[*b"<answer>cat</answer>", 0], stop_reason=0)
    return AgentTrajectory(
        prompt_ids=[1],
        response_ids=[1],
        response_mask=[1],
        source_images=[source],
        observation_images=[],
        steps=[*steps, final],
        status=AgentStatus.ANSWERED,
        metrics=AgentMetrics(),
        final_answer="cat",
        effective_response_tokens=1,
        response_token_budget=32,
        assistant_termination_token_ids=(0,),
    )


def test_pixel_and_norm1000_tool_calls_on_the_same_region_score_identically(tmp_path):
    source = Image.new("RGB", (1600, 1200))
    # the model sees the image at 800x600 (data-side resize + smart_resize)
    pixel_tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[(800, 600)])
    norm_tool = ImageZoomInTool(bbox_format="norm1000")
    pixel_run = _trajectory([_zoom_step(pixel_tool, {"bbox_2d": [80, 60, 400, 300]}, source)], source)
    norm_run = _trajectory([_zoom_step(norm_tool, {"bbox_2d": [100, 100, 500, 500]}, source)], source)

    rows = []
    for trajectory, fmt in ((pixel_run, "pixel"), (norm_run, "norm1000")):
        output = trajectory_to_generation_output(trajectory, tokenizer=_Tokenizer())
        (region,) = output.diagnostics["agent"]["committed_tool_regions"]
        assert region["bbox_norm1000"] == [100, 100, 500, 500] and region["bbox_format"] == fmt
        rows.append(
            {
                "sample_id": fmt,
                "target": "cat",
                "responses": [output.text],
                "agent_diagnostics": [output.diagnostics["agent"]],
                "eval_metadata": {"box_format": fmt, "interaction_mode": "agentic", "agent_output_contract": "native"},
                "extra_info": {
                    "bboxs": [[160, 120, 800, 600]],
                    "bbox_format": "pixel_xyxy",
                    "width": 1600,
                    "height": 1200,
                },
            }
        )
    assert rows[0]["agent_diagnostics"][0]["committed_tool_regions"][0]["bbox_2d"] == [80, 60, 400, 300]
    assert committed_tool_boxes(rows[0]) == committed_tool_boxes(rows[1]) == [(0.1, 0.1, 0.5, 0.5)]

    spec = BenchmarkSpec(
        key="grit_vsr",
        label="g",
        group="Grounding",
        loader="x",
        scorer="answer_bbox",
        primary_metric="answer_accuracy",
    )
    pixel_result = score_answer_bbox(spec, rows[:1], None, tmp_path / "p")
    norm_result = score_answer_bbox(spec, rows[1:], None, tmp_path / "n")
    assert pixel_result.details["grounding/grit_iou"] == norm_result.details["grounding/grit_iou"] == 1.0


def test_legacy_regions_without_bbox_norm1000_are_read_as_0_1000():
    row = {
        "sample_id": "s",
        "agent_diagnostics": [
            {"status": "answered", "committed_tool_regions": [{"image_idx": 0, "bbox_2d": [0, 0, 500, 1000]}]}
        ],
    }
    assert committed_tool_boxes(row) == [(0.0, 0.0, 0.5, 1.0)]


def test_agentic_backend_passes_bbox_format_and_records_model_frames(monkeypatch):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    class Scheduler:
        def __init__(self, *args, **kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr("easyr1_eval.agentic.VLLMAgentBatchScheduler", Scheduler)
    processor = SimpleNamespace(image_processor=Qwen2VLImageProcessor(min_pixels=3136, max_pixels=12845056))
    backend = object.__new__(AgenticVLLMBackend)
    backend.agent_profile = "deepeyes"
    backend.agent_config = AgentLoopConfig(max_response_tokens=32)
    backend.agent_max_images_per_prompt = 8
    backend.agent_max_batch_images = 8
    backend.agent_tool_image_mode = "original"
    backend.agent_bbox_format = "pixel"
    backend.processor = processor
    backend.tokenizer = _Tokenizer()
    backend.min_pixels = 200704
    backend.max_pixels = 1003520
    backend.max_model_len = 4096
    backend.base = SimpleNamespace(llm=object())
    seen = {}

    async def fake_runner(**kwargs):
        seen.update(kwargs)
        source = kwargs["source_images"][0]
        return _trajectory([], source)

    backend.agent_runner = fake_runner
    sample = EvalSample(
        "vstar", "0", "q", "A", images=[Image.new("RGB", (2246, 1500))], messages=[{"role": "user", "content": "q"}]
    )
    (outputs,) = backend.generate(
        [sample], GenerationConfig(temperature=0.0, top_p=1.0, num_samples=1, max_new_tokens=32, seed=1)
    )

    assert seen["bbox_format"] == "pixel"
    (record,) = outputs[0].diagnostics["image_sizes"]
    prepared = seen["image_cache"].prepare(seen["source_images"][0])
    # the recorded model-input frame is the one the zoom-in tool maps pixel boxes from
    assert tuple(record["model_input"]) == model_input_image_size(processor, prepared)
    assert record["original"] == [2246, 1500] and record["processed"] == list(prepared.size)


def test_build_backend_forwards_bbox_format_and_chat_template(monkeypatch):
    from easyr1_eval import backends

    captured = {}

    class FakeAgentic:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("easyr1_eval.agentic.AgenticVLLMBackend", FakeAgentic)
    args = Namespace(
        backend="vllm",
        model="m",
        tp=1,
        max_model_len=None,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        min_pixels=None,
        max_pixels=None,
        perturbation=None,
        perturbation_seed=0,
        output_dir=None,
        perturbation_save_samples=0,
        interaction_mode="agentic",
        agent_profile="deepeyes",
        agent_max_tool_calls=2,
        agent_max_response_tokens=64,
        agent_max_tokens_per_turn=32,
        agent_max_images_per_prompt=8,
        max_batch_images=8,
        agent_tool_image_mode="original",
        box_format="pixel",
        chat_template=str(TOOL_TEMPLATE),
    )
    assert backends.build_backend is runner.build_backend
    runner.build_eval_backend(args)
    assert captured["agent_bbox_format"] == "pixel"
    assert captured["chat_template"] == str(TOOL_TEMPLATE)


def test_failed_tool_calls_keep_their_error_codes():
    from easyr1_eval.scorers import generation_diagnostics

    source = Image.new("RGB", (1600, 1200))
    tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[(800, 600)])
    failed = tool.execute({"bbox_2d": [10, 10, 12, 12]}, [source])
    assert not failed.success
    step = AgentStep(
        model_text="<tool_call>zoom</tool_call>",
        model_token_ids=[1],
        tool_call=ToolCall(name="image_zoom_in_tool", arguments={"bbox_2d": [10, 10, 12, 12]}, raw_text="zoom"),
        tool_result=failed,
        observation_committed=True,
    )
    output = trajectory_to_generation_output(_trajectory([step], source), tokenizer=_Tokenizer())
    agent = output.diagnostics["agent"]
    assert agent["committed_tool_regions"] == []
    (error,) = agent["tool_errors"]
    assert (error["turn_index"], error["error_code"]) == (0, "bbox_too_small") and "shorter pixel side" in error[
        "message"
    ]
    details = generation_diagnostics([{"responses": [output.text], "agent_diagnostics": [agent]}])
    assert details["agent/tool_error_codes"] == {failed.error_code: 1}
