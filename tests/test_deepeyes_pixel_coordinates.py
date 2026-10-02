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
"""Qwen2-VL / Qwen2.5-VL absolute pixel coordinates in the DeepEyes zoom-in tool."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from verl.utils.dataset import process_image
from verl.workers.agent import NativeToolChatAdapter
from verl.workers.agent.coordinates import (
    check_chat_template_supports_tools,
    check_prompt_matches_bbox_format,
    model_input_image_size,
    resolve_bbox_format,
)
from verl.workers.agent.protocol import ObservationEncodingRequest
from verl.workers.agent.tools import ImageZoomInTool, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
TOOL_TEMPLATE = ROOT / "examples" / "chat_template" / "qwen2_5_vl_tool_call.jinja"
NORM_PROMPT = (ROOT / "examples" / "system_prompt" / "deepeyes.txt").read_text()
PIXEL_PROMPT = (ROOT / "examples" / "system_prompt" / "deepeyes_pixel.txt").read_text()


def _model_dir(tmp_path, model_type):
    path = tmp_path / model_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(path)


def test_resolve_bbox_format_reads_model_type(tmp_path):
    assert resolve_bbox_format("auto", _model_dir(tmp_path, "qwen2_5_vl")) == "pixel"
    assert resolve_bbox_format("auto", _model_dir(tmp_path, "qwen2_vl")) == "pixel"
    assert resolve_bbox_format("auto", _model_dir(tmp_path, "qwen3_vl")) == "norm1000"
    # an explicit setting always wins
    assert resolve_bbox_format("norm1000", _model_dir(tmp_path, "qwen2_5_vl_x")) == "norm1000"
    assert resolve_bbox_format("auto", None) == "norm1000"
    with pytest.raises(ValueError, match="bbox format"):
        resolve_bbox_format("xyxy", None)


def test_resolve_bbox_format_falls_back_to_the_model_name():
    assert resolve_bbox_format("auto", "missing-org/Qwen2.5-VL-7B-Instruct-not-cached") == "pixel"
    assert resolve_bbox_format("auto", "missing-org/Qwen3-VL-8B-Instruct-not-cached") == "norm1000"


def test_system_prompt_must_match_the_coordinate_convention():
    check_prompt_matches_bbox_format(NORM_PROMPT, "norm1000")
    check_prompt_matches_bbox_format(PIXEL_PROMPT, "pixel")
    with pytest.raises(ValueError, match="deepeyes_pixel.txt"):
        check_prompt_matches_bbox_format(NORM_PROMPT, "pixel")
    with pytest.raises(ValueError, match="deepeyes.txt"):
        check_prompt_matches_bbox_format(PIXEL_PROMPT, "norm1000")


def test_stock_qwen25_template_is_refused_for_agentic_training(tmp_path):
    qwen25 = _model_dir(tmp_path, "qwen2_5_vl")
    with pytest.raises(ValueError, match="qwen2_5_vl_tool_call.jinja"):
        check_chat_template_supports_tools(qwen25, None)
    check_chat_template_supports_tools(qwen25, str(TOOL_TEMPLATE))
    check_chat_template_supports_tools(_model_dir(tmp_path, "qwen3_vl"), None)


@pytest.mark.parametrize("size", [(3000, 2000), (640, 480), (37, 1500), (5000, 90), (28, 28)])
def test_model_input_image_size_matches_the_processor_grid(size):
    image_processor = Qwen2VLImageProcessor()
    processor = SimpleNamespace(image_processor=image_processor)
    image = process_image(Image.new("RGB", size), 3136, 1003520)
    grid_t, grid_h, grid_w = image_processor(images=[image], return_tensors="pt")["image_grid_thw"][0].tolist()
    del grid_t
    assert model_input_image_size(processor, image) == (
        grid_w * image_processor.patch_size,
        grid_h * image_processor.patch_size,
    )


def test_pixel_schema_describes_absolute_coordinates():
    tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[(1232, 812)])
    bbox_schema = tool.schema(1)["function"]["parameters"]["properties"]["bbox_2d"]
    assert bbox_schema["items"]["type"] == "number"
    assert "absolute pixel" in bbox_schema["description"]
    with pytest.raises(ValueError, match="frame size"):
        ImageZoomInTool(bbox_format="pixel")


def test_pixel_bbox_is_mapped_from_the_model_frame_to_the_source_image():
    # The source image is 3000x2000; the model sees it as 1232x812 after resizing.
    tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[(1232, 812)])
    source = Image.new("RGB", (3000, 2000))
    result = tool.execute({"bbox_2d": [616, 406, 1232, 812]}, [source])

    assert result.success
    assert result.metadata["pixel_bbox"] == [1500, 1000, 3000, 2000]
    assert result.metadata["bbox_norm1000"] == [500, 500, 1000, 1000]
    assert result.metadata["bbox_format"] == "pixel"
    assert result.images[0].size == (1500, 1000)
    # the model only ever sees coordinates of its own frame
    assert result.content["bbox_2d"] == [616, 406, 1232, 812]
    assert "pixel_bbox" not in result.content


def test_pixel_bbox_is_clipped_to_the_frame_and_uses_the_selected_image():
    tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[(100, 100), (1232, 812)])
    sources = [Image.new("RGB", (100, 100)), Image.new("RGB", (3000, 2000))]
    result = tool.execute({"bbox_2d": [-50, -10, 2000, 406.5], "image_idx": 1}, sources)

    assert result.success
    assert result.metadata["pixel_bbox"] == [0, 0, 3000, 1002]  # ceil(406.5 * 2000 / 812)


def test_pixel_bbox_errors_do_not_leak_source_pixels():
    tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[(1232, 812)])
    source = Image.new("RGB", (3000, 2000))

    degenerate = tool.execute({"bbox_2d": [500, 500, 400, 600]}, [source])
    assert not degenerate.success and degenerate.error_code == "degenerate_bbox"
    assert degenerate.content["image_size"] == [1232, 812]

    too_small = tool.execute({"bbox_2d": [10, 10, 15, 15]}, [source])
    assert not too_small.success and too_small.error_code == "bbox_too_small"
    assert "pixel_bbox" not in too_small.content

    wrong_type = tool.execute({"bbox_2d": [10, 10, "20", 20]}, [source])
    assert wrong_type.error_code == "invalid_bbox_type"


def test_tool_template_matches_the_stock_template_without_tools():
    pytest.importorskip("jinja2")
    from transformers.utils.chat_template_utils import render_jinja_template

    stock = (
        "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in "
        "messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful "
        "assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string "
        "%}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if "
        "content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = "
        "image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif "
        "%}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content "
        "%}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value "
        "}}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ "
        "content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt "
        "%}<|im_start|>assistant\n{% endif %}"
    )
    conversations = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What?"}]}],
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Q"}]},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "more"},
        ],
    ]
    for generation_prompt in (False, True):
        expected, _ = render_jinja_template(
            conversations=conversations, chat_template=stock, add_generation_prompt=generation_prompt
        )
        actual, _ = render_jinja_template(
            conversations=conversations,
            chat_template=TOOL_TEMPLATE.read_text(),
            add_generation_prompt=generation_prompt,
        )
        assert actual == expected


def _qwen25_vl_path():
    candidates = [
        os.environ.get("QWEN25_VL_PROCESSOR_PATH"),
        os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct"),
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "preprocessor_config.json")):
            return candidate
        try:
            from huggingface_hub import try_to_load_from_cache

            if isinstance(try_to_load_from_cache(candidate, "preprocessor_config.json"), str):
                return candidate
        except Exception:
            continue
    return None


@pytest.fixture(scope="module")
def qwen25_vl_processor():
    path = _qwen25_vl_path()
    if path is None:
        pytest.skip("Qwen2.5-VL processor files are not available locally")
    from verl.utils.tokenizer import get_processor

    return get_processor(path, override_chat_template=str(TOOL_TEMPLATE), use_fast=True), path


def _messages(prompt):
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the cup?"}]},
    ]


def test_qwen25_vl_adapter_renders_tools_and_tool_responses(qwen25_vl_processor):
    processor, _ = qwen25_vl_processor
    source = Image.new("RGB", (3000, 2000))
    frame = model_input_image_size(processor, process_image(source, 3136, 1003520))
    tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[frame])
    adapter = NativeToolChatAdapter(processor, ToolRegistry([tool]), 1, max_tool_calls=3)

    prompt = adapter.encode_initial_prompt(_messages(PIXEL_PROMPT))
    assert "<tools>\n" in prompt.rendered_text and "image_zoom_in_tool" in prompt.rendered_text
    assert "absolute pixel coordinates of the image as you see it" in prompt.rendered_text
    assert prompt.rendered_text.endswith("<|im_start|>assistant\n")
    assert adapter.assistant_termination_token_ids == (processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),)

    result = tool.execute({"bbox_2d": [0, 0, frame[0] // 2, frame[1] // 2]}, [source])
    observation = asyncio.run(adapter.encode(ObservationEncodingRequest(result=result)))
    text = processor.tokenizer.decode(observation.token_ids)
    assert text.startswith("\n<|im_start|>user\n<tool_response>\n<|vision_start|><|image_pad|><|vision_end|>{")
    assert text.endswith("\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
    assert observation.visual_token_count and observation.visual_token_count > 0


def test_qwen25_vl_stock_template_is_refused_by_the_adapter(qwen25_vl_processor):
    _, path = qwen25_vl_processor
    from verl.utils.tokenizer import get_processor

    stock = get_processor(path, use_fast=True)
    tool = ImageZoomInTool(bbox_format="pixel", frame_sizes=[(1232, 812)])
    adapter = NativeToolChatAdapter(stock, ToolRegistry([tool]), 1, max_tool_calls=3)
    with pytest.raises(RuntimeError, match="override_chat_template"):
        adapter.encode_initial_prompt(_messages(PIXEL_PROMPT))
