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
"""Bounding-box coordinate conventions of the agent tools.

Qwen3-VL and InternVL ground objects with integer coordinates normalized to 0-1000. Qwen2-VL and
Qwen2.5-VL ground with *absolute pixel coordinates of the image they see*: the image after the
data-side resize (``min_pixels`` / ``max_pixels``) and the processor's ``smart_resize`` (rounding
to multiples of patch_size * merge_size). Those coordinates must be mapped back to the source image
before cropping, otherwise every crop is shifted and scaled by the resize ratio.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Optional


BBOX_FORMATS = ("norm1000", "pixel")
PIXEL_COORDINATE_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl"})
_PIXEL_NAME_HINTS = ("qwen2.5-vl", "qwen2_5_vl", "qwen2-vl", "qwen2_vl", "qwen2.5vl")


def _local_model_type(model_path: str) -> Optional[str]:
    """Read ``model_type`` from a local directory or the local HF cache (never the network)."""
    config_path = None
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, "config.json")
        config_path = candidate if os.path.isfile(candidate) else None
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_path, "config.json")
            config_path = cached if isinstance(cached, str) else None
        except Exception:
            config_path = None
    if config_path is None:
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("model_type")
    except (OSError, ValueError):
        return None


# Stock chat templates of these models ignore ``tools=...`` and render the ``tool`` role verbatim.
_NO_TOOL_TEMPLATE_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl"})
TOOL_CALL_TEMPLATE_HINT = "examples/chat_template/qwen2_5_vl_tool_call.jinja"


def check_chat_template_supports_tools(model_path: Optional[str], override_chat_template: Optional[str]) -> None:
    """Refuse agentic training of Qwen2-VL / Qwen2.5-VL with their stock (tool-less) chat template."""
    if override_chat_template is not None or not model_path:
        return
    if _local_model_type(model_path) in _NO_TOOL_TEMPLATE_MODEL_TYPES:
        raise ValueError(
            "the stock Qwen2-VL / Qwen2.5-VL chat template does not render tool definitions or tool responses; "
            f"set data.override_chat_template={TOOL_CALL_TEMPLATE_HINT}"
        )


def resolve_bbox_format(setting: str, model_path: Optional[str]) -> str:
    """Resolve ``auto`` to ``pixel`` for Qwen2-VL / Qwen2.5-VL and to ``norm1000`` otherwise."""
    if setting in BBOX_FORMATS:
        return setting
    if setting != "auto":
        raise ValueError(f"bbox format must be one of auto, {', '.join(BBOX_FORMATS)}; got {setting!r}")
    if not model_path:
        return "norm1000"
    model_type = _local_model_type(model_path)
    if model_type is not None:
        return "pixel" if model_type in PIXEL_COORDINATE_MODEL_TYPES else "norm1000"
    lowered = model_path.lower()
    return "pixel" if any(hint in lowered for hint in _PIXEL_NAME_HINTS) else "norm1000"


def check_prompt_matches_bbox_format(prompt_text: str, bbox_format: str) -> None:
    """Refuse a system prompt that describes the other coordinate convention."""
    text = prompt_text.lower()
    if bbox_format == "pixel" and "0-1000" in text:
        raise ValueError(
            "the agent tool expects absolute pixel coordinates (Qwen2-VL / Qwen2.5-VL), but the system prompt "
            "asks for coordinates normalized to 0-1000; use examples/system_prompt/deepeyes_pixel.txt or set "
            "worker.rollout.agent_bbox_format=norm1000"
        )
    if bbox_format == "norm1000" and "absolute pixel" in text:
        raise ValueError(
            "the agent tool expects coordinates normalized to 0-1000, but the system prompt asks for absolute "
            "pixel coordinates; use examples/system_prompt/deepeyes.txt or set worker.rollout.agent_bbox_format=pixel"
        )


def model_input_image_size(processor: Any, image: Any) -> tuple[int, int]:
    """``(width, height)`` of an already data-side-resized image as the vision encoder receives it.

    Mirrors the Qwen image processor's ``smart_resize`` with the processor's own pixel limits, i.e.
    the frame in which a Qwen2-VL / Qwen2.5-VL model reads and writes absolute coordinates.
    """
    width, height = image.size
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return int(width), int(height)
    module = importlib.import_module(type(image_processor).__module__)
    smart_resize = getattr(module, "smart_resize", None)
    if not callable(smart_resize):
        return int(width), int(height)
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    size = getattr(image_processor, "size", None)
    kwargs: dict[str, Any] = {"factor": patch_size * merge_size}
    if isinstance(size, dict) and "shortest_edge" in size and "longest_edge" in size:
        kwargs.update(min_pixels=int(size["shortest_edge"]), max_pixels=int(size["longest_edge"]))
    elif size is not None and hasattr(size, "shortest_edge"):
        kwargs.update(min_pixels=int(size.shortest_edge), max_pixels=int(size.longest_edge))
    else:
        for name in ("min_pixels", "max_pixels"):
            value = getattr(image_processor, name, None)
            if value is not None:
                kwargs[name] = int(value)
    resized_height, resized_width = smart_resize(int(height), int(width), **kwargs)
    return int(resized_width), int(resized_height)
