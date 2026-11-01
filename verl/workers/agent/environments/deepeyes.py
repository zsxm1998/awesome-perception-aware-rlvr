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
"""DeepEyes' zoom-in task as an in-process environment."""

from __future__ import annotations

from typing import Any, Mapping

from ....models.transformers.internvl import is_internvl_processor
from ..chat import NativeToolChatAdapter, OfficialDeepEyesChatAdapter, PlainChatAdapter
from ..coordinates import BBOX_FORMATS, model_input_image_size
from ..protocol import AgentImageConfig, EnvReset
from ..tools import ImageZoomInTool, ToolRegistry
from ..tools.image_zoom import OUTPUT_IMAGE_MODES
from .local import LocalToolEnvironment
from .registry import EnvironmentContext, register_environment


PROMPT_STYLES = ("native", "official")


@register_environment("deepeyes")
class DeepEyesEnvironment(LocalToolEnvironment):
    """The data row is the task: its chat messages and source images; ``image_zoom_in_tool`` crops a source image.

    ``reset`` config keys: ``messages``, ``images`` and ``tools_enabled`` (False for DeepEyes rows without an
    environment, e.g. ThinkLite, which answer in one turn without tools). ``prompt_style`` selects the native
    tool template or DeepEyes' own prompt format; ``tool_image_mode`` is the evaluation-only ablation of what a
    zoom-in returns.
    """

    def __init__(
        self,
        context: EnvironmentContext,
        *,
        bbox_format: str = "norm1000",
        prompt_style: str = "native",
        tool_image_mode: str = "original",
    ) -> None:
        super().__init__()
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be 'native' or 'official', got {prompt_style!r}")
        if bbox_format not in BBOX_FORMATS:
            raise ValueError(f"bbox_format must be one of {list(BBOX_FORMATS)}, got {bbox_format!r}")
        if tool_image_mode not in OUTPUT_IMAGE_MODES:
            raise ValueError(f"tool_image_mode must be one of {sorted(OUTPUT_IMAGE_MODES)}, got {tool_image_mode!r}")
        if context.max_tool_calls_per_turn != 1:
            raise ValueError(
                "DeepEyes runs one tool call per turn (its prompts say so); set max_tool_calls_per_turn=1"
            )
        self.context = context
        self.bbox_format = bbox_format
        self.prompt_style = prompt_style
        self.tool_image_mode = tool_image_mode

    async def reset(self, config: Mapping[str, Any], seed: int) -> EnvReset:
        images = list(config["images"])
        tools_enabled = bool(config.get("tools_enabled", True))
        frame_sizes = None
        if tools_enabled and self.bbox_format == "pixel":
            if is_internvl_processor(self.context.processor):
                raise ValueError("InternVL grounds with 0-1000 coordinates; use bbox_format='norm1000'")
            # The frame a Qwen2-VL / Qwen2.5-VL policy reads absolute coordinates in.
            frame_sizes = [
                model_input_image_size(self.context.processor, self.context.image_preprocessor(image))
                for image in images
            ]
        self.registry = ToolRegistry(
            [
                ImageZoomInTool(
                    output_image_mode=self.tool_image_mode,
                    bbox_format=self.bbox_format,
                    frame_sizes=frame_sizes,
                )
            ]
            if tools_enabled
            else []
        )
        self.source_images = images
        return EnvReset(
            messages=list(config["messages"]),
            images=images,
            tool_schemas=self.registry.schemas(len(images)),
            max_images_per_step=0 if self.tool_image_mode == "text_skipped" else 1,
        )

    def chat_adapter(
        self,
        processor: Any,
        reset: EnvReset,
        *,
        max_tool_calls: int,
        image_config: AgentImageConfig,
        image_preprocessor: Any,
    ) -> Any:
        if not reset.tool_schemas:
            return PlainChatAdapter(
                processor,
                len(reset.images),
                image_config=image_config,
                image_preprocessor=image_preprocessor,
            )
        adapter_cls = OfficialDeepEyesChatAdapter if self.prompt_style == "official" else NativeToolChatAdapter
        return adapter_cls(
            processor,
            self.registry,
            len(reset.images),
            max_tool_calls=max_tool_calls,
            image_config=image_config,
            image_preprocessor=image_preprocessor,
        )
