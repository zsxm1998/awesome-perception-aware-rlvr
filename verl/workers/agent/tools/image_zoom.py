# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from math import ceil, floor
from typing import Any, Mapping, Optional, Sequence, Union

from PIL import Image

from ..coordinates import BBOX_FORMATS
from ..protocol import ToolResult
from .base import AgentTool, tool_error


TOOL_NAME = "image_zoom_in_tool"
NORMALIZED_COORDINATE_MAX = 1000
MIN_PIXEL_SIDE = 30
MAX_ASPECT_RATIO = 100
FIXED_GRAY_VALUE = 128
TEXT_SKIPPED_OBSERVATION = "[Image output skipped]"
OUTPUT_IMAGE_MODES = frozenset({"original", "fixed_gray", "text_skipped"})


class ImageZoomInTool(AgentTool):
    """DeepEyes zoom tool.

    ``bbox_format="norm1000"`` (Qwen3-VL, InternVL): ``bbox_2d`` holds integers normalized to 0-1000
    of the source image. ``bbox_format="pixel"`` (Qwen2-VL, Qwen2.5-VL): ``bbox_2d`` holds absolute
    pixel coordinates in the frame the model actually sees, given per source image by
    ``frame_sizes`` (``(width, height)`` after the data-side resize and the processor's
    ``smart_resize``); they are clipped to that frame, as in the official DeepEyes tool, and mapped
    back to the source image before cropping. Crops are always taken from the full-resolution source.
    """

    name = TOOL_NAME

    def __init__(
        self,
        *,
        output_image_mode: str = "original",
        bbox_format: str = "norm1000",
        frame_sizes: Optional[Sequence[tuple[int, int]]] = None,
    ) -> None:
        if output_image_mode not in OUTPUT_IMAGE_MODES:
            raise ValueError(f"image_zoom_in_tool output_image_mode must be one of {sorted(OUTPUT_IMAGE_MODES)}")
        if bbox_format not in BBOX_FORMATS:
            raise ValueError(f"image_zoom_in_tool bbox_format must be one of {list(BBOX_FORMATS)}")
        if bbox_format == "pixel" and frame_sizes is None:
            raise ValueError("bbox_format='pixel' requires the model-input frame size of every source image")
        self.output_image_mode = output_image_mode
        self.bbox_format = bbox_format
        self.frame_sizes = None if frame_sizes is None else [(int(w), int(h)) for w, h in frame_sizes]

    def schema(self, num_source_images: int) -> Mapping[str, Any]:
        if num_source_images <= 0:
            raise ValueError("image_zoom_in_tool requires at least one source image")

        if self.bbox_format == "pixel":
            bbox_schema: dict[str, Any] = {
                "type": "array",
                "items": {"type": "number", "minimum": 0},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Bounding box [x1, y1, x2, y2] in absolute pixel coordinates of the image as you see it, "
                    "with (x1, y1) the top-left and (x2, y2) the bottom-right corner."
                ),
            }
        else:
            bbox_schema = {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": NORMALIZED_COORDINATE_MAX,
                },
                "minItems": 4,
                "maxItems": 4,
                "description": ("Bounding box [x1, y1, x2, y2] in integer coordinates normalized to 0-1000."),
            }
        properties: dict[str, Any] = {
            "bbox_2d": bbox_schema,
            "label": {
                "type": "string",
                "description": "Optional name of the object or region to inspect.",
            },
            "image_idx": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Optional 0-based index into the original source images. "
                    "It defaults to 0 for a single source image and is required "
                    "at execution time when multiple source images are present."
                ),
            },
        }
        required = ["bbox_2d"]

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Zoom in on a region by cropping its original source image.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def execute(self, arguments: Mapping[str, Any], source_images: Sequence[Image.Image]) -> ToolResult:
        num_images = len(source_images)
        if num_images == 0:
            return tool_error(self.name, "no_source_images", "no source image is available", num_images=0)

        image_idx_or_error = self._resolve_image_idx(arguments, num_images)
        if isinstance(image_idx_or_error, ToolResult):
            return image_idx_or_error
        image_idx = image_idx_or_error

        if self.bbox_format == "pixel":
            if self.frame_sizes is None or len(self.frame_sizes) != num_images:
                return tool_error(
                    self.name,
                    "missing_frame_size",
                    "the model-input size of every source image is required for pixel coordinates",
                    num_images=num_images,
                )
            bbox_or_error = self._validate_pixel_bbox(arguments.get("bbox_2d"), self.frame_sizes[image_idx])
        else:
            bbox_or_error = self._validate_normalized_bbox(arguments.get("bbox_2d"))
        if isinstance(bbox_or_error, ToolResult):
            return bbox_or_error
        bbox = bbox_or_error

        label = arguments.get("label")
        if label is not None and not isinstance(label, str):
            return tool_error(self.name, "invalid_label", "label must be a string when provided")

        source_image = source_images[image_idx]
        if not isinstance(source_image, Image.Image):
            return tool_error(
                self.name,
                "invalid_source_image",
                f"source image {image_idx} is not a PIL image",
                image_idx=image_idx,
            )

        if self.bbox_format == "pixel":
            pixel_bbox = self._frame_to_source_bbox(
                bbox, self.frame_sizes[image_idx], source_image.width, source_image.height
            )
        else:
            pixel_bbox = self._to_pixel_bbox(bbox, source_image.width, source_image.height)
        bbox_norm1000 = [
            round(pixel_bbox[0] * NORMALIZED_COORDINATE_MAX / source_image.width),
            round(pixel_bbox[1] * NORMALIZED_COORDINATE_MAX / source_image.height),
            round(pixel_bbox[2] * NORMALIZED_COORDINATE_MAX / source_image.width),
            round(pixel_bbox[3] * NORMALIZED_COORDINATE_MAX / source_image.height),
        ]
        left, top, right, bottom = pixel_bbox
        width = right - left
        height = bottom - top
        # Source-image pixels are only meaningful to the model in the normalized convention; a
        # pixel-coordinate model reads coordinates in its own (resized) frame.
        error_detail: dict[str, Any] = {"image_idx": image_idx, "bbox_2d": bbox}
        if self.bbox_format == "norm1000":
            error_detail["pixel_bbox"] = pixel_bbox
        if width <= 0 or height <= 0:
            return tool_error(
                self.name,
                "degenerate_bbox",
                "bbox_2d maps to an empty pixel region",
                **error_detail,
            )
        if min(width, height) <= MIN_PIXEL_SIDE:
            return tool_error(
                self.name,
                "bbox_too_small",
                f"the shorter pixel side must be greater than {MIN_PIXEL_SIDE}",
                **error_detail,
            )
        if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
            return tool_error(
                self.name,
                "bbox_aspect_ratio",
                f"pixel aspect ratio must not exceed {MAX_ASPECT_RATIO}",
                **error_detail,
            )

        crop = None
        if self.output_image_mode != "text_skipped":
            crop = source_image.crop(pixel_bbox)
        if self.output_image_mode == "fixed_gray":
            assert crop is not None
            # Evaluation-only visual-information ablation: preserve the crop's
            # dimensions and protocol while returning a sample-independent
            # neutral gray image. Unlike a per-crop mean, this cannot leak the
            # selected region's color or brightness statistics.
            crop = Image.new(
                "RGB",
                crop.size,
                color=(FIXED_GRAY_VALUE,) * 3,
            )
        content: dict[str, Any] = {
            "status": "success",
            "tool_name": self.name,
            "image_idx": image_idx,
            "bbox_2d": bbox if self.bbox_format == "norm1000" else arguments.get("bbox_2d"),
        }
        if self.bbox_format == "norm1000":
            content["pixel_bbox"] = pixel_bbox
        content["message"] = (
            TEXT_SKIPPED_OBSERVATION if self.output_image_mode == "text_skipped" else "The requested crop is attached."
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            content=content,
            images=[] if crop is None else [crop],
            metadata={
                "image_idx": image_idx,
                "bbox_2d": bbox,
                "bbox_format": self.bbox_format,
                "pixel_bbox": pixel_bbox,
                "bbox_norm1000": bbox_norm1000,
                "label": label,
                "output_image_mode": self.output_image_mode,
            },
            observation_text=(TEXT_SKIPPED_OBSERVATION if self.output_image_mode == "text_skipped" else None),
        )

    def _resolve_image_idx(self, arguments: Mapping[str, Any], num_images: int) -> Union[int, ToolResult]:
        if "image_idx" not in arguments:
            if num_images == 1:
                return 0
            return self._image_idx_error(
                "missing_image_idx",
                "image_idx is required when multiple source images are present",
                num_images,
            )

        image_idx = arguments["image_idx"]
        if isinstance(image_idx, bool) or not isinstance(image_idx, int):
            return self._image_idx_error(
                "invalid_image_idx_type",
                "image_idx must be a JSON integer",
                num_images,
            )
        if image_idx < 0 or image_idx >= num_images:
            return self._image_idx_error(
                "image_idx_out_of_range",
                f"image_idx must be between 0 and {num_images - 1}",
                num_images,
            )
        return image_idx

    def _image_idx_error(self, code: str, message: str, num_images: int) -> ToolResult:
        return tool_error(
            self.name,
            code,
            message,
            num_images=num_images,
            valid_image_idx_range=[0, num_images - 1],
        )

    def _validate_normalized_bbox(self, bbox: Any) -> Union[list[int], ToolResult]:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return tool_error(
                self.name,
                "invalid_bbox_shape",
                "bbox_2d must be a JSON array containing exactly four integers",
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in bbox):
            return tool_error(
                self.name,
                "invalid_bbox_type",
                "bbox_2d coordinates must be JSON integers",
            )
        if any(value < 0 or value > NORMALIZED_COORDINATE_MAX for value in bbox):
            return tool_error(
                self.name,
                "bbox_out_of_range",
                "bbox_2d coordinates must be in the range 0-1000",
                bbox_2d=bbox,
            )

        left, top, right, bottom = bbox
        if left >= right or top >= bottom:
            return tool_error(
                self.name,
                "degenerate_bbox",
                "bbox_2d must satisfy x1 < x2 and y1 < y2",
                bbox_2d=bbox,
            )
        return list(bbox)

    def _validate_pixel_bbox(self, bbox: Any, frame_size: tuple[int, int]) -> Union[list[float], ToolResult]:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return tool_error(
                self.name,
                "invalid_bbox_shape",
                "bbox_2d must be a JSON array containing exactly four numbers",
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
            return tool_error(self.name, "invalid_bbox_type", "bbox_2d coordinates must be JSON numbers")
        frame_width, frame_height = frame_size
        left, top, right, bottom = (float(value) for value in bbox)
        # Clip to the visible frame like the official DeepEyes tool, then require a proper box.
        left, right = (min(max(value, 0.0), float(frame_width)) for value in (left, right))
        top, bottom = (min(max(value, 0.0), float(frame_height)) for value in (top, bottom))
        if left >= right or top >= bottom:
            return tool_error(
                self.name,
                "degenerate_bbox",
                "bbox_2d must satisfy x1 < x2 and y1 < y2 inside the image",
                bbox_2d=bbox,
                image_size=[frame_width, frame_height],
            )
        return [left, top, right, bottom]

    @staticmethod
    def _frame_to_source_bbox(
        bbox: Sequence[float], frame_size: tuple[int, int], width: int, height: int
    ) -> list[int]:
        frame_width, frame_height = frame_size
        scale_x = width / frame_width
        scale_y = height / frame_height
        left, top, right, bottom = bbox
        return [
            max(0, floor(left * scale_x)),
            max(0, floor(top * scale_y)),
            min(width, ceil(right * scale_x)),
            min(height, ceil(bottom * scale_y)),
        ]

    @staticmethod
    def _to_pixel_bbox(bbox: Sequence[int], width: int, height: int) -> list[int]:
        left, top, right, bottom = bbox
        return [
            floor(left * width / NORMALIZED_COORDINATE_MAX),
            floor(top * height / NORMALIZED_COORDINATE_MAX),
            ceil(right * width / NORMALIZED_COORDINATE_MAX),
            ceil(bottom * height / NORMALIZED_COORDINATE_MAX),
        ]
