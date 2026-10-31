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
"""SFT samples for ms-swift from replayed agent episodes (``replay.replay_episode``), tokenized like the rollout.

A sample is ms-swift's multimodal chat format: ``messages`` with system, user, assistant and tool roles, ``<image>``
tags where the images go and ``images`` in that order. To make ms-swift encode it exactly as the rollout did:

- the tool definitions go in ms-swift's ``tools`` field (a JSON string, which its dataset loader keeps as written)
  and the system message as the environment wrote it: ms-swift then renders the tool descriptions as the chat
  template does. Without a system message ms-swift would add two newlines before them, so such a sample carries
  the system block that the chat template rendered instead, tool descriptions included;
- assistant contents are the turns' texts as the policy wrote them; tool contents are the observations as the
  environment's chat adapter renders them (ms-swift's hermes agent template, the default for Qwen models, wraps them
  in the same ``<tool_response>`` turn as the chat template);
- images are saved at the size the model's image processor resizes them to, resized the way it resizes them, so
  neither side resizes them again.

``scripts/agent/check_swift_export.py`` encodes the samples with ms-swift and compares them with the rollout tokens
recorded next to them (``rollout`` below).
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from .episode import EpisodeRun
from .protocol import AgentStatus


SWIFT_MEDIA_TAGS = ("<image>", "<video>", "<audio>")
_CHATML_SYSTEM_OPEN = "<|im_start|>system\n"
_CHATML_TURN_END = "<|im_end|>\n"


class ExportError(ValueError):
    """An episode that cannot be written as an SFT sample with the rollout's tokens."""


@dataclass
class SwiftSample:
    record: dict[str, Any]  # one JSONL line for ms-swift, without its "images"
    images: list[Image.Image]  # in the order of the <image> tags, at the model's input size
    rollout: dict[str, Any] = field(default_factory=dict)  # the rollout's tokens and image grids, for the check


def model_input_image(processor: Any, prepared: Image.Image) -> Image.Image:
    """A prepared rollout image at the size the image processor gives the model, resized the way it resizes.

    The rollout prepares an image (the data-side pixel bounds) and the processor then resizes it to its patch grid.
    Saving the result lets a training framework read the same pixels without resizing.
    """
    prepared = prepared.convert("RGB")
    image_processor = getattr(processor, "image_processor", None)
    module = importlib.import_module(type(image_processor).__module__) if image_processor is not None else None
    smart_resize = getattr(module, "smart_resize", None)
    size = getattr(image_processor, "size", None)
    if not callable(smart_resize) or size is None or not hasattr(image_processor, "merge_size"):
        raise ExportError("SFT export supports Qwen-style image processors (smart_resize to a patch grid)")
    min_pixels = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
    max_pixels = size["longest_edge"] if isinstance(size, dict) else size.longest_edge
    height, width = smart_resize(
        prepared.height,
        prepared.width,
        factor=int(image_processor.patch_size) * int(image_processor.merge_size),
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    if (width, height) == prepared.size:
        return prepared
    from transformers.image_utils import SizeDict, pil_torch_interpolation_mapping

    if not hasattr(image_processor, "rescale_and_normalize"):  # a slow (PIL) processor resizes differently
        raise ExportError("SFT export needs the fast image processor (the one training uses, use_fast=True)")
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).unsqueeze(0).contiguous()
    resized = image_processor.resize(
        image=pixels,
        size=SizeDict(height=height, width=width),
        interpolation=pil_torch_interpolation_mapping[image_processor.resample],
    )
    return Image.fromarray(resized[0].permute(1, 2, 0).numpy())


def image_sha256(image: Image.Image) -> str:
    """The SHA-256 of an image's RGB pixels (not of its file)."""
    import hashlib

    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def check_model_input_image(processor: Any, prepared: Any, final: Image.Image) -> None:
    """The image processor gives the same patches for the rollout's image and the saved one."""
    image_processor = processor.image_processor
    original = image_processor(images=[prepared], return_tensors="pt")
    saved = image_processor(images=[final], return_tensors="pt")
    if not torch.equal(original["image_grid_thw"], saved["image_grid_thw"]) or not torch.equal(
        original["pixel_values"], saved["pixel_values"]
    ):
        raise ExportError("the saved image does not reproduce the processor's input for the rollout image")


def swift_sample(
    run: EpisodeRun,
    processor: Any,
    image_preprocessor: Callable[[Any], Any],
    *,
    check_images: bool = True,
) -> SwiftSample:
    """The ms-swift sample of a replayed episode, with the rollout's tokens for the check."""
    adapter = run.chat_adapter
    trajectory = run.trajectory
    if not hasattr(adapter, "tool_message_content"):
        raise ExportError(
            "SFT export supports environments rendered by EnvironmentChatAdapter (remote_http and environments that "
            f"use it); this one uses {type(adapter).__name__}"
        )
    rendered = adapter.encode_initial_prompt(run.reset.messages).rendered_text
    messages: list[dict[str, Any]] = []
    if rendered.startswith(_CHATML_SYSTEM_OPEN):
        end = rendered.find(_CHATML_TURN_END)
        if end < 0:
            raise ExportError("the rendered system block has no end")
        messages.append({"role": "system", "content": rendered[len(_CHATML_SYSTEM_OPEN) : end]})
    elif any(message.get("role") == "system" for message in run.reset.messages) or adapter.tools:
        raise ExportError("SFT export supports ChatML chat templates (<|im_start|>system ... <|im_end|>)")
    record: dict[str, Any] = {}
    system = next((message for message in run.reset.messages if message.get("role") == "system"), None)
    if adapter.tools and system is not None and isinstance(system.get("content"), str):
        # ms-swift renders the tool descriptions from its tools field as the chat template does after a system
        # message (a JSON string, so that the dataset loader keeps each schema as written)
        messages[0] = {"role": "system", "content": system["content"]}
        record["tools"] = json.dumps(list(adapter.tools), ensure_ascii=False)
    for message in run.reset.messages:
        if message.get("role") == "system":
            continue
        messages.append({"role": message["role"], "content": _swift_content(message.get("content"))})
    for step in trajectory.steps:
        messages.append({"role": "assistant", "content": step.model_text})
        if step.observation_committed:
            messages.append(
                {"role": "tool", "content": _swift_content(adapter.tool_message_content(step.tool_result))}
            )
    for message in messages:
        for tag in SWIFT_MEDIA_TAGS:
            if tag in message["content"] and not (tag == "<image>" and message["role"] in {"user", "tool"}):
                raise ExportError(f"a {message['role']} text contains {tag!r}, which ms-swift reads as a media tag")
    num_tags = sum(message["content"].count("<image>") for message in messages)
    images = trajectory.all_images
    if num_tags != len(images):
        raise ExportError(
            f"the sample has {num_tags} <image> tags for {len(images)} images; an environment text may contain "
            "'<image>', which ms-swift reads as an image"
        )
    final_images = []
    for image in images:
        prepared = image_preprocessor(image)
        prepared = getattr(prepared, "image", prepared)
        final = model_input_image(processor, prepared)
        if check_images:
            check_model_input_image(processor, prepared, final)
        final_images.append(final)
    image_processor = processor.image_processor
    grids = (
        image_processor(images=final_images, return_tensors="pt")["image_grid_thw"].tolist() if final_images else []
    )
    rollout = {
        # the pixels of each saved image, so that the check also catches an image file that changed after export
        "image_sha256": [image_sha256(image) for image in final_images],
        "input_ids": [*trajectory.prompt_ids, *trajectory.response_ids],
        "loss_mask": [0] * len(trajectory.prompt_ids) + list(trajectory.response_mask),
        "image_grid_thw": grids,
        "image_token_id": int(processor.image_token_id),
        "assistant_termination_token_ids": list(trajectory.assistant_termination_token_ids),
    }
    return SwiftSample(record={"messages": messages, **record}, images=final_images, rollout=rollout)


def keep_episode(run: EpisodeRun, *, keep: str = "answered", min_reward: Optional[float] = None) -> Optional[str]:
    """Why an episode is left out of the export, or None to keep it."""
    trajectory = run.trajectory
    if keep == "answered" and trajectory.status != AgentStatus.ANSWERED:
        return f"status {trajectory.status.value}"
    if min_reward is not None:
        outcome = trajectory.env_outcome
        if outcome is None or outcome.reward is None:
            return "no episode reward"
        # the episode's return: its reward plus the rewards the environment gave for single tool calls
        step_rewards = sum(step.env_reward for step in trajectory.steps if step.env_reward is not None)
        if outcome.reward + step_rewards < min_reward:
            return "reward below --min-reward"
    return None


def _swift_content(content: Any) -> str:
    """Chat content parts as ms-swift text: texts in order, an image as an <image> tag."""
    if isinstance(content, str):
        return content
    pieces = []
    for part in content or []:
        if part.get("type") == "image":
            pieces.append("<image>")
        elif part.get("type") == "text":
            pieces.append(part.get("text", ""))
        else:
            raise ExportError(f"unsupported content part {part.get('type')!r}")
    return "".join(pieces)
