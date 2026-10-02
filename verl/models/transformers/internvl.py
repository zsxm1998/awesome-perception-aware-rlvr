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

import json
import os
import types
from typing import Any, Optional

import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
INTERNVL_SYSTEM_MESSAGE = (
    "\u4f60\u662f\u4e66\u751f\u00b7\u4e07\u8c61\uff0c\u82f1\u6587\u540d\u662fInternVL\uff0c"
    "\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u3001"
    "\u6e05\u534e\u5927\u5b66\u53ca\u591a\u5bb6\u5408\u4f5c\u5355\u4f4d\u8054\u5408\u5f00\u53d1"
    "\u7684\u591a\u6a21\u6001\u5927\u8bed\u8a00\u6a21\u578b\u3002"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio

    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    target_aspect_ratio = _dynamic_target_ratio(
        orig_width,
        orig_height,
        min_num=min_num,
        max_num=max_num,
        image_size=image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height), resample=Image.BICUBIC)

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), resample=Image.BICUBIC))

    return processed_images


def _dynamic_target_ratio(
    width: int,
    height: int,
    *,
    min_num: int,
    max_num: int,
    image_size: int,
) -> tuple[int, int]:
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    return _find_closest_aspect_ratio(
        width / height,
        target_ratios,
        width,
        height,
        image_size,
    )


def _dynamic_patch_count(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> int:
    """Return InternVL's dynamic tile count without materializing image tensors."""

    width, height = image.size
    target_ratio = _dynamic_target_ratio(
        width,
        height,
        min_num=min_num,
        max_num=max_num,
        image_size=image_size,
    )
    blocks = target_ratio[0] * target_ratio[1]
    return blocks + int(use_thumbnail and blocks != 1)


def _image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


class InternVLImageProcessorAdapter:
    def __init__(self, processor: "InternVLProcessorAdapter"):
        self.processor = processor

    def __call__(
        self,
        images: Optional[list[Image.Image]] = None,
        videos: Optional[list[Any]] = None,
        return_tensors: str = "pt",
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if videos:
            raise NotImplementedError("InternVL3.5 video preprocessing is not supported in this EasyR1 adapter.")
        if return_tensors != "pt":
            raise ValueError("InternVL3.5 adapter only supports return_tensors='pt'.")
        if not images:
            return {}

        pixel_values = []
        image_flags = []
        for image in images:
            tiles = self.processor.preprocess_image(image)
            pixel_values.extend(_image_to_tensor(tile, self.processor.image_size) for tile in tiles)
            image_flags.extend([1] * len(tiles))

        return {
            "pixel_values": torch.stack(pixel_values),
            "image_flags": torch.tensor(image_flags, dtype=torch.long).view(-1, 1),
        }


class InternVLProcessorAdapter:
    image_token = IMG_CONTEXT_TOKEN
    image_token_id: Optional[int] = None
    model_input_names = ["pixel_values", "image_flags"]

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.image_size = config.force_image_size or config.vision_config.image_size
        self.min_dynamic_patch = config.min_dynamic_patch
        self.max_dynamic_patch = config.max_dynamic_patch
        self.use_thumbnail = config.use_thumbnail
        self.num_image_token = int(
            (self.image_size // config.vision_config.patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.image_processor = InternVLImageProcessorAdapter(self)
        self.chat_template = None

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        config = AutoConfig.from_pretrained(model_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(tokenizer, config)

    def preprocess_image(self, image: Image.Image) -> list[Image.Image]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return _dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=self.max_dynamic_patch,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

    def get_num_image_patches(self, image: Image.Image) -> int:
        """Mirror ``preprocess_image`` layout without resizing/cropping tiles."""

        return _dynamic_patch_count(
            image,
            min_num=self.min_dynamic_patch,
            max_num=self.max_dynamic_patch,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        processor_config = {
            "processor_class": self.__class__.__name__,
            "image_size": self.image_size,
            "min_dynamic_patch": self.min_dynamic_patch,
            "max_dynamic_patch": self.max_dynamic_patch,
            "use_thumbnail": self.use_thumbnail,
            "num_image_token": self.num_image_token,
            "image_token": self.image_token,
        }
        with open(os.path.join(save_directory, "preprocessor_config.json"), "w", encoding="utf-8") as f:
            json.dump(processor_config, f, ensure_ascii=False, indent=2)
        return (save_directory,)

    def apply_chat_template(self, messages, add_generation_prompt: bool = True, tokenize: bool = False, **kwargs):
        tools = kwargs.pop("tools", None)
        if tools is not None:
            # Agentic-only path: InternVL3.5's tokenizer owns the native tool
            # protocol. Keep the legacy formatter byte-for-byte unchanged when
            # tools is not supplied.
            native_messages = []
            for message in messages:
                native_message = dict(message)
                native_message["content"] = self._format_content(message["content"])
                native_messages.append(native_message)
            return self.tokenizer.apply_chat_template(
                native_messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                **kwargs,
            )

        prompt = self._format_messages(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return self.tokenizer(prompt, add_special_tokens=False, **kwargs)["input_ids"]
        return prompt

    def get_raw_prompt_ids(self, prompt: str) -> list[int]:
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        image_token_ids = self.tokenizer.encode("<image>", add_special_tokens=False)
        image_newline_ids = self.tokenizer.encode("<image>\n", add_special_tokens=False)
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)

        # vLLM applies InternVL prompt replacement by looking for the tokenized
        # "<image>" target. The tokenizer can merge ">\n" into one token, so keep
        # the raw placeholder unexpanded while making that target discoverable.
        return self._replace_token_sequence(raw_prompt_ids, image_newline_ids, image_token_ids + newline_ids)

    @staticmethod
    def _replace_token_sequence(token_ids: list[int], old: list[int], new: list[int]) -> list[int]:
        if not old:
            return token_ids

        output = []
        index = 0
        while index < len(token_ids):
            if token_ids[index : index + len(old)] == old:
                output.extend(new)
                index += len(old)
            else:
                output.append(token_ids[index])
                index += 1
        return output

    def __call__(
        self,
        images: Optional[list[Image.Image]] = None,
        text: Optional[list[str]] = None,
        add_special_tokens: bool = False,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("InternVL3.5 adapter only supports return_tensors='pt'.")

        prompts = list(text or [""])
        if len(prompts) != 1:
            raise ValueError("InternVL3.5 adapter currently expects one prompt per preprocessing call.")

        prompt = prompts[0]
        if images:
            missing_placeholders = len(images) - prompt.count("<image>")
            if missing_placeholders > 0:
                prompt = ("<image>\n" * missing_placeholders) + prompt

            for image in images:
                num_patches = len(self.preprocess_image(image))
                image_tokens = (
                    IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * self.num_image_token * num_patches) + IMG_END_TOKEN
                )
                prompt = prompt.replace("<image>", image_tokens, 1)

        return self.tokenizer([prompt], add_special_tokens=add_special_tokens, return_tensors=return_tensors, **kwargs)

    def _format_messages(self, messages, add_generation_prompt: bool) -> str:
        prompt = f"<|im_start|>system\n{INTERNVL_SYSTEM_MESSAGE}<|im_end|>\n"
        for message in messages:
            role = message["role"]
            content = self._format_content(message["content"])
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "system":
                prompt = f"<|im_start|>system\n{content}<|im_end|>\n"
            else:
                raise ValueError(f"Unsupported role for InternVL3.5 chat template: {role}")

        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"

        return prompt

    @staticmethod
    def _format_content(content) -> str:
        if isinstance(content, str):
            return content

        parts = []
        for item in content:
            if item["type"] == "image":
                parts.append("<image>\n")
            elif item["type"] == "text":
                parts.append(item["text"])
            else:
                raise ValueError(f"Unsupported InternVL3.5 content type: {item['type']}")

        return "".join(parts)


def is_internvl_processor(processor: Any) -> bool:
    return isinstance(processor, InternVLProcessorAdapter)


def set_internvl_image_context_token_id(model: Any, processor: Any) -> None:
    if is_internvl_processor(processor) and hasattr(model, "img_context_token_id"):
        model.img_context_token_id = processor.image_token_id


def patch_internvl_vision_checkpointing(model: torch.nn.Module) -> int:
    """Use non-reentrant checkpointing in InternVL's remote vision encoder.

    InternVL3.5 remote code calls torch.utils.checkpoint.checkpoint directly in
    the vision encoder, so model.gradient_checkpointing_enable(... use_reentrant=False)
    does not reach it. Reentrant checkpointing can fail with FSDP/offload during
    backward recomputation because tensors may be materialized lazily.
    """

    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "InternVisionEncoder":
            continue
        if getattr(module, "_easy_r1_non_reentrant_checkpoint", False):
            continue

        def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_states = () if output_hidden_states else None
            hidden_states = inputs_embeds

            for encoder_layer in self.layers:
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                if self.gradient_checkpointing and self.training:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        encoder_layer,
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = encoder_layer(hidden_states)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, encoder_states] if v is not None)
            return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states)

        module.forward = types.MethodType(forward, module)
        module._easy_r1_non_reentrant_checkpoint = True
        patched += 1

    return patched
