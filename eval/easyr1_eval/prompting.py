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
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Template

from .schemas import EvalSample


@dataclass(frozen=True)
class PromptConfig:
    format_prompt: str | None = None
    system_prompt: str | None = None
    prompt_mode: str = "raw"


def prompt_config_from_args(args: Any) -> PromptConfig:
    return PromptConfig(
        format_prompt=_read_optional_text(getattr(args, "format_prompt", None), strip=False),
        system_prompt=_read_optional_text(getattr(args, "system_prompt", None), strip=True),
        prompt_mode=str(getattr(args, "prompt_mode", "raw")),
    )


def apply_prompt_config(samples: Iterable[EvalSample], config: PromptConfig) -> list[EvalSample]:
    return [apply_prompt_to_sample(sample, config) for sample in samples]


def apply_prompt_to_sample(sample: EvalSample, config: PromptConfig) -> EvalSample:
    formatted_prompt = _format_user_prompt(sample.prompt, config.format_prompt)
    image_count = len(sample.images)
    metadata = dict(sample.metadata)
    if config.format_prompt is not None:
        metadata["format_prompt_applied"] = True
    if config.system_prompt is not None:
        metadata["system_prompt_applied"] = True
    metadata["prompt_mode"] = config.prompt_mode

    if config.prompt_mode == "chat":
        return EvalSample(
            benchmark=sample.benchmark,
            sample_id=sample.sample_id,
            prompt=formatted_prompt,
            target=sample.target,
            images=list(sample.images),
            messages=_build_messages(formatted_prompt, image_count, config.system_prompt),
            extra_info=dict(sample.extra_info),
            metadata=metadata,
            native_agentic_prompt=sample.native_agentic_prompt,
        )
    if config.prompt_mode != "raw":
        raise ValueError(f"unknown prompt mode: {config.prompt_mode}")
    return EvalSample(
        benchmark=sample.benchmark,
        sample_id=sample.sample_id,
        prompt=_prepend_system_prompt(_ensure_image_placeholders(formatted_prompt, image_count), config.system_prompt),
        target=sample.target,
        images=list(sample.images),
        messages=None,
        extra_info=dict(sample.extra_info),
        metadata=metadata,
        native_agentic_prompt=sample.native_agentic_prompt,
    )


def render_chat_prompt(sample: EvalSample, template_owner: Any) -> str:
    if not sample.messages:
        return sample.prompt
    return template_owner.apply_chat_template(sample.messages, add_generation_prompt=True, tokenize=False)


def _format_user_prompt(prompt: str, format_prompt: str | None) -> str:
    if format_prompt is None:
        return prompt
    return Template(format_prompt.strip()).render(content=prompt)


def _build_messages(prompt: str, image_count: int, system_prompt: str | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image_count:
        content_list: list[dict[str, str]] = []
        placeholder_count = prompt.count("<image>")
        if placeholder_count == 0:
            content_list.extend({"type": "image"} for _ in range(image_count))
            if prompt:
                content_list.append({"type": "text", "text": prompt})
        else:
            for index, content in enumerate(prompt.split("<image>")):
                if index != 0:
                    content_list.append({"type": "image"})
                if content:
                    content_list.append({"type": "text", "text": content})
            for _ in range(max(0, image_count - placeholder_count)):
                content_list.append({"type": "image"})
        messages.append({"role": "user", "content": content_list})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def _prepend_system_prompt(prompt: str, system_prompt: str | None) -> str:
    if not system_prompt:
        return prompt
    return f"{system_prompt}\n\n{prompt}"


def _ensure_image_placeholders(prompt: str, image_count: int) -> str:
    if image_count <= 0:
        return prompt
    missing = image_count - prompt.count("<image>")
    if missing <= 0:
        return prompt
    prefix = "\n".join("<image>" for _ in range(missing))
    return f"{prefix}\n{prompt}" if prompt else prefix


def _read_optional_text(path: str | None, *, strip: bool) -> str | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    return text.strip() if strip else text
