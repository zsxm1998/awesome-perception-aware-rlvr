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
"""What the adapters of external environment standards share."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from typing import Any, Mapping, Optional

from ....utils.dataset import process_image


# The environments of these standards give the task, not the rules of this repository's agent loop (the tool-call
# limits, the answer inside <answer></answer>), so the adapters add them as a system message. The example of the
# answer matters: with the same task prompt, Qwen3-VL-4B left the tags out on 16 of the toy search's 64 validation
# tasks without it, on 1 with it.
ANSWER_SYSTEM_PROMPT = (
    "Work on the task with the tools provided. You can make at most {max_tool_calls} tool calls in total, at most "
    "{max_tool_calls_per_turn} in one turn. When you are done, give your final answer inside <answer></answer>, for "
    "example <answer>42</answer>."
)
TOOLS_SYSTEM_PROMPT = (
    "Complete the task with the tools provided. You can make at most {max_tool_calls} tool calls in total, at most "
    "{max_tool_calls_per_turn} in one turn. The task ends when a tool says that it is finished."
)


def check_system_prompt(name: str, template: Optional[str]) -> None:
    """``None`` (the default prompt), ``""`` (no system message) or a template with ``{max_tool_calls}`` and
    ``{max_tool_calls_per_turn}``."""
    if template is None:
        return
    if not isinstance(template, str):
        raise ValueError(f"{name} system_prompt must be a string or null")
    try:
        template.format(max_tool_calls=1, max_tool_calls_per_turn=1)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"{name} system_prompt may only use the placeholders {{max_tool_calls}} and {{max_tool_calls_per_turn}} "
            f"(write literal braces as {{{{ and }}}}): {exc!r}"
        ) from exc


def system_messages(
    template: Optional[str], default: str, *, max_tool_calls: int, max_tool_calls_per_turn: int
) -> list[dict[str, Any]]:
    text = (default if template is None else template).format(
        max_tool_calls=max_tool_calls, max_tool_calls_per_turn=max_tool_calls_per_turn
    )
    return [{"role": "system", "content": text}] if text else []


def check_count(name: str, value: Any, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")


def finite_or_none(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or null, got {value!r}")
    return float(value)


def function_schema(name: Any, description: Any, parameters: Any) -> dict[str, Any]:
    """An environment's tool as an OpenAI function schema, which the chat template renders (``tools=``)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"a tool needs a non-empty name, got {name!r}")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, Mapping):
        raise ValueError(f"the input schema of tool {name!r} must be a JSON object")
    return {
        "type": "function",
        "function": {"name": name, "description": description or "", "parameters": dict(parameters)},
    }


class DecodedImages:
    """Base64 images decoded once per distinct content: the rollouts of a group get the same image object, which
    hits vLLM's multimodal caches (as remote_http does)."""

    def __init__(self) -> None:
        self._images: dict[str, Any] = {}

    def decode(self, data: Any) -> Any:
        if not isinstance(data, str):
            raise ValueError("an image needs its data as a base64 string")
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("an image has invalid base64 data") from exc
        return self.decode_bytes(raw)

    def decode_bytes(self, raw: bytes) -> Any:
        key = hashlib.sha256(raw).hexdigest()
        if key not in self._images:
            self._images[key] = process_image(raw, None, None)
        return self._images[key]
