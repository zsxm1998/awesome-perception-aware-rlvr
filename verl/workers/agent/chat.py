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

import copy
import json
import re
from typing import Any, Mapping, Sequence

from jinja2 import Environment, StrictUndefined, TemplateError, meta

from .coordinates import TOOL_CALL_TEMPLATE_HINT
from .protocol import AgentImageConfig, EncodedObservation, EncodedPrompt, ObservationEncodingRequest
from .tools.base import ToolRegistry
from .trajectory import measure_expanded_observation


DEEPEYES_MAX_TOOL_CALLS_PLACEHOLDER = "{{ max_tool_calls }}"


def render_deepeyes_system_prompt(
    template_text: str,
    *,
    max_tool_calls: int,
) -> str:
    """Strictly render the file-backed DeepEyes method contract."""

    if not isinstance(template_text, str) or not template_text.strip():
        raise ValueError("DeepEyes requires a non-empty system prompt template")
    if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 0:
        raise ValueError("max_tool_calls must be a non-negative integer")
    if template_text.count(DEEPEYES_MAX_TOOL_CALLS_PLACEHOLDER) != 1:
        raise ValueError(
            f"DeepEyes system prompt must contain exactly one {DEEPEYES_MAX_TOOL_CALLS_PLACEHOLDER} placeholder"
        )

    environment = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    try:
        parsed = environment.parse(template_text)
        variables = meta.find_undeclared_variables(parsed)
        if variables != {"max_tool_calls"}:
            raise ValueError(
                f"DeepEyes system prompt may only reference the max_tool_calls placeholder; found {sorted(variables)}"
            )
        return environment.from_string(template_text).render(max_tool_calls=max_tool_calls).strip()
    except TemplateError as exc:
        raise ValueError(f"invalid DeepEyes system prompt template: {exc}") from exc


class NativeToolChatAdapter:
    """Use a model's native chat template without re-encoding model actions.

    The initial prompt is rendered normally with ``tools=...``. For later
    observations, a unique assistant sentinel is rendered through the same
    template and only the suffix after that sentinel is tokenized. The actual
    generated tool-call IDs therefore remain byte-for-byte untouched.
    """

    _ASSISTANT_SENTINEL = "__EASYR1_AGENT_ACTION_SENTINEL_7F4A0E__"
    _THINK_TAG_PATTERN = re.compile(r"</?think>")

    def __init__(
        self,
        processor: Any,
        tool_registry: ToolRegistry,
        num_source_images: int,
        max_tool_calls: int,
        image_config: AgentImageConfig = AgentImageConfig(),
        image_preprocessor: Any = None,
    ):
        if processor is None or not hasattr(processor, "apply_chat_template"):
            raise TypeError("NativeToolChatAdapter requires a multimodal processor with a chat template")
        if not hasattr(processor, "tokenizer"):
            raise TypeError("NativeToolChatAdapter requires processor.tokenizer")
        if num_source_images <= 0:
            raise ValueError("DeepEyes requires at least one source image")
        if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a non-negative integer")

        self.processor = processor
        self.tools = tool_registry.schemas(num_source_images)
        self.num_source_images = num_source_images
        self.max_tool_calls = max_tool_calls
        if not isinstance(image_config, AgentImageConfig):
            raise TypeError("image_config must be an AgentImageConfig")
        self.image_config = image_config
        self.image_preprocessor = image_preprocessor
        self.reasoning_prefilled = False
        self.assistant_termination_token_ids = self._derive_assistant_termination_token_ids()

    def encode_initial_prompt(self, messages: Sequence[Mapping[str, Any]]) -> EncodedPrompt:
        prepared_messages = self._prepare_deepeyes_messages(messages)
        rendered = self._render_with_generation_prompt(prepared_messages)
        self._check_tools_rendered(prepared_messages, rendered)
        return EncodedPrompt(token_ids=self._encode_raw_text(rendered), rendered_text=rendered)

    def _check_tools_rendered(self, messages: Sequence[Mapping[str, Any]], rendered: str) -> None:
        """Fail loudly when the chat template silently drops ``tools=`` (e.g. stock Qwen2.5-VL)."""
        without_tools = self.processor.apply_chat_template(
            messages, tools=None, add_generation_prompt=True, tokenize=False
        )
        if without_tools == rendered:
            raise RuntimeError(
                "the processor's chat template ignores the tool definitions passed via tools=...; for Qwen2-VL / "
                f"Qwen2.5-VL set data.override_chat_template={TOOL_CALL_TEMPLATE_HINT}"
            )

    async def encode(self, request: ObservationEncodingRequest) -> EncodedObservation:
        if request.result.observation_text is not None:
            content_json = request.result.observation_text
        else:
            content_json = json.dumps(
                request.result.content,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            # Tool arguments are model-generated. Keep strings from prematurely
            # closing the native XML wrapper when they are echoed in an error.
            content_json = content_json.replace("<", "\\u003c").replace(">", "\\u003e")
        if request.result.images:
            tool_content: Any = [{"type": "image"}, {"type": "text", "text": content_json}]
        else:
            tool_content = content_json

        synthetic_messages = [
            {"role": "assistant", "content": self._ASSISTANT_SENTINEL},
            {"role": "tool", "content": tool_content},
        ]
        rendered = self._render_with_generation_prompt(synthetic_messages)
        if rendered.count(self._ASSISTANT_SENTINEL) != 1:
            raise RuntimeError("native chat template did not preserve the agent action sentinel exactly once")

        suffix = rendered.split(self._ASSISTANT_SENTINEL, 1)[1]
        suffix_token_ids = self._encode_raw_text(suffix)
        termination_token_ids = list(self.assistant_termination_token_ids)
        if suffix_token_ids[: len(termination_token_ids)] != termination_token_ids:
            raise RuntimeError(
                "native chat template observation suffix does not begin with the derived assistant termination tokens"
            )

        # vLLM preserves the model-generated EOS/end-of-turn token in the
        # action. Remove only that overlapping token prefix from the synthetic
        # suffix; keep the following template separator (for Qwen, "\n").
        suffix_token_ids = suffix_token_ids[len(termination_token_ids) :]
        effective_token_count = None
        visual_token_count = None if request.result.images else 0
        if request.result.images and hasattr(self.processor, "image_processor"):
            effective_token_count, visual_token_count = measure_expanded_observation(
                self.processor,
                suffix_token_ids,
                request.result.images,
                min_pixels=self.image_config.min_pixels,
                max_pixels=self.image_config.max_pixels,
                image_preprocessor=self.image_preprocessor,
            )
        return EncodedObservation(
            token_ids=suffix_token_ids,
            images=request.result.images,
            effective_token_count=effective_token_count,
            visual_token_count=visual_token_count,
        )

    def _prepare_deepeyes_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        prepared = copy.deepcopy(list(messages))
        if not prepared or prepared[0].get("role") != "system":
            raise ValueError("DeepEyes requires its file-backed system prompt as the first message")
        system_content = prepared[0].get("content")
        if not isinstance(system_content, str):
            raise TypeError("DeepEyes system prompt content must be a string")
        prepared[0]["content"] = render_deepeyes_system_prompt(
            system_content,
            max_tool_calls=self.max_tool_calls,
        )

        self._validate_and_label_source_images(prepared)
        return prepared

    def _validate_and_label_source_images(self, messages: list[dict[str, Any]]) -> None:
        image_idx = 0
        for message in messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue

            labeled_content = []
            for item in message["content"]:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    if self.num_source_images > 1:
                        labeled_content.append({"type": "text", "text": f"Image {image_idx}:\n"})
                    image_idx += 1
                labeled_content.append(item)
            message["content"] = labeled_content

        if image_idx != self.num_source_images:
            raise ValueError(
                "source image count does not match image placeholders in messages: "
                f"{self.num_source_images} images but {image_idx} placeholders"
            )

    def _encode_raw_text(self, text: str) -> list[int]:
        if hasattr(self.processor, "get_raw_prompt_ids"):
            return list(self.processor.get_raw_prompt_ids(text))
        return list(self.processor.tokenizer.encode(text, add_special_tokens=False))

    def _derive_assistant_termination_token_ids(self) -> tuple[int, ...]:
        """Derive the model-owned assistant turn terminator from its template."""

        rendered = self.processor.apply_chat_template(
            [{"role": "assistant", "content": self._ASSISTANT_SENTINEL}],
            tools=self.tools,
            add_generation_prompt=False,
            tokenize=False,
        )
        if rendered.count(self._ASSISTANT_SENTINEL) != 1:
            raise RuntimeError(
                "native chat template did not preserve the assistant sentinel while deriving its turn terminator"
            )

        suffix_token_ids = self._encode_raw_text(rendered.split(self._ASSISTANT_SENTINEL, 1)[1])
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if not isinstance(eos_token_id, int):
            raise RuntimeError("native tool chat template requires one integer tokenizer eos_token_id")
        try:
            eos_index = suffix_token_ids.index(eos_token_id)
        except ValueError as exc:
            raise RuntimeError(
                "native chat template assistant suffix does not contain tokenizer eos_token_id"
            ) from exc

        termination_token_ids = tuple(suffix_token_ids[: eos_index + 1])
        if not termination_token_ids:
            raise RuntimeError("native chat template produced an empty assistant terminator")
        return termination_token_ids

    def _render_with_generation_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> str:
        render_kwargs = {
            "tools": self.tools,
            "tokenize": False,
        }
        without_generation_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            **render_kwargs,
        )
        rendered = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **render_kwargs,
        )
        if not rendered.startswith(without_generation_prompt):
            raise RuntimeError(
                "native chat template must append its generation prompt without rewriting the rendered conversation"
            )

        generation_prefix = rendered[len(without_generation_prompt) :]
        self.reasoning_prefilled = self._has_unclosed_think(generation_prefix)
        return rendered

    def _has_unclosed_think(self, text: str) -> bool:
        depth = 0
        for match in self._THINK_TAG_PATTERN.finditer(text):
            if match.group(0) == "<think>":
                depth += 1
            elif depth:
                depth -= 1
        return depth > 0


class PlainChatAdapter(NativeToolChatAdapter):
    """One model turn without tools, with the prompt's own messages (no tool schema, no system template).

    DeepEyes rows whose official ``env_name`` is empty (ThinkLite) are rolled out this way, as in the
    official environment, which gives them no tool.
    """

    def __init__(
        self,
        processor: Any,
        num_source_images: int,
        image_config: AgentImageConfig = AgentImageConfig(),
        image_preprocessor: Any = None,
    ):
        super().__init__(
            processor,
            ToolRegistry([]),
            num_source_images,
            max_tool_calls=0,
            image_config=image_config,
            image_preprocessor=image_preprocessor,
        )
        self.tools = None
        self.assistant_termination_token_ids = self._derive_assistant_termination_token_ids()

    def encode_initial_prompt(self, messages: Sequence[Mapping[str, Any]]) -> EncodedPrompt:
        prepared = copy.deepcopy(list(messages))
        for message in prepared:
            if message.get("role") == "system" and DEEPEYES_MAX_TOOL_CALLS_PLACEHOLDER in str(message.get("content")):
                raise ValueError(
                    "a row without a tool got the tool system prompt; give such rows their own system prompt "
                    "through data.system_prompt_key (DeepEyes: row_system_prompt or official_system_prompt)"
                )
        self._validate_and_label_source_images(prepared)
        rendered = self._render_with_generation_prompt(prepared)
        return EncodedPrompt(token_ids=self._encode_raw_text(rendered), rendered_text=rendered)

    async def encode(self, request: ObservationEncodingRequest) -> EncodedObservation:
        raise RuntimeError("a rollout without tools has no tool observations")


# DeepEyes' tool response (verl/workers/agent/envs/mm_process_engine: visual_toolbox_v2.py with PROMPT.USER_PROMPT_V2)
OFFICIAL_DEEPEYES_TOOL_RESPONSE_SUFFIX = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  "
    "<tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
)


class OfficialDeepEyesChatAdapter(NativeToolChatAdapter):
    """DeepEyes' own prompt format: the tool schema is written in the dataset's system prompt (not passed to the
    chat template), and a tool result comes back as a user turn
    ``<tool_response><image>{format instruction}</tool_response>`` (``Error: ...`` when the call fails)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.tools = None  # the schema is part of the system prompt text
        self.assistant_termination_token_ids = self._derive_assistant_termination_token_ids()

    def encode_initial_prompt(self, messages: Sequence[Mapping[str, Any]]) -> EncodedPrompt:
        prepared = copy.deepcopy(list(messages))
        self._validate_and_label_source_images(prepared)
        rendered = self._render_with_generation_prompt(prepared)
        return EncodedPrompt(token_ids=self._encode_raw_text(rendered), rendered_text=rendered)

    async def encode(self, request: ObservationEncodingRequest) -> EncodedObservation:
        result = request.result
        if result.images:
            content: Any = [
                {"type": "text", "text": "<tool_response>"},
                *({"type": "image"} for _ in result.images),
                {"type": "text", "text": OFFICIAL_DEEPEYES_TOOL_RESPONSE_SUFFIX + "</tool_response>"},
            ]
        else:
            error = result.content.get("error") if isinstance(result.content, Mapping) else None
            message = error.get("message") if isinstance(error, Mapping) else None
            content = f"Error: {message or result.error_code or 'tool call failed'}"
        rendered = self._render_with_generation_prompt(
            [{"role": "assistant", "content": self._ASSISTANT_SENTINEL}, {"role": "user", "content": content}]
        )
        if rendered.count(self._ASSISTANT_SENTINEL) != 1:
            raise RuntimeError("chat template did not preserve the agent action sentinel exactly once")
        suffix_token_ids = self._encode_raw_text(rendered.split(self._ASSISTANT_SENTINEL, 1)[1])
        termination_token_ids = list(self.assistant_termination_token_ids)
        if suffix_token_ids[: len(termination_token_ids)] != termination_token_ids:
            raise RuntimeError("observation suffix does not begin with the derived assistant termination tokens")
        suffix_token_ids = suffix_token_ids[len(termination_token_ids) :]
        effective_token_count = None
        visual_token_count = None if result.images else 0
        if result.images and hasattr(self.processor, "image_processor"):
            effective_token_count, visual_token_count = measure_expanded_observation(
                self.processor,
                suffix_token_ids,
                result.images,
                min_pixels=self.image_config.min_pixels,
                max_pixels=self.image_config.max_pixels,
                image_preprocessor=self.image_preprocessor,
            )
        return EncodedObservation(
            token_ids=suffix_token_ids,
            images=result.images,
            effective_token_count=effective_token_count,
            visual_token_count=visual_token_count,
        )
