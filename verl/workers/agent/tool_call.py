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
import re
from typing import Protocol

from .protocol import ParsedAction, ToolCall


class ToolCallCodec(Protocol):
    def parse(self, text: str, *, reasoning_open: bool = False) -> ParsedAction: ...


class NativeToolCallCodec:
    """Parse the tool XML emitted by Qwen3-VL and InternVL3.5 templates."""

    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    _THINK_TAG_PATTERN = re.compile(r"</?think>")
    _TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def parse(self, text: str, *, reasoning_open: bool = False) -> ParsedAction:
        answer_matches = list(self._ANSWER_PATTERN.finditer(text))
        if answer_matches:
            valid_answer_matches = [
                match
                for match in answer_matches
                if not self.is_inside_reasoning(
                    text,
                    match.start(),
                    reasoning_open=reasoning_open,
                )
            ]
            if not valid_answer_matches:
                return ParsedAction(final_answer_error="final answer appears before reasoning is closed")

            # Match the original DeepEyes behavior: a final answer takes
            # precedence if a generation contains both an answer and a call.
            final_answer = valid_answer_matches[-1].group(1).strip()
            if not final_answer:
                return ParsedAction(final_answer_error="final answer must not be empty")
            return ParsedAction(final_answer=final_answer)

        tool_call_matches = list(self._TOOL_CALL_PATTERN.finditer(text))
        if not tool_call_matches:
            if "<tool_call>" in text or "</tool_call>" in text:
                return ParsedAction(tool_call_error="tool call tags are incomplete")
            return ParsedAction()

        # Match the original DeepEyes environment: a complete tool-call block is
        # executable even when the model has not closed its reasoning tag yet.
        # Reasoning-tag balance remains a format-reward concern; rejecting the
        # action here would also withhold the crop observation and tool reward.
        if len(tool_call_matches) != 1:
            return ParsedAction(
                tool_call_error="exactly one tool call is allowed per generation turn",
            )

        raw_text = tool_call_matches[0].group(1).strip()
        try:
            payload = json.loads(raw_text)
        except (TypeError, ValueError) as exc:
            return ParsedAction(
                tool_call_error=f"tool call is not valid JSON: {exc}",
            )

        if not isinstance(payload, dict):
            return ParsedAction(tool_call_error="tool call must be a JSON object")

        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name:
            return ParsedAction(
                tool_call_error="tool call field 'name' must be a non-empty string",
            )
        if not isinstance(arguments, dict):
            return ParsedAction(
                tool_call_error="tool call field 'arguments' must be a JSON object",
            )

        return ParsedAction(tool_call=ToolCall(name=name, arguments=arguments, raw_text=raw_text))

    def contains_answer_inside_reasoning(
        self,
        text: str,
        *,
        reasoning_open: bool = False,
    ) -> bool:
        """Whether any complete answer tag is quoted before reasoning closes."""

        return any(
            self.is_inside_reasoning(
                text,
                match.start(),
                reasoning_open=reasoning_open,
            )
            for match in self._ANSWER_PATTERN.finditer(text)
        )

    def count_tool_calls_inside_reasoning(
        self,
        text: str,
        *,
        reasoning_open: bool = False,
    ) -> int:
        """Count complete tool-call blocks emitted before reasoning closes."""

        return sum(
            self.is_inside_reasoning(
                text,
                match.start(),
                reasoning_open=reasoning_open,
            )
            for match in self._TOOL_CALL_PATTERN.finditer(text)
        )

    def is_inside_reasoning(
        self,
        text: str,
        position: int,
        *,
        reasoning_open: bool,
    ) -> bool:
        depth = 1 if reasoning_open else 0
        for match in self._THINK_TAG_PATTERN.finditer(text, 0, position):
            if match.group(0) == "<think>":
                depth += 1
            elif depth:
                depth -= 1
        return depth > 0
