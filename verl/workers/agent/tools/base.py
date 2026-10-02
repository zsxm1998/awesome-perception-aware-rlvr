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

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image

from ..protocol import ToolCall, ToolResult


logger = logging.getLogger(__name__)
_MAX_INTERNAL_ERROR_LOGS = 5
_INTERNAL_ERROR_LOG_WINDOW_SECONDS = 300.0


@dataclass
class _InternalErrorLogState:
    window_started_at: float
    traceback_count: int = 0
    suppressed_count: int = 0


_internal_error_log_states: dict[
    tuple[str, type[BaseException]],
    _InternalErrorLogState,
] = {}
_internal_error_log_lock = threading.Lock()
_monotonic = time.monotonic


def _internal_error_log_decision(
    tool_name: str,
    exception_type: type[BaseException],
) -> tuple[bool, bool, int]:
    """Return traceback, suppression-warning, and prior-window summary decisions."""

    key = (tool_name, exception_type)
    now = _monotonic()
    with _internal_error_log_lock:
        state = _internal_error_log_states.get(key)
        prior_window_suppressed = 0
        if state is None or now - state.window_started_at >= _INTERNAL_ERROR_LOG_WINDOW_SECONDS:
            if state is not None:
                prior_window_suppressed = state.suppressed_count
            state = _InternalErrorLogState(window_started_at=now)
            _internal_error_log_states[key] = state

        should_log_traceback = state.traceback_count < _MAX_INTERNAL_ERROR_LOGS
        if should_log_traceback:
            state.traceback_count += 1
        else:
            state.suppressed_count += 1
        should_warn_suppression = state.traceback_count == _MAX_INTERNAL_ERROR_LOGS and (state.suppressed_count == 0)
    return should_log_traceback, should_warn_suppression, prior_window_suppressed


def _reset_internal_error_log_state_for_testing() -> None:
    """Reset process-level limiter state for deterministic tests."""

    with _internal_error_log_lock:
        _internal_error_log_states.clear()


def tool_error(
    tool_name: str,
    error_code: str,
    message: str,
    **details: Any,
) -> ToolResult:
    content = {
        "status": "error",
        "tool_name": tool_name,
        "error": {"code": error_code, "message": message},
        **details,
    }
    return ToolResult(
        tool_name=tool_name,
        success=False,
        content=content,
        error_code=error_code,
        metadata=dict(details),
    )


class AgentTool(ABC):
    name: str

    @abstractmethod
    def schema(self, num_source_images: int) -> Mapping[str, Any]: ...

    @abstractmethod
    def execute(self, arguments: Mapping[str, Any], source_images: Sequence[Image.Image]) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: Sequence[AgentTool]):
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self, num_source_images: int) -> list[Mapping[str, Any]]:
        return [tool.schema(num_source_images) for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall, source_images: Sequence[Image.Image]) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if tool is None:
            return tool_error(
                tool_call.name,
                "unknown_tool",
                f"unknown tool {tool_call.name!r}",
                available_tools=sorted(self._tools),
            )

        try:
            return tool.execute(tool_call.arguments, source_images)
        except Exception as exc:
            # An environment failure becomes an observation instead of
            # aborting the whole rollout worker.
            should_log, should_warn, prior_window_suppressed = _internal_error_log_decision(
                tool_call.name,
                type(exc),
            )
            if prior_window_suppressed:
                logger.warning(
                    "Suppressed %d %s tracebacks from agent tool %s in the previous "
                    "logging window; per-trajectory metrics include every occurrence",
                    prior_window_suppressed,
                    type(exc).__name__,
                    tool_call.name,
                )
            if should_log:
                logger.exception("Agent tool %s raised an internal error", tool_call.name)
            if should_warn:
                logger.warning(
                    "Further %s exceptions from agent tool %s in this worker process "
                    "will not log tracebacks until the current %.0f-second logging "
                    "window resets",
                    type(exc).__name__,
                    tool_call.name,
                    _INTERNAL_ERROR_LOG_WINDOW_SECONDS,
                )
            return tool_error(tool_call.name, "tool_internal_error", str(exc))
