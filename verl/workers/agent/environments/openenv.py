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
"""OpenEnv environments over their WebSocket session protocol (docs/agent_environments.md).

An OpenEnv server keeps one environment per WebSocket connection (``/ws``) and answers JSON messages: ``reset``,
``step`` (an action), ``state`` and ``close`` (``openenv.core.env_server.types``; written against openenv 0.11.0).
Its HTTP ``/reset`` and ``/step`` build a new environment for every request, so a multi-turn episode needs the
WebSocket. Environments with MCP tools are supported as they are: ``list_tools`` gives the tools and every tool call
is a ``call_tool`` action. An environment with its own action types needs a subclass that lists its tools and maps a
tool call to an action (``list_tools``, ``action``, ``tool_result``, ``observation_parts``), as TRL's OpenEnv examples
wrap each environment in a class of tools.

The client is written here instead of using the ``openenv`` package, which needs a huggingface_hub that
transformers 4.x does not allow.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from typing import Any, Mapping, Optional, Sequence

import aiohttp
from PIL import Image

from ..chat import EnvironmentChatAdapter
from ..protocol import AgentImageConfig, AgentStatus, EnvOutcome, EnvReset, EnvStep, ToolCall, ToolResult
from .registry import EnvironmentContext, register_environment
from .standards import (
    TOOLS_SYSTEM_PROMPT,
    DecodedImages,
    check_count,
    check_system_prompt,
    finite_or_none,
    function_schema,
    system_messages,
)


logger = logging.getLogger(__name__)
OPENENV_TESTED_VERSION = "0.11.0"
_CONFIG_KEYS = frozenset({"prompt", "reset"})
# errors a server answers before an environment exists, which a new connection may not meet
_RETRYABLE_CODES = frozenset({"CAPACITY_REACHED"})
# MCP errors that the call caused, which the model reads; the others (transport_error, timeout) fail the step.
# execution_error is the tool's own error: OpenEnv 0.11 reports a tool that raises (FastMCP's ToolError, the way a
# tool tells the caller that a call failed) with it, and cannot tell it from a tool that crashed.
_CALL_ERRORS = frozenset({"tool_not_found", "invalid_args", "execution_error"})
# the fields of every observation that are not part of what the model reads
_OBSERVATION_CONTROL_FIELDS = ("done", "reward")


class OpenEnvEnvironmentError(RuntimeError):
    """An OpenEnv server that failed (an error response, a lost connection, a timeout) or answered outside the
    protocol; it fails the training step, as a remote_http server error does."""


class _Retryable(Exception):
    pass


class _Connections:
    """What the environments of one rollout batch share: an aiohttp session (it belongs to the batch's event loop),
    the session slots and the decoded images."""

    def __init__(self, max_sessions: int, trust_env: bool) -> None:
        self.http = aiohttp.ClientSession(trust_env=trust_env)
        self.slots = asyncio.Semaphore(max_sessions)
        self.images = DecodedImages()

    async def aclose(self) -> None:
        await self.http.close()


def websocket_url(url: str) -> str:
    """``http(s)://host[/path]`` or ``ws(s)://...`` -> the session endpoint ``ws(s)://host[/path]/ws``."""
    for scheme, ws_scheme in (("https://", "wss://"), ("http://", "ws://")):
        if url.startswith(scheme):
            url = ws_scheme + url[len(scheme) :]
    return url.rstrip("/") + "/ws"


@register_environment("openenv")
class OpenEnvEnvironment:
    """An episode of an OpenEnv environment: one WebSocket session.

    The row's configuration (``agent_environment_config_key``) may hold ``reset``, the arguments of the environment's
    ``reset`` (``seed`` is the group's seed unless given), and ``prompt``, a text put before the reset observation in the
    user message. Options: ``url`` (``http(s)://`` or ``ws(s)://``, e.g. a Hugging Face Space), ``timeout_s`` per
    message, ``retries`` and ``backoff_s`` for opening a session (a connection error, or a server that has no room yet),
    ``max_sessions`` (sessions open at a time per rollout worker, each held from ``reset`` to ``close``),
    ``max_images_per_step``, ``image_fields`` (observation fields that hold an image), ``reset_observation`` (whether
    the reset observation is shown), ``system_prompt`` (the system message with the loop's rules; ``""`` for none),
    ``token_env`` (an environment variable holding a bearer token, e.g. for a private Space) and ``trust_env``
    (whether the proxy variables apply).
    """

    # its episodes live on a server, so every rank of a tensor-parallel group would open them again
    remote = True

    def __init__(
        self,
        context: EnvironmentContext,
        *,
        url: str,
        timeout_s: float = 60.0,
        retries: int = 3,
        backoff_s: float = 1.0,
        max_sessions: int = 8,
        max_images_per_step: int = 1,
        image_fields: Sequence[str] = (),
        reset_observation: bool = True,
        system_prompt: Optional[str] = None,
        token_env: Optional[str] = None,
        trust_env: bool = False,
    ) -> None:
        if not isinstance(url, str) or not url.startswith(("http://", "https://", "ws://", "wss://")):
            raise ValueError(f"openenv url must be an http(s) or ws(s) address, got {url!r}")
        for name, value in (("timeout_s", timeout_s), ("backoff_s", backoff_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
                raise ValueError(f"openenv {name} must be a non-negative number")
        check_count("openenv retries", retries, 0)
        check_count("openenv max_sessions", max_sessions, 1)
        check_count("openenv max_images_per_step", max_images_per_step, 0)
        if isinstance(image_fields, str) or not all(isinstance(name, str) and name for name in image_fields):
            raise ValueError("openenv image_fields must be a list of field names")
        if not isinstance(reset_observation, bool) or not isinstance(trust_env, bool):
            raise ValueError("openenv reset_observation and trust_env must be booleans")
        if token_env is not None and (not isinstance(token_env, str) or not token_env):
            raise ValueError("openenv token_env must be the name of an environment variable")
        check_system_prompt("openenv", system_prompt)
        self.context = context
        self.url = url
        self.timeout_s = float(timeout_s)
        self.retries = retries
        self.backoff_s = float(backoff_s)
        self.max_sessions = max_sessions
        self.max_images_per_step = max_images_per_step
        self.image_fields = tuple(image_fields)
        self.reset_observation = reset_observation
        self.system_prompt = system_prompt
        self.token_env = token_env
        self.trust_env = trust_env
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._slot: Optional[asyncio.Semaphore] = None
        self._reset_arguments: dict[str, Any] = {}
        self._done = False
        self._step_index = 0
        self._label = f"openenv {url}"

    def _connections(self) -> _Connections:
        return self.context.shared(
            ("openenv", self.trust_env), lambda: _Connections(self.max_sessions, self.trust_env)
        )

    # ------------------------------------------------------------------ messages
    async def request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Send one message and return the server's response (``{"type": ..., "data": ...}``, errors included)."""
        if self._ws is None:
            raise OpenEnvEnvironmentError(f"{self._label}: no open session")
        ws = self._ws

        async def exchange() -> aiohttp.WSMessage:
            await ws.send_json(dict(message))  # may wait too, for a server that stopped reading
            return await ws.receive()

        try:
            frame = await asyncio.wait_for(exchange(), self.timeout_s)
        except asyncio.TimeoutError as exc:
            raise OpenEnvEnvironmentError(f"{self._label}: no answer within {self.timeout_s:g} s") from exc
        except (aiohttp.ClientError, ConnectionError) as exc:
            raise OpenEnvEnvironmentError(f"{self._label}: {type(exc).__name__}: {exc}") from exc
        if frame.type != aiohttp.WSMsgType.TEXT:
            raise OpenEnvEnvironmentError(f"{self._label}: the session ended ({frame.type.name}: {frame.data!r:.200})")
        try:
            response = json.loads(frame.data)
        except ValueError as exc:
            raise OpenEnvEnvironmentError(f"{self._label}: not JSON: {frame.data[:300]}") from exc
        if not isinstance(response, dict) or not isinstance(response.get("data"), dict):
            raise OpenEnvEnvironmentError(f"{self._label}: a response needs a data object, got {frame.data[:300]}")
        return response

    def _observation_data(self, response: Mapping[str, Any], what: str) -> dict[str, Any]:
        if response.get("type") == "error":
            data = response["data"]
            raise OpenEnvEnvironmentError(f"{self._label}: {what} failed: {data.get('code')}: {data.get('message')}")
        if response.get("type") != "observation" or not isinstance(response["data"].get("observation"), dict):
            raise OpenEnvEnvironmentError(f"{self._label}: {what} answered {json.dumps(response)[:300]}")
        return response["data"]

    async def _open(self, reset_message: Mapping[str, Any]) -> dict[str, Any]:
        """Connect and reset, retrying a connection error or a server with no room yet."""
        headers = {}
        if self.token_env is not None:
            token = os.environ.get(self.token_env)
            if not token:
                raise ValueError(f"openenv token_env names an unset environment variable {self.token_env!r}")
            headers["Authorization"] = f"Bearer {token}"
        last_error = ""
        for attempt in range(self.retries + 1):
            if attempt:
                await asyncio.sleep(self.backoff_s * 2 ** (attempt - 1))
            try:
                self._ws = await asyncio.wait_for(
                    self._connections().http.ws_connect(
                        websocket_url(self.url), headers=headers, heartbeat=20.0, max_msg_size=100 * 2**20
                    ),
                    self.timeout_s,
                )
                response = await self.request(reset_message)
                if response.get("type") == "error" and response["data"].get("code") in _RETRYABLE_CODES:
                    raise _Retryable(f"{response['data'].get('code')}: {response['data'].get('message')}")
                return self._observation_data(response, "reset")
            except (_Retryable, aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except OpenEnvEnvironmentError as exc:
                if not isinstance(exc.__cause__, (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)):
                    raise
                last_error = str(exc)
            await self._disconnect()
        raise OpenEnvEnvironmentError(
            f"{self._label}: opening a session failed after {self.retries + 1} attempts: {last_error}"
        )

    async def _disconnect(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None or ws.closed:
            return
        # both bounded by timeout_s, so that a server that stopped reading cannot hold up the cleanup
        try:
            await asyncio.wait_for(ws.send_json({"type": "close"}), self.timeout_s)
        except Exception:  # the server may be gone already
            pass
        try:
            await asyncio.wait_for(ws.close(), self.timeout_s)
        except Exception:  # the session ends with the batch's HTTP client in any case
            pass

    # ------------------------------------------------------------------ what a subclass may change
    async def list_tools(self) -> list[dict[str, Any]]:
        """The tools as OpenAI function schemas; the server's MCP tools by default."""
        data = self._observation_data(
            await self.request({"type": "step", "data": {"type": "list_tools"}}), "list_tools"
        )
        tools = data["observation"].get("tools")
        if not isinstance(tools, list):
            raise OpenEnvEnvironmentError(
                f"{self._label}: the environment lists no MCP tools; for one with its own actions, write a subclass "
                "of OpenEnvEnvironment that lists its tools and maps a call to an action (docs/agent_environments.md)"
            )
        try:
            return [
                function_schema(
                    tool.get("name"), tool.get("description"), tool.get("input_schema", tool.get("inputSchema"))
                )
                for tool in tools
            ]
        except (AttributeError, ValueError) as exc:
            raise OpenEnvEnvironmentError(f"{self._label}: {exc}") from exc

    def action(self, call: ToolCall) -> dict[str, Any]:
        """The ``step`` action of a tool call: an MCP ``call_tool`` by default."""
        return {"type": "call_tool", "tool_name": call.name, "arguments": dict(call.arguments)}

    def observation_parts(self, observation: Mapping[str, Any], images: list) -> list[dict[str, Any]]:
        """What the model reads of an observation (the reset's, or a step's that is not an MCP tool result): the
        ``image_fields`` as images, then the other fields as text (a single text field as it is, else JSON); ``done``,
        ``reward`` and empty metadata are left out."""
        fields = {key: value for key, value in observation.items() if key not in _OBSERVATION_CONTROL_FIELDS}
        if not fields.get("metadata"):
            fields.pop("metadata", None)
        parts: list[dict[str, Any]] = []
        for name in self.image_fields:
            if fields.get(name) is not None:
                images.append(self._image(fields.pop(name), name))
                parts.append({"type": "image"})
            else:
                fields.pop(name, None)
        if len(fields) == 1 and isinstance(next(iter(fields.values())), str):
            parts.append({"type": "text", "text": next(iter(fields.values()))})
        elif fields:
            parts.append({"type": "text", "text": json.dumps(fields, ensure_ascii=False)})
        return parts

    def tool_result(self, call: ToolCall, data: Mapping[str, Any]) -> ToolResult:
        """The result of a step (``data``: its observation, reward, done and metadata) as the model reads it."""
        observation = data["observation"]
        images: list = []
        if "tool_name" not in observation:  # not an MCP tool result (e.g. the observation that ends the task)
            parts = self.observation_parts(observation, images)
            text = "".join(part["text"] for part in parts if part["type"] == "text")
            return ToolResult(call.name, True, {"status": "ok", "text": text}, images=images, parts=parts)
        error = observation.get("error")
        if error:
            code = error.get("error_type") if isinstance(error, dict) else None
            message = error.get("message", "") if isinstance(error, dict) else str(error)
            if code not in _CALL_ERRORS:
                # the environment could not carry out the call (a transport error, a timeout, an unknown kind)
                raise OpenEnvEnvironmentError(f"{self._label}, {call.name}: {code}: {message}")
            # what the call caused (unknown tool, invalid arguments, the tool's own error); it may echo the model's
            # arguments, so it is shown as escaped JSON, like the loop's own errors (no parts)
            content = {"status": "error", "tool_name": call.name, "error": {"code": code, "message": message}}
            return ToolResult(call.name, False, content, error_code=str(code))
        result = observation.get("result")
        parts = []
        items = result.get("content") if isinstance(result, dict) else None
        for item in items or []:
            kind = item.get("type") if isinstance(item, dict) else None
            if kind == "text":
                parts.append({"type": "text", "text": item.get("text", "")})
            elif kind == "image":
                images.append(self._image(item.get("data"), "an MCP image"))
                parts.append({"type": "image"})
            else:  # e.g. an embedded resource
                parts.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
        if not parts and isinstance(result, dict) and result.get("data") is not None:
            parts.append({"type": "text", "text": json.dumps(result["data"], ensure_ascii=False)})
        elif not parts and result is not None and not isinstance(result, dict):
            parts.append({"type": "text", "text": json.dumps(result, ensure_ascii=False)})
        text = "".join(part["text"] for part in parts if part["type"] == "text")
        if isinstance(result, dict) and result.get("is_error"):  # the tool's own error message
            content = {"status": "error", "tool_name": call.name, "error": {"code": "tool_error", "message": text}}
            return ToolResult(call.name, False, content, images=images, error_code="tool_error", parts=parts)
        return ToolResult(call.name, True, {"status": "ok", "text": text}, images=images, parts=parts)

    def _image(self, value: Any, where: str) -> Any:
        images = self._connections().images
        try:
            if isinstance(value, str):
                if value.startswith("data:"):
                    value = value.partition(",")[2]
                return images.decode(value)
            if isinstance(value, list):  # an H x W (x C) array of 0-255 values, e.g. a screenshot
                import numpy as np

                buffer = io.BytesIO()
                Image.fromarray(np.asarray(value, dtype=np.uint8)).save(buffer, format="PNG")
                return images.decode_bytes(buffer.getvalue())
        except (ValueError, TypeError) as exc:
            raise OpenEnvEnvironmentError(f"{self._label}: {where} is not an image: {exc}") from exc
        raise OpenEnvEnvironmentError(f"{self._label}: {where} must be a base64 image or a pixel array")

    # ------------------------------------------------------------------ the environment interface
    async def reset(self, config: Mapping[str, Any], seed: int) -> EnvReset:
        unknown = sorted(set(config) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"openenv task configuration has unknown keys {unknown}; known: {sorted(_CONFIG_KEYS)}")
        prompt = config.get("prompt")
        arguments = config.get("reset", {})
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("openenv prompt must be a string")
        if not isinstance(arguments, Mapping):
            raise ValueError("openenv reset must be a JSON object of the environment's reset arguments")
        self._reset_arguments = {"seed": seed, **arguments}
        connections = self._connections()
        await connections.slots.acquire()
        self._slot = connections.slots
        data = await self._open({"type": "reset", "data": self._reset_arguments})
        schemas = await self.list_tools()
        images: list = []
        user: list[dict[str, Any]] = [{"type": "text", "text": prompt}] if prompt else []
        if self.reset_observation:
            user += self.observation_parts(data["observation"], images)
        if not user:
            raise ValueError(
                "an openenv episode needs a prompt: give one in the row's configuration, or show the reset observation"
            )
        messages = system_messages(
            self.system_prompt,
            TOOLS_SYSTEM_PROMPT,
            max_tool_calls=self.context.max_tool_calls,
            max_tool_calls_per_turn=self.context.max_tool_calls_per_turn,
        )
        messages.append({"role": "user", "content": user})
        return EnvReset(
            messages=messages, images=images, tool_schemas=schemas, max_images_per_step=self.max_images_per_step
        )

    async def step(self, call: ToolCall) -> EnvStep:
        label = f"call {self._step_index} ({call.name})"
        self._step_index += 1
        response = await self.request({"type": "step", "data": self.action(call)})
        if response.get("type") == "error" and response["data"].get("code") == "VALIDATION_ERROR":
            # the server did not accept the action (it does not fit the environment's action type): the model's call
            content = {
                "status": "error",
                "tool_name": call.name,
                "error": {"code": "invalid_action", "message": json.dumps(response["data"])},
            }
            return EnvStep(result=ToolResult(call.name, False, content, error_code="invalid_action"))
        data = self._observation_data(response, label)
        result = self.tool_result(call, data)
        if len(result.images) > self.max_images_per_step:
            raise OpenEnvEnvironmentError(
                f"{self._label}, {label}: {len(result.images)} images, more than max_images_per_step={self.max_images_per_step}"
            )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise OpenEnvEnvironmentError(f"{self._label}, {label}: metadata must be a JSON object")
        try:
            reward = finite_or_none(data.get("reward"), "a step's reward")
        except ValueError as exc:
            raise OpenEnvEnvironmentError(f"{self._label}, {label}: {exc}") from exc
        self._done = bool(data.get("done"))
        return EnvStep(result=result, done=self._done, reward=reward, info=metadata)

    async def finalize(self, final_answer: Optional[str], status: AgentStatus) -> EnvOutcome:
        details: dict[str, Any] = {"reset": dict(self._reset_arguments)}
        response = await self.request({"type": "state"})
        if response.get("type") == "state":
            details["state"] = response["data"]
        else:  # the state is only recorded; an environment without one still ends its episode
            details["state_error"] = response["data"]
        state = details.get("state", {})
        episode_id = state.get("episode_id")
        return EnvOutcome(
            reward=0.0,  # OpenEnv's rewards come with the steps, which are in the return already
            metrics={"done": float(self._done)},
            details=details,
            episode_id=episode_id if isinstance(episode_id, str) else None,
        )

    async def close(self) -> None:
        slot, self._slot = self._slot, None
        if self._ws is not None or slot is not None:
            await self.context.run_cleanup(self._end(slot))

    async def _end(self, slot: Optional[asyncio.Semaphore]) -> None:
        try:
            await self._disconnect()
        except Exception as exc:  # the server drops the environment when the connection ends
            logger.warning("%s: closing the session failed: %s", self._label, exc)
        finally:
            if slot is not None:
                slot.release()

    def chat_adapter(
        self,
        processor: Any,
        reset: EnvReset,
        *,
        max_tool_calls: int,
        image_config: AgentImageConfig,
        image_preprocessor: Any,
    ) -> EnvironmentChatAdapter:
        return EnvironmentChatAdapter(
            processor,
            reset.tool_schemas,
            len(reset.images),
            max_tool_calls=max_tool_calls,
            image_config=image_config,
            image_preprocessor=image_preprocessor,
        )
