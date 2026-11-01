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
"""Environments served over HTTP (the wire format is documented in docs/agent_environments.md)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import math
from typing import Any, Mapping, Optional

import httpx

from ....utils.dataset import process_image
from ..chat import EnvironmentChatAdapter
from ..protocol import AgentImageConfig, AgentStatus, EnvOutcome, EnvReset, EnvStep, ToolCall, ToolResult
from .registry import EnvironmentContext, register_environment


logger = logging.getLogger(__name__)
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class RemoteEnvironmentError(RuntimeError):
    """A remote environment call that failed after its retries, or answered outside the protocol."""


class _HTTPSession:
    """What the environments of one rollout batch share for one server: a client, a limit on concurrent
    requests and the images decoded so far (one object per distinct PNG, so the rollouts of a group see the
    same image object and hit vLLM's multimodal caches)."""

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float,
        max_concurrency: int,
        trust_env: bool = False,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.client = httpx.AsyncClient(
            base_url=url,
            timeout=timeout_s,
            trust_env=trust_env,
            limits=httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency),
            transport=transport,
        )
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.images: dict[str, Any] = {}

    async def post(self, path: str, payload: Mapping[str, Any], *, retries: int, backoff_s: float) -> dict:
        last_error = None
        for attempt in range(retries + 1):
            if attempt:
                await asyncio.sleep(backoff_s * 2 ** (attempt - 1))
            try:
                async with self.semaphore:
                    response = await self.client.post(path, json=payload)
            except httpx.TransportError as exc:  # connection errors and timeouts
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                continue
            if response.status_code != 200:
                raise RemoteEnvironmentError(
                    f"POST {path} returned HTTP {response.status_code}: {response.text[:300]}"
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise RemoteEnvironmentError(f"POST {path} did not return JSON: {response.text[:300]}") from exc
            if not isinstance(body, dict):
                raise RemoteEnvironmentError(f"POST {path} must return a JSON object")
            return body
        raise RemoteEnvironmentError(f"POST {path} failed after {retries + 1} attempts: {last_error}")

    def decode_image(self, png_base64: Any) -> Any:
        if not isinstance(png_base64, str):
            raise RemoteEnvironmentError("an image part needs png_base64 as a string")
        try:
            data = base64.b64decode(png_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RemoteEnvironmentError("an image part has invalid base64 data") from exc
        key = hashlib.sha256(data).hexdigest()
        if key not in self.images:
            self.images[key] = process_image(data, None, None)
        return self.images[key]

    async def aclose(self) -> None:
        await self.client.aclose()


@register_environment("remote_http")
class RemoteHTTPEnvironment:
    """An environment served over HTTP: ``POST /reset``, ``/step``, ``/finalize`` and ``/close``.

    Requests that fail with a network error, a timeout, HTTP 408, 429, 500, 502, 503 or 504 are retried with
    exponential backoff (``retries`` times). ``/step`` carries ``step_index``, so a server can answer a
    repeated step without executing it twice, and ``tool_calls_used``, the trajectory's calls that counted towards
    ``max_tool_calls`` before this one (malformed calls, which the server never sees, included). Images arrive as base64 PNG. ``max_images_per_step`` bounds the
    images of one step (it sizes vLLM's per-prompt image limit); a step that returns more fails. Proxy settings
    in the environment variables (HTTP_PROXY, ALL_PROXY, ...) are ignored unless ``trust_env`` is set, since an
    environment server usually runs on the same machine or cluster as the trainer.
    """

    def __init__(
        self,
        context: EnvironmentContext,
        *,
        url: str,
        timeout_s: float = 60.0,
        retries: int = 3,
        backoff_s: float = 0.5,
        max_concurrency: int = 64,
        max_images_per_step: int = 1,
        trust_env: bool = False,
    ) -> None:
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(f"remote_http needs an http(s) url, got {url!r}")
        for name, value in (("timeout_s", timeout_s), ("backoff_s", backoff_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"remote_http {name} must be a non-negative number")
        for name, value, low in (("retries", retries, 0), ("max_concurrency", max_concurrency, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"remote_http {name} must be an integer >= {low}")
        if (
            isinstance(max_images_per_step, bool)
            or not isinstance(max_images_per_step, int)
            or max_images_per_step < 0
        ):
            raise ValueError("remote_http max_images_per_step must be a non-negative integer")
        self.context = context
        self.url = url
        self.timeout_s = float(timeout_s)
        self.retries = retries
        self.backoff_s = float(backoff_s)
        self.max_concurrency = max_concurrency
        self.max_images_per_step = max_images_per_step
        self.trust_env = bool(trust_env)
        self.episode_id: Optional[str] = None
        self.step_index = 0

    @property
    def session_key(self) -> tuple:
        return ("remote_http", self.url, self.trust_env)

    def _session(self) -> _HTTPSession:
        return self.context.shared(
            self.session_key,
            lambda: _HTTPSession(
                self.url,
                timeout_s=self.timeout_s,
                max_concurrency=self.max_concurrency,
                trust_env=self.trust_env,
            ),
        )

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict:
        return await self._session().post(path, payload, retries=self.retries, backoff_s=self.backoff_s)

    def _chat_parts(self, items: Any, images: list) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            raise RemoteEnvironmentError("content must be a list of {type: text|image} parts")
        parts = []
        for item in items:
            kind = item.get("type") if isinstance(item, dict) else None
            if kind == "text":
                if not isinstance(item.get("text"), str):
                    raise RemoteEnvironmentError("a text part needs text as a string")
                parts.append({"type": "text", "text": item["text"]})
            elif kind == "image":
                images.append(self._session().decode_image(item.get("png_base64")))
                parts.append({"type": "image"})
            else:
                raise RemoteEnvironmentError(f"unsupported content part {item!r:.100}")
        return parts

    async def reset(self, config: Mapping[str, Any], seed: int) -> EnvReset:
        response = await self._post(
            "/reset",
            {
                "config": dict(config),
                "seed": seed,
                "max_tool_calls": self.context.max_tool_calls,
                "max_tool_calls_per_turn": self.context.max_tool_calls_per_turn,
            },
        )
        episode_id = response.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise RemoteEnvironmentError("/reset must return a non-empty episode_id")
        self.episode_id = episode_id
        self.step_index = 0
        images: list = []
        messages: list[dict[str, Any]] = []
        system_prompt = response.get("system_prompt")
        if system_prompt is not None:
            if not isinstance(system_prompt, str):
                raise RemoteEnvironmentError("/reset system_prompt must be a string or null")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self._chat_parts(response.get("user_content"), images)})
        if "tool_schemas" not in response:  # an empty list is a task without tools; a missing field is a bug
            raise RemoteEnvironmentError("/reset must return tool_schemas (an empty list for a task without tools)")
        tool_schemas = response["tool_schemas"]
        if not isinstance(tool_schemas, list) or not all(_is_function_schema(schema) for schema in tool_schemas):
            raise RemoteEnvironmentError(
                '/reset tool_schemas must be a list of {"type": "function", "function": {"name": ..., ...}}'
            )
        state_key = response.get("state_key")
        if state_key is not None and not isinstance(state_key, str):
            raise RemoteEnvironmentError("/reset state_key must be a string or null")
        return EnvReset(
            messages=messages,
            images=images,
            tool_schemas=tool_schemas,
            max_images_per_step=self.max_images_per_step,
            state_key=state_key,
        )

    async def step(self, call: ToolCall, *, tool_calls_used: Optional[int] = None) -> EnvStep:
        if self.episode_id is None:
            raise RemoteEnvironmentError("step() before reset()")
        step_index = self.step_index
        try:
            return await self._step(call, tool_calls_used)
        except RemoteEnvironmentError as exc:
            raise RemoteEnvironmentError(f"episode {self.episode_id}, step {step_index}: {exc}") from exc

    async def _step(self, call: ToolCall, tool_calls_used: Optional[int]) -> EnvStep:
        payload: dict[str, Any] = {"episode_id": self.episode_id, "step_index": self.step_index}
        if tool_calls_used is not None:
            payload["tool_calls_used"] = tool_calls_used
        payload["tool_call"] = {"name": call.name, "arguments": dict(call.arguments)}
        response = await self._post("/step", payload)
        self.step_index += 1
        images: list = []
        parts = self._chat_parts(response.get("content"), images)
        if len(images) > self.max_images_per_step:
            raise RemoteEnvironmentError(
                f"/step returned {len(images)} images, more than max_images_per_step={self.max_images_per_step}"
            )
        success = response.get("success")
        if not isinstance(success, bool):
            raise RemoteEnvironmentError("/step success must be a boolean")
        text = "".join(part["text"] for part in parts if part["type"] == "text")
        error_code = None
        if success:
            content: dict[str, Any] = {"status": "ok", "text": text}
        else:
            error_code = response.get("error_code") or "environment_error"
            if not isinstance(error_code, str):
                raise RemoteEnvironmentError("/step error_code must be a string")
            content = {"status": "error", "tool_name": call.name, "error": {"code": error_code, "message": text}}
        info = response.get("info", {})
        if not isinstance(info, dict):
            raise RemoteEnvironmentError("/step info must be an object")
        done = response.get("done", False)
        if not isinstance(done, bool):
            raise RemoteEnvironmentError("/step done must be a boolean")
        state_key = response.get("state_key")
        if state_key is not None and not isinstance(state_key, str):
            raise RemoteEnvironmentError("/step state_key must be a string or null")
        return EnvStep(
            result=ToolResult(
                tool_name=call.name,
                success=success,
                content=content,
                images=images,
                error_code=error_code,
                metadata=dict(info),
                parts=parts,
            ),
            done=done,
            reward=_optional_number(response.get("reward"), "/step reward"),
            state_key=state_key,
            cost=_optional_number(response.get("cost", 0.0), "/step cost") or 0.0,
            info=info,
        )

    async def finalize(self, final_answer: Optional[str], status: AgentStatus) -> EnvOutcome:
        if self.episode_id is None:
            raise RemoteEnvironmentError("finalize() before reset()")
        try:
            response = await self._post(
                "/finalize",
                {"episode_id": self.episode_id, "final_answer": final_answer, "status": AgentStatus(status).value},
            )
            details = response.get("details", {})
            if not isinstance(details, dict):
                raise RemoteEnvironmentError("/finalize details must be a JSON object")
            return EnvOutcome(
                reward=_optional_number(response.get("reward"), "/finalize reward"),
                components=_number_map(response.get("components", {}), "/finalize components"),
                metrics=_number_map(response.get("metrics", {}), "/finalize metrics"),
                details=details,
                episode_id=self.episode_id,
            )
        except RemoteEnvironmentError as exc:
            raise RemoteEnvironmentError(f"episode {self.episode_id}, finalize: {exc}") from exc

    async def close(self) -> None:
        if self.episode_id is None:
            return
        episode_id, self.episode_id = self.episode_id, None
        await self.context.run_cleanup(self._close_episode(episode_id))

    async def _close_episode(self, episode_id: str) -> None:
        try:
            await self._post("/close", {"episode_id": episode_id})
        except (RemoteEnvironmentError, httpx.HTTPError) as exc:
            # The episode's data is complete; the server reclaims unclosed episodes after its idle timeout.
            logger.warning("closing remote episode %s failed: %s", episode_id, exc)

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


def _is_function_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    function = schema.get("function")
    return (
        schema.get("type") == "function"
        and isinstance(function, dict)
        and isinstance(function.get("name"), str)
        and bool(function["name"])
        and isinstance(function.get("parameters", {}), dict)
    )


def _optional_number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RemoteEnvironmentError(f"{name} must be a finite number or null")
    return float(value)


def _number_map(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or any(item is None for item in value.values()):
        raise RemoteEnvironmentError(f"{name} must be an object of numbers")
    return {str(key): _optional_number(item, f"{name}.{key}") for key, item in value.items()}
