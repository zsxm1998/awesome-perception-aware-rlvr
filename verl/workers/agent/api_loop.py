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
"""An agent loop over messages, for models behind an OpenAI-compatible Chat Completions API.

It runs one episode of an agent environment like ``AgentLoop`` runs it for a local policy, with the same rules:
an ``<answer>`` in the reply ends the episode and takes precedence over a tool call; one tool call per turn; after
``max_tool_calls`` calls one more turn may only answer; ``max_tokens_per_turn`` caps each reply and
``max_response_tokens`` the tokens the model generates (as the API reports them; observations are not counted,
since the API's tokenizer is not ours). Tool calls come as the API's native ``tool_calls`` or, for endpoints
without function calling, as ``<tool_call>`` text in the reply (parsed like the local loop parses it).

Observations: the result of a native call is a ``tool`` message with the observation's text, followed by a
``user`` message with its images (Chat Completions tool messages take text only), or, with ``tool_images="tool"``
for endpoints that accept it (vLLM's server), one ``tool`` message with the text and images in order, as local
rollouts render it; the result of a text call is one ``user`` message with the observation's text and images in
order. Images are sent as PNG data URLs.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

import httpx
from PIL import Image

from ...utils.dataset import process_image
from .protocol import (
    AgentEnvironment,
    AgentLoopConfig,
    AgentMetrics,
    AgentSeedContext,
    AgentStatus,
    EnvOutcome,
    ToolCall,
    ToolResult,
)
from .tool_call import NativeToolCallCodec
from .tools.base import tool_error


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class APIError(RuntimeError):
    """A request that failed for good (after its retries) or a reply that is not a chat completion."""


class APIBudgetExceeded(APIError):
    """The client's token budget (prompt + completion tokens reported by the API) is used up."""


@dataclass(frozen=True)
class APISampling:
    temperature: float = 0.0
    top_p: Optional[float] = None
    seed: Optional[int] = None


def uses_environment_proxies(base_url: str) -> bool:
    """Whether requests to ``base_url`` follow the proxy variables (HTTPS_PROXY, ALL_PROXY, ...): not for an endpoint
    on this machine (a local vLLM server), which a proxy could not reach; yes otherwise."""
    return urlparse(base_url).hostname not in _LOCAL_HOSTS


def _token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class ChatCompletionsClient:
    """A shared client for one OpenAI-compatible endpoint: concurrency limit, retries and token accounting.

    The API key is sent in the Authorization header only. ``extra_body`` is merged into every request (a key set
    to None is removed from it). ``trust_env`` (proxy variables) defaults to off for local endpoints and on
    otherwise. ``max_total_tokens`` stops new requests once the reported prompt + completion tokens reach it (requests
    already sent finish); with it, a reply without token usage is an error.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        *,
        timeout_s: float = 300.0,
        retries: int = 5,
        backoff_s: float = 2.0,
        max_concurrency: int = 8,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_tokens_field: str = "max_tokens",
        max_total_tokens: Optional[int] = None,
        trust_env: Optional[bool] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = retries
        self.backoff_s = backoff_s
        self.extra_body = dict(extra_body or {})
        self.max_tokens_field = max_tokens_field
        self.max_total_tokens = max_total_tokens
        if trust_env is None:
            trust_env = uses_environment_proxies(self.base_url)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            self._client = httpx.AsyncClient(
                timeout=timeout_s,
                trust_env=trust_env,
                headers=headers,
                limits=httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency),
                transport=transport,
            )
        except ImportError as exc:  # e.g. ALL_PROXY=socks5://... without httpx's SOCKS support
            raise APIError(
                f"the proxy variables of this shell cannot be used ({exc}); install httpx[socks] or unset them"
            ) from exc
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]],
        *,
        max_tokens: int,
        sampling: APISampling,
        seed: Optional[int],
    ) -> dict[str, Any]:
        """``{"message", "finish_reason", "prompt_tokens", "completion_tokens"}`` of one chat completion."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            self.max_tokens_field: max_tokens,
            "temperature": sampling.temperature,
        }
        if sampling.top_p is not None:
            payload["top_p"] = sampling.top_p
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = list(tools)
        payload.update(self.extra_body)
        payload = {key: value for key, value in payload.items() if value is not None}
        last_error = ""
        for attempt in range(self.retries + 1):
            delay = 0.0
            try:
                async with self._semaphore:
                    # checked once the request has its slot (requests waiting for one see the usage of those before
                    # them), and the usage is counted before the slot is given up
                    if self.max_total_tokens is not None and self.total_tokens >= self.max_total_tokens:
                        raise APIBudgetExceeded(
                            f"the API token budget is used up ({self.total_tokens} >= {self.max_total_tokens} tokens)"
                        )
                    response = await self._client.post(self.base_url + "/chat/completions", json=payload)
                    self.requests += 1
                    if response.status_code == 200:
                        return self._read(response)
            except httpx.TransportError as exc:  # connection errors and timeouts
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise APIError(f"chat completion failed: {last_error}")
                retry_after = response.headers.get("retry-after")
                if retry_after is not None:
                    try:
                        delay = min(60.0, float(retry_after))
                    except ValueError:
                        delay = 0.0
            if attempt < self.retries:
                await asyncio.sleep(max(delay, self.backoff_s * 2**attempt))
        raise APIError(f"chat completion failed after {self.retries + 1} attempts: {last_error}")

    def _read(self, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise APIError(f"not a chat completion: {response.text[:300]}") from exc
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_tokens, completion_tokens = (
            usage.get(key) if _token_count(usage.get(key)) else None for key in ("prompt_tokens", "completion_tokens")
        )
        if self.max_total_tokens is not None and (prompt_tokens is None or completion_tokens is None):
            raise APIError(
                "the reply has no token usage (usage.prompt_tokens and usage.completion_tokens), which "
                "max_total_tokens counts; this endpoint cannot run with a token budget"
            )
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0
        return {
            "message": message,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass
class APITurn:
    """One reply of the model and what followed from it."""

    content: str
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    is_finalization_turn: bool = False
    native_tool_call: bool = False
    tool_call: Optional[ToolCall] = None
    tool_call_error: Optional[str] = None
    final_answer_error: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    env_reward: Optional[float] = None
    state_key: Optional[str] = None
    cost: float = 0.0
    env_done: bool = False
    env_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class APIEpisode:
    turns: list[APITurn]
    status: AgentStatus
    metrics: AgentMetrics
    final_answer: Optional[str] = None
    env_outcome: Optional[EnvOutcome] = None


def image_data_url(image: Image.Image, pixels: tuple[Optional[int], Optional[int]] = (None, None)) -> str:
    """A PNG data URL of ``image``, resized into ``pixels`` (min, max), as a local rollout prepares its images."""
    if pixels != (None, None):
        image = process_image(image, *pixels)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def chat_content(
    parts: Sequence[Mapping[str, Any]], images: Sequence[Image.Image], pixels: tuple[Optional[int], Optional[int]]
) -> Any:
    """Chat parts with ``{"type": "image"}`` placeholders as OpenAI content (images filled in order)."""
    content = []
    images = iter(images)
    for part in parts:
        if part.get("type") == "image":
            url = image_data_url(next(images), pixels)
            content.append({"type": "image_url", "image_url": {"url": url}})
        elif part.get("type") == "text":
            content.append({"type": "text", "text": part.get("text", "")})
        else:
            raise ValueError(f"unsupported content part {part.get('type')!r}")
    return content


def observation_parts(result: ToolResult) -> list[dict[str, Any]]:
    """An observation's text and image placeholders in order (as EnvironmentChatAdapter renders them)."""
    if result.parts is not None:
        return [dict(part) for part in result.parts]
    if result.observation_text is not None:
        text = result.observation_text
    else:  # e.g. the loop's malformed_tool_call error
        text = json.dumps(result.content, ensure_ascii=False, separators=(",", ":"))
    return [*({"type": "image"} for _ in result.images), {"type": "text", "text": text}]


def _reply_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):  # some endpoints return content parts
        return "".join(part.get("text", "") for part in content if isinstance(part, Mapping))
    return content or ""


def _native_tool_call(tool_calls: Sequence[Any]) -> tuple[Optional[ToolCall], Optional[str]]:
    if len(tool_calls) != 1:
        return None, "exactly one tool call is allowed per generation turn"
    function = tool_calls[0].get("function") if isinstance(tool_calls[0], Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    if not isinstance(name, str) or not name:
        return None, "tool call field 'name' must be a non-empty string"
    raw = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as exc:
        return None, f"tool call is not valid JSON: {exc}"
    if not isinstance(arguments, dict):
        return None, "tool call field 'arguments' must be a JSON object"
    return ToolCall(name=name, arguments=arguments, raw_text=raw if isinstance(raw, str) else json.dumps(raw)), None


async def run_api_episode(
    environment: AgentEnvironment,
    env_config: Mapping[str, Any],
    client: ChatCompletionsClient,
    *,
    env_seed: int,
    config: AgentLoopConfig,
    sampling: APISampling,
    seed_context: Optional[AgentSeedContext] = None,
    pixels: tuple[Optional[int], Optional[int]] = (None, None),
    tool_images: str = "user",
) -> APIEpisode:
    """Run one episode of ``environment`` with the model behind ``client``, then finalize and close it.

    ``pixels`` (min, max) resizes every image before it is sent, as a local rollout prepares images (the model's
    view of small text depends on it); (None, None) sends them as the environment gives them.

    ``tool_images`` places the images of a native call's observation: ``user`` in a user message after the tool
    message (Chat Completions tool messages take text only), ``tool`` in the tool message with its text, in order,
    for endpoints that accept that (vLLM's server), which renders observations as local rollouts do.
    """
    if tool_images not in {"user", "tool"}:
        raise ValueError("tool_images must be 'user' or 'tool'")

    codec = NativeToolCallCodec()
    turns: list[APITurn] = []
    metrics = AgentMetrics(visual_tokens=None)
    final_answer: Optional[str] = None
    status = AgentStatus.NO_ACTION
    try:
        reset = await environment.reset(env_config, env_seed)
        max_tool_calls = config.max_tool_calls if reset.tool_schemas else 0
        images = iter(reset.images)
        messages: list[dict[str, Any]] = []
        for message in reset.messages:
            content = message.get("content")
            if not isinstance(content, str):
                parts = list(content or [])
                content = chat_content(parts, [next(images) for part in parts if part.get("type") == "image"], pixels)
            messages.append({"role": message["role"], "content": content})
        tools = [dict(schema) for schema in reset.tool_schemas] or None
        completion_used = 0
        while True:
            remaining = config.max_response_tokens - completion_used
            if remaining <= 0:
                metrics.truncated = metrics.trajectory_length_limited = True
                status = AgentStatus.LENGTH_LIMIT
                break
            tools_allowed = metrics.tool_call_attempts < max_tool_calls
            max_tokens = min(remaining, config.max_tokens_per_turn)
            seed = sampling.seed if seed_context is None else seed_context.for_turn(len(turns))
            reply = await client.complete(messages, tools, max_tokens=max_tokens, sampling=sampling, seed=seed)
            if reply["completion_tokens"] is None:
                raise APIError(
                    "the reply has no token usage (usage.completion_tokens), which the trajectory's token budget "
                    "(max_response_tokens) counts"
                )
            message = reply["message"]
            text = _reply_text(message)
            completion_used += reply["completion_tokens"]
            metrics.action_tokens += reply["completion_tokens"]
            turn = APITurn(
                content=text,
                finish_reason=reply["finish_reason"],
                reasoning=message.get("reasoning_content") or message.get("reasoning"),
                prompt_tokens=reply["prompt_tokens"],
                completion_tokens=reply["completion_tokens"],
                is_finalization_turn=not tools_allowed,
            )
            turns.append(turn)
            if reply["finish_reason"] == "length":  # never act on a cut-off reply
                metrics.truncated = True
                metrics.turn_length_limited = max_tokens == config.max_tokens_per_turn
                metrics.trajectory_length_limited = max_tokens == remaining
                status = AgentStatus.LENGTH_LIMIT
                break
            parsed = codec.parse(text)
            if parsed.final_answer is not None:
                final_answer = parsed.final_answer
                status = AgentStatus.ANSWERED
                break
            if parsed.final_answer_error is not None:
                turn.final_answer_error = parsed.final_answer_error
                metrics.invalid_final_answers += 1
                status = AgentStatus.INVALID_FINAL_ANSWER
                break
            native_calls = list(message.get("tool_calls") or [])
            if native_calls:
                turn.native_tool_call = True
                call, call_error = _native_tool_call(native_calls)
            elif parsed.has_tool_attempt:
                call, call_error = parsed.tool_call, parsed.tool_call_error
            else:
                if completion_used >= config.max_response_tokens:
                    metrics.truncated = metrics.trajectory_length_limited = True
                    status = AgentStatus.LENGTH_LIMIT
                else:
                    status = AgentStatus.NO_ACTION
                break
            if not tools_allowed:
                metrics.tool_call_limit_reached = True
                status = AgentStatus.TOOL_CALL_LIMIT
                break
            metrics.tool_call_attempts += 1
            if call_error is not None:
                turn.tool_call_error = call_error
                result = tool_error("unknown", "malformed_tool_call", call_error)
            else:
                turn.tool_call = call
                env_step = await environment.step(call)
                result = env_step.result
                turn.env_reward, turn.state_key, turn.cost = env_step.reward, env_step.state_key, env_step.cost
                turn.env_info = dict(env_step.info)
            turn.tool_result = result
            if result.success:
                metrics.tool_execution_successes += 1
            else:
                metrics.tool_call_errors += 1
                if result.error_code == "tool_internal_error":
                    metrics.tool_internal_errors += 1
            if call_error is None and env_step.done:
                turn.env_done = True
                status = AgentStatus.ENVIRONMENT_DONE
                break
            parts = observation_parts(result)
            observation_text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
            if native_calls and tool_images == "tool":
                messages.append({"role": "assistant", "content": text, "tool_calls": native_calls})
                content = chat_content(parts, result.images, pixels) if result.images else observation_text
                for native in native_calls:
                    messages.append({"role": "tool", "tool_call_id": native.get("id"), "content": content})
            elif native_calls:
                messages.append({"role": "assistant", "content": text, "tool_calls": native_calls})
                for native in native_calls:
                    messages.append({"role": "tool", "tool_call_id": native.get("id"), "content": observation_text})
                if result.images:
                    image_parts = [{"type": "image"} for _ in result.images]
                    messages.append({"role": "user", "content": chat_content(image_parts, result.images, pixels)})
            else:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": chat_content(parts, result.images, pixels)})
            if result.success:
                metrics.tool_call_successes += 1
            metrics.visual_observations += len(result.images)
        outcome = await environment.finalize(final_answer, status)
        return APIEpisode(turns=turns, status=status, metrics=metrics, final_answer=final_answer, env_outcome=outcome)
    finally:
        await environment.close()
