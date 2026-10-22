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
"""Replay a scripted policy in an agent environment, e.g. to make demonstrations for SFT warm-up.

The policy writes each assistant turn; the turn goes through the same agent loop, environment, chat adapter and
image handling as a rollout (``run_environment_episode``), so a replayed episode is tokenized exactly as the RL
rollout would tokenize the same actions. A policy is any object with

    act(step)                       -> the next assistant turn; ``step`` is the EnvStep of the previous tool call
                                       (None on the first turn); may be async
    start(env_config, seed, reset)  -> optional, called once before the first turn; may be async

A turn is the raw assistant text (``str`` or ``{"text": ...}``), tokenized as is, or
``{"content": ..., "tool_call": {"name": ..., "arguments": {...}}}``, which the model's chat template renders
into its native tool-call text (``content`` alone is a turn without a tool call).
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .episode import EpisodeRun, run_environment_episode
from .protocol import (
    AgentEnvironment,
    AgentImageConfig,
    AgentLoopConfig,
    AgentStatus,
    EnvOutcome,
    EnvReset,
    EnvStep,
    GenerationOutput,
    GenerationRequest,
    ToolCall,
)
from .tool_call import NativeToolCallCodec


class ReplayError(ValueError):
    """A scripted episode that the rollout could not have produced (e.g. a malformed call or an over-long turn)."""


@dataclass(frozen=True)
class PolicyTurn:
    text: Optional[str] = None
    content: Optional[str] = None
    tool_call: Optional[Mapping[str, Any]] = None


def policy_turn(value: Any) -> PolicyTurn:
    """A policy's turn as given: a string, ``{"text": ...}`` or ``{"content": ..., "tool_call": {...}}``."""
    if isinstance(value, PolicyTurn):
        return value
    if isinstance(value, str):
        return PolicyTurn(text=value)
    if not isinstance(value, Mapping):
        raise ReplayError(f"a policy turn must be a string or an object, got {type(value).__name__}")
    unknown = set(value) - {"text", "content", "tool_call"}
    if unknown:
        raise ReplayError(f"unknown policy turn fields {sorted(unknown)}; use text, or content and tool_call")
    if "text" in value:
        if set(value) != {"text"} or not isinstance(value["text"], str):
            raise ReplayError("a raw turn is {'text': <string>} alone")
        return PolicyTurn(text=value["text"])
    content = value.get("content", "")
    if not isinstance(content, str):
        raise ReplayError("turn content must be a string")
    tool_call = value.get("tool_call")
    if tool_call is not None:
        if (
            not isinstance(tool_call, Mapping)
            or not isinstance(tool_call.get("name"), str)
            or not tool_call["name"]
            or not isinstance(tool_call.get("arguments", {}), Mapping)
            or set(tool_call) - {"name", "arguments"}
        ):
            raise ReplayError("tool_call must be {'name': <non-empty string>, 'arguments': <object>}")
        tool_call = {"name": tool_call["name"], "arguments": dict(tool_call.get("arguments", {}))}
    return PolicyTurn(content=content, tool_call=tool_call)


class AssistantTurnRenderer:
    """Native assistant text of a turn, rendered by the model's chat template (with the episode's tools)."""

    _SENTINEL = "__EASYR1_REPLAY_ASSISTANT_SENTINEL_5C1B__"
    _USER = [{"role": "user", "content": "x"}]

    def __init__(self, processor: Any, tool_schemas: Sequence[Mapping[str, Any]]):
        self.processor = processor
        self.tools = [dict(schema) for schema in tool_schemas] or None
        self._prefix = self._render(self._USER, add_generation_prompt=True)
        rendered = self._render([*self._USER, {"role": "assistant", "content": self._SENTINEL}])
        if not rendered.startswith(self._prefix) or rendered.count(self._SENTINEL) != 1:
            raise ReplayError("the chat template does not render an assistant turn after its generation prompt")
        self._suffix = rendered.split(self._SENTINEL, 1)[1]

    def render(self, turn: PolicyTurn) -> str:
        if turn.text is not None:
            return turn.text
        message: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_call is not None:
            message["tool_calls"] = [{"type": "function", "function": dict(turn.tool_call)}]
        rendered = self._render([*self._USER, message])
        if not rendered.startswith(self._prefix) or not rendered.endswith(self._suffix):
            raise ReplayError("the chat template rendered the assistant turn outside its generation prompt")
        return rendered[len(self._prefix) : len(rendered) - len(self._suffix)]

    def _render(self, messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
        return self.processor.apply_chat_template(
            messages, tools=self.tools, add_generation_prompt=add_generation_prompt, tokenize=False
        )


class RecordedPolicy:
    """A fixed sequence of turns, e.g. one line of an episodes file."""

    def __init__(self, turns: Iterable[Any]):
        self.turns = [policy_turn(turn) for turn in turns]
        self.position = 0

    def act(self, step: Optional[EnvStep]) -> PolicyTurn:
        if self.position >= len(self.turns):
            raise ReplayError(
                f"the episode asked for turn {self.position + 1}, but only {len(self.turns)} are recorded"
            )
        turn = self.turns[self.position]
        self.position += 1
        return turn


class _ReplayEnvironment:
    """The episode's environment as the loop sees it: starts the policy after ``reset``, binds the backend to the
    episode's chat template and keeps the last step for the policy."""

    def __init__(self, environment: AgentEnvironment, policy: Any, backend: "ScriptedBackend"):
        self.environment = environment
        self.policy = policy
        self.backend = backend
        self.last_step: Optional[EnvStep] = None
        self.steps = 0

    async def reset(self, config: Mapping[str, Any], seed: int) -> EnvReset:
        reset = await self.environment.reset(config, seed)
        start = getattr(self.policy, "start", None)
        if start is not None:
            await _maybe_await(start(dict(config), seed, reset))
        return reset

    async def step(self, call: ToolCall) -> EnvStep:
        self.last_step = await self.environment.step(call)
        self.steps += 1
        return self.last_step

    async def finalize(self, final_answer: Optional[str], status: AgentStatus) -> Optional[EnvOutcome]:
        return await self.environment.finalize(final_answer, status)

    async def close(self) -> None:
        await self.environment.close()

    def chat_adapter(self, processor: Any, reset: EnvReset, **kwargs: Any) -> Any:
        adapter = self.environment.chat_adapter(processor, reset, **kwargs)
        self.backend.bind(reset, adapter.assistant_termination_token_ids)
        return adapter


class ScriptedBackend:
    """A generation backend whose actions come from a policy, tokenized with the model's tokenizer and ended
    with the template's assistant terminator, as a model that stops at its end-of-turn token would end them."""

    def __init__(self, policy: Any, processor: Any):
        self.policy = policy
        self.environment: Optional[_ReplayEnvironment] = None
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.codec = NativeToolCallCodec()
        self.renderer: Optional[AssistantTurnRenderer] = None
        self.termination_token_ids: tuple[int, ...] = ()
        self.special_tokens = sorted({token for token in getattr(self.tokenizer, "all_special_tokens", []) if token})
        self.turns = 0
        self._steps_at_last_turn = 0
        self._last_text = ""

    def bind(self, reset: EnvReset, termination_token_ids: Sequence[int]) -> None:
        self.renderer = AssistantTurnRenderer(self.processor, reset.tool_schemas)
        self.termination_token_ids = tuple(termination_token_ids)

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        if self.renderer is None or self.environment is None:
            raise RuntimeError("ScriptedBackend.bind() must run before the first turn")
        step = None
        if self.turns:
            if self.environment.steps == self._steps_at_last_turn:
                parsed = self.codec.parse(self._last_text)
                raise ReplayError(
                    f"turn {self.turns} is not a well-formed tool call: {parsed.tool_call_error or 'no tool call'}"
                )
            step = self.environment.last_step
        turn = policy_turn(await _maybe_await(self.policy.act(step)))
        text = self.renderer.render(turn)
        for token in self.special_tokens:
            if token in text:
                raise ReplayError(f"turn {self.turns + 1} contains the special token {token!r}")
        token_ids = [*self.tokenizer.encode(text, add_special_tokens=False), *self.termination_token_ids]
        if len(token_ids) > request.max_new_tokens:
            raise ReplayError(
                f"turn {self.turns + 1} has {len(token_ids)} tokens, more than the {request.max_new_tokens} the "
                "rollout would allow it (max_tokens_per_turn, the response budget or the context window)"
            )
        self.turns += 1
        self._steps_at_last_turn = self.environment.steps
        self._last_text = text
        return GenerationOutput(token_ids=token_ids, text=text, finish_reason="stop")


async def replay_episode(
    environment: AgentEnvironment,
    env_config: Mapping[str, Any],
    policy: Any,
    processor: Any,
    *,
    env_seed: int,
    config: AgentLoopConfig,
    image_config: AgentImageConfig,
    image_preprocessor: Any,
    max_model_len: int,
) -> EpisodeRun:
    """Run ``policy`` for one episode of ``environment`` as the rollout would; the trajectory's tokens, masks and
    images are what training on the same actions would see."""

    backend = ScriptedBackend(policy, processor)
    backend.environment = replay_environment = _ReplayEnvironment(environment, policy, backend)
    run = await run_environment_episode(
        replay_environment,
        env_config,
        processor,
        env_seed=env_seed,
        config=config,
        image_config=image_config,
        image_preprocessor=image_preprocessor,
        make_backend=lambda: backend,
        seed_context=None,
        max_model_len=max_model_len,
    )
    for index, step in enumerate(run.trajectory.steps):
        if step.tool_call_error is not None:  # e.g. a malformed last turn, which ended the episode
            raise ReplayError(f"turn {index + 1} is not a well-formed tool call: {step.tool_call_error}")
    return run


def load_episodes(path: str) -> list[dict[str, Any]]:
    """An episodes file: one JSON object per line, ``{"env_config": {...}, "turns": [...]}`` with optional
    ``id`` and ``seed`` (else derived from the run's seed and the line number)."""
    episodes = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            episode = json.loads(line)
            if not isinstance(episode, dict) or not isinstance(episode.get("env_config"), dict):
                raise ReplayError(f"{path}:{line_number}: an episode needs env_config (an object)")
            if not isinstance(episode.get("turns"), list) or not episode["turns"]:
                raise ReplayError(f"{path}:{line_number}: an episode needs a non-empty turns list")
            episodes.append(episode)
    return episodes


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value
