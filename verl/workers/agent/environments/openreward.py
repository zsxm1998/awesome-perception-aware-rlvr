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
"""Environments of the Open Reward Standard (ORS), through the official ``openreward`` SDK (docs/agent_environments.md).

One ORS session per trajectory: the session's task prompt and tools start the episode, every tool call goes to the
session, and the environment's ``@terminal`` tool, if it has one, scores the text the model writes inside
<answer></answer>. TRL, ROLL and SkyRL use the same SDK; it handles the platform's session creation, keep-alive
pings, resumed event streams and secrets.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
from typing import Any, Mapping, Optional, Sequence

from ..chat import EnvironmentChatAdapter
from ..protocol import AgentImageConfig, AgentStatus, EnvOutcome, EnvReset, EnvStep, ToolCall, ToolResult
from .registry import EnvironmentContext, register_environment
from .standards import (
    ANSWER_SYSTEM_PROMPT,
    TOOLS_SYSTEM_PROMPT,
    DecodedImages,
    check_count,
    check_system_prompt,
    finite_or_none,
    function_schema,
    system_messages,
)


logger = logging.getLogger(__name__)
OPENREWARD_TESTED_VERSION = "0.1.167"
_TASK_KEYS = frozenset({"environment", "variant", "split", "index", "task_spec"})


def _sdk() -> tuple[Any, Any]:
    try:
        import openreward
        from openreward.api import errors
        from openreward.api.environments import client as environments_client
    except ImportError as exc:
        raise ImportError(
            "the openreward environment needs the OpenReward SDK: pip install openreward "
            f"(tested with {OPENREWARD_TESTED_VERSION})"
        ) from exc
    _call_once(environments_client)
    return openreward, errors


def _call_once(environments_client: Any) -> None:
    """Make the SDK send a tool call at most once.

    The SDK streams ``/call`` with retries; a retry after the server has named the call's task resumes that task, but
    one before it re-sends the call, which the server may run a second time. ``resumable_sse`` refuses the latter with
    ``resumable_only=True``, which ``call_tool`` does not pass (0.1.167). This sets it for every ``/call`` stream, so
    a failure before the task is named raises (and fails the training step) instead of running the action again.
    """
    stream = environments_client.resumable_sse
    if getattr(stream, "_calls_once", False):
        return
    if "resumable_only" not in inspect.signature(stream).parameters:
        raise ImportError(
            "this openreward SDK cannot be kept from sending a tool call twice (its resumable_sse has no "
            f"resumable_only); use the tested version {OPENREWARD_TESTED_VERSION}"
        )

    @functools.wraps(stream)
    async def call_stream(client: Any, path: str, *args: Any, **kwargs: Any) -> Any:
        if path.endswith("/call"):
            kwargs["resumable_only"] = True
        return await stream(client, path, *args, **kwargs)

    call_stream._calls_once = True
    environments_client.resumable_sse = call_stream


class OpenRewardEnvironmentError(RuntimeError):
    """An ORS environment that failed (its session ended, a tool raised, the network) or answered outside the
    standard; it fails the training step, as a remote_http server error does."""


class _Client:
    """What the environments of one rollout batch share: one SDK client (its HTTP sessions belong to the batch's
    event loop), the session slots and the decoded images."""

    def __init__(self, url: Optional[str], max_sessions: int) -> None:
        openreward, _ = _sdk()
        self.url = url
        client = openreward.AsyncOpenReward(base_url=url) if url else openreward.AsyncOpenReward()
        self.environments_api = client.environments
        self.slots = asyncio.Semaphore(max_sessions)
        self.images = DecodedImages()
        self._environments: dict[tuple[str, Optional[str]], Any] = {}

    def environment(self, name: str, variant: Optional[str]) -> Any:
        key = (name, variant)
        if key not in self._environments:
            # a self-hosted server serves its sessions at its own address (base_url), not on the platform's hosts
            self._environments[key] = self.environments_api.get(name, variant=variant, base_url=self.url)
        return self._environments[key]

    async def aclose(self) -> None:
        await self.environments_api.aclose()


@register_environment("openreward")
class OpenRewardEnvironment:
    """An episode of an ORS environment: one session of the task that the row's configuration names.

    The row's configuration (``agent_environment_config_key``) holds ``environment`` (default: the option), ``split``
    (default: the option) and ``index``, or ``task_spec`` instead of the split and index; ``variant`` names one
    environment of a server that serves several. Options: ``environment`` (an openreward.ai name such as
    ``owner/Name``, or the environment's name on a self-hosted server), ``split``, ``url`` (a self-hosted server;
    the platform otherwise, with ``OPENREWARD_API_KEY``), ``max_sessions`` (sessions open at a time per rollout
    worker, each held from ``reset`` to ``close``), ``max_images_per_step``, ``system_prompt`` (the system message
    with the loop's rules; ``""`` for none) and ``secrets_from_env`` (names of environment variables whose values
    the session gets as secrets).
    """

    # its episodes live on a server, so every rank of a tensor-parallel group would open them again
    remote = True

    def __init__(
        self,
        context: EnvironmentContext,
        *,
        environment: Optional[str] = None,
        split: str = "train",
        url: Optional[str] = None,
        max_sessions: int = 64,
        max_images_per_step: int = 1,
        system_prompt: Optional[str] = None,
        secrets_from_env: Sequence[str] = (),
    ) -> None:
        if environment is not None and (not isinstance(environment, str) or not environment):
            raise ValueError("openreward environment must be a non-empty string")
        if not isinstance(split, str) or not split:
            raise ValueError("openreward split must be a non-empty string")
        if url is not None and (not isinstance(url, str) or not url.startswith(("http://", "https://"))):
            raise ValueError(f"openreward url must be an http(s) address, got {url!r}")
        check_count("openreward max_sessions", max_sessions, 1)
        check_count("openreward max_images_per_step", max_images_per_step, 0)
        check_system_prompt("openreward", system_prompt)
        if isinstance(secrets_from_env, str) or not all(isinstance(name, str) and name for name in secrets_from_env):
            raise ValueError("openreward secrets_from_env must be a list of environment variable names")
        self.context = context
        self.environment = environment
        self.split = split
        self.url = url
        self.max_sessions = max_sessions
        self.max_images_per_step = max_images_per_step
        self.system_prompt = system_prompt
        self.secrets_from_env = list(secrets_from_env)
        self._session: Any = None
        self._slot: Optional[asyncio.Semaphore] = None
        self._task: dict[str, Any] = {}
        self._label = "openreward episode"
        self._terminal_tool: Any = None
        self._finished = False
        self._step_index = 0

    def _client(self) -> _Client:
        return self.context.shared(("openreward", self.url), lambda: _Client(self.url, self.max_sessions))

    def _task_of(self, config: Mapping[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(config) - _TASK_KEYS)
        if unknown:
            raise ValueError(f"openreward task configuration has unknown keys {unknown}; known: {sorted(_TASK_KEYS)}")
        task = {
            "environment": config.get("environment", self.environment),
            "variant": config.get("variant"),
            "split": config.get("split", self.split),
        }
        if not isinstance(task["environment"], str) or not task["environment"]:
            raise ValueError("openreward needs the environment's name, in the row's configuration or the options")
        if task["variant"] is not None and not isinstance(task["variant"], str):
            raise ValueError("openreward variant must be a string")
        if ("index" in config) == ("task_spec" in config):
            raise ValueError("an openreward task configuration gives either index or task_spec")
        if "task_spec" in config:
            if not isinstance(config["task_spec"], Mapping):
                raise ValueError("openreward task_spec must be a JSON object")
            task["task_spec"] = dict(config["task_spec"])
            del task["split"]
        else:
            check_count("openreward index", config["index"], 0)
            task["index"] = config["index"]
        return task

    def _secrets(self) -> Optional[dict[str, str]]:
        missing = [name for name in self.secrets_from_env if not os.environ.get(name)]
        if missing:
            raise ValueError(f"openreward secrets_from_env names unset environment variables {missing}")
        return {name: os.environ[name] for name in self.secrets_from_env} or None

    def _parts(self, blocks: Any, images: list, where: str) -> list[dict[str, Any]]:
        parts = []
        for block in blocks or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                parts.append({"type": "text", "text": block.text})
            elif kind == "image":
                images.append(self._client().images.decode(block.data))
                parts.append({"type": "image"})
            else:
                raise OpenRewardEnvironmentError(f"{self._label}: {where} has a {kind!r} block; text and images only")
        return parts

    async def reset(self, config: Mapping[str, Any], seed: int) -> EnvReset:
        # ``seed`` is not used: an ORS task is fixed by its index, and every rollout of a group opens the same one
        task = self._task_of(config)
        _sdk()  # a missing SDK fails here, with the install command
        secrets = self._secrets()
        if self.url is None and not os.environ.get("OPENREWARD_API_KEY"):
            raise ValueError("openreward environments on openreward.ai need OPENREWARD_API_KEY (or set url)")
        client = self._client()
        where = task.get("task_spec", f"{task.get('split')}/{task.get('index')}")
        self._label = f"openreward {task['environment']} task {where}"
        self._task = task
        await client.slots.acquire()
        self._slot = client.slots
        try:
            environment = client.environment(task["environment"], task["variant"])
            if "task_spec" in task:
                from openreward.api.environments.types import Task

                sdk_task = Task(
                    server_name=environment.server,
                    environment_name=environment.name,
                    task_spec=task["task_spec"],
                    namespace=environment.namespace,
                )
                self._session = environment.session(sdk_task, secrets)
            else:
                self._session = environment.session(secrets=secrets, split=task["split"], index=task["index"])
            await self._session.__aenter__()
            # one after the other: a failure of one leaves no request running into the session's cleanup
            blocks = await self._session.get_prompt()
            tools = await self._session.list_tools()
            self._terminal_tool = await self._session.terminal_tool()  # None when the environment has none
        except Exception as exc:  # the SDK's errors, aiohttp's, timeouts
            raise OpenRewardEnvironmentError(f"{self._label}: starting the session failed: {exc}") from exc
        images: list = []
        user = self._parts(blocks, images, "the task prompt")
        template = ANSWER_SYSTEM_PROMPT if self._terminal_tool is not None else TOOLS_SYSTEM_PROMPT
        messages = system_messages(
            self.system_prompt,
            template,
            max_tool_calls=self.context.max_tool_calls,
            max_tool_calls_per_turn=self.context.max_tool_calls_per_turn,
        )
        messages.append({"role": "user", "content": user})
        try:
            schemas = [function_schema(tool.name, tool.description, tool.input_schema) for tool in tools]
        except ValueError as exc:
            raise OpenRewardEnvironmentError(f"{self._label}: {exc}") from exc
        return EnvReset(
            messages=messages, images=images, tool_schemas=schemas, max_images_per_step=self.max_images_per_step
        )

    async def step(self, call: ToolCall) -> EnvStep:
        if self._session is None:
            raise OpenRewardEnvironmentError("step() before reset()")
        _, errors = _sdk()
        label = f"{self._label}, call {self._step_index} ({call.name})"
        self._step_index += 1
        try:
            output = await self._session.call_tool(call.name, dict(call.arguments))
        except errors.ToolCallError as exc:
            # the model's call was wrong (unknown tool, invalid input): an ordinary answer it reads. The text may echo
            # the model's input, so it is shown as escaped JSON, like the loop's own errors (no parts).
            return EnvStep(
                result=ToolResult(
                    tool_name=call.name,
                    success=False,
                    content={
                        "status": "error",
                        "tool_name": call.name,
                        "error": {"code": exc.reason, "message": exc.detail},
                    },
                    error_code=str(exc.reason),
                )
            )
        except Exception as exc:  # the session ended (a tool that raised ends it), the network, a timeout
            raise OpenRewardEnvironmentError(f"{label}: {type(exc).__name__}: {exc}") from exc
        images: list = []
        parts = self._parts(output.blocks, images, f"the result of {call.name}")
        if len(images) > self.max_images_per_step:
            raise OpenRewardEnvironmentError(
                f"{label}: {len(images)} images, more than max_images_per_step={self.max_images_per_step}"
            )
        metadata = output.metadata if output.metadata is not None else {}
        if not isinstance(metadata, dict):
            raise OpenRewardEnvironmentError(f"{label}: the result's metadata must be a JSON object")
        try:
            reward = finite_or_none(output.reward, "a tool's reward")
        except ValueError as exc:
            raise OpenRewardEnvironmentError(f"{label}: {exc}") from exc
        self._finished = bool(output.finished)
        text = "".join(part["text"] for part in parts if part["type"] == "text")
        return EnvStep(
            result=ToolResult(
                tool_name=call.name,
                success=True,
                content={"status": "ok", "text": text},
                images=images,
                metadata=dict(metadata),
                parts=parts,
            ),
            done=self._finished,
            reward=reward,
            info=metadata,
        )

    async def finalize(self, final_answer: Optional[str], status: AgentStatus) -> EnvOutcome:
        if self._session is None:
            raise OpenRewardEnvironmentError("finalize() before reset()")
        _, errors = _sdk()
        details: dict[str, Any] = {key: value for key, value in self._task.items() if value is not None}
        reward = 0.0  # the tools' rewards are in the return already (step rewards)
        terminal = self._terminal_tool
        if terminal is not None and not self._finished and AgentStatus(status) == AgentStatus.ANSWERED:
            record: dict[str, Any] = {"name": terminal.name}
            try:
                output = await self._session.call_terminal_tool(final_answer or "")
            except errors.ToolCallError as exc:
                record["error"] = {"code": exc.reason, "message": exc.detail}
            except Exception as exc:
                raise OpenRewardEnvironmentError(f"{self._label}, terminal tool: {type(exc).__name__}: {exc}") from exc
            else:
                try:
                    terminal_reward = finite_or_none(output.reward, "the terminal tool's reward")
                except ValueError as exc:
                    raise OpenRewardEnvironmentError(f"{self._label}: {exc}") from exc
                reward = terminal_reward or 0.0
                self._finished = bool(output.finished)
                record.update(
                    reward=terminal_reward,
                    text="".join(
                        block.text for block in output.blocks or [] if getattr(block, "type", None) == "text"
                    ),
                    metadata=output.metadata,
                )
            details["terminal_tool"] = record
        return EnvOutcome(
            reward=reward,
            metrics={"finished": float(self._finished)},
            details=details,
            episode_id=getattr(self._session, "sid", None),
        )

    async def close(self) -> None:
        session, self._session = self._session, None
        slot, self._slot = self._slot, None
        if session is not None or slot is not None:
            await self.context.run_cleanup(self._end(session, slot))

    async def _end(self, session: Any, slot: Optional[asyncio.Semaphore]) -> None:
        try:
            if session is not None:
                await session.__aexit__(None, None, None)  # deletes the session on the server
        except Exception as exc:  # the server drops idle sessions; the episode's data is complete
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
