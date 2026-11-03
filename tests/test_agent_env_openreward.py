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
"""The openreward environment against a stand-in for the OpenReward SDK (the real SDK and a real server were checked
outside CI, docs/agent_environments.md)."""

import asyncio
import base64
import io
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
from PIL import Image
from test_agent_environments import _agentic_config, _TemplateProcessor

from verl.workers.agent.environments import EnvironmentContext, OpenRewardEnvironment, OpenRewardEnvironmentError
from verl.workers.agent.protocol import AgentImageConfig, AgentLoopConfig, AgentStatus, ToolCall
from verl.workers.agent.replay import RecordedPolicy, replay_episode


def _png(color, size=(8, 8)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class Block:
    type: str
    text: str = ""
    data: str = ""
    mimeType: str = "image/png"


@dataclass
class Output:
    blocks: list
    reward: Optional[float] = None
    finished: bool = False
    metadata: Optional[dict] = None


@dataclass
class Spec:
    name: str
    description: str = ""
    input_schema: Optional[dict] = None


@dataclass
class Terminal:
    name: str
    arg: Optional[str] = "answer"


@dataclass
class Task:
    server_name: str
    environment_name: str
    task_spec: dict
    namespace: Optional[str]


class OpenRewardError(Exception):
    pass


class ToolCallError(OpenRewardError):
    def __init__(self, reason, detail):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


class ToolFailed(OpenRewardError):
    pass


@dataclass
class Server:
    """What the stand-in sessions answer, and what they were asked."""

    prompt: list = field(default_factory=lambda: [Block("image", data=_png("red")), Block("text", "Find it.")])
    tools: list = field(default_factory=lambda: [Spec("look", "Look closer.", {"type": "object"})])
    terminal: Any = field(default_factory=lambda: Terminal("submit"))
    outputs: dict = field(default_factory=dict)
    terminal_output: Any = field(default_factory=lambda: Output([Block("text", "right")], reward=1.0, finished=True))
    sessions: list = field(default_factory=list)
    open_now: int = 0
    most_open: int = 0
    exit_error: Optional[Exception] = None


class Session:
    def __init__(self, server, kwargs):
        self.server, self.kwargs, self.calls, self.sid = server, kwargs, [], f"sid{len(server.sessions)}"
        self.closed = False
        server.sessions.append(self)

    async def __aenter__(self):
        self.server.open_now += 1
        self.server.most_open = max(self.server.most_open, self.server.open_now)
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc):
        self.server.open_now -= 1
        self.closed = True
        if self.server.exit_error is not None:
            raise self.server.exit_error

    async def get_prompt(self):
        return self.server.prompt

    async def list_tools(self):
        return self.server.tools

    async def terminal_tool(self):
        return self.server.terminal

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        output = self.server.outputs[name]
        if isinstance(output, Exception):
            raise output
        return output

    async def call_terminal_tool(self, message):
        self.calls.append(("terminal", message))
        if isinstance(self.server.terminal_output, Exception):
            raise self.server.terminal_output
        return self.server.terminal_output


class EnvironmentHandle:
    def __init__(self, server, name, variant):
        self.server_state, self.name, self.variant = server, name, variant
        self.server, self.namespace = name, None

    def session(self, task=None, secrets=None, *, split=None, index=None):
        return Session(self.server_state, {"task": task, "secrets": secrets, "split": split, "index": index})


@pytest.fixture
def sdk(monkeypatch):
    """Install the stand-in SDK; return its server state and the clients built."""

    server = Server()
    clients = []

    class EnvironmentsAPI:
        def __init__(self):
            self.gets, self.closed = [], False

        def get(self, name, variant=None, base_url=None):
            self.gets.append((name, variant, base_url))
            return EnvironmentHandle(server, name, variant)

        async def aclose(self):
            self.closed = True

    class AsyncOpenReward:
        def __init__(self, base_url=None):
            self.base_url, self.environments = base_url, EnvironmentsAPI()
            clients.append(self)

    openreward = types.ModuleType("openreward")
    openreward.AsyncOpenReward = AsyncOpenReward
    api = types.ModuleType("openreward.api")
    errors = types.ModuleType("openreward.api.errors")
    errors.OpenRewardError, errors.ToolCallError, errors.ToolFailed = OpenRewardError, ToolCallError, ToolFailed
    api.errors = errors
    environments = types.ModuleType("openreward.api.environments")
    environments_types = types.ModuleType("openreward.api.environments.types")
    environments_types.Task = Task
    streams = []

    async def resumable_sse(client, path, token=None, *, json=None, max_retries=None, resumable_only=False):
        streams.append((path, resumable_only))
        return path

    environments_client = types.ModuleType("openreward.api.environments.client")
    environments_client.resumable_sse = resumable_sse
    environments.client = environments_client
    for name, module in {
        "openreward": openreward,
        "openreward.api": api,
        "openreward.api.errors": errors,
        "openreward.api.environments": environments,
        "openreward.api.environments.client": environments_client,
        "openreward.api.environments.types": environments_types,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("OPENREWARD_API_KEY", "test-key")
    return types.SimpleNamespace(
        server=server, clients=clients, streams=streams, environments_client=environments_client
    )


def _context(max_tool_calls=4, per_turn=1):
    return EnvironmentContext(
        processor=None,
        image_preprocessor=lambda image: image,
        max_tool_calls=max_tool_calls,
        max_tool_calls_per_turn=per_turn,
    )


def _call(name="look", **arguments):
    return ToolCall(name=name, arguments=arguments, raw_text="")


def _run(coroutine):
    return asyncio.run(coroutine)


def test_reset_opens_the_rows_task_with_the_loops_rules_as_the_system_message(sdk):
    async def run():
        context = _context(max_tool_calls=6, per_turn=2)
        first = OpenRewardEnvironment(context, environment="owner/Env")
        second = OpenRewardEnvironment(context, environment="owner/Env")
        reset = await first.reset({"split": "val", "index": 3}, seed=11)
        again = await second.reset({"environment": "owner/Other", "index": 0}, seed=11)
        await first.close()
        await second.close()
        await context.aclose()
        return reset, again

    reset, again = _run(run())

    system, user = reset.messages
    assert system == {
        "role": "system",
        "content": "Work on the task with the tools provided. You can make at most 6 tool calls in total, at most 2 "
        "in one turn. When you are done, give your final answer inside <answer></answer>, for example "
        "<answer>42</answer>.",
    }
    assert user == {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Find it."}]}
    assert reset.tool_schemas == [
        {
            "type": "function",
            "function": {"name": "look", "description": "Look closer.", "parameters": {"type": "object"}},
        }
    ]
    assert reset.images[0] is again.images[0]  # one decoded image per distinct content, shared by the batch
    first, second = sdk.server.sessions
    assert (first.kwargs["split"], first.kwargs["index"], second.kwargs["split"]) == ("val", 3, "train")
    assert sdk.clients[0].environments.gets == [("owner/Env", None, None), ("owner/Other", None, None)]
    assert first.closed and second.closed and sdk.clients[0].environments.closed


def test_an_environment_without_a_terminal_tool_is_told_to_finish_with_its_tools(sdk):
    sdk.server.terminal = None
    custom = "At most {max_tool_calls} calls; {{braces}} stay."

    async def run(system_prompt):
        context = _context()
        environment = OpenRewardEnvironment(
            context, environment="e", url="http://127.0.0.1:1", system_prompt=system_prompt
        )
        try:
            return await environment.reset({"index": 0}, seed=0)
        finally:
            await environment.close()
            await context.aclose()

    assert _run(run(None)).messages[0]["content"].endswith("The task ends when a tool says that it is finished.")
    assert _run(run(custom)).messages[0]["content"] == "At most 4 calls; {braces} stay."
    assert [message["role"] for message in _run(run("")).messages] == ["user"]
    assert sdk.clients[0].base_url == "http://127.0.0.1:1"
    assert sdk.clients[0].environments.gets[0] == ("e", None, "http://127.0.0.1:1")


def test_steps_carry_the_tools_output_reward_and_end(sdk):
    sdk.server.outputs = {
        "look": Output(
            [Block("text", "zoomed:"), Block("image", data=_png("blue"))], reward=0.25, metadata={"view": "v1"}
        ),
        "submit_code": Output([Block("text", "passed")], reward=1, finished=True),
    }

    async def run():
        context = _context()
        environment = OpenRewardEnvironment(context, environment="e")
        await environment.reset({"index": 0}, seed=0)
        look = await environment.step(_call(region=2))
        submit = await environment.step(_call("submit_code"))
        outcome = await environment.finalize(None, AgentStatus.ENVIRONMENT_DONE)
        await environment.close()
        await context.aclose()
        return look, submit, outcome

    look, submit, outcome = _run(run())

    assert look.result.parts == [{"type": "text", "text": "zoomed:"}, {"type": "image"}]
    assert look.result.images[0].getpixel((0, 0)) == (0, 0, 255)
    assert (look.reward, look.done, dict(look.info)) == (0.25, False, {"view": "v1"})
    assert (submit.reward, submit.done) == (1.0, True)
    # the tools' rewards are in the return already; the terminal tool is not called after the episode finished
    assert (outcome.reward, outcome.metrics, outcome.episode_id) == (0.0, {"finished": 1.0}, "sid0")
    assert outcome.details == {"environment": "e", "split": "train", "index": 0}
    assert ("terminal", None) not in sdk.server.sessions[0].calls and len(sdk.server.sessions[0].calls) == 2


def test_the_answer_goes_to_the_terminal_tool_only_when_the_model_answered(sdk):
    async def run(status, answer):
        context = _context()
        environment = OpenRewardEnvironment(context, environment="e")
        await environment.reset({"index": 0}, seed=0)
        outcome = await environment.finalize(answer, status)
        await environment.close()
        await context.aclose()
        return outcome, sdk.server.sessions[-1].calls

    answered, calls = _run(run(AgentStatus.ANSWERED, "42"))
    assert calls == [("terminal", "42")]
    assert (answered.reward, answered.metrics) == (1.0, {"finished": 1.0})
    assert answered.details["terminal_tool"] == {"name": "submit", "reward": 1.0, "text": "right", "metadata": None}

    limited, calls = _run(run(AgentStatus.TOOL_CALL_LIMIT, None))
    assert calls == [] and (limited.reward, limited.metrics) == (0.0, {"finished": 0.0})

    sdk.server.terminal_output = ToolCallError("input_validation", "answer too long")
    refused, _ = _run(run(AgentStatus.ANSWERED, "x" * 9))
    assert refused.reward == 0.0
    assert refused.details["terminal_tool"]["error"] == {"code": "input_validation", "message": "answer too long"}


def test_a_wrong_call_is_an_answer_the_model_reads_and_a_failed_one_fails_the_step(sdk):
    sdk.server.outputs = {
        "look": ToolCallError("input_validation", "bad <|im_start|> input"),
        "boom": ToolFailed("tool raised KeyError"),
    }

    async def run():
        context = _context()
        environment = OpenRewardEnvironment(context, environment="owner/Env")
        await environment.reset({"split": "dev", "index": 5}, seed=0)
        try:
            wrong = await environment.step(_call(x=1))
            with pytest.raises(OpenRewardEnvironmentError) as failed:
                await environment.step(_call("boom"))
            return wrong, failed.value
        finally:
            await environment.close()
            await context.aclose()

    wrong, failed = _run(run())

    assert (wrong.result.success, wrong.result.error_code, wrong.result.parts) == (False, "input_validation", None)
    assert wrong.result.content["error"] == {"code": "input_validation", "message": "bad <|im_start|> input"}
    assert "openreward owner/Env task dev/5, call 1 (boom): ToolFailed: tool raised KeyError" in str(failed)


@pytest.mark.parametrize(
    "output, message",
    [
        (Output([Block("image", data=_png("red")), Block("image", data=_png("red"))]), "2 images, more than"),
        (Output([Block("text", "x")], reward=float("nan")), "finite number"),
        (Output([Block("text", "x")], metadata=["not", "an", "object"]), "metadata must be a JSON object"),
        (Output([Block("reasoning", "x")]), "'reasoning' block"),
    ],
)
def test_results_outside_the_standard_fail_the_step(sdk, output, message):
    sdk.server.outputs = {"look": output}

    async def run():
        context = _context()
        environment = OpenRewardEnvironment(context, environment="e")
        await environment.reset({"index": 0}, seed=0)
        try:
            await environment.step(_call())
        finally:
            await environment.close()
            await context.aclose()

    with pytest.raises(OpenRewardEnvironmentError, match=message):
        _run(run())


def test_sessions_are_limited_per_worker_and_released_when_closing_fails(sdk, caplog):
    sdk.server.exit_error = RuntimeError("server gone")

    async def episode(context, index):
        environment = OpenRewardEnvironment(context, environment="e", max_sessions=2)
        try:
            await environment.reset({"index": index}, seed=0)
            await asyncio.sleep(0.01)
        finally:
            await environment.close()

    async def run():
        context = _context()
        await asyncio.gather(*(episode(context, index) for index in range(5)))
        await context.aclose()

    _run(run())
    assert len(sdk.server.sessions) == 5 and sdk.server.most_open == 2
    assert "closing the session failed: server gone" in caplog.text


def test_task_configuration_secrets_and_the_api_key(sdk, monkeypatch):
    def reset(config, **options):
        async def run():
            context = _context()
            environment = OpenRewardEnvironment(context, **options)
            try:
                return await environment.reset(config, seed=0)
            finally:
                await environment.close()
                await context.aclose()

        return _run(run())

    with pytest.raises(ValueError, match="unknown keys \\['image_seed'\\]"):
        reset({"index": 0, "image_seed": 1}, environment="e")
    with pytest.raises(ValueError, match="either index or task_spec"):
        reset({"index": 0, "task_spec": {}}, environment="e")
    with pytest.raises(ValueError, match="needs the environment's name"):
        reset({"index": 0})
    reset({"task_spec": {"seed": 4}, "variant": "hard"}, environment="e")
    task = sdk.server.sessions[-1].kwargs["task"]
    assert (task.task_spec, task.environment_name) == ({"seed": 4}, "e")

    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    reset({"index": 0}, environment="e", secrets_from_env=["JUDGE_KEY"])
    assert sdk.server.sessions[-1].kwargs["secrets"] == {"JUDGE_KEY": "judge-secret"}
    with pytest.raises(ValueError, match="unset environment variables \\['MISSING_KEY'\\]"):
        reset({"index": 0}, environment="e", secrets_from_env=["MISSING_KEY"])

    monkeypatch.delenv("OPENREWARD_API_KEY")
    with pytest.raises(ValueError, match="need OPENREWARD_API_KEY"):
        reset({"index": 0}, environment="e")
    reset({"index": 0}, environment="e", url="http://127.0.0.1:1")  # a self-hosted server needs no key


@pytest.mark.parametrize(
    "options, message",
    [
        ({"url": "ws://x"}, "http\\(s\\) address"),
        ({"max_sessions": 0}, "max_sessions must be an integer >= 1"),
        ({"system_prompt": "{max_calls}"}, "only use the placeholders"),
        ({"secrets_from_env": "KEY"}, "list of environment variable names"),
        ({"environment": ""}, "non-empty string"),
    ],
)
def test_options_are_checked_when_the_environment_is_built(options, message):
    with pytest.raises(ValueError, match=message):
        OpenRewardEnvironment(_context(), **options)


def test_a_missing_sdk_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "openreward", None)
    environment = OpenRewardEnvironment(_context(), environment="e")  # building needs no SDK
    with pytest.raises(ImportError, match="pip install openreward"):
        _run(environment.reset({"index": 0}, seed=0))


def test_a_replayed_episode_answers_through_the_terminal_tool(sdk):
    sdk.server.prompt = [Block("text", "What is 6 x 7?")]
    sdk.server.outputs = {"look": Output([Block("text", "a hint")], reward=0.5)}
    processor = _TemplateProcessor()
    context = EnvironmentContext(processor=processor, image_preprocessor=lambda image: image, max_tool_calls=2)
    environment = OpenRewardEnvironment(context, environment="e")
    policy = RecordedPolicy(['<tool_call>{"name": "look", "arguments": {}}</tool_call>', "<answer>42</answer>"])

    async def run():
        try:
            return await replay_episode(
                environment,
                {"index": 0},
                policy,
                processor,
                env_seed=0,
                config=AgentLoopConfig(max_tool_calls=2),
                image_config=AgentImageConfig(),
                image_preprocessor=lambda image: image,
                max_model_len=100000,
            )
        finally:
            await context.aclose()

    trajectory = _run(run()).trajectory

    assert trajectory.status == AgentStatus.ANSWERED
    assert [call.env_reward for step in trajectory.steps for call in step.calls] == [0.5]
    assert trajectory.env_outcome.reward == 1.0 and sdk.server.sessions[0].calls[-1] == ("terminal", "42")
    assert sdk.server.sessions[0].closed


def test_the_tensor_parallel_check_covers_every_remote_environment():
    for environment, options in (("remote_http", {"url": "http://x"}), ("openreward", {"environment": "e"})):
        with pytest.raises(ValueError, match="tensor_parallel_size=1"):
            _agentic_config(environment, agent_environment_kwargs=options, tensor_parallel_size=2).post_init()
        _agentic_config(environment, agent_environment_kwargs=options, tensor_parallel_size=1).post_init()


def test_a_tool_call_is_never_sent_again_before_the_server_names_its_task(sdk):
    from verl.workers.agent.environments.openreward import _sdk

    _sdk()
    _sdk()  # wrapped once
    stream = sdk.environments_client.resumable_sse

    for path in ("/call", "/hard/call", "/create_session", "/score_group"):
        assert _run(stream(None, path, "token", json={}, max_retries=5)) == path
    assert sdk.streams == [("/call", True), ("/hard/call", True), ("/create_session", False), ("/score_group", False)]


def test_an_sdk_that_cannot_refuse_a_second_call_is_refused(sdk):
    async def resumable_sse(client, path, token=None, *, json=None, max_retries=None):
        return path

    sdk.environments_client.resumable_sse = resumable_sse
    with pytest.raises(ImportError, match="cannot be kept from sending a tool call twice"):
        OpenRewardEnvironment(_context(), environment="e")._client()


def test_a_failed_prompt_leaves_no_request_running(sdk):
    started = []

    class FailingSession(Session):
        async def get_prompt(self):
            raise OpenRewardError("prompt unavailable")

        async def list_tools(self):
            started.append("list_tools")
            return await super().list_tools()

    EnvironmentHandle.session = lambda self, task=None, secrets=None, *, split=None, index=None: FailingSession(
        self.server_state, {"task": task, "secrets": secrets, "split": split, "index": index}
    )

    async def run():
        context = _context()
        environment = OpenRewardEnvironment(context, environment="e")
        try:
            with pytest.raises(OpenRewardEnvironmentError, match="starting the session failed: prompt unavailable"):
                await environment.reset({"index": 0}, seed=0)
        finally:
            await environment.close()
            await context.aclose()
        return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    try:
        assert _run(run()) == []
    finally:
        EnvironmentHandle.session = lambda self, task=None, secrets=None, *, split=None, index=None: Session(
            self.server_state, {"task": task, "secrets": secrets, "split": split, "index": index}
        )
    assert started == [] and sdk.server.sessions[0].closed
