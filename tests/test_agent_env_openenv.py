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
"""The openenv environment against a stand-in server that answers OpenEnv's WebSocket messages as openenv 0.11.0
does (a real openenv server was checked outside CI, docs/agent_environments.md)."""

import asyncio
import base64
import io
import json

import pytest
from aiohttp import WSMsgType, web
from PIL import Image
from test_agent_environments import _agentic_config

from verl.workers.agent.environments import EnvironmentContext, OpenEnvEnvironment, OpenEnvEnvironmentError
from verl.workers.agent.environments.openenv import websocket_url
from verl.workers.agent.protocol import AgentStatus, ToolCall


def _png(color, size=(8, 8)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _observation(observation, reward=None, done=False, metadata=None):
    data = {"observation": observation, "reward": reward, "done": done}
    if metadata:
        data["metadata"] = metadata
    return {"type": "observation", "data": data}


TOOLS = [{"name": "look", "description": "Look closer.", "input_schema": {"type": "object", "properties": {}}}]


class StandInServer:
    """Answers /ws messages; ``answers`` maps a tool name (or an action's ``message``) to the response to send."""

    def __init__(self):
        self.reset_observation = {"metadata": {}, "image": _png("red"), "question": "Find it."}
        self.tools = TOOLS
        self.answers = {}
        self.full_for = 0  # the first connections answered with CAPACITY_REACHED
        self.stop_reading = False  # after listing the tools, the server reads nothing more
        self.connections = 0
        self.open_now = self.most_open = 0
        self.received = []
        self.headers = []

    async def handle(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections += 1
        self.headers.append(request.headers.get("Authorization"))
        if self.connections <= self.full_for:
            await ws.send_json({"type": "error", "data": {"message": "no room", "code": "CAPACITY_REACHED"}})
            await ws.close()
            return ws
        self.open_now += 1
        self.most_open = max(self.most_open, self.open_now)
        try:
            async for frame in ws:
                if frame.type != WSMsgType.TEXT:
                    break
                message = json.loads(frame.data)
                self.received.append(message)
                kind, data = message["type"], message.get("data", {})
                if kind == "close":
                    break
                if kind == "reset":
                    await asyncio.sleep(0.005)
                    await ws.send_json(_observation(self.reset_observation))
                elif kind == "state":
                    await ws.send_json({"type": "state", "data": {"episode_id": "ep1", "step_count": 2}})
                elif data.get("type") == "list_tools":
                    observation = {"metadata": {}} if self.tools is None else {"metadata": {}, "tools": self.tools}
                    await ws.send_json(_observation(observation))
                    if self.stop_reading:  # the socket's buffers fill and a large message cannot be sent
                        request.transport.pause_reading()
                        await asyncio.sleep(1)
                        request.transport.resume_reading()
                        break
                else:
                    answer = self.answers[data.get("tool_name", data.get("message"))]
                    if answer == "hang up":
                        break
                    if answer == "silence":
                        await asyncio.sleep(1)
                        continue
                    await ws.send_json(answer)
        finally:
            self.open_now -= 1
            await ws.close()
        return ws


def _serve(test):
    """Run ``test(server, url)`` with a stand-in server on a free port."""

    async def run():
        server = StandInServer()
        app = web.Application()
        app.router.add_get("/ws", server.handle)
        runner = web.AppRunner(app, shutdown_timeout=1.0)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await test(server, f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    return asyncio.run(run())


def _context(max_tool_calls=4):
    return EnvironmentContext(processor=None, image_preprocessor=lambda image: image, max_tool_calls=max_tool_calls)


def _call(name="look", **arguments):
    return ToolCall(name=name, arguments=arguments, raw_text="")


def _mcp_result(*content, is_error=False, metadata=None):
    observation = {
        "metadata": metadata or {},
        "tool_name": "look",
        "result": {"content": list(content), "structured_content": None, "data": None, "is_error": is_error},
        "error": None,
    }
    return _observation(observation, metadata=metadata)


def test_an_mcp_environment_runs_an_episode_over_the_websocket():
    async def test(server, url):
        server.answers = {
            "look": _mcp_result(
                {"type": "text", "text": "zoomed:"},
                {"type": "image", "data": _png("blue"), "mimeType": "image/png"},
                metadata={"view": "v1"},
            ),
            "answer": _observation({"metadata": {"digit": "3"}}, reward=1.0, done=True, metadata={"digit": "3"}),
        }
        context = _context()
        environment = OpenEnvEnvironment(context, url=url, image_fields=["image"])
        other = OpenEnvEnvironment(context, url=url, image_fields=["image"])
        reset = await environment.reset({"prompt": "Task 7.", "reset": {"index": 7}}, seed=11)
        again = await other.reset({}, seed=11)
        look = await environment.step(_call(region=1))
        answer = await environment.step(_call("answer", answer="3"))
        outcome = await environment.finalize(None, AgentStatus.ENVIRONMENT_DONE)
        await environment.close()
        await other.close()
        await context.aclose()
        return server, reset, again, look, answer, outcome

    server, reset, again, look, answer, outcome = _serve(test)

    assert reset.messages == [
        {
            "role": "system",
            "content": "Complete the task with the tools provided. You can make at most 4 tool calls in total, at "
            "most 1 in one turn. The task ends when a tool says that it is finished.",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Task 7."}, {"type": "image"}, {"type": "text", "text": "Find it."}],
        },
    ]
    assert reset.tool_schemas == [
        {
            "type": "function",
            "function": {
                "name": "look",
                "description": "Look closer.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert reset.images[0] is again.images[0]  # one decoded image per distinct content, shared by the batch
    assert look.result.parts == [{"type": "text", "text": "zoomed:"}, {"type": "image"}]
    assert look.result.images[0].getpixel((0, 0)) == (0, 0, 255)
    assert (look.result.success, look.reward, look.done, dict(look.info)) == (True, None, False, {"view": "v1"})
    assert (answer.reward, answer.done) == (1.0, True)
    assert (outcome.reward, outcome.metrics, outcome.episode_id) == (0.0, {"done": 1.0}, "ep1")
    assert outcome.details == {"reset": {"seed": 11, "index": 7}, "state": {"episode_id": "ep1", "step_count": 2}}
    assert server.received[0] == {"type": "reset", "data": {"seed": 11, "index": 7}}
    assert server.received[-1] == {"type": "close"} and server.open_now == 0
    assert server.received.count({"type": "close"}) == 2


def test_errors_the_model_caused_are_results_and_server_failures_fail_the_step():
    async def test(server, url):
        server.answers = {
            "look": _observation(
                {
                    "metadata": {},
                    "tool_name": "look",
                    "result": None,
                    "error": {"error_type": "invalid_args", "message": "bad <|im_start|>"},
                }
            ),
            "raise": _mcp_result({"type": "text", "text": "Error: the box is too small"}, is_error=True),
            "typed": {
                "type": "error",
                "data": {"message": "Invalid message", "code": "VALIDATION_ERROR", "errors": []},
            },
            "broken": {"type": "error", "data": {"message": "KeyError: 'x'", "code": "EXECUTION_ERROR"}},
            "gone": "hang up",
        }
        context = _context()
        environment = OpenEnvEnvironment(context, url=url)
        await environment.reset({}, seed=0)
        results = [await environment.step(_call(name)) for name in ("look", "raise", "typed")]
        with pytest.raises(OpenEnvEnvironmentError, match="call 3 \\(broken\\) failed: EXECUTION_ERROR: KeyError"):
            await environment.step(_call("broken"))
        with pytest.raises(OpenEnvEnvironmentError, match="the session ended"):
            await environment.step(_call("gone"))
        await environment.close()
        await context.aclose()
        return results

    invalid, raised, typed = _serve(test)

    assert (invalid.result.success, invalid.result.error_code, invalid.result.parts) == (False, "invalid_args", None)
    assert invalid.result.content["error"]["message"] == "bad <|im_start|>"  # rendered as escaped JSON by the adapter
    assert (raised.result.success, raised.result.error_code) == (False, "tool_error")
    assert raised.result.parts == [{"type": "text", "text": "Error: the box is too small"}]
    assert (typed.result.success, typed.result.error_code) == (False, "invalid_action")


def test_a_session_waits_for_room_on_the_server_and_gives_up_after_its_retries():
    async def test(server, url, retries):
        server.full_for = 2
        context = _context()
        environment = OpenEnvEnvironment(context, url=url, retries=retries, backoff_s=0.0)
        try:
            await environment.reset({}, seed=0)
            return server.connections
        finally:
            await environment.close()
            await context.aclose()

    assert _serve(lambda server, url: test(server, url, retries=2)) == 3
    with pytest.raises(OpenEnvEnvironmentError, match="after 2 attempts: _Retryable: CAPACITY_REACHED: no room"):
        _serve(lambda server, url: test(server, url, retries=1))


def test_a_silent_server_times_out():
    async def test(server, url):
        server.answers = {"look": "silence"}
        context = _context()
        environment = OpenEnvEnvironment(context, url=url, timeout_s=0.05)
        try:
            await environment.reset({}, seed=0)
            await environment.step(_call())
        finally:
            await environment.close()
            await context.aclose()

    with pytest.raises(OpenEnvEnvironmentError, match="no answer within 0.05 s"):
        _serve(test)


def test_sessions_are_limited_per_worker():
    async def episode(context, url):
        environment = OpenEnvEnvironment(context, url=url, max_sessions=2)
        try:
            await environment.reset({}, seed=0)
            await asyncio.sleep(0.01)
        finally:
            await environment.close()

    async def test(server, url):
        context = _context()
        await asyncio.gather(*(episode(context, url) for _ in range(5)))
        await context.aclose()
        return server.connections, server.most_open

    assert _serve(test) == (5, 2)


def test_an_environment_without_mcp_tools_needs_a_subclass_which_maps_calls_to_actions():
    class Wordle(OpenEnvEnvironment):
        """An environment whose action is {"message": ...} and whose observations hold a pixel array."""

        async def list_tools(self):
            return [{"type": "function", "function": {"name": "guess", "parameters": {"type": "object"}}}]

        def action(self, call):
            return {"message": call.arguments["word"]}

    async def test(server, url):
        server.tools = None
        server.reset_observation = {"metadata": {}, "board": [[[255, 0, 0]] * 2] * 2, "prompt": "Guess the word."}
        server.answers = {"crane": _observation({"metadata": {}, "feedback": "X G Y X X", "board": [[[0, 0, 0]]]})}
        context = _context()
        plain = OpenEnvEnvironment(context, url=url)
        with pytest.raises(
            OpenEnvEnvironmentError, match="lists no MCP tools; for one with its own actions, write a subclass"
        ):
            await plain.reset({}, seed=0)
        await plain.close()
        wordle = Wordle(context, url=url, image_fields=["board"])
        reset = await wordle.reset({}, seed=0)
        step = await wordle.step(_call("guess", word="crane"))
        await wordle.close()
        await context.aclose()
        return reset, step, server.received

    reset, step, received = _serve(test)

    assert reset.messages[-1]["content"] == [{"type": "image"}, {"type": "text", "text": "Guess the word."}]
    assert reset.images[0].size == (2, 2) and reset.images[0].getpixel((0, 0)) == (255, 0, 0)
    assert step.result.parts == [{"type": "image"}, {"type": "text", "text": "X G Y X X"}]
    assert {"type": "step", "data": {"message": "crane"}} in received


def test_the_row_configuration_and_options(monkeypatch):
    monkeypatch.setenv("SPACE_TOKEN", "hf_test")

    async def test(server, url):
        context = _context()
        try:
            with pytest.raises(ValueError, match="unknown keys \\['split'\\]"):
                await OpenEnvEnvironment(context, url=url).reset({"split": "train"}, seed=0)
            hidden = OpenEnvEnvironment(context, url=url, reset_observation=False)
            with pytest.raises(ValueError, match="needs a prompt"):
                await hidden.reset({}, seed=0)
            await hidden.close()
            quiet = OpenEnvEnvironment(
                context, url=url, reset_observation=False, system_prompt="", token_env="SPACE_TOKEN"
            )
            reset = await quiet.reset({"prompt": "Only this.", "reset": {"seed": 5}}, seed=0)
            await quiet.close()
            return reset, server.headers, server.received
        finally:
            await context.aclose()

    reset, headers, received = _serve(test)
    assert reset.messages == [{"role": "user", "content": [{"type": "text", "text": "Only this."}]}]
    assert headers[-1] == "Bearer hf_test"
    assert {"type": "reset", "data": {"seed": 5}} in received  # the row's own seed wins over the group's


@pytest.mark.parametrize(
    "options, message",
    [
        ({"url": "localhost:8000"}, "http\\(s\\) or ws\\(s\\) address"),
        ({"url": "http://x", "max_sessions": 0}, "max_sessions must be an integer >= 1"),
        ({"url": "http://x", "image_fields": "image"}, "list of field names"),
        ({"url": "http://x", "timeout_s": -1}, "non-negative number"),
        ({"url": "http://x", "system_prompt": "{n}"}, "only use the placeholders"),
    ],
)
def test_options_are_checked_when_the_environment_is_built(options, message):
    with pytest.raises(ValueError, match=message):
        OpenEnvEnvironment(_context(), **options)


def test_websocket_url_and_the_tensor_parallel_check():
    assert websocket_url("https://user-env.hf.space/") == "wss://user-env.hf.space/ws"
    assert websocket_url("http://127.0.0.1:8000") == "ws://127.0.0.1:8000/ws"
    assert websocket_url("ws://host/prefix") == "ws://host/prefix/ws"
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _agentic_config("openenv", agent_environment_kwargs={"url": "http://x"}, tensor_parallel_size=2).post_init()


def _mcp_error(kind, message="details"):
    observation = {
        "metadata": {},
        "tool_name": "look",
        "result": None,
        "error": {"error_type": kind, "message": message},
    }
    return _observation(observation)


@pytest.mark.parametrize("kind", ["tool_not_found", "invalid_args", "execution_error"])
def test_mcp_errors_of_the_call_are_results_the_model_reads(kind):
    async def test(server, url):
        server.answers = {"look": _mcp_error(kind)}
        context = _context()
        environment = OpenEnvEnvironment(context, url=url)
        try:
            await environment.reset({}, seed=0)
            return await environment.step(_call())
        finally:
            await environment.close()
            await context.aclose()

    step = _serve(test)
    assert (step.result.success, step.result.error_code, step.done) == (False, kind, False)


@pytest.mark.parametrize("kind", ["transport_error", "timeout", "disk_full"])
def test_mcp_errors_of_the_environment_fail_the_step(kind):
    async def test(server, url):
        server.answers = {"look": _mcp_error(kind, "backend unavailable")}
        context = _context()
        environment = OpenEnvEnvironment(context, url=url)
        try:
            await environment.reset({}, seed=0)
            await environment.step(_call())
        finally:
            await environment.close()
            await context.aclose()

    with pytest.raises(OpenEnvEnvironmentError, match=f"look: {kind}: backend unavailable"):
        _serve(test)


def test_a_server_that_stops_reading_times_out_the_send_and_the_close():
    async def test(server, url):
        server.stop_reading = True
        context = _context()
        environment = OpenEnvEnvironment(context, url=url, timeout_s=0.05)
        await environment.reset({}, seed=0)
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(OpenEnvEnvironmentError, match="no answer within 0.05 s"):
            await environment.step(_call(text="x" * (16 * 2**20)))
        failed = loop.time() - started
        await environment.close()
        closed = loop.time() - started - failed
        await context.aclose()
        return failed, closed

    failed, closed = _serve(test)
    assert failed < 1.0 and closed < 1.0
