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
"""``tool_calls_used``: the agent loop tells an environment how many of the trajectory's calls counted towards
``max_tool_calls`` before each call, malformed calls (which never reach the environment) included."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

from verl.workers.agent.environments import EnvironmentContext, LocalToolEnvironment
from verl.workers.agent.loop import AgentLoop
from verl.workers.agent.protocol import (
    AgentLoopConfig,
    AgentStatus,
    EncodedObservation,
    EnvReset,
    EnvStep,
    GenerationOutput,
    ToolCall,
    ToolResult,
)


ROOT = Path(__file__).resolve().parents[1]


def _call(region):
    return "<tool_call>" + json.dumps({"name": "view", "arguments": {"region": region}}) + "</tool_call>"


class _Backend:
    def __init__(self, texts):
        self.texts = list(texts)

    async def generate(self, request):
        text = self.texts.pop(0)
        return GenerationOutput(token_ids=[ord(char) for char in text][:6] or [1], text=text, finish_reason="stop")


class _Encoder:
    max_tool_calls = None

    async def encode(self, request):
        return EncodedObservation(
            token_ids=[7] * len(request.results), effective_token_count=len(request.results), visual_token_count=0
        )


class _CountingEnvironment(LocalToolEnvironment):
    """Records the region and the calls used of each call it receives."""

    def __init__(self):
        super().__init__()
        self.received = []

    async def step(self, call, *, tool_calls_used):
        self.received.append((call.arguments["region"], tool_calls_used))
        return EnvStep(result=ToolResult(call.name, True, {}, observation_text="ok"))


class _PlainEnvironment(LocalToolEnvironment):
    """An environment written before ``tool_calls_used``."""

    def __init__(self):
        super().__init__()
        self.received = []

    async def step(self, call):
        self.received.append(call.arguments["region"])
        return EnvStep(result=ToolResult(call.name, True, {}, observation_text="ok"))


def _run(environment, texts, *, per_turn=1, max_tool_calls=6):
    config = AgentLoopConfig(max_tool_calls=max_tool_calls, max_tool_calls_per_turn=per_turn)
    loop = AgentLoop(_Backend(texts), _Encoder(), environment, config)
    return asyncio.run(loop.run(prompt_ids=[1], source_images=[]))


def test_the_loop_counts_malformed_and_earlier_calls_of_the_turn():
    environment = _CountingEnvironment()
    trajectory = _run(
        environment,
        ["<tool_call>not json</tool_call>", _call(1), _call(2) + _call(3), "<answer>A</answer>"],
        per_turn=2,
    )

    # the malformed call of the first turn counts but never reaches the environment
    assert environment.received == [(1, 1), (2, 2), (3, 3)]
    assert trajectory.status == AgentStatus.ANSWERED and trajectory.metrics.tool_call_attempts == 4


def test_an_environment_without_the_keyword_runs_as_before():
    environment = _PlainEnvironment()
    trajectory = _run(environment, ["<tool_call>not json</tool_call>", _call(1), "<answer>A</answer>"])

    assert environment.received == [1] and trajectory.status == AgentStatus.ANSWERED


def test_remote_http_sends_the_calls_used_with_each_step():
    from verl.workers.agent.environments import RemoteHTTPEnvironment
    from verl.workers.agent.environments.remote import _HTTPSession

    requests = []

    def server(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/reset":
            tool = {"type": "function", "function": {"name": "view"}}
            text = {"type": "text", "text": "Task"}
            return httpx.Response(200, json={"episode_id": "e1", "user_content": [text], "tool_schemas": [tool]})
        if request.url.path == "/step":
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "success": True})
        return httpx.Response(200, json={})

    context = EnvironmentContext(processor=object(), image_preprocessor=lambda image: image, max_tool_calls=4)
    environment = RemoteHTTPEnvironment(context, url="http://env.test", backoff_s=0.0)
    context.shared(
        environment.session_key,
        lambda: _HTTPSession(
            "http://env.test", timeout_s=5.0, max_concurrency=4, transport=httpx.MockTransport(server)
        ),
    )

    async def episode():
        try:
            await environment.reset({}, seed=0)
            await environment.step(ToolCall(name="view", arguments={"region": 1}, raw_text=""), tool_calls_used=2)
            await environment.step(ToolCall(name="view", arguments={"region": 2}, raw_text=""))
        finally:
            await environment.close()
            await context.aclose()

    asyncio.run(episode())
    steps = [body for path, body in requests if path == "/step"]
    assert steps[0] == {
        "episode_id": "e1",
        "step_index": 0,
        "tool_calls_used": 2,
        "tool_call": {"name": "view", "arguments": {"region": 1}},
    }
    assert steps[1]["step_index"] == 1 and "tool_calls_used" not in steps[1]  # called without it, as by hand


def test_the_example_server_gives_the_calls_used_to_an_episode_that_takes_them():
    sys.path.insert(0, str(ROOT / "examples" / "agent_env"))
    from server import EpisodeServer, text_part

    class _Episode:
        def __init__(self):
            self.received = []

        def reset(self, config, seed, max_tool_calls):
            return {"user_content": [text_part("q")], "tool_schemas": [], "system_prompt": None}

        def finalize(self, final_answer, status):
            return {"reward": 0.0}

    class CountingEpisode(_Episode):
        def step(self, name, arguments, tool_calls_used):
            self.received.append(tool_calls_used)
            return {"content": [text_part(f"{4 - tool_calls_used - 1} calls left")], "success": True}

    class PlainEpisode(_Episode):
        def step(self, name, arguments):
            self.received.append(None)
            return {"content": [text_part("ok")], "success": True}

    reset = {"config": {}, "seed": 0, "max_tool_calls": 4}
    call = {"name": "view", "arguments": {}}
    for episode_class, expected in ((CountingEpisode, [1]), (PlainEpisode, [None])):
        server = EpisodeServer(episode_class)
        episode_id = server.handle("/reset", reset)["episode_id"]
        server.handle("/step", {"episode_id": episode_id, "step_index": 0, "tool_calls_used": 1, "tool_call": call})
        assert server._episodes[episode_id].episode.received == expected


class _ReplayTarget:
    """What a replay forwards to: an environment that takes the calls used."""

    def __init__(self):
        self.received = []

    async def step(self, call, *, tool_calls_used):
        self.received.append(tool_calls_used)
        return EnvStep(result=ToolResult(call.name, True, {}))


def test_replay_and_the_api_loop_pass_the_calls_used_on():
    from verl.workers.agent.api_loop import APISampling, ChatCompletionsClient, run_api_episode
    from verl.workers.agent.replay import _ReplayEnvironment

    target = _ReplayTarget()
    wrapper = _ReplayEnvironment(target, policy=None, backend=None)
    asyncio.run(wrapper.step(ToolCall(name="view", arguments={}, raw_text=""), tool_calls_used=3))
    assert target.received == [3] and len(wrapper.history) == 1

    class APIEnvironment:
        def __init__(self):
            self.received = []

        async def reset(self, config, seed):
            tool = {"type": "function", "function": {"name": "view", "parameters": {"type": "object"}}}
            return EnvReset(messages=[{"role": "user", "content": "Which region?"}], images=[], tool_schemas=[tool])

        async def step(self, call, *, tool_calls_used):
            self.received.append(tool_calls_used)
            return EnvStep(result=ToolResult(call.name, True, {}, observation_text="ok"))

        async def finalize(self, final_answer, status):
            return None

        async def close(self):
            pass

    native = {"id": "a", "type": "function", "function": {"name": "view", "arguments": "{}"}}
    replies = iter(
        [
            {"content": "", "tool_calls": [{**native, "function": {"name": "view", "arguments": "not json"}}]},
            {"content": "", "tool_calls": [native]},
            {"content": "<answer>2</answer>"},
        ]
    )

    def endpoint(request):
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return httpx.Response(
            200, json={"choices": [{"message": next(replies), "finish_reason": "stop"}], "usage": usage}
        )

    client = ChatCompletionsClient("http://api.test/v1", "m", None, transport=httpx.MockTransport(endpoint))
    environment = APIEnvironment()

    async def run():
        try:
            config = AgentLoopConfig(max_tool_calls=4)
            return await run_api_episode(
                environment, {}, client, env_seed=0, config=config, sampling=APISampling(seed=1)
            )
        finally:
            await client.aclose()

    episode = asyncio.run(run())
    assert environment.received == [1] and episode.status == AgentStatus.ANSWERED
