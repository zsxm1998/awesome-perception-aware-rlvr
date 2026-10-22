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
"""API models in agent environments (verl/workers/agent/api_loop.py, eval --backend api), against a scripted
OpenAI-compatible endpoint."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from verl.workers.agent.api_loop import (
    APIBudgetExceeded,
    APIError,
    APISampling,
    ChatCompletionsClient,
    run_api_episode,
    uses_environment_proxies,
)
from verl.workers.agent.protocol import (
    AgentLoopConfig,
    AgentStatus,
    EnvOutcome,
    EnvReset,
    EnvStep,
    ToolCall,
    ToolResult,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

TOOL = {"type": "function", "function": {"name": "zoom_in", "parameters": {"type": "object", "properties": {}}}}


class _Environment:
    """One image, one zoom tool that returns a text and an image, and an outcome with details."""

    def __init__(self, context=None):
        self.calls, self.closed = [], False

    async def reset(self, config, seed):
        self.seed = seed
        return EnvReset(
            messages=[
                {"role": "system", "content": "Find the digit."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Which digit?"}]},
            ],
            images=[Image.new("RGB", (64, 48), (200, 0, 0))],
            tool_schemas=[TOOL],
        )

    async def step(self, call):
        self.calls.append((call.name, dict(call.arguments)))
        result = ToolResult(
            call.name,
            True,
            {},
            images=[Image.new("RGB", (32, 32))],
            parts=[{"type": "text", "text": "Zoomed:"}, {"type": "image"}],
        )
        return EnvStep(result=result, state_key="s1", cost=1.0)

    async def finalize(self, final_answer, status):
        return EnvOutcome(reward=float(final_answer == "7"), details={"status": status.value}, episode_id="e1")

    async def close(self):
        self.closed = True

    def chat_adapter(self, *args, **kwargs):
        raise AssertionError("an API episode renders no tokens")


class _Endpoint:
    """Hands out scripted chat completions and keeps the requests."""

    def __init__(self, replies, statuses=()):
        self.replies, self.statuses, self.requests = list(replies), list(statuses), []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return httpx.Response(status, text="busy", headers={"retry-after": "0"})
        message, finish = self.replies.pop(0)
        usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": finish}], "usage": usage, "model": "m"}
        )


def _client(endpoint, **kwargs):
    return ChatCompletionsClient(
        "http://api.test/v1", "m", "secret", backoff_s=0.0, transport=httpx.MockTransport(endpoint), **kwargs
    )


def _native_call(name="zoom_in", arguments='{"bbox_2d": [0, 0, 500, 500]}', call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _run(endpoint, environment=None, config=AgentLoopConfig(max_tool_calls=2), episode_options=None, **kwargs):
    environment = environment or _Environment()
    client = _client(endpoint, **kwargs)

    async def run():
        try:
            return await run_api_episode(
                environment,
                {"task": 1},
                client,
                env_seed=5,
                config=config,
                sampling=APISampling(seed=3),
                **(episode_options or {}),
            )
        finally:
            await client.aclose()

    return asyncio.run(run()), environment, client


def test_native_tool_calls_run_in_the_environment_and_the_answer_ends_the_episode():
    endpoint = _Endpoint(
        [
            ({"role": "assistant", "content": "Look closer.", "tool_calls": [_native_call()]}, "tool_calls"),
            ({"role": "assistant", "content": "It is 7.\n<answer>7</answer>"}, "stop"),
        ]
    )

    episode, environment, client = _run(endpoint)

    assert episode.status == AgentStatus.ANSWERED and episode.final_answer == "7"
    assert environment.calls == [("zoom_in", {"bbox_2d": [0, 0, 500, 500]})] and environment.closed
    assert episode.env_outcome.reward == 1.0 and episode.turns[0].state_key == "s1"
    first, second = endpoint.requests
    assert first["tools"] == [TOOL] and first["max_tokens"] == 10240 and first["temperature"] == 0.0
    assert first["messages"][0] == {"role": "system", "content": "Find the digit."}
    assert [part["type"] for part in first["messages"][1]["content"]] == ["image_url", "text"]
    assert first["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assistant, tool, images = second["messages"][2:]
    assert assistant["tool_calls"] == [_native_call()] and assistant["content"] == "Look closer."
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "Zoomed:"}
    assert images["role"] == "user" and [part["type"] for part in images["content"]] == ["image_url"]
    assert (client.requests, client.prompt_tokens, client.completion_tokens) == (2, 200, 20)


def test_tool_images_can_stay_in_the_tool_message_and_images_are_resized_as_local_runs_do():
    import base64
    import io

    endpoint = _Endpoint(
        [({"content": "", "tool_calls": [_native_call()]}, "tool_calls"), ({"content": "<answer>7</answer>"}, "stop")]
    )

    _run(endpoint, episode_options={"tool_images": "tool", "pixels": (None, 1000)})

    first, second = endpoint.requests
    tool = second["messages"][-1]
    assert tool["role"] == "tool" and [part["type"] for part in tool["content"]] == ["text", "image_url"]
    url = first["messages"][1]["content"][0]["image_url"]["url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert sent.width * sent.height <= 1000  # the 64x48 source image, within the pixel bound


def test_text_tool_calls_follow_the_local_rules_and_the_limit():
    text_call = '<tool_call>\n{"name": "zoom_in", "arguments": {"bbox_2d": [1, 2, 3, 4]}}\n</tool_call>'
    endpoint = _Endpoint([({"content": text_call}, "stop")] * 2)

    episode, environment, _ = _run(endpoint, config=AgentLoopConfig(max_tool_calls=1))

    # the second call comes in the answer-only turn after the one allowed call
    assert episode.status == AgentStatus.TOOL_CALL_LIMIT and len(environment.calls) == 1
    observation = endpoint.requests[1]["messages"][-1]
    assert observation["role"] == "user" and [part["type"] for part in observation["content"]] == ["text", "image_url"]
    assert episode.turns[1].is_finalization_turn and episode.metrics.tool_call_limit_reached


def test_malformed_and_parallel_native_calls_get_error_observations():
    endpoint = _Endpoint(
        [
            ({"content": "", "tool_calls": [_native_call(call_id="a"), _native_call(call_id="b")]}, "tool_calls"),
            ({"content": "", "tool_calls": [_native_call(arguments="{not json")]}, "tool_calls"),
            ({"content": "<answer>3</answer>"}, "stop"),
        ]
    )

    episode, environment, _ = _run(endpoint)

    assert environment.calls == [] and episode.metrics.tool_call_errors == 2
    replies = [message for message in endpoint.requests[1]["messages"] if message["role"] == "tool"]
    assert [reply["tool_call_id"] for reply in replies] == ["a", "b"]
    assert "exactly one tool call" in replies[0]["content"]
    assert episode.turns[1].tool_call_error.startswith("tool call is not valid JSON")
    assert episode.status == AgentStatus.ANSWERED and episode.env_outcome.reward == 0.0


def test_a_cut_off_reply_is_never_acted_on():
    endpoint = _Endpoint([({"content": "<answer>7</answer>"}, "length")])

    episode, _, _ = _run(endpoint, config=AgentLoopConfig(max_tool_calls=2, max_tokens_per_turn=64))

    assert episode.status == AgentStatus.LENGTH_LIMIT and episode.final_answer is None
    assert episode.metrics.turn_length_limited and endpoint.requests[0]["max_tokens"] == 64


def test_the_client_retries_busy_answers_and_fails_closed():
    endpoint = _Endpoint([({"content": "<answer>7</answer>"}, "stop")], statuses=[429, 503, 200])
    episode, _, client = _run(endpoint)
    assert episode.status == AgentStatus.ANSWERED and len(endpoint.requests) == client.requests == 3

    with pytest.raises(APIError, match="HTTP 400"):
        _run(_Endpoint([], statuses=[400]))
    with pytest.raises(APIBudgetExceeded):
        _run(
            _Endpoint([({"content": "x", "tool_calls": [_native_call()]}, "tool_calls")] * 2),
            max_total_tokens=50,
        )


def test_the_key_goes_in_the_header_only_and_local_endpoints_skip_proxies():
    seen = {}

    def endpoint(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content.decode()
        usage = {"prompt_tokens": 5, "completion_tokens": 3}
        return httpx.Response(200, json={"choices": [{"message": {"content": "<answer>7</answer>"}}], "usage": usage})

    _run(endpoint)
    assert seen["auth"] == "Bearer secret" and "secret" not in seen["body"]
    assert not uses_environment_proxies("http://127.0.0.1:8001/v1") and not uses_environment_proxies(
        "http://localhost:1"
    )
    assert uses_environment_proxies("https://api.example.com/v1")


# ----------------------------------------------------------------------------- eval --backend api
def _backend(tmp_path, endpoint, **kwargs):
    from easyr1_eval.api_agent import AgenticAPIBackend

    environment_file = tmp_path / "env.py"
    environment_file.write_text(
        f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_api_agent import _Environment as Environment\n"
    )
    return AgenticAPIBackend(
        "m",
        base_url="http://api.test/v1",
        api_key_env="TEST_API_KEY_UNSET",
        agent_config=AgentLoopConfig(max_tool_calls=2),
        agent_environment=f"{environment_file}:Environment",
        output_dir=tmp_path,
        transport=httpx.MockTransport(endpoint),
        **kwargs,
    )


def test_eval_backend_records_the_episode_and_reuses_finished_trajectories(tmp_path):
    from easyr1_eval.schemas import EvalSample, GenerationConfig

    endpoint = _Endpoint(
        [({"content": "", "tool_calls": [_native_call()]}, "tool_calls"), ({"content": "<answer>7</answer>"}, "stop")]
    )
    sample = EvalSample("env_tasks", "t1", "", None, extra_info={"env_config": {"task": "F1"}})
    broken = EvalSample("env_tasks", "t2", "", None, extra_info={})
    config = GenerationConfig(temperature=0.0, top_p=1.0, num_samples=1, max_new_tokens=20480, seed=1)

    (output,), (failed,) = _backend(tmp_path, endpoint).generate([sample, broken], config)
    again = _backend(tmp_path, _Endpoint([])).generate([sample], config)  # no request: read from the cache

    agent = output.diagnostics["agent"]
    assert output.text == "<answer>7</answer>" and agent["status"] == "answered"
    assert agent["env_outcome"] == {
        "reward": 1.0,
        "components": {},
        "metrics": {},
        "details": {"status": "answered"},
        "episode_id": "e1",
    }
    assert agent["env_steps"][0]["state_key"] == "s1" and agent["api_usage"]["requests"] == 2
    assert agent["turns"][0]["tool_call"] == {"name": "zoom_in", "arguments": {"bbox_2d": [0, 0, 500, 500]}}
    assert failed.diagnostics["agent"]["status"] == "sample_error"
    assert again[0][0].diagnostics == output.diagnostics
    assert "authorization" not in json.dumps(output.diagnostics).lower()


def test_runner_options_and_fingerprint_of_the_api_backend(monkeypatch):
    from easyr1_eval import runner

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-recorded")
    base = ["--model", "qwen-vl-max", "--backend", "api", "--api-base-url", "http://127.0.0.1:8001/v1"]
    with pytest.raises(SystemExit):
        runner.parse_args(base)  # needs agent environments
    args = runner.expand_eval_runs(
        runner.parse_args(
            [
                *base,
                "--interaction-mode",
                "agentic",
                "--agent-profile",
                "environment",
                "--agent-environment-kwargs",
                '{"url": "http://127.0.0.1:8000"}',
                "--api-extra-body",
                '{"enable_thinking": false}',
            ]
        )
    )[0]

    fields = runner._api_fingerprint_fields(args)
    assert fields["api"]["base_url"] == "http://127.0.0.1:8001/v1"
    assert fields["api"]["extra_body"] == {"enable_thinking": False}
    assert "sk-test-not-recorded" not in json.dumps(fields) and args.chat_template is None
    assert runner._api_fingerprint_fields(runner.parse_args(["--model", "m"])) == {}


# ----------------------------------------------------------------------------- API models as demonstration policies
def _policy(endpoint, **kwargs):
    from verl.workers.agent.api_loop import APIPolicy

    client = _client(endpoint)
    policy = APIPolicy(client, sampling=APISampling(temperature=1.0), max_tokens=256, **kwargs)
    reset = asyncio.run(_Environment().reset({}, 0))
    policy.start({}, 0, reset)
    return policy, client


def test_an_api_policy_turns_replies_into_turns_for_the_rollouts_loop():
    endpoint = _Endpoint(
        [
            ({"content": "Look closer.", "tool_calls": [_native_call()]}, "tool_calls"),
            ({"content": "It is 7.\n<answer>7</answer>", "tool_calls": [_native_call()]}, "stop"),
        ]
    )
    policy, client = _policy(endpoint, pixels=(None, 1000), observation_pixels=(None, 500))

    async def run():
        first = await policy.act(None)
        step = await _Environment().step(ToolCall("zoom_in", {}, ""))
        second = await policy.act(step)
        await client.aclose()
        return first, second

    first, second = asyncio.run(run())

    assert first == {
        "content": "Look closer.",
        "tool_call": {"name": "zoom_in", "arguments": {"bbox_2d": [0, 0, 500, 500]}},
    }
    assert second == {"text": "It is 7.\n<answer>7</answer>"}  # the answer takes precedence over the call
    assert endpoint.requests[0]["max_tokens"] == 256 and endpoint.requests[0]["tools"] == [TOOL]
    assistant, tool, images = endpoint.requests[1]["messages"][2:]
    assert assistant["tool_calls"] == [_native_call()] and tool["content"] == "Zoomed:"
    assert images["role"] == "user" and images["content"][0]["type"] == "image_url"


def test_an_api_policy_leaves_out_replies_the_rollout_could_not_take():
    from verl.workers.agent.replay import ReplayError

    for reply, match in (
        (({"content": "half a th"}, "length"), "cut off at 256 tokens"),
        (({"content": "", "tool_calls": [_native_call(), _native_call(call_id="c2")]}, "tool_calls"), "exactly one"),
        (({"content": "", "tool_calls": [_native_call(arguments="{")]}, "tool_calls"), "not valid JSON"),
    ):
        policy, client = _policy(_Endpoint([reply]))
        with pytest.raises(ReplayError, match=match):
            asyncio.run(policy.act(None))
        asyncio.run(client.aclose())


def test_export_options_for_api_models():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_sft", Path(__file__).resolve().parents[1] / "scripts/agent/export_sft.py"
    )
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    base = ["--model", "m", "--environment", "remote_http", "--out", "x"]

    args = export.parse_args(
        [
            *base,
            "--api-model",
            "glm",
            "--api-base-url",
            "http://h/v1",
            "--tasks",
            "t.jsonl",
            "--samples-per-task",
            "4",
            "--temperature",
            "1.0",
        ]
    )
    assert (args.api_model, args.samples_per_task, args.temperature) == ("glm", 4, 1.0)
    for extra in (
        ["--api-model", "glm", "--api-base-url", "http://h/v1"],
        ["--episodes", "e.jsonl", "--samples-per-task", "2"],
    ):
        with pytest.raises(SystemExit):
            export.parse_args([*base, *extra])


def test_a_used_up_budget_stops_requests_that_waited_for_a_slot():
    scripted = _Endpoint([({"content": "<answer>7</answer>"}, "stop")] * 5)

    async def endpoint(request):  # takes a while, so that the other requests wait for the slot
        await asyncio.sleep(0.01)
        return scripted(request)

    client = _client(endpoint, max_concurrency=1, max_total_tokens=10)  # each reply reports 110 tokens

    async def run():
        replies = await asyncio.gather(
            *(client.complete([], None, max_tokens=8, sampling=APISampling(), seed=None) for _ in range(5)),
            return_exceptions=True,
        )
        await client.aclose()
        return replies

    replies = asyncio.run(run())

    assert len(scripted.requests) == 1 and client.total_tokens == 110
    assert sum(isinstance(reply, APIBudgetExceeded) for reply in replies) == 4


def test_budgets_need_the_apis_token_usage():
    def endpoint(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "<answer>7</answer>"}}]})

    with pytest.raises(APIError, match="token usage"):
        _run(endpoint)  # the trajectory budget counts completion tokens
    client = _client(endpoint, max_total_tokens=1000)
    with pytest.raises(APIError, match="token usage"):
        asyncio.run(client.complete([], None, max_tokens=8, sampling=APISampling(), seed=None))
    asyncio.run(client.aclose())


def test_one_api_backend_and_one_budget_for_all_benchmarks_of_a_run(monkeypatch, tmp_path):
    from easyr1_eval import runner

    built, used = [], []
    monkeypatch.setattr(runner, "build_eval_backend", lambda args: built.append(object()) or built[-1])
    monkeypatch.setattr(
        runner,
        "infer_benchmark_shard",
        lambda spec, args, *, shard_index, num_shards, backend=None: used.append(backend),
    )
    monkeypatch.setattr(runner, "merge_prediction_shards", lambda *args: None)
    args = SimpleNamespace(backend="api", output_dir=tmp_path)
    specs = [SimpleNamespace(key="a"), SimpleNamespace(key="b")]

    runner.run_inference(specs, args, [])

    assert len(built) == 1 and used == [built[0], built[0]]
