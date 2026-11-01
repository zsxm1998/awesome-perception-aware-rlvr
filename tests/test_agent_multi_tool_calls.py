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
"""Several tool calls per model turn (worker.rollout.agent_max_tool_calls_per_turn): the first ones of a turn run in
the order written, the others get an error result; the observation holds every result."""

import asyncio
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest
from PIL import Image

from verl.workers.agent.environments import EnvironmentContext, LocalToolEnvironment
from verl.workers.agent.loop import AgentLoop
from verl.workers.agent.protocol import (
    TOO_MANY_TOOL_CALLS_IN_TURN,
    TOOL_CALL_LIMIT_REACHED,
    AgentImageConfig,
    AgentLoopConfig,
    AgentStatus,
    EncodedObservation,
    EnvReset,
    EnvStep,
    GenerationOutput,
    ObservationEncodingRequest,
    ToolResult,
    turn_env_reward,
)
from verl.workers.agent.step_records import step_records, turn_is_valid
from verl.workers.agent.trajectory import turn_action_spans


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))


def _call(name="view", **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments or {"region": 1}}) + "</tool_call>"


class _Backend:
    def __init__(self, texts):
        self.texts = list(texts)

    async def generate(self, request):
        text = self.texts.pop(0)
        return GenerationOutput(token_ids=[ord(char) for char in text][:6] or [1], text=text, finish_reason="stop")


class _Encoder:
    """Records the results of each observation; one token per result."""

    max_tool_calls = None

    def __init__(self):
        self.requests = []

    async def encode(self, request):
        self.requests.append(request.results)
        return EncodedObservation(
            token_ids=[7] * len(request.results), effective_token_count=len(request.results), visual_token_count=0
        )


class _Environment(LocalToolEnvironment):
    """Answers each call with its region; ``done_on`` ends the episode at that call."""

    def __init__(self, done_on=None):
        super().__init__()
        self.calls, self.done_on = [], done_on

    async def step(self, call):
        self.calls.append(dict(call.arguments))
        region = call.arguments.get("region")
        result = ToolResult(call.name, True, {}, observation_text=f"region {region}")
        return EnvStep(result=result, reward=0.25, state_key=f"r{region}", done=region == self.done_on)


def _run(texts, *, per_turn, max_tool_calls=6, environment=None):
    environment = environment or _Environment()
    encoder = _Encoder()
    config = AgentLoopConfig(max_tool_calls=max_tool_calls, max_tool_calls_per_turn=per_turn)
    loop = AgentLoop(_Backend(texts), encoder, environment, config)
    return asyncio.run(loop.run(prompt_ids=[1], source_images=[])), environment, encoder


def test_the_first_calls_of_a_turn_run_in_order_and_the_others_get_an_error():
    trajectory, environment, encoder = _run(
        [_call(region=1) + _call(region=2) + _call(region=3), "<answer>A</answer>"], per_turn=2
    )

    assert environment.calls == [{"region": 1}, {"region": 2}]
    first = trajectory.steps[0]
    assert [call.executed for call in first.calls] == [True, True, False]
    assert first.calls[2].tool_result.error_code == TOO_MANY_TOOL_CALLS_IN_TURN
    assert first.calls[2].tool_result.content["error"]["message"] == "not run: at most 2 tool calls run per turn"
    # one observation with every result, in call order
    assert [result.observation_text for result in encoder.requests[0][:2]] == ["region 1", "region 2"]
    assert len(encoder.requests[0]) == 3 and first.observation_committed
    assert trajectory.metrics.tool_call_attempts == 2 and trajectory.metrics.tool_call_errors == 1
    assert (trajectory.metrics.malformed_tool_calls, trajectory.metrics.unrun_tool_calls) == (0, 1)
    assert trajectory.metrics.tool_call_successes == 2
    assert turn_env_reward(first.calls) == 0.5
    assert trajectory.status == AgentStatus.ANSWERED
    # more calls than a turn runs make the turn invalid (GiGPO), as several search tags do in verl-agent
    assert not turn_is_valid(trajectory, 0) and turn_is_valid(trajectory, 1)
    with pytest.raises(ValueError, match="several tool calls"):
        first.tool_call


def test_with_one_call_per_turn_the_first_call_runs():
    trajectory, environment, _ = _run([_call(region=1) + _call(region=2), "<answer>A</answer>"], per_turn=1)

    assert environment.calls == [{"region": 1}]
    message = trajectory.steps[0].calls[1].tool_result.content["error"]["message"]
    assert message == "not run: at most 1 tool call run per turn"
    assert trajectory.metrics.tool_call_attempts == 1


def test_calls_beyond_the_remaining_calls_get_a_limit_error_and_the_turn_stays_valid():
    trajectory, environment, _ = _run(
        [_call(region=1) + _call(region=2) + _call(region=3), "<answer>A</answer>"], per_turn=3, max_tool_calls=2
    )

    assert environment.calls == [{"region": 1}, {"region": 2}]
    first, second = trajectory.steps
    assert first.calls[2].tool_result.error_code == TOOL_CALL_LIMIT_REACHED
    assert trajectory.metrics.tool_call_attempts == 2 and trajectory.metrics.tool_call_limit_reached
    assert (trajectory.metrics.malformed_tool_calls, trajectory.metrics.unrun_tool_calls) == (0, 1)
    assert trajectory.metrics.tool_call_errors == 1
    assert second.is_finalization_turn and trajectory.status == AgentStatus.ANSWERED
    assert turn_is_valid(trajectory, 0)


def test_a_malformed_call_counts_gets_its_error_and_the_other_calls_run():
    trajectory, environment, _ = _run(
        ["<tool_call>{oops}</tool_call>" + _call(region=2), "<answer>A</answer>"], per_turn=2
    )

    assert environment.calls == [{"region": 2}]
    malformed, ran = trajectory.steps[0].calls
    assert malformed.tool_result.error_code == "malformed_tool_call" and ran.executed
    assert trajectory.metrics.tool_call_attempts == 2
    assert (trajectory.metrics.malformed_tool_calls, trajectory.metrics.unrun_tool_calls) == (1, 0)
    assert not turn_is_valid(trajectory, 0)


def test_a_call_that_ends_the_episode_stops_the_turn():
    trajectory, environment, encoder = _run(
        [_call(region=1) + _call(region=2), "<answer>never</answer>"], per_turn=2, environment=_Environment(done_on=1)
    )

    assert environment.calls == [{"region": 1}] and encoder.requests == []
    done, later = trajectory.steps[0].calls
    assert done.env_done and not later.executed and later.tool_result is None
    # the call after the one that ended the episode did not run, and gets no result: no error
    assert (trajectory.metrics.unrun_tool_calls, trajectory.metrics.tool_call_errors) == (1, 0)
    assert trajectory.status == AgentStatus.ENVIRONMENT_DONE and not trajectory.steps[0].observation_committed


def test_calls_beyond_the_limit_count_as_errors_also_when_an_earlier_call_ends_the_episode():
    trajectory, environment, _ = _run(
        [_call(region=1) + _call(region=2), "<answer>never</answer>"], per_turn=1, environment=_Environment(done_on=1)
    )

    assert environment.calls == [{"region": 1}]
    assert trajectory.steps[0].calls[1].tool_result.error_code == TOO_MANY_TOOL_CALLS_IN_TURN
    assert trajectory.metrics.tool_call_attempts == 1 and trajectory.metrics.tool_call_errors == 1
    assert trajectory.metrics.unrun_tool_calls == 1
    assert not turn_is_valid(trajectory, 0)


def test_a_malformed_call_beyond_the_limit_is_unrun_not_malformed():
    trajectory, environment, _ = _run(
        [_call(region=1) + "<tool_call>{oops}</tool_call>", "<answer>A</answer>"], per_turn=1
    )

    assert environment.calls == [{"region": 1}]
    metrics = trajectory.metrics
    assert (metrics.malformed_tool_calls, metrics.unrun_tool_calls, metrics.tool_call_errors) == (0, 1, 1)
    assert metrics.tool_call_attempts == 1


def test_calls_in_the_answer_only_turn_are_neither_unrun_nor_errors():
    trajectory, environment, _ = _run([_call(region=1), _call(region=2)], per_turn=1, max_tool_calls=1)

    assert environment.calls == [{"region": 1}] and trajectory.status == AgentStatus.TOOL_CALL_LIMIT
    metrics = trajectory.metrics
    assert (metrics.unrun_tool_calls, metrics.tool_call_errors, metrics.tool_call_attempts) == (0, 0, 1)


@pytest.mark.parametrize("per_turn, unrun, errors", [(1, 2, 2), (2, 2, 1), (3, 2, 0)])
def test_after_a_call_ends_the_episode_only_calls_beyond_the_limit_get_an_error(per_turn, unrun, errors):
    trajectory, environment, _ = _run(
        [_call(region=1) + _call(region=2) + _call(region=3)], per_turn=per_turn, environment=_Environment(done_on=1)
    )

    assert environment.calls == [{"region": 1}] and trajectory.status == AgentStatus.ENVIRONMENT_DONE
    metrics = trajectory.metrics
    assert (metrics.unrun_tool_calls, metrics.tool_call_errors, metrics.malformed_tool_calls) == (unrun, errors, 0)
    # tool_call_errors = malformed + calls not run for the per-turn limit or the remaining calls + environment refusals
    with_result = [call for call in trajectory.steps[0].calls if not call.executed and call.tool_result is not None]
    assert errors == metrics.malformed_tool_calls + len(with_result)


def test_step_records_hold_every_call_the_environment_ran():
    trajectory, _, _ = _run([_call(region=1) + _call(region=2), _call(region=3), "<answer>A</answer>"], per_turn=2)
    spans = turn_action_spans(trajectory.steps, trajectory.response_mask)

    record = step_records(trajectory, spans, [], rollout_index=0)

    first, second, third = record["turns"]
    assert first["env_calls"] == [{"state_key": "r1", "reward": 0.25}, {"state_key": "r2", "reward": 0.25}]
    assert first["observation_text"] == "region 1\nregion 2" and first["valid"]
    assert second["env_calls"] == [{"state_key": "r3", "reward": 0.25}] and third["valid"]


def test_an_observation_request_gives_one_or_several_results():
    one = ToolResult("view", True, {})
    request = ObservationEncodingRequest(result=one)
    assert request.results == (one,) and request.result is one
    several = ObservationEncodingRequest(results=(one, one))
    assert several.result is None and len(several.results) == 2
    with pytest.raises(ValueError, match="at least one tool result"):
        ObservationEncodingRequest()


# ----------------------------------------------------------------------------- rendering with Qwen3-VL's template
def _qwen3_vl_processor():
    from verl.utils.tokenizer import get_processor

    for repo in ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"):
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo, "preprocessor_config.json")
        local = Path.home() / ".cache" / "modelscope" / "hub" / "models" / repo
        path = str(Path(cached).parent) if isinstance(cached, str) else str(local)
        if (Path(path) / "preprocessor_config.json").exists():
            return get_processor(path, use_fast=True)
    pytest.skip("no Qwen3-VL processor in the local Hugging Face or ModelScope cache")


def test_the_results_of_a_turn_render_as_the_chat_template_joins_tool_messages():
    from verl.workers.agent.chat import EnvironmentChatAdapter

    processor = _qwen3_vl_processor()
    tools = [{"type": "function", "function": {"name": "view", "parameters": {"type": "object", "properties": {}}}}]
    adapter = EnvironmentChatAdapter(processor, tools, 0, max_tool_calls=4)
    image = Image.new("RGB", (64, 64), (10, 200, 30))
    first = ToolResult("view", True, {}, observation_text="region 1")
    second = ToolResult("view", True, {}, images=[image], parts=[{"type": "text", "text": "crop:"}, {"type": "image"}])

    encoded = asyncio.run(adapter.encode(ObservationEncodingRequest(results=(first, second))))

    text = processor.tokenizer.decode(encoded.token_ids, skip_special_tokens=False)
    assert text == (
        "\n<|im_start|>user\n<tool_response>\nregion 1\n</tool_response>\n<tool_response>\ncrop:"
        "<|vision_start|><|image_pad|><|vision_end|>\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    assert list(encoded.images) == [image]
    rendered = processor.apply_chat_template(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "region 1"},
            {"role": "tool", "content": [{"type": "text", "text": "crop:"}, {"type": "image"}]},
        ],
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert rendered.endswith(text)


def test_the_official_deepeyes_style_puts_the_results_of_a_turn_in_one_user_turn():
    from verl.workers.agent.chat import OFFICIAL_DEEPEYES_FORMAT_INSTRUCTION, OfficialDeepEyesChatAdapter
    from verl.workers.agent.tools import ImageZoomInTool, ToolRegistry

    processor = _qwen3_vl_processor()
    adapter = OfficialDeepEyesChatAdapter(processor, ToolRegistry([ImageZoomInTool()]), 1, max_tool_calls=6)
    crop = ToolResult("image_zoom_in_tool", True, {}, images=[Image.new("RGB", (64, 64))])
    not_run = ToolResult(
        "image_zoom_in_tool",
        False,
        {"error": {"code": TOO_MANY_TOOL_CALLS_IN_TURN, "message": "not run: at most 1 tool call run per turn"}},
    )

    encoded = asyncio.run(adapter.encode(ObservationEncodingRequest(results=(crop, not_run))))

    assert processor.tokenizer.decode(encoded.token_ids, skip_special_tokens=False) == (
        "\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|vision_end|>"
        + OFFICIAL_DEEPEYES_FORMAT_INSTRUCTION
        + "</tool_response>\nError: not run: at most 1 tool call run per turn<|im_end|>\n<|im_start|>assistant\n"
    )
    assert list(encoded.images) == crop.images and encoded.visual_token_count > 0


# ----------------------------------------------------------------------------- API models
class _APIEnvironment:
    def __init__(self):
        self.calls = []

    async def reset(self, config, seed):
        tool = {"type": "function", "function": {"name": "view", "parameters": {"type": "object", "properties": {}}}}
        return EnvReset(messages=[{"role": "user", "content": "Which region?"}], images=[], tool_schemas=[tool])

    async def step(self, call):
        self.calls.append(dict(call.arguments))
        image = Image.new("RGB", (8, 8))
        return EnvStep(result=ToolResult(call.name, True, {}, images=[image], parts=[{"type": "image"}]))

    async def finalize(self, final_answer, status):
        return None

    async def close(self):
        pass


def _native(call_id, region):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "view", "arguments": json.dumps({"region": region})},
    }


def _api_episode(replies, *, per_turn):
    from verl.workers.agent.api_loop import APISampling, ChatCompletionsClient, run_api_episode

    replies, requests = iter(replies), []

    def endpoint(request):
        requests.append(json.loads(request.content))
        message = next(replies)
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage})

    client = ChatCompletionsClient("http://api.test/v1", "m", None, transport=httpx.MockTransport(endpoint))
    environment = _APIEnvironment()

    async def run():
        try:
            config = AgentLoopConfig(max_tool_calls=4, max_tool_calls_per_turn=per_turn)
            return await run_api_episode(
                environment, {}, client, env_seed=0, config=config, sampling=APISampling(seed=1)
            )
        finally:
            await client.aclose()

    return asyncio.run(run()), environment, requests


def test_an_api_turn_with_native_calls_gets_a_tool_message_per_call_then_the_images():
    episode, environment, requests = _api_episode(
        [{"content": "", "tool_calls": [_native("a", 1), _native("b", 2)]}, {"content": "<answer>2</answer>"}],
        per_turn=2,
    )

    assert environment.calls == [{"region": 1}, {"region": 2}] and episode.status == AgentStatus.ANSWERED
    messages = requests[1]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool", "user"]
    assert [message["tool_call_id"] for message in messages[2:4]] == ["a", "b"]
    assert [part["type"] for part in messages[4]["content"]] == ["image_url", "image_url"]
    assert len(episode.turns[0].calls) == 2


def test_api_records_list_the_calls_of_a_turn_with_several():
    from easyr1_eval.api_agent import api_episode_to_generation_output

    episode, _, _ = _api_episode(
        [{"content": "", "tool_calls": [_native("a", 1), _native("b", 2)]}, {"content": "<answer>2</answer>"}],
        per_turn=1,
    )

    turns = api_episode_to_generation_output(episode).diagnostics["agent"]["turns"]
    assert turns[0]["tool_call"] is None and [call["tool_success"] for call in turns[0]["calls"]] == [True, False]
    assert turns[0]["calls"][1]["tool_error_code"] == TOO_MANY_TOOL_CALLS_IN_TURN
    assert "calls" not in turns[1]


# ----------------------------------------------------------------------------- replay and SFT export
def test_a_replayed_turn_with_two_calls_exports_two_tool_messages_with_the_rollout_tokens():
    sys.path.insert(0, str(ROOT / "examples" / "agent_env"))
    from server import EpisodeServer
    from toy_search import ToySearchEpisode

    from verl.utils.dataset import process_image
    from verl.workers.agent.environments import RemoteHTTPEnvironment
    from verl.workers.agent.replay import RecordedPolicy, replay_episode
    from verl.workers.agent.sft_export import swift_sample

    processor = _qwen3_vl_processor()
    http = EpisodeServer(ToySearchEpisode).http_server("127.0.0.1", 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()

    def prepare(image):
        return process_image(image, 200704, 1048576)

    context = EnvironmentContext(
        processor=processor, image_preprocessor=prepare, max_tool_calls=4, max_tool_calls_per_turn=2
    )
    environment = RemoteHTTPEnvironment(context, url=f"http://127.0.0.1:{http.server_address[1]}")
    seen = []

    class Policy(RecordedPolicy):
        def act(self, step):
            seen.append(step)
            return super().act(step)

    calls = [
        {"name": "zoom_in", "arguments": {"bbox_2d": [0, 0, 500, 500]}},
        {"name": "zoom_in", "arguments": {"bbox_2d": [500, 500, 1000, 1000]}},
    ]
    policy = Policy([{"content": "Two regions.", "tool_calls": calls}, {"content": "<answer>3</answer>"}])

    async def run():
        try:
            return await replay_episode(
                environment,
                {"image_seed": 3},
                policy,
                processor,
                env_seed=0,
                config=AgentLoopConfig(
                    max_tool_calls=4, max_response_tokens=8192, max_tokens_per_turn=2048, max_tool_calls_per_turn=2
                ),
                image_config=AgentImageConfig(min_pixels=200704, max_pixels=1048576),
                image_preprocessor=prepare,
                max_model_len=100000,
            )
        finally:
            await context.aclose()

    try:
        episode = asyncio.run(run())
    finally:
        http.shutdown()

    assert seen[0] is None and isinstance(seen[1], tuple) and len(seen[1]) == 2
    assert [call.executed for call in episode.trajectory.steps[0].calls] == [True, True]
    sample = swift_sample(episode, processor, prepare)
    messages = sample.record["messages"]
    assert "up to 2 zoom_in calls in one turn" in messages[0]["content"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert sample.rollout["input_ids"] == episode.trajectory.input_ids
    assert len(sample.images) == 3


def test_the_example_server_refuses_a_per_turn_limit_its_episode_does_not_take():
    sys.path.insert(0, str(ROOT / "examples" / "agent_env"))
    from server import EpisodeServer, ProtocolError, text_part
    from toy_search import ToySearchEpisode

    class OneCallEpisode:
        def reset(self, config, seed, max_tool_calls):
            return {"user_content": [text_part("q")], "tool_schemas": [], "system_prompt": None}

    body = {"config": {}, "seed": 0, "max_tool_calls": 4}
    assert "episode_id" in EpisodeServer(OneCallEpisode).handle("/reset", {**body, "max_tool_calls_per_turn": 1})
    with pytest.raises(ProtocolError, match="one tool call per turn"):
        EpisodeServer(OneCallEpisode).handle("/reset", {**body, "max_tool_calls_per_turn": 2})
    one = EpisodeServer(ToySearchEpisode).handle("/reset", {**body, "config": {"image_seed": 1}})
    two = EpisodeServer(ToySearchEpisode).handle(
        "/reset", {**body, "config": {"image_seed": 1}, "max_tool_calls_per_turn": 2}
    )
    assert "per turn" not in one["system_prompt"] and "in one turn" not in one["system_prompt"]
    assert "up to 2 zoom_in calls in one turn" in two["system_prompt"]


# ----------------------------------------------------------------------------- configuration
def test_the_per_turn_limit_is_checked_where_it_is_configured():
    from easyr1_eval import runner

    from verl.trainer.config import PPOConfig
    from verl.workers.agent.environments import DeepEyesEnvironment
    from verl.workers.rollout.config import RolloutConfig

    with pytest.raises(ValueError, match="at least 1"):
        RolloutConfig(agent_max_tool_calls_per_turn=0).post_init()
    with pytest.raises(ValueError, match="positive integer"):
        AgentLoopConfig(max_tool_calls_per_turn=0)

    one_shot = PPOConfig()
    one_shot.worker.rollout.agent_max_tool_calls_per_turn = 2
    with pytest.raises(ValueError, match="needs worker.rollout.interaction_mode=agentic"):
        one_shot.post_init()
    deepeyes = PPOConfig()
    deepeyes.worker.rollout.interaction_mode = "agentic"
    deepeyes.worker.rollout.agent_max_tool_calls_per_turn = 2
    deepeyes.data.format_prompt = None
    with pytest.raises(ValueError, match="DeepEyes runs one tool call per turn"):
        deepeyes.post_init()
    context = EnvironmentContext(processor=object(), image_preprocessor=lambda image: image, max_tool_calls_per_turn=2)
    with pytest.raises(ValueError, match="DeepEyes runs one tool call per turn"):
        DeepEyesEnvironment(context)

    def args(*extra):
        parsed = runner.parse_args(["--model", "m", "--interaction-mode", "agentic", *extra])
        return runner.expand_eval_runs(parsed)[0]

    environment = ("--agent-profile", "environment", "--agent-environment-kwargs", '{"url": "http://127.0.0.1:1"}')
    default = runner._agent_fingerprint_fields(args(*environment))
    two = runner._agent_fingerprint_fields(args(*environment, "--agent-max-tool-calls-per-turn", "2"))
    # the option enters the fingerprint only when it is not 1; agentic results from before several calls per turn
    # (runner version 6, where such a turn ran none of its calls) are not reused
    assert "agent_max_tool_calls_per_turn" not in default and "max_tool_calls_per_turn" not in default["agent_config"]
    assert default["agent_runner_version"] == 7
    assert two["agent_max_tool_calls_per_turn"] == 2 and two["agent_config"]["max_tool_calls_per_turn"] == 2
    with pytest.raises(SystemExit):
        args(
            "--system-prompt",
            str(ROOT / "examples/system_prompt/deepeyes.txt"),
            "--agent-max-tool-calls-per-turn",
            "2",
        )
