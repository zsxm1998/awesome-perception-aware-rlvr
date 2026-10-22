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
"""Scripted demonstrations replayed through the rollout's loop (verl/workers/agent/replay.py) and exported for
ms-swift (verl/workers/agent/sft_export.py, scripts/agent/)."""

import asyncio
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image
from test_agent_environments import _TemplateProcessor

from verl.workers.agent.chat import EnvironmentChatAdapter
from verl.workers.agent.protocol import (
    AgentImageConfig,
    AgentLoopConfig,
    AgentStatus,
    EnvOutcome,
    EnvReset,
    EnvStep,
    ToolResult,
)
from verl.workers.agent.replay import RecordedPolicy, ReplayError, load_episodes, replay_episode


ROOT = Path(__file__).resolve().parents[1]
TOOL = {"type": "function", "function": {"name": "view"}}
CALL = '<tool_call>{"name": "view", "arguments": {"region": 2}}</tool_call>'


class _Environment:
    """Text-only: one tool whose observation names the region, and an outcome for the answer."""

    def __init__(self, processor):
        self.processor, self.calls = processor, []

    async def reset(self, config, seed):
        return EnvReset(
            messages=[{"role": "user", "content": f"Task {config['task']}"}], images=[], tool_schemas=[TOOL]
        )

    async def step(self, call):
        self.calls.append(dict(call.arguments))
        parts = [{"type": "text", "text": f"region {call.arguments['region']}: empty"}]
        return EnvStep(result=ToolResult("view", True, {}, parts=parts), state_key=f"r{call.arguments['region']}")

    async def finalize(self, final_answer, status):
        return EnvOutcome(reward=float(final_answer == "A"))

    async def close(self):
        pass

    def chat_adapter(self, processor, reset, *, max_tool_calls, image_config, image_preprocessor):
        return EnvironmentChatAdapter(processor, reset.tool_schemas, len(reset.images), max_tool_calls)


def _replay(policy, config=AgentLoopConfig(max_tool_calls=2), processor=None):
    processor = processor or _TemplateProcessor()
    environment = _Environment(processor)
    run = asyncio.run(
        replay_episode(
            environment,
            {"task": 1},
            policy,
            processor,
            env_seed=0,
            config=config,
            image_config=AgentImageConfig(),
            image_preprocessor=lambda image: image,
            max_model_len=100000,
        )
    )
    return run, environment


def test_a_replayed_episode_is_tokenized_as_the_rollout_tokenizes_the_same_turns():
    class Policy:
        def __init__(self):
            self.seen = []

        def act(self, step):
            self.seen.append(None if step is None else step.state_key)
            return CALL if step is None else {"content": "<answer>A</answer>"}

    policy = Policy()
    run, environment = _replay(policy)
    trajectory, tokenizer = run.trajectory, _TemplateProcessor().tokenizer

    assert trajectory.status == AgentStatus.ANSWERED and trajectory.env_outcome.reward == 1.0
    assert environment.calls == [{"region": 2}] and policy.seen == [None, "r2"]
    first, second = trajectory.steps
    assert first.model_token_ids == [*tokenizer.encode(CALL), 0]  # the text, then the end-of-turn token
    assert second.model_token_ids == [*tokenizer.encode("<answer>A</answer>"), 0]
    assert first.observation_token_ids == tokenizer.encode(
        "\n<|im_start|>user\n<tool_response>\nregion 2: empty\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    assert trajectory.response_ids == [*first.model_token_ids, *first.observation_token_ids, *second.model_token_ids]
    assert trajectory.response_mask == (
        [1] * len(first.model_token_ids) + [0] * len(first.observation_token_ids) + [1] * len(second.model_token_ids)
    )


def test_turns_the_rollout_could_not_produce_are_rejected():
    with pytest.raises(ReplayError, match="turn 1 is not a well-formed tool call"):
        _replay(RecordedPolicy(["<tool_call>{oops}</tool_call>", "<answer>A</answer>"]))
    with pytest.raises(ReplayError, match="more than the 8"):
        _replay(RecordedPolicy([CALL]), config=AgentLoopConfig(max_tool_calls=2, max_tokens_per_turn=8))
    with pytest.raises(ReplayError, match="only 1 are recorded"):
        _replay(RecordedPolicy([CALL]))
    with pytest.raises(ReplayError, match="special token"):
        _replay(RecordedPolicy(["x<|im_start|>system"]))
    with pytest.raises(ReplayError, match="unknown policy turn fields"):
        RecordedPolicy([{"answer": "A"}])


def test_episodes_files_need_a_configuration_and_turns(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"env_config": {"task": 1}, "turns": [CALL, "<answer>A</answer>"]}) + "\n\n")
    assert load_episodes(str(path))[0]["turns"][1] == "<answer>A</answer>"
    path.write_text(json.dumps({"env_config": {"task": 1}, "turns": []}) + "\n")
    with pytest.raises(ReplayError, match="non-empty turns"):
        load_episodes(str(path))


def _check_module():
    spec = importlib.util.spec_from_file_location("check_swift_export", ROOT / "scripts/agent/check_swift_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_check_expands_images_and_allows_only_the_newline_after_the_end_of_turn():
    check = _check_module()
    # prompt [5, IMG, 6], action [7, END], observation [NL, 8], action [9, END]
    rollout = {
        "input_ids": [5, 99, 6, 7, 0, 10, 8, 9, 0],
        "loss_mask": [0, 0, 0, 1, 1, 0, 0, 1, 1],
        "image_grid_thw": [[1, 2, 4]],
        "image_token_id": 99,
        "assistant_termination_token_ids": [0],
    }

    ids, mask, allowed = check.expected_tokens(rollout, merge_size=2, extra=[10])

    assert ids == [5, 99, 99, 6, 7, 0, 10, 8, 9, 0] and mask == [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    assert allowed == {6}  # the newline right after the first action's end-of-turn token


# ----------------------------------------------------------------------------- with a real Qwen3-VL processor
def _qwen3_vl_processor():
    from verl.utils.tokenizer import get_processor

    for repo in ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"):
        local = Path.home() / ".cache" / "modelscope" / "hub" / "models" / repo
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo, "preprocessor_config.json")
        path = str(Path(cached).parent) if isinstance(cached, str) else str(local)
        if (Path(path) / "preprocessor_config.json").exists():
            return get_processor(path, use_fast=True)
    pytest.skip("no Qwen3-VL processor in the local Hugging Face or ModelScope cache")


def test_structured_turns_render_as_the_chat_template_writes_tool_calls():
    from verl.workers.agent.replay import AssistantTurnRenderer, policy_turn

    renderer = AssistantTurnRenderer(_qwen3_vl_processor(), [TOOL])
    turn = {"content": "Look closer.", "tool_call": {"name": "zoom_in", "arguments": {"bbox_2d": [0, 0, 500, 500]}}}

    assert renderer.render(policy_turn(turn)) == (
        'Look closer.\n<tool_call>\n{"name": "zoom_in", "arguments": {"bbox_2d": [0, 0, 500, 500]}}\n</tool_call>'
    )
    assert renderer.render(policy_turn({"content": "<answer>A</answer>"})) == "<answer>A</answer>"


def test_without_a_system_message_the_tool_descriptions_stay_in_the_rendered_system_block():
    from verl.workers.agent.sft_export import swift_sample

    processor = _qwen3_vl_processor()
    run, _ = _replay(RecordedPolicy([{"content": "<answer>A</answer>"}]), processor=processor)

    sample = swift_sample(run, processor, lambda image: image)

    assert "tools" not in sample.record
    system, user = sample.record["messages"][:2]
    assert (
        system["role"] == "system"
        and system["content"].startswith("# Tools")
        and '"name": "view"' in system["content"]
    )
    assert user == {"role": "user", "content": "Task 1"}


@pytest.mark.parametrize("size", [(500, 333), (1001, 1001), (40, 40), (1024, 512)])
def test_saved_images_give_the_processor_the_rollouts_input(size):
    from verl.workers.agent.sft_export import check_model_input_image, model_input_image

    processor = _qwen3_vl_processor()
    prepared = Image.effect_mandelbrot(size, (-2.0, -1.5, 1.0, 1.5), 60).convert("RGB")

    final = model_input_image(processor, prepared)

    assert final.width % 32 == 0 and final.height % 32 == 0
    check_model_input_image(processor, prepared, final)  # the same patches as the rollout's image


def test_a_replayed_toy_episode_becomes_a_swift_sample_with_the_rollout_tokens():
    sys.path.insert(0, str(ROOT / "examples" / "agent_env"))
    from server import EpisodeServer
    from toy_search import ToySearchEpisode
    from toy_search_policy import ToySearchPolicy

    from verl.utils.dataset import process_image
    from verl.workers.agent.environments import EnvironmentContext, RemoteHTTPEnvironment
    from verl.workers.agent.sft_export import swift_sample

    processor = _qwen3_vl_processor()
    http = EpisodeServer(ToySearchEpisode).http_server("127.0.0.1", 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()

    def prepare(image):
        return process_image(image, 200704, 1048576)

    context = EnvironmentContext(processor=processor, image_preprocessor=prepare, max_tool_calls=4)
    environment = RemoteHTTPEnvironment(context, url=f"http://127.0.0.1:{http.server_address[1]}")

    async def run():
        try:
            return await replay_episode(
                environment,
                {"image_seed": 3},
                ToySearchPolicy(),
                processor,
                env_seed=0,
                config=AgentLoopConfig(max_tool_calls=4, max_response_tokens=8192, max_tokens_per_turn=2048),
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
    sample = swift_sample(episode, processor, prepare)

    messages = sample.record["messages"]
    assert episode.trajectory.status == AgentStatus.ANSWERED and episode.trajectory.env_outcome.reward == 1.0
    assert [message["role"] for message in messages][:3] == ["system", "user", "assistant"]
    assert messages[0]["content"].startswith("Among the small letters") and "# Tools" not in messages[0]["content"]
    assert [tool["function"]["name"] for tool in json.loads(sample.record["tools"])] == ["zoom_in"]
    assert messages[1]["content"].startswith("<image>")
    assert messages[-1]["role"] == "assistant" and messages[-1]["content"].endswith("</answer>")
    assert sum(message["content"].count("<image>") for message in messages) == len(sample.images)
    assert sample.rollout["input_ids"] == episode.trajectory.input_ids
    assert len(sample.rollout["image_grid_thw"]) == len(sample.images) == len(episode.trajectory.all_images)


_FAILING_ENVIRONMENT = """
import asyncio

from verl.workers.agent.chat import EnvironmentChatAdapter
from verl.workers.agent.protocol import EnvOutcome, EnvReset

EVENTS = []


class Environment:
    def __init__(self, context):
        self.context = context

    async def reset(self, config, seed):
        self.name = config["name"]
        EVENTS.append(("reset", self.name))
        return EnvReset(messages=[{"role": "user", "content": "q"}], images=[], tool_schemas=[{"type": "function", "function": {"name": "view"}}])

    async def step(self, call):
        if self.name == "A":
            await asyncio.sleep(0.05)
            raise ValueError("A broke")
        EVENTS.append(("step running", self.name))
        await asyncio.sleep(30)

    async def finalize(self, final_answer, status):
        return EnvOutcome(reward=1.0)

    async def close(self):
        await self.context.run_cleanup(self._close())

    async def _close(self):
        await asyncio.sleep(0.01)  # a request to the server
        EVENTS.append(("close", self.name, "after the context closed" if self.context._closed else "in time"))

    def chat_adapter(self, processor, reset, *, max_tool_calls, image_config, image_preprocessor):
        return EnvironmentChatAdapter(processor, reset.tool_schemas, 0, max_tool_calls)
"""


def test_an_export_that_fails_closes_every_episode_before_the_shared_resources(tmp_path, monkeypatch):
    """One episode fails with an unexpected error while another waits on the environment: the other is cancelled and
    closes its episode before the batch's shared resources (the environment context) are closed."""
    spec = importlib.util.spec_from_file_location("export_sft", ROOT / "scripts/agent/export_sft.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    (tmp_path / "failing_env.py").write_text(_FAILING_ENVIRONMENT)
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "".join(
            json.dumps({"env_config": {"name": name}, "turns": [CALL, "<answer>A</answer>"]}) + "\n" for name in "AB"
        )
    )
    monkeypatch.setattr(export, "get_processor", lambda *args, **kwargs: _TemplateProcessor())
    args = export.parse_args(
        [
            "--model",
            "m",
            "--environment",
            f"{tmp_path / 'failing_env.py'}:Environment",
            "--episodes",
            str(episodes),
            "--min-pixels",
            "1",
            "--max-pixels",
            "100",
            "--out",
            str(tmp_path / "out"),
            "--no-check",
        ]
    )

    with pytest.raises(ValueError, match="A broke"):
        asyncio.run(export.export(args))

    events = sys.modules[
        next(
            name
            for name in sys.modules
            if name.startswith("agent_environment_") and hasattr(sys.modules[name], "EVENTS")
        )
    ].EVENTS
    assert ("close", "A", "in time") in events and ("close", "B", "in time") in events


def test_an_environment_error_of_one_episode_is_counted_and_the_export_goes_on(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("export_sft_2", ROOT / "scripts/agent/export_sft.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    from verl.workers.agent.environments import RemoteEnvironmentError

    async def failing_replay(environment, env_config, *args, **kwargs):
        raise RemoteEnvironmentError("episode e1, step 0: POST /step returned HTTP 400")

    monkeypatch.setattr(export, "replay_episode", failing_replay)
    monkeypatch.setattr(export, "get_processor", lambda *args, **kwargs: _TemplateProcessor())
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"env_config": {}, "turns": [CALL]}) + "\n")
    args = export.parse_args(
        [
            "--model",
            "m",
            "--environment",
            "remote_http",
            "--environment-kwargs",
            '{"url": "http://env.test"}',
            "--episodes",
            str(episodes),
            "--out",
            str(tmp_path / "out"),
            "--no-check",
        ]
    )

    summary = asyncio.run(export.export(args))

    assert summary["failed"] == 1 and "HTTP 400" in summary["failures"][0]["error"]
