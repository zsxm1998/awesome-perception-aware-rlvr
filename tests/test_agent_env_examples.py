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
"""The agent environment examples: the HTTP server template, the toy search environment and DeepEyes over HTTP."""

import asyncio
import io
import json
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "agent_env"))

from server import EpisodeServer, ProtocolError  # noqa: E402
from toy_search import ToySearchEpisode, render_task  # noqa: E402

from verl.workers.agent.environments import EnvironmentContext, RemoteHTTPEnvironment  # noqa: E402
from verl.workers.agent.protocol import AgentStatus, ToolCall  # noqa: E402


class _CountingEpisode:
    steps = 0

    def reset(self, config, seed, max_tool_calls):
        return {"user_content": [], "tool_schemas": [], "system_prompt": None}

    def step(self, name, arguments):
        type(self).steps += 1
        return {"content": [{"type": "text", "text": str(type(self).steps)}], "success": True}

    def finalize(self, final_answer, status):
        return {"reward": 1.0}


def test_episode_server_deduplicates_steps_and_drops_idle_episodes():
    now = [0.0]
    server = EpisodeServer(_CountingEpisode, idle_timeout_s=10.0, clock=lambda: now[0])
    episode_id = server.handle("/reset", {"config": {}, "seed": 0})["episode_id"]
    call = {"name": "t", "arguments": {}}

    first = server.handle("/step", {"episode_id": episode_id, "step_index": 0, "tool_call": call})
    repeated = server.handle("/step", {"episode_id": episode_id, "step_index": 0, "tool_call": call})

    assert first == repeated and _CountingEpisode.steps == 1  # executed once
    assert first["done"] is False and first["cost"] == 0.0
    with pytest.raises(ProtocolError, match="out of order"):
        server.handle("/step", {"episode_id": episode_id, "step_index": 2, "tool_call": call})
    outcome = server.handle("/finalize", {"episode_id": episode_id, "final_answer": None, "status": "no_action"})
    assert outcome == {"reward": 1.0, "components": {}, "metrics": {}}
    assert server.handle("/finalize", {"episode_id": episode_id, "final_answer": "x", "status": "answered"}) == outcome
    now[0] = 11.0  # idle longer than the timeout
    with pytest.raises(ProtocolError, match="unknown or expired"):
        server.handle("/step", {"episode_id": episode_id, "step_index": 1, "tool_call": call})
    assert server.handle("/close", {"episode_id": episode_id}) == {}


def test_toy_search_runs_over_http_with_the_remote_environment():
    http = EpisodeServer(ToySearchEpisode).http_server("127.0.0.1", 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    context = EnvironmentContext(processor=object(), image_preprocessor=lambda image: image, max_tool_calls=4)
    environment = RemoteHTTPEnvironment(context, url=f"http://127.0.0.1:{http.server_address[1]}")
    _, digit = render_task(5)

    async def episode():
        reset = await environment.reset({"image_seed": 5}, seed=0)
        zoom = await environment.step(ToolCall("zoom_in", {"bbox_2d": [0, 0, 400, 300]}, ""))
        bad = await environment.step(ToolCall("zoom_in", {"bbox_2d": [5, 5, 6, 6]}, ""))
        outcome = await environment.finalize(digit, AgentStatus.ANSWERED)
        await environment.close()
        await context.aclose()
        return reset, zoom, bad, outcome

    try:
        reset, zoom, bad, outcome = asyncio.run(episode())
    finally:
        http.shutdown()

    assert "at most 4 times" in reset.messages[0]["content"] and reset.images[0].size == (1024, 1024)
    assert zoom.result.success and zoom.state_key == "level2:cell00" and max(zoom.result.images[0].size) == 512
    assert not bad.result.success and bad.result.error_code == "invalid_bbox"
    assert outcome.reward == 1.0 and outcome.metrics == {"zoom_calls": 1.0}


def test_deepeyes_http_episode_writes_the_native_prompt_and_observation_layout(tmp_path):
    from deepeyes_http import DeepEyesEpisode, DeepEyesRows, write_tasks

    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), color=(9, 9, 9)).save(buffer, format="PNG")
    row = {
        "images": [{"bytes": buffer.getvalue(), "path": None}],
        "env_name": "visual_toolbox_v2",
        "row_system_prompt": "",
    }
    pd.DataFrame([{**row, "problem": "<image>\nWhat is it?", "answer": "x"}]).to_parquet(tmp_path / "train.parquet")
    pd.DataFrame([{**row, "problem": "<image>\nQ?", "answer": "y"}]).to_parquet(tmp_path / "val.parquet")
    template = (ROOT / "examples/system_prompt/deepeyes.txt").read_text()
    episode = DeepEyesEpisode(DeepEyesRows(tmp_path), template)

    reset = episode.reset({"split": "train", "index": 0}, seed=0, max_tool_calls=3)
    step = episode.step("image_zoom_in_tool", {"bbox_2d": [0, 0, 500, 1000]})

    assert "at most 3 tool calls" in reset["system_prompt"]
    assert [part["type"] for part in reset["user_content"]] == ["image", "text"]
    assert reset["user_content"][1]["text"] == "\nWhat is it?"
    assert reset["tool_schemas"][0]["function"]["name"] == "image_zoom_in_tool"
    assert step["success"] and [part["type"] for part in step["content"]] == ["image", "text"]
    assert json.loads(step["content"][1]["text"])["status"] == "success"
    assert episode.finalize("x", "answered") == {"reward": None}

    write_tasks(tmp_path, tmp_path / "tasks")
    tasks = pd.read_parquet(tmp_path / "tasks" / "train.parquet")
    assert "images" not in tasks.columns and json.loads(tasks["env_config"][0]) == {"split": "train", "index": 0}


def test_episode_server_does_not_rerun_a_failed_step_or_drop_a_busy_episode():
    class ChargingEpisode(_CountingEpisode):
        charges = 0

        def step(self, name, arguments):
            type(self).charges += 1
            raise RuntimeError("failed after charging")

    server = EpisodeServer(ChargingEpisode)
    episode_id = server.handle("/reset", {"config": {}, "seed": 0})["episode_id"]
    request = {"episode_id": episode_id, "step_index": 0, "tool_call": {"name": "order", "arguments": {}}}

    with pytest.raises(RuntimeError, match="failed after charging"):
        server.handle("/step", request)
    with pytest.raises(ProtocolError, match="takes no more calls") as retried:
        server.handle("/step", request)  # the client's retry: answered, not executed again
    assert ChargingEpisode.charges == 1 and int(retried.value.status) == 409
    with pytest.raises(ProtocolError, match="takes no more calls"):
        server.handle("/finalize", {"episode_id": episode_id, "final_answer": None, "status": "no_action"})

    now = [0.0]
    entered, release = threading.Event(), threading.Event()

    class SlowEpisode(_CountingEpisode):
        def step(self, name, arguments):
            entered.set()
            assert release.wait(5)
            return {"content": [], "success": True}

    server = EpisodeServer(SlowEpisode, idle_timeout_s=10.0, clock=lambda: now[0])
    episode_id = server.handle("/reset", {"config": {}, "seed": 0})["episode_id"]
    step = threading.Thread(
        target=server.handle,
        args=("/step", {"episode_id": episode_id, "step_index": 0, "tool_call": {"name": "t", "arguments": {}}}),
    )
    step.start()
    assert entered.wait(5)
    now[0] = 11.0  # the step runs longer than the idle timeout
    server.handle("/close", {"episode_id": "another"})  # any request drops idle episodes
    release.set()
    step.join(5)
    outcome = server.handle("/finalize", {"episode_id": episode_id, "final_answer": None, "status": "no_action"})
    assert outcome["reward"] == 1.0  # the busy episode was kept, and its last request counts as activity
