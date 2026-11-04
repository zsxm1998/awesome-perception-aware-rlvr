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
"""The toy visual search of toy_search.py as an OpenEnv MCP environment, trained with ``agent_environment=openenv``.

The same tasks, canvas and zoom-ins as the HTTP version. ``reset`` (with ``split`` and ``index``) returns the
canvas as a base64 PNG in its ``image`` field and the question in ``question``; the MCP tool ``zoom_in`` returns the
region's text and image (a bad box is a tool error); the tool ``answer`` ends the episode with reward 1 for the right
digit. Tasks are (split, index): ``train`` (1024) and ``val`` (64).

The server needs the ``openenv`` package (pip install openenv; tested with 0.11.0), which requires a newer
huggingface_hub than transformers 4.x allows, so run it in its own Python environment:

    python examples/agent_env/toy_search_openenv.py tasks --out data/agent_env/toy_search_openenv
    python examples/agent_env/toy_search_openenv.py serve --port 8020        # in the openenv environment
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_search import SPLIT_SEEDS, ToySearchEpisode, render_task  # noqa: E402


NUM_TASKS = {"train": 1024, "val": 64}
QUESTION = (
    "Among the small letters in the image there is exactly one digit (0-9). zoom_in shows a region of the image "
    "more closely. Find the digit, then call answer with it."
)


def write_tasks(out: Path) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    for split, count in NUM_TASKS.items():
        rows = [
            {
                "prompt": "Find the hidden digit.",  # the environment writes the real prompt
                "answer": render_task(SPLIT_SEEDS[split] + index)[1],
                "env_config": json.dumps({"reset": {"split": split, "index": index}}),
            }
            for index in range(count)
        ]
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet")
        print(f"[toy_search_openenv] {len(rows)} tasks -> {out / f'{split}.parquet'}")


def serve(host: str, port: int, max_concurrent_envs: int) -> None:
    import uvicorn
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from fastmcp.utilities.types import Image
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Observation, State

    class ToySearchObservation(Observation):
        image: Optional[str] = None  # the canvas, a base64 PNG (reset only)
        question: Optional[str] = None

    class ToySearchEnvironment(MCPEnvironment):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self) -> None:
            mcp = FastMCP("toy_search")
            self.episode = ToySearchEpisode()
            self._state = State(step_count=0)

            @mcp.tool(output_schema=None)  # content blocks (text and an image), no structured output
            def zoom_in(bbox_2d: list) -> list:
                """Show a region of the original image at a higher resolution.

                Args:
                    bbox_2d: The region [x1, y1, x2, y2], in 0-1000 coordinates of the image.
                """
                result = self.episode.step("zoom_in", {"bbox_2d": bbox_2d})
                if not result["success"]:
                    raise ToolError(result["content"][0]["text"])
                text, image = result["content"]
                return [text["text"], Image(data=base64.b64decode(image["png_base64"]), format="png")]

            @mcp.tool
            def answer(answer: str) -> str:
                """Give the digit; this ends the task.

                Args:
                    answer: The digit.
                """
                return "submitted"

            super().__init__(mcp)

        def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
            image_seed = SPLIT_SEEDS[kwargs.get("split", "train")] + int(kwargs.get("index", 0))
            self.episode = ToySearchEpisode()
            reset = self.episode.reset({"image_seed": image_seed}, seed=seed or 0, max_tool_calls=0)
            self._state = State(episode_id=episode_id, step_count=0)
            return ToySearchObservation(image=reset["user_content"][0]["png_base64"], question=QUESTION)

        def _scored(self, action: Any, observation: Observation) -> Observation:
            self._state.step_count += 1
            if isinstance(action, CallToolAction) and action.tool_name == "answer":
                outcome = self.episode.finalize(str(action.arguments.get("answer", "")), "answered")
                return Observation(done=True, reward=outcome["reward"], metadata={"digit": self.episode.digit})
            if isinstance(observation, CallToolObservation) and observation.error is None:
                observation.metadata = {"view": self.episode.view}
            return observation

        def step(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
            return self._scored(action, super().step(action, timeout_s=timeout_s, **kwargs))

        async def step_async(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
            return self._scored(action, await super().step_async(action, timeout_s=timeout_s, **kwargs))

        def _step_impl(self, action: Any, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
            return Observation(metadata={"error": f"unknown action {type(action).__name__}; use the MCP tools"})

        @property
        def state(self) -> State:
            return self._state

    app = create_app(
        ToySearchEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="toy_search",
        max_concurrent_envs=max_concurrent_envs,
    )
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks", help="write train/val task parquet files")
    tasks.add_argument("--out", type=Path, default=Path("data/agent_env/toy_search_openenv"))
    run = commands.add_parser("serve", help="serve the environment")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8020)
    run.add_argument("--max-concurrent-envs", type=int, default=64, help="sessions the server keeps at a time")
    args = parser.parse_args()
    if args.command == "tasks":
        write_tasks(args.out)
    else:
        serve(args.host, args.port, args.max_concurrent_envs)


if __name__ == "__main__":
    main()
