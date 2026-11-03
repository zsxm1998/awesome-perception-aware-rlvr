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
"""The toy visual search of toy_search.py as an environment of the Open Reward Standard (ORS), served by the
``openreward`` SDK (pip install openreward; tested with 0.1.167) and trained with ``agent_environment=openreward``.

The same tasks, canvas and zoom-ins as the HTTP version: the prompt is the canvas and the question, ``zoom_in``
returns the region's text and image (with the view's name in its metadata), and ``answer`` is the environment's
terminal tool: it is not shown to the model, which writes the digit inside <answer></answer>; the adapter passes
that text to it, and it gives 1 for the right digit. Tasks are (split, index): ``train`` (1024) and ``val`` (64).

    python examples/agent_env/toy_search_openreward.py tasks --out data/agent_env/toy_search_openreward
    python examples/agent_env/toy_search_openreward.py serve --port 8010
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from openreward.environments import Environment, ImageBlock, Server, TextBlock, ToolOutput, terminal, tool
from pydantic import BaseModel, Field


sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_search import SPLIT_SEEDS, ToySearchEpisode, image_seed_of, render_task  # noqa: E402


NUM_TASKS = {"train": 1024, "val": 64}
# how to answer is left to the adapter's system message (the answer inside <answer></answer>), which the terminal
# tool then compares with the digit
QUESTION = (
    "Among the small letters in the image there is exactly one digit (0-9). zoom_in shows a region of the image "
    "more closely. Which digit is it?"
)


class ZoomIn(BaseModel):
    # loosely typed, so that the environment's own checks answer a bad box (as the HTTP version does)
    bbox_2d: list[Any] = Field(description="The region [x1, y1, x2, y2], in 0-1000 coordinates of the image.")


class Answer(BaseModel):
    answer: str


def _blocks(parts: list[dict[str, Any]]) -> list[Any]:
    return [
        TextBlock(text=part["text"])
        if part["type"] == "text"
        else ImageBlock(data=part["png_base64"], mimeType="image/png")
        for part in parts
    ]


class ToySearch(Environment):
    def __init__(self, task_spec: dict[str, Any] = {}, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.episode = ToySearchEpisode()
        # the HTTP version's reset gives the canvas; its system prompt is replaced by the adapter's
        reset = self.episode.reset({"image_seed": image_seed_of(task_spec)}, seed=0, max_tool_calls=0)
        self.canvas_part = reset["user_content"][0]

    @classmethod
    def list_splits(cls) -> list[str]:
        return list(NUM_TASKS)

    @classmethod
    def list_tasks(cls, split: str) -> list[dict[str, Any]]:
        return [{"split": split, "index": index} for index in range(NUM_TASKS[split])]

    def get_prompt(self) -> list[Any]:
        return _blocks([self.canvas_part]) + [TextBlock(text=QUESTION)]

    @tool
    async def zoom_in(self, params: ZoomIn) -> ToolOutput:
        """Show a region of the original image at a higher resolution."""
        result = self.episode.step("zoom_in", {"bbox_2d": params.bbox_2d})
        metadata = {"view": result["state_key"], **result.get("info", {})}
        if not result["success"]:
            metadata["error_code"] = result["error_code"]
        return ToolOutput(blocks=_blocks(result["content"]), metadata=metadata)

    @terminal
    @tool
    async def answer(self, params: Answer) -> ToolOutput:
        """Give the digit."""
        outcome = self.episode.finalize(params.answer, "answered")
        correct = outcome["reward"] == 1.0
        return ToolOutput(
            blocks=[TextBlock(text="correct" if correct else "wrong")],
            reward=outcome["reward"],
            finished=True,
            metadata={"digit": self.episode.digit, "zoom_calls": self.episode.zoom_calls},
        )


def write_tasks(out: Path) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    for split, count in NUM_TASKS.items():
        rows = [
            {
                "prompt": "Find the hidden digit.",  # the environment writes the real prompt
                "answer": render_task(SPLIT_SEEDS[split] + index)[1],
                "env_config": json.dumps({"split": split, "index": index}),
            }
            for index in range(count)
        ]
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet")
        print(f"[toy_search_openreward] {len(rows)} tasks -> {out / f'{split}.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks", help="write train/val task parquet files")
    tasks.add_argument("--out", type=Path, default=Path("data/agent_env/toy_search_openreward"))
    run = commands.add_parser("serve", help="serve the environment")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    if args.command == "tasks":
        write_tasks(args.out)
    else:
        Server([ToySearch]).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
