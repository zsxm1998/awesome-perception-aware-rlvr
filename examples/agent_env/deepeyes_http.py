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
"""DeepEyes' zoom-in task served over HTTP: the remote counterpart of the in-process "deepeyes" environment.

The server reads the DeepEyes parquet files (scripts/data/prepare_train_data.py deepeyes) and serves a row:
the system prompt (the native DeepEyes prompt with ``{{ max_tool_calls }}`` rendered, or the row's own for
rows without a tool), the row's images and question, and ``image_zoom_in_tool`` with 0-1000 boxes (the same
tool code). Prompts and observations have the layout of the in-process environment's native prompt, so the
policy reads the same tokens; only the batching of vLLM requests differs. The reward stays with DeepEyes'
reward function on the trainer (finalize reports none). Only the native prompt with 0-1000 boxes (Qwen3-VL).

    python examples/agent_env/deepeyes_http.py tasks --data-dir data/deepeyes --out data/agent_env/deepeyes_http
    python examples/agent_env/deepeyes_http.py serve --data-dir data/deepeyes --port 8001
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import image_part, serve, text_part  # noqa: E402

from verl.utils.dataset import process_image  # noqa: E402
from verl.workers.agent.chat import render_deepeyes_system_prompt  # noqa: E402
from verl.workers.agent.protocol import ToolCall  # noqa: E402
from verl.workers.agent.tools import ImageZoomInTool, ToolRegistry  # noqa: E402


DEFAULT_SYSTEM_PROMPT = ROOT / "examples/system_prompt/deepeyes.txt"


class DeepEyesRows:
    """The rows of data/deepeyes/{split}.parquet, read once per server."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._frames: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def row(self, split: str, index: int) -> dict[str, Any]:
        with self._lock:
            if split not in self._frames:
                self._frames[split] = pd.read_parquet(self.data_dir / f"{split}.parquet")
        return self._frames[split].iloc[index].to_dict()

    @functools.lru_cache(maxsize=512)  # the rollouts of a group reset the same row
    def source_images(self, split: str, index: int) -> tuple[Any, tuple[dict[str, Any], ...]]:
        images = [process_image(image, None, None) for image in self.row(split, index)["images"]]
        return images, tuple(image_part(image) for image in images)


class DeepEyesEpisode:
    def __init__(self, rows: DeepEyesRows, system_prompt_template: str):
        self.rows = rows
        self.system_prompt_template = system_prompt_template

    def reset(self, config: dict[str, Any], seed: int, max_tool_calls: int) -> dict[str, Any]:
        split, index = config.get("split", "train"), int(config["index"])
        row = self.rows.row(split, index)
        self.source_images, image_parts = self.rows.source_images(split, index)
        tools_enabled = bool(row.get("env_name"))
        self.registry = ToolRegistry([ImageZoomInTool()] if tools_enabled else [])
        if tools_enabled:
            system_prompt = render_deepeyes_system_prompt(self.system_prompt_template, max_tool_calls=max_tool_calls)
        else:  # rows without a tool (ThinkLite) keep their own system prompt and answer in one turn
            system_prompt = str(row.get("row_system_prompt") or "").strip() or None
        # the user message as the dataset builds it ("<image>" splits the text), with DeepEyes' image labels
        user_content: list[dict[str, Any]] = []
        for position, text in enumerate(str(row["problem"]).split("<image>")):
            if position:
                if len(self.source_images) > 1:
                    user_content.append(text_part(f"Image {position - 1}:\n"))
                user_content.append(image_parts[position - 1])
            if text:
                user_content.append(text_part(text))
        return {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "tool_schemas": self.registry.schemas(len(self.source_images)),
        }

    def step(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.registry.execute(ToolCall(name=name, arguments=arguments, raw_text=""), self.source_images)
        # the observation text of the native DeepEyes prompt: the result's JSON, crop first
        content_json = json.dumps(result.content, ensure_ascii=False, separators=(",", ":"))
        content_json = content_json.replace("<", "\\u003c").replace(">", "\\u003e")
        return {
            "content": [*(image_part(image) for image in result.images), text_part(content_json)],
            "success": result.success,
            **({} if result.success else {"error_code": result.error_code}),
            "info": result.metadata,
        }

    def finalize(self, final_answer: Optional[str], status: str) -> dict[str, Any]:
        return {"reward": None}  # scored on the trainer by examples/reward_function/deepeyes.py


def write_tasks(data_dir: Path, out: Path) -> None:
    """DeepEyes rows without their images, plus env_config pointing at the row the server serves."""
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        frame = pd.read_parquet(data_dir / f"{split}.parquet").drop(columns=["images"])
        frame["env_config"] = [json.dumps({"split": split, "index": index}) for index in range(len(frame))]
        frame.to_parquet(out / f"{split}.parquet")
        print(f"[deepeyes_http] {len(frame)} tasks -> {out / f'{split}.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks", help="write the training rows of the remote task")
    tasks.add_argument("--data-dir", type=Path, default=Path("data/deepeyes"))
    tasks.add_argument("--out", type=Path, default=Path("data/agent_env/deepeyes_http"))
    run = commands.add_parser("serve", help="serve the environment")
    run.add_argument("--data-dir", type=Path, default=Path("data/deepeyes"))
    run.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.command == "tasks":
        write_tasks(args.data_dir, args.out)
        return
    rows = DeepEyesRows(args.data_dir)
    template = args.system_prompt.read_text(encoding="utf-8")
    serve(lambda: DeepEyesEpisode(rows, template), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
