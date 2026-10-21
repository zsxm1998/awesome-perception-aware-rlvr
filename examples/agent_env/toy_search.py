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
"""A toy visual-search environment served over HTTP: find the one digit among small letters on a large canvas.

Each task draws colored boxes, small letters and one small digit on a 1024x1024 canvas, determined by the
task's ``image_seed``. The model sees the canvas, may call ``zoom_in`` (a 0-1000 box, returned at 512 px on its long
side) and answers the digit in <answer></answer>. The episode reward is 1 for the right digit. ``state_key``
is the view after a zoom-in (zoom level and 4x4 grid cell of its center), for grouping steps across rollouts.

    python examples/agent_env/toy_search.py tasks --out data/agent_env/toy_search   # train/val parquet
    python examples/agent_env/toy_search.py serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont


sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import image_part, serve, text_part  # noqa: E402


CANVAS = 1024
ZOOM_SIDE = 512
SYSTEM_PROMPT = (
    "Among the small letters in the image there is exactly one digit (0-9). "
    "You may call zoom_in to look at a region more closely; you can call it at most {max_tool_calls} times. "
    "Think briefly, then give the digit inside <answer></answer>, for example <answer>7</answer>."
)
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "zoom_in",
        "description": "Show a region of the original image at a higher resolution.",
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "The region [x1, y1, x2, y2], in 0-1000 coordinates of the image.",
                }
            },
            "required": ["bbox_2d"],
        },
    },
}


def render_task(image_seed: int) -> tuple[Image.Image, str]:
    """The canvas and its digit; the same seed always gives the same task."""
    rng = random.Random(image_seed)
    canvas = Image.new("RGB", (CANVAS, CANVAS), (236, 236, 228))
    draw = ImageDraw.Draw(canvas)
    for _ in range(40):
        x, y = rng.randrange(0, CANVAS - 40), rng.randrange(0, CANVAS - 40)
        w, h = rng.randrange(20, 160), rng.randrange(20, 160)
        color = tuple(rng.randrange(80, 230) for _ in range(3))
        draw.rectangle((x, y, min(CANVAS - 1, x + w), min(CANVAS - 1, y + h)), fill=color)
    font = ImageFont.load_default(size=11)
    for _ in range(80):  # distractors: letters that look alike at a glance
        letter = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        draw.text(
            (rng.randrange(20, CANVAS - 30), rng.randrange(20, CANVAS - 30)), letter, fill=(30, 30, 30), font=font
        )
    digit = str(rng.randrange(10))
    draw.text((rng.randrange(20, CANVAS - 30), rng.randrange(20, CANVAS - 30)), digit, fill=(30, 30, 30), font=font)
    return canvas, digit


class ToySearchEpisode:
    def reset(self, config: dict[str, Any], seed: int, max_tool_calls: int) -> dict[str, Any]:
        # the task is fully given by the configuration; ``seed`` (shared by a group) is not needed
        self.canvas, self.digit = render_task(int(config["image_seed"]))
        self.zoom_calls = 0
        return {
            "system_prompt": SYSTEM_PROMPT.format(max_tool_calls=max_tool_calls),
            "user_content": [image_part(self.canvas), text_part("Which digit is hidden in this image?")],
            "tool_schemas": [TOOL_SCHEMA],
        }

    def step(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != "zoom_in":
            return self._error("unknown_tool", f"unknown tool {name!r}; the only tool is zoom_in")
        box = arguments.get("bbox_2d")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box)
        ):
            return self._error("invalid_bbox", "bbox_2d must be four numbers [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (min(1000.0, max(0.0, float(value))) for value in box)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return self._error(
                "invalid_bbox", "the box must be at least 10 units wide and high, with x1 < x2, y1 < y2"
            )
        self.zoom_calls += 1
        scale = CANVAS / 1000.0
        region = self.canvas.crop((round(x1 * scale), round(y1 * scale), round(x2 * scale), round(y2 * scale)))
        factor = ZOOM_SIDE / max(region.size)
        region = region.resize((max(1, round(region.width * factor)), max(1, round(region.height * factor))))
        side = max(x2 - x1, y2 - y1)
        level = 1 if side > 500 else 2 if side > 250 else 3
        cell = (min(3, int((x1 + x2) / 2 / 250)), min(3, int((y1 + y2) / 2 / 250)))
        return {
            "content": [text_part(f"Zoomed into [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]:"), image_part(region)],
            "success": True,
            "state_key": f"level{level}:cell{cell[0]}{cell[1]}",
            "info": {"zoom_level": level},
        }

    def finalize(self, final_answer: Optional[str], status: str) -> dict[str, Any]:
        correct = float(final_answer is not None and final_answer.strip() == self.digit)
        return {"reward": correct, "components": {"correct": correct}, "metrics": {"zoom_calls": self.zoom_calls}}

    def _error(self, code: str, message: str) -> dict[str, Any]:
        return {"content": [text_part(f"Error: {message}")], "success": False, "error_code": code}


def write_tasks(out: Path, num_train: int, num_val: int) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    for split, first, count in (("train", 0, num_train), ("val", 10**6, num_val)):
        rows = []
        for image_seed in range(first, first + count):
            _, digit = render_task(image_seed)
            rows.append(
                {
                    "prompt": "Find the hidden digit.",  # the server writes the real prompt
                    "answer": digit,
                    "env_config": json.dumps({"image_seed": image_seed}),
                }
            )
        pd.DataFrame(rows).to_parquet(out / f"{split}.parquet")
        print(f"[toy_search] {len(rows)} tasks -> {out / f'{split}.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks", help="write train/val task parquet files")
    tasks.add_argument("--out", type=Path, default=Path("data/agent_env/toy_search"))
    tasks.add_argument("--num-train", type=int, default=1024)
    tasks.add_argument("--num-val", type=int, default=64)
    run = commands.add_parser("serve", help="serve the environment")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "tasks":
        write_tasks(args.out, args.num_train, args.num_val)
    else:
        serve(ToySearchEpisode, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
