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
"""A scripted policy for the toy search environment, to make SFT demonstrations (scripts/agent/export_sft.py).

It zooms into the four quarters of the image in order (they overlap, so the digit lies whole in at least one)
and answers as soon as a zoomed view shows the digit. It knows where the digit is drawn (the task's layout) and
uses that only to decide what a view shows; its words describe only what the views show.

    python scripts/agent/export_sft.py --policy examples/agent_env/toy_search_policy.py:ToySearchPolicy ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy_search import CANVAS, render_task_layout  # noqa: E402


QUARTERS = (
    ("top-left", [0, 0, 550, 550]),
    ("top-right", [450, 0, 1000, 550]),
    ("bottom-left", [0, 450, 550, 1000]),
    ("bottom-right", [450, 450, 1000, 1000]),
)
GLYPH = (7, 11)  # the digit's size in canvas pixels (the environment's font, size 11)


class ToySearchPolicy:
    def start(self, env_config: dict[str, Any], seed: int, reset: Any) -> None:
        _, self.digit, (x, y) = render_task_layout(int(env_config["image_seed"]))
        scale = 1000.0 / CANVAS
        self.glyph = (x * scale, y * scale, (x + GLYPH[0]) * scale, (y + GLYPH[1]) * scale)
        self.quarter = 0

    def act(self, step: Optional[Any]) -> dict[str, Any]:
        if step is not None and self._shows_digit(QUARTERS[self.quarter - 1][1]):
            name = QUARTERS[self.quarter - 1][0]
            return {"content": f"The {name} part shows the digit {self.digit}.\n<answer>{self.digit}</answer>"}
        name, box = QUARTERS[self.quarter]
        if step is None:
            content = f"The characters are too small to read at this size. I will zoom into the {name} part first."
        else:
            content = f"I see only letters in the {QUARTERS[self.quarter - 1][0]} part. Next, the {name} part."
        self.quarter += 1
        return {"content": content, "tool_call": {"name": "zoom_in", "arguments": {"bbox_2d": box}}}

    def _shows_digit(self, box: list[int]) -> bool:
        x1, y1, x2, y2 = self.glyph
        return box[0] <= x1 and box[1] <= y1 and x2 <= box[2] and y2 <= box[3]
