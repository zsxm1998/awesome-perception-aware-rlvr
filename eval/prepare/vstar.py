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
"""V* Bench (HF ``craigwu/vstar_bench``): 191 high-resolution MCQs (115 attribute, 76 spatial).

``test_questions.jsonl`` holds the official prompts with shuffled option letters and the
gold ``label``; images live in ``direct_attributes/`` and ``relative_position/``. Layout::

    <data_root>/vstar/test_questions.jsonl
    <data_root>/vstar/{direct_attributes,relative_position}/<image>.jpg
"""

from __future__ import annotations

import shutil
from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, hf_snapshot, move_file, read_jsonl


REPO_ID = "craigwu/vstar_bench"
CATEGORIES = ("direct_attributes", "relative_position")


def prepare_vstar(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    target = spec.target_dir(ctx.data_root)
    raw = hf_snapshot(ctx, REPO_ID, ["test_questions.jsonl", *(f"{category}/*" for category in CATEGORIES)])
    rows = read_jsonl(raw / "test_questions.jsonl")
    if len(rows) != 191:
        raise PrepareError(f"vstar: expected 191 questions, found {len(rows)}")
    for category in CATEGORIES:
        (target / category).mkdir(parents=True, exist_ok=True)
        for path in (raw / category).iterdir():
            if path.is_file():
                shutil.move(str(path), str(target / category / path.name))
    move_file(raw / "test_questions.jsonl", target / "test_questions.jsonl")
    missing = [row["image"] for row in rows if not (target / row["image"]).is_file()]
    if missing:
        raise PrepareError(f"vstar: {len(missing)} images missing, e.g. {missing[:3]}")
    ctx.discard_raw(raw)
    return {"repo_id": REPO_ID, "rows": len(rows), "images": len({row["image"] for row in rows})}


SOURCES = [
    BenchmarkSource(
        key="vstar",
        target="vstar",
        outputs=("vstar/test_questions.jsonl",),
        source=f"{REPO_ID} (test_questions.jsonl)",
        approx_size="230 MB",
        prepare=prepare_vstar,
    )
]
