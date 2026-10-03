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
"""VPPO-Eval (HF ``chamber111/VPPO-Eval``): the two VPPO benchmarks missing from PAPO-Eval.

The repository is a LLaMA-Factory ``data/`` folder: ``data/vppo/<name>.json`` (list of
``{messages, images}``) and loose images under ``data/images/<name>/``. The other six VPPO
columns have the same items as the PAPO-Eval splits and are prepared from there. Layout::

    <data_root>/<key>/test.json
    <data_root>/<key>/images/<name>/<file>.png
"""

from __future__ import annotations

import json
import shutil
from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, copy_file, hf_snapshot, place_referenced_files
from .papo_eval import sharegpt_image_relpath


REPO_ID = "chamber111/VPPO-Eval"

# key -> (file stem, approximate size, expected rows)
SPLITS: dict[str, tuple[str, str, int]] = {
    "dynamath": ("DynaMath_DynaMath_Sample", "190 MB", 3666),
    "mathvision": ("MathLLMs_MathVision", "120 MB", 2907),
}


def prepare_vppo_split(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    name, _, expected_rows = SPLITS[spec.key]
    target = spec.target_dir(ctx.data_root)
    raw = hf_snapshot(ctx, REPO_ID, [f"data/vppo/{name}.json", f"data/images/{name}/*"])
    rows = json.loads((raw / "data" / "vppo" / f"{name}.json").read_text(encoding="utf-8"))
    if len(rows) != expected_rows:
        raise PrepareError(f"{spec.key}: expected {expected_rows} rows in {name}.json, found {len(rows)}")
    copy_file(raw / "data" / "vppo" / f"{name}.json", target / "test.json")

    source_images = raw / "data" / "images" / name
    dest_images = target / "images" / name
    dest_images.mkdir(parents=True, exist_ok=True)
    for path in source_images.iterdir():
        if path.is_file():
            shutil.move(str(path), str(dest_images / path.name))
    relpaths = [sharegpt_image_relpath(item) for row in rows for item in row.get("images") or []]
    place_referenced_files(target, relpaths, search_root=target / "images")
    ctx.discard_raw(raw / "data" / "images" / name)
    ctx.discard_raw(raw / "data" / "vppo" / f"{name}.json")
    return {"repo_id": REPO_ID, "file": f"data/vppo/{name}.json", "rows": len(rows), "images": len(set(relpaths))}


SOURCES = [
    BenchmarkSource(
        key=key,
        target=key,
        outputs=(f"{key}/test.json",),
        source=f"{REPO_ID} (data/vppo/{name}.json)",
        approx_size=size,
        prepare=prepare_vppo_split,
    )
    for key, (name, size, _) in SPLITS.items()
]
