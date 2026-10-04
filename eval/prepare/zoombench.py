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
"""ZoomBench (HF ``inclusionAI/ZoomBench``): 845 fine-grained perception questions.

621 questions are multiple choice (``question_type`` ``mcq``, 2 or 4 options written into
``query``, gold letter in ``response``) and 224 ask for a count (``blank``, gold integer). The
single parquet file (4 GB) embeds every full image and the crop of its key region (``bbox``) as
PNG. Images are decoded once, de-duplicated by content and written to disk; the crops are kept
for the region-view protocol of the ZoomBench paper, while the registry entry evaluates the full
image::

    <data_root>/zoombench/annotations.jsonl
    <data_root>/zoombench/images/<sha1 prefix>.<ext>
    <data_root>/zoombench/crops/<sha1 prefix>.<ext>
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, hf_download, image_extension, write_jsonl


REPO_ID = "inclusionAI/ZoomBench"
FILENAME = "data/test.parquet"
EXPECTED_ROWS = 845


def _write_image(target: Path, folder: str, payload: bytes) -> str:
    relpath = f"{folder}/{hashlib.sha1(payload).hexdigest()[:20]}{image_extension(payload)}"
    path = target / relpath
    if not path.exists() or path.stat().st_size != len(payload):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    return relpath


def prepare_zoombench(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    import pyarrow.parquet as pq

    target = spec.target_dir(ctx.data_root)
    for folder in ("images", "crops"):
        (target / folder).mkdir(parents=True, exist_ok=True)
    raw = hf_download(ctx, REPO_ID, FILENAME)
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=8):
        for record in batch.to_pylist():
            rows.append(
                {
                    "id": record["id"],
                    "query": record["query"],
                    "response": record["response"],
                    "question_type": record["question_type"],
                    "bbox": list(record["bbox"] or []),
                    "image": _write_image(target, "images", record["image"]["bytes"]),
                    "crop_image": _write_image(target, "crops", record["crop_image"]["bytes"]),
                }
            )
    if len(rows) != EXPECTED_ROWS:
        raise PrepareError(f"{spec.key}: expected {EXPECTED_ROWS} rows, found {len(rows)}")
    write_jsonl(target / "annotations.jsonl", rows)
    ctx.discard_raw(raw)
    return {
        "repo_id": REPO_ID,
        "file": FILENAME,
        "rows": len(rows),
        "images": len({row["image"] for row in rows}),
        "crops": len({row["crop_image"] for row in rows}),
    }


SOURCES = [
    BenchmarkSource(
        key="zoombench",
        target="zoombench",
        outputs=("zoombench/annotations.jsonl",),
        source=f"{REPO_ID} ({FILENAME})",
        approx_size="4.0 GB download, ~3.8 GB on disk",
        prepare=prepare_zoombench,
        disk_gb=8.5,
    )
]
