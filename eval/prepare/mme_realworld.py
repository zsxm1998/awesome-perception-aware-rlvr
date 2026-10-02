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
"""MME-RealWorld-Lite (HF ``yifanzhang114/MME-RealWorld-lite-lmms-eval``): 1,919 A-E MCQs.

The lmms-eval export stores every image as a base64 string next to ``question``,
``multi-choice options`` (``"(A) ..."``), ``answer`` (letter), ``category``
(``Perception/<task>`` or ``Reasoning/<task>``) and ``l2-category``. The four parquet shards
(1.9 GB) are processed one at a time; images are decoded once, de-duplicated by content and
written to disk::

    <data_root>/mme_realworld_lite/annotations.jsonl
    <data_root>/mme_realworld_lite/images/<sha1 prefix>.<ext>
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, hf_download, image_extension, write_jsonl


REPO_ID = "yifanzhang114/MME-RealWorld-lite-lmms-eval"
SHARDS = [f"data/train-0000{index}-of-00004.parquet" for index in range(4)]
EXPECTED_ROWS = 1919


def prepare_mme_realworld_lite(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    import pyarrow.parquet as pq

    target = spec.target_dir(ctx.data_root)
    (target / "images").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    written: dict[str, str] = {}
    for shard in SHARDS:
        raw = hf_download(ctx, REPO_ID, shard)
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=16):
            for record in batch.to_pylist():
                encoded = record["bytes"]
                digest = hashlib.sha1(encoded.encode("ascii")).hexdigest()[:20]
                relpath = written.get(digest)
                if relpath is None:
                    payload = base64.b64decode(encoded)
                    relpath = f"images/{digest}{image_extension(payload)}"
                    path = target / relpath
                    if not path.exists() or path.stat().st_size != len(payload):
                        path.write_bytes(payload)
                    written[digest] = relpath
                rows.append(
                    {
                        "index": record["index"],
                        "question": record["question"],
                        "options": list(record["multi-choice options"] or []),
                        "answer": record["answer"],
                        "category": record["category"],
                        "l2_category": record["l2-category"],
                        "image": relpath,
                    }
                )
        ctx.discard_raw(raw)
    if len(rows) != EXPECTED_ROWS:
        raise PrepareError(f"{spec.key}: expected {EXPECTED_ROWS} rows, found {len(rows)}")
    rows.sort(key=lambda row: row["index"])
    write_jsonl(target / "annotations.jsonl", rows)
    return {"repo_id": REPO_ID, "rows": len(rows), "images": len(written)}


SOURCES = [
    BenchmarkSource(
        key="mme_realworld_lite",
        target="mme_realworld_lite",
        outputs=("mme_realworld_lite/annotations.jsonl",),
        source=f"{REPO_ID} (4 parquet shards)",
        approx_size="1.9 GB download, ~1.3 GB on disk",
        prepare=prepare_mme_realworld_lite,
        disk_gb=3.5,
    )
]
