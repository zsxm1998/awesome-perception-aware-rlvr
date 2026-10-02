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
"""HR-Bench 4K / 8K (HF ``DreamMr/HR-Bench``): 800 four-option MCQs per version.

The parquet stores every image as a base64 string, and each image appears in four rows
(one per cyclic option permutation). Images are decoded once, de-duplicated by content and
written to disk so that evaluation workers do not hold gigabytes of base64 in memory::

    <data_root>/hrbench_{4k,8k}/annotations.jsonl
    <data_root>/hrbench_{4k,8k}/images/<sha1 prefix>.<ext>
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, hf_download, image_extension, write_jsonl


REPO_ID = "DreamMr/HR-Bench"
VERSIONS = {
    "hrbench_4k": ("hr_bench_4k.parquet", "430 MB"),
    "hrbench_8k": ("hr_bench_8k.parquet", "2.8 GB download, ~0.7 GB on disk"),
}
ANNOTATION_COLUMNS = ("index", "question", "answer", "category", "A", "B", "C", "D", "cycle_category")


def prepare_hrbench(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    import pyarrow.parquet as pq

    filename, _ = VERSIONS[spec.key]
    target = spec.target_dir(ctx.data_root)
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    raw = hf_download(ctx, REPO_ID, filename)
    parquet = pq.ParquetFile(raw)
    rows: list[dict[str, Any]] = []
    written: dict[str, str] = {}
    for batch in parquet.iter_batches(batch_size=8):
        for record in batch.to_pylist():
            encoded = record["image"]
            digest = hashlib.sha1(encoded.encode("ascii") if isinstance(encoded, str) else encoded).hexdigest()[:20]
            relpath = written.get(digest)
            if relpath is None:
                payload = base64.b64decode(encoded)
                relpath = f"images/{digest}{image_extension(payload)}"
                path = target / relpath
                if not path.exists() or path.stat().st_size != len(payload):
                    path.write_bytes(payload)
                written[digest] = relpath
            row = {column: record.get(column) for column in ANNOTATION_COLUMNS}
            row["image"] = relpath
            rows.append(row)
    if len(rows) != 800:
        raise PrepareError(f"{spec.key}: expected 800 rows, found {len(rows)}")
    write_jsonl(target / "annotations.jsonl", rows)
    ctx.discard_raw(raw)
    return {"repo_id": REPO_ID, "file": filename, "rows": len(rows), "images": len(written)}


SOURCES = [
    BenchmarkSource(
        key=key,
        target=key,
        outputs=(f"{key}/annotations.jsonl",),
        source=f"{REPO_ID} ({filename})",
        approx_size=size,
        prepare=prepare_hrbench,
    )
    for key, (filename, size) in VERSIONS.items()
]
