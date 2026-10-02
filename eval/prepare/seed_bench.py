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
"""SEED-Bench (HF ``lmms-lab/SEED-Bench``): 17,990 four-option MCQs, ~27 GB of parquet.

14,233 image questions (dimensions 1-9) carry one image; 3,757 video questions (dimensions
10-12) carry the 8 frames sampled by lmms-eval. The 273 parquet shards embed the images, which
would have to be held in memory by every evaluation worker, so they are converted shard by
shard into image files plus one annotation file. Each shard is deleted right after its
conversion (unless ``--keep-raw``), keeping the peak disk use close to the final ~25 GB, and a
per-shard marker makes interrupted runs resume where they stopped::

    <data_root>/seed_bench/annotations.jsonl
    <data_root>/seed_bench/images/<sha1 prefix>.<ext>
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, hf_download, image_extension, write_jsonl


REPOS = ["lmms-lab/SEED-Bench", "lmms-lab-encoder/SEED-Bench"]
EXPECTED_ROWS = 17990
COLUMNS = (
    "question_id",
    "question",
    "choice_a",
    "choice_b",
    "choice_c",
    "choice_d",
    "answer",
    "data_type",
    "question_type_id",
    "data_id",
)


def _list_shards() -> tuple[str, list[str]]:
    from huggingface_hub import list_repo_files

    last_error: Exception | None = None
    for repo_id in REPOS:
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
            return repo_id, sorted(
                name for name in files if name.startswith("data/test-") and name.endswith(".parquet")
            )
        except Exception as exc:  # noqa: BLE001 - try the next repo id
            last_error = exc
    raise PrepareError(f"could not list {REPOS}: {last_error}")


def convert_shard(parquet: Path, target: Path) -> list[dict[str, Any]]:
    """Write the images of one shard to ``target/images`` and return its annotation rows."""
    import pyarrow.parquet as pq

    rows = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32):
        for record in batch.to_pylist():
            relpaths = []
            for image in record.get("image") or []:
                payload = image.get("bytes") if isinstance(image, dict) else None
                if not payload:
                    continue
                relpath = f"images/{hashlib.sha1(payload).hexdigest()[:20]}{image_extension(payload)}"
                path = target / relpath
                if not path.exists() or path.stat().st_size != len(payload):
                    tmp = path.with_name(path.name + ".tmp")
                    tmp.write_bytes(payload)
                    tmp.replace(path)
                relpaths.append(relpath)
            row = {column: record.get(column) for column in COLUMNS}
            row["images"] = relpaths
            rows.append(row)
    return rows


def prepare_seed_bench(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    target = spec.target_dir(ctx.data_root)
    (target / "images").mkdir(parents=True, exist_ok=True)
    shard_dir = target / ".shards"
    shard_dir.mkdir(exist_ok=True)
    repo_id, shards = _list_shards()
    for index, shard in enumerate(shards, start=1):
        done = shard_dir / (Path(shard).name + ".jsonl")
        if done.exists():
            continue
        local = target / shard  # an earlier in-place download of the parquet shards
        parquet = local if local.is_file() else hf_download(ctx, repo_id, shard)
        rows = convert_shard(parquet, target)
        write_jsonl(done, rows)
        if not ctx.keep_raw:
            parquet.unlink(missing_ok=True)
        if index % 20 == 0 or index == len(shards):
            ctx.log(f"  [seed_bench] converted {index}/{len(shards)} shards")
    rows = []
    for shard in shards:
        with (shard_dir / (Path(shard).name + ".jsonl")).open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if len(rows) != EXPECTED_ROWS:
        raise PrepareError(f"seed_bench: expected {EXPECTED_ROWS} rows, found {len(rows)}")
    write_jsonl(target / "annotations.jsonl", rows)
    shutil.rmtree(shard_dir, ignore_errors=True)
    if not ctx.keep_raw:
        shutil.rmtree(target / "data", ignore_errors=True)
        shutil.rmtree(target / ".cache", ignore_errors=True)
    images = sum(1 for _ in (target / "images").iterdir())
    return {"repo_id": repo_id, "rows": len(rows), "images": images, "shards": len(shards)}


SOURCES = [
    BenchmarkSource(
        key="seed_bench",
        target="seed_bench",
        outputs=("seed_bench/annotations.jsonl",),
        source="lmms-lab/SEED-Bench (273 parquet shards, converted to image files)",
        approx_size="27 GB download, ~25 GB on disk",
        prepare=prepare_seed_bench,
        disk_gb=30,
    )
]
