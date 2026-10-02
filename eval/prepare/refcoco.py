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
"""RefCOCO / RefCOCO+ / RefCOCOg val (HF ``PaDT-MLLM/RefCOCO`` + COCO train2014 images).

The PaDT release stores one referring expression per JSON line with the target box already
normalized to [0, 1] xyxy. Only the COCO images referenced by the val splits are fetched
(about 3k images, ~0.5 GB) instead of the 13 GB COCO train2014 archive. The three
benchmarks share one image folder::

    <data_root>/refcoco/{refcoco,refcoco+,refcocog}_val.jsonl
    <data_root>/refcoco/images/COCO_train2014_<id>.jpg
"""

from __future__ import annotations

import json
from typing import Any

from .base import BenchmarkSource
from .common import DownloadItem, PrepareContext, PrepareError, hf_download, http_download_many, write_jsonl


REPO_ID = "PaDT-MLLM/RefCOCO"
COCO_URL = "http://images.cocodataset.org"

# key -> (PaDT file, output file, expected rows)
SPLITS: dict[str, tuple[str, str, int]] = {
    "refcoco_val": ("refcoco_val.json", "refcoco_val.jsonl", 10834),
    "refcoco_plus_val": ("refcoco+_val.json", "refcoco+_val.jsonl", 10758),
    "refcocog_val": ("refcocog_val.json", "refcocog_val.jsonl", 4896),
}


def _compact_rows(raw_lines: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        item = json.loads(line)
        objects = item.get("objects") or []
        if len(objects) != 1:
            raise PrepareError(f"RefCOCO row {index} has {len(objects)} target objects; expected exactly one")
        target = objects[0]
        bbox = [float(value) for value in target["bbox"]]
        if len(bbox) != 4 or not (0.0 <= bbox[0] < bbox[2] <= 1.0 and 0.0 <= bbox[1] < bbox[3] <= 1.0):
            raise PrepareError(f"RefCOCO row {index} has an invalid normalized box: {bbox}")
        rows.append(
            {
                "sample_id": f"{item.get('id')}:{index}",
                "coco_image_id": item.get("id"),
                "image": f"images/{item['image']}",
                "expression": str(target.get("label") or "").strip(),
                "bbox": bbox,
            }
        )
    return rows


def prepare_refcoco(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    source_file, output_file, expected_rows = SPLITS[spec.key]
    target = spec.target_dir(ctx.data_root)
    raw = hf_download(ctx, REPO_ID, source_file)
    rows = _compact_rows(raw.read_text(encoding="utf-8").splitlines())
    if len(rows) != expected_rows:
        raise PrepareError(f"{spec.key}: expected {expected_rows} expressions, found {len(rows)}")
    items = {
        row["image"]: DownloadItem(
            urls=(f"{COCO_URL}/train2014/{row['image'].split('/', 1)[1]}",),
            dest=target / row["image"],
        )
        for row in rows
    }
    failures = http_download_many(ctx, list(items.values()), desc=f"{spec.key} COCO train2014 images")
    if failures:
        examples = "; ".join(f"{path.name}: {error}" for path, error in list(failures.items())[:3])
        raise PrepareError(f"{spec.key}: {len(failures)} images could not be downloaded ({examples}); re-run to retry")
    write_jsonl(target / output_file, rows)
    ctx.discard_raw(raw)
    return {"repo_id": REPO_ID, "file": source_file, "rows": len(rows), "images": len(items)}


SOURCES = [
    BenchmarkSource(
        key=key,
        target="refcoco",
        outputs=(f"refcoco/{output_file}",),
        source=f"{REPO_ID} ({source_file}) + COCO train2014 images",
        approx_size="~0.5 GB shared by the three splits",
        prepare=prepare_refcoco,
    )
    for key, (source_file, output_file, _) in SPLITS.items()
]
