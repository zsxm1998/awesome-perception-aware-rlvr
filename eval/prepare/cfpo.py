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
"""CFPO real-world benchmarks (HF ``RavenInJuly/CFPO_Datasets``, apache-2.0).

Each benchmark is a JSON list of ``{images, problem, answer, id, type, is_cf}``. Images ship as
zips in the same repository; only what is needed is fetched:

- C-VQA-Real (6,288 questions, 3,026 images) and MARS-Bench (5,110 questions, 1,022 images)
  use COCO val2014 images. They are downloaded one by one from ``images.cocodataset.org``
  (byte-identical to the files in CFPO's 6.6 GB ``COCO_val2014_images.zip``); images that
  cannot be fetched there are read out of that zip with HTTP range requests.
- TextVQA (5,000 validation questions, 3,166 images): the 3.1 GB ``Textvqa_images.zip`` holds
  exactly these images, so it is downloaded (resumable) and extracted, then deleted.

Layout::

    <data_root>/<key>/test.json
    <data_root>/<key>/images/<file name>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BenchmarkSource
from .common import (
    DownloadItem,
    PrepareContext,
    PrepareError,
    copy_file,
    extract_remote_zip_members,
    hf_download,
    http_download_many,
)


REPO_ID = "RavenInJuly/CFPO_Datasets"
COCO_URL = "http://images.cocodataset.org/val2014"

# key -> (annotation file, image archive, expected rows, image source)
SPLITS: dict[str, tuple[str, str, int, str]] = {
    "cvqa_real": ("C-VQA-Real_test.json", "COCO_val2014_images.zip", 6288, "coco"),
    "mars_bench": ("MARS_Bench_test.json", "COCO_val2014_images.zip", 5110, "coco"),
    "textvqa": ("Textvqa_test.json", "Textvqa_images.zip", 5000, "archive"),
}


def _annotations(ctx: PrepareContext, spec: BenchmarkSource) -> list[dict[str, Any]]:
    filename, _, expected, _ = SPLITS[spec.key]
    raw = hf_download(ctx, REPO_ID, filename)
    output = copy_file(raw, spec.target_dir(ctx.data_root) / "test.json")
    ctx.discard_raw(raw)
    rows = json.loads(output.read_text(encoding="utf-8"))
    if len(rows) != expected:
        raise PrepareError(f"{spec.key}: expected {expected} rows in {filename}, found {len(rows)}")
    return rows


def _extract_archive(ctx: PrepareContext, archive_name: str, wanted: dict[str, Path], desc: str) -> None:
    """Download the whole archive (resumable) and extract the wanted members."""
    import zipfile

    pending = {name: path for name, path in wanted.items() if not path.exists() or path.stat().st_size == 0}
    ctx.log(f"  [zip] {desc}: {len(wanted) - len(pending)} present, {len(pending)} to extract")
    if not pending:
        return
    archive_path = hf_download(ctx, REPO_ID, archive_name)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            destination = pending.get(Path(info.filename).name)
            if destination is None or info.is_dir():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_name(destination.name + ".tmp")
            with archive.open(info) as src, tmp.open("wb") as dst:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    dst.write(block)
            tmp.replace(destination)
    missing = [name for name, path in pending.items() if not path.exists()]
    if missing:
        raise PrepareError(f"{desc}: {len(missing)} image(s) missing from {archive_name}, e.g. {missing[:3]}")
    ctx.discard_raw(archive_path)


def prepare_cfpo(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    _, archive_name, _, image_source = SPLITS[spec.key]
    rows = _annotations(ctx, spec)
    image_dir = spec.target_dir(ctx.data_root) / "images"
    wanted = {name: image_dir / name for row in rows for name in row["images"]}
    if image_source == "coco":
        items = [DownloadItem(urls=(f"{COCO_URL}/{name}",), dest=path) for name, path in wanted.items()]
        failures = http_download_many(ctx, items, desc=f"{spec.key} COCO val2014 images")
        if failures:
            ctx.log(f"  [http] {len(failures)} image(s) failed on cocodataset.org; reading them from {archive_name}")
            retry = {path.name: path for path in failures}
            extract_remote_zip_members(ctx, REPO_ID, archive_name, retry, desc=f"{spec.key} fallback")
    else:
        _extract_archive(ctx, archive_name, wanted, f"{spec.key} images")
    missing = [name for name, path in wanted.items() if not path.is_file()]
    if missing:
        raise PrepareError(f"{spec.key}: {len(missing)} image(s) missing, e.g. {missing[:3]}; re-run to resume")
    return {"repo_id": REPO_ID, "rows": len(rows), "images": len(wanted)}


SOURCES = [
    BenchmarkSource(
        key="cvqa_real",
        target="cvqa_real",
        outputs=("cvqa_real/test.json",),
        source=f"{REPO_ID} (C-VQA-Real_test.json) + COCO val2014 images",
        approx_size="0.5 GB",
        prepare=prepare_cfpo,
    ),
    BenchmarkSource(
        key="mars_bench",
        target="mars_bench",
        outputs=("mars_bench/test.json",),
        source=f"{REPO_ID} (MARS_Bench_test.json) + COCO val2014 images",
        approx_size="0.2 GB",
        prepare=prepare_cfpo,
    ),
    BenchmarkSource(
        key="textvqa",
        target="textvqa",
        outputs=("textvqa/test.json",),
        source=f"{REPO_ID} (Textvqa_test.json = TextVQA val) + Textvqa_images.zip",
        approx_size="3.1 GB (+3.1 GB temporary zip)",
        prepare=prepare_cfpo,
        disk_gb=6.5,
    ),
]
