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
"""GRIT evaluation sets (HF ``yfan1997/GRIT_data``, official release of the GRIT paper).

GRIT_data ships annotation files only. Images are fetched individually so that only the
referenced ones are downloaded instead of the full COCO / Visual Genome / GQA archives:

- VSR (288 rows, 247 COCO 2017 images): ``images.cocodataset.org/{train,val}2017``
- TallyQA (491 rows, Visual Genome images): ``cs.stanford.edu/people/rak248/VG_100K{,_2}``
- GQA (509 rows, Visual Genome images, same files as the GQA ``images.zip``)
- OVDEval position (2146 rows): ``position.tar.gz`` (216 MB) from HF ``omlab/OVDEval``

The relabeled TallyQA set (HF ``zsxm1998/GRIT-TallyQA-Relabeled``: the same 491 questions with one box
per counted instance and corrected answers) ships its 465 Visual Genome images with the annotations.

Layout::

    <data_root>/grit_vsr/vsr_val.jsonl            + images/<coco file name>
    <data_root>/grit_tallyqa/tallyqa_val.jsonl    + images/VG_100K{,_2}/<id>.jpg
    <data_root>/grit_gqa/gqa_val.jsonl            + images/<id>.jpg
    <data_root>/ovdeval_position/ovd_position_val.jsonl + images/<file>.jpg
    <data_root>/tallyqa_relabeled/tallyqa_val.jsonl + images/VG_100K{,_2}/<id>.jpg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BenchmarkSource
from .common import (
    DownloadItem,
    PrepareContext,
    PrepareError,
    copy_file,
    extract_tar,
    hf_download,
    hf_snapshot,
    http_download_many,
    move_file,
    place_referenced_files,
    read_jsonl,
)


REPO_ID = "yfan1997/GRIT_data"
OVDEVAL_REPO_ID = "omlab/OVDEval"
RELABELED_TALLYQA_REPO_ID = "zsxm1998/GRIT-TallyQA-Relabeled"
COCO_URL = "http://images.cocodataset.org"
VG_URL = "https://cs.stanford.edu/people/rak248"


def _annotations(ctx: PrepareContext, spec: BenchmarkSource) -> tuple[Path, list[dict[str, Any]]]:
    filename = spec.options["annotation"]
    raw = hf_download(ctx, REPO_ID, filename)
    output = copy_file(raw, spec.target_dir(ctx.data_root) / filename)
    ctx.discard_raw(raw)
    rows = read_jsonl(output)
    expected = spec.options["expected_rows"]
    if len(rows) != expected:
        raise PrepareError(f"{spec.key}: expected {expected} rows in {filename}, found {len(rows)}")
    return output, rows


def _image_items(spec: BenchmarkSource, rows: list[dict[str, Any]], image_root: Path) -> list[DownloadItem]:
    items: dict[str, DownloadItem] = {}
    for row in rows:
        image = str(row["image"])
        if image in items:
            continue
        name = Path(image).name
        if spec.key == "grit_vsr":
            urls = (f"{COCO_URL}/train2017/{name}", f"{COCO_URL}/val2017/{name}")
        elif spec.key == "grit_tallyqa":
            urls = (
                (f"{VG_URL}/{image}",)
                if image.startswith("VG_100K")
                else (f"{VG_URL}/VG_100K/{name}", f"{VG_URL}/VG_100K_2/{name}")
            )
        elif spec.key == "grit_gqa":
            urls = (f"{VG_URL}/VG_100K/{name}", f"{VG_URL}/VG_100K_2/{name}")
        else:  # pragma: no cover - guarded by SOURCES
            raise PrepareError(f"no image source for {spec.key}")
        items[image] = DownloadItem(urls=urls, dest=image_root / image)
    return list(items.values())


def prepare_grit_http(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    _, rows = _annotations(ctx, spec)
    image_root = spec.target_dir(ctx.data_root) / spec.options["image_subdir"]
    items = _image_items(spec, rows, image_root)
    failures = http_download_many(ctx, items, desc=f"{spec.key} images")
    if failures:
        examples = "; ".join(f"{path.name}: {error}" for path, error in list(failures.items())[:3])
        raise PrepareError(
            f"{spec.key}: {len(failures)}/{len(items)} images could not be downloaded ({examples}). "
            "Re-run the command to retry; set https_proxy/http_proxy if these hosts are blocked."
        )
    return {"repo_id": REPO_ID, "annotation": spec.options["annotation"], "rows": len(rows), "images": len(items)}


def prepare_ovdeval_position(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    _, rows = _annotations(ctx, spec)
    target = spec.target_dir(ctx.data_root)
    archive = hf_download(ctx, OVDEVAL_REPO_ID, "position.tar.gz")
    extract_tar(archive, target / "images", log=ctx.log)
    place_referenced_files(target / "images", [str(row["image"]) for row in rows])
    ctx.discard_raw(archive)
    return {
        "repo_id": REPO_ID,
        "image_repo_id": OVDEVAL_REPO_ID,
        "annotation": spec.options["annotation"],
        "rows": len(rows),
        "images": len({row["image"] for row in rows}),
    }


def prepare_tallyqa_relabeled(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    target = spec.target_dir(ctx.data_root)
    annotation = spec.options["annotation"]
    raw = hf_snapshot(ctx, RELABELED_TALLYQA_REPO_ID, [annotation, "images/*"])
    rows = read_jsonl(raw / annotation)
    if len(rows) != spec.options["expected_rows"]:
        raise PrepareError(
            f"{spec.key}: expected {spec.options['expected_rows']} rows in {annotation}, found {len(rows)}"
        )
    images = sorted({str(row["image"]) for row in rows})
    for image in images:
        if (raw / "images" / image).is_file():
            move_file(raw / "images" / image, target / "images" / image)
    missing = [image for image in images if not (target / "images" / image).is_file()]
    if missing:
        raise PrepareError(f"{spec.key}: {len(missing)} images missing, e.g. {missing[:3]}")
    move_file(raw / annotation, target / annotation)
    ctx.discard_raw(raw)
    return {"repo_id": RELABELED_TALLYQA_REPO_ID, "annotation": annotation, "rows": len(rows), "images": len(images)}


SOURCES = [
    BenchmarkSource(
        key="grit_vsr",
        target="grit_vsr",
        outputs=("grit_vsr/vsr_val.jsonl",),
        source=f"{REPO_ID} vsr_val.jsonl + COCO 2017 images",
        approx_size="45 MB",
        prepare=prepare_grit_http,
        options={"annotation": "vsr_val.jsonl", "expected_rows": 288, "image_subdir": "images"},
    ),
    BenchmarkSource(
        key="grit_tallyqa",
        target="grit_tallyqa",
        outputs=("grit_tallyqa/tallyqa_val.jsonl",),
        source=f"{REPO_ID} tallyqa_val.jsonl + Visual Genome images",
        approx_size="50 MB",
        prepare=prepare_grit_http,
        options={"annotation": "tallyqa_val.jsonl", "expected_rows": 491, "image_subdir": "images"},
    ),
    BenchmarkSource(
        key="grit_gqa",
        target="grit_gqa",
        outputs=("grit_gqa/gqa_val.jsonl",),
        source=f"{REPO_ID} gqa_val.jsonl + Visual Genome (GQA) images",
        approx_size="55 MB",
        prepare=prepare_grit_http,
        # GRIT stores the GQA image as "images/<id>.jpg" relative to the benchmark dir.
        options={"annotation": "gqa_val.jsonl", "expected_rows": 509, "image_subdir": "."},
    ),
    BenchmarkSource(
        key="ovdeval_position",
        target="ovdeval_position",
        outputs=("ovdeval_position/ovd_position_val.jsonl",),
        source=f"{REPO_ID} ovd_position_val.jsonl + {OVDEVAL_REPO_ID} position.tar.gz",
        approx_size="215 MB",
        prepare=prepare_ovdeval_position,
        options={"annotation": "ovd_position_val.jsonl", "expected_rows": 2146},
    ),
    BenchmarkSource(
        key="tallyqa_relabeled",
        target="tallyqa_relabeled",
        outputs=("tallyqa_relabeled/tallyqa_val.jsonl",),
        source=f"{RELABELED_TALLYQA_REPO_ID} (relabeled GRIT TallyQA + its Visual Genome images)",
        approx_size="35 MB",
        prepare=prepare_tallyqa_relabeled,
        options={"annotation": "tallyqa_val.jsonl", "expected_rows": 491},
    ),
]
