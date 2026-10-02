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
"""PAPO-Eval (HF ``PAPO-Galaxy/PAPO_eval``): the 9 benchmark columns of PAPO Table 1.

Each split ships as ``data/<split>-00000-of-00001.parquet`` (ShareGPT ``messages`` +
relative ``images`` paths such as ``./data/images/<folder>/<file>.png``) plus an image
zip at the repository root. Layout produced::

    <data_root>/<key>/test.parquet
    <data_root>/<key>/images/<folder>/<file>.png
"""

from __future__ import annotations

from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, copy_file, extract_zip, hf_download, place_referenced_files


REPO_ID = "PAPO-Galaxy/PAPO_eval"

# key -> (split name, image zip, approximate size, expected rows)
SPLITS: dict[str, tuple[str, str, str, int]] = {
    "geo3k": ("hiyouga_geometry3k", "hiyouga_geometry3k_images.zip", "12 MB", 601),
    "mathvista": ("AI4Math_MathVista", "AI4Math_MathVista_images.zip", "250 MB", 1000),
    "wemath": ("We_Math", "We-Math_We-Math_images.zip", "40 MB", 1740),
    "mmk12": ("PAPO_MMK12", "PAPO_MMK12_test_images.zip", "165 MB", 2000),
    "mathverse": ("AI4Math_MathVerse", "AI4Math_MathVerse_images.zip", "170 MB", 2180),
    "mathverse_v": (
        "AI4Math_MathVerse_vision_dependent",
        "AI4Math_MathVerse_vision_dependent_images.zip",
        "76 MB",
        1308,
    ),
    "logicvista": ("lscpku_LogicVista", "lscpku_LogicVista_images.zip", "23 MB", 447),
    "clevr_count": ("BUAADreamer_clevr_count_70k", "BUAADreamer_clevr_count_70k_images.zip", "63 MB", 200),
    "mmmu_pro": ("MMMU_MMMU_Pro", "MMMU_MMMU_Pro_images.zip", "1.45 GB", 1730),
}


def sharegpt_image_relpath(value: str) -> str:
    """``./data/images/<folder>/<file>`` (LLaMA-Factory layout) -> ``images/<folder>/<file>``."""
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("data/images/"):
        text = text[len("data/") :]
    return text


def prepare_papo_split(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    import pandas as pd

    split, zip_name, _, expected_rows = SPLITS[spec.key]
    target = spec.target_dir(ctx.data_root)
    parquet = hf_download(ctx, REPO_ID, f"data/{split}-00000-of-00001.parquet")
    output = copy_file(parquet, target / "test.parquet")
    frame = pd.read_parquet(output)
    if len(frame) != expected_rows:
        raise PrepareError(f"{spec.key}: expected {expected_rows} rows in {split}, found {len(frame)}")
    relpaths = [sharegpt_image_relpath(item) for images in frame["images"] for item in list(images)]

    archive = hf_download(ctx, REPO_ID, zip_name)
    extract_zip(archive, target / "images", log=ctx.log)
    place_referenced_files(target, relpaths, search_root=target / "images")
    ctx.discard_raw(archive)
    ctx.discard_raw(parquet)
    return {"repo_id": REPO_ID, "split": split, "rows": len(frame), "images": len(set(relpaths))}


SOURCES = [
    BenchmarkSource(
        key=key,
        target=key,
        outputs=(f"{key}/test.parquet",),
        source=f"{REPO_ID} (split {split})",
        approx_size=size,
        prepare=prepare_papo_split,
    )
    for key, (split, _, size, _) in SPLITS.items()
]
