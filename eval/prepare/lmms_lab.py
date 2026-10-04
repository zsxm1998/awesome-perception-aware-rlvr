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
"""lmms-eval formatted perception benchmarks (SEED-Bench: see seed_bench.py) (HF ``lmms-lab/*``; images embedded in parquet).

The Hub currently redirects ``lmms-lab/<name>`` to ``lmms-lab-encoder/<name>``; both ids
are tried. Layouts::

    <data_root>/pope/{random,popular,adversarial}.parquet
    <data_root>/mme/test-0000X-of-00004.parquet
    <data_root>/hallusionbench/{image,non_image}.parquet
    <data_root>/gqa/testdev_balanced_{instructions,images}.parquet
    <data_root>/mm_vet/test.parquet
    <data_root>/ai2d/test-0000X-of-00002.parquet
    <data_root>/mmmu_val/validation.parquet
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError, hf_download, hf_snapshot, move_file


def _repos(name: str) -> list[str]:
    return [f"lmms-lab/{name}", f"lmms-lab-encoder/{name}"]


def _parquet_rows(paths: list[Path]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _download_renamed(ctx: PrepareContext, spec: BenchmarkSource, files: dict[str, str]) -> list[Path]:
    """Download ``{repo path: output name}`` into the benchmark directory."""
    target = spec.target_dir(ctx.data_root)
    outputs = []
    for repo_path, output_name in files.items():
        raw = hf_download(ctx, spec.options["repos"], repo_path)
        outputs.append(move_file(raw, target / output_name))
    return outputs


def _download_glob(ctx: PrepareContext, spec: BenchmarkSource, pattern: str) -> list[Path]:
    target = spec.target_dir(ctx.data_root)
    raw_dir = hf_snapshot(ctx, spec.options["repos"], [pattern])
    matches = sorted(
        path for path in raw_dir.rglob("*.parquet") if fnmatch.fnmatch(path.relative_to(raw_dir).as_posix(), pattern)
    )
    if not matches:
        raise PrepareError(f"{spec.key}: no files matched {pattern}")
    return [move_file(path, target / path.name) for path in matches]


def prepare_renamed(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    outputs = _download_renamed(ctx, spec, spec.options["files"])
    counted = [path for path in outputs if path.name in spec.options.get("rows_from", [path.name for path in outputs])]
    rows = _parquet_rows(counted)
    expected = spec.options.get("expected_rows")
    if expected is not None and rows != expected:
        raise PrepareError(f"{spec.key}: expected {expected} rows, found {rows}")
    return {"repo_id": spec.options["repos"][0], "files": [path.name for path in outputs], "rows": rows}


def prepare_glob(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    outputs = _download_glob(ctx, spec, spec.options["pattern"])
    rows = _parquet_rows(outputs)
    expected = spec.options.get("expected_rows")
    if expected is not None and rows != expected:
        raise PrepareError(f"{spec.key}: expected {expected} rows, found {rows}")
    return {"repo_id": spec.options["repos"][0], "files": [path.name for path in outputs], "rows": rows}


SOURCES = [
    BenchmarkSource(
        key="pope",
        target="pope",
        outputs=("pope/random.parquet", "pope/popular.parquet", "pope/adversarial.parquet"),
        source="lmms-lab/POPE (config Full)",
        approx_size="255 MB",
        prepare=prepare_renamed,
        options={
            "repos": _repos("POPE"),
            "files": {
                "Full/random-00000-of-00001.parquet": "random.parquet",
                "Full/popular-00000-of-00001.parquet": "popular.parquet",
                "Full/adversarial-00000-of-00001.parquet": "adversarial.parquet",
            },
            "expected_rows": 9000,
        },
    ),
    BenchmarkSource(
        key="mme",
        target="mme",
        outputs=("mme",),
        source="lmms-lab/MME",
        approx_size="865 MB",
        prepare=prepare_glob,
        options={"repos": _repos("MME"), "pattern": "data/test-*.parquet", "expected_rows": 2374},
    ),
    BenchmarkSource(
        key="hallusionbench",
        target="hallusionbench",
        outputs=("hallusionbench/image.parquet",),
        source="lmms-lab/HallusionBench (split image)",
        approx_size="147 MB",
        prepare=prepare_renamed,
        options={
            "repos": _repos("HallusionBench"),
            "files": {
                "data/image-00000-of-00001.parquet": "image.parquet",
                "data/non_image-00000-of-00001.parquet": "non_image.parquet",
            },
            # Only the 951 image questions are evaluated (lmms-eval hallusion_bench_image).
            "rows_from": ["image.parquet"],
            "expected_rows": 951,
        },
    ),
    BenchmarkSource(
        key="gqa",
        target="gqa",
        outputs=("gqa/testdev_balanced_instructions.parquet", "gqa/testdev_balanced_images.parquet"),
        source="lmms-lab/GQA (testdev balanced)",
        approx_size="68 MB",
        prepare=prepare_renamed,
        options={
            "repos": _repos("GQA"),
            "files": {
                "testdev_balanced_instructions/testdev-00000-of-00001.parquet": "testdev_balanced_instructions.parquet",
                "testdev_balanced_images/testdev-00000-of-00001.parquet": "testdev_balanced_images.parquet",
            },
            "rows_from": ["testdev_balanced_instructions.parquet"],
            "expected_rows": 12578,
        },
    ),
    BenchmarkSource(
        key="mm_vet",
        target="mm_vet",
        outputs=("mm_vet/test.parquet",),
        source="lmms-lab/MMVet",
        approx_size="67 MB",
        prepare=prepare_renamed,
        options={
            "repos": _repos("MMVet"),
            "files": {"data/test-00000-of-00001.parquet": "test.parquet"},
            "expected_rows": 218,
        },
    ),
    BenchmarkSource(
        key="ai2d",
        target="ai2d",
        outputs=("ai2d",),
        source="lmms-lab/ai2d (test)",
        approx_size="140 MB",
        prepare=prepare_glob,
        options={"repos": _repos("ai2d"), "pattern": "data/test-*.parquet", "expected_rows": 3088},
    ),
    BenchmarkSource(
        key="mmmu_val",
        target="mmmu_val",
        outputs=("mmmu_val/validation.parquet",),
        source="lmms-lab/MMMU (validation)",
        approx_size="340 MB",
        prepare=prepare_renamed,
        options={
            "repos": _repos("MMMU"),
            "files": {"data/validation-00000-of-00001.parquet": "validation.parquet"},
            "expected_rows": 900,
        },
    ),
]
