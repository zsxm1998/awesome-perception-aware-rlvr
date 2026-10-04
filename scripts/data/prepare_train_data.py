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
"""Download training/validation data into ``<repo>/data`` as one parquet file per split.

Every dataset is written to ``$DATA_ROOT/<name>/<split>.parquet`` with the columns the
training scripts expect (``problem``, ``answer``, ``images`` and, for some datasets,
extra columns). Run ``python scripts/data/prepare_train_data.py --list`` to see all names.
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from converters import CONVERTERS  # noqa: E402


@dataclass
class TrainDataset:
    name: str
    repo_id: str
    # split name in the output dir -> glob patterns of the source files in the HF repo
    splits: dict[str, list[str]]
    description: str
    used_by: list[str]
    converter: Optional[str] = None  # name in converters.CONVERTERS; None = plain shard merge
    extra_patterns: list[str] = field(default_factory=list)
    # columns the current converter writes and the scripts read; existing output without them was written by an
    # older version and is prepared again
    required_columns: tuple[str, ...] = ()


DATASETS: dict[str, TrainDataset] = {
    "virl39k": TrainDataset(
        name="virl39k",
        repo_id="PAPOGalaxy/PAPO_ViRL39K_train",
        splits={"train": ["data/train-*.parquet"]},
        description="ViRL39K (38,870 multimodal reasoning problems) as preprocessed by PAPO",
        used_by=["papo", "vppo", "dvrp", "pgpo", "cgpo", "comparison", "opd_comparison", "va_opd", "vcsd"],
    ),
    "mmk12": TrainDataset(
        name="mmk12",
        repo_id="PAPOGalaxy/PAPO_MMK12_test",
        splits={"test": ["data/train-*.parquet"]},
        description="MMK12 test (2,000 problems), the validation set used by PAPO/VPPO",
        used_by=[
            "papo",
            "vppo",
            "dvrp",
            "pgpo",
            "cfpo",
            "cgpo",
            "comparison",
            "opd_comparison",
            "va_opd",
            "vcsd",
            "vision_opd",
        ],
    ),
    "geometry3k": TrainDataset(
        name="geometry3k",
        repo_id="hiyouga/geometry3k",
        splits={
            "train": ["data/train-*.parquet"],
            "validation": ["data/validation-*.parquet"],
            "test": ["data/test-*.parquet"],
        },
        description="Geometry3K (2,101 train / 300 val / 601 test)",
        used_by=["tor", "pepo", "vepo", "va_opd"],  # VEPO: through the vepo dataset
    ),
    "cfpo": TrainDataset(
        name="cfpo",
        repo_id="RavenInJuly/CFPO_Datasets",
        splits={"train": []},
        description="CFPO's ViRL39K (38,870 problems, CFPO's problem text) with the ViRL39K images",
        used_by=["cfpo"],
        converter="cfpo",
    ),
    "vepo": TrainDataset(
        name="vepo",
        repo_id="xyliu6/k12-freeform",
        splits={"train": [], "test": []},
        description="VEPO's 4.2K: Geometry3K train (2,101) + k12-freeform mini_train (2,100); k12-freeform test (808)",
        used_by=["vepo"],
        converter="vepo",
    ),
    "grit": TrainDataset(
        name="grit",
        repo_id="yfan1997/GRIT_data",
        splits={"train": [], "test": []},
        description="GRIT training data (VSR + TallyQA subsets with grounded answers)",
        used_by=["grit"],
        converter="grit",
    ),
    "vision_opd": TrainDataset(
        name="vision_opd",
        repo_id="yuanqianhao/Vision-OPD-6K",
        splits={"train": []},
        description="Vision-OPD-6K (6,241 SA-1B images with a red box, the teacher's zoomed crops, 4-option questions)",
        used_by=["vision_opd"],
        converter="vision_opd",
        required_columns=("teacher_images",),
    ),
    "vision_sr1": TrainDataset(
        name="vision_sr1",
        repo_id="tomyoon2/OPD_train",
        splits={"train": ["data/train-*.parquet"]},
        description="Vision-SR1-47K as split by the VGS authors for training (45,246 problems)",
        used_by=["vgs"],
    ),
    "vision_sr1_val": TrainDataset(
        name="vision_sr1_val",
        repo_id="tomyoon2/OPD_val",
        splits={"val": ["data/train-*.parquet"]},
        description="Vision-SR1-47K as split by the VGS authors for validation (2,382 problems)",
        used_by=["vgs"],
    ),
    "deepeyes": TrainDataset(
        name="deepeyes",
        repo_id="ChenShawn/DeepEyes-Datasets-47k",
        splits={"train": [], "val": []},
        description="DeepEyes-Datasets-47k (fine-grained perception, chart, and reasoning subsets)",
        used_by=["deepeyes"],
        converter="deepeyes",
        required_columns=("env_name", "official_system_prompt", "official_prompt", "row_system_prompt"),
    ),
}

METHOD_GROUPS: dict[str, list[str]] = {
    "papo": ["virl39k", "mmk12"],
    "vppo": ["virl39k", "mmk12"],
    "dvrp": ["virl39k", "mmk12"],
    "pgpo": ["virl39k", "mmk12"],
    "cfpo": ["cfpo", "mmk12"],
    "cgpo": ["virl39k", "mmk12"],
    "comparison": ["virl39k", "mmk12"],
    "tor": ["geometry3k"],
    "pepo": ["geometry3k"],
    "vepo": ["geometry3k", "vepo"],
    "grit": ["grit"],
    "deepeyes": ["deepeyes"],
    "opd_comparison": ["virl39k", "mmk12"],
    "va_opd": ["geometry3k", "virl39k"],
    "vgs": ["vision_sr1", "vision_sr1_val"],
    "vcsd": ["virl39k", "mmk12"],
    "vision_opd": ["vision_opd", "mmk12"],
}


def _snapshot(repo_id: str, local_dir: Path, patterns: Optional[list[str]], retries: int = 4) -> Path:
    """Download (a subset of) a dataset repo, retrying with less concurrency on network errors.

    Mirrors such as https://hf-mirror.com sometimes reject many parallel requests; finished
    files are kept between attempts, so a retry only fetches what is still missing.
    """
    import time

    from huggingface_hub import snapshot_download

    workers = int(os.environ.get("HF_DOWNLOAD_WORKERS", "8"))
    for attempt in range(1, retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                allow_patterns=patterns,
                max_workers=workers,
            )
            return local_dir
        except Exception as exc:  # network errors surface as several different exception types
            if attempt == retries:
                raise
            workers = max(1, workers // 2)
            print(f"[retry {attempt}/{retries - 1}] {type(exc).__name__}: {exc}; retrying with {workers} worker(s)")
            time.sleep(5 * attempt)
    return local_dir


def _merge_parquet(files: list[Path], output: Path) -> int:
    import pyarrow.parquet as pq

    if not files:
        raise FileNotFoundError("no source parquet files matched")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(".parquet.tmp")
    writer = None
    rows = 0
    try:
        for path in sorted(files):
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp_output), table.schema)
            writer.write_table(table.cast(writer.schema) if table.schema != writer.schema else table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp_output, output)
    return rows


def _missing_columns(path: Path, columns: tuple[str, ...]) -> list[str]:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    return [column for column in columns if column not in names]


def prepare(dataset: TrainDataset, data_root: Path, keep_raw: bool, force: bool) -> None:
    out_dir = data_root / dataset.name
    expected = [out_dir / f"{split}.parquet" for split in dataset.splits]
    if not force and all(path.exists() for path in expected):
        stale = {path.name: _missing_columns(path, dataset.required_columns) for path in expected}
        stale = {name: columns for name, columns in stale.items() if columns}
        if not stale:
            print(f"[skip] {dataset.name}: already prepared at {out_dir}")
            return
        print(f"[rebuild] {dataset.name}: {stale} lack columns of the current version; preparing again")

    raw_dir = data_root / ".raw" / dataset.repo_id.replace("/", "__")
    print(f"[download] {dataset.repo_id} -> {raw_dir}")
    if dataset.converter is None:
        patterns = sorted({p for globs in dataset.splits.values() for p in globs} | set(dataset.extra_patterns))
        _snapshot(dataset.repo_id, raw_dir, patterns)
        summary = {}
        for split, globs in dataset.splits.items():
            files = [path for pattern in globs for path in raw_dir.glob(pattern)]
            rows = _merge_parquet(files, out_dir / f"{split}.parquet")
            summary[split] = rows
            print(f"[write] {out_dir / f'{split}.parquet'} ({rows} rows)")
    else:
        converter: Callable = CONVERTERS[dataset.converter]
        summary = converter(
            repo_id=dataset.repo_id,
            raw_dir=raw_dir,
            out_dir=out_dir,
            snapshot=lambda patterns: _snapshot(dataset.repo_id, raw_dir, patterns),
        )
    (out_dir / "SOURCE.json").write_text(
        json.dumps({"repo_id": dataset.repo_id, "rows": summary, "description": dataset.description}, indent=2)
    )
    if not keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)


def _bypass_proxy_for_mirror() -> None:
    """hf-mirror.com redirects to huggingface.co when it is reached through an HTTP proxy, which
    makes downloads fail; talk to the mirror directly unless HF_MIRROR_BYPASS_PROXY=0."""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if "hf-mirror" not in endpoint or os.environ.get("HF_MIRROR_BYPASS_PROXY", "1") == "0":
        return
    hosts = ["hf-mirror.com", ".hf-mirror.com", ".hf.co"]
    for key in ("NO_PROXY", "no_proxy"):
        current = [item for item in os.environ.get(key, "").split(",") if item]
        os.environ[key] = ",".join(current + [host for host in hosts if host not in current])


def main() -> None:
    _bypass_proxy_for_mirror()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="dataset names, method names (e.g. papo), or 'all'")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(ROOT_DIR / "data")))
    parser.add_argument("--keep-raw", action="store_true", help="keep the raw HF snapshot under <data-root>/.raw")
    parser.add_argument("--force", action="store_true", help="re-download even if the output exists")
    parser.add_argument("--list", action="store_true", help="list datasets and method groups")
    args = parser.parse_args()

    if args.list or not args.targets:
        print("Datasets:")
        for ds in DATASETS.values():
            print(f"  {ds.name:<12} {ds.repo_id:<36} used by: {', '.join(ds.used_by)}")
            print(f"  {'':<12} {ds.description}")
        print("Method groups:")
        for method, names in METHOD_GROUPS.items():
            print(f"  {method:<12} -> {', '.join(names)}")
        return

    names: list[str] = []
    for target in args.targets:
        if target == "all":
            names.extend(DATASETS)
        elif target in METHOD_GROUPS:
            names.extend(METHOD_GROUPS[target])
        elif target in DATASETS:
            names.append(target)
        else:
            parser.error(f"unknown target {target!r}; run with --list")
    data_root = Path(args.data_root).resolve()
    for name in dict.fromkeys(names):
        prepare(DATASETS[name], data_root, keep_raw=args.keep_raw, force=args.force)
    print(f"Done. Data root: {data_root}")


if __name__ == "__main__":
    main()
