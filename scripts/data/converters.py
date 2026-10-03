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
"""Dataset-specific converters from an upstream HF layout to EasyR1 parquet files.

A converter receives the repo id, a raw download directory, the output directory and a
``snapshot(patterns)`` callable, and returns ``{split: num_rows}``.
"""

from pathlib import Path
from typing import Callable


Converter = Callable[..., dict[str, int]]
CONVERTERS: dict[str, Converter] = {}


def register(name: str):
    def wrapper(fn: Converter) -> Converter:
        CONVERTERS[name] = fn
        return fn

    return wrapper


# ---------------------------------------------------------------------------
# GRIT (yfan1997/GRIT_data): jsonl annotations without images. VSR images come from
# COCO 2017 and TallyQA images from Visual Genome; only the referenced files are fetched.
# ---------------------------------------------------------------------------

_COCO_URLS = (
    "http://images.cocodataset.org/train2017/{name}",
    "http://images.cocodataset.org/val2017/{name}",
)
_VG_URL = "https://cs.stanford.edu/people/rak248/{name}"


def _fetch_image(urls: list[str], target: Path, retries: int = 3) -> bytes:
    import time
    import urllib.request

    if target.exists():
        return target.read_bytes()
    last_error: Exception | None = None
    for url in urls:
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                return payload
            except Exception as exc:  # 404 on the first COCO split is expected
                last_error = exc
                if getattr(exc, "code", None) == 404:
                    break
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {target.name} from {urls}: {last_error}")


def _grit_image_urls(row: dict) -> list[str]:
    name = row["image"]
    if row.get("dataset") == "tallyqa" or name.startswith("VG_100K"):
        return [_VG_URL.format(name=name)]
    return [url.format(name=name) for url in _COCO_URLS]


@register("grit")
def convert_grit(repo_id: str, raw_dir: Path, out_dir: Path, snapshot) -> dict[str, int]:
    import json
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    files = {
        "train": ["vsr_cot_train_10.jsonl", "tallyqa_train_10.jsonl"],
        "test": ["vsr_val.jsonl", "tallyqa_val.jsonl"],
    }
    snapshot([name for names in files.values() for name in names])
    image_dir = raw_dir / "images"
    summary: dict[str, int] = {}
    for split, names in files.items():
        rows = []
        for name in names:
            with open(raw_dir / name, encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())

        def load(row: dict) -> bytes:
            return _fetch_image(_grit_image_urls(row), image_dir / row["image"].replace("/", "__"))

        with ThreadPoolExecutor(max_workers=8) as pool:
            payloads = list(pool.map(load, rows))
        table = pa.table(
            {
                "problem": ["<image>" + row["question"] for row in rows],
                "answer": [str(row["answer"]) for row in rows],
                "images": [[{"bytes": payload, "path": row["image"]}] for row, payload in zip(rows, payloads)],
                "data_source": [row.get("dataset", "") for row in rows],
                "bboxs": [json.dumps(row.get("bboxs", [])) for row in rows],
                "width": [int(row["width"]) for row in rows],
                "height": [int(row["height"]) for row in rows],
            }
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_dir / f"{split}.parquet")
        summary[split] = len(rows)
        print(f"[write] {out_dir / f'{split}.parquet'} ({len(rows)} rows)")
    return summary


# ---------------------------------------------------------------------------
# DeepEyes (ChenShawn/DeepEyes-Datasets-47k): verl-format parquet files whose prompts embed
# the original tool system prompt. `problem` keeps the user question (with its <image>
# placeholders and, for charts, the answer options) for the native prompt style, where
# examples/system_prompt/deepeyes*.txt provide the tool protocol through the model's chat
# template; `official_system_prompt` / `official_prompt` keep the official messages verbatim
# for the official style. Rows with an empty `env_name` (ThinkLite) get no tool in the
# official environment; they keep the official prompts in both styles (`problem` is the
# official user text and `row_system_prompt` the official system prompt, empty for tool rows).
# ---------------------------------------------------------------------------

_DEEPEYES_FILES = (
    "data_0.1.2_visual_toolbox_v2.parquet",  # V*-derived fine-grained perception (22,362)
    "data_v0.8_visual_toolbox_v2.parquet",  # ArxivQA charts (13,659)
    "data_thinklite_reasoning_acc.parquet",  # ThinkLite-VL reasoning (11,031)
)
_DEEPEYES_TOOL_SUFFIX = "\nThink first, call **image_zoom_in_tool** if needed"


def _deepeyes_message(prompt: list[dict], role: str) -> str:
    return next((message["content"] for message in prompt if message["role"] == role), "")


def _deepeyes_question(prompt: list[dict]) -> str:
    user = _deepeyes_message(prompt, "user")
    if _DEEPEYES_TOOL_SUFFIX in user:
        return user.split(_DEEPEYES_TOOL_SUFFIX, 1)[0].strip()
    return user.strip()


@register("deepeyes")
def convert_deepeyes(repo_id: str, raw_dir: Path, out_dir: Path, snapshot) -> dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    snapshot(list(_DEEPEYES_FILES))
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "train.parquet"
    tmp_output = output.with_suffix(".parquet.tmp")
    schema = pa.schema(
        [
            ("problem", pa.string()),
            ("answer", pa.string()),
            ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
            ("data_source", pa.string()),
            ("question", pa.string()),
            ("env_name", pa.string()),
            ("official_system_prompt", pa.string()),
            ("official_prompt", pa.string()),
            ("row_system_prompt", pa.string()),
        ]
    )
    rows = 0
    with pq.ParquetWriter(str(tmp_output), schema) as writer:
        for name in _DEEPEYES_FILES:
            parquet = pq.ParquetFile(raw_dir / name)
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(
                    group, columns=["prompt", "images", "reward_model", "extra_info", "data_source", "env_name"]
                )
                prompts = table.column("prompt").to_pylist()
                rewards = table.column("reward_model").to_pylist()
                extras = table.column("extra_info").to_pylist()
                env_names = [str(env_name or "") for env_name in table.column("env_name").to_pylist()]
                systems = [_deepeyes_message(prompt, "system") for prompt in prompts]
                users = [_deepeyes_message(prompt, "user") for prompt in prompts]
                batch = pa.table(
                    {
                        "problem": [
                            _deepeyes_question(prompt) if env_name else user.strip()
                            for prompt, user, env_name in zip(prompts, users, env_names)
                        ],
                        "answer": [str(reward["ground_truth"]) for reward in rewards],
                        "images": table.column("images"),
                        "data_source": table.column("data_source"),
                        "question": [str(extra.get("question") or "") for extra in extras],
                        "env_name": env_names,
                        "official_system_prompt": systems,
                        "official_prompt": users,
                        "row_system_prompt": [
                            "" if env_name else system for system, env_name in zip(systems, env_names)
                        ],
                    },
                    schema=schema,
                )
                writer.write_table(batch)
                rows += batch.num_rows
            print(f"[convert] {name}: done ({rows} rows so far)")
    tmp_output.replace(output)
    print(f"[write] {output} ({rows} rows)")
    val_rows = _write_deepeyes_monitor_split(output, out_dir / "val.parquet")
    return {"train": rows, "val": val_rows}


def _write_deepeyes_monitor_split(train_path: Path, val_path: Path, per_source: int = 100, seed: int = 0) -> int:
    """Small monitoring split (100 rows per data source, drawn from the training file).

    The official release has no held-out validation set (its script validates on a training
    file); use the evaluation suite for real numbers.
    """
    import random

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(train_path)
    sources = pq.read_table(train_path, columns=["data_source"]).column("data_source").to_pylist()
    rng = random.Random(seed)
    selected: set[int] = set()
    for source in sorted(set(sources)):
        source_indices = [index for index, value in enumerate(sources) if value == source]
        selected.update(rng.sample(source_indices, min(per_source, len(source_indices))))
    pieces, offset = [], 0
    for group in range(parquet.num_row_groups):  # row-group-wise to stay below 2 GB per array
        size = parquet.metadata.row_group(group).num_rows
        local = [index - offset for index in range(offset, offset + size) if index in selected]
        if local:
            pieces.append(parquet.read_row_group(group).take(local))
        offset += size
    subset = pa.concat_tables(pieces)
    pq.write_table(subset, val_path)
    print(f"[write] {val_path} ({subset.num_rows} rows, monitoring only)")
    return subset.num_rows


# ---------------------------------------------------------------------------
# VEPO: "4.2K" training problems, read as Geometry3K train (2,101) plus the mini_train split of
# xyliu6/k12-freeform (2,100), which VEPO's code base (NoisyRollout) trains on; validation on the
# k12-freeform test split (808), as VEPO's script. Geometry3K must be prepared first.
# ---------------------------------------------------------------------------


@register("vepo")
def convert_vepo(repo_id: str, raw_dir: Path, out_dir: Path, snapshot) -> dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    geometry3k = out_dir.parent / "geometry3k" / "train.parquet"
    if not geometry3k.exists():
        raise FileNotFoundError(f"{geometry3k} is missing; prepare geometry3k first")
    snapshot(["data/mini_train-*.parquet", "data/test-*.parquet"])
    columns = ["problem", "answer", "images"]

    def read(pattern: str) -> pa.Table:
        return pa.concat_tables(pq.read_table(path, columns=columns) for path in sorted(raw_dir.glob(pattern)))

    splits = {
        "train": pa.concat_tables([pq.read_table(geometry3k, columns=columns), read("data/mini_train-*.parquet")]),
        "test": read("data/test-*.parquet"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for split, table in splits.items():
        pq.write_table(table, out_dir / f"{split}.parquet")
        summary[split] = table.num_rows
        print(f"[write] {out_dir / f'{split}.parquet'} ({table.num_rows} rows)")
    return summary


# ---------------------------------------------------------------------------
# CFPO (RavenInJuly/CFPO_Datasets): CFPO's copy of ViRL39K (38,870 problems) with its own problem
# text (option format, <image> placement), and the ViRL39K images it references by file name.
# ---------------------------------------------------------------------------


@register("cfpo")
def convert_cfpo(repo_id: str, raw_dir: Path, out_dir: Path, snapshot) -> dict[str, int]:
    import json
    import zipfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    snapshot(["ViRL39K_train.json", "ViRL39K_images.zip"])
    with open(raw_dir / "ViRL39K_train.json", encoding="utf-8") as f:
        rows = json.load(f)
    schema = pa.schema(
        [
            ("problem", pa.string()),
            ("answer", pa.string()),
            ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "train.parquet"
    tmp_output = output.with_suffix(".parquet.tmp")
    with zipfile.ZipFile(raw_dir / "ViRL39K_images.zip") as archive, pq.ParquetWriter(tmp_output, schema) as writer:
        members = {name.rsplit("/", 1)[-1]: name for name in archive.namelist() if not name.endswith("/")}
        for start in range(0, len(rows), 2000):  # row groups stay far below 2 GB
            chunk = rows[start : start + 2000]
            writer.write_table(
                pa.table(
                    {
                        "problem": [row["problem"] for row in chunk],
                        "answer": [str(row["answer"]) for row in chunk],
                        "images": [
                            [{"bytes": archive.read(members[name]), "path": name} for name in row["images"]]
                            for row in chunk
                        ],
                    },
                    schema=schema,
                )
            )
    tmp_output.replace(output)
    print(f"[write] {output} ({len(rows)} rows)")
    return {"train": len(rows)}
