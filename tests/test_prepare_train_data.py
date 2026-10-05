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
"""scripts/data/prepare_train_data.py prepares again what an older converter wrote."""

import importlib.util
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "prepare_train_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_train_data", SCRIPT)
prepare_train_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_train_data)


def _write(out_dir: Path, columns: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        pq.write_table(pa.table({name: ["x"] for name in columns}), out_dir / f"{split}.parquet")


def test_output_without_the_current_columns_is_prepared_again(tmp_path, monkeypatch):
    calls = []

    def converter(repo_id, raw_dir, out_dir, snapshot):
        calls.append(out_dir)
        _write(out_dir, ["problem", "env_name"])
        return {"train": 1, "val": 1}

    monkeypatch.setitem(prepare_train_data.CONVERTERS, "fake", converter)
    dataset = prepare_train_data.TrainDataset(
        name="fake",
        repo_id="org/fake",
        splits={"train": [], "val": []},
        description="",
        used_by=[],
        converter="fake",
        required_columns=("env_name",),
    )
    _write(tmp_path / "fake", ["problem"])  # written by an older converter
    prepare_train_data.prepare(dataset, tmp_path, keep_raw=False, force=False)
    assert len(calls) == 1 and "env_name" in pq.read_schema(tmp_path / "fake" / "train.parquet").names

    prepare_train_data.prepare(dataset, tmp_path, keep_raw=False, force=False)  # up to date: skipped
    assert len(calls) == 1


def test_deepeyes_requires_the_official_prompt_columns():
    deepeyes = prepare_train_data.DATASETS["deepeyes"]
    assert {"env_name", "official_prompt", "official_system_prompt", "row_system_prompt"} <= set(
        deepeyes.required_columns
    )


def test_every_method_group_prepares_the_data_of_its_scripts():
    """`bash scripts/prepare_data.sh <method>` covers every $DATA_ROOT/<dataset>/ its launch scripts read."""
    root = SCRIPT.resolve().parents[2]
    folders = {
        "comparison": "examples/comparison/qwen3_vl_4b",
        "opd_comparison": "examples/comparison/opd_qwen3_vl_2b",
    }
    pattern = re.compile(r"\$\{?DATA_ROOT\}?/([A-Za-z0-9_]+)/")
    for group, datasets in prepare_train_data.METHOD_GROUPS.items():
        folder = root / folders.get(group, f"examples/reproduction/{group}")
        scripts = sorted(folder.glob("*.sh"))
        assert scripts, f"no launch scripts for the data group {group} in {folder}"
        used = {name for script in scripts for name in pattern.findall(script.read_text(encoding="utf-8"))}
        assert used <= set(datasets), f"{group}: scripts read {sorted(used - set(datasets))}, not prepared"
        assert set(datasets) <= set(prepare_train_data.DATASETS)


def _image(seed: int) -> list[dict]:
    return [{"bytes": bytes([seed]) * 8, "path": f"{seed}.png"}]


def _vapo_shard(path: Path, rows: list[tuple[str, str, int]]) -> None:
    columns = {
        "problem": [problem for problem, _, _ in rows],
        "answer": [answer for _, answer, _ in rows],
        "images": [_image(seed) for _, _, seed in rows],
    }
    for idx in range(1, 21):
        columns[f"claim_{idx}"] = [f"claim {idx} of {problem}" for problem, _, _ in rows]
        columns[f"label_{idx}"] = ["CORRECT" if idx % 2 else "WRONG"] * len(rows)
    table = pa.table(columns)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)


def _virl39k(data_root: Path, rows: list[tuple[str, str, int, int]]) -> None:
    (data_root / "virl39k").mkdir(parents=True)
    table = pa.table(
        {
            "problem": [problem for problem, _, _, _ in rows],
            "answer": [answer for _, answer, _, _ in rows],
            "images": [_image(seed) * count for _, _, seed, count in rows],
        }
    )
    pq.write_table(table, data_root / "virl39k" / "train.parquet", row_group_size=2)


def _convert(name: str, data_root: Path, raw_dir: Path) -> dict:
    return prepare_train_data.CONVERTERS[name](
        repo_id="org/vapo", raw_dir=raw_dir, out_dir=data_root / name, snapshot=lambda patterns: raw_dir
    )


def test_vapo_claims_are_written_as_json(tmp_path):
    import json

    raw = tmp_path / "raw"
    raw.mkdir()
    _vapo_shard(raw / "data-00000-of-00002.arrow", [("<image>a", "1", 1)])
    _vapo_shard(raw / "data-00001-of-00002.arrow", [("<image>b", "2", 2)])
    assert _convert("vapo", tmp_path, raw) == {"train": 2}
    table = pq.read_table(tmp_path / "vapo" / "train.parquet").to_pylist()
    assert [row["problem"] for row in table] == ["<image>a", "<image>b"]
    claims = json.loads(table[1]["visual_claims"])
    assert len(claims) == 20 and claims[0] == {"claim": "claim 1 of <image>b", "correct": True}
    assert claims[1]["correct"] is False
    assert table[1]["images"] == _image(2)


def test_virl39k_claims_align_with_the_single_image_rows(tmp_path):
    import json

    raw = tmp_path / "raw"
    raw.mkdir()
    _vapo_shard(raw / "data-00000-of-00001.arrow", [("<image>a", "1", 1), ("<image>c", "3", 3)])
    virl = [("<image>a", "1", 1, 1), ("<image><image>b", "2", 2, 2), ("<image>c", "3", 3, 1)]
    _virl39k(tmp_path, virl)
    assert _convert("virl39k_claims", tmp_path, raw) == {"train": 3}
    out = pq.read_table(tmp_path / "virl39k_claims" / "train.parquet")
    source = pq.read_table(tmp_path / "virl39k" / "train.parquet")
    assert out.drop(["visual_claims"]).equals(source)  # the ViRL39K rows unchanged, in order
    claims = [json.loads(value) for value in out.column("visual_claims").to_pylist()]
    assert [len(c) for c in claims] == [20, 0, 20] and claims[2][0]["claim"] == "claim 1 of <image>c"


def test_virl39k_claims_reject_misaligned_rows(tmp_path):
    cases = {
        "image": ([("<image>a", "1", 9, 1), ("<image>c", "3", 3, 1)], "does not match"),
        "text": ([("<image>x", "1", 1, 1), ("<image>c", "3", 3, 1)], "does not match"),
        "extra VAPO row": ([("<image>a", "1", 1, 1)], "remain"),
        "missing VAPO row": ([("<image>a", "1", 1, 1), ("<image>c", "3", 3, 1), ("<image>d", "4", 4, 1)], "ran out"),
    }
    for idx, (virl, match) in enumerate(cases.values()):
        root = tmp_path / str(idx)
        raw = root / "raw"
        raw.mkdir(parents=True)
        _vapo_shard(raw / "data-00000-of-00001.arrow", [("<image>a", "1", 1), ("<image>c", "3", 3)])
        _virl39k(root, virl)
        try:
            _convert("virl39k_claims", root, raw)
        except ValueError as error:
            assert match in str(error), (match, error)
        else:
            raise AssertionError(f"misaligned data was accepted: {match}")
