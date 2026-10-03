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
