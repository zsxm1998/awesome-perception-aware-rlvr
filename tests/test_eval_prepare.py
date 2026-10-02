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
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.loaders import load_samples, sharegpt_image_relpath  # noqa: E402
from easyr1_eval.registry import load_benchmark_specs  # noqa: E402

from eval.prepare import cli, hrbench, papo_eval  # noqa: E402
from eval.prepare.base import BenchmarkSource  # noqa: E402
from eval.prepare.common import (  # noqa: E402
    PrepareContext,
    PrepareError,
    extract_zip,
    is_prepared,
    place_referenced_files,
)


def _png(color=(255, 0, 0)):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_resolve_targets_expands_suites_and_all():
    assert cli.resolve_targets(["papo"]) == [
        "geo3k",
        "mathvista",
        "wemath",
        "mmk12",
        "mathverse",
        "logicvista",
        "clevr_count",
        "mmmu_pro",
        "mathverse_v",
    ]
    assert cli.resolve_targets(["geo3k", "papo"])[0] == "geo3k"
    everything = cli.resolve_targets(["all"])
    assert "refcoco_val" not in everything and "seed_bench" not in everything
    assert "mm_vet" in everything and "hrbench_8k" in everything
    assert "refcoco_val" in cli.resolve_targets(["all"], include_optional=True)
    with pytest.raises(SystemExit):
        cli.resolve_targets(["not_a_benchmark"])


def test_prepare_one_is_idempotent_and_force_redoes(tmp_path, monkeypatch):
    calls = []

    def fake_prepare(ctx, source):
        calls.append(source.key)
        out = ctx.data_root / "fake" / "data.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}\n", encoding="utf-8")
        return {"rows": 1}

    source = BenchmarkSource(
        key="fake",
        target="fake",
        outputs=("fake/data.jsonl",),
        source="unit test",
        approx_size="1 B",
        prepare=fake_prepare,
    )
    monkeypatch.setitem(cli.SOURCES, "fake", source)
    ctx = PrepareContext(data_root=tmp_path, log=lambda message: None)

    assert cli.prepare_one(ctx, "fake")[0] == "prepared"
    assert is_prepared(tmp_path / "fake", "fake", [tmp_path / "fake/data.jsonl"])
    marker = json.loads((tmp_path / "fake" / ".prepared-fake.json").read_text())
    assert marker["rows"] == 1 and marker["source"] == "unit test"
    assert cli.prepare_one(ctx, "fake")[0] == "skipped"
    ctx.force = True
    assert cli.prepare_one(ctx, "fake")[0] == "prepared"
    assert calls == ["fake", "fake"]
    # A removed output invalidates the marker.
    ctx.force = False
    (tmp_path / "fake" / "data.jsonl").unlink()
    assert cli.prepare_one(ctx, "fake")[0] == "prepared"


def test_cli_main_continues_after_a_failure_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    def broken(ctx, source):
        raise PrepareError("mirror unreachable")

    def fine(ctx, source):
        (ctx.data_root / "ok").mkdir(parents=True, exist_ok=True)
        (ctx.data_root / "ok" / "x.jsonl").write_text("{}\n")
        return {"rows": 1}

    monkeypatch.setitem(
        cli.SOURCES, "geo3k", BenchmarkSource("geo3k", "geo3k", ("geo3k/test.parquet",), "s", "1", broken)
    )
    monkeypatch.setitem(cli.SOURCES, "pope", BenchmarkSource("pope", "ok", ("ok/x.jsonl",), "s", "1", fine))
    assert cli.main(["geo3k", "pope", "--data-root", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "geo3k: FAILED" in output and "pope: prepared" in output


def test_place_referenced_files_moves_by_basename(tmp_path):
    (tmp_path / "images" / "nested" / "extra").mkdir(parents=True)
    (tmp_path / "images" / "nested" / "extra" / "a.png").write_bytes(b"a")
    place_referenced_files(tmp_path, ["images/geo/a.png"], search_root=tmp_path / "images")
    assert (tmp_path / "images" / "geo" / "a.png").read_bytes() == b"a"
    with pytest.raises(PrepareError, match="missing after extraction"):
        place_referenced_files(tmp_path, ["images/geo/b.png"])


def test_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(PrepareError, match="unsafe"):
        extract_zip(archive, tmp_path / "out", log=lambda message: None)


def test_papo_split_prepare_produces_loader_layout(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {
            "messages": [{"role": "user", "content": f"<image>Q{i}?"}, {"role": "assistant", "content": str(i)}],
            "images": [f"./data/images/hiyouga_geometry3k/hiyouga_geometry3k_{i}_0.png"],
            "id": f"hiyouga_geometry3k_{i}",
        }
        for i in range(2)
    ]
    pd.DataFrame(rows).to_parquet(raw / "split.parquet")
    with zipfile.ZipFile(raw / "images.zip", "w") as zf:
        for i in range(2):
            zf.writestr(f"hiyouga_geometry3k/hiyouga_geometry3k_{i}_0.png", _png())

    def fake_download(ctx, repo_ids, filename):
        return raw / ("split.parquet" if filename.endswith(".parquet") else "images.zip")

    monkeypatch.setattr(papo_eval, "hf_download", fake_download)
    monkeypatch.setitem(papo_eval.SPLITS, "geo3k", ("hiyouga_geometry3k", "images.zip", "1 KB", 2))
    data_root = tmp_path / "data"
    ctx = PrepareContext(data_root=data_root, keep_raw=True, log=lambda message: None)
    source = next(item for item in papo_eval.SOURCES if item.key == "geo3k")

    info = papo_eval.prepare_papo_split(ctx, source)

    assert info["rows"] == 2
    spec = next(spec for spec in load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml") if spec.key == "geo3k")
    samples = load_samples(spec, data_root)
    assert [sample.prompt for sample in samples] == ["Q0?", "Q1?"]
    assert all(Path(sample.images[0]).is_file() for sample in samples)


def test_hrbench_prepare_decodes_and_deduplicates_images(tmp_path, monkeypatch):
    images = [base64.b64encode(_png((255, 0, 0))).decode(), base64.b64encode(_png((0, 255, 0))).decode()]
    rows = [
        {
            "index": i,
            "question": f"q{i}",
            "answer": "ABCD"[i % 4],
            "category": "single" if i < 400 else "cross",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "cycle_category": f"c{i % 4}",
            "image": images[i % 2],
        }
        for i in range(800)
    ]
    parquet = tmp_path / "hr.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    monkeypatch.setattr(hrbench, "hf_download", lambda ctx, repo, filename: parquet)
    ctx = PrepareContext(data_root=tmp_path / "data", keep_raw=True, log=lambda message: None)
    source = next(item for item in hrbench.SOURCES if item.key == "hrbench_4k")

    info = hrbench.prepare_hrbench(ctx, source)

    assert info == {"repo_id": hrbench.REPO_ID, "file": "hr_bench_4k.parquet", "rows": 800, "images": 2}
    spec = next(
        spec for spec in load_benchmark_specs(ROOT / "eval/config/benchmarks.yaml") if spec.key == "hrbench_4k"
    )
    samples = load_samples(spec, tmp_path / "data", limit=3)
    assert samples[0].target == "A" and samples[1].target == "B"
    assert samples[0].images[0].endswith(".png")
    assert len(list((tmp_path / "data" / "hrbench_4k" / "images").iterdir())) == 2


def test_prepare_and_loader_agree_on_image_paths():
    assert papo_eval.sharegpt_image_relpath("./data/images/a/b.png") == sharegpt_image_relpath("./data/images/a/b.png")
