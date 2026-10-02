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
"""Prepare evaluation data: ``python -m eval.prepare <benchmark|suite|all> ... [--data-root DIR]``.

Targets are benchmark keys (eval/config/benchmarks.yaml), suite names
(eval/config/suites.yaml) or ``all`` (every non-optional benchmark; add
``--include-optional`` for RefCOCO and SEED-Bench). Already prepared benchmarks are
skipped unless ``--force`` is given. Downloads use the Hugging Face Hub; set
``HF_ENDPOINT=https://hf-mirror.com`` if huggingface.co is slow or blocked.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from easyr1_eval.paths import DEFAULT_CONFIG, DEFAULT_SUITES, default_data_root  # noqa: E402
from easyr1_eval.registry import load_benchmark_specs  # noqa: E402
from easyr1_eval.suites import load_suites  # noqa: E402

from .common import (  # noqa: E402
    PrepareContext,
    PrepareError,
    bypass_proxy_for_mirror,
    directory_size,
    free_disk_gb,
    is_prepared,
    write_marker,
)
from .sources import SOURCES  # noqa: E402


def resolve_targets(targets: list[str], *, include_optional: bool = False) -> list[str]:
    specs = load_benchmark_specs(DEFAULT_CONFIG)
    by_key = {spec.key: spec for spec in specs}
    suites = load_suites(DEFAULT_SUITES, specs)
    keys: list[str] = []
    for target in targets:
        if target == "all":
            keys.extend(spec.key for spec in specs if include_optional or not spec.optional)
        elif target in by_key:
            keys.append(target)
        elif target in suites:
            keys.extend(suites[target].benchmarks)
        else:
            raise SystemExit(
                f"unknown target {target!r}. Benchmarks: {', '.join(by_key)}. Suites: {', '.join(suites)}."
            )
    missing = [key for key in keys if key not in SOURCES]
    if missing:
        raise SystemExit(f"no preparation recipe for: {', '.join(missing)}")
    return list(dict.fromkeys(keys))


def print_listing() -> None:
    specs = load_benchmark_specs(DEFAULT_CONFIG)
    suites = load_suites(DEFAULT_SUITES, specs)
    print("Benchmarks (key, approx. size, source):")
    for spec in specs:
        source = SOURCES.get(spec.key)
        flag = " [optional]" if spec.optional else ""
        flag += " [needs judge]" if spec.requires_judge else ""
        size = source.approx_size if source else "?"
        print(f"  {spec.key:<18} {size:<34} {source.source if source else spec.source}{flag}")
    print("\nSuites:")
    for name, suite in suites.items():
        print(f"  {name:<14} {', '.join(suite.benchmarks)}")


def prepare_one(ctx: PrepareContext, key: str) -> tuple[str, str]:
    source = SOURCES[key]
    target_dir = source.target_dir(ctx.data_root)
    outputs = source.output_paths(ctx.data_root)
    if not ctx.force and is_prepared(target_dir, key, outputs):
        return "skipped", "already prepared"
    ctx.log(f"[prepare] {key} <- {source.source} (about {source.approx_size})")
    free = free_disk_gb(ctx.data_root)
    if source.disk_gb > 2 and free < source.disk_gb * 1.1 and not ctx.skip_space_check:
        raise PrepareError(
            f"{key} needs about {source.disk_gb:.0f} GB of free space under {ctx.data_root} "
            f"but only {free:.1f} GB is free; free some space, use --data-root on a larger disk, "
            "or pass --skip-space-check"
        )
    if source.disk_gb >= 5:
        ctx.log(
            f"  [disk] large download: ~{source.disk_gb:.0f} GB needed, {free:.0f} GB free; interrupted runs resume"
        )
    started = time.time()
    info = source.prepare(ctx, source)
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"preparation finished but expected outputs are missing: {missing}")
    write_marker(target_dir, key, {"source": source.source, **info})
    elapsed = time.time() - started
    rows = info.get("rows", "?")
    return "prepared", f"{rows} rows in {elapsed:.0f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="benchmark keys, suite names, or 'all'")
    parser.add_argument(
        "--data-root", default=None, help="default: $EVAL_DATA_ROOT, $DATA_ROOT/eval or <repo>/data/eval"
    )
    parser.add_argument("--force", action="store_true", help="re-download even if already prepared")
    parser.add_argument("--keep-raw", action="store_true", help="keep raw downloads under <data-root>/.raw")
    parser.add_argument("--include-optional", action="store_true", help="make 'all' include optional benchmarks")
    parser.add_argument("--hf-workers", type=int, default=8, help="parallel Hugging Face downloads (default 8)")
    parser.add_argument(
        "--http-workers", type=int, default=16, help="parallel image downloads from COCO/VG (default 16)"
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--skip-space-check", action="store_true", help="do not abort when free disk space looks too small"
    )
    parser.add_argument("--list", action="store_true", help="list benchmarks, sources and suites")
    args = parser.parse_args(argv)

    if args.list:
        print_listing()
        return 0
    if not args.targets:
        parser.print_help()
        return 2

    if not sys.stdout.isatty():  # keep logs readable when redirected to a file
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    bypass_proxy_for_mirror()
    keys = resolve_targets(args.targets, include_optional=args.include_optional)
    data_root = Path(args.data_root).expanduser() if args.data_root else default_data_root()
    data_root = data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    ctx = PrepareContext(
        data_root=data_root,
        force=args.force,
        keep_raw=args.keep_raw,
        hf_workers=args.hf_workers,
        http_workers=args.http_workers,
        retries=max(1, args.retries),
        skip_space_check=args.skip_space_check,
    )
    print(f"[prepare] data root: {data_root}")
    print(f"[prepare] benchmarks: {', '.join(keys)}")

    results: list[tuple[str, str, str]] = []
    for key in keys:
        try:
            status, detail = prepare_one(ctx, key)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - keep going with the other benchmarks
            traceback.print_exc()
            status, detail = "FAILED", f"{type(exc).__name__}: {exc}"
        results.append((key, status, detail))
        print(f"[prepare] {key}: {status} ({detail})", flush=True)
    ctx.finalize()

    print("\nSummary")
    for key, status, detail in results:
        target = SOURCES[key].target_dir(data_root)
        size = directory_size(target) / 1e6 if target.exists() else 0.0
        print(f"  {key:<18} {status:<9} {size:9.1f} MB  {detail}")
    failed = [key for key, status, _ in results if status == "FAILED"]
    if failed:
        print(f"\n{len(failed)} benchmark(s) failed: {', '.join(failed)}. Re-run the same command to retry.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
