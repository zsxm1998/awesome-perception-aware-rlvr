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
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import BenchmarkSpec


def load_benchmark_specs(config_path: Path) -> list[BenchmarkSpec]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("benchmarks"), list):
        raise ValueError(f"{config_path} must contain a 'benchmarks' list")

    specs: list[BenchmarkSpec] = []
    seen: set[str] = set()
    for item in data["benchmarks"]:
        if not isinstance(item, dict):
            raise ValueError("each benchmark entry must be a mapping")
        spec = BenchmarkSpec(**item)
        if spec.key in seen:
            raise ValueError(f"duplicate benchmark key: {spec.key}")
        seen.add(spec.key)
        specs.append(spec)
    return specs


def select_benchmarks(
    specs: list[BenchmarkSpec],
    *,
    include: list[str] | None = None,
    skip: list[str] | None = None,
) -> list[BenchmarkSpec]:
    by_key = {spec.key: spec for spec in specs}
    skip_set = set(skip or [])
    if include:
        missing = sorted(set(include) - set(by_key))
        if missing:
            raise KeyError(f"unknown benchmark(s): {', '.join(missing)}")
        selected = [by_key[key] for key in include]
    else:
        selected = list(specs)
    missing_skip = sorted(skip_set - set(by_key))
    if missing_skip:
        raise KeyError(f"unknown benchmark(s) in skip list: {', '.join(missing_skip)}")
    return [spec for spec in selected if spec.key not in skip_set]


def specs_fingerprint_payload(specs: list[BenchmarkSpec]) -> list[dict[str, Any]]:
    return [spec.__dict__ for spec in specs]
