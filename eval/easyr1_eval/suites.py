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
"""Named benchmark suites (eval/config/suites.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schemas import BenchmarkSpec


SUITE_DEFAULT_KEYS = (
    "format_prompt",
    "system_prompt",
    "interaction_mode",
    "agent_profile",
    "min_pixels",
    "max_pixels",
)


@dataclass(frozen=True)
class Suite:
    name: str
    benchmarks: tuple[str, ...]
    description: str = ""
    notes: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)


def load_suites(path: Path, specs: list[BenchmarkSpec]) -> dict[str, Suite]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw = data.get("suites")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a 'suites' mapping")
    known = {spec.key for spec in specs}
    non_optional = [spec.key for spec in specs if not spec.optional]

    resolved: dict[str, Suite] = {}

    def resolve(name: str, stack: tuple[str, ...]) -> Suite:
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ValueError(f"suite include cycle: {' -> '.join(stack + (name,))}")
        entry = raw.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"unknown suite: {name}")
        keys: list[str] = []
        for included in entry.get("include") or []:
            keys.extend(resolve(str(included), stack + (name,)).benchmarks)
        benchmarks = entry.get("benchmarks") or []
        if benchmarks == "all":
            benchmarks = non_optional
        if not isinstance(benchmarks, list):
            raise ValueError(f"suite {name}: benchmarks must be a list or 'all'")
        keys.extend(str(key) for key in benchmarks)
        unknown = sorted(set(keys) - known)
        if unknown:
            raise ValueError(f"suite {name} references unknown benchmark(s): {', '.join(unknown)}")
        defaults = dict(entry.get("defaults") or {})
        bad = sorted(set(defaults) - set(SUITE_DEFAULT_KEYS))
        if bad:
            raise ValueError(f"suite {name}: unsupported defaults {bad}; allowed: {list(SUITE_DEFAULT_KEYS)}")
        suite = Suite(
            name=name,
            benchmarks=tuple(dict.fromkeys(keys)),
            description=str(entry.get("description") or ""),
            notes=" ".join(str(entry.get("notes") or "").split()),
            defaults=defaults,
        )
        resolved[name] = suite
        return suite

    for name in raw:
        resolve(str(name), ())
    return resolved


def suite_benchmarks(suites: dict[str, Suite], names: list[str]) -> list[str]:
    keys: list[str] = []
    for name in names:
        if name not in suites:
            raise KeyError(f"unknown suite: {name} (available: {', '.join(sorted(suites))})")
        keys.extend(suites[name].benchmarks)
    return list(dict.fromkeys(keys))


def merged_suite_defaults(
    suites: dict[str, Suite], names: list[str], *, ignore: set[str] | frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Defaults shared by the selected suites; conflicting values raise (keys in ``ignore`` are skipped)."""
    merged: dict[str, Any] = {}
    for name in names:
        for key, value in suites[name].defaults.items():
            if key in ignore:
                continue
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"suites {names} disagree on default {key!r} ({merged[key]!r} vs {value!r}); "
                    f"pass --{key.replace('_', '-')} explicitly"
                )
            merged[key] = value
    return merged
