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
    "chat_template",
    "plain_think_tokens",
    "interaction_mode",
    "agent_profile",
    "min_pixels",
    "max_pixels",
    "grounding_instruction",
    "answer_protocol",
)


@dataclass(frozen=True)
class Suite:
    name: str
    benchmarks: tuple[str, ...]
    description: str = ""
    notes: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    # (group name, benchmarks) in order; when set, the summaries add each group's mean and the mean of the
    # group means next to the registry groups and the mean over all benchmarks
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()


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
        if "plain_think_tokens" in defaults:
            from verl.utils.plain_think import normalize_plain_think_tokens

            defaults["plain_think_tokens"] = normalize_plain_think_tokens(defaults["plain_think_tokens"])
        unique_keys = tuple(dict.fromkeys(keys))
        groups = _parse_groups(name, entry.get("groups"), unique_keys)
        includes = entry.get("include") or []
        if not groups and len(includes) == 1 and not benchmarks:
            groups = resolve(str(includes[0]), stack + (name,)).groups  # an alias keeps the groups
        suite = Suite(
            name=name,
            benchmarks=unique_keys,
            description=str(entry.get("description") or ""),
            notes=" ".join(str(entry.get("notes") or "").split()),
            defaults=defaults,
            groups=groups,
        )
        resolved[name] = suite
        return suite

    for name in raw:
        resolve(str(name), ())
    return resolved


def _parse_groups(name: str, raw: Any, benchmarks: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``groups: {group: [benchmark, ...]}``: every benchmark of the suite in exactly one group."""
    if raw is None:
        return ()
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"suite {name}: groups must be a non-empty mapping of group name -> benchmarks")
    groups = []
    assigned: dict[str, str] = {}
    for group, members in raw.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"suite {name}: group {group!r} must list benchmarks")
        for key in members:
            if key not in benchmarks:
                raise ValueError(f"suite {name}: group {group!r} lists {key!r}, which is not in the suite")
            if key in assigned:
                raise ValueError(f"suite {name}: {key!r} is in groups {assigned[key]!r} and {group!r}")
            assigned[key] = str(group)
        groups.append((str(group), tuple(str(key) for key in members)))
    ungrouped = [key for key in benchmarks if key not in assigned]
    if ungrouped:
        raise ValueError(f"suite {name}: benchmarks without a group: {ungrouped}")
    return tuple(groups)


def suite_groupings(suites: dict[str, Suite], names: list[str]) -> dict[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """The groups of the selected suites that define them (aliases of the same groups appear once)."""
    selected: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    for name in names:
        groups = suites[name].groups
        if groups and groups not in selected.values():
            selected[name] = groups
    return selected


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
