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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_OUTPUT_CONTRACT_NATIVE = "native"
PRIMARY_NA_STATUS = "primary_na"
SKIPPED_STATUS = "skipped"


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    label: str
    group: str
    loader: str
    scorer: str
    primary_metric: str
    path: str | list[str] | None = None
    paths: list[str] | None = None
    image_root: str | None = None
    split: str | None = None
    max_new_tokens: int = 128
    temperature: float | None = None
    num_samples: int | None = None
    top_p: float = 1.0
    normalize: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Upstream dataset the files come from (informational; shown by --list commands).
    source: str | None = None
    # Benchmarks whose metric needs an LLM judge are skipped (with a warning) when no
    # judge is configured instead of failing the run.
    requires_judge: bool = False
    # Optional benchmarks (large downloads) are left out of the `all` suite.
    optional: bool = False

    def data_paths(self, data_root: Path) -> list[Path]:
        raw_paths: list[str] = []
        if self.paths:
            raw_paths.extend(self.paths)
        elif isinstance(self.path, list):
            raw_paths.extend(self.path)
        elif isinstance(self.path, str):
            raw_paths.append(self.path)
        return [resolve_data_path(item, data_root) for item in raw_paths]

    def resolved_image_root(self, data_root: Path) -> Path | None:
        if self.image_root is None:
            return None
        return resolve_data_path(self.image_root, data_root)


@dataclass
class EvalSample:
    benchmark: str
    sample_id: str
    prompt: str
    target: Any
    images: list[Any] = field(default_factory=list)
    messages: list[dict[str, Any]] | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Some benchmarks append a textual-grounding output contract for
    # grounded-reasoning models. Native visual agents instead expose evidence
    # through tool actions, so retain the unmodified question explicitly.
    native_agentic_prompt: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    num_samples: int
    max_new_tokens: int
    seed: int
    top_k: int | None = None  # None: no top-k limit


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    finish_reason: str | None = None
    stop_reason: Any = None
    token_count: int | None = None
    truncated: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricResult:
    benchmark: str
    group: str
    primary_metric: str
    raw_score: float | None
    normalized_score_0_100: float | None
    num_examples: int
    status: str = "ok"
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_data_path(value: str, data_root: Path) -> Path:
    text = value.replace("${data_root}", str(data_root))
    path = Path(text)
    if not path.is_absolute():
        path = data_root / path
    return path
