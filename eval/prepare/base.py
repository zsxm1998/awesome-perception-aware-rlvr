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
from typing import Any, Callable

from .common import PrepareContext


@dataclass(frozen=True)
class BenchmarkSource:
    """How one registry benchmark is obtained.

    ``target`` is the directory under the data root the benchmark lives in and ``outputs``
    lists the files (relative to the data root) the registry entry reads; a benchmark counts
    as prepared once its marker and all outputs exist.
    """

    key: str
    target: str
    outputs: tuple[str, ...]
    source: str
    approx_size: str
    prepare: Callable[[PrepareContext, "BenchmarkSource"], dict[str, Any]]
    options: dict[str, Any] = field(default_factory=dict)
    # Free space (GB) the preparation needs on the data-root filesystem, including temporary
    # downloads; checked before anything is downloaded.
    disk_gb: float = 1.0

    def target_dir(self, data_root: Path) -> Path:
        return data_root / self.target

    def output_paths(self, data_root: Path) -> list[Path]:
        return [data_root / output for output in self.outputs]
