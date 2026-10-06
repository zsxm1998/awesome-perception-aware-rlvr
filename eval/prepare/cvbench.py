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
"""CV-Bench (HF ``nyu-visionx/CV-Bench``, Cambrian-1): 2,638 vision-centric multiple-choice questions.

``test_2d.parquet`` (1,438: counting and spatial relations on ADE20K and COCO images) and
``test_3d.parquet`` (1,200: depth order and relative distance on Omni3D images) embed the images next
to ``prompt`` (question and ``(A) ...`` options), ``choices``, ``answer`` (``(A)``), ``task`` and
``source``. Layout::

    <data_root>/cvbench/test_2d.parquet
    <data_root>/cvbench/test_3d.parquet
"""

from __future__ import annotations

from .base import BenchmarkSource
from .lmms_lab import prepare_renamed


SOURCES = [
    BenchmarkSource(
        key="cvbench",
        target="cvbench",
        outputs=("cvbench/test_2d.parquet", "cvbench/test_3d.parquet"),
        source="nyu-visionx/CV-Bench (test_2d.parquet, test_3d.parquet)",
        approx_size="405 MB",
        prepare=prepare_renamed,
        options={
            "repos": ["nyu-visionx/CV-Bench"],
            "files": {"test_2d.parquet": "test_2d.parquet", "test_3d.parquet": "test_3d.parquet"},
            "expected_rows": 2638,
        },
    )
]
