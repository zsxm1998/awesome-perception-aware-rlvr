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
"""BLINK (HF ``BLINK-Benchmark/BLINK``): the 1,901 validation questions of the 14 subtasks.

Each subtask is a dataset config with a ``val`` parquet file that embeds up to four images
(``image_1`` .. ``image_4``) next to ``prompt`` (question + "(A) ..." options), ``choices``,
``answer`` (``"(A)"``) and ``sub_task``. Layout::

    <data_root>/blink/<Subtask>.parquet
"""

from __future__ import annotations

from .base import BenchmarkSource
from .lmms_lab import prepare_renamed


SUBTASKS = (
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
)

SOURCES = [
    BenchmarkSource(
        key="blink",
        target="blink",
        outputs=tuple(f"blink/{subtask}.parquet" for subtask in SUBTASKS),
        source="BLINK-Benchmark/BLINK (14 subtasks, split val)",
        approx_size="400 MB",
        prepare=prepare_renamed,
        options={
            "repos": ["BLINK-Benchmark/BLINK"],
            "files": {f"{subtask}/val-00000-of-00001.parquet": f"{subtask}.parquet" for subtask in SUBTASKS},
            "expected_rows": 1901,
        },
    )
]
