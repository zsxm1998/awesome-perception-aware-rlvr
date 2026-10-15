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
"""VisualPuzzles (HF ``neulab/VisualPuzzles``): 1,168 four-option puzzles, the dataset's only split (``train``).

The single parquet file embeds one PNG image per question (``image``) next to ``id``, ``question``,
``options`` (four option texts, or null for the 669 questions whose options are drawn in the image),
``answer`` (letter A-D), ``category`` (algorithmic, analogical, deductive, inductive, spatial) and
``difficulty`` (easy, medium, hard). Layout::

    <data_root>/visualpuzzles/train.parquet
"""

from __future__ import annotations

from .base import BenchmarkSource
from .lmms_lab import prepare_renamed


SOURCES = [
    BenchmarkSource(
        key="visualpuzzles",
        target="visualpuzzles",
        outputs=("visualpuzzles/train.parquet",),
        source="neulab/VisualPuzzles (split train)",
        approx_size="143 MB",
        prepare=prepare_renamed,
        options={
            "repos": ["neulab/VisualPuzzles"],
            "files": {"data/train-00000-of-00001.parquet": "train.parquet"},
            "expected_rows": 1168,
        },
    )
]
