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
"""MMStar (HF ``Lin-Chen/MMStar``, the dataset lmms-eval's ``mmstar`` task reads): 1,500 four-option MCQs.

The single parquet file embeds the images (raw bytes) next to ``question`` (options included),
``answer`` (letter), ``category`` and ``l2_category``. Layout::

    <data_root>/mmstar/val.parquet
"""

from __future__ import annotations

from .base import BenchmarkSource
from .lmms_lab import prepare_renamed


SOURCES = [
    BenchmarkSource(
        key="mmstar",
        target="mmstar",
        outputs=("mmstar/val.parquet",),
        source="Lin-Chen/MMStar (mmstar.parquet, split val)",
        approx_size="42 MB",
        prepare=prepare_renamed,
        options={"repos": ["Lin-Chen/MMStar"], "files": {"mmstar.parquet": "val.parquet"}, "expected_rows": 1500},
    )
]
