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
"""ZeroBench sub-questions (HF ``jonathan-roberts1/zerobench``, split ``zerobench_subquestions``): 334 open questions.

The dataset is gated: accept its terms on the dataset page once (they ask not to publish the answers or train
on the data) and log in (``hf auth login``, or ``HF_TOKEN``) before preparing it. The ``main`` branch holds the
current version (v3); earlier versions are on the ``v1`` and ``v2`` branches. The parquet file embeds the
images of each question (``question_images_decoded``, one or more) next to ``question_id``, ``question_text``,
``question_answer`` and ``question_images`` (their file names). Layout::

    <data_root>/zerobench_sub/zerobench_subquestions.parquet
"""

from __future__ import annotations

from typing import Any

from .base import BenchmarkSource
from .common import PrepareContext, PrepareError
from .lmms_lab import prepare_renamed


REPO_ID = "jonathan-roberts1/zerobench"


def prepare_zerobench(ctx: PrepareContext, spec: BenchmarkSource) -> dict[str, Any]:
    try:
        return prepare_renamed(ctx, spec)
    except PrepareError as exc:
        if "gated" not in str(exc).lower():
            raise
        raise PrepareError(
            f"{spec.key}: {REPO_ID} is a gated dataset. Accept its terms at "
            f"https://huggingface.co/datasets/{REPO_ID} with your Hugging Face account, then log in with "
            f"`hf auth login` (or set HF_TOKEN) and re-run. ({exc})"
        ) from exc


SOURCES = [
    BenchmarkSource(
        key="zerobench_sub",
        target="zerobench_sub",
        outputs=("zerobench_sub/zerobench_subquestions.parquet",),
        source=f"{REPO_ID} (split zerobench_subquestions; gated, needs a Hugging Face login)",
        approx_size="96 MB",
        prepare=prepare_zerobench,
        options={
            "repos": [REPO_ID],
            "files": {"data/zerobench_subquestions-00000-of-00001.parquet": "zerobench_subquestions.parquet"},
            "expected_rows": 334,
        },
    )
]
