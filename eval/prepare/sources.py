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
"""Benchmark key -> how to obtain it. One module per upstream source."""

from __future__ import annotations

from . import (
    blink,
    cfpo,
    cvbench,
    grit,
    hrbench,
    lmms_lab,
    mme_realworld,
    mmstar,
    papo_eval,
    refcoco,
    seed_bench,
    vppo_eval,
    vstar,
    zoombench,
)
from .base import BenchmarkSource


SOURCES: dict[str, BenchmarkSource] = {
    source.key: source
    for module in (
        papo_eval,
        vppo_eval,
        lmms_lab,
        mmstar,
        blink,
        cvbench,
        seed_bench,
        cfpo,
        grit,
        vstar,
        hrbench,
        mme_realworld,
        zoombench,
        refcoco,
    )
    for source in module.SOURCES
}
