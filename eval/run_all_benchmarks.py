#!/usr/bin/env python3
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
"""Evaluate a model on the registered benchmarks (see eval/README.md)."""

from __future__ import annotations

import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
# Prefer this repository's eval/ and verl/ packages over any other installed copy.
for _path in (EVAL_ROOT.parent, EVAL_ROOT):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

from easyr1_eval.runner import main  # noqa: E402


if __name__ == "__main__":
    main()
