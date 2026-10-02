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
"""Default locations shared by the runner, the data preparation CLI and the viz server."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = PROJECT_ROOT / "eval"
DEFAULT_CONFIG = EVAL_ROOT / "config" / "benchmarks.yaml"
DEFAULT_SUITES = EVAL_ROOT / "config" / "suites.yaml"
DEFAULT_RESULTS_DIR = EVAL_ROOT / "results"
DEFAULT_FORMAT_PROMPT = PROJECT_ROOT / "examples" / "format_prompt" / "math_perception.jinja"


def default_data_root() -> Path:
    """Evaluation data root: ``$EVAL_DATA_ROOT``, else ``$DATA_ROOT/eval``, else ``<repo>/data/eval``.

    ``DATA_ROOT`` is the variable the training-data preparation script and the training
    launchers use for ``<repo>/data``; evaluation data lives in its ``eval/`` subdirectory.
    """
    explicit = os.environ.get("EVAL_DATA_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        return Path(data_root).expanduser() / "eval"
    return PROJECT_ROOT / "data" / "eval"
