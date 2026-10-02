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
"""Download and lay out evaluation benchmarks under ``data/eval/<benchmark>/``.

Usage (from the repository root)::

    python -m eval.prepare papo            # every benchmark of a suite
    python -m eval.prepare geo3k pope      # individual benchmarks
    python -m eval.prepare all             # every non-optional benchmark
    python -m eval.prepare --list          # benchmarks, sources and suites

``bash scripts/prepare_eval_data.sh`` is a thin wrapper around the same CLI.
"""
