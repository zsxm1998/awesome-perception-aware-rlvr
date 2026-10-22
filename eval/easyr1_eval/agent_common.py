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
"""What agent evaluation with a local model (agentic.py, needs vLLM) and with an API (api_agent.py) share."""

from __future__ import annotations

import hashlib

from .schemas import EvalSample, GenerationOutput


class AgenticSampleError(ValueError):
    """A malformed or unreadable individual evaluation sample."""


def failed_trajectory_output(exc: BaseException) -> GenerationOutput:
    return GenerationOutput(
        text="",
        finish_reason="error",
        token_count=0,
        truncated=False,
        diagnostics={
            "agent": {
                "status": "sample_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        },
    )


def stable_sample_index(sample: EvalSample) -> int:
    """The sample's index for seeds, the same in every run and shard."""
    payload = f"{sample.benchmark}\0{sample.sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)
