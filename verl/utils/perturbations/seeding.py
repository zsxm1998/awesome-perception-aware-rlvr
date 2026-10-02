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
from __future__ import annotations

import hashlib
import json
from typing import Any


def derive_seed(global_seed: int, sample_id: str, image_idx: int, *purpose: Any) -> int:
    payload = json.dumps(
        [int(global_seed), str(sample_id), int(image_idx), *[str(item) for item in purpose]],
        sort_keys=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**32)
