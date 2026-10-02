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
from datetime import datetime
from pathlib import Path
from typing import Any

from .json_utils import read_json, write_json


def fingerprint(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_path(out_dir: Path, key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "__")
    return out_dir / "state" / f"{safe}.json"


def is_complete(out_dir: Path, key: str, expected_fingerprint: str, outputs: list[Path]) -> bool:
    path = state_path(out_dir, key)
    if not path.exists():
        return False
    try:
        state = read_json(path)
    except Exception:
        return False
    if state.get("status") != "ok" or state.get("fingerprint") != expected_fingerprint:
        return False
    return all(output.exists() and output.stat().st_size > 0 for output in outputs)


def mark_complete(
    out_dir: Path, key: str, expected_fingerprint: str, outputs: list[Path], extra: dict[str, Any] | None = None
) -> None:
    payload = {
        "status": "ok",
        "fingerprint": expected_fingerprint,
        "outputs": [{"path": str(path), "size": path.stat().st_size if path.exists() else 0} for path in outputs],
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    write_json(state_path(out_dir, key), payload)


def mark_failed(out_dir: Path, key: str, expected_fingerprint: str, error: str) -> None:
    write_json(
        state_path(out_dir, key),
        {
            "status": "failed",
            "fingerprint": expected_fingerprint,
            "error": error,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
