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
"""Write the tasks of an OpenReward (ORS) environment as data rows for agent_environment=openreward.

One row per task: ``env_config`` names the environment, split and index (the session is opened during the rollout),
``prompt`` is a short label (only for logs and the prompt-length filter; the environment writes the prompt) and
``answer`` is empty (the environment scores the episode). Only the task count is asked from the server.

    python scripts/agent/prepare_openreward_tasks.py --environment GeneralReasoning/WhoDunit --split train \\
        --num-tasks 256 --out data/openreward/whodunit/train.parquet          # needs OPENREWARD_API_KEY
    python scripts/agent/prepare_openreward_tasks.py --url http://127.0.0.1:8010 --environment toysearch \\
        --split val --out data/openreward/toysearch/val.parquet               # a self-hosted server
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def parse_indices(text: str) -> list[int]:
    """``0,5,10-19`` -> [0, 5, 10, ..., 19]."""
    indices: list[int] = []
    for item in text.split(","):
        first, _, last = item.strip().partition("-")
        indices.extend(range(int(first), int(last or first) + 1))
    return indices


def task_rows(environment: str, split: str, indices: list[int], variant: Optional[str] = None) -> list[dict[str, str]]:
    rows = []
    for index in indices:
        config = {"environment": environment, "split": split, "index": index}
        if variant is not None:
            config["variant"] = variant
        rows.append({"prompt": f"{environment} {split} task {index}", "answer": "", "env_config": json.dumps(config)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--environment", required=True, help="owner/Name on openreward.ai, or the name on --url")
    parser.add_argument("--split", default="train")
    parser.add_argument("--variant", help="one environment of a server that serves several")
    parser.add_argument("--url", help="a self-hosted ORS server (default: openreward.ai, with OPENREWARD_API_KEY)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--num-tasks", type=int, help="the first N tasks (default: all)")
    which.add_argument("--indices", type=parse_indices, help="task indices, e.g. 0,5,10-19")
    parser.add_argument("--out", type=Path, required=True, help="parquet file to write")
    args = parser.parse_args()

    from openreward import OpenReward

    client = OpenReward(base_url=args.url) if args.url else OpenReward()
    try:
        environment = client.environments.get(args.environment, variant=args.variant, base_url=args.url)
        count = environment.num_tasks(args.split)
    finally:
        client.close()
    indices = args.indices if args.indices is not None else list(range(min(count, args.num_tasks or count)))
    outside = [index for index in indices if not 0 <= index < count]
    if outside:
        parser.error(f"{args.environment} has {count} {args.split} tasks; indices out of range: {outside[:10]}")

    import pandas as pd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(task_rows(args.environment, args.split, indices, args.variant)).to_parquet(args.out)
    print(f"[prepare_openreward_tasks] {len(indices)} of {count} {args.split} tasks -> {args.out}")


if __name__ == "__main__":
    main()
