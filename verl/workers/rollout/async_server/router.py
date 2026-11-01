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
"""Which rollout server a model turn goes to.

As verl v0.9.1's ``GlobalRequestLoadBalancer`` (``verl/workers/rollout/router.py``): every turn of a trajectory goes to
the server of its first turn, whose prefix cache holds the trajectory's context; a new trajectory goes to the server
with the fewest requests in flight, ties broken at random (verl #7613). One router serves all agent workers, so the
counts are global. With ``group_affinity`` a group's new trajectories go to the server of its first trajectory
unless that server has more than ``affinity_slack`` requests in flight above the least busy one (an option of this
repository, off by default: no framework found pins a group's multi-turn trajectories to one server; ROLL pins the
samples of a prompt in its single-turn pipeline, PromptAffinityRouter).
"""

import random
from typing import Hashable, Optional


class ReplicaRouter:
    def __init__(self, num_replicas: int, *, group_affinity: bool = False, affinity_slack: int = 0, seed: int = 0):
        if num_replicas <= 0:
            raise ValueError("the router needs at least one replica")
        self.inflight = [0] * num_replicas
        self.group_affinity = group_affinity
        self.affinity_slack = affinity_slack
        self.random = random.Random(seed)
        self.sticky: dict[Hashable, int] = {}
        self.group_home: dict[Hashable, int] = {}

    def acquire(self, sticky_key: Hashable, group_key: Optional[Hashable] = None) -> int:
        """The replica of the next turn of trajectory ``sticky_key`` (of group ``group_key``); counts it in flight
        until ``release``."""
        replica = self.sticky.get(sticky_key)
        if replica is None:
            least = min(self.inflight)
            home = self.group_home.get(group_key) if self.group_affinity else None
            if home is not None and self.inflight[home] <= least + self.affinity_slack:
                replica = home
            else:
                replica = self.random.choice([index for index, count in enumerate(self.inflight) if count == least])
            if self.group_affinity and group_key is not None:
                self.group_home.setdefault(group_key, replica)
            self.sticky[sticky_key] = replica
        self.inflight[replica] += 1
        return replica

    def release(self, replica: int) -> None:
        if self.inflight[replica] <= 0:
            raise RuntimeError(f"replica {replica} released more requests than it took")
        self.inflight[replica] -= 1

    def reset(self) -> None:
        """Forget the trajectories of the last rollout (a new rollout batch starts)."""
        if any(self.inflight):
            raise RuntimeError(f"a rollout batch starts while requests are in flight: {self.inflight}")
        self.sticky.clear()
        self.group_home.clear()

    def snapshot(self) -> list[int]:
        return list(self.inflight)
