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
"""Names of agent environments and how one rollout batch builds them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from ..protocol import AgentEnvironment


_ENVIRONMENTS: dict[str, type] = {}


def register_environment(name: str) -> Callable[[type], type]:
    """Register an environment class under ``name``; it is built as ``cls(context, **kwargs)``."""

    def decorator(cls: type) -> type:
        registered = _ENVIRONMENTS.get(name)
        if registered is not None and registered is not cls:
            raise ValueError(f"agent environment {name!r} is already registered by {registered.__qualname__}")
        _ENVIRONMENTS[name] = cls
        return cls

    return decorator


def registered_environments() -> list[str]:
    return sorted(_ENVIRONMENTS)


@dataclass(frozen=True)
class EnvironmentContext:
    """Resources of one rollout batch, shared by the environments of its trajectories.

    ``image_preprocessor`` is the batch's model-side image resize (the vLLM image cache), so an environment
    that reasons about the model's view of an image (e.g. pixel coordinates) sees the frames the policy sees.
    """

    processor: Any
    image_preprocessor: Callable[[Any], Any]


def environment_factory(
    name: str,
    context: EnvironmentContext,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[[], AgentEnvironment]:
    """Return a builder of fresh environments, one per trajectory.

    One instance is built here and discarded, so unknown names, unknown arguments and invalid values fail
    before any rollout starts. Constructors must therefore not acquire resources (do that in ``reset``).
    """

    cls = _ENVIRONMENTS.get(name)
    if cls is None:
        raise ValueError(f"unknown agent environment {name!r}; registered: {registered_environments()}")
    kwargs = dict(kwargs or {})
    cls(context, **kwargs)

    def build() -> AgentEnvironment:
        return cls(context, **kwargs)

    return build
