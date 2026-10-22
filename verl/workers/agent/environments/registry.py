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

import asyncio
import hashlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Mapping, Optional

from ..protocol import AgentEnvironment


_ENVIRONMENTS: dict[str, type] = {}
_LOADED_FILES: dict[str, Any] = {}


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


def environment_class(name: str) -> type:
    """The class of a registered name, or of ``path/to/file.py:ClassName`` (loaded once per process, so a
    user's environment class reaches the rollout workers through the configuration alone)."""

    cls = _ENVIRONMENTS.get(name)
    if cls is not None:
        return cls
    if ":" not in name:
        raise ValueError(
            f"unknown agent environment {name!r}; registered: {registered_environments()}, "
            "or give path/to/file.py:ClassName"
        )
    return class_from_file(name, what="agent environment")


def class_from_file(spec: str, *, what: str = "class") -> type:
    """The class of ``path/to/file.py:ClassName``; each file is loaded once per process."""

    path, class_name = spec.rsplit(":", 1)
    path = os.path.abspath(path)
    module = _LOADED_FILES.get(path)
    if module is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{what} file not found: {path}")
        module_name = "agent_environment_" + hashlib.sha1(path.encode()).hexdigest()[:12]
        spec_ = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec_)
        sys.modules[module_name] = module
        spec_.loader.exec_module(module)
        _LOADED_FILES[path] = module
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        raise ValueError(f"{path} has no class {class_name!r}")
    return cls


@dataclass
class EnvironmentContext:
    """Resources of one rollout batch, shared by the environments of its trajectories.

    ``image_preprocessor`` is the batch's model-side image resize (the vLLM image cache), so an environment
    that reasons about the model's view of an image (e.g. pixel coordinates) sees the frames the policy sees.
    ``max_tool_calls`` is the configured tool-call limit of a trajectory. ``shared`` keeps objects that the
    environments of the batch share (an HTTP client, a concurrency limit); ``run_cleanup`` runs an episode's
    cleanup so that cancelling its trajectory (when another trajectory of the batch fails) does not cut it off.
    The batch owner calls ``aclose`` inside its event loop: it waits for the cleanups, then closes the shared
    objects.
    """

    processor: Any
    image_preprocessor: Callable[[Any], Any]
    max_tool_calls: int = 0
    _shared: dict[Hashable, Any] = field(default_factory=dict, repr=False)
    _cleanups: set = field(default_factory=set, repr=False)
    _closed: bool = field(default=False, repr=False)

    def shared(self, key: Hashable, create: Callable[[], Any]) -> Any:
        if self._closed:
            raise RuntimeError("the environment context of this rollout batch is closed")
        if key not in self._shared:
            self._shared[key] = create()
        return self._shared[key]

    async def run_cleanup(self, cleanup: Awaitable[Any]) -> None:
        """Await ``cleanup``, which keeps running to the end even if the awaiting trajectory is cancelled."""
        task = asyncio.ensure_future(cleanup)
        self._cleanups.add(task)
        task.add_done_callback(self._cleanups.discard)
        await asyncio.shield(task)

    async def aclose(self) -> None:
        while self._cleanups:  # cleanups of cancelled trajectories may still be running
            await asyncio.gather(*list(self._cleanups), return_exceptions=True)
        self._closed = True
        resources, self._shared = list(self._shared.values()), {}
        for resource in resources:
            close = getattr(resource, "aclose", None)
            if close is not None:
                await close()


def validate_environment_options(name: str, kwargs: Optional[Mapping[str, Any]] = None) -> None:
    """Check at configuration time that ``name`` is registered and accepts ``kwargs`` (values are checked
    when the rollout builds the environment)."""

    cls = environment_class(name)
    try:
        inspect.signature(cls).bind(None, **dict(kwargs or {}))
    except TypeError as exc:
        raise ValueError(f"invalid options for agent environment {name!r}: {exc}") from exc


def environment_factory(
    name: str,
    context: EnvironmentContext,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[[], AgentEnvironment]:
    """Return a builder of fresh environments, one per trajectory.

    One instance is built here and discarded, so unknown names, unknown arguments and invalid values fail
    before any rollout starts. Constructors must therefore not acquire resources (do that in ``reset``).
    """

    cls = environment_class(name)
    kwargs = dict(kwargs or {})
    cls(context, **kwargs)

    def build() -> AgentEnvironment:
        return cls(context, **kwargs)

    return build
