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
"""In-process environments whose tools are ``AgentTool`` objects."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from PIL import Image

from ..protocol import AgentStatus, EnvOutcome, EnvReset, EnvStep, ToolCall
from ..tools.base import ToolRegistry


class LocalToolEnvironment:
    """Run tool calls with the stateless ``AgentTool`` objects of a ``ToolRegistry``.

    Subclasses build the task in ``reset`` (tools and source images) and choose the chat adapter. No
    coroutine here suspends, so batched rollouts keep their dispatch order (see ``AgentEnvironment``).
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        source_images: Sequence[Image.Image] = (),
    ) -> None:
        self.registry = registry if registry is not None else ToolRegistry([])
        self.source_images = list(source_images)

    async def reset(self, config: Mapping[str, Any], seed: int) -> EnvReset:
        raise NotImplementedError(f"{type(self).__name__} must build its task in reset()")

    async def step(self, call: ToolCall) -> EnvStep:
        return EnvStep(result=self.registry.execute(call, self.source_images))

    async def finalize(self, final_answer: Optional[str], status: AgentStatus) -> Optional[EnvOutcome]:
        return None

    async def close(self) -> None:
        return None

    def chat_adapter(self, processor: Any, reset: EnvReset, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must choose its chat adapter")
