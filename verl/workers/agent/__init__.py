# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from .chat import NativeToolChatAdapter, render_deepeyes_system_prompt
from .loop import AgentLoop
from .protocol import (
    AgentImageConfig,
    AgentLoopConfig,
    AgentSeedContext,
    AgentStatus,
    AgentTrajectory,
    EncodedObservation,
    EncodedPrompt,
    GenerationOutput,
    GenerationRequest,
    ObservationEncodingRequest,
    derive_agent_seed,
)
from .tool_call import NativeToolCallCodec
from .trajectory import (
    AgentTrajectoryMaterializer,
    AgentTrajectoryRejected,
    MaterializedAgentTrajectory,
    collate_materialized_agent_trajectories,
)


__all__ = [
    "AgentImageConfig",
    "AgentLoop",
    "AgentLoopConfig",
    "AgentSeedContext",
    "AgentStatus",
    "AgentTrajectory",
    "AgentTrajectoryMaterializer",
    "AgentTrajectoryRejected",
    "EncodedObservation",
    "EncodedPrompt",
    "GenerationOutput",
    "GenerationRequest",
    "MaterializedAgentTrajectory",
    "NativeToolCallCodec",
    "NativeToolChatAdapter",
    "ObservationEncodingRequest",
    "collate_materialized_agent_trajectories",
    "derive_agent_seed",
    "render_deepeyes_system_prompt",
]
