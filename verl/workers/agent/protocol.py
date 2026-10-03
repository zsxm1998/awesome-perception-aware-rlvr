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

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence

from PIL import Image


@dataclass(frozen=True)
class AgentImageConfig:
    """One immutable image-preprocessing contract for an agent trajectory."""

    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    # ``None`` is unbounded. EasyR1's public rollout config uses 0 as the
    # equivalent sentinel, which is normalized here at the boundary.
    limit_images: Optional[int] = None
    # Lower pixel bound of tool observations (crops); ``None`` uses ``min_pixels``. Observations are resized
    # once when the tool returns them, so rollout and training see the same size.
    observation_min_pixels: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("min_pixels", "max_pixels", "observation_min_pixels"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer when provided")
        if self.min_pixels is not None and self.max_pixels is not None and self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels cannot exceed max_pixels")

        if isinstance(self.limit_images, bool):
            raise ValueError("limit_images must be a non-negative integer when provided")
        if self.limit_images == 0:
            object.__setattr__(self, "limit_images", None)
        elif self.limit_images is not None and (not isinstance(self.limit_images, int) or self.limit_images < 0):
            raise ValueError("limit_images must be a non-negative integer when provided")


@dataclass(frozen=True)
class AgentLoopConfig:
    """Budgets for one agentic response trajectory.

    ``max_response_tokens`` covers model actions and environment observations,
    but not the immutable input prompt. ``max_tokens_per_turn`` caps one model
    action independently from that full-trajectory budget.
    """

    max_tool_calls: int = 6
    max_response_tokens: int = 20480
    max_tokens_per_turn: int = 10240

    def __post_init__(self) -> None:
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if self.max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if self.max_tokens_per_turn <= 0:
            raise ValueError("max_tokens_per_turn must be positive")


class AgentStatus(str, Enum):
    ANSWERED = "answered"
    INVALID_FINAL_ANSWER = "invalid_final_answer"
    INVALID_TURN_END = "invalid_turn_end"
    NO_ACTION = "no_action"
    TOOL_CALL_LIMIT = "tool_call_limit"
    LENGTH_LIMIT = "length_limit"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    raw_text: str


@dataclass(frozen=True)
class ParsedAction:
    final_answer: Optional[str] = None
    final_answer_error: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_call_error: Optional[str] = None

    @property
    def has_tool_attempt(self) -> bool:
        return self.tool_call is not None or self.tool_call_error is not None


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    content: Mapping[str, Any]
    images: list[Image.Image] = field(default_factory=list)
    error_code: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional model-facing text for carrier-swap ablations. Structured
    # ``content`` remains available for diagnostics, while the observation
    # encoder emits this literal instead of serializing that mapping.
    observation_text: Optional[str] = None


@dataclass(frozen=True)
class GenerationRequest:
    prompt_token_ids: Sequence[int]
    images: Sequence[Image.Image]
    max_new_tokens: int
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenerationOutput:
    token_ids: Sequence[int]
    text: str
    finish_reason: Optional[str] = None
    # Backend-native termination detail. vLLM may return either a token ID or
    # a string, so keep the structured value instead of flattening it into a
    # human-readable error.
    stop_reason: Optional[Any] = None
    backend_text: Optional[str] = None


@dataclass(frozen=True)
class ObservationEncodingRequest:
    result: ToolResult


@dataclass(frozen=True)
class EncodedObservation:
    token_ids: Sequence[int]
    images: Sequence[Image.Image] = ()
    # Number of model-visible tokens after multimodal placeholder expansion.
    # ``None`` is allowed only for inference adapters that cannot preprocess
    # vision; training materialization must obtain the exact value.
    effective_token_count: Optional[int] = None
    # ``None`` means that the expanded model-side visual token count is not
    # available yet. It must not be reported as a real zero.
    visual_token_count: Optional[int] = None


@dataclass(frozen=True)
class EncodedPrompt:
    token_ids: Sequence[int]
    rendered_text: str


class GenerationBackend(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationOutput: ...


class ObservationEncoder(Protocol):
    async def encode(self, request: ObservationEncodingRequest) -> EncodedObservation: ...


@dataclass
class AgentStep:
    model_text: str
    model_token_ids: list[int]
    finish_reason: Optional[str] = None
    stop_reason: Optional[Any] = None
    backend_text: Optional[str] = None
    reasoning_prefilled: bool = False
    is_finalization_turn: bool = False
    turn_end_error: Optional[str] = None
    expected_termination_token_ids: list[int] = field(default_factory=list)
    actual_termination_tail_ids: list[int] = field(default_factory=list)
    final_answer_error: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    tool_call_error: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    observation_token_ids: list[int] = field(default_factory=list)
    observation_committed: bool = False


@dataclass
class AgentMetrics:
    tool_call_attempts: int = 0
    tool_execution_successes: int = 0
    # A success is reward-eligible only after its complete observation has
    # been committed to the trajectory.
    tool_call_successes: int = 0
    # Errors describe parse/execution outcomes and are counted even if their
    # feedback observation cannot be committed. They are diagnostic only and
    # must not be interpreted as observation state by reward code.
    tool_call_errors: int = 0
    tool_internal_errors: int = 0
    image_idx_errors: int = 0
    invalid_final_answers: int = 0
    turn_end_errors: int = 0
    action_tokens: int = 0
    raw_observation_tokens: int = 0
    observation_tokens: int = 0
    # Starts at a known zero and becomes unavailable once any committed
    # visual observation lacks an expanded model-side token count.
    visual_tokens: Optional[int] = 0
    visual_observations: int = 0
    truncated: bool = False
    # These configured limits can bind simultaneously when the remaining
    # trajectory budget equals the per-turn generation budget.
    turn_length_limited: bool = False
    trajectory_length_limited: bool = False
    # vLLM reports the model context window and requested generation cap with
    # the same finish reason. Stopping before the requested cap is sufficient
    # evidence that the context window bound first.
    context_window_limited: bool = False
    tool_call_limit_reached: bool = False


@dataclass
class AgentTrajectory:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    source_images: list[Image.Image]
    observation_images: list[Image.Image]
    steps: list[AgentStep]
    status: AgentStatus
    metrics: AgentMetrics
    final_answer: Optional[str] = None
    effective_prompt_tokens: Optional[int] = None
    prompt_visual_tokens: Optional[int] = None
    effective_response_tokens: Optional[int] = None
    response_token_budget: Optional[int] = None
    image_config: AgentImageConfig = field(default_factory=AgentImageConfig)
    # Exact native assistant suffix derived from the active chat template.
    # Evaluation uses it to turn the final action into a standalone,
    # score-facing reply without deleting other model-specific special tokens.
    assistant_termination_token_ids: tuple[int, ...] = ()

    @property
    def all_images(self) -> list[Image.Image]:
        return [*self.source_images, *self.observation_images]

    @property
    def input_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.response_ids]


def derive_agent_seed(
    base_seed: int,
    sample_index: int,
    rollout_index: int,
    turn_index: int = 0,
) -> int:
    """Derive a stable, process-independent seed for one agent generation."""

    values = (base_seed, sample_index, rollout_index, turn_index)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("agent seed components must be non-negative integers")
    payload = ":".join(str(value) for value in values).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


@dataclass(frozen=True)
class AgentSeedContext:
    """Stable sampling identity shared by every turn of one group member."""

    base_seed: int
    sample_index: int
    rollout_index: int

    def __post_init__(self) -> None:
        derive_agent_seed(self.base_seed, self.sample_index, self.rollout_index)

    def for_turn(self, turn_index: int) -> int:
        return derive_agent_seed(
            self.base_seed,
            self.sample_index,
            self.rollout_index,
            turn_index,
        )
