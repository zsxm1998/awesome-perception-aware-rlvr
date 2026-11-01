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
import math
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
    # Tool calls run per turn: the first ones of a turn, in the order written; the others get an error result.
    max_tool_calls_per_turn: int = 1

    def __post_init__(self) -> None:
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if (
            isinstance(self.max_tool_calls_per_turn, bool)
            or not isinstance(self.max_tool_calls_per_turn, int)
            or self.max_tool_calls_per_turn < 1
        ):
            raise ValueError("max_tool_calls_per_turn must be a positive integer")
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
    # The environment ended the episode after a tool call (EnvStep.done).
    ENVIRONMENT_DONE = "environment_done"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    raw_text: str


@dataclass(frozen=True)
class ParsedToolCall:
    """One ``<tool_call>`` block of a turn: the call, or why it is malformed."""

    call: Optional[ToolCall] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ParsedAction:
    final_answer: Optional[str] = None
    final_answer_error: Optional[str] = None
    # The turn's tool-call blocks in the order written (one malformed entry for tags without a complete block).
    tool_calls: tuple[ParsedToolCall, ...] = ()

    @property
    def has_tool_attempt(self) -> bool:
        return bool(self.tool_calls)

    @property
    def tool_call(self) -> Optional[ToolCall]:
        """The call of a turn with exactly one well-formed block."""
        return self.tool_calls[0].call if len(self.tool_calls) == 1 else None

    @property
    def tool_call_error(self) -> Optional[str]:
        """Why the only block of a turn is malformed."""
        return self.tool_calls[0].error if len(self.tool_calls) == 1 else None


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
    # Model-facing content in order, as chat parts: {"type": "text", "text": ...} and {"type": "image"}
    # placeholders filled from ``images``. Set by environments that write their own observations
    # (rendered by EnvironmentChatAdapter); None for the DeepEyes tool.
    parts: Optional[list[dict[str, Any]]] = None


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
    """The results of a turn's tool calls in call order, rendered as the one observation the model reads next.
    A request gives ``result`` (a turn with one call) or ``results``; ``results`` is then always set, ``result``
    only for one result. A turn may have several results with any per-turn limit: the calls beyond it get
    error results, so every encoder renders ``results``."""

    result: Optional[ToolResult] = None
    results: tuple[ToolResult, ...] = ()

    def __post_init__(self) -> None:
        if self.result is not None:
            if self.results and tuple(self.results) != (self.result,):
                raise ValueError("give result or results, not both")
            object.__setattr__(self, "results", (self.result,))
        else:
            if not self.results:
                raise ValueError("an observation needs at least one tool result")
            object.__setattr__(self, "results", tuple(self.results))
            if len(self.results) == 1:
                object.__setattr__(self, "result", self.results[0])


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
class EnvReset:
    """The task one episode starts from.

    ``messages`` are chat messages whose image parts are ``{"type": "image"}`` placeholders, filled from
    ``images`` in order. Empty ``tool_schemas`` mean that the model answers in one turn without tools.
    ``max_images_per_step`` bounds the images one step returns; it sizes the per-prompt image limit.
    ``state_key`` optionally names the initial state, as ``EnvStep.state_key`` names the state after a call.
    """

    messages: list[dict[str, Any]]
    images: list[Image.Image]
    tool_schemas: list[Mapping[str, Any]] = field(default_factory=list)
    max_images_per_step: int = 1
    state_key: Optional[str] = None


@dataclass
class EnvStep:
    """The environment's response to one well-formed tool call.

    ``done`` ends the episode after this step (its observation is not shown to the model). ``reward`` is what
    this call earned: it is added to the episode's return, the reward function's score being the reward of the
    last turn. ``state_key`` names the environment state after this call, the state the next action is taken in,
    so that actions of different rollouts taken in the same state can be grouped (GiGPO with
    ``step_state_key=environment``); the first actions of a group share the initial state (``reset`` gets one
    seed per group). ``cost`` and ``info`` are recorded for diagnostics.
    """

    result: ToolResult
    done: bool = False
    reward: Optional[float] = None
    state_key: Optional[str] = None
    cost: float = 0.0
    info: Mapping[str, Any] = field(default_factory=dict)


def checked_step_reward(reward: Any) -> Optional[float]:
    """``EnvStep.reward`` as a float, or None; anything but a finite number is an environment error (it would make
    the return NaN or infinite)."""
    if reward is None:
        return None
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"EnvStep.reward must be a finite number or None, got {reward!r}")
    return float(reward)


@dataclass
class EnvOutcome:
    """What an environment reports when its episode ends.

    ``reward`` None means that the environment does not score the episode (a reward function that reads the answer
    scores it, e.g. DeepEyes over HTTP); the environment reward function and evaluation scorer then fail rather than
    count it as 0. ``components`` and ``metrics`` are finite numbers, logged as ``reward/env/<name>``; an episode may
    report only the names that apply to it. ``details`` is any JSON object, kept verbatim in the trajectory records
    (not aggregated, not given to reward functions). ``episode_id`` is the environment's own name of the episode
    (remote_http: the server's episode_id), to match the records with the environment's logs.
    """

    reward: Optional[float]
    components: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    episode_id: Optional[str] = None


class AgentEnvironment(Protocol):
    """One episode of an environment; every trajectory gets its own instance.

    In the batched rollout the event loop switches between trajectories only at awaits that suspend.
    In-process environments therefore must not suspend (no awaiting futures, threads, I/O or sleeps): an
    ``await`` on them completes at once and leaves the scheduler's batches as they were.
    ``chat_adapter`` returns the encoder that renders this environment's prompt and observations.
    ``step`` may take a keyword argument ``tool_calls_used``: the agent loop then passes the number of the
    trajectory's calls that counted towards ``max_tool_calls`` before this one, malformed calls included (they never
    reach the environment), so that the environment can tell the model how many calls remain.
    """

    async def reset(self, config: Mapping[str, Any], seed: int) -> EnvReset: ...

    async def step(self, call: ToolCall) -> EnvStep: ...

    async def finalize(self, final_answer: Optional[str], status: AgentStatus) -> Optional[EnvOutcome]: ...

    async def close(self) -> None: ...

    def chat_adapter(
        self,
        processor: Any,
        reset: EnvReset,
        *,
        max_tool_calls: int,
        image_config: AgentImageConfig,
        image_preprocessor: Any,
    ) -> Any: ...


# Error codes of the results that the agent loop gives to calls it does not run.
TOO_MANY_TOOL_CALLS_IN_TURN = "too_many_tool_calls_in_turn"
TOOL_CALL_LIMIT_REACHED = "tool_call_limit_reached"


@dataclass
class AgentCall:
    """One tool-call block of a model turn and what followed from it.

    ``tool_call`` is None for a malformed block (``tool_call_error`` says why). ``tool_result`` is what the model is
    shown for it: the environment's result, or the loop's error for a malformed block or a call it did not run
    (beyond ``max_tool_calls_per_turn``, ``TOO_MANY_TOOL_CALLS_IN_TURN``, or beyond the trajectory's remaining calls,
    ``TOOL_CALL_LIMIT_REACHED``); None for a call left after the environment ended the episode. The ``env_*``,
    ``state_key`` and ``cost`` fields are what the environment reported (EnvStep); they keep their defaults when the
    call did not reach the environment, which leaves its state unchanged.
    """

    tool_call: Optional[ToolCall] = None
    tool_call_error: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    executed: bool = False
    env_reward: Optional[float] = None
    state_key: Optional[str] = None
    cost: float = 0.0
    env_done: bool = False
    env_info: dict[str, Any] = field(default_factory=dict)


class OneCallView:
    """The fields of a turn's only tool call (``calls``), as turns had before several calls per turn: None or the
    default for a turn without a call; a turn with several calls has to be read through ``calls``."""

    calls: list[AgentCall]

    def _only_call(self) -> Optional[AgentCall]:
        if len(self.calls) > 1:
            raise ValueError("the turn has several tool calls; read them from .calls")
        return self.calls[0] if self.calls else None

    @property
    def tool_call(self) -> Optional[ToolCall]:
        call = self._only_call()
        return None if call is None else call.tool_call

    @property
    def tool_call_error(self) -> Optional[str]:
        call = self._only_call()
        return None if call is None else call.tool_call_error

    @property
    def tool_result(self) -> Optional[ToolResult]:
        call = self._only_call()
        return None if call is None else call.tool_result

    @property
    def env_reward(self) -> Optional[float]:
        call = self._only_call()
        return None if call is None else call.env_reward

    @property
    def state_key(self) -> Optional[str]:
        call = self._only_call()
        return None if call is None else call.state_key

    @property
    def cost(self) -> float:
        call = self._only_call()
        return 0.0 if call is None else call.cost

    @property
    def env_done(self) -> bool:
        call = self._only_call()
        return False if call is None else call.env_done

    @property
    def env_info(self) -> dict[str, Any]:
        call = self._only_call()
        return {} if call is None else call.env_info


def turn_env_reward(calls: Sequence[AgentCall]) -> Optional[float]:
    """What the environment's rewards of a turn's calls add up to (None when it gave none)."""
    rewards = [call.env_reward for call in calls if call.env_reward is not None]
    return float(sum(rewards)) if rewards else None


@dataclass
class AgentStep(OneCallView):
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
    # The turn's tool-call blocks in the order written (AgentCall).
    calls: list[AgentCall] = field(default_factory=list)
    observation_token_ids: list[int] = field(default_factory=list)
    observation_committed: bool = False
    # How many of the trajectory's observation images this step's committed observation added.
    observation_image_count: int = 0


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
    # tool_call_errors = malformed_tool_calls + the calls not run for the per-turn limit or the remaining calls
    # + the calls the environment answered with success: false. malformed_tool_calls are malformed calls that
    # count towards max_tool_calls (the environment receives tool_call_attempts - malformed_tool_calls calls).
    # unrun_tool_calls are calls the loop did not run: beyond max_tool_calls_per_turn or the remaining calls (a
    # malformed one among them too), each with an error result, and calls within the per-turn limit after one
    # that ended the episode, which get no result. Calls written in the answer-only turn are in neither.
    malformed_tool_calls: int = 0
    unrun_tool_calls: int = 0
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
    # What the environment reported at the end of the episode (None: it reports nothing).
    env_outcome: Optional[EnvOutcome] = None
    # The environment's name of the initial state (EnvReset.state_key), if it gives one.
    initial_state_key: Optional[str] = None

    @property
    def all_images(self) -> list[Image.Image]:
        return [*self.source_images, *self.observation_images]

    @property
    def input_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.response_ids]


def environment_report(
    steps: Sequence[Any], outcome: Optional[EnvOutcome], initial_state_key: Optional[str] = None
) -> dict[str, Any]:
    """What an environment reported during an episode, as trajectory records keep it (training generations,
    evaluation outputs, SFT exports): ``initial_state_key`` when it named the initial state, ``env_steps`` for the
    tool calls it answered with a state, cost, reward, done or info (by turn and call index), and ``env_outcome``. Empty for an environment
    that reports none of them (DeepEyes)."""

    env_steps = [
        {
            "turn_index": turn_index,
            "call_index": call_index,
            "tool": call.tool_call.name,
            "state_key": call.state_key,
            "cost": call.cost,
            "reward": call.env_reward,
            "done": call.env_done,
            "info": dict(call.env_info),
        }
        for turn_index, step in enumerate(steps)
        for call_index, call in enumerate(step.calls)
        if call.executed
        and (call.state_key is not None or call.cost or call.env_reward is not None or call.env_done or call.env_info)
    ]
    report: dict[str, Any] = {}
    if initial_state_key is not None:
        report["initial_state_key"] = initial_state_key
    if env_steps:
        report["env_steps"] = env_steps
    if outcome is not None:
        report["env_outcome"] = {
            "reward": outcome.reward,
            "components": dict(outcome.components),
            "metrics": dict(outcome.metrics),
            "details": dict(outcome.details),
            "episode_id": outcome.episode_id,
        }
    return report


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


def derive_environment_seed(base_seed: int, sample_index: int) -> int:
    """Seed of ``AgentEnvironment.reset``.

    It depends on the prompt only, not on the rollout index or the retry, so all rollouts of a group and
    their retries start from the same task state.
    """

    values = (base_seed, sample_index)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("environment seed components must be non-negative integers")
    payload = f"env:{base_seed}:{sample_index}".encode("ascii")
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
