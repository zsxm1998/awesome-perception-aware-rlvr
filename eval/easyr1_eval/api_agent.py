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
"""--backend api: models behind an OpenAI-compatible Chat Completions API in agent environments.

The environment, its tasks, the tool-call limit and the scorer are those of ``--agent-profile environment``
with a local model; the policy is a message-level loop (``verl/workers/agent/api_loop.py``). Finished
trajectories are appended to ``<output_dir>/api_trajectories/<benchmark>.jsonl`` and reused by a later run
with the same settings, so a failed or interrupted run does not pay for them again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from verl.workers.agent.api_loop import (
    APIBudgetExceeded,
    APIEpisode,
    APISampling,
    ChatCompletionsClient,
    run_api_episode,
)
from verl.workers.agent.environments import EnvironmentContext, environment_factory, validate_environment_options
from verl.workers.agent.protocol import (
    AgentLoopConfig,
    AgentSeedContext,
    AgentStatus,
    derive_environment_seed,
    environment_report,
)

from .agent_common import AgenticSampleError, failed_trajectory_output, stable_sample_index
from .schemas import EvalSample, GenerationConfig, GenerationOutput


API_RUNNER_VERSION = 1
_TOOL_CALL_BLOCK = "<tool_call>"


class AgenticAPIBackend:
    """Run agent-environment tasks with a model behind an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key_env: str,
        agent_config: AgentLoopConfig,
        agent_environment: str,
        agent_environment_kwargs: dict[str, Any] | None = None,
        concurrency: int = 8,
        request_retries: int = 5,
        timeout_s: float = 300.0,
        extra_body: dict[str, Any] | None = None,
        max_tokens_field: str = "max_tokens",
        max_total_tokens: int | None = None,
        pixels: tuple[int | None, int | None] = (None, None),
        tool_images: str = "user",
        output_dir: Any = None,
        transport: Any = None,
    ):
        if not base_url:
            raise ValueError("--backend api needs --api-base-url (or OPENAI_BASE_URL)")
        validate_environment_options(agent_environment, agent_environment_kwargs)
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.agent_config = agent_config
        self.agent_environment = agent_environment
        self.agent_environment_kwargs = dict(agent_environment_kwargs or {})
        self.concurrency = concurrency
        self.request_retries = request_retries
        self.timeout_s = timeout_s
        self.extra_body = dict(extra_body or {})
        self.max_tokens_field = max_tokens_field
        self.max_total_tokens = max_total_tokens
        self.pixels = tuple(pixels)
        self.tool_images = tool_images
        self.cache_dir = None if output_dir is None else Path(output_dir) / "api_trajectories"
        self.tokens_used = 0
        self._cache: dict[str, dict[str, Any]] = {}
        self._transport = transport  # tests replace the network

    def generate(self, samples: list[EvalSample], config: GenerationConfig) -> list[list[GenerationOutput]]:
        return asyncio.run(self._generate_async(samples, config))

    async def _generate_async(
        self, samples: list[EvalSample], config: GenerationConfig
    ) -> list[list[GenerationOutput]]:
        if config.max_new_tokens != self.agent_config.max_response_tokens:
            raise ValueError(
                "agentic generation and loop disagree on response budget: "
                f"{config.max_new_tokens} != {self.agent_config.max_response_tokens}"
            )
        if config.top_k is not None:
            raise ValueError(
                "top_k is not a Chat Completions parameter; pass it in --api-extra-body if the API takes it"
            )
        sampling = APISampling(temperature=config.temperature, top_p=config.top_p)
        client = ChatCompletionsClient(
            self.base_url,
            self.model,
            os.environ.get(self.api_key_env) or None,
            timeout_s=self.timeout_s,
            retries=self.request_retries,
            max_concurrency=self.concurrency,
            extra_body=self.extra_body,
            max_tokens_field=self.max_tokens_field,
            max_total_tokens=None if self.max_total_tokens is None else self.max_total_tokens - self.tokens_used,
            transport=self._transport,
        )
        context = EnvironmentContext(
            processor=None, image_preprocessor=lambda image: image, max_tool_calls=self.agent_config.max_tool_calls
        )
        make_environment = environment_factory(self.agent_environment, context, self.agent_environment_kwargs)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(sample: EvalSample, rollout_index: int) -> GenerationOutput:
            env_config = sample.extra_info.get("env_config")
            if not isinstance(env_config, dict):
                raise AgenticSampleError("the environment profile needs extra_info['env_config'] (a JSON object)")
            key = self._cache_key(sample, env_config, rollout_index, config)
            cached = self._cached(sample.benchmark, key)
            if cached is not None:
                return cached
            sample_index = stable_sample_index(sample)
            async with semaphore:  # environments started at a time (the client limits its requests itself)
                episode = await run_api_episode(
                    make_environment(),
                    env_config,
                    client,
                    env_seed=derive_environment_seed(config.seed, sample_index),
                    config=self.agent_config,
                    sampling=sampling,
                    seed_context=AgentSeedContext(config.seed, sample_index, rollout_index),
                    pixels=self.pixels,
                    tool_images=self.tool_images,
                )
            output = api_episode_to_generation_output(episode)
            self._store(sample.benchmark, key, output)
            return output

        identities = [(position, k) for position in range(len(samples)) for k in range(config.num_samples)]
        try:
            results = await asyncio.gather(
                *(run_one(samples[position], k) for position, k in identities), return_exceptions=True
            )
        finally:
            self.tokens_used += client.total_tokens
            await client.aclose()
            await context.aclose()
        budget = [result for result in results if isinstance(result, APIBudgetExceeded)]
        if budget:
            raise RuntimeError(f"{budget[0]}; finished trajectories are kept for a rerun")
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures and len(failures) == len(results):
            raise RuntimeError(
                "every trajectory of the batch failed: " + "; ".join(sorted({str(error) for error in failures})[:3])
            )
        grouped: list[list[GenerationOutput]] = [[] for _ in samples]
        for (position, _), result in zip(identities, results):
            grouped[position].append(failed_trajectory_output(result) if isinstance(result, BaseException) else result)
        return grouped

    def _cache_key(self, sample: EvalSample, env_config: dict, rollout_index: int, config: GenerationConfig) -> str:
        payload = {
            "version": API_RUNNER_VERSION,
            "model": self.model,
            "base_url": self.base_url,
            "extra_body": self.extra_body,
            "max_tokens_field": self.max_tokens_field,
            "pixels": list(self.pixels),
            "tool_images": self.tool_images,
            "agent_config": self.agent_config.__dict__,
            "environment": self.agent_environment,
            "environment_kwargs": self.agent_environment_kwargs,
            "generation": {"temperature": config.temperature, "top_p": config.top_p, "seed": config.seed},
            "sample": [sample.benchmark, sample.sample_id, env_config, rollout_index],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _cache_path(self, benchmark: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{benchmark}.jsonl"

    def _cached(self, benchmark: str, key: str) -> GenerationOutput | None:
        path = self._cache_path(benchmark)
        if path is None:
            return None
        if benchmark not in self._cache:
            entries: dict[str, Any] = {}
            if path.is_file():
                with path.open(encoding="utf-8") as file:
                    for line in file:
                        if line.strip():
                            entry = json.loads(line)
                            entries[entry["key"]] = entry["output"]
            self._cache[benchmark] = entries
        output = self._cache[benchmark].get(key)
        return None if output is None else GenerationOutput(**output)

    def _store(self, benchmark: str, key: str, output: GenerationOutput) -> None:
        path = self._cache_path(benchmark)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = output.__dict__
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"key": key, "output": record}, ensure_ascii=False) + "\n")
        self._cache.setdefault(benchmark, {})[key] = record


def api_episode_to_generation_output(episode: APIEpisode) -> GenerationOutput:
    """The score-facing final reply and the trajectory's diagnostics, as for local agent runs."""
    metrics = episode.metrics
    final = episode.turns[-1] if episode.turns else None
    text = "" if final is None else final.content
    if _TOOL_CALL_BLOCK in text:
        text = text[: text.index(_TOOL_CALL_BLOCK)]
    agent = {
        "status": episode.status.value,
        "final_answer": episode.final_answer,
        "num_steps": len(episode.turns),
        "tool_call_attempts": metrics.tool_call_attempts,
        "tool_execution_successes": metrics.tool_execution_successes,
        "tool_call_successes": metrics.tool_call_successes,
        "tool_call_errors": metrics.tool_call_errors,
        "tool_internal_errors": metrics.tool_internal_errors,
        "invalid_final_answers": metrics.invalid_final_answers,
        "completion_tokens": metrics.action_tokens,
        "visual_observations": metrics.visual_observations,
        "truncated": metrics.truncated,
        "turn_length_limited": metrics.turn_length_limited,
        "trajectory_length_limited": metrics.trajectory_length_limited,
        "tool_call_limit_reached": metrics.tool_call_limit_reached,
        "no_action": episode.status == AgentStatus.NO_ACTION,
        "invalid_answer": episode.status == AgentStatus.INVALID_FINAL_ANSWER,
        "api_usage": {
            "requests": len(episode.turns),
            "prompt_tokens": sum(turn.prompt_tokens or 0 for turn in episode.turns),
            "completion_tokens": sum(turn.completion_tokens or 0 for turn in episode.turns),
        },
        # the conversation without images: each reply and what its tool call got
        "turns": [
            {
                "content": turn.content,
                "reasoning": turn.reasoning,
                "finish_reason": turn.finish_reason,
                "native_tool_call": turn.native_tool_call,
                "tool_call": (
                    None
                    if turn.tool_call is None
                    else {"name": turn.tool_call.name, "arguments": dict(turn.tool_call.arguments)}
                ),
                "tool_call_error": turn.tool_call_error,
                "tool_success": None if turn.tool_result is None else turn.tool_result.success,
                "tool_error_code": None if turn.tool_result is None else turn.tool_result.error_code,
            }
            for turn in episode.turns
        ],
    }
    agent.update(environment_report(episode.turns, episode.env_outcome))
    return GenerationOutput(
        text=text.strip(),
        finish_reason="length" if episode.status == AgentStatus.LENGTH_LIMIT else (final and final.finish_reason),
        token_count=None if final is None else final.completion_tokens,
        truncated=metrics.truncated,
        diagnostics={"agent": agent},
    )
