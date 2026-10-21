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
"""
Rollout config
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    interaction_mode: str = "one_shot"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    # vLLM's multimodal processor cache (GB, per engine). Zero disables it, which
    # is right for one-shot rollout where every image is submitted exactly once.
    # Agentic rollout resubmits the whole image set on every turn under stable
    # ``multi_modal_uuids``, so a non-zero budget turns those repeats into hits.
    mm_processor_cache_gb: float = 0.0
    agent_max_tool_calls: int = 6
    agent_max_tokens_per_turn: int = 10240
    agent_max_batch_images: int = 0
    agent_max_trajectory_retries: int = 2
    # Coordinate convention of the zoom-in tool: "norm1000" (Qwen3-VL, InternVL), "pixel" (absolute
    # coordinates in the resized frame the model sees; Qwen2-VL / Qwen2.5-VL) or "auto" (from the
    # model's config.json). Resolved during config validation.
    agent_bbox_format: str = "auto"
    # Lower pixel bound of the zoom-in tool's crops (None: data.min_pixels); DeepEyes uses 3136, so a small crop
    # stays small instead of being upscaled to data.min_pixels.
    agent_observation_min_pixels: Optional[int] = None
    # "native": the method contract in data.system_prompt with the tool schema rendered by the chat template;
    # "official": DeepEyes' own prompts from the data (tool schema in the system prompt text) and its tool response
    # format. Either way, rows whose env_name column is empty are rolled out once without tools.
    agent_prompt_style: str = "native"
    # The environment of agentic rollouts (docs/agent_environments.md): a registered name ("deepeyes",
    # "remote_http"), path/to/file.py:ClassName for your own class, or null for "deepeyes"; the options of
    # its constructor (remote_http: url, timeout_s, retries, ...); and the data column holding each row's
    # environment configuration, a JSON object given to reset (not used by "deepeyes", whose task is the
    # row's prompt and images).
    agent_environment: Optional[str] = None
    agent_environment_kwargs: dict[str, Any] = field(default_factory=dict)
    agent_environment_config_key: str = "env_config"
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)
    max_dynamic_patch: Optional[int] = field(default=None, init=False)  # copied from worker.actor.model
    # on-policy distillation: forbid sampling ids at or beyond len(tokenizer) (the padding rows of the LM head)
    ban_ids_beyond_tokenizer: bool = field(default=False, init=False)

    def post_init(self):
        if self.interaction_mode not in {"one_shot", "agentic"}:
            raise ValueError("rollout.interaction_mode must be 'one_shot' or 'agentic'")
        if self.agent_prompt_style not in {"native", "official"}:
            raise ValueError("rollout.agent_prompt_style must be 'native' or 'official'")
        if self.agent_observation_min_pixels is not None and self.agent_observation_min_pixels <= 0:
            raise ValueError("rollout.agent_observation_min_pixels must be positive when set")
        if self.mm_processor_cache_gb < 0:
            raise ValueError("mm_processor_cache_gb must be non-negative")
        if self.agent_max_tool_calls < 0:
            raise ValueError("agent_max_tool_calls must be non-negative")
        if self.agent_max_tokens_per_turn <= 0:
            raise ValueError("agent_max_tokens_per_turn must be positive")
        if self.agent_max_batch_images < 0:
            raise ValueError("agent_max_batch_images must be non-negative")
        if self.agent_bbox_format not in {"auto", "norm1000", "pixel"}:
            raise ValueError("agent_bbox_format must be 'auto', 'norm1000' or 'pixel'")
        if self.agent_max_trajectory_retries < 0:
            raise ValueError("agent_max_trajectory_retries must be non-negative")
        if self.agent_environment is not None and self.interaction_mode != "agentic":
            raise ValueError("rollout.agent_environment requires rollout.interaction_mode=agentic")
        if self.agent_environment is not None and ":" in self.agent_environment:
            # path/to/file.py:ClassName, made absolute so that the rollout workers find the same file
            path, class_name = self.agent_environment.rsplit(":", 1)
            self.agent_environment = f"{os.path.abspath(path)}:{class_name}"
        if self.agent_environment_kwargs and self.interaction_mode != "agentic":
            raise ValueError("rollout.agent_environment_kwargs requires rollout.interaction_mode=agentic")
        if (
            self.interaction_mode == "agentic"
            and self.response_length > 0
            and self.agent_max_tokens_per_turn > self.response_length
        ):
            raise ValueError("agent_max_tokens_per_turn cannot exceed the full response budget")

    @property
    def resolved_agent_environment(self) -> str:
        """The agent environment of this rollout; null means DeepEyes, the original agentic setting."""
        return self.agent_environment or "deepeyes"

    def to_dict(self):
        return asdict(self)
