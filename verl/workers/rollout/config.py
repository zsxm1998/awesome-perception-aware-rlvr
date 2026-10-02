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
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def post_init(self):
        if self.interaction_mode not in {"one_shot", "agentic"}:
            raise ValueError("rollout.interaction_mode must be 'one_shot' or 'agentic'")
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
        if (
            self.interaction_mode == "agentic"
            and self.response_length > 0
            and self.agent_max_tokens_per_turn > self.response_length
        ):
            raise ValueError("agent_max_tokens_per_turn cannot exceed the full response budget")

    def to_dict(self):
        return asdict(self)
