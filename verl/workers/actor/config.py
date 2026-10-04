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
Actor config
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoraConfig:
    rank: int = 0
    alpha: int = 64
    target_modules: str = "all-linear"
    exclude_modules: Optional[str] = None

    def post_init(self):
        if not isinstance(self.target_modules, str):
            raise TypeError("lora.target_modules must be a string like 'all-linear' or 'q_proj,k_proj,v_proj,o_proj'.")

        self.target_modules = self.target_modules.strip()
        if self.exclude_modules is not None:
            if not isinstance(self.exclude_modules, str):
                raise TypeError("lora.exclude_modules must be a string like '.*visual.*'.")

            self.exclude_modules = self.exclude_modules.strip()


@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)
    # auto | true | false: tokenize <think>/</think> as plain text when they are untrained added tokens
    # (Qwen3-VL Instruct); see verl/utils/plain_think.py
    plain_think_tokens: Any = "auto"
    # InternVL only: maximum number of tiles per image (a thumbnail is added when an image has more than one);
    # None keeps the model config's max_dynamic_patch. Applies to the trainer's processor and the vLLM rollout;
    # pass the same value to the evaluation (--max-dynamic-patch).
    max_dynamic_patch: Optional[int] = None
    # below are auto keys
    override_chat_template: Optional[str] = field(default=None, init=False)  # copied from data.override_chat_template

    def post_init(self):
        from ...utils.plain_think import normalize_plain_think_tokens

        self.plain_think_tokens = normalize_plain_think_tokens(self.plain_think_tokens)
        if self.max_dynamic_patch is not None and self.max_dynamic_patch < 1:
            raise ValueError(f"model.max_dynamic_patch must be a positive integer, but got {self.max_dynamic_patch}.")
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    min_lr_ratio: Optional[float] = None
    lr_scheduler_type: str = "constant"
    # below are auto keys
    training_steps: int = field(default=-1, init=False)


@dataclass
class FSDPConfig:
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    enable_rank0_init: bool = True
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    fsdp_size: int = -1
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_buffer_dtype: str = "fp32"


@dataclass
class OffloadConfig:
    offload_params: bool = False
    offload_optimizer: bool = False


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    global_batch_size: int = 256
    """number of samples per minibatch for updating actor"""
    micro_batch_size_per_device_for_update: int = 4
    """number of samples per forward pass for updating actor"""
    micro_batch_size_per_device_for_experience: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.3
    """clip ratio in PPO & DAPO"""
    clip_ratio_dual: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    loss_avg_mode: str = "token"
    """loss average mode: `token` weights every response token of the mini-batch (all ranks) equally;
    `seq` averages over the tokens of each response, then weights every response equally (verl's
    seq-mean-token-mean)"""
    loss_type: str = "default"
    """loss type: `default`, `gspo`, `cispo`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = True
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = True
    """enable torch compile"""
    tau_positive: float = 1.0
    """temperature for positive tokens"""
    tau_negative: float = 1.05
    """temperature for negative tokens"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)
    disable_kl: bool = field(default=False, init=False)
    use_kl_loss: bool = field(default=False, init=False)
    kl_penalty: str = field(default="kl", init=False)
    kl_coef: float = field(default=0.0, init=False)


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    # below are auto keys
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)


@dataclass
class TeacherModelConfig:
    model_path: Optional[str] = None
    """the teacher checkpoint (a model name or a local path), loaded with its own config"""
    trust_remote_code: bool = True
    override_config: dict[str, Any] = field(default_factory=dict)

    def post_init(self):
        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)


@dataclass
class TeacherConfig:
    source: str = "none"
    """where the teacher of on-policy distillation comes from: `none` (no teacher), `model` (a frozen model
    loaded from `model.model_path`) or `ema` (an exponential moving average of the actor, updated after each
    training step and saved with the actor's checkpoints)"""
    model: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    ema_rate: float = 0.05
    """`source=ema`: phi <- (1 - ema_rate) * phi + ema_rate * theta after each training step"""
    # below are auto keys
    micro_batch_size_per_device_for_experience: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)

    def post_init(self):
        if self.source not in {"none", "model", "ema"}:
            raise ValueError(
                f"worker.teacher.source must be one of ['ema', 'model', 'none'], but got {self.source!r}."
            )
        if self.source == "ema":
            if self.model.model_path:
                raise ValueError(
                    "worker.teacher.source=ema copies the actor; leave worker.teacher.model.model_path unset."
                )
            if self.fsdp.torch_dtype not in (None, "fp32"):
                # a 0.05 step of an EMA is often below the relative precision of bf16 (2^-8) and would be rounded away
                raise ValueError(
                    "worker.teacher.source=ema keeps its weights in fp32; leave worker.teacher.fsdp.torch_dtype unset."
                )
            if not 0.0 < self.ema_rate <= 1.0:
                raise ValueError(f"worker.teacher.ema_rate must be in (0, 1], but got {self.ema_rate}.")
        if self.source == "model" and not self.model.model_path:
            raise ValueError("worker.teacher.source=model requires worker.teacher.model.model_path.")
        if self.source == "none" and self.model.model_path:
            raise ValueError(
                "worker.teacher.model.model_path is set but worker.teacher.source=none; set source=model to use it."
            )
        if self.offload.offload_optimizer:
            raise ValueError("worker.teacher has no optimizer; set worker.teacher.offload.offload_optimizer=false.")

    @property
    def enabled(self) -> bool:
        return self.source != "none"
