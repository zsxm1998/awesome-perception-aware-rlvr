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
The main entry point to run the PPO algorithm
"""

import importlib.util
from contextlib import nullcontext
from typing import Any, Literal, Optional, Union, cast

import numpy as np
import peft
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from peft import TaskType, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..models.transformers.internvl import patch_internvl_vision_checkpointing, set_internvl_image_context_token_id
from ..models.transformers.qwen3_5 import register_qwen3_5
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import (
    AnyPrecisionAdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


# vLLM sets environment variables (CUDA / inductor) when it is imported, so it is imported together with
# this module, before torch.distributed starts. It is optional only so that the CPU unit tests can import
# this module on machines without vLLM.
if importlib.util.find_spec("vllm") is not None:
    from .rollout import vLLMRollout
    from .sharding_manager import FSDPVLLMShardingManager


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._lora_rank = self.config.actor.model.lora.rank
        self._is_lora = self._lora_rank > 0

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_ref_param_offload = self.config.ref.offload.offload_params

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        # create main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        # create ulysses device mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # validate and normalize config
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = config.global_batch_size // (world_size // config.ulysses_size)
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        register_qwen3_5()
        if role != "ref":  # ref model's tokenizer is same as actor
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                override_chat_template=model_config.override_chat_template,
                plain_think_tokens=model_config.plain_think_tokens,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                override_chat_template=model_config.override_chat_template,
                plain_think_tokens=model_config.plain_think_tokens,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if role == "critic":
            AutoClass = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        model = cast(PreTrainedModel, model)  # lint
        model.tie_weights()  # avoid hanging
        set_internvl_image_context_token_id(model, getattr(self, "processor", None))

        if role == "ref":
            model.requires_grad_(False)

        is_lora_model = self._is_lora and role == "actor"
        if is_lora_model:
            self.print_rank0("Applying LoRA to actor module")
            model.enable_input_require_grads()
            if model_config.lora.target_modules == "all-linear":
                target_modules = model_config.lora.target_modules
            else:
                target_modules = [item.strip() for item in model_config.lora.target_modules.split(",") if item.strip()]

            lora_config = peft.LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=model_config.lora.rank,
                lora_alpha=model_config.lora.alpha,
                target_modules=target_modules,
                exclude_modules=model_config.lora.exclude_modules,
            )
            model = get_peft_model(model, lora_config)
            for p in model.parameters():
                if not p.requires_grad:
                    p.data = p.to(torch.bfloat16)
                else:
                    p.data = p.to(torch_dtype)
        else:
            model = model.to(torch_dtype)

        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        patched_encoders = patch_internvl_vision_checkpointing(model)
        if patched_encoders:
            self.print_rank0(f"InternVL vision checkpointing uses non-reentrant mode ({patched_encoders} encoder(s)).")

        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):  # transformers >= 4.52.0
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):  # transformers < 4.52.0
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
            cast_forward_inputs=True,
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model, is_lora_model=is_lora_model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if role in ["actor", "critic"]:
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            if optim_config.lr_scheduler_type == "constant":
                self.lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
                )
            elif optim_config.lr_scheduler_type == "cosine":
                total_steps = optim_config.training_steps
                min_lr_ratio = optim_config.min_lr_ratio
                num_cycles = 0.5
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {optim_config.lr_scheduler_type} is not supported")
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        lora_kwargs = (
            {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
            if self._is_lora
            else {}
        )
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            tokenizer_path=self.tokenizer.name_or_path,
            processor=self.processor,
            **lora_kwargs,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                padding_free=self.config.critic.padding_free,
                role="critic",
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )

        if self._has_ref:
            if self._is_lora:
                self.ref_fsdp_module = self.fsdp_module
            else:
                self._build_model_optimizer(
                    model_config=self.config.actor.model,
                    fsdp_config=self.config.ref.fsdp,
                    optim_config=None,
                    padding_free=self.config.ref.padding_free,
                    role="ref",
                )

        if self._has_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:  # must after actor
            self._build_rollout()

        if self._has_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_fsdp_module,
            )

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(
        self,
        data: DataProto,
        source_key: str = "multi_modal_data",
        output_key: str = "multi_modal_inputs",
        cache_namespace: str | None = None,
        cache_key_field: str = "uid",
    ):
        from .agent.trajectory import validate_agent_multi_modal_layout

        layouts = data.non_tensor_batch.get("multi_modal_layout")

        if output_key in data.non_tensor_batch:
            if len(data.non_tensor_batch[output_key]) != len(data):
                raise ValueError(f"precomputed {output_key} does not align with the DataProto batch")
            provenance = data.non_tensor_batch.get(f"{output_key}_provenance")
            if (
                provenance is None
                or len(provenance) != len(data)
                or any(value != "processor_atomic" for value in provenance)
            ):
                raise ValueError(
                    f"precomputed {output_key} requires one aligned 'processor_atomic' provenance marker per sample"
                )
            if layouts is not None:
                if len(layouts) != len(data):
                    raise ValueError("multi_modal_layout does not align with the DataProto batch")
                for layout, inputs in zip(
                    layouts,
                    data.non_tensor_batch[output_key],
                ):
                    validate_agent_multi_modal_layout(layout, inputs)
            return
        if source_key not in data.non_tensor_batch:
            return

        cache_namespace = cache_namespace or source_key
        uid_cache_key = f"{cache_namespace}:uid"
        mm_cache_key = f"{cache_namespace}:{output_key}"
        # Native agent rollouts need per-trajectory cache isolation by default,
        # because members of one GRPO group may commit different crops. An
        # explicit branch-specific key (for example a corrupted-view cache ID)
        # is stronger and must never be silently replaced.
        if cache_key_field == "uid" and "agent_trajectory_id" in data.non_tensor_batch:
            cache_key_field = "agent_trajectory_id"
        if cache_key_field not in data.non_tensor_batch:
            raise KeyError(f"multimodal cache key field {cache_key_field!r} is missing")
        cache_indices = data.non_tensor_batch[cache_key_field]

        cached_indices = self._cache.get(uid_cache_key)
        if cached_indices is not None and (
            len(cached_indices) != len(cache_indices) or not np.all(cache_indices == cached_indices)
        ):
            # differing lengths (e.g. a probe batch between training batches) must
            # invalidate the cache instead of crashing the broadcast comparison
            self._cache.pop(uid_cache_key, None)
            self._cache.pop(mm_cache_key, None)

        if mm_cache_key not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}  # avoid repeated processing for n > 1 samples
            for index, multi_modal_data in zip(cache_indices, data.non_tensor_batch[source_key]):  # process per sample
                if index not in multi_modal_inputs_cache:
                    if multi_modal_data is None:
                        multi_modal_inputs_cache[index] = {}
                        batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])
                        continue
                    images, videos = [], []
                    if "images" in multi_modal_data:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels, max_pixels))

                    if "videos" in multi_modal_data:
                        for video in multi_modal_data["videos"]:
                            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

                    if len(images) != 0:
                        # it's necessary to add `dict` to properly convert batch features to dict
                        # otherwise the batch features will be converted to dict keys
                        # see https://github.com/hiyouga/EasyR1/pull/339
                        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    elif len(videos) != 0:
                        multi_modal_inputs = dict(
                            self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                        )
                    else:
                        multi_modal_inputs = {}

                    multi_modal_inputs_cache[index] = multi_modal_inputs

                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

            self._cache[uid_cache_key] = cache_indices
            self._cache[mm_cache_key] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch[output_key] = self._cache[mm_cache_key]
        if layouts is not None:
            if len(layouts) != len(data):
                raise ValueError("multi_modal_layout does not align with the DataProto batch")
            for layout, inputs in zip(
                layouts,
                data.non_tensor_batch[output_key],
            ):
                validate_agent_multi_modal_layout(layout, inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.DP_COMPUTE)
    def detect_grounding_dino(
        self,
        requests: list[dict[str, Any]],
        batch_size: int,
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ) -> list[tuple[int, list[tuple[int, int, int, int]]]]:
        from ..trainer.grounding_consistency import _run_grounding_dino_detection_requests

        if not requests:
            return []

        processor, model, device = self._get_grounding_dino_detector()
        try:
            return _run_grounding_dino_detection_requests(
                indexed_requests=[
                    (int(request["request_idx"]), str(request["region_name"]), request["image"])
                    for request in requests
                ],
                processor=processor,
                model=model,
                device=device,
                batch_size=batch_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        finally:
            model = None
            self.release_grounding_dino()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_grounding_dino(self):
        model = getattr(self, "_grounding_dino_model", None)
        self._grounding_dino_model = None
        self._grounding_dino_device = None
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_grounding_dino_detector(self) -> tuple[Any, Any, torch.device]:
        if getattr(self, "_grounding_dino_processor", None) is None:
            from transformers import AutoProcessor

            from ..trainer.grounding_consistency import _GROUNDING_DINO_MODEL_ID

            self._grounding_dino_processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_MODEL_ID)

        if getattr(self, "_grounding_dino_model", None) is None:
            from transformers import AutoModelForZeroShotObjectDetection

            from ..trainer.grounding_consistency import _GROUNDING_DINO_MODEL_ID, _resolve_grounding_dino_torch_dtype

            device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
            torch_dtype = _resolve_grounding_dino_torch_dtype(device)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                _GROUNDING_DINO_MODEL_ID,
                torch_dtype=torch_dtype,
            ).to(device)
            model.eval()
            self._grounding_dino_model = model
            self._grounding_dino_device = device
            print(
                "Grounding DINO worker detector loaded "
                f"rank={self.rank} local_rank={self._local_rank} model={_GROUNDING_DINO_MODEL_ID} "
                f"device={device} dtype={torch_dtype}",
                flush=True,
            )
        return self._grounding_dino_processor, self._grounding_dino_model, self._grounding_dino_device

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if self.generation_config is not None:
            if self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
            if self.generation_config.pad_token_id is not None:
                pad_token_id = self.generation_config.pad_token_id

        meta_info = {
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # the rollout policy's entropy is returned for batch-level entropy masks (entropy_thr_granularity=batch)
        return_entropy = bool(data.meta_info.get("return_old_entropies", False))
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(
                data=data,
                return_entropy=return_entropy,
                entropy_top_p=float(data.meta_info.get("old_entropy_top_p", 1.0)),
            )
            tensors = (
                {"old_log_probs": output[0], "old_entropies": output[1]}
                if return_entropy
                else {"old_log_probs": output}
            )
            output = DataProto.from_dict(tensors=tensors, meta_info={"temperature": self.config.rollout.temperature})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_aux_log_probs(self, data: DataProto):
        assert self._has_actor

        output_key = data.meta_info["aux_log_probs_output_key"]
        entropy_output_key = data.meta_info.get("aux_entropy_output_key")
        return_entropy = entropy_output_key is not None

        # Auxiliary images are rebuilt per branch/step, so stale worker-side features must not survive across calls.
        self._cache.pop(f"{output_key}:uid", None)
        self._cache.pop(f"{output_key}:multi_modal_inputs", None)
        self._process_multi_modal_inputs(
            data,
            source_key="multi_modal_data",
            output_key="multi_modal_inputs",
            cache_namespace=output_key,
            cache_key_field="multi_modal_cache_id",
        )
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data, return_entropy=return_entropy)
            if return_entropy:
                log_probs, entropies = output
                output = DataProto.from_dict(tensors={output_key: log_probs, entropy_output_key: entropies})
            else:
                output = DataProto.from_dict(tensors={output_key: output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_visual_sensitivity_scores(self, data: DataProto):
        assert self._has_actor

        self._cache.pop("visual_sensitivity:uid", None)
        self._cache.pop("visual_sensitivity:multi_modal_inputs", None)
        self._cache.pop("visual_sensitivity_aux:uid", None)
        self._cache.pop("visual_sensitivity_aux:auxiliary_multi_modal_inputs", None)
        self._process_multi_modal_inputs(
            data,
            source_key="multi_modal_data",
            output_key="multi_modal_inputs",
            cache_namespace="visual_sensitivity",
            cache_key_field="uid",
        )
        self._process_multi_modal_inputs(
            data,
            source_key="auxiliary_multi_modal_data",
            output_key="auxiliary_multi_modal_inputs",
            cache_namespace="visual_sensitivity_aux",
            cache_key_field="auxiliary_multi_modal_cache_id",
        )
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            scores, metrics = self.actor.compute_visual_sensitivity_scores(data=data)
            output = DataProto.from_dict(
                tensors={"per_token_sensitivity_scores": scores},
                meta_info={"visual_sensitivity_metrics": metrics},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs_and_visual_sensitivity_scores(self, data: DataProto):
        assert self._has_actor

        self._cache.pop("old_visual_sensitivity:uid", None)
        self._cache.pop("old_visual_sensitivity:multi_modal_inputs", None)
        self._cache.pop("old_visual_sensitivity_aux:uid", None)
        self._cache.pop("old_visual_sensitivity_aux:auxiliary_multi_modal_inputs", None)
        self._process_multi_modal_inputs(
            data,
            source_key="multi_modal_data",
            output_key="multi_modal_inputs",
            cache_namespace="old_visual_sensitivity",
            cache_key_field="uid",
        )
        self._process_multi_modal_inputs(
            data,
            source_key="auxiliary_multi_modal_data",
            output_key="auxiliary_multi_modal_inputs",
            cache_namespace="old_visual_sensitivity_aux",
            cache_key_field="auxiliary_multi_modal_cache_id",
        )
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output_tensors, metrics = self.actor.compute_log_prob_and_visual_sensitivity_scores(data=data)
            output = DataProto.from_dict(
                tensors=output_tensors,
                meta_info={
                    "temperature": self.config.rollout.temperature,
                    "visual_sensitivity_metrics": metrics,
                },
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        if self.world_size > 1:
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        adapter_ctx = self.ref_fsdp_module.disable_adapter() if self._is_lora else nullcontext()

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        # the fsdp module is the same as the ref fsdp module when lora is enabled
        if self._use_ref_param_offload or (self._is_lora and self._use_param_offload):
            load_fsdp_model(self.ref_fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager, adapter_ctx:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_fsdp_module._handle.reshard(True)

        if self._use_ref_param_offload or (self._is_lora and self._use_param_offload):
            offload_fsdp_model(self.ref_fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
