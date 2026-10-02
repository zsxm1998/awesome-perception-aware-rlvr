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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import GenerationSample, Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.reasoning import (
    canonicalize_response_for_prefilled_think,
    compact_vision_pad_runs,
    decode_prompt_from_batch,
)
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .grounding_consistency import (
    GroundingConsistencyRewardResult,
    GroundingConsistencyRewardScorer,
    compute_group_eligibility_mask,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from .perception_reasoning_data import (
    PerceptionReasoningCorruptionBuilder,
    build_perception_reasoning_loss_config,
    build_region_token_mask,
    needs_auxiliary_log_probs,
    needs_decremental_entropy,
    needs_dvrp_auxiliary_views,
    needs_full_vocab_visual_sensitivity,
    needs_hidden_state_visual_sensitivity,
    needs_incremental_auxiliary,
    needs_incremental_entropy,
    needs_media_corruption_builder,
    resolve_visual_token_ids,
    uses_model_level_visual_corruption,
)
from .perception_reasoning_loss import (
    build_sensitivity_advantage_shaping_context,
    has_perception_reasoning,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score: Optional[float] = None  # None until a validation has run
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

        self._latest_vision_metrics: dict[str, float] = {}
        self.visual_token_ids: list[int] = []
        if needs_hidden_state_visual_sensitivity(config.algorithm):
            visual_token_ids = resolve_visual_token_ids(tokenizer, config.algorithm.visual_token)
            if not visual_token_ids:
                raise ValueError(
                    "hidden_state_similarity visual sensitivity could not resolve any visual token ids. "
                    "Set algorithm.visual_token to a tokenizer-recognized image/video placeholder token."
                )
            self.visual_token_ids = sorted(visual_token_ids)
        self.perception_reasoning_corruption_builder: PerceptionReasoningCorruptionBuilder | None = None
        self.grounding_consistency_scorer: GroundingConsistencyRewardScorer | None = None
        if needs_media_corruption_builder(config.algorithm):
            if processor is None:
                raise ValueError("Perception/reasoning auxiliary views require a multimodal processor.")
            image_processor = getattr(processor, "image_processor", None)
            image_patch_size = getattr(image_processor, "patch_size", None) or 14
            if isinstance(image_patch_size, (list, tuple)):
                image_patch_size = image_patch_size[0]
            self.perception_reasoning_corruption_builder = PerceptionReasoningCorruptionBuilder(
                tokenizer=tokenizer,
                processor=processor,
                image_patch_size=int(image_patch_size),
                min_pixels=config.data.min_pixels,
                max_pixels=config.data.max_pixels,
                video_fps=config.data.video_fps,
            )
        if config.algorithm.use_grounding_consistency_reward:
            if processor is None:
                raise ValueError("Grounding consistency reward requires a multimodal processor.")
            self.grounding_consistency_scorer = GroundingConsistencyRewardScorer(
                tokenizer=tokenizer,
                processor=processor,
                max_prompt_length=config.data.max_prompt_length,
                min_pixels=config.data.min_pixels,
                max_pixels=config.data.max_pixels,
                video_fps=config.data.video_fps,
                reward_weight=config.algorithm.grounding_consistency_reward_weight,
                detector=config.algorithm.grounding_consistency_detector,
                grounding_dino_device=config.algorithm.grounding_dino_device,
                grounding_dino_batch_size=config.algorithm.grounding_dino_batch_size,
            )

    def _build_perception_reasoning_loss_config(self) -> dict[str, Any]:
        loss_config = build_perception_reasoning_loss_config(
            self.config.algorithm,
            decremental_stats=self._latest_vision_metrics,
        )
        if needs_hidden_state_visual_sensitivity(self.config.algorithm):
            loss_config["visual_token_ids"] = self.visual_token_ids
        if loss_config.get("advantage_scaling_schedule") == "linear":
            total_steps = max(int(self.training_steps), 1)
            if total_steps == 1:
                progress = 1.0
            else:
                progress = min(max(float(self.global_step - 1) / (total_steps - 1), 0.0), 1.0)
            loss_config["advantage_scaling_schedule_progress"] = progress
        return loss_config

    def _compute_auxiliary_old_log_probs(self, batch: DataProto) -> DataProto:
        model_level_corruption = uses_model_level_visual_corruption(self.config.algorithm)
        self._latest_vision_metrics = {}

        def _compute_branch(branch_batch: DataProto, output_key: str, entropy_output_key: str | None) -> DataProto:
            branch_batch.meta_info["aux_log_probs_output_key"] = output_key
            if entropy_output_key is not None:
                branch_batch.meta_info["aux_entropy_output_key"] = entropy_output_key
            return self.actor_rollout_ref_wg.compute_aux_log_probs(branch_batch)

        if model_level_corruption:
            decremental_batch = DataProto(
                batch=batch.batch,
                non_tensor_batch=dict(batch.non_tensor_batch),
                meta_info=dict(batch.meta_info),
            )
            # The images are unchanged; key the worker-side multimodal cache by sample like the
            # base forward (the worker requires an explicit cache key for auxiliary branches).
            cache_ids = batch.non_tensor_batch.get("agent_trajectory_id", batch.non_tensor_batch.get("uid"))
            if cache_ids is not None:
                decremental_batch.non_tensor_batch["multi_modal_cache_id"] = cache_ids
            decremental_batch.meta_info["model_level_visual_corruption"] = {
                "name": self.config.algorithm.corrupt_image,
                "kwargs": dict(self.config.algorithm.corrupt_image_kwargs or {}),
            }
            auxiliary_output = _compute_branch(
                decremental_batch,
                output_key="decremental_old_log_probs",
                entropy_output_key="decremental_entropies"
                if needs_decremental_entropy(self.config.algorithm)
                else None,
            )
            if needs_dvrp_auxiliary_views(self.config.algorithm):
                incremental_output = self._compute_incremental_old_log_probs(batch)
                auxiliary_output = auxiliary_output.union(incremental_output)
            return auxiliary_output

        if self.perception_reasoning_corruption_builder is None:
            raise ValueError("Auxiliary old-log-prob computation requires a configured corruption builder.")

        if needs_dvrp_auxiliary_views(self.config.algorithm):
            decremental_result, incremental_result = self.perception_reasoning_corruption_builder.build_dvrp_batches(
                batch=batch,
                config=self.config.algorithm,
                global_step=self.global_step,
                total_training_steps=max(int(self.training_steps), 1),
            )
            decremental_output = _compute_branch(
                decremental_result.batch,
                output_key="decremental_old_log_probs",
                entropy_output_key="decremental_entropies"
                if needs_decremental_entropy(self.config.algorithm)
                else None,
            )
            incremental_output = _compute_branch(
                incremental_result.batch,
                output_key="incremental_old_log_probs",
                entropy_output_key="incremental_entropies"
                if needs_incremental_entropy(self.config.algorithm)
                else None,
            )
            auxiliary_output = decremental_output.union(incremental_output)
            self._latest_vision_metrics.update(decremental_result.stats)
            self._latest_vision_metrics.update(incremental_result.stats)
        else:
            decremental_result = self.perception_reasoning_corruption_builder.build_batch(
                batch=batch,
                config=self.config.algorithm,
                global_step=self.global_step,
            )
            auxiliary_output = _compute_branch(
                decremental_result.batch,
                output_key="decremental_old_log_probs",
                entropy_output_key="decremental_entropies"
                if needs_decremental_entropy(self.config.algorithm)
                else None,
            )
            self._latest_vision_metrics.update(decremental_result.stats)

        return auxiliary_output

    def _compute_incremental_old_log_probs(self, batch: DataProto) -> DataProto:
        if self.perception_reasoning_corruption_builder is None:
            raise ValueError("Incremental old-log-prob computation requires a configured corruption builder.")

        incremental_result = self.perception_reasoning_corruption_builder.build_incremental_batch(
            batch=batch,
            config=self.config.algorithm,
            global_step=self.global_step,
            total_training_steps=max(int(self.training_steps), 1),
        )
        incremental_batch = incremental_result.batch
        incremental_batch.meta_info["aux_log_probs_output_key"] = "incremental_old_log_probs"
        if needs_incremental_entropy(self.config.algorithm):
            incremental_batch.meta_info["aux_entropy_output_key"] = "incremental_entropies"
        self._latest_vision_metrics.update(incremental_result.stats)
        return self.actor_rollout_ref_wg.compute_aux_log_probs(incremental_batch)

    def _build_full_vocab_visual_sensitivity_batch(self, batch: DataProto) -> DataProto:
        if self.perception_reasoning_corruption_builder is None:
            raise ValueError("Full-vocab visual sensitivity requires a configured corruption builder.")

        self._latest_vision_metrics = {}
        auxiliary_result = self.perception_reasoning_corruption_builder.build_batch(
            batch=batch,
            config=self.config.algorithm,
            global_step=self.global_step,
        )
        self._latest_vision_metrics.update(auxiliary_result.stats)
        return self._pack_visual_sensitivity_metric_batch(batch, auxiliary_result.batch)

    def _pack_visual_sensitivity_metric_batch(self, batch: DataProto, auxiliary_batch: DataProto) -> DataProto:
        metric_batch = DataProto.from_dict(
            tensors={
                "input_ids": batch.batch["input_ids"],
                "attention_mask": batch.batch["attention_mask"],
                "position_ids": batch.batch["position_ids"],
                "responses": batch.batch["responses"],
                "response_mask": batch.batch["response_mask"],
                "auxiliary_input_ids": auxiliary_batch.batch["input_ids"],
                "auxiliary_attention_mask": auxiliary_batch.batch["attention_mask"],
                "auxiliary_position_ids": auxiliary_batch.batch["position_ids"],
            },
            non_tensors={
                key: batch.non_tensor_batch[key]
                for key in ("uid", "multi_modal_data")
                if key in batch.non_tensor_batch
            },
            meta_info=dict(batch.meta_info),
        )
        metric_batch.non_tensor_batch["auxiliary_multi_modal_cache_id"] = auxiliary_batch.non_tensor_batch[
            "multi_modal_cache_id"
        ]
        metric_batch.non_tensor_batch["auxiliary_multi_modal_data"] = auxiliary_batch.non_tensor_batch[
            "multi_modal_data"
        ]
        metric_batch.meta_info["visual_sensitivity_metric"] = self.config.algorithm.visual_sensitivity_metric
        metric_batch.meta_info["visual_sensitivity_jsd_weight"] = self.config.algorithm.visual_sensitivity_jsd_weight
        metric_batch.meta_info["visual_sensitivity_entropy_gate"] = (
            self.config.algorithm.visual_sensitivity_entropy_gate
        )
        return metric_batch

    def _compute_old_log_probs_and_full_vocab_visual_sensitivity(self, batch: DataProto) -> DataProto:
        metric_batch = self._build_full_vocab_visual_sensitivity_batch(batch)
        if self._full_vocab_fused_needs_decremental_log_probs():
            metric_batch.meta_info["return_decremental_log_probs"] = True
        if needs_decremental_entropy(self.config.algorithm):
            metric_batch.meta_info["return_decremental_entropy"] = True
        return self.actor_rollout_ref_wg.compute_log_probs_and_visual_sensitivity_scores(metric_batch)

    def _full_vocab_fused_needs_decremental_log_probs(self) -> bool:
        return (
            self.config.algorithm.visual_sensitivity_loss_coef != 0.0
            or self.config.algorithm.decremental_entropy_coef != 0.0
            # Sampled diagnostic estimators piggyback on the same corrupted-view forward.
            or bool(self.config.algorithm.visual_sensitivity_log_metrics)
        )

    def _maybe_attach_region_token_mask(self, batch: DataProto) -> None:
        if not self.config.algorithm.include_region_tokens_in_perception_mask:
            return
        if self.config.algorithm.top_perception_quantile >= 1.0:
            return
        if "region_token_mask" in batch.batch.keys():
            return
        batch.batch["region_token_mask"] = build_region_token_mask(
            responses=batch.batch["responses"],
            response_mask=batch.batch["response_mask"],
            tokenizer=self.tokenizer,
        )

    def _compute_grounding_consistency_reward(
        self,
        batch: DataProto,
        rollout_config: dict[str, Any],
        eligible_sample_mask: Optional[list[bool]] = None,
        cache_token: Any = None,
    ) -> GroundingConsistencyRewardResult | None:
        if self.grounding_consistency_scorer is None:
            return None
        return self.grounding_consistency_scorer.score_batch(
            batch=batch,
            rollout_worker_group=self.actor_rollout_ref_wg,
            rollout_config=rollout_config,
            eligible_sample_mask=eligible_sample_mask,
            cache_token=cache_token,
        )

    def _attach_grounding_consistency_reward_inputs(
        self,
        batch: DataProto,
        rollout_config: dict[str, Any],
        eligible_sample_mask: Optional[list[bool]] = None,
        cache_token: Any = None,
    ) -> GroundingConsistencyRewardResult | None:
        if "grounding_consistency" in batch.non_tensor_batch:
            # already attached (e.g. during online filtering); recomputing would re-run detection
            return None
        grounding_reward_result = self._compute_grounding_consistency_reward(
            batch, rollout_config, eligible_sample_mask, cache_token
        )
        if grounding_reward_result is None:
            return None

        batch.non_tensor_batch["grounding_consistency"] = np.array(
            grounding_reward_result.weighted_scores,
            dtype=np.float32,
        )
        batch.non_tensor_batch["grounding_consistency_raw"] = np.array(
            grounding_reward_result.raw_scores,
            dtype=np.float32,
        )
        batch.batch["grounding_token_level_scores"] = grounding_reward_result.reward_tensor
        return grounding_reward_result

    def _attach_grounding_rewards(
        self,
        batch: DataProto,
        rollout_config: dict[str, Any],
        reward_fn: Any,
        cache_token: Any,
    ) -> GroundingConsistencyRewardResult | None:
        """Attach GCR inputs, skipping detection for groups without a correct answer.

        A rule-based reward pre-pass (grounding inputs absent, so they contribute 0) provides
        per-sample accuracy; GCR is gated on accuracy == 1.0 downstream, so groups with no
        correct answer receive zeros either way and their detection can be skipped. If the
        reward function reports no accuracy metric, every group is treated as eligible.
        """
        eligible_sample_mask = None
        if self.grounding_consistency_scorer is not None and "grounding_consistency" not in batch.non_tensor_batch:
            _, pre_reward_metrics = ray.get(reward_fn.compute_reward.remote(self._reward_rpc_input(batch)))
            accuracy_values = pre_reward_metrics.get("accuracy")
            uids = batch.non_tensor_batch.get("uid")
            if accuracy_values is not None and uids is not None:
                eligible_sample_mask = compute_group_eligibility_mask(uids, accuracy_values)
        return self._attach_grounding_consistency_reward_inputs(
            batch, rollout_config, eligible_sample_mask, cache_token
        )

    def _reward_rpc_input(self, batch: DataProto) -> DataProto:
        """Build the minimal payload required by an agentic reward actor.

        Full source/crop images, compact image layouts, and processor tensors
        remain on the driver/rollout-worker path. The reward actor only needs
        response tensors, ground truth, and compact trajectory summaries.
        """
        if self.config.worker.rollout.interaction_mode != "agentic":
            return batch

        return batch.select(
            batch_keys=["responses", "response_mask"],
            non_tensor_batch_keys=[
                "ground_truth",
                "agent_reward_input",
                "agent_diagnostics",
                "grounding_consistency",
                "grounding_consistency_raw",
                "data_source",
                "question",
            ],
        )

    def _pop_rollout_inputs(
        self,
        batch: DataProto,
        *,
        meta_info_keys: Optional[list[str]] = None,
    ) -> DataProto:
        """Detach generation inputs without losing driver-owned agent messages.

        Native rollout consumes ``raw_prompt`` on the rollout worker, while
        perception/intervention builders may inspect the same messages after
        generation. Keep the driver copy in ``batch`` instead of depending on
        the worker to serialize it back with the materialized trajectory.
        """

        rollout_non_tensor_keys = ["raw_prompt_ids", "multi_modal_data"]
        raw_prompt = None
        if self.config.worker.rollout.interaction_mode == "agentic":
            rollout_non_tensor_keys.extend(["raw_prompt", "uid"])
            raw_prompt = batch.non_tensor_batch.get("raw_prompt")
            if raw_prompt is None:
                raise KeyError("agentic rollout requires 'raw_prompt' in non_tensor_batch")

        rollout_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=rollout_non_tensor_keys,
            meta_info_keys=meta_info_keys,
        )
        if raw_prompt is not None:
            batch.non_tensor_batch["raw_prompt"] = raw_prompt
        return rollout_batch

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score is not None and self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(f"{checkpointer_tracker_path}.tmp", "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)
        os.replace(f"{checkpointer_tracker_path}.tmp", checkpointer_tracker_path)

        # Remove older checkpoints only once the new one and the tracker pointing to it are complete:
        # removing them first leaves nothing to resume from if saving fails.
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    @staticmethod
    def _get_non_tensor_values(batch: DataProto, key: str, default: Any = None) -> list[Any]:
        values = batch.non_tensor_batch.get(key)
        if values is None:
            return [default] * len(batch)
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    @staticmethod
    def _to_float_list(values: Any) -> list[Optional[float]]:
        if values is None:
            return []
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            values = values.tolist()

        result = []
        for value in values:
            if value is None:
                result.append(None)
            else:
                result.append(float(value))
        return result

    @staticmethod
    def _extract_generation_media(batch: DataProto, index: int) -> tuple[Optional[list[Any]], Optional[list[Any]]]:
        multi_modal_batch = batch.non_tensor_batch.get("multi_modal_data")
        if multi_modal_batch is None:
            return None, None

        multi_modal_data = multi_modal_batch[index]
        if not isinstance(multi_modal_data, dict):
            return None, None

        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        return list(images) if images is not None else None, list(videos) if videos is not None else None

    @staticmethod
    def _build_reward_details(reward_metrics: Optional[dict[str, list[Any]]], batch_size: int) -> list[dict[str, Any]]:
        details = [dict() for _ in range(batch_size)]
        if not reward_metrics:
            return details

        for key, values in reward_metrics.items():
            if hasattr(values, "detach"):
                values = values.detach()
            if hasattr(values, "cpu"):
                values = values.cpu()
            if hasattr(values, "tolist"):
                values = values.tolist()
            for index, value in enumerate(values[:batch_size]):
                details[index][key] = value
        return details

    @staticmethod
    def _select_generation_samples(
        samples: list[GenerationSample],
        limit: int,
        seed: int,
        sort_key: Optional[Any] = None,
    ) -> list[GenerationSample]:
        if limit == 0 or not samples:
            return []

        selected = list(samples)
        if sort_key is not None:
            selected.sort(key=sort_key)
        if limit < 0 or len(selected) <= limit:
            return selected

        rng = np.random.RandomState(seed)
        rng.shuffle(selected)
        return selected[:limit]

    def _sequence_scores(
        self,
        batch: DataProto,
        reward_tensor: Optional[torch.Tensor] = None,
    ) -> list[Optional[float]]:
        score_tensor = reward_tensor
        if score_tensor is None and "token_level_scores" in batch.batch:
            score_tensor = batch.batch["token_level_scores"]
        if score_tensor is None:
            return [None] * len(batch)

        return self._to_float_list(score_tensor.sum(-1))

    def _sequence_advantages(self, batch: DataProto) -> list[Optional[float]]:
        if "advantages" not in batch.batch:
            return [None] * len(batch)

        advantages = batch.batch["advantages"]
        if "response_mask" not in batch.batch:
            return self._to_float_list(advantages.mean(-1))

        response_mask = batch.batch["response_mask"]
        denom = response_mask.sum(-1).clamp(min=1)
        return self._to_float_list((advantages * response_mask).sum(-1) / denom)

    def _generation_reward_details(
        self,
        batch: DataProto,
        reward_metrics: Optional[dict[str, list[Any]]] = None,
    ) -> list[dict[str, Any]]:
        if reward_metrics is not None:
            return self._build_reward_details(reward_metrics, len(batch))

        stored_details = batch.non_tensor_batch.get("reward_details")
        if stored_details is None:
            return [dict() for _ in range(len(batch))]
        return stored_details.tolist() if hasattr(stored_details, "tolist") else list(stored_details)

    def _build_generation_samples(
        self,
        batch: DataProto,
        split: str,
        reward_tensor: Optional[torch.Tensor] = None,
        reward_metrics: Optional[dict[str, list[Any]]] = None,
    ) -> list[GenerationSample]:
        raw_prompt_texts = [
            decode_prompt_from_batch(self.tokenizer, batch.batch, index) for index in range(len(batch))
        ]
        prompt_texts = [compact_vision_pad_runs(raw_prompt or "", self.tokenizer) for raw_prompt in raw_prompt_texts]
        response_mask = batch.batch.get("response_mask", None)
        completion_texts = []
        for index, ids in enumerate(batch.batch["responses"]):
            if response_mask is not None:
                ids = ids[response_mask[index].to(torch.bool)]
            else:
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_token_id is not None:
                    valid_len = ids.numel()
                    while valid_len > 0 and ids[valid_len - 1].item() == pad_token_id:
                        valid_len -= 1
                    ids = ids[:valid_len]
            completion_texts.append(self.tokenizer.decode(ids, skip_special_tokens=False))
        completion_texts = [
            canonicalize_response_for_prefilled_think(raw_prompt, completion)
            for raw_prompt, completion in zip(raw_prompt_texts, completion_texts)
        ]
        ground_truths = self._get_non_tensor_values(batch, "ground_truth")
        uids = self._get_non_tensor_values(batch, "uid")
        scores = self._sequence_scores(batch, reward_tensor=reward_tensor)
        advantages = self._sequence_advantages(batch)
        reward_details = self._generation_reward_details(batch, reward_metrics=reward_metrics)

        samples = []
        for index, (prompt, completion) in enumerate(zip(prompt_texts, completion_texts)):
            images, videos = self._extract_generation_media(batch, index)
            samples.append(
                GenerationSample(
                    step=self.global_step,
                    split=split,
                    uid=str(uids[index]) if uids[index] is not None else None,
                    prompt=prompt,
                    completion=completion,
                    ground_truth=ground_truths[index],
                    score=scores[index],
                    reward_details=reward_details[index],
                    advantages=advantages[index],
                    image_count=len(images or []),
                    video_count=len(videos or []),
                    images=images,
                    videos=videos,
                )
            )
        return samples

    def _maybe_log_val_generations(self, samples: list[GenerationSample]) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        samples = self._select_generation_samples(
            samples,
            limit=self.config.trainer.val_generations_to_log,
            seed=42,
            sort_key=lambda sample: sample.prompt,
        )
        self.logger.log_generation(samples, self.global_step, split="val")

    def _maybe_log_train_generations(
        self,
        batch: DataProto,
        reward_metrics: Optional[dict[str, list[Any]]] = None,
    ) -> None:
        """Log a table of generated training samples that reached optimization."""
        if self.config.trainer.train_generations_to_log == 0:
            return

        samples = self._build_generation_samples(batch, split="train", reward_metrics=reward_metrics)
        samples = self._select_generation_samples(
            samples,
            limit=self.config.trainer.train_generations_to_log,
            seed=self.global_step,
        )
        self.logger.log_generation(samples, self.global_step, split="train")

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        generation_samples = []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                dtype=object,
            )
            test_gen_batch = self._pop_rollout_inputs(test_batch)
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            val_rollout_config = dict(self.config.worker.rollout.to_dict())
            val_rollout_config.update(self.config.worker.rollout.val_override_config)
            self._attach_grounding_rewards(
                test_batch,
                val_rollout_config,
                reward_fn=self.val_reward_fn,
                cache_token=f"val-step-{self.global_step}",
            )

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(
                self.val_reward_fn.compute_reward.remote(self._reward_rpc_input(test_batch))
            )

            if self.config.trainer.val_generations_to_log > 0:
                generation_samples.extend(
                    self._build_generation_samples(
                        test_batch,
                        split="val",
                        reward_tensor=reward_tensor,
                        reward_metrics=reward_metrics,
                    )
                )
                generation_samples = self._select_generation_samples(
                    generation_samples,
                    limit=self.config.trainer.val_generations_to_log,
                    seed=42,
                    sort_key=lambda sample: sample.prompt,
                )

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(generation_samples)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {
            "val/reward_score": self.val_reward_score,
            **val_reward_metrics,
            **val_length_metrics,
        }

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _select_filtered_sample_idxs(self, uids: Any, filter_scores: list[float]) -> list[int]:
        uid2scores = defaultdict(list)
        for uid, score in zip(uids, filter_scores):
            uid2scores[uid].append(score)

        uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
        kept_uids = {
            uid
            for uid, avg_score in uid2mean.items()
            if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
        }
        kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
        if len(kept_sample_idxs) == 0:
            raise RuntimeError("No sample is kept after filtering. Please check your data.")

        return kept_sample_idxs

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        scorer_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # Pop generation inputs while retaining the driver-owned raw
            # messages for any post-rollout consumer.
            gen_batch = self._pop_rollout_inputs(
                new_batch,
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                baseline_batch = new_batch.union(gen_baseline_output)
                baseline_rollout_config = dict(self.config.worker.rollout.to_dict())
                baseline_rollout_config.update({"temperature": 0.0, "n": 1})
                self._attach_grounding_consistency_reward_inputs(
                    baseline_batch, baseline_rollout_config, cache_token=f"train-step-{self.global_step}"
                )
                reward_baseline_tensor, _ = ray.get(
                    self.reward_fn.compute_reward.remote(self._reward_rpc_input(baseline_batch))
                )
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output, baseline_batch

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                filter_rollout_config = dict(self.config.worker.rollout.to_dict())
                cache_token = f"train-step-{self.global_step}"
                # When the filter decision does not depend on grounding rewards, filter on a
                # rule-based reward pass first and run detection only for the kept groups
                # (skipping groups without a correct answer). Kept samples end up with the
                # same token_level_scores as the detect-everything order.
                use_two_pass_reward = (
                    self.grounding_consistency_scorer is not None
                    and self.config.algorithm.filter_key in {"accuracy", "format"}
                )
                if use_two_pass_reward:
                    _, pre_reward_metrics = ray.get(
                        self.reward_fn.compute_reward.remote(self._reward_rpc_input(new_batch))
                    )
                    for k, v in pre_reward_metrics.items():
                        # accuracy/format metrics still describe all candidates; grounding rewards
                        # are only computed for kept samples and get appended after the second pass
                        if k != "grounding_consistency":
                            all_metrics[k].extend(v)

                    kept_sample_idxs = self._select_filtered_sample_idxs(
                        new_batch.non_tensor_batch["uid"],
                        pre_reward_metrics[self.config.algorithm.filter_key],
                    )
                    accuracy_values = pre_reward_metrics.get("accuracy")
                    new_batch = new_batch[kept_sample_idxs]
                    eligible_sample_mask = None
                    if accuracy_values is not None:
                        eligible_sample_mask = compute_group_eligibility_mask(
                            new_batch.non_tensor_batch["uid"],
                            [accuracy_values[idx] for idx in kept_sample_idxs],
                        )
                    grounding_reward_result = self._attach_grounding_consistency_reward_inputs(
                        new_batch, filter_rollout_config, eligible_sample_mask, cache_token
                    )
                    reward_tensor, reward_metrics = ray.get(
                        self.reward_fn.compute_reward.remote(self._reward_rpc_input(new_batch))
                    )
                    if "grounding_consistency" in reward_metrics:
                        all_metrics["grounding_consistency"].extend(reward_metrics["grounding_consistency"])
                    new_batch.batch["token_level_scores"] = reward_tensor
                    new_batch.non_tensor_batch["reward_details"] = np.array(
                        self._build_reward_details(reward_metrics, len(new_batch)),
                        dtype=object,
                    )
                else:
                    grounding_reward_result = self._attach_grounding_consistency_reward_inputs(
                        new_batch, filter_rollout_config, cache_token=cache_token
                    )
                    reward_tensor, reward_metrics = ray.get(
                        self.reward_fn.compute_reward.remote(self._reward_rpc_input(new_batch))
                    )
                    new_batch.batch["token_level_scores"] = reward_tensor
                    new_batch.non_tensor_batch["reward_details"] = np.array(
                        self._build_reward_details(reward_metrics, len(new_batch)),
                        dtype=object,
                    )
                    for k, v in reward_metrics.items():
                        all_metrics[k].extend(v)

                    kept_sample_idxs = self._select_filtered_sample_idxs(
                        new_batch.non_tensor_batch["uid"],
                        reward_metrics[self.config.algorithm.filter_key],
                    )
                    new_batch = new_batch[kept_sample_idxs]

                if grounding_reward_result is not None:
                    for key, value in grounding_reward_result.metrics.items():
                        scorer_metrics[key].append(value)

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                if scorer_metrics:
                    metrics.update(
                        {
                            key: (
                                float(np.sum(values))
                                if "detection_time_s" in key or "detection_request_count" in key
                                else float(np.mean(values))
                            )
                            for key, values in scorer_metrics.items()
                        }
                    )

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            train_reward_metrics: Optional[dict[str, list[Any]]] = None
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    grounding_reward_result = self._attach_grounding_rewards(
                        batch,
                        dict(self.config.worker.rollout.to_dict()),
                        reward_fn=self.reward_fn,
                        cache_token=f"train-step-{self.global_step}",
                    )
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(self._reward_rpc_input(batch))

                self._latest_vision_metrics = {}
                if needs_full_vocab_visual_sensitivity(self.config.algorithm):
                    with timer("old_aux-old_full-vocab-visual-sensitivity", timing_raw):
                        sensitivity_output = self._compute_old_log_probs_and_full_vocab_visual_sensitivity(batch)
                        sensitivity_metrics = sensitivity_output.meta_info.get("visual_sensitivity_metrics", {})
                        for key, values in sensitivity_metrics.items():
                            if values:
                                metrics[key] = float(np.mean(values))
                        sensitivity_output.meta_info.pop("visual_sensitivity_metrics", None)
                        batch = batch.union(sensitivity_output)
                else:
                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                if needs_auxiliary_log_probs(self.config.algorithm):
                    if needs_full_vocab_visual_sensitivity(self.config.algorithm):
                        if needs_incremental_auxiliary(self.config.algorithm):
                            with timer("incremental_old", timing_raw):
                                incremental_old_log_probs = self._compute_incremental_old_log_probs(batch)
                                batch = batch.union(incremental_old_log_probs)
                    else:
                        with timer("aux_old", timing_raw):
                            auxiliary_old_log_probs = self._compute_auxiliary_old_log_probs(batch)
                            batch = batch.union(auxiliary_old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        train_reward_metrics = reward_metrics
                        metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})
                        if grounding_reward_result is not None:
                            metrics.update(grounding_reward_result.metrics)
                        batch.batch["token_level_scores"] = reward_tensor
                        if "reward_details" not in batch.non_tensor_batch:
                            batch.non_tensor_batch["reward_details"] = np.array(
                                self._build_reward_details(reward_metrics, len(batch)), dtype=object
                            )
                    else:
                        metrics["reward/overall"] = batch.batch["token_level_scores"].sum(-1).mean().item()
                    if "token_level_scores" in batch.batch and grounding_reward_result is not None:
                        metrics.update(grounding_reward_result.metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                self._maybe_log_train_generations(batch, reward_metrics=train_reward_metrics)

                perception_reasoning_config = self._build_perception_reasoning_loss_config()
                batch.meta_info["perception_reasoning_config"] = perception_reasoning_config
                self._maybe_attach_region_token_mask(batch)
                if has_perception_reasoning(perception_reasoning_config):
                    shaping_context = build_sensitivity_advantage_shaping_context(perception_reasoning_config, batch)
                    if shaping_context is not None:
                        batch.meta_info["advantage_shaping_context"] = shaping_context

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
