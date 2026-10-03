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
Implement Actor
"""

import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...models.transformers.flash_attention_utils import use_model_level_visual_corruption
from ...protocol import DataProto, collate_multi_modal_inputs
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...trainer.perception_reasoning_loss import (
    compute_perception_reasoning_policy_loss,
    current_policy_entropy_requires_grad,
    has_perception_reasoning,
    needs_current_policy_entropy,
)
from ...trainer.visual_sensitivity import (
    _required_full_vocab_component_names,
    combine_full_vocab_visual_sensitivity_scores,
    compute_full_vocab_visual_sensitivity_components,
    compute_full_vocab_visual_sensitivity_scores,
)
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


def _pad_and_slice_mm_feature_indices(feature_indices: torch.Tensor, pad_size: int) -> torch.Tensor:
    if pad_size > 0:
        feature_indices = torch.nn.functional.pad(feature_indices, (0, pad_size), value=-1)
    return slice_input_tensor(feature_indices, dim=-1, padding=False)


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
        # Keep entropy eager: the chunked helper has shape-dependent Python
        # loops and tends to trigger expensive recompiles under highly dynamic
        # multimodal batches.
        self.entropy_from_logits = VF.entropy_from_logits

    def _get_actor_model_config(self):
        module = self.actor_module
        if isinstance(module, FSDP):
            module = module._fsdp_wrapped_module
        return getattr(module, "config", None)

    def _requires_single_sample_sp_microbatch(self) -> bool:
        if self.config.ulysses_size <= 1:
            return False
        model_config = self._get_actor_model_config()
        return getattr(model_config, "model_type", None) == "qwen3_5"

    def _visual_corruption_context(
        self,
        visual_corruption: dict[str, Any] | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        if visual_corruption is None:
            return nullcontext()
        if self.config.ulysses_size > 1:
            raise ValueError("Model-level visual corruption currently requires worker.actor.ulysses_size=1.")

        model_config = self._get_actor_model_config()
        model_type = getattr(model_config, "model_type", None)
        if model_type not in {"qwen2_5_vl", "qwen3_vl"}:
            raise ValueError(
                f"Model-level visual corruption currently supports qwen2_5_vl and qwen3_vl, but got {model_type!r}."
            )
        vision_start_token_id = getattr(model_config, "vision_start_token_id", None)
        vision_end_token_id = getattr(model_config, "vision_end_token_id", None)
        if vision_start_token_id is None or vision_end_token_id is None:
            raise ValueError(f"Model config {model_type!r} must define vision_start_token_id and vision_end_token_id.")

        return use_model_level_visual_corruption(
            name=visual_corruption["name"],
            kwargs=visual_corruption.get("kwargs"),
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_start_token_id=int(vision_start_token_id),
            vision_end_token_id=int(vision_end_token_id),
        )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        return_entropy: bool = False,
        entropy_requires_grad: bool = True,
        entropy_top_p: float = 1.0,
        return_hidden_visual_scores: bool = False,
        visual_token_ids: list[int] | tuple[int, ...] | None = None,
        hidden_visual_metric: str = "cosine",
        model_level_visual_corruption: dict[str, Any] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        hidden_state_kwargs = {"output_hidden_states": True} if return_hidden_visual_scores else {}
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = collate_multi_modal_inputs(
                micro_batch["multi_modal_inputs"],
                device=input_ids.device,
            )
        else:
            multi_modal_inputs = {}

        use_padding_free = self.config.padding_free and model_level_visual_corruption is None
        if use_padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            mm_feature_indices = {}
            if self.config.ulysses_size > 1 and multi_modal_inputs:
                model_config = self._get_actor_model_config()
                for token_attr, input_key, index_key in (
                    ("image_token_id", "pixel_values", "image_token_feature_indices"),
                    ("video_token_id", "pixel_values_videos", "video_token_feature_indices"),
                ):
                    token_id = getattr(model_config, token_attr, None)
                    if token_id is None or input_key not in multi_modal_inputs:
                        continue

                    token_mask = input_ids_rmpad == token_id
                    feature_indices = token_mask.long().cumsum(dim=-1) - 1
                    mm_feature_indices[index_key] = feature_indices.masked_fill(~token_mask, -1)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )
                for key, feature_indices in mm_feature_indices.items():
                    multi_modal_inputs[key] = _pad_and_slice_mm_feature_indices(feature_indices, pad_size)

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
                **hidden_state_kwargs,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            entropy_rmpad = None
            if return_entropy:
                entropy_rmpad = self._entropy(logits_rmpad, entropy_requires_grad, entropy_top_p)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                if return_entropy:
                    entropy_rmpad = gather_outputs_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
            if return_entropy:
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
            hidden_visual_scores = None
            if return_hidden_visual_scores:
                if not output.hidden_states:
                    raise ValueError("hidden_state_similarity requires model outputs to include hidden_states.")
                visual_mask, response_mask_bool = self._prepare_hidden_visual_masks(
                    input_ids=input_ids,
                    response_length=response_length,
                    response_mask=micro_batch.get("response_mask"),
                    visual_token_ids=visual_token_ids,
                )
                hidden_scores_accum = None
                num_layers = 0
                for layer_hidden in output.hidden_states:
                    layer_hidden = layer_hidden.squeeze(0)
                    if self.config.ulysses_size > 1:
                        layer_hidden = gather_outputs_and_unpad(
                            layer_hidden, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    full_hidden = pad_input(
                        hidden_states=layer_hidden, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                    layer_scores = self._compute_hidden_visual_scores_for_layer(
                        layer_hidden=full_hidden,
                        visual_mask=visual_mask,
                        response_length=response_length,
                        metric=hidden_visual_metric,
                    )
                    if hidden_scores_accum is None:
                        hidden_scores_accum = layer_scores
                    else:
                        hidden_scores_accum = hidden_scores_accum + layer_scores
                    num_layers += 1
                    del full_hidden, layer_scores
                hidden_visual_scores = self._finalize_hidden_visual_scores(
                    scores_accum=hidden_scores_accum,
                    num_layers=num_layers,
                    response_mask_bool=response_mask_bool,
                )
                del visual_mask, response_mask_bool
        else:
            with self._visual_corruption_context(
                model_level_visual_corruption,
                input_ids=input_ids,
                attention_mask=attention_mask,
            ):
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **hidden_state_kwargs,
                )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
            if return_entropy:
                entropy = self._entropy(logits, entropy_requires_grad, entropy_top_p, padded=True)
            hidden_visual_scores = None
            if return_hidden_visual_scores:
                hidden_visual_scores = self._compute_hidden_visual_scores(
                    hidden_states=output.hidden_states,
                    input_ids=input_ids,
                    response_length=response_length,
                    response_mask=micro_batch.get("response_mask"),
                    visual_token_ids=visual_token_ids,
                    metric=hidden_visual_metric,
                )

        if return_hidden_visual_scores:
            return log_probs, entropy if return_entropy else None, hidden_visual_scores
        if return_entropy:
            return log_probs, entropy
        return log_probs

    def _entropy(
        self, logits: torch.Tensor, requires_grad: bool, top_p: float = 1.0, padded: bool = False
    ) -> torch.Tensor:
        """Token entropy of ``logits``; ``top_p < 1`` gives the entropy of the top-p truncated distribution
        (no gradient, used for the rollout-policy entropy of batch-level entropy masks)."""
        if top_p < 1.0:
            if requires_grad:
                raise ValueError("the top-p truncated entropy is only computed without gradient.")
            with torch.no_grad():
                return VF.top_p_entropy_from_logits(logits.detach(), top_p)
        entropy_logits = logits if requires_grad else logits.detach()
        if padded:
            with torch.enable_grad() if requires_grad else torch.no_grad():
                return self.entropy_from_logits(entropy_logits)
        with nullcontext() if requires_grad else torch.no_grad():
            return self.entropy_from_logits(entropy_logits)

    def _compute_hidden_visual_scores(
        self,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        input_ids: torch.Tensor,
        response_length: int,
        response_mask: torch.Tensor | None,
        visual_token_ids: list[int] | tuple[int, ...] | None,
        metric: str,
    ) -> torch.Tensor:
        if not hidden_states:
            raise ValueError("hidden_state_similarity requires model outputs to include hidden_states.")
        visual_mask, response_mask_bool = self._prepare_hidden_visual_masks(
            input_ids=input_ids,
            response_length=response_length,
            response_mask=response_mask,
            visual_token_ids=visual_token_ids,
        )
        scores_accum = None
        num_layers = 0
        for layer_hidden in hidden_states:
            layer_hidden = layer_hidden[:, -input_ids.size(1) :, :]
            layer_scores = self._compute_hidden_visual_scores_for_layer(
                layer_hidden=layer_hidden,
                visual_mask=visual_mask,
                response_length=response_length,
                metric=metric,
            )
            if scores_accum is None:
                scores_accum = layer_scores
            else:
                scores_accum = scores_accum + layer_scores
            num_layers += 1

        return self._finalize_hidden_visual_scores(
            scores_accum=scores_accum,
            num_layers=num_layers,
            response_mask_bool=response_mask_bool,
        )

    def _prepare_hidden_visual_masks(
        self,
        input_ids: torch.Tensor,
        response_length: int,
        response_mask: torch.Tensor | None,
        visual_token_ids: list[int] | tuple[int, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not visual_token_ids:
            raise ValueError("hidden_state_similarity requires non-empty visual_token_ids from the driver.")

        visual_ids = torch.tensor(list(visual_token_ids), dtype=input_ids.dtype, device=input_ids.device)
        visual_mask = torch.isin(input_ids, visual_ids)
        if response_mask is not None:
            response_mask_bool = response_mask.to(torch.bool)
        else:
            response_mask_bool = torch.ones(
                (input_ids.size(0), response_length), dtype=torch.bool, device=input_ids.device
            )
        return visual_mask, response_mask_bool

    def _compute_hidden_visual_scores_for_layer(
        self,
        layer_hidden: torch.Tensor,
        visual_mask: torch.Tensor,
        response_length: int,
        metric: str,
    ) -> torch.Tensor:
        response_hidden = layer_hidden[:, -response_length:, :]
        layer_scores = response_hidden.new_zeros((layer_hidden.size(0), response_length), dtype=torch.float32)
        for row_idx in range(layer_hidden.size(0)):
            row_visual_hidden = layer_hidden[row_idx, visual_mask[row_idx]]
            if row_visual_hidden.numel() == 0:
                raise ValueError(
                    "hidden_state_similarity could not find visual tokens in a training sample. "
                    "Check algorithm.visual_token and the processed multimodal prompt."
                )
            row_response_hidden = response_hidden[row_idx]
            layer_scores[row_idx] = self._hidden_similarity_scores(
                response_hidden=row_response_hidden,
                visual_hidden=row_visual_hidden,
                metric=metric,
            )
        return layer_scores

    @staticmethod
    def _finalize_hidden_visual_scores(
        scores_accum: torch.Tensor | None,
        num_layers: int,
        response_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        if scores_accum is None or num_layers <= 0:
            raise ValueError("hidden_state_similarity requires at least one hidden-state layer.")
        scores = scores_accum / num_layers
        scores = scores.masked_fill(~response_mask_bool, 0.0)
        return scores.detach()

    @staticmethod
    def _hidden_similarity_scores(
        response_hidden: torch.Tensor,
        visual_hidden: torch.Tensor,
        metric: str,
    ) -> torch.Tensor:
        response_hidden = response_hidden.float()
        visual_hidden = visual_hidden.float()
        if metric == "cosine":
            response_norm = F.normalize(response_hidden, p=2, dim=-1)
            visual_norm = F.normalize(visual_hidden, p=2, dim=-1)
            return response_norm.matmul(visual_norm.transpose(0, 1)).mean(dim=-1)
        if metric == "dot":
            return response_hidden.matmul(visual_hidden.transpose(0, 1)).mean(dim=-1)
        diff = response_hidden.unsqueeze(1) - visual_hidden.unsqueeze(0)
        if metric == "l1":
            return -diff.abs().sum(dim=-1).mean(dim=-1)
        if metric == "l2":
            dist_sq = diff.pow(2).sum(dim=-1).clamp_min(0.0)
            return -torch.sqrt(dist_sq + 1e-6).mean(dim=-1)
        raise ValueError(f"Unsupported visual_sensitivity_hidden_metric: {metric}")

    def _forward_micro_batch_response_logits(
        self, micro_batch: dict[str, torch.Tensor], temperature: float
    ) -> torch.Tensor | dict[str, torch.Tensor | tuple[int, int]]:
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = collate_multi_modal_inputs(
                micro_batch["multi_modal_inputs"],
                device=input_ids.device,
            )
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            mm_feature_indices = {}
            if self.config.ulysses_size > 1 and multi_modal_inputs:
                model_config = self._get_actor_model_config()
                for token_attr, input_key, index_key in (
                    ("image_token_id", "pixel_values", "image_token_feature_indices"),
                    ("video_token_id", "pixel_values_videos", "video_token_feature_indices"),
                ):
                    token_id = getattr(model_config, token_attr, None)
                    if token_id is None or input_key not in multi_modal_inputs:
                        continue
                    token_mask = input_ids_rmpad == token_id
                    feature_indices = token_mask.long().cumsum(dim=-1) - 1
                    mm_feature_indices[index_key] = feature_indices.masked_fill(~token_mask, -1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                for key, feature_indices in mm_feature_indices.items():
                    multi_modal_inputs[key] = _pad_and_slice_mm_feature_indices(feature_indices, pad_size)

            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)
            if self.config.ulysses_size > 1:
                logits_rmpad = gather_outputs_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            response_position_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            response_position_mask[:, -response_length - 1 : -1] = True
            response_position_mask = response_position_mask & attention_mask.to(torch.bool)
            flat_response_mask = response_position_mask.reshape(-1)[indices]
            flat_positions = torch.arange(batch_size * seqlen, device=input_ids.device)[indices][flat_response_mask]
            response_rows = flat_positions // seqlen
            response_cols = flat_positions % seqlen - (seqlen - response_length - 1)
            logits = logits_rmpad[flat_response_mask]
            return {
                "logits": logits.contiguous(),
                "response_rows": response_rows,
                "response_cols": response_cols,
                "response_shape": (batch_size, response_length),
            }

        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            use_cache=False,
        )
        logits: torch.Tensor = output.logits
        logits.div_(temperature)
        return logits[:, -response_length - 1 : -1, :].contiguous()

    def _log_probs_from_response_logits(
        self,
        logits_output: torch.Tensor | dict[str, torch.Tensor | tuple[int, int]],
        responses: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(logits_output, dict):
            response_rows = logits_output["response_rows"]
            response_cols = logits_output["response_cols"]
            response_shape = logits_output["response_shape"]
            labels = responses[response_rows, response_cols]
            sparse_log_probs = self.log_probs_from_logits(logits_output["logits"], labels)
            log_probs = sparse_log_probs.new_zeros((int(response_shape[0]), int(response_shape[1])))
            log_probs[response_rows, response_cols] = sparse_log_probs
            return log_probs

        return self.log_probs_from_logits(logits_output, responses)

    def _entropy_from_response_logits(
        self,
        logits_output: torch.Tensor | dict[str, torch.Tensor | tuple[int, int]],
    ) -> torch.Tensor:
        if isinstance(logits_output, dict):
            response_rows = logits_output["response_rows"]
            response_cols = logits_output["response_cols"]
            response_shape = logits_output["response_shape"]
            sparse_entropies = self.entropy_from_logits(logits_output["logits"])
            entropies = sparse_entropies.new_zeros((int(response_shape[0]), int(response_shape[1])))
            entropies[response_rows, response_cols] = sparse_entropies
            return entropies

        return self.entropy_from_logits(logits_output)

    @staticmethod
    def _full_vocab_visual_sensitivity_from_logits_outputs(
        logits_output: torch.Tensor | dict[str, torch.Tensor | tuple[int, int]],
        auxiliary_logits_output: torch.Tensor | dict[str, torch.Tensor | tuple[int, int]],
        response_mask: torch.Tensor,
        metric: str,
        jsd_weight: float,
        entropy_gate: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(logits_output, dict):
            if not isinstance(auxiliary_logits_output, dict):
                raise RuntimeError("Normal and auxiliary visual sensitivity forwards used different output formats.")
            response_rows = logits_output["response_rows"]
            response_cols = logits_output["response_cols"]
            if not torch.equal(response_rows, auxiliary_logits_output["response_rows"]) or not torch.equal(
                response_cols, auxiliary_logits_output["response_cols"]
            ):
                raise RuntimeError("Normal and auxiliary response token positions do not match.")

            sparse_components = compute_full_vocab_visual_sensitivity_components(
                logits=logits_output["logits"],
                corrupted_logits=auxiliary_logits_output["logits"],
                component_names=_required_full_vocab_component_names(metric=metric, entropy_gate=entropy_gate),
            )
            response_shape = logits_output["response_shape"]
            scatter_shape = (int(response_shape[0]), int(response_shape[1]))
            components = {}
            for name, sparse_values in sparse_components.items():
                dense_values = sparse_values.new_zeros(scatter_shape)
                dense_values[response_rows, response_cols] = sparse_values
                components[name] = dense_values
            return combine_full_vocab_visual_sensitivity_scores(
                components=components,
                response_mask=response_mask,
                metric=metric,
                jsd_weight=jsd_weight,
                entropy_gate=entropy_gate,
            )

        return compute_full_vocab_visual_sensitivity_scores(
            logits=logits_output,
            corrupted_logits=auxiliary_logits_output,
            response_mask=response_mask,
            metric=metric,
            jsd_weight=jsd_weight,
            entropy_gate=entropy_gate,
        )

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(
        self, data: DataProto, return_entropy: bool = False, entropy_top_p: float = 1.0
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        model_level_visual_corruption = data.meta_info.get("model_level_visual_corruption")
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        force_single_sample_sp = self._requires_single_sample_sp_microbatch()
        if force_single_sample_sp:
            micro_batches = data.split(1)
            batch_idx_list = None
        elif self.config.dynamic_batching and model_level_visual_corruption is None:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            # Effective-token dynamic batching assumes the forward is padding-free.
            # Model-level visual interventions require a padded 2D batch, so short
            # samples could otherwise be packed far beyond the configured sample
            # limit and materialize an oversized dense logits tensor.
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)
            batch_idx_list = None

        log_probs_lst = []
        entropy_lst = [] if return_entropy else None
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            output = self._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                return_entropy=return_entropy,
                entropy_top_p=entropy_top_p,
                model_level_visual_corruption=model_level_visual_corruption,
            )
            if return_entropy:
                log_probs, entropy = output
                entropy_lst.append(entropy)
            else:
                log_probs = output
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropies = torch.concat(entropy_lst, dim=0) if return_entropy else None

        if batch_idx_list is not None:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if return_entropy:
                entropies = restore_dynamic_batch(entropies, batch_idx_list)

        if return_entropy:
            return log_probs, entropies
        return log_probs

    @torch.no_grad()
    def compute_visual_sensitivity_scores(self, data: DataProto) -> tuple[torch.Tensor, dict[str, list[float]]]:
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        metric = data.meta_info["visual_sensitivity_metric"]
        jsd_weight = data.meta_info.get("visual_sensitivity_jsd_weight", 0.5)
        entropy_gate = data.meta_info.get("visual_sensitivity_entropy_gate", "none")
        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "auxiliary_input_ids",
            "auxiliary_attention_mask",
            "auxiliary_position_ids",
        ]
        non_tensor_select_keys = ["multi_modal_inputs", "auxiliary_multi_modal_inputs"]
        data = data.select(select_keys, non_tensor_select_keys)

        force_single_sample_sp = self._requires_single_sample_sp_microbatch()
        if force_single_sample_sp:
            micro_batches = data.split(1)
            batch_idx_list = None
        elif self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)
            batch_idx_list = None

        scores_lst = []
        metrics: dict[str, list[float]] = defaultdict(list)
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute visual sensitivity", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            normal_inputs = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "position_ids": model_inputs["position_ids"],
                "responses": model_inputs["responses"],
            }
            if "multi_modal_inputs" in model_inputs:
                normal_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]

            auxiliary_inputs = {
                "input_ids": model_inputs["auxiliary_input_ids"],
                "attention_mask": model_inputs["auxiliary_attention_mask"],
                "position_ids": model_inputs["auxiliary_position_ids"],
                "responses": model_inputs["responses"],
            }
            if "auxiliary_multi_modal_inputs" in model_inputs:
                auxiliary_inputs["multi_modal_inputs"] = model_inputs["auxiliary_multi_modal_inputs"]

            logits_output = self._forward_micro_batch_response_logits(normal_inputs, temperature=temperature)
            auxiliary_logits_output = self._forward_micro_batch_response_logits(
                auxiliary_inputs, temperature=temperature
            )
            scores, batch_metrics = self._full_vocab_visual_sensitivity_from_logits_outputs(
                logits_output=logits_output,
                auxiliary_logits_output=auxiliary_logits_output,
                response_mask=model_inputs["response_mask"],
                metric=metric,
                jsd_weight=jsd_weight,
                entropy_gate=entropy_gate,
            )
            scores_lst.append(scores)
            for key, value in batch_metrics.items():
                metrics[key].append(value)

        scores = torch.concat(scores_lst, dim=0)
        if batch_idx_list is not None:
            scores = restore_dynamic_batch(scores, batch_idx_list)

        return scores, metrics

    @torch.no_grad()
    def compute_log_prob_and_visual_sensitivity_scores(
        self, data: DataProto
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        metric = data.meta_info["visual_sensitivity_metric"]
        jsd_weight = data.meta_info.get("visual_sensitivity_jsd_weight", 0.5)
        entropy_gate = data.meta_info.get("visual_sensitivity_entropy_gate", "none")
        return_decremental_log_probs = data.meta_info.get("return_decremental_log_probs", False)
        return_decremental_entropy = data.meta_info.get("return_decremental_entropy", False)
        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "auxiliary_input_ids",
            "auxiliary_attention_mask",
            "auxiliary_position_ids",
        ]
        non_tensor_select_keys = ["multi_modal_inputs", "auxiliary_multi_modal_inputs"]
        data = data.select(select_keys, non_tensor_select_keys)

        force_single_sample_sp = self._requires_single_sample_sp_microbatch()
        if force_single_sample_sp:
            micro_batches = data.split(1)
            batch_idx_list = None
        elif self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)
            batch_idx_list = None

        log_probs_lst = []
        decremental_log_probs_lst = [] if return_decremental_log_probs else None
        decremental_entropy_lst = [] if return_decremental_entropy else None
        scores_lst = []
        metrics: dict[str, list[float]] = defaultdict(list)
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs + visual sensitivity", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            normal_inputs = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "position_ids": model_inputs["position_ids"],
                "responses": model_inputs["responses"],
            }
            if "multi_modal_inputs" in model_inputs:
                normal_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]

            auxiliary_inputs = {
                "input_ids": model_inputs["auxiliary_input_ids"],
                "attention_mask": model_inputs["auxiliary_attention_mask"],
                "position_ids": model_inputs["auxiliary_position_ids"],
                "responses": model_inputs["responses"],
            }
            if "auxiliary_multi_modal_inputs" in model_inputs:
                auxiliary_inputs["multi_modal_inputs"] = model_inputs["auxiliary_multi_modal_inputs"]

            logits_output = self._forward_micro_batch_response_logits(normal_inputs, temperature=temperature)
            log_probs = self._log_probs_from_response_logits(logits_output, model_inputs["responses"])
            auxiliary_logits_output = self._forward_micro_batch_response_logits(
                auxiliary_inputs, temperature=temperature
            )
            if return_decremental_log_probs:
                decremental_log_probs = self._log_probs_from_response_logits(
                    auxiliary_logits_output,
                    model_inputs["responses"],
                )
                decremental_log_probs_lst.append(decremental_log_probs)
            if return_decremental_entropy:
                decremental_entropy = self._entropy_from_response_logits(auxiliary_logits_output)
                decremental_entropy_lst.append(decremental_entropy)
            scores, batch_metrics = self._full_vocab_visual_sensitivity_from_logits_outputs(
                logits_output=logits_output,
                auxiliary_logits_output=auxiliary_logits_output,
                response_mask=model_inputs["response_mask"],
                metric=metric,
                jsd_weight=jsd_weight,
                entropy_gate=entropy_gate,
            )
            log_probs_lst.append(log_probs)
            scores_lst.append(scores)
            for key, value in batch_metrics.items():
                metrics[key].append(value)

        log_probs = torch.concat(log_probs_lst, dim=0)
        scores = torch.concat(scores_lst, dim=0)
        output_tensors = {"old_log_probs": log_probs, "per_token_sensitivity_scores": scores}
        if return_decremental_log_probs:
            output_tensors["decremental_old_log_probs"] = torch.concat(decremental_log_probs_lst, dim=0)
        if return_decremental_entropy:
            output_tensors["decremental_entropies"] = torch.concat(decremental_entropy_lst, dim=0)
        if batch_idx_list is not None:
            for key, value in output_tensors.items():
                output_tensors[key] = restore_dynamic_batch(value, batch_idx_list)

        return output_tensors, metrics

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        loss_config = data.meta_info.get("perception_reasoning_config")
        use_perception_reasoning = has_perception_reasoning(loss_config)
        need_entropy = needs_current_policy_entropy(loss_config)
        entropy_requires_grad = current_policy_entropy_requires_grad(loss_config)
        need_hidden_visual_scores = (
            use_perception_reasoning
            and loss_config is not None
            and loss_config.get("visual_sensitivity_metric") == "hidden_state_similarity"
            and (
                loss_config.get("top_perception_quantile", 1.0) < 1.0
                or loss_config.get("advantage_scaling_method") in {"pgpo", "pepo"}
            )
        )
        advantage_shaping_context = data.meta_info.get("advantage_shaping_context")
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        for optional_key in [
            "decremental_old_log_probs",
            "decremental_entropies",
            "incremental_old_log_probs",
            "incremental_entropies",
            "region_token_mask",
            "per_token_sensitivity_scores",
            "batch_entropy_mask",
            "batch_perception_mask",
        ]:
            if optional_key in data.batch.keys():
                select_keys.append(optional_key)
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                # token mode weights every response token equally across the mini-batch on all ranks;
                # seq mode weights every response (with at least one token) equally, as verl's seq-mean-token-mean
                if self.config.loss_avg_mode == "seq":
                    total_responses = torch.sum(mini_batch.batch["response_mask"].sum(-1) > 0)
                    dist.all_reduce(total_responses, op=dist.ReduceOp.SUM)
                else:
                    total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                    dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                force_single_sample_sp = self._requires_single_sample_sp_microbatch()
                if force_single_sample_sp:
                    micro_batches = mini_batch.split(1)
                elif self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    forward_kwargs = {
                        "temperature": temperature,
                        "return_entropy": need_entropy,
                        "entropy_requires_grad": entropy_requires_grad,
                    }
                    if need_hidden_visual_scores:
                        visual_token_ids = loss_config.get("visual_token_ids")
                        if not visual_token_ids:
                            raise ValueError(
                                "hidden_state_similarity requires visual_token_ids in perception_reasoning_config."
                            )
                        forward_kwargs.update(
                            {
                                "return_hidden_visual_scores": True,
                                "visual_token_ids": visual_token_ids,
                                "hidden_visual_metric": loss_config.get("visual_sensitivity_hidden_metric", "cosine"),
                            }
                        )
                    output = self._forward_micro_batch(model_inputs, **forward_kwargs)
                    hidden_visual_scores = None
                    if need_hidden_visual_scores:
                        log_probs, entropy, hidden_visual_scores = output
                    elif need_entropy:
                        log_probs, entropy = output
                    else:
                        log_probs = output
                        entropy = None

                    if use_perception_reasoning:
                        pg_loss, batch_metrics = compute_perception_reasoning_policy_loss(
                            actor_config=self.config,
                            loss_config=loss_config,
                            log_prob=log_probs,
                            old_log_prob=old_log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            entropy=entropy,
                            decremental_old_log_probs=model_inputs.get("decremental_old_log_probs"),
                            decremental_entropies=model_inputs.get("decremental_entropies"),
                            incremental_old_log_probs=model_inputs.get("incremental_old_log_probs"),
                            incremental_entropies=model_inputs.get("incremental_entropies"),
                            region_token_mask=model_inputs.get("region_token_mask"),
                            per_token_sensitivity_scores=(
                                hidden_visual_scores
                                if hidden_visual_scores is not None
                                else model_inputs.get("per_token_sensitivity_scores")
                            ),
                            advantage_shaping_context=advantage_shaping_context,
                            batch_entropy_mask=model_inputs.get("batch_entropy_mask"),
                            batch_perception_mask=model_inputs.get("batch_perception_mask"),
                        )
                    else:
                        pg_loss, pg_metrics = compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                            tau_positive=self.config.tau_positive,
                            tau_negative=self.config.tau_negative,
                            loss_type=self.config.loss_type,
                            loss_avg_mode=self.config.loss_avg_mode,
                        )
                        batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                        batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                        if entropy is not None and loss_config and loss_config.get("log_entropy", False):
                            batch_metrics["actor/policy_entropy"] = (
                                VF.masked_mean(entropy, response_mask).detach().item()
                            )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        batch_metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    if self.config.loss_avg_mode == "seq":
                        # average_loss(mode="seq") divides by the rows of this micro-batch
                        loss = loss * response_mask.size(0) * self.world_size / total_responses
                    else:
                        loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
