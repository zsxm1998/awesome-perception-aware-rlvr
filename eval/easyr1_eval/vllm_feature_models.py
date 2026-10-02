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
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from verl.utils.perturbations.feature_views import perturb_visual_features


ENV_CONFIG = "EASYR1_VLLM_FEATURE_PERTURBATION_JSON"
ENV_CONFIG_PATH = "EASYR1_VLLM_FEATURE_PERTURBATION_CONFIG_PATH"
ENV_MARKER = "EASYR1_VLLM_FEATURE_PERTURBATION_MARKER"

logger = logging.getLogger(__name__)
_REGISTERED = False


def register_vllm_feature_models() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm import ModelRegistry

    registrations = {
        "Qwen3VLForConditionalGeneration": "eval.easyr1_eval.vllm_feature_models:EasyR1Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration": "eval.easyr1_eval.vllm_feature_models:EasyR1Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration": "eval.easyr1_eval.vllm_feature_models:EasyR1Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration": "eval.easyr1_eval.vllm_feature_models:EasyR1Qwen3_5MoeForConditionalGeneration",
    }
    for arch, target in registrations.items():
        ModelRegistry.register_model(arch, target)
    _REGISTERED = True


class EasyR1VLLMFeatureMixin:
    def _process_image_input(self, image_input):
        config = _load_feature_config()
        if not _feature_enabled(config):
            return super()._process_image_input(image_input)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        if getattr(self, "use_data_parallel", False):
            raise ValueError("vLLM feature-space perturbations require mm_encoder_tp_mode != 'data'.")

        source_chunks = None
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            raw_sizes = grid_thw.prod(-1).detach().cpu().long().tolist()
            source_chunks = list(torch.split(pixel_values.detach(), raw_sizes, dim=0))
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        image_embeds = self._easy_perturb_vllm_image_embeds(
            image_embeds=image_embeds,
            image_grid_thw=grid_thw,
            source_chunks=source_chunks,
            config=config,
        )

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input):
        if _feature_enabled(_load_feature_config()):
            raise ValueError(
                "vLLM feature-space perturbations currently support image inputs only; video inputs are unsupported."
            )
        return super()._process_video_input(video_input)

    def _easy_perturb_vllm_image_embeds(
        self,
        *,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        source_chunks: list[torch.Tensor] | None,
        config: dict[str, Any],
    ) -> torch.Tensor:
        visual_dim = int(getattr(self, "visual_dim", getattr(self.config.vision_config, "out_hidden_size")))
        deepstack_num_level = int(getattr(self, "deepstack_num_level", 0))
        expected_dim = visual_dim * (1 + deepstack_num_level)
        if int(image_embeds.shape[-1]) != expected_dim:
            raise ValueError(
                f"Unexpected vLLM visual embedding dim {int(image_embeds.shape[-1])}; expected {expected_dim}."
            )

        merge_size = int(self.visual.spatial_merge_size)
        sizes = (image_grid_thw.prod(-1) // merge_size // merge_size).detach().cpu().long().tolist()
        embed_chunks = list(torch.split(image_embeds, sizes, dim=0))
        perturbed_chunks: list[torch.Tensor] = []
        first_batch_l2: dict[str, float] = {}

        for image_ordinal, (chunk, grid_row) in enumerate(zip(embed_chunks, image_grid_thw)):
            source = source_chunks[image_ordinal] if source_chunks is not None else chunk.detach()
            sample_key = _image_sample_key(source, grid_row)
            main = chunk[:, :visual_dim]
            deepstack = [
                chunk[:, visual_dim * (level + 1) : visual_dim * (level + 2)] for level in range(deepstack_num_level)
            ]
            main_out, deepstack_out, diagnostics = perturb_visual_features(
                main,
                deepstack,
                image_grid_thw=grid_row.view(1, 3),
                merge_size=merge_size,
                perturbation_type=str(config["type"]),
                params=dict(config.get("params") or {}),
                global_seed=int(config["seed"]),
                sample_id=sample_key,
            )
            by_level = {item["level"]: float(item["l2_change"]) for item in diagnostics}
            for level_name, l2_change in by_level.items():
                first_batch_l2[level_name] = first_batch_l2.get(level_name, 0.0) + l2_change
            if deepstack_out is None:
                deepstack_out = []
            perturbed_chunks.append(torch.cat([main_out, *deepstack_out], dim=-1))

        _record_activation(config, first_batch_l2)
        return torch.cat(perturbed_chunks, dim=0)


def _load_feature_config() -> dict[str, Any]:
    config_path = os.environ.get(ENV_CONFIG_PATH)
    if config_path:
        path = Path(config_path)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raw = os.environ.get(ENV_CONFIG)
    if not raw:
        return {"type": "none", "params": {}, "seed": 0}
    return json.loads(raw)


def _feature_enabled(config: dict[str, Any]) -> bool:
    return str(config.get("type") or "none") in {"feature_divergence", "feature_consistency"}


def _image_sample_key(source: torch.Tensor, grid_row: torch.Tensor) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(tuple(int(item) for item in grid_row.detach().cpu().long().tolist())).encode("utf-8"))
    cpu = source.detach().contiguous().view(torch.uint8).cpu()
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(str(source.dtype).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _record_activation(config: dict[str, Any], l2_by_level: dict[str, float]) -> None:
    marker = config.get("marker_path") or os.environ.get(ENV_MARKER)
    if not marker:
        return
    path = Path(str(marker))
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
        payload["count"] = int(payload.get("count") or 0) + 1
        payload["type"] = config.get("type")
        payload["seed"] = config.get("seed")
        if "first_batch_l2" not in payload:
            payload["first_batch_l2"] = l2_by_level
            logger.info("EasyR1 vLLM feature perturbation activated: %s", json.dumps(l2_by_level, sort_keys=True))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fcntl.flock(lock, fcntl.LOCK_UN)


try:
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    class EasyR1Qwen3VLForConditionalGeneration(EasyR1VLLMFeatureMixin, Qwen3VLForConditionalGeneration):
        pass

except Exception:  # pragma: no cover
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

    class EasyR1Qwen3VLMoeForConditionalGeneration(EasyR1VLLMFeatureMixin, Qwen3VLMoeForConditionalGeneration):
        pass

except Exception:  # pragma: no cover
    pass

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    class EasyR1Qwen3_5ForConditionalGeneration(EasyR1VLLMFeatureMixin, Qwen3_5ForConditionalGeneration):
        pass

except Exception:  # pragma: no cover
    pass

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

    class EasyR1Qwen3_5MoeForConditionalGeneration(EasyR1VLLMFeatureMixin, Qwen3_5MoeForConditionalGeneration):
        pass

except Exception:  # pragma: no cover
    pass
