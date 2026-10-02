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

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from .feature_hooks import install_feature_perturbation_hook
from .perturbations import (
    PerturbationConfig,
    apply_pixel_perturbation,
    write_sample_image,
)
from .prompting import render_chat_prompt
from .schemas import EvalSample, GenerationConfig, GenerationOutput


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VLLM_FEATURE_PLUGIN_NAME = "easyr1_vllm_feature"


class Backend(Protocol):
    def generate(self, samples: list[EvalSample], config: GenerationConfig) -> list[list[GenerationOutput]]: ...


@dataclass
class DummyBackend:
    def generate(self, samples: list[EvalSample], config: GenerationConfig) -> list[list[GenerationOutput]]:
        outputs = []
        for sample in samples:
            response = sample.metadata.get("dummy_response")
            if response is None:
                response = str(sample.target)
            outputs.append(
                [GenerationOutput(text=str(response), finish_reason="dummy", truncated=False)] * config.num_samples
            )
        return outputs


def process_eval_image(image, min_pixels: int | None, max_pixels: int | None):
    from PIL import Image

    from verl.utils.dataset import ProcessedImageInput

    if isinstance(image, ProcessedImageInput):
        image = image.image
    elif isinstance(image, str):
        image = _open_image_safely(image)
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            image = _open_image_safely(BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = _open_image_safely(image["path"])
        else:
            raise ValueError("Image dictionaries must contain either non-null 'bytes' or 'path'.")
    elif isinstance(image, bytes):
        image = _open_image_safely(BytesIO(image))

    width, height = image.size
    if max_pixels is not None and width * height > max_pixels:
        width, height = _scaled_size(width, height, max_pixels)
        try:
            image.draft("RGB", (width, height))
        except (AttributeError, OSError):
            pass
        image.thumbnail((width, height), _resample_filter(Image))
    image.load()

    width, height = image.size
    if max_pixels is not None and width * height > max_pixels:
        image = image.resize(_scaled_size(width, height, max_pixels), _resample_filter(Image))

    width, height = image.size
    if min_pixels is not None and width * height < min_pixels:
        image = image.resize(_scaled_size(width, height, min_pixels), _resample_filter(Image))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def source_image_size(image) -> tuple[int, int] | None:
    """(width, height) of an evaluation image without decoding its pixels."""
    from PIL import Image

    from verl.utils.dataset import ProcessedImageInput

    try:
        if isinstance(image, ProcessedImageInput):
            image = image.image
        if isinstance(image, Image.Image):
            return image.size
        if isinstance(image, dict):
            image = BytesIO(image["bytes"]) if image.get("bytes") is not None else image.get("path")
        elif isinstance(image, bytes):
            image = BytesIO(image)
        with _open_image_safely(image) as opened:
            return opened.size
    except Exception:  # noqa: BLE001 - sizes are diagnostics; never fail generation for them
        return None


def qwen_smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Qwen2-VL / Qwen2.5-VL / Qwen3-VL image resizing (same rule as transformers' ``smart_resize``).

    Returns ``(height, width)`` rounded to multiples of ``factor`` (patch size x merge size) with
    the pixel count clamped to ``[min_pixels, max_pixels]``.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def model_input_image_size(width: int, height: int, processor) -> tuple[int, int]:
    """(width, height) of the image the vision tower sees after the processor's own resize.

    Qwen-VL processors snap images to multiples of ``patch_size * merge_size`` within their
    ``min_pixels``/``max_pixels``; absolute-pixel boxes emitted by Qwen2(.5)-VL refer to this size.
    Other processors are assumed not to resize further.
    """
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", None)
    if not patch_size or not merge_size:
        return width, height
    size = getattr(image_processor, "size", None)
    size = size if isinstance(size, dict) else {}
    min_pixels = getattr(image_processor, "min_pixels", None) or size.get("shortest_edge") or 56 * 56
    max_pixels = getattr(image_processor, "max_pixels", None) or size.get("longest_edge") or 14 * 14 * 4 * 1280
    resized_height, resized_width = qwen_smart_resize(
        height, width, factor=int(patch_size) * int(merge_size), min_pixels=int(min_pixels), max_pixels=int(max_pixels)
    )
    return resized_width, resized_height


def image_size_records(sample: EvalSample, processed_images: list, processor) -> list[dict]:
    """Per-image sizes stored with every prediction (needed to read absolute-pixel boxes)."""
    records = []
    for source, processed in zip(sample.images, processed_images):
        width, height = processed.size
        model_width, model_height = model_input_image_size(width, height, processor)
        original = source_image_size(source)
        records.append(
            {
                "original": list(original) if original else None,
                "processed": [width, height],
                "model_input": [model_width, model_height],
            }
        )
    return records


def _open_image_safely(source):
    from PIL import Image

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        return Image.open(source)


def _scaled_size(width: int, height: int, target_pixels: int) -> tuple[int, int]:
    resize_factor = math.sqrt(target_pixels / (width * height))
    return max(1, int(width * resize_factor)), max(1, int(height * resize_factor))


def _resample_filter(image_module):
    return getattr(image_module, "Resampling", image_module).LANCZOS


class VLLMBackend:
    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        limit_images: int = 8,
        dtype: str = "bfloat16",
        perturbation: PerturbationConfig | None = None,
        perturbation_seed: int = 0,
        output_dir: Path | str | None = None,
        save_perturbation_samples: int = 0,
        force_vllm_feature_wrapper: bool = False,
        chat_template: str | None = None,
    ):
        from vllm import LLM

        from verl.utils.tokenizer import get_processor, get_tokenizer

        self.perturbation = perturbation or PerturbationConfig()
        self.perturbation_seed = perturbation_seed
        self.output_dir = Path(output_dir) if output_dir is not None else Path("eval/results/latest")
        self._vllm_feature_marker_path: Path | None = None
        self._vllm_feature_expected_levels: list[str] = []
        if self.perturbation.is_feature and tensor_parallel_size != 1:
            raise ValueError("vLLM feature-space perturbations require --tp 1 in Phase 1.")
        if not (self.perturbation.is_feature or force_vllm_feature_wrapper):
            os.environ.pop("EASYR1_VLLM_FEATURE_PERTURBATION_JSON", None)
            os.environ.pop("EASYR1_VLLM_FEATURE_PERTURBATION_CONFIG_PATH", None)
            os.environ.pop("EASYR1_VLLM_FEATURE_PERTURBATION_MARKER", None)
        active_payload = None
        if self.perturbation.is_feature or force_vllm_feature_wrapper:
            from .vllm_feature_models import ENV_CONFIG, ENV_CONFIG_PATH, ENV_MARKER, register_vllm_feature_models

            _prepare_vllm_feature_plugin_env(self.output_dir)
            register_vllm_feature_models()
            marker_path = self.output_dir / f"vllm_feature_activation_marker_{os.getpid()}.json"
            config_path = self.output_dir / f"vllm_feature_config_{os.getpid()}.json"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            self._vllm_feature_marker_path = marker_path
            if marker_path.exists():
                marker_path.unlink()
            if marker_path.with_suffix(marker_path.suffix + ".lock").exists():
                marker_path.with_suffix(marker_path.suffix + ".lock").unlink()
            active_payload = {
                "type": self.perturbation.type,
                "params": dict(self.perturbation.params),
                "seed": self.perturbation_seed,
                "marker_path": str(marker_path),
            }
            inactive_payload = {**active_payload, "type": "none", "params": {}}
            _write_vllm_feature_config(config_path, inactive_payload)
            os.environ.pop(ENV_CONFIG, None)
            os.environ[ENV_CONFIG_PATH] = str(config_path)
            os.environ[ENV_MARKER] = str(marker_path)
            self._vllm_feature_expected_levels = _expected_vllm_feature_levels(model, trust_remote_code)

        engine_kwargs = {
            "model": model,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "limit_mm_per_prompt": {"image": limit_images},
        }
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        # vLLM V1 snapshots environment variables when EngineCore workers spawn.
        # Build one LLM per perturbation config/seed; never mutate feature env on a live engine.
        self.llm = LLM(**engine_kwargs)
        if active_payload is not None:
            _write_vllm_feature_config(
                Path(os.environ["EASYR1_VLLM_FEATURE_PERTURBATION_CONFIG_PATH"]), active_payload
            )
        self.processor = get_processor(
            model, override_chat_template=chat_template, trust_remote_code=trust_remote_code
        )
        self.tokenizer = get_tokenizer(model, trust_remote_code=trust_remote_code)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.save_perturbation_samples = save_perturbation_samples
        self._saved_perturbation_samples = 0

    def generate(self, samples: list[EvalSample], config: GenerationConfig) -> list[list[GenerationOutput]]:
        from vllm import SamplingParams

        inputs = []
        batch_diagnostics = []
        batch_sizes = []
        has_images = False
        for sample in samples:
            item = {"prompt": self._prompt_text(sample)}
            if sample.images:
                has_images = True
                images, diagnostics = self._prepare_images(sample)
                item["multi_modal_data"] = {"image": images}
                sizes = image_size_records(sample, images, self.processor)
            else:
                diagnostics = []
                sizes = []
            batch_diagnostics.append(diagnostics)
            batch_sizes.append(sizes)
            inputs.append(item)
        sampling = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            n=config.num_samples,
            max_tokens=config.max_new_tokens,
            seed=config.seed,
        )
        completions = self.llm.generate(inputs, sampling_params=sampling)
        if self.perturbation.is_feature and has_images:
            self._validate_vllm_feature_activation()
        return [
            [
                GenerationOutput(
                    text=output.text,
                    finish_reason=getattr(output, "finish_reason", None),
                    stop_reason=getattr(output, "stop_reason", None),
                    token_count=len(getattr(output, "token_ids", []) or []),
                    truncated=str(getattr(output, "finish_reason", "")).lower() == "length",
                    diagnostics={"perturbation": batch_diagnostics[index], "image_sizes": batch_sizes[index]},
                )
                for output in completion.outputs
            ]
            for index, completion in enumerate(completions)
        ]

    def _prepare_images(self, sample: EvalSample) -> tuple[list, list[dict]]:
        images = []
        diagnostics = []
        for image_idx, image in enumerate(sample.images):
            processed = process_eval_image(image, self.min_pixels, self.max_pixels)
            processed, diag = apply_pixel_perturbation(
                processed,
                config=self.perturbation,
                global_seed=self.perturbation_seed,
                sample_id=sample.sample_id,
                image_idx=image_idx,
            )
            if diag:
                self._saved_perturbation_samples = write_sample_image(
                    processed,
                    output_dir=self.output_dir,
                    config=self.perturbation,
                    sample_id=sample.sample_id,
                    image_idx=image_idx,
                    max_samples=self.save_perturbation_samples,
                    current_count=self._saved_perturbation_samples,
                )
                diagnostics.append(diag)
            images.append(processed)
        return images, diagnostics

    def _prompt_text(self, sample: EvalSample) -> str:
        if not sample.messages:
            return sample.prompt
        template_owner = self.processor if self.processor is not None else self.tokenizer
        return render_chat_prompt(sample, template_owner)

    def _validate_vllm_feature_activation(self) -> None:
        if self._vllm_feature_marker_path is None:
            raise RuntimeError("vLLM feature activation marker path was not configured.")
        if not self._vllm_feature_marker_path.exists():
            raise RuntimeError(
                f"vLLM feature perturbation did not activate; missing marker {self._vllm_feature_marker_path}."
            )
        payload = json.loads(self._vllm_feature_marker_path.read_text(encoding="utf-8"))
        if int(payload.get("count") or 0) <= 0:
            raise RuntimeError("vLLM feature perturbation marker has non-positive activation count.")
        l2_by_level = payload.get("first_batch_l2") or {}
        expected = self._vllm_feature_expected_levels or ["main"]
        missing = [level for level in expected if level not in l2_by_level]
        non_positive = [level for level in expected if level in l2_by_level and float(l2_by_level[level]) <= 0.0]
        if missing or non_positive:
            raise RuntimeError(
                "vLLM feature perturbation activation proof failed: "
                f"missing_levels={missing}, non_positive_l2={non_positive}, marker={self._vllm_feature_marker_path}"
            )


def _expected_vllm_feature_levels(model: str, trust_remote_code: bool) -> list[str]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    vision_config = getattr(config, "vision_config", None)
    indexes = getattr(vision_config, "deepstack_visual_indexes", None)
    if indexes is None and isinstance(vision_config, dict):
        indexes = vision_config.get("deepstack_visual_indexes")
    deepstack_count = len(indexes or [])
    return ["main"] + [f"deepstack_{index}" for index in range(deepstack_count)]


def _prepare_vllm_feature_plugin_env(output_dir: Path) -> None:
    plugin_root = output_dir / ".vllm_feature_plugin"
    dist_info = plugin_root / f"{VLLM_FEATURE_PLUGIN_NAME}-0.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {VLLM_FEATURE_PLUGIN_NAME}\nVersion: 0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[vllm.general_plugins]\n"
        f"{VLLM_FEATURE_PLUGIN_NAME} = eval.easyr1_eval.vllm_feature_models:register_vllm_feature_models\n",
        encoding="utf-8",
    )
    _prepend_env_path("PYTHONPATH", str(plugin_root))
    _prepend_env_path("PYTHONPATH", str(PROJECT_ROOT))
    _prepend_sys_path(str(plugin_root))
    _prepend_sys_path(str(PROJECT_ROOT))
    allowed_plugins = os.environ.get("VLLM_PLUGINS")
    if allowed_plugins is not None:
        items = [item for item in allowed_plugins.split(",") if item]
        if VLLM_FEATURE_PLUGIN_NAME not in items:
            items.append(VLLM_FEATURE_PLUGIN_NAME)
        os.environ["VLLM_PLUGINS"] = ",".join(items)


def _write_vllm_feature_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _prepend_env_path(name: str, value: str) -> None:
    parts = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value in parts:
        parts.remove(value)
    os.environ[name] = os.pathsep.join([value, *parts])


def _prepend_sys_path(value: str) -> None:
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


class TransformersBackend:
    def __init__(
        self,
        model: str,
        *,
        trust_remote_code: bool = True,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        dtype: str = "bfloat16",
        perturbation: PerturbationConfig | None = None,
        perturbation_seed: int = 0,
        output_dir: Path | str | None = None,
        save_perturbation_samples: int = 0,
        chat_template: str | None = None,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        from verl.utils.tokenizer import get_processor, get_tokenizer

        model_config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.processor = get_processor(
            model, override_chat_template=chat_template, trust_remote_code=trust_remote_code
        )
        self.tokenizer = get_tokenizer(model, trust_remote_code=trust_remote_code)
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else torch.float16
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.perturbation = perturbation or PerturbationConfig()
        self.perturbation_seed = perturbation_seed
        self.output_dir = Path(output_dir) if output_dir is not None else Path("eval/results/latest")
        self.save_perturbation_samples = save_perturbation_samples
        self._saved_perturbation_samples = 0
        self._write_model_class_gate(model_config)
        self.feature_perturbation = install_feature_perturbation_hook(
            self.model,
            self.perturbation,
            self.perturbation_seed,
        )
        self.model.eval()

    def _write_model_class_gate(self, model_config) -> None:
        class_name = self.model.__class__.__name__
        model_type = str(getattr(model_config, "model_type", ""))
        architectures = list(getattr(model_config, "architectures", None) or [])
        if model_type == "qwen3_5" and "Qwen3_5" not in class_name:
            raise RuntimeError(
                f"Checkpoint model_type=qwen3_5 loaded unexpected class {class_name}; stopping for manual review."
            )
        if model_type in {"qwen3_vl", "qwen3_vl_moe"} and "Qwen3VL" not in class_name:
            raise RuntimeError(
                f"Checkpoint model_type={model_type} loaded unexpected class {class_name}; stopping for manual review."
            )
        payload = {
            "model_type": model_type,
            "architectures": architectures,
            "loaded_class": f"{self.model.__class__.__module__}.{class_name}",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "model_class_gate.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def generate(self, samples: list[EvalSample], config: GenerationConfig) -> list[list[GenerationOutput]]:
        import torch

        outputs: list[list[GenerationOutput]] = []
        with torch.no_grad():
            for sample in samples:
                sample_outputs = []
                images, base_diagnostics = self._prepare_images(sample)
                sizes = image_size_records(sample, images, self.processor) if images else []
                prompt = self._prompt_text(sample)
                if self.processor is not None and images:
                    inputs = self.processor(
                        images=images,
                        text=[prompt],
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                else:
                    inputs = self.tokenizer([prompt], return_tensors="pt")
                if self.perturbation.is_feature and "pixel_values_videos" in inputs:
                    raise ValueError(
                        "Feature-space perturbations currently support image inputs only; video inputs are unsupported."
                    )
                inputs = {
                    key: value.to(self.model.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                for sample_idx in range(config.num_samples):
                    diagnostics = [dict(item) for item in base_diagnostics]
                    if self.feature_perturbation is not None:
                        self.feature_perturbation.begin_sample(sample.sample_id)
                    try:
                        generated = self.model.generate(
                            **inputs,
                            do_sample=config.temperature > 0,
                            temperature=max(config.temperature, 1e-6),
                            top_p=config.top_p,
                            max_new_tokens=config.max_new_tokens,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    finally:
                        if self.feature_perturbation is not None:
                            diagnostics = diagnostics + self.feature_perturbation.end_sample()
                    prompt_len = inputs["input_ids"].shape[-1]
                    generated_ids = generated[0][prompt_len:]
                    token_count = int(generated_ids.shape[-1])
                    stopped_on_eos = _contains_eos(generated_ids.tolist(), self.tokenizer.eos_token_id)
                    truncated = token_count >= config.max_new_tokens and not stopped_on_eos
                    sample_outputs.append(
                        GenerationOutput(
                            text=self.tokenizer.decode(generated_ids, skip_special_tokens=True),
                            finish_reason="length" if truncated else "stop",
                            token_count=token_count,
                            truncated=truncated,
                            diagnostics={"perturbation": diagnostics, "image_sizes": sizes},
                        )
                    )
                outputs.append(sample_outputs)
        return outputs

    def _prepare_images(self, sample: EvalSample) -> tuple[list, list[dict]]:
        images = []
        diagnostics = []
        for image_idx, image in enumerate(sample.images):
            processed = process_eval_image(image, self.min_pixels, self.max_pixels)
            processed, diag = apply_pixel_perturbation(
                processed,
                config=self.perturbation,
                global_seed=self.perturbation_seed,
                sample_id=sample.sample_id,
                image_idx=image_idx,
            )
            if diag:
                self._saved_perturbation_samples = write_sample_image(
                    processed,
                    output_dir=self.output_dir,
                    config=self.perturbation,
                    sample_id=sample.sample_id,
                    image_idx=image_idx,
                    max_samples=self.save_perturbation_samples,
                    current_count=self._saved_perturbation_samples,
                )
                diagnostics.append(diag)
            images.append(processed)
        return images, diagnostics

    def _prompt_text(self, sample: EvalSample) -> str:
        if not sample.messages:
            return sample.prompt
        template_owner = self.processor if self.processor is not None else self.tokenizer
        return render_chat_prompt(sample, template_owner)


def _contains_eos(token_ids: list[int], eos_token_id) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple, set)):
        return any(token_id in eos_token_id for token_id in token_ids)
    return eos_token_id in token_ids


def build_backend(name: str, model: str, **kwargs) -> Backend:
    interaction_mode = kwargs.pop("interaction_mode", "one_shot")
    agent_profile = kwargs.pop("agent_profile", "deepeyes")
    agent_config = kwargs.pop("agent_config", None)
    agent_max_images_per_prompt = kwargs.pop(
        "agent_max_images_per_prompt",
        None,
    )
    agent_max_batch_images = kwargs.pop("agent_max_batch_images", None)
    agent_tool_image_mode = kwargs.pop("agent_tool_image_mode", "original")
    agent_bbox_format = kwargs.pop("agent_bbox_format", "norm1000")
    if interaction_mode == "agentic":
        if name != "vllm":
            raise ValueError("agentic interaction currently requires the vllm backend")
        if agent_config is None or agent_max_images_per_prompt is None or agent_max_batch_images is None:
            raise ValueError(
                "agentic interaction requires agent_config and agent_max_images_per_prompt and agent_max_batch_images"
            )
        from .agentic import AgenticVLLMBackend

        return AgenticVLLMBackend(
            model,
            agent_profile=agent_profile,
            agent_config=agent_config,
            agent_max_images_per_prompt=agent_max_images_per_prompt,
            agent_max_batch_images=agent_max_batch_images,
            agent_tool_image_mode=agent_tool_image_mode,
            agent_bbox_format=agent_bbox_format,
            **kwargs,
        )
    if interaction_mode != "one_shot":
        raise ValueError(f"unknown interaction mode: {interaction_mode}")
    if name == "dummy":
        return DummyBackend()
    if name == "vllm":
        return VLLMBackend(model, **kwargs)
    if name == "transformers":
        allowed = {
            "trust_remote_code",
            "min_pixels",
            "max_pixels",
            "dtype",
            "perturbation",
            "perturbation_seed",
            "output_dir",
            "save_perturbation_samples",
            "chat_template",
        }
        return TransformersBackend(model, **{key: value for key, value in kwargs.items() if key in allowed})
    raise ValueError(f"unknown backend: {name}")
