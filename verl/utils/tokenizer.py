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
"""Utils for tokenization."""

from typing import Any, Optional

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.internvl import InternVLProcessorAdapter
from ..models.transformers.qwen3_5 import register_qwen3_5
from .plain_think import THINK_TOKENS, plain_think_tokenizer_path


def get_tokenizer(
    model_path: str, override_chat_template: Optional[str] = None, plain_think_tokens: Any = "auto", **kwargs
) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer (see verl/utils/plain_think.py for ``plain_think_tokens``)."""
    register_qwen3_5()
    source = plain_think_tokenizer_path(model_path, plain_think_tokens, **kwargs) or model_path
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if override_chat_template is not None:
        with open(override_chat_template) as f:
            tokenizer.chat_template = f.read()

        print(f"Using chat template {override_chat_template}")

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        # the EOS token in gemma2 & gemma3 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(
    model_path: str,
    override_chat_template: Optional[str] = None,
    plain_think_tokens: Any = "auto",
    max_dynamic_patch: Optional[int] = None,
    **kwargs,
) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor (see verl/utils/plain_think.py for ``plain_think_tokens``).

    ``max_dynamic_patch`` caps the tiles per image of an InternVL processor (None: the model config's value).
    """
    register_qwen3_5()
    config = None
    source = plain_think_tokenizer_path(model_path, plain_think_tokens, **kwargs) or model_path
    try:
        processor = AutoProcessor.from_pretrained(source, **kwargs)
    except Exception:
        config = AutoConfig.from_pretrained(model_path, **kwargs)
        if config.model_type != "internvl_chat":
            raise

        processor = InternVLProcessorAdapter.from_pretrained(model_path, **kwargs)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/auto/processing_auto.py#L386
    if processor is not None and "Processor" not in processor.__class__.__name__:
        config = config or AutoConfig.from_pretrained(model_path, **kwargs)
        processor = (
            InternVLProcessorAdapter.from_pretrained(model_path, **kwargs)
            if config.model_type == "internvl_chat"
            else None
        )

    if processor is not None:
        config = config or AutoConfig.from_pretrained(model_path, **kwargs)
        if config.model_type == "qwen3_5":
            processor._easy_r1_model_type = "qwen3_5"
            for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
                if not hasattr(processor, name) and hasattr(config, name):
                    setattr(processor, name, getattr(config, name))

    if max_dynamic_patch is not None:
        if not isinstance(processor, InternVLProcessorAdapter):
            raise ValueError("max_dynamic_patch only applies to InternVL models.")
        if max_dynamic_patch < processor.min_dynamic_patch:
            raise ValueError(f"max_dynamic_patch must be at least min_dynamic_patch ({processor.min_dynamic_patch}).")
        processor.max_dynamic_patch = int(max_dynamic_patch)

    if processor is not None and override_chat_template is not None:
        with open(override_chat_template) as f:
            processor.chat_template = f.read()

        print(f"Using chat template {override_chat_template}")

    return processor


def _model_family(model_type: str) -> str:
    return model_type[: -len("_moe")] if model_type.endswith("_moe") else model_type


def check_teacher_compatibility(student_path: str, teacher_path: str, trust_remote_code: bool = True) -> None:
    """Fail unless a teacher can read the student's inputs and score the student's vocabulary.

    The teacher is fed the student's token ids and pixel values, and the distillation losses compare the two
    distributions id by id, so the token-to-id map, the vocabulary size, the visual special tokens, the multimodal
    rotary layout and the image preprocessing must agree. The architecture (size, depth) may differ.
    """
    register_qwen3_5()
    problems = []
    student_config = AutoConfig.from_pretrained(student_path, trust_remote_code=trust_remote_code)
    teacher_config = AutoConfig.from_pretrained(teacher_path, trust_remote_code=trust_remote_code)
    if _model_family(student_config.model_type) != _model_family(teacher_config.model_type):
        problems.append(f"model_type {student_config.model_type!r} vs {teacher_config.model_type!r}")

    student_text, teacher_text = student_config.get_text_config(), teacher_config.get_text_config()
    if student_text.vocab_size != teacher_text.vocab_size:
        problems.append(f"vocab_size {student_text.vocab_size} vs {teacher_text.vocab_size}")

    for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if getattr(student_config, name, None) != getattr(teacher_config, name, None):
            problems.append(f"{name} {getattr(student_config, name, None)} vs {getattr(teacher_config, name, None)}")

    def _mrope_section(text_config) -> Any:
        rope = getattr(text_config, "rope_scaling", None) or getattr(text_config, "rope_parameters", None) or {}
        return rope.get("mrope_section")

    if _mrope_section(student_text) != _mrope_section(teacher_text):
        problems.append(f"mrope_section {_mrope_section(student_text)} vs {_mrope_section(teacher_text)}")

    student_vision = getattr(student_config, "vision_config", None)
    teacher_vision = getattr(teacher_config, "vision_config", None)
    for name in ("patch_size", "spatial_merge_size", "temporal_patch_size"):
        student_value, teacher_value = getattr(student_vision, name, None), getattr(teacher_vision, name, None)
        if student_value != teacher_value:
            problems.append(f"vision_config.{name} {student_value} vs {teacher_value}")

    student_tokenizer = AutoTokenizer.from_pretrained(student_path, trust_remote_code=trust_remote_code)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=trust_remote_code)
    # <think> / </think> may be removed from either side by the plain-think option (verl/utils/plain_think.py);
    # the teacher never tokenizes text, so only the ids of the other tokens have to agree
    student_vocab, teacher_vocab = (
        {token: index for token, index in tokenizer.get_vocab().items() if token not in THINK_TOKENS}
        for tokenizer in (student_tokenizer, teacher_tokenizer)
    )
    if student_vocab != teacher_vocab:
        differing = sorted(set(student_vocab.items()) ^ set(teacher_vocab.items()), key=lambda item: item[1])
        problems.append(f"tokenizer vocabularies differ ({len(differing)} entries, e.g. {differing[:3]})")
    if len(student_tokenizer) > student_text.vocab_size:
        problems.append(f"the tokenizer has {len(student_tokenizer)} tokens, more than vocab_size")

    if getattr(student_config, "vision_config", None) is not None:
        student_image = AutoProcessor.from_pretrained(student_path, trust_remote_code=trust_remote_code)
        teacher_image = AutoProcessor.from_pretrained(teacher_path, trust_remote_code=trust_remote_code)
        student_image = getattr(student_image, "image_processor", student_image)
        teacher_image = getattr(teacher_image, "image_processor", teacher_image)
        for name in ("patch_size", "merge_size", "temporal_patch_size", "image_mean", "image_std"):
            student_value, teacher_value = getattr(student_image, name, None), getattr(teacher_image, name, None)
            if student_value != teacher_value:
                problems.append(f"image processor {name} {student_value} vs {teacher_value}")

    if problems:
        raise ValueError(
            f"The teacher {teacher_path} cannot be distilled into the student {student_path}: " + "; ".join(problems)
        )
