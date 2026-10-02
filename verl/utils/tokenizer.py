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

from typing import Optional

from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.internvl import InternVLProcessorAdapter
from ..models.transformers.qwen3_5 import register_qwen3_5


def get_tokenizer(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> PreTrainedTokenizer:
    """Create a huggingface pretrained tokenizer."""
    register_qwen3_5()
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
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


def get_processor(model_path: str, override_chat_template: Optional[str] = None, **kwargs) -> Optional[ProcessorMixin]:
    """Create a huggingface pretrained processor."""
    register_qwen3_5()
    config = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
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

    if processor is not None and override_chat_template is not None:
        with open(override_chat_template) as f:
            processor.chat_template = f.read()

        print(f"Using chat template {override_chat_template}")

    return processor
