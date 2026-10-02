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

import re
from typing import Any, Optional

import torch


QWEN_THINK_PREFILL_SUFFIX = "<|im_start|>assistant\n<think>"
_FALLBACK_VISUAL_PAD_TOKENS = ("<|image_pad|>", "<|video_pad|>", "<IMG_CONTEXT>")
_VISUAL_TOKEN_MARKERS = ("image", "img", "video", "vision", "visual", "im_patch")
_VISUAL_REPEAT_MARKERS = ("pad", "context", "patch")


def _special_token_to_str(token: Any) -> Optional[str]:
    if isinstance(token, str):
        return token

    content = getattr(token, "content", None)
    if isinstance(content, str):
        return content

    return None


def _iter_special_token_strings(value: Any):
    token = _special_token_to_str(value)
    if token is not None:
        yield token
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_special_token_strings(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_special_token_strings(item)


def _is_visual_repeat_token(token: str) -> bool:
    lowered = token.lower()
    return any(marker in lowered for marker in _VISUAL_TOKEN_MARKERS) and any(
        marker in lowered for marker in _VISUAL_REPEAT_MARKERS
    )


def _get_visual_pad_tokens(tokenizer: Optional[Any]) -> tuple[str, ...]:
    tokens = set(_FALLBACK_VISUAL_PAD_TOKENS)
    if tokenizer is not None:
        for attr in ("all_special_tokens", "additional_special_tokens", "special_tokens_map_extended"):
            for token in _iter_special_token_strings(getattr(tokenizer, attr, None)):
                if _is_visual_repeat_token(token):
                    tokens.add(token)

    return tuple(sorted(tokens, key=len, reverse=True))


def compact_vision_pad_runs(text: str, tokenizer: Optional[Any] = None) -> str:
    for token in _get_visual_pad_tokens(tokenizer):
        text = re.sub(f"(?:{re.escape(token)}){{2,}}", token, text)
    return text


def prompt_prefills_think(prompt: Optional[str]) -> bool:
    if prompt is None:
        return False
    return prompt.rstrip().endswith(QWEN_THINK_PREFILL_SUFFIX)


def canonicalize_response_for_prefilled_think(prompt: Optional[str], response: str) -> str:
    """Restore the opening think tag when the chat template already put it in the prompt.

    Qwen3.5's thinking chat template ends the prompt with ``<think>\n``. Rollout responses
    only contain newly generated tokens, so they can start with reasoning text and later emit
    ``</think>``. For reward/log display, reconstruct the semantic assistant message only
    when that exact template prefill is present.
    """

    if not prompt_prefills_think(prompt):
        return response

    stripped_response = response.lstrip()
    if stripped_response.startswith("<think>"):
        return response

    closing_idx = response.find("</think>")
    if closing_idx < 0:
        return response

    opening_idx = response.find("<think>")
    if opening_idx >= 0 and opening_idx < closing_idx:
        return response

    return "<think>\n" + response


def decode_prompt_from_batch(tokenizer: Any, batch: Any, idx: int) -> Optional[str]:
    if "prompts" not in batch.keys():
        return None

    prompt_ids = batch["prompts"][idx]
    attention_mask = batch["attention_mask"][idx] if "attention_mask" in batch.keys() else None
    if attention_mask is not None and attention_mask.ndim == 1 and attention_mask.numel() >= prompt_ids.numel():
        prompt_mask = attention_mask[: prompt_ids.numel()].to(torch.bool)
        prompt_ids = prompt_ids[prompt_mask]
    else:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            prompt_ids = prompt_ids[prompt_ids != pad_token_id]

    return tokenizer.decode(prompt_ids, skip_special_tokens=False)
