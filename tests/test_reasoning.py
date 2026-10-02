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
from verl.utils.reasoning import (
    canonicalize_response_for_prefilled_think,
    compact_vision_pad_runs,
    prompt_prefills_think,
)


QWEN35_PROMPT = "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_prompt_prefills_think_requires_assistant_suffix():
    assert prompt_prefills_think(QWEN35_PROMPT)
    assert not prompt_prefills_think("user asked for <think>\n")
    assert not prompt_prefills_think("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_canonicalize_response_restores_template_prefilled_opening_tag():
    response = "reasoning</think>\n\\boxed{A}"

    assert canonicalize_response_for_prefilled_think(QWEN35_PROMPT, response) == (
        "<think>\nreasoning</think>\n\\boxed{A}"
    )


def test_canonicalize_response_does_not_grant_format_without_fixed_prompt_prefill():
    response = "reasoning</think>\n\\boxed{A}"

    assert canonicalize_response_for_prefilled_think("user asked for <think>\n", response) == response


def test_canonicalize_response_does_not_duplicate_existing_opening_tag():
    response = "<think>\nreasoning</think>\n\\boxed{A}"

    assert canonicalize_response_for_prefilled_think(QWEN35_PROMPT, response) == response


def test_canonicalize_response_requires_closing_tag():
    response = "reasoning only \\boxed{A}"

    assert canonicalize_response_for_prefilled_think(QWEN35_PROMPT, response) == response


def test_compact_vision_pad_runs_uses_tokenizer_special_tokens():
    class Tokenizer:
        all_special_tokens = ["<MY_IMAGE_CONTEXT>"]

    text = "<image><MY_IMAGE_CONTEXT><MY_IMAGE_CONTEXT><MY_IMAGE_CONTEXT></image>"

    assert compact_vision_pad_runs(text, Tokenizer()) == "<image><MY_IMAGE_CONTEXT></image>"
