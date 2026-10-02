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
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.prompting import PromptConfig, apply_prompt_to_sample  # noqa: E402
from easyr1_eval.schemas import EvalSample  # noqa: E402


def test_chat_prompt_config_matches_training_message_shape():
    sample = EvalSample(
        benchmark="pope",
        sample_id="1",
        prompt="<image>\nIs there a cat?",
        target="yes",
        images=["cat.jpg"],
    )
    configured = apply_prompt_to_sample(
        sample,
        PromptConfig(
            format_prompt="{{ content | trim }}\nAnswer in \\boxed{}.",
            system_prompt="You are a grounded VLM.",
            prompt_mode="chat",
        ),
    )

    assert configured.prompt.endswith(r"Answer in \boxed{}.")
    assert configured.messages == [
        {"role": "system", "content": "You are a grounded VLM."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "\nIs there a cat?\nAnswer in \\boxed{}."},
            ],
        },
    ]


def test_raw_prompt_config_prepends_system_prompt():
    sample = EvalSample(benchmark="gqa", sample_id="1", prompt="What is shown?", target="cat")
    configured = apply_prompt_to_sample(
        sample,
        PromptConfig(format_prompt="{{ content }}\nBe brief.", system_prompt="System text.", prompt_mode="raw"),
    )

    assert configured.messages is None
    assert configured.prompt == "System text.\n\nWhat is shown?\nBe brief."


def test_chat_prompt_config_inserts_missing_image_placeholder():
    sample = EvalSample(
        benchmark="pope",
        sample_id="1",
        prompt="Is there a snowboard in the image?",
        target="yes",
        images=["snowboard.jpg"],
    )
    configured = apply_prompt_to_sample(sample, PromptConfig(system_prompt="System text.", prompt_mode="chat"))

    assert configured.messages == [
        {"role": "system", "content": "System text."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Is there a snowboard in the image?"},
            ],
        },
    ]


def test_raw_prompt_config_inserts_missing_image_placeholder():
    sample = EvalSample(
        benchmark="pope",
        sample_id="1",
        prompt="Is there a snowboard in the image?",
        target="yes",
        images=["snowboard.jpg"],
    )
    configured = apply_prompt_to_sample(sample, PromptConfig(system_prompt="System text.", prompt_mode="raw"))

    assert configured.prompt == "System text.\n\n<image>\nIs there a snowboard in the image?"
