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
"""vLLMRollout.generate_from_raw_prompts sends vLLM the requests generate_sequences would send, without the padded
input tensors, and processes each distinct image once per call (grounding consistency self-detection)."""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from vllm import SamplingParams

from verl.protocol import DataProto
from verl.workers.rollout import vllm_rollout_spmd as rollout_module
from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout


PAD, EOS, MAX_TOKENS = 0, 99, 6


def _png(value):
    buffer = io.BytesIO()
    Image.fromarray(np.full((40, 30, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue()}


class _Engine:
    """Answers every request with its first prompt id, then EOS; records what it was given."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
        self.calls.append((prompts, sampling_params))
        return [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[request["prompt_token_ids"][0], EOS])])
            for request in prompts
        ]


def _rollout():
    rollout = vLLMRollout.__new__(vLLMRollout)
    rollout.config = SimpleNamespace(interaction_mode="one_shot", response_length=MAX_TOKENS)
    rollout.pad_token_id = PAD
    rollout.lora_kwargs = {}
    rollout.use_tqdm = False
    rollout.return_video_metadata = False
    rollout.sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=1.0, top_p=0.99, detokenize=False)
    rollout.inference_engine = _Engine()
    return rollout


META = {"n": 1, "temperature": 0.0, "top_p": 1.0, "min_pixels": 64, "max_pixels": 4096, "video_fps": 2.0}


def _requests(engine_call):
    prompts, params = engine_call
    images = [
        [np.asarray(image) for image in (request.get("multi_modal_data") or {}).get("image", [])]
        for request in prompts
    ]
    return [request["prompt_token_ids"] for request in prompts], images, (params.n, params.temperature, params.top_p)


def test_requests_equal_generate_sequences_and_images_are_processed_once(monkeypatch):
    calls = []
    real_process_image = rollout_module.process_image

    def counting_process_image(image, min_pixels, max_pixels):
        calls.append(image)
        return real_process_image(image, min_pixels, max_pixels)

    monkeypatch.setattr(rollout_module, "process_image", counting_process_image)
    shared, other = _png(10), _png(200)
    raw = [[5, 6, 7], [8, 9], [5, 6, 7]]
    images = [{"images": [shared]}, {"images": [other]}, {"images": [shared]}]

    rollout = _rollout()
    full = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(3, 4, dtype=torch.long),
            "attention_mask": torch.ones(3, 4, dtype=torch.long),
            "position_ids": torch.zeros(3, 4, dtype=torch.long),
        },
        non_tensors={
            "raw_prompt_ids": np.array(raw + [None], dtype=object)[:-1],
            "multi_modal_data": np.array(images, dtype=object),
        },
        meta_info={**META, "eos_token_id": EOS},
    )
    rollout.generate_sequences(full)
    assert len(calls) == 3  # once per request

    calls.clear()
    light = DataProto.from_dict(
        tensors={"request_index": torch.arange(3)},
        non_tensors={
            "raw_prompt_ids": np.array(raw + [None], dtype=object)[:-1],
            "multi_modal_data": np.array(images, dtype=object),
        },
        meta_info={**META, "eos_token_id": EOS},
    )
    output = rollout.generate_from_raw_prompts(light)
    assert len(calls) == 2  # once per distinct image

    old_ids, old_images, old_params = _requests(rollout.inference_engine.calls[0])
    new_ids, new_images, new_params = _requests(rollout.inference_engine.calls[1])
    assert new_ids == old_ids and new_params == old_params == (1, 0.0, 1.0)
    assert all(np.array_equal(a, b) for row_a, row_b in zip(old_images, new_images) for a, b in zip(row_a, row_b))
    assert rollout.sampling_params.temperature == 1.0  # the override is temporary

    assert output.batch["responses"].tolist() == [[5, EOS] + [PAD] * 4, [8, EOS] + [PAD] * 4, [5, EOS] + [PAD] * 4]
    assert output.batch["response_lengths"].tolist() == [2, 2, 2]


def test_one_completion_per_prompt():
    rollout = _rollout()
    prompts = DataProto.from_dict(
        tensors={"request_index": torch.arange(1)},
        non_tensors={"raw_prompt_ids": np.array([[1], None], dtype=object)[:-1], "multi_modal_data": np.array([None])},
        meta_info={**META, "n": 2},
    )
    with pytest.raises(ValueError, match="n=1"):
        rollout.generate_from_raw_prompts(prompts)
