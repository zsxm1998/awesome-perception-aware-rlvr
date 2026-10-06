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
"""With LoRA, every vLLM request of the policy (rollout, claim probes) goes to the current adapter: the adapter is
loaded next to the base weights, not merged into them, so a request without it reads the untrained base model."""

import json
from types import SimpleNamespace

import numpy as np
import torch
from vllm import SamplingParams

from verl.protocol import DataProto
from verl.workers.rollout.vllm_rollout_spmd import vLLMRollout


ADAPTER = 17
WORDS = ["yes", "Yes", "True", "true", "1", "0", "no", "No", "false", "False"]


class _Tokenizer:
    word_ids = {word: 1000 + idx for idx, word in enumerate(WORDS)}

    def encode(self, text, add_special_tokens=False):
        return [self.word_ids[text]] if text in self.word_ids else [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return "".join(chr(i) for i in ids if i < 1000)


class _Engine:
    def __init__(self):
        self.lora_requests = []
        self.llm_engine = SimpleNamespace(list_loras=lambda: {ADAPTER})

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
        self.lora_requests.append(lora_request)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1001])]) for _ in prompts]


def _rollout():
    rollout = vLLMRollout.__new__(vLLMRollout)
    rollout.config = SimpleNamespace(
        interaction_mode="one_shot", prompt_length=64, response_length=64, max_model_len=256
    )
    rollout.tokenizer = _Tokenizer()
    rollout.pad_token_id = 0
    rollout.lora_kwargs = {"enable_lora": True}
    rollout.use_tqdm = False
    rollout.return_video_metadata = False
    rollout.sampling_params = SamplingParams(max_tokens=4, detokenize=False)
    rollout.inference_engine = _Engine()
    return rollout


def test_claim_probes_use_the_current_adapter():
    rollout = _rollout()
    response = [ord(c) for c in "a, b, c, d. \\boxed{1}"]
    claims = json.dumps([{"claim": f"c{i}", "correct": i % 2 == 0} for i in range(4)])
    probes = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([response]),
            "response_mask": torch.ones(1, len(response), dtype=torch.long),
            "attention_mask": torch.ones(1, len(response) + 3, dtype=torch.long),
        },
        non_tensors={
            "raw_prompt_ids": np.array([[7, 8, 9], None], dtype=object)[:-1],
            "visual_claims": np.array([claims], dtype=object),
            "claim_probe_seed": np.array([1], dtype=np.uint64),
            "multi_modal_data": np.array([None], dtype=object),
        },
        meta_info={
            "claim_probe_count": 4,
            "claim_probe_question": "\n{claim}? (Yes/No): ",
            "claim_probe_yes_tokens": WORDS[:5],
            "claim_probe_no_tokens": WORDS[5:],
            "min_pixels": None,
            "max_pixels": None,
        },
    )
    rollout.answer_claim_probes(probes)
    (requests,) = rollout.inference_engine.lora_requests
    assert requests is not None and len(requests) == 4
    assert {request.lora_int_id for request in requests} == {ADAPTER}
