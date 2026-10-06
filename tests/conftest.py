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
"""Test collection rules shared by every test module.

The CPU-only CI machine has no vLLM. These modules import it at import time (DeepEyes agent loop,
vLLM rollout and agentic evaluation), so they are only collected where vLLM is installed.
"""

import importlib.util


collect_ignore = []
if importlib.util.find_spec("vllm") is None:
    collect_ignore += [
        "test_deepeyes_agent.py",
        "test_deepeyes_training.py",
        "test_eval_agentic_pixel.py",
        "test_eval_batching_and_images.py",
        "test_rollout_lora_requests.py",
        "test_rollout_raw_prompts.py",
    ]
