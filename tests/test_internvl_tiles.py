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
"""InternVL tiles per image (worker.actor.model.max_dynamic_patch) and prompts filtered by length."""

from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from verl.models.transformers.internvl import InternVLProcessorAdapter
from verl.trainer.config import PPOConfig
from verl.utils import tokenizer as tokenizer_module
from verl.utils.dataset import RLHFDataset
from verl.utils.tokenizer import get_processor, get_tokenizer


def _internvl_adapter() -> InternVLProcessorAdapter:
    adapter = object.__new__(InternVLProcessorAdapter)
    adapter.image_size, adapter.min_dynamic_patch, adapter.max_dynamic_patch, adapter.use_thumbnail = 448, 1, 12, True
    return adapter


def _patch_loading(monkeypatch, processor, model_type):
    monkeypatch.setattr(tokenizer_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(
        tokenizer_module.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(model_type=model_type)
    )


def test_max_dynamic_patch_caps_the_tiles(monkeypatch):
    _patch_loading(monkeypatch, _internvl_adapter(), "internvl_chat")
    wide = Image.new("RGB", (2400, 900))
    assert len(get_processor("internvl", plain_think_tokens=False).preprocess_image(wide)) > 3  # model config: 12

    _patch_loading(monkeypatch, _internvl_adapter(), "internvl_chat")
    processor = get_processor("internvl", plain_think_tokens=False, max_dynamic_patch=2)
    assert len(processor.preprocess_image(wide)) == 3 == processor.get_num_image_patches(wide)  # 2 tiles + thumbnail
    assert len(processor.preprocess_image(Image.new("RGB", (448, 448)))) == 1


def test_max_dynamic_patch_is_internvl_only(monkeypatch):
    class Qwen2_5_VLProcessor:
        pass

    _patch_loading(monkeypatch, Qwen2_5_VLProcessor(), "qwen2_5_vl")
    with pytest.raises(ValueError, match="InternVL"):
        get_processor("qwen", plain_think_tokens=False, max_dynamic_patch=2)


def test_rollout_gets_the_tile_limit():
    config = PPOConfig()
    config.trainer.n_gpus_per_node = 1
    config.data.rollout_batch_size = config.worker.actor.global_batch_size = 2
    config.worker.actor.micro_batch_size_per_device_for_update = 1
    config.worker.actor.micro_batch_size_per_device_for_experience = 1
    config.worker.actor.model.max_dynamic_patch = 2
    config.post_init()
    assert config.worker.rollout.max_dynamic_patch == 2


def test_prompts_filtered_to_nothing_raise(tmp_path):
    path = tmp_path / "train.parquet"
    pd.DataFrame({"problem": ["a long enough question " * 4] * 3, "answer": ["1"] * 3}).to_parquet(path)
    tokenizer = get_tokenizer("Qwen/Qwen2.5-VL-7B-Instruct")
    kwargs = dict(data_path=str(path), tokenizer=tokenizer, processor=None, prompt_key="problem")
    assert len(RLHFDataset(max_prompt_length=1024, filter_overlong_prompts_workers=1, **kwargs)) == 3
    with pytest.raises(ValueError, match="all 3 prompts are longer than data.max_prompt_length=8"):
        RLHFDataset(max_prompt_length=8, filter_overlong_prompts_workers=1, **kwargs)
