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
import json

import pytest
import torch

import verl.utils.logger.gen_logger as gen_logger
from verl.protocol import DataProto
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.utils.logger.gen_logger import (
    FileGenerationLogger,
    GenerationSample,
    WandbGenerationLogger,
    prepare_generation_samples,
)


def _config(tmp_path, *, log_images=False):
    return {
        "trainer": {
            "save_checkpoint_path": str(tmp_path),
            "generations_log_file": "completions.jsonl",
            "log_images": log_images,
            "generation_log_image_max_size": 64,
            "generation_log_caption_max_chars": 32,
            "train_generations_to_log": 0,
        }
    }


class _SpecialTokenTokenizer:
    pad_token_id = 0

    _tokens = {
        0: "<|pad|>",
        1: "<|im_start|>",
        2: "user",
        3: "\n",
        4: "Question",
        5: "<|im_end|>",
        6: "assistant",
        7: "<think>",
        8: "\n\n",
        9: "reasoning",
        10: "</think>",
        11: "<|vision_start|>",
        12: "<|image_pad|>",
        13: "<|vision_end|>",
        14: "<|video_pad|>",
        15: "<img>",
        16: "<IMG_CONTEXT>",
        17: "</img>",
    }
    all_special_tokens = list(_tokens.values())

    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if skip_special_tokens:
            ids = [idx for idx in ids if idx not in {0, 1, 5}]
        return "".join(self._tokens[int(idx)] for idx in ids)


def test_file_generation_logger_writes_jsonl(tmp_path):
    logger = FileGenerationLogger(_config(tmp_path))
    sample = GenerationSample(
        step=3,
        split="train",
        uid="sample-1",
        prompt="What is shown?",
        completion="A square.",
        ground_truth="square",
        score=1.0,
        reward_details={"format": 0.5, "overall": 1.0},
        advantages=0.25,
    )

    logger.log([sample], step=3, split="train")

    rows = [json.loads(line) for line in (tmp_path / "completions.jsonl").read_text().splitlines()]
    assert rows == [
        {
            "step": 3,
            "split": "train",
            "uid": "sample-1",
            "prompt": "What is shown?",
            "completion": "A square.",
            "ground_truth": "square",
            "score": 1.0,
            "advantages": 0.25,
            "image_preview_path": None,
            "image_count": 0,
            "video_count": 0,
            "video_paths": [],
            "reward/format": 0.5,
            "reward/overall": 1.0,
        }
    ]


def test_prepare_generation_samples_creates_image_grid(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    sample = GenerationSample(
        step=7,
        split="val",
        uid="image sample",
        prompt="prompt",
        completion="completion",
        images=[
            Image.new("RGB", (80, 40), color=(255, 0, 0)),
            Image.new("RGB", (40, 80), color=(0, 255, 0)),
        ],
    )

    prepared = prepare_generation_samples([sample], _config(tmp_path, log_images=True), step=7, split="val")

    preview_path = prepared[0].image_preview_path
    assert preview_path is not None
    assert prepared[0].image_count == 2

    preview = Image.open(preview_path)
    assert preview.mode == "RGB"
    assert max(preview.size) <= 64
    assert "images" not in prepared[0].to_record()


def test_wandb_generation_logger_uses_one_row_per_sample(monkeypatch, tmp_path):
    class FakeImage:
        def __init__(self, path, caption=None):
            self.path = path
            self.caption = caption

    class FakeTable:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    class FakeWandb:
        Image = FakeImage
        Table = FakeTable
        logs = []

        @classmethod
        def log(cls, data, step=None):
            cls.logs.append((data, step))

    monkeypatch.setattr(gen_logger, "wandb", FakeWandb, raising=False)
    logger = WandbGenerationLogger(_config(tmp_path))
    sample = GenerationSample(
        step=9,
        split="train",
        uid="uid-1",
        prompt="prompt",
        completion="completion",
        score=0.75,
        reward_details={"overall": 0.75},
    )

    logger.log([sample], step=9, split="train")

    data, step = FakeWandb.logs[-1]
    table = data["train/generations"]
    assert step == 9
    assert table.columns.count("prompt") == 1
    assert "reward/overall" in table.columns
    assert len(table.data) == 1
    assert table.data[0][table.columns.index("prompt")] == "prompt"
    assert table.data[0][table.columns.index("score")] == 0.75


def test_swanlab_generation_logger_logs_table_and_images(monkeypatch, tmp_path):
    class FakeImage:
        def __init__(self, path, caption=None):
            self.path = path
            self.caption = caption

    class FakeTable:
        def add(self, headers, rows):
            return {"headers": headers, "rows": rows}

    class FakeEcharts:
        Table = FakeTable

    class FakeSwanlab:
        Image = FakeImage
        echarts = FakeEcharts
        logs = []

        @classmethod
        def log(cls, data, step=None):
            cls.logs.append((data, step))

    monkeypatch.setattr(gen_logger, "swanlab", FakeSwanlab, raising=False)
    logger = gen_logger.SwanlabGenerationLogger(_config(tmp_path, log_images=True))
    sample = GenerationSample(
        step=4,
        split="val",
        uid="uid-2",
        prompt="prompt",
        completion="completion",
        image_preview_path="/tmp/preview.png",
    )

    logger.log([sample], step=4, split="val")

    data, step = FakeSwanlab.logs[-1]
    assert step == 4
    assert "val/generations" in data
    assert "val/generation_images" in data
    assert data["val/generation_images"][0].path == "/tmp/preview.png"


def test_trainer_generation_samples_keep_special_tokens_without_padding():
    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = _SpecialTokenTokenizer()
    trainer.global_step = 12
    prompt_ids = [
        0,
        1,
        2,
        3,
        11,
        12,
        12,
        12,
        13,
        3,
        11,
        14,
        14,
        13,
        3,
        15,
        16,
        16,
        16,
        17,
        3,
        4,
        5,
        3,
        1,
        6,
        3,
        7,
    ]
    response_ids = [9, 10, 5, 0, 0]
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([prompt_ids], dtype=torch.long),
            "responses": torch.tensor([response_ids], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[0] + [1] * (len(prompt_ids) - 1) + [1, 1, 1, 0, 0]],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long),
        },
        non_tensors={"ground_truth": ["answer"], "uid": ["uid-1"]},
    )

    samples = trainer._build_generation_samples(batch, split="val")

    assert samples[0].prompt == (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "<|vision_start|><|video_pad|><|vision_end|>\n"
        "<img><IMG_CONTEXT></img>\n"
        "Question<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )
    assert samples[0].completion == "<think>\nreasoning</think><|im_end|>"
    assert "<|pad|>" not in samples[0].prompt
    assert "<|pad|>" not in samples[0].completion
    assert "<|image_pad|><|image_pad|>" not in samples[0].prompt
    assert "<|video_pad|><|video_pad|>" not in samples[0].prompt
    assert "<IMG_CONTEXT><IMG_CONTEXT>" not in samples[0].prompt
