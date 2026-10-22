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
"""Reward metrics stay with their samples when the samples of a batch report different keys (an environment's
components that apply to some tasks only)."""

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.metrics import extend_reward_metrics, reduce_metrics, reduce_reward_metrics
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.workers.reward.config import RewardConfig
from verl.workers.reward.function import AutoRewardManager, collect_reward_metrics


# A batch reward function whose samples report different metrics, as an environment's components that apply to
# some tasks only.
_REWARD_FUNCTION = """
REWARD_NAME = "partial"
REWARD_TYPE = "batch"


def compute_score(reward_inputs):
    scores = []
    for reward_input in reward_inputs:
        if reward_input["ground_truth"] == "density":
            scores.append({"overall": 1.0, "density": 0.8})
        else:
            scores.append({"overall": 0.0, "region": 0.25, "orders": 3.0})
    return scores
"""


class _Tokenizer:
    def decode(self, *args, **kwargs):
        return "<answer>x</answer>"


def test_two_samples_with_different_keys_keep_their_own_reward_details(tmp_path):
    path = tmp_path / "partial.py"
    path.write_text(_REWARD_FUNCTION)
    config = RewardConfig(reward_function=f"{path}:compute_score")
    config.post_init()
    manager = AutoRewardManager(config, _Tokenizer())
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[5, 6], [7, 8]]),
            "responses": torch.tensor([[1, 2], [3, 4]]),
            "response_mask": torch.tensor([[1, 1], [1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
        },
        non_tensors={"ground_truth": np.array(["density", "region"], dtype=object)},
    )

    _, metrics = manager.compute_reward(batch)
    details = RayPPOTrainer._build_reward_details(metrics, len(batch))
    logged = RayPPOTrainer._reward_metric_values(metrics)

    assert metrics["density"] == [0.8, None] and metrics["region"] == [None, 0.25]
    assert details == [{"overall": 1.0, "density": 0.8}, {"overall": 0.0, "region": 0.25, "orders": 3.0}]
    assert logged["reward/overall"] == 0.5
    assert (logged["reward/density"], logged["reward_count/density"]) == (0.8, 1)
    assert (logged["reward/region"], logged["reward_count/region"]) == (0.25, 1)
    assert "reward_count/overall" not in logged  # every sample reported it


def test_uniform_keys_reduce_as_before():
    scores = [{"overall": 1.0, "accuracy": 1.0}, {"overall": 0.0, "accuracy": 0.0}, {"overall": 0.5, "accuracy": 1.0}]
    metrics = collect_reward_metrics(scores)

    means, counts = reduce_reward_metrics(metrics)

    assert dict(metrics) == {"overall": [1.0, 0.0, 0.5], "accuracy": [1.0, 0.0, 1.0]}
    assert means == reduce_metrics(metrics) and counts == {}
    assert RayPPOTrainer._build_reward_details(metrics, 3) == [dict(score) for score in scores]


def test_metrics_used_per_sample_must_come_from_every_sample():
    metrics = collect_reward_metrics([{"overall": 1.0, "accuracy": 1.0}, {"overall": 0.0}])

    assert RayPPOTrainer._per_sample_reward_values(metrics, "overall", "online filtering") == [1.0, 0.0]
    assert RayPPOTrainer._per_sample_reward_values(metrics, "missing", "gating", required=False) is None
    with pytest.raises(ValueError, match="'accuracy' from every sample"):
        RayPPOTrainer._per_sample_reward_values(metrics, "accuracy", "online filtering (filter_key)")


def test_validation_batches_with_different_keys_stay_aligned():
    accumulated: dict = {}
    count = extend_reward_metrics(accumulated, collect_reward_metrics([{"overall": 1.0}, {"overall": 0.0}]), 0)
    count = extend_reward_metrics(accumulated, collect_reward_metrics([{"overall": 1.0, "density": 0.5}]), count)

    assert count == 3
    assert accumulated == {"overall": [1.0, 0.0, 1.0], "density": [None, None, 0.5]}
    assert reduce_reward_metrics(accumulated) == ({"overall": pytest.approx(2 / 3), "density": 0.5}, {"density": 1})
