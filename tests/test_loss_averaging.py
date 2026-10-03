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
"""Loss averaging across micro-batches and ranks: token mode weights tokens, seq mode weights responses equally."""

import numpy as np
import pytest
import torch
from torch import nn

from verl.protocol import DataProto
from verl.workers.actor import dp_actor as dp_actor_module
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor


LENGTHS = [1, 7, 2, 0, 5, 3, 8, 4]  # the response with no token gets no weight in either mode
RESPONSE_LENGTH = max(LENGTHS)


def _data(rows: list[int]) -> DataProto:
    lengths = torch.tensor([LENGTHS[row] for row in rows])
    response_mask = (torch.arange(RESPONSE_LENGTH)[None, :] < lengths[:, None]).long()
    prompt = torch.ones(len(rows), 2, dtype=torch.long)
    attention_mask = torch.cat([prompt, response_mask], dim=-1)
    input_ids = torch.cat([torch.tensor(rows)[:, None], prompt[:, :1], torch.zeros_like(response_mask)], dim=-1)
    advantages = torch.tensor([float(row + 1) * (-1) ** row for row in rows])[:, None].expand(-1, RESPONSE_LENGTH)
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.cumsum(attention_mask, dim=-1),
            "responses": torch.zeros_like(response_mask),
            "response_mask": response_mask,
            "old_log_probs": torch.zeros(len(rows), RESPONSE_LENGTH),
            "advantages": advantages * response_mask,
        },
        non_tensors={"multi_modal_inputs": np.array([None] * len(rows), dtype=object)},
        meta_info={"temperature": 1.0},
    )


def _rank_grad(monkeypatch, mode: str, rows: list[int], world: list[list[int]], micro: int, dynamic: bool):
    """The gradient one rank accumulates over its micro-batches; ``world`` holds the rows of every rank."""
    config = ActorConfig(
        global_batch_size=len(rows),
        micro_batch_size_per_device_for_update=micro,
        micro_batch_size_per_device_for_experience=micro,
        loss_avg_mode=mode,
        padding_free=False,
        dynamic_batching=dynamic,
        use_torch_compile=False,
    )
    config.global_batch_size_per_device = len(rows)
    weights = nn.Parameter(torch.zeros(len(LENGTHS), RESPONSE_LENGTH))  # log-prob of every (response, token)
    actor = DataParallelPPOActor(config=config, actor_module=nn.Linear(1, 1), actor_optimizer=None)
    actor.rank, actor.world_size = 1, len(world)

    def forward(model_inputs, temperature, **kwargs):
        return weights[model_inputs["input_ids"][:, 0]]

    flat = [LENGTHS[row] for ranks in world for row in ranks]
    total = sum(length > 0 for length in flat) if mode == "seq" else sum(flat)
    monkeypatch.setattr(actor, "_forward_micro_batch", forward)
    monkeypatch.setattr(actor, "_optimizer_step", lambda: torch.tensor(0.0))
    monkeypatch.setattr(dp_actor_module.dist, "all_reduce", lambda tensor, op=None: tensor.fill_(total))
    actor.update_policy(_data(rows))
    return weights.grad


def _reference_grad(mode: str) -> torch.Tensor:
    data = _data(list(range(len(LENGTHS))))
    mask = data.batch["response_mask"].float()
    lengths = mask.sum(-1, keepdim=True)
    if mode == "seq":  # mean over the tokens of each response, then over the responses with a token
        weights = mask / lengths.clamp(min=1) / (lengths > 0).sum()
    else:
        weights = mask / mask.sum()
    return -data.batch["advantages"] * weights  # d(-A * exp(log p - old log p)) / d log p at ratio 1


@pytest.mark.parametrize("mode", ["seq", "token"])
@pytest.mark.parametrize("micro,dynamic", [(8, False), (4, False), (1, False), (2, True)])
def test_one_rank_matches_the_reference_for_any_split(monkeypatch, mode, micro, dynamic):
    rows = list(range(len(LENGTHS)))
    grad = _rank_grad(monkeypatch, mode, rows, [rows], micro, dynamic)
    torch.testing.assert_close(grad, _reference_grad(mode))


@pytest.mark.parametrize("mode", ["seq", "token"])
def test_ranks_with_different_lengths_match_the_reference(monkeypatch, mode):
    """FSDP averages the gradients of the ranks; each rank holds responses of different lengths."""
    world = [[0, 2, 3, 5], [1, 4, 6, 7]]
    grads = [_rank_grad(monkeypatch, mode, rows, world, micro=2, dynamic=False) for rows in world]
    torch.testing.assert_close(sum(grads) / len(world), _reference_grad(mode))
