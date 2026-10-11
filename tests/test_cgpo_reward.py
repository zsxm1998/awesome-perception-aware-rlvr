# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import importlib.util
from pathlib import Path

import pytest
import torch

from verl.protocol import DataProto
from verl.workers.reward.function import BatchFunctionRewardManagerMixin


_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "reward_function" / "xml_grounded_reasoning.py"
_SPEC = importlib.util.spec_from_file_location("xml_grounded_reasoning_reward", _MODULE_PATH)
xml_grounded_reasoning = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(xml_grounded_reasoning)


def _wrap_response(think: str, boxed: str = "A", trailing: str = "") -> str:
    return f"<think>{think}</think>{trailing}\\boxed{{{boxed}}}"


def test_cgpo_format_reward_accepts_valid_multi_image_regions():
    response = _wrap_response(
        'Observe <region name="cat" image_idx="1" id="0">[[10, 20, 110, 220]]</region> '
        'next to <region name="dog" image_idx="0" id="1">[[300, 300, 500, 700], [520, 310, 700, 720]]</region>.',
    )
    assert xml_grounded_reasoning.format_reward(response) == 1.0


def test_cgpo_format_reward_accepts_same_name_on_different_images():
    response = _wrap_response(
        'Compare <region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region> '
        'with <region name="cat" image_idx="1" id="1">[[210, 220, 310, 420]]</region>.'
    )
    assert xml_grounded_reasoning.format_reward(response) == 1.0


def test_cgpo_format_reward_requires_at_least_one_region():
    response = _wrap_response("Only reason about the image without grounding evidence.")
    assert xml_grounded_reasoning.format_reward(response) == 0.5


def test_cgpo_format_reward_rejects_duplicate_same_image_same_name_regions():
    response = _wrap_response(
        'See <region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>. '
        'Another <region name="cat" image_idx="0" id="1">[[120, 220, 180, 300]]</region>.'
    )
    assert xml_grounded_reasoning.format_reward(response) == 0.8


def test_cgpo_compute_score_uses_new_format_reward():
    scores = xml_grounded_reasoning.compute_score(
        [
            {
                "response": (
                    "<think>"
                    'Observe <region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>.'
                    "</think>"
                    "\\boxed{42}"
                ),
                "ground_truth": "42",
                "response_length": 1,
            }
        ]
    )
    assert scores == [{"overall": 1.0, "format": 1.0, "accuracy": 1.0}]


def test_cgpo_compute_score_adds_grounding_consistency_to_breakdown_and_overall():
    scores = xml_grounded_reasoning.compute_score(
        [
            {
                "response": (
                    "<think>"
                    'Observe <region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>.'
                    "</think>"
                    "\\boxed{42}"
                ),
                "ground_truth": "42",
                "response_length": 1,
                "grounding_consistency": 0.3,
                "grounding_consistency_raw": 3.0,
            }
        ]
    )
    assert scores == [{"overall": 1.3, "format": 1.0, "accuracy": 1.0, "grounding_consistency": 3.0}]


def test_cgpo_compute_score_gates_grounding_consistency_on_correctness():
    scores = xml_grounded_reasoning.compute_score(
        [
            {
                "response": (
                    "<think>"
                    'Observe <region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>.'
                    "</think>"
                    "\\boxed{24}"
                ),
                "ground_truth": "42",
                "response_length": 1,
                "grounding_consistency": 0.3,
                "grounding_consistency_raw": 3.0,
            }
        ]
    )
    assert scores == [{"overall": 0.5, "format": 1.0, "accuracy": 0.0, "grounding_consistency": 0.0}]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("plain answer \\boxed{A}", 0.0),
        ("<think>reason only</think>", 0.0),
    ],
)
def test_cgpo_format_reward_rejects_invalid_basic_format(response, expected):
    assert xml_grounded_reasoning.format_reward(response) == expected


@pytest.mark.parametrize(
    "think_text",
    [
        'See <region name="cat" image_idx="1" id="1">[[10, 20, 110, 220]]</region>.',
        'See <region name="cat" image_idx="1" id="0">[[10, 20, 110, 220]]</region> '
        'and <region name="dog" image_idx="0" id="2">[[210, 220, 310, 420]]</region>.',
        'See <region name="cat" image_idx="1" id="0">[[10, 20, 110, 220]]</region> '
        'and <region name="dog" image_idx="0" id="0">[[210, 220, 310, 420]]</region>.',
    ],
)
def test_cgpo_format_reward_rejects_invalid_region_id_progression(think_text):
    assert xml_grounded_reasoning.format_reward(_wrap_response(think_text)) == 0.6


@pytest.mark.parametrize(
    "region_text",
    [
        '<region name="cat" image_idx="-1" id="0">[[10, 20, 110, 220]]</region>',
        '<region name="cat" image_idx="left" id="0">[[10, 20, 110, 220]]</region>',
        '<region name="" image_idx="0" id="0">[[10, 20, 110, 220]]</region>',
        '<region name="cat" id="0">[[10, 20, 110, 220]]</region>',
        '<region name="cat" image_idx="0" id="0" role="evidence">[[10, 20, 110, 220]]</region>',
    ],
)
def test_cgpo_format_reward_rejects_invalid_region_attributes(region_text):
    assert xml_grounded_reasoning.format_reward(_wrap_response(f"See {region_text}.")) == 0.6


def test_cgpo_format_reward_rejects_region_outside_think():
    response = (
        '<think>Reason first.</think><region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>\\boxed{A}'
    )
    assert xml_grounded_reasoning.format_reward(response) == 0.5


@pytest.mark.parametrize(
    "think_text",
    [
        'I identify the visible evidence.\n<region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>.\nIt is relevant.',
        '<region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>\n'
        '<region name="dog" image_idx="0" id="1">[[300, 300, 500, 700]]</region>\n'
        "The two animals are visible.",
        'Evidence: <region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>. It is relevant.',
        '- Circle b: <region name="Circle b" image_idx="0" id="0">[[200, 300, 600, 500]]</region>.',
        '- Circle "b": <region name="Circle b" image_idx="0" id="0">[[200, 300, 600, 500]]</region>.',
    ],
)
def test_cgpo_format_reward_caps_standalone_grounding_tags(think_text):
    assert xml_grounded_reasoning.format_reward(_wrap_response(think_text)) == 0.5


def test_cgpo_format_reward_accepts_region_as_sentence_subject_when_reasoning_continues():
    response = _wrap_response(
        '<region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region> is the animal I need to compare.'
    )
    assert xml_grounded_reasoning.format_reward(response) == 1.0


@pytest.mark.parametrize(
    "think_text",
    [
        "The correct answer is B. The key visible evidence is "
        '<region name="orbit b" image_idx="0" id="0">[[200, 300, 600, 500]]</region>.',
        "Thus, the only correct statement is B. "
        '<region name="orbit b" image_idx="0" id="0">[[200, 300, 600, 500]]</region> lies in the equatorial plane.',
        "So Option C is the correct additional condition. Now ground "
        '<region name="triangle ABC" image_idx="0" id="0">[[0, 0, 500, 300]]</region>.',
    ],
)
def test_cgpo_format_reward_caps_post_answer_grounding(think_text):
    assert xml_grounded_reasoning.format_reward(_wrap_response(think_text)) == 0.5


def test_cgpo_format_reward_accepts_grounding_before_answer_decision():
    response = _wrap_response(
        'The <region name="orbit b" image_idx="0" id="0">[[200, 300, 600, 500]]</region> '
        "lies in the equatorial plane, so it can match the geostationary orbit. "
        "The correct answer is B."
    )
    assert xml_grounded_reasoning.format_reward(response) == 1.0


def test_cgpo_compute_score_preserves_raw_newlines_for_standalone_grounding_cap():
    scores = xml_grounded_reasoning.compute_score(
        [
            {
                "response": (
                    "<think>First inspect the image.\n"
                    '<region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region>.\n'
                    "Then answer.</think>"
                    "\\boxed{42}"
                ),
                "ground_truth": "42",
                "response_length": 1,
            }
        ]
    )
    assert scores[0] == pytest.approx({"overall": 0.75, "format": 0.5, "accuracy": 1.0})


_LATE_THINK = (
    "Figure A gives the growth ratio over the years and Figure B the population counts. "
    "The ratio stays at one for four years, then drops, then rises again, so the population shrinks and recovers. "
    "The counts in Figure B swing up and down without a fixed period, so they are not strictly periodic. "
    'The <region name="figure a" image_idx="0" id="0">[[270, 36, 720, 400]]</region> shows this ratio curve.'
)
_EARLY_THINK = (
    'The <region name="figure a" image_idx="0" id="0">[[270, 36, 720, 400]]</region> gives the growth ratio over '
    "the years. It stays at one for four years, then drops, then rises again, so the population shrinks and "
    "recovers. The counts in Figure B swing up and down without a fixed period, so they are not strictly periodic."
)


def test_cgpo_format_reward_caps_grounding_that_follows_the_reasoning():
    # the region starts after 60% of the reasoning: appended once the reasoning is done
    assert xml_grounded_reasoning.format_reward(_wrap_response(_LATE_THINK)) == 1.0  # off by default
    assert xml_grounded_reasoning.format_reward(_wrap_response(_LATE_THINK), late_grounding_fraction=0.6) == 0.5
    assert xml_grounded_reasoning.format_reward(_wrap_response(_EARLY_THINK), late_grounding_fraction=0.6) == 1.0
    # no region at all: the late check does not apply (0.5 from the missing grounding either way)
    assert xml_grounded_reasoning.format_reward(_wrap_response("plain reasoning"), late_grounding_fraction=0.6) == 0.5


def test_cgpo_compute_score_forwards_late_grounding_fraction():
    inputs = [{"response": _wrap_response(_LATE_THINK, boxed="42"), "ground_truth": "42", "response_length": 1}]
    assert xml_grounded_reasoning.compute_score(inputs)[0]["format"] == 1.0
    scores = xml_grounded_reasoning.compute_score(inputs, format_weight=0.1, late_grounding_fraction=0.6)
    assert scores[0] == pytest.approx({"overall": 0.9 + 0.1 * 0.5, "format": 0.5, "accuracy": 1.0})
    # the position is read on the raw response too (whitespace around tags is normalized before the ladder)
    spaced = [
        {
            "response": _wrap_response(_LATE_THINK.replace("<region", "< region")),
            "ground_truth": "A",
            "response_length": 1,
        }
    ]
    assert xml_grounded_reasoning.compute_score(spaced, late_grounding_fraction=0.6)[0]["format"] == 0.5
    with pytest.raises(ValueError, match="late_grounding_fraction"):
        xml_grounded_reasoning.compute_score(inputs, late_grounding_fraction=1.0)


def test_cgpo_first_region_position_counts_the_reasoning_prose():
    region = '<region name="a" image_idx="0" id="0">[[1, 2, 3, 4]]</region>'
    assert xml_grounded_reasoning.first_region_position("no regions") is None
    assert xml_grounded_reasoning.first_region_position(f"{region} abcd") == 0.0
    # 3 of the 6 non-whitespace prose characters come before the region, whatever the spacing or the regions after it
    assert xml_grounded_reasoning.first_region_position(f"abc {region} def") == 0.5
    assert xml_grounded_reasoning.first_region_position(f"  a b   c\n{region}\n\nd e f  {region}{region}") == 0.5
    assert xml_grounded_reasoning.first_region_position(region) == 0.0


def test_cgpo_late_grounding_ignores_whitespace_and_later_regions():
    def format_score(think):
        inputs = [{"response": _wrap_response(think), "ground_truth": "A", "response_length": 1}]
        return xml_grounded_reasoning.compute_score(inputs, late_grounding_fraction=0.6)[0]["format"]

    assert format_score(_LATE_THINK) == 0.5
    # whitespace inside the box list does not move the region earlier
    assert format_score(_LATE_THINK.replace("[[270", "[[" + " " * 400 + "270")) == 0.5
    # nor do more regions appended after the first one
    more_regions = "".join(
        f' The <region name="part {idx}" image_idx="0" id="{idx}">[[{100 + idx}, 210, 330, 440]]</region> is relevant.'
        for idx in range(1, 6)
    )
    assert format_score(_LATE_THINK + more_regions) == 0.5
    # and leading whitespace does not move an early region later
    early = 'Here <region name="figure a" image_idx="0" id="0">[[270, 36, 720, 400]]</region> ' + _EARLY_THINK[85:]
    assert format_score(early) == 1.0
    assert format_score(" " * 2000 + early) == 1.0


def test_cgpo_late_grounding_fraction_is_checked_at_start():
    from verl.workers.reward.config import RewardConfig
    from verl.workers.reward.function import AutoRewardManager

    def manager(kwargs):
        config = RewardConfig(reward_function=f"{_MODULE_PATH}:compute_score", reward_function_kwargs=kwargs)
        config.post_init()
        return AutoRewardManager(config, tokenizer=None)

    manager({"format_weight": 0.1, "late_grounding_fraction": 0.6})
    manager({"format_weight": 0.1})
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="late_grounding_fraction"):
            manager({"format_weight": 0.1, "late_grounding_fraction": bad})


def test_cgpo_compute_score_caps_post_answer_grounding():
    scores = xml_grounded_reasoning.compute_score(
        [
            {
                "response": (
                    "<think>The correct answer is B.\n"
                    'The key visible evidence is <region name="orbit b" image_idx="0" id="0">[[200, 300, 600, 500]]</region>.'
                    "</think>"
                    "\\boxed{42}"
                ),
                "ground_truth": "42",
                "response_length": 1,
            }
        ]
    )
    assert scores[0] == pytest.approx({"overall": 0.75, "format": 0.5, "accuracy": 1.0})


@pytest.mark.parametrize(
    "bbox_literal",
    [
        "[]",
        "[10, 20, 110, 220]",
        "[[10, 20, 110]]",
        "[[10.5, 20, 110, 220]]",
        "[[True, 20, 110, 220]]",
        "{[[10, 20, 110, 220]]}",
        "[[-1, 20, 110, 220]]",
        "[[10, 20, 1001, 220]]",
        "[[10, 20, 10, 220]]",
        "[[10, 20, 110, 20]]",
    ],
)
def test_cgpo_format_reward_rejects_invalid_bbox_lists(bbox_literal):
    response = _wrap_response(f'See <region name="cat" image_idx="0" id="0">{bbox_literal}</region>.')
    assert xml_grounded_reasoning.format_reward(response) == 0.7


def test_cgpo_format_reward_rejects_image_idx_out_of_range_when_num_images_is_provided():
    response = _wrap_response('See <region name="cat" image_idx="1" id="0">[[10, 20, 110, 220]]</region>.')
    assert xml_grounded_reasoning.format_reward(response, num_images=1) == 0.6
    assert xml_grounded_reasoning.format_reward(response, num_images=None) == 1.0


def test_cgpo_format_reward_rejects_duplicate_region_after_name_normalization():
    response = _wrap_response(
        'See <region name="  Cat " image_idx="0" id="0">[[10, 20, 110, 220]]</region>. '
        'Again <region name="cat" image_idx="0" id="1">[[210, 220, 310, 420]]</region>.'
    )
    assert xml_grounded_reasoning.format_reward(response) == 0.8


def test_cgpo_format_reward_allows_valid_response_without_repeated_grounding():
    response = _wrap_response(
        'Observe <region name="cat" image_idx="0" id="0">[[10, 20, 110, 220]]</region> only once.'
    )
    assert xml_grounded_reasoning.format_reward(response) == 1.0


class _DummyTokenizer:
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids.tolist() if token_id != self.pad_token_id)


class _DummyRewardManager(BatchFunctionRewardManagerMixin):
    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.config = type("Config", (), {"skip_special_tokens": False})()
        self.captured_inputs = None

        def reward_fn(reward_inputs):
            self.captured_inputs = reward_inputs
            return [{"overall": 0.0, "format": 0.0, "accuracy": 0.0} for _ in reward_inputs]

        self.reward_fn = reward_fn


def test_response_ids_reach_only_the_reward_functions_that_ask_for_them(tmp_path):
    """REWARD_INPUT_RESPONSE_IDS = True in the reward module (GRIT) adds the response token ids."""
    from verl.workers.reward.config import RewardConfig
    from verl.workers.reward.function import AutoRewardManager

    root = Path(__file__).resolve().parents[1] / "examples" / "reward_function"
    for name, expected in (("grit.py", True), ("math.py", False)):
        config = RewardConfig(reward_function=f"{root / name}:compute_score")
        config.post_init()
        assert AutoRewardManager(config, _DummyTokenizer()).add_response_ids is expected

    manager = _DummyRewardManager()
    manager.add_response_ids = True
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[97, 98, 99, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        },
        non_tensors={"ground_truth": ["abc"]},
    )
    manager.compute_reward_batch(data)
    assert manager.captured_inputs[0]["response_ids"] == [97, 98, 99]


def test_reward_manager_passes_num_images_to_reward_function():
    manager = _DummyRewardManager()
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[97, 98, 99]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "ground_truth": ["abc"],
            "multi_modal_data": [{"images": [object(), object(), object()]}],
        },
    )
    manager.compute_reward_batch(data)
    assert manager.captured_inputs == [
        {
            "response": "abc",
            "response_length": 3,
            "ground_truth": "abc",
            "num_images": 3,
        }
    ]


def test_reward_manager_passes_grounding_consistency_to_reward_function():
    manager = _DummyRewardManager()
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[97, 98, 99]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "ground_truth": ["abc"],
            "grounding_consistency": [0.25],
        },
    )
    manager.compute_reward_batch(data)
    assert manager.captured_inputs == [
        {
            "response": "abc",
            "response_length": 3,
            "ground_truth": "abc",
            "grounding_consistency": 0.25,
        }
    ]


def test_reward_manager_forwards_raw_grounding_keys():
    manager = _DummyRewardManager()
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[97, 98, 99]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        },
        non_tensors={
            "ground_truth": ["abc"],
            "grounding_consistency": [0.1],
            "grounding_consistency_raw": [1.0],
        },
    )
    manager.compute_reward_batch(data)
    assert manager.captured_inputs == [
        {
            "response": "abc",
            "response_length": 3,
            "ground_truth": "abc",
            "grounding_consistency": 0.1,
            "grounding_consistency_raw": 1.0,
        }
    ]


def test_reward_manager_canonicalizes_qwen35_prefilled_think_response():
    manager = _DummyRewardManager()
    prompt = "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n<think>\n"
    response = "reasoning</think>\\boxed{A}"
    prompt_ids = [ord(char) for char in prompt]
    response_ids = [ord(char) for char in response]
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([prompt_ids], dtype=torch.long),
            "responses": torch.tensor([response_ids], dtype=torch.long),
            "response_mask": torch.ones(1, len(response_ids), dtype=torch.long),
            "attention_mask": torch.ones(1, len(prompt_ids) + len(response_ids), dtype=torch.long),
        },
        non_tensors={"ground_truth": ["A"]},
    )

    manager.compute_reward_batch(data)

    assert manager.captured_inputs == [
        {
            "response": "<think>\nreasoning</think>\\boxed{A}",
            "response_length": len(response_ids),
            "ground_truth": "A",
        }
    ]
