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
"""Claim probes (VAPO's visual anchors): cuts, claim choice, answers, perception score, reward, config and flow."""

import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from verl.protocol import DataProto
from verl.trainer import claim_probes as cp
from verl.trainer import ray_trainer as ray_trainer_module
from verl.trainer.config import CLAIM_PROBE_ENGINE_HEADROOM, AlgorithmConfig, PPOConfig
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.workers.reward.function import _add_perception_score


ROOT = Path(__file__).resolve().parents[1]
K = 4
DEFAULTS = AlgorithmConfig()
QUESTION = DEFAULTS.claim_probe_question  # the released question, ending with ": "
YES_WORDS, NO_WORDS = DEFAULTS.claim_probe_yes_tokens, DEFAULTS.claim_probe_no_tokens
WORDS = YES_WORDS + NO_WORDS + [" Yes", " No"]


class _Tokenizer:
    """One token per character, plus one token for each yes/no candidate word (ids from 1000)."""

    word_ids = {word: 1000 + idx for idx, word in enumerate(WORDS)}

    def encode(self, text, add_special_tokens=False):
        if text in self.word_ids:
            return [self.word_ids[text]]
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return "".join(chr(i) for i in ids if i < 1000)


TOKENIZER = _Tokenizer()
YES, NO = TOKENIZER.word_ids["Yes"], TOKENIZER.word_ids["No"]


def _claims(num=20):
    # claim i is correct when i is even
    return json.dumps([{"claim": f"claim-{idx}", "correct": idx % 2 == 0} for idx in range(num)])


def _png(seed):
    buffer = io.BytesIO()
    Image.fromarray(np.full((20, 30, 3), seed, dtype=np.uint8)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue()}


# --- cuts, sampling, text ------------------------------------------------------------------------------------


def test_reasoning_span():
    assert cp.reasoning_span("<think>a, b.</think> so \\boxed{1}") == "<think>a, b."
    assert cp.reasoning_span("a, b. so \\boxed{1}") == "a, b. so "
    assert cp.reasoning_span("a, b") == "a, b"
    # <answer> ends the reasoning when </think> is missing, before \boxed
    assert cp.reasoning_span("a, b. <answer>\\boxed{1}</answer>") == "a, b. "
    assert cp.reasoning_span("<think>a, \\boxed{2}</think><answer>1</answer>") == "<think>a, \\boxed{2}"


def test_cut_positions():
    def reference(text, max_position):
        kept, last = [], -3
        for idx, char in enumerate(text):
            if idx < max_position and char in ",.\n!?;:" and idx - last >= 2:
                kept.append(idx)
                last = idx
        return kept

    for text in ["a,,b.c", "x..y:z;w!q?\nr", "no separator", ",", "a, b, c. d: e"]:
        for max_position in (0, 3, len(text)):
            assert cp.cut_positions(text, max_position) == reference(text, max_position)
    assert cp.cut_positions("a,,b", 4) == [1]  # the second comma is one character after the first
    assert cp.cut_positions("a,b,", 4) == [1, 3]


def test_plan_samples_sorted_cuts_and_distinct_claims():
    response = "<think>" + "word, " * 30 + "</think>\\boxed{1}"
    candidates = cp.cut_positions(cp.reasoning_span(response), len(cp.reasoning_span(response)))
    positions, claims, fallback = cp.plan_claim_probes(response, 20, K, False, np.random.default_rng(3))
    assert not fallback and list(positions) == sorted(positions) and set(positions) <= set(candidates)
    assert len(set(positions)) == K and len(set(claims)) == K and all(0 <= c < 20 for c in claims)
    # the same generator state gives the same plan: positions first, then the claims
    rng = np.random.default_rng(3)
    assert list(positions) == sorted(rng.choice(np.asarray(candidates), size=K, replace=False))
    assert list(claims) == list(rng.choice(20, size=K, replace=False))


def test_plan_with_few_or_no_cuts():
    positions, _, fallback = cp.plan_claim_probes("a, b. c \\boxed{1}", 20, K, False, np.random.default_rng(0))
    assert list(positions) == [1, 4, 4, 4] and not fallback  # all cuts, then the last one repeated
    response = "abc \\boxed{1}"
    positions, claims, fallback = cp.plan_claim_probes(response, 20, K, False, np.random.default_rng(0))
    assert fallback and list(positions) == [len(response) - 1] * K and len(set(claims)) == K
    with pytest.raises(ValueError, match="count"):
        cp.plan_claim_probes(response, 3, K, False, np.random.default_rng(0))


def test_long_response_skips_the_reasoning_tail():
    reasoning = "word, " * 60  # 360 characters, cuts up to the end
    response = reasoning + "\\boxed{1}"
    assert max(cp.cut_positions(reasoning, len(reasoning))) >= len(reasoning) - cp.CLAIM_PROBE_LONG_RESPONSE_TAIL
    positions, _, _ = cp.plan_claim_probes(response, 20, 20, True, np.random.default_rng(0))
    assert max(positions) < len(reasoning) - cp.CLAIM_PROBE_LONG_RESPONSE_TAIL


def test_probe_text():
    text = cp.probe_text("<prompt>", "a, b. c", 4, "There is a {cat}.", QUESTION)
    assert text == "<prompt>a, b.\n<anchor>There is a {cat}. Is this claim correct? Answer (Yes/No): "
    assert cp.probe_text("", "a, b", 1, "x", "<{claim}>?") == "a,<x>?"


def test_parse_visual_claims():
    assert cp.parse_visual_claims("[]") == []
    assert cp.parse_visual_claims(_claims(2))[1] == {"claim": "claim-1", "correct": False}
    for bad in ['{"claim": "x"}', '[{"claim": "x", "correct": "yes"}]', '[{"text": "x", "correct": true}]']:
        with pytest.raises(ValueError):
            cp.parse_visual_claims(bad)


def test_answer_ids_must_be_single_tokens():
    yes_ids, no_ids = cp.claim_probe_answer_ids(TOKENIZER, YES_WORDS, NO_WORDS)
    assert yes_ids == [TOKENIZER.word_ids[w] for w in YES_WORDS]
    assert no_ids == [TOKENIZER.word_ids[w] for w in NO_WORDS]
    assert cp.claim_probe_answer_ids(TOKENIZER, [" Yes"], [" No"]) == (
        [TOKENIZER.word_ids[" Yes"]],
        [TOKENIZER.word_ids[" No"]],
    )

    class Split(_Tokenizer):
        word_ids = {word: 1000 + idx for idx, word in enumerate(WORDS) if word != "False"}

    with pytest.raises(ValueError, match="'False' is not a single token"):
        cp.claim_probe_answer_ids(Split(), YES_WORDS, NO_WORDS)
    with pytest.raises(ValueError, match="repeated"):
        cp.claim_probe_answer_ids(TOKENIZER, ["Yes"], ["Yes"])


def test_perception_score():
    def reference(correct, positions, chars, beta):
        weights = [math.exp(beta * p / chars) for p in positions]
        mean = sum(w * c for w, c in zip(weights, correct)) / sum(weights)
        return max(0.0, (mean - 0.5) / 0.5)

    for correct, positions in [([1, 0], [0, 50]), ([1, 1, 0, 1], [3, 10, 60, 99]), ([0, 0], [5, 9]), ([1], [99])]:
        for beta in (0.0, 1.5):
            assert cp.perception_score(np.array(correct), np.array(positions), 100, beta) == pytest.approx(
                reference(correct, positions, 100, beta)
            )
    assert cp.perception_score(np.array([1, 0]), np.array([0, 0]), 10, 1.5) == 0.0  # chance maps to 0


def test_seed_depends_on_seed_step_and_row():
    seeds = {cp.claim_probe_seed(s, t, r) for s in (1, 2) for t in (1, 2) for r in (0, 1)}
    assert len(seeds) == 8 and cp.claim_probe_seed(1, 2, 3) == cp.claim_probe_seed(1, 2, 3)


# --- answering on a worker ------------------------------------------------------------------------------------


def _probe_batch(responses, claims, images, seeds, prompt=(7, 8, 9), extra_prompt_tokens=0):
    width = max(len(r) for r in responses)
    ids = torch.zeros(len(responses), width, dtype=torch.long)
    mask = torch.zeros(len(responses), width, dtype=torch.long)
    for idx, response in enumerate(responses):
        ids[idx, : len(response)] = torch.tensor([ord(c) for c in response])
        mask[idx, : len(response)] = 1
    prompt_mask = torch.ones(len(responses), len(prompt) + extra_prompt_tokens, dtype=torch.long)
    return DataProto.from_dict(
        tensors={"responses": ids, "response_mask": mask, "attention_mask": torch.cat([prompt_mask, mask], -1)},
        non_tensors={
            "raw_prompt_ids": np.array([list(prompt)] * len(responses) + [None], dtype=object)[:-1],
            "visual_claims": np.array(claims, dtype=object),
            "claim_probe_seed": np.array(seeds),
            "multi_modal_data": np.array(images + [None], dtype=object)[:-1],
        },
    )


def _answer(
    probes,
    answer=lambda text: YES,
    max_model_len=10_000,
    long_response_tokens=10_000,
    calls=None,
    question=QUESTION,
    yes_words=YES_WORDS,
    no_words=NO_WORDS,
):
    requests = []

    def generate(inputs):
        requests.extend(inputs)
        return [answer(TOKENIZER.decode(request["prompt_token_ids"])) for request in inputs]

    def process(image, min_pixels, max_pixels):
        if calls is not None:
            calls.append(image)
        return ("processed", id(image), min_pixels, max_pixels)

    output = cp.answer_claim_probes(
        probes,
        tokenizer=TOKENIZER,
        generate_fn=generate,
        process_image_fn=process,
        count=K,
        question=question,
        yes_words=yes_words,
        no_words=no_words,
        long_response_tokens=long_response_tokens,
        max_model_len=max_model_len,
        min_pixels=4,
        max_pixels=400,
    )
    return output, requests


def test_answers_are_compared_with_the_asked_claims_label():
    response = "<think>" + "step, " * 12 + "</think>\\boxed{1}"
    probes = _probe_batch([response, response], [_claims(), _claims()], [{"images": [_png(1)]}] * 2, [11, 12])
    output, requests = _answer(probes)  # always "Yes": right exactly when the asked claim is correct
    for row, seed in enumerate([11, 12]):
        positions, claims, _ = cp.plan_claim_probes(response, 20, K, False, np.random.default_rng(seed))
        assert output.batch["claim_probe_position"][row].tolist() == positions.tolist()
        assert output.batch["claim_probe_correct"][row].tolist() == [c % 2 == 0 for c in claims]
        for j, (position, claim) in enumerate(zip(positions, claims)):
            expected = "\x07\x08\x09" + cp.probe_text("", response, int(position), f"claim-{claim}", QUESTION)
            assert TOKENIZER.decode(requests[row * K + j]["prompt_token_ids"]) == expected
    assert output.batch["claim_probe_answer_yes"].all()
    assert output.batch["claim_probe_response_chars"].tolist() == [len(response)] * 2
    assert not output.batch["claim_probe_fallback"].any()
    assert all("multi_modal_uuids" not in request for request in requests)

    output, _ = _answer(probes, answer=lambda text: TOKENIZER.word_ids["false"])  # any no candidate reads "no"
    assert not output.batch["claim_probe_answer_yes"].any()
    with pytest.raises(ValueError, match="not a yes/no candidate"):
        _answer(probes, answer=lambda text: 5)


def test_question_and_candidates_are_options():
    response = "a, b, c, d, e. \\boxed{1}"
    probes = _probe_batch([response], [_claims()], [None], [3])
    space_yes = TOKENIZER.word_ids[" Yes"]
    output, requests = _answer(
        probes, answer=lambda text: space_yes, question="\n{claim}? (Yes/No):", yes_words=[" Yes"], no_words=[" No"]
    )
    assert all(TOKENIZER.decode(r["prompt_token_ids"]).endswith("? (Yes/No):") for r in requests)
    assert output.batch["claim_probe_answer_yes"].all()
    with pytest.raises(ValueError, match="not a yes/no candidate"):  # "Yes" is not a candidate here
        _answer(probes, question="\n{claim}? (Yes/No):", yes_words=[" Yes"], no_words=[" No"])


def test_each_image_is_processed_once_per_call_and_the_batch_is_unchanged():
    response = "a, b, c, d, e. \\boxed{1}"
    shared, other, same_bytes = _png(1), _png(2), _png(1)
    images = [{"images": [shared]}, {"images": [shared]}, {"images": [other]}, {"images": [same_bytes]}]
    probes = _probe_batch([response] * 4, [_claims()] * 4, images, [1, 2, 3, 4])
    before = [dict(row) for row in probes.non_tensor_batch["multi_modal_data"]]
    calls = []
    _, requests = _answer(probes, calls=calls)
    assert len(calls) == 2  # one per distinct image content, though it reaches 4 x K probes
    assert requests[0]["multi_modal_data"]["image"][0] is requests[-1]["multi_modal_data"]["image"][0]
    assert requests[2 * K]["multi_modal_data"] != requests[0]["multi_modal_data"]
    for row, original in zip(probes.non_tensor_batch["multi_modal_data"], before):
        assert row == original and row["images"][0] is original["images"][0]

    cache = cp.ImageCache(lambda image, lo, hi: (lo, hi))
    image = _png(3)
    assert cache.get(image, 1, 2) != cache.get(image, 1, 3) and cache.misses == 2  # pixel bounds are in the key
    pil = Image.new("RGB", (2, 2))
    cache.get(pil, 1, 2)
    cache.get(pil, 1, 2)
    assert cache.misses == 4  # decoded images are not keyed by identity


def test_long_responses_and_engine_limit():
    response = "<think>" + "word, " * 60 + "</think>\\boxed{1}"
    probes = _probe_batch([response], [_claims()], [None], [5])
    total = int(probes.batch["attention_mask"].sum())
    output, _ = _answer(probes, long_response_tokens=total - 1)
    reasoning = cp.reasoning_span(response)
    assert int(output.batch["claim_probe_position"].max()) < len(reasoning) - cp.CLAIM_PROBE_LONG_RESPONSE_TAIL
    # an image placeholder in the raw prompt expands to the prompt tokens of the attention mask
    probes = _probe_batch([response], [_claims()], [None], [5], extra_prompt_tokens=100)
    output, requests = _answer(probes)
    longest = max(len(request["prompt_token_ids"]) for request in requests) + 100
    assert int(output.batch["claim_probe_max_tokens"][0]) == longest
    with pytest.raises(ValueError, match="max_model_len"):
        _answer(probes, max_model_len=longest - 1)


# --- driver: rows, seeds, scores ------------------------------------------------------------------------------


def _training_batch(accuracy, claims):
    responses = ["<think>" + f"line {idx}, more. " * 4 + "</think>\\boxed{1}" for idx in range(len(accuracy))]
    probes = _probe_batch(responses, claims, [{"images": [_png(idx)]} for idx in range(len(accuracy))], [0] * 4)
    batch = DataProto.from_dict(
        tensors={key: probes.batch[key] for key in ("responses", "response_mask", "attention_mask")},
        non_tensors={key: probes.non_tensor_batch[key] for key in ("raw_prompt_ids", "visual_claims")},
    )
    batch.non_tensor_batch["multi_modal_data"] = probes.non_tensor_batch["multi_modal_data"]
    return batch


def test_probe_rows_and_seeds():
    batch = _training_batch([1.0, 0.0, 1.0, 1.0], [_claims(), _claims(), "[]", _claims()])
    assert cp.claim_probe_rows(batch, [1.0, 0.0, 1.0, 1.0], K) == [0, 3]
    probes = cp.build_claim_probe_batch(batch, [0, 3], data_seed=7, global_step=2)
    assert probes.non_tensor_batch["claim_probe_seed"].tolist() == [
        cp.claim_probe_seed(7, 2, 0),
        cp.claim_probe_seed(7, 2, 3),
    ]
    assert probes.batch["responses"].tolist() == batch.batch["responses"][[0, 3]].tolist()
    bad = _training_batch([1.0] * 4, [_claims(), _claims(K - 1), _claims(), _claims()])
    with pytest.raises(ValueError, match="row 1 has 3 visual claims"):
        cp.claim_probe_rows(bad, [0.0] * 4, K)


def test_scores_are_written_to_probed_rows_only():
    batch = _training_batch([1.0] * 4, [_claims()] * 4)
    output = DataProto.from_dict(
        tensors={
            "claim_probe_correct": torch.tensor([[True, True, False, True], [False, False, False, False]]),
            "claim_probe_answer_yes": torch.tensor([[True, False, True, True], [True, True, True, True]]),
            "claim_probe_position": torch.tensor([[1, 2, 30, 40], [1, 2, 3, 4]]),
            "claim_probe_response_chars": torch.tensor([50, 50]),
            "claim_probe_fallback": torch.tensor([False, True]),
            "claim_probe_max_tokens": torch.tensor([90, 70]),
        }
    )
    metrics = cp.attach_perception_scores(batch, [1, 2], output, beta=1.5)
    scores = batch.non_tensor_batch[cp.PERCEPTION_SCORE_KEY]
    assert np.isnan(scores[[0, 3]]).all()
    assert scores[1] == pytest.approx(cp.perception_score(np.array([1, 1, 0, 1]), np.array([1, 2, 30, 40]), 50, 1.5))
    assert scores[2] == 0.0
    assert metrics["claim_probe/probed_fraction"] == 0.5 and metrics["claim_probe/accuracy"] == 3 / 8
    assert metrics["claim_probe/accuracy_early"] == pytest.approx(2 / 6)
    assert metrics["claim_probe/accuracy_late"] == pytest.approx(1 / 2)
    assert metrics["claim_probe/yes_rate"] == 7 / 8 and metrics["claim_probe/fallback_rate"] == 0.5
    assert metrics["claim_probe/max_prompt_tokens"] == 90

    empty = _training_batch([0.0] * 4, [_claims()] * 4)
    assert cp.attach_perception_scores(empty, [], None, 1.5) == {"claim_probe/probed_fraction": 0.0}
    assert np.isnan(empty.non_tensor_batch[cp.PERCEPTION_SCORE_KEY]).all()


# --- reward ---------------------------------------------------------------------------------------------------


def _math_reward():
    spec = importlib.util.spec_from_file_location("math_reward", ROOT / "examples/reward_function/math.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_math_reward():
    math_reward = _math_reward()
    right, wrong = "<think>x</think> \\boxed{2}", "<think>x</think> \\boxed{3}"
    inputs = [{"response": right, "ground_truth": "2"}, {"response": wrong, "ground_truth": "2"}]
    assert math_reward.compute_score(inputs) == [
        {"overall": 0.9 * 1.0 + 0.1 * 1.0, "format": 1.0, "accuracy": 1.0},
        {"overall": 0.1 * 1.0, "format": 1.0, "accuracy": 0.0},
    ]
    probed = [dict(inputs[0], perception_score=0.5), dict(inputs[1], perception_score=0.5), inputs[0]]
    assert math_reward.compute_score(probed, perception_weight=0.1) == [
        {
            "overall": 0.8 + 0.1 + 0.1 * 0.5,
            "format": 1.0,
            "accuracy": 1.0,
            "perception": 0.5,
            "perception_scored": 1.0,
        },
        {"overall": 0.1, "format": 1.0, "accuracy": 0.0, "perception": 0.5, "perception_scored": 1.0},
        {"overall": 0.9 + 0.1, "format": 1.0, "accuracy": 1.0, "perception": 0.0, "perception_scored": 0.0},
    ]
    assert math_reward.compute_score(probed[:1]) == math_reward.compute_score(inputs[:1])  # weight 0 ignores it


def test_reward_function_kwargs_are_checked_at_start(tmp_path):
    from verl.workers.reward.config import RewardConfig
    from verl.workers.reward.function import AutoRewardManager

    def manager(function, kwargs):
        config = RewardConfig(
            reward_function=f"{ROOT}/examples/reward_function/{function}", reward_function_kwargs=kwargs
        )
        config.post_init()
        return AutoRewardManager(config, tokenizer=None)

    manager("math.py:compute_score", {"format_weight": 0.1, "perception_weight": 0.1})
    with pytest.raises(TypeError, match="perception_weight"):
        manager("math.py:compute_score_wo_format", {"perception_weight": 0.1})
    with pytest.raises(TypeError, match="perception_weight"):
        manager("r1v.py:compute_score", {"perception_weight": 0.1})


def test_reward_manager_passes_finite_scores_only():
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros(2, 1)},
        non_tensors={cp.PERCEPTION_SCORE_KEY: np.array([0.25, np.nan])},
    )
    inputs = [{}, {}]
    for idx, reward_input in enumerate(inputs):
        _add_perception_score(reward_input, data, idx)
    assert inputs == [{"perception_score": 0.25}, {}]
    plain = {}
    _add_perception_score(plain, DataProto.from_dict(tensors={"responses": torch.zeros(1, 1)}), 0)
    assert plain == {}


# --- config ---------------------------------------------------------------------------------------------------


def _config(count=K, weight=0.1, **algorithm):
    config = PPOConfig()
    config.trainer.n_gpus_per_node = 1
    config.data.rollout_batch_size = 2
    config.worker.actor.global_batch_size = 2
    config.worker.actor.micro_batch_size_per_device_for_update = 1
    config.worker.actor.micro_batch_size_per_device_for_experience = 1
    config.worker.rollout.n = 2
    config.algorithm.claim_probe_count = count
    if weight is not None:
        config.worker.reward.reward_function_kwargs = {"perception_weight": weight}
    for key, value in algorithm.items():
        setattr(config.algorithm, key, value)
    return config


def test_question_and_candidates_from_the_command_line():
    """As verl/trainer/main.py merges them: a double-quoted question keeps \\n, the colon and a trailing space; an
    unquoted Yes/True would be a YAML boolean and is refused instead of becoming "True"."""

    def algorithm(*args, count=4):
        args = [f"algorithm.claim_probe_count={count}", *args]
        config = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.from_cli(args))
        config = OmegaConf.to_object(config).algorithm
        config.post_init()
        return config

    config = algorithm(
        'algorithm.claim_probe_question="\\n<anchor>{claim} Is this claim correct? Answer (Yes/No):"',
        'algorithm.claim_probe_yes_tokens=["Yes"]',
        'algorithm.claim_probe_no_tokens=["No"]',
    )
    assert config.claim_probe_question == "\n<anchor>{claim} Is this claim correct? Answer (Yes/No):"
    assert (config.claim_probe_yes_tokens, config.claim_probe_no_tokens) == (["Yes"], ["No"])
    assert algorithm('algorithm.claim_probe_question="\\n{claim}: "').claim_probe_question == "\n{claim}: "
    assert algorithm().claim_probe_question.endswith("(Yes/No): ")  # the released question by default
    for tokens in ("[Yes,true]", "[1]", '[["Yes"]]', "[]", '[""]'):
        with pytest.raises(ValueError, match="quote every item"):
            algorithm(f"algorithm.claim_probe_yes_tokens={tokens}")
    with pytest.raises(ValueError, match="repeats"):
        algorithm('algorithm.claim_probe_yes_tokens=["Yes","Yes"]')
    with pytest.raises(ValueError, match="literal backslash"):  # single quotes keep \\n as two characters
        algorithm("algorithm.claim_probe_question='\\n<anchor>{claim} Answer (Yes/No):'")
    with pytest.raises(Exception, match="DictConfig"):  # unquoted, YAML reads "...: " as a mapping
        algorithm("algorithm.claim_probe_question=<anchor>{claim} Answer (Yes/No): ")
    with pytest.raises(Exception, match="ListConfig"):
        algorithm('algorithm.claim_probe_yes_tokens="Yes"')
    with pytest.raises(ValueError, match="only apply to claim probes"):
        algorithm('algorithm.claim_probe_yes_tokens=["Yes"]', count=0)
    algorithm(count=0)  # the defaults are fine without probes
    with pytest.raises(ValueError, match="exactly once"):
        algorithm('algorithm.claim_probe_question="Is it correct?"')
    with pytest.raises(ValueError, match="must not share"):
        algorithm('algorithm.claim_probe_yes_tokens=["Yes"]', 'algorithm.claim_probe_no_tokens=["Yes"]')


def test_config_checks():
    config = _config()
    config.post_init()
    rollout = config.worker.rollout
    assert rollout.max_model_len == rollout.prompt_length + rollout.response_length + CLAIM_PROBE_ENGINE_HEADROOM

    config = _config(count=0, weight=None)
    config.post_init()
    assert config.worker.rollout.max_model_len is None

    for config, match in [
        (_config(weight=None), "perception_weight > 0"),
        (_config(count=0), "claim_probe_count > 0"),
        (_config(adv_estimator="remax"), "remax"),
        (_config(use_grounding_consistency_reward=True), "use_grounding_consistency_reward"),
        (_config(rollout_image_transform="vp_diffusion"), "rollout_image_transform"),
        (_config(online_filtering=True, filter_key="overall"), "filter_key=accuracy or format"),
    ]:
        with pytest.raises(ValueError, match=match):
            config.deep_post_init()
    config = _config()
    config.worker.rollout.interaction_mode = "agentic"
    with pytest.raises(ValueError, match="agentic"):
        config._validate_claim_probes()
    config = _config(online_filtering=True, filter_key="accuracy")
    config.worker.rollout.max_model_len = 1024
    with pytest.raises(ValueError, match="no room for the claim probe question"):
        config.post_init()
    with pytest.raises(ValueError, match="non-negative"):
        _config(count=-1).algorithm.post_init()


# --- trainer flow ---------------------------------------------------------------------------------------------


class _WorkerGroup:
    world_size = 2

    def __init__(self, n):
        self.n = n
        self.probe_calls = []

    def generate_sequences(self, gen_batch):
        rows = len(gen_batch) * self.n
        responses = torch.tensor([[ord(c) for c in "a, b, c, d. \\boxed{1}"]] * rows)
        return DataProto(
            batch=DataProto.from_dict(
                {
                    "responses": responses,
                    "response_mask": torch.ones_like(responses),
                    "attention_mask": torch.ones(rows, responses.size(1) + 3, dtype=torch.long),
                }
            ).batch,
            non_tensor_batch={"multi_modal_data": np.repeat(gen_batch.non_tensor_batch["multi_modal_data"], self.n)},
        )

    def answer_claim_probes(self, probes):
        self.probe_calls.append(probes)
        assert len(probes) % self.world_size == 0
        meta = probes.meta_info
        output, _ = _answer(
            probes,
            question=meta["claim_probe_question"],
            yes_words=meta["claim_probe_yes_tokens"],
            no_words=meta["claim_probe_no_tokens"],
        )
        return output


class _Reward:
    def __init__(self, accuracy):
        self.accuracy = accuracy
        self.calls = 0
        self.compute_reward = SimpleNamespace(remote=self._compute)

    def _compute(self, batch):
        self.calls += 1
        return torch.zeros(len(batch), 1), {"accuracy": list(self.accuracy[: len(batch)])}


def _trainer(monkeypatch, config, accuracy):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.global_step = 3
    trainer.grounding_consistency_scorer = None
    trainer.reward_fn = _Reward(accuracy)
    num_prompts = config.data.rollout_batch_size

    def batches():
        while True:
            yield {
                "input_ids": torch.arange(num_prompts)[:, None],
                "attention_mask": torch.ones(num_prompts, 1, dtype=torch.long),
                "position_ids": torch.zeros(num_prompts, 1, dtype=torch.long),
                "raw_prompt_ids": np.array([[7, 8, 9 + idx] for idx in range(num_prompts)] + [None], dtype=object)[
                    :-1
                ],
                "multi_modal_data": np.array([{"images": [_png(idx)]} for idx in range(num_prompts)]),
                "visual_claims": np.array([_claims()] * num_prompts, dtype=object),
            }

    trainer.data_iterator = batches()
    trainer.actor_rollout_ref_wg = _WorkerGroup(config.worker.rollout.n)
    monkeypatch.setattr(ray_trainer_module.ray, "get", lambda value: value)
    return trainer


def test_step_probes_correct_responses_and_rescores(monkeypatch):
    config = _config()
    config.post_init()
    trainer = _trainer(monkeypatch, config, accuracy=[1.0, 0.0, 0.0, 1.0])
    batch = trainer._make_batch_data({})
    # the prompt ids are kept for the probes, repeated to align with the responses
    assert [list(ids) for ids in batch.non_tensor_batch["raw_prompt_ids"]] == [[7, 8, 9]] * 2 + [[7, 8, 10]] * 2
    batch.batch["token_level_scores"] = torch.zeros(len(batch), 1)  # as online filtering leaves it
    batch.non_tensor_batch["reward_details"] = np.array([{}] * len(batch))
    metrics = {}
    trainer._attach_claim_probe_scores(batch, metrics)

    (probes,) = trainer.actor_rollout_ref_wg.probe_calls
    assert probes.non_tensor_batch["claim_probe_seed"].tolist() == [
        cp.claim_probe_seed(config.data.seed, 3, 0),
        cp.claim_probe_seed(config.data.seed, 3, 3),
    ]
    assert probes.meta_info["claim_probe_question"] == config.algorithm.claim_probe_question
    assert probes.meta_info["claim_probe_yes_tokens"] == config.algorithm.claim_probe_yes_tokens
    assert [list(ids) for ids in probes.non_tensor_batch["raw_prompt_ids"]] == [[7, 8, 9], [7, 8, 10]]
    scores = batch.non_tensor_batch[cp.PERCEPTION_SCORE_KEY]
    assert np.isfinite(scores[[0, 3]]).all() and np.isnan(scores[[1, 2]]).all()
    assert metrics["claim_probe/probed_fraction"] == 0.5
    assert "raw_prompt_ids" not in batch.non_tensor_batch
    assert "token_level_scores" not in batch.batch.keys() and "reward_details" not in batch.non_tensor_batch


def test_scores_do_not_depend_on_which_rows_are_probed(monkeypatch):
    def scores(accuracy):
        config = _config()
        config.post_init()
        trainer = _trainer(monkeypatch, config, accuracy)
        batch = trainer._make_batch_data({})
        trainer._attach_claim_probe_scores(batch, {})
        return batch.non_tensor_batch[cp.PERCEPTION_SCORE_KEY]

    assert scores([1.0, 1.0, 1.0, 1.0])[3] == scores([0.0, 0.0, 0.0, 1.0])[3]


def test_no_correct_response_skips_the_workers(monkeypatch):
    config = _config()
    config.post_init()
    trainer = _trainer(monkeypatch, config, accuracy=[0.0] * 4)
    batch = trainer._make_batch_data({})
    trainer._attach_claim_probe_scores(batch, {})
    assert trainer.actor_rollout_ref_wg.probe_calls == []
    assert np.isnan(batch.non_tensor_batch[cp.PERCEPTION_SCORE_KEY]).all()


def test_filtered_groups_are_not_probed(monkeypatch):
    """With online filtering the probes run on the kept groups only, once, after the filter."""
    config = _config(online_filtering=True, filter_key="accuracy")
    config.data.rollout_batch_size = 1
    config.worker.actor.global_batch_size = 1
    config.post_init()
    trainer = _trainer(monkeypatch, config, accuracy=None)
    trainer.data_iterator = _two_prompt_batches()  # a round of two prompts; group 0 is solved, group 1 mixed

    def reward(batch):
        values = [1.0, 1.0, 1.0, 0.0] if len(batch) == 4 else [1.0, 0.0]
        return torch.tensor(values)[:, None], {"accuracy": values}

    trainer.reward_fn.compute_reward = SimpleNamespace(remote=reward)
    batch = trainer._make_batch_data({})
    assert len(batch) == 2 and [list(ids) for ids in batch.non_tensor_batch["raw_prompt_ids"]] == [[7, 8, 10]] * 2
    trainer._attach_claim_probe_scores(batch, {})
    (probes,) = trainer.actor_rollout_ref_wg.probe_calls  # the kept correct response, padded to the world size
    assert [list(ids) for ids in probes.non_tensor_batch["raw_prompt_ids"]] == [[7, 8, 10]] * 2
    assert probes.non_tensor_batch["claim_probe_seed"].tolist() == [cp.claim_probe_seed(config.data.seed, 3, 0)] * 2
    scores = batch.non_tensor_batch[cp.PERCEPTION_SCORE_KEY]
    assert np.isfinite(scores[0]) and np.isnan(scores[1])


def _two_prompt_batches():
    while True:
        yield {
            "input_ids": torch.arange(2)[:, None],
            "attention_mask": torch.ones(2, 1, dtype=torch.long),
            "position_ids": torch.zeros(2, 1, dtype=torch.long),
            "raw_prompt_ids": np.array([[7, 8, 9], [7, 8, 10], None], dtype=object)[:-1],
            "multi_modal_data": np.array([{"images": [_png(0)]}, {"images": [_png(1)]}]),
            "visual_claims": np.array([_claims()] * 2, dtype=object),
        }


def test_disabled_probes_keep_the_batch_unchanged(monkeypatch):
    config = _config(count=0, weight=None)
    config.post_init()
    trainer = _trainer(monkeypatch, config, accuracy=[1.0] * 4)
    batch = trainer._make_batch_data({})
    assert "raw_prompt_ids" not in batch.non_tensor_batch
    assert trainer.reward_fn.calls == 0 and trainer.actor_rollout_ref_wg.probe_calls == []


def test_filtered_runs_log_the_perception_of_the_final_rescoring():
    """Online filtering logs the reward metrics of every candidate before the probes; the perception metrics of
    that pass are placeholders (0), so they come from the rescoring after the probes, as reward/overall does."""
    config = _config(online_filtering=True, filter_key="accuracy")
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config
    metrics = {
        "reward/overall": 0.4,
        "reward/accuracy": 0.5,
        "reward/perception": 0.0,
        "reward/perception_scored": 0.0,
    }
    rescored = {"overall": [0.95], "accuracy": [1.0], "perception": [0.8], "perception_scored": [1.0]}
    trainer._log_reward_metrics(metrics, rescored)
    assert metrics == {
        "reward/overall": 0.95,
        "reward/accuracy": 0.5,  # every candidate, as logged by online filtering
        "reward/perception": 0.8,
        "reward/perception_scored": 1.0,
    }

    unfiltered = {}
    trainer.config = _config()
    trainer._log_reward_metrics(unfiltered, rescored)
    assert unfiltered == {f"reward/{k}": v[0] for k, v in rescored.items()}


def test_math_reward_reports_the_perception_metrics_the_trainer_rescores():
    scores = _math_reward().compute_score(
        [{"response": "<think>x</think> \\boxed{2}", "ground_truth": "2", "perception_score": 0.5}],
        perception_weight=0.1,
    )
    assert set(cp.PERCEPTION_REWARD_METRICS) == set(scores[0]) - {"overall", "format", "accuracy"}
