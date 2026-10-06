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
"""Visual-claim probes inside the responses (VAPO's visual anchors, arXiv 2509.25848).

Each training row carries pre-generated claims about its image, labeled true or false (the ``visual_claims``
column, a JSON list of ``{"claim": str, "correct": bool}``). After the rollout, every response with
accuracy 1 is cut at ``K = algorithm.claim_probe_count`` random punctuation positions of its reasoning; at
each cut the policy reads the prompt, the response up to the cut and one claim, and answers yes/no in one
greedy token. A probe is right when the answer matches the label of the claim it was asked. The perception
score is the right-answer rate weighted toward later cuts and rescaled so chance (0.5) maps to 0:

    R_perc = max(0, (sum_k w_k s_k / sum_k w_k - 0.5) / 0.5),    w_k = exp(beta * a_k / T),

with a_k the character position of the cut and T the number of characters of the decoded response (beta is
``algorithm.claim_probe_late_emphasis``). The reward function adds it for correct answers (``perception_weight``
of ``examples/reward_function/math.py``). Cuts, claim choice, prompt text, the candidate answer tokens and the
long-response rule follow the released VAPO code (the question and the candidates are options whose defaults are
the released ones), except that the reasoning also ends at ``<answer>``; the randomness comes from the config
seed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional, Sequence

import numpy as np
import torch

from ..protocol import DataProto
from ..utils.dataset import ImageCache


VISUAL_CLAIMS_KEY = "visual_claims"
"""dataset column: JSON list of {"claim": str, "correct": bool}; "[]" for rows without claims."""
PERCEPTION_SCORE_KEY = "perception_score"
"""float column of the training batch: R_perc of a probed response, NaN for responses that were not probed."""
PERCEPTION_REWARD_METRICS = ("perception", "perception_scored")
"""reward metrics of examples/reward_function/math.py that depend on the probes; with online filtering, their values
from the filtering pass (before the probes) are placeholders and the rescoring after the probes replaces them"""

CLAIM_PROBE_SEPARATORS = frozenset({",", ".", "\n", "!", "?", ";", ":"})
"""characters after which a response can be cut"""
CLAIM_PROBE_MIN_CUT_DISTANCE = 2
"""a cut is kept only if it is at least this many characters after the previous kept cut"""
CLAIM_PROBE_LONG_RESPONSE_MARGIN = 144
"""a response whose prompt + response tokens exceed prompt_length + response_length - this margin is not cut in
the last CLAIM_PROBE_LONG_RESPONSE_TAIL characters of its reasoning (the released code: 6000 for 4096 + 2048)"""
CLAIM_PROBE_LONG_RESPONSE_TAIL = 128


def reasoning_span(response: str) -> str:
    """The text before ``</think>``; without it, the text before ``<answer>``, then before ``\\boxed``; otherwise
    the whole response. The released code checks ``</think>`` and ``\\boxed``; ``<answer>`` keeps the cuts out of
    the answer of formats that put it there."""
    for marker in ("</think>", "<answer>"):
        end = response.find(marker)
        if end != -1:
            return response[:end]
    boxed_start = response.find("\\boxed")
    if boxed_start != -1:
        return response[:boxed_start]
    return response


def cut_positions(reasoning: str, max_position: int) -> list[int]:
    """Character positions of separators before ``max_position``, each at least CLAIM_PROBE_MIN_CUT_DISTANCE
    after the previous kept one."""
    positions, last = [], -CLAIM_PROBE_MIN_CUT_DISTANCE - 1
    for position, char in enumerate(reasoning[: max(max_position, 0)]):
        if char in CLAIM_PROBE_SEPARATORS and position - last >= CLAIM_PROBE_MIN_CUT_DISTANCE:
            positions.append(position)
            last = position
    return positions


def plan_claim_probes(
    response: str, num_claims: int, count: int, long_response: bool, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sorted cut positions, the claim asked at each cut, and whether no cut existed.

    ``count`` cuts are drawn without replacement; with fewer candidates, all are used and the last one is
    repeated; without any, every probe reads the whole response (position ``len(response) - 1``). The claims are
    ``count`` distinct indices drawn after the positions, from the same generator.
    """
    if not 0 < count <= num_claims:
        raise ValueError(f"claim probes need 0 < count <= number of claims, got {count} and {num_claims}")
    reasoning = reasoning_span(response)
    max_position = len(reasoning) - CLAIM_PROBE_LONG_RESPONSE_TAIL if long_response else len(reasoning)
    candidates = np.asarray(cut_positions(reasoning, max_position), dtype=np.int64)
    fallback = len(candidates) == 0
    if fallback:
        positions = np.full(count, len(response) - 1, dtype=np.int64)
    elif count <= len(candidates):
        positions = np.sort(rng.choice(candidates, size=count, replace=False))
    else:
        positions = np.sort(np.concatenate([candidates, np.full(count - len(candidates), candidates[-1])]))
    claim_indices = rng.choice(num_claims, size=count, replace=False)
    return positions, claim_indices, fallback


def probe_text(prompt_text: str, response: str, position: int, claim: str, question: str) -> str:
    """The prompt, the response up to and including the cut character, and the question (``{claim}`` replaced)."""
    return prompt_text + response[: position + 1] + question.replace("{claim}", claim)


def parse_visual_claims(value: Any) -> list[dict[str, Any]]:
    """The claims of one row; fails on anything but a JSON list of {"claim": str, "correct": bool}."""
    claims = json.loads(value) if isinstance(value, str) else value
    if not isinstance(claims, list):
        raise ValueError(f"{VISUAL_CLAIMS_KEY} must be a JSON list, got {type(claims).__name__}")
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or not isinstance(claim.get("claim"), str)
            or not isinstance(claim.get("correct"), bool)
        ):
            raise ValueError(f'each visual claim must be {{"claim": str, "correct": bool}}, got {claim!r}')
    return claims


def claim_probe_answer_ids(
    tokenizer: Any, yes_words: Sequence[str], no_words: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Token ids of the yes and no candidates (the probe answer is the greedy token among them); fails unless each
    word is one token of ``tokenizer``."""

    def ids(words: Sequence[str]) -> list[int]:
        result = []
        for word in words:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"the claim probe answer {word!r} is not a single token: {encoded}")
            result.append(int(encoded[0]))
        return result

    yes_ids, no_ids = ids(yes_words), ids(no_words)
    if len(set(yes_ids + no_ids)) != len(yes_ids) + len(no_ids):
        raise ValueError(f"the claim probe answers map to repeated token ids: {yes_ids} {no_ids}")
    return yes_ids, no_ids


def perception_score(correct: np.ndarray, positions: np.ndarray, response_chars: int, beta: float) -> float:
    """R_perc of one response from its probes (see the module docstring)."""
    weights = np.exp(beta * (np.asarray(positions) / response_chars))
    weighted = float(np.sum(np.asarray(correct, dtype=np.float64) * weights) / np.sum(weights))
    return max((weighted - 0.5) / 0.5, 0.0)


def claim_probe_seed(data_seed: int, global_step: int, row: int) -> int:
    """Seed of one response's probes: fixed by the config seed, the step and the response's row in the step's
    batch, so it does not depend on which other responses are probed."""
    payload = f"claim_probe_v1:{data_seed}:{global_step}:{row}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little", signed=False)


def check_visual_claims(values: Sequence[Any], count: int) -> int:
    """Check the training column before training: every row has no claim or at least ``count``, and some row has
    claims. Returns the number of rows with claims."""
    sizes = [len(parse_visual_claims(value)) for value in values]
    short = [row for row, size in enumerate(sizes) if 0 < size < count]
    if short:
        raise ValueError(
            f"{len(short)} training rows (e.g. rows {short[:5]}) have fewer {VISUAL_CLAIMS_KEY} than "
            f"algorithm.claim_probe_count={count} (the fewest: {min(sizes[row] for row in short)}); lower the count"
        )
    with_claims = sum(size > 0 for size in sizes)
    if with_claims == 0:
        raise ValueError(f"no training row has {VISUAL_CLAIMS_KEY}; the claim probes would never run")
    return with_claims


def claim_probe_rows(batch: DataProto, accuracy: Sequence[float], count: int) -> list[int]:
    """Rows to probe: accuracy 1 and at least one claim. Fails on a row with 1 to count - 1 claims."""
    rows = []
    for row, (value, row_accuracy) in enumerate(zip(batch.non_tensor_batch[VISUAL_CLAIMS_KEY], accuracy)):
        num_claims = len(parse_visual_claims(value))
        if 0 < num_claims < count:
            raise ValueError(
                f"row {row} has {num_claims} visual claims, fewer than algorithm.claim_probe_count={count}"
            )
        if num_claims > 0 and float(row_accuracy) == 1.0:
            rows.append(row)
    return rows


def build_claim_probe_batch(batch: DataProto, rows: Sequence[int], data_seed: int, global_step: int) -> DataProto:
    """The probe requests of ``rows``: response tensors (for the long-response rule) plus the prompt ids, images,
    claims and per-row seed."""
    index = list(rows)
    probe_batch = DataProto.from_dict(
        tensors={key: batch.batch[key][index] for key in ("responses", "response_mask", "attention_mask")},
        non_tensors={
            "raw_prompt_ids": batch.non_tensor_batch["raw_prompt_ids"][index],
            VISUAL_CLAIMS_KEY: batch.non_tensor_batch[VISUAL_CLAIMS_KEY][index],
            "claim_probe_seed": np.array(
                [claim_probe_seed(data_seed, global_step, row) for row in index], dtype=np.uint64
            ),
        },
    )
    if "multi_modal_data" in batch.non_tensor_batch:
        probe_batch.non_tensor_batch["multi_modal_data"] = batch.non_tensor_batch["multi_modal_data"][index]
    return probe_batch


def attach_perception_scores(
    batch: DataProto, rows: Sequence[int], probe_output: Optional[DataProto], beta: float
) -> dict[str, float]:
    """Write R_perc of the probed rows into ``batch`` (NaN elsewhere) and return the claim_probe/ metrics."""
    scores = np.full(len(batch), np.nan, dtype=np.float64)
    metrics = {"claim_probe/probed_fraction": len(rows) / max(len(batch), 1)}
    if rows:
        if probe_output is None or len(probe_output) != len(rows):
            raise RuntimeError("the claim probe output does not align with the probed rows")
        out = probe_output.batch
        correct = out["claim_probe_correct"].numpy().astype(np.float64)
        positions = out["claim_probe_position"].numpy()
        chars = out["claim_probe_response_chars"].numpy()
        for i, row in enumerate(rows):
            scores[row] = perception_score(correct[i], positions[i], int(chars[i]), beta)
        late = positions / chars[:, None] >= 0.5
        metrics.update(
            {
                "claim_probe/score": float(np.mean(scores[list(rows)])),
                "claim_probe/accuracy": float(correct.mean()),
                "claim_probe/yes_rate": float(out["claim_probe_answer_yes"].float().mean()),
                "claim_probe/fallback_rate": float(out["claim_probe_fallback"].float().mean()),
                "claim_probe/max_prompt_tokens": float(out["claim_probe_max_tokens"].max()),
            }
        )
        if (~late).any():
            metrics["claim_probe/accuracy_early"] = float(correct[~late].mean())
        if late.any():
            metrics["claim_probe/accuracy_late"] = float(correct[late].mean())
    batch.non_tensor_batch[PERCEPTION_SCORE_KEY] = scores
    return metrics


def answer_claim_probes(
    probes: DataProto,
    *,
    tokenizer: Any,
    generate_fn: Any,
    process_image_fn: Any,
    count: int,
    question: str,
    yes_words: Sequence[str],
    no_words: Sequence[str],
    long_response_tokens: int,
    max_model_len: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> DataProto:
    """Run the probes of one worker's rows.

    ``generate_fn(inputs)`` takes vLLM prompt dicts and returns the first answer token id of each; the caller
    restricts it to the claim probe answer ids, greedy, one token. Returns per row: right/wrong, position and
    yes/no of each probe, the response's character count, whether no cut existed, and the longest probe in
    tokens (images expanded).
    """
    yes_ids, no_ids = claim_probe_answer_ids(tokenizer, yes_words, no_words)
    yes_set, no_set = set(yes_ids), set(no_ids)
    images = ImageCache(process_image_fn)
    responses, response_mask = probes.batch["responses"], probes.batch["response_mask"].bool()
    total_tokens = probes.batch["attention_mask"].sum(-1)
    prompt_tokens = total_tokens - response_mask.sum(-1)
    multi_modal = probes.non_tensor_batch.get("multi_modal_data")

    inputs, labels, positions, chars, fallbacks, max_tokens = [], [], [], [], [], []
    for i in range(len(probes)):
        response = tokenizer.decode(responses[i][response_mask[i]], skip_special_tokens=True)
        claims = parse_visual_claims(probes.non_tensor_batch[VISUAL_CLAIMS_KEY][i])
        rng = np.random.default_rng(int(probes.non_tensor_batch["claim_probe_seed"][i]))
        row_positions, claim_indices, fallback = plan_claim_probes(
            response, len(claims), count, bool(total_tokens[i] > long_response_tokens), rng
        )
        raw_prompt_ids = list(probes.non_tensor_batch["raw_prompt_ids"][i])
        prompt_text = tokenizer.decode(raw_prompt_ids, skip_special_tokens=False)
        row_multi_modal = None
        if multi_modal is not None and multi_modal[i] is not None:
            if multi_modal[i].get("videos"):
                raise NotImplementedError("claim probes support image inputs only")
            row_images = [images.get(image, min_pixels, max_pixels) for image in multi_modal[i].get("images", [])]
            row_multi_modal = {"image": row_images} if row_images else None
        row_max = 0
        for position, claim_index in zip(row_positions, claim_indices):
            claim = claims[int(claim_index)]
            ids = tokenizer.encode(
                probe_text(prompt_text, response, int(position), claim["claim"], question), add_special_tokens=False
            )
            # the raw prompt ids hold one placeholder per image; the engine expands it as in the rollout
            num_tokens = len(ids) - len(raw_prompt_ids) + int(prompt_tokens[i])
            if num_tokens > max_model_len:
                raise ValueError(
                    f"a claim probe has {num_tokens} tokens, more than the rollout engine's max_model_len "
                    f"{max_model_len}; raise worker.rollout.max_model_len"
                )
            row_max = max(row_max, num_tokens)
            request = {"prompt_token_ids": ids}
            if row_multi_modal is not None:
                request["multi_modal_data"] = row_multi_modal
            inputs.append(request)
            labels.append(claim["correct"])
        positions.append(row_positions)
        chars.append(len(response))
        fallbacks.append(fallback)
        max_tokens.append(row_max)

    answers = list(generate_fn(inputs)) if inputs else []
    if len(answers) != len(inputs):
        raise RuntimeError(f"expected {len(inputs)} claim probe answers, got {len(answers)}")
    for token_id in answers:
        if token_id not in yes_set and token_id not in no_set:
            raise ValueError(f"claim probe answer token {token_id} is not a yes/no candidate")
    answer_yes = np.array([token_id in yes_set for token_id in answers], dtype=bool).reshape(len(probes), count)
    correct = answer_yes == np.array(labels, dtype=bool).reshape(len(probes), count)
    return DataProto.from_dict(
        tensors={
            "claim_probe_correct": torch.from_numpy(correct),
            "claim_probe_answer_yes": torch.from_numpy(answer_yes),
            "claim_probe_position": torch.from_numpy(np.stack(positions).astype(np.int64)),
            "claim_probe_response_chars": torch.tensor(chars, dtype=torch.long),
            "claim_probe_fallback": torch.tensor(fallbacks, dtype=torch.bool),
            "claim_probe_max_tokens": torch.tensor(max_tokens, dtype=torch.long),
        },
    )
