#!/usr/bin/env python3
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
"""Check that ms-swift trains on SFT data from scripts/agent/export_sft.py with the RL rollout's tokens.

Run it in an environment with ms-swift (it does not import this repository):

    python scripts/agent/check_swift_export.py --model Qwen/Qwen3-VL-4B-Instruct --export data/agent_env/toy_search_sft

Each sample of train.jsonl is encoded by ms-swift's own template in training mode (the encoding ``swift sft``
uses) and compared with the rollout's record in rollout_tokens.jsonl:

- token ids, with each image placeholder expanded to its visual tokens;
- trained tokens: exactly the model's actions (rollout loss mask 1). The one difference allowed is ms-swift's own:
  it also trains the rest of its template's turn suffix after the model's end-of-turn token (for Qwen, the newline
  after ``<|im_end|>``), which the model never generates because generation stops at ``<|im_end|>``;
- each image: its pixels against those exported, its patch grid, and its pixel values against the model's own
  (fast) image processor.

Exit code 0 when every sample matches. Tested with ms-swift 4.x (``swift.template``) and 3.x (``swift.llm``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def load_swift(model: str, agent_template: str | None) -> tuple[Any, Any, str]:
    """ms-swift's processor and training template for ``model``, and the version."""
    import swift

    version = getattr(swift, "__version__", "unknown")
    kwargs = {"agent_template": agent_template} if agent_template else {}
    try:  # ms-swift 4
        from swift.model import get_processor
        from swift.template import get_template

        processor = get_processor(model)
        template = get_template(processor, **kwargs)
    except ImportError:  # ms-swift 3
        from swift.llm import get_model_tokenizer, get_template

        _, processor = get_model_tokenizer(model, load_model=False)
        template = get_template(processor.model_meta.template, processor, **kwargs)
    template.set_mode("train")
    return processor, template, version


def suffix_after_termination(template: Any, tokenizer: Any, termination: list[int]) -> list[int]:
    """The tokens ms-swift trains after the model's end-of-turn tokens (its turn suffix minus the overlap)."""
    suffix = "".join(part for part in template.template_meta.suffix if isinstance(part, str))
    ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    return ids[len(termination) :] if termination and ids[: len(termination)] == termination else []


def expected_tokens(
    rollout: dict[str, Any], merge_size: int, extra: list[int]
) -> tuple[list[int], list[int], set[int]]:
    """The rollout's ids and loss mask with images expanded, and the positions where ms-swift may also train."""
    image_token = rollout["image_token_id"]
    grids = iter(rollout["image_grid_thw"])
    termination = rollout["assistant_termination_token_ids"]
    ids: list[int] = []
    mask: list[int] = []
    for token, trained in zip(rollout["input_ids"], rollout["loss_mask"]):
        if token == image_token:
            t, h, w = next(grids)
            count = t * h * w // (merge_size**2)
            ids += [image_token] * count
            mask += [trained] * count
        else:
            ids.append(token)
            mask.append(trained)
    allowed: set[int] = set()
    if extra:
        width = len(termination)
        for end in range(width, len(ids) + 1):
            if ids[end - width : end] == termination and all(mask[end - width : end]):
                for offset in range(len(extra)):
                    if end + offset < len(ids) and ids[end + offset] == extra[offset] and not mask[end + offset]:
                        allowed.add(end + offset)
    return ids, mask, allowed


def compare(sample: dict[str, Any], rollout: dict[str, Any], template: Any, hf_image_processor: Any) -> str | None:
    """None when ms-swift's encoding of ``sample`` matches the rollout, else what differs."""
    import torch
    from PIL import Image

    tokenizer = template.tokenizer
    merge_size = int(hf_image_processor.merge_size)
    extra = suffix_after_termination(template, tokenizer, rollout["assistant_termination_token_ids"])
    ids, mask, allowed = expected_tokens(rollout, merge_size, extra)
    encoded = template.encode({key: sample[key] for key in ("messages", "images", "tools") if key in sample})
    swift_ids = list(encoded["input_ids"])
    swift_mask = [int(label != -100) for label in encoded["labels"]]
    if len(swift_ids) == len(ids) + len(extra) and extra and swift_ids[len(ids) :] == extra:
        # ms-swift ends the sample with its full turn suffix
        allowed |= set(range(len(ids), len(swift_ids)))
        ids = ids + extra
        mask = mask + [0] * len(extra)
    if swift_ids != ids:
        position = next((i for i, (a, b) in enumerate(zip(swift_ids, ids)) if a != b), min(len(swift_ids), len(ids)))
        return (
            f"token ids differ at {position} (ms-swift {len(swift_ids)}, rollout {len(ids)}):\n"
            f"  ms-swift: {tokenizer.decode(swift_ids[max(0, position - 20) : position + 20])!r}\n"
            f"  rollout:  {tokenizer.decode(ids[max(0, position - 20) : position + 20])!r}"
        )
    for position, (got, want) in enumerate(zip(swift_mask, mask)):
        if got != want and not (got == 1 and position in allowed):
            return (
                f"trained tokens differ at {position} ({tokenizer.decode([ids[position]])!r}: ms-swift "
                f"{'trains' if got else 'does not train'} it, the rollout {'does' if want else 'does not'})"
            )
    grids = encoded.get("image_grid_thw")
    grids = [] if grids is None else grids.tolist()
    if grids != rollout["image_grid_thw"]:
        return f"image grids differ: ms-swift {grids}, rollout {rollout['image_grid_thw']}"
    if sample["images"]:
        images = [Image.open(path).convert("RGB") for path in sample["images"]]
        for position, (image, digest) in enumerate(zip(images, rollout.get("image_sha256", []))):
            if hashlib.sha256(image.tobytes()).hexdigest() != digest:
                return f"image {position} ({sample['images'][position]}) is not the image that was exported"
        reference = hf_image_processor(images=images, return_tensors="pt")["pixel_values"]
        pixels = encoded["pixel_values"]
        if reference.shape != pixels.shape or not torch.equal(reference.to(pixels.dtype), pixels):
            difference = (
                (reference.float() - pixels.float()).abs().max().item() if reference.shape == pixels.shape else "shape"
            )
            return f"pixel values differ from the model's image processor (max abs difference {difference})"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the model given to swift sft")
    parser.add_argument("--export", required=True, type=Path, help="the --out directory of export_sft.py")
    parser.add_argument("--agent-template", help="as swift sft --agent_template (default: the template's own)")
    parser.add_argument("--limit", type=int, help="check the first N samples only")
    args = parser.parse_args(argv)

    from transformers import AutoProcessor

    processor, template, version = load_swift(args.model, args.agent_template)
    model_dir = getattr(getattr(processor, "model_info", None), "model_dir", None) or args.model
    hf_image_processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=True
    ).image_processor
    with (args.export / "train.jsonl").open(encoding="utf-8") as file:
        samples = [json.loads(line) for line in file if line.strip()]
    with (args.export / "rollout_tokens.jsonl").open(encoding="utf-8") as file:
        rollouts = [json.loads(line) for line in file if line.strip()]
    if len(samples) != len(rollouts):
        print(f"[check] {len(samples)} samples but {len(rollouts)} rollout records", flush=True)
        return 1
    pairs = list(zip(samples, rollouts))[: args.limit] if args.limit is not None else list(zip(samples, rollouts))
    failures = 0
    for sample, rollout in pairs:
        problem = compare(sample, rollout, template, hf_image_processor)
        if problem is not None:
            failures += 1
            if failures <= 5:
                print(f"[check] sample {rollout['id']}: {problem}", flush=True)
    print(
        f"[check] ms-swift {version}, template {template.template_meta.template_type}: "
        f"{len(pairs) - failures} of {len(pairs)} samples match the rollout tokens",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
