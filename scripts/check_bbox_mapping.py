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
"""Check that the zoom-in tool reads a backbone's box coordinates correctly.

The model grounds every V* target object in its native format, on exactly the image the agent
loop feeds it (same ``min_pixels`` / ``max_pixels``), and the tool maps the box to the source image
as it would during DeepEyes training. The mapped boxes are compared with the V* annotations, next
to two wrong readings of the same coordinates. Run it when adding a backbone:

    bash scripts/prepare_eval_data.sh vstar
    python scripts/check_bbox_mapping.py --model Qwen/Qwen2.5-VL-7B-Instruct            # auto -> pixel
    python scripts/check_bbox_mapping.py --model Qwen/Qwen3-VL-8B-Instruct              # auto -> norm1000

A correct convention gives a mean IoU far above both alternatives.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUNDING_PROMPT = "Locate the {name} in the image, output its bbox coordinates using JSON format."


def _iou(box, gt):
    iw = max(0.0, min(box[2], gt[2]) - max(box[0], gt[0]))
    ih = max(0.0, min(box[3], gt[3]) - max(box[1], gt[1]))
    inter = iw * ih
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    gt_area = max(1e-6, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    return inter / (area + gt_area - inter)


def _parse_box(text):
    match = re.search(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text
    )
    return None if match is None else [float(value) for value in match.groups()]


def _readings(box, bbox_format, frame, source, prepared):
    """The tool's mapping and two wrong readings of the same coordinates, in source pixels."""
    from verl.workers.agent.tools.image_zoom import NORMALIZED_COORDINATE_MAX, ImageZoomInTool

    (frame_w, frame_h), (source_w, source_h), (prepared_w, prepared_h) = frame, source, prepared
    if bbox_format == "pixel":
        clipped = [min(max(value, 0.0), float(limit)) for value, limit in zip(box, (frame_w, frame_h) * 2)]
        return {
            "tool": ImageZoomInTool._frame_to_source_bbox(clipped, frame, source_w, source_h),
            "as source pixels": box,
            "data-side resize only": [
                box[0] * source_w / prepared_w,
                box[1] * source_h / prepared_h,
                box[2] * source_w / prepared_w,
                box[3] * source_h / prepared_h,
            ],
        }
    normalized = [min(max(round(value), 0), NORMALIZED_COORDINATE_MAX) for value in box]
    return {
        "tool": ImageZoomInTool._to_pixel_bbox(normalized, source_w, source_h),
        "as source pixels": box,
        "as model-frame pixels": [
            box[0] * source_w / frame_w,
            box[1] * source_h / frame_h,
            box[2] * source_w / frame_w,
            box[3] * source_h / frame_h,
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bbox-format", default="auto", choices=["auto", "norm1000", "pixel"])
    parser.add_argument("--min-pixels", type=int, default=200704, help="data.min_pixels of the training script")
    parser.add_argument("--max-pixels", type=int, default=1003520, help="data.max_pixels of the training script")
    parser.add_argument("--data-dir", default=None, help="prepared V* directory (default: <eval data root>/vstar)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None, help="optional JSON file with per-object records")
    args = parser.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams

    from verl.utils.tokenizer import get_processor
    from verl.workers.agent.backends.vllm import VLLMAgentImageCache
    from verl.workers.agent.coordinates import model_input_image_size, resolve_bbox_format

    eval_root = os.environ.get("EVAL_DATA_ROOT") or os.path.join(
        os.environ.get("DATA_ROOT", str(ROOT / "data")), "eval"
    )
    data_dir = Path(args.data_dir or os.path.join(eval_root, "vstar"))
    questions = data_dir / "test_questions.jsonl"
    if not questions.is_file():
        sys.exit(f"{questions} not found; run 'bash scripts/prepare_eval_data.sh vstar' first")
    bbox_format = resolve_bbox_format(args.bbox_format, args.model)
    rows = [json.loads(line) for line in questions.open()][: args.limit]

    processor = get_processor(args.model, use_fast=True)
    cache = VLLMAgentImageCache(min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    requests, objects = [], []
    for row in rows:
        annotation = json.loads((data_dir / row["image"]).with_suffix(".json").read_text())
        source = Image.open(data_dir / row["image"]).convert("RGB")
        prepared = cache.prepare(source)
        frame = model_input_image_size(processor, prepared)
        for name, (x, y, w, h) in zip(annotation["target_object"], annotation["bbox"]):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": GROUNDING_PROMPT.format(name=name)}],
                }
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": prepared}})
            objects.append(
                {
                    "image": row["image"],
                    "object": name,
                    "gt": [x, y, x + w, y + h],
                    "source": source.size,
                    "prepared": prepared.size,
                    "frame": frame,
                }
            )

    llm = LLM(model=args.model, limit_mm_per_prompt={"image": 1}, max_model_len=8192, seed=0)
    outputs = llm.generate(requests, SamplingParams(temperature=0.0, max_tokens=128))
    scores: dict[str, list[float]] = {}
    for item, output in zip(objects, outputs):
        item["text"] = output.outputs[0].text
        box = _parse_box(item["text"])
        if box is None:
            continue
        item["readings"] = _readings(box, bbox_format, item["frame"], item["source"], item["prepared"])
        for reading, mapped in item["readings"].items():
            scores.setdefault(reading, []).append(_iou(mapped, item["gt"]))

    parsed = len(next(iter(scores.values()), []))
    print(f"model: {args.model}\nbbox format: {bbox_format}\nobjects: {len(objects)}, parsed boxes: {parsed}")
    print(f"{'reading':<24}{'mean IoU':>10}{'IoU>=0.5':>10}")
    for reading, values in scores.items():
        print(f"{reading:<24}{sum(values) / len(values):>10.3f}{sum(v >= 0.5 for v in values) / len(values):>10.3f}")
    if args.output:
        Path(args.output).write_text(json.dumps(objects, indent=1, default=list))


if __name__ == "__main__":
    main()
