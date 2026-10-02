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
"""Benchmark loaders: turn the files prepared under the data root into ``EvalSample`` lists.

Every loader returns the *bare* question as ``EvalSample.prompt``; the runner then applies
the training-time ``--format-prompt`` template and ``--system-prompt`` (see prompting.py),
so models are evaluated with the same instruction they were trained with.
"""

from __future__ import annotations

import ast
import glob
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image

from .json_utils import read_jsonl
from .schemas import BenchmarkSpec, EvalSample


Loader = Callable[[BenchmarkSpec, Path, int | None], list[EvalSample]]

# Instruction suffix that PAPO-Eval / VPPO-Eval bake into every question. It is removed so
# that the runner's --format-prompt (examples/format_prompt/math_perception.jinja by
# default, i.e. the same sentence) is the single source of the output-format instruction.
PAPO_INSTRUCTION_RE = re.compile(
    r"\s*You first think through the reasoning process as an internal monologue, enclosed within "
    r"<think> </think> tags\.\s*Then, provide your final answer enclosed within \\boxed\{\}\.?\s*$"
)
YES_NO_SUFFIX = " Please answer yes or no."
MCQ_LETTER_SUFFIX = "Answer with the option's letter from the given choices."


def load_samples(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    loaders: dict[str, Loader] = {
        "sharegpt": load_sharegpt,
        "easy_parquet": load_easy_parquet,
        "pope": load_pope,
        "hallusionbench": load_hallusionbench,
        "gqa": load_gqa,
        "seed_bench": load_seed_bench,
        "mme": load_mme,
        "mmvet": load_mmvet,
        "vstar": load_vstar,
        "hrbench": load_hrbench,
        "mme_realworld": load_mme_realworld,
        "cfpo_json": load_cfpo_json,
        "grit_jsonl": load_grit_jsonl,
        "refcoco": load_refcoco,
    }
    try:
        loader = loaders[spec.loader]
    except KeyError as exc:
        raise KeyError(f"unknown loader for {spec.key}: {spec.loader}") from exc
    samples = loader(spec, data_root, limit)
    if not samples:
        raise RuntimeError(f"{spec.key} produced no samples")
    return samples


def missing_data_files(spec: BenchmarkSpec, data_root: Path) -> list[Path]:
    """Data files the registry entry needs but that do not exist (cheap pre-flight check)."""
    missing = []
    for path in spec.data_paths(data_root):
        if glob.has_magic(str(path)):
            if not glob.glob(str(path)):
                missing.append(path)
        elif not path.exists():
            missing.append(path)
    image_root = spec.resolved_image_root(data_root)
    if image_root is not None and not image_root.exists():
        missing.append(image_root)
    return missing


_NORM1000_INSTRUCTION_RE = re.compile(r"(?:integer )?coordinates normalized to 0-1000")


def apply_box_format_to_prompts(samples: list[EvalSample], box_format: str) -> list[EvalSample]:
    """Ask for absolute pixel coordinates instead of 0-1000 ones when boxes are read as pixels.

    The grounding loaders (GRIT, OVDEval, RefCOCO) request ``coordinates normalized to 0-1000``;
    with ``--box-format pixel`` the instruction is rewritten so that it matches how the boxes are
    interpreted. Prompts without that phrase are returned unchanged.
    """
    if box_format != "pixel":
        return samples
    return [
        replace(sample, prompt=_NORM1000_INSTRUCTION_RE.sub("absolute pixel coordinates", sample.prompt))
        if _NORM1000_INSTRUCTION_RE.search(sample.prompt)
        else sample
        for sample in samples
    ]


def shard_samples(
    samples: list[EvalSample], shard_index: int, num_shards: int, limit: int | None = None
) -> list[EvalSample]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    selected = [sample for index, sample in enumerate(samples) if index % num_shards == shard_index]
    if limit is not None:
        selected = selected[:limit]
    return selected


# ---------------------------------------------------------------------------
# Generic ShareGPT / LLaMA-Factory rows (PAPO-Eval, VPPO-Eval)
# ---------------------------------------------------------------------------


def sharegpt_image_relpath(value: str) -> str:
    """``./data/images/<folder>/<file>`` (LLaMA-Factory layout) -> ``images/<folder>/<file>``."""
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("data/images/"):
        text = text[len("data/") :]
    return text


def strip_sharegpt_question(content: str, image_count: int) -> str:
    """Bare question from a ShareGPT user turn.

    Removes the PAPO-Eval output-format instruction and the leading ``<image>`` placeholders.
    Leading placeholders carry no position information (prompting.py puts images before the
    text when the prompt has no placeholder); placeholders in the middle of the text are kept
    so that interleaved multi-image prompts keep their layout.
    """
    text = PAPO_INSTRUCTION_RE.sub("", str(content))
    stripped = text.lstrip()
    leading = 0
    while stripped.startswith("<image>") and leading < max(image_count, 1):
        stripped = stripped[len("<image>") :].lstrip()
        leading += 1
    if leading and "<image>" not in stripped:
        text = stripped
    return text.strip()


def _sharegpt_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of ShareGPT rows")
    return payload


def _turns(messages: Any) -> list[dict[str, Any]]:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return [dict(item) for item in messages or []]


def load_sharegpt(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """ShareGPT-style rows: ``messages`` (user question + assistant answer) and ``images``.

    Image paths are resolved against ``image_root`` (default: the data file's directory);
    the LLaMA-Factory prefix ``./data/images/`` maps to ``images/``.
    """
    samples: list[EvalSample] = []
    for path in _expand_paths(spec, data_root):
        image_root = spec.resolved_image_root(data_root) or path.parent
        for index, row in enumerate(_sharegpt_rows(path)):
            if limit is not None and len(samples) >= limit:
                return samples
            messages = row.get("messages")
            turns = _turns(messages if messages is not None else row.get("conversations"))
            user = next((turn for turn in turns if turn.get("role", turn.get("from")) in {"user", "human"}), None)
            answer = next((turn for turn in turns if turn.get("role", turn.get("from")) in {"assistant", "gpt"}), None)
            if user is None or answer is None:
                raise ValueError(f"{path}:{index} needs a user and an assistant turn")
            raw_images = row.get("images")
            if hasattr(raw_images, "tolist"):
                raw_images = raw_images.tolist()
            if isinstance(raw_images, str):
                raw_images = [raw_images]
            images = [_required_image(image_root, sharegpt_image_relpath(item)) for item in raw_images or []]
            question = strip_sharegpt_question(str(user.get("content", user.get("value", ""))), len(images))
            row_id = row.get("id")
            samples.append(
                EvalSample(
                    benchmark=spec.key,
                    sample_id=str(row_id) if row_id not in (None, "") else f"{spec.key}:{index}",
                    prompt=question,
                    target=str(answer.get("content", answer.get("value", ""))).strip(),
                    images=images,
                    metadata={"row_index": index, "question": question, "source_file": path.name},
                )
            )
    return samples


def load_easy_parquet(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """EasyR1 training-format parquet (``problem``/``answer``/``images``), e.g. data/geometry3k."""
    paths = _require_paths(spec, data_root)
    image_root = spec.resolved_image_root(data_root)
    rows = _read_parquets(paths)
    samples = []
    for index, row in rows.iterrows():
        if limit is not None and len(samples) >= limit:
            break
        prompt = str(row.get("problem") or row.get("question") or "")
        images = _resolve_image_list(row.get("images"), image_root)
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=str(row.get("id") or f"{spec.key}:{index}"),
                prompt=prompt,
                target=row.get("answer"),
                images=images,
                extra_info=_mapping(row.get("extra_info")),
                metadata={"row_index": int(index), "question": prompt},
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Perception (lmms-eval parquet files with embedded images)
# ---------------------------------------------------------------------------


def load_pope(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    samples = []
    for path in _expand_paths(spec, data_root):
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            if limit is not None and len(samples) >= limit:
                return samples
            question = str(row["question"]).strip()
            samples.append(
                EvalSample(
                    benchmark=spec.key,
                    sample_id=f"{row.get('category', path.stem)}:{row.get('question_id', index)}",
                    prompt=question + YES_NO_SUFFIX,
                    target=str(row["answer"]),
                    images=[_decode_image(row.get("image"))],
                    metadata={
                        "category": str(row.get("category") or path.stem),
                        "row_index": int(index),
                        "question": question,
                    },
                )
            )
    return samples


def load_hallusionbench(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """HallusionBench (lmms-lab ``image`` split): yes/no questions, gt_answer "1"=yes / "0"=no."""
    samples = []
    for path in _expand_paths(spec, data_root):
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            if limit is not None and len(samples) >= limit:
                return samples
            question = str(row["question"]).strip()
            gt = str(row["gt_answer"]).strip()
            if gt not in {"0", "1"}:
                raise ValueError(f"{spec.key}:{index} has unexpected gt_answer {gt!r}")
            image_value = row.get("image")
            images = [] if _is_missing(image_value) else [_decode_image(image_value)]
            ids = {
                name: str(row.get(name)) for name in ("category", "subcategory", "set_id", "figure_id", "question_id")
            }
            samples.append(
                EvalSample(
                    benchmark=spec.key,
                    sample_id="_".join(
                        ids[name] for name in ("category", "subcategory", "set_id", "figure_id", "question_id")
                    ),
                    prompt=question + YES_NO_SUFFIX,
                    target="yes" if gt == "1" else "no",
                    images=images,
                    metadata={
                        **ids,
                        "visual_input": str(row.get("visual_input")),
                        "row_index": int(index),
                        "question": question,
                    },
                )
            )
    return samples


def load_gqa(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    paths = _require_paths(spec, data_root)
    instructions = next((path for path in paths if "instructions" in path.name), None)
    image_path = next((path for path in paths if "images" in path.name), None)
    if instructions is None or image_path is None:
        raise FileNotFoundError(f"{spec.key} requires both instructions and images parquet files")
    images = {str(row["id"]): _decode_image(row["image"]) for _, row in pd.read_parquet(image_path).iterrows()}
    df = pd.read_parquet(instructions)
    samples = []
    for index, row in df.iterrows():
        if limit is not None and len(samples) >= limit:
            break
        image_id = str(row["imageId"])
        if image_id not in images:
            raise KeyError(f"GQA image id not found: {image_id}")
        question = str(row["question"]).strip()
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=str(row.get("id") or f"{spec.key}:{index}"),
                prompt=f"{question}\nAnswer the question using a single word or phrase.",
                target=str(row["answer"]),
                images=[images[image_id]],
                metadata={"image_id": image_id, "row_index": int(index), "question": question},
            )
        )
    return samples


def load_seed_bench(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """SEED-Bench annotations written by the prepare step (images converted to files).

    Image questions carry one image; video questions carry the 8 frames of the lmms-lab release,
    which are passed to the model as a multi-image prompt.
    """
    path = _require_paths(spec, data_root)[0]
    image_root = spec.resolved_image_root(data_root) or path.parent
    samples = []
    for index, row in enumerate(read_jsonl(path)):
        if limit is not None and len(samples) >= limit:
            break
        options = [str(row[f"choice_{letter}"]) for letter in "abcd"]
        question = str(row["question"]).strip()
        prompt = (
            f"{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n"
            "Answer with the option letter from the given choices directly."
        )
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=str(row.get("question_id") or f"{spec.key}:{index}"),
                prompt=prompt,
                target=str(row["answer"]).strip().upper(),
                images=[_required_image(image_root, str(image)) for image in row.get("images") or []],
                extra_info={"options": options},
                metadata={
                    "data_type": str(row.get("data_type")),
                    "question_type_id": row.get("question_type_id"),
                    "row_index": index,
                    "question": question,
                },
            )
        )
    return samples


def load_mme(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    samples = []
    for path in _expand_paths(spec, data_root):
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            if limit is not None and len(samples) >= limit:
                return samples
            question = str(row["question"]).strip()
            samples.append(
                EvalSample(
                    benchmark=spec.key,
                    sample_id=f"{row['category']}:{row['question_id']}",
                    prompt=question,
                    target=str(row["answer"]),
                    images=[_decode_image(row["image"])],
                    metadata={
                        "category": str(row["category"]),
                        "question_id": str(row["question_id"]),
                        "row_index": int(index),
                        "question": question,
                    },
                )
            )
    return samples


def load_mmvet(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    samples = []
    for path in _require_paths(spec, data_root):
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            if limit is not None and len(samples) >= limit:
                return samples
            question = str(row["question"])
            samples.append(
                EvalSample(
                    benchmark=spec.key,
                    sample_id=str(row.get("id") or f"{spec.key}:{index}"),
                    prompt=question,
                    target=str(row["answer"]),
                    images=[_decode_image(row["image"])] if row.get("image") is not None else [],
                    metadata={
                        "capability": _stringify_capability(row.get("capability")),
                        "row_index": int(index),
                        "question": question,
                    },
                )
            )
    return samples


# ---------------------------------------------------------------------------
# High-resolution MCQ (DeepEyes tables)
# ---------------------------------------------------------------------------

_VSTAR_OPTION_RE = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def _mcq_prompt(question: str, options: list[str]) -> str:
    lines = [question.strip()]
    lines.extend(f"({chr(ord('A') + index)}) {option}" for index, option in enumerate(options))
    lines.append(MCQ_LETTER_SUFFIX)
    return "\n".join(lines)


def load_vstar(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """V* Bench ``test_questions.jsonl``: official prompts with shuffled options + gold letter."""
    path = _require_paths(spec, data_root)[0]
    image_root = spec.resolved_image_root(data_root) or path.parent
    samples = []
    for index, row in enumerate(read_jsonl(path)):
        if limit is not None and len(samples) >= limit:
            break
        question_lines: list[str] = []
        options: list[str] = []
        for line in str(row["text"]).splitlines():
            match = _VSTAR_OPTION_RE.match(line.strip())
            if match:
                options.append(match.group(2).strip())
            elif line.strip().lower().startswith("answer with the option"):
                continue
            elif not options:
                question_lines.append(line)
        question = "\n".join(question_lines).strip()
        if len(options) < 2:
            raise ValueError(f"{path}:{index + 1} has fewer than two options")
        label = str(row["label"]).strip().upper()
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=f"{row.get('category')}:{row.get('question_id', index)}",
                prompt=_mcq_prompt(question, options),
                target=label,
                images=[_required_image(image_root, str(row["image"]))],
                extra_info={"options": options},
                metadata={"category": str(row.get("category")), "row_index": index, "question": question},
            )
        )
    return samples


def load_hrbench(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """HR-Bench annotations written by the prepare step (images decoded to files)."""
    path = _require_paths(spec, data_root)[0]
    image_root = spec.resolved_image_root(data_root) or path.parent
    samples = []
    for index, row in enumerate(read_jsonl(path)):
        if limit is not None and len(samples) >= limit:
            break
        options = [str(row[letter]) for letter in "ABCD" if row.get(letter) not in (None, "")]
        question = str(row["question"]).strip()
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=str(row.get("index", index)),
                prompt=_mcq_prompt(question, options),
                target=str(row["answer"]).strip().upper(),
                images=[_required_image(image_root, str(row["image"]))],
                extra_info={"options": options},
                metadata={
                    "category": str(row.get("category")),
                    "cycle_category": str(row.get("cycle_category")),
                    "row_index": index,
                    "question": question,
                },
            )
        )
    return samples


MME_REALWORLD_TASKS = {
    "OCR with Complex Context": "OCR",
    "Remote Sensing": "RS",
    "Diagram and Table": "DT",
    "Monitoring": "MO",
    "Autonomous_Driving": "AD",
}
_OPTION_PREFIX_RE = re.compile(r"^\s*\(?([A-Z])\)\s*")


def mme_realworld_prompt(question: str, options: list[str]) -> str:
    """lmms-eval's MME-RealWorld prompt without the trailing completion cue "The best answer is:"."""
    return (
        f"{question.strip()} The choices are listed below:\n"
        + "\n".join(options)
        + "\nSelect the best answer to the above multiple-choice question based on the image. "
        "Respond with only the letter (A, B, C, D, or E) of the correct option."
    )


def _mme_realworld_option_texts(options: list[str]) -> list[str]:
    """Option texts indexed by their printed letter ("(A) ..." -> position 0).

    Two Autonomous_Driving rows print "(D)" twice; like lmms-eval, which reads the letter out of
    the option string, the first "(D)" keeps position D and "(E)" stays E.
    """
    by_letter: dict[str, str] = {}
    for position, option in enumerate(options):
        match = _OPTION_PREFIX_RE.match(option)
        letter = match.group(1) if match else chr(ord("A") + position)
        by_letter.setdefault(letter, _OPTION_PREFIX_RE.sub("", option))
    return [by_letter.get(chr(ord("A") + index), "") for index in range(len(by_letter))]


def load_mme_realworld(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """MME-RealWorld-Lite annotations written by the prepare step (A-E MCQ, one image)."""
    path = _require_paths(spec, data_root)[0]
    image_root = spec.resolved_image_root(data_root) or path.parent
    samples = []
    for index, row in enumerate(read_jsonl(path)):
        if limit is not None and len(samples) >= limit:
            break
        options = [str(option).strip() for option in row["options"]]
        task, _, subtask = str(row["category"]).partition("/")
        short = MME_REALWORLD_TASKS.get(subtask, subtask)
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=str(row["index"]),
                prompt=mme_realworld_prompt(str(row["question"]), options),
                target=str(row["answer"]).strip().upper(),
                images=[_required_image(image_root, str(row["image"]))],
                extra_info={"options": _mme_realworld_option_texts(options)},
                metadata={
                    "task": task,
                    "category": f"{task}/{short}",
                    "l2_category": row.get("l2_category"),
                    "row_index": index,
                    "question": str(row["question"]).strip(),
                },
            )
        )
    return samples


def load_cfpo_json(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """CFPO_Datasets rows ``{images, problem, answer, id, type, is_cf}``.

    ``answer`` is a string, or the list of the 10 human answers for TextVQA.
    """
    path = _require_paths(spec, data_root)[0]
    image_root = spec.resolved_image_root(data_root) or path.parent
    rows = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for index, row in enumerate(rows):
        if limit is not None and len(samples) >= limit:
            break
        images = [_required_image(image_root, str(name)) for name in row.get("images") or []]
        question = strip_sharegpt_question(str(row["problem"]), len(images))
        answer = row["answer"]
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=str(row.get("id") or f"{spec.key}:{index}"),
                prompt=question,
                target=[str(item) for item in answer] if isinstance(answer, list) else str(answer),
                images=images,
                metadata={
                    "type": str(row.get("type") or ""),
                    "is_cf": bool(row.get("is_cf", False)),
                    "row_index": index,
                    "question": question,
                },
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def load_grit_jsonl(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    path = _require_paths(spec, data_root)[0]
    image_root = spec.resolved_image_root(data_root)
    if image_root is None:
        raise ValueError(f"{spec.key} requires image_root")
    samples = []
    for index, row in enumerate(read_jsonl(path)):
        if limit is not None and len(samples) >= limit:
            break
        try:
            width, height = float(row["width"]), float(row["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{index + 1} must contain numeric width and height") from exc
        if not math.isfinite(width) or width <= 0.0 or not math.isfinite(height) or height <= 0.0:
            raise ValueError(f"{path}:{index + 1} has invalid image dimensions: width={width}, height={height}")
        boxes = _normalize_boxes(row.get("bboxs") or [], width=width, height=height, scale="pixel")
        base_prompt = str(row["question"]).strip()
        prompt = base_prompt
        native_agentic_prompt = None
        if spec.scorer == "answer_bbox":
            prompt += (
                "\nProvide a concise final answer and ground every image region needed to justify it "
                "with bounding boxes using coordinates normalized to 0-1000."
            )
            native_agentic_prompt = base_prompt
        elif spec.scorer == "grounding_iou":
            prompt += "\nReturn only the bounding box using coordinates normalized to 0-1000."
        target = row.get("answer")
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=f"{spec.key}:{index}",
                prompt=prompt,
                target=target if isinstance(target, (list, dict)) else str(target),
                images=[_required_image(image_root, str(row["image"]))],
                native_agentic_prompt=native_agentic_prompt,
                extra_info={
                    "bboxs": row.get("bboxs") or [],
                    "bboxs_normalized": boxes,
                    "bbox_format": "pixel_xyxy",
                    "width": width,
                    "height": height,
                },
                metadata={
                    "dataset": row.get("dataset"),
                    "split": row.get("split"),
                    "row_index": index,
                    "question": base_prompt,
                },
            )
        )
    return samples


def refcoco_prompt(expression: str) -> str:
    return (
        f'Locate the object described by: "{expression}". '
        "Return its bounding box as [x1, y1, x2, y2] using integer coordinates normalized to 0-1000."
    )


def load_refcoco(spec: BenchmarkSpec, data_root: Path, limit: int | None = None) -> list[EvalSample]:
    """RefCOCO-family rows prepared from PaDT-MLLM/RefCOCO (one expression, one normalized box)."""
    path = _require_paths(spec, data_root)[0]
    image_root = spec.resolved_image_root(data_root) or path.parent
    samples = []
    for index, row in enumerate(read_jsonl(path)):
        if limit is not None and len(samples) >= limit:
            break
        expression = str(row["expression"]).strip()
        bbox = [float(value) for value in row["bbox"]]
        samples.append(
            EvalSample(
                benchmark=spec.key,
                sample_id=str(row.get("sample_id") or f"{spec.key}:{index}"),
                prompt=refcoco_prompt(expression),
                target=expression,
                images=[_required_image(image_root, str(row["image"]))],
                extra_info={"bbox": bbox, "bbox_format": "normalized_xyxy", "expression": expression},
                metadata={"row_index": index, "question": expression, "coco_image_id": row.get("coco_image_id")},
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_paths(spec: BenchmarkSpec, data_root: Path) -> list[Path]:
    paths = spec.data_paths(data_root)
    if not paths:
        raise ValueError(f"{spec.key} has no data path configured")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{spec.key} missing data file(s): {missing}; run `bash scripts/prepare_eval_data.sh {spec.key}`"
        )
    return paths


def _expand_paths(spec: BenchmarkSpec, data_root: Path) -> list[Path]:
    expanded: list[Path] = []
    for path in spec.data_paths(data_root):
        matches = sorted(Path(item) for item in glob.glob(str(path)))
        expanded.extend(matches or [path])
    missing = [path for path in expanded if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{spec.key} missing data file(s): {missing}; run `bash scripts/prepare_eval_data.sh {spec.key}`"
        )
    return expanded


def _read_parquets(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if path.suffix == ".parquet"]
    if not frames:
        raise ValueError("no parquet files found")
    return pd.concat(frames, ignore_index=True)


def _required_image(image_root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = image_root / path
    if not path.is_file() and not Path(value).is_absolute():
        split_candidates = [image_root / split / value for split in ("train2017", "val2017")]
        path = next((candidate for candidate in split_candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark image: {path}")
    return str(path)


def _normalize_boxes(values: Any, *, width: float, height: float, scale: str) -> list[list[float]]:
    boxes = []
    for value in values:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        x1, y1, x2, y2 = (float(item) for item in value)
        if scale == "pixel":
            normalized = [x1 / width, y1 / height, x2 / width, y2 / height]
        elif scale == "1000":
            normalized = [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
        else:
            normalized = [x1, y1, x2, y2]
        if 0.0 <= normalized[0] < normalized[2] <= 1.0 and 0.0 <= normalized[1] < normalized[3] <= 1.0:
            boxes.append(normalized)
    return boxes


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _resolve_image_list(value: Any, image_root: Path | None) -> list[Any]:
    if value is None:
        return []
    items = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(items, str):
        items = [items]
    images = []
    for item in items:
        if image_root is not None and isinstance(item, str):
            images.append(str(image_root / item))
        else:
            images.append(_decode_image(item))
    return images


def _decode_image(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if len(value) == 1:
            return _decode_image(value[0])
        raise ValueError("image list must be decoded by the caller")
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value
        if value.get("path") is not None:
            return value
    if isinstance(value, bytes):
        return value
    if isinstance(value, str) and value and value != "None":
        path = Path(value)
        if path.exists():
            return str(path)
    raise ValueError(f"unsupported image value: {type(value).__name__}")


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        parsed = _parse_maybe_literal(value)
        return parsed if isinstance(parsed, dict) else {}
    return dict(value) if hasattr(value, "items") else {}


def _parse_maybe_literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "None"}:
            return None
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value
    return value


def _stringify_capability(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    text = str(value).strip()
    parsed = _parse_maybe_literal(text)
    if isinstance(parsed, (list, tuple)):
        return ",".join(str(item) for item in parsed)
    return text
