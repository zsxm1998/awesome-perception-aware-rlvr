# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional, Sequence

from ..py_functional import is_package_available


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


def _trainer_config(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    return (config or {}).get("trainer", {})


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return str(value)


def _import_pil():
    from PIL import Image, ImageOps

    return Image, ImageOps


def load_pil_image(image: Any) -> Any:
    """Load image-like inputs into a RGB PIL image."""
    Image, _ = _import_pil()

    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            raise ValueError("Image list must contain exactly one image when loading a single image.")
        image = image[0]

    if hasattr(image, "image") and isinstance(image.image, Image.Image):
        image = image.image

    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        pil_image = Image.open(image)
    elif isinstance(image, bytes):
        pil_image = Image.open(BytesIO(image))
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            pil_image = Image.open(BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            pil_image = Image.open(image["path"])
        else:
            raise ValueError("Image dictionary must contain either 'bytes' or 'path'.")
    else:
        raise ValueError(f"Unsupported image type for generation logging: {type(image)!r}.")

    pil_image.load()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def _resize_to_longest_edge(image: Any, max_size: int) -> Any:
    Image, ImageOps = _import_pil()
    del Image
    return ImageOps.contain(image, (max_size, max_size))


def make_image_grid(images: Sequence[Any], max_size: int) -> Any:
    """Create one RGB preview image from one or more image-like objects."""
    Image, ImageOps = _import_pil()
    pil_images = [load_pil_image(image) for image in images]
    if not pil_images:
        return None

    max_size = max(1, int(max_size))
    if len(pil_images) == 1:
        return _resize_to_longest_edge(pil_images[0], max_size)

    cols = math.ceil(math.sqrt(len(pil_images)))
    rows = math.ceil(len(pil_images) / cols)
    cell_size = max(1, max_size // max(cols, rows))
    canvas = Image.new("RGB", (cols * cell_size, rows * cell_size), color=(255, 255, 255))

    for index, image in enumerate(pil_images):
        thumb = ImageOps.contain(image, (cell_size, cell_size))
        row, col = divmod(index, cols)
        offset = (
            col * cell_size + (cell_size - thumb.width) // 2,
            row * cell_size + (cell_size - thumb.height) // 2,
        )
        canvas.paste(thumb, offset)

    return canvas


def _safe_filename(value: Any) -> str:
    text = "none" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text[:64] or "sample"


def _media_list(media: Any) -> list[Any]:
    if media is None:
        return []
    if isinstance(media, (str, bytes, dict)):
        return [media]
    try:
        return list(media)
    except TypeError:
        return [media]


def _media_paths(media: Any) -> list[str]:
    paths = []
    for item in _media_list(media):
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict) and item.get("path") is not None:
            paths.append(str(item["path"]))
    return paths


@dataclass
class GenerationSample:
    step: int
    split: str
    uid: Optional[str]
    prompt: str
    completion: str
    ground_truth: Any = None
    score: Optional[float] = None
    reward_details: dict[str, Any] = field(default_factory=dict)
    advantages: Any = None
    image_preview_path: Optional[str] = None
    image_count: int = 0
    video_count: int = 0
    video_paths: list[str] = field(default_factory=list)
    images: Any = field(default=None, repr=False)
    videos: Any = field(default=None, repr=False)

    def to_record(self) -> dict[str, Any]:
        record = {
            "step": self.step,
            "split": self.split,
            "uid": self.uid,
            "prompt": self.prompt,
            "completion": self.completion,
            "ground_truth": self.ground_truth,
            "score": self.score,
            "advantages": self.advantages,
            "image_preview_path": self.image_preview_path,
            "image_count": self.image_count,
            "video_count": self.video_count,
            "video_paths": self.video_paths,
        }
        for key, value in sorted((self.reward_details or {}).items()):
            record[f"reward/{key}"] = value

        return _json_safe(record)


def _coerce_sample(sample: GenerationSample | tuple[str, str, str, float], step: int, split: str) -> GenerationSample:
    if isinstance(sample, GenerationSample):
        sample.step = step
        sample.split = split
        return sample

    prompt, completion, ground_truth, score = sample
    return GenerationSample(
        step=step,
        split=split,
        uid=None,
        prompt=prompt,
        completion=completion,
        ground_truth=ground_truth,
        score=score,
    )


def _save_preview(sample: GenerationSample, config: Optional[dict[str, Any]], index: int) -> None:
    trainer_config = _trainer_config(config)
    if not trainer_config.get("log_images", True):
        return

    images = _media_list(sample.images)
    sample.image_count = len(images)
    if not images:
        return

    save_root = trainer_config.get("save_checkpoint_path")
    if not save_root:
        return

    max_size = trainer_config.get("generation_log_image_max_size", 768)
    media_dir = os.path.join(save_root, "generation_media", sample.split, f"step_{sample.step}")
    os.makedirs(media_dir, exist_ok=True)
    filename = f"{sample.split}_{sample.step}_{index}_{_safe_filename(sample.uid)}.png"
    preview_path = os.path.join(media_dir, filename)
    preview = make_image_grid(images, max_size=max_size)
    if preview is not None:
        preview.save(preview_path)
        sample.image_preview_path = preview_path


def prepare_generation_samples(
    samples: Sequence[GenerationSample | tuple[str, str, str, float]],
    config: Optional[dict[str, Any]],
    step: int,
    split: str,
) -> list[GenerationSample]:
    prepared = [_coerce_sample(sample, step=step, split=split) for sample in samples]
    for index, sample in enumerate(prepared):
        sample.video_count = len(_media_list(sample.videos))
        sample.video_paths = _media_paths(sample.videos)
        if _trainer_config(config).get("log_images", True):
            try:
                _save_preview(sample, config, index)
            except Exception as exc:
                print(f"Failed to create generation image preview for sample {index}: {exc}")
    return prepared


def _table_columns(samples: Sequence[GenerationSample]) -> list[str]:
    reward_columns = sorted({f"reward/{key}" for sample in samples for key in (sample.reward_details or {})})
    return [
        "step",
        "split",
        "uid",
        "image",
        "image_preview_path",
        "image_count",
        "video_count",
        "video_paths",
        "prompt",
        "completion",
        "ground_truth",
        "score",
        "advantages",
        *reward_columns,
    ]


def _table_row(sample: GenerationSample, columns: Sequence[str], image_value: Any = None) -> list[Any]:
    record = sample.to_record()
    record["image"] = image_value
    return [record.get(column) for column in columns]


def _caption(sample: GenerationSample, config: Optional[dict[str, Any]]) -> str:
    limit = int(_trainer_config(config).get("generation_log_caption_max_chars", 512))
    text = f"{sample.split} step={sample.step} uid={sample.uid}\nscore={sample.score}\n{sample.prompt}"
    return text[:limit]


@dataclass
class GenerationLogger(ABC):
    config: Optional[dict[str, Any]]

    @abstractmethod
    def log(self, samples: Sequence[GenerationSample], step: int, split: str) -> None: ...


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: Sequence[GenerationSample], step: int, split: str) -> None:
        for sample in samples:
            print(
                f"[{split}/generation step={step} uid={sample.uid}]\n"
                f"[image_preview] {sample.image_preview_path}\n"
                f"[prompt] {sample.prompt}\n"
                f"[output] {sample.completion}\n"
                f"[ground_truth] {sample.ground_truth}\n"
                f"[score] {sample.score}\n"
                f"[advantages] {sample.advantages}\n"
            )


@dataclass
class FileGenerationLogger(GenerationLogger):
    def __post_init__(self) -> None:
        save_root = _trainer_config(self.config).get("save_checkpoint_path")
        if save_root:
            os.makedirs(save_root, exist_ok=True)
            with open(self._jsonl_path, "w", encoding="utf-8"):
                pass

    @property
    def _jsonl_path(self) -> str:
        trainer_config = _trainer_config(self.config)
        save_root = trainer_config["save_checkpoint_path"]
        filename = trainer_config.get("generations_log_file", "completions.jsonl")
        return os.path.join(save_root, filename)

    def log(self, samples: Sequence[GenerationSample], step: int, split: str) -> None:
        del step, split
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def log(self, samples: Sequence[GenerationSample], step: int, split: str) -> None:
        columns = _table_columns(samples)
        data = []
        log_images = _trainer_config(self.config).get("log_images", True)
        for sample in samples:
            image_value = None
            if log_images and sample.image_preview_path is not None:
                image_value = wandb.Image(sample.image_preview_path, caption=_caption(sample, self.config))
            data.append(_table_row(sample, columns, image_value=image_value))

        wandb.log({f"{split}/generations": wandb.Table(columns=columns, data=data)}, step=step)


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: Sequence[GenerationSample], step: int, split: str) -> None:
        columns = [column for column in _table_columns(samples) if column != "image"]
        rows = [_table_row(sample, columns) for sample in samples]
        log_data = {f"{split}/generations": swanlab.echarts.Table().add(columns, rows)}

        images = []
        if _trainer_config(self.config).get("log_images", True):
            images = [
                swanlab.Image(sample.image_preview_path, caption=_caption(sample, self.config))
                for sample in samples
                if sample.image_preview_path is not None
            ]
        if images:
            log_data[f"{split}/generation_images"] = images

        swanlab.log(log_data, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "file": FileGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


class AggregateGenerationsLogger:
    def __init__(self, loggers: Sequence[str], config: Optional[dict[str, Any]] = None):
        self.config = config
        self.loggers: list[GenerationLogger] = []

        logger_names = list(loggers)
        if _trainer_config(config).get("train_generations_to_log", 0) != 0 and "file" not in logger_names:
            logger_names.append("file")

        for logger in logger_names:
            if logger in GEN_LOGGERS:
                self.loggers.append(GEN_LOGGERS[logger](config))

    def log(
        self,
        samples: Sequence[GenerationSample | tuple[str, str, str, float]],
        step: int,
        split: str = "val",
    ) -> None:
        if not self.loggers or not samples:
            return
        samples = prepare_generation_samples(samples, self.config, step=step, split=split)
        for logger in self.loggers:
            logger.log(samples, step, split)
