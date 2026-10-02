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
"""Evaluation results visualization server (default http://127.0.0.1:7777).

Usage:
    python eval/viz_server.py
    python eval/viz_server.py --results eval/results --data-root data/eval --port 7777

The server binds to 127.0.0.1 by default; use an SSH tunnel to browse it from another
machine (or pass --host 0.0.0.0 on a trusted network). Files are only served from the
results directory and the evaluation data root (plus absolute image roots configured in
eval/config/benchmarks.yaml); every other path is rejected with HTTP 403.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import math
import os
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EVAL_DIR.parent))
sys.path.insert(0, str(EVAL_DIR))

from easyr1_eval.paths import default_data_root  # noqa: E402
from flask import Flask, Response, jsonify, request, send_file  # noqa: E402


# ── Defaults ─────────────────────────────────────────────────────────────────
RESULTS_DIR = EVAL_DIR / "results"
CONFIG_FILE = EVAL_DIR / "config" / "benchmarks.yaml"
DEFAULT_DATA_ROOT = default_data_root()
DEFAULT_HOST = "127.0.0.1"
PORT = 7777
MAX_RESULT_DEPTH = 4

app = Flask(__name__)

# ── Runtime config (overridable via CLI) ─────────────────────────────────────
_cfg: dict = {
    "results_dir": str(RESULTS_DIR),
    "data_root": str(DEFAULT_DATA_ROOT),
}

# ── In-memory caches ──────────────────────────────────────────────────────────
_benchmark_specs: dict | None = None  # key -> BenchmarkSpec
_image_index: dict[str, dict] = {}  # bmark_key -> {sid -> [raw image/path]}
_image_index_lock = threading.Lock()
_loading_status: dict[str, str] = {}  # bmark_key -> "loading"|"done"|"error:<msg>"
_display_order_index: dict[str, dict[str, tuple]] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _jsonify(value: Any):
    return jsonify(_json_safe(value))


# ══════════════════════════════════════════════════════════════════════════════
# Path access control
# ══════════════════════════════════════════════════════════════════════════════


def _lexical(path: str | Path) -> str:
    """Absolute, normalized path without resolving symlinks (``..`` is collapsed)."""
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))


def _is_within(path: str | Path, roots: list[str | Path]) -> bool:
    candidate = _lexical(path)
    for root in roots:
        base = _lexical(root)
        if candidate == base or candidate.startswith(base.rstrip(os.sep) + os.sep):
            return True
    return False


def _allowed_file_roots() -> list[str]:
    """Directories /api/image may serve from: results dir, data root, absolute image roots."""
    roots = [_cfg["results_dir"], _cfg["data_root"]]
    data_root = Path(_cfg["data_root"])
    for spec in _get_specs().values():
        try:
            image_root = spec.resolved_image_root(data_root)
        except Exception:
            image_root = None
        if image_root is not None:
            roots.append(str(image_root))
    return roots


def _allowed_result_dir(value: str) -> bool:
    return bool(value) and _is_within(value, [_cfg["results_dir"]])


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark config helpers
# ══════════════════════════════════════════════════════════════════════════════


def _get_specs() -> dict:
    global _benchmark_specs
    if _benchmark_specs is None:
        try:
            from easyr1_eval.registry import load_benchmark_specs

            specs = load_benchmark_specs(CONFIG_FILE)
            _benchmark_specs = {s.key: s for s in specs}
        except Exception:
            _benchmark_specs = {}
    return _benchmark_specs


def _spec_label_group(key: str) -> tuple[str, str]:
    specs = _get_specs()
    s = specs.get(key)
    return (s.label if s else key, s.group if s else "Other")


# ══════════════════════════════════════════════════════════════════════════════
# Grounding extraction
# ══════════════════════════════════════════════════════════════════════════════

# JSON grounding: {"label": "...", "bbox_list": [[x1,y1,x2,y2],...], "image_idx": 0}
_JSON_GND_RE = re.compile(
    r'\{\s*"label"\s*:\s*"([^"]*)"\s*,\s*"bbox_list"\s*:\s*(\[(?:\[[^\]]*\](?:\s*,\s*)?)+\])'
    r'(?:\s*,\s*"image_idx"\s*:\s*(\d+))?\s*\}'
    r"|"
    r'\{\s*"label"\s*:\s*"([^"]*)"\s*,\s*"image_idx"\s*:\s*(\d+)\s*,\s*"bbox_list"\s*:\s*(\[(?:\[[^\]]*\](?:\s*,\s*)?)+\])\s*\}',
)

# XML grounding: <region name="..." image_idx="0" id="0">[[x1,y1,x2,y2],...]</region>
_XML_GND_RE = re.compile(r"<region\s+([^>]*?)>(.*?)</region>", re.DOTALL)


def _extract_grounding(text: str) -> list[dict]:
    """Return unified list of {label, bbox_list, image_idx} from response text."""
    results: list[dict] = []

    # XML format
    for m in _XML_GND_RE.finditer(text):
        attrs, body = m.group(1), m.group(2).strip()
        nm = re.search(r'name=["\']([^"\']*)["\']', attrs)
        im = re.search(r'image_idx=["\']?(\d+)["\']?', attrs)
        label = nm.group(1) if nm else "region"
        img_idx = int(im.group(1)) if im else 0
        try:
            bbox_list = ast.literal_eval(body)
            if not isinstance(bbox_list, list):
                continue
            if bbox_list and isinstance(bbox_list[0], (int, float)):
                bbox_list = [bbox_list]
            results.append({"label": label, "bbox_list": bbox_list, "image_idx": img_idx, "fmt": "xml"})
        except Exception:
            pass

    # JSON format (only if no XML found, or as supplement)
    # Use a broader pattern that handles both field orders
    _JSON_PAT = re.compile(r'\{(?=[^{}]*"label"\s*:)(?=[^{}]*"bbox_list"\s*:)[^{}]*\}', re.DOTALL)
    for m in _JSON_PAT.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if "label" in obj and "bbox_list" in obj:
                bl = obj["bbox_list"]
                if isinstance(bl, list):
                    if bl and isinstance(bl[0], (int, float)):
                        bl = [bl]
                    results.append(
                        {
                            "label": obj["label"],
                            "bbox_list": bl,
                            "image_idx": int(obj.get("image_idx", 0)),
                            "fmt": "json",
                        }
                    )
        except Exception:
            pass

    # De-duplicate (XML and JSON might both match the same region)
    seen = set()
    deduped = []
    for r in results:
        key = (r["label"], r["image_idx"], str(r["bbox_list"]))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _split_response(text: str) -> dict:
    """Split response into {think, answer, raw} parts."""
    think = ""
    answer = ""

    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_m:
        think = think_m.group(1).strip()

    answer_m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_m:
        answer = answer_m.group(1).strip()
    else:
        # \boxed{...} or \\boxed{...} (model may double-escape)
        box_m = re.search(r"\\{1,2}boxed\{(.*?)\}", text, re.DOTALL)
        if box_m:
            answer = box_m.group(1).strip()
        else:
            # fallback: everything after </think>
            after = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            answer = after

    return {"think": think, "answer": answer, "raw": text}


# ══════════════════════════════════════════════════════════════════════════════
# Image loading
# ══════════════════════════════════════════════════════════════════════════════


def _mime_from_image_format(fmt: str | None) -> str:
    if not fmt:
        return "image/jpeg"
    fmt = fmt.lower()
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(fmt, f"image/{fmt}")


def _mime_from_bytes(raw: bytes) -> str:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            return _mime_from_image_format(im.format)
    except Exception:
        return "image/jpeg"


def _img_to_data_url(img_data: Any) -> str:
    """Convert PIL/bytes/dict/path image to a data URL string."""
    try:
        from PIL import Image

        if isinstance(img_data, str):
            p = Path(img_data)
            if p.exists():
                with open(p, "rb") as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode()
                return f"data:{_mime_from_image_format(p.suffix.lower().lstrip('.'))};base64,{b64}"

        elif isinstance(img_data, Image.Image):
            buf = io.BytesIO()
            img_data.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"

        elif isinstance(img_data, bytes):
            mime = _mime_from_bytes(img_data)
            b64 = base64.b64encode(img_data).decode()
            return f"data:{mime};base64,{b64}"

        elif isinstance(img_data, dict):
            raw_bytes = img_data.get("bytes")
            if raw_bytes:
                return _img_to_data_url(raw_bytes)
            path = img_data.get("path")
            if path:
                return _img_to_data_url(path)

    except Exception:
        pass
    return ""


def _image_to_response(img_data: Any) -> Response:
    """Return a Flask response for PIL/bytes/dict/path image data."""
    try:
        from PIL import Image

        if isinstance(img_data, str):
            path = Path(img_data)
            if not path.exists():
                return Response("not found", status=404)
            return send_file(str(path))

        if isinstance(img_data, Image.Image):
            buf = io.BytesIO()
            img_data.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype="image/png")

        if isinstance(img_data, bytes):
            return Response(img_data, mimetype=_mime_from_bytes(img_data))

        if isinstance(img_data, dict):
            raw_bytes = img_data.get("bytes")
            if raw_bytes:
                return _image_to_response(raw_bytes)
            path = img_data.get("path")
            if path:
                return _image_to_response(path)

    except Exception as e:
        return Response(f"error: {e}", status=500)
    return Response("unsupported image", status=404)


def _load_image_index_bg(benchmark_key: str) -> None:
    """Background thread: build sample_id -> [raw image/path] index."""
    data_root = Path(_cfg["data_root"])
    specs = _get_specs()
    spec = specs.get(benchmark_key)
    if spec is None:
        with _image_index_lock:
            _loading_status[benchmark_key] = "error:unknown benchmark"
        return

    try:
        from easyr1_eval.loaders import load_samples

        samples = load_samples(spec, data_root)
        index: dict[str, list[Any]] = {}
        for s in samples:
            index[s.sample_id] = list(s.images)
        with _image_index_lock:
            _image_index[benchmark_key] = index
            _loading_status[benchmark_key] = "done"
    except Exception as e:
        with _image_index_lock:
            _image_index[benchmark_key] = {}
            _loading_status[benchmark_key] = f"error:{e}"


def _ensure_image_index(benchmark_key: str) -> None:
    """Trigger background loading if not started yet."""
    with _image_index_lock:
        if benchmark_key in _loading_status:
            return
        _loading_status[benchmark_key] = "loading"
    t = threading.Thread(target=_load_image_index_bg, args=(benchmark_key,), daemon=True)
    t.start()


def _ensure_images_for_samples(benchmark_key: str, sample_ids: list[str], limit_hint: int | None = None) -> None:
    """Synchronously cache embedded images needed for the current page."""
    wanted = [str(sid) for sid in sample_ids if sid]
    if not wanted:
        return

    with _image_index_lock:
        cached = _image_index.setdefault(benchmark_key, {})
        missing = [sid for sid in wanted if sid not in cached]
        if not missing:
            _loading_status[benchmark_key] = "done"
            return
        _loading_status[benchmark_key] = "loading"

    data_root = Path(_cfg["data_root"])
    spec = _get_specs().get(benchmark_key)
    if spec is None:
        with _image_index_lock:
            _loading_status[benchmark_key] = "error:unknown benchmark"
        return

    try:
        from easyr1_eval.loaders import load_samples

        samples = load_samples(spec, data_root, limit=limit_hint)
        found = {s.sample_id for s in samples}
        if any(sid not in found for sid in missing):
            samples = load_samples(spec, data_root)

        with _image_index_lock:
            index = _image_index.setdefault(benchmark_key, {})
            for sample in samples:
                index[sample.sample_id] = list(sample.images)
            still_missing = [sid for sid in wanted if sid not in index]
            if still_missing:
                _loading_status[benchmark_key] = f"error:missing image samples: {still_missing[:3]}"
            else:
                _loading_status[benchmark_key] = "done"
    except Exception as e:
        with _image_index_lock:
            _loading_status[benchmark_key] = f"error:{e}"


def _get_images_for_sample(benchmark_key: str, sample_id: str, image_refs: list[str]) -> list[str]:
    """Return list of image URLs/data-URLs for a sample."""
    urls: list[str] = []
    with _image_index_lock:
        index = _image_index.get(benchmark_key, {})
        cached = index.get(sample_id, [])

    for i, ref in enumerate(image_refs):
        if ref != "<embedded>":
            # File path – serve via /api/image endpoint
            b64p = base64.urlsafe_b64encode(ref.encode()).decode()
            urls.append(f"/api/image?p={b64p}")
        else:
            # Embedded – serve cached raw bytes/PIL data lazily.
            if i < len(cached):
                c = cached[i]
                if isinstance(c, str) and c.startswith("data:"):
                    urls.append(c)
                elif isinstance(c, str) and c.startswith("/"):
                    b64p = base64.urlsafe_b64encode(c.encode()).decode()
                    urls.append(f"/api/image?p={b64p}")
                elif c:
                    b64b = base64.urlsafe_b64encode(benchmark_key.encode()).decode()
                    b64s = base64.urlsafe_b64encode(str(sample_id).encode()).decode()
                    urls.append(f"/api/embedded_image?b={b64b}&sid={b64s}&i={i}")
                else:
                    urls.append("")
            else:
                urls.append("")
    return urls


# ══════════════════════════════════════════════════════════════════════════════
# Prediction loading
# ══════════════════════════════════════════════════════════════════════════════


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


_jsonl_cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
_jsonl_cache_lock = threading.Lock()
_JSONL_CACHE_MAX = 8


def _load_jsonl_cached(path: Path) -> list[dict]:
    """mtime-keyed prediction cache; compare mode parses two files per page."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    key = str(path)
    with _jsonl_cache_lock:
        entry = _jsonl_cache.get(key)
        if entry is not None and entry[0] == mtime:
            _jsonl_cache.move_to_end(key)
            return entry[1]
    records = _load_jsonl(path)
    with _jsonl_cache_lock:
        _jsonl_cache[key] = (mtime, records)
        _jsonl_cache.move_to_end(key)
        while len(_jsonl_cache) > _JSONL_CACHE_MAX:
            _jsonl_cache.popitem(last=False)
    return records


def _group_rollouts(records: list[dict]) -> list[dict]:
    """Group by sample_id, preserving first-appearance order."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in records:
        sid = r.get("sample_id", "")
        groups.setdefault(sid, []).append(r)
    result = []
    for sid, rollouts in groups.items():
        base = dict(rollouts[0])
        base["_rollouts"] = rollouts
        result.append(base)
    return result


def _natural_sort_key(value: Any) -> tuple:
    """Return a deterministic human/numeric ordering key for identifiers."""
    parts = re.split(r"(\d+)", str(value or "").lower())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts if part)


def _record_display_sort_key(
    benchmark_key: str,
    group: dict,
    fallback_index: int,
) -> tuple:
    """Restore loader/source order after multi-shard prediction merging."""
    metadata = group.get("metadata") or {}
    sample_id = str(group.get("sample_id", ""))

    row_index = metadata.get("row_index")
    if row_index is not None:
        return (
            0,
            _natural_sort_key(metadata.get("source_file") or ""),
            int(row_index),
            _natural_sort_key(sample_id),
            fallback_index,
        )

    return (1, _natural_sort_key(sample_id), fallback_index)


def _normalized_gt_box(
    values: Any,
    *,
    scale: str,
    width: float | None = None,
    height: float | None = None,
) -> list[float] | None:
    """Convert one GT box to the 0-1000 canvas coordinate convention."""
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None

    if scale == "normalized":
        x1, y1, x2, y2 = (value * 1000.0 for value in (x1, y1, x2, y2))
    elif scale == "pixel":
        if width is None or height is None or width <= 0 or height <= 0:
            return None
        x1, x2 = x1 / width * 1000.0, x2 / width * 1000.0
        y1, y2 = y1 / height * 1000.0, y2 / height * 1000.0

    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return [x1, y1, x2, y2]


def _ground_truth_regions(extra_info: dict) -> list[dict]:
    """Build display-only multi-box GT regions from all benchmark schemas."""
    if not isinstance(extra_info, dict):
        return []

    width = extra_info.get("width")
    height = extra_info.get("height")
    try:
        width = float(width) if width is not None else None
        height = float(height) if height is not None else None
    except (TypeError, ValueError):
        width = height = None
    bbox_format = str(extra_info.get("bbox_format") or "").lower()

    if "bboxs_normalized" in extra_info:
        boxes = extra_info.get("bboxs_normalized") or []
        scale = "normalized"
    elif "bbox_1000" in extra_info:
        boxes = [extra_info.get("bbox_1000")]
        scale = "1000"
    elif "bbox" in extra_info:
        boxes = [extra_info.get("bbox")]
        scale = "normalized" if "normalized" in bbox_format else "pixel"
    else:
        boxes = extra_info.get("bboxs") or []
        if "normalized" in bbox_format:
            scale = "normalized"
        elif "1000" in bbox_format:
            scale = "1000"
        else:
            scale = "pixel"

    names = extra_info.get("bbox_names") or []
    objects = extra_info.get("objects") or []
    regions: list[dict] = []
    for index, raw_box in enumerate(boxes):
        box = _normalized_gt_box(
            raw_box,
            scale=scale,
            width=width,
            height=height,
        )
        if box is None:
            continue
        label = names[index] if index < len(names) else None
        if label is None and index < len(objects) and isinstance(objects[index], dict):
            label = objects[index].get("label")
        if not label:
            label = "GT" if len(boxes) == 1 else f"GT {index + 1}"
        regions.append({"label": str(label), "bbox_list": [box], "image_idx": 0})
    return regions


def _get_display_order_index(benchmark_key: str) -> dict[str, tuple]:
    """Return sample_id -> sort key for display-only benchmark ordering."""
    if benchmark_key in _display_order_index:
        return _display_order_index[benchmark_key]

    if benchmark_key != "pope":
        _display_order_index[benchmark_key] = {}
        return {}

    specs = _get_specs()
    spec = specs.get(benchmark_key)
    if spec is None:
        _display_order_index[benchmark_key] = {}
        return {}

    try:
        import glob

        import pandas as pd

        data_root = Path(_cfg["data_root"])
        paths: list[Path] = []
        for raw_path in spec.data_paths(data_root):
            matches = sorted(Path(item) for item in glob.glob(str(raw_path)))
            paths.extend(matches or [raw_path])

        index: dict[str, tuple] = {}
        image_order: dict[str, int] = {}
        split_order: dict[str, int] = {}
        for path_order, path in enumerate(paths):
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["question_id", "image_source", "category"])
            per_image_counts: dict[str, int] = {}
            for row_index, row in df.iterrows():
                category = str(row.get("category") or path.stem)
                image_source = str(row.get("image_source") or "")
                question_id = str(row.get("question_id"))
                sample_id = f"{category}:{question_id}"
                if image_source not in image_order:
                    image_order[image_source] = len(image_order)
                if category not in split_order:
                    split_order[category] = path_order
                in_image_order = per_image_counts.get(image_source, 0)
                per_image_counts[image_source] = in_image_order + 1
                index[sample_id] = (
                    image_order[image_source],
                    split_order[category],
                    in_image_order,
                    int(row_index),
                )
        _display_order_index[benchmark_key] = index
        return index
    except Exception:
        _display_order_index[benchmark_key] = {}
        return {}


def _sort_groups_for_display(benchmark_key: str, groups: list[dict]) -> list[dict]:
    enumerated = list(enumerate(groups))
    if benchmark_key == "pope":
        order_index = _get_display_order_index(benchmark_key)
        if order_index:
            fallback_base = len(order_index) + len(groups) + 1
            enumerated.sort(
                key=lambda item: order_index.get(
                    str(item[1].get("sample_id", "")),
                    (fallback_base + item[0], 999, 999, item[0]),
                )
            )
            return [group for _, group in enumerated]

    enumerated.sort(
        key=lambda item: _record_display_sort_key(
            benchmark_key,
            item[1],
            item[0],
        )
    )
    return [group for _, group in enumerated]


def _to_canvas_groundings(groundings: list[dict], record: dict) -> list[dict]:
    """Rescale pixel-format boxes (runner --box-format pixel) to the 0-1000 display canvas."""
    metadata = record.get("eval_metadata") if isinstance(record.get("eval_metadata"), dict) else {}
    sizes = record.get("image_sizes")
    if metadata.get("box_format") != "pixel" or not isinstance(sizes, list):
        return groundings
    scaled = []
    for item in groundings:
        idx = int(item.get("image_idx") or 0)
        size = sizes[idx].get("model_input") if idx < len(sizes) and isinstance(sizes[idx], dict) else None
        if not size or not size[0] or not size[1]:
            scaled.append(item)
            continue
        width, height = float(size[0]), float(size[1])
        boxes = []
        for box in item.get("bbox_list") or []:
            try:
                x1, y1, x2, y2 = (float(value) for value in box)
            except (TypeError, ValueError):
                continue
            boxes.append([x1 / width * 1000, y1 / height * 1000, x2 / width * 1000, y2 / height * 1000])
        scaled.append({**item, "bbox_list": boxes})
    return scaled


def _serialize_rollouts(group: dict) -> list[dict]:
    """Parse a grouped prediction record into display-ready rollouts."""
    rollouts_out = []
    for rollout in group.get("_rollouts", [group]):
        responses = rollout.get("responses", [])
        rmeta = rollout.get("response_metadata", [{}] * len(responses))
        parsed = []
        for resp in responses:
            sp = _split_response(resp)
            parsed.append(
                {
                    "think": sp["think"],
                    "answer": sp["answer"],
                    "raw": resp,
                    "groundings": _to_canvas_groundings(_extract_grounding(resp), rollout),
                }
            )
        rollouts_out.append({"responses": parsed, "metadata": rmeta})
    return rollouts_out


def _load_per_sample_metrics(model_dir: Path, benchmark_key: str) -> dict[str, dict]:
    """Load per-sample metric files → {sample_id -> metric_dict}."""
    metrics_dir = model_dir / "metrics"
    idx: dict[str, dict] = {}
    paths = [metrics_dir / f"{benchmark_key}_per_sample.jsonl"]
    paths.extend(sorted(metrics_dir.glob(f"{benchmark_key}_per_sample_*.jsonl")))
    for p in paths:
        if not p.is_file():
            continue
        for rec in _load_jsonl(p):
            sid = rec.get("sample_id")
            if sid:
                idx.setdefault(sid, {}).update(rec)
    return idx


# ══════════════════════════════════════════════════════════════════════════════
# Compare-mode diff filter
# ══════════════════════════════════════════════════════════════════════════════

# Primary grounding metric, first present on both sides wins.
_DIFF_GROUNDING_KEYS = ("grit_iou", "best_iou")
# Behavioral flips count as grounding diffs regardless of threshold.
_DIFF_FLIP_KEYS = ("missing_prediction", "parse_failed")


def _metric_answer_diff(ma: dict, mb: dict) -> bool:
    if "answer_correct" in ma and "answer_correct" in mb:
        return bool(ma["answer_correct"]) != bool(mb["answer_correct"])
    va, vb = ma.get("mean_accuracy"), mb.get("mean_accuracy")
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        # No boolean verdict: flag only when most rollouts changed verdict.
        return abs(float(va) - float(vb)) >= 0.5
    return False


def _metric_grounding_diff(ma: dict, mb: dict, threshold: float) -> bool:
    for key in _DIFF_FLIP_KEYS:
        if key in ma and key in mb and bool(ma[key]) != bool(mb[key]):
            return True
    for key in _DIFF_GROUNDING_KEYS:
        va, vb = ma.get(key), mb.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            return abs(float(va) - float(vb)) >= threshold
    return False


def _is_diff_sample(
    sid: str,
    primary_metrics: dict[str, dict],
    compare_groups: dict[str, dict],
    compare_metrics: dict[str, dict],
    mode: str,
    threshold: float,
) -> bool:
    if sid not in compare_groups:
        return False
    ma = primary_metrics.get(sid, {})
    mb = compare_metrics.get(sid, {})
    if mode == "answer":
        return _metric_answer_diff(ma, mb)
    if mode == "grounding":
        return _metric_grounding_diff(ma, mb, threshold)
    return _metric_answer_diff(ma, mb) or _metric_grounding_diff(ma, mb, threshold)


# ══════════════════════════════════════════════════════════════════════════════
# Flask routes
# ══════════════════════════════════════════════════════════════════════════════


@app.route("/")
def index():
    return HTML_PAGE


def _find_result_dirs(results_dir: Path, max_depth: int = MAX_RESULT_DEPTH) -> list[Path]:
    """Run directories (those holding predictions/) up to ``max_depth`` levels deep."""
    found: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        try:
            children = sorted(child for child in directory.iterdir() if child.is_dir())
        except OSError:
            return
        for child in children:
            if child.name.startswith((".", "_")) or child.name in {"predictions", "metrics", "state"}:
                continue
            if (child / "predictions").is_dir():
                found.append(child)
            elif depth < max_depth:
                walk(child, depth + 1)

    walk(results_dir, 1)
    return found


@app.route("/api/models")
def api_models():
    results_dir = Path(_cfg["results_dir"])
    if not results_dir.exists():
        return _jsonify([])
    dirs = [{"name": d.relative_to(results_dir).as_posix(), "path": str(d)} for d in _find_result_dirs(results_dir)]
    return _jsonify(dirs)


@app.route("/api/benchmarks")
def api_benchmarks():
    model_dir = request.args.get("model_dir", "")
    if not model_dir:
        return _jsonify([])
    if not _allowed_result_dir(model_dir):
        return _jsonify({"error": "model_dir must be inside the results directory"}), 403
    pred_dir = Path(model_dir) / "predictions"
    if not pred_dir.exists():
        return _jsonify([])

    out = []
    for d in sorted(pred_dir.iterdir()):
        if d.is_dir() and (d / "predictions.jsonl").exists():
            key = d.name
            label, group = _spec_label_group(key)
            # Quick count
            count = sum(1 for _ in open(d / "predictions.jsonl"))
            out.append({"key": key, "label": label, "group": group, "count": count})

    # Follow benchmarks.yaml order so the sidebar matches the evaluation table.
    spec_order = {key: index for index, key in enumerate(_get_specs())}
    out.sort(
        key=lambda item: (
            (0, spec_order[item["key"]]) if item["key"] in spec_order else (1, item["group"], item["label"])
        )
    )

    # Load summary metrics if available
    metrics_map: dict[str, float] = {}
    summary_csv = Path(model_dir) / "summary.csv"
    if summary_csv.exists():
        try:
            import pandas as pd

            df = pd.read_csv(summary_csv)
            df = df[df["row_type"] == "benchmark"]
            for _, row in df.iterrows():
                if row.get("status") != "ok":
                    continue
                score = pd.to_numeric(row.get("raw_score"), errors="coerce")
                if pd.notna(score):
                    metrics_map[str(row["benchmark"])] = float(score)
        except Exception:
            pass

    for item in out:
        if item["key"] in metrics_map:
            item["score"] = round(metrics_map[item["key"]], 2)

    # Trigger image pre-loading for the first benchmark (heuristic)
    return _jsonify(out)


@app.route("/api/samples")
def api_samples():
    model_dir_s = request.args.get("model_dir", "")
    benchmark = request.args.get("benchmark", "")
    compare_dir_s = request.args.get("compare_dir", "").strip()
    diff_mode = request.args.get("diff_mode", "").strip().lower()
    if diff_mode not in ("any", "answer", "grounding"):
        diff_mode = ""
    try:
        diff_threshold = float(request.args.get("diff_threshold", 0.3))
    except (TypeError, ValueError):
        diff_threshold = 0.3
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(100, max(1, int(request.args.get("size", 20))))
    search = request.args.get("search", "").strip().lower()

    if not model_dir_s or not benchmark:
        return _jsonify({"samples": [], "total": 0, "page": 1, "pages": 0})
    if not _allowed_result_dir(model_dir_s) or (compare_dir_s and not _allowed_result_dir(compare_dir_s)):
        return _jsonify({"error": "model_dir/compare_dir must be inside the results directory"}), 403
    if "/" in benchmark or "\\" in benchmark or benchmark in {".", ".."}:
        return _jsonify({"error": "invalid benchmark key"}), 400

    model_dir = Path(model_dir_s)
    jsonl_path = model_dir / "predictions" / benchmark / "predictions.jsonl"
    if not jsonl_path.exists():
        return _jsonify({"error": "predictions.jsonl not found"}), 404

    records = _load_jsonl_cached(jsonl_path)
    grouped = _group_rollouts(records)
    grouped = _sort_groups_for_display(benchmark, grouped)

    # Search filter
    if search:
        filtered = []
        for g in grouped:
            prompt = str(g.get("prompt", "")).lower()
            sid = str(g.get("sample_id", "")).lower()
            if search in prompt or search in sid:
                filtered.append(g)
        grouped = filtered

    # Per-sample metrics (loaded before pagination — the diff filter needs them)
    per_sample_metrics = _load_per_sample_metrics(model_dir, benchmark)

    # Compare model: align its records by sample_id
    compare_available = False
    compare_groups: dict[str, dict] = {}
    compare_metrics: dict[str, dict] = {}
    if compare_dir_s:
        compare_dir = Path(compare_dir_s)
        compare_jsonl = compare_dir / "predictions" / benchmark / "predictions.jsonl"
        if compare_jsonl.exists():
            compare_available = True
            compare_grouped = _group_rollouts(_load_jsonl_cached(compare_jsonl))
            compare_groups = {str(g.get("sample_id", "")): g for g in compare_grouped}
            compare_metrics = _load_per_sample_metrics(compare_dir, benchmark)

    # Diff-only filter: keep samples where the two models disagree
    diff_active = bool(diff_mode) and compare_available
    unfiltered_total = len(grouped)
    if diff_active:
        grouped = [
            g
            for g in grouped
            if _is_diff_sample(
                str(g.get("sample_id", "")),
                per_sample_metrics,
                compare_groups,
                compare_metrics,
                diff_mode,
                diff_threshold,
            )
        ]

    total = len(grouped)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    start = (page - 1) * page_size
    page_data = grouped[start : start + page_size]

    # Ensure embedded images for the current page are cached before URLs are built.
    if page_data and any("<embedded>" in (g.get("image_refs") or []) for g in page_data):
        page_sample_ids = [str(g.get("sample_id", "")) for g in page_data]
        # Merged shard outputs are often interleaved, so a page can reference
        # later loader rows than its displayed offset suggests. Diff filtering
        # breaks the offset heuristic entirely, so load the full index then.
        if diff_active:
            limit_hint = None
        else:
            limit_hint = min(unfiltered_total, max(1, (start + len(page_data)) * 16))
        _ensure_images_for_samples(benchmark, page_sample_ids, limit_hint=limit_hint)

    # Build response
    samples_out = []
    for g in page_data:
        sid = g.get("sample_id", "")
        image_refs = g.get("image_refs", [])

        # Images
        img_urls = _get_images_for_sample(benchmark, sid, image_refs)

        sample_out = {
            "sample_id": sid,
            "benchmark": benchmark,
            "prompt": g.get("prompt", ""),
            "target": g.get("target", ""),
            "image_urls": img_urls,
            "image_refs": image_refs,
            "rollouts": _serialize_rollouts(g),
            "extra_info": g.get("extra_info", {}),
            "metadata": g.get("metadata", {}),
            "metrics": per_sample_metrics.get(sid, {}),
            "gt_regions": _ground_truth_regions(g.get("extra_info") or {}),
        }
        if compare_dir_s:
            cmp_group = compare_groups.get(str(sid))
            if cmp_group is None:
                sample_out["compare"] = None
            else:
                sample_out["compare"] = {
                    "rollouts": _serialize_rollouts(cmp_group),
                    "metrics": compare_metrics.get(sid, {}),
                }
        samples_out.append(sample_out)

    # Image loading status
    with _image_index_lock:
        img_status = _loading_status.get(benchmark, "not_started")

    return _jsonify(
        {
            "samples": samples_out,
            "total": total,
            "page": page,
            "pages": pages,
            "page_size": page_size,
            "image_status": img_status,
            "compare_available": compare_available if compare_dir_s else None,
            "diff_active": diff_active,
            "unfiltered_total": unfiltered_total,
        }
    )


@app.route("/api/image")
def api_image():
    p_b64 = request.args.get("p", "")
    if not p_b64:
        return "missing path", 400
    try:
        path = Path(base64.urlsafe_b64decode(p_b64.encode()).decode())
    except Exception:
        return "bad path", 400
    if not _is_within(path, _allowed_file_roots()):
        return "forbidden: path is outside the results and data directories", 403
    if not path.is_file():
        return "not found", 404
    try:
        return send_file(_lexical(path))
    except Exception as e:
        return f"error: {e}", 500


@app.route("/api/embedded_image")
def api_embedded_image():
    b_b64 = request.args.get("b", "")
    sid_b64 = request.args.get("sid", "")
    idx_s = request.args.get("i", "0")
    if not b_b64 or not sid_b64:
        return "missing image key", 400
    try:
        benchmark = base64.urlsafe_b64decode(b_b64.encode()).decode()
        sample_id = base64.urlsafe_b64decode(sid_b64.encode()).decode()
        idx = int(idx_s)
    except Exception as e:
        return f"bad image key: {e}", 400

    with _image_index_lock:
        images = _image_index.get(benchmark, {}).get(sample_id, [])
        img_data = images[idx] if 0 <= idx < len(images) else None

    if img_data is None:
        return "not found", 404
    return _image_to_response(img_data)


@app.route("/api/image_status")
def api_image_status():
    benchmark = request.args.get("benchmark", "")
    with _image_index_lock:
        status = _loading_status.get(benchmark, "not_started")
    return _jsonify({"status": status})


@app.route("/api/config")
def api_config():
    return _jsonify(
        {
            "results_dir": _cfg["results_dir"],
            "data_root": _cfg["data_root"],
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTML Page (single-file, all CSS/JS embedded)
# ══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EasyR1 Eval Visualizer</title>
<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:       #0d1117;
  --surface:  #161b22;
  --surface2: #21262d;
  --surface3: #2d333b;
  --border:   #30363d;
  --text:     #c9d1d9;
  --muted:    #8b949e;
  --accent:   #58a6ff;
  --green:    #3fb950;
  --red:      #f85149;
  --orange:   #e3b341;
  --purple:   #bc8cff;
  --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  --sidebar-w: 260px;
}
html, body { height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); font-size: 13px; line-height: 1.5; }

/* ── Layout ── */
#app { display: flex; height: 100vh; overflow: hidden; }
#sidebar { width: var(--sidebar-w); flex-shrink: 0; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }

/* ── Sidebar ── */
.sb-header { padding: 12px 14px 10px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; }
.sb-header svg { flex-shrink: 0; }
.sb-title { font-weight: 700; font-size: 14px; color: var(--accent); letter-spacing: .3px; }
.sb-body { flex: 1; overflow-y: auto; padding: 10px 0; }
.sb-section { padding: 6px 14px; }
.sb-label { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); margin-bottom: 5px; display: block; }
.sb-select { width: 100%; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 5px 8px; font-size: 12px; appearance: none; cursor: pointer; }
.sb-select:focus { outline: none; border-color: var(--accent); }
.sb-input { width: 100%; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 5px 8px; font-size: 12px; }
.sb-input:focus { outline: none; border-color: var(--accent); }
.sb-divider { height: 1px; background: var(--border); margin: 8px 14px; }
.bmark-group { margin-bottom: 4px; }
.bmark-group-label { padding: 3px 14px; font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--muted); letter-spacing: .7px; }
.bmark-item { display: flex; align-items: center; justify-content: space-between; padding: 5px 14px 5px 18px; cursor: pointer; border-radius: 0; transition: background .1s; gap: 6px; }
.bmark-item:hover { background: var(--surface2); }
.bmark-item.active { background: rgba(88,166,255,.12); border-left: 2px solid var(--accent); padding-left: 16px; }
.bmark-name { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12px; }
.bmark-score { font-size: 11px; color: var(--orange); font-weight: 600; white-space: nowrap; }
.bmark-count { font-size: 10px; color: var(--muted); white-space: nowrap; }
.sb-stats { padding: 8px 14px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); line-height: 1.8; }

/* ── Toolbar ── */
#toolbar { padding: 8px 14px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; flex-shrink: 0; flex-wrap: wrap; }
.breadcrumb { color: var(--muted); font-size: 12px; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.breadcrumb .sep { margin: 0 5px; }
.breadcrumb .model-name { color: var(--purple); font-weight: 600; }
.breadcrumb .bmark-name-bc { color: var(--accent); font-weight: 600; }
#search-box { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 4px 8px; font-size: 12px; width: 180px; }
#search-box:focus { outline: none; border-color: var(--accent); }
.page-size-sel { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 4px 6px; font-size: 12px; }
.page-size-sel:focus { outline: none; }
.pagination { display: flex; align-items: center; gap: 4px; }
.pg-btn { background: var(--surface2); border: 1px solid var(--border); border-radius: 5px; color: var(--text); padding: 3px 9px; font-size: 12px; cursor: pointer; }
.pg-btn:hover:not(:disabled) { background: var(--surface3); border-color: var(--accent); }
.pg-btn:disabled { opacity: .35; cursor: default; }
.pg-info { font-size: 11px; color: var(--muted); white-space: nowrap; }
.img-status { font-size: 11px; padding: 2px 7px; border-radius: 10px; }
.img-status.loading { background: rgba(227,179,65,.15); color: var(--orange); }
.img-status.done { background: rgba(63,185,80,.1); color: var(--green); }
.img-status.error { background: rgba(248,81,73,.1); color: var(--red); }

/* ── Samples area ── */
#samples-wrap { flex: 1; overflow-y: auto; padding: 10px 12px; }
#samples-container { display: flex; flex-direction: column; gap: 10px; }

/* ── Sample card ── */
.sample-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
.card-header {
  padding: 6px 12px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: flex-start; gap: 10px; flex-wrap: wrap;
}
.card-sid { font-size: 10px; color: var(--muted); font-family: var(--font-mono); white-space: nowrap; margin-top: 2px; }
.card-prompt { flex: 1; min-width: 0; font-size: 12px; color: var(--text); line-height: 1.4; }
.card-prompt.expandable { cursor: pointer; }
.card-prompt.expanded { white-space: pre-wrap; }
.prompt-toggle { color: var(--accent); font-size: 11px; white-space: nowrap; user-select: none; }
.card-target { font-size: 11px; background: var(--surface3); border-radius: 4px; padding: 1px 7px; color: var(--orange); white-space: nowrap; font-family: var(--font-mono); margin-top: 2px; }
.card-body { display: flex; gap: 0; min-height: 0; }

/* Images column */
.card-images { flex-shrink: 0; padding: 8px; display: flex; flex-direction: column; gap: 6px; max-width: 340px; min-width: 120px; width: 35%; background: #0a0d12; }
.img-wrapper { position: relative; display: inline-block; max-width: 100%; cursor: zoom-in; }
.img-wrapper img { display: block; max-width: 100%; max-height: 260px; width: auto; height: auto; border-radius: 4px; }
.img-wrapper canvas { position: absolute; top: 0; left: 0; pointer-events: none; border-radius: 4px; }
.img-label { font-size: 10px; color: var(--muted); margin-top: 2px; text-align: center; font-family: var(--font-mono); }
.image-modal { position: fixed; inset: 0; z-index: 1000; display: none; align-items: center; justify-content: center; background: rgba(0,0,0,.88); padding: 22px; overflow: hidden; }
.image-modal.open { display: flex; }
.image-modal-stage { position: relative; width: 96vw; height: 94vh; overflow: hidden; cursor: grab; }
.image-modal-stage.dragging { cursor: grabbing; }
.image-modal-scene { position: absolute; top: 0; left: 0; transform-origin: 0 0; }
.image-modal-scene img { display: block; max-width: 96vw; max-height: 94vh; width: auto; height: auto; border-radius: 4px; user-select: none; -webkit-user-drag: none; }
.image-modal-scene canvas { position: absolute; top: 0; left: 0; pointer-events: none; border-radius: 4px; }
.image-modal-close { position: fixed; top: 14px; right: 16px; width: 34px; height: 34px; border: 1px solid rgba(255,255,255,.25); border-radius: 50%; background: rgba(13,17,23,.85); color: #fff; font-size: 22px; line-height: 30px; z-index: 1001; }
.no-img { color: var(--muted); font-size: 11px; padding: 20px; text-align: center; }

/* Rollouts column */
.card-rollouts { flex: 1; min-width: 0; display: flex; flex-direction: column; border-left: 1px solid var(--border); }
.rollout-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); flex-wrap: wrap; background: var(--surface2); padding: 0 8px; }
.rollout-tab { padding: 4px 10px; font-size: 11px; color: var(--muted); cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; white-space: nowrap; display: flex; align-items: center; gap: 5px; }
.rollout-tab:hover { color: var(--text); }
.rollout-tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.rollout-tab .tab-badge { font-size: 9px; padding: 1px 4px; border-radius: 8px; }
.rollout-tab .tab-badge.correct { background: rgba(63,185,80,.2); color: var(--green); }
.rollout-tab .tab-badge.wrong { background: rgba(248,81,73,.15); color: var(--red); }
.rollout-content { flex: 1; overflow-y: auto; padding: 8px 10px; display: none; }
.rollout-content.active { display: block; }

/* Response parts */
.resp-section { margin-bottom: 7px; }
.resp-section-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin-bottom: 3px; }
.think-box { background: #0f1419; border: 1px solid #1e2b38; border-radius: 5px; padding: 6px 9px; font-family: var(--font-mono); font-size: 11px; line-height: 1.6; color: #a8bbbf; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
.think-box.collapsed { max-height: 56px; overflow: hidden; -webkit-mask-image: linear-gradient(180deg, #000 60%, transparent); }
.think-toggle { font-size: 10px; color: var(--accent); cursor: pointer; margin-top: 2px; display: inline-block; }
.answer-box { background: rgba(88,166,255,.07); border: 1px solid rgba(88,166,255,.2); border-radius: 5px; padding: 5px 9px; font-family: var(--font-mono); font-size: 12px; color: #d2e3ff; word-break: break-word; white-space: pre-wrap; }
.verdict { display: flex; align-items: center; gap: 5px; margin-top: 5px; }
.verdict-correct { color: var(--green); font-size: 11px; font-weight: 600; }
.verdict-wrong { color: var(--red); font-size: 11px; font-weight: 600; }

/* Grounding summary */
.gnd-section { margin-top: 6px; }
.gnd-label { font-size: 10px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
.gnd-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.gnd-chip { font-size: 10px; padding: 2px 7px; border-radius: 10px; color: #fff; font-weight: 500; cursor: default; white-space: nowrap; }

/* Metrics row */
.metrics-row { display: flex; gap: 8px; flex-wrap: wrap; padding: 4px 10px 5px; border-top: 1px solid var(--border); background: var(--surface2); }
.metric-item { font-size: 10px; color: var(--muted); }
.metric-item span { color: var(--text); font-weight: 600; }
.metric-item .cmp-arrow { color: var(--muted); font-weight: 400; margin: 0 1px; }
.metric-delta.pos { color: var(--green); }
.metric-delta.neg { color: var(--red); }

/* ── Compare mode ── */
.breadcrumb .model-a { color: var(--accent); font-weight: 600; }
.breadcrumb .model-b { color: var(--purple); font-weight: 600; }
.breadcrumb .vs-sep { color: var(--muted); font-style: italic; margin: 0 4px; }
.bmark-score .sc-a { color: var(--accent); }
.bmark-score .sc-arrow { color: var(--muted); font-weight: 400; margin: 0 3px; }
.bmark-score .sc-b { color: var(--purple); }
.bmark-score .sc-b.na { color: var(--muted); font-weight: 400; }
.cmp-img-stack { flex-shrink: 0; width: 30%; max-width: 320px; min-width: 140px; background: #0a0d12; display: flex; flex-direction: column; }
.cmp-img-stack .card-images { width: 100%; max-width: none; background: transparent; }
.cmp-duo { flex: 1; min-width: 0; display: flex; border-left: 1px solid var(--border); }
.cmp-col { flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }
.cmp-col + .cmp-col { border-left: 1px solid var(--border); }
.cmp-col .card-rollouts { border-left: none; flex: 1; min-height: 0; }
.cmp-model-tag { font-size: 10px; font-weight: 700; letter-spacing: .4px; padding: 3px 10px; border-left: 3px solid; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 0; }
.cmp-model-tag.cmp-a { color: var(--accent); border-left-color: var(--accent); background: rgba(88,166,255,.08); }
.cmp-model-tag.cmp-b { color: var(--purple); border-left-color: var(--purple); background: rgba(188,140,255,.08); }
.cmp-missing { padding: 26px 12px; color: var(--muted); font-size: 12px; text-align: center; }
@media (max-width: 1500px) {
  .cmp-duo { flex-direction: column; }
  .cmp-col + .cmp-col { border-left: none; border-top: 1px solid var(--border); }
}

/* Empty / loading states */
.state-msg { text-align: center; padding: 60px 20px; color: var(--muted); }
.state-msg h2 { font-size: 18px; font-weight: 500; margin-bottom: 8px; color: var(--muted); }
.state-msg p { font-size: 13px; }
.spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #444c56; }

/* Misc */
button { cursor: pointer; }
a { color: var(--accent); text-decoration: none; }
.mono { font-family: var(--font-mono); }
</style>
</head>
<body>
<div id="app">

<!-- ── Sidebar ──────────────────────────────────────────────────────── -->
<aside id="sidebar">
  <div class="sb-header">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2">
      <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>
    </svg>
    <span class="sb-title">EasyR1 Viz</span>
  </div>

  <div class="sb-body">
    <div class="sb-section">
      <span class="sb-label">Results Root</span>
      <div style="display:flex;gap:5px">
        <input class="sb-input" id="root-input" style="flex:1" placeholder="/path/to/results">
        <button class="pg-btn" onclick="loadModels()">Load</button>
      </div>
    </div>

    <div class="sb-section" style="margin-top:6px">
      <span class="sb-label">Model</span>
      <select class="sb-select" id="model-select" onchange="onModelChange()">
        <option value="">— select model —</option>
      </select>
    </div>

    <div class="sb-section" style="margin-top:6px">
      <span class="sb-label">Compare (optional)</span>
      <select class="sb-select" id="compare-select" onchange="onCompareChange()">
        <option value="">— none —</option>
      </select>
    </div>

    <div class="sb-divider"></div>
    <div id="benchmark-list"></div>

    <div class="sb-divider"></div>
    <div id="sb-stats" class="sb-stats"></div>
  </div>
</aside>

<!-- ── Main ─────────────────────────────────────────────────────────── -->
<div id="main">
  <!-- Toolbar -->
  <div id="toolbar">
    <div class="breadcrumb" id="breadcrumb">Select a model and benchmark to begin</div>
    <div id="diff-controls" style="display:none;align-items:center;gap:5px">
      <select class="page-size-sel" id="diff-mode-sel" onchange="onDiffChange()" title="Show only samples where the two models disagree">
        <option value="">diff: off</option>
        <option value="any">diff: any</option>
        <option value="answer">diff: answer</option>
        <option value="grounding">diff: grounding</option>
      </select>
      <input class="sb-input" id="diff-thresh" type="number" min="0" max="1" step="0.05" value="0.3" style="width:58px" onchange="onDiffChange()" title="Grounding diff threshold (|Δ grit_iou / best_iou|)">
    </div>
    <input class="sb-input" id="search-box" placeholder="🔍 filter prompts…" oninput="onSearch()" style="width:160px">
    <select class="page-size-sel" id="page-size-sel" onchange="onPageSizeChange()">
      <option value="10">10 / page</option>
      <option value="20" selected>20 / page</option>
      <option value="50">50 / page</option>
    </select>
    <div class="pagination">
      <button class="pg-btn" id="pg-prev" onclick="goPrev()" disabled>&#8249;</button>
      <span class="pg-info" id="pg-info">—</span>
      <button class="pg-btn" id="pg-next" onclick="goNext()" disabled>&#8250;</button>
    </div>
    <span class="img-status" id="img-status" style="display:none"></span>
  </div>

  <!-- Samples -->
  <div id="samples-wrap">
    <div id="samples-container">
      <div class="state-msg">
        <h2>EasyR1 Evaluation Visualizer</h2>
        <p>Select a model result directory and a benchmark from the sidebar to begin.</p>
      </div>
    </div>
  </div>
</div>
</div><!-- #app -->

<script>
'use strict';

// ── State ────────────────────────────────────────────────────────────────────
const S = {
  modelDir: '',
  modelName: '',
  compareDir: '',
  compareName: '',
  compareBmarkKeys: null,   // Set of benchmark keys the compare model has
  compareBmarkScores: {},   // key -> summary score of compare model
  diffMode: '',             // '' | 'any' | 'answer' | 'grounding'
  diffThreshold: 0.3,
  bmarks: null,             // benchmark list of the primary model
  benchmark: null,     // {key, label, group, count, score?}
  page: 1,
  pageSize: 20,
  totalPages: 0,
  total: 0,
  search: '',
  searchTimer: null,
  imgStatusTimer: null,
};

// ── Color palette for grounding labels ───────────────────────────────────────
const PALETTE = [
  '#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#c77dff',
  '#ff9f43','#54a0ff','#00d2d3','#ff9ff3','#48dbfb',
  '#ff6348','#2ed573','#1e90ff','#a29bfe','#fd79a8',
  '#e17055','#00cec9','#6c5ce7','#fdcb6e','#55efc4',
];
const _colorCache = {};
function labelColor(label) {
  if (!_colorCache[label]) {
    let h = 0;
    for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
    _colorCache[label] = PALETTE[h % PALETTE.length];
  }
  return _colorCache[label];
}
function hexToRgba(hex, a) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── API calls ─────────────────────────────────────────────────────────────────
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ── Sidebar init ──────────────────────────────────────────────────────────────
async function init() {
  // fetch config defaults
  try {
    const cfg = await apiGet('/api/config');
    document.getElementById('root-input').value = cfg.results_dir;
  } catch(e) {}
  // Single delegated listener (attached once — survives innerHTML re-renders)
  document.getElementById('benchmark-list').addEventListener('click', (e) => {
    const item = e.target.closest('.bmark-item');
    if (!item) return;
    const b = _bmarkMap[item.dataset.key];
    if (b) selectBenchmark(b);
  });
  await loadModels();
}

async function loadModels() {
  const root = document.getElementById('root-input').value.trim();
  const models = await apiGet('/api/models?root=' + encodeURIComponent(root));
  const sel = document.getElementById('model-select');
  const cmpSel = document.getElementById('compare-select');
  sel.innerHTML = '<option value="">— select model —</option>';
  cmpSel.innerHTML = '<option value="">— none —</option>';
  for (const m of models) {
    for (const s of [sel, cmpSel]) {
      const opt = document.createElement('option');
      opt.value = m.path;
      opt.textContent = m.name;
      s.appendChild(opt);
    }
  }
  S.compareDir = '';
  S.compareName = '';
  S.compareBmarkKeys = null;
  S.compareBmarkScores = {};
  if (models.length === 1) { sel.value = models[0].path; onModelChange(); }
}

async function onModelChange() {
  const sel = document.getElementById('model-select');
  S.modelDir = sel.value;
  S.modelName = sel.options[sel.selectedIndex]?.text || '';
  S.benchmark = null;
  S.bmarks = null;
  S.page = 1;
  document.getElementById('benchmark-list').innerHTML = '';
  document.getElementById('sb-stats').innerHTML = '';
  if (!S.modelDir) return;

  document.getElementById('benchmark-list').innerHTML =
    '<div style="padding:8px 14px;color:var(--muted);font-size:11px"><span class="spinner"></span> Loading…</div>';

  try {
    const bmarks = await apiGet('/api/benchmarks?model_dir=' + encodeURIComponent(S.modelDir));
    S.bmarks = bmarks;
    renderBenchmarkList(bmarks);
    // Update stats
    const groups = {};
    for (const b of bmarks) { groups[b.group] = (groups[b.group] || 0) + 1; }
    let statsHtml = `<strong style="color:var(--text)">${bmarks.length}</strong> benchmarks`;
    for (const [g, c] of Object.entries(groups)) {
      statsHtml += ` · <span>${g}: ${c}</span>`;
    }
    document.getElementById('sb-stats').innerHTML = statsHtml;
  } catch(e) {
    document.getElementById('benchmark-list').innerHTML =
      `<div style="padding:8px 14px;color:var(--red);font-size:11px">Error: ${e.message}</div>`;
  }
}

// Global benchmark lookup (key -> benchmark object), avoids encoding issues in onclick
const _bmarkMap = {};

function renderBenchmarkList(bmarks) {
  // Populate global map for event delegation
  for (const b of bmarks) _bmarkMap[b.key] = b;

  const container = document.getElementById('benchmark-list');
  const groups = {};
  for (const b of bmarks) (groups[b.group] = groups[b.group] || []).push(b);

  let html = '';
  for (const [grp, items] of Object.entries(groups)) {
    html += `<div class="bmark-group">
      <div class="bmark-group-label">${escHtml(grp)}</div>`;
    for (const b of items) {
      let score = '';
      if (S.compareDir) {
        // Dual score "A → B"; count is dropped to keep the row readable
        const aTxt = b.score != null ? b.score.toFixed(1) : '—';
        const cmpHas = S.compareBmarkKeys && S.compareBmarkKeys.has(b.key);
        const cs = S.compareBmarkScores[b.key];
        const bTxt = cs != null ? cs.toFixed(1) : (cmpHas ? '—' : 'n/a');
        score = `<span class="bmark-score"><span class="sc-a">${aTxt}</span><span class="sc-arrow">→</span><span class="sc-b${cmpHas ? '' : ' na'}">${bTxt}</span></span>`;
      } else if (b.score != null) {
        score = `<span class="bmark-score">${b.score.toFixed(1)}</span>`;
      }
      const count = S.compareDir ? '' : `<span class="bmark-count">${fmtN(b.count)}</span>`;
      // Use data-key + event delegation — avoids JSON-in-attribute quoting bug
      html += `<div class="bmark-item" data-key="${escHtml(b.key)}" title="${escHtml(b.key)}">
        <span class="bmark-name">${escHtml(b.label)}</span>${score}${count}
      </div>`;
    }
    html += '</div>';
  }
  container.innerHTML = html;
  markActiveBenchmark();
}

function markActiveBenchmark() {
  for (const el of document.querySelectorAll('.bmark-item')) {
    el.classList.toggle('active', !!S.benchmark && el.dataset.key === S.benchmark.key);
  }
}

async function selectBenchmark(b) {
  S.benchmark = b;
  S.page = 1;
  S.search = '';
  document.getElementById('search-box').value = '';
  markActiveBenchmark();
  await loadSamples();
}

// ── Compare model ─────────────────────────────────────────────────────────────
async function onCompareChange() {
  const sel = document.getElementById('compare-select');
  S.compareDir = sel.value;
  S.compareName = sel.options[sel.selectedIndex]?.text || '';
  // Diff filter only makes sense in compare mode
  document.getElementById('diff-controls').style.display = S.compareDir ? 'flex' : 'none';
  if (!S.compareDir) {
    S.diffMode = '';
    document.getElementById('diff-mode-sel').value = '';
  }
  S.page = 1;
  await fetchCompareBenchmarks();
  if (S.bmarks) renderBenchmarkList(S.bmarks);
  if (S.modelDir && S.benchmark) loadSamples();
}

function onDiffChange() {
  S.diffMode = document.getElementById('diff-mode-sel').value;
  const input = document.getElementById('diff-thresh');
  const t = parseFloat(input.value);
  S.diffThreshold = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0.3;
  input.value = S.diffThreshold;
  S.page = 1;
  if (S.modelDir && S.benchmark) loadSamples();
}

async function fetchCompareBenchmarks() {
  S.compareBmarkKeys = null;
  S.compareBmarkScores = {};
  if (!S.compareDir) return;
  try {
    const bmarks = await apiGet('/api/benchmarks?model_dir=' + encodeURIComponent(S.compareDir));
    S.compareBmarkKeys = new Set(bmarks.map(b => b.key));
    for (const b of bmarks) {
      if (b.score != null) S.compareBmarkScores[b.key] = b.score;
    }
  } catch(e) {
    S.compareBmarkKeys = new Set();
  }
}

// ── Search & pagination ───────────────────────────────────────────────────────
function onSearch() {
  clearTimeout(S.searchTimer);
  S.searchTimer = setTimeout(() => {
    S.search = document.getElementById('search-box').value.trim();
    S.page = 1;
    loadSamples();
  }, 400);
}

function onPageSizeChange() {
  S.pageSize = parseInt(document.getElementById('page-size-sel').value);
  S.page = 1;
  loadSamples();
}

function goPrev() { if (S.page > 1) { S.page--; loadSamples(); } }
function goNext() { if (S.page < S.totalPages) { S.page++; loadSamples(); } }

function fmtN(n) {
  if (n >= 1000) return (n/1000).toFixed(1)+'k';
  return String(n);
}

// ── Sample loading ────────────────────────────────────────────────────────────
async function loadSamples() {
  if (!S.modelDir || !S.benchmark) return;
  const cont = document.getElementById('samples-container');
  cont.innerHTML = '<div class="state-msg"><span class="spinner"></span> Loading…</div>';

  try {
    let url = `/api/samples?model_dir=${encodeURIComponent(S.modelDir)}&benchmark=${encodeURIComponent(S.benchmark.key)}&page=${S.page}&size=${S.pageSize}&search=${encodeURIComponent(S.search)}&compare_dir=${encodeURIComponent(S.compareDir || '')}`;
    if (S.compareDir && S.diffMode) {
      url += `&diff_mode=${encodeURIComponent(S.diffMode)}&diff_threshold=${S.diffThreshold}`;
    }
    const data = await apiGet(url);

    S.total = data.total;
    S.totalPages = data.pages;
    S.page = data.page;

    const compareActive = !!S.compareDir && data.compare_available !== false;
    updateToolbar(data);
    updateImageStatus(data.image_status, S.benchmark.key);
    renderSamples(data.samples, compareActive);
    startImageStatusPolling(data.image_status, S.benchmark.key);
  } catch(e) {
    cont.innerHTML = `<div class="state-msg"><p style="color:var(--red)">Error: ${e.message}</p></div>`;
  }
}

function updateToolbar(data) {
  const bc = document.getElementById('breadcrumb');
  const bname = S.benchmark ? S.benchmark.label : '';
  const bkey = S.benchmark ? S.benchmark.key : '';
  let modelHtml;
  if (S.compareDir) {
    modelHtml = `<span class="model-a">${escHtml(S.modelName)}</span>
      <span class="vs-sep">vs</span>
      <span class="model-b">${escHtml(S.compareName)}</span>`;
  } else {
    modelHtml = `<span class="model-name">${escHtml(S.modelName)}</span>`;
  }
  let warn = '';
  if (S.compareDir && data.compare_available === false) {
    warn = `<span class="sep">·</span><span style="color:var(--orange)">compare model has no results for this benchmark</span>`;
  }
  const countHtml = data.diff_active
    ? `<span style="color:var(--orange)">${fmtN(data.total)} diff</span><span style="color:var(--muted)"> / ${fmtN(data.unfiltered_total)} groups</span>`
    : `<span style="color:var(--muted)">${fmtN(data.total)} groups</span>`;
  bc.innerHTML = `${modelHtml}
    <span class="sep">›</span>
    <span class="bmark-name-bc" title="${escHtml(bkey)}">${escHtml(bname)}</span>
    <span class="sep">·</span>
    ${countHtml}${warn}`;

  document.getElementById('pg-info').textContent = `${data.page} / ${data.pages}`;
  document.getElementById('pg-prev').disabled = data.page <= 1;
  document.getElementById('pg-next').disabled = data.page >= data.pages;
}

function updateImageStatus(status, bmark) {
  const el = document.getElementById('img-status');
  if (status === 'not_started' || status === 'done' && !status.startsWith('error')) {
    if (status === 'done') {
      el.style.display = 'inline-block';
      el.className = 'img-status done';
      el.textContent = '✓ images loaded';
      setTimeout(() => { el.style.display = 'none'; }, 3000);
    } else {
      el.style.display = 'none';
    }
  } else if (status === 'loading') {
    el.style.display = 'inline-block';
    el.className = 'img-status loading';
    el.textContent = '⏳ loading images…';
  } else if (status && status.startsWith('error')) {
    el.style.display = 'inline-block';
    el.className = 'img-status error';
    el.textContent = '✗ images unavailable';
  } else {
    el.style.display = 'none';
  }
}

function startImageStatusPolling(status, bmark) {
  clearInterval(S.imgStatusTimer);
  if (status === 'loading') {
    S.imgStatusTimer = setInterval(async () => {
      try {
        const d = await apiGet(`/api/image_status?benchmark=${encodeURIComponent(bmark)}`);
        updateImageStatus(d.status, bmark);
        if (d.status !== 'loading') {
          clearInterval(S.imgStatusTimer);
          // Reload current page to get images
          if (d.status === 'done') loadSamples();
        }
      } catch(e) { clearInterval(S.imgStatusTimer); }
    }, 2000);
  }
}

// ── Sample rendering ──────────────────────────────────────────────────────────
function renderSamples(samples, compareActive) {
  const cont = document.getElementById('samples-container');
  if (!samples || samples.length === 0) {
    cont.innerHTML = '<div class="state-msg"><h2>No results</h2><p>Try adjusting the search or diff filter.</p></div>';
    return;
  }
  cont.innerHTML = '';
  for (const s of samples) {
    cont.appendChild(buildSampleCard(s, compareActive));
  }
}

function buildSampleCard(sample, compareActive) {
  const card = document.createElement('div');
  card.className = 'sample-card';

  // ── Header ──
  const header = document.createElement('div');
  header.className = 'card-header';

  const sid = document.createElement('span');
  sid.className = 'card-sid';
  sid.textContent = sample.sample_id;

  const prompt = document.createElement('span');
  prompt.className = 'card-prompt';
  const promptFull = sample.prompt;
  const promptFlat = promptFull.replace(/\n+/g, ' ');
  if (promptFlat.length > 300) {
    prompt.classList.add('expandable');
    const promptText = document.createTextNode('');
    const promptToggle = document.createElement('span');
    promptToggle.className = 'prompt-toggle';
    prompt.append(promptText, promptToggle);
    let promptExpanded = false;
    const renderPrompt = () => {
      promptText.textContent = promptExpanded ? promptFull + ' ' : promptFlat.slice(0, 300) + '… ';
      promptToggle.textContent = promptExpanded ? 'less ▴' : 'more ▾';
      prompt.classList.toggle('expanded', promptExpanded);
      prompt.title = promptExpanded ? 'Click to collapse' : 'Click to expand';
    };
    prompt.onclick = () => { promptExpanded = !promptExpanded; renderPrompt(); };
    renderPrompt();
  } else {
    prompt.textContent = promptFlat;
  }

  const target = document.createElement('span');
  target.className = 'card-target';
  target.title = 'Ground truth';
  const gtBbox = sample.extra_info && Array.isArray(sample.extra_info.bbox_1000)
    ? sample.extra_info.bbox_1000 : null;
  const gtRegions = sample.gt_regions || [];
  const gtBoxCount = gtRegions.reduce((count, region) => count + (region.bbox_list || []).length, 0);
  if (gtBbox) {
    target.textContent = 'GT: [' + gtBbox.join(', ') + ']';
  } else if (gtBoxCount) {
    const answerText = sample.target ? String(sample.target).slice(0, 60) + ' · ' : '';
    const boxText = gtBoxCount + ' box' + (gtBoxCount > 1 ? 'es' : '');
    target.textContent = 'GT: ' + answerText + boxText;
  } else {
    target.textContent = 'GT: ' + String(sample.target).slice(0, 80);
  }

  header.append(sid, prompt, target);
  card.appendChild(header);

  if (compareActive) {
    card.appendChild(buildCompareBody(sample));
    appendCompareMetricsRow(card, sample.metrics, sample.compare ? sample.compare.metrics : null);
    return card;
  }

  // ── Body (single-model) ──
  const body = document.createElement('div');
  body.className = 'card-body';

  // Images column
  const imgCol = buildImagesColumn(sample);
  body.appendChild(imgCol);

  // Rollouts column
  const rolloutCol = buildRolloutsColumn(sample, imgCol);
  body.appendChild(rolloutCol);

  card.appendChild(body);
  appendMetricsRow(card, sample.metrics);
  return card;
}

// ── Compare-mode card body ────────────────────────────────────────────────────
function cmpModelTag(side, name) {
  const el = document.createElement('div');
  el.className = 'cmp-model-tag cmp-' + side;
  el.textContent = (side === 'a' ? 'A · ' : 'B · ') + name;
  el.title = name;
  return el;
}

function buildCompareBody(sample) {
  const body = document.createElement('div');
  body.className = 'card-body';
  const hasCmp = !!sample.compare;

  // Left: stacked per-model annotated copies of the same image(s)
  const imgStack = document.createElement('div');
  imgStack.className = 'cmp-img-stack';
  const imgColA = buildImagesColumn(sample);
  imgStack.append(cmpModelTag('a', S.modelName), imgColA);
  let imgColB = null;
  if (hasCmp) {
    imgColB = buildImagesColumn(sample);
    imgStack.append(cmpModelTag('b', S.compareName), imgColB);
  }
  body.appendChild(imgStack);

  // Right: side-by-side response columns, each wired to its own image copy
  const duo = document.createElement('div');
  duo.className = 'cmp-duo';

  const colA = document.createElement('div');
  colA.className = 'cmp-col';
  colA.append(cmpModelTag('a', S.modelName), buildRolloutsColumn(sample, imgColA));

  const colB = document.createElement('div');
  colB.className = 'cmp-col';
  colB.appendChild(cmpModelTag('b', S.compareName));
  if (hasCmp) {
    const sampleB = { ...sample, rollouts: sample.compare.rollouts, metrics: sample.compare.metrics || {} };
    colB.appendChild(buildRolloutsColumn(sampleB, imgColB));
  } else {
    const miss = document.createElement('div');
    miss.className = 'cmp-missing';
    miss.textContent = 'This sample was not evaluated by the compare model.';
    colB.appendChild(miss);
  }
  duo.append(colA, colB);
  body.appendChild(duo);
  return body;
}

// ── Metrics rows ──────────────────────────────────────────────────────────────
const METRIC_SKIP = new Set(['sample_id', 'has_valid_gt', 'malformed_gt', 'scores']);
const METRIC_ORDER = ['answer_correct', 'answer_relaxed_correct', 'mean_accuracy',
  'grit_iou', 'best_iou', 'box_f1_at_0_5', 'box_precision_at_0_5', 'box_recall_at_0_5',
  'matched_mean_iou', 'pred_count', 'gt_count', 'correct'];
const METRIC_HIGHER_BETTER = new Set(['answer_correct', 'answer_relaxed_correct', 'mean_accuracy',
  'grit_iou', 'box_f1_at_0_5', 'box_precision_at_0_5', 'box_recall_at_0_5', 'matched_mean_iou',
  'best_iou', 'correct']);
const METRIC_BOOL_COLOR = new Set(['answer_correct', 'answer_relaxed_correct', 'correct']);

function sortMetricKeys(keys) {
  return [...keys].sort((a,b) => {
    const ai = METRIC_ORDER.indexOf(a), bi = METRIC_ORDER.indexOf(b);
    if (ai >= 0 && bi < 0) return -1;
    if (bi >= 0 && ai < 0) return 1;
    if (ai >= 0 && bi >= 0) return ai - bi;
    return a.localeCompare(b);
  });
}

function fmtMetricValue(v) {
  if (v === undefined || v === null) return '—';
  if (typeof v === 'boolean') return v ? '✓' : '✗';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

function appendMetricsRow(card, metrics) {
  const m = metrics || {};
  const keys = Object.keys(m).filter(k => !METRIC_SKIP.has(k));
  if (keys.length === 0) return;
  const mrow = document.createElement('div');
  mrow.className = 'metrics-row';
  for (const k of sortMetricKeys(keys).slice(0, 12)) {
    const v = m[k];
    const color = (k === 'answer_correct' || k === 'correct') ? (v ? 'var(--green)' : 'var(--red)') : '';
    mrow.innerHTML += `<span class="metric-item">${escHtml(k.replace(/^.*\//,''))}: <span style="${color ? 'color:'+color : ''}">${escHtml(fmtMetricValue(v))}</span></span>`;
  }
  card.appendChild(mrow);
}

function appendCompareMetricsRow(card, metricsA, metricsB) {
  const mA = metricsA || {}, mB = metricsB || {};
  const keys = [...new Set([...Object.keys(mA), ...Object.keys(mB)])].filter(k => !METRIC_SKIP.has(k));
  if (keys.length === 0) return;
  const boolStyle = v => (v === undefined || v === null) ? '' : (v ? 'color:var(--green)' : 'color:var(--red)');
  const mrow = document.createElement('div');
  mrow.className = 'metrics-row';
  for (const k of sortMetricKeys(keys).slice(0, 12)) {
    const va = mA[k], vb = mB[k];
    let styleA = '', styleB = '';
    if (METRIC_BOOL_COLOR.has(k) && (typeof va === 'boolean' || typeof vb === 'boolean')) {
      styleA = boolStyle(va);
      styleB = boolStyle(vb);
    }
    let delta = '';
    if (typeof va === 'number' && typeof vb === 'number' && METRIC_HIGHER_BETTER.has(k)) {
      const d = vb - va;
      if (Number.isFinite(d) && Math.abs(d) > 1e-9) {
        delta = ` <span class="metric-delta ${d > 0 ? 'pos' : 'neg'}">(${d > 0 ? '+' : ''}${d.toFixed(3)})</span>`;
      }
    }
    mrow.innerHTML += `<span class="metric-item">${escHtml(k.replace(/^.*\//,''))}: `
      + `<span style="${styleA}">${escHtml(fmtMetricValue(va))}</span>`
      + `<span class="cmp-arrow">→</span>`
      + `<span style="${styleB}">${escHtml(fmtMetricValue(vb))}</span>${delta}</span>`;
  }
  card.appendChild(mrow);
}

// ── Images column ──────────────────────────────────────────────────────────────
function buildImagesColumn(sample) {
  const col = document.createElement('div');
  col.className = 'card-images';
  col.dataset.sampleId = sample.sample_id;

  if (!sample.image_urls || sample.image_urls.length === 0) {
    const no = document.createElement('div');
    no.className = 'no-img';
    no.textContent = 'no image';
    col.appendChild(no);
    return col;
  }

  for (let imgIdx = 0; imgIdx < sample.image_urls.length; imgIdx++) {
    const url = sample.image_urls[imgIdx];
    const wrapper = document.createElement('div');
    wrapper.className = 'img-wrapper';
    wrapper.dataset.imgIdx = imgIdx;

    wrapper.addEventListener('click', () => {
      const img = wrapper.querySelector('img');
      if (img && img.src) openImageModal(img.src, col._activeGroundings || [], col._gtRegions || [], imgIdx);
    });

    const img = document.createElement('img');
    img.style.display = 'block';
    if (url) {
      img.src = url;
      img.onerror = () => { img.alt = '[image unavailable]'; img.style.color='var(--muted)'; };
    } else {
      img.alt = '[image loading…]';
    }

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    img.title = 'Click to enlarge';

    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = img.width + 'px';
      canvas.style.height = img.height + 'px';
      // Resize observer to keep canvas in sync
      const ro = new ResizeObserver(() => {
        canvas.style.width = img.offsetWidth + 'px';
        canvas.style.height = img.offsetHeight + 'px';
        redrawCanvas(canvas, img, col._activeGroundings || [], col._gtRegions || [], imgIdx);
      });
      ro.observe(img);
      redrawCanvas(canvas, img, col._activeGroundings || [], col._gtRegions || [], imgIdx);
    };

    if (sample.image_urls.length > 1) {
      const lbl = document.createElement('div');
      lbl.className = 'img-label';
      lbl.textContent = `Image ${imgIdx}`;
      wrapper.append(img, canvas, lbl);
    } else {
      wrapper.append(img, canvas);
    }
    col.appendChild(wrapper);
  }

  col._activeGroundings = [];
  col._gtRegions = sample.gt_regions || [];

  col.setGroundings = (groundings) => {
    col._activeGroundings = groundings;
    for (const wrapper of col.querySelectorAll('.img-wrapper')) {
      const imgIdx = parseInt(wrapper.dataset.imgIdx);
      const canvas = wrapper.querySelector('canvas');
      const img = wrapper.querySelector('img');
      if (canvas && img && img.complete && img.naturalWidth) {
        redrawCanvas(canvas, img, groundings, col._gtRegions, imgIdx);
      }
    }
  };

  return col;
}

function redrawCanvas(canvas, img, groundings, gtRegions, imgIdx) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const lw = Math.max(1.5, W / 400);

  // Draw every GT box as a dashed lime region.
  for (const region of (gtRegions || [])) {
    if ((region.image_idx || 0) !== imgIdx) continue;
    for (const box of (region.bbox_list || [])) {
      if (!Array.isArray(box) || box.length < 4) continue;
      const [gx1, gy1, gx2, gy2] = box;
      const rx1 = gx1/1000*W, ry1 = gy1/1000*H, rw = (gx2-gx1)/1000*W, rh = (gy2-gy1)/1000*H;
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = '#39d353';
      ctx.lineWidth = lw + 1;
      ctx.strokeRect(rx1, ry1, rw, rh);
      ctx.restore();
      const fs = Math.max(10, Math.min(13, W/32));
      ctx.font = `bold ${fs}px sans-serif`;
      const lbl = region.label || 'GT';
      const tw = ctx.measureText(lbl).width;
      const lx = Math.min(rx1, W - tw - 6);
      const ly = ry1 > fs + 4 ? ry1 - fs - 2 : ry1 + rh + 2;
      ctx.fillStyle = 'rgba(57,211,83,0.9)';
      ctx.fillRect(lx, ly, tw + 6, fs + 4);
      ctx.fillStyle = '#000';
      ctx.fillText(lbl, lx + 3, ly + fs);
    }
  }

  if (!groundings) return;

  for (const det of groundings) {
    if (det.image_idx !== imgIdx) continue;
    const color = det.source === 'answer' ? '#ff9f43' : labelColor(det.label);
    for (const box of det.bbox_list) {
      if (!Array.isArray(box) || box.length < 4) continue;
      const [x1, y1, x2, y2] = box;
      const rx1 = x1/1000*W, ry1 = y1/1000*H, rw = (x2-x1)/1000*W, rh = (y2-y1)/1000*H;
      // Fill
      ctx.fillStyle = hexToRgba(color, 0.18);
      ctx.fillRect(rx1, ry1, rw, rh);
      // Stroke
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.setLineDash([]);
      ctx.strokeRect(rx1, ry1, rw, rh);
      // Label
      const lbl = det.label;
      const fs = Math.max(10, Math.min(14, W/30));
      ctx.font = `bold ${fs}px sans-serif`;
      const tw = ctx.measureText(lbl).width;
      const lx = Math.min(rx1, W - tw - 6);
      const ly = ry1 > fs + 4 ? ry1 - fs - 2 : ry1 + rh + 2;
      ctx.fillStyle = hexToRgba(color, 0.9);
      ctx.fillRect(lx, ly, tw + 6, fs + 4);
      ctx.fillStyle = '#fff';
      ctx.fillText(lbl, lx + 3, ly + fs);
    }
  }
}

// ── Rollouts column ────────────────────────────────────────────────────────────
function responseCorrectness(sample, responseIndex, answer) {
  const metrics = sample.metrics || {};
  if (Array.isArray(metrics.scores) && responseIndex < metrics.scores.length) {
    return Number(metrics.scores[responseIndex]) >= 1;
  }
  if ('answer_correct' in metrics) return Boolean(metrics.answer_correct);
  if ('correct' in metrics) return Boolean(metrics.correct);
  if ((sample.gt_regions || []).length > 0) return null;
  const target = String(sample.target || '').trim().toLowerCase();
  if (!target) return null;
  return String(answer || '').trim().toLowerCase() === target;
}

function buildRolloutsColumn(sample, imgCol) {
  const col = document.createElement('div');
  col.className = 'card-rollouts';

  const rollouts = sample.rollouts || [];
  const numRollouts = rollouts.reduce((s, r) => s + (r.responses || []).length, 0);

  // Flatten all individual responses with rollout/response indices
  const allResponses = [];
  for (let ri = 0; ri < rollouts.length; ri++) {
    for (let pi = 0; pi < (rollouts[ri].responses || []).length; pi++) {
      allResponses.push({ rolloutIdx: ri, respIdx: pi, data: rollouts[ri].responses[pi] });
    }
  }

  // Tab bar
  const tabs = document.createElement('div');
  tabs.className = 'rollout-tabs';

  // Content panes
  const panes = document.createElement('div');
  panes.style.flex = '1';
  panes.style.overflow = 'hidden';
  panes.style.display = 'flex';
  panes.style.flexDirection = 'column';

  const contents = [];

  for (let i = 0; i < allResponses.length; i++) {
    const { rolloutIdx, respIdx, data } = allResponses[i];

    // Determine correctness badge
    const ans = data.answer || '';
    const correct = responseCorrectness(sample, i, ans);
    const badge = correct !== null
      ? `<span class="tab-badge ${correct ? 'correct' : 'wrong'}">${correct ? '✓' : '✗'}</span>`
      : '';

    const tabLabel = allResponses.length === 1 ? 'Response' : `R${i+1}`;
    const rolloutMeta = sample.rollouts[allResponses[i].rolloutIdx] || {};
    const rmeta = (rolloutMeta.metadata || [])[allResponses[i].respIdx] || {};
    const tokCount = rmeta.token_count ? ` <span style="color:var(--muted);font-size:9px">${rmeta.token_count}tok</span>` : '';
    const tab = document.createElement('div');
    tab.className = 'rollout-tab' + (i === 0 ? ' active' : '');
    tab.innerHTML = `${tabLabel}${badge}${tokCount}`;
    tab.dataset.idx = i;
    tabs.appendChild(tab);

    const content = document.createElement('div');
    content.className = 'rollout-content' + (i === 0 ? ' active' : '');
    content.dataset.idx = i;
    contents.push(content);
    panes.appendChild(content);

    // Build content
    buildResponseContent(content, data, sample, imgCol, i === 0, i);
  }

  // Tab click handlers
  tabs.addEventListener('click', (e) => {
    const tab = e.target.closest('.rollout-tab');
    if (!tab) return;
    const idx = parseInt(tab.dataset.idx);
    for (const t of tabs.querySelectorAll('.rollout-tab')) t.classList.remove('active');
    for (const c of contents) c.classList.remove('active');
    tab.classList.add('active');
    contents[idx].classList.add('active');
    // Update grounging on images
    const data = allResponses[idx].data;
    imgCol.setGroundings && imgCol.setGroundings(getDisplayGroundings(data, sample));
  });

  // Init image grounging for first tab
  if (allResponses.length > 0) {
    const first = allResponses[0].data;
    const firstGroundings = getDisplayGroundings(first, sample);
    imgCol._activeGroundings = firstGroundings;
    imgCol.setGroundings && imgCol.setGroundings(firstGroundings);
  }

  col.append(tabs, panes);
  return col;
}

function buildResponseContent(el, respData, sample, imgCol, isActive, responseIndex) {
  const think = respData.think || '';
  const answer = respData.answer || '';
  const groundings = getDisplayGroundings(respData, sample);

  // Think section
  if (think) {
    const sec = document.createElement('div');
    sec.className = 'resp-section';
    const lbl = document.createElement('div');
    lbl.className = 'resp-section-label';
    lbl.textContent = 'Reasoning';

    const box = document.createElement('div');
    box.className = 'think-box';
    box.textContent = think;

    const toggle = document.createElement('span');
    toggle.className = 'think-toggle';
    toggle.textContent = '▲ collapse';
    toggle.onclick = () => {
      const collapsed = box.classList.toggle('collapsed');
      toggle.textContent = collapsed ? '▼ expand' : '▲ collapse';
    };

    sec.append(lbl, box, toggle);
    el.appendChild(sec);
  }

  // Answer section
  const answerSec = document.createElement('div');
  answerSec.className = 'resp-section';
  const answerLbl = document.createElement('div');
  answerLbl.className = 'resp-section-label';
  answerLbl.textContent = 'Answer';
  const answerBox = document.createElement('div');
  answerBox.className = 'answer-box';
  answerBox.textContent = answer || '(no answer parsed)';
  answerSec.append(answerLbl, answerBox);

  // Verdict
  const tgt = String(sample.target || '').trim();
  const ei = sample.extra_info || {};
  const gtBbox1000 = Array.isArray(ei.bbox_1000) ? ei.bbox_1000
    : (Array.isArray(ei.bbox) && sample.metrics && 'best_iou' in sample.metrics ? ei.bbox.map(v => v * 1000) : null);
  const isRefcoco = gtBbox1000 !== null;

  if (tgt || isRefcoco || (sample.metrics && 'grit_iou' in sample.metrics)) {
    const verdict = document.createElement('div');
    verdict.className = 'verdict';
    const metrics = sample.metrics || {};

    if (isRefcoco) {
      // RefCOCO: bbox IoU verdict
      let iou = null, correct = null;
      if ('best_iou' in metrics) {
        iou = metrics.best_iou;
        correct = metrics.correct;
      } else {
        // Compute client-side from answer bbox
        const predBbox = extractBboxFromText(answer);
        if (predBbox && gtBbox1000) {
          iou = computeIoU(predBbox, gtBbox1000);
          correct = iou >= 0.5;
        }
      }
      if (iou !== null) {
        const iouStr = iou.toFixed(4);
        const cls = correct ? 'verdict-correct' : 'verdict-wrong';
        const mark = correct ? '✓' : '✗';
        const thresh = correct ? '' : ' <span style="color:var(--muted)">(< 0.5)</span>';
        verdict.innerHTML = `<span class="${cls}">${mark} IoU: ${iouStr}${thresh}</span>`;
      } else {
        verdict.innerHTML = '<span style="color:var(--muted)">IoU: —</span>';
      }
    } else if (Array.isArray(metrics.scores)) {
      const correct = responseCorrectness(sample, responseIndex, answer);
      if (correct) {
        verdict.innerHTML = '<span class="verdict-correct">✓ Correct</span>';
      } else {
        verdict.innerHTML = `<span class="verdict-wrong">✗ Expected: ${escHtml(tgt.slice(0,80))}</span>`;
      }
    } else if ('grit_iou' in metrics && !('answer_correct' in metrics)) {
      const iou = Number(metrics.grit_iou);
      const boxF1 = Number(metrics.box_f1_at_0_5);
      const iouText = Number.isFinite(iou) ? iou.toFixed(4) : '—';
      const f1Text = Number.isFinite(boxF1) ? ` · Box F1@0.5: ${boxF1.toFixed(4)}` : '';
      verdict.innerHTML = `<span style="color:var(--orange)">Grounding IoU: ${iouText}${f1Text}</span>`;
    } else if ('answer_correct' in metrics) {
      // Use authoritative metric
      if (metrics.answer_correct) {
        verdict.innerHTML = '<span class="verdict-correct">✓ Correct</span>';
      } else {
        verdict.innerHTML = `<span class="verdict-wrong">✗ Expected: ${escHtml(tgt.slice(0,80))}</span>`;
      }
    } else {
      // Fallback string comparison
      const correct = tgt && answer.trim().toLowerCase() === tgt.toLowerCase();
      if (correct) {
        verdict.innerHTML = '<span class="verdict-correct">✓ Correct</span>';
      } else {
        verdict.innerHTML = `<span class="verdict-wrong">✗ Expected: ${escHtml(tgt.slice(0,80))}</span>`;
      }
    }
    answerSec.appendChild(verdict);
  }
  el.appendChild(answerSec);

  // Grounding chips
  if (groundings.length > 0) {
    const gndSec = document.createElement('div');
    gndSec.className = 'gnd-section';
    const gndLbl = document.createElement('div');
    gndLbl.className = 'gnd-label';
    gndLbl.textContent = `Grounding (${groundings.length})`;
    const chips = document.createElement('div');
    chips.className = 'gnd-chips';
    for (const g of groundings) {
      const chip = document.createElement('span');
      chip.className = 'gnd-chip';
      const chipColor = g.source === 'answer' ? '#ff9f43' : labelColor(g.label);
      chip.style.background = hexToRgba(chipColor, 0.85);
      const nb = g.bbox_list.length;
      const imgTxt = sample.image_urls && sample.image_urls.length > 1 ? ` img${g.image_idx}` : '';
      chip.title = `${g.label}${imgTxt} · ${nb} box${nb>1?'es':''}`;
      chip.textContent = `${g.label}${imgTxt}`;
      chips.appendChild(chip);
    }
    gndSec.append(gndLbl, chips);
    el.appendChild(gndSec);

    // On hover, highlight on image
    chips.addEventListener('mouseover', (e) => {
      const chip = e.target.closest('.gnd-chip');
      if (!chip) return;
      const idx = Array.from(chips.children).indexOf(chip);
      if (idx >= 0) {
        imgCol.setGroundings && imgCol.setGroundings([groundings[idx]]);
      }
    });
    chips.addEventListener('mouseleave', () => {
      imgCol.setGroundings && imgCol.setGroundings(groundings);
    });
  }

  // Activate grounging when this pane becomes visible
  if (isActive) {
    imgCol._activeGroundings = groundings;
  }
}

function getDisplayGroundings(respData, sample) {
  const groundings = cloneGroundings(respData.groundings || []);
  if ((sample.gt_regions || []).length === 0) return groundings;

  const answerBbox = extractBboxFromText(respData.answer || '');
  if (!answerBbox) return groundings;

  const hasAnswerBox = groundings.some(g =>
    (g.image_idx || 0) === 0 && (g.bbox_list || []).some(box => boxesClose(box, answerBbox))
  );
  if (!hasAnswerBox) {
    groundings.push({
      label: 'answer bbox',
      bbox_list: [answerBbox],
      image_idx: 0,
      fmt: 'answer',
      source: 'answer',
    });
  }
  return groundings;
}

function cloneGroundings(groundings) {
  return groundings.map(g => ({
    ...g,
    bbox_list: (g.bbox_list || []).map(box => Array.isArray(box) ? [...box] : box),
  }));
}

function boxesClose(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length < 4 || b.length < 4) return false;
  const tol = 1;
  for (let i = 0; i < 4; i++) {
    if (Math.abs(Number(a[i]) - Number(b[i])) > tol) return false;
  }
  return true;
}

function openImageModal(src, groundings, gtRegions, imgIdx) {
  let modal = document.getElementById('image-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'image-modal';
    modal.className = 'image-modal';
    modal.innerHTML = '<button class="image-modal-close" aria-label="Close">×</button><div class="image-modal-stage"><div class="image-modal-scene"><img draggable="false"><canvas></canvas></div></div>';
    document.body.appendChild(modal);
    modal._zoom = { scale: 1, x: 0, y: 0, dragging: false, lastX: 0, lastY: 0 };

    modal.addEventListener('click', (e) => {
      if (e.target === modal || e.target.closest('.image-modal-close')) {
        modal.classList.remove('open');
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') modal.classList.remove('open');
    });

    const stage = modal.querySelector('.image-modal-stage');
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      const z = modal._zoom;
      const rect = stage.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const oldScale = z.scale;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const nextScale = Math.min(8, Math.max(0.25, oldScale * factor));
      const imageX = (mx - z.x) / oldScale;
      const imageY = (my - z.y) / oldScale;
      z.scale = nextScale;
      z.x = mx - imageX * nextScale;
      z.y = my - imageY * nextScale;
      applyModalTransform(modal);
    }, { passive: false });

    stage.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const z = modal._zoom;
      z.dragging = true;
      z.lastX = e.clientX;
      z.lastY = e.clientY;
      stage.classList.add('dragging');
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener('pointermove', (e) => {
      const z = modal._zoom;
      if (!z.dragging) return;
      z.x += e.clientX - z.lastX;
      z.y += e.clientY - z.lastY;
      z.lastX = e.clientX;
      z.lastY = e.clientY;
      applyModalTransform(modal);
    });
    stage.addEventListener('pointerup', (e) => endModalDrag(modal, e));
    stage.addEventListener('pointercancel', (e) => endModalDrag(modal, e));
  }

  const stage = modal.querySelector('.image-modal-stage');
  const scene = modal.querySelector('.image-modal-scene');
  const img = modal.querySelector('img');
  const canvas = modal.querySelector('canvas');
  modal._zoom = { scale: 1, x: 0, y: 0, dragging: false, lastX: 0, lastY: 0 };
  stage.classList.remove('dragging');
  scene.style.transform = 'translate(0px, 0px) scale(1)';

  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = img.width + 'px';
    canvas.style.height = img.height + 'px';
    scene.style.width = img.width + 'px';
    scene.style.height = img.height + 'px';
    const rect = stage.getBoundingClientRect();
    modal._zoom.x = Math.max(0, (rect.width - img.width) / 2);
    modal._zoom.y = Math.max(0, (rect.height - img.height) / 2);
    applyModalTransform(modal);
    redrawCanvas(canvas, img, groundings || [], gtRegions || [], imgIdx);
  };
  modal.classList.add('open');
  img.src = src;
  if (img.complete && img.naturalWidth) img.onload();
}

function applyModalTransform(modal) {
  const z = modal._zoom || { scale: 1, x: 0, y: 0 };
  const scene = modal.querySelector('.image-modal-scene');
  if (scene) scene.style.transform = `translate(${z.x}px, ${z.y}px) scale(${z.scale})`;
}

function endModalDrag(modal, e) {
  const z = modal._zoom;
  if (!z || !z.dragging) return;
  z.dragging = false;
  const stage = modal.querySelector('.image-modal-stage');
  if (stage) {
    stage.classList.remove('dragging');
    try { stage.releasePointerCapture(e.pointerId); } catch(err) {}
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function computeIoU(a, b) {
  // Both [x1,y1,x2,y2] in same coordinate space
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const ua = (a[2]-a[0]) * (a[3]-a[1]);
  const ub = (b[2]-b[0]) * (b[3]-b[1]);
  const union = ua + ub - inter;
  return union > 0 ? inter / union : 0;
}

function extractBboxFromText(text) {
  // Try direct JSON parse [x1,y1,x2,y2]
  try {
    const v = JSON.parse(text.trim());
    if (Array.isArray(v) && v.length === 4 && v.every(x => typeof x === 'number')) return v;
  } catch(e) {}
  // Regex fallback: first occurrence of [n, n, n, n]
  const m = text.match(/\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]/);
  if (m) return [parseFloat(m[1]), parseFloat(m[2]), parseFloat(m[3]), parseFloat(m[4])];
  return null;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ── Boot ──────────────────────────────────────────────────────────────────────
init();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS_DIR), help="Root directory containing model result folders.")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Evaluation data root (default: $EVAL_DATA_ROOT, $DATA_ROOT/eval or <repo>/data/eval).",
    )
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Bind address (default 127.0.0.1). Use 0.0.0.0 only on a trusted network."
    )
    args = parser.parse_args()

    _cfg["results_dir"] = str(Path(args.results).resolve())
    _cfg["data_root"] = str(Path(args.data_root).resolve())

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"WARNING: binding to {args.host}; anyone who can reach this port can browse the results.")
    print("Evaluation Visualization Server")
    print(f"  Results dir : {_cfg['results_dir']}")
    print(f"  Data root   : {_cfg['data_root']}")
    print(f"  Listening   : http://{args.host}:{args.port}")
    print()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
