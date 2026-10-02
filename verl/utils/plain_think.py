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
"""Tokenize ``<think>`` / ``</think>`` as plain text for models that never trained them.

Qwen3-VL *Instruct* checkpoints ship ``<think>`` and ``</think>`` as added tokens (ids 151667 and
151668) but were never trained on them: their chat template never emits them, unlike the Thinking
checkpoints. A format prompt that asks for ``<think> ... </think>`` is then encoded with two
untrained tokens, and the policy does not follow it (on ViRL39K with the comparison prompt the
format reward of Qwen3-VL-4B-Instruct drops from 0.47 to 0.00, accuracy is unchanged).

``plain_think_tokenizer_path`` returns a tokenizer directory in which the two tokens are removed
from the added tokens, so they are tokenized as ordinary text, or ``None`` when nothing has to be
changed. The directory is derived from the tokenizer as it is normally loaded (Hugging Face hub,
ModelScope via ``USE_MODELSCOPE_HUB=1``, or a local path) and cached under
``$PARLVR_CACHE_DIR/tokenizers`` (default ``~/.cache/parlvr/tokenizers``); model weights and every
downloaded file stay untouched.

Settings (``worker.actor.model.plain_think_tokens``):
  auto   apply when the tokenizer has the tokens but the chat template never uses them (default)
  true   always remove the tokens when present
  false  keep the tokenizer as released
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional


THINK_TOKENS = ("<think>", "</think>")
CHOICES = ("auto", "true", "false")
_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off", "none", "null"}
_RESOLVED: dict[tuple[str, str], Optional[str]] = {}
_COMPLETE_MARKER = ".plain_think_complete"


def normalize_plain_think_tokens(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in _TRUE:
        return "true"
    if text in _FALSE:
        return "false"
    if text == "auto":
        return "auto"
    raise ValueError(f"plain_think_tokens must be one of {', '.join(CHOICES)}; got {value!r}")


def cache_root() -> Path:
    base = os.environ.get("PARLVR_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "parlvr")
    return Path(base) / "tokenizers"


def _chat_template_text(template: Any) -> str:
    if isinstance(template, dict):
        return "\n".join(str(value) for value in template.values())
    return template or ""


def _display_name(model_path: str) -> str:
    """Model name for the cache directory, also for Hugging Face cache snapshots (.../models--Org--Name/snapshots/<sha>)."""
    parts = Path(model_path.rstrip("/")).parts
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        return parts[-3].split("--")[-1]
    return parts[-1] if parts else model_path


def _load_processor(model_path: str, **kwargs):
    from transformers import AutoProcessor, ProcessorMixin

    try:
        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    except Exception:
        return None
    return processor if isinstance(processor, ProcessorMixin) else None


def remove_think_tokens(directory: Path, tokens: tuple[str, ...] = THINK_TOKENS) -> None:
    """Drop ``tokens`` from the added tokens of a saved tokenizer (tokenizer.json, tokenizer_config.json,
    added_tokens.json, special_tokens_map.json), in place."""
    path = directory / "tokenizer.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["added_tokens"] = [item for item in data.get("added_tokens", []) if item.get("content") not in tokens]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    path = directory / "tokenizer_config.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        decoder = data.get("added_tokens_decoder", {})
        data["added_tokens_decoder"] = {
            key: item for key, item in decoder.items() if item.get("content") not in tokens
        }
        for key in ("additional_special_tokens", "extra_special_tokens"):
            if isinstance(data.get(key), list):
                data[key] = [item for item in data[key] if item not in tokens]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    path = directory / "added_tokens.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps({key: value for key, value in data.items() if key not in tokens}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    path = directory / "special_tokens_map.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        extra = data.get("additional_special_tokens")
        if isinstance(extra, list):
            data["additional_special_tokens"] = [
                item for item in extra if (item.get("content") if isinstance(item, dict) else item) not in tokens
            ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _build(target: Path, model_path: str, tokenizer, kwargs: dict[str, Any]) -> None:
    from transformers import AutoConfig, AutoTokenizer

    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    try:
        processor = _load_processor(model_path, **kwargs)
        (processor or tokenizer).save_pretrained(tmp)
        if processor is not None and _load_processor(str(tmp), **kwargs) is None:
            # Only when the processor class cannot be resolved from its own files: a config.json written by
            # transformers 4.57.3+ makes AutoTokenizer print a (spurious) Mistral regex warning.
            config_kwargs = {
                key: value for key, value in kwargs.items() if key in ("trust_remote_code", "revision", "token")
            }
            AutoConfig.from_pretrained(model_path, **config_kwargs).save_pretrained(tmp)
        remove_think_tokens(tmp)
        check = AutoTokenizer.from_pretrained(tmp, **kwargs)
        left = [token for token in THINK_TOKENS if token in check.get_added_vocab()]
        if left:
            raise RuntimeError(f"failed to remove {left} from the added tokens of {model_path}")
        (tmp / "PLAIN_THINK_SOURCE.json").write_text(
            json.dumps({"source": model_path, "removed_added_tokens": list(THINK_TOKENS)}, indent=2), encoding="utf-8"
        )
        (tmp / _COMPLETE_MARKER).touch()
        try:
            os.replace(tmp, target)
        except OSError:  # another process finished first
            if not (target / _COMPLETE_MARKER).exists():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def plain_think_tokenizer_path(model_path: Optional[str], plain_think_tokens: Any = "auto", **kwargs) -> Optional[str]:
    """Directory of a tokenizer with ``<think>``/``</think>`` as plain text, or ``None`` to keep ``model_path``."""
    setting = normalize_plain_think_tokens(plain_think_tokens)
    if setting == "false" or not model_path:
        return None
    key = (str(model_path), setting)
    if key in _RESOLVED:
        return _RESOLVED[key]

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    except Exception as exc:  # the regular loader reports real problems; do not mask them here
        print(
            f"[plain_think_tokens] could not inspect the tokenizer of {model_path} ({type(exc).__name__}); using it as is"
        )
        return None
    present = [token for token in THINK_TOKENS if token in tokenizer.get_added_vocab()]
    result: Optional[str] = None
    if present:
        template = _chat_template_text(getattr(tokenizer, "chat_template", None))
        if not template:
            processor = _load_processor(model_path, **kwargs)
            template = _chat_template_text(getattr(processor, "chat_template", None))
        uses_think = "<think>" in template
        if setting == "true" or not uses_think:
            digest = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()[:12]
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", _display_name(str(model_path))) or "tokenizer"
            target = cache_root() / f"{name}-plain-think-{digest}"
            if not (target / _COMPLETE_MARKER).exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                _build(target, str(model_path), tokenizer, kwargs)
            result = str(target)
            print(
                f"[plain_think_tokens={setting}] {model_path}: {', '.join(present)} are added tokens that the chat "
                "template never uses (untrained in Qwen3-VL Instruct models); they are tokenized as plain text "
                f"with the tokenizer in {target}. Disable with worker.actor.model.plain_think_tokens=false."
            )
    _RESOLVED[key] = result
    return result
