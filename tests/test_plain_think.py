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
"""<think>/</think> as plain text for checkpoints that never trained them (verl/utils/plain_think.py)."""

import os
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from verl.utils import plain_think
from verl.utils.plain_think import normalize_plain_think_tokens, plain_think_tokenizer_path
from verl.utils.tokenizer import get_tokenizer
from verl.workers.actor.config import ModelConfig


INSTRUCT_TEMPLATE = "{% for m in messages %}<|im_start|>{{ m['content'] }}<|im_end|>{% endfor %}"
THINKING_TEMPLATE = INSTRUCT_TEMPLATE + "{% if add_generation_prompt %}<|im_start|><think>\n{% endif %}"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PARLVR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(plain_think, "_RESOLVED", {})


def _tiny_tokenizer(directory: Path, chat_template: str, think_tokens: bool = True) -> str:
    """Character-level tokenizer with Qwen-like added tokens, saved like a released checkpoint."""
    chars = sorted(set("abcdefghijklmnopqrstuvwxyz<>/\\{}()_ .,:\n0123456789|"))
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|im_end|>")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    if think_tokens:
        tokenizer.add_tokens(["<think>", "</think>"])  # added, non-special: as in Qwen3-VL
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(directory)
    return str(directory)


def _think_ids(tokenizer):
    return tokenizer("<think>", add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize(
    "value,expected",
    [(True, "true"), (False, "false"), ("AUTO", "auto"), ("off", "false"), ("1", "true"), (None, "false")],
)
def test_normalize(value, expected):
    assert normalize_plain_think_tokens(value) == expected


def test_normalize_rejects_unknown_values():
    with pytest.raises(ValueError, match="plain_think_tokens"):
        normalize_plain_think_tokens("sometimes")


def test_auto_rewrites_untrained_think_tokens(tmp_path):
    model = _tiny_tokenizer(tmp_path / "instruct", INSTRUCT_TEMPLATE)
    original = get_tokenizer(model, plain_think_tokens="false")
    patched_dir = plain_think_tokenizer_path(model, "auto")
    patched = get_tokenizer(model, plain_think_tokens="auto")

    assert patched_dir is not None and Path(patched_dir).is_relative_to(Path(os.environ["PARLVR_CACHE_DIR"]))
    assert len(_think_ids(original)) == 1
    assert len(_think_ids(patched)) == len("<think>")
    assert len(patched) == len(original) - 2
    assert "<think>" not in patched.get_added_vocab()
    # everything else is untouched
    text = "<|im_start|>ab 12<|im_end|>"
    assert (
        patched(text, add_special_tokens=False)["input_ids"] == original(text, add_special_tokens=False)["input_ids"]
    )
    assert patched.chat_template == original.chat_template
    # the source checkpoint is not modified
    assert len(_think_ids(get_tokenizer(model, plain_think_tokens="false"))) == 1


def test_auto_keeps_thinking_checkpoints_and_models_without_the_tokens(tmp_path):
    thinking = _tiny_tokenizer(tmp_path / "thinking", THINKING_TEMPLATE)
    without = _tiny_tokenizer(tmp_path / "without", INSTRUCT_TEMPLATE, think_tokens=False)

    assert plain_think_tokenizer_path(thinking, "auto") is None
    assert len(_think_ids(get_tokenizer(thinking))) == 1
    assert plain_think_tokenizer_path(without, "auto") is None
    assert plain_think_tokenizer_path(thinking, "true") is not None
    assert plain_think_tokenizer_path(_tiny_tokenizer(tmp_path / "off", INSTRUCT_TEMPLATE), "false") is None


def test_patched_tokenizer_is_cached_and_reused(tmp_path, monkeypatch):
    model = _tiny_tokenizer(tmp_path / "instruct", INSTRUCT_TEMPLATE)
    first = plain_think_tokenizer_path(model, "auto")
    marker = Path(first) / ".plain_think_complete"
    stamp = marker.stat().st_mtime_ns

    monkeypatch.setattr(plain_think, "_RESOLVED", {})  # a new process
    monkeypatch.setattr(plain_think, "_build", lambda *a, **k: pytest.fail("rebuilt an existing tokenizer"))
    assert plain_think_tokenizer_path(model, "auto") == first
    assert marker.stat().st_mtime_ns == stamp


def test_saving_a_patched_tokenizer_keeps_it_patched(tmp_path):
    """Checkpoints save the tokenizer they trained with; reloading them must not change it again."""
    model = _tiny_tokenizer(tmp_path / "instruct", INSTRUCT_TEMPLATE)
    get_tokenizer(model).save_pretrained(tmp_path / "checkpoint")
    reloaded_dir = str(tmp_path / "checkpoint")

    assert plain_think_tokenizer_path(reloaded_dir, "auto") is None
    assert len(_think_ids(get_tokenizer(reloaded_dir))) == len("<think>")


def test_model_config_normalizes_the_setting():
    config = ModelConfig(model_path=None, plain_think_tokens=False)
    config.post_init()
    assert config.plain_think_tokens == "false"
    with pytest.raises(ValueError):
        ModelConfig(plain_think_tokens="maybe").post_init()


def _cached_qwen3_vl_instruct():
    from huggingface_hub import try_to_load_from_cache

    for repo in ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"):
        found = try_to_load_from_cache(repo, "tokenizer.json")
        if isinstance(found, str):
            return str(Path(found).parent)
    return None


def test_real_qwen3_vl_instruct_processor():
    model = _cached_qwen3_vl_instruct()
    if model is None:
        pytest.skip("no Qwen3-VL Instruct tokenizer in the local Hugging Face cache")
    from verl.utils.tokenizer import get_processor

    processor = get_processor(model, use_fast=True)
    assert processor.tokenizer("<think>", add_special_tokens=False)["input_ids"] == [13708, 766, 29]
    assert get_processor(model, plain_think_tokens="false", use_fast=True).tokenizer(
        "<think>", add_special_tokens=False
    )["input_ids"] == [151667]
