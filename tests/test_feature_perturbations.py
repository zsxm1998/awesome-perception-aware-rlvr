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
from __future__ import annotations

import csv
import importlib.metadata as importlib_metadata
import json
import os
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from eval.easyr1_eval.backends import VLLM_FEATURE_PLUGIN_NAME, VLLMBackend, _prepare_vllm_feature_plugin_env
from eval.easyr1_eval.feature_hooks import install_feature_perturbation_hook
from eval.easyr1_eval.perturbation_summary import write_perturbation_aggregate_summary
from eval.easyr1_eval.perturbations import PerturbationConfig, expand_perturbation_configs, perturbation_metadata
from eval.easyr1_eval.vllm_feature_models import (
    ENV_CONFIG,
    ENV_CONFIG_PATH,
    ENV_MARKER,
    EasyR1VLLMFeatureMixin,
    _image_sample_key,
)
from verl.utils.perturbations.feature_views import perturb_visual_features


def test_feature_divergence_perturbs_main_and_deepstack_with_shared_token_mask():
    torch.manual_seed(7)
    grid = torch.tensor([[1, 8, 8], [1, 4, 8]])
    main = torch.randn(24, 12, dtype=torch.float32)
    deepstack = [torch.randn(24, 10, dtype=torch.float32), torch.randn(24, 14, dtype=torch.float32)]

    main_out, deepstack_out, diagnostics = perturb_visual_features(
        main,
        deepstack,
        image_grid_thw=grid,
        merge_size=2,
        perturbation_type="feature_divergence",
        params={"k": 4, "rho": 0.75, "token_p": 0.6, "auto_rogue": False},
        global_seed=123,
        sample_id="sample-a",
        include_token_indices=True,
    )

    assert torch.linalg.vector_norm(main_out - main).item() > 0
    assert all(torch.linalg.vector_norm(out - original).item() > 0 for out, original in zip(deepstack_out, deepstack))
    for image_idx in {0, 1}:
        by_level = [item for item in diagnostics if item["image_idx"] == image_idx]
        token_sets = {tuple(item["token_indices"]) for item in by_level}
        assert len(token_sets) == 1
        assert all(item["token_mask_count"] == len(by_level[0]["token_indices"]) for item in by_level)


def test_feature_consistency_ignores_token_p_and_perturbs_all_tokens():
    torch.manual_seed(11)
    grid = torch.tensor([[1, 8, 8]])
    main = torch.randn(16, 12, dtype=torch.float32)
    deepstack = [torch.randn(16, 12, dtype=torch.float32)]

    main_out, deepstack_out, diagnostics = perturb_visual_features(
        main,
        deepstack,
        image_grid_thw=grid,
        merge_size=2,
        perturbation_type="feature_consistency",
        params={"k": 3, "token_p": 0.0, "noise_lambda": 0.3, "auto_rogue": False},
        global_seed=999,
        sample_id="sample-b",
    )

    assert torch.linalg.vector_norm(main_out - main).item() > 0
    assert torch.linalg.vector_norm(deepstack_out[0] - deepstack[0]).item() > 0
    assert {item["token_mask_count"] for item in diagnostics} == {16}


def test_feature_divergence_handles_empty_deepstack_list():
    torch.manual_seed(13)
    main = torch.randn(16, 12, dtype=torch.float32)

    main_out, deepstack_out, diagnostics = perturb_visual_features(
        main,
        [],
        image_grid_thw=torch.tensor([[1, 8, 8]]),
        merge_size=2,
        perturbation_type="feature_divergence",
        params={"k": 3, "rho": 1.0, "token_p": 1.0, "auto_rogue": False},
        global_seed=2,
        sample_id="empty-deepstack",
    )

    assert torch.linalg.vector_norm(main_out - main).item() > 0
    assert deepstack_out == []
    assert {item["level"] for item in diagnostics} == {"main"}


def test_feature_none_is_bitwise_unchanged():
    main = torch.randn(8, 4, dtype=torch.float32)
    deepstack = [torch.randn(8, 4, dtype=torch.float32)]
    main_out, deepstack_out, diagnostics = perturb_visual_features(
        main,
        deepstack,
        image_grid_thw=torch.tensor([[1, 4, 8]]),
        merge_size=2,
        perturbation_type="none",
        params={},
        global_seed=0,
        sample_id="clean",
    )
    assert main_out is main
    assert deepstack_out is deepstack
    assert diagnostics == []


class _FakeVisual(torch.nn.Module):
    config = type("Config", (), {"spatial_merge_size": 2})()

    def __init__(self):
        super().__init__()

    def forward(self, pixel_values, grid_thw=None):
        main = pixel_values.clone()
        deepstack = [pixel_values.clone() + 0.25, pixel_values.clone() - 0.25]
        return main, deepstack


class _FakeModel:
    def __init__(self):
        self.visual = _FakeVisual()

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.visual(pixel_values_videos, grid_thw=video_grid_thw)


def test_feature_hook_changes_main_and_deepstack_outputs():
    torch.manual_seed(23)
    model = _FakeModel()
    controller = install_feature_perturbation_hook(
        model,
        PerturbationConfig(
            type="feature_divergence",
            params={"k": 3, "rho": 1.0, "token_p": 1.0, "auto_rogue": False},
        ),
        global_seed=5,
    )
    inputs = torch.randn(16, 8)
    controller.begin_sample("hooked")
    main_out, deepstack_out = model.get_image_features(inputs, image_grid_thw=torch.tensor([[1, 8, 8]]))
    diagnostics = controller.end_sample()

    assert torch.linalg.vector_norm(main_out - inputs).item() > 0
    assert all(
        torch.linalg.vector_norm(out - original).item() > 0
        for out, original in zip(deepstack_out, [inputs + 0.25, inputs - 0.25])
    )
    assert {item["level"] for item in diagnostics} == {"main", "deepstack_0", "deepstack_1"}


def test_feature_hook_reinstall_does_not_double_wrap_visual():
    torch.manual_seed(29)
    model = _FakeModel()
    first = install_feature_perturbation_hook(
        model,
        PerturbationConfig(
            type="feature_divergence",
            params={"k": 3, "rho": 1.0, "token_p": 1.0, "auto_rogue": False},
        ),
        global_seed=5,
    )
    second = install_feature_perturbation_hook(
        model,
        PerturbationConfig(
            type="feature_divergence",
            params={"k": 3, "rho": 1.0, "token_p": 1.0, "auto_rogue": False},
        ),
        global_seed=5,
    )
    inputs = torch.randn(16, 8)

    second.begin_sample("hooked")
    model.get_image_features(inputs, image_grid_thw=torch.tensor([[1, 8, 8]]))
    diagnostics = second.end_sample()

    assert first.sample_id is None
    assert len(diagnostics) == 3
    assert {item["level"] for item in diagnostics} == {"main", "deepstack_0", "deepstack_1"}


def test_feature_hook_rejects_video_inputs():
    model = _FakeModel()
    controller = install_feature_perturbation_hook(
        model,
        PerturbationConfig(type="feature_divergence", params={"k": 2, "token_p": 1.0, "auto_rogue": False}),
        global_seed=5,
    )
    controller.begin_sample("video")
    with pytest.raises(ValueError, match="video inputs are unsupported"):
        model.get_video_features(torch.randn(16, 8), video_grid_thw=torch.tensor([[1, 8, 8]]))
    controller.end_sample()


class _FakeVLLMVisual(torch.nn.Module):
    spatial_merge_size = 2
    dtype = torch.float32

    def forward(self, pixel_values, grid_thw=None):
        chunks = torch.split(pixel_values, grid_thw.prod(-1).detach().cpu().long().tolist(), dim=0)
        merged_chunks = []
        for chunk in chunks:
            merged_chunks.append(chunk.reshape(-1, 4, chunk.shape[-1]).mean(dim=1))
        main = torch.cat(merged_chunks, dim=0)
        return torch.cat([main, main + 0.25, main - 0.25], dim=-1)


class _FakeVLLMBase:
    def __init__(self):
        self.visual = _FakeVLLMVisual()
        self.visual_dim = 8
        self.deepstack_num_level = 2
        self.config = SimpleNamespace(vision_config=SimpleNamespace(out_hidden_size=8))
        self.use_data_parallel = False

    def _process_image_input(self, image_input):
        return ("native", image_input)

    def _process_video_input(self, video_input):
        return ("native_video", video_input)


class _FakeVLLMModel(EasyR1VLLMFeatureMixin, _FakeVLLMBase):
    pass


def _feature_env(tmp_path, perturbation_type="feature_divergence", params=None, seed=17):
    marker = tmp_path / f"{perturbation_type}_{seed}.json"
    payload = {
        "type": perturbation_type,
        "params": params if params is not None else {"k": 2, "rho": 1.0, "token_p": 1.0, "auto_rogue": False},
        "seed": seed,
        "marker_path": str(marker),
    }
    return marker, payload


def test_vllm_feature_wrapper_changes_all_paths_and_writes_marker(tmp_path, monkeypatch):
    torch.manual_seed(37)
    marker, payload = _feature_env(tmp_path)
    monkeypatch.setenv(ENV_CONFIG, json.dumps(payload))
    monkeypatch.setenv(ENV_MARKER, str(marker))
    model = _FakeVLLMModel()
    pixel_values = torch.randn(16, 8)
    grid = torch.tensor([[1, 4, 4]])

    (output,) = model._process_image_input(
        {"type": "pixel_values", "pixel_values": pixel_values, "image_grid_thw": grid}
    )
    clean = model.visual(pixel_values, grid_thw=grid)
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))

    assert torch.linalg.vector_norm(output - clean).item() > 0
    assert marker_payload["count"] == 1
    assert set(marker_payload["first_batch_l2"]) == {"main", "deepstack_0", "deepstack_1"}
    assert all(value > 0 for value in marker_payload["first_batch_l2"].values())


def test_vllm_feature_consistency_writes_positive_marker(tmp_path, monkeypatch):
    torch.manual_seed(39)
    marker, payload = _feature_env(
        tmp_path,
        perturbation_type="feature_consistency",
        params={"k": 1, "noise_lambda": 0.5, "auto_rogue": False},
        seed=23,
    )
    monkeypatch.setenv(ENV_CONFIG, json.dumps(payload))
    monkeypatch.setenv(ENV_MARKER, str(marker))
    model = _FakeVLLMModel()
    pixel_values = torch.randn(16, 8)

    model._process_image_input(
        {"type": "pixel_values", "pixel_values": pixel_values, "image_grid_thw": torch.tensor([[1, 4, 4]])}
    )
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))

    assert set(marker_payload["first_batch_l2"]) == {"main", "deepstack_0", "deepstack_1"}
    assert all(value > 0 for value in marker_payload["first_batch_l2"].values())


def test_vllm_feature_wrapper_type_none_delegates_to_native(monkeypatch):
    monkeypatch.delenv(ENV_CONFIG, raising=False)
    monkeypatch.delenv(ENV_MARKER, raising=False)
    model = _FakeVLLMModel()
    image_input = {
        "type": "pixel_values",
        "pixel_values": torch.randn(16, 8),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }

    assert model._process_image_input(image_input) == ("native", image_input)


def test_vllm_feature_wrapper_rejects_video(tmp_path, monkeypatch):
    marker, payload = _feature_env(tmp_path)
    monkeypatch.setenv(ENV_CONFIG, json.dumps(payload))
    monkeypatch.setenv(ENV_MARKER, str(marker))
    model = _FakeVLLMModel()

    with pytest.raises(ValueError, match="video inputs are unsupported"):
        model._process_video_input({"type": "video"})


def test_vllm_feature_wrapper_is_deterministic_for_same_image_fingerprint(tmp_path, monkeypatch):
    torch.manual_seed(41)
    pixel_values = torch.randn(16, 8)
    grid = torch.tensor([[1, 4, 4]])
    outputs = []
    for index in range(2):
        marker, payload = _feature_env(tmp_path, seed=19 + index)
        payload["seed"] = 19
        payload["marker_path"] = str(marker)
        monkeypatch.setenv(ENV_CONFIG, json.dumps(payload))
        monkeypatch.setenv(ENV_MARKER, str(marker))
        model = _FakeVLLMModel()
        (output,) = model._process_image_input(
            {"type": "pixel_values", "pixel_values": pixel_values.clone(), "image_grid_thw": grid.clone()}
        )
        outputs.append(output)

    assert torch.equal(outputs[0], outputs[1])


def test_vllm_feature_wrapper_two_configs_use_current_env(tmp_path, monkeypatch):
    torch.manual_seed(43)
    pixel_values = torch.randn(16, 8)
    grid = torch.tensor([[1, 4, 4]])
    outputs = []
    for rho in (0.5, 1.0):
        marker, payload = _feature_env(tmp_path, params={"k": 3, "rho": rho, "token_p": 1.0, "auto_rogue": False})
        monkeypatch.setenv(ENV_CONFIG, json.dumps(payload))
        monkeypatch.setenv(ENV_MARKER, str(marker))
        model = _FakeVLLMModel()
        (output,) = model._process_image_input(
            {"type": "pixel_values", "pixel_values": pixel_values.clone(), "image_grid_thw": grid.clone()}
        )
        outputs.append(output)
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        assert marker_payload["type"] == "feature_divergence"

    assert not torch.equal(outputs[0], outputs[1])


def test_vllm_image_sample_key_uses_image_content_not_batch_position():
    image = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    grid = torch.tensor([1, 2, 2])

    assert _image_sample_key(image, grid) == _image_sample_key(image.clone(), grid.clone())
    changed = image.clone()
    changed[0, 0] = -1
    assert _image_sample_key(image, grid) != _image_sample_key(changed, grid)


def test_vllm_image_sample_key_supports_bfloat16():
    image = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
    grid = torch.tensor([1, 2, 2])

    assert _image_sample_key(image, grid) == _image_sample_key(image.clone(), grid.clone())


def test_vllm_activation_validation_requires_positive_expected_levels(tmp_path):
    marker = tmp_path / "marker.json"
    marker.write_text(
        json.dumps({"count": 1, "first_batch_l2": {"main": 1.0, "deepstack_0": 0.0}}),
        encoding="utf-8",
    )
    backend = VLLMBackend.__new__(VLLMBackend)
    backend._vllm_feature_marker_path = marker
    backend._vllm_feature_expected_levels = ["main", "deepstack_0"]

    with pytest.raises(RuntimeError, match="non_positive_l2"):
        backend._validate_vllm_feature_activation()


def test_vllm_backend_sets_feature_env_before_each_engine_and_clears_clean(tmp_path, monkeypatch):
    vllm = pytest.importorskip("vllm")

    import eval.easyr1_eval.backends as backend_module
    import eval.easyr1_eval.vllm_feature_models as feature_model_module
    import verl.utils.tokenizer as tokenizer_module

    captured_env = []

    class _FakeLLM:
        def __init__(self, **kwargs):
            config_path = os.environ.get(ENV_CONFIG_PATH)
            captured_env.append(
                {
                    "config": os.environ.get(ENV_CONFIG),
                    "config_path": config_path,
                    "init_config": json.loads(open(config_path, encoding="utf-8").read()) if config_path else None,
                    "pythonpath": os.environ.get("PYTHONPATH", ""),
                    "plugins": os.environ.get("VLLM_PLUGINS"),
                }
            )

    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("VLLM_PLUGINS", os.environ.get("VLLM_PLUGINS", ""))
    monkeypatch.setattr(vllm, "LLM", _FakeLLM)
    monkeypatch.setattr(tokenizer_module, "get_processor", lambda *args, **kwargs: None)
    monkeypatch.setattr(tokenizer_module, "get_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_module, "_expected_vllm_feature_levels", lambda *args, **kwargs: ["main"])
    monkeypatch.setattr(feature_model_module, "register_vllm_feature_models", lambda: None)

    VLLMBackend(
        "fake-model",
        perturbation=PerturbationConfig(type="feature_divergence", params={"k": 2, "rho": 0.5}),
        perturbation_seed=1,
        output_dir=tmp_path / "rho05",
    )
    VLLMBackend(
        "fake-model",
        perturbation=PerturbationConfig(type="feature_divergence", params={"k": 2, "rho": 1.0}),
        perturbation_seed=1,
        output_dir=tmp_path / "rho10",
    )
    VLLMBackend("fake-model", perturbation=PerturbationConfig(), output_dir=tmp_path / "clean")

    first = json.loads(open(captured_env[0]["config_path"], encoding="utf-8").read())
    second = json.loads(open(captured_env[1]["config_path"], encoding="utf-8").read())
    assert captured_env[0]["init_config"]["type"] == "none"
    assert captured_env[1]["init_config"]["type"] == "none"
    assert captured_env[0]["config"] is None
    assert captured_env[1]["config"] is None
    assert first["params"]["rho"] == 0.5
    assert second["params"]["rho"] == 1.0
    assert first["type"] == "feature_divergence"
    assert second["type"] == "feature_divergence"
    assert str(tmp_path / "rho05" / ".vllm_feature_plugin") in captured_env[0]["pythonpath"]
    assert str(tmp_path / "rho10" / ".vllm_feature_plugin") in captured_env[1]["pythonpath"]
    assert VLLM_FEATURE_PLUGIN_NAME in captured_env[0]["plugins"].split(",")
    assert VLLM_FEATURE_PLUGIN_NAME in captured_env[1]["plugins"].split(",")
    assert captured_env[2]["config"] is None
    assert captured_env[2]["config_path"] is None


def test_vllm_feature_plugin_env_writes_discoverable_entry_point(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    _prepare_vllm_feature_plugin_env(tmp_path)

    plugin_root = tmp_path / ".vllm_feature_plugin"
    distributions = list(importlib_metadata.distributions(path=[str(plugin_root)]))
    # one distinct entry point (some environments expose the same distribution through two finders)
    entry_points = list(
        {
            (entry_point.name, entry_point.value): entry_point
            for dist in distributions
            for entry_point in dist.entry_points
            if entry_point.group == "vllm.general_plugins"
        }.values()
    )
    assert len(entry_points) == 1
    assert entry_points[0].name == VLLM_FEATURE_PLUGIN_NAME
    assert entry_points[0].value == "eval.easyr1_eval.vllm_feature_models:register_vllm_feature_models"
    assert str(plugin_root) in os.environ["PYTHONPATH"]
    assert str(plugin_root) in sys.path


def test_feature_consistency_rejects_explicit_m_tail_beyond_candidates():
    torch.manual_seed(31)
    with pytest.raises(ValueError, match="m_tail=50 exceeds available"):
        perturb_visual_features(
            torch.randn(16, 12, dtype=torch.float32),
            [],
            image_grid_thw=torch.tensor([[1, 8, 8]]),
            merge_size=2,
            perturbation_type="feature_consistency",
            params={"k": 3, "m_tail": 50, "noise_lambda": 0.2, "auto_rogue": False},
            global_seed=3,
            sample_id="bad-tail",
        )


def test_feature_consistency_config_rejects_token_p():
    args = Namespace(
        perturbation_type="feature_consistency",
        perturbation_k="3",
        perturbation_ev_threshold=None,
        perturbation_n_skip=None,
        perturbation_auto_rogue=None,
        perturbation_rogue_corr_threshold=None,
        perturbation_noise_lambda="0.1",
        perturbation_m_tail=None,
        perturbation_token_p="0.5",
    )
    with pytest.raises(ValueError, match="token_p only applies"):
        expand_perturbation_configs(args)


def test_perturbation_metadata_serializes_params_as_json():
    metadata = perturbation_metadata(PerturbationConfig(type="feature_divergence", params={"k": 3}), 7)
    assert metadata["perturbation_params"] == '{"k": 3}'


def test_perturbation_summary_skips_clean_only_runs(tmp_path):
    run_dir = tmp_path / "clean"
    run_dir.mkdir()
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "backend",
                "row_type",
                "group",
                "benchmark",
                "normalized_score_0_100",
                "num_examples",
                "status",
                "perturbation_type",
                "perturbation_summary",
                "perturbation_seed",
                "perturbation_params",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "m",
                "backend": "vllm",
                "row_type": "benchmark",
                "group": "g",
                "benchmark": "b",
                "normalized_score_0_100": "90.0",
                "num_examples": "200",
                "status": "ok",
                "perturbation_type": "none",
                "perturbation_summary": "clean",
                "perturbation_seed": "42",
                "perturbation_params": "{}",
            }
        )

    output = write_perturbation_aggregate_summary([run_dir], tmp_path)

    assert output is None
    assert not (tmp_path / "perturbation_summary.csv").exists()


def test_perturbation_summary_requires_same_backend_clean(tmp_path):
    run_dir = tmp_path / "feature"
    run_dir.mkdir()
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "backend",
                "row_type",
                "group",
                "benchmark",
                "normalized_score_0_100",
                "num_examples",
                "status",
                "perturbation_type",
                "perturbation_summary",
                "perturbation_seed",
                "perturbation_params",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "m",
                "backend": "transformers",
                "row_type": "benchmark",
                "group": "g",
                "benchmark": "b",
                "normalized_score_0_100": "88.0",
                "num_examples": "200",
                "status": "ok",
                "perturbation_type": "feature_divergence",
                "perturbation_summary": "feature_divergence_k-3",
                "perturbation_seed": "1",
                "perturbation_params": '{"k": 3}',
            }
        )

    output = write_perturbation_aggregate_summary([run_dir], tmp_path)
    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["delta_vs_clean_mean"] == ""
    assert rows[0]["delta_missing_reason"] == "missing_same_backend_clean"
