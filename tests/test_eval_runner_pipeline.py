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
import json
import sys
import threading
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval import runner  # noqa: E402
from easyr1_eval.schemas import (  # noqa: E402
    BenchmarkSpec,
    EvalSample,
    GenerationOutput,
    MetricResult,
)


def _spec(key: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        key=key,
        label=key,
        group="Test",
        loader="dummy",
        scorer="dummy",
        primary_metric="score",
    )


def test_native_agentic_prompt_contract_removes_textual_bbox_enumeration_only_for_agent():
    sample = EvalSample(
        benchmark="grit_tallyqa",
        sample_id="one",
        prompt=("How many birds?\nProvide a concise final answer and ground every image region needed to justify it."),
        native_agentic_prompt="How many birds?",
        target="2",
    )

    agentic = runner.apply_interaction_prompt_contract(
        [sample],
        SimpleNamespace(
            interaction_mode="agentic",
            agent_output_contract="native",
        ),
    )
    one_shot = runner.apply_interaction_prompt_contract(
        [sample],
        SimpleNamespace(interaction_mode="one_shot"),
    )

    assert agentic[0].prompt == "How many birds?"
    assert one_shot[0].prompt == sample.prompt
    assert sample.prompt.endswith("image region needed to justify it.")


def test_parse_args_rejects_one_shot_token_override_in_agentic_mode(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--model",
            "model",
            "--interaction-mode",
            "agentic",
            "--max-new-tokens",
            "4096",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        runner.parse_args()

    assert "does not use --max-new-tokens" in capsys.readouterr().err


def test_parse_args_requires_strict_deepeyes_system_prompt_in_agentic_mode(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--model",
            "model",
            "--interaction-mode",
            "agentic",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args()
    assert "requires --system-prompt" in capsys.readouterr().err

    invalid_prompt = tmp_path / "json_GR.txt"
    invalid_prompt.write_text("Think and answer with grounded boxes.", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--model",
            "model",
            "--interaction-mode",
            "agentic",
            "--system-prompt",
            str(invalid_prompt),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        runner.parse_args()
    assert "exactly one" in capsys.readouterr().err

    deepeyes_prompt = ROOT / "examples/system_prompt/deepeyes.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--model",
            "model",
            "--interaction-mode",
            "agentic",
            "--system-prompt",
            str(deepeyes_prompt),
            "--agent-max-tool-calls",
            "2",
            "--agent-tool-image-mode",
            "text_skipped",
        ],
    )

    args = runner.parse_args()

    assert args.system_prompt == str(deepeyes_prompt)
    assert args.agent_max_tool_calls == 2
    assert args.agent_tool_image_mode == "text_skipped"
    assert args.format_prompt is None


def test_parse_args_rejects_format_prompt_for_native_deepeyes(
    monkeypatch,
    tmp_path,
    capsys,
):
    system_prompt = tmp_path / "deepeyes.txt"
    system_prompt.write_text(
        "At most {{ max_tool_calls }} calls.",
        encoding="utf-8",
    )
    format_prompt = tmp_path / "format.txt"
    format_prompt.write_text("{{ content }}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--model",
            "model",
            "--interaction-mode",
            "agentic",
            "--system-prompt",
            str(system_prompt),
            "--format-prompt",
            str(format_prompt),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        runner.parse_args()

    assert "does not use --format-prompt" in capsys.readouterr().err


def test_agentic_shard_with_only_sample_errors_is_not_marked_successful(
    monkeypatch,
    tmp_path,
):
    sample = EvalSample(
        benchmark="bench",
        sample_id="bad",
        prompt="question",
        target="yes",
    )

    class FailedBackend:
        def generate(self, samples, config):
            del config
            return [
                [
                    GenerationOutput(
                        text="",
                        finish_reason="error",
                        diagnostics={
                            "agent": {
                                "status": "sample_error",
                                "error": "bad image",
                            }
                        },
                    )
                ]
                for _ in samples
            ]

    args = SimpleNamespace(
        data_root=tmp_path,
        output_dir=tmp_path,
        limit=None,
        batch_size=1,
        max_batch_images=1,
        interaction_mode="agentic",
        temperature=None,
        num_samples=1,
        max_new_tokens=None,
        top_p=1.0,
        seed=42,
        agent_max_tool_calls=6,
        agent_max_response_tokens=32,
        agent_max_tokens_per_turn=16,
        agent_max_images_per_prompt=8,
    )
    monkeypatch.setattr(runner, "load_samples", lambda *args, **kwargs: [sample])
    monkeypatch.setattr(
        runner,
        "apply_prompt_config",
        lambda samples, config: samples,
    )
    monkeypatch.setattr(runner, "prompt_config_from_args", lambda args: None)
    monkeypatch.setattr(runner, "metric_metadata_for", lambda spec, args: {})

    with pytest.raises(
        RuntimeError,
        match="every agent trajectory in shard 0 failed",
    ):
        runner.infer_benchmark_shard(
            _spec("bench"),
            args,
            shard_index=0,
            num_shards=1,
            backend=FailedBackend(),
        )


def test_run_all_overlaps_inference_with_serial_benchmark_scoring(monkeypatch, tmp_path):
    specs = [_spec("first"), _spec("second")]
    score_started = threading.Event()
    inference_continued = threading.Event()
    lock = threading.Lock()
    events = []
    active_scores = 0
    max_active_scores = 0

    def fake_score_benchmark(spec, args, judge_config):
        nonlocal active_scores, max_active_scores
        with lock:
            active_scores += 1
            max_active_scores = max(max_active_scores, active_scores)
            events.append(f"score_start:{spec.key}")
        if spec.key == "first":
            score_started.set()
            assert inference_continued.wait(timeout=2)
        with lock:
            events.append(f"score_end:{spec.key}")
            active_scores -= 1
        return MetricResult(spec.key, spec.group, spec.primary_metric, 1.0, 1.0, 1)

    def fake_run_inference(specs, args, gpu_groups, *, on_benchmark_ready, failures=None):
        events.append("inference_ready:first")
        on_benchmark_ready(specs[0])
        assert score_started.wait(timeout=2)
        events.append("inference_continued")
        inference_continued.set()
        events.append("inference_ready:second")
        on_benchmark_ready(specs[1])

    args = SimpleNamespace(
        output_dir=tmp_path / "run",
        summary_only=False,
        score_only=False,
        gpus="0",
        tp=1,
        judge_provider="none",
    )
    monkeypatch.setattr(runner, "score_benchmark", fake_score_benchmark)
    monkeypatch.setattr(runner, "run_inference", fake_run_inference)
    monkeypatch.setattr(runner, "collect_run_results", lambda args, results: results)
    monkeypatch.setattr(runner, "write_run_summaries", lambda results, args, run_id: None)

    runner.run_all(specs, args)

    assert max_active_scores == 1
    assert events.index("score_start:first") < events.index("inference_continued")
    assert events.index("score_end:first") < events.index("score_start:second")


def test_run_inference_marks_fully_resumed_benchmarks_ready(monkeypatch, tmp_path):
    specs = [_spec("first"), _spec("second")]
    args = SimpleNamespace(
        output_dir=tmp_path,
        resume=True,
        backend="vllm",
        model="model",
        temperature=None,
        num_samples=None,
        max_new_tokens=None,
        top_p=1.0,
        seed=42,
        limit=None,
        min_pixels=1,
        max_pixels=2,
        max_model_len=128,
        format_prompt=None,
        system_prompt=None,
        prompt_mode="chat",
        judge_provider="none",
        judge_model=None,
        judge_max_tokens=None,
        judge_request_retries=5,
        judge_thinking="disabled",
        perturbation=SimpleNamespace(type="none", params={}),
        perturbation_seed=42,
        perturbation_vllm_force_feature_wrapper=False,
    )
    gpu_groups = ["0", "1"]
    for spec in specs:
        for shard_index in range(len(gpu_groups)):
            output = runner.shard_prediction_path(tmp_path, spec, shard_index)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                f'{{"sample_id": "{spec.key}-{shard_index}"}}\n',
                encoding="utf-8",
            )
            fingerprint = runner.task_fingerprint(
                args,
                spec,
                phase="infer",
                shard_index=shard_index,
                num_shards=len(gpu_groups),
            )
            runner.mark_complete(
                tmp_path,
                f"infer:{spec.key}:shard{shard_index}",
                fingerprint,
                [output],
            )

    ready = []
    monkeypatch.setattr(
        runner.mp,
        "get_context",
        lambda method: (_ for _ in ()).throw(AssertionError("resume should not start workers")),
    )

    runner.run_inference(specs, args, gpu_groups, on_benchmark_ready=lambda spec: ready.append(spec.key))

    assert ready == ["first", "second"]
    for spec in specs:
        merged = runner.merged_prediction_path(tmp_path, spec)
        assert len(merged.read_text(encoding="utf-8").splitlines()) == len(gpu_groups)


def test_agentic_budget_changes_invalidate_only_agentic_inference_resume(tmp_path):
    spec = _spec("bench")
    args = SimpleNamespace(
        output_dir=tmp_path,
        backend="vllm",
        model="model",
        temperature=None,
        num_samples=None,
        max_new_tokens=None,
        top_p=1.0,
        seed=42,
        limit=None,
        min_pixels=1,
        max_pixels=2,
        max_model_len=128,
        format_prompt=None,
        system_prompt=None,
        prompt_mode="chat",
        judge_provider="none",
        judge_model=None,
        judge_max_tokens=None,
        judge_request_retries=5,
        judge_thinking="disabled",
        perturbation=SimpleNamespace(type="none", params={}),
        perturbation_seed=42,
        perturbation_vllm_force_feature_wrapper=False,
        interaction_mode="one_shot",
        agent_profile="deepeyes",
        agent_max_tool_calls=6,
        agent_max_response_tokens=20480,
        agent_max_tokens_per_turn=10240,
        agent_max_images_per_prompt=16,
        agent_tool_image_mode="original",
    )
    one_shot = runner.task_fingerprint(args, spec, phase="infer")
    changed_one_shot_args = copy(args)
    changed_one_shot_args.agent_max_tool_calls = 2
    changed_one_shot = runner.task_fingerprint(
        changed_one_shot_args,
        spec,
        phase="infer",
    )

    agentic_args = copy(args)
    agentic_args.interaction_mode = "agentic"
    system_prompt = tmp_path / "deepeyes.txt"
    system_prompt.write_text(
        "At most {{ max_tool_calls }} calls.",
        encoding="utf-8",
    )
    agentic_args.system_prompt = str(system_prompt)
    agentic = runner.task_fingerprint(agentic_args, spec, phase="infer")
    changed_agentic_args = copy(agentic_args)
    changed_agentic_args.agent_max_tool_calls = 2
    changed_agentic = runner.task_fingerprint(
        changed_agentic_args,
        spec,
        phase="infer",
    )

    assert changed_one_shot == one_shot
    assert agentic != one_shot

    # --top-k enters the fingerprint only when set, so earlier results stay valid
    unset_top_k, top_k_50 = copy(args), copy(args)
    unset_top_k.top_k, top_k_50.top_k = None, 50
    assert runner.task_fingerprint(unset_top_k, spec, phase="infer") == one_shot
    assert runner.task_fingerprint(top_k_50, spec, phase="infer") != one_shot
    assert runner.generation_config_for(spec, top_k_50).top_k == 50
    assert changed_agentic != agentic

    fixed_gray_agentic_args = copy(agentic_args)
    fixed_gray_agentic_args.agent_tool_image_mode = "fixed_gray"
    assert (
        runner.task_fingerprint(
            fixed_gray_agentic_args,
            spec,
            phase="infer",
        )
        != agentic
    )

    text_skipped_agentic_args = copy(agentic_args)
    text_skipped_agentic_args.agent_tool_image_mode = "text_skipped"
    assert runner.task_fingerprint(
        text_skipped_agentic_args,
        spec,
        phase="infer",
    ) not in {agentic, changed_agentic}


def test_score_failure_does_not_cancel_later_benchmarks(monkeypatch, tmp_path):
    specs = [_spec("first"), _spec("second")]
    scored = []

    def fake_score_benchmark(spec, args, judge_config):
        scored.append(spec.key)
        if spec.key == "first":
            raise RuntimeError("score failed")
        return MetricResult(spec.key, spec.group, spec.primary_metric, 1.0, 1.0, 1)

    def fake_run_inference(specs, args, gpu_groups, *, on_benchmark_ready, failures=None):
        for spec in specs:
            on_benchmark_ready(spec)

    summaries = []
    args = SimpleNamespace(
        output_dir=tmp_path / "run",
        summary_only=False,
        score_only=False,
        gpus="0",
        tp=1,
        judge_provider="none",
    )
    monkeypatch.setattr(runner, "score_benchmark", fake_score_benchmark)
    monkeypatch.setattr(runner, "run_inference", fake_run_inference)
    monkeypatch.setattr(runner, "collect_run_results", lambda args, results: results)
    monkeypatch.setattr(runner, "write_run_summaries", lambda results, args, run_id: summaries.append(results))

    failures = runner.run_all(specs, args)

    assert scored == ["first", "second"]
    assert list(failures) == ["first"]
    assert "score failed" in failures["first"]
    assert [result.benchmark for result in summaries[0]] == ["second"]


def test_inference_failure_skips_scoring_of_that_benchmark_only(monkeypatch, tmp_path):
    specs = [_spec("broken"), _spec("fine")]
    scored = []

    def fake_score_benchmark(spec, args, judge_config):
        scored.append(spec.key)
        return MetricResult(spec.key, spec.group, spec.primary_metric, 1.0, 1.0, 1)

    def fake_run_inference(specs, args, gpu_groups, *, on_benchmark_ready, failures=None):
        failures["broken"] = "inference failed: FileNotFoundError"
        for spec in specs:
            on_benchmark_ready(spec)

    args = SimpleNamespace(
        output_dir=tmp_path / "run", summary_only=False, score_only=False, gpus="0", tp=1, judge_provider="none"
    )
    monkeypatch.setattr(runner, "score_benchmark", fake_score_benchmark)
    monkeypatch.setattr(runner, "run_inference", fake_run_inference)
    monkeypatch.setattr(runner, "collect_run_results", lambda args, results: results)
    monkeypatch.setattr(runner, "write_run_summaries", lambda results, args, run_id: None)

    failures = runner.run_all(specs, args)

    assert scored == ["fine"]
    assert list(failures) == ["broken"]


def test_judge_only_benchmark_is_skipped_without_judge(monkeypatch, tmp_path):
    judged = BenchmarkSpec(
        key="mm_vet",
        label="MM-Vet",
        group="Perception",
        loader="mmvet",
        scorer="mmvet",
        primary_metric="judge_score",
        requires_judge=True,
    )
    specs = [_spec("first"), judged]
    inferred = []
    scored = []

    def fake_run_inference(specs, args, gpu_groups, *, on_benchmark_ready, failures=None):
        for spec in specs:
            inferred.append(spec.key)
            on_benchmark_ready(spec)

    def fake_score_benchmark(spec, args, judge_config):
        scored.append(spec.key)
        return MetricResult(spec.key, spec.group, spec.primary_metric, 1.0, 1.0, 1)

    summaries = []
    args = SimpleNamespace(
        output_dir=tmp_path / "run", summary_only=False, score_only=False, gpus="0", tp=1, judge_provider="none"
    )
    monkeypatch.setattr(runner, "score_benchmark", fake_score_benchmark)
    monkeypatch.setattr(runner, "run_inference", fake_run_inference)
    monkeypatch.setattr(runner, "collect_run_results", lambda args, results: results)
    monkeypatch.setattr(runner, "write_run_summaries", lambda results, args, run_id: summaries.append(results))

    failures = runner.run_all(specs, args)

    assert failures == {}
    assert inferred == ["first"] and scored == ["first"]
    by_key = {result.benchmark: result for result in summaries[0]}
    assert by_key["mm_vet"].status == "skipped"
    assert "judge" in by_key["mm_vet"].details["skip_reason"]
    metric = json.loads((tmp_path / "run" / "metrics" / "mm_vet.json").read_text())
    assert metric["result"]["status"] == "skipped"


def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py", "--model", "m", *argv])
    return runner.parse_args()


def test_parse_args_default_prompt_and_data_root(monkeypatch):
    monkeypatch.delenv("EVAL_DATA_ROOT", raising=False)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    args = _parse(monkeypatch)
    assert Path(args.format_prompt) == ROOT / "examples/format_prompt/math_perception.jinja"
    assert args.system_prompt is None
    assert args.interaction_mode == "one_shot"
    assert Path(args.data_root) == ROOT / "data" / "eval"
    assert args.judge_provider == "none"
    assert args.top_p is None

    assert _parse(monkeypatch, "--format-prompt", "none").format_prompt is None
    monkeypatch.setenv("DATA_ROOT", "/tmp/somewhere")
    assert Path(_parse(monkeypatch).data_root) == Path("/tmp/somewhere/eval")
    monkeypatch.setenv("EVAL_DATA_ROOT", "/tmp/eval_data")
    assert Path(_parse(monkeypatch).data_root) == Path("/tmp/eval_data")


def test_suite_defaults_apply_only_when_flags_are_not_given(monkeypatch):
    args = _parse(monkeypatch, "--suite", "grit")
    assert args.format_prompt is None
    assert Path(args.system_prompt) == ROOT / "examples/system_prompt/grit_GR.txt"

    args = _parse(monkeypatch, "--suite", "cgpo", "--format-prompt", str(ROOT / "examples/format_prompt/math.jinja"))
    assert Path(args.format_prompt) == ROOT / "examples/format_prompt/math.jinja"

    args = _parse(monkeypatch, "--suite", "cgpo")
    assert Path(args.format_prompt) == ROOT / "examples/format_prompt/xml_grounded_reasoning.jinja"

    args = _parse(monkeypatch, "--suite", "deepeyes")
    assert args.interaction_mode == "agentic"
    assert args.format_prompt is None
    assert Path(args.system_prompt) == ROOT / "examples/system_prompt/deepeyes.txt"

    args = _parse(
        monkeypatch,
        "--suite",
        "deepeyes",
        "--interaction-mode",
        "one_shot",
        "--format-prompt",
        str(ROOT / "examples/format_prompt/math_perception.jinja"),
        "--system-prompt",
        "none",
    )
    assert args.interaction_mode == "one_shot"
    assert args.system_prompt is None


def test_conflicting_suite_defaults_require_explicit_flags(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--suite", "grit,cgpo")
    assert "disagree" in capsys.readouterr().err
    args = _parse(monkeypatch, "--suite", "grit,cgpo", "--format-prompt", "none", "--system-prompt", "none")
    assert args.format_prompt is None and args.system_prompt is None


def test_selected_specs_union_of_suite_and_benchmarks_minus_skip(monkeypatch):
    args = _parse(
        monkeypatch, "--suite", "tor", "--benchmarks", "pope,mathvista", "--skip-benchmarks", "hallusionbench"
    )
    assert [spec.key for spec in runner.selected_specs(args)] == [
        "mathverse",
        "mathvision",
        "mathvista",
        "wemath",
        "pope",
    ]
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--suite", "nope")


def test_missing_data_aborts_before_inference_or_is_skipped(monkeypatch, tmp_path):
    args = _parse(monkeypatch, "--benchmarks", "geo3k,pope", "--data-root", str(tmp_path))
    args.data_root = tmp_path
    specs = runner.selected_specs(args)
    with pytest.raises(SystemExit, match="prepare_eval_data.sh geo3k pope"):
        runner.check_benchmark_data(specs, args)

    (tmp_path / "geo3k" / "images").mkdir(parents=True)
    (tmp_path / "geo3k" / "test.parquet").write_bytes(b"x")
    args.skip_missing_data = True
    assert [spec.key for spec in runner.check_benchmark_data(specs, args)] == ["geo3k"]


@pytest.mark.parametrize(
    "extra,expected",
    [
        ([], (200704, 1003520)),
        (["--suite", "pepo_geometry"], (3136, 12845056)),
        (["--suite", "pepo_geometry", "--max-pixels", "1003520"], (3136, 1003520)),
    ],
)
def test_image_size_defaults_follow_the_suite(monkeypatch, extra, expected):
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py", "--model", "model", *extra])
    args = runner.parse_args()
    assert (args.min_pixels, args.max_pixels) == expected


def test_grit_suite_asks_the_bare_question(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py", "--model", "model", "--suite", "grit"])
    grit_args = runner.parse_args()
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py", "--model", "model", "--suite", "cgpo"])
    cgpo_args = runner.parse_args()
    assert grit_args.grounding_instruction == "none" and cgpo_args.grounding_instruction == "append"

    sample = EvalSample(
        benchmark="grit_vsr",
        sample_id="grit_vsr:0",
        prompt="Is the cat left of the dog?\nProvide a concise final answer and ground every image region ...",
        target="yes",
        question_only_prompt="Is the cat left of the dog?",
    )
    (bare,) = runner.apply_grounding_instruction([sample], grit_args)
    (kept,) = runner.apply_grounding_instruction([sample], cgpo_args)
    assert bare.prompt == "Is the cat left of the dog?" and kept.prompt == sample.prompt
