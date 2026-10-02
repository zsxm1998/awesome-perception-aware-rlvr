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
import sys
from argparse import Namespace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval.agentic import (  # noqa: E402
    AgenticConfigurationError,
    AgenticVLLMBackend,
    trajectory_to_generation_output,
)
from easyr1_eval.backends import process_eval_image  # noqa: E402
from easyr1_eval.json_utils import write_jsonl  # noqa: E402
from easyr1_eval.runner import generation_config_for, iter_sample_batches, prediction_row  # noqa: E402
from easyr1_eval.schemas import (  # noqa: E402
    BenchmarkSpec,
    EvalSample,
    GenerationConfig,
    GenerationOutput,
)
from easyr1_eval.scorers import extract_normalized_boxes, generation_diagnostics  # noqa: E402

from verl.workers.agent.backends import (  # noqa: E402
    VLLMAgentRequestError,
    VLLMAgentSchedulerError,
)
from verl.workers.agent.protocol import (  # noqa: E402
    AgentLoopConfig,
    AgentMetrics,
    AgentStatus,
    AgentStep,
    AgentTrajectory,
    ToolCall,
    ToolResult,
)


def test_iter_sample_batches_respects_sample_and_image_limits():
    samples = [
        EvalSample("bench", "1", "p", "t", images=["a.jpg"]),
        EvalSample("bench", "2", "p", "t", images=["a.jpg"] * 8),
        EvalSample("bench", "3", "p", "t", images=["a.jpg"] * 8),
        EvalSample("bench", "4", "p", "t", images=[]),
    ]

    batches = list(iter_sample_batches(samples, max_samples=8, max_images=8))

    assert [[sample.sample_id for sample in batch] for batch in batches] == [["1"], ["2"], ["3", "4"]]


def test_process_eval_image_downscales_before_returning_rgb_image():
    image = Image.new("RGB", (400, 200), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    processed = process_eval_image(buffer.getvalue(), min_pixels=None, max_pixels=10_000)

    assert processed.mode == "RGB"
    assert processed.width * processed.height <= 10_000


def test_prediction_jsonl_keeps_message_order_and_compacts_extra_info(tmp_path):
    sample = EvalSample(
        benchmark="refcoco",
        sample_id="1",
        prompt="Locate the person.",
        target="person",
        messages=[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Locate the person."}]}],
        extra_info={
            "bbox": [0, 0, 1, 1],
            "objects": [
                {
                    "label": "person",
                    "bbox": [0, 0, 1, 1],
                    "patches": [1, 2, 3],
                    "rle": {"counts": "compressed-mask", "size": [10, 10]},
                }
            ],
        },
        images=["image.jpg"],
    )

    row = prediction_row(
        sample,
        [GenerationOutput("[0, 0, 1000, 1000]", finish_reason="length", token_count=1024, truncated=True)],
        eval_metadata={"max_model_len": 32768},
    )
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [row])
    text = path.read_text(encoding="utf-8")

    assert text.index('"role"') < text.index('"content"')
    assert "compressed-mask" not in text
    assert '"patches"' not in text
    assert row["responses"] == ["[0, 0, 1000, 1000]"]
    assert row["response_metadata"][0]["finish_reason"] == "length"
    assert row["response_metadata"][0]["truncated"] is True
    assert row["eval_metadata"]["max_model_len"] == 32768


def test_generation_config_allows_global_max_new_tokens_override():
    spec = BenchmarkSpec(
        key="pope",
        label="POPE",
        group="Grounding",
        loader="pope",
        scorer="pope",
        primary_metric="macro_f1",
        max_new_tokens=1024,
        temperature=0.0,
        num_samples=1,
    )
    args = Namespace(temperature=None, num_samples=None, top_p=1.0, max_new_tokens=4096, seed=42)

    config = generation_config_for(spec, args)

    assert config.max_new_tokens == 4096


class _DiagnosticTokenizer:
    all_special_ids = [0, 255]

    def decode(self, token_ids, **kwargs):
        del kwargs
        if len(token_ids) > 1 and all(isinstance(token_id, int) and 32 <= token_id < 127 for token_id in token_ids):
            return bytes(token_ids).decode("ascii")
        return ",".join(str(token_id) for token_id in token_ids)


def _install_tracking_agent_scheduler(monkeypatch):
    instances = []

    class TrackingAgentScheduler:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.close_calls = 0
            instances.append(self)

        async def close(self):
            self.close_calls += 1

    monkeypatch.setattr(
        "easyr1_eval.agentic.VLLMAgentBatchScheduler",
        TrackingAgentScheduler,
    )
    return instances


def _agent_trajectory(
    *,
    text="<think>Done.</think><answer>yes</answer>",
    status=AgentStatus.ANSWERED,
):
    step = AgentStep(
        model_text=text,
        model_token_ids=[*text.encode("ascii"), 0],
        finish_reason="stop",
        stop_reason=0,
    )
    return AgentTrajectory(
        prompt_ids=[1, 2],
        response_ids=[*text.encode("ascii"), 0],
        response_mask=[1] * (len(text.encode("ascii")) + 1),
        source_images=[Image.new("RGB", (8, 8))],
        observation_images=[],
        steps=[step],
        status=status,
        metrics=AgentMetrics(action_tokens=2),
        final_answer="yes" if status == AgentStatus.ANSWERED else None,
        effective_response_tokens=2,
        response_token_budget=32,
        assistant_termination_token_ids=(0,),
    )


def test_agentic_prediction_preserves_action_text_and_structured_diagnostics():
    trajectory = _agent_trajectory()
    output = trajectory_to_generation_output(
        trajectory,
        tokenizer=_DiagnosticTokenizer(),
    )
    sample = EvalSample(
        benchmark="bench",
        sample_id="1",
        prompt="Question",
        target="yes",
        images=[Image.new("RGB", (8, 8))],
    )

    row = prediction_row(sample, [output])
    details = generation_diagnostics([row])

    assert row["responses"] == ["<think>Done.</think><answer>yes</answer>"]
    assert row["agent_diagnostics"][0]["status"] == "answered"
    assert row["agent_diagnostics"][0]["final_answer"] == "yes"
    assert details["agent/trajectory_count"] == 1
    assert details["agent/statuses"] == {"answered": 1}
    assert details["agent/no_action_rate"] == 0.0


def test_agentic_diagnostics_count_executable_tool_calls_inside_reasoning():
    tool_call = '<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,0,1000,1000]}}</tool_call>'
    trajectory = _agent_trajectory(text="<think>Inspect." + tool_call)
    output = trajectory_to_generation_output(
        trajectory,
        tokenizer=_DiagnosticTokenizer(),
    )
    sample = EvalSample(
        benchmark="bench",
        sample_id="inside-reasoning",
        prompt="Question",
        target="yes",
        images=[Image.new("RGB", (8, 8))],
    )

    row = prediction_row(sample, [output])
    details = generation_diagnostics([row])

    assert row["agent_diagnostics"][0]["tool_calls_inside_reasoning"] == 1
    assert details["agent/tool_calls_inside_reasoning_total"] == 1


def test_agentic_scoring_uses_only_final_reply_and_excludes_tool_bbox():
    tool_text = (
        "<think>I need a closer look.</think>"
        '<tool_call>{"name":"image_zoom_in_tool","arguments":'
        '{"bbox_2d":[100,200,300,400]}}</tool_call>'
    )
    final_text = '<think>{"label":"target","bbox_list":[[600,650,800,900]]}</think><answer>yes</answer>'
    steps = [
        AgentStep(
            model_text=tool_text,
            model_token_ids=[*tool_text.encode("ascii"), 0],
        ),
        AgentStep(
            model_text=final_text,
            model_token_ids=[*final_text.encode("ascii"), 0],
        ),
    ]
    trajectory = AgentTrajectory(
        prompt_ids=[1],
        response_ids=[],
        response_mask=[],
        source_images=[Image.new("RGB", (8, 8))],
        observation_images=[],
        steps=steps,
        status=AgentStatus.ANSWERED,
        metrics=AgentMetrics(),
        final_answer="yes",
        assistant_termination_token_ids=(0,),
    )

    output = trajectory_to_generation_output(
        trajectory,
        tokenizer=_DiagnosticTokenizer(),
    )

    assert "image_zoom_in_tool" not in output.text
    assert [0.1, 0.2, 0.3, 0.4] not in [list(box) for box in extract_normalized_boxes(output.text)]
    assert extract_normalized_boxes(output.text) == [(0.6, 0.65, 0.8, 0.9)]


def test_agentic_diagnostics_persist_only_committed_successful_tool_regions():
    trajectory = _agent_trajectory()
    committed = AgentStep(
        model_text="<tool_call>committed</tool_call>",
        model_token_ids=[1],
        tool_call=ToolCall(
            name="image_zoom_in_tool",
            arguments={"bbox_2d": [100, 200, 300, 400]},
            raw_text="committed",
        ),
        tool_result=ToolResult(
            tool_name="image_zoom_in_tool",
            success=True,
            content={"status": "success"},
            metadata={
                "image_idx": 0,
                "bbox_2d": [100, 200, 300, 400],
                "label": "cat",
            },
        ),
        observation_committed=True,
    )
    uncommitted = AgentStep(
        model_text="<tool_call>uncommitted</tool_call>",
        model_token_ids=[2],
        tool_call=ToolCall(
            name="image_zoom_in_tool",
            arguments={"bbox_2d": [500, 500, 700, 700]},
            raw_text="uncommitted",
        ),
        tool_result=ToolResult(
            tool_name="image_zoom_in_tool",
            success=True,
            content={"status": "success"},
            metadata={
                "image_idx": 0,
                "bbox_2d": [500, 500, 700, 700],
            },
        ),
        observation_committed=False,
    )
    trajectory.steps = [committed, uncommitted, *trajectory.steps]

    output = trajectory_to_generation_output(
        trajectory,
        tokenizer=_DiagnosticTokenizer(),
    )

    assert output.diagnostics["agent"]["committed_tool_regions"] == [
        {
            "turn_index": 0,
            "image_idx": 0,
            "bbox_norm1000": [100, 200, 300, 400],
            "bbox_2d": [100, 200, 300, 400],
            "bbox_format": "norm1000",
            "label": "cat",
        }
    ]


def test_agentic_scoring_restores_template_prefilled_think_scope():
    text = "reasoning</think><answer>yes</answer>"
    trajectory = _agent_trajectory(text=text)
    trajectory.steps[0].reasoning_prefilled = True

    output = trajectory_to_generation_output(
        trajectory,
        tokenizer=_DiagnosticTokenizer(),
    )

    assert output.text == "<think>reasoning</think><answer>yes</answer>"


def test_agentic_scoring_strips_alternative_special_stop_token():
    text = "<answer>yes</answer>"
    trajectory = _agent_trajectory(text=text, status=AgentStatus.INVALID_TURN_END)
    trajectory.steps[0].model_token_ids = [*text.encode("ascii"), 255]
    trajectory.steps[0].stop_reason = 255

    output = trajectory_to_generation_output(
        trajectory,
        tokenizer=_DiagnosticTokenizer(),
    )

    assert output.text == text


def test_agentic_turn_end_diagnostics_keep_backend_reason_and_token_classification():
    trajectory = _agent_trajectory(status=AgentStatus.INVALID_TURN_END)
    trajectory.metrics.turn_end_errors = 1
    trajectory.steps[0].turn_end_error = "noncanonical"
    trajectory.steps[0].stop_reason = 255
    trajectory.steps[0].expected_termination_token_ids = [0]
    trajectory.steps[0].actual_termination_tail_ids = [255]

    output = trajectory_to_generation_output(
        trajectory,
        tokenizer=_DiagnosticTokenizer(),
    )
    failure = output.diagnostics["agent"]["turn_end_failures"][0]

    assert output.stop_reason == 255
    assert failure == {
        "turn_index": 0,
        "stop_reason": 255,
        "stop_reason_text": "255",
        "expected_token_ids": [0],
        "expected_text": "0",
        "actual_tail_token_ids": [255],
        "actual_tail_text": "255",
    }


def test_agentic_backend_isolates_one_bad_sample_and_preserves_rollout_order(
    monkeypatch,
):
    schedulers = _install_tracking_agent_scheduler(monkeypatch)
    backend = object.__new__(AgenticVLLMBackend)
    backend.agent_profile = "deepeyes"
    backend.agent_config = AgentLoopConfig(max_response_tokens=32)
    backend.agent_max_images_per_prompt = 8
    backend.agent_max_batch_images = 8
    backend.agent_tool_image_mode = "text_skipped"
    backend.agent_bbox_format = "norm1000"
    backend.processor = object()
    backend.tokenizer = _DiagnosticTokenizer()
    backend.min_pixels = None
    backend.max_pixels = None
    backend.max_model_len = None
    backend.base = SimpleNamespace(llm=object())

    observed_tool_image_modes = []

    async def fake_agent_runner(**kwargs):
        observed_tool_image_modes.append(kwargs["tool_image_mode"])
        if kwargs["messages"][0]["content"] == "bad":
            raise ValueError("bad sample")
        return _agent_trajectory()

    backend.agent_runner = fake_agent_runner
    samples = [
        EvalSample(
            "bench",
            "good",
            "Question",
            "yes",
            images=[Image.new("RGB", (8, 8))],
            messages=[{"role": "user", "content": "good"}],
        ),
        EvalSample(
            "bench",
            "bad",
            "Question",
            "yes",
            images=[Image.new("RGB", (8, 8))],
            messages=[{"role": "user", "content": "bad"}],
        ),
    ]
    config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        num_samples=2,
        max_new_tokens=32,
        seed=42,
    )

    outputs = backend.generate(samples, config)

    assert [len(items) for items in outputs] == [2, 2]
    assert observed_tool_image_modes == ["text_skipped"] * 4
    assert all(output.diagnostics["agent"]["status"] == "answered" for output in outputs[0])
    assert all(output.diagnostics["agent"]["status"] == "sample_error" for output in outputs[1])
    assert all(output.text == "" for output in outputs[1])
    assert len(schedulers) == 1
    assert schedulers[0].close_calls == 1


def test_agentic_backend_fails_fast_when_every_sample_has_the_same_engine_failure():
    backend = object.__new__(AgenticVLLMBackend)
    backend.agent_profile = "deepeyes"
    backend.agent_config = AgentLoopConfig(max_response_tokens=32)
    backend.agent_max_images_per_prompt = 8
    backend.agent_max_batch_images = 8
    backend.agent_tool_image_mode = "original"
    backend.agent_bbox_format = "norm1000"
    backend.processor = object()
    backend.tokenizer = _DiagnosticTokenizer()
    backend.min_pixels = None
    backend.max_pixels = None
    backend.max_model_len = None
    backend.base = SimpleNamespace(llm=object())

    async def failed_agent_runner(**kwargs):
        del kwargs
        raise VLLMAgentRequestError("engine unavailable")

    backend.agent_runner = failed_agent_runner
    samples = [
        EvalSample(
            "bench",
            sample_id,
            "Question",
            "yes",
            images=[Image.new("RGB", (8, 8))],
            messages=[{"role": "user", "content": "question"}],
        )
        for sample_id in ("one", "two")
    ]
    config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        num_samples=1,
        max_new_tokens=32,
        seed=42,
    )

    with pytest.raises(
        RuntimeError,
        match="every trajectory in the agentic batch failed inside vLLM",
    ):
        backend.generate(samples, config)


def test_agentic_backend_fails_fast_for_single_engine_failure():
    backend = object.__new__(AgenticVLLMBackend)
    backend.agent_profile = "deepeyes"
    backend.agent_config = AgentLoopConfig(max_response_tokens=32)
    backend.agent_max_images_per_prompt = 8
    backend.agent_max_batch_images = 8
    backend.agent_tool_image_mode = "original"
    backend.agent_bbox_format = "norm1000"
    backend.processor = object()
    backend.tokenizer = _DiagnosticTokenizer()
    backend.min_pixels = None
    backend.max_pixels = None
    backend.max_model_len = None
    backend.base = SimpleNamespace(llm=object())

    async def failed_agent_runner(**kwargs):
        del kwargs
        raise VLLMAgentRequestError("engine unavailable")

    backend.agent_runner = failed_agent_runner
    samples = [
        EvalSample(
            "bench",
            "one",
            "Question",
            "yes",
            images=[Image.new("RGB", (8, 8))],
            messages=[{"role": "user", "content": "question"}],
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="every trajectory in the agentic batch failed inside vLLM",
    ):
        backend.generate(
            samples,
            GenerationConfig(0.0, 1.0, 1, 32, 42),
        )


def test_agentic_backend_rejects_image_capacity_as_configuration_error(
    monkeypatch,
):
    schedulers = _install_tracking_agent_scheduler(monkeypatch)
    backend = object.__new__(AgenticVLLMBackend)
    backend.agent_profile = "deepeyes"
    backend.agent_config = AgentLoopConfig(
        max_tool_calls=6,
        max_response_tokens=32,
    )
    backend.agent_max_images_per_prompt = 8
    backend.agent_max_batch_images = 8
    backend.agent_tool_image_mode = "original"
    backend.agent_bbox_format = "norm1000"
    backend.processor = object()
    backend.tokenizer = _DiagnosticTokenizer()
    backend.min_pixels = None
    backend.max_pixels = None
    backend.max_model_len = None
    backend.base = SimpleNamespace(llm=object())
    backend.agent_runner = pytest.fail
    sample = EvalSample(
        "bench",
        "many-images",
        "Question",
        "yes",
        images=[Image.new("RGB", (8, 8)) for _ in range(3)],
        messages=[{"role": "user", "content": "question"}],
    )

    with pytest.raises(
        AgenticConfigurationError,
        match="required at least 9",
    ):
        backend.generate(
            [sample],
            GenerationConfig(0.0, 1.0, 1, 32, 42),
        )
    assert schedulers == []


def test_agentic_backend_never_downgrades_scheduler_failure_to_sample_error(
    monkeypatch,
):
    schedulers = _install_tracking_agent_scheduler(monkeypatch)
    backend = object.__new__(AgenticVLLMBackend)
    backend.agent_profile = "deepeyes"
    backend.agent_config = AgentLoopConfig(max_response_tokens=32)
    backend.agent_max_images_per_prompt = 8
    backend.agent_max_batch_images = 8
    backend.agent_tool_image_mode = "original"
    backend.agent_bbox_format = "norm1000"
    backend.processor = object()
    backend.tokenizer = _DiagnosticTokenizer()
    backend.min_pixels = None
    backend.max_pixels = None
    backend.max_model_len = 128
    backend.base = SimpleNamespace(llm=object())

    async def partially_failed_agent_runner(**kwargs):
        if kwargs["messages"][0]["content"] == "failed":
            raise VLLMAgentSchedulerError("dispatcher stopped")
        return _agent_trajectory()

    backend.agent_runner = partially_failed_agent_runner
    samples = [
        EvalSample(
            "bench",
            content,
            "Question",
            "yes",
            images=[Image.new("RGB", (8, 8))],
            messages=[{"role": "user", "content": content}],
        )
        for content in ("good", "failed")
    ]

    with pytest.raises(RuntimeError, match="shared agentic vLLM scheduler failed"):
        backend.generate(
            samples,
            GenerationConfig(0.0, 1.0, 1, 32, 42),
        )
    assert len(schedulers) == 1
    assert schedulers[0].close_calls == 1
