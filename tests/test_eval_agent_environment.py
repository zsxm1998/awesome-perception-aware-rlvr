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
"""Evaluation with --agent-profile environment: task loading, the runner options, fingerprints and scoring."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from easyr1_eval import runner  # noqa: E402
from easyr1_eval.loaders import load_samples  # noqa: E402
from easyr1_eval.schemas import BenchmarkSpec  # noqa: E402
from easyr1_eval.scorers import score_environment_reward  # noqa: E402


def _spec(tmp_path, path="tasks.parquet"):
    return BenchmarkSpec(
        key="env_tasks",
        label="Environment tasks",
        group="Agent",
        loader="environment_tasks",
        scorer="environment_reward",
        primary_metric="mean_reward",
        path=path,
    )


def test_environment_tasks_loader_reads_each_rows_configuration(tmp_path):
    pd.DataFrame(
        {
            "id": ["t1", "t2"],
            "prompt": ["immune phenotype", "budget 4"],
            "env_config": [json.dumps({"task": "F1", "window": 3}), json.dumps({"task": "F2"})],
        }
    ).to_parquet(tmp_path / "tasks.parquet")

    samples = load_samples(_spec(tmp_path), tmp_path, limit=1)

    assert [(sample.sample_id, sample.prompt, sample.images) for sample in samples] == [("t1", "immune phenotype", [])]
    assert samples[0].extra_info == {"env_config": {"task": "F1", "window": 3}}


def test_environment_reward_scorer_averages_the_reported_rewards(tmp_path):
    rows = [
        {
            "sample_id": "a",
            "agent_diagnostics": [
                {"env_outcome": {"reward": 1.0, "components": {"task": 1.0}, "metrics": {"orders": 2}}},
                {"env_outcome": {"reward": 0.5, "components": {"task": 0.5}, "metrics": {"orders": 4}}},
            ],
        },
        {"sample_id": "b", "agent_diagnostics": [{"status": "sample_error"}, {"env_outcome": {"reward": 0.0}}]},
    ]

    result = score_environment_reward(_spec(tmp_path), rows, None, tmp_path)

    assert result.raw_score == pytest.approx(0.375) and result.normalized_score_0_100 == pytest.approx(37.5)
    assert result.details["trajectories_without_outcome"] == 1
    assert result.details["components"] == {"task": 0.75} and result.details["metrics"] == {"orders": 3.0}
    assert result.details["component_counts"] == {"task": 2} and result.details["metric_counts"] == {"orders": 2}


def test_environment_reward_scorer_averages_each_name_over_the_episodes_that_report_it(tmp_path):
    rows = [
        {
            "sample_id": "a",
            "agent_diagnostics": [
                {"env_outcome": {"reward": 1.0, "components": {"density": 0.8}, "metrics": {}}},
                {"env_outcome": {"reward": 0.0, "components": {"region": 0.2}, "metrics": {}}},
            ],
        }
    ]

    result = score_environment_reward(_spec(tmp_path), rows, None, tmp_path)

    assert result.details["components"] == {"density": 0.8, "region": 0.2}
    assert result.details["component_counts"] == {"density": 1, "region": 1}


def test_environment_reward_scorer_rejects_an_unscored_episode(tmp_path):
    """reward: null means that the environment does not score its episodes, as in training."""
    rows = [{"sample_id": "a", "agent_diagnostics": [{"env_outcome": {"reward": None, "components": {}}}]}]

    with pytest.raises(ValueError, match="no episode reward"):
        score_environment_reward(_spec(tmp_path), rows, None, tmp_path)


def _args(*extra):
    args = runner.parse_args(["--model", "m", "--interaction-mode", "agentic", *extra])
    return runner.expand_eval_runs(args)[0]


def test_environment_profile_options_reach_the_backend_and_the_fingerprint(monkeypatch):
    captured = {}

    def fake_build_backend(name, model, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(runner, "build_backend", fake_build_backend)
    args = _args(
        "--agent-profile",
        "environment",
        "--agent-environment-kwargs",
        '{"url": "http://127.0.0.1:8000"}',
    )
    runner.build_eval_backend(args)
    fields = runner._agent_fingerprint_fields(args)

    assert captured["agent_environment"] == "remote_http"
    assert captured["agent_environment_kwargs"] == {"url": "http://127.0.0.1:8000"}
    assert fields["agent_environment"] == "remote_http"
    assert fields["agent_environment_kwargs"] == {"url": "http://127.0.0.1:8000"}

    deepeyes = runner._agent_fingerprint_fields(
        _args("--agent-profile", "deepeyes", "--system-prompt", str(ROOT / "examples/system_prompt/deepeyes.txt"))
    )
    assert "agent_environment" not in deepeyes and "agent_environment_kwargs" not in deepeyes
    with pytest.raises(SystemExit):
        _args("--agent-profile", "deepeyes", "--agent-environment", "remote_http")
    with pytest.raises(SystemExit):
        _args("--agent-profile", "environment", "--agent-environment-kwargs", '{"uri": "http://x"}')


def test_agentic_backend_runs_environment_tasks_and_records_the_outcome(monkeypatch):
    pytest.importorskip("vllm")
    from types import SimpleNamespace

    from easyr1_eval.agentic import AgenticVLLMBackend
    from easyr1_eval.schemas import EvalSample, GenerationConfig

    from verl.workers.agent.protocol import (
        AgentCall,
        AgentLoopConfig,
        AgentMetrics,
        AgentStatus,
        AgentStep,
        AgentTrajectory,
        EnvOutcome,
        ToolCall,
    )

    class Scheduler:
        def __init__(self, *args, **kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr("easyr1_eval.agentic.VLLMAgentBatchScheduler", Scheduler)
    backend = object.__new__(AgenticVLLMBackend)
    backend.agent_profile = "environment"
    backend.agent_environment = "remote_http"
    backend.agent_environment_kwargs = {"url": "http://env.test"}
    backend.agent_config = AgentLoopConfig(max_tool_calls=20, max_response_tokens=32)
    backend.agent_max_images_per_prompt = 22
    backend.agent_max_batch_images = 8
    backend.agent_tool_image_mode = "original"
    backend.agent_bbox_format = "norm1000"
    backend.processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: ""))
    backend.tokenizer = backend.processor.tokenizer
    backend.min_pixels = backend.max_pixels = None
    backend.max_model_len = 4096
    backend.base = SimpleNamespace(llm=object())
    seen = []

    async def fake_runner(**kwargs):
        seen.append(
            (type(kwargs["environment"]).__name__, kwargs["env_config"], kwargs["environment"].context.max_tool_calls)
        )
        step = AgentStep(
            model_text="<answer>x</answer>",
            model_token_ids=[1],
            calls=[
                AgentCall(
                    tool_call=ToolCall("order", {}, ""),
                    state_key="s",
                    cost=1.0,
                    executed=True,
                )
            ],
        )
        return AgentTrajectory(
            prompt_ids=[1],
            response_ids=[1],
            response_mask=[1],
            source_images=[],
            observation_images=[],
            steps=[step],
            status=AgentStatus.ANSWERED,
            metrics=AgentMetrics(),
            final_answer="x",
            env_outcome=EnvOutcome(reward=0.5, components={"task": 1.0}, details={"note": "ok"}, episode_id="e7"),
        )

    backend.agent_runner = fake_runner
    sample = EvalSample("env_tasks", "t1", "", None, extra_info={"env_config": {"task": "F1"}})

    ((output,),) = backend.generate(
        [sample], GenerationConfig(temperature=1.0, top_p=1.0, num_samples=1, max_new_tokens=32, seed=1)
    )

    assert seen == [("RemoteHTTPEnvironment", {"task": "F1"}, 20)]
    agent = output.diagnostics["agent"]
    assert agent["env_outcome"] == {
        "reward": 0.5,
        "components": {"task": 1.0},
        "metrics": {},
        "details": {"note": "ok"},
        "episode_id": "e7",
    }
    assert agent["env_steps"] == [
        {
            "turn_index": 0,
            "call_index": 0,
            "tool": "order",
            "state_key": "s",
            "cost": 1.0,
            "reward": None,
            "done": False,
            "info": {},
        }
    ]


def test_environment_reward_counts_the_environments_step_rewards():
    from easyr1_eval.scorers import environment_row_rewards

    row = {
        "agent_diagnostics": [
            {"env_outcome": {"reward": 1.0}, "env_steps": [{"reward": -0.25}, {"reward": None}]},
            {"env_outcome": {"reward": 0.5}},
            {},  # failed before the episode ended
        ]
    }
    assert environment_row_rewards(row) == [0.75, 0.5, None]
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="not finite"):
            environment_row_rewards(
                {"agent_diagnostics": [{"env_outcome": {"reward": 1.0}, "env_steps": [{"reward": bad}]}]}
            )
        with pytest.raises(ValueError, match="not finite"):
            environment_row_rewards({"agent_diagnostics": [{"env_outcome": {"reward": bad}}]})


def test_the_environment_reward_scores_of_the_released_version_are_recomputed():
    """environment_reward changed (2: a null reward is an error, counts in the details; 3: the environment's step
    rewards count in the return): its score fingerprints change, so finished scores of earlier versions are not
    reused."""
    from easyr1_eval.scorers import SCORER_REVISIONS

    assert SCORER_REVISIONS["environment_reward"] == 3
