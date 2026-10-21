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
"""Agent environments: the registry, in-process environments and the shared inference entry."""

import asyncio

import pytest
from PIL import Image

from verl.workers.agent.environments import (
    DeepEyesEnvironment,
    EnvironmentContext,
    LocalToolEnvironment,
    environment_factory,
    register_environment,
    registered_environments,
)
from verl.workers.agent.protocol import AgentStatus, ToolCall, derive_agent_seed, derive_environment_seed
from verl.workers.agent.tools import ImageZoomInTool, ToolRegistry


def _context():
    return EnvironmentContext(processor=object(), image_preprocessor=lambda image: image)


def _completes_without_suspending(coroutine):
    """Drive a coroutine by hand: the first send must finish it (StopIteration) instead of yielding."""

    try:
        coroutine.send(None)
    except StopIteration as stop:
        return stop.value
    coroutine.close()
    raise AssertionError("the coroutine suspended")


def _zoom_call():
    return ToolCall(name="image_zoom_in_tool", arguments={"bbox_2d": [0, 0, 500, 500]}, raw_text="")


@pytest.mark.parametrize("tools_enabled", [True, False])
def test_deepeyes_environment_methods_never_suspend(tools_enabled):
    """The batched rollout's dispatch order relies on in-process environments completing every await at once."""

    environment = DeepEyesEnvironment(_context())
    image = Image.new("RGB", (100, 100))
    reset = _completes_without_suspending(
        environment.reset({"messages": [], "images": [image], "tools_enabled": tools_enabled}, seed=0)
    )
    assert reset.images == [image]
    assert len(reset.tool_schemas) == (1 if tools_enabled else 0)
    step = _completes_without_suspending(environment.step(_zoom_call()))
    assert step.result.success is tools_enabled
    assert _completes_without_suspending(environment.finalize("A", AgentStatus.ANSWERED)) is None
    assert _completes_without_suspending(environment.close()) is None


def test_local_tool_environment_runs_tools_on_its_source_images():
    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    environment = LocalToolEnvironment(ToolRegistry([ImageZoomInTool()]), [image])

    step = _completes_without_suspending(environment.step(_zoom_call()))

    assert step.result.success
    assert step.result.images[0].getpixel((0, 0)) == (255, 0, 0)
    with pytest.raises(NotImplementedError):
        _completes_without_suspending(environment.reset({}, seed=0))


def test_deepeyes_environment_declares_images_per_step_for_the_capacity_check():
    image = Image.new("RGB", (100, 100))
    config = {"messages": [], "images": [image]}

    original = asyncio.run(DeepEyesEnvironment(_context()).reset(config, seed=0))
    text_only = asyncio.run(DeepEyesEnvironment(_context(), tool_image_mode="text_skipped").reset(config, seed=0))

    assert original.max_images_per_step == 1
    assert text_only.max_images_per_step == 0


def test_environment_factory_builds_fresh_instances_and_fails_early():
    build = environment_factory("deepeyes", _context(), {"prompt_style": "official"})
    first, second = build(), build()

    assert first is not second and first.prompt_style == second.prompt_style == "official"
    with pytest.raises(ValueError, match="unknown agent environment 'nope'"):
        environment_factory("nope", _context())
    with pytest.raises(TypeError):
        environment_factory("deepeyes", _context(), {"no_such_option": 1})
    with pytest.raises(ValueError, match="prompt_style"):
        environment_factory("deepeyes", _context(), {"prompt_style": "fancy"})


def test_register_environment_rejects_a_second_class_under_one_name():
    assert "deepeyes" in registered_environments()
    with pytest.raises(ValueError, match="already registered"):

        @register_environment("deepeyes")
        class Other(LocalToolEnvironment):
            pass


def test_environment_seed_is_shared_by_a_group_and_differs_from_generation_seeds():
    seed = derive_environment_seed(7, 3)

    assert seed == derive_environment_seed(7, 3)
    assert seed != derive_environment_seed(7, 4)
    assert seed not in {derive_agent_seed(7, 3, rollout, turn) for rollout in range(8) for turn in range(4)}
    with pytest.raises(ValueError):
        derive_environment_seed(-1, 3)


def test_run_agent_inference_closes_the_environment_when_the_rollout_fails():
    SamplingParams = pytest.importorskip("vllm").SamplingParams
    from verl.workers.agent.backends import VLLMAgentBatchScheduler
    from verl.workers.agent.inference import run_agent_inference
    from verl.workers.agent.protocol import AgentImageConfig

    class RecordingEnvironment(DeepEyesEnvironment):
        closed = False

        async def close(self):
            type(self).closed = True

    with pytest.raises(ValueError, match="image capacity is too small"):
        asyncio.run(
            run_agent_inference(
                environment=RecordingEnvironment(_context()),
                env_config={"messages": [], "images": [Image.new("RGB", (8, 8))]},
                inference_engine=object(),
                sampling_params=SamplingParams(n=1, seed=1),
                processor=object(),
                sample_index=0,
                rollout_index=0,
                image_config=AgentImageConfig(limit_images=2),
                env_seed=0,
                scheduler=VLLMAgentBatchScheduler(object()),
                max_model_len=128,
            )
        )
    assert RecordingEnvironment.closed
