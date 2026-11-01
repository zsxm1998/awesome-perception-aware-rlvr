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
"""The server mode's generation backend and the driver's manager, with stand-ins for the Ray actors."""

import asyncio

import numpy as np
import pytest
import torch
from PIL import Image
from vllm import SamplingParams

from verl.protocol import DataProto
from verl.workers.agent.backends import VLLMAgentImageCache, build_agent_vllm_request
from verl.workers.agent.backends.async_server import AsyncVLLMServerBackend
from verl.workers.agent.protocol import GenerationRequest
from verl.workers.agent.rollout_batch import AgentRolloutTask
from verl.workers.rollout.async_server import agent_rollout
from verl.workers.rollout.async_server.agent_rollout import AgentRolloutManager, split_by_group
from verl.workers.rollout.config import RolloutConfig


class _Call:
    """``obj.method.remote(...)`` of a Ray actor, answered at once."""

    def __init__(self, function):
        self.remote = function


class _ImmediateRay:
    """The ``ray`` calls the manager makes, for stand-ins whose ``.remote`` returns the value itself."""

    @staticmethod
    def get(value):
        return value

    @staticmethod
    def wait(pending, num_returns):
        return pending[:num_returns], pending[num_returns:]

    @staticmethod
    def cancel(value):
        pass


class _Tokenizer:
    def decode(self, token_ids, **kwargs):
        return " ".join(map(str, token_ids))


def test_the_server_backend_sends_the_batch_modes_request_without_detokenization():
    image = Image.new("RGB", (28, 28), "red")
    request = GenerationRequest(prompt_token_ids=[1, 2], images=[image, image], max_new_tokens=9, seed=41)
    base = SamplingParams(n=1, temperature=1.0, logit_bias={5: -100.0}, logprobs=0)
    cache = VLLMAgentImageCache(min_pixels=None, max_pixels=None)
    sent, released = [], []

    class Server:
        generate = _Call(
            lambda prompt, sampling_params, request_id: (
                sent.append((prompt, sampling_params))
                or {
                    "token_ids": [7, 8],
                    "text": None,
                    "finish_reason": "stop",
                    "stop_reason": 8,
                    "log_probs": [-1.0, -2.0],
                }
            )
        )

    router = type("Router", (), {"acquire": _Call(lambda key, group: 1), "release": _Call(released.append)})()
    backend = AsyncVLLMServerBackend(
        servers=[None, Server()],
        router=router,
        sticky_key=("u", 0, 0),
        group_key="u",
        sampling_params=base,
        tokenizer=_Tokenizer(),
        image_cache=cache,
        forbidden_token_ids=(3,),
    )

    output = asyncio.run(backend.generate(request))

    expected_prompt, expected_params = build_agent_vllm_request(
        request, base, image_cache=cache, forbidden_token_ids=(3,)
    )
    ((prompt, params),) = sent
    assert prompt == expected_prompt  # the same tokens, images and multimodal uuids as the batch mode's request
    assert params.seed == 41 and params.max_tokens == 9 and params.logit_bias == {5: -100.0, 3: -100.0}
    assert params.detokenize is False and expected_params.detokenize is True
    assert (output.token_ids, output.text, output.log_probs, output.backend_text) == (
        [7, 8],
        "7 8",
        [-1.0, -2.0],
        None,
    )
    assert released == [1]


def test_a_failed_release_fails_the_turn_unless_the_turn_failed_first(caplog):
    async def release(replica):
        raise RuntimeError("the router is gone")

    async def failed_generation(*args):
        raise ValueError("the server failed")

    def backend(generate):
        return AsyncVLLMServerBackend(
            servers=[type("Server", (), {"generate": _Call(generate)})()],
            router=type("Router", (), {"acquire": _Call(lambda key, group: 0), "release": _Call(release)})(),
            sticky_key="t",
            group_key="g",
            sampling_params=SamplingParams(n=1, seed=1),
            tokenizer=_Tokenizer(),
            image_cache=VLLMAgentImageCache(min_pixels=None, max_pixels=None),
        )

    request = GenerationRequest(prompt_token_ids=[1], images=[], max_new_tokens=2)
    reply = {"token_ids": [7], "text": None, "finish_reason": "stop", "stop_reason": None, "log_probs": None}
    # the generation succeeded, but the router still counts the turn: the turn fails
    with pytest.raises(RuntimeError, match="the router is gone"):
        asyncio.run(backend(lambda *args: reply).generate(request))
    # the generation failed first: its error is the one raised, the release's is logged
    with pytest.raises(ValueError, match="the server failed"):
        asyncio.run(backend(failed_generation).generate(request))
    assert "the router is gone" in caplog.text


def test_a_failed_environment_close_fails_the_worker_unless_a_trajectory_failed_first(monkeypatch, caplog):
    class Context:  # the environment context of the worker's call
        def __init__(self, **kwargs):
            pass

        async def aclose(self):
            raise RuntimeError("closing the environments failed")

    monkeypatch.setattr(agent_rollout, "EnvironmentContext", Context)
    monkeypatch.setattr(agent_rollout, "environment_factory", lambda *args: None)
    config = RolloutConfig(interaction_mode="agentic", agent_environment="remote_http")
    config.prompt_length, config.response_length = 16, 32
    processor = type("Processor", (), {"tokenizer": _Tokenizer()})()
    worker = agent_rollout.AgentRolloutWorker(config, processor, servers=[], router=None, bbox_format="qwen")

    def run():
        return asyncio.run(worker.run([], SamplingParams(n=1, seed=1), {}))

    with pytest.raises(RuntimeError, match="closing the environments failed"):
        run()

    async def failed_trajectory(coroutines):
        raise ValueError("a trajectory failed")

    monkeypatch.setattr(agent_rollout, "run_all", failed_trajectory)
    with pytest.raises(ValueError, match="a trajectory failed"):
        run()
    assert "closing the environments failed" in caplog.text


def _tasks(groups, n):
    return [
        AgentRolloutTask(
            uid=f"u{row}", sample_position=row, rollout_index=index, env_config={}, env_seed=0, sample_index=row
        )
        for row in range(groups)
        for index in range(n)
    ]


def test_groups_stay_whole_and_the_split_is_even():
    chunks = split_by_group(_tasks(5, 3), 2)
    assert [len(chunk) for chunk in chunks] == [6, 9]
    assert all(len({task.sample_position for task in chunk}) * 3 == len(chunk) for chunk in chunks)
    assert len(split_by_group(_tasks(1, 4), 8)) == 1  # never more parts than groups


def test_the_manager_restores_the_batch_order_and_leaves_the_drivers_batch_alone(monkeypatch):
    config = RolloutConfig(interaction_mode="agentic", agent_environment="remote_http", n=2)
    config.prompt_length, config.response_length = 16, 32
    received, captured = [], {}

    class Worker:
        def __init__(self):
            self.run = _Call(lambda tasks, sampling_params, meta: self._run(tasks, sampling_params, meta))

        def _run(self, tasks, sampling_params, meta):
            received.append(([task.sample_position for task in tasks], sampling_params.n, meta["eos_token_id"]))
            return [(f"trajectory {task.uid}/{task.rollout_index}", "sample", 0) for task in tasks]

    def fake_output(tasks, trajectories, materialized, retry_counts, **kwargs):
        captured.update(tasks=tasks, trajectories=trajectories, meta=kwargs["meta_info"])
        return "batch"

    monkeypatch.setattr(agent_rollout, "agent_rollout_output", fake_output)
    resets = []
    router = type("Router", (), {"reset": _Call(lambda: resets.append(True))})()
    manager = AgentRolloutManager(
        config=config,
        tokenizer=type("T", (), {"pad_token_id": 0})(),
        base_sampling_params=SamplingParams(n=2, seed=5),
        generation_meta_info={"eos_token_id": 9, "pad_token_id": 0},
        workers=[Worker(), Worker()],
        router=router,
    )
    prompts = DataProto.from_dict(
        tensors={"input_ids": torch.ones((3, 2), dtype=torch.long)},
        non_tensors={
            "uid": np.array(["a", "b", "c"], dtype=object),
            "env_config": np.array(["{}", "{}", "{}"], dtype=object),
        },
        meta_info={"min_pixels": None},
    )

    assert manager.generate_sequences(prompts, ray_api=_ImmediateRay) == "batch"
    assert received == [([0, 0], 1, 9), ([1, 1, 2, 2], 1, 9)]  # whole groups, one trajectory per request
    assert captured["trajectories"] == [f"trajectory {uid}/{index}" for uid in "abc" for index in range(2)]
    assert captured["meta"]["eos_token_id"] == 9 and resets == [True]
    # the driver's batch is intact
    assert "uid" in prompts.non_tensor_batch and "eos_token_id" not in prompts.meta_info
    with pytest.raises(ValueError, match="integer sampling seed"):
        manager.base_sampling_params = SamplingParams(n=2)
        manager.generate_sequences(prompts, ray_api=_ImmediateRay)


def _slow_actors():
    """A server whose generation runs until cancelled, and a router whose acquire takes a while (defined here, so that
    Ray ships them to its workers by value)."""
    from verl.workers.rollout.async_server.router import ReplicaRouter

    class SlowServer:
        def __init__(self):
            self.active, self.cancelled = False, False

        async def generate(self, *args):
            self.active = True
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            finally:
                await asyncio.sleep(0.2)  # vLLM aborts the request
                self.active = False

        def status(self):
            return self.active, self.cancelled

    class SlowRouter(ReplicaRouter):
        async def acquire(self, sticky_key, group_key=None):
            await asyncio.sleep(1.0)
            return super().acquire(sticky_key, group_key)

    return SlowServer, SlowRouter


@pytest.fixture(scope="module")
def local_ray():
    import ray

    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, log_to_driver=False)
    yield ray
    ray.shutdown()


def test_cancelling_a_turn_cancels_the_servers_generation_before_the_router_lets_it_go(local_ray):
    from verl.workers.rollout.async_server.router import ReplicaRouter

    slow_server, slow_router = _slow_actors()
    server = local_ray.remote(slow_server).options(num_cpus=0).remote()
    router = local_ray.remote(ReplicaRouter).options(num_cpus=0).remote(1)
    request = GenerationRequest(prompt_token_ids=[1], images=[], max_new_tokens=2)

    def backend(router):
        return AsyncVLLMServerBackend(
            servers=[server],
            router=router,
            sticky_key="t",
            group_key="g",
            sampling_params=SamplingParams(n=1, seed=1),
            tokenizer=_Tokenizer(),
            image_cache=VLLMAgentImageCache(min_pixels=None, max_pixels=None),
        )

    async def cancel_while_generating():
        task = asyncio.ensure_future(backend(router).generate(request))
        while not (await server.status.remote())[0]:
            await asyncio.sleep(0.05)
        assert await router.snapshot.remote() == [1]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await server.status.remote(), await router.snapshot.remote()

    # the server's call ended (cancelled: vLLM aborts the request) before the router stopped counting it
    assert asyncio.run(cancel_while_generating()) == ((False, True), [0])

    counting = local_ray.remote(slow_router).options(num_cpus=0).remote(1)

    async def cancel_while_acquiring():
        task = asyncio.ensure_future(backend(counting).generate(request))
        await asyncio.sleep(0.3)  # the router is still choosing a server
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await counting.snapshot.remote()

    assert asyncio.run(cancel_while_acquiring()) == [0]  # the turn it counted meanwhile is released


def test_a_worker_cancelled_while_it_stops_its_own_turns_still_waits_for_them(local_ray):
    """A trajectory of a worker fails, so its other turn is cancelled and the worker waits for the server to end that
    request; meanwhile another worker fails and the manager cancels this one. The manager returns only after the
    server's request has ended and the router has let it go."""
    from verl.workers.rollout.async_server.router import ReplicaRouter
    from verl.workers.rollout.async_server.tasks import gather_or_cancel

    slow_server, _ = _slow_actors()

    def worker_classes():
        class Worker:
            def __init__(self, server, router):
                self.server, self.router = server, router

            async def run(self):
                from verl.workers.rollout.async_server.tasks import run_all

                backend = AsyncVLLMServerBackend(
                    servers=[self.server],
                    router=self.router,
                    sticky_key="t",
                    group_key="g",
                    sampling_params=SamplingParams(n=1, seed=1),
                    tokenizer=None,  # a cancelled turn decodes nothing
                    image_cache=VLLMAgentImageCache(min_pixels=None, max_pixels=None),
                )

                async def fails_once_the_server_runs():
                    while not (await self.server.status.remote())[0]:
                        await asyncio.sleep(0.02)
                    raise ValueError("a trajectory of this worker failed")

                request = GenerationRequest(prompt_token_ids=[1], images=[], max_new_tokens=2)
                await run_all([backend.generate(request), fails_once_the_server_runs()])

        class OtherWorker:
            async def run(self, server):
                while not (await server.status.remote())[0]:
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.1)  # the first worker is waiting for the server's request to end
                raise ValueError("another worker failed")

        return Worker, OtherWorker

    worker_class, other_class = worker_classes()
    server = local_ray.remote(slow_server).options(num_cpus=0).remote()
    router = local_ray.remote(ReplicaRouter).options(num_cpus=0).remote(1)
    worker = local_ray.remote(worker_class).options(num_cpus=0).remote(server, router)
    other = local_ray.remote(other_class).options(num_cpus=0).remote()

    with pytest.raises(local_ray.exceptions.RayTaskError, match="another worker failed"):
        gather_or_cancel([worker.run.remote(), other.run.remote(server)])
    # the server's request ended (cancelled) before the router let it go, and both before the manager returned
    assert local_ray.get([server.status.remote(), router.snapshot.remote()]) == [(False, True), [0]]
