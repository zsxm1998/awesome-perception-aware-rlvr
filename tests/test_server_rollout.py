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
"""The server mode of agentic rollouts (worker.rollout.agent_engine_mode=server) without vLLM: the router, the
bucketed weight transfer (its shared-memory path) and the configuration checks."""

import asyncio
import threading
import time

import pytest
import torch

from verl.trainer.config import PPOConfig
from verl.workers.rollout.async_server.router import ReplicaRouter
from verl.workers.rollout.async_server.weight_transfer import BucketedWeightReceiver, BucketedWeightSender
from verl.workers.rollout.config import RolloutConfig


def test_a_trajectory_stays_on_its_replica_and_new_ones_go_to_the_least_busy():
    router = ReplicaRouter(3, seed=0)
    first = router.acquire("a")
    second = router.acquire("b")
    third = router.acquire("c")
    assert sorted([first, second, third]) == [0, 1, 2]  # each new trajectory takes an idle replica
    router.release(second)
    assert router.acquire("d") == second  # the only replica with nothing in flight
    router.release(first)
    assert router.acquire("a") == first  # the next turn of "a" goes where its context is cached
    assert router.acquire("a") == first and router.snapshot()[first] == 2  # even when that replica is busier
    for _ in range(2):
        router.release(first)
    with pytest.raises(RuntimeError, match="released more"):
        router.release(first)


def test_ties_are_broken_at_random_and_a_new_batch_forgets_the_trajectories():
    picks = {ReplicaRouter(4, seed=seed).acquire("t") for seed in range(20)}
    assert len(picks) > 1  # not always the first replica (verl #7613)
    router = ReplicaRouter(2, seed=0)
    replica = router.acquire("t")
    with pytest.raises(RuntimeError, match="in flight"):
        router.reset()
    router.release(replica)
    router.reset()
    assert router.sticky == {}


def test_group_affinity_keeps_a_group_together_within_the_slack():
    router = ReplicaRouter(2, group_affinity=True, affinity_slack=1, seed=0)
    home = router.acquire(("g", 0), "g")
    assert router.acquire(("g", 1), "g") == home  # one more in flight than the idle replica: within the slack
    assert router.acquire(("g", 2), "g") == 1 - home  # two more: the least busy replica takes it
    plain = ReplicaRouter(2, seed=0)
    first = plain.acquire(("g", 0), "g")
    assert plain.acquire(("g", 1), "g") == 1 - first  # without affinity a group spreads (as verl)


def test_bucketed_weights_arrive_whole_in_order_through_shared_memory(tmp_path):
    torch.manual_seed(0)
    weights = [
        ("embed", torch.randn(300, 512)),  # 600 KiB, about half a bucket
        ("norm", torch.randn(7).to(torch.bfloat16)),  # odd size: the next tensor is realigned
        ("proj", torch.randn(256, 512)),  # does not fit in what is left of the bucket: a new bucket
        ("bias", torch.arange(5, dtype=torch.int64)),
    ]
    handle = f"ipc://{tmp_path}/weights.sock"
    received, buckets = [], []

    def receive():
        def on_bucket(bucket, is_last):
            buckets.append(([name for name, _ in bucket], is_last))
            received.extend((name, tensor.clone()) for name, tensor in bucket)

        BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True).receive_weights(on_bucket)

    thread = threading.Thread(target=receive)
    thread.start()
    BucketedWeightSender(handle, bucket_size_mb=1, use_shm=True).send_weights(iter(weights))
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert [name for name, _ in received] == [name for name, _ in weights]
    for (_, sent), (_, got) in zip(weights, received):
        assert got.dtype == sent.dtype and torch.equal(got, sent)
    assert [is_last for _, is_last in buckets] == [False, True] and buckets[0][0] == ["embed", "norm"]


def _receive_in_thread(handle, on_bucket, errors, **kwargs):
    def receive():
        try:
            BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True, **kwargs).receive_weights(
                on_bucket
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    return thread


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_a_receiver_error_reaches_the_sender_instead_of_a_wait_without_end(tmp_path):
    handle = f"ipc://{tmp_path}/weights.sock"
    errors = []

    def fail(bucket, is_last):
        raise ValueError("shape mismatch while loading")

    thread = _receive_in_thread(handle, fail, errors)
    with pytest.raises(RuntimeError, match="weight receiver failed: ValueError: shape mismatch"):
        BucketedWeightSender(handle, bucket_size_mb=1, use_shm=True).send_weights([("x", torch.ones(2))])
    thread.join(timeout=30)
    assert not thread.is_alive() and isinstance(errors[0], ValueError)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize("explicit", [True, False], ids=["raise-from", "raised-while-handling"])
def test_a_chained_receiver_error_frees_the_buffer_too(tmp_path, explicit):
    # the loader wraps a lower-level error, whose frames hold views of the buffer
    handle = f"ipc://{tmp_path}/weights.sock"
    errors = []

    def load(bucket):
        raise ValueError("shape mismatch while loading")

    def on_bucket(bucket, is_last):
        try:
            load(bucket)
        except ValueError as exc:
            if explicit:
                raise RuntimeError("loading the model failed") from exc
            raise RuntimeError("loading the model failed")  # the ValueError is its context

    thread = _receive_in_thread(handle, on_bucket, errors)
    with pytest.raises(RuntimeError, match="weight receiver failed: RuntimeError: loading the model failed"):
        BucketedWeightSender(handle, bucket_size_mb=1, use_shm=True).send_weights([("x", torch.ones(2))])
    thread.join(timeout=30)
    assert not thread.is_alive() and isinstance(errors[0], RuntimeError)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_a_buffer_that_cannot_be_freed_after_a_receiver_error_does_not_hide_that_error(tmp_path, caplog):
    handle = f"ipc://{tmp_path}/weights.sock"
    receiver = BucketedWeightReceiver(handle, device=torch.device("cpu"), use_shm=True)
    kept, errors = [], []

    def on_bucket(bucket, is_last):
        kept.extend(bucket)  # against the contract: views of the buffer outlive the call
        raise ValueError("shape mismatch while loading")

    def receive():
        try:
            receiver.receive_weights(on_bucket)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    with pytest.raises(RuntimeError, match="weight receiver failed: ValueError: shape mismatch"):
        BucketedWeightSender(handle, bucket_size_mb=1, use_shm=True).send_weights([("x", torch.ones(2))])
    thread.join(timeout=30)
    assert not thread.is_alive() and isinstance(errors[0], ValueError)  # not the BufferError of closing it
    assert "releasing the weight buffer failed" in caplog.text
    kept.clear()
    receiver.shm.close()  # the views are gone now


def test_the_sender_stops_waiting_when_the_receiving_side_has_failed_or_never_answers(tmp_path):
    # nobody receives: the receiving call is known to have failed (its process died), or the wait times out
    class ReceiverDied(Exception):
        pass

    def check():
        raise ReceiverDied("the server's update_weights call failed")

    started = time.monotonic()
    with pytest.raises(ReceiverDied):
        BucketedWeightSender(f"ipc://{tmp_path}/a.sock", bucket_size_mb=1, use_shm=True).send_weights(
            [("x", torch.ones(2))], check_receiver=check
        )
    with pytest.raises(TimeoutError, match="did not answer within 1 s"):
        BucketedWeightSender(f"ipc://{tmp_path}/b.sock", bucket_size_mb=1, use_shm=True, timeout_s=1).send_weights(
            [("x", torch.ones(2))]
        )
    assert time.monotonic() - started < 15


@pytest.fixture(scope="module")
def local_ray():
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, log_to_driver=False)
    yield ray
    ray.shutdown()


def _worker_class():
    """An agent worker stand-in (defined here, so that Ray ships the class to its workers by value): ``run`` fails
    once its peer is running, or runs until cancelled; ``status`` says whether its run is still active and whether
    its cleanup ran."""

    class Worker:
        def __init__(self, fails):
            self.fails, self.active, self.closed, self.peer = fails, False, False, None

        def set_peer(self, peer):
            self.peer = peer

        async def run(self, value):
            self.active = True
            try:
                if self.fails:
                    while not (await self.peer.status.remote())[0]:
                        await asyncio.sleep(0.05)
                    raise ValueError("injected worker failure")
                if value is not None:
                    return value
                await asyncio.sleep(60)
            finally:
                await asyncio.sleep(0.2)  # closing the environments takes a moment
                self.active, self.closed = False, True

        def status(self):
            return self.active, self.closed

    return Worker


def test_a_failed_worker_stops_the_others_after_their_cleanup(local_ray):
    from verl.workers.rollout.async_server.tasks import gather_or_cancel

    worker = local_ray.remote(_worker_class()).options(num_cpus=0)
    failing, running = worker.remote(True), worker.remote(False)
    local_ray.get(failing.set_peer.remote(running))

    with pytest.raises(local_ray.exceptions.RayTaskError, match="injected worker failure"):
        gather_or_cancel([failing.run.remote(None), running.run.remote(None)])
    assert local_ray.get(running.status.remote()) == (False, True)  # cancelled, and its cleanup ran

    ok = [worker.remote(False) for _ in range(2)]
    assert gather_or_cancel([ok[0].run.remote("a"), ok[1].run.remote("b")]) == ["a", "b"]


def test_a_second_cancellation_does_not_cut_the_first_ones_cleanup_short():
    from verl.workers.rollout.async_server.tasks import call_and_cancel_with, finish, run_all

    async def scenario():
        ended = []

        async def remote_call():  # a server's generation: it ends a moment after its cancellation
            await asyncio.sleep(0.5)
            ended.append("remote call")

        class RemoteRef:  # like a Ray ObjectRef: cancelling a local wait for it leaves the remote call running
            def __init__(self, task):
                self.task = task

            def __await__(self):
                return asyncio.shield(self.task).__await__()

        remote = RemoteRef(asyncio.ensure_future(remote_call()))

        async def failing():
            await asyncio.sleep(0.05)
            raise ValueError("a trajectory of this worker failed")

        worker = asyncio.ensure_future(run_all([call_and_cancel_with(remote), failing()]))
        await asyncio.sleep(0.15)  # the worker waits for the remote call to end
        worker.cancel()  # the manager cancels the worker, as another worker failed
        with pytest.raises((ValueError, asyncio.CancelledError)):
            await worker
        assert ended == ["remote call"]  # the worker ended only after the call it was waiting for

        async def cleanup():
            await asyncio.sleep(0.2)
            ended.append("cleanup")

        waiter = asyncio.ensure_future(finish(cleanup()))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        return ended

    # finish waited for the cleanup, then raised the cancellation
    assert asyncio.run(scenario()) == ["remote call", "cleanup"]


def test_a_cleanups_error_is_raised_unless_another_error_goes_on(caplog):
    from verl.workers.rollout.async_server.tasks import finish

    async def broken_close():
        await asyncio.sleep(0.05)
        raise RuntimeError("closing the HTTP pool failed")

    with pytest.raises(RuntimeError, match="closing the HTTP pool failed"):
        asyncio.run(finish(broken_close()))

    async def cancelled_meanwhile():
        waiter = asyncio.ensure_future(finish(broken_close()))
        await asyncio.sleep(0.01)
        waiter.cancel()
        await waiter

    with pytest.raises(RuntimeError, match="closing the HTTP pool failed"):  # the error, not only the cancellation
        asyncio.run(cancelled_meanwhile())

    async def while_failing():
        try:
            raise ValueError("a trajectory failed")
        except ValueError:
            await finish(broken_close(), quiet=True)
            raise

    with pytest.raises(ValueError, match="a trajectory failed"):  # the original error stays the one raised
        asyncio.run(while_failing())
    assert "closing the HTTP pool failed" in caplog.text  # and the cleanup's is logged


def _server_rollout(**overrides):
    config = RolloutConfig(interaction_mode="agentic", agent_engine_mode="server", tensor_parallel_size=1)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_the_server_mode_is_checked_where_it_is_configured():
    _server_rollout().post_init()
    assert _server_rollout().resolved_record_rollout_log_probs  # on by default in the server mode, as verl
    assert not _server_rollout(record_rollout_log_probs=False).resolved_record_rollout_log_probs
    for overrides, message in (
        ({"interaction_mode": "one_shot"}, "requires rollout.interaction_mode=agentic"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=1 only"),
        ({"agent_max_batch_images": 256}, "set it to 0"),
        ({"server_sleep_level": 3}, "server_sleep_level"),
        ({"agent_num_workers": 0}, "agent_num_workers"),
        ({"agent_engine_mode": "continuous"}, "'batch' or 'server'"),
    ):
        with pytest.raises(ValueError, match=message):
            _server_rollout(**overrides).post_init()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"worker.actor.model.lora.rank": 8}, "does not support LoRA"),
        ({"algorithm.adv_estimator": "remax"}, "does not support adv_estimator=remax"),
    ],
)
def test_the_trainer_refuses_what_the_server_mode_does_not_support(overrides, message):
    config = PPOConfig()
    config.worker.rollout.agent_engine_mode = "server"
    for path, value in overrides.items():
        target = config
        *parents, name = path.split(".")
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, name, value)
    with pytest.raises(ValueError, match=message):
        config.post_init()


def _sync_manager(monkeypatch, result, send_error=None):
    """The training side of a weight sync, with stand-ins for the server, Ray and the FSDP module."""
    from types import SimpleNamespace

    from verl.workers.sharding_manager import fsdp_async_server

    class Call:
        def __init__(self, value):
            self.value = value

    class FakeRay:
        @staticmethod
        def get(call):
            if isinstance(call.value, Exception):
                raise call.value
            return call.value

        @staticmethod
        def wait(calls, timeout=None):
            return list(calls), []

    class Sender:
        def __init__(self, *args, **kwargs):
            pass

        def send_weights(self, weights, check_receiver=None):
            if send_error is not None:
                raise send_error

    server = SimpleNamespace(
        wake_up=SimpleNamespace(remote=lambda tags: Call(None)),
        update_weights=SimpleNamespace(remote=lambda handle: Call(result)),
        reset_caches=SimpleNamespace(remote=lambda: Call(None)),
    )
    monkeypatch.setattr(fsdp_async_server, "ray", FakeRay)
    monkeypatch.setattr(fsdp_async_server, "BucketedWeightSender", Sender)
    monkeypatch.setattr(fsdp_async_server, "get_model_state_dict", lambda module: {})
    monkeypatch.setattr(fsdp_async_server.dist, "barrier", lambda: None)
    monkeypatch.setattr(fsdp_async_server, "print_gpu_memory_usage", lambda message: None)
    monkeypatch.setattr(fsdp_async_server.torch.cuda, "empty_cache", lambda: None)
    module = SimpleNamespace(_fsdp_wrapped_module=object())
    manager = fsdp_async_server.FSDPAsyncServerShardingManager(module, use_param_offload=False, bucket_size_mb=1)
    manager.attach(server, "ipc:///unused")
    return manager


def test_the_first_sync_fails_unless_every_parameter_is_known_to_be_loaded(monkeypatch):
    for missing, message in ((None, "does not report the parameters"), (["lm_head.weight"], "left 1 parameters")):
        with pytest.raises(RuntimeError, match=message):
            _sync_manager(monkeypatch, {"missing": missing, "received": 3}).load_vllm_and_sync_weights()
    manager = _sync_manager(monkeypatch, {"missing": [], "received": 3})
    manager.load_vllm_and_sync_weights()
    assert manager.synced_once and manager.loaded


def test_a_failed_sync_raises_the_receivers_own_error(monkeypatch):
    receiver_error = ValueError("shape mismatch in the vLLM worker")
    manager = _sync_manager(monkeypatch, receiver_error, send_error=RuntimeError("the weight receiver failed"))
    with pytest.raises(ValueError, match="shape mismatch in the vLLM worker"):
        manager.load_vllm_and_sync_weights()
