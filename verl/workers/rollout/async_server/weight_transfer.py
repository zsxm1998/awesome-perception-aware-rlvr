# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Bucketed weight transfer over ZMQ with CUDA IPC (or shared memory), from the training process to the vLLM worker
on the same GPU.

Ported from verl v0.9.1 (``verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py``): the sender packs the
tensors into one communication buffer whose CUDA IPC handle the receiver opened once, and sends a bucket's metadata
per round; a tensor larger than the buffer goes by its own handle. The receiver acknowledges the last bucket only
after releasing the buffer (verl #7873). Here the sender is synchronous, as the training worker's methods are, and
both sides wait for each other for a bounded time: a receiver that fails answers with its error, and the sender
checks meanwhile whether the receiving call has failed (verl's sender waits without a limit).
"""

import logging
import os
import time
import traceback
from multiprocessing import shared_memory
from typing import Any, Callable, Iterable, Optional, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor


logger = logging.getLogger(__name__)


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int = None) -> torch.Tensor:
    """Open a CUDA IPC handle; ``device_id`` is the receiver's index of the same GPU (the two processes may see
    different CUDA_VISIBLE_DEVICES)."""
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        list_args[6] = device_id
    return func(*list_args)


def create_shared_memory(size: int, name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        if shm.size < size:
            raise RuntimeError(f"stale shared memory segment {name!r}: expected {size} bytes, got {shm.size}")
        return shm


def _clear_frames(error: BaseException) -> None:
    """Clear the local variables of the finished frames that an error went through, and of those that the errors it
    chains went through (``raise ... from``, or raised while handling another), each error once."""
    pending, seen = [error], set()
    while pending:
        error = pending.pop()
        if not isinstance(error, BaseException) or id(error) in seen:
            continue
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        pending += [error.__cause__, error.__context__, *getattr(error, "exceptions", ())]


def _remove_ipc_path(zmq_handle: str) -> None:
    if zmq_handle.startswith("ipc://"):
        try:
            os.remove(zmq_handle[len("ipc://") :])
        except OSError:
            pass


def _open_socket(context: zmq.Context, kind: int) -> zmq.Socket:
    socket = context.socket(kind)
    socket.setsockopt(zmq.LINGER, 1000)  # closing never waits long for a peer that is gone
    return socket


def _wait_for_peer(
    socket: zmq.Socket,
    timeout_s: float,
    peer: str,
    check: Optional[Callable[[], None]] = None,
    event: int = zmq.POLLIN,
) -> None:
    """Wait until ``socket`` has a message (``event=POLLIN``) or can send one (``POLLOUT``: a REQ socket blocks until
    its peer has connected), at most ``timeout_s``; ``check()`` runs every second meanwhile and raises when the peer is
    known to have failed."""
    deadline = time.monotonic() + timeout_s
    while not socket.poll(1000, event):
        if check is not None:
            check()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"the weight {peer} did not answer within {timeout_s:.0f} s")


class BucketedWeightSender:
    """Send ``(name, tensor)`` pairs to a ``BucketedWeightReceiver`` in buckets of ``bucket_size_mb``.

    Each message waits for the receiver's answer: an acknowledgement, or the error that stopped the receiver, which
    is raised here. Waiting stops after ``timeout_s`` without an answer, and earlier when ``check_receiver`` (see
    ``send_weights``) raises."""

    def __init__(self, zmq_handle: str, bucket_size_mb: int = 2048, use_shm: bool = False, timeout_s: float = 600.0):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm
        self.timeout_s = timeout_s
        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None
        self.check_receiver: Optional[Callable[[], None]] = None

    def send_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], check_receiver: Optional[Callable[[], None]] = None
    ) -> None:
        """``check_receiver()`` is called while an answer is awaited and raises if the receiving side has failed
        (e.g. its process died before it could answer)."""
        self.check_receiver = check_receiver
        try:
            self._init_socket()
            self._init_buffer()
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            for name, weight in weights:
                # align each tensor for the receiver's view of the byte buffer in that tensor's dtype
                alignment = weight.element_size()
                offset = (offset + alignment - 1) // alignment * alignment
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    self._synchronize()
                    self._request({"bucket_meta": bucket_meta, "is_last": False})
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    if self.use_shm:
                        raise ValueError(
                            f"weight {name} {tuple(weight.shape)} {weight.dtype} does not fit in the "
                            f"{self.bucket_size_mb} MB bucket; increase worker.rollout.server_weight_bucket_mb"
                        )
                    self._send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].view(dtype=weight.dtype).view(weight.shape).copy_(
                    weight, non_blocking=True
                )
                offset += weight.nbytes

            self._synchronize()
            self._request({"bucket_meta": bucket_meta, "is_last": True})
        finally:
            self._cleanup()

    def _request(self, payload: Any) -> None:
        _wait_for_peer(self.socket, self.timeout_s, "receiver", self.check_receiver, zmq.POLLOUT)
        self.socket.send_pyobj(payload)
        _wait_for_peer(self.socket, self.timeout_s, "receiver", self.check_receiver)
        answer = self.socket.recv_pyobj()
        if answer is not None:
            raise RuntimeError(f"the weight receiver failed: {answer['error']}")

    def _synchronize(self) -> None:
        if not self.use_shm:
            torch.cuda.synchronize()

    def _init_socket(self) -> None:
        _remove_ipc_path(self.zmq_handle)
        self.socket = _open_socket(self.zmq_context, zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self) -> None:
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"cuda:{torch.cuda.current_device()}")
            self.buffer = buffer
            self._request(reduce_tensor(buffer))
        else:
            import uuid

            shm_name = f"easyr1_weights_{uuid.uuid4().hex}"
            self.shm = create_shared_memory(self.bucket_size, shm_name)
            self.buffer = torch.frombuffer(self.shm.buf, dtype=torch.uint8)
            self._request({"name": shm_name, "size": self.bucket_size})

    def _send_large_weight(self, name: str, weight: torch.Tensor) -> None:
        # the receiver reads the tensor itself: whatever produced it (the gather of its shards) must be done
        self._synchronize()
        bucket_meta: dict[str, TensorMetadata] = {
            name: {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": 0,
                "handle": reduce_tensor(weight),
            }
        }
        self._request({"bucket_meta": bucket_meta, "is_last": False})

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        _remove_ipc_path(self.zmq_handle)
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
        if not self.use_shm:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


class BucketedWeightReceiver:
    """Receive the weights of a ``BucketedWeightSender`` and hand each bucket to a callback. An error while
    receiving (in the callback too) is sent back to the sender before it is raised here."""

    def __init__(self, zmq_handle: str, device: torch.device, use_shm: bool = False, timeout_s: float = 600.0):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm
        self.timeout_s = timeout_s
        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: Callable[[list[tuple[str, torch.Tensor]], bool], None]) -> None:
        """``on_bucket_received(weights, is_last)`` gets each bucket's ``(name, tensor)`` pairs; the tensors are views
        of the shared buffer, valid during the call only."""
        answered = True
        try:
            self._init_socket()
            comm_metadata = self._receive()
            answered = False
            self._init_buffer(comm_metadata)
            self.socket.send_pyobj(None)
            answered = True
            while True:
                metadata = self._receive()
                answered = False
                is_last = self._load_bucket(metadata, on_bucket_received)
                if is_last:
                    # Release the buffer before the last answer: the sender frees its bucket as soon as the answer
                    # arrives, and CUDA IPC frees the sender's block only once this process no longer references it.
                    self._release_buffer()
                self.socket.send_pyobj(None)
                answered = True
                if is_last:
                    break
        except BaseException as exc:
            # the frames the error went through (and those of the errors it chains) hold views of the buffer; drop
            # them so that the buffer can be freed
            _clear_frames(exc)
            if not answered and self.socket is not None:
                try:
                    self.socket.send_pyobj({"error": f"{type(exc).__name__}: {exc}"})
                except zmq.ZMQError:
                    pass
            try:
                self._release_buffer()
            except Exception:  # the original error is the one raised
                logger.exception("releasing the weight buffer failed after an error")
            raise
        finally:
            if self.socket is not None:
                self.socket.close()
                self.socket = None

    def _load_bucket(
        self, metadata: Any, on_bucket_received: Callable[[list[tuple[str, torch.Tensor]], bool], None]
    ) -> bool:
        """Hand one bucket's tensors (views of the buffer, or opened handles) to the callback; returns is_last."""
        weights = []
        for name, meta in metadata["bucket_meta"].items():
            shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
            if handle is not None:
                weights.append((name, rebuild_ipc(handle, self.device.index)))
                continue
            size = dtype.itemsize * shape.numel()
            tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
            if self.use_shm:
                tensor = tensor.to(self.device)
            weights.append((name, tensor))
        on_bucket_received(weights, metadata["is_last"])
        self._synchronize()
        return metadata["is_last"]

    def _receive(self) -> Any:
        _wait_for_peer(self.socket, self.timeout_s, "sender")
        return self.socket.recv_pyobj()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _init_socket(self) -> None:
        self.socket = _open_socket(self.zmq_context, zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self, comm_metadata: Any) -> None:
        if not self.use_shm:
            self.buffer = rebuild_ipc(comm_metadata, self.device.index)
        else:
            self.shm = shared_memory.SharedMemory(name=comm_metadata["name"])
            self.buffer = torch.frombuffer(self.shm.buf[: comm_metadata["size"]], dtype=torch.uint8)

    def _release_buffer(self) -> None:
        self._synchronize()
        self.buffer = None
        if self.shm is not None:
            self.shm.close()  # every view of it is gone by now (see receive_weights)
            self.shm = None
        if self.device.type == "cuda":
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
