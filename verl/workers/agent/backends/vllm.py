# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from vllm import SamplingParams

from ....utils.dataset import process_image
from ..protocol import GenerationOutput, GenerationRequest


class VLLMAgentRequestError(RuntimeError):
    """One vLLM request that still failed after batch isolation."""


class VLLMAgentSchedulerError(RuntimeError):
    """The shared dispatcher failed outside an individual vLLM request."""


@dataclass
class _CachedImage:
    source: Any
    processed: Any


class VLLMAgentImageCache:
    """Cache model-side resize results and stable vLLM multimodal identities."""

    def __init__(
        self,
        *,
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._namespace = uuid4().hex
        self._records: dict[int, _CachedImage] = {}
        self._uuids: dict[tuple[int, int], str] = {}

    def prepare(self, image: Any) -> Any:
        return self._record(image).processed

    def uuid_for(self, image: Any, *, occurrence: int = 0) -> str:
        if occurrence < 0:
            raise ValueError("image occurrence must be non-negative")
        self._record(image)
        key = (id(image), occurrence)
        if key not in self._uuids:
            self._uuids[key] = f"easyr1-agent-{self._namespace}-{len(self._uuids)}"
        return self._uuids[key]

    def _record(self, image: Any) -> _CachedImage:
        key = id(image)
        record = self._records.get(key)
        if record is not None and record.source is image:
            return record
        record = _CachedImage(
            source=image,
            processed=process_image(image, self.min_pixels, self.max_pixels),
        )
        self._records[key] = record
        return record


@dataclass
class _PendingVLLMRequest:
    prompt: dict[str, Any]
    sampling_params: SamplingParams
    lora_request: Optional[Any]
    future: asyncio.Future


class VLLMAgentBatchScheduler:
    """Coalesce concurrent agent turns into non-blocking batched LLM calls."""

    def __init__(
        self,
        inference_engine: Any,
        *,
        use_tqdm: bool = False,
        max_batch_size: Optional[int] = None,
        max_batch_images: Optional[int] = None,
    ):
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when provided")
        if max_batch_images is not None and max_batch_images <= 0:
            raise ValueError("max_batch_images must be positive when provided")
        self.inference_engine = inference_engine
        self.use_tqdm = use_tqdm
        self.max_batch_size = max_batch_size
        self.max_batch_images = max_batch_images
        self._queue: asyncio.Queue[_PendingVLLMRequest] = asyncio.Queue()
        self._dispatcher: Optional[asyncio.Task] = None
        self._closed = False
        self._terminal_error: Optional[BaseException] = None

    async def generate(
        self,
        *,
        prompt: dict[str, Any],
        sampling_params: SamplingParams,
        lora_request: Optional[Any],
    ) -> Any:
        if self._closed:
            if self._terminal_error is not None:
                raise VLLMAgentSchedulerError(
                    f"agent scheduler is unavailable: {self._terminal_error}"
                ) from self._terminal_error
            raise VLLMAgentSchedulerError("agent scheduler is closed")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(
            _PendingVLLMRequest(
                prompt=prompt,
                sampling_params=sampling_params,
                lora_request=lora_request,
                future=future,
            )
        )
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch())
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _dispatch(self) -> None:
        pending: list[_PendingVLLMRequest] = []
        try:
            while True:
                first = await self._queue.get()
                pending = [first]
                pending_images = self._request_image_count(first)
                # Let concurrently-created trajectories enqueue their current
                # turn, both before and after the blocking vLLM call.
                await asyncio.sleep(0)
                while not self._queue.empty() and (self.max_batch_size is None or len(pending) < self.max_batch_size):
                    candidate = self._queue.get_nowait()
                    candidate_images = self._request_image_count(candidate)
                    if (
                        self.max_batch_images is not None
                        and pending
                        and pending_images + candidate_images > self.max_batch_images
                    ):
                        # The image limit is an aggregate batching cap. As with
                        # the one-shot runner, one oversized request may still
                        # run alone instead of being stranded forever.
                        self._queue.put_nowait(candidate)
                        break
                    pending.append(candidate)
                    pending_images += candidate_images

                try:
                    results = await asyncio.to_thread(
                        self._generate_isolated,
                        pending,
                    )
                except asyncio.CancelledError:
                    self._closed = True
                    for item in pending:
                        item.future.cancel()
                    self._fail_queued(VLLMAgentSchedulerError("agent scheduler dispatcher was cancelled"))
                    raise
                except BaseException as exc:
                    scheduler_error = VLLMAgentSchedulerError(
                        f"agent vLLM dispatcher failed before request isolation: {type(exc).__name__}: {exc}"
                    )
                    scheduler_error.__cause__ = exc
                    self._terminal_error = scheduler_error
                    self._closed = True
                    self._settle_with_exception(pending, scheduler_error)
                    self._fail_queued(scheduler_error)
                    return
                for item, result in zip(pending, results):
                    if item.future.cancelled():
                        continue
                    if isinstance(result, BaseException):
                        item.future.set_exception(result)
                    else:
                        item.future.set_result(result)
                pending = []

                # Completed trajectories resume on this yield and may enqueue
                # their next turn. Do not strand such a request between a
                # dispatcher-empty check and task completion.
                await asyncio.sleep(0)
                if self._queue.empty():
                    return
        except asyncio.CancelledError:
            self._closed = True
            self._fail_queued(VLLMAgentSchedulerError("agent scheduler dispatcher was cancelled"))
            raise
        except BaseException as exc:
            scheduler_error = VLLMAgentSchedulerError(
                f"agent scheduler dispatcher failed: {type(exc).__name__}: {exc}"
            )
            scheduler_error.__cause__ = exc
            self._terminal_error = scheduler_error
            self._closed = True
            self._settle_with_exception(pending, scheduler_error)
            self._fail_queued(scheduler_error)
        finally:
            if pending:
                for item in pending:
                    if not item.future.done():
                        item.future.cancel()
            self._dispatcher = None
            if not self._closed and not self._queue.empty():
                self._dispatcher = asyncio.create_task(self._dispatch())

    async def close(self) -> None:
        """Stop dispatching and settle every waiter owned by this scheduler."""

        self._closed = True
        dispatcher = self._dispatcher
        if dispatcher is not None and not dispatcher.done():
            dispatcher.cancel()
            try:
                await dispatcher
            except asyncio.CancelledError:
                pass
        self._fail_queued(VLLMAgentSchedulerError("agent scheduler is closed"))

    @staticmethod
    def _request_image_count(item: _PendingVLLMRequest) -> int:
        multi_modal_data = item.prompt.get("multi_modal_data")
        if not isinstance(multi_modal_data, dict):
            return 0
        images = multi_modal_data.get("image")
        if images is None:
            return 0
        try:
            return len(images)
        except TypeError:
            return 1

    @staticmethod
    def _settle_with_exception(
        pending: list[_PendingVLLMRequest],
        exc: BaseException,
    ) -> None:
        for item in pending:
            if not item.future.done():
                item.future.set_exception(exc)

    def _fail_queued(self, exc: BaseException) -> None:
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(exc)

    def _generate_isolated(
        self,
        pending: list[_PendingVLLMRequest],
    ) -> list[Any | BaseException]:
        try:
            lora_requests = [item.lora_request for item in pending]
            sampling_params: SamplingParams | list[SamplingParams]
            sampling_params = [item.sampling_params for item in pending]
            if len(sampling_params) == 1:
                sampling_params = sampling_params[0]
            request_outputs = self.inference_engine.generate(
                prompts=[item.prompt for item in pending],
                sampling_params=sampling_params,
                lora_request=(None if all(request is None for request in lora_requests) else lora_requests),
                use_tqdm=self.use_tqdm,
            )
            if len(request_outputs) != len(pending):
                raise RuntimeError(
                    "vLLM returned an unexpected number of batched agent outputs: "
                    f"{len(request_outputs)} != {len(pending)}"
                )
            return list(request_outputs)
        except Exception as exc:  # noqa: BLE001
            if len(pending) == 1:
                error = VLLMAgentRequestError(f"{type(exc).__name__}: {exc}")
                error.__cause__ = exc
                return [error]
            midpoint = len(pending) // 2
            return [
                *self._generate_isolated(pending[:midpoint]),
                *self._generate_isolated(pending[midpoint:]),
            ]


class VLLMGenerationBackend:
    """Single-trajectory vLLM adapter for the inference-only AgentLoop."""

    def __init__(
        self,
        inference_engine: Any,
        sampling_params: SamplingParams,
        tokenizer: Any,
        scheduler: VLLMAgentBatchScheduler,
        lora_request: Optional[Any] = None,
        use_tqdm: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        forbidden_token_ids: tuple[int, ...] = (),
        image_cache: Optional[VLLMAgentImageCache] = None,
    ):
        if sampling_params.n != 1:
            raise ValueError("VLLMGenerationBackend expects sampling_params.n == 1 per agent trajectory")
        if sampling_params.stop:
            raise ValueError(
                "VLLMGenerationBackend does not allow string stop sequences; "
                "native EOS/end-of-turn tokens must delimit DeepEyes actions"
            )
        self.inference_engine = inference_engine
        self.sampling_params = sampling_params.clone()
        self.sampling_params.stop = None
        self.tokenizer = tokenizer
        self.lora_request = lora_request
        self.use_tqdm = use_tqdm
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if not isinstance(scheduler, VLLMAgentBatchScheduler):
            raise TypeError("VLLMGenerationBackend requires one shared VLLMAgentBatchScheduler")
        self.scheduler = scheduler
        self.image_cache = image_cache or VLLMAgentImageCache(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.forbidden_token_ids = tuple(forbidden_token_ids)
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in self.forbidden_token_ids):
            raise TypeError("forbidden token IDs must be integers")

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        sampling_params = self.sampling_params.clone()
        sampling_params.max_tokens = request.max_new_tokens
        if request.seed is not None:
            sampling_params.seed = request.seed
        sampling_params.detokenize = True
        # Match the original DeepEyes rollout: do not add XML closing tags as
        # static stops. Let the model's native EOS/end-of-turn token delimit a
        # complete assistant action so tags quoted inside reasoning cannot
        # terminate generation prematurely.
        # Keeping stop terminators in the output preserves the native EOS in
        # model action IDs, where it receives the same policy mask as one-shot
        # EasyR1 responses.
        sampling_params.include_stop_str_in_output = True
        if self.forbidden_token_ids:
            logit_bias = dict(sampling_params.logit_bias or {})
            logit_bias.update(dict.fromkeys(self.forbidden_token_ids, -100.0))
            sampling_params.logit_bias = logit_bias

        prompt: dict[str, Any] = {"prompt_token_ids": list(request.prompt_token_ids)}
        if request.images:
            processed_images = [self.image_cache.prepare(image) for image in request.images]
            image_occurrences: dict[int, int] = {}
            image_uuids = []
            for image in request.images:
                key = id(image)
                occurrence = image_occurrences.get(key, 0)
                image_occurrences[key] = occurrence + 1
                image_uuids.append(
                    self.image_cache.uuid_for(
                        image,
                        occurrence=occurrence,
                    )
                )
            prompt["multi_modal_data"] = {"image": processed_images}
            prompt["multi_modal_uuids"] = {"image": image_uuids}

        request_output = await self.scheduler.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            lora_request=self.lora_request,
        )
        if len(request_output.outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected number of agent trajectory outputs")

        completion = request_output.outputs[0]
        token_ids = list(completion.token_ids)
        backend_text = completion.text
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return GenerationOutput(
            token_ids=token_ids,
            text=text,
            finish_reason=completion.finish_reason,
            stop_reason=getattr(completion, "stop_reason", None),
            backend_text=backend_text if backend_text != text else None,
        )
