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
"""The generation backend of one trajectory in the server mode: each turn goes to a rollout server picked by the
router, as one request (continuous batching on the server)."""

from __future__ import annotations

import asyncio
from typing import Any, Hashable, Optional, Sequence
from uuid import uuid4

from vllm import SamplingParams

from ...rollout.async_server.tasks import call_and_cancel_with, finish, resolve, settle
from ..protocol import GenerationOutput, GenerationRequest
from .vllm import VLLMAgentImageCache, agent_generation_output, build_agent_vllm_request


class AsyncVLLMServerBackend:
    """Builds each turn's request as the batched rollout does (``build_agent_vllm_request``: the same sampling
    parameters, seed, logit bias, images and multimodal uuids), without detokenization (token in, token out, as verl
    #7679; the loop decodes the tokens itself)."""

    def __init__(
        self,
        *,
        servers: Sequence[Any],
        router: Any,
        sticky_key: Hashable,
        group_key: Optional[Hashable],
        sampling_params: SamplingParams,
        tokenizer: Any,
        image_cache: VLLMAgentImageCache,
        forbidden_token_ids: tuple[int, ...] = (),
    ):
        if sampling_params.n != 1:
            raise ValueError("AsyncVLLMServerBackend expects sampling_params.n == 1 per agent trajectory")
        if sampling_params.stop:
            raise ValueError("agent turns end at the model's end-of-turn token, not at string stop sequences")
        self.servers = list(servers)
        self.router = router
        self.sticky_key = sticky_key
        self.group_key = group_key
        self.sampling_params = sampling_params.clone()
        self.sampling_params.stop = None
        self.tokenizer = tokenizer
        self.image_cache = image_cache
        self.forbidden_token_ids = tuple(forbidden_token_ids)

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        prompt, sampling_params = build_agent_vllm_request(
            request,
            self.sampling_params,
            image_cache=self.image_cache,
            forbidden_token_ids=self.forbidden_token_ids,
        )
        sampling_params.detokenize = False
        replica = await self._acquire()
        try:
            # cancelling the trajectory cancels the server's generation (vLLM aborts the request) and waits for it
            result = await call_and_cancel_with(
                self.servers[replica].generate.remote(prompt, sampling_params, uuid4().hex)
            )
        except BaseException:
            # the turn counts as in flight until the server's call has ended
            await finish(self._release(replica), quiet=True)
            raise
        await finish(self._release(replica))
        return agent_generation_output(
            result["token_ids"],
            result["text"],
            finish_reason=result["finish_reason"],
            stop_reason=result["stop_reason"],
            tokenizer=self.tokenizer,
            log_probs=result["log_probs"],
        )

    async def _acquire(self) -> int:
        call = self.router.acquire.remote(self.sticky_key, self.group_key)
        try:
            return await resolve(call)
        except asyncio.CancelledError:

            async def give_back() -> None:  # the router may have counted the turn already
                replica = await settle(call)
                if replica is not None:
                    await self._release(replica)

            await finish(give_back(), quiet=True)
            raise

    async def _release(self, replica: int) -> None:
        await resolve(self.router.release.remote(replica))
