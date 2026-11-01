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
"""Waiting on the Ray calls of a rollout so that a failure does not leave the others running."""

import asyncio
import inspect
import logging
from typing import Any, Iterable, Optional, Sequence


logger = logging.getLogger(__name__)


def gather_or_cancel(refs: Sequence[Any], api: Optional[Any] = None) -> list[Any]:
    """The results of the Ray calls ``refs``, in order. When one fails, the others are cancelled (a call of an async
    actor gets asyncio's cancellation, so the trajectories it runs end and close their environments), and their end
    is awaited before the first error is raised; ``api`` is the ``ray`` module or a stand-in."""
    if api is None:
        import ray as api
    pending = list(refs)
    try:
        while pending:
            ready, pending = api.wait(pending, num_returns=1)
            api.get(ready[0])
    except BaseException:
        for ref in pending:
            # not recursive: a cancelled worker cancels its own calls and waits for them (a recursive cancellation
            # could also cancel the calls that release the router's counts)
            api.cancel(ref, recursive=False)
        for ref in pending:
            try:
                api.get(ref)
            except BaseException:  # cancelled, or failed on its own: the first error is the one raised
                pass
        raise
    return [api.get(ref) for ref in refs]


async def resolve(value: Any) -> Any:
    """Await a Ray call's result; a stand-in may return the value itself."""
    return await value if inspect.isawaitable(value) else value


async def settle(value: Any) -> Any:
    """The result of a call that is ending anyway (after its cancellation), or None if it failed."""
    try:
        return await resolve(value)
    except BaseException:
        return None


async def finish(*awaitables: Any, quiet: bool = False) -> None:
    """Wait until every awaitable is done, also when this coroutine is cancelled meanwhile (``asyncio.shield`` alone
    protects the work but not its waiter, which a second cancellation would let go on early). Then the first error
    of the awaitables is raised, or else the cancellation if one came.

    ``quiet`` is for a cleanup while another error goes on (the caller raises that one next, so it stays the error
    raised): nothing is raised, and the cleanup's errors are logged."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    cancelled = await _wait_through_cancellations(tasks)
    cancelled = cancelled or any(task.cancelled() for task in tasks)
    errors = [task.exception() for task in tasks if not task.cancelled() and task.exception() is not None]
    if quiet:
        for error in errors:
            logger.warning("a cleanup failed while another error went on: %r", error, exc_info=error)
    elif errors:
        raise errors[0]
    elif cancelled:
        raise asyncio.CancelledError


async def _wait_through_cancellations(tasks: Sequence[asyncio.Future]) -> bool:
    """Wait until the tasks are done; returns whether this coroutine was cancelled meanwhile."""
    cancelled = False
    while not all(task.done() for task in tasks):
        try:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


async def run_all(coroutines: Iterable[Any]) -> list[Any]:
    """Run the coroutines concurrently and return their results in order. When one fails, or this coroutine is
    cancelled, the others are cancelled and all of them are waited for (also through further cancellations) before
    the error or the cancellation is raised."""
    tasks = [asyncio.ensure_future(coroutine) for coroutine in coroutines]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await _wait_through_cancellations(tasks)  # the first error (or the cancellation) is the one raised
        raise


async def call_and_cancel_with(ref: Any) -> Any:
    """Await the Ray call ``ref``. If this coroutine is cancelled, the remote call is cancelled too (an async actor's
    method gets asyncio's cancellation) and its end is awaited before the cancellation goes on: awaiting a Ray call
    locally does not stop it remotely."""
    try:
        return await resolve(ref)
    except asyncio.CancelledError:
        cancel_remote(ref)
        await finish(settle(ref), quiet=True)
        raise


def cancel_remote(ref: Any) -> None:
    try:
        import ray
    except ImportError:  # a stand-in without Ray
        return
    if isinstance(ref, ray.ObjectRef):
        ray.cancel(ref)
