"""Non-abandoning aggregate drain for ingestion worker Futures.

The ingestion CLI schedules a bounded set of synchronous region-writer worker
Futures via ``run_in_executor``. When the outer coroutine is cancelled, the
worker threads cannot be stopped; their session-level advisory locks and
source-file usage must be allowed to finish before the coroutine propagates
cancellation.

This module provides the single aggregate await used after workers are
scheduled. It receives the first cancellation itself (it is the primary
await), sets a cooperative ``threading.Event``, and keeps waiting on the same
``asyncio.gather`` aggregate until every worker Future is actually done —
repeated ``asyncio.CancelledError`` cannot abandon the drain.

Source cleanup and executor shutdown happen only after every worker has
finished, released its locks, and stopped using its resources.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)


class IngestionCancellationDrain:
    """Holds the aggregate drain state and the cooperative cancel token.

    Attributes:
        cancel_event: A ``threading.Event`` the workers check cooperatively
            between lock acquisitions and region-key steps.
    """

    def __init__(self, cancel_event: threading.Event | None = None) -> None:
        # The cooperative token is a threading.Event set on cancellation; sync
        # workers poll it (they cannot await).
        self.cancel_event: threading.Event = (
            cancel_event if cancel_event is not None else threading.Event()
        )


async def await_all_workers_non_abandoning(
    worker_futures: Iterable["asyncio.Future[Any]"],
    cancel_event: threading.Event,
) -> tuple[list[Any], bool]:
    """Await every worker Future without abandoning the drain on cancellation.

    Args:
        worker_futures: The raw asyncio worker Futures (strong references
            retained by the caller).
        cancel_event: The cooperative cancellation token (a ``threading.Event``
            the sync workers poll).

    Returns:
        A ``(results, cancellation_requested)`` pair. ``results`` contains the
        worker return values or exceptions (``return_exceptions=True``).
    """
    if cancel_event is None:
        cancel_event = threading.Event()
    futures = list(worker_futures)
    drain_future = asyncio.gather(*futures, return_exceptions=True)
    cancellation_requested = False

    while not drain_future.done():
        try:
            await asyncio.shield(drain_future)
        except asyncio.CancelledError:
            # The cooperative token is a threading.Event; the sync workers poll
            # it.
            cancel_event.set()
            cancellation_requested = True
            # Continue waiting on the SAME drain_future; repeated cancellation
            # must not abandon the aggregate drain.
            continue

    results = list(drain_future.result())

    # Inspect + record every worker result/exception.
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error(
                "ingestion worker %d raised during drain: %s",
                index,
                result,
            )

    # The caller is responsible for verifying all worker cleanup (locks
    # released, Connections closed/invalidated, source files stopped being
    # used) BEFORE calling cleanup_source_files/executor.shutdown.

    return results, cancellation_requested


async def await_blocking_settled(
    fn: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> Any:
    """Run a blocking callable off the event loop, settling it on cancellation.

    The callable executes in a worker thread (``asyncio.to_thread``). If the
    awaiting task is cancelled, the worker is shielded from the initial
    cancellation and the caller keeps waiting — non-abandoningly, tolerating
    repeated cancellation — until the worker thread has actually finished,
    before :class:`asyncio.CancelledError` is re-raised.

    This guarantees that callers holding bounded resources across the call
    (semaphore slots, staging/residency ownership) never release them while
    the blocking work is still running: a replacement worker cannot start
    until this one has settled.

    On the normal path the worker's return value is returned and worker
    exceptions propagate unchanged. On the cancellation path the worker's
    result/exception is retrieved and discarded (debug-logged on failure) so
    no "exception never retrieved" warning is deferred.
    """
    task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    try:
        # Shield: cancelling the outer task must not cancel the worker Task.
        # A bare ``await task`` propagates cancellation into it, and the
        # asyncio side would report done/cancelled while the thread runs.
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        if not task.cancelled():
            with contextlib.suppress(Exception):
                task.result()
        raise
