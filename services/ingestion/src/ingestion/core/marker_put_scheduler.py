"""Rolling bounded marker-PUT scheduler for the wave pre-update.

The wave pre-update writes one ``UPDATING`` marker per target logical region
while holding the EXCLUSIVE store gate. Submitting all PUTs to a thread pool at
once would not bound the number of in-flight MinIO/local PUTs. This scheduler
keeps at most ``MARKER_PUT_CONCURRENCY`` PUT calls in flight, submits the next
target when a slot frees, and — on the first failure or caller cancellation —
stops submitting new targets, cancels not-started futures, and drains every
running PUT **before** the caller is allowed to release EXCLUSIVE ownership.

No data worker may start unless every target marker PUT succeeded.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MarkerPutResult:
    """Outcome of a wave pre-update marker-PUT batch."""

    successes: list[str] = field(default_factory=list)
    failures: list[tuple[str, BaseException]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.cancelled


class MarkerPutError(RuntimeError):
    """Raised when one or more marker PUTs failed or were cancelled."""


def put_markers_rolling(
    targets: Iterable[str],
    put_one: Callable[[str], None],
    *,
    concurrency: int,
    cancel_event: threading.Event,
    timeout_seconds: float,
    executor: concurrent.futures.ThreadPoolExecutor,
    observer: object | None = None,
) -> MarkerPutResult:
    """Submit marker PUTs with a rolling concurrency bound.

    Args:
        targets: The region ids to write UPDATING markers for.
        put_one: A callable that writes one marker and raises on failure.
        concurrency: Maximum in-flight PUT calls.
        cancel_event: Cooperative cancellation token.
        timeout_seconds: Per-PUT storage/network timeout (a slow PUT bounds the
            drain).
        executor: The thread pool to run PUTs on.
        observer: Optional observer receiving progress updates.

    Returns:
        A :class:`MarkerPutResult` with successes, failures, and cancelled
        targets.
    """
    target_list = list(targets)
    successes: list[str] = []
    failures: list[tuple[str, BaseException]] = []
    lock = threading.Lock()

    def _notify_observer(active_count: int) -> None:
        if observer is not None and hasattr(observer, "set_marker_progress"):
            with lock:
                done_count = len(successes) + len(failures)
            observer.set_marker_progress(
                done=done_count,
                total=len(target_list),
                active=active_count,
            )

    def run_one(region_id: str) -> None:
        try:
            put_one(region_id)
            with lock:
                successes.append(region_id)
        except BaseException as exc:  # noqa: BLE001 - record; the scheduler drains
            with lock:
                failures.append((region_id, exc))
        _notify_observer(len(in_flight))

    # Rolling submission: an in-flight map future -> region id. Initially submit
    # up to `concurrency`. As futures complete, submit the next target. On the
    # first failure or cancellation, stop submitting new targets and cancel
    # not-started futures (the ThreadPoolExecutor cancels queued-not-started
    # futures on .cancel()).
    in_flight: dict[concurrent.futures.Future[None], str] = {}
    idx = 0
    stop = threading.Event()
    cancelled: list[str] = []

    def submit_next() -> bool:
        nonlocal idx
        if idx >= len(target_list):
            return False
        if stop.is_set():
            return False
        region_id = target_list[idx]
        idx += 1
        fut = executor.submit(run_one, region_id)
        in_flight[fut] = region_id
        return True

    def cancel_not_started() -> None:
        for fut, region in list(in_flight.items()):
            if not fut.running() and not fut.done():
                if fut.cancel():
                    cancelled.append(region)
                    in_flight.pop(fut, None)

    # Initial fill.
    for _ in range(min(concurrency, len(target_list))):
        if not submit_next():
            break
    _notify_observer(len(in_flight))

    while True:
        # Collect completed futures.
        done = [f for f in in_flight if f.done()]
        for f in done:
            in_flight.pop(f, None)
        # Re-evaluate failure/cancel and cancel not-started futures.
        if failures or cancel_event.is_set():
            stop.set()
            cancel_not_started()
        # Fill slots with new targets when not stopped and work remains.
        if not stop.is_set():
            while len(in_flight) < concurrency and idx < len(target_list):
                if not submit_next():
                    break
        # Exit only when nothing is in flight and no work remains (or stopped).
        if not in_flight and (idx >= len(target_list) or stop.is_set()):
            break
        # Wait for progress (bounded).
        if in_flight:
            concurrent.futures.wait(list(in_flight), timeout=min(0.1, timeout_seconds))
        if failures or cancel_event.is_set():
            stop.set()
            cancel_not_started()

    # Every started PUT has now finished (drained) OR was cancelled as
    # not-started (removed from in_flight). The remaining in_flight entries are
    # futures that finished; nothing is left running.

    # Targets that were never submitted (because the wave was stopped early by
    # a failure or cancellation) are also "cancelled" — they never wrote a
    # marker and must not be treated as success.
    if stop.is_set() and idx < len(target_list):
        cancelled.extend(target_list[idx:])

    return MarkerPutResult(
        successes=sorted(successes),
        failures=list(failures),
        cancelled=sorted(set(cancelled)),
    )
