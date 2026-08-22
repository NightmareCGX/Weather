"""Deterministic tests for the rolling bounded marker-PUT scheduler."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ingestion.core.marker_put_scheduler import put_markers_rolling


def _mk_targets(n: int) -> list[str]:
    return [f"det_L{i:04d}" for i in range(n)]


def _fake_put(completed: list[str], block: threading.Event | None = None):
    """Return a put_one that records completed targets, optionally blocking."""

    def put(region: str) -> None:
        if block is not None and region == "blocked":
            block.wait(5.0)  # bounded block for the test
        completed.append(region)

    return put


def test_no_more_than_concurrency_in_flight() -> None:
    targets = _mk_targets(30)
    completed: list[str] = []
    active = 0
    peak = 0
    lock = threading.Lock()
    gate = threading.Event()

    def put(region: str) -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        gate.wait(5.0)
        completed.append(region)
        with lock:
            active -= 1

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = put_markers_rolling(
            targets, put, concurrency=4, cancel_event=threading.Event(),
            timeout_seconds=5.0, executor=ex,
        )
    gate.set()
    assert res.ok
    assert peak <= 4, f"peak in-flight PUTs was {peak}, expected <= 4"
    assert len(completed) == 30


def test_first_failure_stops_submission_and_drains() -> None:
    targets = _mk_targets(20)

    def put(region: str) -> None:
        if region == "det_L0004":
            raise RuntimeError("boom")
        time.sleep(0.01)

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = put_markers_rolling(
            targets, put, concurrency=3, cancel_event=threading.Event(),
            timeout_seconds=5.0, executor=ex,
        )
    assert not res.ok
    assert len(res.failures) >= 1
    assert "det_L0004" in {r for r, _ in res.failures}
    # Not-started targets were cancelled (or never submitted).
    assert len(res.successes) < 20


def test_cancellation_stops_submission_and_drains() -> None:
    targets = _mk_targets(20)
    cancel_event = threading.Event()

    def put(region: str) -> None:
        if region == "det_L0002":
            cancel_event.set()  # simulate cancellation during a PUT
        time.sleep(0.02)

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = put_markers_rolling(
            targets, put, concurrency=3, cancel_event=cancel_event,
            timeout_seconds=5.0, executor=ex,
        )
    assert not res.ok
    assert res.cancelled or len(res.successes) < 20


def test_all_success_ok() -> None:
    targets = _mk_targets(10)
    completed: list[str] = []

    def put(region: str) -> None:
        completed.append(region)

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = put_markers_rolling(
            targets, put, concurrency=4, cancel_event=threading.Event(),
            timeout_seconds=5.0, executor=ex,
        )
    assert res.ok
    assert len(res.successes) == 10
    assert not res.failures
    assert not res.cancelled


def test_late_failed_wave_put_cannot_overwrite_newer_wave() -> None:
    """A blocked PUT from an old wave must not complete after EXCLUSIVE release.

    Deterministic: the old wave's blocked PUT is held; the scheduler must not
    return (and the caller must not release EXCLUSIVE) until the blocked PUT
    finishes. We model "EXCLUSIVE release" by the scheduler returning only
    after every started PUT drained.
    """
    targets = _mk_targets(5)
    blocked_gate = threading.Event()

    def put(region: str) -> None:
        if region == "det_L0000":
            blocked_gate.wait(5.0)  # deliberately blocked
            # After release, this PUT must have already drained before the
            # caller releases EXCLUSIVE.
        time.sleep(0.01)

    result_holder: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        # Start the rolling scheduler; it will block on the first PUT.
        t = threading.Thread(
            target=lambda: result_holder.update(
                res=put_markers_rolling(
                    targets, put, concurrency=2, cancel_event=threading.Event(),
                    timeout_seconds=5.0, executor=ex,
                )
            )
        )
        t.start()
        time.sleep(0.2)
        # EXCLUSIVE must NOT be released while the blocked PUT is still running:
        # the scheduler thread has not returned, so the caller (this test) has
        # not released EXCLUSIVE.
        assert t.is_alive(), "scheduler returned before the blocked PUT drained"
        # Release the blocked PUT and let the wave finish.
        blocked_gate.set()
        t.join(5.0)
        assert not t.is_alive()
        assert result_holder["res"].ok
