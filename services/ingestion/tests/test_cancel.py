"""Tests for the non-abandoning aggregate cancellation drain."""

from __future__ import annotations

import asyncio
import threading

import pytest

from ingestion.core.cancel import await_all_workers_non_abandoning


def test_normal_completion_does_not_set_cancel_event() -> None:
    async def main() -> None:
        cancel_event = threading.Event()
        tasks = [asyncio.create_task(_worker(0.01, "a")()), asyncio.create_task(_worker(0.01, "b")())]
        results, cancelled = await await_all_workers_non_abandoning(tasks, cancel_event)
        assert cancelled is False
        assert results == ["a", "b"]
        assert not cancel_event.is_set()

    asyncio.run(main())


def _worker(delay: float, result: str = "ok"):
    async def _inner():
        await asyncio.sleep(delay)
        return result

    return _inner


async def _run_drain_until_cancelled(worker_futures, cancel_event, results_holder):
    """Mirror Checkpoint 2H: the helper returns the flag; the caller raises."""
    try:
        results, cancelled = await await_all_workers_non_abandoning(
            worker_futures, cancel_event
        )
        results_holder["results"] = results
        results_holder["cancelled"] = cancelled
        if cancelled:
            raise asyncio.CancelledError  # propagate cancellation outward
    except asyncio.CancelledError:
        results_holder["cancelled_exc"] = True


def test_first_cancellation_sets_event_and_waits() -> None:
    async def main() -> None:
        cancel_event = threading.Event()
        release = asyncio.Event()
        done = {"n": 0}

        async def slow():
            await release.wait()
            done["n"] += 1
            return "slow"

        worker = asyncio.create_task(slow())
        results_holder: dict[str, object] = {}

        # The run task awaits the aggregate drain; we cancel THAT task.
        run_task = asyncio.create_task(
            _run_drain_until_cancelled([worker], cancel_event, results_holder)
        )
        # Let the drain start waiting on the worker.
        await asyncio.sleep(0.02)
        run_task.cancel()  # first cancellation arrives at the drain await
        await asyncio.sleep(0.05)
        # The drain must NOT have returned yet (the worker is still blocked).
        assert "cancelled_exc" not in results_holder
        assert cancel_event.is_set()
        # Release the worker; the drain completes and propagates cancellation.
        release.set()
        await asyncio.sleep(0.1)
        assert results_holder.get("cancelled_exc") is True
        assert done["n"] == 1
        # The run task finished (it raised CancelledError).
        assert run_task.done()

    asyncio.run(main())


def test_repeated_cancellation_cannot_abandon() -> None:
    async def main() -> None:
        cancel_event = threading.Event()
        release = asyncio.Event()
        done = {"n": 0}

        async def slow():
            await release.wait()
            done["n"] += 1
            return "slow"

        worker = asyncio.create_task(slow())
        results_holder: dict[str, object] = {}
        run_task = asyncio.create_task(
            _run_drain_until_cancelled([worker], cancel_event, results_holder)
        )
        await asyncio.sleep(0.02)
        # Two consecutive cancellations.
        run_task.cancel()
        await asyncio.sleep(0.02)
        run_task.cancel()
        await asyncio.sleep(0.05)
        # The drain must not have abandoned the still-running worker.
        assert "cancelled_exc" not in results_holder
        release.set()
        await asyncio.sleep(0.1)
        assert results_holder.get("cancelled_exc") is True
        assert done["n"] == 1

    asyncio.run(main())


def test_worker_exception_is_recorded_not_lost() -> None:
    async def main() -> None:
        cancel_event = threading.Event()

        async def boom():
            raise ValueError("worker blew up")

        task = asyncio.create_task(boom())
        results, cancelled = await await_all_workers_non_abandoning([task], cancel_event)
        assert cancelled is False
        assert len(results) == 1
        assert isinstance(results[0], ValueError)

    asyncio.run(main())


# ---------------------------------------------------------------------------
# await_blocking_settled: settle-before-release invariant for blocking decode
# work executed off the event loop while the caller holds bounded resources
# (decode/staging permits).
# ---------------------------------------------------------------------------

from ingestion.core.base import PredecessorState  # noqa: E402
from ingestion.core.cancel import await_blocking_settled  # noqa: E402


def _blocked_worker(gate: threading.Event, log: list[str]) -> float:
    """Blocking stand-in for _decode_and_normalize: parks until released."""
    log.append("worker_started")
    gate.wait(timeout=30.0)
    log.append("worker_settled")
    return 42.0


def test_blocking_settled_returns_result_normally() -> None:
    async def main() -> None:
        gate = threading.Event()
        gate.set()
        log: list[str] = []
        result = await await_blocking_settled(_blocked_worker, gate, log)
        assert result == 42.0
        assert log == ["worker_started", "worker_settled"]

    asyncio.run(main())


def test_blocking_settled_propagates_worker_exception() -> None:
    async def main() -> None:
        def boom() -> float:
            raise ValueError("decode worker blew up")

        try:
            await await_blocking_settled(boom)
        except ValueError as exc:
            assert str(exc) == "decode worker blew up"
        else:
            raise AssertionError("worker exception was not propagated")

    asyncio.run(main())


def test_cancellation_waits_for_blocked_worker_before_releasing_permits() -> None:
    """While the worker is blocked, no permit may be released and no
    replacement decode may start; only after the worker settles does the
    cancellation propagate."""
    async def main() -> None:
        gate = threading.Event()
        log: list[str] = []
        # Single permit: exhaustion models "all decode slots busy", so any
        # replacement acquisition proves the cancelled owner released its
        # permit (which the invariant forbids while the worker runs).
        decode_sem = asyncio.Semaphore(1)

        owner_started = asyncio.Event()

        async def owner() -> None:
            async with decode_sem:
                owner_started.set()
                try:
                    await await_blocking_settled(_blocked_worker, gate, log)
                except asyncio.CancelledError:
                    log.append("owner_cancelled")
                    raise

        owner_task = asyncio.create_task(owner())
        await owner_started.wait()
        await asyncio.sleep(0.05)  # let the worker enter the blocked section
        assert log == ["worker_started"]

        owner_task.cancel()
        await asyncio.sleep(0.1)  # deliver cancellation while worker is blocked

        # Worker still blocked: the settle-before-release guarantee means the
        # decode permit has NOT been released, so a replacement cannot enter.
        replacement_entered: list[bool] = []

        async def replacement() -> None:
            async with decode_sem:
                replacement_entered.append(True)

        replacement_task = asyncio.create_task(replacement())
        await asyncio.sleep(0.1)
        assert replacement_entered == []
        assert log == ["worker_started"]  # cancellation has not surfaced yet

        # Release the worker: it must settle, then the permit is released and
        # the replacement finally enters, and cancellation propagates.
        gate.set()
        await asyncio.wait_for(replacement_task, timeout=10.0)
        assert replacement_entered == [True]
        assert log == ["worker_started", "worker_settled", "owner_cancelled"]
        with pytest.raises(asyncio.CancelledError):
            await owner_task

    asyncio.run(main())


def test_cancellation_restores_predecessor_state_only_after_worker_settles() -> None:
    async def main() -> None:
        gate = threading.Event()
        log: list[str] = []
        predecessor_states: dict[tuple[int | None, int, bool], PredecessorState] = {}
        predecessor_lock = threading.Lock()
        pred_item = (None, 3, False)
        pred_state = PredecessorState(precip_raw=None, cloud_raw=None)

        async def owner() -> None:
            with predecessor_lock:
                popped = predecessor_states.pop(pred_item, None)
            assert popped is pred_state  # consume, as the pipeline does
            try:
                await await_blocking_settled(_blocked_worker, gate, log)
            except asyncio.CancelledError:
                # Restore only after the worker has settled (helper guarantee).
                with predecessor_lock:
                    predecessor_states.setdefault(pred_item, popped)
                raise

        predecessor_states[pred_item] = pred_state
        owner_task = asyncio.create_task(owner())
        await asyncio.sleep(0.05)
        assert log == ["worker_started"]

        owner_task.cancel()
        await asyncio.sleep(0.1)
        # Worker still blocked: predecessor state must NOT be restored yet.
        assert predecessor_states.get(pred_item) is None

        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await owner_task
        assert log == ["worker_started", "worker_settled"]
        assert predecessor_states.get(pred_item) is pred_state

    asyncio.run(main())
