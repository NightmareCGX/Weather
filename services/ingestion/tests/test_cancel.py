"""Tests for the non-abandoning aggregate cancellation drain."""

from __future__ import annotations

import asyncio
import threading

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
