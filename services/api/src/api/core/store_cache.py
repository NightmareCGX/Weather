"""Generation-aware store-handle reuse for the API serving tier.

Phase 2 of the serving-performance work: the Phase 1 bounded-selection path
still pays a per-request setup cost — ``S3FileSystem``/mapper resolution,
``xr.open_zarr``, the ``.zmetadata`` consolidated-metadata GET, and the CPU
reconstruction of the zarr group / xarray variables / coordinate indexes from
byte-identical metadata. Measured against MinIO, that setup is ~60% of a warm
tile/point request and dominates viewport bursts (every tile re-opens the same
store).

What is reused
--------------

The lazily-opened :class:`xarray.Dataset` returned by
:func:`api.core.zarr.read_dataset`, cached per ``(store_path,
serving_generation)``. The dataset is **lazy**: its variables wrap
``LazilyIndexedArray(ZarrArrayWrapper(...))`` so any actual chunk I/O happens
at selection time, exactly as before. Reusing it skips only the repeated
open/metadata work — never the data reads.

What is deliberately NOT reused
-------------------------------

* **Decoded meteorological fields.** Serving selectors materialize bounded
  windows (tile crop / 2×2 neighborhood / member vector) into fresh numpy
  arrays; those per-request results stay per-request. The cached dataset never
  has a full field materialized into it (no selector calls ``.values`` /
  ``compute()`` on an unselected variable), so the cache retains metadata +
  coordinate arrays only (kilobytes).
* Any result/PNG/response caching (Phase 1 tile LRU and the Redis response
  cache are separate layers).

Why generation identity is part of the key
------------------------------------------

A same-cycle re-ingestion replaces the data behind the SAME ``store_path``
and commits a new committed-manifest generation. A cache keyed only by
``store_path`` would serve generation-A metadata (lead axis, variable set,
chunk layout) to generation-B readers. The key therefore includes the
generation; :func:`read_dataset_cached` re-probes the committed manifest on
every call (one small GET — far cheaper than the open it saves), so the
moment a writer commits generation B the next reader computes a B-keyed
identity, misses, and opens fresh. Legacy stores without a manifest use the
same deterministic legacy token as ``manifest_generation``; the first
marker-aware finalization writes a real generation and naturally rotates the
key. A malformed manifest fails closed (no caching) rather than risking a
stale key.

Why lazy I/O must remain under the reader lock
----------------------------------------------

This cache changes *when setup happens*, not *when data reads happen*. Every
gated selector still runs inside the SHARED reader gate, so all chunk fetches
— including those served through a cached dataset — occur while the SHARED
advisory lock is held (the Phase 1 invariant). Nothing here returns a lazy
object past the gate boundary.

Lifecycle
---------

* **Key:** ``(canonical store root, serving_generation)``.
* **Value:** the lazy ``xr.Dataset`` from ``read_dataset``.
* **Max size:** ``MAX_ENTRIES`` (insertion-order eviction of the oldest
  entry; bounded, never unbounded module growth).
* **TTL:** none — generation keying is the invalidation mechanism; entries
  for superseded generations simply become unreachable and age out of the
  small LRU. A repaired/replaced newest store commits a new generation and
  is therefore never shadowed by a stale fallback entry.
* **Thread synchronization:** a module-level :class:`threading.Lock` guards
  the internal dictionary and in-flight flight table. Network I/O (``opener()``)
  runs **outside the lock**. A minimal per-key single-flight coordination
  ensures that concurrent misses for the **same key** execute exactly one
  underlying open while followers await that leader result; concurrent misses
  for **distinct keys** (e.g. GFS vs GEFS) open concurrently without
  cross-store serialization.
* **Process scope:** process-local (per uvicorn worker), matching the
  serving tier's other caches.
* **Failure behavior:** exceptions from ``opener`` (missing/broken store,
  transient S3 error) propagate to the leader and any concurrent waiters, and
  are NOT cached — there is no negative caching, so a transient failure
  cannot poison the key and the next request retries the open fresh. A
  ``ManifestReadError`` (malformed manifest) bypasses the cache entirely and
  opens directly (fail closed).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable

import xarray as xr

#: Maximum number of cached (store_path, generation) entries. Small by
#: design: the working set is the handful of newest GFS/GEFS cycle stores;
#: decoded fields are never retained, so entries cost kilobytes.
MAX_ENTRIES = 8


class _Flight:
    """Represents an in-flight store open for a specific cache key."""

    __slots__ = ("event", "result", "exception")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: xr.Dataset | None = None
        self.exception: BaseException | None = None


class StoreHandleCache:
    """A bounded, thread-safe, generation-keyed lazy-dataset reuse cache.

    Includes per-key single-flight coordination to collapse concurrent cold
    misses for the same key into a single underlying open without serializing
    unrelated stores or generations.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], xr.Dataset] = OrderedDict()
        self._in_flight: dict[tuple[str, str], _Flight] = {}
        #: Diagnostics: total lookups / misses (opened) since process start.
        self.hits = 0
        self.misses = 0

    def get_or_open(
        self,
        key: tuple[str, str],
        opener: Callable[[], xr.Dataset],
    ) -> tuple[xr.Dataset, bool]:
        """Return ``(dataset, hit)`` for the key, opening via ``opener`` on miss.

        The lock guards dictionary operations only; ``opener`` (network I/O)
        runs outside it. Multiple concurrent requests for the **same key** share
        a single open via per-key single-flight; requests for **different keys**
        open concurrently. Exceptions from ``opener`` propagate and are not
        cached.
        """
        is_leader = False
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return entry, True

            flight = self._in_flight.get(key)
            if flight is None:
                flight = _Flight()
                self._in_flight[key] = flight
                is_leader = True

        if not is_leader:
            # Wait for the lead opener to complete outside the lock.
            flight.event.wait()
            if flight.exception is not None:
                raise flight.exception
            assert flight.result is not None
            with self._lock:
                self.hits += 1
            return flight.result, True

        # Lead thread: execute open OUTSIDE the lock.
        try:
            dataset = opener()
        except BaseException as exc:
            with self._lock:
                flight.exception = exc
                self._in_flight.pop(key, None)
                flight.event.set()
            raise
        else:
            with self._lock:
                self.misses += 1
                self._entries[key] = dataset
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
                flight.result = dataset
                self._in_flight.pop(key, None)
                flight.event.set()
            return dataset, False

    def clear(self) -> None:
        """Drop every cached entry (used by tests; not on the serving path)."""
        with self._lock:
            self._entries.clear()
            self._in_flight.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


#: Process-wide store-handle cache (per uvicorn worker).
store_handle_cache = StoreHandleCache()
