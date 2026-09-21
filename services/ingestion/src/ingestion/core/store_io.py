"""Normalize the three store shapes a staging read/write may encounter.

The ingestion code passes stores around as one of: an ``s3://`` URL, a local path, or an
in-memory mapping (tests). Resolving that union in every function that touches the staging
area is how the three shapes drift apart, so it is resolved once here and everything else
speaks store-relative keys.

S3 access goes through the same ``resolve_s3_mapper`` the rest of ingestion uses, so
credentials, endpoint and filesystem caching behave identically to the member-shard path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from os import PathLike
from typing import Any


#: The store shapes the ingestion code passes around: an ``s3://`` URL, a local path, or an
#: in-memory mapping (tests).
StoreRef = str | PathLike[str] | Mapping[str, bytes]


def _range_to_offset_length(
    size: int, start: int | None, end: int | None
) -> tuple[int, int]:
    """Resolve a ``cat_file``-style range to ``(offset, length)`` for a local read.

    The semantics are the ones a range GET has, because :meth:`StoreIO.read_range` has to behave
    identically on all three backends: a negative ``start`` with no ``end`` is a *suffix* request
    (the last ``-start`` bytes, or the whole object when it is shorter), a negative ``end`` counts
    back from the end, ``end`` is exclusive, and a range running past the end is clamped.
    """
    if start is None and end is None:
        return 0, max(0, size)
    resolved_start = 0 if start is None else start
    if resolved_start < 0 and end is None:
        offset = max(0, size + resolved_start)
        return offset, max(0, size - offset)
    if resolved_start < 0:
        resolved_start = max(0, size + resolved_start)
    resolved_end = size if end is None else end
    if resolved_end < 0:
        resolved_end = max(0, size + resolved_end)
    offset = max(0, min(resolved_start, size))
    stop = max(0, min(resolved_end, size))
    return offset, max(0, stop - offset)


def _slice_like_a_range_get(
    blob: bytes, start: int | None, end: int | None
) -> bytes:
    """Slice an in-memory object the way a range GET would have answered."""
    offset, length = _range_to_offset_length(len(blob), start, end)
    return blob[offset : offset + length]


class StoreAccessError(RuntimeError):
    """Raised when a store reference cannot be used for I/O."""


class StoreIO:
    """Uniform list/read/write/delete over a store reference.

    Attributes:
        kind: One of ``"mapping"``, ``"s3"``, ``"local"``.
        root: For ``"s3"``, the bucket/prefix; for ``"local"``, the resolved directory.
    """

    def __init__(self, store: StoreRef) -> None:
        self.kind: str
        self.root: str
        self._mapping: Mapping[str, bytes] | None = None
        self._fs: Any = None
        self._loop: Any = None

        if isinstance(store, Mapping):
            self.kind = "mapping"
            self._mapping = store
            self.root = ""
            return

        if not isinstance(store, (str, PathLike)):
            raise StoreAccessError(f"unsupported store reference: {type(store).__name__}")
        path = os.fspath(store)
        if not isinstance(path, str):
            raise StoreAccessError(f"unsupported store path: {path!r}")

        if path.startswith("s3://"):
            from typing import cast

            from ingestion.core.config import settings
            from ingestion.core.s3 import resolve_s3_mapper

            # resolve_s3_mapper returns FSMap, whose .fs/.root are untyped in s3fs; the
            # existing member-shard write path casts for the same reason.
            resolved = cast(Any, resolve_s3_mapper(path, settings))
            self.kind = "s3"
            self._fs = resolved.fs
            self._loop = resolved.fs.loop
            self.root = str(resolved.root)
            return

        from ingestion.core.zarr_writer import _resolve_store

        self.kind = "local"
        self.root = str(_resolve_store(path))

    def __repr__(self) -> str:
        return f"StoreIO(kind={self.kind!r}, root={self.root!r})"

    # -- helpers -----------------------------------------------------------------

    def _sync(self, coroutine_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run one s3fs **coroutine** against its loop and return its result.

        ``coroutine_name`` is the private coroutine's name without its underscore -- ``"find"``
        for ``fs._find`` -- and the mapping is deliberate rather than incidental.

        **The public ``fs.find`` is not a coroutine, and that is the whole reason this exists.**
        s3fs replaces a public method with a blocking wrapper (``sync_wrapper``) whenever its
        private coroutine exists, so ``fs.find(...)`` runs the whole operation and returns a
        ``list``. Handing that to :func:`fsspec.asyn.sync` makes fsspec call it, get the finished
        result, and try to ``await`` it -- which is an error, not a slow path::

            TypeError: object list can't be used in 'await' expression

        The mistake is invisible against a local directory and against an in-memory mapping,
        because neither takes this branch at all; every store the staging and aggregate tests use
        is one of those two. It only appears on ``s3://``, which is the only shape production
        runs -- so the first symptom is a warning in a real cycle's log and a step that silently
        did nothing.

        Taking the *name* rather than a bound method keeps that trap out of reach: a caller cannot
        pass the blocking wrapper by accident, and a name s3fs renames fails loudly here instead of
        quietly becoming a no-op.

        Raises:
            StoreAccessError: if the named coroutine does not exist on this filesystem. A missing
                name means the pinned s3fs no longer provides it, which must not read as "nothing
                to do".
        """
        import fsspec.asyn  # type: ignore[import-untyped]

        private = f"_{coroutine_name}"
        coroutine = getattr(self._fs, private, None)
        if coroutine is None or not callable(coroutine):
            raise StoreAccessError(
                f"s3fs provides no {private!r} on {type(self._fs).__name__}; "
                "the pinned s3fs version no longer exposes this operation"
            )
        return fsspec.asyn.sync(self._loop, coroutine, *args, **kwargs)

    def _local_full(self, relative_key: str) -> str:
        return os.path.join(self.root, *relative_key.split("/"))

    # -- listing -----------------------------------------------------------------

    def list_under(self, relative_prefix: str) -> list[str]:
        """Return store-relative keys under ``relative_prefix``.

        Recursive and unordered; callers that need determinism sort. A missing prefix
        yields an empty list rather than an error, because an empty staging area is a normal
        state (nothing staged yet, or already aggregated).
        """
        if self.kind == "mapping":
            assert self._mapping is not None
            return [
                key
                for key in self._mapping
                if isinstance(key, str) and key.startswith(relative_prefix)
            ]
        if self.kind == "s3":
            found = self._sync("find", f"{self.root}/{relative_prefix}")
            return [str(key)[len(self.root) + 1 :] for key in found]
        base = self._local_full(relative_prefix.rstrip("/"))
        if not os.path.isdir(base):
            return []
        keys: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(base):
            for name in filenames:
                full = os.path.join(dirpath, name)
                keys.append(os.path.relpath(full, self.root).replace(os.sep, "/"))
        return keys

    # -- reading -----------------------------------------------------------------

    def read(self, relative_key: str) -> bytes:
        """Read one object.

        Raises:
            StoreAccessError: if the object is absent or unreadable.
        """
        if self.kind == "mapping":
            assert self._mapping is not None
            try:
                return bytes(self._mapping[relative_key])
            except KeyError as exc:
                raise StoreAccessError(f"missing object {relative_key!r}") from exc
        if self.kind == "s3":
            try:
                data = self._sync("cat_file", f"{self.root}/{relative_key}")
            except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
                raise StoreAccessError(
                    f"cannot read {relative_key!r}: {type(exc).__name__}: {exc}"
                ) from exc
            return bytes(data)
        try:
            with open(self._local_full(relative_key), "rb") as handle:
                return handle.read()
        except OSError as exc:
            raise StoreAccessError(f"cannot read {relative_key!r}: {exc}") from exc

    def exists(self, relative_key: str) -> bool:
        """Whether an object is present, without raising."""
        if self.kind == "mapping":
            assert self._mapping is not None
            return relative_key in self._mapping
        if self.kind == "s3":
            return bool(self._sync("exists", f"{self.root}/{relative_key}"))
        return os.path.isfile(self._local_full(relative_key))

    def read_range(
        self,
        relative_key: str,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> bytes:
        """Read part of one object, with ``start``/``end`` as ``cat_file`` takes them.

        **A negative ``start`` with no ``end`` asks for the object's last ``-start`` bytes**, and
        that form is the reason this method exists. An S3 range request of ``bytes=-N`` is one GET
        answered from the object's tail, which is how a ``sharded_v2`` container is meant to be
        opened: its descriptor and trailer live in the last 52 bytes, and its index length is
        declared there. Reading the whole object to parse those is O(object) traffic for O(tail)
        information -- measured on a real 11.5 MB aggregate container at 2.61 ms against 0.13 ms
        locally, and on S3 the difference is the transfer rather than the syscall.

        A range that runs past the object's end yields the bytes that exist, exactly as the three
        backends behave for a range GET; the caller compares the length against what it asked for.
        An object shorter than the requested suffix yields the whole object.

        Raises:
            StoreAccessError: if the object is absent or unreadable.
        """
        if self.kind == "mapping":
            assert self._mapping is not None
            try:
                blob = bytes(self._mapping[relative_key])
            except KeyError as exc:
                raise StoreAccessError(f"missing object {relative_key!r}") from exc
            return _slice_like_a_range_get(blob, start, end)
        if self.kind == "s3":
            try:
                data = self._sync(
                    "cat_file", f"{self.root}/{relative_key}", start=start, end=end
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
                raise StoreAccessError(
                    f"cannot read {relative_key!r}: {type(exc).__name__}: {exc}"
                ) from exc
            return bytes(data)
        full = self._local_full(relative_key)
        try:
            size = os.path.getsize(full)
            with open(full, "rb") as handle:
                offset, length = _range_to_offset_length(size, start, end)
                handle.seek(offset)
                return handle.read(length)
        except OSError as exc:
            raise StoreAccessError(f"cannot read {relative_key!r}: {exc}") from exc

    # -- writing and deleting ----------------------------------------------------

    def write(self, relative_key: str, data: bytes) -> None:
        """Write one object, creating parent prefixes as needed.

        Raises:
            StoreAccessError: on a read-only mapping store or a failed write.
        """
        if self.kind == "mapping":
            if not isinstance(self._mapping, dict):
                raise StoreAccessError("the supplied mapping store is read-only")
            self._mapping[relative_key] = data
            return
        if self.kind == "s3":
            try:
                self._sync("pipe_file", f"{self.root}/{relative_key}", data)
            except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
                raise StoreAccessError(
                    f"cannot write {relative_key!r}: {type(exc).__name__}: {exc}"
                ) from exc
            return
        full = self._local_full(relative_key)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(data)

    def delete_many(self, relative_keys: list[str]) -> int:
        """Delete objects, tolerating ones already gone. Returns the number removed.

        Staging cleanup is best-effort by design: a leftover staging object costs bytes
        until the next pass, whereas failing the aggregate would strand a published result
        over a cleanup problem.
        """
        if not relative_keys:
            return 0
        if self.kind == "mapping":
            if not isinstance(self._mapping, dict):
                return 0
            removed = 0
            for key in relative_keys:
                if self._mapping.pop(key, None) is not None:
                    removed += 1
            return removed
        if self.kind == "s3":
            targets = [f"{self.root}/{key}" for key in relative_keys]
            try:
                self._sync("rm", targets)
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                raise StoreAccessError(
                    f"cannot remove staging objects: {type(exc).__name__}: {exc}"
                ) from exc
            return len(relative_keys)
        removed = 0
        for key in relative_keys:
            try:
                os.remove(self._local_full(key))
                removed += 1
            except FileNotFoundError:
                pass
        return removed
