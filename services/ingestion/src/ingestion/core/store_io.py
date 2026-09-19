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

    def _sync(self, func: Any, *args: Any) -> Any:
        """Run one s3fs coroutine against its loop."""
        import fsspec.asyn  # type: ignore[import-untyped]

        return fsspec.asyn.sync(self._loop, func, *args)

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
            found = self._sync(self._fs.find, f"{self.root}/{relative_prefix}")
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
                data = self._sync(self._fs.cat_file, f"{self.root}/{relative_key}")
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
            return bool(self._sync(self._fs.exists, f"{self.root}/{relative_key}"))
        return os.path.isfile(self._local_full(relative_key))

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
                self._sync(self._fs._pipe_file, f"{self.root}/{relative_key}", data)
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
                self._sync(self._fs.rm, targets)
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
