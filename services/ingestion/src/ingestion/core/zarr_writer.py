"""Chunked, compressed Zarr storage for normalized forecast datasets.

A single ``xarray.Dataset`` is written to a Zarr store (local directory or
an ``s3://`` bucket) with per-dimension chunking and ``Zstd`` compression,
then read back for round-trip verification.

The ``s3://`` scheme is resolved to a MinIO/S3-compatible store using
:class:`s3fs.S3FileSystem` configured from :class:`IngestionSettings`.
This is intentionally narrow: no bucket lifecycle or multi-file
management lives here.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Hashable, Mapping, MutableMapping

import numpy as np
import numpy.typing as npt
import s3fs  # type: ignore[import-untyped]
import xarray as xr
from numcodecs import Zstd  # type: ignore[import-untyped]

from ingestion.core.config import IngestionSettings, settings

#: Default chunks applied per dimension when none are provided.
DEFAULT_CHUNKS: Mapping[str, int] = {
    "time": 1,
    "lead_time_hours": 1,
    "isobaricInhPa": 1,
    "latitude": 100,
    "longitude": 100,
}


Store = str | PathLike[str] | Mapping[str, bytes]


def _resolve_store(
    store: str | PathLike[str] | Mapping[str, bytes],
) -> str | PathLike[str] | MutableMapping[str, bytes]:
    """Resolve a Zarr store target to an xarray-compatible mapping.

    Args:
        store: A local directory path, an ``s3://`` URL, or an existing
            mutable mapping (e.g. ``fsspec`` ``FSMap``).

    Returns:
        An object accepted by :func:`xarray.Dataset.to_zarr` /
        :func:`xarray.open_zarr`.

    Raises:
        ValueError: If the ``s3://`` URL cannot be parsed or the scheme is
            unsupported.
    """
    if isinstance(store, MutableMapping):
        return store
    if isinstance(store, Mapping):
        return dict(store)
    path = os.fspath(store)
    if path.startswith("s3://"):
        return _resolve_s3_store(path, settings)
    if path.startswith("file://"):
        return path[len("file://") :]
    return path


def _resolve_s3_store(
    path: str, conn_settings: IngestionSettings
) -> MutableMapping[str, bytes]:
    """Build an ``FSMap`` over an ``s3://`` URL using S3 settings.

    Args:
        path: An ``s3://bucket/prefix`` URL.
        conn_settings: Ingestion settings providing MinIO credentials.

    Returns:
        An ``s3fs`` mapping object accepted by xarray.

    Raises:
        ValueError: If the bucket/prefix cannot be derived from the URL.
    """
    rest = path[len("s3://") :].strip("/")
    if not rest:
        raise ValueError(f"Invalid S3 store URL: {path!r}")

    fs = s3fs.S3FileSystem(
        key=conn_settings.MINIO_ACCESS_KEY,
        secret=conn_settings.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": _endpoint_url(conn_settings)},
        use_listings_cache=False,
    )
    # ``get_mapper`` is untyped in the s3fs stub (import-untyped), so mypy
    # sees ``Any``. Narrowing through a ``MutableMapping[str, bytes]``-typed
    # intermediate enforces the declared return type instead of suppressing it.
    mapper: MutableMapping[str, bytes] = fs.get_mapper(rest)
    return mapper


def _endpoint_url(conn_settings: IngestionSettings) -> str:
    """Build the S3 endpoint URL from MinIO settings."""
    scheme = "https" if conn_settings.MINIO_SECURE else "http"
    return f"{scheme}://{conn_settings.MINIO_ENDPOINT}"


def write_dataset(
    dataset: xr.Dataset,
    store: str | PathLike[str] | Mapping[str, bytes],
    *,
    chunks: Mapping[str, int] | None = None,
) -> str:
    """Write a normalized dataset to a Zarr store.

    Chunk sizes default to :data:`DEFAULT_CHUNKS` (falling back to the
    full extent of any dimension not covered) and every data variable is
    stored with ``Zstd`` compression. No dask backend is required; the
    chunk grid and compressor are recorded in the Zarr metadata.

    Args:
        dataset: Normalized dataset (e.g. from
            :func:`ingestion.providers.noaa.parser.parse_grib2`).
        store: Local path, ``s3://`` URL, or a mutable mapping.
        chunks: Optional per-dimension chunk sizes.

    Returns:
        The store target as a string (for reporting).

    Raises:
        ValueError: If the store target is unsupported.
    """
    resolved = _resolve_store(store)
    defaults = chunks or DEFAULT_CHUNKS
    encoding = {
        name: {"chunks": _chunk_sizes(dataset, defaults)[name], "compressor": Zstd(level=5)}
        for name in dataset.data_vars
    }
    dataset.to_zarr(resolved, mode="w", encoding=encoding)
    return os.fspath(store) if isinstance(store, PathLike) else str(store)


def _chunk_sizes(
    dataset: xr.Dataset,
    defaults: Mapping[str, int],
) -> dict[Hashable, tuple[int, ...]]:
    """Compute per-variable chunk tuples from the defaults, clamped to dims.

    A dimension not covered by ``defaults`` is chunked at its full extent so no
    variable is left without an explicit chunk grid. The result is clamped to
    the dimension size so a small test grid never requests a chunk larger than
    the axis.

    Args:
        dataset: The dataset whose dims drive the chunk grid.
        defaults: Per-dimension default chunk sizes (e.g.
            :data:`DEFAULT_CHUNKS`).

    Returns:
        A mapping of data-variable name to its chunk tuple.
    """
    out: dict[Hashable, tuple[int, ...]] = {}

    def _size(dim: Hashable) -> int:
        size = dataset.sizes[dim]
        assert size is not None
        return size

    def _default(dim: Hashable) -> int:
        return defaults.get(str(dim), _size(dim))

    for name in dataset.data_vars:
        out[name] = tuple(
            min(_default(dim), _size(dim)) for dim in dataset[name].dims
        )
    return out


def write_dataset_atomic(
    dataset: xr.Dataset,
    store: str | PathLike[str] | Mapping[str, bytes],
    *,
    chunks: Mapping[str, int] | None = None,
) -> str:
    """Write a dataset to a Zarr store without ever exposing a partial store.

    Cycle stores accumulate leads over successive invocations, so a
    partially-written store could otherwise be served by the API between the
    first write and the final rename. This writes to a unique staged sibling
    and then atomically swaps it into the final path (remove-then-rename for
    local paths; a best-effort staging for ``s3://`` stores). If the write or
    swap fails, the staged sibling is removed and the previous store (if any)
    is left untouched.

    Args:
        dataset: Normalized dataset to write.
        store: Local path, ``s3://`` URL, or a mutable mapping.
        chunks: Optional per-dimension chunk sizes (defaults to
            :data:`DEFAULT_CHUNKS`).

    Returns:
        The store target as a string (for reporting).

    Raises:
        ValueError: If the store target is unsupported.
    """
    if isinstance(store, (Mapping, PathLike)):
        # Mappings and path-like targets have no atomic-rename semantics here;
        # fall back to the plain write (callers use atomic writes for the
        # cycle-store path, which is a local dir or s3:// URL).
        return write_dataset(dataset, store, chunks=chunks)

    target = os.fspath(store)
    if target.startswith("s3://"):
        # S3-compatible stores: write to a unique staged prefix then list-copy
        # it into the final prefix, removing the stage. This is best-effort
        # atomicity (S3 has no rename); a partial final store is bounded by the
        # copy window and the API's run-identity validation refuses mismatched
        # cycles.
        stage = f"{target.rstrip('/')}.tmp"
        try:
            _delete_store_path(stage)
            write_dataset(dataset, stage, chunks=chunks)
            _delete_store_path(target)
            _move_store_path(stage, target)
            return target
        except Exception:
            _delete_store_path(stage)
            raise

    # Local directory target: write to a staged sibling then atomically swap.
    stage = f"{target}.tmp"
    _delete_store_path(stage)
    write_dataset(dataset, stage, chunks=chunks)
    _delete_store_path(target)
    os.replace(stage, target)
    return target


def _delete_store_path(path: str) -> None:
    """Best-effort recursive delete of a local path or ``s3://`` store."""
    if path.startswith("s3://"):
        resolved = _resolve_store(path)
        if isinstance(resolved, MutableMapping):
            try:
                list(resolved.keys())
                for key in list(resolved.keys()):
                    resolved.pop(key, None)
            except Exception:
                pass
        return
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _move_store_path(src: str, dst: str) -> None:
    """Best-effort copy of an ``s3://`` store from ``src`` to ``dst``."""
    src_resolved = _resolve_store(src)
    dst_resolved = _resolve_store(dst)
    if isinstance(src_resolved, MutableMapping) and isinstance(dst_resolved, MutableMapping):
        for key in list(src_resolved.keys()):
            dst_resolved[key] = src_resolved[key]
        for key in list(src_resolved.keys()):
            src_resolved.pop(key, None)


def prepare_run_store(
    dataset: xr.Dataset,
    store: str | PathLike[str] | Mapping[str, bytes],
    *,
    expected_lead_time_hours: tuple[int, ...],
    expected_members: tuple[int, ...] = (),
) -> str:
    """Pre-allocate the serving store for a run before region writes begin.

    The store is initialized with the full expected coordinate structure:
    every expected lead (and, for GEFS, every expected member) as dimensions,
    NaN-filled data variables. Subsequent per-file ``commit_region`` calls
    write only their own (lead, member) slice, so writes are independent and
    no whole-store read-modify-write ever happens.

    Args:
        dataset: A parsed (normalized) file of the run used to derive the grid
            axes and data-variable layout (dtype, dims, attributes).
        store: Local path, ``s3://`` URL, or a mutable mapping.
        expected_lead_time_hours: The full set of leads the run will serve.
        expected_members: The full set of GEFS member identities (1..30). Empty
            for deterministic models.

    Returns:
        The store target as a string.

    Raises:
        ValueError: If the dataset lacks the grid axes or a lead coordinate.
    """
    if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
        raise ValueError("prepare_run_store requires latitude/longitude coords")
    if "lead_time_hours" not in dataset.coords:
        raise ValueError("prepare_run_store requires a lead_time_hours coord")

    lat = dataset.coords["latitude"].values
    lon = dataset.coords["longitude"].values
    # When the caller declares no expected leads, fall back to the dataset's own
    # lead coordinate (a library caller ingesting a single file treats that
    # file as the whole run). The CLI always declares the full expected set.
    if expected_lead_time_hours:
        leads = list(expected_lead_time_hours)
    else:
        lead_values = dataset.coords["lead_time_hours"].values
        if np.ndim(lead_values) == 0:
            leads = [int(lead_values)]
        else:
            leads = [int(v) for v in lead_values]
    members = list(expected_members) if expected_members else []

    data_vars: dict[str, tuple[tuple[str, ...], npt.NDArray[np.float32]]] = {}
    for name, da in dataset.data_vars.items():
        # Only the spatial axes come from the source file; the lead (and for
        # GEFS, member) axes are the run's expected dimensions.
        base_dims: tuple[str, ...] = tuple(
            str(d) for d in da.dims if str(d) in ("latitude", "longitude")
        )
        grid_shape = tuple(int(dataset.sizes[d]) for d in base_dims)
        if members:
            dims: tuple[str, ...] = ("member", "lead_time_hours") + base_dims
            shape = (len(members), len(leads)) + grid_shape
        else:
            dims = ("lead_time_hours",) + base_dims
            shape = (len(leads),) + grid_shape
        filled: npt.NDArray[np.float32] = np.full(
            shape, np.nan, dtype=da.dtype
        )
        data_vars[name] = (dims, filled)

    coords: dict[str, object] = {
        "lead_time_hours": leads,
        "latitude": lat,
        "longitude": lon,
    }
    if expected_members:
        coords["member"] = members

    store_ds = xr.Dataset(data_vars=data_vars, coords=coords)
    # Copy the dataset and variable attributes (model_id, cycle_time, units,
    # ...) from the source file so the store is self-describing about its
    # forecast run and its canonical units.
    store_ds.attrs = dict(dataset.attrs)
    for name, da in dataset.data_vars.items():
        store_ds[name].attrs = dict(da.attrs)
    # Preserve the cycle identity on the store even when the source dataset
    # carried it only as a ``time`` coordinate (the parser derives cycle_time
    # from the GRIB ``time`` coord; synthetic datasets in tests may carry only
    # ``time``). The pipeline's store-identity guard reads ``cycle_time``
    # attribute then falls back to the ``time`` coordinate, so carrying either
    # keeps the store identifiable.
    if "cycle_time" not in store_ds.attrs and "time" in dataset.coords:
        import numpy as _np

        value = dataset.coords["time"].values
        item = value.item() if _np.ndim(value) != 0 else value
        store_ds.attrs["cycle_time"] = str(
            _np.datetime_as_string(
                _np.asarray(item, dtype="datetime64[ns]"), unit="s"
            ).item()
        )
    # Don't write empty (all-NaN) chunks for the un-committed regions; only
    # committed regions materialize chunks.
    chunk_map = _chunk_sizes(store_ds, DEFAULT_CHUNKS)
    encoding = {
        name: {
            "chunks": chunk_map[name],
            "compressor": Zstd(level=5),
            "write_empty_chunks": False,
        }
        for name in store_ds.data_vars
    }
    store_ds.to_zarr(_resolve_store(store), mode="w", encoding=encoding)
    return os.fspath(store) if isinstance(store, PathLike) else str(store)


def commit_region(
    dataset: xr.Dataset,
    store: str | PathLike[str] | Mapping[str, bytes],
    *,
    lead_time_hours: int | None = None,
    member: int | None = None,
) -> str:
    """Commit a single-lead (and optional single-member) file into an existing store.

    This is the region-write primitive that replaces whole-store read-modify-
    write. The dataset is written into the store at the coordinate-driven
    ``(lead_time_hours[, member])`` region:

    * only the target lead (and member) slice is written;
    * existing data in other leads/members is never read or rewritten;
    * xarray ``to_zarr(mode="r+", region=...)`` is used, which in region mode
      writes chunk data only and does not mutate shared metadata.

    The region is derived from the dataset's ``lead_time_hours`` coordinate
    value (and, when ``member`` is given, its ``member`` coordinate value) via
    ``region="auto"`` so the mapping is coordinate-driven (a ``gep17`` file
    always lands in member 17, never in member 0 because of completion order).

    Args:
        dataset: The normalized single-lead (optionally single-member)
            dataset to commit.
        store: Local path, ``s3://`` URL, or a mutable mapping of the existing
            serving store.
        lead_time_hours: Explicit lead to target. When ``None``, the lead is
            derived from the dataset's ``lead_time_hours`` coordinate.
        member: Explicit member identity (``1..30``) to target. When ``None``
            but the dataset has a ``member`` dimension of length 1, the member
            is derived from its ``member`` coordinate value.

    Returns:
        The store target as a string.

    Raises:
        ValueError: If the store target is unsupported or the region cannot be
            derived.
    """
    if lead_time_hours is None:
        if "lead_time_hours" not in dataset.coords:
            raise ValueError(
                "commit_region requires a 'lead_time_hours' coordinate to "
                "derive the target region."
            )
        lead_value = dataset.coords["lead_time_hours"].values
        lead_time_hours = int(lead_value.item() if np.ndim(lead_value) else lead_value)

    if member is None and "member" in dataset.dims:
        member_value = dataset.coords["member"].values
        if np.ndim(member_value) == 0:
            member = int(member_value)
        elif member_value.size == 1:
            member = int(member_value.reshape(-1)[0])

    resolved = _resolve_store(store)

    # Resolve the lead/member coordinate values to positional indices against
    # the pre-allocated store's coordinates. This is coordinate-driven: a
    # gep17 file always targets the member position whose value is 17, never a
    # completion-order index.
    existing = read_dataset(store)
    lead_index = _coordinate_index(existing, "lead_time_hours", lead_time_hours)
    member_index = None
    if member is not None:
        if "member" not in existing.coords:
            raise ValueError(
                "commit_region: the store has no 'member' coordinate but a "
                "member identity was requested."
            )
        member_index = _coordinate_index(existing, "member", member)

    # The dataset is single-lead (and, for GEFS, single-member). Build the
    # positional region slices. Data variables already carry the lead (and
    # member) dims; re-assert coordinates so the region write is exact.
    coords: dict[str, object] = {"lead_time_hours": [lead_time_hours]}
    if member is not None:
        coords["member"] = [member]
    target = dataset.assign_coords(coords)

    region: dict[str, slice] = {
        "lead_time_hours": slice(lead_index, lead_index + 1),
        "latitude": slice(None),
        "longitude": slice(None),
    }
    if member_index is not None:
        region["member"] = slice(member_index, member_index + 1)

    # Scalar coordinates not in the region dims (e.g. the parser's ``time``
    # coordinate) cannot be written by a region write; drop them. They are
    # already captured in the store's attributes at pre-allocation time.
    drop_vars = [
        name
        for name in target.coords
        if name not in region and name not in ("latitude", "longitude")
    ]
    if drop_vars:
        target = target.drop_vars(drop_vars)

    target.to_zarr(resolved, mode="r+", region=region)
    return os.fspath(store) if isinstance(store, PathLike) else str(store)


def _coordinate_index(
    dataset: xr.Dataset, dim: str, value: int
) -> int:
    """Return the positional index of ``value`` along ``dim`` in ``dataset``.

    Args:
        dataset: A dataset (e.g. the pre-allocated store) with the coordinate.
        dim: The dimension/coordinate name.
        value: The coordinate value whose index is needed.

    Returns:
        The positional index.

    Raises:
        ValueError: If the value is absent from the coordinate.
    """
    if dim not in dataset.coords:
        raise ValueError(f"commit_region: store has no '{dim}' coordinate.")
    values = dataset.coords[dim].values
    flat = np.atleast_1d(values).reshape(-1)
    for i in range(flat.size):
        candidate = flat[i]
        if int(candidate) == int(value):
            return i
    raise ValueError(
        f"commit_region: value {value} not found in store coordinate "
        f"'{dim}' (available: {[int(c) for c in flat]})."
    )


def read_dataset(store: str | PathLike[str] | Mapping[str, bytes]) -> xr.Dataset:
    """Read a Zarr store back into a dataset.

    The dataset is returned numpy-backed (no dask required); the chunk
    grid and compressor persisted at write time remain available via
    ``encoding``.

    Args:
        store: The same local path, ``s3://`` URL, or mapping used to
            write.

    Returns:
        The dataset read back from the Zarr store.
    """
    resolved = _resolve_store(store)
    # ``xr.open_zarr`` is overloaded and infers ``Any`` for the ``Any``-typed
    # store; the value is always a concrete ``Dataset`` at runtime, so narrow
    # through a typed intermediate to satisfy the declared return type.
    dataset: xr.Dataset = xr.open_zarr(resolved)
    return dataset


def store_exists(store: str | PathLike[str] | Mapping[str, bytes]) -> bool:
    """Return whether a readable Zarr store already exists at ``store``.

    Used by the ingestion orchestration to decide whether a write is a fresh
    ingest (the target may be written in place) or a re-ingest (the target is
    already served by a ``ready`` catalog row and must not be truncated until
    the new store is known-good). A store that exists but fails to open is
    treated as absent so it is never written over destructively.

    Args:
        store: A local path, ``s3://`` URL, or mapping.

    Returns:
        True when a store can be opened at ``store``, False otherwise.
    """
    resolved = _resolve_store(store)
    if isinstance(resolved, str):
        return os.path.exists(resolved)
    try:
        dataset = xr.open_zarr(resolved)
    except Exception:
        return False
    dataset.close()
    return True
