"""Shadow validation for the sharded_v2 storage format (storage + numerical).

Re-encodes an existing canonical store's committed shards into a ``sharded_v2``
shadow store and compares the two across the storage and numerical dimensions.
The shadow store lives in its own namespace, is never registered in the catalog
(no ``model_runs`` row), and therefore never enters canonical serving selection,
the reclamation queue, or the tombstone lifecycle — its lifetime is managed
entirely by this tooling.

This module intentionally depends on the ingestion package only. The
serving-tier comparison (which needs the API reader) lives in
``scripts/shadow_v2.py`` and the cross-package contract suite
(``tests/contracts``), because the per-package CI environments install only
their own package.

CLI entry point: ``scripts/shadow_v2.py`` (``validate`` / ``cleanup``).
"""

from __future__ import annotations

import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from domain.storage_dtype import (
    SHARDED_V2_FORMAT_VERSION,
    resolve_storage_dtype,
)

from ingestion.core.zarr_writer import (
    TRAILER_SIZE,
    encode_region_sharded_v2,
    read_slice,
    write_encoded_chunks,
)

#: Numeric difference tolerance (max absolute, internal units) per variable
#: class. f32-exception variables must be byte-identical (diff exactly 0.0);
#: f16 variables were measured at <= 0.125 mm / 0.03 degC worst case on real
#: data (see QUANTIZATION_BENCHMARK.md), so 0.5 is a generous regression bound.
F32_EXCEPTION_TOLERANCE = 0.0
F16_VARIABLE_TOLERANCE = 0.5

#: Threshold-coupled variables: the shadow comparison asserts ZERO behavior
#: difference for their production predicates (they are f32 exceptions).
_PRECIP_THRESHOLD_MM = 0.10
_CEILING_THRESHOLD_KM = 19.99

SHADOW_NAMESPACE_SEGMENT = "shadow-v2"


@dataclass
class VariableReport:
    """Per-variable shadow comparison result."""

    variable: str
    n_slices_compared: int = 0
    max_abs_diff: float = 0.0
    mae: float = 0.0
    p95_abs_diff: float = 0.0
    p99_abs_diff: float = 0.0
    precip_threshold_flips: int = 0
    ceiling_threshold_flips: int = 0
    tolerance: float = F16_VARIABLE_TOLERANCE
    notes: list[str] = field(default_factory=list)


def default_shadow_store_path(store: str | PathLike[str]) -> str:
    """Derive the shadow-store path from a canonical store path.

    Local: ``<cycle>.zarr`` -> ``<cycle>.shadow-v2.zarr`` (sibling).
    S3: ``.../{model}/{date}/{HH}/cycle.zarr`` -> ``.../{model}/{date}/{HH}/shadow-v2/``.
    """
    path = os.fspath(store)
    if path.startswith("s3://"):
        return path.rstrip("/") + "/" + SHADOW_NAMESPACE_SEGMENT
    parent = Path(path).parent
    name = Path(path).name
    return str(parent / f"{name}.{SHADOW_NAMESPACE_SEGMENT}")


def is_shadow_store_path(path: str | PathLike[str]) -> bool:
    """True only when the path lives inside the dedicated shadow namespace."""
    p = os.fspath(path).rstrip("/")
    return p.endswith(SHADOW_NAMESPACE_SEGMENT) or p.endswith(
        f"{SHADOW_NAMESPACE_SEGMENT}.zarr"
    )


def _iter_local_shards(store: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    root = Path(store)
    if not root.is_dir():
        return out
    for var_dir in sorted(root.iterdir()):
        if not var_dir.is_dir():
            continue
        shards = sorted(var_dir.glob("shard.*.shard"))
        if shards:
            out[var_dir.name] = shards
    return out


def _shard_member_lead_is_mean(path: Path) -> tuple[int | None, int, bool]:
    from ingestion.core.zarr_writer import _parse_shard_filename

    return _parse_shard_filename(path.name)


def _shard_payload_size(path: Path) -> int:
    """Compressed payload byte size of one shard container (index+trailer excluded)."""
    data = path.read_bytes()
    if len(data) < TRAILER_SIZE:
        return 0
    num_chunks, index_size, _ = struct.unpack("<III", data[-TRAILER_SIZE:])
    return int(len(data) - TRAILER_SIZE - int(index_size))


def write_shadow_store(
    source_store: str | PathLike[str],
    shadow_store: str | PathLike[str],
) -> dict[str, Any]:
    """Re-encode every committed shard of a canonical store into sharded_v2.

    Reads each committed slice through the source store's own read path
    (:func:`ingestion.core.zarr_writer.read_slice`, dtype-aware) and re-encodes
    it with the sharded_v2 encoder into the shadow namespace. The shadow store
    gets its own committed manifest stamped ``sharded_v2`` so readers dispatch
    to :class:`~api.core.zarr_v2.ShardedV2Reader`. Nothing is written to the
    catalog.

    Returns:
        Statistics: shard counts, source/shadow payload byte totals, and the
        wall-clock re-encode duration.
    """
    from ingestion.core.markers import write_manifest

    source = os.fspath(source_store)
    shadow = os.fspath(shadow_store)
    if not source.startswith("s3://") and not shadow.startswith("s3://"):
        if Path(source).resolve() == Path(shadow).resolve():
            raise ValueError("Shadow store path must differ from the source store path.")
    if not is_shadow_store_path(shadow):
        raise ValueError(
            f"Shadow store path {shadow!r} must live inside the dedicated "
            f"{SHADOW_NAMESPACE_SEGMENT!r} namespace."
        )

    started = time.perf_counter()
    source_bytes = 0
    shadow_bytes = 0
    n_shards = 0
    for var_name, shard_paths in _iter_local_shards(source).items():
        for shard_path in shard_paths:
            member, lead, is_mean = _shard_member_lead_is_mean(shard_path)
            if is_mean and member is not None:
                continue
            source_bytes += _shard_payload_size(shard_path)
            slice_vals = read_slice(
                source, var_name, lead_time_hours=lead, member=member, is_mean=is_mean
            )
            if slice_vals is None:
                continue
            dims = ("latitude", "longitude")
            ds = xr.Dataset({var_name: (dims, slice_vals)})
            encoded = encode_region_sharded_v2(
                ds,
                member=member,
                lead_time_hours=lead,
                is_mean=is_mean,
            )
            write_encoded_chunks(shadow, encoded)
            shadow_bytes += sum(len(payload) for _, payload in encoded)
            n_shards += 1

    write_manifest(
        shadow,
        {
            "manifest_schema_version": 1,
            "store_protocol_mode": "HYBRID",
            "storage_format_version": SHARDED_V2_FORMAT_VERSION,
            "generation": "shadow",
            "run_identity": {
                "model_version_id": "shadow",
                "cycle_time": "shadow",
                "is_ensemble": False,
            },
        },
    )
    duration = time.perf_counter() - started
    return {
        "source_store": source,
        "shadow_store": shadow,
        "shards_reencoded": n_shards,
        "source_payload_bytes": source_bytes,
        "shadow_payload_bytes": shadow_bytes,
        "storage_ratio": (shadow_bytes / source_bytes) if source_bytes else None,
        "reencode_seconds": duration,
    }


def compare_stores_numerical(
    source_store: str | PathLike[str],
    shadow_store: str | PathLike[str],
) -> list[VariableReport]:
    """Compare every committed slice of both stores numerically.

    The f32 semantic exceptions (``precipitation_amount_3h``, ``cloud_ceiling``)
    must be bit-identical; their production predicates (exactly-0.10 mm,
    19.99 km sentinel) must show zero flips.
    """
    source = os.fspath(source_store)
    reports: list[VariableReport] = []
    for var_name, shard_paths in _iter_local_shards(source).items():
        resolved = resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, var_name, "det")
        report = VariableReport(
            variable=var_name,
            tolerance=F32_EXCEPTION_TOLERANCE
            if resolved.itemsize == 4
            else F16_VARIABLE_TOLERANCE,
        )
        abs_diffs: list[float] = []
        for shard_path in shard_paths:
            member, lead, is_mean = _shard_member_lead_is_mean(shard_path)
            if is_mean and member is not None:
                continue
            source_slice = read_slice(
                source, var_name, lead_time_hours=lead, member=member, is_mean=is_mean
            )
            shadow_slice = read_slice(
                shadow_store, var_name, lead_time_hours=lead, member=member, is_mean=is_mean
            )
            if source_slice is None or shadow_slice is None:
                continue
            report.n_slices_compared += 1
            a = source_slice.astype(np.float64)
            b = shadow_slice.astype(np.float64)
            finite = np.isfinite(a) & np.isfinite(b)
            if not finite.any():
                continue
            diff = np.abs(b[finite] - a[finite])
            abs_diffs.extend(diff.tolist())
            if var_name == "precipitation_amount_3h":
                report.precip_threshold_flips += int(
                    np.sum((b[finite] <= _PRECIP_THRESHOLD_MM) != (a[finite] <= _PRECIP_THRESHOLD_MM))
                )
            if var_name == "cloud_ceiling":
                report.ceiling_threshold_flips += int(
                    np.sum((b[finite] >= _CEILING_THRESHOLD_KM) != (a[finite] >= _CEILING_THRESHOLD_KM))
                )
        if abs_diffs:
            arr = np.asarray(abs_diffs)
            report.max_abs_diff = float(arr.max())
            report.mae = float(arr.mean())
            report.p95_abs_diff = float(np.percentile(arr, 95))
            report.p99_abs_diff = float(np.percentile(arr, 99))
        if resolved.itemsize == 4 and report.n_slices_compared:
            if report.max_abs_diff != 0.0:
                report.notes.append("f32 exception shows non-zero diff")
            if var_name == "precipitation_amount_3h" and report.precip_threshold_flips:
                report.notes.append("precip 0.10mm predicate flipped")
            if var_name == "cloud_ceiling" and report.ceiling_threshold_flips:
                report.notes.append("ceiling 19.99km predicate flipped")
        reports.append(report)
    return reports


def cleanup_shadow_stores(
    shadow_path: str | PathLike[str],
    *,
    older_than_days: float | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Delete a shadow store, guarded to the shadow namespace only.

    Refuses any path outside the ``shadow-v2`` namespace (the guard is the
    safety contract: cleanup can never accept an arbitrary production prefix).
    With ``older_than_days`` set, deletion only happens when the store's mtime
    is older than the age bound; ``dry_run`` reports without deleting.
    """
    path = os.fspath(shadow_path)
    if not is_shadow_store_path(path):
        raise ValueError(
            f"Refusing cleanup: {path!r} is not inside the dedicated "
            f"{SHADOW_NAMESPACE_SEGMENT!r} shadow namespace."
        )
    target = Path(path)
    info: dict[str, Any] = {"path": path, "exists": target.is_dir(), "deleted": False}
    if not info["exists"]:
        return info
    if older_than_days is not None:
        age_days = (time.time() - target.stat().st_mtime) / 86400
        info["age_days"] = age_days
        if age_days < older_than_days:
            info["skipped"] = f"younger than {older_than_days} days"
            return info
    info["dry_run"] = dry_run
    if not dry_run:
        shutil.rmtree(target)
        info["deleted"] = True
    return info
