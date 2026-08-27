"""Tests proving legacy member_chunk=30 and new member_chunk=1 store compatibility.

Invariants verified:
1. NEW GEFS stores initialize with member_chunk=1 (chunks=[1, 1, ...]).
2. EXISTING legacy stores (member_chunk=30) preserve their persisted .zarray
   chunk geometry when subsequent single-member commits occur.
3. Same-cycle re-ingestion into both legacy and new stores preserves logical
   and numerical equivalence without corrupting adjacent members.
4. An existing array never mutates its chunk geometry on write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Zstd

from ingestion.core.pipeline import _commit_region
from ingestion.core.zarr_writer import (
    _resolve_store,
    prepare_run_store,
    read_dataset,
)


def _make_gefs_file(lead: int, member: int) -> xr.Dataset:
    lat = np.array([40.0, 41.0, 42.0, 43.0], dtype=np.float32)
    lon = np.array([250.0, 251.0, 252.0, 253.0], dtype=np.float32)
    dims = ("latitude", "longitude")
    shape = (4, 4)
    data = np.full(shape, 280.0 + member * 0.5 + lead * 0.1, dtype=np.float32)
    coords = {
        "lead_time_hours": lead,
        "member": member,
        "latitude": lat,
        "longitude": lon,
        "time": np.datetime64("2026-08-27T00:00:00", "ns"),
    }
    return xr.Dataset(
        data_vars={"t2m": (dims, data)},
        coords=coords,
    )


def test_new_store_initializes_with_member_chunk_1(tmp_path: Path) -> None:
    """A fresh prepare_run_store for GEFS uses member_chunk=1 in .zarray."""
    store_path = str(tmp_path / "gefs_new.zarr")
    seed = _make_gefs_file(0, 1)

    prepare_run_store(
        seed,
        store_path,
        expected_lead_time_hours=(0, 3),
        expected_members=tuple(range(1, 31)),
    )

    za_path = os.path.join(store_path, "t2m", ".zarray")
    assert os.path.exists(za_path)
    with open(za_path, "r", encoding="utf-8") as f:
        za = json.load(f)

    # 30 members, 2 leads, 4 lat, 4 lon
    assert za["shape"] == [30, 2, 4, 4]
    # member_chunk=1, lead_chunk=1, lat=4, lon=4 (clamped to axis length)
    assert za["chunks"] == [1, 1, 4, 4]


def test_legacy_store_preserves_member_chunk_30_on_reingest(tmp_path: Path) -> None:
    """An existing store initialized with member_chunk=30 preserves its chunks
    when subsequent _commit_region calls execute."""
    store_path = str(tmp_path / "gefs_legacy.zarr")
    seed = _make_gefs_file(0, 1)

    # Manually initialize legacy store with member_chunk=30
    lat = seed.coords["latitude"].values
    lon = seed.coords["longitude"].values
    store_ds = xr.Dataset(
        data_vars={
            "t2m": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                np.full((30, 2, 4, 4), np.nan, dtype=np.float32),
            )
        },
        coords={
            "member": list(range(1, 31)),
            "lead_time_hours": [0, 3],
            "latitude": lat,
            "longitude": lon,
        },
    )
    store_ds.attrs["cycle_time"] = "2026-08-27T00:00:00"
    # Legacy chunks: member=30
    legacy_encoding = {
        "t2m": {
            "chunks": (30, 1, 4, 4),
            "compressor": Zstd(level=5),
            "write_empty_chunks": False,
        }
    }
    store_ds.to_zarr(_resolve_store(store_path), mode="w", encoding=legacy_encoding)

    za_path = os.path.join(store_path, "t2m", ".zarray")
    with open(za_path, "r", encoding="utf-8") as f:
        za_initial = json.load(f)
    assert za_initial["chunks"] == [30, 1, 4, 4]

    # Commit member 1, lead 0 through pipeline _commit_region
    f1 = _make_gefs_file(0, 1)
    _commit_region(
        f1,
        store_path,
        member=1,
        expected_lead_time_hours=(0, 3),
        expected_members=tuple(range(1, 31)),
    )

    # Commit member 15, lead 0
    f15 = _make_gefs_file(0, 15)
    _commit_region(
        f15,
        store_path,
        member=15,
        expected_lead_time_hours=(0, 3),
        expected_members=tuple(range(1, 31)),
    )

    # Verify .zarray chunk geometry was NOT modified
    with open(za_path, "r", encoding="utf-8") as f:
        za_after = json.load(f)
    assert za_after["chunks"] == [30, 1, 4, 4]

    # Read back dataset and verify values
    read_ds = read_dataset(store_path)
    val_m1 = read_ds["t2m"].sel(member=1, lead_time_hours=0).values
    val_m15 = read_ds["t2m"].sel(member=15, lead_time_hours=0).values
    val_m2 = read_ds["t2m"].sel(member=2, lead_time_hours=0).values

    assert np.allclose(val_m1, 280.0 + 1 * 0.5)
    assert np.allclose(val_m15, 280.0 + 15 * 0.5)
    # Member 2 was not committed; must be NaN
    assert np.all(np.isnan(val_m2))


def test_member_chunk_1_patch_isolation(tmp_path: Path) -> None:
    """In a member_chunk=1 store, updating member 2 replaces only member 2's chunk
    and leaves member 1 and member 3 intact."""
    store_path = str(tmp_path / "gefs_mc1_patch.zarr")
    seed = _make_gefs_file(0, 1)

    prepare_run_store(
        seed,
        store_path,
        expected_lead_time_hours=(0,),
        expected_members=(1, 2, 3),
    )

    # Commit members 1, 2, 3 via pipeline _commit_region
    for m in (1, 2, 3):
        _commit_region(
            _make_gefs_file(0, m),
            store_path,
            member=m,
            expected_lead_time_hours=(0,),
            expected_members=(1, 2, 3),
        )

    # Verify initial values
    ds1 = read_dataset(store_path)
    assert np.allclose(ds1["t2m"].sel(member=2, lead_time_hours=0).values, 280.0 + 2 * 0.5)

    # Re-ingest member 2 with new data (e.g. corrected values)
    corrected_file = _make_gefs_file(0, 2)
    corrected_file["t2m"].values[:] = 999.0
    _commit_region(
        corrected_file,
        store_path,
        member=2,
        expected_lead_time_hours=(0,),
        expected_members=(1, 2, 3),
    )

    # Verify member 2 was updated while members 1 and 3 are unchanged
    ds2 = read_dataset(store_path)
    assert np.allclose(ds2["t2m"].sel(member=1, lead_time_hours=0).values, 280.0 + 1 * 0.5)
    assert np.allclose(ds2["t2m"].sel(member=2, lead_time_hours=0).values, 999.0)
    assert np.allclose(ds2["t2m"].sel(member=3, lead_time_hours=0).values, 280.0 + 3 * 0.5)
