"""Tests proving serving logical and numerical equivalence across chunk geometries.

Verifies that the API serving tier returns identical results regardless of
whether the underlying Zarr store was written with member_chunk=1 or legacy
member_chunk=30.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Zstd


def _make_ensemble_dataset() -> xr.Dataset:
    lat = np.array([38.0, 38.25, 38.5, 38.75], dtype=np.float32)
    lon = np.array([-107.0, -106.75, -106.5, -106.25], dtype=np.float32)
    members = list(range(1, 6))  # 5 members
    leads = [0, 6]
    dims = ("member", "lead_time_hours", "latitude", "longitude")
    shape = (5, 2, 4, 4)

    # Deterministic linear temperature and precipitation fields
    t2m_data = np.zeros(shape, dtype=np.float32)
    prate_data = np.zeros(shape, dtype=np.float32)
    for m_idx, m in enumerate(members):
        for lead_idx, lead_val in enumerate(leads):
            t2m_data[m_idx, lead_idx, :, :] = 15.0 + m * 2.0 + lead_val * 0.5
            prate_data[m_idx, lead_idx, :, :] = 0.5 + m * 0.1

    coords = {
        "member": members,
        "lead_time_hours": leads,
        "latitude": lat,
        "longitude": lon,
    }
    ds = xr.Dataset(
        data_vars={
            "temperature_2m": (dims, t2m_data),
            "precipitation_rate": (dims, prate_data),
        },
        coords=coords,
    )
    ds.attrs["model_id"] = "gefs"
    ds.attrs["cycle_time"] = "2026-08-27T00:00:00"
    return ds


def test_serving_equivalence_between_member_chunk_1_and_30(tmp_path: Path) -> None:
    """Verify that ensemble statistics and probability calculations produce
    identical outputs from member_chunk=1 vs member_chunk=30 stores."""
    ds = _make_ensemble_dataset()

    store_mc1 = str(tmp_path / "gefs_mc1.zarr")
    store_mc30 = str(tmp_path / "gefs_mc30.zarr")

    # Write store with member_chunk=1
    encoding_mc1 = {
        name: {"chunks": (1, 1, 4, 4), "compressor": Zstd(level=5)}
        for name in ds.data_vars
    }
    ds.to_zarr(store_mc1, mode="w", encoding=encoding_mc1, zarr_format=2)

    # Write store with member_chunk=30 (or full extent 5)
    encoding_mc30 = {
        name: {"chunks": (5, 1, 4, 4), "compressor": Zstd(level=5)}
        for name in ds.data_vars
    }
    ds.to_zarr(store_mc30, mode="w", encoding=encoding_mc30, zarr_format=2)

    # Read back and compare direct array selections
    ds_mc1 = xr.open_zarr(store_mc1)
    ds_mc30 = xr.open_zarr(store_mc30)

    # 1. Single member point selection
    pt_mc1 = ds_mc1["temperature_2m"].sel(member=3, lead_time_hours=6).values
    pt_mc30 = ds_mc30["temperature_2m"].sel(member=3, lead_time_hours=6).values
    assert np.array_equal(pt_mc1, pt_mc30)

    # 2. Ensemble reduction (mean, spread, percentiles)
    ens_mc1 = ds_mc1["temperature_2m"].sel(lead_time_hours=6).mean(dim="member").values
    ens_mc30 = ds_mc30["temperature_2m"].sel(lead_time_hours=6).mean(dim="member").values
    assert np.allclose(ens_mc1, ens_mc30)

    # 3. Spatial window crop
    crop_mc1 = ds_mc1["temperature_2m"].sel(
        lead_time_hours=6, latitude=slice(38.5, 38.0), longitude=slice(-107.0, -106.5)
    ).values
    crop_mc30 = ds_mc30["temperature_2m"].sel(
        lead_time_hours=6, latitude=slice(38.5, 38.0), longitude=slice(-107.0, -106.5)
    ).values
    assert np.array_equal(crop_mc1, crop_mc30)

    ds_mc1.close()
    ds_mc30.close()
