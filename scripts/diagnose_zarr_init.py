"""Diagnostic script for Zarr initialization latency and S3 request analysis (Tasks 15-20)."""

from __future__ import annotations

import collections
import time
from typing import Any
import uuid

import numpy as np
import xarray as xr
import zarr
from numcodecs import Zstd

from ingestion.core.config import settings
from ingestion.core.s3 import get_s3_fs, resolve_s3_mapper
from ingestion.core.zarr_writer import DEFAULT_CHUNKS


def make_dummy_gfs_dataset() -> xr.Dataset:
    lat = np.linspace(-90.0, 90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    coords = {
        "lead_time_hours": [0],
        "latitude": lat,
        "longitude": lon,
        "time": np.datetime64("2026-09-12T06:00:00"),
    }
    dims = ("latitude", "longitude")
    shape = (721, 1440)
    # 14 standard forecast variables
    vars_names = [
        "temperature_2m", "precipitation_rate", "precipitation_amount_3h",
        "crain", "csnow", "cfrzr", "cicep", "relative_humidity_2m",
        "wind_gust", "visibility", "snow_depth", "wind_u_10m", "wind_v_10m",
        "cloud_cover_3h"
    ]
    data_vars = {v: (dims, np.zeros(shape, dtype=np.float32)) for v in vars_names}
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["model_id"] = "gfs"
    ds.attrs["cycle_time"] = "2026-09-12T06:00:00"
    return ds


class S3CallCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = collections.defaultdict(int)
        self.original_call_s3 = None

    def __enter__(self) -> S3CallCounter:
        import s3fs
        self.original_call_s3 = s3fs.S3FileSystem._call_s3
        counter = self

        async def _mock_call_s3(self_fs, method, *args, **kwargs):
            counter.counts[method] += 1
            return await counter.original_call_s3(self_fs, method, *args, **kwargs)

        s3fs.S3FileSystem._call_s3 = _mock_call_s3
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import s3fs
        s3fs.S3FileSystem._call_s3 = self.original_call_s3


def instrumented_prepare_run_store(
    dataset: xr.Dataset,
    store_path: str,
    expected_leads: tuple[int, ...] = (),
    expected_members: tuple[int, ...] = (),
) -> dict[str, Any]:
    timings = {}

    with S3CallCounter() as counter:
        t_total_start = time.monotonic()

        # Step 1: Resolve store
        t0 = time.monotonic()
        resolved = resolve_s3_mapper(store_path, settings)
        timings["resolve_store_s"] = time.monotonic() - t0

        # Step 2: Write coordinates (ds_coords.to_zarr)
        t0 = time.monotonic()
        lat = dataset.coords["latitude"].values
        lon = dataset.coords["longitude"].values
        leads = list(expected_leads) if expected_leads else [0]
        members = list(expected_members) if expected_members else []
        coords: dict[str, object] = {
            "lead_time_hours": leads,
            "latitude": lat,
            "longitude": lon,
        }
        if expected_members:
            coords["member"] = members
        ds_coords = xr.Dataset(coords=coords)
        ds_coords.attrs = dict(dataset.attrs)
        ds_coords.to_zarr(resolved, mode="w", consolidated=False, zarr_format=2)
        timings["coords_to_zarr_s"] = time.monotonic() - t0

        # Step 3: Open group
        t0 = time.monotonic()
        root = zarr.open_group(resolved, mode="a", zarr_format=2)
        timings["open_group_s"] = time.monotonic() - t0

        # Step 4: Create arrays & update attrs
        t0 = time.monotonic()
        var_create_times = []
        var_attrs_times = []
        for name, da in dataset.data_vars.items():
            base_dims = tuple(str(d) for d in da.dims if str(d) in ("latitude", "longitude"))
            grid_shape = (len(lat), len(lon))
            if members:
                dims = ("member", "lead_time_hours") + base_dims
                shape = (len(members), len(leads)) + grid_shape
            else:
                dims = ("lead_time_hours",) + base_dims
                shape = (len(leads),) + grid_shape
            chunks = tuple(min(DEFAULT_CHUNKS.get(str(d), s), s) for d, s in zip(dims, shape))
            fill_val = "NaN" if np.issubdtype(da.dtype, np.floating) else None

            tc0 = time.monotonic()
            arr = root.create_array(
                str(name),
                shape=shape,
                chunks=chunks,
                dtype=da.dtype,
                compressor=Zstd(level=5),
                fill_value=fill_val,
                order="C",
            )
            var_create_times.append(time.monotonic() - tc0)

            ta0 = time.monotonic()
            var_attrs: dict[str, object] = {"_ARRAY_DIMENSIONS": list(dims)}
            var_attrs.update(da.attrs)
            arr.attrs.update(var_attrs)
            var_attrs_times.append(time.monotonic() - ta0)

        timings["all_vars_create_array_s"] = sum(var_create_times)
        timings["all_vars_attrs_update_s"] = sum(var_attrs_times)
        timings["step4_total_vars_s"] = time.monotonic() - t0

        # Step 5: Consolidate metadata
        t0 = time.monotonic()
        zarr.consolidate_metadata(resolved)
        timings["consolidate_metadata_s"] = time.monotonic() - t0

        timings["total_prepare_run_store_s"] = time.monotonic() - t_total_start

    return {
        "timings": timings,
        "s3_counts": dict(counter.counts),
        "num_vars": len(dataset.data_vars),
    }


def run_raw_minio_control_test(store_path: str, count: int = 100) -> dict[str, Any]:
    """Task 19: Direct MinIO small-object control test."""
    fs = get_s3_fs(settings)
    prefix = f"weather-data/control_test_{uuid.uuid4().hex[:8]}"

    data = b'{"protocol_version": 1, "state": "updating"}'

    # 1. First PUT (cold setup)
    t0 = time.monotonic()
    fs.pipe(f"{prefix}/first_put.json", data)
    first_put_dur = (time.monotonic() - t0) * 1000.0

    # 2. Sequential 100 PUTs
    t0 = time.monotonic()
    for i in range(count):
        fs.pipe(f"{prefix}/put_{i}.json", data)
    total_puts_time = time.monotonic() - t0
    avg_put_ms = (total_puts_time / count) * 1000.0

    # 3. Sequential 100 GETs
    t0 = time.monotonic()
    for i in range(count):
        _ = fs.cat(f"{prefix}/put_{i}.json")
    total_gets_time = time.monotonic() - t0
    avg_get_ms = (total_gets_time / count) * 1000.0

    # 4. Sequential 100 HEADs
    t0 = time.monotonic()
    for i in range(count):
        _ = fs.info(f"{prefix}/put_{i}.json")
    total_heads_time = time.monotonic() - t0
    avg_head_ms = (total_heads_time / count) * 1000.0

    # 5. LIST
    t0 = time.monotonic()
    _ = fs.ls(prefix)
    list_ms = (time.monotonic() - t0) * 1000.0

    # Cleanup
    try:
        fs.rm(prefix, recursive=True)
    except Exception:
        pass

    return {
        "first_put_ms": first_put_dur,
        "avg_put_ms": avg_put_ms,
        "avg_get_ms": avg_get_ms,
        "avg_head_ms": avg_head_ms,
        "list_ms": list_ms,
        "count": count,
    }


if __name__ == "__main__":
    leads = tuple(range(0, 243, 3))
    members = tuple(range(1, 31))
    ds = make_dummy_gfs_dataset()

    print("================================================================================")
    print("                    TASK 15 & 16: ZARR INIT TIMING & S3 COUNTS")
    print("================================================================================")

    # GFS Test
    gfs_store = f"s3://weather-data/benchmarks/zarr_init_gfs_{uuid.uuid4().hex[:8]}.zarr"
    gfs_res = instrumented_prepare_run_store(ds, gfs_store, expected_leads=leads)
    print("\n--- GFS (81 leads, 14 variables) ---")
    print(f"Total Prepare Run Store: {gfs_res['timings']['total_prepare_run_store_s']:.3f}s")
    for step, dur in gfs_res["timings"].items():
        if step != "total_prepare_run_store_s":
            pct = (dur / gfs_res["timings"]["total_prepare_run_store_s"]) * 100
            print(f"  {step:<26}: {dur:6.3f}s ({pct:5.1f}%)")
    print("S3 Calls:", gfs_res["s3_counts"])

    # GEFS Test
    gefs_store = f"s3://weather-data/benchmarks/zarr_init_gefs_{uuid.uuid4().hex[:8]}.zarr"
    gefs_res = instrumented_prepare_run_store(ds, gefs_store, expected_leads=leads, expected_members=members)
    print("\n--- GEFS (81 leads, 30 members, 14 variables) ---")
    print(f"Total Prepare Run Store: {gefs_res['timings']['total_prepare_run_store_s']:.3f}s")
    for step, dur in gefs_res["timings"].items():
        if step != "total_prepare_run_store_s":
            pct = (dur / gefs_res["timings"]["total_prepare_run_store_s"]) * 100
            print(f"  {step:<26}: {dur:6.3f}s ({pct:5.1f}%)")
    print("S3 Calls:", gefs_res["s3_counts"])

    print("\n================================================================================")
    print("                    TASK 17: COLD VS WARM FRESH STORE RUNS")
    print("================================================================================")
    for run_name in ["Run A", "Run B", "Run C"]:
        st = f"s3://weather-data/benchmarks/fresh_{run_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.zarr"
        res = instrumented_prepare_run_store(ds, st, expected_leads=leads)
        print(f"{run_name}: {res['timings']['total_prepare_run_store_s']:.3f}s | "
              f"coords={res['timings']['coords_to_zarr_s']:.3f}s | "
              f"vars={res['timings']['step4_total_vars_s']:.3f}s | "
              f"consolidate={res['timings']['consolidate_metadata_s']:.3f}s | "
              f"S3 calls: {sum(res['s3_counts'].values())}")

    print("\n================================================================================")
    print("                    TASK 19: RAW MINIO CONTROL BENCHMARK (100 objs)")
    print("================================================================================")
    control = run_raw_minio_control_test("s3://weather-data/control", count=100)
    print(f"First PUT (cold connection): {control['first_put_ms']:.2f} ms")
    print(f"Average PUT (100 small objs): {control['avg_put_ms']:.2f} ms")
    print(f"Average GET (100 small objs): {control['avg_get_ms']:.2f} ms")
    print(f"Average HEAD (100 small objs): {control['avg_head_ms']:.2f} ms")
    print(f"LIST (100 small objs):        {control['list_ms']:.2f} ms")
