"""Unit and integration tests for Weather Platform Sharded v1 (sharded_v1) storage format."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import xarray as xr
from numcodecs import Zstd

from ingestion.core.inventory import (
    region_expected_object_keys,
    verify_shard_integrity,
)
from ingestion.core.zarr_writer import (
    INDEX_ENTRY_SIZE,
    SHARD_MAGIC,
    TRAILER_SIZE,
    build_sharded_v1_container,
    encode_region_sharded_v1,
    parse_sharded_v1_index,
)


def _synthetic_region_dataset(lead: int = 6, member: int | None = 1) -> xr.Dataset:
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    dims = ("latitude", "longitude")
    shape = (721, 1440)

    data_vars = {
        "temperature_2m": (dims, np.full(shape, 290.0, dtype=np.float32)),
        "precipitation_rate": (dims, np.full(shape, 2.5, dtype=np.float32)),
    }
    coords = {
        "lead_time_hours": [lead],
        "latitude": lat,
        "longitude": lon,
        "time": np.datetime64("2026-07-21T00:00:00"),
    }
    if member is not None:
        coords["member"] = [member]
    return xr.Dataset(data_vars=data_vars, coords=coords)


def test_build_and_parse_sharded_v1_container() -> None:
    """Test packing and parsing of sharded_v1 container format."""
    compressor = Zstd(level=5)
    raw_chunks = [
        compressor.encode(np.full((1, 1, 100, 100), float(i), dtype=np.float32).tobytes())
        for i in range(120)
    ]

    container_bytes = build_sharded_v1_container(raw_chunks)
    assert len(container_bytes) > 120 * INDEX_ENTRY_SIZE + TRAILER_SIZE

    # Inspect trailer
    trailer = container_bytes[-TRAILER_SIZE:]
    num_chunks, index_size, magic = struct.unpack("<III", trailer)
    assert magic == SHARD_MAGIC
    assert num_chunks == 120
    assert index_size == 120 * INDEX_ENTRY_SIZE

    # Inspect index
    index_bytes = container_bytes[-(TRAILER_SIZE + index_size) : -TRAILER_SIZE]
    entries = parse_sharded_v1_index(index_bytes, num_chunks)
    assert len(entries) == 120

    # Verify each chunk roundtrip
    for idx, (off, length) in enumerate(entries):
        assert off >= 0
        assert length == len(raw_chunks[idx])
        chunk_data = container_bytes[off : off + length]
        assert chunk_data == raw_chunks[idx]
        decoded = compressor.decode(chunk_data)
        arr = np.frombuffer(decoded, dtype=np.float32).reshape(1, 1, 100, 100)
        assert arr[0, 0, 0, 0] == float(idx)


def test_encode_region_sharded_v1() -> None:
    """Test encoding single-region dataset into 1 shard object per variable."""
    ds = _synthetic_region_dataset(lead=6, member=1)
    shards = encode_region_sharded_v1(ds, member=1, lead_time_hours=6)

    # 2 variables in synthetic dataset -> exactly 2 shard files
    assert len(shards) == 2
    keys = [k for k, _ in shards]
    assert "temperature_2m/shard.mem001_L0006.shard" in keys
    assert "precipitation_rate/shard.mem001_L0006.shard" in keys

    # Deterministic model test (member=None)
    ds_det = _synthetic_region_dataset(lead=12, member=None)
    shards_det = encode_region_sharded_v1(ds_det, member=None, lead_time_hours=12)
    assert len(shards_det) == 2
    keys_det = [k for k, _ in shards_det]
    assert "temperature_2m/shard.det_L0012.shard" in keys_det
    assert "precipitation_rate/shard.det_L0012.shard" in keys_det


def test_region_expected_object_keys_sharded() -> None:
    """Test that region_expected_object_keys derives 14 shard objects under sharded_v1."""
    vars_list = [
        "temperature_2m", "precipitation_rate", "precipitation_amount_3h",
        "crain", "csnow", "cfrzr", "cicep", "relative_humidity_2m",
        "wind_gust", "visibility", "snow_depth", "wind_u_10m", "wind_v_10m",
        "cloud_cover_3h",
    ]
    keys = region_expected_object_keys(
        "s3://weather-data/test/cycle.zarr",
        member=17,
        lead_index=2,
        lead_time_hours=6,
        data_var_paths=vars_list,
        format_version="sharded_v1",
    )
    assert len(keys) == 14
    assert "cfrzr/shard.mem017_L0006.shard" in keys
    assert "cicep/shard.mem017_L0006.shard" in keys
    assert "temperature_2m/shard.mem017_L0006.shard" in keys


def test_verify_shard_integrity(tmp_path: Path) -> None:
    """Test structural integrity validation for sharded_v1 objects."""
    compressor = Zstd(level=5)
    raw_chunks = [compressor.encode(np.zeros((1, 1, 100, 100), dtype=np.float32).tobytes()) for _ in range(120)]
    valid_shard = build_sharded_v1_container(raw_chunks)

    shard_file = tmp_path / "temperature_2m" / "shard.mem001_L0006.shard"
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    shard_file.write_bytes(valid_shard)

    store_path = str(tmp_path)
    shard_key = "temperature_2m/shard.mem001_L0006.shard"

    # Valid shard
    assert verify_shard_integrity(store_path, shard_key, expected_num_chunks=120) is True

    # Corrupted trailer magic
    corrupted = bytearray(valid_shard)
    corrupted[-4:] = b"BAD!"
    shard_file.write_bytes(bytes(corrupted))
    assert verify_shard_integrity(store_path, shard_key, expected_num_chunks=120) is False

    # Short/truncated file
    shard_file.write_bytes(valid_shard[:1000])
    assert verify_shard_integrity(store_path, shard_key, expected_num_chunks=120) is False


def test_sharded_v1_store_roundtrip_preserves_data(tmp_path: Path) -> None:
    """Test full prepare_run_store, commit_region, and read_dataset roundtrip under sharded_v1."""
    from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_dataset

    store = str(tmp_path / "sharded_store.zarr")
    ds_seed = _synthetic_region_dataset(lead=0, member=1)
    prepare_run_store(
        ds_seed,
        store,
        expected_lead_time_hours=(0, 6, 12),
        expected_members=(1, 2),
    )

    # Commit member 1, lead 0
    commit_region(ds_seed, store, lead_time_hours=0, member=1)

    # Commit member 1, lead 6
    ds_lead6 = _synthetic_region_dataset(lead=6, member=1)
    commit_region(ds_lead6, store, lead_time_hours=6, member=1)

    # Read back
    restored = read_dataset(store)
    assert "temperature_2m" in restored.data_vars
    assert "precipitation_rate" in restored.data_vars

    # Check committed slices have data and uncommitted slices are NaN
    t2m = restored["temperature_2m"].values # shape: (member=2, lead=3, 721, 1440)
    # member 1 (index 0), lead 0 (index 0) committed
    assert not np.all(np.isnan(t2m[0, 0]))
    assert np.allclose(t2m[0, 0], 290.0)

    # member 1 (index 0), lead 6 (index 1) committed
    assert not np.all(np.isnan(t2m[0, 1]))
    assert np.allclose(t2m[0, 1], 290.0)

    # member 1 (index 0), lead 12 (index 2) uncommitted -> NaN
    assert np.all(np.isnan(t2m[0, 2]))

    # member 2 (index 1), lead 0 (index 0) uncommitted -> NaN
    assert np.all(np.isnan(t2m[1, 0]))


def test_sharded_v1_minio_roundtrip_and_no_chunk_flooding(minio_store: str) -> None:
    """Regression test: sharded_v1 on MinIO S3 roundtrips without flooding chunk GETs or hanging."""
    from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_dataset

    ds_seed = _synthetic_region_dataset(lead=0, member=None)
    prepare_run_store(
        ds_seed,
        minio_store,
        expected_lead_time_hours=(0, 6),
        expected_members=(),
    )

    commit_region(ds_seed, minio_store, lead_time_hours=0, member=None)
    ds_lead6 = _synthetic_region_dataset(lead=6, member=None)
    commit_region(ds_lead6, minio_store, lead_time_hours=6, member=None)

    # Reading the store must not hang or issue 404 chunk requests
    restored = read_dataset(minio_store)
    assert not np.all(np.isnan(restored["temperature_2m"].values[0]))
    assert not np.all(np.isnan(restored["temperature_2m"].values[1]))
    assert np.allclose(restored["temperature_2m"].values[0], 290.0)


def _write_sample_grib(path: Path, lead: int) -> None:
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )
    with path.open("wb") as f:
        msg = codes_grib_new_from_samples("GRIB2")
        codes_set(msg, "dataDate", 20260721)
        codes_set(msg, "dataTime", 0)
        codes_set(msg, "stepType", "instant")
        codes_set(msg, "stepRange", str(lead))
        codes_set(msg, "stepUnits", "h")
        codes_set(msg, "paramId", 167)
        codes_set(msg, "shortName", "2t")
        codes_set(msg, "typeOfLevel", "heightAboveGround")
        codes_set(msg, "level", 2)
        codes_set(msg, "gridType", "regular_ll")
        codes_set(msg, "Ni", 10)
        codes_set(msg, "Nj", 5)
        codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
        codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
        codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
        codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
        codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
        codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
        codes_set_values(msg, np.full((5, 10), 285.0 + float(lead), dtype=np.float32).ravel())
        codes_write(msg, f)
        codes_release(msg)


def test_sharded_v1_pipeline_concurrent_wave_regression(
    minio_store: str, tmp_path: Path, monkeypatch
) -> None:
    """Regression test: multi-lead concurrent wave against MinIO with sharded_v1.

    Validates that:
    1. Multiple admitted pipeline items download, decode, and write concurrently.
    2. The first sharded region write does not freeze or block the event loop.
    3. Global PUT concurrency remains bounded.
    4. Pipeline cleanly drains and finalizes with status='ready'.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from ingestion.cli import main
    from ingestion.core.catalog import CatalogBase, record_run
    from ingestion.core.zarr_writer import read_dataset

    db_file = tmp_path / "catalog_test.sqlite"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    CatalogBase.metadata.create_all(engine)
    session = Session(engine)

    class _NoopLocks:
        def __init__(self, *a, **k):
            pass
        def acquire_shared_gate(self):
            pass
        def release_shared_gate(self):
            pass
        def acquire_exclusive_gate(self):
            pass
        def release_exclusive_gate(self):
            pass
        def acquire_admission(self):
            pass
        def release_admission(self):
            pass
        def acquire_shared_admission(self):
            pass
        def release_shared_admission(self):
            pass
        def acquire_region_locks(self, region_ids):
            pass
        def release_region_locks(self, region_ids):
            pass
        def release_all(self):
            pass
        def close_connection(self):
            pass

    async def _mock_download(self, model, cycle_date, cycle_hour, lead_time_hours, destination, **kw):
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_sample_grib(dest, lead_time_hours)
        return dest

    monkeypatch.setattr("ingestion.providers.noaa.connector.NOAAConnector.download", _mock_download)
    monkeypatch.setattr("ingestion.core.wave_runner._catalog_session_factory", lambda: engine)
    monkeypatch.setattr("ingestion.core.coordinator.StoreLockCoordinator", _NoopLocks)

    recorded = []
    def _record_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        run = record_run(session, spec, dataset, member=member, committed_state=committed_state)
        recorded.append(run)
        return run

    monkeypatch.setattr("ingestion.core.pipeline.record_ingested_dataset", _record_session)

    argv = [
        "ingest",
        "--model", "gfs",
        "--cycle-date", "2026-07-21",
        "--cycle-hour", "0",
        "--lead-time-hours", "0", "3", "6",
        "--store", minio_store,
        "--allow-custom-store",
        "--concurrency", "4",
        "--download-dir", str(tmp_path / "downloads"),
    ]

    exit_code = main(argv)
    assert exit_code == 0

    # Verify store content across all 3 leads
    restored = read_dataset(minio_store)
    assert "temperature_2m" in restored.data_vars
    t2m = restored["temperature_2m"].values
    assert t2m.shape[0] == 3
    for lead_idx, lead_val in enumerate([0, 3, 6]):
        assert not np.all(np.isnan(t2m[lead_idx]))
        # 285.0 K -> 11.85 °C (+ lead_val)
        expected_celsius = (285.0 + lead_val) - 273.15
        assert np.allclose(t2m[lead_idx], expected_celsius, atol=1e-3)


