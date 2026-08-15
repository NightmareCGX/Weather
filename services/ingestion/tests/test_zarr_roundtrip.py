"""Integration tests for Zarr write/read round-trip on local and MinIO stores."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
import xarray.testing as xrt

from ingestion.core.zarr_writer import read_dataset, write_dataset
from ingestion.providers.noaa.parser import parse_grib2


def _roundtrip(ds: xr.Dataset, store: str | dict[str, bytes]) -> xr.Dataset:
    """Write ``ds`` to ``store`` and read it back."""
    write_dataset(ds, store)
    return read_dataset(store)


def test_local_store_roundtrip_preserves_data(
    grib_fixture: Path, tmp_path: Path
) -> None:
    """The normalized fixture round-trips through a local Zarr store."""
    ds = parse_grib2(grib_fixture)
    store = str(tmp_path / "zarrstore")

    restored = _roundtrip(ds, store)

    # Values, coordinates, and attributes are identical.
    xrt.assert_identical(ds, restored)

    # The persisted Zarr chunk grid is restored on read. The fixture decodes
    # to the ``t2m`` variable (the cfgrib ``cfVarName`` for the 2 m field).
    assert restored.t2m.encoding.get("chunks") == (5, 10)

    # Compression is preserved.
    compressor = restored.t2m.encoding.get("compressor")
    assert compressor is not None
    assert compressor.codec_id == "zstd"


def test_s3_store_roundtrip_preserves_data(grib_fixture, minio_store: str) -> None:
    """The normalized fixture round-trips through a MinIO Zarr store."""
    ds = parse_grib2(grib_fixture)

    restored = _roundtrip(ds, minio_store)

    xrt.assert_identical(ds, restored)
    assert restored.t2m.encoding.get("chunks") == (5, 10)
    compressor = restored.t2m.encoding.get("compressor")
    assert compressor is not None
    assert compressor.codec_id == "zstd"


def test_api_reads_ingestion_written_store(grib_fixture: Path, tmp_path: Path) -> None:
    """Regression: the API serving tier must read a store written by ingestion.

    This is the real cross-service compatibility contract (D3). The ingestion
    writer (numcodecs 0.15.x) serializes Zstd metadata with a ``checksum`` key;
    an older API numcodecs (<0.14) cannot construct that codec and the API's
    ``xr.open_zarr`` would raise ``TypeError``. The API environment must be
    dependency-aligned so it can open the exact store the ingestion writer
    produced — not a store the API wrote itself.

    The store is written by the *ingestion* writer (never the API's test-only
    writer) and opened by the *API's* production reader path
    (``api.core.zarr.read_dataset``).
    """
    from api.core.zarr import read_dataset

    ds = parse_grib2(grib_fixture)
    store = str(tmp_path / "ingestion_written.zarr")
    write_dataset(ds, store)

    # The API production reader must open the ingestion-written store and
    # recover values, coordinates, and Zstd compressor metadata. The fixture
    # decodes to the ``t2m`` variable.
    restored = read_dataset(store)
    assert "t2m" in restored.data_vars
    assert restored.t2m.shape == ds.t2m.shape
    assert float(restored.t2m.values[0, 0]) == pytest.approx(
        float(ds.t2m.values[0, 0]), abs=1e-6
    )
    compressor = restored.t2m.encoding.get("compressor")
    assert compressor is not None
    assert compressor.codec_id == "zstd"


# --- Full-overwrite guard (Phase 2A) ---


def _tiny_ds():
    import numpy as np

    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                np.ones((1, 4, 4)),
            )
        },
        coords={
            "lead_time_hours": [6],
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_full_overwrite_guard_rejects_live_store(tmp_path) -> None:
    """A full overwrite of a store referenced by a live model_runs is rejected."""
    from datetime import datetime, timezone

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from ingestion.core.base import LiveStoreOverwriteError
    from ingestion.core.catalog import (
        CatalogBase,
        CenterRecord,
        ModelRecord,
        ModelRunRecord,
        ModelVersionRecord,
    )
    from ingestion.core.pipeline import guard_full_overwrite

    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    store = str(tmp_path / "live.zarr")
    write_dataset(_tiny_ds(), store)

    with Session(engine) as session:
        session.add(CenterRecord(id="c", center_id="noaa", name="NOAA", country="USA"))
        session.add(
            ModelRecord(
                id="m", model_id="gfs", name="GFS", center_id="noaa",
                is_ensemble=False, resolution_km=25.0,
            )
        )
        session.add(ModelVersionRecord(id="v", model_id="gfs", version_string="v1.0"))
        session.add(
            ModelRunRecord(
                id="r", model_version_id="v",
                cycle_time=datetime(2026, 8, 14, 0, tzinfo=timezone.utc),
                status="ready", zarr_store_path=store,
            )
        )
        session.commit()
        # A full overwrite of a live-run store is refused.
        with pytest.raises(LiveStoreOverwriteError):
            guard_full_overwrite(session, store)


def test_full_overwrite_guard_allows_new_store(tmp_path) -> None:
    """A full overwrite of a non-live (unreferenced) store is allowed."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from ingestion.core.catalog import CatalogBase
    from ingestion.core.pipeline import guard_full_overwrite

    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    store = str(tmp_path / "new.zarr")
    with Session(engine) as session:
        # No model_runs references this store -> allowed.
        guard_full_overwrite(session, store)
    # The write itself still works.
    write_dataset(_tiny_ds(), store)
    assert read_dataset(store) is not None
