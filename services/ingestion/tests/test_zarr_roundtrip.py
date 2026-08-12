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

    # The persisted Zarr chunk grid is restored on read.
    assert restored.t.encoding.get("chunks") == (5, 10)

    # Compression is preserved.
    compressor = restored.t.encoding.get("compressor")
    assert compressor is not None
    assert compressor.codec_id == "zstd"


def test_s3_store_roundtrip_preserves_data(grib_fixture, minio_store: str) -> None:
    """The normalized fixture round-trips through a MinIO Zarr store."""
    ds = parse_grib2(grib_fixture)

    restored = _roundtrip(ds, minio_store)

    xrt.assert_identical(ds, restored)
    assert restored.t.encoding.get("chunks") == (5, 10)
    compressor = restored.t.encoding.get("compressor")
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
    # recover values, coordinates, and Zstd compressor metadata.
    restored = read_dataset(store)
    assert "t" in restored.data_vars
    assert restored.t.shape == ds.t.shape
    assert float(restored.t.values[0, 0]) == pytest.approx(
        float(ds.t.values[0, 0]), abs=1e-6
    )
    compressor = restored.t.encoding.get("compressor")
    assert compressor is not None
    assert compressor.codec_id == "zstd"
