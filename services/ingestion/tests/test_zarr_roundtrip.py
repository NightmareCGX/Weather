"""Integration tests for Zarr write/read round-trip on local and MinIO stores."""

from __future__ import annotations

from pathlib import Path

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
