"""Integration tests for Zarr write/read round-trip on local and MinIO stores."""

from __future__ import annotations

from pathlib import Path

import os

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


def _synthetic(lead: int) -> xr.Dataset:
    """A minimal 2-D dataset for atomic-write tests without a GRIB fixture."""
    import numpy as np

    return xr.Dataset(
        {"temperature_2m": (("latitude", "longitude"), np.full((2, 2), float(lead)))},
        coords={
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
            "lead_time_hours": lead,
        },
    )


def test_store_exists_missing_path_is_false(tmp_path: Path) -> None:
    """A non-existent store reports absent."""
    from ingestion.core.zarr_writer import store_exists

    assert store_exists(str(tmp_path / "absent.zarr")) is False


def test_store_exists_valid_store_is_true(grib_fixture: Path, tmp_path: Path) -> None:
    """A readable written store reports present."""
    from ingestion.core.zarr_writer import store_exists

    ds = parse_grib2(grib_fixture)
    store = str(tmp_path / "present.zarr")
    write_dataset(ds, store)
    assert store_exists(store) is True


def test_store_exists_corrupt_directory_is_false(tmp_path: Path) -> None:
    """A corrupt/half-written store is treated as absent (L2-M3).

    Regression: the local-path branch only checked ``os.path.exists``, so a
    corrupt or half-written directory reported present and could be reused.
    The deep check must fail to consider it readable.
    """
    from ingestion.core.zarr_writer import store_exists

    corrupt = tmp_path / "corrupt.zarr"
    corrupt.mkdir()
    (corrupt / ".zgroup").write_text("{not valid json")
    assert os.path.exists(corrupt)
    assert store_exists(str(corrupt)) is False


def test_write_dataset_atomic_leaves_no_staging_or_old(tmp_path: Path) -> None:
    """A successful atomic local write leaves no ``.staging``/``.old`` residue."""
    import ingestion.core.zarr_writer as zw

    store = str(tmp_path / "cycle.zarr")
    write_dataset(_synthetic(6), store)

    assert not os.path.exists(zw._staging_path(store))
    assert not os.path.exists(zw._old_path(store))
    restored = read_dataset(store)
    assert int(restored["lead_time_hours"].values) == 6


def test_write_dataset_atomic_reingest_replaces_cleanly(tmp_path: Path) -> None:
    """Re-ingesting replaces the store and leaves no staging/old residue."""
    import ingestion.core.zarr_writer as zw

    store = str(tmp_path / "cycle.zarr")
    write_dataset(_synthetic(6), store)
    write_dataset(_synthetic(12), store)

    assert not os.path.exists(zw._staging_path(store))
    assert not os.path.exists(zw._old_path(store))
    restored = read_dataset(store)
    assert int(restored["lead_time_hours"].values) == 12


def test_write_dataset_failure_keeps_previous_store(monkeypatch, tmp_path: Path) -> None:
    """A mid-write failure leaves the previous store intact and no ``.staging``.

    Regression for L2-C1: a crash while writing the staging store must not
    truncate or corrupt the previously-served store at the final path.
    """
    import ingestion.core.zarr_writer as zw

    store = str(tmp_path / "cycle.zarr")
    write_dataset(_synthetic(6), store)

    def _boom(dataset, resolved, chunks):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(zw, "_write_zarr", _boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        write_dataset(_synthetic(12), store)

    # Previous store intact and readable, staging cleaned up.
    assert not os.path.exists(zw._staging_path(store))
    restored = read_dataset(store)
    assert int(restored["lead_time_hours"].values) == 6


def test_write_dataset_mutable_mapping_written_in_place(tmp_path) -> None:
    """A dict store target receives the written bytes in place.

    Regression for MAJOR-4: copying the mapping with dict(store) would write
    to a throwaway copy and silently drop the data.
    """
    import numpy as np
    import xarray as xr

    from ingestion.core.zarr_writer import read_dataset, write_dataset

    dataset = xr.Dataset(
        {"temperature_2m": (("latitude", "longitude"), np.array([[1.0, 2.0]]))},
        coords={"latitude": [0.0], "longitude": [0.0, 1.0]},
    )
    store: dict[str, bytes] = {}
    write_dataset(dataset, store)
    assert len(store) > 0
    restored = read_dataset(store)
    assert "temperature_2m" in restored.data_vars


def test_store_status_classifies_stores(tmp_path) -> None:
    """store_status distinguishes missing, readable, and corrupt targets.

    Regression for MAJOR-2: a corrupt directory must be classified as
    corrupt (not missing) so ingestion refuses to silently rebuild it.
    """
    import os

    import numpy as np
    import xarray as xr

    from ingestion.core.zarr_writer import store_status, write_dataset

    dataset = xr.Dataset(
        {"temperature_2m": (("latitude", "longitude"), np.array([[1.0, 2.0]]))},
        coords={"latitude": [0.0], "longitude": [0.0, 1.0]},
    )

    missing = str(tmp_path / "missing.zarr")
    assert store_status(missing) == "missing"

    readable = str(tmp_path / "ok.zarr")
    write_dataset(dataset, readable)
    assert store_status(readable) == "readable"

    corrupt = str(tmp_path / "bad.zarr")
    os.makedirs(corrupt)
    with open(os.path.join(corrupt, ".zgroup"), "w") as handle:
        handle.write("{not-valid-json")
    assert store_status(corrupt) == "corrupt"
