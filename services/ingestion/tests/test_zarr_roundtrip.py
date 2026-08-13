"""Integration tests for Zarr write/read round-trip on local and MinIO stores."""

from __future__ import annotations

from pathlib import Path

import glob
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

    assert not glob.glob(store + ".staging*")
    assert not os.path.exists(zw._old_path(store))
    restored = read_dataset(store)
    assert int(restored["lead_time_hours"].values) == 6


def test_write_dataset_atomic_reingest_replaces_cleanly(tmp_path: Path) -> None:
    """Re-ingesting replaces the store and leaves no staging/old residue."""
    import ingestion.core.zarr_writer as zw

    store = str(tmp_path / "cycle.zarr")
    write_dataset(_synthetic(6), store)
    write_dataset(_synthetic(12), store)

    assert not glob.glob(store + ".staging*")
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
    assert not glob.glob(store + ".staging*")
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



def test_write_dataset_swap_failure_cleans_staging(tmp_path, monkeypatch) -> None:
    """A failed promote swap raises and leaves no staging residue behind.

    Regression for MAJOR-2: the swap step was previously outside the
    cleanup try/except, so a failed second rename left the full staging
    copy on disk.
    """
    import glob
    import os

    import numpy as np
    import xarray as xr

    from ingestion.core.zarr_writer import read_dataset, write_dataset

    store = str(tmp_path / "cycle.zarr")

    def _dataset(lead: int) -> xr.Dataset:
        return xr.Dataset(
            {
                "temperature_2m": (
                    ("lead_time_hours", "latitude", "longitude"),
                    np.full((1, 2, 2), float(lead)),
                )
            },
            coords={"lead_time_hours": [lead], "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
        )

    write_dataset(_dataset(0), store)

    real_rename = os.rename
    calls = {"count": 0}

    def _flaky_rename(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated second-rename failure")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", _flaky_rename)

    with pytest.raises(OSError):
        write_dataset(_dataset(6), store)

    # Previous store intact, no staging residue.
    restored = read_dataset(store)
    assert restored["lead_time_hours"].values.tolist() == [0]
    assert not glob.glob(store + ".staging*")


def test_store_status_mapping_branch(tmp_path) -> None:
    """store_status classifies mutable mappings without a full listing.

    Regression for the mapping branch: an empty mapping is missing, a
    mapping with a broken .zgroup is corrupt, and a written mapping is
    readable.
    """
    import numpy as np
    import xarray as xr

    from ingestion.core.zarr_writer import store_status, write_dataset

    assert store_status({}) == "missing"
    assert store_status({".zgroup": b"{not-valid-json"}) == "corrupt"

    dataset = xr.Dataset(
        {"temperature_2m": (("latitude", "longitude"), np.array([[1.0, 2.0]]))},
        coords={"latitude": [0.0], "longitude": [0.0, 1.0]},
    )
    store: dict[str, bytes] = {}
    write_dataset(dataset, store)
    assert store_status(store) == "readable"


def test_s3_advisory_lock_acquire_release_and_stale_break(monkeypatch) -> None:
    """The S3 advisory lock protocol claims, verifies, and releases.

    Regression for MAJOR-1 (S3): the lock object is written with a writer
    token, re-read to verify the claim, removed on release, and a stale
    lock is reclaimed.
    """
    import json
    import time
    from io import BytesIO

    from ingestion.core.zarr_writer import _s3_advisory_lock

    class _FakeWriter:
        def __init__(self, fs, key):
            self._fs = fs
            self._key = key
            self._buf = BytesIO()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                self._fs.objects[self._key] = self._buf.getvalue()
            return False

        def write(self, data):
            self._buf.write(data)

    class _FakeS3:
        def __init__(self):
            self.objects: dict[str, bytes] = {}

        def open(self, key, mode):
            if "r" in mode:
                if key not in self.objects:
                    raise FileNotFoundError(key)
                return BytesIO(self.objects[key])
            return _FakeWriter(self, key)

        def rm(self, key, recursive=False):
            self.objects.pop(key, None)

        def glob(self, pattern):
            return []

        def exists(self, key):
            return key in self.objects

    fs = _FakeS3()
    with _s3_advisory_lock(fs, "bucket/cycle.zarr"):
        payload = json.loads(fs.objects["bucket/cycle.zarr.lock"])
        assert "token" in payload
    # Released: the lock object is removed.
    assert "bucket/cycle.zarr.lock" not in fs.objects

    # A stale lock (past TTL) is reclaimed.
    fs.objects["bucket/cycle.zarr.lock"] = json.dumps(
        {"token": "other", "t": time.time() - 100000}
    ).encode("utf-8")
    with _s3_advisory_lock(fs, "bucket/cycle.zarr"):
        payload = json.loads(fs.objects["bucket/cycle.zarr.lock"])
        assert payload["token"] != "other"
    assert "bucket/cycle.zarr.lock" not in fs.objects

    # An active foreign lock is not stolen: the acquirer keeps waiting, so
    # simulate the waiter giving up via a time.sleep patch that raises.
    fs.objects["bucket/cycle.zarr.lock"] = json.dumps(
        {"token": "other", "t": time.time()}
    ).encode("utf-8")

    def _give_up(_seconds):
        raise TimeoutError("waiter gave up")

    monkeypatch.setattr(time, "sleep", _give_up)
    with pytest.raises(TimeoutError):
        with _s3_advisory_lock(fs, "bucket/cycle.zarr"):
            pass  # pragma: no cover - unreachable in the test
    # The foreign lock object was not clobbered by the waiter.
    payload = json.loads(fs.objects["bucket/cycle.zarr.lock"])
    assert payload["token"] == "other"
