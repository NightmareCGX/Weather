"""Focused regression tests for P1-A: Zarr Cold-Start Initialization Optimization.

Validates:
* Test A — No full logical forecast array allocation during initialization.
* Test B — No forecast data chunks created during prepare_run_store.
* Test C — No delete storm (0 DELETE operations on nonexistent data chunks).
* Test D — Schema equivalence (shape, chunks, dtype, compressor, attrs, coords).
* Test E — xarray readback before data write (missing chunks return fill value).
* Test F — Region write after direct initialization (GFS and GEFS).
* Test G — Same-cycle re-ingestion behavior.
* Test H — Consolidated metadata (.zmetadata) generation and compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from ingestion.core.coordinator import RunCoordinator
from ingestion.core.zarr_writer import (
    DEFAULT_CHUNKS,
    commit_region,
    prepare_run_store,
    read_dataset,
)

GRID_LAT = np.array([38.0, 38.25, 38.5, 38.75], dtype=np.float64)
GRID_LON = np.array([-107.0, -106.75, -106.5, -106.25], dtype=np.float64)
GFS_LEADS = (0, 6, 12, 18, 24)
GEFS_MEMBERS = tuple(range(1, 31))
GEFS_LEADS = (0, 6, 12, 18, 24)


def _make_gfs_seed() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("latitude", "longitude"),
                np.full((len(GRID_LAT), len(GRID_LON)), 285.5, dtype=np.float32),
                {"units": "K", "long_name": "2 metre temperature"},
            ),
            "relative_humidity": (
                ("latitude", "longitude"),
                np.full((len(GRID_LAT), len(GRID_LON)), 65.0, dtype=np.float32),
                {"units": "%", "long_name": "2 metre relative humidity"},
            ),
        },
        coords={
            "latitude": ("latitude", GRID_LAT, {"units": "degrees_north"}),
            "longitude": ("longitude", GRID_LON, {"units": "degrees_east"}),
            "lead_time_hours": ("lead_time_hours", [6]),
            "time": ("time", [np.datetime64("2026-07-22T00:00:00", "ns")]),
        },
        attrs={"model_id": "gfs", "cycle_time": "2026-07-22T00:00:00"},
    )


def _make_gefs_seed() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("latitude", "longitude"),
                np.full((len(GRID_LAT), len(GRID_LON)), 285.5, dtype=np.float32),
                {"units": "K", "long_name": "2 metre temperature"},
            ),
        },
        coords={
            "latitude": ("latitude", GRID_LAT),
            "longitude": ("longitude", GRID_LON),
            "lead_time_hours": ("lead_time_hours", [6]),
            "member": ("member", [1]),
            "time": ("time", [np.datetime64("2026-07-22T00:00:00", "ns")]),
        },
        attrs={"model_id": "gefs", "cycle_time": "2026-07-22T00:00:00"},
    )


class TrackingStore(dict[str, bytes]):
    """A dictionary-backed Zarr store that records deletions and operations."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.deletions: list[str] = []
        self.writes: list[str] = []

    def __setitem__(self, key: str, value: bytes) -> None:
        self.writes.append(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self.deletions.append(key)
        super().__delitem__(key)


# ==============================================================================
# Test A — No Full Logical Forecast Allocation
# ==============================================================================


def test_a_no_full_logical_forecast_allocation(tmp_path: Path) -> None:
    """Prove that prepare_run_store never allocates full-grid forecast data arrays."""
    seed = _make_gefs_seed()

    allocated_shapes: list[tuple[int, ...]] = []
    orig_full = np.full
    orig_zeros = np.zeros
    orig_empty = np.empty

    def spy_full(shape, fill_value, *args, **kwargs):
        if isinstance(shape, (tuple, list)):
            allocated_shapes.append(tuple(shape))
        return orig_full(shape, fill_value, *args, **kwargs)

    def spy_zeros(shape, *args, **kwargs):
        if isinstance(shape, (tuple, list)):
            allocated_shapes.append(tuple(shape))
        return orig_zeros(shape, *args, **kwargs)

    def spy_empty(shape, *args, **kwargs):
        if isinstance(shape, (tuple, list)):
            allocated_shapes.append(tuple(shape))
        return orig_empty(shape, *args, **kwargs)

    store = str(tmp_path / "gefs_alloc.zarr")

    with patch("numpy.full", side_effect=spy_full), \
         patch("numpy.zeros", side_effect=spy_zeros), \
         patch("numpy.empty", side_effect=spy_empty):
        prepare_run_store(
            seed,
            store,
            expected_lead_time_hours=GEFS_LEADS,
            expected_members=GEFS_MEMBERS,
        )

    # The full logical forecast shape would be (30, 5, 4, 4)
    full_logical_shape = (len(GEFS_MEMBERS), len(GEFS_LEADS), len(GRID_LAT), len(GRID_LON))
    assert full_logical_shape not in allocated_shapes, (
        f"Full forecast cube {full_logical_shape} was allocated during prepare_run_store"
    )
    # Ensure no large forecast data array of any sort was allocated
    for shape in allocated_shapes:
        assert np.prod(shape) < np.prod(full_logical_shape), (
            f"An array with shape {shape} (size {np.prod(shape)}) was allocated"
        )


# ==============================================================================
# Test B — No Data Chunk Creation During Initialization
# ==============================================================================


def test_b_no_data_chunks_created_gfs(tmp_path: Path) -> None:
    """GFS fresh store has zero forecast data chunks after prepare_run_store."""
    seed = _make_gfs_seed()
    store_dir = tmp_path / "gfs_chunks.zarr"
    prepare_run_store(seed, str(store_dir), expected_lead_time_hours=GFS_LEADS)

    # Check files created
    all_files = [p.relative_to(store_dir).as_posix() for p in store_dir.rglob("*") if p.is_file()]

    # Coordinates have data chunks (e.g. latitude/0, longitude/0, lead_time_hours/0)
    coord_chunks = [f for f in all_files if any(f.startswith(f"{c}/") for c in ("latitude", "longitude", "lead_time_hours")) and not f.endswith((".zarray", ".zattrs"))]
    assert len(coord_chunks) > 0, "Coordinate chunks should exist"

    # Data variables must have ZERO chunk files (only .zarray and .zattrs)
    for var in ("temperature_2m", "relative_humidity"):
        var_chunks = [f for f in all_files if f.startswith(f"{var}/") and not f.endswith((".zarray", ".zattrs"))]
        assert len(var_chunks) == 0, f"Found unexpected data chunks for {var}: {var_chunks}"


def test_b_no_data_chunks_created_gefs(tmp_path: Path) -> None:
    """GEFS fresh store has zero forecast data chunks after prepare_run_store."""
    seed = _make_gefs_seed()
    store_dir = tmp_path / "gefs_chunks.zarr"
    prepare_run_store(seed, str(store_dir), expected_lead_time_hours=GEFS_LEADS, expected_members=GEFS_MEMBERS)

    all_files = [p.relative_to(store_dir).as_posix() for p in store_dir.rglob("*") if p.is_file()]

    # Data variables must have ZERO chunk files
    var_chunks = [f for f in all_files if f.startswith("temperature_2m/") and not f.endswith((".zarray", ".zattrs"))]
    assert len(var_chunks) == 0, f"Found unexpected data chunks for GEFS: {var_chunks}"


# ==============================================================================
# Test C — No Delete Storm
# ==============================================================================


def test_c_no_delete_storm() -> None:
    """Prove initialization executes 0 DELETE operations on forecast chunk keys."""
    seed = _make_gefs_seed()
    store = TrackingStore()

    prepare_run_store(
        seed,
        store,
        expected_lead_time_hours=GEFS_LEADS,
        expected_members=GEFS_MEMBERS,
    )

    # In the old implementation, write_empty_chunks=False caused zarr to delete
    # every empty chunk key (members * leads * spatial_chunks = 30 * 5 * 1 = 150 deletes).
    # In the optimized implementation, deletions must be 0.
    data_chunk_deletes = [k for k in store.deletions if k.startswith("temperature_2m/") and not k.endswith((".zarray", ".zattrs"))]
    assert len(data_chunk_deletes) == 0, (
        f"Expected 0 data chunk DELETE operations, got {len(data_chunk_deletes)}: {data_chunk_deletes}"
    )
    assert len(store.deletions) == 0, f"Expected 0 total deletions, got {store.deletions}"


# ==============================================================================
# Test D — Schema Equivalence
# ==============================================================================


def test_d_schema_equivalence_gfs(tmp_path: Path) -> None:
    """Validate GFS schema metadata (shape, chunks, dtype, compressor, attrs, dimensions)."""
    seed = _make_gfs_seed()
    store_dir = tmp_path / "gfs_schema.zarr"
    prepare_run_store(seed, str(store_dir), expected_lead_time_hours=GFS_LEADS)

    # 1. Root metadata
    zgroup = json.loads((store_dir / ".zgroup").read_text(encoding="utf-8"))
    assert zgroup == {"zarr_format": 2}
    zattrs = json.loads((store_dir / ".zattrs").read_text(encoding="utf-8"))
    assert zattrs["model_id"] == "gfs"
    assert zattrs["cycle_time"] == "2026-07-22T00:00:00"

    # 2. Coordinates
    int_dtype_str = np.dtype(int).str
    for coord, exp_shape, exp_dtype in [
        ("latitude", [len(GRID_LAT)], "<f8"),
        ("longitude", [len(GRID_LON)], "<f8"),
        ("lead_time_hours", [len(GFS_LEADS)], int_dtype_str),
    ]:
        za = json.loads((store_dir / coord / ".zarray").read_text(encoding="utf-8"))
        assert za["shape"] == exp_shape
        assert za["chunks"] == exp_shape
        assert za["dtype"] == exp_dtype
        assert za["zarr_format"] == 2
        zat = json.loads((store_dir / coord / ".zattrs").read_text(encoding="utf-8"))
        assert zat["_ARRAY_DIMENSIONS"] == [coord]

    # 3. Forecast Data Variables
    for var, units, long_name in [
        ("temperature_2m", "K", "2 metre temperature"),
        ("relative_humidity", "%", "2 metre relative humidity"),
    ]:
        za = json.loads((store_dir / var / ".zarray").read_text(encoding="utf-8"))
        assert za["shape"] == [len(GFS_LEADS), len(GRID_LAT), len(GRID_LON)]
        assert za["chunks"] == [1, min(DEFAULT_CHUNKS["latitude"], len(GRID_LAT)), min(DEFAULT_CHUNKS["longitude"], len(GRID_LON))]
        assert za["dtype"] == "<f4"
        assert za["fill_value"] == "NaN"
        assert za["compressor"]["id"] == "zstd"
        assert za["compressor"]["level"] == 5
        assert za["order"] == "C"

        zat = json.loads((store_dir / var / ".zattrs").read_text(encoding="utf-8"))
        assert zat["_ARRAY_DIMENSIONS"] == ["lead_time_hours", "latitude", "longitude"]
        assert zat["units"] == units
        assert zat["long_name"] == long_name


def test_d_schema_equivalence_gefs(tmp_path: Path) -> None:
    """Validate GEFS ensemble schema metadata."""
    seed = _make_gefs_seed()
    store_dir = tmp_path / "gefs_schema.zarr"
    prepare_run_store(seed, str(store_dir), expected_lead_time_hours=GEFS_LEADS, expected_members=GEFS_MEMBERS)

    # Member coordinate
    za_m = json.loads((store_dir / "member" / ".zarray").read_text(encoding="utf-8"))
    assert za_m["shape"] == [len(GEFS_MEMBERS)]
    assert za_m["dtype"] == np.dtype(int).str
    zat_m = json.loads((store_dir / "member" / ".zattrs").read_text(encoding="utf-8"))
    assert zat_m["_ARRAY_DIMENSIONS"] == ["member"]

    # Variable
    za = json.loads((store_dir / "temperature_2m" / ".zarray").read_text(encoding="utf-8"))
    assert za["shape"] == [len(GEFS_MEMBERS), len(GEFS_LEADS), len(GRID_LAT), len(GRID_LON)]
    assert za["chunks"] == [1, 1, min(DEFAULT_CHUNKS["latitude"], len(GRID_LAT)), min(DEFAULT_CHUNKS["longitude"], len(GRID_LON))]
    assert za["dtype"] == "<f4"
    assert za["fill_value"] == "NaN"
    assert za["compressor"]["id"] == "zstd"

    zat = json.loads((store_dir / "temperature_2m" / ".zattrs").read_text(encoding="utf-8"))
    assert zat["_ARRAY_DIMENSIONS"] == ["member", "lead_time_hours", "latitude", "longitude"]


# ==============================================================================
# Test E — xarray Readback Before Data Write
# ==============================================================================


def test_e_xarray_readback_before_data_write(tmp_path: Path) -> None:
    """Open directly initialized store with xarray and verify all missing chunks read as fill value."""
    seed = _make_gfs_seed()
    store_dir = str(tmp_path / "gfs_readback.zarr")
    prepare_run_store(seed, store_dir, expected_lead_time_hours=GFS_LEADS)

    # Read back with production reader
    ds = read_dataset(store_dir)

    # Validate coordinates
    np.testing.assert_array_equal(ds.coords["lead_time_hours"].values, np.array(GFS_LEADS))
    np.testing.assert_array_equal(ds.coords["latitude"].values, GRID_LAT)
    np.testing.assert_array_equal(ds.coords["longitude"].values, GRID_LON)

    # Validate data variables exist and return NaNs
    assert "temperature_2m" in ds.data_vars
    assert "relative_humidity" in ds.data_vars
    assert ds.temperature_2m.shape == (len(GFS_LEADS), len(GRID_LAT), len(GRID_LON))

    # All values must be NaN before data write
    assert np.isnan(ds.temperature_2m.values).all()
    assert np.isnan(ds.relative_humidity.values).all()


# ==============================================================================
# Test F — Region Write After Direct Initialization
# ==============================================================================


def test_f_region_write_after_direct_initialization_gfs(tmp_path: Path) -> None:
    """Write a GFS region into directly initialized store and verify values and physical chunks."""
    seed = _make_gfs_seed()
    store_dir = tmp_path / "gfs_regwrite.zarr"
    store_path = str(store_dir)
    prepare_run_store(seed, store_path, expected_lead_time_hours=GFS_LEADS)

    # Write lead 12 (positional index 2)
    lead_12_ds = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                np.full((1, len(GRID_LAT), len(GRID_LON)), 290.0, dtype=np.float32),
            ),
            "relative_humidity": (
                ("lead_time_hours", "latitude", "longitude"),
                np.full((1, len(GRID_LAT), len(GRID_LON)), 75.0, dtype=np.float32),
            ),
        },
        coords={
            "lead_time_hours": [12],
            "latitude": GRID_LAT,
            "longitude": GRID_LON,
        },
    )

    commit_region(lead_12_ds, store_path, lead_time_hours=12)

    # Verify physical chunks / shards: lead 12 shard or chunk should exist
    t2m_files = [p.name for p in (store_dir / "temperature_2m").glob("*") if not p.name.startswith(".")]
    assert "shard.det_L0012.shard" in t2m_files or "2.0.0" in t2m_files, f"Expected shard or chunk for lead 12, found {t2m_files}"

    # Read back and verify values
    ds = read_dataset(store_path)
    # Lead index 0 (lead 0), 1 (lead 6), 3 (lead 18), 4 (lead 24) should still be NaN
    assert np.isnan(ds.temperature_2m.sel(lead_time_hours=0).values).all()
    assert np.isnan(ds.temperature_2m.sel(lead_time_hours=6).values).all()
    assert np.isnan(ds.temperature_2m.sel(lead_time_hours=18).values).all()

    # Lead 12 must have written values
    np.testing.assert_allclose(
        ds.temperature_2m.sel(lead_time_hours=12).values,
        np.full((len(GRID_LAT), len(GRID_LON)), 290.0, dtype=np.float32),
    )
    np.testing.assert_allclose(
        ds.relative_humidity.sel(lead_time_hours=12).values,
        np.full((len(GRID_LAT), len(GRID_LON)), 75.0, dtype=np.float32),
    )


def test_f_region_write_after_direct_initialization_gefs(tmp_path: Path) -> None:
    """Write a GEFS member-lead slice into directly initialized store and verify isolation."""
    seed = _make_gefs_seed()
    store_dir = tmp_path / "gefs_regwrite.zarr"
    store_path = str(store_dir)
    prepare_run_store(seed, store_path, expected_lead_time_hours=GEFS_LEADS, expected_members=GEFS_MEMBERS)

    # Write member 5, lead 18 (member index 4, lead index 3)
    slice_ds = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                np.full((1, 1, len(GRID_LAT), len(GRID_LON)), 273.15, dtype=np.float32),
            )
        },
        coords={
            "member": [5],
            "lead_time_hours": [18],
            "latitude": GRID_LAT,
            "longitude": GRID_LON,
        },
    )

    commit_region(slice_ds, store_path, lead_time_hours=18, member=5)

    # Verify physical chunks / shards
    t2m_files = [p.name for p in (store_dir / "temperature_2m").glob("*") if not p.name.startswith(".")]
    assert "shard.mem005_L0018.shard" in t2m_files or "4.3.0.0" in t2m_files, f"Expected shard or chunk for member 5 lead 18, found {t2m_files}"

    # Read back and verify values
    ds = read_dataset(store_path)
    # Target slice is written
    np.testing.assert_allclose(
        ds.temperature_2m.sel(member=5, lead_time_hours=18).values,
        np.full((len(GRID_LAT), len(GRID_LON)), 273.15, dtype=np.float32),
    )
    # Other member (e.g. member 1) at lead 18 is NaN
    assert np.isnan(ds.temperature_2m.sel(member=1, lead_time_hours=18).values).all()
    # Same member at other lead (e.g. member 5, lead 0) is NaN
    assert np.isnan(ds.temperature_2m.sel(member=5, lead_time_hours=0).values).all()


# ==============================================================================
# Test G — Same-Cycle Re-Ingestion
# ==============================================================================


def test_g_same_cycle_reingestion(tmp_path: Path) -> None:
    """Same-cycle re-ingestion does not clobber existing store or re-run destructive initialization."""
    from unittest.mock import MagicMock
    from ingestion.core.catalog import RunCatalogSpec, VariableSpec

    seed = _make_gfs_seed()
    store_dir = tmp_path / "gfs_reingest.zarr"
    store_path = str(store_dir)

    # Initial preparation
    prepare_run_store(seed, store_path, expected_lead_time_hours=GFS_LEADS)

    # Write lead 6
    lead_6_ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.full((1, 4, 4), 300.0, dtype=np.float32))},
        coords={"lead_time_hours": [6], "latitude": GRID_LAT, "longitude": GRID_LON},
    )
    commit_region(lead_6_ds, store_path, lead_time_hours=6)

    # Setup RunCoordinator with existing store
    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gfs",
        model_name="GFS",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=seed.attrs["cycle_time"],
        grid_id="global_025deg",
        grid_name="g",
        grid_resolution_km=25.0,
        zarr_store_path=store_path,
        variables=(VariableSpec("temperature_2m", "T", "K", "t2m"),),
        expected_lead_time_hours=GFS_LEADS,
    )

    mock_conn = MagicMock()
    coord = RunCoordinator(spec, store_path)

    # Patch StoreLockCoordinator and Session
    with patch("ingestion.core.coordinator.StoreLockCoordinator"), \
         patch("ingestion.core.coordinator.Session"):
        coord.initialize_run_store(
            mock_conn,
            seed_dataset=seed,
            expected_leads=GFS_LEADS,
            expected_members=(),
            run_id="run-123",
            is_same_cycle=True,
        )

    # Verify that existing store was retained and lead 6 data is intact
    ds = read_dataset(store_path)
    np.testing.assert_allclose(
        ds.temperature_2m.sel(lead_time_hours=6).values,
        np.full((4, 4), 300.0, dtype=np.float32),
    )
    # Lead 12 is still fill value
    assert np.isnan(ds.temperature_2m.sel(lead_time_hours=12).values).all()


# ==============================================================================
# Test H — Consolidated Metadata
# ==============================================================================


def test_h_consolidated_metadata(tmp_path: Path) -> None:
    """Verify that .zmetadata exists and enables consolidated xarray opens."""
    seed = _make_gfs_seed()
    store_dir = tmp_path / "gfs_consolidated.zarr"
    prepare_run_store(seed, str(store_dir), expected_lead_time_hours=GFS_LEADS)

    zmeta_file = store_dir / ".zmetadata"
    assert zmeta_file.exists(), ".zmetadata file must exist in the initialized store"

    zmeta = json.loads(zmeta_file.read_text(encoding="utf-8"))
    assert zmeta.get("zarr_consolidated_format") == 1
    metadata = zmeta.get("metadata", {})

    # Check that root, coords, and data vars are all represented
    assert ".zattrs" in metadata
    assert ".zgroup" in metadata
    assert "latitude/.zarray" in metadata
    assert "latitude/.zattrs" in metadata
    assert "lead_time_hours/.zarray" in metadata
    assert "temperature_2m/.zarray" in metadata
    assert "temperature_2m/.zattrs" in metadata
    assert "relative_humidity/.zarray" in metadata
    assert "relative_humidity/.zattrs" in metadata

    # Ensure xr.open_zarr with consolidated=True works seamlessly
    ds_cons = xr.open_zarr(str(store_dir), consolidated=True)
    assert "temperature_2m" in ds_cons.data_vars
    assert "relative_humidity" in ds_cons.data_vars
    assert ds_cons.temperature_2m.shape == (len(GFS_LEADS), len(GRID_LAT), len(GRID_LON))
    ds_cons.close()
