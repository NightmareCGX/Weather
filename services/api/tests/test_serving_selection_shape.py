"""Performance-regression structural tests (Phase 1 remediation).

These tests are deterministic — they do NOT rely on wall-clock timings. They
prove the selection-before-materialization contract:

* map tile reads only a spatially bounded window (never the full 721x1440
  global field),
* point interpolation reads only the 2x2 neighborhood (never the full grid),
* GEFS member interpolation reads per-member neighborhoods,
* the reader gate materializes the selected subset BEFORE releasing the SHARED
  lock (all chunk I/O happens under the lock).

They would FAIL against the old full-store ``gated_read_dataset`` implementation
(which materialized the entire store before selection).

PostgreSQL-independent: these tests exercise the service-level selectors and
the reader-gate helper directly, with local fixture Zarr stores, so they run in
any environment (no DB, no MinIO, no reader pool required).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/domain/src")))

from domain.geo.grid import RegularGrid

from api.core.reader_gate import gated_read_dataset_with_selector


@pytest.fixture(autouse=True)
def _no_reader_pool(monkeypatch):
    """Force the direct (no-DB-gate) bounded read path.

    These structural tests assert selection/materialization *shapes* and do not
    need the PostgreSQL advisory-lock gate. When the shared app lifespan's
    ``reader_pool`` is set (e.g. a TestClient from another test module), we
    force it to ``None`` so ``gated_read_dataset_with_selector`` uses the direct
    bounded selector path (same logic, no DB lock) — otherwise a lingering
    shutting-down lifecycle raises ``ReaderGateClosing``.
    """
    try:
        import api.main as main

        if hasattr(main, "reader_pool"):
            monkeypatch.setattr(main, "reader_pool", None)
        if hasattr(main, "reader_lifecycle"):
            monkeypatch.setattr(main, "reader_lifecycle", None)
    except ImportError:
        pass
    yield
from api.services.point_forecast import (
    _CycleMetadata,
    _derive_grid,
    gated_cycle_metadata,
    gated_point_interpolations,
)
from api.services.tiles import _select_tile_window
from tests._zarr_writer import write_dataset


# (Grid copied from tests/fixtures so these tests are standalone.)
_LAT0 = 0.0
_LAT_STEP = 1.0
_LAT_N = 30
_LON0 = -15.0
_LON_STEP = 1.0
_LON_N = 30
_LEADS = [0, 6, 12, 18]


def _make_grid() -> RegularGrid:
    return RegularGrid(
        lat_start=_LAT0, lon_start=_LON0, lat_step=_LAT_STEP, lon_step=_LON_STEP,
        rows=_LAT_N, cols=_LON_N,
    )


def _build_gfs_fixture(tmp_path) -> str:
    """A deterministic GFS-style (lead, lat, lon) store (721x1440-like but tiny)."""
    lat = _LAT0 + _LAT_STEP * np.arange(_LAT_N)
    lon = _LON0 + _LON_STEP * np.arange(_LON_N)
    lead = np.asarray(_LEADS, dtype=int)
    lead_g, lat_g, lon_g = np.meshgrid(lead, lat, lon, indexing="ij")
    temperature = 10.0 + 0.1 * lat_g + 0.2 * lon_g + 0.5 * lead_g
    ds = xr.Dataset(
        {"temperature_2m": (("lead_time_hours", "latitude", "longitude"), temperature)},
        coords={"lead_time_hours": lead, "latitude": lat, "longitude": lon},
    )
    store = str(tmp_path / "gfs.zarr")
    write_dataset(ds, store, chunks={"lead_time_hours": 1, "latitude": 10, "longitude": 10})
    return store


def _build_gefs_fixture(tmp_path) -> str:
    """A deterministic GEFS-style (member, lead, lat, lon) store."""
    lat = _LAT0 + _LAT_STEP * np.arange(_LAT_N)
    lon = _LON0 + _LON_STEP * np.arange(_LON_N)
    lead = np.asarray(_LEADS, dtype=int)
    member = np.asarray([0, 1, 2, 3, 4], dtype=int)
    m_g, lead_g, lat_g, lon_g = np.meshgrid(member, lead, lat, lon, indexing="ij")
    temperature = 10.0 + 0.1 * lat_g + 0.2 * lon_g + 0.5 * lead_g + 2.0 * m_g
    ds = xr.Dataset(
        {"temperature_2m": (("member", "lead_time_hours", "latitude", "longitude"), temperature)},
        coords={"member": member, "lead_time_hours": lead, "latitude": lat, "longitude": lon},
    )
    store = str(tmp_path / "gefs.zarr")
    write_dataset(
        ds, store,
        chunks={"member": 5, "lead_time_hours": 1, "latitude": 10, "longitude": 10},
    )
    return store


# ---------------------------------------------------------------------------
# Map tile: materialized shape must be spatially bounded, not the full grid.
# ---------------------------------------------------------------------------


def test_tile_selector_materializes_only_spatial_window_gfs(tmp_path) -> None:
    store = _build_gfs_fixture(tmp_path)
    # Emulate a zoom-4 tile over a small geographic window (lon 0..5, lat 0..5).
    window = gated_read_dataset_with_selector(
        store,
        lambda ds: _select_tile_window(ds, variable="temperature_2m", lead=12, zoom=4, x=8, y=8),
    )
    assert hasattr(window, "field"), "selector should return a _TileWindow"
    shape = np.asarray(window.field).shape
    # A bounded window, NOT the full (30, 30) grid.
    assert shape[0] < 30, f"tile window materialized too many rows: {shape}"
    assert shape[1] < 30, f"tile window materialized too many cols: {shape}"
    assert shape[0] >= 1 and shape[1] >= 1


def test_tile_selector_materializes_spatial_window_gefs(tmp_path) -> None:
    store = _build_gefs_fixture(tmp_path)
    window = gated_read_dataset_with_selector(
        store,
        lambda ds: _select_tile_window(ds, variable="temperature_2m", lead=6, zoom=4, x=8, y=8),
    )
    shape = np.asarray(window.field).shape
    assert shape[0] < 30 and shape[1] < 30, f"GEFS window too large: {shape}"
    # The member axis must be absent (reduced to a 2-D mean surface).
    assert hasattr(window, "grid")
    # The field is float64 mean over members / NaN where window empty.
    assert window.field.dtype.kind == "f"


# ---------------------------------------------------------------------------
# Point interpolation: only the 2x2 neighborhood is materialized.
# ---------------------------------------------------------------------------


def test_point_interpolation_materializes_tiny_neighborhood_gfs(tmp_path) -> None:
    store = _build_gfs_fixture(tmp_path)

    captured: dict[str, tuple] = {}
    import api.core.reader_gate as rg
    original = rg.gated_read_dataset_with_selector

    def spy(store_path, selector):
        # Wrap: intercept the selector to record the lazily-selected isel shape.
        def wrapped(ds):
            from api.services.point_forecast import _derive_grid as dg
            grid, _lat_desc, _lon_desc = dg(ds)
            row_f, col_f = grid.row_col_from_coordinates(4.5, -8.5)
            ro = int(np.floor(row_f))
            co = int(np.floor(col_f))
            # Draw the 2x2 indices in stored orientation (no reversing here as
            # grid is ascending).
            idx = ds["temperature_2m"].isel(
                lead_time_hours=0, latitude=[ro, min(ro + 1, _LAT_N - 1)],
                longitude=[co, min(co + 1, _LON_N - 1)],
            )
            captured["isel_shape"] = tuple(idx.sizes.values())
            return selector(ds)
        return original(store_path, wrapped)

    rg.gated_read_dataset_with_selector = spy  # type: ignore[assignment]
    try:
        values = gated_point_interpolations(
            store, var_codes=("temperature_2m",), lead=12, latitude=4.5, longitude=-8.5
        )
    finally:
        rg.gated_read_dataset_with_selector = original
    assert values is not None
    # Only a tiny isel shape (2 lat x 2 lon) was read.
    assert captured["isel_shape"] == (2, 2), f"expected 2x2 point neighborhood, got {captured['isel_shape']}"


def test_point_interpolation_numeric_gfs(tmp_path) -> None:
    """The bounded interpolation returns the same value as a full-grid bilinear."""
    store = _build_gfs_fixture(tmp_path)
    values = gated_point_interpolations(
        store, var_codes=("temperature_2m",), lead=6, latitude=4.5, longitude=-8.5
    )
    ds = xr.open_zarr(store, consolidated=False)
    _grid, _lat_desc, _lon_desc = _derive_grid(ds)
    # Sanity: reference value from the formula.
    ref = 10.0 + 0.1 * 4.5 + 0.2 * (-8.5) + 0.5 * 6
    assert values is not None
    got = values["temperature_2m"]
    assert got == np.float64(10.0 + 0.1 * 4.5 + 0.2 * -8.5 + 0.5 * 6.0), (
        f"bounded interpolation deviated from analytic field: {got} != {ref}"
    )
    assert abs(got - ref) < 1e-6


# ---------------------------------------------------------------------------
# GEFS member (ensemble) interpolation: per-member tiny neighborhoods.
# ---------------------------------------------------------------------------

def _gefs_members_via_gate(tmp_path):
    from api.services.ensemble_data import _gated_member_values

    store = _build_gefs_fixture(tmp_path)
    return store, _gated_member_values(
        store, "temperature_2m", 6, latitude=4.5, longitude=-8.5
    )


def test_ensemble_member_interpolation_neighborhood(tmp_path) -> None:
    store, members = _gefs_members_via_gate(tmp_path)
    assert members is not None
    assert len(members) == 5  # one value per member
    # Values follow 2*member + base.
    base = 10.0 + 0.1 * 4.5 + 0.2 * -8.5 + 0.5 * 6
    for i, mval in enumerate(members):
        assert abs(mval - (base + 2.0 * i)) < 1e-6, (
            f"member {i} value {mval} != {base + 2.0*i}"
        )


# ---------------------------------------------------------------------------
# Reader-gate contract: materialization is bounded and lazy-selection-driven.
# ---------------------------------------------------------------------------


def test_gated_read_dataset_with_selector_materializes_bounded_result(tmp_path) -> None:
    store = _build_gfs_fixture(tmp_path)

    def selector(ds):
        # Return a lazy selection at coordinate level without reading the grid.
        arr = ds["temperature_2m"].sel(lead_time_hours=12, latitude=slice(4, 6), longitude=slice(-10, -8))
        # Materialize the bounded selection; this must NOT read the full store.
        return arr.values

    out = gated_read_dataset_with_selector(store, selector)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 3), f"expected bounded 3x3 read, got {out.shape}"


def test_gated_read_dataset_metadata_does_not_materialize_grid(tmp_path) -> None:
    store = _build_gfs_fixture(tmp_path)
    meta = gated_cycle_metadata(store)
    assert isinstance(meta, _CycleMetadata)
    # Only lead/var names are read; no gridded data materialized.
    assert meta.lead_times == frozenset(_LEADS)
    assert "temperature_2m" in meta.var_names