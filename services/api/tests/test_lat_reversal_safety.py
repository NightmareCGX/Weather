"""Regression tests: the tile serving path must never apply a NEGATIVE-STEP
slice to a lazily-indexed chunked Zarr array.

Root cause (2026-08-24 correctness-hardening task): ``_slice_field`` reversed
descending axes on the **lazy** dataset via
``field.isel(latitude=slice(None, None, -1))``. xarray's indexer decomposition
(``xarray.core.indexing._decompose_slice``, verified against xarray 2024.11.0)
translates ANY negative-step slice into "forward backend slice + in-memory
reversal", computing ``exact_stop = range(start, stop, step)[-1]`` -- which
raises ``IndexError: range object index out of range`` whenever the effective
negative-step slice is EMPTY (start <= stop). Direct bounded negative-step
slices reproduce this deterministically (e.g.
``isel(latitude=slice(5, 10, -1))`` fails 100% of the time); whether the
production reverse-then-select composition can produce an empty composite is
input-dependent, which made the defect LATENT rather than deterministic in
serving.

The fix orders label-slice arguments to match each stored axis's own monotonic
direction (descending source axis -> ``slice(hi, lo)``), so xarray derives a
plain POSITIVE-step positional read; orientation is normalized IN MEMORY after
the bounded window is materialized.

These tests prove, behaviorally:

A. No negative-step slice ever reaches the Zarr backend from the production
   selector (zarr.Array.__getitem__ spy -- behavioral, not grep).
B. The direct negative-step operation that the fix removed does fail with the
   recorded signature on a chunked store (guards against "it doesn't actually
   crash" regressions of this suite's premise).
C. Materialized windows stay bounded (+1-cell halo, never full-global).
D. High-latitude / pole-edge tiles and descending-latitude windows complete
   successfully and stay numerically equivalent to the full-field reference.

Deterministic: local fixture Zarr stores, no DB / MinIO / reader pool.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import xarray as xr
import zarr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/domain/src")))

from api.services.tiles import _select_tile_window
from tests._zarr_writer import write_dataset
from tests.test_tile_selection_equivalence import (
    _bounded_sample,
    _reference_render,
    _window_coverage,
    _world_dataset,
    _world_gefs_dataset,
)

# Production-like grid geometry: 0.25 deg would make fixtures huge; the
# equivalence fixture's 1.0 deg x 180 rows preserves every structural property
# (descending lat, uniform axis, chunked store) at test scale.


# ---------------------------------------------------------------------------
# Spy infrastructure: record every positional key reaching the Zarr backend.
# ---------------------------------------------------------------------------


class _BackendKeySpy:
    """Record every ``__getitem__`` key applied to zarr core arrays."""

    def __init__(self) -> None:
        self.keys: list[tuple[object, ...]] = []

    def __enter__(self) -> "_BackendKeySpy":
        self._orig = zarr.Array.__getitem__

        def spy(array: zarr.Array, item: object) -> np.ndarray:  # type: ignore[type-arg]
            self.keys.append(item if isinstance(item, tuple) else (item,))
            return self._orig(array, item)  # type: ignore[arg-type]

        zarr.Array.__getitem__ = spy  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc_info: object) -> None:
        zarr.Array.__getitem__ = self._orig  # type: ignore[method-assign]

    def assert_no_negative_step(self) -> None:
        for key in self.keys:
            for component in key:
                if isinstance(component, slice):
                    step = component.step if component.step is not None else 1
                    assert step > 0, (
                        f"negative/zero-step slice {component!r} reached the "
                        f"chunked Zarr backend; lazy negative-step indexing is "
                        f"forbidden on the serving path"
                    )


# ---------------------------------------------------------------------------
# A. Behavioral proof: no negative-step backend indexer from the selector.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gefs", [False, True], ids=["gfs", "gefs"])
def test_selector_never_sends_negative_step_to_backend(tmp_path, gefs: bool) -> None:
    """Sweeping representative tiles (both hemispheres, dateline, poles),
    no key arriving at the chunked Zarr backend may contain a negative-step
    slice."""
    ds = _world_gefs_dataset() if gefs else _world_dataset()
    store = str(tmp_path / ("spy_" + ("gefs" if gefs else "gfs") + ".zarr"))
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    lead = 6 if gefs else 12
    # Representative tiles incl. pole-edge rows and dateline column.
    tiles = [
        (2, 5),   # western hemisphere mid-lat (historical corruption case)
        (0, 0),   # NW corner (pole edge)
        (15, 0),  # NE corner near dateline/pole
        (7, 8),   # equator-ish center
        (15, 11), # SE corner near dateline
    ]
    spy = _BackendKeySpy()
    with spy:
        for x, y in tiles:
            window = _select_tile_window(
                laz, variable="temperature_2m", lead=lead, zoom=4, x=x, y=y
            )
            assert window.field.ndim == 2
            # Materialize fully inside the with-block, mirroring the gate-time
            # contract (all backend reads happen under the spy).
            _ = window.field.sum()
    spy.assert_no_negative_step()


def test_point_interpolation_path_never_sends_negative_step_to_backend(
    tmp_path,
) -> None:
    """The point path uses integer isel only; verify behaviorally too."""
    from api.services.point_forecast import (
        _derive_grid as _derive_point_grid,
        _interpolate_neighborhood,
    )

    ds = _world_dataset()
    store = str(tmp_path / "point_spy.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    grid, lat_desc, lon_desc = _derive_point_grid(laz)
    assert lat_desc and not lon_desc, "fixture must exercise a descending axis"
    field = laz["temperature_2m"].sel(lead_time_hours=12)
    spy = _BackendKeySpy()
    with spy:
        value = _interpolate_neighborhood(
            field, grid, lat_desc, lon_desc, latitude=40.3, longitude=-159.9
        )
        assert np.isfinite(value)
    spy.assert_no_negative_step()


# ---------------------------------------------------------------------------
# B. Premise guard: the REMOVED operation really does fail on chunked stores.
# ---------------------------------------------------------------------------


def test_direct_bounded_negative_step_isel_fails_on_chunked_store(
    tmp_path,
) -> None:
    """Documents the hazard itself: a bounded negative-step isel on a lazy
    chunked-Zarr array raises 'IndexError: range object index out of range'
    from xarray's _decompose_slice. If a future xarray fixes this upstream,
    this test failing is EXPECTED and the suite premise should be revisited."""
    nlat, nlon = 64, 64
    lats = np.arange(90.0, 90.0 - 1.0 * nlat, -1.0)
    lons = np.arange(0.0, 56.25, 0.5625)[:nlon]
    data = np.zeros((nlat, nlon), dtype=np.float32)
    store = str(tmp_path / "hazard.zarr")
    xr.DataArray(
        data,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
        name="t",
    ).to_dataset().to_zarr(store, consolidated=True, encoding={"t": {"chunks": [16, 16]}}, zarr_format=2)
    lazy = xr.open_zarr(store)["t"]
    with pytest.raises(IndexError, match="range object index out of range"):
        # Any bounded start>stop... i.e. an EMPTY negative-step slice. This is
        # exactly what xarray produces when reverse+select compose to nothing.
        lazy.isel(latitude=slice(5, 10, -1)).values


# ---------------------------------------------------------------------------
# C. Bounded materialization preserved.
# ---------------------------------------------------------------------------


def test_point_interpolation_matches_analytic_truth_on_descending_axes() -> None:
    """The 2x2 bilinear window must reproduce exact values on descending
    axes. Regression guard: ``isel`` with a LIST indexer returns rows in LIST
    order (already [row_0, row_1] in the ascending-grid sense); an extra
    in-memory ``[::-1, :]`` normalization swaps the corners and biases every
    off-midpoint interpolation by up to ~(1-2t) * cell gradient (found
    2026-08-24: fixture diffs of 0.04-0.1 on a linear field)."""
    import xarray as xr

    from api.services.point_forecast import (
        _derive_grid as _derive_point_grid,
        _interpolate_neighborhood,
    )

    # Descending lat AND descending lon; field linear in both so the exact
    # analytic value is known for any query point.
    lats = np.arange(90.0, -90.0 - 1.0, -1.0)
    lons = np.arange(10.0, -10.1, -1.0)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    ds = xr.Dataset(
        {"t": (("latitude", "longitude"), 2.0 + 0.05 * lat_g - 0.07 * lon_g)},
        coords={"latitude": lats, "longitude": lons},
    )
    grid, lat_desc, lon_desc = _derive_point_grid(ds)
    assert lat_desc and lon_desc, "fixture must exercise both descending axes"

    # Query points chosen OFF midpoint (t != 0.5) where a corner swap shows.
    for latitude, longitude in [(-12.3, 3.2), (33.7, -4.6), (-77.9, 8.1)]:
        got = _interpolate_neighborhood(ds["t"], grid, lat_desc, lon_desc,
                                        latitude, longitude)
        expected = 2.0 + 0.05 * latitude - 0.07 * longitude
        assert abs(got - expected) < 1e-9, (
            f"bilinear at ({latitude}, {longitude}) = {got!r}, "
            f"expected {expected!r} (corner-order regression)"
        )


def test_windows_stay_bounded_after_fix(tmp_path) -> None:
    """Post-fix windows remain strictly smaller than the full grid."""
    ds = _world_dataset()
    store = str(tmp_path / "bounded.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    lat_n = laz.sizes["latitude"]
    lon_n = laz.sizes["longitude"]
    for x, y in [(0, 0), (2, 5), (7, 8), (15, 0), (15, 11)]:
        window = _select_tile_window(
            laz, variable="temperature_2m", lead=12, zoom=4, x=x, y=y
        )
        assert window.field.shape[0] < lat_n
        assert window.field.shape[1] < lon_n
        assert window.lat_axis[-1] >= window.lat_axis[0]
        assert window.lon_axis[-1] >= window.lon_axis[0]


# ---------------------------------------------------------------------------
# D. High-latitude / pole-edge + descending-window numerical safety.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y", [0, 1, 14, 15], ids=["north-pole", "high-n", "high-s", "south-pole"])
def test_pole_edge_tiles_complete_and_match_reference(tmp_path, y: int) -> None:
    """Tiles at/near both poles complete successfully and equal the full-field
    nearest-neighbor reference over their coverage (the previously hazardous
    latitude/window class)."""
    ds = _world_dataset()
    store = str(tmp_path / f"pole_{y}.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for x in (0, 8, 15):  # W edge, meridian, E/dateline edge
        window = _select_tile_window(
            laz, variable="temperature_2m", lead=12, zoom=4, x=x, y=y
        )
        ref = _reference_render(ds, "temperature_2m", 12, 4, x, y)
        got = _bounded_sample(window, 4, x, y)
        cov = _window_coverage(window, 4, x, y)
        same = np.isclose(got[cov], ref[cov], equal_nan=True)
        assert same.all(), (
            f"pole tile y={y} x={x}: {int(np.count_nonzero(~same))} pixels differ"
        )


def test_gefs_mean_equivalent_after_fix(tmp_path) -> None:
    """GEFS member-mean windows still equal the full-field reference."""
    ds = _world_gefs_dataset()
    store = str(tmp_path / "gefs_post.zarr")
    write_dataset(ds, store)
    laz = xr.open_zarr(store)
    for x, y in [(2, 5), (0, 0), (15, 15)]:
        window = _select_tile_window(
            laz, variable="temperature_2m", lead=6, zoom=4, x=x, y=y
        )
        ref = _reference_render(ds, "temperature_2m", 6, 4, x, y)
        got = _bounded_sample(window, 4, x, y)
        cov = _window_coverage(window, 4, x, y)
        same = np.isclose(got[cov], ref[cov], equal_nan=True)
        assert same.all()


def test_empty_band_degrades_to_transparent_not_crash(tmp_path) -> None:
    """A band containing no coordinate (coarse grid) must yield the graceful
    transparent fallback instead of tripping negative-step decomposition or a
    renderer indexing error."""
    # 37-row coarse grid: 5-degree spacing makes sub-5-degree pixel bands
    # coordinate-free at high zoom.
    lats = np.arange(90.0, -90.0 - 5.0, -5.0)
    lons = np.arange(0.0, 360.0, 5.0)
    data = 5.0 + 0.1 * lats[:, None] + 0.2 * lons[None, :]
    ds = xr.Dataset(
        {"temperature_2m": (("latitude", "longitude"), data)},
        coords={"latitude": lats, "longitude": lons},
    )
    store = str(tmp_path / "coarse.zarr")
    write_dataset(ds, store, chunks={"latitude": 19, "longitude": 36})
    laz = xr.open_zarr(store)
    # z9 tile row 13 band ~84.6..85.1N spans under one 5-degree step; with the
    # +/-1-cell halo it may catch the nearest coordinate (85N) or not depending
    # on exact pixel bounds -- assert only that it completes and stays tiny.
    window = _select_tile_window(
        laz, variable="temperature_2m", lead=12, zoom=9, x=256, y=13
    )
    assert window.field.shape[0] <= 2 and window.field.shape[1] <= 2
    if window.field.size == 1:
        assert np.isnan(window.field[0, 0])
