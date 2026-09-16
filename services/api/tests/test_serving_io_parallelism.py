"""P1 first-cold-load parallelism tests (I/O fan-out and compute de-duplication).

Covers:
1. ``ShardedV1Reader.read_window`` concurrent multi-chunk fetching — exact
   equivalence with the previous strictly sequential implementation, executor
   use for multi-chunk windows only, genuine thread overlap, worker-exception
   propagation, and absence of deadlock on the full chunk grid.
2. ``_slice_field`` wind_10m/``wind_speed_10m`` — the u and v components are
   fetched concurrently while the ``hypot(u, v) * 3.6`` combination stays
   numerically identical.
3. ``_TileWindow`` — the per-pixel tile geometry is computed once during
   selection and reused by the renderer instead of being rebuilt.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import api.core.zarr as zarr_mod
import api.services.tiles as tiles_mod
from api.core.zarr import ShardedV1Reader
from tests.test_sharded_reader import _build_test_shard

#: Windows whose chunk geometry exercises different code paths.
#: (lat_min, lat_max, lon_min, lon_max, expected_chunk_count)
WINDOW_CASES = [
    (10, 20, 10, 20, 1),          # entirely inside one chunk
    (95, 105, 95, 105, 4),        # straddles 2x2 chunk boundaries
    (99, 100, 99, 100, 4),        # minimal 2x2 straddle
    (0, 0, 0, 0, 1),              # first cell of the grid
    (720, 720, 1439, 1439, 1),    # last cell of the grid
    (695, 720, 1385, 1439, 4),    # clipped at the grid's high edge
    (0, 105, 0, 105, 4),          # clipped at the grid's low edge
    (100, 199, 200, 399, 6),      # multi-row, multi-column
]


def _write_shard_store(tmp_path: Path, variable: str = "temperature_2m") -> str:
    """Write a deterministic sharded_v1 store with one 120-chunk shard."""
    store = tmp_path / "store.zarr"
    shard = store / variable / "shard.det_L0000.shard"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(_build_test_shard(val_offset=10.0))
    return str(store)


def _sequential_window_reference(
    reader: ShardedV1Reader,
    variable: str,
    *,
    lat_min: int,
    lat_max: int,
    lon_min: int,
    lon_max: int,
    lead_time_hours: int = 0,
) -> np.ndarray:
    """Oracle: the original strictly sequential nested-loop implementation.

    Kept verbatim (including the NaN fill for non-intersecting chunks) so the
    concurrent implementation is compared against the behaviour it replaced.
    """
    window = np.full(
        (lat_max - lat_min + 1, lon_max - lon_min + 1), np.nan, dtype=np.float32
    )
    for r_chunk in range(lat_min // 100, lat_max // 100 + 1):
        chunk_lat_start = r_chunk * 100
        chunk_lat_end = min((r_chunk + 1) * 100, 721)

        sub_lat_start = max(0, lat_min - chunk_lat_start)
        sub_lat_end = min(chunk_lat_end - chunk_lat_start, lat_max - chunk_lat_start + 1)
        win_lat_start = max(0, chunk_lat_start - lat_min)
        win_lat_end = win_lat_start + (sub_lat_end - sub_lat_start)

        for c_chunk in range(lon_min // 100, lon_max // 100 + 1):
            chunk_lon_start = c_chunk * 100
            chunk_lon_end = min((c_chunk + 1) * 100, 1440)

            sub_lon_start = max(0, lon_min - chunk_lon_start)
            sub_lon_end = min(chunk_lon_end - chunk_lon_start, lon_max - chunk_lon_start + 1)
            win_lon_start = max(0, chunk_lon_start - lon_min)
            win_lon_end = win_lon_start + (sub_lon_end - sub_lon_start)

            if sub_lat_end > sub_lat_start and sub_lon_end > sub_lon_start:
                chunk_arr = reader.read_chunk(
                    variable,
                    member=None,
                    lead_time_hours=lead_time_hours,
                    chunk_row=r_chunk,
                    chunk_col=c_chunk,
                )
                window[win_lat_start:win_lat_end, win_lon_start:win_lon_end] = chunk_arr[
                    sub_lat_start:sub_lat_end, sub_lon_start:sub_lon_end
                ]
    return window


# ---------------------------------------------------------------------------
# 1. read_window concurrent multi-chunk fetching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lat_min", "lat_max", "lon_min", "lon_max", "expected_chunks"),
    WINDOW_CASES,
)
def test_read_window_matches_sequential_reference(
    tmp_path: Path,
    lat_min: int,
    lat_max: int,
    lon_min: int,
    lon_max: int,
    expected_chunks: int,
) -> None:
    """The concurrent read is bit-identical to the sequential implementation.

    Separate reader instances are used so the oracle cannot be served from the
    chunk cache warmed by the concurrent read.
    """
    store = _write_shard_store(tmp_path)
    concurrent_reader = ShardedV1Reader(store)
    oracle_reader = ShardedV1Reader(store)

    got = concurrent_reader.read_window(
        "temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )
    expected = _sequential_window_reference(
        oracle_reader,
        "temperature_2m",
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    assert got.shape == expected.shape
    # Exact equality including the NaN pattern: no precision loss, no shifted
    # chunk placement, no dropped edge cells.
    assert np.array_equal(got, expected, equal_nan=True)


def test_read_window_full_grid_matches_sequential_reference(tmp_path: Path) -> None:
    """The widest possible window (8x15 = 120 chunks) is still exact."""
    store = _write_shard_store(tmp_path)
    got = ShardedV1Reader(store).read_window(
        "temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_min=0,
        lat_max=720,
        lon_min=0,
        lon_max=1439,
    )
    expected = _sequential_window_reference(
        ShardedV1Reader(store),
        "temperature_2m",
        lat_min=0,
        lat_max=720,
        lon_min=0,
        lon_max=1439,
    )

    assert got.shape == (721, 1440)
    assert np.array_equal(got, expected, equal_nan=True)


def test_read_window_missing_shard_is_nan_like_the_sequential_path(tmp_path: Path) -> None:
    """A window over an absent shard stays all-NaN rather than raising."""
    store = _write_shard_store(tmp_path)
    reader = ShardedV1Reader(store)

    window = reader.read_window(
        "no_such_variable",
        member=None,
        lead_time_hours=0,
        lat_min=95,
        lat_max=105,
        lon_min=95,
        lon_max=105,
    )

    assert window.shape == (11, 11)
    assert np.all(np.isnan(window))


def test_read_window_single_chunk_stays_inline(tmp_path: Path, monkeypatch) -> None:
    """One-chunk windows (the common high-zoom case) must not touch the pool."""
    store = _write_shard_store(tmp_path)
    reader = ShardedV1Reader(store)

    pool_calls = 0
    real_get_executor = zarr_mod.get_chunk_executor

    def spy_get_executor():
        nonlocal pool_calls
        pool_calls += 1
        return real_get_executor()

    monkeypatch.setattr(zarr_mod, "get_chunk_executor", spy_get_executor)

    reader.read_window(
        "temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_min=10,
        lat_max=20,
        lon_min=10,
        lon_max=20,
    )
    assert pool_calls == 0

    # A multi-chunk window does go through the shared executor.
    reader.read_window(
        "temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_min=95,
        lat_max=105,
        lon_min=95,
        lon_max=105,
    )
    assert pool_calls == 1


def test_read_window_fetches_chunks_concurrently(tmp_path: Path, monkeypatch) -> None:
    """Chunks of one window are fetched on distinct worker threads."""
    store = _write_shard_store(tmp_path)
    reader = ShardedV1Reader(store)

    thread_names: set[str] = set()
    names_lock = threading.Lock()
    real_read_chunk = reader.read_chunk

    def spy_read_chunk(*args, **kwargs):
        with names_lock:
            thread_names.add(threading.current_thread().name)
        # Widen the overlap window so a sequential implementation could not
        # possibly produce two distinct threads for these four chunks.
        time.sleep(0.02)
        return real_read_chunk(*args, **kwargs)

    monkeypatch.setattr(reader, "read_chunk", spy_read_chunk)

    reader.read_window(
        "temperature_2m",
        member=None,
        lead_time_hours=0,
        lat_min=95,
        lat_max=105,
        lon_min=95,
        lon_max=105,
    )

    assert len(thread_names) > 1, (
        f"expected the 4 chunks to overlap on multiple threads, saw {thread_names}"
    )


def test_read_window_propagates_worker_exception(tmp_path: Path, monkeypatch) -> None:
    """A failing chunk fetch fails the whole read instead of yielding NaNs."""
    store = _write_shard_store(tmp_path)
    reader = ShardedV1Reader(store)

    real_read_chunk = reader.read_chunk
    seen: list[int] = []
    seen_lock = threading.Lock()

    class _ChunkReadFailure(RuntimeError):
        pass

    def failing_read_chunk(variable, *, chunk_row, **kwargs):
        with seen_lock:
            seen.append(chunk_row)
        if chunk_row == 1:
            raise _ChunkReadFailure("simulated chunk fetch failure")
        return real_read_chunk(variable, chunk_row=chunk_row, **kwargs)

    monkeypatch.setattr(reader, "read_chunk", failing_read_chunk)

    with pytest.raises(_ChunkReadFailure, match="simulated chunk fetch failure"):
        reader.read_window(
            "temperature_2m",
            member=None,
            lead_time_hours=0,
            lat_min=95,
            lat_max=105,
            lon_min=95,
            lon_max=105,
        )

    # Both rows were dispatched: the exception was not swallowed and did not
    # prevent the other chunks from being attempted.
    assert 0 in seen and 1 in seen


def test_read_window_full_grid_does_not_deadlock(tmp_path: Path) -> None:
    """A 120-chunk window completes well inside a bounded wall-clock budget."""
    store = _write_shard_store(tmp_path)
    reader = ShardedV1Reader(store)

    result: list[np.ndarray] = []
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            result.append(
                reader.read_window(
                    "temperature_2m",
                    member=None,
                    lead_time_hours=0,
                    lat_min=0,
                    lat_max=720,
                    lon_min=0,
                    lon_max=1439,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced to the assertion below
            failure.append(exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=30.0)

    assert not worker.is_alive(), "read_window deadlocked on the full chunk grid"
    assert not failure, f"read_window raised: {failure!r}"
    assert result[0].shape == (721, 1440)


# ---------------------------------------------------------------------------
# 2. wind_10m u/v concurrency
# ---------------------------------------------------------------------------


class _InlinePool:
    """Sequential stand-in for ``ThreadPoolExecutor`` used as an oracle."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002 - signature parity
        pass

    def __enter__(self) -> "_InlinePool":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def map(self, fn, iterable):
        return [fn(item) for item in iterable]


def _write_wind_sharded_store(tmp_path: Path) -> tuple[str, xr.Dataset]:
    """Sharded store with official mean shards for both wind components.

    ``wind_u_10m`` chunks carry ``chunk_index`` and ``wind_v_10m`` carry
    ``chunk_index + 100`` so the two components are distinguishable.
    """
    store_dir = tmp_path / "gefs_wind.zarr"
    store_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = store_dir / "__commit__" / "v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 1,
                "generation": "gen_wind_test",
                "storage_format_version": "sharded_v1",
            }
        ),
        encoding="utf-8",
    )

    for component, offset in (("wind_u_10m", 0.0), ("wind_v_10m", 100.0)):
        shard = store_dir / component / "shard.mean_L0000.shard"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(_build_test_shard(val_offset=offset))

    latitudes = [90.0 - i * 0.25 for i in range(721)]
    longitudes = [0.0 + j * 0.25 for j in range(1440)]
    dataset = xr.Dataset(
        data_vars={
            "wind_u_10m": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                np.zeros((1, 1, 721, 1440), dtype=np.float32),
            ),
            "wind_v_10m": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                np.zeros((1, 1, 721, 1440), dtype=np.float32),
            ),
        },
        coords={
            "member": [1],
            "lead_time_hours": [0],
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    return str(store_dir), dataset


def _tile_geometry(grid, zoom: int, x: int, y: int):
    """Tile pixel latitudes + grid-native longitudes, as tiles.py derives them."""
    n = 2**zoom
    px_idx, py_idx = np.meshgrid(
        np.arange(tiles_mod.TILE_SIZE, dtype=np.float64),
        np.arange(tiles_mod.TILE_SIZE, dtype=np.float64),
        indexing="xy",
    )
    pixel_lons = ((x + (px_idx + 0.5) / tiles_mod.TILE_SIZE) / n) * 360.0 - 180.0
    y_merc = y + (py_idx + 0.5) / tiles_mod.TILE_SIZE
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * y_merc / n)))
    return np.degrees(lat_rad), tiles_mod._align_longitudes(grid, pixel_lons)


def test_sharded_wind_components_are_fetched_concurrently(
    tmp_path: Path, monkeypatch
) -> None:
    """u and v for the same window are read on two different threads."""
    store_path, dataset = _write_wind_sharded_store(tmp_path)
    reader = ShardedV1Reader(store_path)
    monkeypatch.setattr(zarr_mod, "get_sharded_reader", lambda _store: reader)

    records: list[tuple[str, int, int, str]] = []
    records_lock = threading.Lock()
    real_read_window = reader.read_window

    def spy_read_window(variable, **kwargs):
        with records_lock:
            records.append(
                (
                    variable,
                    kwargs["lon_min"],
                    kwargs["lon_max"],
                    threading.current_thread().name,
                )
            )
        time.sleep(0.03)
        return real_read_window(variable, **kwargs)

    monkeypatch.setattr(reader, "read_window", spy_read_window)

    grid = tiles_mod._derive_grid(dataset)
    pixel_lats, lon_native = _tile_geometry(grid, 4, 7, 8)
    tiles_mod._slice_field(
        dataset,
        "wind_10m",
        0,
        grid,
        pixel_lats,
        lon_native,
        expected_members=30,
        store_path=store_path,
    )

    # Group the observed reads by the window they served.
    by_window: dict[tuple[int, int], dict[str, str]] = {}
    for variable, lon_min, lon_max, thread_name in records:
        by_window.setdefault((lon_min, lon_max), {})[variable] = thread_name

    assert by_window, "no wind component reads were recorded"
    for window, by_component in by_window.items():
        assert set(by_component) == {"wind_u_10m", "wind_v_10m"}, (
            f"window {window} did not read both components: {by_component}"
        )
        assert by_component["wind_u_10m"] != by_component["wind_v_10m"], (
            f"u and v for window {window} ran on the same thread, so they were "
            f"serialized instead of overlapped"
        )


def test_sharded_wind_combination_matches_sequential_oracle(
    tmp_path: Path, monkeypatch
) -> None:
    """The hypot * 3.6 combination is unchanged by the concurrent fetch."""
    store_path, dataset = _write_wind_sharded_store(tmp_path)
    grid = tiles_mod._derive_grid(dataset)
    pixel_lats, lon_native = _tile_geometry(grid, 4, 7, 8)

    def _field() -> np.ndarray:
        monkeypatch.setattr(
            zarr_mod, "get_sharded_reader", lambda _store: ShardedV1Reader(store_path)
        )
        field, _lat, _lon = tiles_mod._slice_field(
            dataset,
            "wind_10m",
            0,
            grid,
            pixel_lats,
            lon_native,
            expected_members=30,
            store_path=store_path,
        )
        return field

    concurrent = _field()

    monkeypatch.setattr(tiles_mod, "ThreadPoolExecutor", _InlinePool)
    sequential = _field()

    assert concurrent.shape == sequential.shape
    assert np.array_equal(concurrent, sequential, equal_nan=True)
    # The store is non-degenerate, so this is a meaningful comparison.
    assert np.any(np.isfinite(concurrent))
    # hypot(u, v) * 3.6 with u in [0, 119] and v in [100, 219] km/h.
    assert float(np.nanmax(concurrent)) <= float(np.hypot(119.0, 219.0) * 3.6) + 1e-6


# ---------------------------------------------------------------------------
# 3. Tile pixel geometry computed once
# ---------------------------------------------------------------------------


def test_tile_window_carries_pixel_geometry(tmp_path: Path) -> None:
    """``_TileWindow`` exposes the geometry and the renderer reuses it.

    The renderer no longer accepts ``zoom``/``x``/``y``: it can only produce the
    correct tile if the window itself carries the per-pixel coordinates, so this
    also guards against the duplicated meshgrid being reintroduced.
    """
    latitudes = np.array([38.0, 38.25, 38.5], dtype=np.float64)
    longitudes = np.array([-107.0, -106.75, -106.5], dtype=np.float64)
    dataset = xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                np.full((1, 3, 3), 12.0, dtype=np.float32),
            )
        },
        coords={"lead_time_hours": [0], "latitude": latitudes, "longitude": longitudes},
    )

    window = tiles_mod._select_tile_window(
        dataset, variable="temperature_2m", lead=0, zoom=4, x=8, y=8
    )

    expected_lats, expected_lons = _tile_geometry(window.grid, 4, 8, 8)
    assert np.array_equal(window.pixel_lats, expected_lats)
    assert np.array_equal(window.pixel_lons_native, expected_lons)

    # Rendering consumes the carried geometry (no zoom/x/y arguments exist).
    png = tiles_mod._render_window_to_png(window, variable="temperature_2m", cache_key=())
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
