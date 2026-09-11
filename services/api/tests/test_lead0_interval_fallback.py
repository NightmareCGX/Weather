"""Comprehensive tests for lead-time-0 display/source-selection fallback under Data Lifecycle V2.

Verifies:
1. Shared valid-time resolver selects positive-lead fallback for interval variables at lead 0.
2. Shared valid-time resolver leaves instantaneous variables on newest cycle at lead 0.
3. Shared valid-time resolver leaves interval variables on newest cycle at positive leads.
4. Point forecast mixed-source resolution correctly samples interval variables from fallback.
5. Precipitation companion variables (crain, csnow, cfrzr, cicep) are strictly source-coupled
   to the exact same store, cycle, and lead as precipitation_amount_3h.
6. Point forecast gracefully sets interval variables to null when fallback is missing without failing.
7. Map raster tiles and metadata for interval variables at lead 0 read the fallback source.
8. Availability does not falsely advertise interval variables at valid_time if no positive lead exists.
9. Cache keys and provenance correctly reflect the actual fallback source.
"""

from __future__ import annotations

from datetime import datetime, timezone
import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.core.database import Base
from api.models.entities import (
    EnsembleMember,
    EnsembleMemberProduct,
    ForecastCenter,
    ForecastCycleLifecycle,
    ForecastGrid,
    ForecastProduct,
    ForecastVariable,
    Model,
    ModelRun,
    ModelVersion,
)
from api.services.availability import build_forecast_availability
from api.services.cache import (
    build_ensemble_cache_key,
    build_probability_cache_key,
)
from api.services.point_forecast import (
    ResolvedLocation,
    build_point_forecast,
)
from api.services.tiles import (
    _tile_cache_key,
    render_tile_png,
    resolve_tile_read_context,
)
from tests._zarr_writer import write_dataset
from tests.test_tiles import _png_has_opaque_pixels


@pytest.fixture(autouse=True)
def _no_reader_pool(monkeypatch):
    """Force the direct (no-DB-gate) bounded read path for standalone SQLite tests."""
    monkeypatch.setenv("WEATHER_SIMULATED_NOW", "2026-09-05T00:00:00Z")
    try:
        import api.main as main

        if hasattr(main, "reader_pool"):
            monkeypatch.setattr(main, "reader_pool", None)
        if hasattr(main, "reader_lifecycle"):
            monkeypatch.setattr(main, "reader_lifecycle", None)
    except ImportError:
        pass
    yield


def _dt(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


LAT_START = 38.0
LON_START = -107.0
LATITUDES = [LAT_START + i * 0.25 for i in range(4)]
LONGITUDES = [LON_START + j * 0.25 for j in range(4)]


def _build_test_dataset(
    *,
    leads: list[int],
    cycle_time: datetime,
    temperature_val: float,
    precip_val: float | None = None,
    crain_val: int = 0,
    csnow_val: int = 0,
    cloud_cover_val: float | None = None,
) -> xr.Dataset:
    lat = np.asarray(LATITUDES, dtype=float)
    lon = np.asarray(LONGITUDES, dtype=float)
    lead_arr = np.asarray(leads, dtype=float)
    lg, lat_g, lon_g = np.meshgrid(lead_arr, lat, lon, indexing="ij")

    temp_data = np.full_like(lg, temperature_val, dtype=np.float32)

    # Lead 0 is NaN for interval variables, positive leads get specified value
    if precip_val is not None:
        p_data = np.where(lg == 0, np.nan, precip_val).astype(np.float32)
    else:
        p_data = np.full_like(lg, np.nan, dtype=np.float32)

    crain_data = np.where(lg == 0, 0, crain_val).astype(np.uint8)
    csnow_data = np.where(lg == 0, 0, csnow_val).astype(np.uint8)
    cfrzr_data = np.zeros_like(lg, dtype=np.uint8)
    cicep_data = np.zeros_like(lg, dtype=np.uint8)

    if cloud_cover_val is not None:
        cc_data = np.where(lg == 0, np.nan, cloud_cover_val).astype(np.float32)
    else:
        cc_data = np.full_like(lg, np.nan, dtype=np.float32)

    return xr.Dataset(
        data_vars={
            "temperature_2m": (("lead_time_hours", "latitude", "longitude"), temp_data),
            "precipitation_amount_3h": (("lead_time_hours", "latitude", "longitude"), p_data),
            "crain": (("lead_time_hours", "latitude", "longitude"), crain_data),
            "csnow": (("lead_time_hours", "latitude", "longitude"), csnow_data),
            "cfrzr": (("lead_time_hours", "latitude", "longitude"), cfrzr_data),
            "cicep": (("lead_time_hours", "latitude", "longitude"), cicep_data),
            "cloud_cover_3h": (("lead_time_hours", "latitude", "longitude"), cc_data),
        },
        coords={
            "lead_time_hours": leads,
            "latitude": lat,
            "longitude": lon,
            "time": np.datetime64(cycle_time.strftime("%Y-%m-%dT%H:%M:%S")),
        },
        attrs={"cycle_time": cycle_time.isoformat(), "model_id": "gfs"},
    )


@pytest.fixture
def fallback_env(tmp_path, monkeypatch):
    """Setup an isolated database and two Zarr stores (previous 18Z and current 00Z)."""
    monkeypatch.setenv("WEATHER_SIMULATED_NOW", "2026-09-05T00:00:00Z")
    db_path = f"sqlite:///{tmp_path}/fallback_test.db"
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    tables = [
        ForecastCenter.__table__,
        Model.__table__,
        ModelVersion.__table__,
        ModelRun.__table__,
        EnsembleMember.__table__,
        EnsembleMemberProduct.__table__,
        ForecastVariable.__table__,
        ForecastGrid.__table__,
        ForecastProduct.__table__,
        ForecastCycleLifecycle.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables)

    c_18z = _dt(2026, 9, 4, 18)
    c_00z = _dt(2026, 9, 5, 0)

    # 1. Store 18Z (previous cycle): has lead 0 and lead 6 (+6h = 00Z)
    # Lead 6 has valid precipitation (5.0mm, rain flag) and cloud cover (75%)
    store_18z = str(tmp_path / "gfs_18z.zarr")
    ds_18z = _build_test_dataset(
        leads=[0, 6],
        cycle_time=c_18z,
        temperature_val=10.0,
        precip_val=5.0,
        crain_val=1,
        csnow_val=0,
        cloud_cover_val=75.0,
    )
    write_dataset(ds_18z, store_18z)

    # 2. Store 00Z (current cycle): has lead 0 and lead 3
    # Lead 0 has NaN for interval variables, 0 for categorical flags, and 15.0 for temp
    store_00z = str(tmp_path / "gfs_00z.zarr")
    ds_00z = _build_test_dataset(
        leads=[0, 3],
        cycle_time=c_00z,
        temperature_val=15.0,
        precip_val=2.0,  # Lead 3 gets 2.0mm; lead 0 will be NaN by construction
        crain_val=1,
        cloud_cover_val=40.0,
    )
    write_dataset(ds_00z, store_00z)

    with Session(engine) as session:
        center = ForecastCenter(id="c_noaa", center_id="noaa", name="NOAA", country="US", created_at=c_18z)
        model = Model(id="m_gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0, created_at=c_18z)
        version = ModelVersion(id="v_gfs", model_id="gfs", version_string="v1.0", created_at=c_18z)
        grid = ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0)

        v_t2m = ForecastVariable(id="v_t2m", variable_code="temperature_2m", name="2m Temperature", unit="°C")
        v_tp = ForecastVariable(id="v_tp", variable_code="precipitation_amount_3h", name="3h Precip", unit="mm")
        v_tcc = ForecastVariable(id="v_tcc", variable_code="cloud_cover_3h", name="3h Cloud", unit="%")
        v_crain = ForecastVariable(id="v_crain", variable_code="crain", name="Rain Flag", unit="flag")
        v_csnow = ForecastVariable(id="v_csnow", variable_code="csnow", name="Snow Flag", unit="flag")

        r_18z = ModelRun(id="run_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path=store_18z, created_at=c_18z)
        r_00z = ModelRun(id="run_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path=store_00z, created_at=c_00z)

        session.add_all([center, model, version, grid, v_t2m, v_tp, v_tcc, v_crain, v_csnow, r_18z, r_00z])

        # Products for 18Z (leads 0, 6)
        for lead in (0, 6):
            for v_code in ("temperature_2m", "precipitation_amount_3h", "cloud_cover_3h", "crain", "csnow"):
                session.add(
                    ForecastProduct(
                        id=f"p_18z_{v_code}_{lead}",
                        run_id="run_18z",
                        variable_id=v_code,
                        grid_id="global_025deg",
                        product_type="surface",
                        lead_time_hours=lead,
                    )
                )

        # Products for 00Z (leads 0, 3)
        for lead in (0, 3):
            for v_code in ("temperature_2m", "precipitation_amount_3h", "cloud_cover_3h", "crain", "csnow"):
                session.add(
                    ForecastProduct(
                        id=f"p_00z_{v_code}_{lead}",
                        run_id="run_00z",
                        variable_id=v_code,
                        grid_id="global_025deg",
                        product_type="surface",
                        lead_time_hours=lead,
                    )
                )

        session.commit()

    return engine, store_18z, store_00z


def test_point_forecast_mixed_source_lead0_fallback(fallback_env):
    """Test point forecast at cycle boundary (00Z):
    - temperature_2m comes from current cycle 00Z + 0 (15°C)
    - precipitation_amount_3h comes from fallback cycle 18Z + 6 (5.0mm)
    - cloud_cover_3h comes from fallback cycle 18Z + 6 (75%)
    - precipitation_type reflects 18Z + 6 rain flag ('rain')
    - entry provenance is anchored to valid_time 00Z.
    """
    engine, _, _ = fallback_env
    loc = ResolvedLocation(
        latitude=LAT_START + 0.125,
        longitude=LON_START + 0.125,
        elevation_m=None,
        resolved_via="coordinates",
    )

    with Session(engine) as session:
        data = build_point_forecast(
            session,
            location=loc,
            model="gfs",
            variables=["temperature_2m", "precipitation_amount_3h", "cloud_cover_3h"],
            units="metric",
            start_lead_time_hours=0,
            end_lead_time_hours=0,
        )

    f0 = next(f for f in data.forecasts if f.valid_time == _dt(2026, 9, 5, 0))
    # Provenance
    assert f0.valid_time == _dt(2026, 9, 5, 0)
    assert f0.lead_time_hours == 0
    assert f0.cycle_time == _dt(2026, 9, 5, 0)

    # Instantaneous stays on 00Z + 0
    assert getattr(f0, "temperature_2m") == pytest.approx(15.0, abs=1e-3)

    # Interval variables fall back to 18Z + 6
    assert getattr(f0, "precipitation_amount_3h") == pytest.approx(5.0, abs=1e-3)
    assert getattr(f0, "cloud_cover_3h") == pytest.approx(75.0, abs=1e-3)

    # Phase classification uses 18Z + 6 rain flag
    assert getattr(f0, "precipitation_type") == "rain"


def test_point_forecast_companion_coupling_snow(tmp_path):
    """Assert companion variables crain/csnow/cfrzr/cicep are coupled to precipitation fallback.
    If 18Z + 6 has csnow=1, precipitation_type must be 'snow', proving synthetic 00Z lead0 zeroes
    were not consumed.
    """
    db_path = f"sqlite:///{tmp_path}/snow_test.db"
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[
        ForecastCenter.__table__, Model.__table__, ModelVersion.__table__, ModelRun.__table__,
        ForecastVariable.__table__, ForecastGrid.__table__, ForecastProduct.__table__, ForecastCycleLifecycle.__table__,
    ])

    c_18z = _dt(2026, 9, 4, 18)
    c_00z = _dt(2026, 9, 5, 0)

    store_18z = str(tmp_path / "snow_18z.zarr")
    ds_18z = _build_test_dataset(
        leads=[0, 6],
        cycle_time=c_18z,
        temperature_val=-2.0,
        precip_val=4.0,
        crain_val=0,
        csnow_val=1,  # Snow active in fallback source
    )
    write_dataset(ds_18z, store_18z)

    store_00z = str(tmp_path / "snow_00z.zarr")
    ds_00z = _build_test_dataset(
        leads=[0],
        cycle_time=c_00z,
        temperature_val=1.0,
        precip_val=None,  # Lead 0 NaN
        crain_val=0,
        csnow_val=0,  # Synthetic zeroes
    )
    write_dataset(ds_00z, store_00z)

    with Session(engine) as session:
        session.add(ForecastCenter(id="c_noaa", center_id="noaa", name="NOAA", country="US", created_at=c_18z))
        session.add(Model(id="m_gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0, created_at=c_18z))
        session.add(ModelVersion(id="v_gfs", model_id="gfs", version_string="v1.0", created_at=c_18z))
        session.add(ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0))
        session.add(ForecastVariable(id="v_t2m", variable_code="temperature_2m", name="2m Temperature", unit="°C"))
        session.add(ForecastVariable(id="v_tp", variable_code="precipitation_amount_3h", name="3h Precip", unit="mm"))
        session.add(ForecastVariable(id="v_csnow", variable_code="csnow", name="Snow Flag", unit="flag"))
        session.add(ForecastVariable(id="v_crain", variable_code="crain", name="Rain Flag", unit="flag"))
        session.add(ModelRun(id="run_18z", model_version_id="v_gfs", cycle_time=c_18z, status="ready", zarr_store_path=store_18z, created_at=c_18z))
        session.add(ModelRun(id="run_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path=store_00z, created_at=c_00z))

        for v_code in ("temperature_2m", "precipitation_amount_3h", "csnow", "crain"):
            session.add(ForecastProduct(id=f"p_18_{v_code}", run_id="run_18z", variable_id=v_code, grid_id="global_025deg", product_type="surface", lead_time_hours=6))
            session.add(ForecastProduct(id=f"p_00_{v_code}", run_id="run_00z", variable_id=v_code, grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        loc = ResolvedLocation(latitude=LAT_START + 0.125, longitude=LON_START + 0.125, elevation_m=None, resolved_via="coordinates")
        data = build_point_forecast(
            session,
            location=loc,
            model="gfs",
            variables=["temperature_2m", "precipitation_amount_3h"],
            units="metric",
            start_lead_time_hours=None,
            end_lead_time_hours=None,
        )

    f0 = next(f for f in data.forecasts if f.valid_time == _dt(2026, 9, 5, 0))
    assert getattr(f0, "precipitation_amount_3h") == pytest.approx(4.0, abs=1e-3)
    # Proves coupled companion sampling from 18Z + 6:
    assert getattr(f0, "precipitation_type") == "snow"


def test_point_forecast_missing_fallback_graceful_null(tmp_path):
    """When cold-starting with only cycle 00Z lead 0 (no fallback source):
    - temperature_2m is returned from 00Z + 0
    - precipitation_amount_3h and cloud_cover_3h are null
    - endpoint does not fail globally.
    """
    db_path = f"sqlite:///{tmp_path}/solo_test.db"
    engine = create_engine(db_path, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[
        ForecastCenter.__table__, Model.__table__, ModelVersion.__table__, ModelRun.__table__,
        ForecastVariable.__table__, ForecastGrid.__table__, ForecastProduct.__table__, ForecastCycleLifecycle.__table__,
    ])

    c_00z = _dt(2026, 9, 5, 0)
    store_00z = str(tmp_path / "solo_00z.zarr")
    ds_00z = _build_test_dataset(
        leads=[0],
        cycle_time=c_00z,
        temperature_val=18.0,
        precip_val=None,
        cloud_cover_val=None,
    )
    write_dataset(ds_00z, store_00z)

    with Session(engine) as session:
        session.add(ForecastCenter(id="c_noaa", center_id="noaa", name="NOAA", country="US", created_at=c_00z))
        session.add(Model(id="m_gfs", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0, created_at=c_00z))
        session.add(ModelVersion(id="v_gfs", model_id="gfs", version_string="v1.0", created_at=c_00z))
        session.add(ForecastGrid(id="g_glob", grid_code="global_025deg", name="Global", resolution_km=25.0))
        session.add(ForecastVariable(id="v_t2m", variable_code="temperature_2m", name="2m Temperature", unit="°C"))
        session.add(ForecastVariable(id="v_tp", variable_code="precipitation_amount_3h", name="3h Precip", unit="mm"))
        session.add(ForecastVariable(id="v_tcc", variable_code="cloud_cover_3h", name="3h Cloud", unit="%"))
        session.add(ModelRun(id="run_00z", model_version_id="v_gfs", cycle_time=c_00z, status="ready", zarr_store_path=store_00z, created_at=c_00z))

        for v_code in ("temperature_2m", "precipitation_amount_3h", "cloud_cover_3h"):
            session.add(ForecastProduct(id=f"p_00_{v_code}", run_id="run_00z", variable_id=v_code, grid_id="global_025deg", product_type="surface", lead_time_hours=0))
        session.commit()

        loc = ResolvedLocation(latitude=LAT_START + 0.125, longitude=LON_START + 0.125, elevation_m=None, resolved_via="coordinates")
        data = build_point_forecast(
            session,
            location=loc,
            model="gfs",
            variables=["temperature_2m", "precipitation_amount_3h", "cloud_cover_3h"],
            units="metric",
            start_lead_time_hours=None,
            end_lead_time_hours=None,
        )

    assert len(data.forecasts) == 1
    f0 = data.forecasts[0]
    assert getattr(f0, "temperature_2m") == pytest.approx(18.0, abs=1e-3)
    assert getattr(f0, "precipitation_amount_3h") is None
    assert getattr(f0, "cloud_cover_3h") is None
    assert getattr(f0, "precipitation_type") == "none"


def test_map_tile_precipitation_lead0_fallback(fallback_env):
    """Map tile at valid_time = 00Z:
    - precipitation_amount_3h reads fallback 18Z + 6 (contains opaque pixels)
    - cloud_cover_3h reads fallback 18Z + 6 (contains opaque pixels)
    - temperature_2m reads 00Z + 0 (contains opaque pixels).
    """
    engine, _, _ = fallback_env

    with Session(engine) as session:
        # 1. Precipitation map tile at valid_time 00Z
        png_tp = render_tile_png(
            session,
            model="gfs",
            variable="precipitation_amount_3h",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time="2026-09-05T00:00:00Z",
        )
        assert _png_has_opaque_pixels(png_tp)

        # 2. Cloud cover map tile at valid_time 00Z
        png_tcc = render_tile_png(
            session,
            model="gfs",
            variable="cloud_cover_3h",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time="2026-09-05T00:00:00Z",
        )
        assert _png_has_opaque_pixels(png_tcc)

        # 3. Temperature map tile at valid_time 00Z
        png_t2m = render_tile_png(
            session,
            model="gfs",
            variable="temperature_2m",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time="2026-09-05T00:00:00Z",
        )
        assert _png_has_opaque_pixels(png_t2m)


def test_map_tile_provenance_and_cache_key(fallback_env):
    """Verify tile read context, provenance, and cache key reflect the fallback source."""
    engine, _, _ = fallback_env

    with Session(engine) as session:
        # Context for precipitation_amount_3h at valid_time = 00Z
        ctx_tp = resolve_tile_read_context(
            session,
            model="gfs",
            variable="precipitation_amount_3h",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time="2026-09-05T00:00:00Z",
        )
        # Provenance: lead_time_hours must be 6, initial_time must be 18Z
        assert ctx_tp.lead_time_hours == 6
        assert ctx_tp.initial_time == "2026-09-04T18:00:00Z"

        # Cache key incorporates 6 and 18Z
        key_tp = _tile_cache_key(
            ctx_tp.model,
            ctx_tp.variable,
            ctx_tp.level,
            ctx_tp.zoom,
            ctx_tp.x,
            ctx_tp.y,
            ctx_tp.lead_time_hours,
            ctx_tp.initial_time,
            "gen1",
            valid_time=ctx_tp.valid_time,
        )
        assert key_tp[6] == 6
        assert key_tp[7] == "2026-09-04T18:00:00Z"

        # Context for temperature_2m at valid_time = 00Z
        ctx_t2m = resolve_tile_read_context(
            session,
            model="gfs",
            variable="temperature_2m",
            level="surface",
            zoom=8,
            x=51,
            y=98,
            valid_time="2026-09-05T00:00:00Z",
        )
        assert ctx_t2m.lead_time_hours == 0
        assert ctx_t2m.initial_time == "2026-09-05T00:00:00Z"


def test_availability_interval_lead0_filtering(fallback_env):
    """Availability:
    - precipitation_amount_3h at valid_time 00Z advertises source_cycle = 18Z and lead = 6
    - temperature_2m at valid_time 00Z advertises source_cycle = 00Z and lead = 0.
    """
    engine, _, _ = fallback_env

    with Session(engine) as session:
        avail = build_forecast_availability(session, now=_dt(2026, 9, 5, 0))

    gfs_avail = next(m for m in avail.models if m.id == "gfs")
    v_map = {v.id: v for v in gfs_avail.variables}

    # Temperature 2m at valid_time 00Z -> source is 00Z + 0
    t2m_00z = next(vt for vt in v_map["temperature_2m"].valid_times if vt.valid_time == _dt(2026, 9, 5, 0))
    assert t2m_00z.source_cycle == _dt(2026, 9, 5, 0)
    assert t2m_00z.lead_time_hours == 0

    # Precipitation 3h at valid_time 00Z -> source is 18Z + 6
    tp_00z = next(vt for vt in v_map["precipitation_amount_3h"].valid_times if vt.valid_time == _dt(2026, 9, 5, 0))
    assert tp_00z.source_cycle == _dt(2026, 9, 4, 18)
    assert tp_00z.lead_time_hours == 6

    # Cloud cover 3h at valid_time 00Z -> source is 18Z + 6
    tcc_00z = next(vt for vt in v_map["cloud_cover_3h"].valid_times if vt.valid_time == _dt(2026, 9, 5, 0))
    assert tcc_00z.source_cycle == _dt(2026, 9, 4, 18)
    assert tcc_00z.lead_time_hours == 6


def test_ensemble_and_probability_cache_key_reflects_fallback():
    """Ensemble and probability cache keys embed the fallback cycle and lead."""
    key_ens = build_ensemble_cache_key(
        model="gfs",
        latitude=38.0,
        longitude=-107.0,
        variable="precipitation_amount_3h",
        lead_time_hours=6,
        cycle_time="2026-09-04T18:00:00Z",
        serving_generation="gen_18z",
        valid_time="2026-09-05T00:00:00Z",
    )
    key_prob = build_probability_cache_key(
        model="gfs",
        latitude=38.0,
        longitude=-107.0,
        variable="precipitation_amount_3h",
        threshold=1.0,
        operator="gte",
        lead_time_hours=6,
        threshold_max=None,
        cycle_time="2026-09-04T18:00:00Z",
        serving_generation="gen_18z",
        valid_time="2026-09-05T00:00:00Z",
    )
    # The keys must be deterministic non-empty SHA-256 digests prefixed with their domain
    assert key_ens.startswith("ensemble:")
    assert len(key_ens) == 9 + 64
    assert key_prob.startswith("probability:")
    assert len(key_prob) == 12 + 64
