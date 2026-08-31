"""Unit tests for the production ingestion orchestration (parse -> Zarr -> catalog).

``ingest_grib_file`` is the thin runtime boundary that wires the library
modules together. These tests use the committed GRIB2 fixture and a local Zarr
store, and verify that a successful Zarr write is followed by a catalog write
with the run marked ``ready``. The catalog write is exercised against an
in-memory SQLite database so no live PostgreSQL is required.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.base import (
    CycleStoreMismatchError,
    LeadTimeMismatchError,
    LiveStoreOverwriteError,
)
from ingestion.core.catalog import (
    CatalogBase,
    CenterRecord,
    GridRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    RunCatalogSpec,
    VariableRecord,
    VariableSpec,
    record_run,
)
from ingestion.core.pipeline import (
    UnitNormalizationError,
    _apply_variable_mapping,
    _normalize_canonical_units,
    ingest_grib_file,
    read_committed_state,
)
from ingestion.core.zarr_writer import (
    commit_region,
    prepare_run_store,
    read_dataset,
    write_dataset,
)
from ingestion.providers.noaa.parser import GribParsingError

#: Path to the committed GRIB2 fixture, resolved from this file so the tests

#: Path to the committed GRIB2 fixture, resolved from this file so the tests
#: run correctly regardless of the current working directory (root-level CI).
FIXTURE = str(Path(__file__).parent / "fixtures" / "gfs.t00z.pgrb2.0p25.f006.grib2")


def _spec(
    zarr_store_path: str,
    *,
    expected_leads: tuple[int, ...] = (6,),
    expected_members: tuple[int, ...] = (),
    **overrides: object,
) -> RunCatalogSpec:
    return RunCatalogSpec(
        center_id="noaa",
        center_name="National Oceanic and Atmospheric Administration",
        center_country="USA",
        model_id="gfs",
        model_name="Global Forecast System",
        is_ensemble=False,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc),
        grid_id="global_025deg",
        grid_name="Global 0.25 Degree Grid",
        grid_resolution_km=25.0,
        product_type="surface",
        zarr_store_path=zarr_store_path,
        variables=(
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),
            VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h", "prate"),
        ),
        expected_lead_time_hours=expected_leads,
        expected_members=expected_members,
        **overrides,
    )


#: Default cycle time used by synthetic test datasets, matching ``_spec``.
DEFAULT_CYCLE = np.datetime64("2026-07-21T00:00:00")


def _dataset_for_lead(lead: int, cycle=np.datetime64("2026-07-21T00:00:00")):
    """A normalized single-lead dataset for a given lead time.

    Mirrors the parser output: ``temperature_2m`` as a 2-D field on the grid,
    a scalar ``lead_time_hours`` coordinate, and the GRIB ``time`` coordinate
    (the forecast-run cycle/reference time). Carrying ``time`` is essential so
    the cycle-identity guard in ``_merge_lead`` can validate same-cycle merges
    and reject cross-cycle merges (the real parser emits ``time``; the previous
    synthetic datasets omitted it, which is why the cross-cycle defect was
    untested).
    """
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("latitude", "longitude"),
                np.ones((4, 4), dtype=float) * lead,
            )
        },
        coords={
            "time": xr.DataArray(np.datetime64(cycle, "ns"), name="time"),
            "lead_time_hours": lead,
            "latitude": [38.0, 38.25, 38.5, 38.75],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )


@pytest.fixture
def session(tmp_path) -> Session:
    # File-backed SQLite so every connection (including the pipeline's
    # live-store guard) shares the same schema/rows across the test.
    db_file = tmp_path / "catalog.sqlite"
    engine = create_engine(
        f"sqlite:///{db_file}", connect_args={"check_same_thread": False}
    )
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


@pytest.fixture(autouse=True)
def _route_live_store_session(session: Session, monkeypatch) -> None:
    """Route the library path's live-store guard to the test SQLite engine."""
    import ingestion.core.pipeline as P

    monkeypatch.setattr(P, "_live_store_session_factory", lambda: session.bind)


def test_apply_variable_mapping_renames_source_to_platform() -> None:
    """cfgrib-emitted variable names (t2m, prate) map to platform codes."""
    import xarray as xr

    dataset = xr.Dataset(
        {
            "t2m": (("latitude", "longitude"), [[1.0]]),
            "prate": (("latitude", "longitude"), [[2.0]]),
        },
        coords={"latitude": [0.0], "longitude": [0.0]},
    )
    renamed = _apply_variable_mapping(dataset, _spec("/tmp/x").variables)
    assert set(renamed.data_vars) == {"temperature_2m", "precipitation_rate"}


def _dataset_with_units(
    variable_code: str, values, *, source_unit: str
):
    """Build a minimal dataset whose data variable carries a GRIB ``units`` attr.

    Mirrors the real GRIB decode shape (``units`` attribute present) so the
    canonical-unit normalization can be exercised without a binary fixture.
    ``values`` must be a 2×2 nested list matching the coordinate grid.
    """
    import xarray as xr

    return xr.Dataset(
        {
            variable_code: (
                ("latitude", "longitude"),
                values,
                {"units": source_unit},
            )
        },
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )


def test_normalize_temperature_kelvin_to_celsius() -> None:
    """GRIB Kelvin temperature is converted to the canonical °C at ingestion.

    A representative source value of 280 K must become approximately
    6.85 °C (the exact conversion is ``value - 273.15``).
    """
    dataset = _dataset_with_units(
        "temperature_2m", [[280.0, 280.0], [280.0, 280.0]], source_unit="K"
    )
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),)
    normalized = _normalize_canonical_units(dataset, variables)
    value = float(normalized["temperature_2m"].values[0, 0])
    assert value == pytest.approx(6.85, abs=1e-9)
    assert normalized["temperature_2m"].attrs["units"] == "°C"


def test_normalize_temperature_real_fixture_is_celsius() -> None:
    """The real GRIB fixture (native Kelvin) is stored as Celsius.

    The committed fixture decodes to ``t`` values of 280.0..300.0 K. After
    mapping and canonicalization, the Zarr store must hold Celsius values with
    a ``units`` attribute of ``°C`` — never raw Kelvin mislabeled as °C.
    """
    dataset = _dataset_with_units(
        "temperature_2m", [[280.0, 285.0], [290.0, 300.0]], source_unit="K"
    )
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),)
    normalized = _normalize_canonical_units(dataset, variables)
    values = normalized["temperature_2m"].values
    assert float(values.min()) == pytest.approx(280.0 - 273.15, abs=1e-9)
    assert float(values.max()) == pytest.approx(300.0 - 273.15, abs=1e-9)
    assert normalized["temperature_2m"].attrs["units"] == "°C"


def test_normalize_precipitation_rate_kg_m2_s_to_mm_h() -> None:
    """GFS ``prate`` (kg m-2 s-1) is converted to the canonical mm/h at ingestion.

    For liquid water, 1 kg m-2 == 1 mm water-equivalent depth, so the rate
    conversion is ``value × 3600`` (seconds → hours). A representative source
    value of 0.0003 kg m-2 s-1 must become approximately 1.08 mm/h.
    """
    dataset = _dataset_with_units(
        "precipitation_rate",
        [[0.0003, 0.0003], [0.0003, 0.0003]],
        source_unit="kg m-2 s-1",
    )
    variables = (VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h", "prate"),)
    normalized = _normalize_canonical_units(dataset, variables)
    value = float(normalized["precipitation_rate"].values[0, 0])
    assert value == pytest.approx(1.08, abs=1e-9)
    assert normalized["precipitation_rate"].attrs["units"] == "mm/h"


def test_normalize_wind_gust_converts_m_s_to_km_h() -> None:
    """GUST (m/s) is converted to canonical km/h (x3.6) at ingestion."""
    dataset = _dataset_with_units(
        "wind_gust",
        [[10.0, 10.0], [10.0, 10.0]],
        source_unit="m s**-1",
    )
    variables = (VariableSpec("wind_gust", "Wind Gust", "km/h", "gust"),)
    normalized = _normalize_canonical_units(dataset, variables)
    value = float(normalized["wind_gust"].values[0, 0])
    assert value == pytest.approx(36.0, abs=1e-9)
    assert normalized["wind_gust"].attrs["units"] == "km/h"


def test_normalize_relative_humidity_preserves_percent() -> None:
    """RH (%) is preserved as % at ingestion."""
    dataset = _dataset_with_units(
        "relative_humidity_2m",
        [[85.0, 85.0], [85.0, 85.0]],
        source_unit="%",
    )
    variables = (VariableSpec("relative_humidity_2m", "2-Meter Relative Humidity", "%", "r2"),)
    normalized = _normalize_canonical_units(dataset, variables)
    value = float(normalized["relative_humidity_2m"].values[0, 0])
    assert value == pytest.approx(85.0, abs=1e-9)
    assert normalized["relative_humidity_2m"].attrs["units"] == "%"


def test_normalize_visibility_preserves_meters() -> None:
    """VIS (m) is preserved as m at ingestion."""
    dataset = _dataset_with_units(
        "visibility",
        [[10000.0, 10000.0], [10000.0, 10000.0]],
        source_unit="m",
    )
    variables = (VariableSpec("visibility", "Visibility", "m", "vis"),)
    normalized = _normalize_canonical_units(dataset, variables)
    value = float(normalized["visibility"].values[0, 0])
    assert value == pytest.approx(10000.0, abs=1e-9)
    assert normalized["visibility"].attrs["units"] == "m"


def test_normalize_snow_depth_preserves_meters() -> None:
    """SNOD (m) is preserved as m at ingestion."""
    dataset = _dataset_with_units(
        "snow_depth",
        [[0.25, 0.25], [0.25, 0.25]],
        source_unit="m",
    )
    variables = (VariableSpec("snow_depth", "Snow Depth", "m", "sde"),)
    normalized = _normalize_canonical_units(dataset, variables)
    value = float(normalized["snow_depth"].values[0, 0])
    assert value == pytest.approx(0.25, abs=1e-9)
    assert normalized["snow_depth"].attrs["units"] == "m"


def test_normalize_wind_u_and_v_preserves_m_s_and_signed_values() -> None:
    """UGRD/VGRD (m/s) are preserved as canonical m/s including negative signs."""
    dataset = _dataset_with_units(
        "wind_u_10m",
        [[-12.5, 8.4], [-5.0, 15.2]],
        source_unit="m s**-1",
    )
    variables = (VariableSpec("wind_u_10m", "10-Meter U Wind Component", "m/s", "u10"),)
    normalized = _normalize_canonical_units(dataset, variables)
    u_vals = normalized["wind_u_10m"].values
    assert float(u_vals[0, 0]) == pytest.approx(-12.5, abs=1e-9)
    assert float(u_vals[0, 1]) == pytest.approx(8.4, abs=1e-9)
    assert normalized["wind_u_10m"].attrs["units"] == "m/s"

    dataset_v = _dataset_with_units(
        "wind_v_10m",
        [[7.5, -14.2], [0.0, -3.1]],
        source_unit="m/s",
    )
    variables_v = (VariableSpec("wind_v_10m", "10-Meter V Wind Component", "m/s", "v10"),)
    normalized_v = _normalize_canonical_units(dataset_v, variables_v)
    v_vals = normalized_v["wind_v_10m"].values
    assert float(v_vals[0, 1]) == pytest.approx(-14.2, abs=1e-9)
    assert float(v_vals[1, 0]) == pytest.approx(0.0, abs=1e-9)
    assert normalized_v["wind_v_10m"].attrs["units"] == "m/s"


def test_normalize_leaves_already_canonical_values_unchanged() -> None:
    """A variable already in the canonical unit is left numerically untouched."""
    dataset = _dataset_with_units(
        "temperature_2m", [[15.0, 15.0], [15.0, 15.0]], source_unit="°C"
    )
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),)
    normalized = _normalize_canonical_units(dataset, variables)
    assert float(normalized["temperature_2m"].values[0, 0]) == 15.0
    assert normalized["temperature_2m"].attrs["units"] == "°C"


def test_normalize_unknown_source_unit_rejected() -> None:
    """An unsupported source unit fails clearly instead of a silent mislabel."""
    dataset = _dataset_with_units(
        "temperature_2m", [[15.0, 15.0], [15.0, 15.0]], source_unit="°F"
    )
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),)
    with pytest.raises(UnitNormalizationError):
        _normalize_canonical_units(dataset, variables)


def test_normalize_without_units_attr_is_noop() -> None:
    """A synthetic dataset with no GRIB units provenance is left untouched."""
    import xarray as xr

    dataset = xr.Dataset(
        {"temperature_2m": (("latitude", "longitude"), [[15.0, 15.0], [15.0, 15.0]])},
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),)
    normalized = _normalize_canonical_units(dataset, variables)
    assert float(normalized["temperature_2m"].values[0, 0]) == 15.0
    assert "units" not in normalized["temperature_2m"].attrs


def test_ingest_grib_file_parses_writes_and_records(
    session: Session, tmp_path, monkeypatch
) -> None:
    """The full orchestration path runs the GRIB fixture end-to-end."""
    store_path = str(tmp_path / "gfs.zarr")

    recorded: list[ModelRunRecord] = []

    # Route the catalog write into the in-memory SQLite session instead of the
    # configured engine (pipeline imports record_ingested_dataset at module
    # level, so patch it on the pipeline module).
    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        run = record_run(session, spec, dataset, committed_state=committed_state)
        recorded.append(run)
        return run

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    run = ingest_grib_file(_spec(store_path), FIXTURE, store_path)

    assert run.status == "ready"
    assert len(recorded) == 1
    assert recorded[0].status == "ready"

    # Zarr store written and non-empty.
    import os

    assert os.path.isdir(store_path)
    assert os.listdir(store_path)

    # Catalog rows created via the SQLite session.
    assert session.query(ModelRunRecord).count() == 1
    assert session.query(ModelVersionRecord).count() == 1
    assert session.query(ModelRecord).count() == 1
    assert session.query(CenterRecord).count() == 1
    assert session.query(GridRecord).count() == 1
    # The catalog records the spec's platform variables (a registry, model-
    # agnostic), but only ONE product row: the parsed fixture (a single-message
    # t2m file) contains only ``temperature_2m``, and the catalog must never
    # advertise a product for a variable the store does not actually carry
    # (the catalog↔store variable-honesty contract). The spec declares two
    # variables; the store holds one.
    assert session.query(VariableRecord).count() == 2
    assert session.query(ProductRecord).count() == 1


def test_ingest_grib_file_store_uses_platform_variable(tmp_path) -> None:
    """The Zarr store data variable uses the platform code after mapping."""
    store_path = str(tmp_path / "gfs.zarr")

    from ingestion.core.zarr_writer import write_dataset
    from ingestion.providers.noaa.parser import parse_grib2

    dataset = parse_grib2(FIXTURE)
    renamed = _apply_variable_mapping(dataset, _spec(store_path).variables)
    write_dataset(renamed, store_path)

    import xarray as xr

    restored = xr.open_zarr(store_path)
    assert set(restored.data_vars) == {"temperature_2m"}
    assert "t2m" not in restored.data_vars
    assert "t" not in restored.data_vars


#: Path to the committed multi-typeOfLevel regression fixture.
MULTI_FIXTURE = str(Path(__file__).parent / "fixtures" / "gfs_multi_typeoflevel.grib2")


def test_ingest_grib_file_multi_typeoflevel_end_to_end(
    session: Session, tmp_path, monkeypatch
) -> None:
    """The realistic multi-typeOfLevel fixture runs end-to-end to Zarr + catalog.

    Regression for the production failure: an unfiltered open of a real GFS
    ``pgrb2`` file raises ``DatasetBuildError``. The compact multi-typeOfLevel
    fixture reproduces that structure. After the fix, the full ingest path
    (parse -> variable mapping -> canonical units -> Zarr write -> catalog
    metadata) must produce the two platform variables in canonical units.
    """
    store_path = str(tmp_path / "gfs_multi.zarr")

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    run = ingest_grib_file(_spec(store_path), MULTI_FIXTURE, store_path)

    assert run.status == "ready"
    assert session.query(ModelRunRecord).count() == 1
    # Two declared variables -> two VariableRecord + two ProductRecord rows.
    assert session.query(VariableRecord).count() == 2
    assert session.query(ProductRecord).count() == 2

    # The Zarr store holds both platform variables in canonical units.
    import xarray as xr

    restored = xr.open_zarr(store_path)
    assert set(restored.data_vars) == {"temperature_2m", "precipitation_rate"}

    # temperature_2m: K -> °C. Fixture t2m = 280.0 K -> 6.85 °C. The values
    # are stored as float32 in Zarr, so allow float32 rounding tolerance.
    assert restored["temperature_2m"].attrs["units"] == "°C"
    assert float(restored["temperature_2m"].values.flat[0]) == pytest.approx(
        280.0 - 273.15, abs=1e-4
    )

    # precipitation_rate: kg m-2 s-1 -> mm/h. Fixture instant prate = 0.0003.
    assert restored["precipitation_rate"].attrs["units"] == "mm/h"
    assert float(restored["precipitation_rate"].values.flat[0]) == pytest.approx(
        0.0003 * 3600.0, rel=1e-6
    )

    # lead_time_hours survives the merge.
    assert int(restored["lead_time_hours"].values[0]) == 6


def test_ingest_grib_file_corrupt_raises(session: Session, tmp_path) -> None:
    """A corrupt GRIB file fails before any catalog write."""
    corrupt = tmp_path / "bad.grib2"
    corrupt.write_bytes(b"not a grib file")

    with pytest.raises(GribParsingError):
        ingest_grib_file(
            _spec(str(tmp_path / "x.zarr")),
            corrupt,
            str(tmp_path / "x.zarr"),
        )
    assert session.query(ModelRunRecord).count() == 0


def test_ingest_grib_file_requested_lead_matches_succeeds(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A requested lead matching the fixture's decoded lead ingests normally."""
    store_path = str(tmp_path / "gfs.zarr")

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    # The fixture decodes to lead 6 (its GRIB step is +6h).
    run = ingest_grib_file(
        _spec(store_path),
        FIXTURE,
        store_path,
        requested_lead_time_hours=6,
    )
    assert run.status == "ready"
    assert session.query(ModelRunRecord).count() == 1


def test_ingest_grib_file_requested_lead_mismatch_aborts(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A requested lead differing from the fixture's lead aborts, writes nothing."""
    store_path = str(tmp_path / "gfs.zarr")

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    with pytest.raises(LeadTimeMismatchError) as excinfo:
        ingest_grib_file(
            _spec(store_path),
            FIXTURE,
            store_path,
            requested_lead_time_hours=12,  # fixture is lead 6
        )
    message = str(excinfo.value)
    assert "6" in message and "12" in message

    # The mismatch aborts before any Zarr write or catalog row.
    assert session.query(ModelRunRecord).count() == 0
    assert not os.path.isdir(store_path)


def test_ingest_grib_file_merges_leads_into_cycle_store(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A re-ingest of the same cycle store merges the new lead into it (one
    ``model_runs`` row = one cycle = one store containing every lead)."""
    store_path = str(tmp_path / "gfs.zarr")

    recorded: list[ModelRunRecord] = []

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        run = record_run(session, spec, dataset, committed_state=committed_state)
        recorded.append(run)
        return run

    # Each call ingests a different lead (the fixture path is identical, so a
    # call counter yields a distinct lead per ingest).
    _call = {"n": 0}
    _leads = [6, 12]

    def _fake_parse(path):
        lead = _leads[_call["n"] % len(_leads)]
        _call["n"] += 1
        return _dataset_for_lead(lead)

    monkeypatch.setattr("ingestion.core.pipeline.record_ingested_dataset", _record_into_session)
    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _fake_parse)

    # The first library-path ingest creates the run/store. Re-ingestion of a
    # live store via the unlocked library path is now refused (the concurrency
    # protocol requires the coordinator); the second call must raise.
    spec = _spec(store_path, expected_leads=(6, 12))
    first = ingest_grib_file(spec, FIXTURE, store_path)
    assert first.id
    assert first.zarr_store_path == store_path
    # The store is now live (recorded in the SQLite catalog). A second library
    # ingest of the same live store is refused.
    with pytest.raises(LiveStoreOverwriteError):
        ingest_grib_file(spec, FIXTURE, store_path)

    # The store now contains both leads.
    import os
    import xarray as xr

    assert os.path.isdir(store_path)
    restored = xr.open_zarr(store_path)
    assert set(restored.data_vars) == {"temperature_2m"}
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6, 12]


def test_ingest_grib_file_reingest_lead_replaces_in_store(
    session: Session, tmp_path, monkeypatch
) -> None:
    """Re-ingesting the same lead replaces that lead's data in the cycle store."""
    store_path = str(tmp_path / "gfs.zarr")

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, committed_state=committed_state)

    _call = {"n": 0}
    _leads = [6, 12, 6]

    def _fake_parse(path):
        lead = _leads[_call["n"] % len(_leads)]
        _call["n"] += 1
        return _dataset_for_lead(lead)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _fake_parse)

    spec = _spec(store_path, expected_leads=(6, 12))
    ingest_grib_file(spec, FIXTURE, store_path)  # lead 6 creates the live store
    # The store is now live; any further library-path ingest of the same store
    # is refused (the concurrency protocol requires the coordinator).
    with pytest.raises(LiveStoreOverwriteError):
        ingest_grib_file(spec, FIXTURE, store_path)  # lead 12

    import os
    import numpy as np
    import xarray as xr

    assert os.path.isdir(store_path)
    restored = xr.open_zarr(store_path)
    # The store coordinate axis is the pre-allocated [6,12], but only lead 6
    # was committed (lead 12 is NaN because its region was never written).
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6, 12]
    assert np.isnan(
        restored["temperature_2m"].sel(lead_time_hours=12).values[0, 0]
    )


# --- Forecast-run identity / cycle validation (ACCEPTANCE_REMEDIATION_PLAN §3-4) ---


def test_merge_same_cycle_multiple_leads_succeeds(tmp_path) -> None:
    """Same model + same cycle + multiple leads merge into one store."""
    from ingestion.core.pipeline import _merge_lead
    from ingestion.core.zarr_writer import read_dataset, write_dataset

    store = str(tmp_path / "cycle.zarr")
    first = _dataset_for_lead(6)
    write_dataset(first.expand_dims("lead_time_hours"), store)
    merged = _merge_lead(_dataset_for_lead(12), store)
    write_dataset(merged, store)
    restored = read_dataset(store)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6, 12]


def test_merge_different_cycle_fails_fast(tmp_path) -> None:
    """A different-cycle file must be refused, never silently merged.

    This is the exact regression for the reported
    ``MergeError: conflicting values for variable 'time'``: the store is 00Z,
    the incoming file is 12Z. The merge must raise the domain
    ``CycleStoreMismatchError`` and leave the store unchanged.
    """
    from ingestion.core.pipeline import _merge_lead
    from ingestion.core.zarr_writer import read_dataset, write_dataset

    store = str(tmp_path / "cycle.zarr")
    write_dataset(_dataset_for_lead(6).expand_dims("lead_time_hours"), store)
    incoming = _dataset_for_lead(18, cycle=np.datetime64("2026-07-21T12:00:00"))

    with pytest.raises(CycleStoreMismatchError) as excinfo:
        _merge_lead(incoming, store)
    message = str(excinfo.value)
    # The error must communicate both cycles and the refusal.
    assert "2026-07-21T00:00:00" in message
    assert "2026-07-21T12:00:00" in message
    assert "Refusing" in message

    # The store is unchanged after the refused merge: same leads, and its
    # recoverable cycle identity is still the 00Z cycle (via the ``time``
    # coordinate fallback, since the synthetic write carries no ``cycle_time``
    # attr the way the real pipeline does).
    from ingestion.core.pipeline import _resolve_cycle_time

    restored = read_dataset(store)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6]
    assert _resolve_cycle_time(restored) == "2026-07-21T00:00:00"


def test_merge_missing_identity_refused(tmp_path) -> None:
    """A dataset with no cycle identity must not silently merge into a store."""
    import xarray as xr

    from ingestion.core.pipeline import _merge_lead
    from ingestion.core.zarr_writer import write_dataset

    store = str(tmp_path / "cycle.zarr")
    write_dataset(_dataset_for_lead(6).expand_dims("lead_time_hours"), store)

    identityless = xr.Dataset(
        data_vars={
            "temperature_2m": (("latitude", "longitude"), np.ones((4, 4), dtype=float))
        },
        coords={
            "lead_time_hours": 12,
            "latitude": [38.0, 38.25, 38.5, 38.75],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )
    with pytest.raises(CycleStoreMismatchError):
        _merge_lead(identityless, store)


def test_store_is_self_describing_after_ingest(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A written cycle store carries model_id + cycle_time attrs (GAP-3)."""
    import xarray as xr

    store_path = str(tmp_path / "gfs.zarr")

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    # The real fixture decodes to lead 6 at cycle 2026-07-21T00:00Z.
    ingest_grib_file(_spec(store_path), FIXTURE, store_path)

    restored = xr.open_zarr(store_path)
    assert restored.attrs["model_id"] == "gfs"
    assert restored.attrs["cycle_time"] == "2026-07-21T00:00:00"


def test_merge_schema_mismatch_axis_rejected(tmp_path) -> None:
    """A same-cycle lead with a different grid must fail fast (not corrupt)."""
    import xarray as xr

    from ingestion.core.base import StoreSchemaMismatchError
    from ingestion.core.pipeline import _merge_lead
    from ingestion.core.zarr_writer import read_dataset, write_dataset

    store = str(tmp_path / "cycle.zarr")
    write_dataset(_dataset_for_lead(6).expand_dims("lead_time_hours"), store)

    different_grid = xr.Dataset(
        data_vars={
            "temperature_2m": (("latitude", "longitude"), np.ones((2, 2), dtype=float))
        },
        coords={
            "time": xr.DataArray(np.datetime64("2026-07-21T00:00:00"), name="time"),
            "lead_time_hours": 12,
            "latitude": [38.0, 38.5],
            "longitude": [-107.0, -106.5],
        },
    )
    with pytest.raises(StoreSchemaMismatchError):
        _merge_lead(different_grid, store)
    restored = read_dataset(store)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6]


# --- Catalog↔Zarr committed-state consistency (Phase 2A) ---


def _committed_state_spec(
    store: str,
    *,
    expected_leads: tuple[int, ...] = (0, 6, 12, 18),
    **overrides: object,
) -> RunCatalogSpec:
    """A deterministic spec for committed-state tests."""
    return _spec(
        store,
        expected_leads=expected_leads,
        **overrides,
    )


def _committed_dataset(lead: int, value: float = 1.0):
    """A single-lead deterministic dataset (2 variables)."""
    import xarray as xr

    lat = np.array([38.0, 38.25, 38.5, 38.75])
    lon = np.array([-107.0, -106.75, -106.5, -106.25])
    lg, lag, log = np.meshgrid(np.array([float(lead)]), lat, lon, indexing="ij")
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("lead_time_hours", "latitude", "longitude"),
                value + 0.0 * lg,
            ),
            "precipitation_rate": (
                ("lead_time_hours", "latitude", "longitude"),
                0.5 * value + 0.0 * lg,
            ),
        },
        coords={
            "lead_time_hours": [float(lead)],
            "latitude": lat,
            "longitude": lon,
            "time": np.datetime64("2026-07-21T00:00:00", "ns"),
        },
        attrs={"model_id": "gfs", "cycle_time": "2026-07-21T00:00:00"},
    )


def test_committed_state_detects_preallocated_axis_vs_committed(tmp_path) -> None:
    """A preallocated axis is NOT the committed set; NaN regions are excluded."""
    store = str(tmp_path / "prealloc.zarr")
    prepare_run_store(
        _committed_dataset(0),
        store,
        expected_lead_time_hours=(0, 6, 12, 18),
    )
    commit_region(_committed_dataset(0), store)
    commit_region(_committed_dataset(6), store)
    # Coordinate axis is the full preallocated set...
    assert sorted(int(v) for v in read_dataset(store).coords["lead_time_hours"].values) == [
        0,
        6,
        12,
        18,
    ]
    # ...but the committed state is only {0,6}.
    state = read_committed_state(store, is_ensemble=False)
    assert sorted(state.lead_set()) == [0, 6]


def test_healthy_partial_run_is_partial_not_falsely_inconsistent(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A partial run whose catalog matches the committed store is partial, not
    falsely 'inconsistent' merely because the coordinate axis is preallocated."""
    store = str(tmp_path / "partial.zarr")
    sp = _committed_state_spec(store, expected_leads=(0, 6, 12, 18))
    prepare_run_store(
        _committed_dataset(0),
        store,
        expected_lead_time_hours=(0, 6, 12, 18),
    )

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, member=member, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    # Commit only leads 0 and 6 -> committed state {0,6}; expected {0,6,12,18}.
    with Session(session.bind) as s:
        for lead in (0, 6):
            commit_region(_committed_dataset(lead), store)
            record_run(s, sp, _committed_dataset(lead), committed_state=read_committed_state(store, is_ensemble=False))
    # The run is partial (not ready) but NOT falsely inconsistent.
    run = session.query(ModelRunRecord).one()
    assert run.status == "partial"
    # Catalog == committed store ({0,6}), so the store↔catalog gate holds; the
    # non-ready status comes from invocation completeness.
    leads = {p.lead_time_hours for p in session.query(ProductRecord).all()}
    assert leads == {0, 6}


def test_stale_catalog_reconciles_to_committed_store(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A stale catalog superset (catalog {0,6,12,18}, committed store {0,6}) is
    reconciled to the store: stale rows are deleted, run is partial."""
    store = str(tmp_path / "stale_catalog.zarr")
    sp = _committed_state_spec(store, expected_leads=(0, 6, 12, 18))
    prepare_run_store(
        _committed_dataset(0),
        store,
        expected_lead_time_hours=(0, 6, 12, 18),
    )

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, member=member, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    # Simulate a stale catalog: record a full {0,6,12,18} product set (as if the
    # store had all four leads), then commit only {0,6} in the store and re-run.
    # The store actually only holds {0,6} (preallocated axis, committed regions
    # {0,6}): region-commit leads 0 and 6 into the store.
    with Session(session.bind) as s:
        commit_region(_committed_dataset(0), store)
        commit_region(_committed_dataset(6), store)
        # Record a stale catalog superset {0,6,12,18} (as if all four leads had
        # been committed at some earlier point), then re-ingest lead 6 and
        # reconcile against the real store (committed {0,6}).
        for lead in (0, 6, 12, 18):
            record_run(s, sp, _committed_dataset(lead), committed_state=None)
        record_run(s, sp, _committed_dataset(6), committed_state=read_committed_state(store, is_ensemble=False))
    leads = {p.lead_time_hours for p in session.query(ProductRecord).all()}
    assert leads == {0, 6}
    run = session.query(ModelRunRecord).one()
    assert run.status == "partial"


def test_external_shrink_not_hidden_by_subset_readiness(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A store externally shrunk to {6} while catalog claims {0,6,12,18} must be
    partial (the old subset rule alone would keep it ready)."""
    store = str(tmp_path / "shrink.zarr")
    sp = _committed_state_spec(store, expected_leads=(0, 6, 12, 18))
    prepare_run_store(
        _committed_dataset(0),
        store,
        expected_lead_time_hours=(0, 6, 12, 18),
    )

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, member=member, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    # Build a healthy READY {0,6,12,18} catalog.
    with Session(session.bind) as s:
        for lead in (0, 6, 12, 18):
            commit_region(_committed_dataset(lead), store)
            record_run(s, sp, _committed_dataset(lead), committed_state=read_committed_state(store, is_ensemble=False))
    assert session.query(ModelRunRecord).one().status == "ready"
    # External shrink: full overwrite the store to {6} only.
    write_dataset(_committed_dataset(6), store)
    # Re-ingest lead 6 and reconcile against the now-{6} store.
    with Session(session.bind) as s:
        record_run(s, sp, _committed_dataset(6), committed_state=read_committed_state(store, is_ensemble=False))
    run = session.query(ModelRunRecord).one()
    assert run.status == "partial"
    leads = {p.lead_time_hours for p in session.query(ProductRecord).all()}
    assert leads == {6}


def test_healthy_patch_preserves_unrelated_and_stays_ready(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A single-lead PATCH (lead 6) preserves unrelated leads and stays READY.

    This is the mandatory PATCH regression: existing {0,6,12,18}, PATCH 6, final
    {0,6,12,18} and READY, with no unrelated catalog rows deleted.
    """
    store = str(tmp_path / "patch.zarr")
    sp = _committed_state_spec(store, expected_leads=(0, 6, 12, 18))
    prepare_run_store(
        _committed_dataset(0),
        store,
        expected_lead_time_hours=(0, 6, 12, 18),
    )

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, member=member, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    with Session(session.bind) as s:
        for lead in (0, 6, 12, 18):
            commit_region(_committed_dataset(lead), store)
            record_run(s, sp, _committed_dataset(lead), committed_state=read_committed_state(store, is_ensemble=False))
    assert session.query(ModelRunRecord).one().status == "ready"
    # PATCH lead 6 (same expected set).
    with Session(session.bind) as s:
        commit_region(_committed_dataset(6), store)
        record_run(s, sp, _committed_dataset(6), committed_state=read_committed_state(store, is_ensemble=False))
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"
    leads = sorted({p.lead_time_hours for p in session.query(ProductRecord).all()})
    assert leads == [0, 6, 12, 18]


def test_repeated_patch_is_idempotent(
    session: Session, tmp_path, monkeypatch
) -> None:
    """Running the same PATCH twice is idempotent: store and catalog unchanged."""
    store = str(tmp_path / "repeat_patch.zarr")
    sp = _committed_state_spec(store, expected_leads=(0, 6, 12, 18))
    prepare_run_store(
        _committed_dataset(0),
        store,
        expected_lead_time_hours=(0, 6, 12, 18),
    )

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, member=member, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    with Session(session.bind) as s:
        for lead in (0, 6, 12, 18):
            commit_region(_committed_dataset(lead), store)
            record_run(s, sp, _committed_dataset(lead), committed_state=read_committed_state(store, is_ensemble=False))
    for _ in range(2):
        with Session(session.bind) as s:
            commit_region(_committed_dataset(6), store)
            record_run(s, sp, _committed_dataset(6), committed_state=read_committed_state(store, is_ensemble=False))
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"
    leads = sorted({p.lead_time_hours for p in session.query(ProductRecord).all()})
    assert leads == [0, 6, 12, 18]


def test_snapshot_in_memory_validation_catches_schema_mismatch(tmp_path) -> None:
    """_validate_lead_schema_from_snapshot rejects incompatible incoming dimensions or coordinates in-memory."""
    from ingestion.core.coordinator import StoreMetadataSnapshot
    from ingestion.core.pipeline import _validate_lead_schema_from_snapshot
    from ingestion.core.base import StoreSchemaMismatchError

    snapshot = StoreMetadataSnapshot(
        store_path=str(tmp_path / "snap.zarr"),
        generation="gen_test",
        is_ensemble=False,
        data_var_paths=("temperature_2m",),
        lead_index_map={0: 0, 6: 1},
        member_index_map={},
        zarray_by_var={"temperature_2m": {"shape": [2, 4, 4], "chunks": [1, 2, 2]}},
        zattrs_by_var={},
        data_var_dims={"temperature_2m": ("lead_time_hours", "latitude", "longitude")},
        coords_values={"latitude": (38.0, 38.25, 38.5, 38.75), "longitude": (-107.0, -106.75, -106.5, -106.25)},
        grid_shape=(4, 4),
        cycle_time="2026-07-21T00:00:00",
        model_id="gfs",
    )

    # Incompatible grid coordinates (wrong latitude)
    bad_ds = _committed_dataset(6)
    bad_ds = bad_ds.assign_coords(latitude=[10.0, 20.0, 30.0, 40.0])

    with pytest.raises(StoreSchemaMismatchError, match="latitude"):
        _validate_lead_schema_from_snapshot(bad_ds, snapshot, snapshot.store_path)


def test_snapshot_in_memory_validation_catches_cycle_mismatch(tmp_path) -> None:
    """_validate_store_identity_from_snapshot rejects incoming dataset with mismatched cycle time."""
    from ingestion.core.coordinator import StoreMetadataSnapshot
    from ingestion.core.pipeline import _validate_store_identity_from_snapshot
    from ingestion.core.base import CycleStoreMismatchError

    snapshot = StoreMetadataSnapshot(
        store_path=str(tmp_path / "snap.zarr"),
        generation="gen_test",
        is_ensemble=False,
        data_var_paths=("temperature_2m",),
        lead_index_map={0: 0},
        member_index_map={},
        zarray_by_var={},
        zattrs_by_var={},
        data_var_dims={},
        coords_values={},
        grid_shape=(4, 4),
        cycle_time="2026-07-21T00:00:00",
        model_id="gfs",
    )

    ds = _committed_dataset(0)
    ds.attrs["cycle_time"] = "2026-07-22T12:00:00"  # different cycle

    with pytest.raises(CycleStoreMismatchError, match="Refusing to merge"):
        _validate_store_identity_from_snapshot(ds, snapshot, snapshot.store_path)


# --- Phase 1C.1A Precipitation Amount Foundation Tests ---


def test_deaccumulate_precipitation_positive() -> None:
    """Positive 6h minus 3h difference derives exact 3h increment."""
    from ingestion.core.pipeline import deaccumulate_precipitation

    curr = np.array([[5.0, 10.0], [0.0, 3.5]], dtype=np.float32)
    pred = np.array([[2.0, 3.0], [0.0, 1.0]], dtype=np.float32)
    result = deaccumulate_precipitation(curr, pred)
    expected = np.array([[3.0, 7.0], [0.0, 2.5]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_deaccumulate_precipitation_tolerance_clamping() -> None:
    """Small negative residual within tolerance bound [-0.10, 0.0) is clamped to 0.0."""
    from ingestion.core.pipeline import deaccumulate_precipitation

    # Simulates GEFS packing quantization difference where f003 has 0.05 mm and f006 rounds to 0.0 mm
    curr = np.array([[5.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    pred = np.array([[2.0, 0.08], [1.0, 1.5]], dtype=np.float32)
    result = deaccumulate_precipitation(curr, pred, tolerance=0.10)
    expected = np.array([[3.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_deaccumulate_precipitation_rejects_large_negative() -> None:
    """Negative residual exceeding tolerance bound (< -0.10 mm) raises DeaccumulationError."""
    from ingestion.core.base import DeaccumulationError
    from ingestion.core.pipeline import deaccumulate_precipitation

    curr = np.array([[5.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    pred = np.array([[2.0, 0.25], [1.0, 1.5]], dtype=np.float32)  # diff = -0.25 mm < -0.10 mm
    with pytest.raises(DeaccumulationError, match="exceeds tolerance bound"):
        deaccumulate_precipitation(curr, pred, tolerance=0.10)


def test_deaccumulate_precipitation_shape_mismatch() -> None:
    """Shape mismatch between current and predecessor raises DeaccumulationError."""
    from ingestion.core.base import DeaccumulationError
    from ingestion.core.pipeline import deaccumulate_precipitation

    curr = np.array([[5.0, 1.0]], dtype=np.float32)
    pred = np.array([[2.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    with pytest.raises(DeaccumulationError, match="shape mismatch"):
        deaccumulate_precipitation(curr, pred)


def test_read_predecessor_precipitation_deterministic_and_ensemble(tmp_path) -> None:
    """read_predecessor_precipitation extracts correct 2D slice for det and ens stores."""
    import xarray as xr
    from ingestion.core.pipeline import read_predecessor_precipitation
    from ingestion.core.zarr_writer import write_dataset

    # Deterministic store
    det_store = str(tmp_path / "det_precip.zarr")
    ds_det = xr.Dataset(
        data_vars={
            "precipitation_amount_3h": (
                ("lead_time_hours", "latitude", "longitude"),
                np.array([[[1.5, 2.5], [0.0, 4.0]]], dtype=np.float32),
            )
        },
        coords={
            "lead_time_hours": [3],
            "latitude": [10.0, 20.0],
            "longitude": [30.0, 40.0],
        },
    )
    write_dataset(ds_det, det_store)
    slice_det = read_predecessor_precipitation(det_store, 3)
    np.testing.assert_allclose(slice_det, [[1.5, 2.5], [0.0, 4.0]], rtol=1e-5)

    # Ensemble store
    ens_store = str(tmp_path / "ens_precip.zarr")
    ds_ens = xr.Dataset(
        data_vars={
            "precipitation_amount_3h": (
                ("member", "lead_time_hours", "latitude", "longitude"),
                np.array([[[[3.0, 6.0], [1.0, 0.5]]]], dtype=np.float32),
            )
        },
        coords={
            "member": [17],
            "lead_time_hours": [3],
            "latitude": [10.0, 20.0],
            "longitude": [30.0, 40.0],
        },
    )
    write_dataset(ds_ens, ens_store)
    slice_ens = read_predecessor_precipitation(ens_store, 3, member=17)
    np.testing.assert_allclose(slice_ens, [[3.0, 6.0], [1.0, 0.5]], rtol=1e-5)


def test_read_predecessor_precipitation_missing_errors(tmp_path) -> None:
    """Missing store, missing variable, or uncommitted lead raises MissingPredecessorLeadError."""
    import xarray as xr
    from ingestion.core.base import MissingPredecessorLeadError
    from ingestion.core.pipeline import read_predecessor_precipitation
    from ingestion.core.zarr_writer import write_dataset

    # Non-existent store
    with pytest.raises(MissingPredecessorLeadError, match="does not exist"):
        read_predecessor_precipitation(str(tmp_path / "nonexistent.zarr"), 3)

    # Store without precipitation_amount_3h
    store_no_precip = str(tmp_path / "no_precip.zarr")
    ds = xr.Dataset(
        data_vars={"t2m": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={"lead_time_hours": [3], "latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
    )
    write_dataset(ds, store_no_precip)
    with pytest.raises(MissingPredecessorLeadError, match="is missing"):
        read_predecessor_precipitation(store_no_precip, 3)

    # Uncommitted (all NaN) lead
    store_nan = str(tmp_path / "nan_precip.zarr")
    ds_nan = xr.Dataset(
        data_vars={
            "precipitation_amount_3h": (
                ("lead_time_hours", "latitude", "longitude"),
                np.full((1, 2, 2), np.nan, dtype=np.float32),
            )
        },
        coords={"lead_time_hours": [3], "latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
    )
    write_dataset(ds_nan, store_nan)
    with pytest.raises(MissingPredecessorLeadError, match="is uncommitted"):
        read_predecessor_precipitation(store_nan, 3)


def test_normalize_precipitation_increments_lead_zero_nan() -> None:
    """Lead 0 normalizes precipitation_amount_3h to NaN (no interval preceding analysis)."""
    import xarray as xr
    from ingestion.core.pipeline import _normalize_precipitation_increments

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={"lead_time_hours": [0], "latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
    )
    variables = (
        VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),
        VariableSpec("precipitation_amount_3h", "3-Hour Precipitation Amount", "mm", "tp"),
    )
    normalized = _normalize_precipitation_increments(ds, variables)
    assert "tp" in normalized.data_vars
    assert np.all(np.isnan(normalized["tp"].values))


def test_normalize_precipitation_increments_direct_3h() -> None:
    """Lead % 6 == 3 (e.g. lead 3, lead 9) keeps direct upstream accumulation."""
    import xarray as xr
    from ingestion.core.pipeline import _normalize_precipitation_increments

    ds = xr.Dataset(
        data_vars={"tp": (("lead_time_hours", "latitude", "longitude"), np.array([[[2.5, 4.0], [0.0, 1.2]]], dtype=np.float32))},
        coords={"lead_time_hours": [9], "latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
    )
    variables = (VariableSpec("precipitation_amount_3h", "3-Hour Precipitation Amount", "mm", "tp"),)
    normalized = _normalize_precipitation_increments(ds, variables)
    np.testing.assert_allclose(normalized["tp"].values, [[[2.5, 4.0], [0.0, 1.2]]], rtol=1e-5)


def test_normalize_precipitation_increments_differenced_6h() -> None:
    """Lead % 6 == 0 (e.g. lead 6, lead 12) de-accumulates against predecessor array."""
    import xarray as xr
    from ingestion.core.pipeline import _normalize_precipitation_increments

    ds = xr.Dataset(
        data_vars={"tp": (("lead_time_hours", "latitude", "longitude"), np.array([[[6.0, 10.0], [1.0, 3.0]]], dtype=np.float32))},
        coords={"lead_time_hours": [12], "latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
    )
    variables = (VariableSpec("precipitation_amount_3h", "3-Hour Precipitation Amount", "mm", "tp"),)
    pred = np.array([[2.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    normalized = _normalize_precipitation_increments(ds, variables, predecessor_array=pred)
    np.testing.assert_allclose(normalized["tp"].values, [[[4.0, 6.0], [0.0, 2.0]]], rtol=1e-5)


def test_normalize_precipitation_preserves_mm() -> None:
    """Upstream kg m**-2 maps to canonical mm with 1.0 factor."""
    from ingestion.core.pipeline import _normalize_canonical_units

    dataset = _dataset_with_units(
        "precipitation_amount_3h",
        [[5.25, 0.0], [12.1, 3.4]],
        source_unit="kg m**-2",
    )
    variables = (VariableSpec("precipitation_amount_3h", "3-Hour Precipitation Amount", "mm", "tp"),)
    normalized = _normalize_canonical_units(dataset, variables)
    np.testing.assert_allclose(
        normalized["precipitation_amount_3h"].values,
        [[5.25, 0.0], [12.1, 3.4]],
        rtol=1e-5,
    )
    assert normalized["precipitation_amount_3h"].attrs["units"] == "mm"


def test_normalize_categorical_flags_preserves_flag_unit_and_uint8() -> None:
    """Categorical precipitation flags map to flag unit and uint8 values."""
    from ingestion.core.pipeline import _normalize_canonical_units

    for code in ("crain", "csnow", "cfrzr", "cicep"):
        dataset = _dataset_with_units(
            code,
            [[1, 0], [0, 1]],
            source_unit="(Code table 4.222)",
        )
        variables = (VariableSpec(code, f"Categorical {code}", "flag", code),)
        normalized = _normalize_canonical_units(dataset, variables)
        assert normalized[code].attrs["units"] == "flag"
        assert normalized[code].values.dtype == np.uint8
        np.testing.assert_array_equal(normalized[code].values, [[1, 0], [0, 1]])


def test_normalize_categorical_flags_lead_zero_zeros() -> None:
    """Lead 0 normalizes categorical flags to all-zero uint8 arrays."""
    import xarray as xr
    from ingestion.core.pipeline import _normalize_precipitation_increments

    ds = xr.Dataset(
        data_vars={"temperature_2m": (("lead_time_hours", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={"lead_time_hours": [0], "latitude": [10.0, 20.0], "longitude": [30.0, 40.0]},
    )
    variables = (
        VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),
        VariableSpec("crain", "Categorical Rain", "flag", "crain"),
        VariableSpec("csnow", "Categorical Snow", "flag", "csnow"),
    )
    normalized = _normalize_precipitation_increments(ds, variables)
    assert "crain" in normalized.data_vars
    assert "csnow" in normalized.data_vars
    assert normalized["crain"].values.dtype == np.uint8
    np.testing.assert_array_equal(normalized["crain"].values, [[[0, 0], [0, 0]]])

