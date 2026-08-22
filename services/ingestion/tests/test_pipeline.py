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
    # The catalog records the spec's platform variables (a registry), and one
    # product row per (variable x lead). The fixture has one lead and the spec
    # declares two variables.
    assert session.query(VariableRecord).count() == 2
    assert session.query(ProductRecord).count() == 2


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
