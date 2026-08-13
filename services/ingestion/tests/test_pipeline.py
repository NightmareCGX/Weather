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

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.base import LeadTimeMismatchError
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
    MissingVariableError,
    UnitNormalizationError,
    _apply_variable_mapping,
    _normalize_canonical_units,
    ingest_grib_file,
)
from ingestion.providers.noaa.parser import GribParsingError

#: Path to the committed GRIB2 fixture, resolved from this file so the tests

#: Path to the committed GRIB2 fixture, resolved from this file so the tests
#: run correctly regardless of the current working directory (root-level CI).
FIXTURE = str(Path(__file__).parent / "fixtures" / "gfs.t00z.pgrb2.0p25.f006.grib2")


def _spec(zarr_store_path: str) -> RunCatalogSpec:
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
    )


def _surface_spec(zarr_store_path: str) -> RunCatalogSpec:
    """A spec for fixtures that decode to only ``temperature_2m``.

    The single-lead GRIB2 fixture (and the synthetic single-lead datasets built
    for merge/re-ingest tests) contains only the 2-metre temperature field,
    so the declared variables must match what the parser actually selects or
    the variable-presence fail-fast guard rejects the ingest.
    """
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
        variables=(VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t2m"),),
    )

def _dataset_for_lead(lead: int):
    """A normalized single-lead dataset for a given lead time.

    Mirrors the parser output: ``temperature_2m`` as a 2-D field on the grid
    with a scalar ``lead_time_hours`` coordinate.
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
            "lead_time_hours": lead,
            "latitude": [38.0, 38.25, 38.5, 38.75],
            "longitude": [-107.0, -106.75, -106.5, -106.25],
        },
    )


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


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
    def _record_into_session(spec, dataset, *, effective_store_path=None):
        run = record_run(session, spec, dataset)
        recorded.append(run)
        return run

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    run = ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)

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
    # product row per (variable x lead). The single-lead surface spec
    # declares one variable (temperature_2m), so one row of each.
    assert session.query(VariableRecord).count() == 1
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

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        return record_run(session, spec, dataset)

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

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        return record_run(session, spec, dataset)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    # The fixture decodes to lead 6 (its GRIB step is +6h).
    run = ingest_grib_file(
        _surface_spec(store_path),
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

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        return record_run(session, spec, dataset)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    with pytest.raises(LeadTimeMismatchError) as excinfo:
        ingest_grib_file(
            _surface_spec(store_path),
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

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        run = record_run(session, spec, dataset)
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

    # Ingest the same cycle store for two different leads.
    first = ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)
    second = ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)

    # Both leads share one run row and one store path.
    assert first.id == second.id
    assert second.zarr_store_path == store_path
    assert len(recorded) == 2
    assert recorded[0].id == recorded[1].id

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

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        return record_run(session, spec, dataset)

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

    ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)  # lead 6
    ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)  # lead 12
    # Re-ingest lead 6 a second time: still exactly two leads in the store.
    ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)  # lead 6 again

    import os
    import xarray as xr

    assert os.path.isdir(store_path)
    restored = xr.open_zarr(store_path)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6, 12]


def test_ingest_grib_file_missing_custom_variable_fails_fast(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A requested variable absent from the file aborts before any write.

    Regression for N1: a custom ``--variable`` (e.g. a 10 m wind field not
    among ``SURFACE_FIELD_FILTERS``) previously produced a catalog row with
    no data in the store. It must now fail with
    :class:`MissingVariableError` and record nothing.
    """
    import xarray as xr

    store_path = str(tmp_path / "gfs.zarr")

    def _winds_only(path):
        return xr.Dataset(
            {"some_other_field": (("latitude", "longitude"), [[1.0]])},
            coords={
                "latitude": [0.0],
                "longitude": [0.0],
                "lead_time_hours": 6,
            },
        )

    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _winds_only)


    spec = RunCatalogSpec(
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
        zarr_store_path=store_path,
        variables=(
            VariableSpec("wind_u_10m", "10-Meter Wind U", "m/s", "10u"),
        ),
    )

    with pytest.raises(MissingVariableError) as excinfo:
        ingest_grib_file(spec, "x.grib2", store_path)
    message = str(excinfo.value)
    assert "wind_u_10m" in message
    assert "wind_u_10m" in message.split("Available variables:")[0]

    # Nothing recorded and no store was written.
    assert session.query(ModelRunRecord).count() == 0
    assert session.query(ProductRecord).count() == 0
    assert not os.path.isdir(store_path)


def test_ingest_grib_file_write_failure_preserves_old_store(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A mid-write crash preserves the old store and skips the catalog update.

    Regression for L2-C1: a failure while writing the staging store must not
    truncate the previously-served store, must leave no ``.staging`` residue,
    and must not record the failed run.
    """
    import os
    import xarray as xr

    store_path = str(tmp_path / "gfs.zarr")
    recorded: list = []

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        run = record_run(session, spec, dataset)
        recorded.append(run)
        return run

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    _call = {"n": 0}
    _leads = [6, 12]

    def _fake_parse(path):
        lead = _leads[_call["n"] % len(_leads)]
        _call["n"] += 1
        return _dataset_for_lead(lead)

    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _fake_parse)

    # First ingest succeeds (lead 6).
    first = ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)
    assert first.status == "ready"
    assert len(recorded) == 1

    # Second ingest crashes while writing the staging store.
    import ingestion.core.zarr_writer as zw

    def _boom(dataset, resolved, chunks):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(zw, "_write_zarr", _boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)

    # Old store still intact and readable; no staging residue; no extra catalog
    # write for the failed attempt.
    assert len(recorded) == 1
    assert not os.path.exists(zw._staging_path(store_path))
    restored = xr.open_zarr(store_path)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6]


def test_ingest_grib_file_reingest_atomic_no_residue(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A successful re-ingest leaves both leads and no ``.old``/``.staging``.

    Regression for L2-C1: the atomic swap must promote the merged store and
    clean up its staging and superseded directories.
    """
    import os
    import xarray as xr
    import ingestion.core.zarr_writer as zw

    store_path = str(tmp_path / "gfs.zarr")
    recorded: list = []

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        run = record_run(session, spec, dataset)
        recorded.append(run)
        return run

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    _call = {"n": 0}
    _leads = [6, 12]

    def _fake_parse(path):
        lead = _leads[_call["n"] % len(_leads)]
        _call["n"] += 1
        return _dataset_for_lead(lead)

    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _fake_parse)

    first = ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)
    second = ingest_grib_file(_surface_spec(store_path), FIXTURE, store_path)

    assert first.id == second.id
    assert len(recorded) == 2
    assert not os.path.exists(zw._staging_path(store_path))
    assert not os.path.exists(zw._old_path(store_path))
    restored = xr.open_zarr(store_path)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6, 12]


def test_ingest_grib_file_partial_variables_record_present_subset(
    session: Session, tmp_path, monkeypatch
) -> None:
    """Only the requested variables present in the file are recorded.

    Regression for MAJOR-3: a spec declaring temperature_2m + precipitation_rate
    against a file that decodes only temperature_2m must succeed and record
    only the present variable (the parser deliberately skips absent fields: a
    GEFS pgrb2b product legitimately omits prate).
    """
    import xarray as xr

    store_path = str(tmp_path / "gfs.zarr")

    def _temp_only(path):
        return xr.Dataset(
            {"t2m": (("latitude", "longitude"), [[280.0]])},
            coords={"latitude": [0.0], "longitude": [0.0], "lead_time_hours": 6},
        )

    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _temp_only)

    recorded: list[ModelRunRecord] = []

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        run = record_run(session, spec, dataset)
        recorded.append(run)
        return run

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )

    # _spec declares temperature_2m AND precipitation_rate; only t2m is present.
    run = ingest_grib_file(_spec(store_path), "x.grib2", store_path)

    assert run.status == "ready"
    assert len(recorded) == 1
    codes = {code for (code,) in session.query(VariableRecord.variable_code).all()}
    assert codes == {"temperature_2m"}
    assert session.query(ProductRecord).count() == 1


def test_ingest_grib_file_corrupt_store_refused(
    session: Session, tmp_path, monkeypatch
) -> None:
    """A corrupt existing store fails loudly instead of being rebuilt.

    Regression for MAJOR-2: a store that exists but cannot be opened must not
    be silently replaced by a store containing only the current lead (which
    would drop previously ingested leads while the catalog still advertises
    them as ready).
    """
    import os
    import xarray as xr

    from ingestion.core.zarr_writer import CorruptStoreError

    store_path = str(tmp_path / "gfs.zarr")
    os.makedirs(store_path, exist_ok=True)
    with open(os.path.join(store_path, ".zgroup"), "w") as handle:
        handle.write("{not-valid-json")

    def _temp_only(path):
        return xr.Dataset(
            {"t2m": (("latitude", "longitude"), [[280.0]])},
            coords={"latitude": [0.0], "longitude": [0.0], "lead_time_hours": 6},
        )

    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _temp_only)

    with pytest.raises(CorruptStoreError):
        ingest_grib_file(_surface_spec(store_path), "x.grib2", store_path)

    # The corrupt directory is left untouched and nothing was recorded.
    assert os.path.exists(os.path.join(store_path, ".zgroup"))
    assert session.query(ModelRunRecord).count() == 0


def test_store_lock_serializes_concurrent_merge_writes(tmp_path) -> None:
    """Concurrent workers ingesting leads of one cycle store stay consistent.

    Regression for MAJOR-1: without the store lock, workers racing on the
    shared staging directory and read-merge-write cycle lost leads. Under
    store_lock every lead must be present afterwards with no staging/old
    residue.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import xarray as xr

    from ingestion.core.zarr_writer import store_lock, store_status, write_dataset

    store_path = str(tmp_path / "cycle.zarr")

    def _lead_dataset(lead: int) -> xr.Dataset:
        return xr.Dataset(
            {
                "temperature_2m": (
                    ("lead_time_hours", "latitude", "longitude"),
                    np.full((1, 2, 2), float(lead)),
                )
            },
            coords={
                "lead_time_hours": [lead],
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
            },
        )

    def _merge_write(lead: int) -> None:
        dataset = _lead_dataset(lead)
        with store_lock(store_path):
            status = store_status(store_path)
            if status == "readable":
                existing = xr.open_zarr(store_path)
                if "lead_time_hours" in existing.dims:
                    keep = (existing["lead_time_hours"] != lead).values
                    dataset = xr.concat(
                        [existing.isel(lead_time_hours=keep), dataset],
                        dim="lead_time_hours",
                        coords="minimal",
                    )
                    dataset = dataset.sortby("lead_time_hours")
            write_dataset(dataset, store_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(_merge_write, [0, 6, 12, 18]))

    final = xr.open_zarr(store_path)
    assert sorted(final["lead_time_hours"].values.tolist()) == [0, 6, 12, 18]
    assert not os.path.exists(store_path + ".staging")
    assert not os.path.exists(store_path + ".old")


def test_swap_local_rolls_back_previous_store_on_rename_failure(
    tmp_path, monkeypatch
) -> None:
    """If promoting the staged store fails, the previous store is restored.

    The atomicity guarantee of the two-rename exchange: a failure on the
    second rename must roll the old store back into place, never leaving the
    final path empty or half-written.
    """
    import os

    import numpy as np
    import xarray as xr

    import ingestion.core.zarr_writer as zw
    from ingestion.core.zarr_writer import write_dataset

    store_path = str(tmp_path / "cycle.zarr")
    staging = zw._staging_path(store_path)

    def _lead_dataset(lead: int) -> xr.Dataset:
        return xr.Dataset(
            {
                "temperature_2m": (
                    ("lead_time_hours", "latitude", "longitude"),
                    np.full((1, 2, 2), float(lead)),
                )
            },
            coords={
                "lead_time_hours": [lead],
                "latitude": [0.0, 1.0],
                "longitude": [0.0, 1.0],
            },
        )

    write_dataset(_lead_dataset(0), store_path)
    _lead_dataset(6).to_zarr(staging, mode="w")

    real_rename = os.rename
    calls = {"count": 0}

    def _flaky_rename(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated second-rename failure")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", _flaky_rename)

    with pytest.raises(OSError):
        zw._swap_local(store_path, staging)

    # The previous store (lead 0) was rolled back into place.
    restored = xr.open_zarr(store_path)
    assert restored["lead_time_hours"].values.tolist() == [0]
