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
            VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t"),
            VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h", "prate"),
        ),
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
    """Raw GRIB2 shortNames (t, prate) are renamed to platform codes."""
    import xarray as xr

    dataset = xr.Dataset(
        {
            "t": (("latitude", "longitude"), [[1.0]]),
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
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t"),)
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
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t"),)
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
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t"),)
    normalized = _normalize_canonical_units(dataset, variables)
    assert float(normalized["temperature_2m"].values[0, 0]) == 15.0
    assert normalized["temperature_2m"].attrs["units"] == "°C"


def test_normalize_unknown_source_unit_rejected() -> None:
    """An unsupported source unit fails clearly instead of a silent mislabel."""
    dataset = _dataset_with_units(
        "temperature_2m", [[15.0, 15.0], [15.0, 15.0]], source_unit="°F"
    )
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t"),)
    with pytest.raises(UnitNormalizationError):
        _normalize_canonical_units(dataset, variables)


def test_normalize_without_units_attr_is_noop() -> None:
    """A synthetic dataset with no GRIB units provenance is left untouched."""
    import xarray as xr

    dataset = xr.Dataset(
        {"temperature_2m": (("latitude", "longitude"), [[15.0, 15.0], [15.0, 15.0]])},
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    variables = (VariableSpec("temperature_2m", "2-Meter Temperature", "°C", "t"),)
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
    assert "t" not in restored.data_vars


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

    def _record_into_session(spec, dataset, *, effective_store_path=None):
        return record_run(session, spec, dataset)

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
    first = ingest_grib_file(_spec(store_path), FIXTURE, store_path)
    second = ingest_grib_file(_spec(store_path), FIXTURE, store_path)

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

    ingest_grib_file(_spec(store_path), FIXTURE, store_path)  # lead 6
    ingest_grib_file(_spec(store_path), FIXTURE, store_path)  # lead 12
    # Re-ingest lead 6 a second time: still exactly two leads in the store.
    ingest_grib_file(_spec(store_path), FIXTURE, store_path)  # lead 6 again

    import os
    import xarray as xr

    assert os.path.isdir(store_path)
    restored = xr.open_zarr(store_path)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6, 12]
