"""Offline SQLite tests for the durable committed-state reader (Phase 5C)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    CatalogBase,
    EnsembleMemberProductRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
)
from ingestion.realtime.committed import read_cycle_committed_state

CYCLE = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)
MEMBERS = tuple(range(1, 31))


@pytest.fixture()
def catalog_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    CatalogBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


def _seed_run(
    session: Session,
    *,
    model_id: str,
    version_id: str,
    run_id: str,
    leads: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...] = (),
) -> None:
    session.add(ModelVersionRecord(id=version_id, model_id=model_id, version_string="v1.0"))
    session.add(
        ModelRunRecord(
            id=run_id,
            model_version_id=version_id,
            cycle_time=CYCLE,
            status="partial",
            zarr_store_path=f"s3://weather-data/{model_id}/2026-07-21/00/cycle.zarr",
        )
    )
    session.flush()
    for lead in leads:
        session.add(
            ProductRecord(
                id=f"p_{model_id}_{lead}",
                run_id=run_id,
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=lead,
            )
        )
    for member, lead in pairs:
        session.add(
            EnsembleMemberProductRecord(
                id=f"emp_{model_id}_{member}_{lead}",
                run_id=run_id,
                member_index=member,
                lead_time_hours=lead,
            )
        )
    session.commit()


def test_reads_committed_leads_and_pairs_per_model(catalog_engine) -> None:
    with Session(catalog_engine) as session:
        _seed_run(
            session,
            model_id="gfs",
            version_id="version_gfs_v1.0",
            run_id="run_gfs",
            leads=(0, 3, 6),
        )
        _seed_run(
            session,
            model_id="gefs",
            version_id="version_gefs_v1.0",
            run_id="run_gefs",
            leads=(0, 3),
            pairs=tuple((m, lead) for lead in (0, 3) for m in MEMBERS),
        )

    gfs, gefs = read_cycle_committed_state(catalog_engine, cycle_time=CYCLE)
    assert gfs.leads == frozenset({0, 3, 6})
    assert gfs.pairs == frozenset()
    # GFS committed ahead of GEFS: both are read independently.
    assert gefs.pairs == frozenset((m, lead) for lead in (0, 3) for m in MEMBERS)
    assert gefs.is_lead_committed(0, ensemble=True, expected_members=MEMBERS)
    assert gefs.is_lead_committed(6, ensemble=True, expected_members=MEMBERS) is False


def test_unknown_cycle_or_model_returns_empty_state(catalog_engine) -> None:
    gfs, gefs = read_cycle_committed_state(catalog_engine, cycle_time=CYCLE)
    assert gfs.leads == frozenset() and gfs.pairs == frozenset()
    assert gefs.leads == frozenset() and gefs.pairs == frozenset()

    other_cycle = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
    with Session(catalog_engine) as session:
        _seed_run(
            session,
            model_id="gfs",
            version_id="version_gfs_v1.0",
            run_id="run_gfs22",
            leads=(0,),
        )
    # A different cycle has no run row → empty state.
    gfs2, _ = read_cycle_committed_state(catalog_engine, cycle_time=other_cycle)
    assert gfs2.leads == frozenset()
    # The seeded cycle is found.
    gfs3, _ = read_cycle_committed_state(catalog_engine, cycle_time=CYCLE)
    assert gfs3.leads == frozenset({0})


def test_version_string_scopes_the_run_lookup(catalog_engine) -> None:
    with Session(catalog_engine) as session:
        session.add(
            ModelVersionRecord(id="version_gfs_v2.0", model_id="gfs", version_string="v2.0")
        )
        session.add(
            ModelRunRecord(
                id="run_gfs_v2",
                model_version_id="version_gfs_v2.0",
                cycle_time=CYCLE,
                status="ready",
            )
        )
        session.commit()
    # Default version_string v1.0 does not see the v2.0 run.
    gfs, _ = read_cycle_committed_state(catalog_engine, cycle_time=CYCLE)
    assert gfs.leads == frozenset()
    gfs_v2, _ = read_cycle_committed_state(
        catalog_engine, cycle_time=CYCLE, version_string="v2.0"
    )
    assert gfs_v2.leads == frozenset()
