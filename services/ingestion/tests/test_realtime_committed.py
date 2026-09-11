"""Offline SQLite tests for the durable committed-state reader (Phase 5C)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from domain.horizon import CANONICAL_MAX_LEAD_HOURS
from ingestion.core.catalog import (
    CatalogBase,
    EnsembleMemberProductRecord,
    ForecastCycleLifecycleRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
)
from ingestion.realtime.committed import (
    discover_incomplete_historical_cycles,
    is_cycle_durably_complete,
    is_cycle_retired_or_deleted,
    read_cycle_committed_state,
)

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


def test_is_cycle_durably_complete_requires_both_models_ready(
    catalog_engine,
) -> None:
    cycle = datetime(2026, 7, 21, 6, 0, tzinfo=timezone.utc)
    with Session(catalog_engine) as session:
        session.add(
            ModelVersionRecord(
                id="v_gfs", model_id="gfs", version_string="v1.0"
            )
        )
        session.add(
            ModelVersionRecord(
                id="v_gefs", model_id="gefs", version_string="v1.0"
            )
        )
        # GFS ready, GEFS partial
        session.add(
            ModelRunRecord(
                id="run_gfs_6z",
                model_version_id="v_gfs",
                cycle_time=cycle,
                status="ready",
            )
        )
        session.add(
            ModelRunRecord(
                id="run_gefs_6z",
                model_version_id="v_gefs",
                cycle_time=cycle,
                status="partial",
            )
        )
        session.commit()

    assert not is_cycle_durably_complete(catalog_engine, cycle_time=cycle)

    with Session(catalog_engine) as session:
        gefs_run = session.get(ModelRunRecord, "run_gefs_6z")
        assert gefs_run is not None
        gefs_run.status = "ready"
        session.commit()

    assert is_cycle_durably_complete(catalog_engine, cycle_time=cycle)


def test_discover_incomplete_historical_cycles_filters(catalog_engine) -> None:
    now_utc = datetime(2026, 7, 21, 19, 0, tzinfo=timezone.utc)
    c18 = datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc)
    c12 = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    c06 = datetime(2026, 7, 21, 6, 0, tzinfo=timezone.utc)
    c00 = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)
    c_fenced = datetime(2026, 7, 20, 18, 0, tzinfo=timezone.utc)

    with Session(catalog_engine) as session:
        session.add(
            ModelVersionRecord(
                id="ver_gfs", model_id="gfs", version_string="v1.0"
            )
        )
        session.add(
            ModelVersionRecord(
                id="ver_gefs", model_id="gefs", version_string="v1.0"
            )
        )

        # c18: Active cycle (should be excluded by < active_cycle_time)
        session.add(
            ModelRunRecord(
                id="r_gfs_18",
                model_version_id="ver_gfs",
                cycle_time=c18,
                status="partial",
            )
        )

        # c12: Incomplete (GFS partial, GEFS missing) -> included
        session.add(
            ModelRunRecord(
                id="r_gfs_12",
                model_version_id="ver_gfs",
                cycle_time=c12,
                status="partial",
            )
        )

        # c06: Incomplete (GFS partial, GEFS partial) -> included
        session.add(
            ModelRunRecord(
                id="r_gfs_06",
                model_version_id="ver_gfs",
                cycle_time=c06,
                status="partial",
            )
        )
        session.add(
            ModelRunRecord(
                id="r_gefs_06",
                model_version_id="ver_gefs",
                cycle_time=c06,
                status="partial",
            )
        )

        # c00: Complete (both ready) -> excluded
        session.add(
            ModelRunRecord(
                id="r_gfs_00",
                model_version_id="ver_gfs",
                cycle_time=c00,
                status="ready",
            )
        )
        session.add(
            ModelRunRecord(
                id="r_gefs_00",
                model_version_id="ver_gefs",
                cycle_time=c00,
                status="ready",
            )
        )

        # c_fenced: Incomplete but lifecycle deletion started -> excluded
        session.add(
            ModelRunRecord(
                id="r_gfs_fen",
                model_version_id="ver_gfs",
                cycle_time=c_fenced,
                status="partial",
            )
        )
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c_fenced,
                deletion_started_at=now_utc,
            )
        )

        session.commit()

    candidates = discover_incomplete_historical_cycles(
        catalog_engine,
        active_cycle_time=c18,
        now_utc=now_utc,
    )
    # Sorted descending by cycle_time: c12 then c06
    assert [c.cycle_time for c in candidates] == [c12, c06]


def test_discover_incomplete_historical_cycles_horizon_bounded_not_48h(
    catalog_engine,
) -> None:
    now_utc = datetime(2026, 7, 26, 0, 0, tzinfo=timezone.utc)
    c_active = datetime(2026, 7, 25, 18, 0, tzinfo=timezone.utc)
    # 72 hours ago (3 days) -> still well within 240h (10 days) horizon!
    c_72h = datetime(2026, 7, 22, 18, 0, tzinfo=timezone.utc)
    # 243 hours ago (> 240h before serving_start) -> expired!
    c_expired = now_utc - timedelta(hours=CANONICAL_MAX_LEAD_HOURS + 6)

    with Session(catalog_engine) as session:
        session.add(
            ModelVersionRecord(
                id="v_gfs_h", model_id="gfs", version_string="v1.0"
            )
        )
        session.add(
            ModelRunRecord(
                id="r_72h",
                model_version_id="v_gfs_h",
                cycle_time=c_72h,
                status="partial",
            )
        )
        session.add(
            ModelRunRecord(
                id="r_exp",
                model_version_id="v_gfs_h",
                cycle_time=c_expired,
                status="partial",
            )
        )
        session.commit()

    candidates = discover_incomplete_historical_cycles(
        catalog_engine,
        active_cycle_time=c_active,
        now_utc=now_utc,
    )
    candidate_times = [c.cycle_time for c in candidates]
    # 72h-old cycle is included because cycle + 240h >= serving_start
    assert c_72h in candidate_times
    # Expired cycle is excluded
    assert c_expired not in candidate_times


def test_discover_incomplete_historical_cycles_zero_evidence_not_synthesized(
    catalog_engine,
) -> None:
    now_utc = datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc)
    c_active = datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc)
    # Zero rows seeded in database!
    candidates = discover_incomplete_historical_cycles(
        catalog_engine,
        active_cycle_time=c_active,
        now_utc=now_utc,
    )
    # Never synthesizes nominal cycles from thin air
    assert candidates == []


def test_discover_incomplete_historical_cycles_excludes_retired_cycle(
    catalog_engine,
) -> None:
    """A partial cycle with retired_at != NULL must be excluded from recovery candidates,

    while its committed data remains completely intact and unretired control cycles remain eligible.
    """
    now_utc = datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc)
    c_active = datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc)
    c_retired = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    c_control = datetime(2026, 7, 21, 6, 0, tzinfo=timezone.utc)

    with Session(catalog_engine) as session:
        session.add(
            ModelVersionRecord(
                id="ver_gfs_r", model_id="gfs", version_string="v1.0"
            )
        )
        session.add(
            ModelVersionRecord(
                id="ver_gefs_r", model_id="gefs", version_string="v1.0"
            )
        )

        # c_retired: Partial, retired_at is set, deletion_started_at is None, deleted_at is None
        session.add(
            ModelRunRecord(
                id="r_gfs_ret",
                model_version_id="ver_gfs_r",
                cycle_time=c_retired,
                status="partial",
            )
        )
        session.add(
            ProductRecord(
                id="p_gfs_ret_0",
                run_id="r_gfs_ret",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=c_retired,
                retired_at=now_utc,
                deletion_started_at=None,
                deleted_at=None,
            )
        )

        # c_control: Partial, retired_at is None, deletion_started_at is None, deleted_at is None
        session.add(
            ModelRunRecord(
                id="r_gfs_ctrl",
                model_version_id="ver_gfs_r",
                cycle_time=c_control,
                status="partial",
            )
        )
        session.add(
            ProductRecord(
                id="p_gfs_ctrl_0",
                run_id="r_gfs_ctrl",
                variable_id="temperature_2m",
                grid_id="global_025deg",
                product_type="surface",
                lead_time_hours=0,
            )
        )

        session.commit()

    # 1. Candidate discovery excludes c_retired and includes c_control
    candidates = discover_incomplete_historical_cycles(
        catalog_engine,
        active_cycle_time=c_active,
        now_utc=now_utc,
    )
    candidate_times = [c.cycle_time for c in candidates]
    assert c_retired not in candidate_times
    assert c_control in candidate_times

    # 2. Lifecycle query confirms status
    with Session(catalog_engine) as session:
        assert is_cycle_retired_or_deleted(session, c_retired) is True
        assert is_cycle_retired_or_deleted(session, c_control) is False

    # 3. Existing committed state for retired cycle remains completely intact!
    gfs_ret, _ = read_cycle_committed_state(catalog_engine, cycle_time=c_retired)
    assert gfs_ret.leads == frozenset({0})

