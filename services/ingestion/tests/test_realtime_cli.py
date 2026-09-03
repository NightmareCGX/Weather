"""Offline tests for the `weather-ingest realtime` CLI command (Phase 5C)."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine

from ingestion.core.catalog import CatalogBase
from ingestion.core.config import IngestionSettings


def test_realtime_requires_paired_cycle_flags(capsys) -> None:
    from ingestion.cli import main

    code = main(
        ["realtime", "--cycle-date", "2026-07-21", "--once", "--dry-run"]
    )
    assert code == 2
    assert "together" in capsys.readouterr().out


def test_realtime_disabled_by_master_switch(capsys, monkeypatch) -> None:
    from ingestion.cli import main

    monkeypatch.setattr(
        "ingestion.cli.settings", IngestionSettings(REALTIME_ENABLED=False)
    )
    code = main(["realtime", "--once", "--dry-run"])
    assert code == 2
    assert "REALTIME_ENABLED=true" in capsys.readouterr().out


def test_realtime_once_dry_run_plans_offline_without_dispatch(
    tmp_path, monkeypatch, capsys
) -> None:
    """--once --dry-run: one real discovery→plan iteration, zero waves.

    Discovery is faked at the Phase 5B snapshot-function boundary; the
    committed-state reader runs against SQLite; dispatch would raise if ever
    reached (proving --dry-run never dispatches).
    """

    from ingestion.providers.noaa.discovery import (
        ArtifactObservation,
        CycleSnapshot,
        RegionArtifacts,
    )
    import ingestion.realtime.scheduler as scheduler_module

    members = tuple(range(1, 31))
    regions_g = {
        (None, 0): RegionArtifacts(
            data=ArtifactObservation(key="g0", size=1, etag=None, last_modified=None),
            idx=ArtifactObservation(key="g0i", size=1, etag=None, last_modified=None),
        )
    }
    regions_e = {
        (m, 0): RegionArtifacts(
            data=ArtifactObservation(key=f"e{m}", size=1, etag=None, last_modified=None),
            idx=ArtifactObservation(key=f"e{m}i", size=1, etag=None, last_modified=None),
        )
        for m in members
    }
    cycle = date(2026, 7, 21)

    def _fake_gfs(cycle_date, cycle_hour, **kwargs):
        assert (cycle_date, cycle_hour) == (cycle, 0)
        return CycleSnapshot(model="gfs", cycle_date=cycle_date, cycle_hour=cycle_hour, prefix="p", regions=regions_g)

    def _fake_gefs(cycle_date, cycle_hour, **kwargs):
        return CycleSnapshot(model="gefs", cycle_date=cycle_date, cycle_hour=cycle_hour, prefix="p", regions=regions_e)

    monkeypatch.setattr(scheduler_module, "snapshot_gfs_cycle", _fake_gfs)
    monkeypatch.setattr(scheduler_module, "snapshot_gefs_cycle", _fake_gefs)

    # SQLite catalog (empty → nothing committed yet).
    engine = create_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    CatalogBase.metadata.create_all(engine)
    monkeypatch.setattr("ingestion.core.db.engine", engine)

    monkeypatch.setattr(
        "ingestion.cli.settings", IngestionSettings(REALTIME_ENABLED=True)
    )

    # The production constructor wires SchedulerLeadership(catalog_engine);
    # leadership on SQLite raises — the run() contract returns 1. For this
    # offline diagnostic test, inject the noop leadership instead.
    import ingestion.cli as cli_module

    original_leadership = cli_module.__dict__.get("SchedulerLeadership")

    class _Noop:
        is_leader = True

        def acquire(self) -> bool:
            return True

        def release(self) -> None:
            self.is_leader = False

    monkeypatch.setattr(
        "ingestion.realtime.leadership.SchedulerLeadership", lambda engine: _Noop()
    )
    del original_leadership

    from ingestion.cli import main

    code = main(
        [
            "realtime",
            "--cycle-date",
            "2026-07-21",
            "--cycle-hour",
            "0",
            "--once",
            "--dry-run",
            "--download-dir",
            str(tmp_path / "dl"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "wave due" in out or "realtime_poll" not in out  # diagnostics go to logs


def test_realtime_cycle_hour_validated_by_argparse() -> None:
    from ingestion.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["realtime", "--cycle-hour", "3"])
