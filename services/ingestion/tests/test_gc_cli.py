"""Unit tests for the weather-ingest gc CLI subcommand."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock
from alembic import command
from alembic.config import Config
from ingestion.cli import _build_parser, _gc_pipeline_pass, main
from tests._integration_db import integration_db_url


def test_gc_subcommand_parser_defaults():
    parser = _build_parser()
    args = parser.parse_args(["gc"])
    assert args.command == "gc"
    assert args.once is False
    assert args.dry_run is False
    assert args.interval_seconds == 1800.0
    assert args.lock_timeout_seconds == 5.0
    assert args.bucket == "weather-data"


def test_gc_subcommand_parser_flags():
    parser = _build_parser()
    args = parser.parse_args([
        "gc",
        "--once",
        "--dry-run",
        "--interval-seconds", "60.0",
        "--lock-timeout-seconds", "2.0",
        "--bucket", "custom-bucket",
    ])
    assert args.command == "gc"
    assert args.once is True
    assert args.dry_run is True
    assert args.interval_seconds == 60.0
    assert args.lock_timeout_seconds == 2.0
    assert args.bucket == "custom-bucket"


def test_gc_dry_run_main_dispatch(monkeypatch):
    """Verify that main(["gc", "--once", "--dry-run"]) executes cleanly without error."""
    db_url = integration_db_url()
    api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../api"))
    alembic_cfg = Config(os.path.join(api_dir, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    alembic_cfg.set_main_option("script_location", os.path.join(api_dir, "alembic"))
    try:
        command.upgrade(alembic_cfg, "head")
    except Exception:
        pass

    # Run against dry-run pass
    code = main(["gc", "--once", "--dry-run"])
    assert code == 0


def test_gc_subcommand_parser_sweep_metadata():
    parser = _build_parser()
    args = parser.parse_args(["gc", "--sweep-metadata"])
    assert args.command == "gc"
    assert args.sweep_metadata is True
    assert args.batch_size == 50
    assert args.dry_run is False

    args2 = parser.parse_args([
        "gc",
        "--sweep-metadata",
        "--batch-size", "25",
        "--dry-run",
    ])
    assert args2.command == "gc"
    assert args2.sweep_metadata is True
    assert args2.batch_size == 25
    assert args2.dry_run is True


def test_gc_sweep_metadata_dry_run_main_dispatch(monkeypatch):
    """Verify that main(["gc", "--sweep-metadata", "--dry-run"]) executes cleanly."""
    # The dispatch reaches the catalog through application settings (which may
    # load a developer .env). Only run when a test DB URL is EXPLICITLY set in
    # the process environment (CI does this); otherwise skip rather than touch
    # whatever database the developer's .env points at.
    integration_db_url()
    code = main(["gc", "--sweep-metadata", "--dry-run"])
    assert code == 0


def test_gc_sweep_metadata_dispatch_results(monkeypatch):
    from unittest.mock import patch
    from datetime import datetime, timezone
    from ingestion.gc.sweeper import SweeperPassResult

    now = datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc)

    # 1. Success case -> exit code 0
    mock_res_ok = SweeperPassResult(
        dry_run=False,
        evaluated_at=now,
        cutoff=now,
        candidates=(),
        swept_cycles=(),
        failed_cycles=(),
        total_model_runs_deleted=0,
    )
    with patch("ingestion.gc.sweeper.run_metadata_sweeper_pass", return_value=mock_res_ok) as m_pass:
        code = main(["gc", "--sweep-metadata", "--batch-size", "10"])
        assert code == 0
        m_pass.assert_called_once()
        assert m_pass.call_args.kwargs["batch_size"] == 10
        assert m_pass.call_args.kwargs["dry_run"] is False

    # 2. Failure case -> exit code 1
    mock_res_fail = SweeperPassResult(
        dry_run=False,
        evaluated_at=now,
        cutoff=now,
        candidates=(),
        swept_cycles=(),
        failed_cycles=(("gfs", now),),
        total_model_runs_deleted=0,
    )
    with patch("ingestion.gc.sweeper.run_metadata_sweeper_pass", return_value=mock_res_fail):
        code = main(["gc", "--sweep-metadata"])
        assert code == 1




# ===========================================================================
# P0-1: Automated pipeline wiring (planner -> worker -> bookkeeping mainline)
# ===========================================================================
def test_gc_subcommand_parser_pipeline_flags():
    parser = _build_parser()
    args = parser.parse_args(["gc"])
    assert args.enable_planner is False
    assert args.enable_delete is False
    assert args.enable_sweeper is False
    assert args.models == "gfs,gefs"

    args2 = parser.parse_args([
        "gc",
        "--enable-planner",
        "--enable-delete",
        "--enable-sweeper",
        "--models", "gfs",
    ])
    assert args2.enable_planner is True
    assert args2.enable_delete is True
    assert args2.enable_sweeper is True
    assert args2.models == "gfs"


def _bk_result(*args, **kwargs):
    return SimpleNamespace(
        dry_run=False,
        claimed_cycles=("c1",),
        finalized_cycles=("c1",),
        blocked_cycles=(),
        failed_cycles=(),
    )


def _plan_result():
    return SimpleNamespace(dry_run=False, enqueued_count=7, reclaimable_shards=9)


def _worker_result():
    return SimpleNamespace(
        claimed_count=7,
        deleted_count=5,
        revalidated_held_count=2,
        failed_count=0,
        markers_cleaned_count=1,
    )


def _sweeper_result():
    return SimpleNamespace(
        dry_run=False,
        swept_cycles=("c1",),
        failed_cycles=(),
        total_model_runs_deleted=3,
    )


def _patch_pipeline_stages(monkeypatch, *, bk=None, plan=None, worker=None, sweeper=None):
    """Patch the four pipeline stages plus SessionLocal (planner/worker DB)."""
    import contextlib

    from ingestion.core import db as db_mod
    from ingestion.gc import finalizer as fin_mod
    from ingestion.gc import planner as plan_mod
    from ingestion.gc import sweeper as sweep_mod
    from ingestion.gc import worker as worker_mod

    monkeypatch.setattr(fin_mod, "run_lifecycle_bookkeeping_pass", bk or _bk_result)
    monkeypatch.setattr(plan_mod, "plan_reclamation_pass", plan or MagicMock(return_value=_plan_result()))
    monkeypatch.setattr(worker_mod, "run_reclamation_worker_pass", worker or MagicMock(return_value=_worker_result()))
    monkeypatch.setattr(sweep_mod, "run_metadata_sweeper_pass", sweeper or MagicMock(return_value=_sweeper_result()))

    fake_session = MagicMock()
    fake_ctx = contextlib.nullcontext(fake_session)
    monkeypatch.setattr(db_mod, "SessionLocal", MagicMock(return_value=fake_ctx))


def test_gc_pipeline_pass_planner_only_staging(monkeypatch):
    """Planner-only mode enqueues targets but NEVER runs the physical worker."""

    worker_mock = MagicMock(return_value=_worker_result())
    _patch_pipeline_stages(monkeypatch, worker=worker_mock)

    summary = _gc_pipeline_pass(
        "engine://fake",
        models=["gfs"],
        enable_planner=True,
        enable_delete=False,
        enable_sweeper=False,
        batch_size=50,
    )

    assert "bookkeeping claimed=1 finalized=1" in summary
    assert "planner enqueued=7" in summary
    assert "worker" not in summary
    worker_mock.assert_not_called()


def test_gc_pipeline_pass_delete_authorized_runs_worker(monkeypatch):
    """With delete authorization the worker stage runs with delete_enabled=True."""

    worker_mock = MagicMock(return_value=_worker_result())
    _patch_pipeline_stages(monkeypatch, worker=worker_mock)

    summary = _gc_pipeline_pass(
        "engine://fake",
        models=["gfs", "gefs"],
        enable_planner=True,
        enable_delete=True,
        enable_sweeper=True,
        batch_size=50,
    )

    assert "worker claimed=7 deleted=5" in summary
    assert "sweeper swept=1" in summary
    assert worker_mock.call_args.kwargs["delete_enabled"] is True


def test_gc_pipeline_pass_stage_failure_isolation(monkeypatch):
    """A failing bookkeeping stage must not prevent planner/worker stages."""
    def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    plan_mock = MagicMock(return_value=_plan_result())
    worker_mock = MagicMock(return_value=_worker_result())
    _patch_pipeline_stages(monkeypatch, bk=_boom, plan=plan_mock, worker=worker_mock)

    summary = _gc_pipeline_pass(
        "engine://fake",
        models=["gfs"],
        enable_planner=True,
        enable_delete=True,
        enable_sweeper=False,
        batch_size=50,
    )

    assert "bookkeeping=ERROR(RuntimeError)" in summary
    assert "planner enqueued=7" in summary
    assert "worker claimed=7" in summary
    plan_mock.assert_called_once()
    worker_mock.assert_called_once()


def test_gc_pipeline_pass_dry_run_never_touches_worker(monkeypatch):
    """Dry-run must never reach the physical deletion worker stage."""

    worker_mock = MagicMock(return_value=_worker_result())
    _patch_pipeline_stages(monkeypatch, worker=worker_mock)

    summary = _gc_pipeline_pass(
        "engine://fake",
        models=["gfs"],
        enable_planner=True,
        enable_delete=True,
        enable_sweeper=False,
        batch_size=50,
        dry_run=True,
    )

    assert "planner enqueued=" in summary
    assert "worker" not in summary
    worker_mock.assert_not_called()
