"""Unit tests for the weather-ingest gc CLI subcommand."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
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
        "--metadata-retention-days", "3",
        "--models", "gfs",
    ])
    assert args2.enable_planner is True
    assert args2.enable_delete is True
    assert args2.enable_sweeper is True
    assert args2.metadata_retention_days == 3
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
    sweeper_mock = MagicMock(return_value=_sweeper_result())
    _patch_pipeline_stages(monkeypatch, worker=worker_mock, sweeper=sweeper_mock)

    summary = _gc_pipeline_pass(
        "engine://fake",
        models=["gfs", "gefs"],
        enable_planner=True,
        enable_delete=True,
        enable_sweeper=True,
        batch_size=50,
        retention_days=2,
    )

    assert "worker claimed=7 deleted=5" in summary
    assert "sweeper swept=1" in summary
    assert worker_mock.call_args.kwargs["delete_enabled"] is True
    assert sweeper_mock.call_args.kwargs["retention_days"] == 2


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


# ===========================================================================
# In-process GC pipeline metrics (planner/worker/sweeper stage observability)
# ===========================================================================
def _counter_value(name: str) -> float:
    from ingestion.monitoring.metrics import REGISTRY

    metric = REGISTRY.get(name)
    lines = [ln for ln in (metric.collect() if metric else []) if not ln.startswith("#")]
    if not lines:
        return 0.0
    return float(lines[-1].split(" ")[-1])


def _gauge_value(name: str, **labels: str) -> float | None:
    from ingestion.monitoring.metrics import REGISTRY

    metric = REGISTRY.get(name)
    if metric is None:
        return None
    for line in metric.collect():
        if line.startswith("#"):
            continue
        if all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.split(" ")[-1])
    return None


def test_gc_pipeline_pass_records_stage_metrics(monkeypatch):
    """One pass increments stage counters matching the mocked result fields."""
    from ingestion.monitoring.gc_metrics import snapshot_alert_state

    worker_mock = MagicMock(return_value=_worker_result())
    _patch_pipeline_stages(monkeypatch, worker=worker_mock)

    enq_before = _counter_value("weather_gc_planner_enqueued_total")
    del_before = _counter_value("weather_gc_worker_deleted_total")
    sweep_before = _counter_value("weather_gc_sweeper_swept_total")

    summary = _gc_pipeline_pass(
        "engine://fake",
        models=["gfs"],
        enable_planner=True,
        enable_delete=True,
        enable_sweeper=True,
        batch_size=50,
    )

    # stdout summary format is unchanged (backward compatible).
    assert "bookkeeping claimed=1 finalized=1" in summary
    assert "planner enqueued=7" in summary
    assert "worker claimed=7 deleted=5" in summary
    assert "sweeper swept=1" in summary

    assert _counter_value("weather_gc_planner_enqueued_total") == enq_before + 7
    assert _counter_value("weather_gc_planner_reclaimable_total") >= 9
    assert _counter_value("weather_gc_worker_deleted_total") == del_before + 5
    assert _counter_value("weather_gc_worker_claimed_total") >= 7
    assert _counter_value("weather_gc_worker_markers_cleaned_total") >= 1
    assert _counter_value("weather_gc_sweeper_swept_total") == sweep_before + 1

    # All instrumented stages reported success in the latest pass.
    for stage in ("bookkeeping", "planner", "worker", "sweeper"):
        assert _gauge_value("weather_gc_pass_success", stage=stage) == 1.0, stage
        assert (
            _gauge_value("weather_gc_pass_last_success_timestamp", stage=stage) or 0
        ) > 0, stage

    # Latest-pass snapshot: no failures in this pass.
    assert snapshot_alert_state()["worker_failed_total"] == 0


def test_gc_pipeline_pass_stage_failure_sets_pass_success_zero(monkeypatch):
    """A raising stage records pass_success{stage}=0 without killing the pass."""
    from ingestion.monitoring.gc_metrics import snapshot_alert_state

    def _boom(*args, **kwargs):
        raise RuntimeError("s3 down")

    plan_mock = MagicMock(return_value=_plan_result())
    worker_mock = MagicMock(return_value=_worker_result())
    _patch_pipeline_stages(monkeypatch, plan=plan_mock, worker=worker_mock)
    monkeypatch.setattr("ingestion.gc.sweeper.run_metadata_sweeper_pass", _boom)

    summary = _gc_pipeline_pass(
        "engine://fake",
        models=["gfs"],
        enable_planner=True,
        enable_delete=True,
        enable_sweeper=True,
        batch_size=50,
    )

    assert "sweeper=ERROR(RuntimeError)" in summary
    assert "planner enqueued=7" in summary  # pass continued after the failure
    assert _gauge_value("weather_gc_pass_success", stage="sweeper") == 0.0
    assert _gauge_value("weather_gc_pass_success", stage="planner") == 1.0

    # The alert snapshot tracks result-reported failures; a stage that RAISED
    # produced no result, so the sweeper failure count stays untouched.
    state = snapshot_alert_state()
    assert state["sweeper_failed_total"] == 0
    assert state["worker_failed_total"] == 0

    # A sweeper result reporting failed cycles feeds the alert snapshot.
    failed_sweeper = SimpleNamespace(
        dry_run=False,
        swept_cycles=(),
        failed_cycles=("c1", "c2"),
        total_model_runs_deleted=0,
    )
    monkeypatch.setattr(
        "ingestion.gc.sweeper.run_metadata_sweeper_pass",
        MagicMock(return_value=failed_sweeper),
    )
    _gc_pipeline_pass(
        "engine://fake",
        models=["gfs"],
        enable_planner=False,
        enable_delete=False,
        enable_sweeper=True,
        batch_size=50,
    )
    assert snapshot_alert_state()["sweeper_failed_total"] == 2


def test_gc_pipeline_pass_duration_histogram_has_stage_series(monkeypatch):
    _patch_pipeline_stages(monkeypatch)
    _gc_pipeline_pass(
        "engine://fake",
        models=["gfs"],
        enable_planner=False,
        enable_delete=False,
        enable_sweeper=False,
        batch_size=50,
    )
    from ingestion.monitoring.metrics import REGISTRY

    duration_lines = REGISTRY.get("weather_gc_pass_duration_seconds").collect()
    assert any('stage="bookkeeping"' in line for line in duration_lines)
    assert any('_count' in line for line in duration_lines)


def test_gc_metrics_exposed_in_registry_exposition(monkeypatch):
    """REGISTRY.generate_latest() exposes every gc metric family name."""
    from ingestion.monitoring.gc_metrics import record_planner_pass, record_sweeper_pass
    from ingestion.monitoring.metrics import REGISTRY

    record_planner_pass(enqueued_count=1, reclaimable_shards=1)
    record_sweeper_pass(swept_cycles=1, failed_cycles=0)

    exposition = REGISTRY.generate_latest()
    for name in (
        "weather_gc_pass_duration_seconds",
        "weather_gc_pass_success",
        "weather_gc_pass_last_success_timestamp",
        "weather_gc_planner_enqueued_total",
        "weather_gc_planner_reclaimable_total",
        "weather_gc_worker_claimed_total",
        "weather_gc_worker_deleted_total",
        "weather_gc_worker_failed_total",
        "weather_gc_worker_markers_cleaned_total",
        "weather_gc_sweeper_swept_total",
        "weather_gc_sweeper_failed_total",
        "weather_gc_inventory_orphans",
        "weather_gc_inventory_last_success_timestamp",
        "weather_gc_inventory_errors_total",
    ):
        assert name in exposition, name


# ===========================================================================
# Scheduled orphan inventory stage (store <-> catalog reconciliation, §9)
# ===========================================================================
def test_gc_subcommand_parser_new_flags():
    parser = _build_parser()
    args = parser.parse_args(["gc"])
    assert args.inventory_interval_hours is None
    assert args.metrics_port is None
    assert args.metrics_host == "127.0.0.1"

    args2 = parser.parse_args([
        "gc",
        "--inventory-interval-hours", "12",
        "--metrics-host", "0.0.0.0",
        "--metrics-port", "9114",
    ])
    assert args2.inventory_interval_hours == 12.0
    assert args2.metrics_host == "0.0.0.0"
    assert args2.metrics_port == 9114


def test_resolve_inventory_interval_hours_precedence(monkeypatch):
    from ingestion.cli import _resolve_inventory_interval_hours

    args = SimpleNamespace(inventory_interval_hours=None)
    monkeypatch.delenv("GC_INVENTORY_INTERVAL_HOURS", raising=False)
    # Default 24h when neither flag nor env is set.
    assert _resolve_inventory_interval_hours(args) == 24.0
    # Env fallback applies when the flag is not given.
    monkeypatch.setenv("GC_INVENTORY_INTERVAL_HOURS", "6")
    assert _resolve_inventory_interval_hours(args) == 6.0
    # An explicit flag wins over the env fallback.
    args_flag = SimpleNamespace(inventory_interval_hours=0.5)
    assert _resolve_inventory_interval_hours(args_flag) == 0.5
    # 0 disables the scheduled stage.
    args_zero = SimpleNamespace(inventory_interval_hours=0)
    assert _resolve_inventory_interval_hours(args_zero) == 0.0
    # Invalid env falls back to the default.
    monkeypatch.setenv("GC_INVENTORY_INTERVAL_HOURS", "not-a-number")
    assert _resolve_inventory_interval_hours(args) == 24.0


class _FakeGcLeadership:
    def __init__(self, engine):
        del engine

    def acquire(self):
        return True

    def check_leadership(self):
        return True

    def release(self):
        return None


def _orphan_inventory_result(orphans_beyond=1, orphans_within=0):
    from datetime import datetime, timezone

    from ingestion.gc.inventory import InventoryResult, OrphanStore

    orphan_list = []
    for _ in range(orphans_beyond):
        orphan_list.append(
            OrphanStore(
                model_id="gfs",
                cycle_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
                store_path="s3://weather-data/gfs/2020-01-01/00/cycle.zarr",
                beyond_frontier=True,
            )
        )
    for _ in range(orphans_within):
        orphan_list.append(
            OrphanStore(
                model_id="gfs",
                cycle_time=datetime(2026, 9, 13, tzinfo=timezone.utc),
                store_path="s3://weather-data/gfs/2026-09-13/00/cycle.zarr",
                beyond_frontier=False,
            )
        )
    return InventoryResult(
        evaluated_at=datetime.now(timezone.utc),
        cataloged_cycles=10,
        discovered_stores=12,
        orphan_stores=tuple(orphan_list),
        reaped_stores=(),
        reap_blocked=(),
        errors=(),
    )


def _patch_daemon_loop(monkeypatch, *, inventory_mock=None, pipeline_pass="bk ok"):
    """Patch everything the gc daemon loop touches so it can be driven offline.

    The loop is terminated by a fake ``time.sleep`` that advances a fake
    monotonic clock and raises KeyboardInterrupt on its second call, so the
    loop executes exactly two passes.
    """
    import time as time_module

    monkeypatch.setattr("ingestion.gc.leadership.GcLeadership", _FakeGcLeadership)
    monkeypatch.setattr(
        "ingestion.cli._gc_pipeline_pass", MagicMock(return_value=pipeline_pass)
    )
    monkeypatch.setattr(
        "ingestion.cli._dispatch_gc_alerts", MagicMock(return_value=None)
    )
    inv_mock = inventory_mock or MagicMock(
        return_value=_orphan_inventory_result(orphans_beyond=1)
    )
    monkeypatch.setattr("ingestion.gc.inventory.run_orphan_inventory", inv_mock)
    monkeypatch.setattr("ingestion.gc.inventory.reap_orphan_store", MagicMock())

    clock = {"t": 1000.0}
    monkeypatch.setattr(time_module, "monotonic", lambda: clock["t"])
    sleep_calls: list[int] = []

    def fake_sleep(_seconds: float) -> None:
        sleep_calls.append(1)
        clock["t"] += 3600.0  # advance past any realistic inventory interval
        if len(sleep_calls) >= 2:
            raise KeyboardInterrupt  # terminate the daemon loop after 2 passes

    monkeypatch.setattr(time_module, "sleep", fake_sleep)
    return inv_mock


def test_gc_daemon_runs_scheduled_inventory_and_never_reaps(monkeypatch, capsys):
    """Daemon mode appends the inventory stage to the due pass summary.

    Locks the scheduling-path invariant: run_orphan_inventory is invoked with
    reap=False and the physical-deletion reap_orphan_store is never touched.
    """
    inv_mock = _patch_daemon_loop(monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        main(["gc", "--inventory-interval-hours", "0.0000001"])

    inv_mock.assert_called_once()
    assert inv_mock.call_args.kwargs["reap"] is False
    out = capsys.readouterr().out
    assert (
        "GC pass: bk ok; inventory discovered=12 orphans=1 beyond_frontier=1 errors=0"
        in out
    )


def test_gc_daemon_scheduled_inventory_updates_orphan_metrics(monkeypatch):
    from ingestion.monitoring.gc_metrics import snapshot_alert_state

    inv_mock = MagicMock(
        return_value=_orphan_inventory_result(orphans_beyond=1, orphans_within=2)
    )
    _patch_daemon_loop(monkeypatch, inventory_mock=inv_mock)

    with pytest.raises(KeyboardInterrupt):
        main(["gc", "--inventory-interval-hours", "0.0000001"])

    assert _gauge_value("weather_gc_inventory_orphans", beyond_frontier="true") == 1.0
    assert _gauge_value("weather_gc_inventory_orphans", beyond_frontier="false") == 2.0
    assert (_gauge_value("weather_gc_inventory_last_success_timestamp") or 0) > 0
    # Within-frontier orphans are the alert condition (first version: > 0 fires).
    assert snapshot_alert_state()["inventory_orphans_within_frontier"] == 2


def test_gc_daemon_inventory_failure_is_fail_open(monkeypatch, capsys):
    """A raising inventory stage is reported in the summary and never kills the loop."""
    from ingestion.monitoring.gc_metrics import snapshot_alert_state

    def _boom(*args, **kwargs):
        raise RuntimeError("store scan failed")

    inv_mock = MagicMock(side_effect=_boom)
    _patch_daemon_loop(monkeypatch, inventory_mock=inv_mock)
    errors_before = _counter_value("weather_gc_inventory_errors_total")
    state_before = dict(snapshot_alert_state())

    with pytest.raises(KeyboardInterrupt):
        main(["gc", "--inventory-interval-hours", "0.0000001"])

    out = capsys.readouterr().out
    assert "inventory=ERROR(RuntimeError)" in out
    assert _counter_value("weather_gc_inventory_errors_total") == errors_before + 1
    # A failed inventory pass publishes no orphan counts, so the alert
    # snapshot stays exactly as it was before the run.
    assert snapshot_alert_state() == state_before


def test_gc_daemon_inventory_disabled_with_zero_interval(monkeypatch):
    """--inventory-interval-hours 0 keeps daemon behavior identical to today."""
    inv_mock = _patch_daemon_loop(monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        main(["gc", "--inventory-interval-hours", "0"])

    inv_mock.assert_not_called()


def test_gc_once_mode_never_runs_scheduled_inventory(monkeypatch, capsys):
    """--once mode keeps minimal semantics: no scheduled inventory stage."""
    inv_mock = _patch_daemon_loop(monkeypatch)

    code = main(["gc", "--once", "--inventory-interval-hours", "0.0000001"])

    assert code == 0
    inv_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "inventory" not in out


def test_gc_inventory_stage_direct_invariant(monkeypatch):
    """_gc_inventory_stage passes reap=False and formats the summary fragment."""
    from ingestion.cli import _gc_inventory_stage

    inv_mock = MagicMock(return_value=_orphan_inventory_result(orphans_beyond=2))
    monkeypatch.setattr("ingestion.gc.inventory.run_orphan_inventory", inv_mock)
    reap_mock = MagicMock()
    monkeypatch.setattr("ingestion.gc.inventory.reap_orphan_store", reap_mock)

    summary = _gc_inventory_stage(
        "engine://fake", store_root="s3://bucket", timeout_seconds=5.0
    )

    assert summary == "inventory discovered=12 orphans=2 beyond_frontier=2 errors=0"
    assert inv_mock.call_args.kwargs["reap"] is False
    assert inv_mock.call_args.kwargs["store_root"] == "s3://bucket"
    assert _gauge_value("weather_gc_inventory_orphans", beyond_frontier="true") == 2.0
    reap_mock.assert_not_called()


def test_gc_inventory_stage_failure_sets_pass_success_zero(monkeypatch):
    """A raising scheduled inventory pass marks the stage gauge 0 (fail-open)."""
    from ingestion.cli import _gc_inventory_stage

    inv_mock = MagicMock(return_value=_orphan_inventory_result(orphans_beyond=1))
    monkeypatch.setattr("ingestion.gc.inventory.run_orphan_inventory", inv_mock)
    _gc_inventory_stage("engine://fake", store_root="s3://b", timeout_seconds=5.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("x")

    monkeypatch.setattr(
        "ingestion.gc.inventory.run_orphan_inventory", MagicMock(side_effect=_boom)
    )
    summary = _gc_inventory_stage("engine://fake", store_root="s3://b", timeout_seconds=5.0)

    assert summary == "inventory=ERROR(RuntimeError)"
    assert _gauge_value("weather_gc_pass_success", stage="inventory") == 0.0


# ===========================================================================
# Alert-engine GC rules
# ===========================================================================
def test_alert_rules_gc_stage_failures_and_orphans():
    from ingestion.monitoring.alerts import AlertEngine, AlertSeverity

    engine = AlertEngine()

    alerts = engine.evaluate_rules(
        gc_data={
            "worker_failed_total": 3,
            "sweeper_failed_total": 1,
            "inventory_orphans_within_frontier": 2,
        }
    )
    names = {a.name: a for a in alerts}
    assert names["gc_worker_failed_shards"].severity == AlertSeverity.WARNING
    assert names["gc_worker_failed_shards"].value == 3
    assert names["gc_sweeper_failed_cycles"].value == 1
    assert names["gc_orphan_stores_detected"].value == 2

    # Clean snapshot -> no gc alerts fire.
    clean = engine.evaluate_rules(
        gc_data={
            "worker_failed_total": 0,
            "sweeper_failed_total": 0,
            "inventory_orphans_within_frontier": 0,
        }
    )
    assert all(not a.name.startswith("gc_") for a in clean)

    # Legacy call sites without gc_data keep working.
    assert engine.evaluate_rules() == []
