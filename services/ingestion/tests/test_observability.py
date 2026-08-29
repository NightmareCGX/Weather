"""Unit and integration tests for ingestion runtime observability, progress UI, and startup timeline."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import CatalogBase
from ingestion.core.observability import (
    NullProgressRenderer,
    PipelineProgressTracker,
    PlainTextSummaryRenderer,
    RichLiveRenderer,
    StartupTimeline,
    create_progress_renderer,
    logger as obs_logger,
)

FIXTURE = str(Path(__file__).parent / "fixtures" / "gfs.t00z.pgrb2.0p25.f006.grib2")


@pytest.fixture
def session(tmp_path) -> Session:
    db_file = tmp_path / "catalog.sqlite"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


class _NoopLockCoordinator:
    def __init__(self, *a, **k):
        pass

    def acquire_shared_gate(self):
        pass

    def release_shared_gate(self):
        pass

    def acquire_exclusive_gate(self):
        pass

    def release_exclusive_gate(self):
        pass

    def acquire_admission(self):
        pass

    def release_admission(self):
        pass

    def acquire_shared_admission(self):
        pass

    def release_shared_admission(self):
        pass

    def acquire_region_locks(self, region_ids):
        pass

    def release_region_locks(self, region_ids):
        pass

    def release_all(self):
        pass

    def close_connection(self):
        pass


async def _fake_download(
    self,
    model,
    cycle_date,
    cycle_hour,
    lead_time_hours,
    destination,
    member=None,
    variables=None,
    **kwargs,
):
    import shutil
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE, destination)
    return destination


def _install_mocks(monkeypatch, session: Session):
    import ingestion.cli as CLI

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        _fake_download,
    )
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: session.bind)
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopLockCoordinator
    )


# -----------------------------------------------------------------------------
# 1. Counter lifecycle & accuracy tests
# -----------------------------------------------------------------------------


def test_counter_lifecycle_full_flow() -> None:
    """Stage counters accurately track progress through queued -> active -> done."""
    tracker = PipelineProgressTracker(
        model="gefs",
        cycle_str="2026-07-21 00:00Z",
        total_items=4,
    )

    # Initial state: 1 seed + 3 non-seed items queued
    assert tracker.counters.total_regions == 4
    assert tracker.counters.download_queued == 3
    assert tracker.counters.download_active == 0
    assert tracker.counters.download_done == 0
    assert tracker.counters.decode_queued == 0
    assert tracker.counters.write_waiting == 0

    # Seed download lifecycle (is_seed=True does not decrement non-seed queued count)
    tracker.on_download_start(member=1, lead=6, is_seed=True)
    assert tracker.counters.download_active == 1
    assert tracker.counters.download_queued == 3

    tracker.on_download_complete(member=1, lead=6, duration_ms=120.0)
    assert tracker.counters.download_active == 0
    assert tracker.counters.download_done == 1
    assert tracker.counters.decode_queued == 1

    # Seed decode lifecycle
    tracker.on_decode_start(member=1, lead=6)
    assert tracker.counters.decode_queued == 0
    assert tracker.counters.decode_active == 1

    tracker.on_decode_complete(member=1, lead=6, duration_ms=80.0)
    assert tracker.counters.decode_active == 0
    assert tracker.counters.decode_done == 1
    assert tracker.counters.write_waiting == 1

    # Non-seed item 2 download lifecycle
    tracker.on_download_start(member=2, lead=6, is_seed=False)
    assert tracker.counters.download_queued == 2
    assert tracker.counters.download_active == 1

    tracker.on_download_complete(member=2, lead=6, duration_ms=110.0)
    assert tracker.counters.download_active == 0
    assert tracker.counters.download_done == 2
    assert tracker.counters.decode_queued == 1

    # Seed write lifecycle (seed leaves write_waiting and enters write_active)
    tracker.on_write_start(member=1, lead=6, is_seed=True)
    assert tracker.counters.write_active == 1
    assert tracker.counters.write_waiting == 0

    tracker.on_write_complete(member=1, lead=6, duration_ms=50.0)
    assert tracker.counters.write_active == 0
    assert tracker.counters.write_done == 1
    assert tracker.counters.overall_done == 1

    # Non-seed item 2 decode lifecycle (enters write_waiting)
    tracker.on_decode_start(member=2, lead=6)
    assert tracker.counters.decode_queued == 0
    assert tracker.counters.decode_active == 1

    tracker.on_decode_complete(member=2, lead=6, duration_ms=90.0)
    assert tracker.counters.decode_active == 0
    assert tracker.counters.decode_done == 2
    assert tracker.counters.write_waiting == 1

    # Non-seed item 2 write lifecycle (leaves write_waiting)
    tracker.on_write_start(member=2, lead=6, is_seed=False)
    assert tracker.counters.write_active == 1
    assert tracker.counters.write_waiting == 0

    tracker.on_write_complete(member=2, lead=6, duration_ms=50.0)
    assert tracker.counters.write_active == 0
    assert tracker.counters.write_done == 2
    assert tracker.counters.overall_done == 2
    assert tracker.counters.write_waiting == 0

    # Finalize lifecycle
    tracker.on_finalize_start()
    assert tracker.counters.finalize_state == "active"

    tracker.on_finalize_complete(duration_ms=45.0)
    assert tracker.counters.finalize_state == "done"


def test_failure_counter_accounting() -> None:
    """Failures cleanly decrement active counters and increment failed counters."""
    tracker = PipelineProgressTracker(
        model="gfs",
        cycle_str="2026-07-21 00:00Z",
        total_items=3,
    )

    # Download failure
    tracker.on_download_start(member=None, lead=12, is_seed=False)
    assert tracker.counters.download_active == 1
    tracker.on_download_failed(member=None, lead=12, duration_ms=50.0)
    assert tracker.counters.download_active == 0
    assert tracker.counters.download_failed == 1

    # Decode failure
    tracker.on_download_start(member=None, lead=18, is_seed=False)
    tracker.on_download_complete(member=None, lead=18, duration_ms=50.0)
    tracker.on_decode_start(member=None, lead=18)
    assert tracker.counters.decode_active == 1
    tracker.on_decode_failed(member=None, lead=18, duration_ms=30.0)
    assert tracker.counters.decode_active == 0
    assert tracker.counters.decode_failed == 1

    # Write failure
    tracker.counters.write_waiting = 1
    tracker.on_write_start(member=None, lead=24)
    assert tracker.counters.write_active == 1
    assert tracker.counters.write_waiting == 0
    tracker.on_write_failed(member=None, lead=24, duration_ms=40.0)
    assert tracker.counters.write_active == 0
    assert tracker.counters.write_failed == 1

    # Finalize failure
    tracker.on_finalize_start()
    tracker.on_finalize_failed(duration_ms=20.0)
    assert tracker.counters.finalize_state == "failed"


# -----------------------------------------------------------------------------
# 2. Startup timeline & report tests
# -----------------------------------------------------------------------------


def test_startup_timeline_milestones_and_durations() -> None:
    """StartupTimeline calculates correct monotonic stage durations and report."""
    timeline = StartupTimeline()

    t0 = 100.0
    timeline.record("run_start", t0)
    timeline.record("seed_download_start", t0 + 0.1)
    timeline.record("seed_download_complete", t0 + 0.5)
    timeline.record("seed_decode_start", t0 + 0.51)
    timeline.record("seed_decode_complete", t0 + 0.71)
    timeline.record("catalog_init_start", t0 + 0.72)
    timeline.record("catalog_init_complete", t0 + 0.75)
    timeline.record("store_gate_wait_start", t0 + 0.76)
    timeline.record("store_gate_acquired", t0 + 0.78)
    timeline.record("prepare_run_store_start", t0 + 0.79)
    timeline.record("prepare_run_store_complete", t0 + 2.50)
    timeline.record("pre_update_start", t0 + 2.51)
    timeline.record("pre_update_complete", t0 + 4.80)
    timeline.record("store_ready", t0 + 4.81)
    timeline.record("wave_tasks_created", t0 + 4.82)
    timeline.record("first_non_seed_download_start", t0 + 4.85)
    timeline.record("seed_write_start", t0 + 4.83)
    timeline.record("seed_write_complete", t0 + 5.10)

    # Durations
    assert pytest.approx(timeline.duration("seed_download_start", "seed_download_complete"), rel=1e-3) == 0.400
    assert pytest.approx(timeline.duration("seed_decode_start", "seed_decode_complete"), rel=1e-3) == 0.200
    assert pytest.approx(timeline.duration("prepare_run_store_start", "prepare_run_store_complete"), rel=1e-3) == 1.710
    assert pytest.approx(timeline.duration("pre_update_start", "pre_update_complete"), rel=1e-3) == 2.290
    assert pytest.approx(timeline.duration("seed_download_start", "first_non_seed_download_start"), rel=1e-3) == 4.750

    # Format report
    report = timeline.format_report(model="gefs", cycle_str="2026-07-21 00:00Z", total_items=510)
    assert "INGESTION STARTUP TIMELINE REPORT" in report
    assert "GEFS" in report
    assert "510" in report
    assert "Prepare Run Store" in report
    assert "Pre-Update Marker PUTs" in report
    assert "Total Cold-Start Delay" in report


# -----------------------------------------------------------------------------
# 3. Target UX & rendering format tests
# -----------------------------------------------------------------------------


def test_progress_lines_format_target_ux() -> None:
    """Tracker output lines match the required Target UX structure."""
    tracker = PipelineProgressTracker(
        model="gefs",
        cycle_str="2026-07-21 00:00Z",
        total_items=510,
    )
    tracker.counters.overall_done = 182
    tracker.counters.download_active = 24
    tracker.counters.download_done = 210
    tracker.counters.download_queued = 276
    tracker.counters.decode_active = 8
    tracker.counters.decode_done = 188
    tracker.counters.decode_queued = 14
    tracker.counters.write_active = 6
    tracker.counters.write_done = 176
    tracker.counters.write_waiting = 28
    tracker.set_init_phase("prepare_run_store")

    lines = tracker.format_progress_lines()
    assert len(lines) == 6
    assert lines[0].startswith("Overall")
    assert "182/510" in lines[0]
    assert "35.7%" in lines[0]

    assert lines[1].startswith("Download")
    assert "active=24" in lines[1]
    assert "done=210" in lines[1]
    assert "queued=276" in lines[1]

    assert lines[2].startswith("Decode")
    assert "active=8" in lines[2]
    assert "done=188" in lines[2]
    assert "queued=14" in lines[2]

    assert lines[3].startswith("Initialize")
    assert "phase=prepare_run_store" in lines[3]

    assert lines[4].startswith("Write")
    assert "active=6" in lines[4]
    assert "done=176" in lines[4]
    assert "waiting=28" in lines[4]

    assert lines[5].startswith("Finalize")
    assert "waiting" in lines[5]


def test_marker_progress_display() -> None:
    """Pre-update marker progress displays done / total / active counts."""
    tracker = PipelineProgressTracker(
        model="gefs",
        cycle_str="2026-07-21 00:00Z",
        total_items=510,
    )
    tracker.set_init_phase("pre_update_markers")
    tracker.set_marker_progress(done=142, total=510, active=8)

    lines = tracker.format_progress_lines()
    init_line = lines[3]
    assert "phase=pre_update_markers (142/510 active=8)" in init_line


def test_plain_text_summary_renderer() -> None:
    """PlainTextSummaryRenderer emits clean progress lines without ANSI codes."""
    tracker = PipelineProgressTracker(
        model="gfs",
        cycle_str="2026-07-21 00:00Z",
        total_items=10,
    )
    stream = io.StringIO()
    renderer = PlainTextSummaryRenderer(tracker, interval_seconds=0.1, stream=stream)

    renderer.start()
    tracker.counters.overall_done = 5
    renderer.update()
    renderer.stop()

    output = stream.getvalue()
    assert "[PROGRESS]" in output
    assert "\033[" not in output  # No ANSI escape codes


def test_create_progress_renderer_factory() -> None:
    """create_progress_renderer instantiates appropriate renderer for TTY/non-TTY/disabled."""
    tracker = PipelineProgressTracker(
        model="gfs",
        cycle_str="2026-07-21 00:00Z",
        total_items=10,
    )
    # Disabled
    null_renderer = create_progress_renderer(tracker, no_progress=True)
    assert isinstance(null_renderer, NullProgressRenderer)

    # Non-TTY
    plain_renderer = create_progress_renderer(tracker, no_progress=False, is_tty=False)
    assert isinstance(plain_renderer, PlainTextSummaryRenderer)

    # TTY
    rich_renderer = create_progress_renderer(tracker, no_progress=False, is_tty=True)
    assert isinstance(rich_renderer, RichLiveRenderer)


# -----------------------------------------------------------------------------
# 4. Bounded memory invariant test
# -----------------------------------------------------------------------------


def test_bounded_memory_invariant() -> None:
    """Tracker memory footprint is strictly bounded O(1) for 10 vs 2430 items."""
    tracker_small = PipelineProgressTracker(model="gfs", cycle_str="2026-07-21", total_items=10)
    tracker_large = PipelineProgressTracker(model="gefs", cycle_str="2026-07-21", total_items=2430)

    size_small = sys.getsizeof(tracker_small.counters) + sys.getsizeof(tracker_small.timeline._milestones)
    size_large = sys.getsizeof(tracker_large.counters) + sys.getsizeof(tracker_large.timeline._milestones)

    # Size difference between tracking 10 items vs 2430 items is 0 bytes
    assert size_small == size_large


# -----------------------------------------------------------------------------
# 5. DEBUG logging efficiency test
# -----------------------------------------------------------------------------


def test_debug_logging_efficiency(caplog: pytest.LogCaptureFixture) -> None:
    """Structured stage transitions log under DEBUG and produce zero output at INFO."""
    tracker = PipelineProgressTracker(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=1)
    logger_name = obs_logger.name
    obs_logger.disabled = False

    # At INFO: logger is not DEBUG-enabled, taking the fast-path with zero records emitted
    caplog.set_level(logging.INFO, logger=logger_name)
    tracker.log_stage_transition(member=None, lead=6, stage="download", event="start")
    obs_records = [r for r in caplog.records if r.name == logger_name]
    assert len(obs_records) == 0

    # At DEBUG: logger is DEBUG-enabled, emitting the expected structured record
    caplog.set_level(logging.DEBUG, logger=logger_name)
    tracker.log_stage_transition(member=1, lead=6, stage="download", event="complete", duration_ms=123.4)
    obs_records = [r for r in caplog.records if r.name == logger_name]
    assert len(obs_records) == 1
    assert obs_records[0].levelno == logging.DEBUG
    assert "stage_transition: model=gfs" in obs_records[0].message
    assert "duration_ms=123.40" in obs_records[0].message


# -----------------------------------------------------------------------------
# 6. End-to-end CLI integration with observability
# -----------------------------------------------------------------------------


def test_cli_ingest_observability_end_to_end(session: Session, tmp_path, monkeypatch) -> None:
    """CLI ingest runs end-to-end with progress tracker and prints startup report."""
    _install_mocks(monkeypatch, session)
    from ingestion.cli import main

    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"

    stdout_capture = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(stdout_capture):
        code = main(
            [
                "ingest",
                "--model",
                "gfs",
                "--cycle-date",
                "2026-07-21",
                "--cycle-hour",
                "0",
                "--lead-time-hours",
                "6",
                "--store",
                store,
                "--allow-custom-store",
                "--download-dir",
                str(dl_dir),
            ]
        )
    assert code == 0
    out = stdout_capture.getvalue()
    assert "INGESTION STARTUP TIMELINE REPORT" in out
    assert "Seed Download" in out
    assert "Prepare Run Store" in out
    assert "Ingested 1 region(s)" in out


def test_cli_ingest_no_progress_flag(session: Session, tmp_path, monkeypatch) -> None:
    """--no-progress flag disables terminal UI and report printing."""
    _install_mocks(monkeypatch, session)
    from ingestion.cli import main

    store = str(tmp_path / "gfs.zarr")
    dl_dir = tmp_path / "dl"

    stdout_capture = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(stdout_capture):
        code = main(
            [
                "ingest",
                "--model",
                "gfs",
                "--cycle-date",
                "2026-07-21",
                "--cycle-hour",
                "0",
                "--lead-time-hours",
                "6",
                "--store",
                store,
                "--allow-custom-store",
                "--download-dir",
                str(dl_dir),
                "--no-progress",
            ]
        )
    assert code == 0
    out = stdout_capture.getvalue()
    assert "INGESTION STARTUP TIMELINE REPORT" not in out
    assert "Ingested 1 region(s)" in out


def test_gefs_30_member_startup_timeline_breakdown(session: Session, tmp_path, monkeypatch) -> None:
    """30-member GEFS run accurately measures and prints the startup timeline breakdown."""
    import numpy as np
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    def _write_gefs_file(path: str, member_num: int):
        with open(path, "wb") as f:
            msg = codes_grib_new_from_samples("GRIB2")
            codes_set(msg, "dataDate", 20260721)
            codes_set(msg, "dataTime", 0)
            codes_set(msg, "stepType", "instant")
            codes_set(msg, "stepRange", "6")
            codes_set(msg, "stepUnits", "h")
            codes_set(msg, "paramId", 167)
            codes_set(msg, "shortName", "2t")
            codes_set(msg, "typeOfLevel", "heightAboveGround")
            codes_set(msg, "level", 2)
            codes_set(msg, "productDefinitionTemplateNumber", 1)
            codes_set(msg, "perturbationNumber", member_num)
            codes_set(msg, "numberOfForecastsInEnsemble", 30)
            codes_set(msg, "typeOfEnsembleForecast", 3)
            codes_set(msg, "gridType", "regular_ll")
            codes_set(msg, "Ni", 10)
            codes_set(msg, "Nj", 5)
            codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
            codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
            codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
            codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
            codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
            codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
            codes_set_values(msg, np.full((5, 10), 280.0, dtype=np.float32).ravel())
            codes_write(msg, f)
            codes_release(msg)

    async def _gefs_download(
        self,
        model,
        cycle_date,
        cycle_hour,
        lead_time_hours,
        destination,
        member=None,
        variables=None,
        **kwargs,
    ):
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_gefs_file(str(dest), member or 1)
        return dest

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        _gefs_download,
    )
    import ingestion.cli as CLI
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: session.bind)
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopLockCoordinator
    )

    from ingestion.cli import main

    store = str(tmp_path / "gefs.zarr")
    dl_dir = str(tmp_path / "dl")

    stdout_capture = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(stdout_capture):
        code = main(
            [
                "ingest",
                "--model",
                "gefs",
                "--cycle-date",
                "2026-07-21",
                "--cycle-hour",
                "0",
                "--lead-time-hours",
                "6",
                "--member",
                *[str(i) for i in range(1, 31)],
                "--store",
                store,
                "--allow-custom-store",
                "--download-dir",
                dl_dir,
                "--concurrency",
                "4",
            ]
        )
    assert code == 0
    out = stdout_capture.getvalue()
    print("\n--- CAPTURED GEFS 30-MEMBER TIMELINE REPORT ---\n" + out)
    assert "INGESTION STARTUP TIMELINE REPORT" in out
    assert "GEFS" in out
    assert "30" in out
    assert "Pre-Update Marker PUTs" in out
    assert "Total Cold-Start Delay" in out
    assert "Pipeline Drain Milestones:" in out
    assert "downloads_drained" in out
    assert "decodes_drained" in out
    assert "writes_drained" in out
    assert "Tail Physical Write Drain (decodes_drained -> writes_drained)" in out


# -----------------------------------------------------------------------------
# 7. Phase 2A Observability Tests (Tests A - G)
# -----------------------------------------------------------------------------


def test_seed_write_waiting_counter_lifecycle() -> None:
    """Test A — Seed lifecycle: verify write_waiting transitions 0 -> 1 -> 0."""
    tracker = PipelineProgressTracker(
        model="gfs",
        cycle_str="2026-07-21 00:00Z",
        total_items=1,
    )
    # Initial state
    assert tracker.counters.write_waiting == 0
    assert tracker.counters.write_active == 0
    assert tracker.counters.write_done == 0

    # 1. Seed download lifecycle
    tracker.on_download_start(member=None, lead=6, is_seed=True)
    tracker.on_download_complete(member=None, lead=6, duration_ms=100.0)

    # 2. Seed decode lifecycle (decode complete -> enters write_waiting)
    tracker.on_decode_start(member=None, lead=6)
    tracker.on_decode_complete(member=None, lead=6, duration_ms=50.0)
    assert tracker.counters.write_waiting == 1
    assert tracker.counters.write_active == 0

    # 3. Seed write start (leaves write_waiting -> enters write_active)
    tracker.on_write_start(member=None, lead=6, is_seed=True)
    assert tracker.counters.write_waiting == 0
    assert tracker.counters.write_active == 1

    # 4. Seed write complete (leaves write_active -> enters write_done)
    tracker.on_write_complete(member=None, lead=6, duration_ms=40.0)
    assert tracker.counters.write_waiting == 0
    assert tracker.counters.write_active == 0
    assert tracker.counters.write_done == 1


def test_full_mixed_lifecycle_counter_accounting() -> None:
    """Test B — Full mixed lifecycle: 1 seed + N non-seed items ending with write_waiting=0."""
    n_non_seed = 5
    total = 1 + n_non_seed
    tracker = PipelineProgressTracker(
        model="gefs",
        cycle_str="2026-07-21 00:00Z",
        total_items=total,
    )

    # Seed flow
    tracker.on_download_start(member=1, lead=6, is_seed=True)
    tracker.on_download_complete(member=1, lead=6, duration_ms=100.0)
    tracker.on_decode_start(member=1, lead=6)
    tracker.on_decode_complete(member=1, lead=6, duration_ms=50.0)
    assert tracker.counters.write_waiting == 1

    tracker.on_write_start(member=1, lead=6, is_seed=True)
    assert tracker.counters.write_waiting == 0
    assert tracker.counters.write_active == 1

    tracker.on_write_complete(member=1, lead=6, duration_ms=40.0)
    assert tracker.counters.write_active == 0
    assert tracker.counters.write_done == 1
    assert tracker.counters.write_waiting == 0

    # Non-seed items flow
    for m in range(2, 2 + n_non_seed):
        tracker.on_download_start(member=m, lead=6, is_seed=False)
        tracker.on_download_complete(member=m, lead=6, duration_ms=80.0)
        tracker.on_decode_start(member=m, lead=6)
        tracker.on_decode_complete(member=m, lead=6, duration_ms=40.0)

    # All 5 non-seed items finished decode -> write_waiting should be 5
    assert tracker.counters.write_waiting == n_non_seed

    # Start writing all 5 non-seed items
    for m in range(2, 2 + n_non_seed):
        tracker.on_write_start(member=m, lead=6, is_seed=False)
    assert tracker.counters.write_waiting == 0
    assert tracker.counters.write_active == n_non_seed

    # Complete writing all 5 non-seed items
    for m in range(2, 2 + n_non_seed):
        tracker.on_write_complete(member=m, lead=6, duration_ms=30.0)

    # Final state: active=0, done=total, waiting=0
    assert tracker.counters.write_active == 0
    assert tracker.counters.write_done == total
    assert tracker.counters.write_waiting == 0


def test_no_negative_waiting_counts_under_valid_and_out_of_order_events() -> None:
    """Test C — No negative waiting counts under valid or edge-case event ordering."""
    tracker = PipelineProgressTracker(
        model="gfs",
        cycle_str="2026-07-21 00:00Z",
        total_items=2,
    )
    assert tracker.counters.write_waiting == 0

    # Starting write when waiting is 0 must not underflow
    tracker.on_write_start(member=None, lead=6, is_seed=False)
    assert tracker.counters.write_waiting == 0
    tracker.on_write_start(member=None, lead=12, is_seed=True)
    assert tracker.counters.write_waiting == 0


def test_tail_drain_milestone_ordering() -> None:
    """Test D — Milestone ordering: downloads_drained <= decodes_drained <= writes_drained <= finalize_start."""
    tracker = PipelineProgressTracker(
        model="gfs",
        cycle_str="2026-07-21 00:00Z",
        total_items=2,
    )

    t0 = 1000.0
    tracker.record_milestone("run_start", t0)

    # Item 1 (seed)
    tracker.on_download_start(member=None, lead=6, is_seed=True)
    tracker.on_download_complete(member=None, lead=6, duration_ms=10.0)
    tracker.on_decode_start(member=None, lead=6)
    tracker.on_decode_complete(member=None, lead=6, duration_ms=10.0)

    # Item 2 (non-seed) download completes -> downloads_drained recorded
    tracker.on_download_start(member=None, lead=12, is_seed=False)
    tracker.timeline.record("downloads_drained", t0 + 5.0)
    tracker.on_download_complete(member=None, lead=12, duration_ms=10.0)

    # Item 2 decode completes -> decodes_drained recorded
    tracker.on_decode_start(member=None, lead=12)
    tracker.timeline.record("decodes_drained", t0 + 7.0)
    tracker.on_decode_complete(member=None, lead=12, duration_ms=10.0)

    # Item 1 and Item 2 writes complete -> writes_drained recorded
    tracker.on_write_start(member=None, lead=6, is_seed=True)
    tracker.on_write_complete(member=None, lead=6, duration_ms=10.0)
    tracker.on_write_start(member=None, lead=12, is_seed=False)
    tracker.timeline.record("writes_drained", t0 + 15.0)
    tracker.on_write_complete(member=None, lead=12, duration_ms=10.0)

    # Finalization start
    tracker.timeline.record("finalize_start", t0 + 15.003)
    tracker.on_finalize_start()
    tracker.timeline.record("finalize_complete", t0 + 19.0)
    tracker.on_finalize_complete(duration_ms=4000.0)

    t_dl = tracker.timeline.get("downloads_drained")
    t_dec = tracker.timeline.get("decodes_drained")
    t_wr = tracker.timeline.get("writes_drained")
    t_fin_start = tracker.timeline.get("finalize_start")
    t_fin_end = tracker.timeline.get("finalize_complete")

    assert t_dl is not None
    assert t_dec is not None
    assert t_wr is not None
    assert t_fin_start is not None
    assert t_fin_end is not None

    assert t_dl <= t_dec
    assert t_dec <= t_wr
    assert t_wr <= t_fin_start
    assert t_fin_start <= t_fin_end


def test_tail_write_interval_calculation_and_reporting() -> None:
    """Test E — Tail-write interval: decodes_drained < writes_drained correctly formatted."""
    timeline = StartupTimeline()
    t0 = 500.0
    timeline.record("run_start", t0)
    timeline.record("seed_download_start", t0 + 0.1)
    timeline.record("seed_download_complete", t0 + 0.5)
    timeline.record("seed_decode_start", t0 + 0.51)
    timeline.record("seed_decode_complete", t0 + 0.71)
    timeline.record("store_ready", t0 + 2.0)
    timeline.record("first_non_seed_download_start", t0 + 2.1)

    # Simulated producer drain vs writer drain
    timeline.record("downloads_drained", t0 + 10.0)
    timeline.record("decodes_drained", t0 + 12.0)
    timeline.record("writes_drained", t0 + 31.2)  # 19.2s of tail physical writes
    timeline.record("finalize_start", t0 + 31.203)  # 3ms of task teardown
    timeline.record("finalize_complete", t0 + 35.0)

    tail_dur = timeline.duration("decodes_drained", "writes_drained")
    assert tail_dur is not None
    assert pytest.approx(tail_dur, rel=1e-3) == 19.200

    teardown_dur = timeline.duration("writes_drained", "finalize_start")
    assert teardown_dur is not None
    assert pytest.approx(teardown_dur, rel=1e-3) == 0.003

    report = timeline.format_report(model="gefs", cycle_str="2026-07-21 00:00Z", total_items=1110)
    assert "Pipeline Drain Milestones:" in report
    assert "downloads_drained" in report
    assert "decodes_drained" in report
    assert "writes_drained" in report
    assert "* Tail Physical Write Drain (decodes_drained -> writes_drained): 19.200s" in report
    assert "* Task Teardown / Gate Transition (writes_drained -> finalize_start): 0.003s" in report


def test_no_false_tail_write_drain_reporting() -> None:
    """Test F — No false tail: when decodes and writes finish at the same time, duration is ~0s."""
    timeline = StartupTimeline()
    t0 = 100.0
    timeline.record("run_start", t0)
    timeline.record("downloads_drained", t0 + 5.0)
    timeline.record("decodes_drained", t0 + 5.0)
    timeline.record("writes_drained", t0 + 5.0)
    timeline.record("finalize_start", t0 + 5.001)

    tail_dur = timeline.duration("decodes_drained", "writes_drained")
    assert tail_dur == 0.0

    report = timeline.format_report(model="gfs", cycle_str="2026-07-21 00:00Z", total_items=1)
    assert "* Tail Physical Write Drain (decodes_drained -> writes_drained): 0.000s" in report
    assert "* Task Teardown / Gate Transition (writes_drained -> finalize_start): 0.001s" in report


def test_gfs_gefs_report_drain_milestones_compatibility() -> None:
    """Test G — GFS and GEFS timeline reports render drain milestones cleanly without errors."""
    for model_name, n_items in [("gfs", 37), ("gefs", 1110)]:
        tracker = PipelineProgressTracker(model=model_name, cycle_str="2026-07-21 00:00Z", total_items=n_items)
        # Record minimal milestones
        tracker.record_milestone("run_start", 100.0)
        tracker.record_milestone("seed_download_start", 100.1)
        tracker.record_milestone("seed_download_complete", 100.4)
        tracker.record_milestone("first_non_seed_download_start", 101.0)
        tracker.record_milestone("downloads_drained", 150.0)
        tracker.record_milestone("decodes_drained", 151.0)
        tracker.record_milestone("writes_drained", 170.0)
        tracker.record_milestone("finalize_start", 170.002)

        report = tracker.timeline.format_report(model=model_name, cycle_str="2026-07-21 00:00Z", total_items=n_items)
        assert model_name.upper() in report
        assert str(n_items) in report
        assert "Pipeline Drain Milestones:" in report
        assert "downloads_drained" in report
        assert "decodes_drained" in report
        assert "writes_drained" in report
        assert "Tail Physical Write Drain" in report
        assert "Task Teardown / Gate Transition" in report



def test_gefs_510_region_startup_delay_measurement(session: Session, tmp_path, monkeypatch) -> None:
    """510-region GEFS run (30 members x 17 leads) measures cold-start delay breakdown."""
    import re
    import numpy as np
    from eccodes import (
        codes_grib_new_from_samples,
        codes_release,
        codes_set,
        codes_set_values,
        codes_write,
    )

    def _write_gefs_file(path: str, member_num: int, lead: int):
        with open(path, "wb") as f:
            msg = codes_grib_new_from_samples("GRIB2")
            codes_set(msg, "dataDate", 20260721)
            codes_set(msg, "dataTime", 0)
            codes_set(msg, "stepType", "instant")
            codes_set(msg, "stepRange", str(lead))
            codes_set(msg, "stepUnits", "h")
            codes_set(msg, "paramId", 167)
            codes_set(msg, "shortName", "2t")
            codes_set(msg, "typeOfLevel", "heightAboveGround")
            codes_set(msg, "level", 2)
            codes_set(msg, "productDefinitionTemplateNumber", 1)
            codes_set(msg, "perturbationNumber", member_num)
            codes_set(msg, "numberOfForecastsInEnsemble", 30)
            codes_set(msg, "typeOfEnsembleForecast", 3)
            codes_set(msg, "gridType", "regular_ll")
            codes_set(msg, "Ni", 10)
            codes_set(msg, "Nj", 5)
            codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
            codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
            codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
            codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
            codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
            codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
            codes_set_values(msg, np.full((5, 10), 280.0, dtype=np.float32).ravel())
            codes_write(msg, f)
            codes_release(msg)

    async def _gefs_download(
        self,
        model,
        cycle_date,
        cycle_hour,
        lead_time_hours,
        destination,
        member=None,
        variables=None,
        **kwargs,
    ):
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_gefs_file(str(dest), member or 1, lead_time_hours)
        return dest

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download",
        _gefs_download,
    )
    import ingestion.cli as CLI
    monkeypatch.setattr(CLI, "_catalog_session_factory", lambda: session.bind)
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopLockCoordinator
    )

    from ingestion.core.decode_worker import DecodePool
    orig_submit = DecodePool.submit

    def _lead_aware_submit(self, path):
        fut = orig_submit(self, path)
        m = re.search(r"\.f(\d{3})\.", str(path))
        if m:
            lead = int(m.group(1))
            orig_result = fut.result
            def _get_adjusted_ds(timeout=None):
                ds = orig_result(timeout=timeout)
                return ds.assign_coords(lead_time_hours=lead)
            fut.result = _get_adjusted_ds
        return fut

    monkeypatch.setattr(DecodePool, "submit", _lead_aware_submit)

    from ingestion.cli import main

    store = str(tmp_path / "gefs.zarr")
    dl_dir = str(tmp_path / "dl")

    # 30 members x 17 leads = 510 regions
    leads = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
    members = list(range(1, 31))

    stdout_capture = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(stdout_capture):
        code = main(
            [
                "ingest",
                "--model",
                "gefs",
                "--cycle-date",
                "2026-07-21",
                "--cycle-hour",
                "0",
                "--lead-time-hours",
                *[str(lead_hr) for lead_hr in leads],
                "--member",
                *[str(m) for m in members],
                "--store",
                store,
                "--allow-custom-store",
                "--download-dir",
                dl_dir,
                "--concurrency",
                "8",
            ]
        )
    assert code == 0
    out = stdout_capture.getvalue()
    print("\n--- CAPTURED GEFS 510-REGION TIMELINE REPORT ---\n" + out)
    assert "INGESTION STARTUP TIMELINE REPORT" in out
    assert "Total Target Regions: 510" in out
    assert "Pre-Update Marker PUTs" in out
    assert "Total Cold-Start Delay" in out

