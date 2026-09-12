"""Comprehensive test suite for ingestion concurrency configuration, precedence, and safety ceilings."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ingestion.cli import _build_parser
from ingestion.core.catalog import RunCatalogSpec
from ingestion.core.config import (
    ABSOLUTE_MAX_DOWNLOAD_CONCURRENCY,
    ABSOLUTE_MAX_MARKER_GET_CONCURRENCY,
    ABSOLUTE_MAX_MARKER_PUT_CONCURRENCY,
    ABSOLUTE_MAX_WRITE_CONCURRENCY,
    IngestionSettings,
)
from ingestion.core.coordinator import RunCoordinator, WaveRegion
from ingestion.core.wave_runner import _resolve_concurrency_plan


def test_default_concurrency_resolution() -> None:
    """Verify clean code defaults when neither CLI nor ENV is supplied."""
    settings = IngestionSettings(DB_POOL_SIZE=10)
    plan = _resolve_concurrency_plan(settings=settings)

    assert plan.download_concurrency == 8
    assert plan.decode_concurrency == min(4, _get_effective_cpus())
    assert plan.write_concurrency == 4
    assert plan.staging_concurrency == plan.download_concurrency + plan.decode_concurrency + plan.write_concurrency
    assert plan.marker_put_concurrency == 32
    assert plan.marker_get_concurrency == 32
    assert plan.requested is None


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that WEATHER_INGEST_* environment variables override defaults."""
    monkeypatch.setenv("WEATHER_INGEST_DOWNLOAD_CONCURRENCY", "12")
    monkeypatch.setenv("WEATHER_INGEST_DECODE_CONCURRENCY", "6")
    monkeypatch.setenv("WEATHER_INGEST_WRITE_CONCURRENCY", "5")
    monkeypatch.setenv("WEATHER_INGEST_MARKER_PUT_CONCURRENCY", "48")
    monkeypatch.setenv("WEATHER_INGEST_MARKER_GET_CONCURRENCY", "64")

    settings = IngestionSettings(DB_POOL_SIZE=10)
    assert settings.DOWNLOAD_CONCURRENCY == 12
    assert settings.DECODE_CONCURRENCY == 6
    assert settings.WRITE_CONCURRENCY == 5
    assert settings.MARKER_PUT_CONCURRENCY == 48
    assert settings.MARKER_GET_CONCURRENCY == 64

    with patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=16):
        plan = _resolve_concurrency_plan(settings=settings)
        assert plan.download_concurrency == 12
        assert plan.decode_concurrency == 6
        assert plan.write_concurrency == 5
        assert plan.staging_concurrency == 12 + 6 + 5
        assert plan.marker_put_concurrency == 48
        assert plan.marker_get_concurrency == 64


def test_legacy_concurrency_cli_precedence() -> None:
    """Verify legacy --concurrency sets download, decode, and write when stage overrides are absent."""
    settings = IngestionSettings(
        DOWNLOAD_CONCURRENCY=24,
        DECODE_CONCURRENCY=8,
        WRITE_CONCURRENCY=6,
        MARKER_PUT_CONCURRENCY=32,
        MARKER_GET_CONCURRENCY=32,
        DB_POOL_SIZE=10,
    )

    with patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=16):
        # --concurrency 4
        p4 = _resolve_concurrency_plan(requested=4, settings=settings)
        assert p4.requested == 4
        assert p4.download_concurrency == 4
        assert p4.decode_concurrency == 4
        assert p4.write_concurrency == 4
        assert p4.staging_concurrency == 12

        # --concurrency 8 (write clamped by WRITE_CONCURRENCY=6)
        p8 = _resolve_concurrency_plan(requested=8, settings=settings)
        assert p8.requested == 8
        assert p8.download_concurrency == 8
        assert p8.decode_concurrency == 8
        assert p8.write_concurrency == 6
        assert p8.staging_concurrency == 22
        # Marker values should remain untouched by generic --concurrency
        assert p8.marker_put_concurrency == 32
        assert p8.marker_get_concurrency == 32


def test_stage_specific_cli_overrides_legacy_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify stage-specific CLI flags win over both legacy --concurrency and ENV."""
    monkeypatch.setenv("WEATHER_INGEST_DOWNLOAD_CONCURRENCY", "8")
    monkeypatch.setenv("WEATHER_INGEST_DECODE_CONCURRENCY", "12")
    monkeypatch.setenv("WEATHER_INGEST_WRITE_CONCURRENCY", "6")

    settings = IngestionSettings(DB_POOL_SIZE=10)

    with patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=16):
        # User supplied: --concurrency 10 --download-concurrency 16
        # Download should be 16 (stage CLI wins over generic 10 and ENV 8)
        # Decode should be 10 (generic CLI 10 wins over ENV 12)
        # Write should be 6 (clamped by write_concurrency 6)
        plan = _resolve_concurrency_plan(
            requested=10,
            settings=settings,
            download_override=16,
            marker_put_override=64,
        )
        assert plan.download_concurrency == 16
        assert plan.decode_concurrency == 10
        assert plan.write_concurrency == 6
        assert plan.marker_put_concurrency == 64
        assert plan.marker_get_concurrency == 32


def test_cli_parser_defaults_are_none() -> None:
    """Guardrail 1: CLI parser flags must default to None so they do not shadow ENV."""
    parser = _build_parser()

    # Ingest subparser without concurrency flags
    args = parser.parse_args(["ingest", "--model", "gfs", "--cycle-date", "2026-07-21", "--cycle-hour", "0", "--lead-time-hours", "0"])
    assert args.concurrency is None
    assert args.download_concurrency is None
    assert args.decode_concurrency is None
    assert args.write_concurrency is None
    assert args.marker_put_concurrency is None
    assert args.marker_get_concurrency is None

    # Realtime subparser without concurrency flags
    rt_args = parser.parse_args(["realtime"])
    assert rt_args.concurrency is None
    assert rt_args.download_concurrency is None
    assert rt_args.decode_concurrency is None
    assert rt_args.write_concurrency is None
    assert rt_args.marker_put_concurrency is None
    assert rt_args.marker_get_concurrency is None


def test_omitted_cli_does_not_shadow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Task 3: Direct proof that omitted CLI arguments do NOT shadow ENV variables."""
    parser = _build_parser()
    # User runs standard CLI command without any concurrency arguments
    args = parser.parse_args([
        "ingest", "--model", "gfs", "--cycle-date", "2026-07-21",
        "--cycle-hour", "0", "--lead-time-hours", "0"
    ])

    # Operator configured deployment via ENV variables
    monkeypatch.setenv("WEATHER_INGEST_DOWNLOAD_CONCURRENCY", "16")
    monkeypatch.setenv("WEATHER_INGEST_DECODE_CONCURRENCY", "6")
    monkeypatch.setenv("WEATHER_INGEST_WRITE_CONCURRENCY", "5")
    monkeypatch.setenv("WEATHER_INGEST_MARKER_PUT_CONCURRENCY", "48")
    monkeypatch.setenv("WEATHER_INGEST_MARKER_GET_CONCURRENCY", "64")

    settings = IngestionSettings(DB_POOL_SIZE=10)

    # Resolve concurrency plan using parsed CLI args and settings
    with patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=16):
        plan = _resolve_concurrency_plan(
            requested=args.concurrency,
            settings=settings,
            download_override=args.download_concurrency,
            decode_override=args.decode_concurrency,
            write_override=args.write_concurrency,
            marker_put_override=args.marker_put_concurrency,
            marker_get_override=args.marker_get_concurrency,
        )

        # Assert plan strictly matches ENV, not overridden by None or default
        assert plan.download_concurrency == 16
        assert plan.decode_concurrency == 6
        assert plan.write_concurrency == 5
        assert plan.marker_put_concurrency == 48
        assert plan.marker_get_concurrency == 64
        assert plan.requested is None


def test_max_env_does_not_override_operational_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """CRITICAL ACCEPTANCE TEST: Configuring MAX ceiling must NOT alter operational concurrency default."""
    # Operator configures MAX ceiling via legacy or WEATHER_INGEST_MAX_* ENV name
    monkeypatch.setenv("WEATHER_INGEST_MAX_DOWNLOAD_CONCURRENCY", "24")
    monkeypatch.setenv("MAX_DECODE_CONCURRENCY", "16")

    settings = IngestionSettings(DB_POOL_SIZE=10)
    # Operational concurrency must remain at code default 8 and 4!
    assert settings.DOWNLOAD_CONCURRENCY == 8
    assert settings.DECODE_CONCURRENCY == 4
    # Configured MAX ceiling must reflect the setting
    assert settings.MAX_DOWNLOAD_CONCURRENCY == 24
    assert settings.MAX_DECODE_CONCURRENCY == 16

    # When resolving plan without CLI overrides, effective values must equal operational defaults (8, 4)
    with patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=16):
        plan = _resolve_concurrency_plan(settings=settings)
        assert plan.download_concurrency == 8
        assert plan.decode_concurrency == 4


def test_combined_operational_and_max_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify combined operational tuning within configured max ceiling."""
    monkeypatch.setenv("WEATHER_INGEST_DOWNLOAD_CONCURRENCY", "20")
    monkeypatch.setenv("WEATHER_INGEST_MAX_DOWNLOAD_CONCURRENCY", "24")

    settings = IngestionSettings(DB_POOL_SIZE=10)
    assert settings.DOWNLOAD_CONCURRENCY == 20
    assert settings.MAX_DOWNLOAD_CONCURRENCY == 24

    with patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=16):
        plan = _resolve_concurrency_plan(settings=settings)
        assert plan.download_concurrency == 20


def test_operational_exceeding_max_ceiling_clamps_to_ceiling() -> None:
    """Task 3 (Option B): Operational concurrency is naturally clamped to configured max ceiling."""
    s = IngestionSettings(
        DOWNLOAD_CONCURRENCY=32,
        MAX_DOWNLOAD_CONCURRENCY=24,
        DECODE_CONCURRENCY=16,
        MAX_DECODE_CONCURRENCY=8,
        DB_POOL_SIZE=10,
    )
    assert s.DOWNLOAD_CONCURRENCY == 24
    assert s.MAX_DOWNLOAD_CONCURRENCY == 24
    assert s.DECODE_CONCURRENCY == 8
    assert s.MAX_DECODE_CONCURRENCY == 8


def test_configured_max_exceeding_absolute_emergency_cap_fails_validation() -> None:
    """Layer 3: Accidental extreme MAX configuration is rejected by code-only absolute caps."""
    with pytest.raises(ValidationError, match="emergency ceiling"):
        IngestionSettings(
            MAX_DOWNLOAD_CONCURRENCY=ABSOLUTE_MAX_DOWNLOAD_CONCURRENCY + 1,
            DB_POOL_SIZE=10,
        )

    with pytest.raises(ValidationError, match="emergency ceiling"):
        IngestionSettings(
            MAX_WRITE_CONCURRENCY=ABSOLUTE_MAX_WRITE_CONCURRENCY + 1,
            DB_POOL_SIZE=50,
        )


def test_linux_does_not_inherit_windows_61_cap() -> None:
    """Guardrail 2: On Linux, decode concurrency scales up to available CPUs without Windows 61 cap."""
    settings = IngestionSettings(
        DECODE_CONCURRENCY=64,
        MAX_DECODE_CONCURRENCY=96,
        DB_POOL_SIZE=10,
    )
    with patch("sys.platform", "linux"), patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=96):
        plan = _resolve_concurrency_plan(requested=96, settings=settings)
        # On Linux, should scale cleanly to 96 (bounded by ABSOLUTE_MAX_DECODE_CONCURRENCY=128)
        assert plan.decode_concurrency == 96


def test_windows_enforces_61_handle_cap() -> None:
    """Guardrail 2: On Windows, decode concurrency is strictly capped at 61."""
    settings = IngestionSettings(
        DECODE_CONCURRENCY=64,
        MAX_DECODE_CONCURRENCY=96,
        DB_POOL_SIZE=10,
    )
    # On Windows, _detect_effective_cpus returns min(cpus, 61) = 61
    with patch("sys.platform", "win32"), patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=61):
        plan = _resolve_concurrency_plan(requested=96, settings=settings)
        assert plan.decode_concurrency == 61


def test_safety_ceiling_clamping_and_validation() -> None:
    """Verify safety ceilings prevent excessive resource allocation."""
    # Settings validation rejects values <= 0 or > emergency ceiling
    with pytest.raises(ValidationError, match="DOWNLOAD_CONCURRENCY"):
        IngestionSettings(DOWNLOAD_CONCURRENCY=0)

    with pytest.raises(ValidationError, match="DOWNLOAD_CONCURRENCY"):
        IngestionSettings(DOWNLOAD_CONCURRENCY=-5)

    with pytest.raises(ValidationError, match="emergency ceiling"):
        IngestionSettings(MAX_DOWNLOAD_CONCURRENCY=ABSOLUTE_MAX_DOWNLOAD_CONCURRENCY + 1)

    with pytest.raises(ValidationError, match="emergency ceiling"):
        IngestionSettings(MAX_MARKER_PUT_CONCURRENCY=ABSOLUTE_MAX_MARKER_PUT_CONCURRENCY + 1)

    with pytest.raises(ValidationError, match="emergency ceiling"):
        IngestionSettings(MAX_MARKER_GET_CONCURRENCY=ABSOLUTE_MAX_MARKER_GET_CONCURRENCY + 1)

    # Runtime clamping in _resolve_concurrency_plan
    settings = IngestionSettings(
        MAX_DOWNLOAD_CONCURRENCY=64,
        MAX_DECODE_CONCURRENCY=64,
        MAX_WRITE_CONCURRENCY=16,
        MAX_MARKER_PUT_CONCURRENCY=64,
        MAX_MARKER_GET_CONCURRENCY=128,
        DB_POOL_SIZE=20,
    )
    with patch("ingestion.core.wave_runner._detect_effective_cpus", return_value=128):
        plan = _resolve_concurrency_plan(
            download_override=200,
            decode_override=200,
            write_override=50,
            marker_put_override=200,
            marker_get_override=500,
            settings=settings,
        )
        assert plan.download_concurrency == 64  # Clamped by settings.MAX_DOWNLOAD_CONCURRENCY
        assert plan.decode_concurrency == 64   # Clamped by settings.MAX_DECODE_CONCURRENCY
        assert plan.write_concurrency == 16    # Clamped by settings.MAX_WRITE_CONCURRENCY (16 <= 20)
        assert plan.marker_put_concurrency == 64
        assert plan.marker_get_concurrency == 128


def _make_spec() -> RunCatalogSpec:
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
        zarr_store_path="s3://weather-data/test/cycle.zarr",
        variables=(),
        expected_lead_time_hours=(0,),
        expected_members=(),
    )


def test_pre_update_marker_put_receives_configured_concurrency() -> None:
    """Task 5: Pre-update marker PUT passes effective concurrency to put_markers_rolling."""


    spec = _make_spec()
    coordinator = RunCoordinator(spec, "s3://weather-data/test/cycle.zarr")
    coordinator._snapshot = MagicMock()

    regions = [
        WaveRegion(lead_time_hours=i, member=None, generation=f"gen_{i}")
        for i in range(10)
    ]

    mock_conn = MagicMock()
    with (
        patch("ingestion.core.coordinator.read_protocol_version", return_value=1),
        patch("ingestion.core.marker_put_scheduler.put_markers_rolling") as mock_put_rolling,
        patch("ingestion.core.coordinator.StoreLockCoordinator"),
        patch.object(coordinator, "_assert_not_fenced_under_gate"),
        patch("ingestion.core.coordinator.set_run_partial"),
    ):
        mock_put_rolling.return_value = MagicMock(ok=True)
        coordinator.pre_update_wave(
            mock_conn,
            regions=regions,
            run_id=None,
            is_same_cycle=False,
            cancel_event=MagicMock(),
            marker_concurrency=16,
        )
        mock_put_rolling.assert_called_once()
        call_kwargs = mock_put_rolling.call_args[1]
        # Concurrency must be min(16, len(regions)=10) = 10
        assert call_kwargs["concurrency"] == 10


def test_marker_get_concurrency_passed_to_reader() -> None:
    """Task 6: Finalization marker validation uses configured marker_concurrency."""

    spec = _make_spec()
    coordinator = RunCoordinator(spec, "s3://weather-data/test/cycle.zarr")

    mock_conn = MagicMock()
    with (
        patch("ingestion.core.coordinator.read_protocol_version", return_value=1),
        patch("ingestion.core.coordinator.list_region_marker_keys", return_value=["marker1.json", "marker2.json"]),
        patch("ingestion.core.coordinator._read_marker_payloads_bounded", return_value=[]) as mock_read_bounded,
        patch("ingestion.core.coordinator.StoreLockCoordinator"),
        patch.object(coordinator, "_assert_not_fenced_under_gate"),
        patch("ingestion.core.coordinator.read_manifest", return_value=None),
        patch("ingestion.core.coordinator.write_manifest"),
        patch("ingestion.core.coordinator.Session"),
    ):
        coordinator.finalize_run(
            mock_conn,
            run_id="run_123",
            spec=spec,
            expected_leads=(0,),
            expected_members=(),
            marker_concurrency=64,
        )
        mock_read_bounded.assert_called_once()
        assert mock_read_bounded.call_args[1]["max_concurrency"] == 64


def _get_effective_cpus() -> int:
    cpus = os.cpu_count() or 1
    if sys.platform == "win32":
        cpus = min(cpus, 61)
    return max(1, cpus)
