"""Unit tests for concurrency plan derivation, CPU detection, and config invariants."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ingestion.cli import (
    _detect_effective_cpus,
    _resolve_concurrency_plan,
)
from ingestion.core.config import IngestionSettings


def test_concurrency_derivation_standard_values() -> None:
    settings = IngestionSettings(
        MAX_DOWNLOAD_CONCURRENCY=24,
        MAX_DECODE_CONCURRENCY=8,
        MAX_WRITE_CONCURRENCY=6,
        DB_POOL_SIZE=10,
    )

    with patch("ingestion.cli._detect_effective_cpus", return_value=8):
        # 1. Low concurrency (4)
        p4 = _resolve_concurrency_plan(4, settings)
        assert p4.requested == 4
        assert p4.download_concurrency == 4
        assert p4.decode_concurrency == 4
        assert p4.write_concurrency == 4
        assert p4.staging_concurrency == 12

        # 2. Medium concurrency (8)
        p8 = _resolve_concurrency_plan(8, settings)
        assert p8.requested == 8
        assert p8.download_concurrency == 8
        assert p8.decode_concurrency == 8
        assert p8.write_concurrency == 6  # clamped by MAX_WRITE_CONCURRENCY
        assert p8.staging_concurrency == 22

        # 3. High concurrency (32)
        p32 = _resolve_concurrency_plan(32, settings)
        assert p32.requested == 32
        assert p32.download_concurrency == 24  # clamped by MAX_DOWNLOAD_CONCURRENCY
        assert p32.decode_concurrency == 8  # clamped by MAX_DECODE_CONCURRENCY
        assert p32.write_concurrency == 6  # clamped by MAX_WRITE_CONCURRENCY
        assert p32.staging_concurrency == 38

        # 4. Extreme concurrency (128)
        p128 = _resolve_concurrency_plan(128, settings)
        assert p128.requested == 128
        assert p128.download_concurrency == 24
        assert p128.decode_concurrency == 8
        assert p128.write_concurrency == 6
        assert p128.staging_concurrency == 38

        # 5. Minimum boundary (1)
        p1 = _resolve_concurrency_plan(1, settings)
        assert p1.requested == 1
        assert p1.download_concurrency == 1
        assert p1.decode_concurrency == 1
        assert p1.write_concurrency == 1
        assert p1.staging_concurrency == 3


def test_concurrency_derivation_cpu_capping() -> None:
    settings = IngestionSettings(
        MAX_DOWNLOAD_CONCURRENCY=24,
        MAX_DECODE_CONCURRENCY=16,
        MAX_WRITE_CONCURRENCY=6,
        DB_POOL_SIZE=10,
    )
    # When host has only 2 CPUs, decode_concurrency is capped at 2 even if requested=16
    with patch("ingestion.cli._detect_effective_cpus", return_value=2):
        plan = _resolve_concurrency_plan(16, settings)
        assert plan.decode_concurrency == 2
        assert plan.download_concurrency == 16
        assert plan.write_concurrency == 6
        assert plan.staging_concurrency == 24


def test_detect_effective_cpus_affinity_and_fallback() -> None:
    # 1. sched_getaffinity available
    with patch("os.sched_getaffinity", return_value={0, 1, 2, 3}, create=True):
        assert _detect_effective_cpus() == 4

    # 2. sched_getaffinity raises OSError / fallback to os.cpu_count
    with patch("os.sched_getaffinity", side_effect=OSError("not allowed"), create=True):
        with patch("os.cpu_count", return_value=6):
            assert _detect_effective_cpus() == 6

    # 3. Windows ceiling of 61
    with patch("sys.platform", "win32"):
        with patch("os.cpu_count", return_value=128):
            with patch("os.sched_getaffinity", side_effect=AttributeError, create=True):
                assert _detect_effective_cpus() == 61


def test_ingestion_settings_relationship_validation() -> None:
    # Valid small-pool test configuration
    s_small = IngestionSettings(
        DB_POOL_SIZE=2,
        DB_MAX_OVERFLOW=1,
        MAX_WRITE_CONCURRENCY=2,
        MAX_DOWNLOAD_CONCURRENCY=4,
        MAX_DECODE_CONCURRENCY=2,
    )
    assert int(s_small.DB_POOL_SIZE) == 2
    assert int(s_small.MAX_WRITE_CONCURRENCY) == 2

    # Invalid: MAX_WRITE_CONCURRENCY > DB_POOL_SIZE
    with pytest.raises(ValidationError, match="must not exceed DB_POOL_SIZE"):
        IngestionSettings(
            DB_POOL_SIZE=5,
            MAX_WRITE_CONCURRENCY=6,
        )

    # Invalid: DB_POOL_SIZE < 1
    with pytest.raises(ValidationError, match="DB_POOL_SIZE must be >= 1"):
        IngestionSettings(DB_POOL_SIZE=0)

    # Invalid: DB_MAX_OVERFLOW < 0
    with pytest.raises(ValidationError, match="DB_MAX_OVERFLOW must be >= 0"):
        IngestionSettings(DB_MAX_OVERFLOW=-1)

    # Invalid: DB_POOL_TIMEOUT_SECONDS <= 0
    with pytest.raises(ValidationError, match="DB_POOL_TIMEOUT_SECONDS must be > 0.0"):
        IngestionSettings(DB_POOL_TIMEOUT_SECONDS=0.0)

    # Invalid: stage concurrency < 1
    with pytest.raises(ValidationError, match="MAX_DOWNLOAD_CONCURRENCY must be >= 1"):
        IngestionSettings(MAX_DOWNLOAD_CONCURRENCY=0)
    with pytest.raises(ValidationError, match="MAX_DECODE_CONCURRENCY must be >= 1"):
        IngestionSettings(MAX_DECODE_CONCURRENCY=0)
    with pytest.raises(ValidationError, match="MAX_WRITE_CONCURRENCY must be >= 1"):
        IngestionSettings(MAX_WRITE_CONCURRENCY=0)
