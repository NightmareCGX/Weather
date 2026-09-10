"""Tests for ingestion repository-root environment configuration loading.

Verifies:
* Deterministic repository-root discovery via canonical markers (.git, docker-compose.yml).
* Resolution of <repo_root>/.env regardless of working directory.
* Precedence hierarchy: real process environment variables override .env values.
* Missing .env resilience: absent root .env does not fail startup; defaults/env vars are used.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from ingestion.core.config import (
    IngestionSettings,
    find_repository_env_file,
    find_repository_root,
)


def test_ingestion_find_repository_root() -> None:
    """find_repository_root identifies the repository root from ingestion."""
    root = find_repository_root()
    assert root is not None
    assert (root / "docker-compose.yml").is_file()
    assert (root / "services" / "ingestion").is_dir()


def test_ingestion_find_repository_env_file() -> None:
    """find_repository_env_file resolves <repo_root>/.env."""
    env_file = find_repository_env_file()
    assert env_file is not None
    root = find_repository_root()
    assert root is not None
    assert env_file == root / ".env"


def test_ingestion_cwd_independence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CWD changes do not affect root .env discovery."""
    root = find_repository_root()
    assert root is not None
    expected_env = root / ".env"

    monkeypatch.chdir(root)
    assert find_repository_env_file() == expected_env

    ingestion_dir = root / "services" / "ingestion"
    if ingestion_dir.is_dir():
        monkeypatch.chdir(ingestion_dir)
        assert find_repository_env_file() == expected_env

    monkeypatch.chdir(tmp_path)
    assert find_repository_env_file() == expected_env


def test_ingestion_process_env_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process environment variables take strict precedence over .env file values."""
    mock_env = tmp_path / ".env"
    mock_env.write_text("NOAA_DOWNLOAD_SOURCE=nomads\n", encoding="utf-8")

    monkeypatch.setenv("NOAA_DOWNLOAD_SOURCE", "aws_s3")

    class TestIngestionSettings(IngestionSettings):
        model_config = SettingsConfigDict(
            env_file=mock_env,
            env_file_encoding="utf-8",
            extra="ignore",
        )

    s = TestIngestionSettings()
    assert s.NOAA_DOWNLOAD_SOURCE == "aws_s3"


def test_ingestion_missing_env_file_resilience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence of .env does not fail initialization; defaults/env vars are used."""
    nonexistent_env = tmp_path / "nonexistent" / ".env"
    assert not nonexistent_env.exists()

    monkeypatch.delenv("NOAA_DOWNLOAD_SOURCE", raising=False)

    class TestIngestionSettings(IngestionSettings):
        model_config = SettingsConfigDict(
            env_file=nonexistent_env,
            env_file_encoding="utf-8",
            extra="ignore",
        )

    s = TestIngestionSettings()
    assert s.NOAA_DOWNLOAD_SOURCE == "aws_s3"
