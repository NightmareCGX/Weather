"""Tests for repository-root environment configuration loading (CWD independence).

Verifies:
* Deterministic repository-root discovery via canonical markers (.git, docker-compose.yml).
* Resolution of <repo_root>/.env regardless of process current working directory.
* CWD invariance: whether running from repo root, services/api, or an arbitrary temp dir,
  the same canonical .env is resolved.
* Elevation regression: ELEVATION_PROVIDER=open_meteo in root .env populates settings.ELEVATION_PROVIDER.
* Precedence hierarchy: real process environment variables override .env values.
* Missing .env resilience: absent root .env does not fail startup; defaults/env vars are used.
* Container runtime safety: absence of repo markers resolves to None without crashing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from api.core.config import (
    Settings,
    find_repository_env_file,
    find_repository_root,
)


def test_find_repository_root_from_current_repo() -> None:
    """find_repository_root identifies the repository root from within the repo."""
    root = find_repository_root()
    assert root is not None
    assert (root / "docker-compose.yml").is_file()
    assert (root / "services" / "api").is_dir()


def test_find_repository_root_from_deep_path() -> None:
    """find_repository_root ascends from any deep subdirectory to find the root."""
    deep_path = Path(__file__).resolve().parent / "fixtures"
    root = find_repository_root(deep_path)
    assert root is not None
    assert (root / "docker-compose.yml").is_file()


def test_find_repository_root_missing_markers_returns_none(tmp_path: Path) -> None:
    """In an environment with no repo markers (e.g. production container), returns None."""
    isolated_dir = tmp_path / "app" / "services" / "api" / "src"
    isolated_dir.mkdir(parents=True)
    fake_file = isolated_dir / "config.py"
    fake_file.write_text("# fake", encoding="utf-8")

    assert find_repository_root(fake_file) is None


def test_find_repository_env_file_resolves_root_env() -> None:
    """find_repository_env_file resolves <repo_root>/.env."""
    env_file = find_repository_env_file()
    assert env_file is not None
    root = find_repository_root()
    assert root is not None
    assert env_file == root / ".env"


def test_cwd_independence_resolves_same_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Changing CWD to repo root, services/api, or temp dir resolves the exact same .env."""
    root = find_repository_root()
    assert root is not None
    expected_env = root / ".env"

    # From repo root
    monkeypatch.chdir(root)
    assert find_repository_env_file() == expected_env

    # From services/api
    api_dir = root / "services" / "api"
    if api_dir.is_dir():
        monkeypatch.chdir(api_dir)
        assert find_repository_env_file() == expected_env

    # From an arbitrary temporary directory outside the repo
    monkeypatch.chdir(tmp_path)
    assert find_repository_env_file() == expected_env


def test_elevation_provider_loaded_from_mock_root_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ELEVATION_PROVIDER=open_meteo in root .env populates settings regardless of CWD."""
    # Build a mock repository structure
    mock_repo = tmp_path / "MockRepo"
    mock_repo.mkdir()
    (mock_repo / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (mock_repo / ".env").write_text("ELEVATION_PROVIDER=open_meteo\n", encoding="utf-8")

    mock_service_dir = mock_repo / "services" / "api"
    mock_service_dir.mkdir(parents=True)
    mock_config_py = mock_service_dir / "src" / "api" / "core" / "config.py"
    mock_config_py.parent.mkdir(parents=True)
    mock_config_py.write_text("# mock config", encoding="utf-8")

    # Ensure ELEVATION_PROVIDER is not set in process env
    monkeypatch.delenv("ELEVATION_PROVIDER", raising=False)

    # Resolve from the mock service location
    discovered_env = find_repository_env_file(mock_config_py)
    assert discovered_env == mock_repo / ".env"

    # Run from service CWD
    monkeypatch.chdir(mock_service_dir)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=discovered_env,
            env_file_encoding="utf-8",
            extra="ignore",
        )

    test_settings = TestSettings()
    assert test_settings.ELEVATION_PROVIDER == "open_meteo"


def test_process_environment_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process environment variables take strict precedence over .env file values."""
    mock_env = tmp_path / ".env"
    mock_env.write_text("ELEVATION_PROVIDER=none\n", encoding="utf-8")

    # Override in process environment
    monkeypatch.setenv("ELEVATION_PROVIDER", "open_meteo")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=mock_env,
            env_file_encoding="utf-8",
            extra="ignore",
        )

    test_settings = TestSettings()
    assert test_settings.ELEVATION_PROVIDER == "open_meteo"


def test_missing_env_file_resilience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence of .env does not cause startup failure; defaults and env vars resolve."""
    nonexistent_env = tmp_path / "nonexistent" / ".env"
    assert not nonexistent_env.exists()

    monkeypatch.delenv("ELEVATION_PROVIDER", raising=False)

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=nonexistent_env,
            env_file_encoding="utf-8",
            extra="ignore",
        )

    # Must instantiate cleanly with default value
    test_settings = TestSettings()
    assert test_settings.ELEVATION_PROVIDER == "none"

    # With process environment variable
    monkeypatch.setenv("ELEVATION_PROVIDER", "open_meteo")
    test_settings_env = TestSettings()
    assert test_settings_env.ELEVATION_PROVIDER == "open_meteo"


def test_container_runtime_no_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulated container environment where env_file is None works purely from defaults/env."""
    monkeypatch.delenv("ELEVATION_PROVIDER", raising=False)

    class ContainerSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=None,
            extra="ignore",
        )

    # Defaults work
    cs = ContainerSettings()
    assert cs.ELEVATION_PROVIDER == "none"

    # Process env works
    monkeypatch.setenv("ELEVATION_PROVIDER", "open_meteo")
    cs2 = ContainerSettings()
    assert cs2.ELEVATION_PROVIDER == "open_meteo"
