"""Unit tests for the weather-ingest gc CLI subcommand."""

from __future__ import annotations

import os
from alembic import command
from alembic.config import Config
from ingestion.cli import _build_parser, main


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
    db_url = os.getenv("DATABASE_URL", "postgresql://weather_user:weather_password@localhost:5432/weather_db")
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
