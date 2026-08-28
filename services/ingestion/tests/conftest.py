"""Shared fixtures for ingestion tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from ingestion.core.config import IngestionSettings

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GRIB_FIXTURE = FIXTURES_DIR / "gfs.t00z.pgrb2.0p25.f006.grib2"


@pytest.fixture(scope="session")
def grib_fixture() -> Path:
    """Path to the committed tiny GRIB2 sample fixture."""
    assert GRIB_FIXTURE.is_file(), f"Missing GRIB2 fixture: {GRIB_FIXTURE}"
    return GRIB_FIXTURE


def _minio_reachable(conn_settings: IngestionSettings) -> bool:
    """Return True when a MinIO/S3 endpoint is reachable.

    Genuine connectivity failures (endpoint down, DNS, timeout) raise a
    transport-level exception (not an ``S3Error``) and are reported as
    ``False`` so the caller can skip. Authentication/configuration
    failures are identified by exception *type*: the MinIO client raises
    :class:`minio.error.S3Error` with an error ``code`` (e.g.
    ``AccessDenied``, ``InvalidAccessKeyId``) rather than a parsed
    message string. Those propagate so a reachable-but-misconfigured
    endpoint fails the test instead of being hidden behind a skip.
    """
    try:
        from minio import Minio  # type: ignore[import-untyped]
        from minio.error import S3Error  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("minio is not installed; required for MinIO tests") from exc

    client = Minio(
        conn_settings.MINIO_ENDPOINT,
        access_key=conn_settings.MINIO_ACCESS_KEY,
        secret_key=conn_settings.MINIO_SECRET_KEY,
        secure=conn_settings.MINIO_SECURE,
    )
    try:
        if not client.bucket_exists(conn_settings.MINIO_BUCKET_NAME):
            client.make_bucket(conn_settings.MINIO_BUCKET_NAME)
        return True
    except S3Error:
        # Reachable but rejected (bad credentials, invalid configuration).
        # The original S3Error carries the error code for diagnosis.
        raise
    except Exception:
        # Transport-level connectivity failure (endpoint down, DNS,
        # timeout): treat as unreachable.
        return False


def _remove_prefix(conn_settings: IngestionSettings, store: str) -> None:
    """Remove every object created under the store's prefix.

    Only objects under the given store prefix (e.g. ``m5-test/<id>/``)
    are removed; unrelated objects are left untouched.

    Args:
        conn_settings: Ingestion settings providing MinIO credentials.
        store: The ``s3://`` store URL whose objects are to be removed.
    """
    from minio import Minio  # type: ignore[import-untyped]

    prefix = store[len("s3://") :]
    bucket, _, object_prefix = prefix.partition("/")
    client = Minio(
        conn_settings.MINIO_ENDPOINT,
        access_key=conn_settings.MINIO_ACCESS_KEY,
        secret_key=conn_settings.MINIO_SECRET_KEY,
        secure=conn_settings.MINIO_SECURE,
    )
    for obj in client.list_objects(bucket, prefix=object_prefix, recursive=True):
        client.remove_object(bucket, obj.object_name)


@pytest.fixture
def minio_store() -> Iterator[str]:
    """Return an ``s3://`` store URL, skipping when MinIO is unavailable.

    Integration coverage for the S3/MinIO path is opt-in: it only runs
    when ``WEATHER_TEST_MINIO=1`` is set AND the configured MinIO endpoint
    is reachable. Otherwise the test that depends on this fixture is
    skipped.

    The teardown removes every object written under the returned store
    prefix, whether the test passes or fails.
    """
    if os.environ.get("WEATHER_TEST_MINIO") != "1":
        pytest.skip("WEATHER_TEST_MINIO != 1; skipping MinIO integration test")

    conn_settings = IngestionSettings()
    if not _minio_reachable(conn_settings):
        pytest.skip("MinIO endpoint is not reachable; skipping MinIO integration test")

    store = f"s3://{conn_settings.MINIO_BUCKET_NAME}/m5-test/{uuid.uuid4().hex}"
    try:
        yield store
    finally:
        _remove_prefix(conn_settings, store)
