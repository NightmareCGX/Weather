"""Ingestion service configuration."""

from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Environment-driven settings for the ingestion service.

    Mirrors the configuration pattern of ``services/api``.
    """

    # Attribute types are ``Any`` because pydantic settings declare them via
    # the class namespace; they are validated at instantiation time. This keeps
    # strict typing clean without suppressing broader checks.
    #: Primary upstream provider for NOAA operational products ("aws_s3" or "nomads").
    #: Defaults to AWS Open Data on S3 to prevent NOMADS rate limiting / anti-abuse bans.
    NOAA_DOWNLOAD_SOURCE: Any = "aws_s3"

    #: Base URL for NOAA GFS on AWS Open Data (us-east-1).
    AWS_GFS_BASE_URL: Any = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

    #: Base URL for NOAA GEFS on AWS Open Data (us-east-1).
    AWS_GEFS_BASE_URL: Any = "https://noaa-gefs-pds.s3.amazonaws.com"

    #: Base URL for NOAA NOMADS operational archive (fallback source).
    NOMADS_BASE_URL: Any = "https://nomads.ncep.noaa.gov"

    #: Enable automatic fallback to NOMADS when AWS S3 returns 404 (e.g. publication lag)
    #: or upstream availability failure.
    ENABLE_NOMADS_FALLBACK: bool = True

    NOAA_USER_AGENT: Any = "weather-platform-ingestion/0.1.0"
    REQUEST_TIMEOUT_SECONDS: Any = 30.0
    DOWNLOAD_RETRIES: Any = 3
    RETRY_BACKOFF_SECONDS: Any = 1.0

    #: Feature switch for NOMADS .idx byte-range selective downloading.
    #: When True, selectively fetches only platform-required GRIB records with
    #: automatic fallback to full downloads on index/range anomalies.
    ENABLE_SELECTIVE_DOWNLOAD: bool = True

    #: PostgreSQL catalog connection used to record ingested runs so the API
    #: serving tier can discover and serve them.
    DATABASE_URL: Any = (
        "postgresql://weather_user:weather_password@localhost:5432/weather_db"
    )

    #: PostgreSQL QueuePool connection pool settings for the ingestion service.
    #: DB_POOL_SIZE provides steady-state base connection capacity for region writers.
    #: DB_MAX_OVERFLOW is burst headroom (never relied upon for steady-state scheduling).
    DB_POOL_SIZE: Any = 10
    DB_MAX_OVERFLOW: Any = 5
    DB_POOL_TIMEOUT_SECONDS: Any = 30.0

    #: Default stage concurrency ceilings (clamped by CLI requested concurrency).
    MAX_DOWNLOAD_CONCURRENCY: Any = 24
    MAX_DECODE_CONCURRENCY: Any = 8
    MAX_WRITE_CONCURRENCY: Any = 6

    MINIO_ENDPOINT: Any = "localhost:9000"
    MINIO_ACCESS_KEY: Any = "minio_admin"
    MINIO_SECRET_KEY: Any = "minio_password"
    # Typed ``bool`` so a ``MINIO_SECURE=false`` env var (documented in
    # .env.example) is coerced to False by pydantic-settings. An untyped
    # ``Any``/str default would resolve to the truthy string ``'false'`` and
    # force HTTPS to a plain-HTTP endpoint (M14 fix).
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: Any = "weather-data"

    # Wave pre-update marker-PUT concurrency / timeout (region-write protocol).
    MARKER_PUT_CONCURRENCY: Any = 8
    MARKER_PUT_TIMEOUT_SECONDS: Any = 30.0
    # Coalesced finalization marker-GET concurrency (P1-B bounded retrieval).
    MARKER_GET_CONCURRENCY: Any = 32
    # Advisory-lock acquisition timeout for the ingestion coordinator.
    ADVISORY_LOCK_TIMEOUT_SECONDS: Any = 30.0

    #: Maximum concurrent connections per S3 client connection pool for data-plane chunk writes (P2 Phase 1).
    #: Sized to allow high concurrent chunk PUT throughput within region writes.
    S3_MAX_POOL_CONNECTIONS: Any = 50

    #: Maximum concurrent connections per S3 client connection pool for control-plane operations (markers, inventory).
    #: Matched to MARKER_GET_CONCURRENCY / MARKER_PUT_CONCURRENCY without excess socket allocation.
    S3_CONTROL_MAX_POOL_CONNECTIONS: Any = 32

    #: Minimum member coverage ratio for serving eligibility (Phase 3).
    ENSEMBLE_MIN_COVERAGE_RATIO: float = 0.85

    @model_validator(mode="after")
    def _validate_pool_and_concurrency_invariants(self) -> "IngestionSettings":
        pool_size = int(self.DB_POOL_SIZE)
        max_overflow = int(self.DB_MAX_OVERFLOW)
        pool_timeout = float(self.DB_POOL_TIMEOUT_SECONDS)
        max_download = int(self.MAX_DOWNLOAD_CONCURRENCY)
        max_decode = int(self.MAX_DECODE_CONCURRENCY)
        max_write = int(self.MAX_WRITE_CONCURRENCY)
        max_marker_get = int(self.MARKER_GET_CONCURRENCY)
        s3_max_pool = int(self.S3_MAX_POOL_CONNECTIONS)
        s3_ctrl_pool = int(self.S3_CONTROL_MAX_POOL_CONNECTIONS)

        if pool_size < 1:
            raise ValueError(f"DB_POOL_SIZE must be >= 1, got {pool_size}")
        if max_overflow < 0:
            raise ValueError(f"DB_MAX_OVERFLOW must be >= 0, got {max_overflow}")
        if pool_timeout <= 0.0:
            raise ValueError(
                f"DB_POOL_TIMEOUT_SECONDS must be > 0.0, got {pool_timeout}"
            )
        if max_download < 1:
            raise ValueError(
                f"MAX_DOWNLOAD_CONCURRENCY must be >= 1, got {max_download}"
            )
        if max_decode < 1:
            raise ValueError(
                f"MAX_DECODE_CONCURRENCY must be >= 1, got {max_decode}"
            )
        if max_write < 1:
            raise ValueError(
                f"MAX_WRITE_CONCURRENCY must be >= 1, got {max_write}"
            )
        if max_write > pool_size:
            raise ValueError(
                f"MAX_WRITE_CONCURRENCY ({max_write}) must not exceed "
                f"DB_POOL_SIZE ({pool_size}) to prevent QueuePool saturation"
            )
        if max_marker_get < 1:
            raise ValueError(
                f"MARKER_GET_CONCURRENCY must be >= 1, got {max_marker_get}"
            )
        if s3_max_pool < 1:
            raise ValueError(
                f"S3_MAX_POOL_CONNECTIONS must be >= 1, got {s3_max_pool}"
            )
        if s3_ctrl_pool < 1:
            raise ValueError(
                f"S3_CONTROL_MAX_POOL_CONNECTIONS must be >= 1, got {s3_ctrl_pool}"
            )
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = IngestionSettings()

from domain.coverage import set_min_coverage_ratio  # noqa: E402
set_min_coverage_ratio(settings.ENSEMBLE_MIN_COVERAGE_RATIO)
