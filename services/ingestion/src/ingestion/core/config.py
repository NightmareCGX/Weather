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

    #: HTTP connection pool configuration for NOAAConnector (P4)
    HTTP_MAX_CONNECTIONS: Any = 100
    HTTP_MAX_KEEPALIVE_CONNECTIONS: Any = 50
    HTTP_KEEPALIVE_EXPIRY_SECONDS: Any = 5.0

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

    #: Primary storage format version for newly initialized forecast cycles ("sharded_v1" or "v2_unsharded").
    #: Defaults to "sharded_v1" (14 physical shard objects per region, 120 inner 100x100 chunks).
    STORAGE_FORMAT_VERSION: Any = "sharded_v1"

    #: Global physical object PUT concurrency for shard writes.
    GLOBAL_PUT_CONCURRENCY: Any = 64

    #: Maximum concurrent connections per S3 client connection pool for data-plane chunk writes (P2 Phase 1).
    #: Sized to allow high concurrent chunk PUT throughput within region writes.
    S3_MAX_POOL_CONNECTIONS: Any = 128

    #: Maximum concurrent connections per S3 client connection pool for control-plane operations (markers, inventory).
    #: Matched to MARKER_GET_CONCURRENCY / MARKER_PUT_CONCURRENCY without excess socket allocation.
    S3_CONTROL_MAX_POOL_CONNECTIONS: Any = 32

    #: Minimum member coverage ratio for serving eligibility (Phase 3).
    ENSEMBLE_MIN_COVERAGE_RATIO: float = 0.85

    # --- Realtime lead-wave scheduler (Phase 5C) ---
    #: Master switch for `weather-ingest realtime`. Big-batch ingestion never
    #: reads these settings.
    REALTIME_ENABLED: bool = False
    #: Normal active poll cadence (seconds) when tracking a cycle.
    REALTIME_ACTIVE_POLL_SECONDS: float = 600.0
    #: Fast poll cadence (seconds) while upstream publication activity is observed.
    REALTIME_PUBLICATION_POLL_SECONDS: float = 120.0
    #: Initial idle-backoff interval (seconds) after polls with no publication
    #: activity; doubles on each further idle poll up to the maximum.
    REALTIME_IDLE_BACKOFF_INITIAL_SECONDS: float = 1800.0
    #: Upper bound for the idle-backoff interval (seconds).
    REALTIME_IDLE_BACKOFF_MAX_SECONDS: float = 3600.0
    #: Relative jitter applied to every poll interval (± fraction). 0 disables.
    REALTIME_POLL_JITTER_FRACTION: float = 0.10
    #: Bounded-batching: emit a wave when this many complete pending leads have
    #: accumulated OR the oldest pending lead has waited wave_max_wait seconds.
    REALTIME_WAVE_MAX_LEADS: int = 8
    #: Bounded-batching maximum wait (seconds) for the oldest pending lead.
    REALTIME_WAVE_MAX_WAIT_SECONDS: float = 1200.0
    #: Retry delay (seconds) after a discovery failure. Discovery failures are
    #: never treated as "upstream idle" and never advance the poll state machine.
    REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS: float = 60.0
    #: Auto cycle selection: delay (seconds) after a cycle's nominal time before
    #: the scheduler considers it eligible for upstream probing (publication
    #: begins roughly 3-3.5h after cycle time, probed in Phase 5A).
    REALTIME_FIRST_PUBLICATION_DELAY_SECONDS: float = 10800.0

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
        active_poll = float(self.REALTIME_ACTIVE_POLL_SECONDS)
        publication_poll = float(self.REALTIME_PUBLICATION_POLL_SECONDS)
        backoff_initial = float(self.REALTIME_IDLE_BACKOFF_INITIAL_SECONDS)
        backoff_max = float(self.REALTIME_IDLE_BACKOFF_MAX_SECONDS)
        jitter = float(self.REALTIME_POLL_JITTER_FRACTION)
        wave_max_leads = int(self.REALTIME_WAVE_MAX_LEADS)
        wave_max_wait = float(self.REALTIME_WAVE_MAX_WAIT_SECONDS)
        failure_retry = float(self.REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS)
        first_publication_delay = float(
            self.REALTIME_FIRST_PUBLICATION_DELAY_SECONDS
        )

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
        if active_poll <= 0.0:
            raise ValueError(
                f"REALTIME_ACTIVE_POLL_SECONDS must be > 0.0, got {active_poll}"
            )
        if publication_poll <= 0.0:
            raise ValueError(
                f"REALTIME_PUBLICATION_POLL_SECONDS must be > 0.0, got {publication_poll}"
            )
        if backoff_initial <= 0.0:
            raise ValueError(
                f"REALTIME_IDLE_BACKOFF_INITIAL_SECONDS must be > 0.0, got {backoff_initial}"
            )
        if backoff_max < backoff_initial:
            raise ValueError(
                f"REALTIME_IDLE_BACKOFF_MAX_SECONDS ({backoff_max}) must be >= "
                f"REALTIME_IDLE_BACKOFF_INITIAL_SECONDS ({backoff_initial})"
            )
        if not 0.0 <= jitter < 1.0:
            raise ValueError(
                f"REALTIME_POLL_JITTER_FRACTION must be in [0.0, 1.0), got {jitter}"
            )
        if wave_max_leads < 1:
            raise ValueError(
                f"REALTIME_WAVE_MAX_LEADS must be >= 1, got {wave_max_leads}"
            )
        if wave_max_wait <= 0.0:
            raise ValueError(
                f"REALTIME_WAVE_MAX_WAIT_SECONDS must be > 0.0, got {wave_max_wait}"
            )
        if failure_retry <= 0.0:
            raise ValueError(
                f"REALTIME_DISCOVERY_FAILURE_RETRY_SECONDS must be > 0.0, got {failure_retry}"
            )
        if first_publication_delay < 0.0:
            raise ValueError(
                f"REALTIME_FIRST_PUBLICATION_DELAY_SECONDS must be >= 0.0, got "
                f"{first_publication_delay}"
            )
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = IngestionSettings()

from domain.coverage import set_min_coverage_ratio  # noqa: E402
set_min_coverage_ratio(settings.ENSEMBLE_MIN_COVERAGE_RATIO)
