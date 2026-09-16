from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_repository_root(start_path: Path | None = None) -> Path | None:
    """Deterministically locate the repository root directory.

    Searches upward from ``start_path`` (or this file's location) for canonical
    repository root markers (e.g. ``.git``, ``docker-compose.yml``).

    Args:
        start_path: Optional starting filesystem path for search (defaults to
            this file's location).

    Returns:
        Absolute Path to the repository root directory if identified, or None
        if executed in a container/installed layout where no repository root
        markers exist.
    """
    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        if (
            (parent / ".git").exists()
            or (parent / "docker-compose.yml").is_file()
            or (parent / "compose.yaml").is_file()
        ):
            return parent

    return None


def find_repository_env_file(start_path: Path | None = None) -> Path | None:
    """Deterministically locate the canonical repository-root ``.env`` file.

    Searches upward from ``start_path`` (or this file's location) for the
    repository root and resolves ``<repository-root>/.env``.

    Args:
        start_path: Optional starting filesystem path for search.

    Returns:
        Absolute Path to ``<repository-root>/.env`` if a repository root is
        identified, or None if no repository root is found (e.g. production
        container).
    """
    root = find_repository_root(start_path)
    if root is not None:
        return root / ".env"

    return None


ENV_FILE: Path | None = find_repository_env_file()


class Settings(BaseSettings):
    """Environment-driven settings for the API service (Pydantic BaseSettings)."""

    # mypy sees ``Any`` for the attribute types below because pydantic
    # settings declare them via the class namespace; they are validated at
    # instantiation time. The ``Any``-typed annotations keep strict mode clean
    # without suppressing broader checks.
    DATABASE_URL: Any = (
        "postgresql://weather_user:weather_password@localhost:5432/weather_db"
    )
    REDIS_URL: Any = "redis://localhost:6379/0"
    MINIO_ENDPOINT: Any = "localhost:9000"
    MINIO_ACCESS_KEY: Any = "minio_admin"
    MINIO_SECRET_KEY: Any = "minio_password"
    # Typed ``bool`` so a ``MINIO_SECURE=false`` env var (documented in
    # .env.example) is coerced to False by pydantic-settings. An untyped
    # ``Any``/str default would resolve to the truthy string ``'false'`` and
    # force HTTPS to a plain-HTTP endpoint (M14 fix).
    MINIO_SECURE: bool = False

    # Location place-autocomplete provider (Phase 1 Location Discovery).
    # ``geoapify`` (default V1) uses Geoapify Address Autocomplete;
    # ``locationiq`` uses LocationIQ Autocomplete;
    # ``google`` uses the Places API (New); ``mapbox`` uses Mapbox Geocoding.
    # The API key/token lives server-side and is never exposed to the browser.
    SEARCH_PROVIDER: Any = "geoapify"
    GEOAPIFY_API_KEY: Any = ""
    GEOAPIFY_API_BASE: Any = "https://api.geoapify.com/v1"
    LOCATIONIQ_API_KEY: Any = ""
    LOCATIONIQ_API_BASE: Any = "https://api.locationiq.com/v1"
    SEARCH_PRIMARY_TIMEOUT_MS: int = 2000
    SEARCH_FALLBACK_TIMEOUT_MS: int = 1500
    SEARCH_CIRCUIT_BREAKER_FAILURES: int = 2
    SEARCH_CIRCUIT_BREAKER_COOLDOWN_S: int = 60
    SEARCH_CACHE_TTL_SECONDS: int = 300
    SEARCH_CACHE_DISABLED: bool = False
    GOOGLE_PLACES_API_KEY: Any = ""
    GOOGLE_PLACES_API_BASE: Any = "https://places.googleapis.com/v1"
    GOOGLE_PLACES_REGION: Any = None
    # Socket timeout for Places HTTP calls (seconds).
    GOOGLE_PLACES_TIMEOUT: Any = 5.0
    MAPBOX_TOKEN: Any = ""

    # Coarse IP geolocation fallback via infrastructure headers (Cloudflare Managed Transforms).
    # Default False: externally supplied headers are NEVER trusted unless explicitly enabled
    # in an environment where requests are strictly authenticated/firewalled to Cloudflare origins.
    TRUST_CLOUDFLARE_LOCATION_HEADERS: bool = False

    # Reader-gate configuration (Zarr region-write concurrency).
    # The API serving tier participates in the SHARED store gate when reading
    # forecast Zarr stores so it never observes a store mid-re-ingest.
    API_MAX_CONCURRENT_GATED_READS: Any = 16
    API_READER_LOCK_POOL_SIZE: Any = 16
    API_READER_LOCK_MAX_OVERFLOW: Any = 8
    API_READER_LOCK_POOL_TIMEOUT_SECONDS: Any = 5.0
    API_READER_GATE_TIMEOUT_SECONDS: Any = 30.0
    API_SHUTDOWN_DRAIN_TIMEOUT_SECONDS: Any = 40.0

    # Map-tile PNG (IDAT) zlib compression level, constrained to the zlib range.
    # Level 1 is the serving default: on a 256x256 RGBA tile it compresses
    # several times faster than the zlib default of 6 for only a small size
    # increase, which shortens the cold-tile CPU phase and reduces the number
    # of tiles competing for the interpreter during a viewport burst. Declared
    # through ``Field`` so a malformed value is rejected at startup instead of
    # failing at import time.
    API_PNG_COMPRESS_LEVEL: int = Field(default=1, ge=0, le=9)

    # Shard chunk-fetch fan-out, per API process.
    #
    # Sized for the deployed host rather than for the worst case: production runs
    # ``UVICORN_WORKERS=4`` on a 4-core ARM64 box shared with ~10 containers
    # (Postgres, Redis, MinIO, three ingestion workers, gateway, frontend), so a
    # value of N means up to 4*N fetch threads fleet-wide, each of which also
    # needs CPU to zstd-decode its chunk.
    #
    # The fan-out only has to cover the chunk span of one tile, which is 1 chunk
    # at z>=4 — the range the map actually uses (it opens at z=5 and moves to
    # 6.5-8) — and 2-8 chunks at z=2-3. Four workers therefore already collapse
    # the common case to a single round trip; raising it only helps the rare
    # fully-zoomed-out view while adding contention.
    API_CHUNK_FETCH_WORKERS: int = Field(default=4, ge=1, le=32)

    # Ensemble member-read fan-out, per API process.
    #
    # Larger than the chunk pool because a single GEFS point read fans out over
    # up to 30 members, but still bounded for the same host reasons: member reads
    # must not monopolize a small shared machine or flood the object store.
    API_MEMBER_FETCH_WORKERS: int = Field(default=16, ge=1, le=32)

    # Elevation resolution for dynamic coordinates (UI metadata only).
    # ``none`` (default): elevation always unavailable (safe offline default).
    # ``open_meteo``: Open-Meteo Elevation API (Copernicus GLO-90 DEM).
    ELEVATION_PROVIDER: Any = "none"
    ELEVATION_BASE_URL: Any = "https://api.open-meteo.com/v1/elevation"
    ELEVATION_API_KEY: Any = ""
    ELEVATION_TIMEOUT_SECONDS: float = 2.0
    ELEVATION_CACHE_MAX: int = 10000
    ELEVATION_CACHE_DISABLED: bool = False

    # Minimum member coverage ratio for serving eligibility (Phase 3).
    # Constrained to (0.0, 1.0]; default 0.85 (85%).
    ENSEMBLE_MIN_COVERAGE_RATIO: float = 0.85

    # Background wind vector-field cache prewarm (cycle publication warm-up).
    # A lifespan-owned task periodically resolves the serving window's valid
    # times and computes cache-missing vector fields so real users never hit
    # the expensive cold path after a new cycle publishes. Passes are bounded
    # and throttled; see api/services/vector_prewarm.py.
    API_VECTOR_PREWARM_ENABLED: bool = True
    API_VECTOR_PREWARM_INTERVAL_SECONDS: float = 300.0
    API_VECTOR_PREWARM_HORIZON_HOURS: int = 48
    API_VECTOR_PREWARM_MODELS: Any = ["gfs", "gefs"]
    API_VECTOR_PREWARM_MAX_COMPUTES_PER_PASS: int = 6
    API_VECTOR_PREWARM_COMPUTE_THROTTLE_SECONDS: float = 2.0

    # Wind vector-field cache layering. The shared Redis L2 lets every worker
    # (and the prewarm loop) see one another's computed payloads so each valid
    # time is computed once fleet-wide; the small per-process L1 keeps serving
    # fast and unaffected when Redis is unavailable.
    API_VECTOR_CACHE_REDIS_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

from domain.coverage import set_min_coverage_ratio  # noqa: E402
set_min_coverage_ratio(settings.ENSEMBLE_MIN_COVERAGE_RATIO)
