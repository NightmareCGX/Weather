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
    #
    # API_MAX_CONCURRENT_GATED_READS bounds concurrent gated reads per process. It
    # is enforced by an admission semaphore in api.core.reader_gate; before that
    # existed the setting was declared and set in production but read by nothing,
    # so the only real limit was the reader-lock pool's connection count.
    API_MAX_CONCURRENT_GATED_READS: Any = 16
    API_READER_LOCK_POOL_SIZE: Any = 16
    API_READER_LOCK_MAX_OVERFLOW: Any = 8
    API_READER_LOCK_POOL_TIMEOUT_SECONDS: Any = 5.0
    API_READER_GATE_TIMEOUT_SECONDS: Any = 30.0
    API_SHUTDOWN_DRAIN_TIMEOUT_SECONDS: Any = 40.0

    # ``max_pool_connections`` for each reader's s3fs client. One client is built
    # per cached ShardedV1Reader (see MAX_READERS in api.core.zarr), so this
    # multiplies across readers and must stay modest; the previous hardcoded 64
    # was far more than the fetch fan-out can use. Note the ingestion-only
    # S3_MAX_POOL_CONNECTIONS setting does not apply to the API.
    API_S3_MAX_POOL_CONNECTIONS: int = Field(default=16, ge=1, le=256)

    # Chunk-cache ceiling per ShardedV1Reader, counted in 100x100 float32 chunks
    # (~40 KB each), so this value multiplies directly into resident memory.
    #
    # The previous 2048 (~82 MB per reader) sat well past the point of any benefit:
    # simulating a realistic serving session (6 viewports x 3 variables x 8 leads,
    # plus a 30-member ensemble read) touches 630 distinct chunks, and the LRU hit
    # rate plateaus at 1024 entries (91.6%). At 512 the hit rate is 90.6% — one
    # point lower for a 4x memory reduction — which matters because every cached
    # reader holds its own cache and the API container runs near its memory limit.
    API_READER_MAX_CACHED_CHUNKS: int = Field(default=512, ge=1, le=8192)

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
    # Sized by measurement on the deployed topology, not by the worst case.
    # Production runs ``UVICORN_WORKERS=4`` on a 4-core ARM64 box shared with ~10
    # containers, so a value of N means up to 4*N fetch threads fleet-wide, and the
    # API container already sits at ~93% of its 4 GiB memory limit.
    #
    # Benchmarked against a same-host MinIO over s3fs, pinned to 4 CPUs, for a
    # 12-chunk window (the z=2 case):
    #     serial 30.5 ms | 2 workers 20.7 ms | 4 -> 22.6 | 8 -> 23.9 | 16 -> 23.6
    # The gain saturates at 2: a chunk fetch costs ~2.5 ms and is dominated by
    # s3fs/Python overhead plus zstd decode rather than wide-area RTT, so extra
    # workers only add contention. Note the pool is not even engaged in normal use
    # — a tile spans 1 chunk at z>=4, the range the map actually uses (it opens at
    # z=5 and moves to 6.5-8) — so this only shortens zoomed-out views.
    API_CHUNK_FETCH_WORKERS: int = Field(default=2, ge=1, le=32)

    # Ensemble member-read fan-out, per API process.
    #
    # A single GEFS point read fans out over up to 30 members. Measured on the same
    # 4-CPU topology against MinIO, for a 30-member point read:
    #     serial 110.4 ms | 2 -> 67.9 | 4 -> 66.2 | 8 -> 67.4 | 16 -> 68.2 | 30 -> 70.1
    # So the parallel win is real (~1.7x) but saturates by 4 workers, and larger
    # pools are measurably slower while holding far more threads on a small shared
    # host. Raise only if this runs on a box with more dedicated cores.
    API_MEMBER_FETCH_WORKERS: int = Field(default=4, ge=1, le=32)

    # Wind U/V component fetch fan-out, per API process.
    #
    # Synthesized wind raster tiles (wind_10m / wind_speed_10m) read independent
    # wind_u_10m and wind_v_10m shard components. A shared process-wide pool avoids
    # per-request OS thread creation/destruction churn while bounding concurrent
    # thread allocation on 4-core hosts.
    API_WIND_FETCH_WORKERS: int = Field(default=2, ge=1, le=16)

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
