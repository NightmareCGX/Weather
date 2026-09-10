from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the API service (Pydantic BaseSettings)."""
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

from domain.coverage import set_min_coverage_ratio  # noqa: E402
set_min_coverage_ratio(settings.ENSEMBLE_MIN_COVERAGE_RATIO)
