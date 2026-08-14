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

    # Location place-autocomplete provider (ACCEPTANCE_REMEDIATION_PLAN §13).
    # ``google`` (default) uses the Places API (New) via
    # ``api/services/places.py``; ``mapbox`` uses the Mapbox Geocoding API.
    # The API key/token lives server-side and is never exposed to the browser.
    SEARCH_PROVIDER: Any = "google"
    GOOGLE_PLACES_API_KEY: Any = ""
    GOOGLE_PLACES_API_BASE: Any = "https://places.googleapis.com/v1"
    GOOGLE_PLACES_REGION: Any = None
    # Socket timeout for Places HTTP calls (seconds).
    GOOGLE_PLACES_TIMEOUT: Any = 5.0
    MAPBOX_TOKEN: Any = ""

    # Elevation resolution (ACCEPTANCE_REMEDIATION_PLAN §15).
    # ``dem`` (default): local/server-side DEM via api/services/elevation.py.
    # ``google``: Google Elevation API (ToS restricts caching; not the default).
    # ``none``: elevation always unavailable.
    ELEVATION_PROVIDER: Any = "dem"
    # Path/URL of the DEM store (a global xarray-readable Zarr/NetCDF with
    # latitude/longitude coords and an ``elevation`` variable in meters).
    # Empty means no DEM configured -> elevation unavailable.
    DEM_DATA_PATH: Any = ""
    ELEVATION_CACHE_MAX: Any = 4096
    ELEVATION_CACHE_DISABLED: Any = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
