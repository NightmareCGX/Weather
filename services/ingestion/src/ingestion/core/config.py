"""Ingestion service configuration."""

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Environment-driven settings for the ingestion service.

    Mirrors the configuration pattern of ``services/api``.
    """

    # Attribute types are ``Any`` because pydantic settings declare them via
    # the class namespace; they are validated at instantiation time. This keeps
    # strict typing clean without suppressing broader checks.
    NOMADS_BASE_URL: Any = "https://nomads.ncep.noaa.gov"
    NOAA_USER_AGENT: Any = "weather-platform-ingestion/0.1.0"
    REQUEST_TIMEOUT_SECONDS: Any = 30.0
    DOWNLOAD_RETRIES: Any = 3
    RETRY_BACKOFF_SECONDS: Any = 1.0

    #: PostgreSQL catalog connection used to record ingested runs so the API
    #: serving tier can discover and serve them.
    DATABASE_URL: Any = (
        "postgresql://weather_user:weather_password@localhost:5432/weather_db"
    )

    MINIO_ENDPOINT: Any = "localhost:9000"
    MINIO_ACCESS_KEY: Any = "minio_admin"
    MINIO_SECRET_KEY: Any = "minio_password"
    # Typed ``bool`` so a ``MINIO_SECURE=false`` env var (documented in
    # .env.example) is coerced to False by pydantic-settings. An untyped
    # ``Any``/str default would resolve to the truthy string ``'false'`` and
    # force HTTPS to a plain-HTTP endpoint (M14 fix).
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: Any = "weather-data"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = IngestionSettings()
