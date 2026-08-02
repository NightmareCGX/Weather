"""Ingestion service configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Environment-driven settings for the ingestion service.

    Mirrors the configuration pattern of ``services/api``.
    """

    NOMADS_BASE_URL: str = "https://nomads.ncep.noaa.gov"
    NOAA_USER_AGENT: str = "weather-platform-ingestion/0.1.0"
    REQUEST_TIMEOUT_SECONDS: float = 30.0
    DOWNLOAD_RETRIES: int = 3
    RETRY_BACKOFF_SECONDS: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = IngestionSettings()
