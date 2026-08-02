from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://weather_user:weather_password@localhost:5432/weather_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minio_admin"
    MINIO_SECRET_KEY: str = "minio_password"
    MINIO_SECURE: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
