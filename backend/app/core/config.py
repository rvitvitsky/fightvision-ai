"""Application settings, loaded from .env via pydantic-settings."""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FV_",
        case_sensitive=False,
        extra="ignore",
    )

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"

    # Media / storage
    media_dir: str = "./media"
    max_upload_mb: int = 500

    # CORS
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Job lifecycle
    job_ttl_seconds: int = 60 * 60 * 24  # 1 day
    queue_name: str = "fightvision"

    @property
    def uploads_dir(self) -> Path:
        return Path(self.media_dir) / "uploads"

    @property
    def results_dir(self) -> Path:
        return Path(self.media_dir) / "results"


settings = Settings()
