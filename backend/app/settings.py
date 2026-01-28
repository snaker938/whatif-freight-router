from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings (env-driven), keeping config out of code for easy extension."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    osrm_base_url: str = Field(default="http://osrm:5000", alias="OSRM_BASE_URL")
    osrm_profile: str = Field(default="driving", alias="OSRM_PROFILE")

    out_dir: str = Field(default="/app/out", alias="OUT_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Batch control (so 100+ OD runs don't overload OSRM)
    batch_concurrency: int = Field(default=8, alias="BATCH_CONCURRENCY")


settings = Settings()
