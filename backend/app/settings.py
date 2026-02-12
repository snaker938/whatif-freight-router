from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _running_in_docker() -> bool:
    """Best-effort check for container execution.

    Used only to pick sensible defaults. Environment variables always win.
    """
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"


def _default_osrm_base_url() -> str:
    # In docker-compose, OSRM is reachable by service name "osrm".
    # When running the backend directly on the host, OSRM is typically exposed on localhost:5000.
    return "http://osrm:5000" if _running_in_docker() else "http://localhost:5000"


class Settings(BaseSettings):
    """Validated settings (env-driven), keeping config out of code for easy extension."""

    model_config = SettingsConfigDict(
        # Support both "repo root/.env" (docker compose) and "backend/.env" (local dev)
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    osrm_base_url: str = Field(default_factory=_default_osrm_base_url, alias="OSRM_BASE_URL")
    osrm_profile: str = Field(default="driving", alias="OSRM_PROFILE")

    out_dir: str = Field(default="/app/out", alias="OUT_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Batch control (so 100+ OD runs don't overload OSRM)
    batch_concurrency: int = Field(default=8, alias="BATCH_CONCURRENCY")
    route_cache_ttl_s: int = Field(default=600, alias="ROUTE_CACHE_TTL_S")
    route_cache_max_entries: int = Field(default=1024, alias="ROUTE_CACHE_MAX_ENTRIES")
    offline_fallback_enabled: bool = Field(default=True, alias="OFFLINE_FALLBACK_ENABLED")
    manifest_signing_secret: str = Field(
        default="dev-manifest-signing-secret",
        alias="MANIFEST_SIGNING_SECRET",
    )
    rbac_enabled: bool = Field(default=False, alias="RBAC_ENABLED")
    rbac_user_token: str = Field(default="dev-user-token", alias="RBAC_USER_TOKEN")
    rbac_admin_token: str = Field(default="dev-admin-token", alias="RBAC_ADMIN_TOKEN")


settings = Settings()
