from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
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


def _default_model_asset_dir() -> str:
    # Keep model artifacts in backend/out by default to avoid polluting source assets.
    return str(Path(__file__).resolve().parents[1] / "out" / "model_assets")


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
    live_runtime_data_enabled: bool = Field(default=True, alias="LIVE_RUNTIME_DATA_ENABLED")
    strict_live_data_required: bool = Field(default=True, alias="STRICT_LIVE_DATA_REQUIRED")
    live_data_cache_ttl_s: int = Field(default=3600, ge=10, alias="LIVE_DATA_CACHE_TTL_S")
    live_data_request_timeout_s: float = Field(default=8.0, ge=1.0, le=60.0, alias="LIVE_DATA_REQUEST_TIMEOUT_S")
    live_bank_holidays_url: str = Field(
        default="https://www.gov.uk/bank-holidays.json",
        alias="LIVE_BANK_HOLIDAYS_URL",
    )
    live_departure_profile_url: str = Field(default="", alias="LIVE_DEPARTURE_PROFILE_URL")
    live_stochastic_regimes_url: str = Field(default="", alias="LIVE_STOCHASTIC_REGIMES_URL")
    live_toll_topology_url: str = Field(default="", alias="LIVE_TOLL_TOPOLOGY_URL")
    live_toll_tariffs_url: str = Field(default="", alias="LIVE_TOLL_TARIFFS_URL")
    live_fuel_price_url: str = Field(default="", alias="LIVE_FUEL_PRICE_URL")
    live_fuel_auth_token: str = Field(default="", alias="LIVE_FUEL_AUTH_TOKEN")
    live_carbon_schedule_url: str = Field(default="", alias="LIVE_CARBON_SCHEDULE_URL")
    live_departure_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_DEPARTURE_MAX_AGE_DAYS")
    live_stochastic_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_STOCHASTIC_MAX_AGE_DAYS")
    live_toll_topology_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_TOLL_TOPOLOGY_MAX_AGE_DAYS")
    live_toll_tariffs_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_TOLL_TARIFFS_MAX_AGE_DAYS")
    live_fuel_max_age_days: int = Field(default=7, ge=1, le=3650, alias="LIVE_FUEL_MAX_AGE_DAYS")
    live_carbon_max_age_days: int = Field(default=180, ge=1, le=3650, alias="LIVE_CARBON_MAX_AGE_DAYS")
    quality_max_dropped_routes: int = Field(default=0, ge=0, alias="QUALITY_MAX_DROPPED_ROUTES")
    quality_min_fixture_routes: int = Field(default=10, ge=1, alias="QUALITY_MIN_FIXTURE_ROUTES")
    quality_min_unique_corridors: int = Field(default=6, ge=1, alias="QUALITY_MIN_UNIQUE_CORRIDORS")
    route_candidate_alternatives_max: int = Field(
        default=24,
        ge=1,
        le=60,
        alias="ROUTE_CANDIDATE_ALTERNATIVES_MAX",
    )
    route_candidate_via_budget: int = Field(
        default=48,
        ge=4,
        le=200,
        alias="ROUTE_CANDIDATE_VIA_BUDGET",
    )
    route_graph_enabled: bool = Field(default=True, alias="ROUTE_GRAPH_ENABLED")
    route_graph_asset_path: str = Field(default="", alias="ROUTE_GRAPH_ASSET_PATH")
    route_graph_k_paths: int = Field(default=24, ge=1, le=128, alias="ROUTE_GRAPH_K_PATHS")
    route_graph_max_hops: int = Field(default=220, ge=8, le=2000, alias="ROUTE_GRAPH_MAX_HOPS")
    route_graph_min_nodes: int = Field(default=100_000, ge=1, alias="ROUTE_GRAPH_MIN_NODES")
    route_graph_min_adjacency: int = Field(default=100_000, ge=1, alias="ROUTE_GRAPH_MIN_ADJACENCY")
    route_graph_max_state_budget: int = Field(
        default=120_000,
        ge=1000,
        le=2_000_000,
        alias="ROUTE_GRAPH_MAX_STATE_BUDGET",
    )
    route_graph_max_repeat_per_node: int = Field(
        default=1,
        ge=0,
        le=8,
        alias="ROUTE_GRAPH_MAX_REPEAT_PER_NODE",
    )
    route_graph_max_detour_ratio: float = Field(
        default=2.5,
        ge=1.0,
        le=8.0,
        alias="ROUTE_GRAPH_MAX_DETOUR_RATIO",
    )
    route_graph_fixture_max_distance_m: float = Field(
        default=15_000.0,
        ge=100.0,
        le=250_000.0,
        alias="ROUTE_GRAPH_FIXTURE_MAX_DISTANCE_M",
    )
    route_graph_via_landmarks_per_path: int = Field(
        default=2,
        ge=1,
        le=4,
        alias="ROUTE_GRAPH_VIA_LANDMARKS_PER_PATH",
    )
    route_graph_streaming_load_enabled: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_STREAMING_LOAD_ENABLED",
    )
    route_graph_streaming_size_threshold_mb: int = Field(
        default=256,
        ge=32,
        le=16_384,
        alias="ROUTE_GRAPH_STREAMING_SIZE_THRESHOLD_MB",
    )
    route_graph_streaming_max_nodes: int = Field(
        default=220_000,
        ge=10_000,
        le=2_000_000,
        alias="ROUTE_GRAPH_STREAMING_MAX_NODES",
    )
    route_graph_streaming_max_edges: int = Field(
        default=480_000,
        ge=20_000,
        le=6_000_000,
        alias="ROUTE_GRAPH_STREAMING_MAX_EDGES",
    )
    route_graph_strict_required: bool = Field(default=True, alias="ROUTE_GRAPH_STRICT_REQUIRED")
    manifest_signing_secret: str = Field(
        default="dev-manifest-signing-secret",
        alias="MANIFEST_SIGNING_SECRET",
    )
    departure_require_empirical_profiles: bool = Field(
        default=True,
        alias="DEPARTURE_REQUIRE_EMPIRICAL_PROFILES",
    )
    departure_allow_synthetic_profiles: bool = Field(
        default=False,
        alias="DEPARTURE_ALLOW_SYNTHETIC_PROFILES",
    )
    stochastic_require_empirical_calibration: bool = Field(
        default=True,
        alias="STOCHASTIC_REQUIRE_EMPIRICAL_CALIBRATION",
    )
    stochastic_allow_synthetic_calibration: bool = Field(
        default=False,
        alias="STOCHASTIC_ALLOW_SYNTHETIC_CALIBRATION",
    )

    risk_cvar_alpha: float = Field(default=0.95, ge=0.5, le=0.999, alias="RISK_CVAR_ALPHA")
    model_asset_dir: str = Field(default_factory=_default_model_asset_dir, alias="MODEL_ASSET_DIR")
    terrain_dem_fail_closed_uk: bool = Field(default=True, alias="TERRAIN_DEM_FAIL_CLOSED_UK")
    terrain_dem_coverage_min_uk: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        alias="TERRAIN_DEM_COVERAGE_MIN_UK",
    )
    terrain_uk_only_support: bool = Field(default=True, alias="TERRAIN_UK_ONLY_SUPPORT")
    terrain_uk_bbox: str = Field(
        default="49.75,61.10,-8.75,2.25",
        alias="TERRAIN_UK_BBOX",
    )
    terrain_dem_resolution_m: float = Field(default=75.0, ge=1.0, alias="TERRAIN_DEM_RESOLUTION_M")
    terrain_dem_tile_cache_max_mb: int = Field(
        default=256,
        ge=32,
        le=4096,
        alias="TERRAIN_DEM_TILE_CACHE_MAX_MB",
    )
    terrain_allow_synthetic_grid: bool = Field(
        default=False,
        alias="TERRAIN_ALLOW_SYNTHETIC_GRID",
    )
    terrain_sample_spacing_m: float = Field(default=75.0, ge=5.0, alias="TERRAIN_SAMPLE_SPACING_M")
    terrain_max_samples_per_route: int = Field(
        default=6000,
        ge=200,
        le=50_000,
        alias="TERRAIN_MAX_SAMPLES_PER_ROUTE",
    )
    terrain_physics_version: str = Field(default="uk_v3", alias="TERRAIN_PHYSICS_VERSION")
    carbon_scope_mode: str = Field(default="ttw", alias="CARBON_SCOPE_MODE")
    carbon_policy_scenario: str = Field(default="central", alias="CARBON_POLICY_SCENARIO")

    @model_validator(mode="after")
    def _enforce_strict_runtime_defaults(self) -> "Settings":
        # Hard-strict runtime policy: production paths are always strict and
        # synthetic generation is disallowed outside explicit test tooling.
        self.live_runtime_data_enabled = True
        self.strict_live_data_required = True
        self.route_graph_enabled = True
        self.route_graph_strict_required = True
        self.departure_require_empirical_profiles = True
        self.departure_allow_synthetic_profiles = False
        self.stochastic_require_empirical_calibration = True
        self.stochastic_allow_synthetic_calibration = False
        self.terrain_allow_synthetic_grid = False
        return self


settings = Settings()
