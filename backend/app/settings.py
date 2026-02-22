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
    live_data_request_timeout_s: float = Field(default=20.0, ge=1.0, le=120.0, alias="LIVE_DATA_REQUEST_TIMEOUT_S")
    live_bank_holidays_url: str = Field(
        default="https://www.gov.uk/bank-holidays.json",
        alias="LIVE_BANK_HOLIDAYS_URL",
    )
    live_departure_profile_url: str = Field(default="", alias="LIVE_DEPARTURE_PROFILE_URL")
    live_departure_require_url_in_strict: bool = Field(
        default=True,
        alias="LIVE_DEPARTURE_REQUIRE_URL_IN_STRICT",
    )
    live_departure_allow_signed_fallback: bool = Field(
        default=False,
        alias="LIVE_DEPARTURE_ALLOW_SIGNED_FALLBACK",
    )
    live_departure_allowed_hosts: str = Field(
        default="",
        alias="LIVE_DEPARTURE_ALLOWED_HOSTS",
    )
    # Legacy scenario profile URL is kept for compatibility, but strict runtime
    # requires LIVE_SCENARIO_COEFFICIENT_URL.
    live_scenario_profile_url: str = Field(default="", alias="LIVE_SCENARIO_PROFILE_URL")
    live_scenario_coefficient_url: str = Field(default="", alias="LIVE_SCENARIO_COEFFICIENT_URL")
    live_scenario_webtris_sites_url: str = Field(
        default="https://webtris.nationalhighways.co.uk/api/v1.0/sites",
        alias="LIVE_SCENARIO_WEBTRIS_SITES_URL",
    )
    live_scenario_webtris_daily_url: str = Field(
        default="https://webtris.nationalhighways.co.uk/api/v1.0/reports/daily",
        alias="LIVE_SCENARIO_WEBTRIS_DAILY_URL",
    )
    live_scenario_traffic_england_url: str = Field(
        default=(
            "https://www.trafficengland.com/api/events/getByRoad"
            "?road={road}&events=CONGESTION,INCIDENT,ROADWORKS"
            "&direction=All&includeUnconfirmedRoadworks=true"
        ),
        alias="LIVE_SCENARIO_TRAFFIC_ENGLAND_URL",
    )
    live_scenario_dft_counts_url: str = Field(
        default="https://roadtraffic.dft.gov.uk/api/raw-counts",
        alias="LIVE_SCENARIO_DFT_COUNTS_URL",
    )
    live_scenario_dft_max_pages: int = Field(
        default=8,
        ge=1,
        le=50,
        alias="LIVE_SCENARIO_DFT_MAX_PAGES",
    )
    live_scenario_webtris_nearest_sites: int = Field(
        default=6,
        ge=1,
        le=20,
        alias="LIVE_SCENARIO_WEBTRIS_NEAREST_SITES",
    )
    live_scenario_dft_nearest_limit: int = Field(
        default=96,
        ge=1,
        le=2000,
        alias="LIVE_SCENARIO_DFT_NEAREST_LIMIT",
    )
    live_scenario_dft_max_distance_km: float = Field(
        default=120.0,
        ge=1.0,
        le=2000.0,
        alias="LIVE_SCENARIO_DFT_MAX_DISTANCE_KM",
    )
    live_scenario_dft_min_station_count: int = Field(
        default=5,
        ge=1,
        le=200,
        alias="LIVE_SCENARIO_DFT_MIN_STATION_COUNT",
    )
    live_scenario_open_meteo_forecast_url: str = Field(
        default=(
            "https://api.open-meteo.com/v1/forecast"
            "?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,wind_speed_10m,precipitation,weather_code"
        ),
        alias="LIVE_SCENARIO_OPEN_METEO_FORECAST_URL",
    )
    live_scenario_open_meteo_archive_url: str = Field(
        default=(
            "https://archive-api.open-meteo.com/v1/archive"
            "?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
            "&hourly=temperature_2m,wind_speed_10m,precipitation"
        ),
        alias="LIVE_SCENARIO_OPEN_METEO_ARCHIVE_URL",
    )
    live_scenario_require_url_in_strict: bool = Field(
        default=True,
        alias="LIVE_SCENARIO_REQUIRE_URL_IN_STRICT",
    )
    live_scenario_allow_signed_fallback: bool = Field(
        default=False,
        alias="LIVE_SCENARIO_ALLOW_SIGNED_FALLBACK",
    )
    live_scenario_allowed_hosts: str = Field(
        default=(
            "webtris.nationalhighways.co.uk,"
            "www.trafficengland.com,"
            "roadtraffic.dft.gov.uk,"
            "api.open-meteo.com,"
            "archive-api.open-meteo.com"
        ),
        alias="LIVE_SCENARIO_ALLOWED_HOSTS",
    )
    live_scenario_cache_ttl_seconds: int = Field(
        default=300,
        ge=30,
        le=86_400,
        alias="LIVE_SCENARIO_CACHE_TTL_SECONDS",
    )
    live_scenario_max_age_minutes: int = Field(
        default=120,
        ge=5,
        le=10_080,
        alias="LIVE_SCENARIO_MAX_AGE_MINUTES",
    )
    live_scenario_coefficient_max_age_minutes: int = Field(
        default=120,
        ge=5,
        le=10_080,
        alias="LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES",
    )
    live_stochastic_regimes_url: str = Field(default="", alias="LIVE_STOCHASTIC_REGIMES_URL")
    live_stochastic_require_url_in_strict: bool = Field(
        default=True,
        alias="LIVE_STOCHASTIC_REQUIRE_URL_IN_STRICT",
    )
    live_stochastic_allow_signed_fallback: bool = Field(
        default=False,
        alias="LIVE_STOCHASTIC_ALLOW_SIGNED_FALLBACK",
    )
    live_stochastic_allowed_hosts: str = Field(
        default="",
        alias="LIVE_STOCHASTIC_ALLOWED_HOSTS",
    )
    live_toll_topology_url: str = Field(default="", alias="LIVE_TOLL_TOPOLOGY_URL")
    live_toll_topology_require_url_in_strict: bool = Field(
        default=True,
        alias="LIVE_TOLL_TOPOLOGY_REQUIRE_URL_IN_STRICT",
    )
    live_toll_topology_allow_signed_fallback: bool = Field(
        default=False,
        alias="LIVE_TOLL_TOPOLOGY_ALLOW_SIGNED_FALLBACK",
    )
    live_toll_tariffs_url: str = Field(default="", alias="LIVE_TOLL_TARIFFS_URL")
    live_toll_tariffs_require_url_in_strict: bool = Field(
        default=True,
        alias="LIVE_TOLL_TARIFFS_REQUIRE_URL_IN_STRICT",
    )
    live_toll_tariffs_allow_signed_fallback: bool = Field(
        default=False,
        alias="LIVE_TOLL_TARIFFS_ALLOW_SIGNED_FALLBACK",
    )
    live_toll_allowed_hosts: str = Field(
        default="",
        alias="LIVE_TOLL_ALLOWED_HOSTS",
    )
    live_fuel_price_url: str = Field(default="", alias="LIVE_FUEL_PRICE_URL")
    live_fuel_auth_token: str = Field(default="", alias="LIVE_FUEL_AUTH_TOKEN")
    live_fuel_api_key: str = Field(default="", alias="LIVE_FUEL_API_KEY")
    live_fuel_api_key_header: str = Field(default="X-API-Key", alias="LIVE_FUEL_API_KEY_HEADER")
    live_fuel_require_url_in_strict: bool = Field(default=True, alias="LIVE_FUEL_REQUIRE_URL_IN_STRICT")
    live_fuel_allow_signed_fallback: bool = Field(default=False, alias="LIVE_FUEL_ALLOW_SIGNED_FALLBACK")
    live_fuel_require_signature: bool = Field(default=True, alias="LIVE_FUEL_REQUIRE_SIGNATURE")
    live_fuel_allowed_hosts: str = Field(default="", alias="LIVE_FUEL_ALLOWED_HOSTS")
    live_carbon_schedule_url: str = Field(default="", alias="LIVE_CARBON_SCHEDULE_URL")
    live_carbon_require_url_in_strict: bool = Field(
        default=True,
        alias="LIVE_CARBON_REQUIRE_URL_IN_STRICT",
    )
    live_carbon_allow_signed_fallback: bool = Field(
        default=False,
        alias="LIVE_CARBON_ALLOW_SIGNED_FALLBACK",
    )
    live_carbon_allowed_hosts: str = Field(default="", alias="LIVE_CARBON_ALLOWED_HOSTS")
    live_departure_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_DEPARTURE_MAX_AGE_DAYS")
    live_stochastic_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_STOCHASTIC_MAX_AGE_DAYS")
    live_toll_topology_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_TOLL_TOPOLOGY_MAX_AGE_DAYS")
    live_toll_tariffs_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_TOLL_TARIFFS_MAX_AGE_DAYS")
    live_fuel_max_age_days: int = Field(default=7, ge=1, le=3650, alias="LIVE_FUEL_MAX_AGE_DAYS")
    live_carbon_max_age_days: int = Field(default=180, ge=1, le=3650, alias="LIVE_CARBON_MAX_AGE_DAYS")
    live_scenario_max_age_days: int = Field(default=30, ge=1, le=3650, alias="LIVE_SCENARIO_MAX_AGE_DAYS")
    scenario_require_signature: bool = Field(default=True, alias="SCENARIO_REQUIRE_SIGNATURE")
    scenario_min_observed_mode_row_share: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        alias="SCENARIO_MIN_OBSERVED_MODE_ROW_SHARE",
    )
    scenario_max_projection_dominant_context_share: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        alias="SCENARIO_MAX_PROJECTION_DOMINANT_CONTEXT_SHARE",
    )
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
    route_graph_scenario_jaccard_max: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        alias="ROUTE_GRAPH_SCENARIO_JACCARD_MAX",
    )
    route_graph_scenario_jaccard_floor: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        alias="ROUTE_GRAPH_SCENARIO_JACCARD_FLOOR",
    )
    route_graph_scenario_separability_fail: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_SCENARIO_SEPARABILITY_FAIL",
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
    risk_family: str = Field(default="cvar_excess", alias="RISK_FAMILY")
    risk_family_theta: float = Field(default=1.0, ge=0.001, le=25.0, alias="RISK_FAMILY_THETA")
    risk_dominance_min_probability: float = Field(
        default=0.55,
        ge=0.50,
        le=0.99,
        alias="RISK_DOMINANCE_MIN_PROBABILITY",
    )
    risk_dominance_pair_samples: int = Field(
        default=96,
        ge=16,
        le=1024,
        alias="RISK_DOMINANCE_PAIR_SAMPLES",
    )
    risk_objective_sample_cap: int = Field(
        default=160,
        ge=16,
        le=2048,
        alias="RISK_OBJECTIVE_SAMPLE_CAP",
    )
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
    live_terrain_dem_url_template: str = Field(
        default="https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif",
        alias="LIVE_TERRAIN_DEM_URL_TEMPLATE",
    )
    live_terrain_require_url_in_strict: bool = Field(
        default=True,
        alias="LIVE_TERRAIN_REQUIRE_URL_IN_STRICT",
    )
    live_terrain_allow_signed_fallback: bool = Field(
        default=False,
        alias="LIVE_TERRAIN_ALLOW_SIGNED_FALLBACK",
    )
    live_terrain_allowed_hosts: str = Field(
        default="s3.amazonaws.com",
        alias="LIVE_TERRAIN_ALLOWED_HOSTS",
    )
    live_terrain_tile_zoom: int = Field(
        default=8,
        ge=0,
        le=15,
        alias="LIVE_TERRAIN_TILE_ZOOM",
    )
    live_terrain_tile_max_age_days: int = Field(
        default=7,
        ge=0,
        le=3650,
        alias="LIVE_TERRAIN_TILE_MAX_AGE_DAYS",
    )
    live_terrain_cache_dir: str = Field(default="", alias="LIVE_TERRAIN_CACHE_DIR")
    live_terrain_cache_max_tiles: int = Field(
        default=1024,
        ge=16,
        le=100_000,
        alias="LIVE_TERRAIN_CACHE_MAX_TILES",
    )
    live_terrain_cache_max_mb: int = Field(
        default=2048,
        ge=64,
        le=262_144,
        alias="LIVE_TERRAIN_CACHE_MAX_MB",
    )
    live_terrain_fetch_retries: int = Field(
        default=2,
        ge=1,
        le=8,
        alias="LIVE_TERRAIN_FETCH_RETRIES",
    )
    live_terrain_max_remote_tiles_per_route: int = Field(
        default=96,
        ge=1,
        le=4096,
        alias="LIVE_TERRAIN_MAX_REMOTE_TILES_PER_ROUTE",
    )
    live_terrain_circuit_breaker_failures: int = Field(
        default=8,
        ge=1,
        le=128,
        alias="LIVE_TERRAIN_CIRCUIT_BREAKER_FAILURES",
    )
    live_terrain_circuit_breaker_cooldown_s: int = Field(
        default=30,
        ge=1,
        le=3600,
        alias="LIVE_TERRAIN_CIRCUIT_BREAKER_COOLDOWN_S",
    )
    live_terrain_enable_in_tests: bool = Field(
        default=False,
        alias="LIVE_TERRAIN_ENABLE_IN_TESTS",
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
        self.live_terrain_require_url_in_strict = True
        self.live_terrain_allow_signed_fallback = False
        self.live_departure_require_url_in_strict = True
        self.live_departure_allow_signed_fallback = False
        self.live_stochastic_require_url_in_strict = True
        self.live_stochastic_allow_signed_fallback = False
        self.live_toll_topology_require_url_in_strict = True
        self.live_toll_topology_allow_signed_fallback = False
        self.live_toll_tariffs_require_url_in_strict = True
        self.live_toll_tariffs_allow_signed_fallback = False
        self.live_fuel_require_url_in_strict = True
        self.live_fuel_allow_signed_fallback = False
        self.live_fuel_require_signature = True
        self.live_carbon_require_url_in_strict = True
        self.live_carbon_allow_signed_fallback = False
        self.live_scenario_require_url_in_strict = True
        self.live_scenario_allow_signed_fallback = False
        self.scenario_require_signature = True
        rf = str(self.risk_family or "cvar_excess").strip().lower()
        if rf not in {"cvar_excess", "entropic", "downside_semivariance"}:
            rf = "cvar_excess"
        self.risk_family = rf
        return self


settings = Settings()
