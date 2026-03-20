from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import gettempdir

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOGGER = logging.getLogger(__name__)


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


def _resolve_writable_out_dir(configured_out_dir: str) -> str:
    candidates = (
        Path(configured_out_dir),
        Path(__file__).resolve().parents[1] / "out",
        Path.cwd() / "out",
        Path(gettempdir()) / "whatif-freight-router" / "out",
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".writetest"
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return str(candidate)
        except OSError:
            continue
    return str(Path.cwd() / "out")


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
    dev_route_debug_console_enabled: bool = Field(
        default=False,
        alias="DEV_ROUTE_DEBUG_CONSOLE_ENABLED",
    )
    dev_route_debug_include_sensitive: bool = Field(
        default=False,
        alias="DEV_ROUTE_DEBUG_INCLUDE_SENSITIVE",
    )
    dev_route_debug_max_calls_per_request: int = Field(
        default=2000,
        ge=50,
        le=20000,
        alias="DEV_ROUTE_DEBUG_MAX_CALLS_PER_REQUEST",
    )
    dev_route_debug_trace_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        alias="DEV_ROUTE_DEBUG_TRACE_TTL_SECONDS",
    )
    dev_route_debug_max_request_traces: int = Field(
        default=128,
        ge=1,
        le=5000,
        alias="DEV_ROUTE_DEBUG_MAX_REQUEST_TRACES",
    )
    dev_route_debug_return_raw_payloads: bool = Field(
        default=False,
        alias="DEV_ROUTE_DEBUG_RETURN_RAW_PAYLOADS",
    )
    dev_route_debug_max_raw_body_chars: int = Field(
        default=2048,
        ge=0,
        le=500_000,
        alias="DEV_ROUTE_DEBUG_MAX_RAW_BODY_CHARS",
    )

    # Batch control (so 100+ OD runs don't overload OSRM)
    batch_concurrency: int = Field(default=8, alias="BATCH_CONCURRENCY")
    route_cache_ttl_s: int = Field(default=600, alias="ROUTE_CACHE_TTL_S")
    route_cache_max_entries: int = Field(default=1024, alias="ROUTE_CACHE_MAX_ENTRIES")
    live_runtime_data_enabled: bool = Field(default=True, alias="LIVE_RUNTIME_DATA_ENABLED")
    strict_live_data_required: bool = Field(default=True, alias="STRICT_LIVE_DATA_REQUIRED")
    live_route_compute_refresh_mode: str = Field(
        default="route_compute",
        alias="LIVE_ROUTE_COMPUTE_REFRESH_MODE",
    )
    live_route_compute_require_all_expected: bool = Field(
        default=False,
        alias="LIVE_ROUTE_COMPUTE_REQUIRE_ALL_EXPECTED",
    )
    live_route_compute_force_no_cache_headers: bool = Field(
        default=False,
        alias="LIVE_ROUTE_COMPUTE_FORCE_NO_CACHE_HEADERS",
    )
    live_route_compute_force_uncached: bool = Field(
        default=True,
        alias="LIVE_ROUTE_COMPUTE_FORCE_UNCACHED",
    )
    live_route_compute_prefetch_timeout_ms: int = Field(
        default=300_000,
        ge=1_000,
        le=600_000,
        alias="LIVE_ROUTE_COMPUTE_PREFETCH_TIMEOUT_MS",
    )
    live_route_compute_prefetch_max_concurrency: int = Field(
        default=8,
        ge=1,
        le=16,
        alias="LIVE_ROUTE_COMPUTE_PREFETCH_MAX_CONCURRENCY",
    )
    live_route_compute_probe_terrain: bool = Field(
        default=False,
        alias="LIVE_ROUTE_COMPUTE_PROBE_TERRAIN",
    )
    live_data_cache_ttl_s: int = Field(default=3600, ge=1, alias="LIVE_DATA_CACHE_TTL_S")
    live_data_request_timeout_s: float = Field(default=20.0, ge=1.0, le=120.0, alias="LIVE_DATA_REQUEST_TIMEOUT_S")
    live_http_max_attempts: int = Field(default=6, ge=1, le=12, alias="LIVE_HTTP_MAX_ATTEMPTS")
    live_http_retry_deadline_ms: int = Field(default=30_000, ge=100, le=300_000, alias="LIVE_HTTP_RETRY_DEADLINE_MS")
    live_http_retry_backoff_base_ms: int = Field(
        default=200,
        ge=0,
        le=30_000,
        alias="LIVE_HTTP_RETRY_BACKOFF_BASE_MS",
    )
    live_http_retry_backoff_max_ms: int = Field(
        default=2_500,
        ge=0,
        le=120_000,
        alias="LIVE_HTTP_RETRY_BACKOFF_MAX_MS",
    )
    live_http_retry_jitter_ms: int = Field(
        default=150,
        ge=0,
        le=10_000,
        alias="LIVE_HTTP_RETRY_JITTER_MS",
    )
    live_http_retry_respect_retry_after: bool = Field(
        default=True,
        alias="LIVE_HTTP_RETRY_RESPECT_RETRY_AFTER",
    )
    live_http_retryable_status_codes: str = Field(
        default="429,500,502,503,504",
        alias="LIVE_HTTP_RETRYABLE_STATUS_CODES",
    )
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
        default=4,
        ge=1,
        le=50,
        alias="LIVE_SCENARIO_DFT_MAX_PAGES",
    )
    live_scenario_webtris_nearest_sites: int = Field(
        default=3,
        ge=1,
        le=20,
        alias="LIVE_SCENARIO_WEBTRIS_NEAREST_SITES",
    )
    live_scenario_dft_nearest_limit: int = Field(
        default=64,
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
        default=3,
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
        ge=1,
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
        default=4320,
        ge=5,
        le=10_080,
        alias="LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES",
    )
    live_scenario_allow_partial_sources_strict: bool = Field(
        default=False,
        alias="LIVE_SCENARIO_ALLOW_PARTIAL_SOURCES_STRICT",
    )
    live_scenario_min_source_count_strict: int = Field(
        default=4,
        ge=1,
        le=4,
        alias="LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT",
    )
    live_scenario_min_coverage_overall_strict: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        alias="LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT",
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
    live_bank_holidays_allowed_hosts: str = Field(
        default="www.gov.uk,gov.uk",
        alias="LIVE_BANK_HOLIDAYS_ALLOWED_HOSTS",
    )
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
    route_candidate_prefilter_multiplier: int = Field(
        default=3,
        ge=1,
        le=12,
        alias="ROUTE_CANDIDATE_PREFILTER_MULTIPLIER",
    )
    route_candidate_prefilter_multiplier_long: int = Field(
        default=2,
        ge=1,
        le=12,
        alias="ROUTE_CANDIDATE_PREFILTER_MULTIPLIER_LONG",
    )
    route_candidate_prefilter_long_distance_threshold_km: float = Field(
        default=180.0,
        ge=0.0,
        le=5000.0,
        alias="ROUTE_CANDIDATE_PREFILTER_LONG_DISTANCE_THRESHOLD_KM",
    )
    route_pareto_backfill_enabled: bool = Field(
        default=False,
        alias="ROUTE_PARETO_BACKFILL_ENABLED",
    )
    route_pareto_backfill_min_alternatives: int = Field(
        default=1,
        ge=1,
        le=60,
        alias="ROUTE_PARETO_BACKFILL_MIN_ALTERNATIVES",
    )
    route_candidate_via_budget: int = Field(
        default=48,
        ge=4,
        le=200,
        alias="ROUTE_CANDIDATE_VIA_BUDGET",
    )
    route_option_segment_cap: int = Field(
        default=160,
        ge=32,
        le=5000,
        alias="ROUTE_OPTION_SEGMENT_CAP",
    )
    route_option_segment_cap_long: int = Field(
        default=40,
        ge=32,
        le=5000,
        alias="ROUTE_OPTION_SEGMENT_CAP_LONG",
    )
    route_option_long_distance_threshold_km: float = Field(
        default=160.0,
        ge=20.0,
        le=5000.0,
        alias="ROUTE_OPTION_LONG_DISTANCE_THRESHOLD_KM",
    )
    route_option_reuse_scenario_policy: bool = Field(
        default=True,
        alias="ROUTE_OPTION_REUSE_SCENARIO_POLICY",
    )
    route_option_tod_bucket_s: int = Field(
        default=900,
        ge=0,
        le=3600,
        alias="ROUTE_OPTION_TOD_BUCKET_S",
    )
    route_option_energy_speed_bin_kph: float = Field(
        default=3.0,
        ge=0.0,
        le=80.0,
        alias="ROUTE_OPTION_ENERGY_SPEED_BIN_KPH",
    )
    route_option_energy_grade_bin_pct: float = Field(
        default=0.5,
        ge=0.0,
        le=20.0,
        alias="ROUTE_OPTION_ENERGY_GRADE_BIN_PCT",
    )
    route_baseline_duration_multiplier: float = Field(
        default=1.16,
        ge=1.0,
        le=2.0,
        alias="ROUTE_BASELINE_DURATION_MULTIPLIER",
    )
    route_baseline_distance_multiplier: float = Field(
        default=1.13,
        ge=1.0,
        le=2.0,
        alias="ROUTE_BASELINE_DISTANCE_MULTIPLIER",
    )
    route_pipeline_default_mode: str = Field(
        default="legacy",
        alias="ROUTE_PIPELINE_DEFAULT_MODE",
    )
    route_pipeline_request_override_enabled: bool = Field(
        default=True,
        alias="ROUTE_PIPELINE_REQUEST_OVERRIDE_ENABLED",
    )
    route_pipeline_default_seed: int = Field(
        default=20260320,
        ge=0,
        alias="ROUTE_PIPELINE_DEFAULT_SEED",
    )
    route_pipeline_search_budget: int = Field(
        default=6,
        ge=1,
        le=128,
        alias="ROUTE_PIPELINE_SEARCH_BUDGET",
    )
    route_pipeline_evidence_budget: int = Field(
        default=3,
        ge=0,
        le=64,
        alias="ROUTE_PIPELINE_EVIDENCE_BUDGET",
    )
    route_pipeline_cert_world_count: int = Field(
        default=64,
        ge=10,
        le=500,
        alias="ROUTE_PIPELINE_CERT_WORLD_COUNT",
    )
    route_pipeline_certificate_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        alias="ROUTE_PIPELINE_CERTIFICATE_THRESHOLD",
    )
    route_pipeline_tau_stop: float = Field(
        default=0.03,
        ge=0.0,
        alias="ROUTE_PIPELINE_TAU_STOP",
    )
    route_pipeline_world_increment: int = Field(
        default=32,
        ge=1,
        le=500,
        alias="ROUTE_PIPELINE_WORLD_INCREMENT",
    )
    route_dccs_overlap_threshold: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        alias="ROUTE_DCCS_OVERLAP_THRESHOLD",
    )
    route_dccs_bootstrap_count: int = Field(
        default=3,
        ge=1,
        le=32,
        alias="ROUTE_DCCS_BOOTSTRAP_COUNT",
    )
    route_dccs_default_baseline_policy: str = Field(
        default="first_n",
        alias="ROUTE_DCCS_DEFAULT_BASELINE_POLICY",
    )
    route_dccs_pflip_bias: float = Field(
        default=-0.15,
        alias="ROUTE_DCCS_PFLIP_BIAS",
    )
    route_dccs_pflip_gap_weight: float = Field(
        default=2.1,
        ge=0.0,
        alias="ROUTE_DCCS_PFLIP_GAP_WEIGHT",
    )
    route_dccs_pflip_mechanism_weight: float = Field(
        default=1.2,
        ge=0.0,
        alias="ROUTE_DCCS_PFLIP_MECHANISM_WEIGHT",
    )
    route_dccs_pflip_overlap_weight: float = Field(
        default=1.4,
        ge=0.0,
        alias="ROUTE_DCCS_PFLIP_OVERLAP_WEIGHT",
    )
    route_dccs_pflip_detour_weight: float = Field(
        default=0.9,
        ge=0.0,
        alias="ROUTE_DCCS_PFLIP_DETOUR_WEIGHT",
    )
    route_refc_evidence_families: str = Field(
        default="scenario,toll,terrain,fuel,carbon,weather,stochastic",
        alias="ROUTE_REFC_EVIDENCE_FAMILIES",
    )
    route_refc_state_catalog: str = Field(
        default="nominal,mildly_stale,severely_stale,low_confidence,proxy,refreshed",
        alias="ROUTE_REFC_STATE_CATALOG",
    )
    route_ors_baseline_duration_multiplier: float = Field(
        default=1.24,
        ge=1.0,
        le=2.0,
        alias="ROUTE_ORS_BASELINE_DURATION_MULTIPLIER",
    )
    route_ors_baseline_distance_multiplier: float = Field(
        default=1.18,
        ge=1.0,
        le=2.0,
        alias="ROUTE_ORS_BASELINE_DISTANCE_MULTIPLIER",
    )
    route_ors_baseline_allow_proxy_fallback: bool = Field(
        default=True,
        alias="ROUTE_ORS_BASELINE_ALLOW_PROXY_FALLBACK",
    )
    ors_directions_api_key: str = Field(default="", alias="ORS_DIRECTIONS_API_KEY")
    ors_directions_url_template: str = Field(
        default="https://api.openrouteservice.org/v2/directions/{profile}/geojson",
        alias="ORS_DIRECTIONS_URL_TEMPLATE",
    )
    ors_directions_timeout_ms: int = Field(
        default=25_000,
        ge=1_000,
        le=300_000,
        alias="ORS_DIRECTIONS_TIMEOUT_MS",
    )
    ors_directions_profile_default: str = Field(
        default="driving-car",
        alias="ORS_DIRECTIONS_PROFILE_DEFAULT",
    )
    ors_directions_profile_hgv: str = Field(
        default="driving-hgv",
        alias="ORS_DIRECTIONS_PROFILE_HGV",
    )
    route_selection_math_profile: str = Field(
        default="modified_vikor_distance",
        alias="ROUTE_SELECTION_MATH_PROFILE",
    )
    route_selection_modified_regret_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=2.0,
        alias="ROUTE_SELECTION_MODIFIED_REGRET_WEIGHT",
    )
    route_selection_modified_balance_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=2.0,
        alias="ROUTE_SELECTION_MODIFIED_BALANCE_WEIGHT",
    )
    route_selection_modified_distance_weight: float = Field(
        default=0.22,
        ge=0.0,
        le=2.0,
        alias="ROUTE_SELECTION_MODIFIED_DISTANCE_WEIGHT",
    )
    route_selection_modified_eta_distance_weight: float = Field(
        default=0.18,
        ge=0.0,
        le=2.0,
        alias="ROUTE_SELECTION_MODIFIED_ETA_DISTANCE_WEIGHT",
    )
    route_selection_modified_entropy_weight: float = Field(
        default=0.08,
        ge=0.0,
        le=2.0,
        alias="ROUTE_SELECTION_MODIFIED_ENTROPY_WEIGHT",
    )
    route_selection_modified_knee_weight: float = Field(
        default=0.12,
        ge=0.0,
        le=2.0,
        alias="ROUTE_SELECTION_MODIFIED_KNEE_WEIGHT",
    )
    route_selection_tchebycheff_rho: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        alias="ROUTE_SELECTION_TCHEBYCHEFF_RHO",
    )
    route_selection_vikor_v: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="ROUTE_SELECTION_VIKOR_V",
    )
    route_compute_attempt_timeout_s: int = Field(
        default=1200,
        ge=30,
        le=3600,
        alias="ROUTE_COMPUTE_ATTEMPT_TIMEOUT_S",
    )
    route_compute_single_attempt_timeout_s: int = Field(
        default=900,
        ge=30,
        le=3600,
        alias="ROUTE_COMPUTE_SINGLE_ATTEMPT_TIMEOUT_S",
    )
    route_context_probe_timeout_ms: int = Field(
        default=2_500,
        ge=100,
        le=120_000,
        alias="ROUTE_CONTEXT_PROBE_TIMEOUT_MS",
    )
    route_context_probe_enabled: bool = Field(
        default=False,
        alias="ROUTE_CONTEXT_PROBE_ENABLED",
    )
    route_context_probe_max_paths: int = Field(
        default=2,
        ge=1,
        le=24,
        alias="ROUTE_CONTEXT_PROBE_MAX_PATHS",
    )
    route_context_probe_max_state_budget: int = Field(
        default=15_000,
        ge=100,
        le=2_000_000,
        alias="ROUTE_CONTEXT_PROBE_MAX_STATE_BUDGET",
    )
    route_context_probe_max_hops: int = Field(
        default=320,
        ge=8,
        le=20_000,
        alias="ROUTE_CONTEXT_PROBE_MAX_HOPS",
    )
    route_graph_warmup_on_startup: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_WARMUP_ON_STARTUP",
    )
    route_graph_warmup_failfast: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_WARMUP_FAILFAST",
    )
    route_graph_warmup_timeout_s: int = Field(
        default=1200,
        ge=60,
        le=86_400,
        alias="ROUTE_GRAPH_WARMUP_TIMEOUT_S",
    )
    route_graph_fast_startup_enabled: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_FAST_STARTUP_ENABLED",
    )
    route_graph_fast_startup_long_corridor_bypass_km: float = Field(
        default=120.0,
        ge=0.0,
        le=5000.0,
        alias="ROUTE_GRAPH_FAST_STARTUP_LONG_CORRIDOR_BYPASS_KM",
    )
    route_graph_status_check_timeout_ms: int = Field(
        default=1000,
        ge=100,
        le=30_000,
        alias="ROUTE_GRAPH_STATUS_CHECK_TIMEOUT_MS",
    )
    route_graph_od_feasibility_timeout_ms: int = Field(
        default=30_000,
        ge=100,
        le=120_000,
        alias="ROUTE_GRAPH_OD_FEASIBILITY_TIMEOUT_MS",
    )
    route_graph_precheck_timeout_fail_closed: bool = Field(
        default=False,
        alias="ROUTE_GRAPH_PRECHECK_TIMEOUT_FAIL_CLOSED",
    )
    route_graph_enabled: bool = Field(default=True, alias="ROUTE_GRAPH_ENABLED")
    route_graph_asset_path: str = Field(default="", alias="ROUTE_GRAPH_ASSET_PATH")
    route_graph_binary_cache_enabled: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_BINARY_CACHE_ENABLED",
    )
    route_graph_binary_cache_path: str = Field(
        default="",
        alias="ROUTE_GRAPH_BINARY_CACHE_PATH",
    )
    route_graph_k_paths: int = Field(default=24, ge=1, le=128, alias="ROUTE_GRAPH_K_PATHS")
    route_graph_max_hops: int = Field(default=220, ge=8, le=2000, alias="ROUTE_GRAPH_MAX_HOPS")
    route_graph_adaptive_hops_enabled: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_ADAPTIVE_HOPS_ENABLED",
    )
    route_graph_hops_per_km: float = Field(
        default=18.0,
        ge=0.1,
        le=500.0,
        alias="ROUTE_GRAPH_HOPS_PER_KM",
    )
    route_graph_hops_detour_factor: float = Field(
        default=1.35,
        ge=1.0,
        le=8.0,
        alias="ROUTE_GRAPH_HOPS_DETOUR_FACTOR",
    )
    route_graph_edge_length_estimate_m: float = Field(
        default=75.0,
        ge=1.0,
        le=2000.0,
        alias="ROUTE_GRAPH_EDGE_LENGTH_ESTIMATE_M",
    )
    route_graph_hops_safety_factor: float = Field(
        default=1.8,
        ge=0.1,
        le=20.0,
        alias="ROUTE_GRAPH_HOPS_SAFETY_FACTOR",
    )
    route_graph_max_hops_cap: int = Field(
        default=15_000,
        ge=8,
        le=100_000,
        alias="ROUTE_GRAPH_MAX_HOPS_CAP",
    )
    route_graph_min_nodes: int = Field(default=100_000, ge=1, alias="ROUTE_GRAPH_MIN_NODES")
    route_graph_min_adjacency: int = Field(default=100_000, ge=1, alias="ROUTE_GRAPH_MIN_ADJACENCY")
    route_graph_max_state_budget: int = Field(
        default=1_200_000,
        ge=1000,
        le=8_000_000,
        alias="ROUTE_GRAPH_MAX_STATE_BUDGET",
    )
    route_graph_state_budget_per_hop: int = Field(
        default=1600,
        ge=10,
        le=200_000,
        alias="ROUTE_GRAPH_STATE_BUDGET_PER_HOP",
    )
    route_graph_state_budget_retry_multiplier: float = Field(
        default=2.5,
        ge=1.0,
        le=8.0,
        alias="ROUTE_GRAPH_STATE_BUDGET_RETRY_MULTIPLIER",
    )
    route_graph_state_budget_retry_cap: int = Field(
        default=8_000_000,
        ge=1_000,
        le=20_000_000,
        alias="ROUTE_GRAPH_STATE_BUDGET_RETRY_CAP",
    )
    route_graph_search_initial_timeout_ms: int = Field(
        default=30_000,
        ge=0,
        le=900_000,
        alias="ROUTE_GRAPH_SEARCH_INITIAL_TIMEOUT_MS",
    )
    route_graph_search_retry_timeout_ms: int = Field(
        default=120_000,
        ge=0,
        le=900_000,
        alias="ROUTE_GRAPH_SEARCH_RETRY_TIMEOUT_MS",
    )
    route_graph_search_rescue_timeout_ms: int = Field(
        default=150_000,
        ge=0,
        le=900_000,
        alias="ROUTE_GRAPH_SEARCH_RESCUE_TIMEOUT_MS",
    )
    route_graph_reduced_initial_for_long_corridor: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_REDUCED_INITIAL_FOR_LONG_CORRIDOR",
    )
    route_graph_long_corridor_threshold_km: float = Field(
        default=150.0,
        ge=10.0,
        le=2_000.0,
        alias="ROUTE_GRAPH_LONG_CORRIDOR_THRESHOLD_KM",
    )
    route_graph_long_corridor_max_paths: int = Field(
        default=4,
        ge=1,
        le=64,
        alias="ROUTE_GRAPH_LONG_CORRIDOR_MAX_PATHS",
    )
    route_graph_skip_initial_search_long_corridor: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_SKIP_INITIAL_SEARCH_LONG_CORRIDOR",
    )
    route_graph_a_star_heuristic_enabled: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_A_STAR_HEURISTIC_ENABLED",
    )
    route_graph_heuristic_max_speed_kph: float = Field(
        default=220.0,
        ge=30.0,
        le=500.0,
        alias="ROUTE_GRAPH_HEURISTIC_MAX_SPEED_KPH",
    )
    route_graph_search_apply_scenario_edge_costs: bool = Field(
        default=False,
        alias="ROUTE_GRAPH_SEARCH_APPLY_SCENARIO_EDGE_COSTS",
    )
    route_graph_state_space_rescue_enabled: bool = Field(
        default=True,
        alias="ROUTE_GRAPH_STATE_SPACE_RESCUE_ENABLED",
    )
    route_graph_state_space_rescue_mode: str = Field(
        default="reduced",
        alias="ROUTE_GRAPH_STATE_SPACE_RESCUE_MODE",
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
        default=False,
        alias="ROUTE_GRAPH_SCENARIO_SEPARABILITY_FAIL",
    )
    route_graph_min_giant_component_nodes: int = Field(
        default=50_000,
        ge=1_000,
        le=10_000_000,
        alias="ROUTE_GRAPH_MIN_GIANT_COMPONENT_NODES",
    )
    route_graph_min_giant_component_ratio: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        alias="ROUTE_GRAPH_MIN_GIANT_COMPONENT_RATIO",
    )
    route_graph_max_nearest_node_distance_m: float = Field(
        default=10_000.0,
        ge=100.0,
        le=500_000.0,
        alias="ROUTE_GRAPH_MAX_NEAREST_NODE_DISTANCE_M",
    )
    route_graph_od_candidate_limit: int = Field(
        default=2048,
        ge=8,
        le=50_000,
        alias="ROUTE_GRAPH_OD_CANDIDATE_LIMIT",
    )
    route_graph_od_candidate_max_radius: int = Field(
        default=12,
        ge=1,
        le=64,
        alias="ROUTE_GRAPH_OD_CANDIDATE_MAX_RADIUS",
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
        default=0.96,
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
    live_terrain_prefetch_probe_fractions: str = Field(
        default="0.5,0.35,0.65,0.2,0.8",
        alias="LIVE_TERRAIN_PREFETCH_PROBE_FRACTIONS",
    )
    live_terrain_prefetch_min_covered_points: int = Field(
        default=1,
        ge=1,
        le=32,
        alias="LIVE_TERRAIN_PREFETCH_MIN_COVERED_POINTS",
    )
    terrain_allow_synthetic_grid: bool = Field(
        default=False,
        alias="TERRAIN_ALLOW_SYNTHETIC_GRID",
    )
    terrain_sample_spacing_m: float = Field(default=180.0, ge=5.0, alias="TERRAIN_SAMPLE_SPACING_M")
    terrain_long_route_threshold_km: float = Field(
        default=180.0,
        ge=10.0,
        le=5_000.0,
        alias="TERRAIN_LONG_ROUTE_THRESHOLD_KM",
    )
    terrain_long_route_sample_spacing_m: float = Field(
        default=320.0,
        ge=20.0,
        le=5_000.0,
        alias="TERRAIN_LONG_ROUTE_SAMPLE_SPACING_M",
    )
    terrain_long_route_max_samples_per_route: int = Field(
        default=900,
        ge=100,
        le=50_000,
        alias="TERRAIN_LONG_ROUTE_MAX_SAMPLES_PER_ROUTE",
    )
    terrain_max_samples_per_route: int = Field(
        default=1500,
        ge=200,
        le=50_000,
        alias="TERRAIN_MAX_SAMPLES_PER_ROUTE",
    )
    terrain_segment_boundary_probe_max_segments: int = Field(
        default=1200,
        ge=0,
        le=50_000,
        alias="TERRAIN_SEGMENT_BOUNDARY_PROBE_MAX_SEGMENTS",
    )
    terrain_physics_version: str = Field(default="uk_v3", alias="TERRAIN_PHYSICS_VERSION")
    carbon_scope_mode: str = Field(default="ttw", alias="CARBON_SCOPE_MODE")
    carbon_policy_scenario: str = Field(default="central", alias="CARBON_POLICY_SCENARIO")

    @model_validator(mode="after")
    def _enforce_strict_runtime_defaults(self) -> Settings:
        # Hard-strict runtime policy: production paths are always strict and
        # synthetic generation is disallowed outside explicit test tooling.
        self.live_runtime_data_enabled = True
        self.strict_live_data_required = True
        self.live_route_compute_require_all_expected = True
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
        self.live_scenario_allow_partial_sources_strict = False
        self.live_scenario_min_source_count_strict = 4
        self.live_scenario_min_coverage_overall_strict = 1.0
        self.scenario_require_signature = True
        legacy_graph_env_keys = (
            "ROUTE_GRAPH_STREAMING_MAX_NODES",
            "ROUTE_GRAPH_STREAMING_MAX_EDGES",
            "ROUTE_GRAPH_COMPACT_LOAD_ENABLED",
            "ROUTE_GRAPH_COMPACT_BACKBONE_SHARE",
        )
        legacy_present = [key for key in legacy_graph_env_keys if str(os.environ.get(key, "")).strip()]
        if legacy_present:
            _LOGGER.warning(
                "Ignoring legacy compact-graph env overrides under strict full-graph runtime: %s",
                ",".join(sorted(legacy_present)),
            )
        self.out_dir = _resolve_writable_out_dir(self.out_dir)
        rf = str(self.risk_family or "cvar_excess").strip().lower()
        if rf not in {"cvar_excess", "entropic", "downside_semivariance"}:
            rf = "cvar_excess"
        self.risk_family = rf
        refresh_mode = str(self.live_route_compute_refresh_mode or "route_compute").strip().lower()
        if refresh_mode not in {"route_compute", "all_sources", "full"}:
            refresh_mode = "route_compute"
        self.live_route_compute_refresh_mode = refresh_mode
        rescue_mode = str(self.route_graph_state_space_rescue_mode or "reduced").strip().lower()
        if rescue_mode not in {"reduced", "full"}:
            rescue_mode = "reduced"
        self.route_graph_state_space_rescue_mode = rescue_mode
        math_profile = str(self.route_selection_math_profile or "modified_vikor_distance").strip().lower()
        if math_profile not in {
            "academic_reference",
            "academic_tchebycheff",
            "academic_vikor",
            "modified_hybrid",
            "modified_distance_aware",
            "modified_vikor_distance",
        }:
            math_profile = "modified_vikor_distance"
        self.route_selection_math_profile = math_profile
        pipeline_mode = str(self.route_pipeline_default_mode or "legacy").strip().lower()
        if pipeline_mode not in {"legacy", "dccs", "dccs_refc", "voi"}:
            pipeline_mode = "legacy"
        self.route_pipeline_default_mode = pipeline_mode
        baseline_policy = str(self.route_dccs_default_baseline_policy or "first_n").strip().lower()
        if baseline_policy not in {"first_n", "random_n", "corridor_uniform"}:
            baseline_policy = "first_n"
        self.route_dccs_default_baseline_policy = baseline_policy
        self.terrain_long_route_sample_spacing_m = max(
            float(self.terrain_sample_spacing_m),
            float(self.terrain_long_route_sample_spacing_m),
        )
        self.terrain_long_route_max_samples_per_route = min(
            int(self.terrain_long_route_max_samples_per_route),
            int(self.terrain_max_samples_per_route),
        )
        # In strict hard-refresh lanes, terrain probe is required for full live-freshness evidence.
        if self.strict_live_data_required and (
            self.live_route_compute_require_all_expected
            or refresh_mode in {"all_sources", "full"}
        ):
            self.live_route_compute_probe_terrain = True
        return self


settings = Settings()
