from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FROZEN_REASON_CODES: frozenset[str] = frozenset(
    {
        "routing_graph_unavailable",
        "routing_graph_fragmented",
        "routing_graph_disconnected_od",
        "routing_graph_coverage_gap",
        "routing_graph_no_path",
        "routing_graph_precheck_timeout",
        "routing_graph_warming_up",
        "routing_graph_warmup_failed",
        "live_source_refresh_failed",
        "route_compute_timeout",
        "departure_profile_unavailable",
        "holiday_data_unavailable",
        "stochastic_calibration_unavailable",
        "scenario_profile_unavailable",
        "scenario_profile_invalid",
        "risk_normalization_unavailable",
        "risk_prior_unavailable",
        "terrain_region_unsupported",
        "terrain_dem_asset_unavailable",
        "terrain_dem_coverage_insufficient",
        "toll_topology_unavailable",
        "toll_tariff_unavailable",
        "toll_tariff_unresolved",
        "fuel_price_auth_unavailable",
        "fuel_price_source_unavailable",
        "vehicle_profile_unavailable",
        "vehicle_profile_invalid",
        "carbon_policy_unavailable",
        "carbon_intensity_unavailable",
        "epsilon_infeasible",
        "no_route_candidates",
        "model_asset_unavailable",
    }
)


@dataclass
class ModelDataError(ValueError):
    reason_code: str
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


def normalize_reason_code(reason_code: str, *, default: str = "model_asset_unavailable") -> str:
    code = str(reason_code or "").strip()
    if code in FROZEN_REASON_CODES:
        return code
    return default
