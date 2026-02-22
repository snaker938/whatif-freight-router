from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .calibration_loader import (
    TollConfidenceCalibration,
    TollSegmentSeed,
    TollTariffRule,
    load_toll_confidence_calibration,
    load_toll_segments_seed,
    load_toll_tariffs,
)
from .settings import settings

try:
    from pyproj import Transformer
    from shapely.geometry import LineString
except Exception:  # pragma: no cover - optional runtime fallback
    Transformer = None  # type: ignore[assignment]
    LineString = None  # type: ignore[assignment]

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = UTC
EARTH_RADIUS_M = 6_371_000.0
_TO_WEB_M = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True) if Transformer else None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def _to_xy_m(lat: float, lon: float, ref_lat: float) -> tuple[float, float]:
    x = math.radians(lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * EARTH_RADIUS_M
    return x, y


def _point_to_segment_distance_m(
    *,
    lat: float,
    lon: float,
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> float:
    ref_lat = (seg_a[0] + seg_b[0]) / 2.0
    px, py = _to_xy_m(lat, lon, ref_lat)
    ax, ay = _to_xy_m(seg_a[0], seg_a[1], ref_lat)
    bx, by = _to_xy_m(seg_b[0], seg_b[1], ref_lat)
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = (abx * abx) + (aby * aby)
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / denom))
    cx = ax + (t * abx)
    cy = ay + (t * aby)
    return math.hypot(px - cx, py - cy)


def _segment_heading_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dy = lat2 - lat1
    dx = lon2 - lon1
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx))
    return (ang + 360.0) % 360.0


def _heading_delta_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _step_has_toll_class(step: dict) -> bool:
    classes = step.get("classes", [])
    if isinstance(classes, list) and any(str(item).strip().lower() == "toll" for item in classes):
        return True
    intersections = step.get("intersections", [])
    if isinstance(intersections, list):
        for inter in intersections:
            if not isinstance(inter, dict):
                continue
            inter_classes = inter.get("classes", [])
            if isinstance(inter_classes, list) and any(
                str(item).strip().lower() == "toll" for item in inter_classes
            ):
                return True
    return False


def _vehicle_class(vehicle_type: str) -> str:
    key = (vehicle_type or "").strip().lower()
    return {
        "artic_hgv": "artic_hgv",
        "rigid_hgv": "rigid_hgv",
        "van": "van",
        "ev_hgv": "rigid_hgv",
        "ev": "rigid_hgv",
    }.get(key, "default")


def _vehicle_axle_class(vehicle_type: str) -> str:
    key = (vehicle_type or "").strip().lower()
    return {
        "artic_hgv": "5plus",
        "rigid_hgv": "3to4",
        "van": "2",
        "ev_hgv": "3to4",
        "ev": "3to4",
    }.get(key, "default")


def _minute_of_day_uk(departure_time_utc: datetime | None) -> int:
    if departure_time_utc is None:
        return 0
    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=UTC
    )
    local = aware.astimezone(UK_TZ)
    return (local.hour * 60) + local.minute


def _in_time_band(minute: int, *, start: int, end: int) -> bool:
    if start <= end:
        return start <= minute <= end
    return minute >= start or minute <= end


def _rule_score(
    rule: TollTariffRule,
    *,
    operator: str,
    crossing_id: str,
    road_class: str,
    route_direction: str,
    vehicle_class: str,
    axle_class: str,
    payment_class: str,
    minute: int,
) -> int:
    if not _in_time_band(minute, start=rule.start_minute, end=rule.end_minute):
        return -1
    score = 0
    if operator and operator == rule.operator:
        score += 4
    elif rule.operator not in ("", "default", "*"):
        return -1
    if crossing_id and crossing_id == rule.crossing_id:
        score += 5
    elif rule.crossing_id not in ("", "default", "*"):
        return -1
    if road_class and road_class == rule.road_class:
        score += 3
    elif rule.road_class not in ("", "default", "*"):
        return -1
    if rule.direction not in ("", "default", "*", "both", "bi", "bidirectional", "either"):
        if route_direction and rule.direction == route_direction:
            score += 2
        else:
            return -1
    if vehicle_class in rule.vehicle_classes:
        score += 3
    elif "default" in rule.vehicle_classes or "*" in rule.vehicle_classes:
        score += 1
    else:
        return -1
    if axle_class in rule.axle_classes:
        score += 2
    elif "default" not in rule.axle_classes and "*" not in rule.axle_classes:
        return -1
    if payment_class in rule.payment_classes:
        score += 1
    elif "default" not in rule.payment_classes and "*" not in rule.payment_classes:
        return -1
    return score


def _pick_tariff_rule(
    *,
    operator: str,
    crossing_id: str,
    road_class: str,
    route_direction: str,
    vehicle_class: str,
    axle_class: str,
    payment_class: str,
    minute: int,
) -> TollTariffRule | None:
    table = load_toll_tariffs()
    candidates: list[tuple[int, TollTariffRule]] = []
    for rule in table.rules:
        score = _rule_score(
            rule,
            operator=operator.strip().lower(),
            crossing_id=crossing_id.strip().lower(),
            road_class=road_class.strip().lower(),
            route_direction=route_direction.strip().lower(),
            vehicle_class=vehicle_class,
            axle_class=axle_class.strip().lower(),
            payment_class=payment_class.strip().lower(),
            minute=minute,
        )
        if score >= 0:
            candidates.append((score, rule))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1].rule_id))
    return candidates[0][1]


def _apply_confidence_calibration(raw_confidence: float, *, calibration: TollConfidenceCalibration) -> float:
    p = max(0.0, min(1.0, float(raw_confidence)))
    for row in calibration.bins:
        if row.minimum <= p <= row.maximum:
            # Blend keeps monotonicity while anchoring to calibrated reliability bins.
            return max(0.0, min(1.0, (0.65 * p) + (0.35 * row.calibrated)))
    return p


def _route_points(route: dict) -> list[tuple[float, float]]:
    geom = route.get("geometry", {})
    if not isinstance(geom, dict):
        return []
    coords = geom.get("coordinates", [])
    if not isinstance(coords, list):
        return []
    out: list[tuple[float, float]] = []
    for point in coords:
        if not isinstance(point, list | tuple) or len(point) < 2:
            continue
        lon = float(point[0])
        lat = float(point[1])
        out.append((lat, lon))
    return out


def _infer_road_class_from_route(route: dict) -> str:
    legs = route.get("legs", [])
    if not isinstance(legs, list):
        return "default"
    class_counts: dict[str, int] = {}
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        steps = leg.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            classes = step.get("classes", [])
            if not isinstance(classes, list):
                continue
            for cls in classes:
                key = str(cls).strip().lower()
                if not key:
                    continue
                class_counts[key] = class_counts.get(key, 0) + 1
    if not class_counts:
        return "default"
    if class_counts.get("motorway", 0) >= max(class_counts.values()):
        return "motorway_toll"
    if class_counts.get("trunk", 0) >= max(class_counts.values()):
        return "trunk_toll"
    if class_counts.get("tunnel", 0) > 0:
        return "tunnel_toll"
    if class_counts.get("bridge", 0) > 0:
        return "bridge_toll"
    return "default"


def _infer_route_direction_from_points(route_points: list[tuple[float, float]]) -> str:
    if len(route_points) < 2:
        return "both"
    lat1, lon1 = route_points[0]
    lat2, lon2 = route_points[-1]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    if abs(dlat) >= abs(dlon):
        return "northbound" if dlat >= 0 else "southbound"
    return "eastbound" if dlon >= 0 else "westbound"


def _segment_near_route(seed: TollSegmentSeed, route_points: list[tuple[float, float]]) -> bool:
    return _segment_overlap_km(seed, route_points) > 0.0


def _direction_matches(direction: str, *, lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
    d = (direction or "").strip().lower()
    if d in ("", "both", "bi", "bidirectional", "either"):
        return True
    heading = _segment_heading_deg(lat1, lon1, lat2, lon2)
    if "north" in d or d in ("n", "nb", "northbound"):
        return _heading_delta_deg(heading, 0.0) <= 55.0
    if "south" in d or d in ("s", "sb", "southbound"):
        return _heading_delta_deg(heading, 180.0) <= 55.0
    if "east" in d or d in ("e", "eb", "eastbound"):
        return _heading_delta_deg(heading, 90.0) <= 55.0
    if "west" in d or d in ("w", "wb", "westbound"):
        return _heading_delta_deg(heading, 270.0) <= 55.0
    if d in ("forward", "with_traffic", "oneway"):
        return True
    if d in ("reverse", "backward", "-1"):
        # reverse means against the nominal way direction; allow only when heading
        # is materially opposite to north-east quadrant defaults.
        return _heading_delta_deg(heading, 225.0) <= 90.0
    return True


def _segment_overlap_km(seed: TollSegmentSeed, route_points: list[tuple[float, float]]) -> float:
    if len(route_points) < 2 or len(seed.coordinates) < 2:
        return 0.0
    if not _direction_matches(
        seed.direction,
        lat1=route_points[0][0],
        lon1=route_points[0][1],
        lat2=route_points[-1][0],
        lon2=route_points[-1][1],
    ):
        return 0.0

    if _TO_WEB_M is not None and LineString is not None:
        try:
            route_line_wgs = LineString([(lon, lat) for lat, lon in route_points])
            seed_line_wgs = LineString([(lon, lat) for lat, lon in seed.coordinates])
            if route_line_wgs.length <= 0 or seed_line_wgs.length <= 0:
                return 0.0
            route_line = LineString([_TO_WEB_M.transform(x, y) for x, y in route_line_wgs.coords])
            seed_line = LineString([_TO_WEB_M.transform(x, y) for x, y in seed_line_wgs.coords])
            # Buffer route corridor to capture near-parallel map-matched overlap.
            corridor_m = max(40.0, min(140.0, 60.0 + (0.15 * float(settings.route_candidate_via_budget))))
            overlap_geom = seed_line.intersection(
                route_line.buffer(corridor_m, cap_style="flat", join_style="mitre")
            )
            overlap_m = float(overlap_geom.length) if not overlap_geom.is_empty else 0.0
            if overlap_m <= 0.0:
                return 0.0
            seed_len_m = max(1.0, float(seed_line.length))
            route_heading = _segment_heading_deg(
                route_points[0][0],
                route_points[0][1],
                route_points[-1][0],
                route_points[-1][1],
            )
            seed_heading = _segment_heading_deg(
                seed.coordinates[0][0],
                seed.coordinates[0][1],
                seed.coordinates[-1][0],
                seed.coordinates[-1][1],
            )
            heading_delta = _heading_delta_deg(route_heading, seed_heading)
            heading_weight = max(0.30, 1.0 - (heading_delta / 180.0))
            overlap_ratio = min(1.0, overlap_m / seed_len_m)
            coverage_weight = max(0.20, overlap_ratio)
            return max(0.0, (overlap_m / 1000.0) * heading_weight * coverage_weight)
        except Exception:
            # Continue with heuristic fallback when geometry libs fail.
            pass

    threshold_m = 140.0
    overlap_m = 0.0
    for idx in range(1, len(route_points)):
        lat1, lon1 = route_points[idx - 1]
        lat2, lon2 = route_points[idx]
        if not _direction_matches(seed.direction, lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2):
            continue
        seg_len_m = _haversine_m(lat1, lon1, lat2, lon2)
        if seg_len_m <= 0.0:
            continue
        route_heading = _segment_heading_deg(lat1, lon1, lat2, lon2)
        min_dist = float("inf")
        best_heading_delta = 180.0
        for seed_idx in range(1, len(seed.coordinates)):
            seg_a = seed.coordinates[seed_idx - 1]
            seg_b = seed.coordinates[seed_idx]
            seed_heading = _segment_heading_deg(seg_a[0], seg_a[1], seg_b[0], seg_b[1])
            heading_delta = _heading_delta_deg(route_heading, seed_heading)
            # Symmetric endpoint/center checks improve overlap reliability over midpoint-only.
            mid_lat = (lat1 + lat2) / 2.0
            mid_lon = (lon1 + lon2) / 2.0
            dist_values = (
                _point_to_segment_distance_m(lat=lat1, lon=lon1, seg_a=seg_a, seg_b=seg_b),
                _point_to_segment_distance_m(lat=lat2, lon=lon2, seg_a=seg_a, seg_b=seg_b),
                _point_to_segment_distance_m(lat=mid_lat, lon=mid_lon, seg_a=seg_a, seg_b=seg_b),
                _point_to_segment_distance_m(lat=seg_a[0], lon=seg_a[1], seg_a=(lat1, lon1), seg_b=(lat2, lon2)),
                _point_to_segment_distance_m(lat=seg_b[0], lon=seg_b[1], seg_a=(lat1, lon1), seg_b=(lat2, lon2)),
            )
            dist = min(dist_values)
            if dist < min_dist:
                min_dist = dist
                best_heading_delta = heading_delta
        if min_dist <= threshold_m:
            proximity_weight = max(0.15, 1.0 - (min_dist / threshold_m))
            heading_weight = max(0.35, 1.0 - (best_heading_delta / 180.0))
            overlap_m += seg_len_m * proximity_weight * heading_weight
    return max(0.0, overlap_m / 1000.0)


def _seed_length_km(seed: TollSegmentSeed) -> float:
    total_m = 0.0
    for idx in range(1, len(seed.coordinates)):
        lat1, lon1 = seed.coordinates[idx - 1]
        lat2, lon2 = seed.coordinates[idx]
        total_m += _haversine_m(lat1, lon1, lat2, lon2)
    return max(0.0, total_m / 1000.0)


@dataclass(frozen=True)
class TollComputation:
    contains_toll: bool
    toll_distance_km: float
    toll_cost_gbp: float
    confidence: float
    source: str
    details: dict[str, float | str | int]


def compute_toll_cost(
    *,
    route: dict,
    distance_km: float,
    vehicle_type: str,
    vehicle_profile: Any | None = None,
    departure_time_utc: datetime | None,
    use_tolls: bool,
    fallback_toll_cost_per_km: float = 0.0,
) -> TollComputation:
    distance_km = max(0.0, float(distance_km))
    candidate_meta = route.get("_candidate_meta", {})
    if isinstance(candidate_meta, dict) and bool(candidate_meta.get("seen_by_exclude_toll", False)):
        return TollComputation(
            contains_toll=False,
            toll_distance_km=0.0,
            toll_cost_gbp=0.0,
            confidence=0.99,
            source="exclude_toll_candidate",
            details={"segments_matched": 0, "classified_steps": 0},
        )

    # Primary classification from step/intersection classes.
    classified_toll_distance_km = 0.0
    classified_step_count = 0
    total_step_distance_km = 0.0
    legs = route.get("legs", [])
    if isinstance(legs, list):
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            steps = leg.get("steps", [])
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                raw_distance = step.get("distance", 0.0)
                step_km = max(0.0, float(raw_distance) / 1000.0) if isinstance(raw_distance, (int, float)) else 0.0
                total_step_distance_km += step_km
                if _step_has_toll_class(step):
                    classified_step_count += 1
                    classified_toll_distance_km += step_km

    if classified_toll_distance_km > 0.0 and total_step_distance_km > 0.0 and distance_km > 0.0:
        classified_toll_distance_km *= distance_km / total_step_distance_km
        classified_toll_distance_km = min(distance_km, classified_toll_distance_km)

    # Secondary classification from open-data seed segments.
    route_points = _route_points(route)
    all_seed_segments = load_toll_segments_seed()
    overlap_rows: list[tuple[TollSegmentSeed, float]] = []
    # Long-haul routes often touch short toll assets (bridges/tunnels) for only a
    # small distance, so thresholding against total route length is brittle.
    min_overlap_km = 0.15
    min_overlap_ratio = 0.25
    for seed in all_seed_segments:
        overlap_km = _segment_overlap_km(seed, route_points)
        if overlap_km < min_overlap_km:
            continue
        seed_len_km = _seed_length_km(seed)
        overlap_ratio = overlap_km / max(1e-6, seed_len_km)
        if overlap_ratio < min_overlap_ratio and overlap_km < 0.22:
            continue
        overlap_rows.append((seed, overlap_km))
    matched_seeds = [seed for seed, _ in overlap_rows]
    seeded_toll_distance_km = min(distance_km, sum(overlap_km for _, overlap_km in overlap_rows))

    contains_toll = (
        bool(route.get("contains_toll", False))
        or (classified_toll_distance_km > 0.0)
        or bool(matched_seeds)
    )
    toll_distance_km = max(
        classified_toll_distance_km,
        seeded_toll_distance_km if classified_toll_distance_km <= 0 else 0.0,
    )
    if contains_toll and toll_distance_km <= 0.0:
        toll_distance_km = distance_km

    if not use_tolls or not contains_toll:
        return TollComputation(
            contains_toll=contains_toll,
            toll_distance_km=round(toll_distance_km, 6),
            toll_cost_gbp=0.0,
            confidence=0.95 if not contains_toll else 0.80,
            source="classification_only",
            details={
                "segments_matched": len(matched_seeds),
                "classified_steps": classified_step_count,
                "fallback_policy_used": False,
                "pricing_unresolved": False,
                "tariff_rule_ids": "",
            },
        )

    minute = _minute_of_day_uk(departure_time_utc)
    vclass = str(getattr(vehicle_profile, "toll_vehicle_class", "")).strip().lower()
    axle_class = str(getattr(vehicle_profile, "toll_axle_class", "")).strip().lower()
    if not vclass:
        vclass = _vehicle_class(vehicle_type)
    if not axle_class:
        axle_class = _vehicle_axle_class(vehicle_type)
    payment_class = "electronic"
    route_direction = _infer_route_direction_from_points(route_points)
    tariffs = load_toll_tariffs()
    confidence_calibration = load_toll_confidence_calibration()

    # Monetary engine: crossing + distance components from matched segments/tariffs.
    total_crossing_fee = 0.0
    total_distance_fee = 0.0
    matched_asset_ids: list[str] = []
    tariff_rule_ids: list[str] = []
    unresolved_assets: list[str] = []
    for seed, overlap_km in overlap_rows:
        rule = _pick_tariff_rule(
            operator=seed.operator,
            crossing_id=seed.crossing_id,
            road_class=seed.road_class,
            route_direction=route_direction,
            vehicle_class=vclass,
            axle_class=axle_class,
            payment_class=payment_class,
            minute=minute,
        )
        if rule is None:
            unresolved_assets.append(seed.segment_id)
            continue
        crossing_fee = rule.crossing_fee_gbp
        distance_fee_per_km = rule.distance_fee_gbp_per_km
        tariff_rule_ids.append(rule.rule_id)
        matched_asset_ids.append(seed.segment_id)
        total_crossing_fee += max(0.0, crossing_fee)
        total_distance_fee += max(0.0, distance_fee_per_km) * max(0.0, overlap_km)

    if not matched_seeds and contains_toll and classified_toll_distance_km > 0.0:
        return TollComputation(
            contains_toll=True,
            toll_distance_km=round(toll_distance_km, 6),
            toll_cost_gbp=0.0,
            confidence=0.0,
            source="unpriced_toll",
            details={
                "segments_matched": 0,
                "classified_steps": classified_step_count,
                "minute_uk": minute,
                "fallback_policy_used": False,
                "pricing_unresolved": True,
                "tariff_rule_ids": "",
                "matched_asset_ids": "",
                "unresolved_asset_ids": "classified_toll_unmapped",
                "unresolved_reason": "strict_mode_no_tariff_match",
                "explicit_user_rate_gbp_per_km": round(max(0.0, float(fallback_toll_cost_per_km)), 6),
                "tariff_catalog_size": len(tariffs.rules),
                },
            )

    if (not matched_seeds) or unresolved_assets:
        return TollComputation(
            contains_toll=True,
            toll_distance_km=round(toll_distance_km, 6),
            toll_cost_gbp=0.0,
            confidence=0.0,
            source="unpriced_toll",
            details={
                "segments_matched": len(matched_seeds),
                "classified_steps": classified_step_count,
                "minute_uk": minute,
                "fallback_policy_used": False,
                "pricing_unresolved": True,
                "tariff_rule_ids": "",
                "matched_asset_ids": "|".join([*matched_asset_ids, *unresolved_assets]),
                "unresolved_asset_ids": "|".join(unresolved_assets),
                "tariff_catalog_size": len(tariffs.rules),
            },
        )

    toll_cost = max(0.0, total_crossing_fee + total_distance_fee)
    class_signal = min(1.0, classified_toll_distance_km / max(distance_km, 1e-6)) if distance_km > 0 else 0.0
    seed_signal = min(1.0, seeded_toll_distance_km / max(distance_km, 1e-6)) if distance_km > 0 else 0.0
    segment_signal = min(1.0, len(matched_seeds) / 3.0)
    logit = (
        float(confidence_calibration.intercept)
        + (float(confidence_calibration.class_signal_coef) * class_signal)
        + (float(confidence_calibration.seed_signal_coef) * seed_signal)
        + (float(confidence_calibration.segment_signal_coef) * segment_signal)
    )
    confidence = 1.0 / (1.0 + math.exp(-logit))
    if classified_toll_distance_km > 0 and matched_seeds:
        source = "class_and_seed"
        confidence += float(confidence_calibration.bonus_both)
    else:
        source = "seed_only"
    confidence = min(0.99, max(0.15, _apply_confidence_calibration(confidence, calibration=confidence_calibration)))

    return TollComputation(
        contains_toll=contains_toll,
        toll_distance_km=round(toll_distance_km, 6),
        toll_cost_gbp=round(toll_cost, 6),
        confidence=round(confidence, 6),
        source=source,
        details={
            "segments_matched": len(matched_seeds),
            "classified_steps": classified_step_count,
            "minute_uk": minute,
            "fallback_policy_used": False,
            "pricing_unresolved": False,
            "tariff_rule_ids": "|".join(tariff_rule_ids),
            "matched_asset_ids": "|".join(matched_asset_ids),
            "calibration_version": confidence_calibration.version,
        },
    )
