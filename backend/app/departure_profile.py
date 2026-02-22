from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .calibration_loader import load_departure_profile, load_uk_bank_holidays

try:
    UK_TZ = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:
    UK_TZ = timezone.utc

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


@dataclass(frozen=True)
class DepartureMultiplier:
    multiplier: float
    profile_source: str
    local_time_iso: str | None
    profile_day: str
    profile_key: str
    profile_version: str = "uk-v3-contextual"
    profile_as_of_utc: str | None = None
    profile_refreshed_at_utc: str | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None


def _region_bucket_from_route(route_points: list[tuple[float, float]] | None) -> str:
    if not route_points:
        return "uk_default"
    lats = [lat for lat, _lon in route_points]
    lons = [lon for _lat, lon in route_points]
    lat = sum(lats) / len(lats)
    lon = sum(lons) / len(lons)
    geohash5 = _geohash5(lat, lon)
    if geohash5:
        return geohash5
    if lat >= 57.2:
        return "scotland_far_north"
    if lat >= 56.0:
        return "scotland_north"
    if lat >= 55.0:
        return "scotland_south"
    if lon <= -4.4 and lat < 55.0:
        return "wales_west"
    if lon <= -3.0 and lat < 53.8:
        return "wales_border"
    if lon > -1.0 and lat <= 52.4:
        return "london_southeast"
    if lat >= 54.0 and lon >= -2.1:
        return "north_east_corridor"
    if lat >= 53.2 and lon <= -2.4:
        return "north_west_corridor"
    if lat >= 53.2:
        return "north_england_central"
    if lat >= 52.4 and lon > -2.0:
        return "midlands_east"
    if lat >= 52.0:
        return "midlands_west"
    return "south_england"


def _geohash5(lat: float, lon: float) -> str | None:
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    out: list[str] = []
    bit = 0
    ch = 0
    even = True
    bits = (16, 8, 4, 2, 1)
    while len(out) < 5:
        if even:
            mid = (lon_min + lon_max) / 2.0
            if lon >= mid:
                ch |= bits[bit]
                lon_min = mid
            else:
                lon_max = mid
        else:
            mid = (lat_min + lat_max) / 2.0
            if lat >= mid:
                ch |= bits[bit]
                lat_min = mid
            else:
                lat_max = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0
    return "".join(out)


def _road_bucket(road_class_counts: dict[str, int] | None) -> str:
    if not road_class_counts:
        return "mixed"
    total = max(sum(max(0, int(v)) for v in road_class_counts.values()), 1)
    motorway_share = max(0, int(road_class_counts.get("motorway", 0))) / total
    trunk_share = max(0, int(road_class_counts.get("trunk", 0))) / total
    local_share = max(0, int(road_class_counts.get("local", 0))) / total
    primary_share = max(0, int(road_class_counts.get("primary", 0))) / total
    secondary_share = max(0, int(road_class_counts.get("secondary", 0))) / total
    motorway_trunk_share = motorway_share + trunk_share
    arterial_share = primary_share + secondary_share
    if motorway_share >= 0.65:
        return "motorway_dominant"
    if motorway_trunk_share >= 0.75:
        return "motorway_trunk_heavy"
    if motorway_share >= 0.50:
        return "motorway_heavy"
    if trunk_share >= 0.50:
        return "trunk_dominant"
    if trunk_share >= 0.35:
        return "trunk_heavy"
    if local_share >= 0.55:
        return "urban_local_heavy"
    if arterial_share >= 0.60:
        return "arterial_heavy"
    if primary_share >= 0.40:
        return "primary_heavy"
    return "mixed"


def _route_shape_bucket(
    route_points: list[tuple[float, float]] | None,
    road_class_counts: dict[str, int] | None,
) -> str:
    if not route_points or len(route_points) < 2:
        return "unknown_shape"
    lats = [lat for lat, _ in route_points]
    lons = [lon for _, lon in route_points]
    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    if road_class_counts:
        total = max(1, sum(max(0, int(v)) for v in road_class_counts.values()))
        motorway_share = max(0, int(road_class_counts.get("motorway", 0))) / total
        local_share = max(0, int(road_class_counts.get("local", 0))) / total
    else:
        motorway_share = 0.0
        local_share = 0.0
    if (lat_span + lon_span) >= 3.0 or motorway_share >= 0.55:
        return "longhaul_corridor"
    if local_share >= 0.55:
        return "urban_mesh"
    return "regional_mixed"


def time_of_day_multiplier_uk(
    departure_time_utc: datetime | None,
    *,
    route_points: list[tuple[float, float]] | None = None,
    road_class_counts: dict[str, int] | None = None,
) -> DepartureMultiplier:
    profile = load_departure_profile()
    holiday_dates = load_uk_bank_holidays()
    if departure_time_utc is None:
        return DepartureMultiplier(
            multiplier=1.0,
            profile_source=profile.source,
            local_time_iso=None,
            profile_day="none",
            profile_key="uk_default.none.none",
            profile_version=profile.version,
            profile_as_of_utc=profile.as_of_utc,
            profile_refreshed_at_utc=profile.refreshed_at_utc,
        )

    aware = departure_time_utc if departure_time_utc.tzinfo is not None else departure_time_utc.replace(
        tzinfo=timezone.utc
    )
    local = aware.astimezone(UK_TZ)
    minute = (local.hour * 60) + local.minute
    local_date = local.date().isoformat()
    if local_date in holiday_dates:
        day_kind = "holiday"
    elif local.weekday() >= 5:
        day_kind = "weekend"
    else:
        day_kind = "weekday"
    region = _region_bucket_from_route(route_points)
    road_bucket = _road_bucket(road_class_counts)
    route_shape = _route_shape_bucket(route_points, road_class_counts)

    # Canonical v2 behavior: use contextual profile directly without legacy blend factors.
    day_series = profile.resolve(day_kind=day_kind, region=region, road_bucket=road_bucket)
    value = max(0.6, min(2.2, float(day_series[minute])))
    envelope = profile.resolve_envelope(day_kind=day_kind, region=region, road_bucket=road_bucket)
    conf_low: float | None = None
    conf_high: float | None = None
    if envelope is not None:
        low_series, high_series = envelope
        conf_low = max(0.5, min(2.5, float(low_series[minute])))
        conf_high = max(conf_low, min(2.6, float(high_series[minute])))

    return DepartureMultiplier(
        multiplier=float(value),
        profile_source=profile.source,
        local_time_iso=local.isoformat(),
        profile_day=day_kind,
        profile_key=f"{region}.{road_bucket}.{day_kind}.{route_shape}.contextual",
        profile_version=profile.version,
        profile_as_of_utc=profile.as_of_utc,
        profile_refreshed_at_utc=profile.refreshed_at_utc,
        confidence_low=conf_low,
        confidence_high=conf_high,
    )
