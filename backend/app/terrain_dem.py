from __future__ import annotations

from bisect import bisect_left
import math
from dataclasses import dataclass
from functools import lru_cache

from .settings import settings
from .terrain_dem_index import load_terrain_manifest, sample_elevation_m, terrain_runtime_status
from .terrain_physics import (
    params_for_vehicle,
    route_emissions_multiplier,
    segment_duration_multiplier,
)


EARTH_RADIUS_M = 6_371_000.0
UK_BBOX = {
    "lat_min": 49.75,
    "lat_max": 61.10,
    "lon_min": -8.75,
    "lon_max": 2.25,
}


def _configured_uk_bbox() -> dict[str, float]:
    raw = (settings.terrain_uk_bbox or "").strip()
    if not raw:
        return dict(UK_BBOX)
    try:
        lat_min_raw, lat_max_raw, lon_min_raw, lon_max_raw = [float(part.strip()) for part in raw.split(",")]
        if lat_max_raw <= lat_min_raw or lon_max_raw <= lon_min_raw:
            return dict(UK_BBOX)
        return {
            "lat_min": lat_min_raw,
            "lat_max": lat_max_raw,
            "lon_min": lon_min_raw,
            "lon_max": lon_max_raw,
        }
    except Exception:
        return dict(UK_BBOX)


def _terrain_profile_floor(profile: str) -> tuple[float, float]:
    duration_floor = {"flat": 1.0, "rolling": 1.02, "hilly": 1.05}.get(profile, 1.0)
    emissions_floor = {"flat": 1.0, "rolling": 1.03, "hilly": 1.07}.get(profile, 1.0)
    return duration_floor, emissions_floor


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


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


def _is_uk_point(lat: float, lon: float) -> bool:
    bbox = _configured_uk_bbox()
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )


def _route_intersects_uk(coordinates_lon_lat: list[tuple[float, float]]) -> bool:
    for lon, lat in coordinates_lon_lat:
        if _is_uk_point(lat, lon):
            return True
    return False


@dataclass(frozen=True)
class TerrainCoverageError(ValueError):
    reason_code: str
    coverage_ratio: float
    required_ratio: float
    version: str
    message: str

    def __str__(self) -> str:
        return self.message

    @classmethod
    def dem_coverage_insufficient(cls, *, coverage_ratio: float, required_ratio: float, version: str) -> "TerrainCoverageError":
        return cls(
            reason_code="terrain_dem_coverage_insufficient",
            coverage_ratio=coverage_ratio,
            required_ratio=required_ratio,
            version=version,
            message=(
                "Terrain DEM coverage below UK fail-closed threshold "
                f"({coverage_ratio:.4f} < {required_ratio:.4f}, dem={version})."
            ),
        )

    @classmethod
    def region_unsupported(cls, *, version: str) -> "TerrainCoverageError":
        return cls(
            reason_code="terrain_region_unsupported",
            coverage_ratio=0.0,
            required_ratio=max(0.0, min(1.0, float(settings.terrain_dem_coverage_min_uk))),
            version=version,
            message="Terrain model is UK-only in strict mode; route is outside the configured UK region.",
        )

    @classmethod
    def asset_unavailable(cls, *, version: str) -> "TerrainCoverageError":
        return cls(
            reason_code="terrain_dem_asset_unavailable",
            coverage_ratio=0.0,
            required_ratio=max(0.0, min(1.0, float(settings.terrain_dem_coverage_min_uk))),
            version=version,
            message="Terrain DEM assets are unavailable or runtime readers are missing.",
        )


@dataclass(frozen=True)
class TerrainSummary:
    duration_multiplier: float
    emissions_multiplier: float
    ascent_m: float
    descent_m: float
    grade_histogram: dict[str, float]
    source: str
    coverage_ratio: float
    sample_spacing_m: float
    confidence: float
    fail_closed_applied: bool
    version: str

    def as_route_option_payload(self) -> dict[str, float | bool | str | dict[str, float]]:
        return {
            "source": self.source,
            "coverage_ratio": round(self.coverage_ratio, 6),
            "sample_spacing_m": round(self.sample_spacing_m, 3),
            "ascent_m": round(self.ascent_m, 3),
            "descent_m": round(self.descent_m, 3),
            "grade_histogram": self.grade_histogram,
            "confidence": round(self.confidence, 6),
            "fail_closed_applied": self.fail_closed_applied,
            "version": self.version,
        }


@dataclass(frozen=True)
class _SampledPoint:
    distance_m: float
    lat: float
    lon: float
    elevation_m: float
    covered: bool


def _densify_polyline(
    coordinates_lon_lat: list[tuple[float, float]],
    *,
    spacing_m: float,
    max_samples: int,
) -> list[tuple[float, float]]:
    if len(coordinates_lon_lat) < 2:
        return coordinates_lon_lat

    out: list[tuple[float, float]] = [coordinates_lon_lat[0]]
    for idx in range(1, len(coordinates_lon_lat)):
        lon1, lat1 = coordinates_lon_lat[idx - 1]
        lon2, lat2 = coordinates_lon_lat[idx]
        seg_m = _haversine_m(lat1, lon1, lat2, lon2)
        steps = max(1, int(math.ceil(seg_m / max(5.0, spacing_m))))
        for step in range(1, steps + 1):
            t = step / steps
            out.append(
                (
                    lon1 + ((lon2 - lon1) * t),
                    lat1 + ((lat2 - lat1) * t),
                )
            )

    if len(out) <= max_samples:
        return out
    keep_every = max(1, int(math.ceil(len(out) / max_samples)))
    trimmed = out[::keep_every]
    if trimmed[-1] != out[-1]:
        trimmed.append(out[-1])
    return trimmed[:max_samples]


def _sampled_route_profile(
    coordinates_lon_lat: list[tuple[float, float]],
    *,
    spacing_m: float,
    max_samples: int,
) -> tuple[list[_SampledPoint], float, str]:
    densified = _densify_polyline(
        coordinates_lon_lat,
        spacing_m=spacing_m,
        max_samples=max_samples,
    )
    if not densified:
        return [], 0.0, load_terrain_manifest().version
    sampled: list[_SampledPoint] = []
    total = 0
    covered = 0
    distance_cursor = 0.0
    version = "missing"

    for idx, (lon, lat) in enumerate(densified):
        if idx > 0:
            prev_lon, prev_lat = densified[idx - 1]
            distance_cursor += _haversine_m(prev_lat, prev_lon, lat, lon)
        elev, ok, version = sample_elevation_m(lat, lon)
        total += 1
        if ok:
            covered += 1
            elevation = elev
        else:
            elevation = math.nan
        sampled.append(
            _SampledPoint(
                distance_m=distance_cursor,
                lat=lat,
                lon=lon,
                elevation_m=elevation,
                covered=ok,
            )
        )

    coverage_ratio = (covered / total) if total > 0 else 0.0
    return sampled, coverage_ratio, version


def _fill_missing_elevation(sampled: list[_SampledPoint]) -> list[_SampledPoint]:
    if not sampled:
        return sampled
    covered_idx = [idx for idx, point in enumerate(sampled) if point.covered and math.isfinite(point.elevation_m)]
    if not covered_idx:
        return sampled
    out = list(sampled)
    first_idx = covered_idx[0]
    first_val = out[first_idx].elevation_m
    for idx in range(0, first_idx):
        point = out[idx]
        out[idx] = _SampledPoint(
            distance_m=point.distance_m,
            lat=point.lat,
            lon=point.lon,
            elevation_m=first_val,
            covered=False,
        )
    for pos in range(1, len(covered_idx)):
        left_idx = covered_idx[pos - 1]
        right_idx = covered_idx[pos]
        left = out[left_idx]
        right = out[right_idx]
        span = max(1e-6, right.distance_m - left.distance_m)
        for idx in range(left_idx + 1, right_idx):
            point = out[idx]
            t = (point.distance_m - left.distance_m) / span
            elev = left.elevation_m + ((right.elevation_m - left.elevation_m) * t)
            out[idx] = _SampledPoint(
                distance_m=point.distance_m,
                lat=point.lat,
                lon=point.lon,
                elevation_m=elev,
                covered=False,
            )
    last_idx = covered_idx[-1]
    last_val = out[last_idx].elevation_m
    for idx in range(last_idx + 1, len(out)):
        point = out[idx]
        out[idx] = _SampledPoint(
            distance_m=point.distance_m,
            lat=point.lat,
            lon=point.lon,
            elevation_m=last_val,
            covered=False,
        )
    return out


def _filter_elevation_spikes(sampled: list[_SampledPoint]) -> list[_SampledPoint]:
    if len(sampled) < 3:
        return sampled
    out = list(sampled)

    # Pass 1: continuity-aware median clamp to remove single-point spikes.
    for idx in range(1, len(out) - 1):
        prev = out[idx - 1]
        curr = out[idx]
        nxt = out[idx + 1]
        local = sorted([prev.elevation_m, curr.elevation_m, nxt.elevation_m])
        median = local[1]
        seg_left_m = max(1.0, curr.distance_m - prev.distance_m)
        seg_right_m = max(1.0, nxt.distance_m - curr.distance_m)
        span_m = 0.5 * (seg_left_m + seg_right_m)
        coverage_score = (
            (1.0 if prev.covered else 0.0)
            + (1.0 if curr.covered else 0.0)
            + (1.0 if nxt.covered else 0.0)
        ) / 3.0
        max_grade = 0.12 + (0.18 * coverage_score)
        max_delta = max_grade * span_m
        delta_from_median = curr.elevation_m - median
        if abs(delta_from_median) > max_delta:
            adjusted = median + math.copysign(max_delta, delta_from_median)
            out[idx] = _SampledPoint(
                distance_m=curr.distance_m,
                lat=curr.lat,
                lon=curr.lon,
                elevation_m=adjusted,
                covered=curr.covered,
            )

    # Pass 2: slope continuity clamp to limit abrupt grade discontinuities.
    for idx in range(1, len(out)):
        prev = out[idx - 1]
        curr = out[idx]
        dist_m = max(1.0, curr.distance_m - prev.distance_m)
        prev_slope = 0.0
        if idx >= 2:
            prev2 = out[idx - 2]
            prev_dist_m = max(1.0, prev.distance_m - prev2.distance_m)
            prev_slope = (prev.elevation_m - prev2.elevation_m) / prev_dist_m
        coverage_pair = (1.0 if prev.covered else 0.0) * 0.5 + (1.0 if curr.covered else 0.0) * 0.5
        base_grade = 0.10 + (0.20 * coverage_pair)
        continuity_allowance = min(0.12, abs(prev_slope) * 0.45)
        max_grade = base_grade + continuity_allowance
        max_delta = max_grade * dist_m
        delta = curr.elevation_m - prev.elevation_m
        if delta > max_delta:
            clamped = prev.elevation_m + max_delta
            out[idx] = _SampledPoint(
                distance_m=curr.distance_m,
                lat=curr.lat,
                lon=curr.lon,
                elevation_m=clamped,
                covered=curr.covered,
            )
        elif delta < -max_delta:
            clamped = prev.elevation_m - max_delta
            out[idx] = _SampledPoint(
                distance_m=curr.distance_m,
                lat=curr.lat,
                lon=curr.lon,
                elevation_m=clamped,
                covered=curr.covered,
            )

    # Pass 3: confidence-weighted local smoothing to reduce nodata interpolation
    # artifacts while preserving covered DEM points.
    if len(out) >= 5:
        smoothed = list(out)
        for idx in range(2, len(out) - 2):
            window = out[idx - 2 : idx + 3]
            weights: list[float] = []
            values: list[float] = []
            for offset, point in enumerate(window):
                base_w = [0.35, 0.8, 1.0, 0.8, 0.35][offset]
                cov_w = 1.0 if point.covered else 0.45
                weights.append(base_w * cov_w)
                values.append(point.elevation_m)
            w_sum = sum(weights)
            if w_sum <= 1e-9:
                continue
            blended = sum(v * w for v, w in zip(values, weights, strict=True)) / w_sum
            anchor = out[idx]
            anchor_w = 0.78 if anchor.covered else 0.38
            elev = (anchor.elevation_m * anchor_w) + (blended * (1.0 - anchor_w))
            smoothed[idx] = _SampledPoint(
                distance_m=anchor.distance_m,
                lat=anchor.lat,
                lon=anchor.lon,
                elevation_m=elev,
                covered=anchor.covered,
            )
        out = smoothed
    return out


def _interp_elevation(profile: list[_SampledPoint], target_m: float) -> float:
    if not profile:
        return 0.0
    if target_m <= profile[0].distance_m:
        return profile[0].elevation_m
    if target_m >= profile[-1].distance_m:
        return profile[-1].elevation_m
    for idx in range(1, len(profile)):
        left = profile[idx - 1]
        right = profile[idx]
        if target_m > right.distance_m:
            continue
        span = max(1e-6, right.distance_m - left.distance_m)
        t = (target_m - left.distance_m) / span
        return left.elevation_m + ((right.elevation_m - left.elevation_m) * t)
    return profile[-1].elevation_m


def _grade_histogram(grades: list[float], distances_m: list[float]) -> dict[str, float]:
    bins = {
        "steep_up": 0.0,
        "gentle_up": 0.0,
        "flat": 0.0,
        "gentle_down": 0.0,
        "steep_down": 0.0,
    }
    for grade, dist_m in zip(grades, distances_m, strict=False):
        km = max(0.0, dist_m) / 1000.0
        if grade >= 0.05:
            bins["steep_up"] += km
        elif grade >= 0.015:
            bins["gentle_up"] += km
        elif grade <= -0.05:
            bins["steep_down"] += km
        elif grade <= -0.015:
            bins["gentle_down"] += km
        else:
            bins["flat"] += km
    total_km = max(1e-9, sum(bins.values()))
    return {k: round(v / total_km, 6) for k, v in bins.items()}


def _route_signature_key(coordinates_lon_lat: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if len(coordinates_lon_lat) <= 200:
        return tuple((round(lon, 6), round(lat, 6)) for lon, lat in coordinates_lon_lat)
    step = max(1, len(coordinates_lon_lat) // 200)
    sampled = coordinates_lon_lat[::step]
    if sampled[-1] != coordinates_lon_lat[-1]:
        sampled = [*sampled, coordinates_lon_lat[-1]]
    return tuple((round(lon, 6), round(lat, 6)) for lon, lat in sampled)


@lru_cache(maxsize=128)
def _cached_profile(
    signature: tuple[tuple[float, float], ...],
    spacing_m: float,
    max_samples: int,
) -> tuple[list[_SampledPoint], float, str]:
    coords = [(lon, lat) for lon, lat in signature]
    profile, coverage, version = _sampled_route_profile(
        coords,
        spacing_m=spacing_m,
        max_samples=max_samples,
    )
    profile = _fill_missing_elevation(profile)
    profile = _filter_elevation_spikes(profile)
    return profile, coverage, version


def segment_grade_profile(
    *,
    coordinates_lon_lat: list[tuple[float, float]],
    segment_distances_m: list[float],
) -> list[float]:
    if not coordinates_lon_lat or not segment_distances_m:
        return [0.0 for _ in segment_distances_m]

    spacing_m = max(
        20.0,
        float(settings.terrain_sample_spacing_m),
        float(settings.terrain_dem_resolution_m),
    )
    max_samples = max(100, int(settings.terrain_max_samples_per_route))
    signature = _route_signature_key(coordinates_lon_lat)
    profile, _coverage, _version = _cached_profile(
        signature,
        spacing_m,
        max_samples,
    )
    if len(profile) < 2:
        return [0.0 for _ in segment_distances_m]

    profile_total_m = max(profile[-1].distance_m, 1.0)
    geom_distances: list[float] = [0.0]
    geom_total_m = 0.0
    for idx in range(1, len(coordinates_lon_lat)):
        lon1, lat1 = coordinates_lon_lat[idx - 1]
        lon2, lat2 = coordinates_lon_lat[idx]
        geom_total_m += _haversine_m(lat1, lon1, lat2, lon2)
        geom_distances.append(geom_total_m)
    geom_total_m = max(geom_total_m, 1.0)

    def _geometry_point_at_distance(target_geom_m: float) -> tuple[float, float]:
        if target_geom_m <= 0.0:
            return coordinates_lon_lat[0]
        if target_geom_m >= geom_total_m:
            return coordinates_lon_lat[-1]
        pos = bisect_left(geom_distances, target_geom_m)
        pos = max(1, min(pos, len(geom_distances) - 1))
        left_d = geom_distances[pos - 1]
        right_d = geom_distances[pos]
        span = max(1e-6, right_d - left_d)
        t = (target_geom_m - left_d) / span
        lon1, lat1 = coordinates_lon_lat[pos - 1]
        lon2, lat2 = coordinates_lon_lat[pos]
        return (
            lon1 + ((lon2 - lon1) * t),
            lat1 + ((lat2 - lat1) * t),
        )

    def _elevation_at_segment_boundary(boundary_m: float) -> float:
        # Strict geometry-aligned projection: use absolute segment boundary distance
        # on the route chain, not normalized ratio remapping.
        geom_boundary = max(0.0, min(geom_total_m, boundary_m))
        lon, lat = _geometry_point_at_distance(geom_boundary)
        elev, ok, _version = sample_elevation_m(lat, lon)
        if ok and math.isfinite(elev):
            return elev
        profile_boundary = max(0.0, min(profile_total_m, geom_boundary))
        return _interp_elevation(profile, profile_boundary)

    cursor_segment_m = 0.0
    grades: list[float] = []
    for seg_m_raw in segment_distances_m:
        seg_m = max(0.0, float(seg_m_raw))
        start_e = _elevation_at_segment_boundary(cursor_segment_m)
        end_e = _elevation_at_segment_boundary(cursor_segment_m + seg_m)
        grade = (end_e - start_e) / max(1.0, seg_m)
        grades.append(_clamp(grade, -0.25, 0.25))
        cursor_segment_m += seg_m
    return grades


def estimate_terrain_summary(
    *,
    coordinates_lon_lat: list[tuple[float, float]],
    terrain_profile: str,
    avg_speed_kmh: float,
    distance_km: float,
    vehicle_type: str = "rigid_hgv",
) -> TerrainSummary:
    if len(coordinates_lon_lat) < 2:
        d_mult, e_mult = _terrain_profile_floor(terrain_profile)
        return TerrainSummary(
            duration_multiplier=d_mult,
            emissions_multiplier=e_mult,
            ascent_m=0.0,
            descent_m=0.0,
            grade_histogram={"flat": 1.0},
            source="missing",
            coverage_ratio=0.0,
            sample_spacing_m=float(settings.terrain_sample_spacing_m),
            confidence=0.0,
            fail_closed_applied=False,
            version=load_terrain_manifest().version,
        )

    spacing_m = max(
        20.0,
        float(settings.terrain_sample_spacing_m),
        float(settings.terrain_dem_resolution_m),
    )
    max_samples = max(100, int(settings.terrain_max_samples_per_route))
    route_intersects_uk = _route_intersects_uk(coordinates_lon_lat)
    manifest = load_terrain_manifest()
    if bool(settings.terrain_uk_only_support) and not route_intersects_uk:
        raise TerrainCoverageError.region_unsupported(version=manifest.version)
    runtime_ok, _runtime_reason = terrain_runtime_status()
    if not runtime_ok:
        raise TerrainCoverageError.asset_unavailable(version=manifest.version)
    signature = _route_signature_key(coordinates_lon_lat)
    profile, coverage_ratio, version = _cached_profile(
        signature,
        spacing_m,
        max_samples,
    )

    required = max(0.0, min(1.0, float(settings.terrain_dem_coverage_min_uk)))
    if route_intersects_uk and bool(settings.terrain_dem_fail_closed_uk) and coverage_ratio < required:
        raise TerrainCoverageError.dem_coverage_insufficient(
            coverage_ratio=coverage_ratio,
            required_ratio=required,
            version=version,
        )

    if len(profile) < 2:
        d_mult, e_mult = _terrain_profile_floor(terrain_profile)
        return TerrainSummary(
            duration_multiplier=d_mult,
            emissions_multiplier=e_mult,
            ascent_m=0.0,
            descent_m=0.0,
            grade_histogram={"flat": 1.0},
            source="missing",
            coverage_ratio=coverage_ratio,
            sample_spacing_m=spacing_m,
            confidence=0.0,
            fail_closed_applied=False,
            version=version,
        )

    ascent_m = 0.0
    descent_m = 0.0
    grades: list[float] = []
    distances_m: list[float] = []
    uphill_grade_distance = 0.0
    downhill_grade_distance = 0.0
    uphill_distance_km = 0.0
    downhill_distance_km = 0.0
    weighted_duration_mult_sum = 0.0
    weighted_distance_m = 0.0
    vehicle_params = params_for_vehicle(vehicle_type)

    for idx in range(1, len(profile)):
        prev = profile[idx - 1]
        curr = profile[idx]
        seg_m = max(0.0, curr.distance_m - prev.distance_m)
        if seg_m <= 0.1:
            continue
        elev_delta = curr.elevation_m - prev.elevation_m
        if elev_delta > 0:
            ascent_m += elev_delta
        else:
            descent_m += abs(elev_delta)
        grade = elev_delta / seg_m
        grade = _clamp(grade, -0.25, 0.25)
        grades.append(grade)
        distances_m.append(seg_m)

        if grade > 0:
            uphill_grade_distance += grade * (seg_m / 1000.0)
            uphill_distance_km += seg_m / 1000.0
        elif grade < 0:
            downhill_grade_distance += abs(grade) * (seg_m / 1000.0)
            downhill_distance_km += seg_m / 1000.0

        seg_speed = max(8.0, float(avg_speed_kmh))
        seg_mult = segment_duration_multiplier(
            grade=grade,
            speed_kmh=seg_speed,
            terrain_profile=terrain_profile,
            params=vehicle_params,
        )
        weighted_duration_mult_sum += seg_mult * seg_m
        weighted_distance_m += seg_m

    if not grades:
        d_mult, e_mult = _terrain_profile_floor(terrain_profile)
        return TerrainSummary(
            duration_multiplier=d_mult,
            emissions_multiplier=e_mult,
            ascent_m=round(ascent_m, 3),
            descent_m=round(descent_m, 3),
            grade_histogram={"flat": 1.0},
            source="missing",
            coverage_ratio=coverage_ratio,
            sample_spacing_m=spacing_m,
            confidence=0.1,
            fail_closed_applied=False,
            version=version,
        )

    duration_multiplier = weighted_duration_mult_sum / max(1.0, weighted_distance_m)
    mean_grade_up = uphill_grade_distance / max(1e-9, uphill_distance_km)
    mean_grade_down = downhill_grade_distance / max(1e-9, downhill_distance_km)
    emissions_multiplier = route_emissions_multiplier(
        mean_grade_up=mean_grade_up,
        mean_grade_down=mean_grade_down,
        terrain_profile=terrain_profile,
        params=vehicle_params,
    )

    if not route_intersects_uk:
        source = "unsupported_region"
    else:
        source = "dem_real" if coverage_ratio >= required else "missing"

    coverage_conf = coverage_ratio
    grade_conf = 1.0 - min(0.4, abs(mean_grade_up - mean_grade_down))
    confidence = _clamp((coverage_conf * 0.75) + (grade_conf * 0.25), 0.0, 1.0)
    duration_floor_by_profile = {"flat": 1.0, "rolling": 1.02, "hilly": 1.05}
    emissions_floor_by_profile = {"flat": 1.0, "rolling": 1.03, "hilly": 1.07}
    duration_multiplier = max(duration_floor_by_profile.get(terrain_profile, 1.0), duration_multiplier)
    emissions_multiplier = max(emissions_floor_by_profile.get(terrain_profile, 1.0), emissions_multiplier)

    return TerrainSummary(
        duration_multiplier=round(_clamp(duration_multiplier, 0.85, 1.65), 6),
        emissions_multiplier=round(_clamp(emissions_multiplier, 1.0, 1.85), 6),
        ascent_m=round(ascent_m, 3),
        descent_m=round(descent_m, 3),
        grade_histogram=_grade_histogram(grades, distances_m),
        source=source,
        coverage_ratio=round(coverage_ratio, 6),
        sample_spacing_m=round(spacing_m, 3),
        confidence=round(confidence, 6),
        fail_closed_applied=bool(settings.terrain_dem_fail_closed_uk and route_intersects_uk),
        version=version,
    )
