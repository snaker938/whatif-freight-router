from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.calibration_loader import load_scenario_profiles
from app.live_data_sources import live_scenario_context
from app.scenario import ScenarioMode, build_scenario_route_context, resolve_scenario_profile


def _signature(payload: dict[str, Any]) -> str:
    unsigned = {k: v for k, v in payload.items() if k != "signature"}
    material = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _canonical_token(raw: Any, *, fallback: str) -> str:
    token = str(raw or "").strip().lower()
    return token or fallback


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_mode_row(row: dict[str, Any]) -> dict[str, float] | None:
    required = (
        "duration_multiplier",
        "incident_rate_multiplier",
        "incident_delay_multiplier",
        "fuel_consumption_multiplier",
        "emissions_multiplier",
        "stochastic_sigma_multiplier",
    )
    out: dict[str, float] = {}
    for field in required:
        if field not in row:
            return None
        value = _safe_float(row.get(field), float("nan"))
        if value != value or value <= 0.0:
            return None
        out[field] = float(value)
    return out


def _mode_source_is_projected(raw_source: str) -> bool:
    source = _canonical_token(raw_source, fallback="unknown")
    projected_tokens = (
        "projection",
        "projected",
        "artifact",
        "counterfactual",
        "heuristic",
        "synthetic",
        "runtime_profile",
    )
    observed_tokens = (
        "observed",
        "ground_truth",
        "empirical_outcome",
        "telematics",
        "fleet_probe",
        "probe_trace",
        "sensor",
    )
    if any(token in source for token in projected_tokens):
        return True
    if any(token in source for token in observed_tokens):
        return False
    return True


def _load_observed_mode_outcomes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Observed mode-outcome JSONL file not found: {path}")
    grouped: dict[tuple[str, str, int, str, str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            continue
        corridor = _canonical_token(
            payload.get("corridor_bucket", payload.get("corridor_geohash5", "uk_default")),
            fallback="uk_default",
        )
        day_kind = _canonical_token(payload.get("day_kind", "weekday"), fallback="weekday")
        hour_slot = int(max(0, min(23, int(_safe_float(payload.get("hour_slot_local"), 12.0)))))
        road_mix = _canonical_token(payload.get("road_mix_bucket", "mixed"), fallback="mixed")
        vehicle_class = _canonical_token(payload.get("vehicle_class", "rigid_hgv"), fallback="rigid_hgv")
        weather = _canonical_token(
            payload.get("weather_bucket", payload.get("weather_regime", "clear")),
            fallback="clear",
        )
        as_of_utc = str(payload.get("as_of_utc", "")).strip() or ""
        key = (corridor, day_kind, hour_slot, road_mix, vehicle_class, weather, as_of_utc)
        mode_source = str(
            payload.get("mode_observation_source", payload.get("mode_source", "unknown"))
        ).strip() or "unknown"
        mode_is_projected = _mode_source_is_projected(mode_source)
        entry = grouped.setdefault(
            key,
            {
                "corridor_bucket": corridor,
                "day_kind": day_kind,
                "hour_slot_local": hour_slot,
                "road_mix_bucket": road_mix,
                "vehicle_class": vehicle_class,
                "weather_bucket": weather,
                "as_of_utc": as_of_utc,
                "mode_observation_source": mode_source,
                "mode_is_projected": bool(mode_is_projected),
                "modes": {},
            },
        )
        entry["mode_observation_source"] = (
            str(entry.get("mode_observation_source", mode_source)).strip() or mode_source
        )
        entry["mode_is_projected"] = bool(entry.get("mode_is_projected", True)) and bool(mode_is_projected)
        modes_raw = payload.get("modes")
        if isinstance(modes_raw, dict):
            for mode_name, mode_row in modes_raw.items():
                if str(mode_name) not in {m.value for m in ScenarioMode}:
                    continue
                if not isinstance(mode_row, dict):
                    continue
                normalized = _normalize_mode_row(mode_row)
                if normalized is None:
                    continue
                entry["modes"][str(mode_name)] = normalized
            continue

        mode_name = str(payload.get("mode", "")).strip().lower()
        if mode_name not in {m.value for m in ScenarioMode}:
            continue
        normalized = _normalize_mode_row(payload)
        if normalized is None:
            continue
        entry["modes"][mode_name] = normalized

    out: list[dict[str, Any]] = []
    for row in grouped.values():
        modes = row.get("modes", {})
        if not isinstance(modes, dict):
            continue
        if not all(mode.value in modes for mode in ScenarioMode):
            continue
        out.append(row)
    return out


def _match_observed_mode_outcome(
    *,
    outcomes: list[dict[str, Any]],
    corridor_bucket: str,
    road_mix_bucket: str,
    vehicle_class: str,
    day_kind: str,
    weather_bucket: str,
    hour_slot_local: int,
) -> dict[str, Any] | None:
    if not outcomes:
        return None
    corridor = _canonical_token(corridor_bucket, fallback="uk_default")
    road_mix = _canonical_token(road_mix_bucket, fallback="mixed")
    vehicle = _canonical_token(vehicle_class, fallback="rigid_hgv")
    day = _canonical_token(day_kind, fallback="weekday")
    weather = _canonical_token(weather_bucket, fallback="clear")
    hour = int(max(0, min(23, int(hour_slot_local))))

    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in outcomes:
        if _canonical_token(row.get("corridor_bucket"), fallback="uk_default") != corridor:
            continue
        if _canonical_token(row.get("day_kind"), fallback="weekday") != day:
            continue
        if _canonical_token(row.get("road_mix_bucket"), fallback="mixed") != road_mix:
            continue
        if _canonical_token(row.get("vehicle_class"), fallback="rigid_hgv") != vehicle:
            continue
        row_hour = int(max(0, min(23, int(_safe_float(row.get("hour_slot_local"), 12.0)))))
        hour_delta = abs(row_hour - hour)
        row_weather = _canonical_token(row.get("weather_bucket"), fallback="clear")
        weather_penalty = 0.0 if row_weather == weather else 0.25
        score = float(hour_delta) + weather_penalty
        candidates.append((score, row))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def build_snapshot(
    *,
    output_json: Path,
    corridor_bucket: str,
    road_mix_bucket: str,
    vehicle_class: str,
    day_kind: str,
    weather_bucket: str,
    centroid_lat: float,
    centroid_lon: float,
    road_hint: str | None,
    hour_slot_local: int = 12,
    project_modes_from_artifact: bool = False,
    allow_partial_sources: bool = False,
    observed_mode_outcomes: list[dict[str, Any]] | None = None,
    require_observed_modes: bool = False,
    persist_output: bool = True,
) -> dict[str, Any]:
    route_context = {
        "corridor_bucket": corridor_bucket,
        "road_mix_bucket": road_mix_bucket,
        "vehicle_class": vehicle_class,
        "day_kind": day_kind,
        "hour_slot_local": int(max(0, min(23, int(hour_slot_local)))),
        "weather_bucket": weather_bucket,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "road_hint": road_hint or "",
    }
    payload = live_scenario_context(route_context, allow_partial_sources=bool(allow_partial_sources))
    if not isinstance(payload, dict):
        raise RuntimeError("Scenario live payload fetch returned no data.")
    if "_live_error" in payload:
        err = payload.get("_live_error", {})
        reason = err.get("reason_code", "scenario_profile_unavailable")
        message = err.get("message", "scenario live source unavailable")
        raise RuntimeError(f"{reason}: {message}")

    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out = {
        **payload,
        **route_context,
        "source": "free_live_apis:scenario_context_uk",
        "generated_at_utc": now_iso,
        "calibration_basis": "empirical_live",
        "signature_algorithm": "sha256",
        "signed": True,
    }
    matched_observed_outcome = _match_observed_mode_outcome(
        outcomes=(observed_mode_outcomes or []),
        corridor_bucket=corridor_bucket,
        road_mix_bucket=road_mix_bucket,
        vehicle_class=vehicle_class,
        day_kind=day_kind,
        weather_bucket=weather_bucket,
        hour_slot_local=hour_slot_local,
    )
    if matched_observed_outcome is not None:
        modes_raw = matched_observed_outcome.get("modes", {})
        mode_rows: dict[str, dict[str, float]] = {}
        if isinstance(modes_raw, dict):
            for mode in (ScenarioMode.NO_SHARING, ScenarioMode.PARTIAL_SHARING, ScenarioMode.FULL_SHARING):
                row = modes_raw.get(mode.value)
                if not isinstance(row, dict):
                    continue
                normalized = _normalize_mode_row(row)
                if normalized is None:
                    continue
                mode_rows[mode.value] = normalized
        if len(mode_rows) == 3:
            mode_source = str(
                matched_observed_outcome.get("mode_observation_source", "unknown")
            ).strip() or "unknown"
            mode_is_projected = bool(
                matched_observed_outcome.get("mode_is_projected", _mode_source_is_projected(mode_source))
            )
            if require_observed_modes and mode_is_projected:
                raise RuntimeError(
                    "Observed mode-outcome match is projection-tagged/unknown and cannot satisfy "
                    "--require-observed-modes."
                )
            out["modes"] = mode_rows
            out["mode_observation_source"] = mode_source
            out["mode_observation_dataset"] = "independent_observed_mode_outcomes"
            out["mode_is_projected"] = bool(mode_is_projected)
    elif project_modes_from_artifact:
        departure_for_context = _departure_for_day_kind(day_kind=day_kind, hour_slot_local=hour_slot_local)
        scenario_context = build_scenario_route_context(
            route_points=[(float(centroid_lat), float(centroid_lon)), (float(centroid_lat), float(centroid_lon))],
            road_class_counts=None,
            vehicle_class=str(vehicle_class),
            departure_time_utc=departure_for_context,
            weather_bucket=str(weather_bucket),
            road_hint=road_hint,
        )
        mode_rows: dict[str, dict[str, float]] = {}
        if allow_partial_sources:
            profiles = load_scenario_profiles()
            ctx = (profiles.contexts or {}).get(scenario_context.context_key) if profiles.contexts else None
            for mode in (ScenarioMode.NO_SHARING, ScenarioMode.PARTIAL_SHARING, ScenarioMode.FULL_SHARING):
                selected = None
                if ctx is not None:
                    selected = ctx.profiles.get(mode.value)
                if selected is None:
                    selected = (profiles.profiles or {}).get(mode.value)
                if selected is None:
                    continue
                mode_rows[mode.value] = {
                    "duration_multiplier": float(selected.duration_multiplier),
                    "incident_rate_multiplier": float(selected.incident_rate_multiplier),
                    "incident_delay_multiplier": float(selected.incident_delay_multiplier),
                    "fuel_consumption_multiplier": float(selected.fuel_consumption_multiplier),
                    "emissions_multiplier": float(selected.emissions_multiplier),
                    "stochastic_sigma_multiplier": float(selected.stochastic_sigma_multiplier),
                }
        else:
            for mode in (ScenarioMode.NO_SHARING, ScenarioMode.PARTIAL_SHARING, ScenarioMode.FULL_SHARING):
                resolved = resolve_scenario_profile(mode, context=scenario_context)
                mode_rows[mode.value] = {
                    "duration_multiplier": float(resolved.duration_multiplier),
                    "incident_rate_multiplier": float(resolved.incident_rate_multiplier),
                    "incident_delay_multiplier": float(resolved.incident_delay_multiplier),
                    "fuel_consumption_multiplier": float(resolved.fuel_consumption_multiplier),
                    "emissions_multiplier": float(resolved.emissions_multiplier),
                    "stochastic_sigma_multiplier": float(resolved.stochastic_sigma_multiplier),
                }
        out["modes"] = mode_rows
        out["mode_observation_source"] = "runtime_projection"
        out["mode_projection_version"] = "scenario_profile_runtime_projection_v1"
        out["mode_is_projected"] = True
    elif require_observed_modes:
        raise RuntimeError(
            "No independent observed mode-outcome row matched this scenario context "
            f"(corridor={corridor_bucket}, day_kind={day_kind}, hour={hour_slot_local}, road_mix={road_mix_bucket}, "
            f"vehicle_class={vehicle_class}, weather={weather_bucket})."
        )
    out["signature"] = _signature(out)
    if persist_output:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _parse_hour_slots(raw: str) -> list[int]:
    slots: list[int] = []
    for token in _parse_csv_list(raw):
        try:
            hour = int(float(token))
        except ValueError:
            continue
        slots.append(max(0, min(23, hour)))
    if not slots:
        return [0, 4, 8, 12, 16, 20]
    return sorted(set(slots))


def _departure_for_day_kind(*, day_kind: str, hour_slot_local: int) -> datetime:
    base = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, hour=max(0, min(23, int(hour_slot_local))))
    normalized_day = str(day_kind).strip().lower() or "weekday"
    if normalized_day == "weekend":
        # Move to the next Saturday if currently weekday.
        if base.weekday() < 5:
            base = base + timedelta(days=(5 - base.weekday()))
        return base
    # For weekday/holiday alignment, avoid weekend timestamps.
    if base.weekday() >= 5:
        base = base + timedelta(days=(7 - base.weekday()))
    return base


def _default_corridors() -> list[dict[str, Any]]:
    return [
        {"corridor": "london_southeast", "lat": 51.5074, "lon": -0.1278, "road_hint": "M25"},
        {"corridor": "north_west_corridor", "lat": 53.4808, "lon": -2.2426, "road_hint": "M6"},
        {"corridor": "north_east_corridor", "lat": 54.9783, "lon": -1.6178, "road_hint": "A1"},
        {"corridor": "midlands_west", "lat": 52.4862, "lon": -1.8904, "road_hint": "M6"},
        {"corridor": "south_england", "lat": 50.8225, "lon": -0.1372, "road_hint": "M27"},
        {"corridor": "wales_west", "lat": 51.4816, "lon": -3.1791, "road_hint": "M4"},
        {"corridor": "scotland_south", "lat": 55.9533, "lon": -3.1883, "road_hint": "M8"},
        {"corridor": "uk_default", "lat": 54.2, "lon": -2.3, "road_hint": "A1"},
    ]


def _load_corridors(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _default_corridors()
    if not path.exists():
        raise FileNotFoundError(f"Corridor config file not found: {path}")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise RuntimeError("Corridor config must be a JSON array.")
    out: list[dict[str, Any]] = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        corridor = str(row.get("corridor", row.get("corridor_bucket", ""))).strip()
        if not corridor:
            continue
        lat = _safe_float(row.get("lat"), float("nan"))
        lon = _safe_float(row.get("lon"), float("nan"))
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        road_hint = str(row.get("road_hint", "")).strip() or None
        out.append(
            {
                "corridor": corridor,
                "lat": lat,
                "lon": lon,
                "road_hint": road_hint,
            }
        )
    if not out:
        raise RuntimeError("Corridor config did not contain any valid rows.")
    return out


def build_batch_snapshots(
    *,
    output_json: Path,
    output_jsonl: Path,
    corridors: list[dict[str, Any]],
    road_mix_bucket: str,
    vehicle_class: str,
    day_kinds: list[str],
    weather_bucket: str,
    hour_slots: list[int],
    project_modes_from_artifact: bool,
    allow_partial_sources: bool = False,
    observed_mode_outcomes: list[dict[str, Any]] | None = None,
    require_observed_modes: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    tasks: list[dict[str, Any]] = []
    for corridor_row in corridors:
        corridor = str(corridor_row.get("corridor", "uk_default")).strip() or "uk_default"
        lat = float(corridor_row.get("lat", 54.2))
        lon = float(corridor_row.get("lon", -2.3))
        road_hint = str(corridor_row.get("road_hint", "")).strip() or None
        for day_kind in day_kinds:
            normalized_day = str(day_kind).strip().lower() or "weekday"
            for hour in hour_slots:
                tasks.append(
                    {
                        "corridor": corridor,
                        "lat": lat,
                        "lon": lon,
                        "road_hint": road_hint,
                        "day_kind": normalized_day,
                        "hour": int(hour),
                    }
                )

    batch_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def _worker(task: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        corridor = str(task.get("corridor", "uk_default"))
        day_kind = str(task.get("day_kind", "weekday"))
        hour_slot = int(task.get("hour", 12))
        try:
            snapshot = build_snapshot(
                output_json=output_json,
                corridor_bucket=corridor,
                road_mix_bucket=road_mix_bucket,
                vehicle_class=vehicle_class,
                day_kind=day_kind,
                weather_bucket=weather_bucket,
                centroid_lat=float(task.get("lat", 54.2)),
                centroid_lon=float(task.get("lon", -2.3)),
                road_hint=(str(task.get("road_hint", "")).strip() or None),
                hour_slot_local=hour_slot,
                project_modes_from_artifact=project_modes_from_artifact,
                allow_partial_sources=bool(allow_partial_sources),
                observed_mode_outcomes=observed_mode_outcomes,
                require_observed_modes=bool(require_observed_modes),
                persist_output=False,
            )
            return snapshot, None
        except Exception as exc:  # pragma: no cover - defensive batch continuity
            return None, {
                "corridor": corridor,
                "day_kind": day_kind,
                "hour_slot_local": hour_slot,
                "error": str(exc),
            }

    max_workers = max(1, min(int(workers), len(tasks) if tasks else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for fut in as_completed(futures):
            snapshot, err = fut.result()
            if snapshot is not None:
                batch_rows.append(snapshot)
            if err is not None:
                errors.append(err)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if batch_rows:
        with output_jsonl.open("a", encoding="utf-8") as handle:
            for row in batch_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")

    summary: dict[str, Any] = {
        "source": "free_live_apis:scenario_context_uk_batch",
        "generated_at_utc": now_iso,
        "as_of_utc": now_iso,
        "calibration_basis": "empirical_live",
        "batch_count": int(len(batch_rows)),
        "error_count": int(len(errors)),
        "corridor_count": int(len(corridors)),
        "hour_slot_count": int(len(set(hour_slots))),
        "day_kind_count": int(len(set(day_kinds))),
        "observed_mode_outcomes_loaded": int(len(observed_mode_outcomes or [])),
        "require_observed_modes": bool(require_observed_modes),
        "output_jsonl": str(output_jsonl),
        "errors": errors[:200],
        "signature_algorithm": "sha256",
        "signed": True,
    }
    summary["signature"] = _signature(summary)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live UK scenario context from free APIs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "out" / "model_assets" / "scenario_live_snapshot_uk.json",
        help="Output JSON snapshot path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "scenario_live_observed.jsonl",
        help="Append-only JSONL corpus output path for batch pulls.",
    )
    parser.add_argument("--batch", action="store_true", help="Fetch a corridor/day/hour grid and append JSONL corpus.")
    parser.add_argument("--corridor-config", type=Path, default=None, help="Optional JSON corridor config file.")
    parser.add_argument("--corridor", type=str, default="uk_default")
    parser.add_argument("--road-mix", type=str, default="mixed")
    parser.add_argument("--vehicle-class", type=str, default="rigid_hgv")
    parser.add_argument("--day-kind", type=str, default="weekday")
    parser.add_argument("--day-kinds", type=str, default="weekday,weekend")
    parser.add_argument("--hour-slots", type=str, default="0,4,8,12,16,20")
    parser.add_argument("--weather", type=str, default="clear")
    parser.add_argument("--hour-slot", type=int, default=12)
    parser.add_argument("--lat", type=float, default=54.2)
    parser.add_argument("--lon", type=float, default=-2.3)
    parser.add_argument("--road-hint", type=str, default="")
    parser.add_argument(
        "--project-modes-from-artifact",
        action="store_true",
        help="Populate optional modes{} using the current strict scenario artifact.",
    )
    parser.add_argument(
        "--allow-partial-sources",
        action="store_true",
        help=(
            "Allow corpus collection to continue when one or more live sources fail. "
            "This only affects this data-collection script; strict runtime endpoints remain unchanged."
        ),
    )
    parser.add_argument(
        "--observed-mode-outcomes-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional independent observed per-mode outcomes dataset (JSONL). "
            "When provided, matched contexts write observed modes instead of runtime projection."
        ),
    )
    parser.add_argument(
        "--require-observed-modes",
        action="store_true",
        help="Fail snapshot collection when no observed mode-outcome row matches a scenario context.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count for --batch mode.",
    )
    args = parser.parse_args()
    observed_mode_outcomes = (
        _load_observed_mode_outcomes(args.observed_mode_outcomes_jsonl)
        if args.observed_mode_outcomes_jsonl is not None
        else None
    )

    if bool(args.batch):
        summary = build_batch_snapshots(
            output_json=args.output,
            output_jsonl=args.output_jsonl,
            corridors=_load_corridors(args.corridor_config),
            road_mix_bucket=args.road_mix,
            vehicle_class=args.vehicle_class,
            day_kinds=_parse_csv_list(args.day_kinds) or [args.day_kind],
            weather_bucket=args.weather,
            hour_slots=_parse_hour_slots(args.hour_slots),
            project_modes_from_artifact=bool(args.project_modes_from_artifact),
            allow_partial_sources=bool(args.allow_partial_sources),
            observed_mode_outcomes=observed_mode_outcomes,
            require_observed_modes=bool(args.require_observed_modes),
            workers=max(1, int(args.workers)),
        )
        print(
            f"Wrote scenario batch summary to {args.output} "
            f"(rows={summary.get('batch_count')}, errors={summary.get('error_count')}, corpus={args.output_jsonl})."
        )
        return

    payload = build_snapshot(
        output_json=args.output,
        corridor_bucket=args.corridor,
        road_mix_bucket=args.road_mix,
        vehicle_class=args.vehicle_class,
        day_kind=args.day_kind,
        weather_bucket=args.weather,
        centroid_lat=float(args.lat),
        centroid_lon=float(args.lon),
        road_hint=(args.road_hint or None),
        hour_slot_local=int(max(0, min(23, int(args.hour_slot)))),
        project_modes_from_artifact=bool(args.project_modes_from_artifact),
        allow_partial_sources=bool(args.allow_partial_sources),
        observed_mode_outcomes=observed_mode_outcomes,
        require_observed_modes=bool(args.require_observed_modes),
    )
    print(
        f"Wrote scenario live snapshot to {args.output} "
        f"(as_of_utc={payload.get('as_of_utc')}, coverage={payload.get('coverage', {}).get('overall')})."
    )


if __name__ == "__main__":
    main()
