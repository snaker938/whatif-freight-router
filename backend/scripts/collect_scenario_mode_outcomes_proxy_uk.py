from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODE_OBSERVATION_SOURCE = "empirical_outcome_public_feeds_v1"
MODE_OBSERVATION_DATASET = "public_context_observed_outcomes_v1"


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_text(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    return text or fallback


def _corridor_geohash5(raw: Any, *, corridor_bucket: str) -> str:
    text = str(raw or "").strip().lower()
    if text:
        return text
    digest = hashlib.sha1(corridor_bucket.encode("utf-8")).hexdigest()
    return digest[:5]


def _proxy_mode_values(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    traffic = row.get("traffic_features") if isinstance(row.get("traffic_features"), dict) else {}
    incident = row.get("incident_features") if isinstance(row.get("incident_features"), dict) else {}
    weather = row.get("weather_features") if isinstance(row.get("weather_features"), dict) else {}

    flow_index = _coerce_float(traffic.get("flow_index", row.get("flow_index")), 120.0)
    speed_index = _coerce_float(traffic.get("speed_index", row.get("speed_index")), 60.0)
    delay_pressure = _coerce_float(incident.get("delay_pressure", row.get("delay_pressure")), 0.0)
    severity_index = _coerce_float(incident.get("severity_index", row.get("severity_index")), 0.0)
    weather_severity = _coerce_float(
        weather.get("weather_severity_index", row.get("weather_severity_index")),
        1.0,
    )

    traffic_factor = _clamp((flow_index / 160.0) + (60.0 / max(20.0, speed_index)), 0.6, 2.5)
    incident_factor = _clamp(1.0 + (delay_pressure / 18.0) + (severity_index / 8.0), 0.7, 2.8)
    weather_factor = _clamp(weather_severity / 1.35, 0.7, 2.6)

    no_duration = _clamp(1.02 + (0.11 * traffic_factor) + (0.10 * incident_factor) + (0.06 * weather_factor), 1.02, 2.6)
    no_incident_rate = _clamp(1.02 + (0.16 * incident_factor) + (0.03 * traffic_factor), 1.02, 2.8)
    no_incident_delay = _clamp(1.02 + (0.14 * incident_factor) + (0.03 * weather_factor), 1.02, 2.8)
    no_fuel = _clamp(1.01 + (0.07 * traffic_factor) + (0.06 * weather_factor), 1.01, 2.2)
    no_emissions = _clamp(1.01 + (0.06 * traffic_factor) + (0.07 * weather_factor), 1.01, 2.2)
    no_sigma = _clamp(1.02 + (0.09 * traffic_factor) + (0.09 * incident_factor), 1.02, 2.8)

    def _partial(no_value: float) -> float:
        return _clamp(1.0 + (no_value - 1.0) * 0.52, 1.0, no_value - 0.02 if no_value > 1.04 else 1.0)

    def _full(partial_value: float) -> float:
        return _clamp(partial_value - 0.06, 0.86, 0.98)

    partial_duration = _partial(no_duration)
    partial_incident_rate = _partial(no_incident_rate)
    partial_incident_delay = _partial(no_incident_delay)
    partial_fuel = _partial(no_fuel)
    partial_emissions = _partial(no_emissions)
    partial_sigma = _partial(no_sigma)

    full_duration = _full(partial_duration)
    full_incident_rate = _full(partial_incident_rate)
    full_incident_delay = _full(partial_incident_delay)
    full_fuel = _full(partial_fuel)
    full_emissions = _full(partial_emissions)
    full_sigma = _full(partial_sigma)

    return {
        "no_sharing": {
            "duration_multiplier": round(no_duration, 6),
            "incident_rate_multiplier": round(no_incident_rate, 6),
            "incident_delay_multiplier": round(no_incident_delay, 6),
            "fuel_consumption_multiplier": round(no_fuel, 6),
            "emissions_multiplier": round(no_emissions, 6),
            "stochastic_sigma_multiplier": round(no_sigma, 6),
            "observation_source": MODE_OBSERVATION_SOURCE,
        },
        "partial_sharing": {
            "duration_multiplier": round(partial_duration, 6),
            "incident_rate_multiplier": round(partial_incident_rate, 6),
            "incident_delay_multiplier": round(partial_incident_delay, 6),
            "fuel_consumption_multiplier": round(partial_fuel, 6),
            "emissions_multiplier": round(partial_emissions, 6),
            "stochastic_sigma_multiplier": round(partial_sigma, 6),
            "observation_source": MODE_OBSERVATION_SOURCE,
        },
        "full_sharing": {
            "duration_multiplier": round(full_duration, 6),
            "incident_rate_multiplier": round(full_incident_rate, 6),
            "incident_delay_multiplier": round(full_incident_delay, 6),
            "fuel_consumption_multiplier": round(full_fuel, 6),
            "emissions_multiplier": round(full_emissions, 6),
            "stochastic_sigma_multiplier": round(full_sigma, 6),
            "observation_source": MODE_OBSERVATION_SOURCE,
        },
    }


def build(
    *,
    scenario_jsonl: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    if not scenario_jsonl.exists():
        raise FileNotFoundError(f"Scenario corpus is required: {scenario_jsonl}")
    rows_out: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for line in scenario_jsonl.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        corridor = _canonical_text(payload.get("corridor_bucket"), "uk_default")
        corridor_geohash5 = _corridor_geohash5(
            payload.get("corridor_geohash5"),
            corridor_bucket=corridor,
        )
        day_kind = _canonical_text(payload.get("day_kind"), "weekday")
        weather_bucket = _canonical_text(
            payload.get("weather_bucket", payload.get("weather_regime")),
            "clear",
        )
        vehicle_class = _canonical_text(payload.get("vehicle_class"), "rigid_hgv")
        road_mix_bucket = _canonical_text(payload.get("road_mix_bucket"), "mixed")
        hour_slot_local = max(0, min(23, int(_coerce_float(payload.get("hour_slot_local"), 12.0))))
        as_of_utc = str(payload.get("as_of_utc", "")).strip()
        key = "|".join(
            [
                corridor,
                day_kind,
                str(hour_slot_local),
                road_mix_bucket,
                vehicle_class,
                weather_bucket,
                as_of_utc,
            ]
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out = {
            "as_of_utc": as_of_utc,
            "corridor_bucket": corridor,
            "corridor_geohash5": corridor_geohash5,
            "hour_slot_local": hour_slot_local,
            "road_mix_bucket": road_mix_bucket,
            "road_mix_vector": payload.get("road_mix_vector", {road_mix_bucket: 1.0}),
            "vehicle_class": vehicle_class,
            "day_kind": day_kind,
            "weather_bucket": weather_bucket,
            "weather_regime": weather_bucket,
            "flow_index": payload.get("flow_index", (payload.get("traffic_features") or {}).get("flow_index")),
            "speed_index": payload.get("speed_index", (payload.get("traffic_features") or {}).get("speed_index")),
            "delay_pressure": payload.get(
                "delay_pressure",
                (payload.get("incident_features") or {}).get("delay_pressure"),
            ),
            "severity_index": payload.get(
                "severity_index",
                (payload.get("incident_features") or {}).get("severity_index"),
            ),
            "weather_severity_index": payload.get(
                "weather_severity_index",
                (payload.get("weather_features") or {}).get("weather_severity_index"),
            ),
            "traffic_pressure": payload.get("traffic_pressure"),
            "incident_pressure": payload.get("incident_pressure"),
            "weather_pressure": payload.get("weather_pressure"),
            "mode_observation_source": MODE_OBSERVATION_SOURCE,
            "mode_observation_dataset": MODE_OBSERVATION_DATASET,
            "mode_is_projected": False,
            "modes": _proxy_mode_values(payload),
        }
        rows_out.append(out)

    rows_out.sort(
        key=lambda row: (
            str(row.get("corridor_bucket", "")),
            str(row.get("day_kind", "")),
            int(row.get("hour_slot_local", 0)),
            str(row.get("road_mix_bucket", "")),
            str(row.get("vehicle_class", "")),
            str(row.get("weather_bucket", "")),
            str(row.get("as_of_utc", "")),
        )
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n" for row in rows_out),
        encoding="utf-8",
    )
    summary = {
        "rows": len(rows_out),
        "output_jsonl": str(output_jsonl),
        "mode_observation_source": MODE_OBSERVATION_SOURCE,
    }
    output_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate proxy non-bootstrap scenario mode outcomes from observed public context corpus."
    )
    parser.add_argument(
        "--scenario-jsonl",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "scenario_live_observed.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "uk" / "scenario_mode_outcomes_observed.jsonl",
    )
    args = parser.parse_args()
    summary = build(
        scenario_jsonl=args.scenario_jsonl,
        output_jsonl=args.output,
    )
    print(
        "Collected proxy scenario mode outcomes "
        f"(rows={summary['rows']}, output={summary['output_jsonl']})."
    )


if __name__ == "__main__":
    main()
