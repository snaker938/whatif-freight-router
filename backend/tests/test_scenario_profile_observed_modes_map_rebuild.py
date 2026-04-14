from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.build_scenario_profiles_uk as build_scenario_profiles_uk


CORRIDOR_KEYS = [f"uk{idx:03d}" for idx in range(8)]
HOUR_SLOTS = [0, 4, 8, 12, 16, 20]
WEATHER_BY_HOUR = {0: "clear", 4: "clear", 8: "rain", 12: "clear", 16: "rain", 20: "clear"}
OBSERVED_SOURCE = "empirical_outcome_public_feeds_v1"
PROJECTED_SOURCE = "runtime_profile_projection_v1"


def _context_specs() -> list[tuple[int, str, int, int]]:
    specs: list[tuple[int, str, int, int]] = []
    for corridor_index, corridor in enumerate(CORRIDOR_KEYS):
        for hour_index, hour in enumerate(HOUR_SLOTS):
            specs.append((corridor_index, corridor, hour_index, hour))
    return specs


def _mode_triplet(base: float) -> tuple[tuple[str, float], ...]:
    return (
        ("no_sharing", base),
        ("partial_sharing", base - 0.14),
        ("full_sharing", base - 0.28),
    )


def _projected_mode_rows(*, as_of_utc: str, base_multiplier: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for corridor_index, corridor, hour_index, hour in _context_specs():
        for mode_name, multiplier in _mode_triplet(base_multiplier + (0.01 * corridor_index)):
            rows.append(
                {
                    "corridor_bucket": corridor,
                    "corridor_geohash5": corridor,
                    "hour_slot_local": hour,
                    "road_mix_bucket": "mixed",
                    "road_mix_vector": {"mixed": 1.0},
                    "vehicle_class": "rigid_hgv",
                    "day_kind": "weekday" if hour < 20 else "weekend",
                    "weather_bucket": WEATHER_BY_HOUR[hour],
                    "weather_regime": WEATHER_BY_HOUR[hour],
                    "as_of_utc": as_of_utc,
                    "flow_index": 0.0,
                    "speed_index": 0.0,
                    "dft_count_per_hour": 3200.0 + (40.0 * corridor_index) + (15.0 * hour_index),
                    "delay_pressure": 0.7 + (0.01 * hour_index),
                    "severity_index": 0.9,
                    "weather_severity_index": 0.7 if WEATHER_BY_HOUR[hour] == "clear" else 1.2,
                    "mode": mode_name,
                    "mode_observation_source": PROJECTED_SOURCE,
                    "duration_multiplier": multiplier,
                    "incident_rate_multiplier": multiplier,
                    "incident_delay_multiplier": multiplier,
                    "fuel_consumption_multiplier": multiplier,
                    "emissions_multiplier": multiplier,
                    "stochastic_sigma_multiplier": multiplier,
                }
            )
    return rows


def _observed_modes_map_rows(*, as_of_utc: str, base_multiplier: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for corridor_index, corridor, hour_index, hour in _context_specs():
        modes: dict[str, dict[str, float]] = {}
        for mode_name, multiplier in _mode_triplet(base_multiplier + (0.01 * corridor_index)):
            modes[mode_name] = {
                "duration_multiplier": multiplier,
                "incident_rate_multiplier": multiplier,
                "incident_delay_multiplier": multiplier,
                "fuel_consumption_multiplier": multiplier,
                "emissions_multiplier": multiplier,
                "stochastic_sigma_multiplier": multiplier,
            }
        rows.append(
            {
                "corridor_bucket": corridor,
                "corridor_geohash5": corridor,
                "hour_slot_local": hour,
                "road_mix_bucket": "mixed",
                "road_mix_vector": {"mixed": 1.0},
                "vehicle_class": "rigid_hgv",
                "day_kind": "weekday" if hour < 20 else "weekend",
                "weather_bucket": WEATHER_BY_HOUR[hour],
                "weather_regime": WEATHER_BY_HOUR[hour],
                "as_of_utc": as_of_utc,
                "flow_index": 0.0,
                "speed_index": 0.0,
                "dft_count_per_hour": 3300.0 + (40.0 * corridor_index) + (15.0 * hour_index),
                "delay_pressure": 0.8 + (0.01 * hour_index),
                "severity_index": 1.0,
                "weather_severity_index": 0.8 if WEATHER_BY_HOUR[hour] == "clear" else 1.3,
                "mode_observation_source": OBSERVED_SOURCE,
                "mode_observation_dataset": "public_context_observed_outcomes_v1",
                "modes": modes,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_scenario_profiles_counts_nested_modes_map_rows_toward_observed_share(
    tmp_path: Path,
) -> None:
    raw_jsonl = tmp_path / "scenario_live_observed.jsonl"
    observed_modes_jsonl = tmp_path / "scenario_mode_outcomes_observed.jsonl"
    output_json = tmp_path / "scenario_profiles_uk.json"

    raw_rows = _projected_mode_rows(as_of_utc="2026-02-22T23:47:40Z", base_multiplier=1.18)
    observed_rows = _observed_modes_map_rows(as_of_utc="2026-02-22T23:47:40Z", base_multiplier=1.24)
    _write_jsonl(raw_jsonl, raw_rows)
    _write_jsonl(observed_modes_jsonl, observed_rows)

    payload = build_scenario_profiles_uk.build(
        raw_jsonl=raw_jsonl,
        observed_modes_jsonl=observed_modes_jsonl,
        output_json=output_json,
        min_contexts=8,
        min_observed_mode_row_share=0.2,
        max_projection_dominant_context_share=0.8,
    )

    observed_expanded_count = len(observed_rows) * 3
    combined_count = len(raw_rows) + observed_expanded_count

    assert output_json.exists()
    assert payload["mode_outcomes_source"] == str(observed_modes_jsonl)
    assert payload["source_observation_window"]["row_count"] == combined_count
    assert payload["source_observation_window"]["observed_mode_row_count"] == observed_expanded_count
    assert payload["source_observation_filter"]["selected_row_count"] == combined_count
    assert payload["source_observation_filter"]["dropped_row_count"] == 0
    assert payload["holdout_metrics"]["observed_mode_row_share"] == pytest.approx(0.5)


def test_build_scenario_profiles_recent_window_drops_stale_nested_modes_map_rows_before_share_gate(
    tmp_path: Path,
) -> None:
    raw_jsonl = tmp_path / "scenario_live_observed.jsonl"
    observed_modes_jsonl = tmp_path / "scenario_mode_outcomes_observed.jsonl"
    output_json = tmp_path / "scenario_profiles_uk.json"

    raw_rows = _projected_mode_rows(as_of_utc="2026-02-22T23:47:40Z", base_multiplier=1.18)
    stale_observed_rows = _observed_modes_map_rows(as_of_utc="2026-02-01T00:00:00Z", base_multiplier=1.20)
    fresh_observed_rows = _observed_modes_map_rows(as_of_utc="2026-02-22T23:47:40Z", base_multiplier=1.24)
    _write_jsonl(raw_jsonl, raw_rows)
    _write_jsonl(observed_modes_jsonl, stale_observed_rows + fresh_observed_rows)

    payload = build_scenario_profiles_uk.build(
        raw_jsonl=raw_jsonl,
        observed_modes_jsonl=observed_modes_jsonl,
        output_json=output_json,
        min_contexts=8,
        min_observed_mode_row_share=0.2,
        max_projection_dominant_context_share=0.8,
        max_observation_window_minutes=2 * 24 * 60,
    )

    fresh_observed_expanded_count = len(fresh_observed_rows) * 3
    stale_observed_expanded_count = len(stale_observed_rows) * 3
    selected_count = len(raw_rows) + fresh_observed_expanded_count
    input_count = len(raw_rows) + fresh_observed_expanded_count + stale_observed_expanded_count

    assert output_json.exists()
    assert payload["source_observation_window"]["row_count"] == selected_count
    assert payload["source_observation_window"]["observed_mode_row_count"] == fresh_observed_expanded_count
    assert payload["source_observation_filter"]["input_row_count"] == input_count
    assert payload["source_observation_filter"]["selected_row_count"] == selected_count
    assert payload["source_observation_filter"]["dropped_row_count"] == stale_observed_expanded_count
    assert payload["holdout_metrics"]["observed_mode_row_share"] == pytest.approx(0.5)
