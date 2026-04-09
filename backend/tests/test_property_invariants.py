from __future__ import annotations

import json
import random

from app.models import (
    DecisionLaneManifest,
    DecisionPackage,
    DecisionTheoremHookRecord,
    DecisionTheoremHookSummary,
    GeoJSONLineString,
    RouteMetrics,
    RouteOption,
)
from app.objectives_selection import normalise_weights, pick_best_by_weighted_sum
from app.pareto import dominates, pareto_filter
from app.provenance_store import provenance_event, write_provenance
from app.replay_oracle import (
    action_replay_record_from_payload,
    build_action_replay_record,
    build_action_value_estimate,
    build_replay_oracle_summary,
)
from app.route_cache import clear_route_cache, get_cached_routes, set_cached_routes
from app.run_store import (
    is_safe_artifact_name,
    schema_version_for_artifact,
    schema_version_for_surface,
    versioned_json_payload,
)
from app.settings import settings
from app.signatures import sign_payload, verify_payload_signature


def _make_option(option_id: str, *, duration_s: float, money: float, co2: float) -> RouteOption:
    return RouteOption(
        id=option_id,
        geometry=GeoJSONLineString(type="LineString", coordinates=[(-1.0, 52.0), (-0.1, 51.5)]),
        metrics=RouteMetrics(
            distance_km=10.0,
            duration_s=duration_s,
            monetary_cost=money,
            emissions_kg=co2,
            avg_speed_kmh=40.0,
        ),
    )


def _theorem_hook_summary(*artifact_names: str) -> DecisionTheoremHookSummary:
    return DecisionTheoremHookSummary(
        hooks=[
            DecisionTheoremHookRecord(
                hook_id=f"hook_{index}",
                artifact_name=artifact_name,
                status="present",
            )
            for index, artifact_name in enumerate(artifact_names, start=1)
        ]
    )


def test_pareto_randomized_invariants() -> None:
    rng = random.Random(20260212)

    for _ in range(30):
        items = []
        for idx in range(25):
            vec = (
                round(rng.uniform(1.0, 100.0) + (idx * 1e-6), 6),
                round(rng.uniform(1.0, 100.0) + (idx * 1e-6), 6),
                round(rng.uniform(1.0, 100.0) + (idx * 1e-6), 6),
            )
            assert not dominates(vec, vec)
            items.append({"id": idx, "vec": vec})

        kept = pareto_filter(items, key=lambda item: item["vec"])
        kept_ids = {item["id"] for item in kept}

        for i, item_i in enumerate(kept):
            for j, item_j in enumerate(kept):
                if i == j:
                    continue
                assert not dominates(item_i["vec"], item_j["vec"])

        for item in items:
            if item["id"] in kept_ids:
                continue
            assert any(dominates(k["vec"], item["vec"]) for k in kept)


def test_weighted_selection_randomized_invariants() -> None:
    rng = random.Random(42)

    for _ in range(40):
        options = [
            _make_option(
                f"route_{idx}",
                duration_s=rng.uniform(1000.0, 6000.0),
                money=rng.uniform(100.0, 800.0),
                co2=rng.uniform(20.0, 200.0),
            )
            for idx in range(6)
        ]
        w_time = rng.uniform(0.0, 10.0)
        w_money = rng.uniform(0.0, 10.0)
        w_co2 = rng.uniform(0.0, 10.0)

        selected = pick_best_by_weighted_sum(
            options,
            w_time=w_time,
            w_money=w_money,
            w_co2=w_co2,
        )
        scaled = pick_best_by_weighted_sum(
            options,
            w_time=w_time * 7.0,
            w_money=w_money * 7.0,
            w_co2=w_co2 * 7.0,
        )
        assert selected.id == scaled.id

        wt, wm, we = normalise_weights(w_time, w_money, w_co2)
        assert abs((wt + wm + we) - 1.0) < 1e-9


def test_signature_and_cache_invariants_under_varied_inputs() -> None:
    rng = random.Random(99)
    clear_route_cache()

    for idx in range(12):
        payload = {
            "index": idx,
            "value": rng.randint(1, 1000),
            "nested": {"flag": idx % 2 == 0},
        }
        secret = f"secret-{idx}"
        signature = sign_payload(payload, secret=secret)
        valid, expected = verify_payload_signature(payload, signature, secret=secret)
        assert valid is True
        assert expected == signature

        tampered = dict(payload)
        tampered["value"] = int(payload["value"]) + 1
        valid_tampered, _ = verify_payload_signature(tampered, signature, secret=secret)
        assert valid_tampered is False

        key = f"cache_key_{idx}"
        value = (
            [{"distance": 1000.0 + idx, "duration": 100.0 + idx}],
            [f"warn_{idx}"],
            3,
        )
        set_cached_routes(key, value)
        assert get_cached_routes(key) == value

    assert get_cached_routes("missing_key") is None


def test_provenance_event_order_is_preserved(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path / "out"))

    run_id = "11111111-1111-1111-1111-111111111111"
    events = [
        provenance_event(run_id, "input_received", pair_count=2),
        provenance_event(run_id, "candidates_fetched", candidate_count=4),
        provenance_event(run_id, "options_built", option_count=4),
        provenance_event(run_id, "pareto_selected", pareto_count=2),
        provenance_event(run_id, "artifacts_written", artifact_count=5),
    ]
    path = write_provenance(run_id, events)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.exists()
    assert payload["event_count"] == 5
    ordered_events = [event["event"] for event in payload["events"]]
    assert ordered_events == [
        "input_received",
        "candidates_fetched",
        "options_built",
        "pareto_selected",
        "artifacts_written",
    ]


def test_dominates_raises_on_dimension_mismatch() -> None:
    try:
        dominates((1.0, 2.0), (1.0, 2.0, 3.0))
    except ValueError as exc:
        assert "dimension mismatch" in str(exc)
    else:
        raise AssertionError("dominates should reject mismatched dimensions")


def test_dominates_treats_equal_vectors_as_nondominating() -> None:
    assert dominates((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) is False


def test_dominates_respects_tolerance_band() -> None:
    assert dominates((1.0, 2.0, 3.0), (1.0 + 5e-10, 2.0, 3.0), tol=1e-9) is False


def test_dominates_detects_single_axis_improvement() -> None:
    assert dominates((1.0, 2.0, 3.0), (1.1, 2.0, 3.0)) is True


def test_pareto_filter_keeps_nondominated_extremes() -> None:
    items = [
        {"id": "time", "vec": (1.0, 5.0, 5.0)},
        {"id": "money", "vec": (5.0, 1.0, 5.0)},
        {"id": "co2", "vec": (5.0, 5.0, 1.0)},
        {"id": "dominated", "vec": (6.0, 6.0, 6.0)},
    ]

    kept = pareto_filter(items, key=lambda item: item["vec"])

    assert [item["id"] for item in kept] == ["time", "money", "co2"]


def test_pareto_filter_removes_prior_dominated_rows_when_better_point_arrives() -> None:
    items = [
        {"id": "worse", "vec": (4.0, 4.0, 4.0)},
        {"id": "better", "vec": (3.0, 3.0, 3.0)},
        {"id": "tradeoff", "vec": (2.0, 5.0, 2.0)},
    ]

    kept = pareto_filter(items, key=lambda item: item["vec"])

    assert [item["id"] for item in kept] == ["better", "tradeoff"]


def test_pareto_filter_preserves_duplicate_vectors() -> None:
    items = [
        {"id": "a", "vec": (1.0, 2.0, 3.0)},
        {"id": "b", "vec": (1.0, 2.0, 3.0)},
    ]

    kept = pareto_filter(items, key=lambda item: item["vec"])

    assert [item["id"] for item in kept] == ["a", "b"]


def test_normalise_weights_returns_uniform_for_zero_sum() -> None:
    assert normalise_weights(0.0, 0.0, 0.0) == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def test_normalise_weights_returns_uniform_for_negative_sum() -> None:
    assert normalise_weights(-1.0, 0.0, 0.0) == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def test_normalise_weights_scaling_is_invariant() -> None:
    base = normalise_weights(2.0, 3.0, 5.0)
    scaled = normalise_weights(20.0, 30.0, 50.0)

    assert base == scaled


def test_pick_best_by_weighted_sum_prefers_min_time() -> None:
    options = [
        _make_option("route_fast", duration_s=100.0, money=400.0, co2=90.0),
        _make_option("route_slow", duration_s=140.0, money=100.0, co2=20.0),
    ]

    selected = pick_best_by_weighted_sum(options, w_time=1.0, w_money=0.0, w_co2=0.0)

    assert selected.id == "route_fast"


def test_pick_best_by_weighted_sum_prefers_min_money() -> None:
    options = [
        _make_option("route_expensive", duration_s=100.0, money=400.0, co2=20.0),
        _make_option("route_cheap", duration_s=140.0, money=100.0, co2=90.0),
    ]

    selected = pick_best_by_weighted_sum(options, w_time=0.0, w_money=1.0, w_co2=0.0)

    assert selected.id == "route_cheap"


def test_pick_best_by_weighted_sum_prefers_min_co2() -> None:
    options = [
        _make_option("route_dirty", duration_s=100.0, money=100.0, co2=90.0),
        _make_option("route_clean", duration_s=140.0, money=400.0, co2=20.0),
    ]

    selected = pick_best_by_weighted_sum(options, w_time=0.0, w_money=0.0, w_co2=1.0)

    assert selected.id == "route_clean"


def test_pick_best_by_weighted_sum_breaks_exact_tie_by_route_id() -> None:
    options = [
        _make_option("route_b", duration_s=100.0, money=100.0, co2=100.0),
        _make_option("route_a", duration_s=100.0, money=100.0, co2=100.0),
    ]

    selected = pick_best_by_weighted_sum(options, w_time=1.0, w_money=1.0, w_co2=1.0)

    assert selected.id == "route_a"


def test_sign_payload_is_stable_under_key_reordering() -> None:
    payload_a = {"a": 1, "b": {"x": 2, "y": 3}}
    payload_b = {"b": {"y": 3, "x": 2}, "a": 1}

    assert sign_payload(payload_a, secret="stable") == sign_payload(payload_b, secret="stable")


def test_verify_payload_signature_detects_nested_payload_change() -> None:
    payload = {"a": 1, "b": {"x": 2}}
    signature = sign_payload(payload, secret="stable")

    valid, _expected = verify_payload_signature({"a": 1, "b": {"x": 3}}, signature, secret="stable")

    assert valid is False


def test_route_cache_clear_removes_prior_entry() -> None:
    clear_route_cache()
    set_cached_routes("key_a", ([{"distance": 1.0}], [], 1))

    clear_route_cache()

    assert get_cached_routes("key_a") is None


def test_route_cache_distinguishes_different_keys() -> None:
    clear_route_cache()
    set_cached_routes("key_a", ([{"distance": 1.0}], ["a"], 1))
    set_cached_routes("key_b", ([{"distance": 2.0}], ["b"], 2))

    assert get_cached_routes("key_a") == ([{"distance": 1.0}], ["a"], 1)
    assert get_cached_routes("key_b") == ([{"distance": 2.0}], ["b"], 2)


def test_route_cache_preserves_extended_payload_shape() -> None:
    clear_route_cache()
    value = ([{"distance": 3.0}], ["warn"], 2, {"cache_scope": "property"})
    set_cached_routes("extended", value)

    assert get_cached_routes("extended") == value


def test_provenance_event_includes_supplied_context() -> None:
    event = provenance_event("run-1", "custom_event", pair_count=2, lane="property")

    assert event["run_id"] == "run-1"
    assert event["event"] == "custom_event"
    assert event["pair_count"] == 2
    assert event["lane"] == "property"


def test_write_provenance_counts_empty_event_list(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "out_dir", str(tmp_path / "out"))

    path = write_provenance("empty-run", [])
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["run_id"] == "empty-run"
    assert payload["event_count"] == 0
    assert payload["events"] == []


def test_schema_version_for_surface_covers_theorem_hook_map() -> None:
    assert schema_version_for_surface("theorem_hook_map") == "0.1.0"


def test_schema_version_for_surface_covers_lane_manifest() -> None:
    assert schema_version_for_surface("lane_manifest") == "0.1.0"


def test_schema_version_for_surface_covers_route_evidence_and_evaluator_surfaces() -> None:
    expected_surfaces = (
        "dccs_summary",
        "winner_summary",
        "certificate_summary",
        "route_fragility_map",
        "sampled_world_manifest",
        "evidence_snapshot_manifest",
        "voi_controller_state",
        "thesis_metrics",
        "thesis_plots",
        "evaluation_manifest",
    )

    for surface in expected_surfaces:
        assert schema_version_for_surface(surface) == "0.1.0"


def test_schema_version_for_artifact_covers_theorem_hook_map() -> None:
    assert schema_version_for_artifact("theorem_hook_map.json") == "0.1.0"


def test_schema_version_for_artifact_covers_lane_manifest() -> None:
    assert schema_version_for_artifact("lane_manifest.json") == "0.1.0"


def test_schema_version_for_artifact_covers_theorem_hook_targets_and_route_exports() -> None:
    expected_artifacts = (
        "winner_summary.json",
        "certificate_summary.json",
        "dccs_summary.json",
        "strict_frontier.jsonl",
        "sampled_world_manifest.json",
        "evidence_snapshot_manifest.json",
        "voi_controller_state.jsonl",
        "thesis_metrics.json",
        "thesis_plots.json",
        "evaluation_manifest.json",
    )

    for artifact_name in expected_artifacts:
        assert schema_version_for_artifact(artifact_name) == "0.1.0"


def test_versioned_json_payload_adds_theorem_hook_schema_version() -> None:
    payload = versioned_json_payload({"hooks": []}, artifact_name="theorem_hook_map.json")

    assert payload["schema_version"] == "0.1.0"


def test_versioned_json_payload_adds_schema_version_to_theorem_hook_target_artifact() -> None:
    payload = versioned_json_payload({"winner_route_id": "route_a"}, artifact_name="winner_summary.json")

    assert payload["schema_version"] == "0.1.0"


def test_versioned_json_payload_preserves_existing_schema_version() -> None:
    payload = versioned_json_payload(
        {"schema_version": "9.9.9", "hooks": []},
        artifact_name="theorem_hook_map.json",
    )

    assert payload["schema_version"] == "9.9.9"


def test_is_safe_artifact_name_accepts_theorem_hook_map() -> None:
    assert is_safe_artifact_name("theorem_hook_map.json") is True


def test_is_safe_artifact_name_rejects_path_traversal() -> None:
    assert is_safe_artifact_name("../theorem_hook_map.json") is False


def test_decision_package_theorem_hook_summary_round_trips() -> None:
    package = DecisionPackage(
        selected_route_id="route_a",
        theorem_hook_summary=_theorem_hook_summary("theorem_hook_map.json", "lane_manifest.json"),
        lane_manifest=DecisionLaneManifest(
            lane_id="property_lane",
            artifact_names=["theorem_hook_map.json", "lane_manifest.json"],
        ),
    )

    payload = package.model_dump(mode="json")

    assert payload["theorem_hook_summary"]["hooks"][0]["artifact_name"] == "theorem_hook_map.json"
    assert payload["theorem_hook_summary"]["hooks"][1]["artifact_name"] == "lane_manifest.json"
    assert payload["lane_manifest"]["artifact_names"] == ["theorem_hook_map.json", "lane_manifest.json"]


def test_action_replay_record_round_trips_payload() -> None:
    estimate = build_action_value_estimate(
        action_id="refresh_weather",
        action_kind="refresh",
        action_target="weather",
        cost_evidence=1,
        predicted_delta_certificate=0.12,
        lambda_certificate=1.0,
    )
    record = build_action_replay_record(
        estimate,
        trace_entry={
            "realization": {
                "realized_certificate_delta": 0.08,
                "realized_productive": True,
            }
        },
        trace_metadata={"trace_source": "property"},
    )

    parsed = action_replay_record_from_payload(record.as_dict())

    assert parsed is not None
    assert parsed.estimate.action_id == "refresh_weather"
    assert parsed.realization is not None
    assert parsed.realization.realized_productive is True
    assert parsed.trace_metadata["trace_source"] == "property"


def test_build_replay_oracle_summary_preserves_schema_version() -> None:
    estimate = build_action_value_estimate(
        action_id="refresh_weather",
        action_kind="refresh",
        action_target="weather",
        cost_evidence=1,
        predicted_delta_certificate=0.12,
        lambda_certificate=1.0,
    )
    record = build_action_replay_record(
        estimate,
        trace_entry={"realization": {"realized_certificate_delta": 0.04}},
    )

    summary = build_replay_oracle_summary([record], trace_source="property")

    assert summary.schema_version == "0.1.0"
    assert summary.trace_source == "property"
