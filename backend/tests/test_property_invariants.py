from __future__ import annotations

import json
import random

import app.evidence_certification as evidence_certification_module
import pytest
from app.abstention import build_abstention_record
from app.audit_correction import build_proxy_audit_record, summarize_proxy_audit_records
from app.confidence_sequences import anytime_hoeffding_interval
from app.decision_critical import DCCSConfig, build_candidate_ledger
from app.models import GeoJSONLineString, RouteCertificationSummary, RouteMetrics, RouteOption, RouteResponse, Weights
from app.objectives_selection import normalise_weights, pick_best_by_weighted_sum
from app.pareto import dominates, pareto_filter
from app.preference_model import build_preference_state
from app.preference_queries import PairwisePreferenceQuery
from app.preference_update import append_preference_query, validate_preference_invariants
from app.provenance_store import provenance_event, write_provenance
from app.route_cache import (
    build_route_cache_key,
    build_route_cache_key_state,
    clear_route_cache,
    get_cached_routes,
    set_cached_routes,
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


def _make_route_response(
    *,
    selected: RouteOption,
    candidates: list[RouteOption],
    selected_certificate: RouteCertificationSummary | None,
    abstention=None,
    certified_set: list[RouteOption] | None = None,
    world_support_summary: dict[str, object] | None = None,
) -> RouteResponse:
    return RouteResponse(
        selected=selected,
        candidates=candidates,
        selected_certificate=selected_certificate,
        abstention=abstention,
        certified_set=[] if certified_set is None else certified_set,
        run_id="run-property-invariant",
        manifest_endpoint="/runs/run-property-invariant/manifest",
        artifacts_endpoint="/runs/run-property-invariant/artifacts",
        provenance_endpoint="/runs/run-property-invariant/provenance",
        world_support_summary=
            world_support_summary
            or {
                "schema_version": "world-support-summary-v1",
                "selected_route_id": selected.id,
                "selected_certificate_basis": "selected_certificate",
                "support_flag": bool(selected_certificate.certified) if selected_certificate is not None else True,
                "support_state": {
                    "support_flag": bool(selected_certificate.certified) if selected_certificate is not None else True,
                    "support_bin": (
                        "supported" if (selected_certificate is None or bool(selected_certificate.certified)) else "unsupported"
                    ),
                    "calibration_bin": "empirical",
                },
            },
    )


def _make_dccs_candidate(
    candidate_id: str,
    *,
    objective: tuple[float, float, float],
    road_mix: dict[str, float],
    toll_share: float,
    terrain_burden: float,
    straight_line_km: float,
    mechanism: dict[str, float],
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "graph_path": [f"{candidate_id}:a", f"{candidate_id}:b", f"{candidate_id}:c"],
        "graph_length_km": float(sum(objective) / 9.0),
        "straight_line_km": straight_line_km,
        "road_class_mix": road_mix,
        "toll_share": toll_share,
        "terrain_burden": terrain_burden,
        "proxy_objective": objective,
        "mechanism_descriptor": mechanism,
        "proxy_confidence": {"time": 0.9, "money": 0.85, "co2": 0.88},
    }


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


def test_exact_synthetic_pairwise_gap_states_are_certificate_consistent() -> None:
    corpus = [
        (winner_a, winner_b, winner_c)
        for winner_a in ("route_a", "route_b", "route_c")
        for winner_b in ("route_a", "route_b", "route_c")
        for winner_c in ("route_a", "route_b", "route_c")
    ]

    for selected_route_id in ("route_a", "route_b", "route_c"):
        for world_winners in corpus:
            route_scores = {
                route_id: [1.0 if winner == route_id else 0.0 for winner in world_winners]
                for route_id in ("route_a", "route_b", "route_c")
            }
            certificate = {
                route_id: sum(route_scores[route_id]) / float(len(world_winners))
                for route_id in ("route_a", "route_b", "route_c")
            }
            competitor_fragility_breakdown = {
                selected_route_id: {
                    challenger_id: {
                        "weather": challenger_index + 1,
                        "scenario": challenger_index,
                    }
                    for challenger_index, challenger_id in enumerate(
                        sorted(
                            route_id
                            for route_id in ("route_a", "route_b", "route_c")
                            if route_id != selected_route_id
                        ),
                        start=1,
                    )
                }
            }
            certificate_result = evidence_certification_module.CertificateResult(
                winner_id=max(
                    certificate.items(),
                    key=lambda item: (float(item[1]), str(item[0])),
                )[0],
                certificate=certificate,
                threshold=0.49,
                certified=bool(max(certificate.values()) >= 0.49),
                selected_route_id=selected_route_id,
                route_scores=route_scores,
                world_manifest={
                    "world_count": len(world_winners),
                    "unique_world_count": len(world_winners),
                    "support_flag": True,
                    "selected_certificate_basis": "empirical",
                },
                selector_config={"selector_weights": [1.0, 1.0, 1.0]},
            )
            pairwise_states = evidence_certification_module.build_pairwise_gap_states(
                certificate_result,
                selected_route_id=selected_route_id,
                competitor_fragility_breakdown=competitor_fragility_breakdown,
            )

            assert len(pairwise_states) == 2
            nearest_expected = min(
                (
                    (
                        round(
                            max(
                                0.0,
                                float(certificate[selected_route_id]) - float(certificate[challenger_id]),
                            ),
                            6,
                        ),
                        challenger_id,
                    )
                    for challenger_id in ("route_a", "route_b", "route_c")
                    if challenger_id != selected_route_id
                ),
                key=lambda item: (item[0], item[1]),
            )[1]
            nearest_rows = [state for state in pairwise_states if state.nearest_challenger]
            assert [state.challenger_id for state in nearest_rows] == [nearest_expected]

            for state in pairwise_states:
                expected_gap = round(
                    max(
                        0.0,
                        float(certificate[selected_route_id]) - float(certificate[state.challenger_id]),
                    ),
                    6,
                )
                expected_pressure = {
                    "weather": round(
                        float(
                            competitor_fragility_breakdown[selected_route_id][state.challenger_id]["weather"]
                        )
                        / float(len(world_winners)),
                        6,
                    ),
                    "scenario": round(
                        float(
                            competitor_fragility_breakdown[selected_route_id][state.challenger_id]["scenario"]
                        )
                        / float(len(world_winners)),
                        6,
                    ),
                }

                assert state.challenger_id != selected_route_id
                assert state.pairwise_gap_lower_bound == expected_gap
                assert state.pairwise_gap_upper_bound == expected_gap
                assert state.support_flag is True
                assert state.provenance["selected_route_id"] == selected_route_id
                assert state.provenance["selected_certificate"] == pytest.approx(
                    float(certificate[selected_route_id]),
                    rel=0.0,
                    abs=1e-12,
                )
                assert state.provenance["challenger_certificate"] == pytest.approx(
                    float(certificate[state.challenger_id]),
                    rel=0.0,
                    abs=1e-12,
                )
                assert state.provenance["challenger_family_pressure"] == expected_pressure
                assert state.provenance["dominant_evidence_family"] == "weather"
                assert state.challenger_audit_sensitivity == expected_pressure["weather"]
                if expected_gap > 0.0:
                    assert state.challenger_radius == expected_gap
                    assert state.flip_budget == expected_gap
                else:
                    assert state.challenger_radius is None
                    assert state.flip_budget is None


def test_legacy_weight_aliases_remain_backward_compatible_under_varied_payloads() -> None:
    rng = random.Random(20260412)

    for idx in range(24):
        time_weight = round(rng.uniform(0.1, 9.0), 6)
        money_weight = round(rng.uniform(0.1, 9.0), 6)
        co2_weight = round(rng.uniform(0.1, 9.0), 6)

        if idx % 3 == 0:
            payload = {"time": time_weight, "cost": money_weight, "emissions": co2_weight}
        elif idx % 3 == 1:
            payload = {"time": time_weight, "monetary_cost": money_weight, "co2e": co2_weight}
        else:
            payload = {"time": time_weight, "money": money_weight, "emissions_kg": co2_weight}

        parsed = Weights.model_validate(payload)
        assert parsed.time == pytest.approx(time_weight, rel=0.0, abs=1e-12)
        assert parsed.money == pytest.approx(money_weight, rel=0.0, abs=1e-12)
        assert parsed.co2 == pytest.approx(co2_weight, rel=0.0, abs=1e-12)


def test_legacy_route_response_payloads_remain_backward_compatible() -> None:
    rng = random.Random(20260413)

    for idx in range(18):
        selected = _make_option(
            f"legacy_route_{idx}",
            duration_s=rng.uniform(90.0, 180.0),
            money=rng.uniform(12.0, 40.0),
            co2=rng.uniform(2.0, 10.0),
        )
        challenger = _make_option(
            f"legacy_route_{idx}_alt",
            duration_s=rng.uniform(90.0, 180.0),
            money=rng.uniform(12.0, 40.0),
            co2=rng.uniform(2.0, 10.0),
        )
        selected_certificate = RouteCertificationSummary(
            route_id=selected.id,
            certificate=0.9 if idx % 3 == 0 else 0.72,
            certified=idx % 3 == 0,
            threshold=0.8,
            active_families=["scenario"] if idx % 2 == 0 else [],
            top_fragility_families=[] if idx % 3 != 2 else ["scenario"],
        )
        if idx % 3 == 0:
            response = _make_route_response(
                selected=selected,
                candidates=[selected, challenger],
                selected_certificate=selected_certificate,
            )
            expected_support_flag = True
        elif idx % 3 == 1:
            response = _make_route_response(
                selected=selected,
                candidates=[selected, challenger],
                selected_certificate=selected_certificate,
                certified_set=[selected, challenger],
            )
            expected_support_flag = True
        else:
            response = _make_route_response(
                selected=selected,
                candidates=[selected, challenger],
                selected_certificate=selected_certificate,
                abstention=build_abstention_record(
                    stop_reason="search_incomplete_no_action_worth_it",
                    support_flag=False,
                    support_reason="out_of_support_world_model",
                    credible_search_uncertainty=True,
                    active_families=[],
                    top_fragility_families=[],
                    detail={"legacy_case": idx},
                ),
            )
            expected_support_flag = False

        payload = response.model_dump(mode="json")
        payload["support_summary"] = {"supported": expected_support_flag}
        payload["manifest_endpoint"] = f"/runs/legacy-{idx}/manifest"
        payload["artifacts_endpoint"] = f"/runs/legacy-{idx}/artifacts"
        payload["provenance_endpoint"] = f"/runs/legacy-{idx}/provenance"
        payload.pop("artifact_pointers", None)
        payload.pop("frontier_summary", None)
        payload.pop("certified_set_summary", None)
        payload.pop("abstention_summary", None)
        payload.pop("witness_summary", None)

        validated = RouteResponse.model_validate(payload)

        assert validated.support_summary["support_flag"] is expected_support_flag
        assert validated.support_summary["supported"] is expected_support_flag
        assert validated.artifact_pointers == {
            "manifest_endpoint": f"/runs/legacy-{idx}/manifest",
            "artifacts_endpoint": f"/runs/legacy-{idx}/artifacts",
            "provenance_endpoint": f"/runs/legacy-{idx}/provenance",
        }
        assert validated.frontier_summary["selected_route_id"] == selected.id
        assert validated.certified_set_summary["witness"]["route_id"] == selected.id


def test_route_cache_key_is_stable_under_equivalent_input_forms() -> None:
    rng = random.Random(20260414)

    for idx in range(24):
        extra_items = [("alpha", f"value-{idx}"), ("beta", str(idx % 5))]
        rng.shuffle(extra_items)
        extra_left = dict(extra_items)
        rng.shuffle(extra_items)
        extra_right = dict(extra_items)
        support_flag = idx % 2 == 0

        key_left = build_route_cache_key(
            artifact_kind=" route_artifact ",
            run_id=f" run-{idx} ",
            lane_id=" lane-main ",
            variant_id=" default ",
            cache_mode=" cold ",
            support_flag="supported" if support_flag else "unsupported",
            support_status="In Support" if support_flag else "out-of-support",
            fidelity_class="Proxy Only",
            terminal_type="Typed Abstention",
            seed=idx,
            extra=extra_left,
        )
        key_right = build_route_cache_key_state(
            artifact_kind="route_artifact",
            run_id=f"run-{idx}",
            lane_id="lane-main",
            variant_id="default",
            cache_mode="cold",
            support_flag=support_flag,
            support_status="in/support" if support_flag else "out of support",
            fidelity_class="proxy-only",
            terminal_type="typed abstention",
            seed=idx,
            extra=extra_right,
        ).cache_key()
        changed_key = build_route_cache_key(
            artifact_kind="route_artifact",
            run_id=f"run-{idx}",
            lane_id="lane-main",
            variant_id="default",
            cache_mode="cold",
            support_flag=support_flag,
            support_status="in_support" if support_flag else "out_of_support",
            fidelity_class="proxy_only",
            terminal_type="typed_abstention",
            seed=idx + 1,
            extra=extra_right,
        )

        assert key_left == key_right
        assert changed_key != key_left


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


def test_route_response_terminal_consistency_and_artifact_pointer_invariants() -> None:
    rng = random.Random(20260409)

    for idx in range(24):
        selected = _make_option(
            f"route_{idx}",
            duration_s=rng.uniform(90.0, 190.0),
            money=rng.uniform(10.0, 45.0),
            co2=rng.uniform(2.0, 14.0),
        )
        challenger = _make_option(
            f"route_{idx}_alt",
            duration_s=rng.uniform(90.0, 190.0),
            money=rng.uniform(10.0, 45.0),
            co2=rng.uniform(2.0, 14.0),
        )
        selected_certificate = RouteCertificationSummary(
            route_id=selected.id,
            certificate=0.9 if idx % 3 == 0 else 0.74,
            certified=idx % 3 == 0,
            threshold=0.8,
            active_families=["scenario", "toll"] if idx % 4 else [],
            top_fragility_families=[],
        )
        mode = idx % 3

        if mode == 0:
            response = _make_route_response(
                selected=selected,
                candidates=[selected, challenger],
                selected_certificate=selected_certificate,
            )
            assert response.terminal_type == "certified_singleton"
            assert [route.id for route in response.certified_set] == [selected.id]
            assert response.recommended_route is selected
            assert response.support_summary["supported"] is True
            assert response.world_support_summary["schema_version"] == "world-support-summary-v1"
            assert response.world_support_summary["selected_route_id"] == selected.id
            assert response.world_support_summary["selected_certificate_basis"] == "selected_certificate"
            assert response.certified_set_summary["member_route_ids"] == []
            assert response.certified_set_summary["certified"] is False
            assert response.certified_set_summary["set_size"] == 0
            assert response.certified_set_summary["not_applicable_reason"] == "singleton_terminal"
        elif mode == 1:
            response = _make_route_response(
                selected=selected,
                candidates=[selected, challenger],
                selected_certificate=selected_certificate,
                certified_set=[selected, challenger],
            )
            assert response.terminal_type == "certified_set"
            assert [route.id for route in response.certified_set] == [selected.id, challenger.id]
            assert response.certified_set_summary["member_route_ids"] == [selected.id, challenger.id]
            assert response.certified_set_summary["excluded_route_ids"] == []
            assert response.certified_set_summary["certified"] is True
            assert response.certified_set_summary["set_size"] == 2
            assert response.certified_set_summary["not_applicable_reason"] is None
            assert response.world_support_summary["schema_version"] == "world-support-summary-v1"
            assert response.world_support_summary["selected_route_id"] == selected.id
            assert response.world_support_summary["selected_certificate_basis"] == "selected_certificate"
        else:
            abstention = build_abstention_record(
                stop_reason="search_incomplete_no_action_worth_it",
                support_flag=False,
                support_reason="out_of_support_world_model",
                credible_search_uncertainty=True,
                active_families=[],
                top_fragility_families=[],
                detail={"case": idx},
            )
            response = _make_route_response(
                selected=selected,
                candidates=[selected, challenger],
                selected_certificate=selected_certificate,
                abstention=abstention,
                certified_set=[selected, challenger],
            )
            assert response.terminal_type == "typed_abstention"
            assert response.certified_set == []
            assert response.abstention is not None
            assert response.abstention.reason_code == "uncertified_due_to_out_of_support_world_model"
            assert response.certified_set_summary["member_route_ids"] == []
            assert response.certified_set_summary["excluded_route_ids"] == [selected.id, challenger.id]
            assert response.certified_set_summary["certified"] is False
            assert response.certified_set_summary["set_size"] == 0
            assert response.certified_set_summary["not_applicable_reason"] == "abstention_terminal"
            assert response.world_support_summary["schema_version"] == "world-support-summary-v1"
            assert response.world_support_summary["selected_route_id"] == selected.id
            assert response.world_support_summary["selected_certificate_basis"] == "selected_certificate"
            assert response.world_support_summary["support_flag"] is False

        assert response.artifact_pointers == {
            "manifest_endpoint": "/runs/run-property-invariant/manifest",
            "artifacts_endpoint": "/runs/run-property-invariant/artifacts",
            "provenance_endpoint": "/runs/run-property-invariant/provenance",
        }
        assert response.frontier_summary["candidate_count"] == 2
        assert response.frontier_summary["selected_route_id"] == selected.id
        assert response.certified_set_summary["witness"]["route_id"] == selected.id


def test_anytime_hoeffding_interval_is_bounded_and_contains_empirical_mean() -> None:
    rng = random.Random(20260410)

    for _ in range(40):
        sample_count = rng.randint(1, 80)
        success_count = rng.randint(0, sample_count)
        lower_bound, upper_bound = anytime_hoeffding_interval(success_count, sample_count, delta=0.05)
        empirical = success_count / float(sample_count)

        assert 0.0 <= lower_bound <= empirical <= upper_bound <= 1.0


def test_anytime_hoeffding_interval_tightens_with_more_samples_at_fixed_rate() -> None:
    counts = [(2, 3), (4, 6), (8, 12), (16, 24), (32, 48)]
    widths = []

    for success_count, sample_count in counts:
        lower_bound, upper_bound = anytime_hoeffding_interval(success_count, sample_count, delta=0.05)
        widths.append(upper_bound - lower_bound)

    assert widths == sorted(widths, reverse=True)


def test_anytime_hoeffding_threshold_crossing_is_monotone_once_crossed() -> None:
    counts = [(2, 3), (20, 30), (40, 60), (80, 120), (200, 300)]
    threshold = 0.49
    widths = []
    lower_bounds = []

    for success_count, sample_count in counts:
        lower_bound, upper_bound = anytime_hoeffding_interval(success_count, sample_count, delta=0.05)
        widths.append(upper_bound - lower_bound)
        lower_bounds.append(lower_bound)

    assert widths == sorted(widths, reverse=True)
    assert lower_bounds == sorted(lower_bounds)
    crossing_index = next(
        (index for index, lower_bound in enumerate(lower_bounds) if lower_bound >= threshold),
        None,
    )
    assert crossing_index is not None
    assert all(lower_bound < threshold for lower_bound in lower_bounds[:crossing_index])
    assert all(lower_bound >= threshold for lower_bound in lower_bounds[crossing_index:])


def test_exact_synthetic_preference_robustness_invariants_hold_after_valid_query_updates() -> None:
    cases = [
        (["route_a", "route_b"], 1, 0.4),
        (["route_a", "route_b", "route_c"], 2, 0.6),
        (["route_a", "route_b", "route_c", "route_d"], 3, 0.75),
    ]

    for route_ids, after_size, after_volume_proxy in cases:
        state = build_preference_state(
            route_ids=route_ids,
            weights={"time": 2.0, "money": 1.0, "co2": 0.5},
        )
        updated = append_preference_query(
            state,
            PairwisePreferenceQuery(
                preferred_route_id=route_ids[0],
                challenger_route_id=route_ids[1],
            ),
            before_size=len(route_ids),
            after_size=after_size,
            before_volume_proxy=1.0,
            after_volume_proxy=after_volume_proxy,
            target_route_id=route_ids[1],
            query_reason="exact synthetic robustness invariant",
            preference_irrelevance=after_size <= 1,
        )

        invariants = validate_preference_invariants(updated)
        assert invariants["necessary_best_prob_le_possible_best_prob"] is True
        assert invariants["no_necessary_best_without_possible_best"] is True
        assert invariants["compatible_set_volume_nonincreasing_after_query"] is True
        assert invariants["preference_contradiction_free"] is True
        assert set(updated.compatible_set_summary.necessary_best_route_ids).issubset(
            set(updated.compatible_set_summary.possible_best_route_ids)
        )
        if after_size <= 1:
            assert updated.compatible_set_summary.necessary_best_route_ids == [route_ids[0]]
            assert updated.compatible_set_summary.possible_best_route_ids == [route_ids[0]]
        assert updated.query_count == 1
        assert updated.compatible_set_summary.compatible_set_size == after_size
        assert updated.compatible_set_summary.compatible_set_volume_proxy == pytest.approx(
            after_volume_proxy,
            rel=0.0,
            abs=1e-12,
        )


def test_exact_synthetic_preference_ambiguity_persists_without_enough_elicitation() -> None:
    for route_ids in (
        ["route_a", "route_b"],
        ["route_a", "route_b", "route_c"],
        ["route_a", "route_b", "route_c", "route_d"],
    ):
        state = build_preference_state(
            route_ids=route_ids,
            weights={"time": 2.0, "money": 1.0, "co2": 0.5},
        )

        assert state.query_count == 0
        assert state.preference_irrelevance_proven is False
        assert state.no_query_reason == "no_preference_query_issued"
        assert state.compatible_set_summary.compatible_set_size == len(route_ids)
        assert state.compatible_set_summary.necessary_best_prob == pytest.approx(0.0, rel=0.0, abs=1e-12)
        assert state.compatible_set_summary.possible_best_prob == pytest.approx(1.0, rel=0.0, abs=1e-12)

        updated = append_preference_query(
            state,
            PairwisePreferenceQuery(
                preferred_route_id=route_ids[0],
                challenger_route_id=route_ids[1],
            ),
            before_size=len(route_ids),
            after_size=len(route_ids),
            before_volume_proxy=1.0,
            after_volume_proxy=1.0,
            target_route_id=route_ids[1],
            query_reason="insufficient elicitation leaves ambiguity unresolved",
            preference_irrelevance=False,
        )

        invariants = validate_preference_invariants(updated)
        assert invariants["necessary_best_prob_le_possible_best_prob"] is True
        assert invariants["no_necessary_best_without_possible_best"] is True
        assert invariants["compatible_set_volume_nonincreasing_after_query"] is True
        assert updated.compatible_set_summary.necessary_best_route_ids == []
        assert updated.compatible_set_summary.possible_best_route_ids == route_ids
        assert updated.query_count == 1
        assert updated.preference_irrelevance_proven is False
        assert updated.compatible_set_summary.compatible_set_size == len(route_ids)
        assert updated.compatible_set_summary.necessary_best_prob == pytest.approx(0.0, rel=0.0, abs=1e-12)
        assert updated.compatible_set_summary.possible_best_prob == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_uncertified_bounds_preserve_typed_abstention_terminal_safety() -> None:
    selected = _make_option("route_cert", duration_s=120.0, money=16.0, co2=4.0)
    challenger = _make_option("route_alt", duration_s=122.0, money=15.5, co2=4.1)
    selected_certificate = RouteCertificationSummary(
        route_id=selected.id,
        certificate=0.92,
        certified=False,
        threshold=0.95,
        active_families=["scenario"],
        top_fragility_families=["scenario"],
    )
    abstention = build_abstention_record(
        stop_reason="certificate_below_threshold_no_action_worth_it",
        support_flag=True,
        support_reason=None,
        credible_search_uncertainty=False,
        active_families=["scenario"],
        top_fragility_families=["scenario"],
        detail={
            "winner_confidence_state": {
                "empirical_win": 0.92,
                "lower_bound": 0.74,
                "upper_bound": 0.98,
                "threshold": 0.95,
            }
        },
    )

    response = _make_route_response(
        selected=selected,
        candidates=[selected, challenger],
        selected_certificate=selected_certificate,
        abstention=abstention,
    )

    assert response.terminal_type == "typed_abstention"
    assert response.certified_set == []
    assert response.abstention is not None
    assert response.abstention_summary["terminal_type"] == "typed_abstention"
    assert response.abstention_summary["detail"]["winner_confidence_state"]["lower_bound"] < selected_certificate.threshold


def test_certified_set_state_requires_exclusion_and_no_singleton_justification() -> None:
    certified_set_state = evidence_certification_module.build_certified_set_state(
        evidence_certification_module.CertificateResult(
            winner_id="route_a",
            certificate={"route_a": 0.5, "route_b": 0.5, "route_c": 0.0},
            threshold=0.49,
            certified=True,
            selected_route_id="route_a",
            route_scores={"route_a": [1.0, 0.0], "route_b": [0.0, 1.0], "route_c": [0.0, 0.0]},
            world_manifest={
                "world_count": 2,
                "unique_world_count": 2,
                "support_flag": True,
                "selected_certificate_basis": "empirical",
            },
            selector_config={"selector_weights": [1.0, 1.0, 1.0]},
        ),
        frontier_route_ids=["route_a", "route_b"],
        selected_route_id="route_a",
    )

    assert certified_set_state.certified is True
    assert certified_set_state.set_size >= 2
    assert set(certified_set_state.member_route_ids).isdisjoint(certified_set_state.excluded_route_ids)
    assert certified_set_state.witness["outside_routes_excluded"] is True
    assert certified_set_state.witness["singleton_justified"] is False
    assert certified_set_state.witness["singleton_not_justified_reasons"]


def test_exact_synthetic_corpus_certified_set_safety_invariant() -> None:
    corpus = [
        (winner_a, winner_b)
        for winner_a in ("route_a", "route_b", "route_c")
        for winner_b in ("route_a", "route_b", "route_c")
    ]

    for world_winners in corpus:
        route_scores = {
            route_id: [1.0 if winner == route_id else 0.0 for winner in world_winners]
            for route_id in ("route_a", "route_b", "route_c")
        }
        certificate = {
            route_id: sum(route_scores[route_id]) / float(len(world_winners))
            for route_id in ("route_a", "route_b", "route_c")
        }
        winner_id = min(certificate.items(), key=lambda item: (-float(item[1]), str(item[0])))[0]
        certified_set_state = evidence_certification_module.build_certified_set_state(
            evidence_certification_module.CertificateResult(
                winner_id=winner_id,
                certificate=certificate,
                threshold=0.49,
                certified=bool(certificate[winner_id] >= 0.49),
                selected_route_id="route_a",
                route_scores=route_scores,
                world_manifest={
                    "world_count": len(world_winners),
                    "unique_world_count": len(world_winners),
                    "support_flag": True,
                    "selected_certificate_basis": "empirical",
                },
                selector_config={"selector_weights": [1.0, 1.0, 1.0]},
            ),
            frontier_route_ids=["route_a", "route_b"],
            selected_route_id="route_a",
        )

        if certified_set_state.certified:
            assert certified_set_state.set_size >= 2
            assert set(certified_set_state.member_route_ids).isdisjoint(certified_set_state.excluded_route_ids)
            assert certified_set_state.witness["outside_routes_safely_excluded"] is True
            assert certified_set_state.witness["singleton_justified"] is False
            assert certified_set_state.witness["singleton_not_justified_reasons"]
            assert certificate["route_c"] < certificate["route_a"]
        elif certificate["route_c"] > 0.0 and certificate["route_c"] >= certificate["route_a"]:
            assert certified_set_state.witness["outside_routes_safely_excluded"] is False
            assert certified_set_state.witness["excluded_route_safety_reasons"]


def test_exact_synthetic_batch_certified_set_validity_invariant() -> None:
    corpus = [
        (winner_a, winner_b, winner_c)
        for winner_a in ("route_a", "route_b", "route_c")
        for winner_b in ("route_a", "route_b", "route_c")
        for winner_c in ("route_a", "route_b", "route_c")
    ]

    for world_winners in corpus:
        route_scores = {
            route_id: [1.0 if winner == route_id else 0.0 for winner in world_winners]
            for route_id in ("route_a", "route_b", "route_c")
        }
        certificate = {
            route_id: sum(route_scores[route_id]) / float(len(world_winners))
            for route_id in ("route_a", "route_b", "route_c")
        }
        winner_id = min(certificate.items(), key=lambda item: (-float(item[1]), str(item[0])))[0]
        certified_set_state = evidence_certification_module.build_certified_set_state(
            evidence_certification_module.CertificateResult(
                winner_id=winner_id,
                certificate=certificate,
                threshold=0.49,
                certified=bool(certificate[winner_id] >= 0.49),
                selected_route_id="route_a",
                route_scores=route_scores,
                world_manifest={
                    "world_count": len(world_winners),
                    "unique_world_count": len(world_winners),
                    "support_flag": True,
                    "selected_certificate_basis": "empirical",
                },
                selector_config={"selector_weights": [1.0, 1.0, 1.0]},
            ),
            frontier_route_ids=["route_a", "route_b"],
            selected_route_id="route_a",
        )

        if certified_set_state.certified:
            assert certified_set_state.set_size >= 2
            assert set(certified_set_state.member_route_ids).isdisjoint(certified_set_state.excluded_route_ids)
            assert certified_set_state.witness["outside_routes_safely_excluded"] is True
            assert certified_set_state.witness["excluded_route_safety_reasons"] == []
            assert certified_set_state.witness["singleton_justified"] is False
            assert certified_set_state.witness["singleton_not_justified_reasons"]
            assert certificate["route_c"] < certificate["route_a"]
        elif certificate["route_c"] > 0.0 and certificate["route_c"] >= certificate["route_a"]:
            assert certified_set_state.witness["outside_routes_safely_excluded"] is False
            assert certified_set_state.witness["excluded_route_safety_reasons"]


def test_exact_synthetic_batch_safe_elimination_has_zero_false_safe_prunes() -> None:
    frontier_anchor = _make_dccs_candidate(
        "frontier_anchor",
        objective=(10.0, 10.0, 10.0),
        road_mix={"motorway_share": 0.5, "a_road_share": 0.3, "urban_share": 0.2},
        toll_share=0.05,
        terrain_burden=0.10,
        straight_line_km=10.0,
        mechanism={"motorway_share": 0.5, "toll_share": 0.05, "terrain_burden": 0.10},
    )
    dominated_candidates = [
        _make_dccs_candidate(
            f"cand_dominated_{index}",
            objective=(10.0 + duration_gap, 10.0 + money_gap, 10.0 + co2_gap),
            road_mix={"motorway_share": 0.45, "a_road_share": 0.35, "urban_share": 0.2},
            toll_share=0.05 + (0.01 * index),
            terrain_burden=0.10 + (0.01 * index),
            straight_line_km=9.8,
            mechanism={
                "motorway_share": 0.45,
                "toll_share": 0.05 + (0.01 * index),
                "terrain_burden": 0.10 + (0.01 * index),
            },
        )
        for index, (duration_gap, money_gap, co2_gap) in enumerate(
            [
                (0.25, 0.20, 0.30),
                (0.50, 0.40, 0.60),
                (1.00, 0.80, 0.90),
                (1.50, 1.20, 1.10),
            ],
            start=1,
        )
    ]
    live_challenger = _make_dccs_candidate(
        "cand_live",
        objective=(9.8, 10.8, 10.6),
        road_mix={"motorway_share": 0.35, "a_road_share": 0.45, "urban_share": 0.2},
        toll_share=0.04,
        terrain_burden=0.09,
        straight_line_km=9.9,
        mechanism={"motorway_share": 0.35, "toll_share": 0.04, "terrain_burden": 0.09},
    )

    ledger = build_candidate_ledger(
        [*dominated_candidates, live_challenger],
        frontier=[frontier_anchor],
        config=DCCSConfig(mode="challenger", search_budget=5),
    )
    records = {record.candidate_id: record for record in ledger}

    for candidate in dominated_candidates:
        record = records[str(candidate["candidate_id"])]
        assert record.safe_eliminated is True
        assert record.necessary_dominated is True
        assert record.dominated_by_route_id == "frontier_anchor"
        assert record.safe_prune_consistent is True
        assert all(
            float(candidate_value) >= float(frontier_value)
            for candidate_value, frontier_value in zip(
                record.proxy_objective,
                frontier_anchor["proxy_objective"],
                strict=False,
            )
        )

    assert records["cand_live"].safe_eliminated is False
    safe_pruned_count = sum(1 for record in ledger if record.safe_eliminated)
    false_safe_prune_count = sum(
        1 for record in ledger if record.safe_eliminated and not record.safe_prune_consistent
    )

    assert safe_pruned_count == len(dominated_candidates)
    assert false_safe_prune_count == 0
    assert false_safe_prune_count / float(max(1, safe_pruned_count)) == 0.0


def test_exact_synthetic_batch_flip_radius_invariant() -> None:
    corpus = [
        (winner_a, winner_b, winner_c)
        for winner_a in ("route_a", "route_b", "route_c")
        for winner_b in ("route_a", "route_b", "route_c")
        for winner_c in ("route_a", "route_b", "route_c")
    ]

    for world_winners in corpus:
        route_scores = {
            route_id: [1.0 if winner == route_id else 0.0 for winner in world_winners]
            for route_id in ("route_a", "route_b", "route_c")
        }
        certificate = {
            route_id: sum(route_scores[route_id]) / float(len(world_winners))
            for route_id in ("route_a", "route_b", "route_c")
        }
        winner_id = min(certificate.items(), key=lambda item: (-float(item[1]), str(item[0])))[0]
        certificate_result = evidence_certification_module.CertificateResult(
            winner_id=winner_id,
            certificate=certificate,
            threshold=0.49,
            certified=bool(certificate[winner_id] >= 0.49),
            selected_route_id="route_a",
            route_scores=route_scores,
            world_manifest={
                "world_count": len(world_winners),
                "unique_world_count": len(world_winners),
                "support_flag": True,
                "selected_certificate_basis": "empirical",
            },
            selector_config={"selector_weights": [1.0, 1.0, 1.0]},
        )
        pairwise_states = evidence_certification_module.build_pairwise_gap_states(
            certificate_result,
            selected_route_id="route_a",
        )
        flip_state = evidence_certification_module.build_flip_radius_state(
            certificate_result,
            None,
            selected_route_id="route_a",
            pairwise_states=pairwise_states,
        )

        selected_certificate = float(certificate["route_a"])
        challenger_certificates = [float(certificate["route_b"]), float(certificate["route_c"])]
        if any(challenger >= selected_certificate for challenger in challenger_certificates):
            assert flip_state.minimum_flip_budget is None
        else:
            expected_budget = round(
                min(selected_certificate - challenger for challenger in challenger_certificates),
                6,
            )
            assert flip_state.minimum_flip_budget == expected_budget


def test_exact_synthetic_batch_flip_radius_with_fragility_invariant() -> None:
    corpus = [
        (winner_a, winner_b, winner_c)
        for winner_a in ("route_a", "route_b", "route_c")
        for winner_b in ("route_a", "route_b", "route_c")
        for winner_c in ("route_a", "route_b", "route_c")
    ]
    fragility = evidence_certification_module.FragilityResult(
        route_fragility_map={"route_a": {"scenario": 0.125, "weather": 0.2}},
        competitor_fragility_breakdown={},
        value_of_refresh={
            "top_refresh_family": "weather",
            "top_refresh_family_controller": "weather",
        },
    )

    for world_winners in corpus:
        route_scores = {
            route_id: [1.0 if winner == route_id else 0.0 for winner in world_winners]
            for route_id in ("route_a", "route_b", "route_c")
        }
        certificate = {
            route_id: sum(route_scores[route_id]) / float(len(world_winners))
            for route_id in ("route_a", "route_b", "route_c")
        }
        winner_id = min(certificate.items(), key=lambda item: (-float(item[1]), str(item[0])))[0]
        certificate_result = evidence_certification_module.CertificateResult(
            winner_id=winner_id,
            certificate=certificate,
            threshold=0.49,
            certified=bool(certificate[winner_id] >= 0.49),
            selected_route_id="route_a",
            route_scores=route_scores,
            world_manifest={
                "world_count": len(world_winners),
                "unique_world_count": len(world_winners),
                "support_flag": True,
                "selected_certificate_basis": "empirical",
            },
            selector_config={"selector_weights": [1.0, 1.0, 1.0]},
        )
        pairwise_states = evidence_certification_module.build_pairwise_gap_states(
            certificate_result,
            selected_route_id="route_a",
        )
        flip_state = evidence_certification_module.build_flip_radius_state(
            certificate_result,
            fragility,
            selected_route_id="route_a",
            pairwise_states=pairwise_states,
        )

        selected_certificate = float(certificate["route_a"])
        challenger_certificates = [float(certificate["route_b"]), float(certificate["route_c"])]
        if any(challenger >= selected_certificate for challenger in challenger_certificates):
            assert flip_state.minimum_flip_budget is None
        else:
            expected_budget = round(
                min(selected_certificate - challenger for challenger in challenger_certificates),
                6,
            )
            expected_budget = min(expected_budget, 0.125, 0.2)
            assert flip_state.minimum_flip_budget == expected_budget
            assert flip_state.dominant_fragility_family == "weather"


def test_exact_synthetic_batch_decision_region_with_fragility_invariant() -> None:
    corpus = [
        (winner_a, winner_b, winner_c)
        for winner_a in ("route_a", "route_b", "route_c")
        for winner_b in ("route_a", "route_b", "route_c")
        for winner_c in ("route_a", "route_b", "route_c")
    ]
    fragility = evidence_certification_module.FragilityResult(
        route_fragility_map={"route_a": {"scenario": 0.125, "weather": 0.2}},
        competitor_fragility_breakdown={},
        value_of_refresh={
            "top_refresh_family": "weather",
            "top_refresh_family_controller": "weather",
        },
    )

    for world_winners in corpus:
        route_scores = {
            route_id: [1.0 if winner == route_id else 0.0 for winner in world_winners]
            for route_id in ("route_a", "route_b", "route_c")
        }
        certificate = {
            route_id: sum(route_scores[route_id]) / float(len(world_winners))
            for route_id in ("route_a", "route_b", "route_c")
        }
        winner_id = min(certificate.items(), key=lambda item: (-float(item[1]), str(item[0])))[0]
        certificate_result = evidence_certification_module.CertificateResult(
            winner_id=winner_id,
            certificate=certificate,
            threshold=0.49,
            certified=bool(certificate[winner_id] >= 0.49),
            selected_route_id="route_a",
            route_scores=route_scores,
            world_manifest={
                "world_count": len(world_winners),
                "unique_world_count": len(world_winners),
                "support_flag": True,
                "selected_certificate_basis": "empirical",
            },
            selector_config={"selector_weights": [1.0, 1.0, 1.0]},
        )
        pairwise_states = evidence_certification_module.build_pairwise_gap_states(
            certificate_result,
            selected_route_id="route_a",
        )
        flip_state = evidence_certification_module.build_flip_radius_state(
            certificate_result,
            fragility,
            selected_route_id="route_a",
            pairwise_states=pairwise_states,
        )
        decision_state = evidence_certification_module.build_decision_region_state(
            certificate_result,
            fragility,
            selected_route_id="route_a",
            pairwise_states=pairwise_states,
            flip_radius_state=flip_state,
        )

        selected_certificate = float(certificate["route_a"])
        challenger_certificates = [float(certificate["route_b"]), float(certificate["route_c"])]
        if any(challenger >= selected_certificate for challenger in challenger_certificates):
            assert decision_state.nearest_certificate_boundary == "pairwise_gap"
            assert decision_state.nearest_threat_axis == "search"
            assert decision_state.minimum_joint_perturbation == pytest.approx(0.0, rel=0.0, abs=1e-12)
        else:
            assert decision_state.nearest_certificate_boundary == "flip_radius"
            assert decision_state.nearest_threat_axis == "evidence"
            assert decision_state.minimum_joint_perturbation == pytest.approx(0.125, rel=0.0, abs=1e-12)
            assert decision_state.dominant_evidence_family == "weather"


def test_default_proxy_audit_metadata_preserves_leakage_guards_under_varied_inputs() -> None:
    rng = random.Random(20260415)

    for idx in range(24):
        proxy_value = round(rng.uniform(0.0, 200.0), 6)
        audited_value = round(proxy_value + rng.uniform(-25.0, 25.0), 6)
        record = build_proxy_audit_record(
            row_id=f"row-{idx}",
            route_id=f"route-{idx}",
            evidence_family="weather" if idx % 2 == 0 else "scenario",
            proxy_value=proxy_value,
            audited_value=audited_value,
            audit_probability=rng.uniform(-0.5, 1.5),
            propensity_score=rng.uniform(-0.5, 1.5),
            provenance={"seed": idx},
        )
        payload = record.as_dict()
        summary = summarize_proxy_audit_records(
            [record],
            proxy_world_count=1 + (idx % 4),
        )

        assert payload["correction_metadata"]["cross_fitted"] is True
        assert payload["correction_metadata"]["out_of_fold_only"] is True
        assert payload["correction_metadata"]["same_row_fit_prohibited"] is True
        assert payload["propensity_metadata"]["cross_fitted"] is True
        assert payload["propensity_metadata"]["out_of_fold_only"] is True
        assert payload["propensity_metadata"]["same_row_fit_prohibited"] is True
        assert 0.0 <= float(payload["audit_probability"]) <= 1.0
        assert 0.0 <= float(payload["propensity_score"]) <= 1.0
        assert summary.proxy_bias_model_version == record.correction_metadata.model_version
        assert summary.audit_propensity_version == record.propensity_metadata.model_version
        assert summary.positivity_diagnostics.audited_route_pair_count == 1
