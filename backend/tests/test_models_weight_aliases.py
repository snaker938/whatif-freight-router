from app.models import BatchParetoRequest, ParetoRequest, RouteRequest, Weights


def test_route_request_accepts_cost_and_emissions_weight_aliases() -> None:
    req = RouteRequest.model_validate(
        {
            "origin": {"lat": 51.5074, "lon": -0.1278},
            "destination": {"lat": 52.4862, "lon": -1.8904},
            "weights": {"time": 0.5, "cost": 0.3, "emissions": 0.2},
        }
    )
    assert req.weights.time == 0.5
    assert req.weights.money == 0.3
    assert req.weights.co2 == 0.2


def test_explicit_weights_fields_take_precedence_over_aliases() -> None:
    weights = Weights.model_validate(
        {
            "time": 0.5,
            "money": 0.7,
            "cost": 0.3,
            "co2": 0.4,
            "emissions": 0.2,
        }
    )
    assert weights.money == 0.7
    assert weights.co2 == 0.4


def test_pareto_request_accepts_cost_and_emissions_aliases() -> None:
    req = ParetoRequest.model_validate(
        {
            "origin": {"lat": 51.5074, "lon": -0.1278},
            "destination": {"lat": 52.4862, "lon": -1.8904},
            "weights": {"time": 0.2, "cost": 0.4, "emissions": 0.4},
        }
    )
    assert req.weights.time == 0.2
    assert req.weights.money == 0.4
    assert req.weights.co2 == 0.4


def test_route_request_accepts_and_serializes_ambiguity_context_fields() -> None:
    req = RouteRequest.model_validate(
        {
            "origin": {"lat": 51.5074, "lon": -0.1278},
            "destination": {"lat": 52.4862, "lon": -1.8904},
            "od_ambiguity_index": 0.62,
            "od_engine_disagreement_prior": 0.35,
            "od_hard_case_prior": 0.71,
            "od_candidate_path_count": 5,
            "od_corridor_family_count": 3,
            "od_objective_spread": 0.41,
            "od_nominal_margin_proxy": 0.22,
            "od_toll_disagreement_rate": 0.18,
            "ambiguity_budget_prior": 0.66,
            "ambiguity_budget_band": "high",
        }
    )

    payload = req.model_dump(mode="json")
    assert payload["od_ambiguity_index"] == 0.62
    assert payload["od_engine_disagreement_prior"] == 0.35
    assert payload["od_hard_case_prior"] == 0.71
    assert payload["od_candidate_path_count"] == 5
    assert payload["od_corridor_family_count"] == 3
    assert payload["od_objective_spread"] == 0.41
    assert payload["od_nominal_margin_proxy"] == 0.22
    assert payload["od_toll_disagreement_rate"] == 0.18
    assert payload["ambiguity_budget_prior"] == 0.66
    assert payload["ambiguity_budget_band"] == "high"


def test_batch_pareto_request_accepts_top_level_ambiguity_context_fields() -> None:
    req = BatchParetoRequest.model_validate(
        {
            "pairs": [
                {
                    "origin": {"lat": 51.5074, "lon": -0.1278},
                    "destination": {"lat": 52.4862, "lon": -1.8904},
                }
            ],
            "od_ambiguity_index": 0.4,
            "od_engine_disagreement_prior": 0.2,
            "ambiguity_budget_band": "medium",
        }
    )

    payload = req.model_dump(mode="json")
    assert payload["od_ambiguity_index"] == 0.4
    assert payload["od_engine_disagreement_prior"] == 0.2
    assert payload["ambiguity_budget_band"] == "medium"
