from app.models import ParetoRequest, RouteRequest, Weights


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
