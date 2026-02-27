from decimal import Decimal

from app.routing_graph import _parse_edge, _parse_node


def test_parse_node_accepts_decimal_coordinates() -> None:
    parsed = _parse_node({"id": "n1", "lat": Decimal("52.5001"), "lon": Decimal("-1.9002")})
    assert parsed == ("n1", 52.5001, -1.9002)


def test_parse_edge_accepts_decimal_numeric_fields() -> None:
    parsed = _parse_edge(
        {
            "u": "n1",
            "v": "n2",
            "distance_m": Decimal("123.4"),
            "generalized_cost": Decimal("234.5"),
            "oneway": False,
            "highway": "primary",
            "toll": True,
            "maxspeed_kph": Decimal("80.0"),
        }
    )
    assert parsed is not None
    u, v, edge, oneway = parsed
    assert (u, v, oneway) == ("n1", "n2", False)
    assert edge.distance_m == 123.4
    assert edge.cost == 234.5
    assert edge.maxspeed_kph == 80.0
