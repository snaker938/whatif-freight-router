from __future__ import annotations

from datetime import UTC, datetime

import app.toll_engine as toll_engine
from app.calibration_loader import (
    TollConfidenceBin,
    TollConfidenceCalibration,
    TollSegmentSeed,
    TollTariffRule,
    TollTariffTable,
)


def _calibration() -> TollConfidenceCalibration:
    return TollConfidenceCalibration(
        intercept=-0.9,
        class_signal_coef=1.5,
        seed_signal_coef=1.4,
        segment_signal_coef=0.5,
        bonus_both=0.05,
        bonus_class=0.02,
        bins=(
            TollConfidenceBin(minimum=0.0, maximum=0.5, calibrated=0.35),
            TollConfidenceBin(minimum=0.5, maximum=1.0, calibrated=0.82),
        ),
        source="unit",
        version="v1",
        as_of_utc="2026-02-23T00:00:00Z",
    )


def _seed() -> TollSegmentSeed:
    return TollSegmentSeed(
        segment_id="seg_a",
        name="Unit toll segment",
        operator="nh",
        road_class="motorway",
        crossing_id="dartford",
        direction="both",
        crossing_fee_gbp=2.5,
        distance_fee_gbp_per_km=0.10,
        coordinates=((51.5000, -0.1000), (51.5100, -0.0500)),
    )


def _route() -> dict:
    return {
        "geometry": {
            "type": "LineString",
            "coordinates": [[-0.1000, 51.5000], [-0.0500, 51.5100]],
        },
        "legs": [],
        "contains_toll": False,
    }


def test_pick_tariff_rule_prefers_specific_match(monkeypatch) -> None:
    specific = TollTariffRule(
        rule_id="specific",
        operator="nh",
        crossing_id="dartford",
        road_class="motorway",
        direction="both",
        start_minute=0,
        end_minute=1439,
        crossing_fee_gbp=2.5,
        distance_fee_gbp_per_km=0.2,
        vehicle_classes=("rigid_hgv",),
        axle_classes=("3to4",),
        payment_classes=("electronic",),
        exemptions=(),
    )
    fallback = TollTariffRule(
        rule_id="fallback",
        operator="default",
        crossing_id="default",
        road_class="default",
        direction="both",
        start_minute=0,
        end_minute=1439,
        crossing_fee_gbp=1.0,
        distance_fee_gbp_per_km=0.05,
        vehicle_classes=("default",),
        axle_classes=("default",),
        payment_classes=("default",),
        exemptions=(),
    )

    table = TollTariffTable(
        default_crossing_fee_gbp=0.0,
        default_distance_fee_gbp_per_km=0.0,
        rules=(fallback, specific),
        source="unit",
    )
    monkeypatch.setattr(toll_engine, "load_toll_tariffs", lambda: table)

    chosen = toll_engine._pick_tariff_rule(
        operator="nh",
        crossing_id="dartford",
        road_class="motorway",
        route_direction="eastbound",
        vehicle_class="rigid_hgv",
        axle_class="3to4",
        payment_class="electronic",
        minute=120,
    )
    assert chosen is not None
    assert chosen.rule_id == "specific"

    # table object included to ensure serialization shape is valid when reused in patches
    assert len(table.rules) == 2


def test_compute_toll_cost_priced_seed_path(monkeypatch) -> None:
    tariff = TollTariffRule(
        rule_id="rule_a",
        operator="nh",
        crossing_id="dartford",
        road_class="motorway",
        direction="both",
        start_minute=0,
        end_minute=1439,
        crossing_fee_gbp=2.5,
        distance_fee_gbp_per_km=0.2,
        vehicle_classes=("rigid_hgv",),
        axle_classes=("3to4",),
        payment_classes=("electronic",),
        exemptions=(),
    )
    table = TollTariffTable(
        default_crossing_fee_gbp=0.0,
        default_distance_fee_gbp_per_km=0.0,
        rules=(tariff,),
        source="unit",
    )

    monkeypatch.setattr(toll_engine, "load_toll_tariffs", lambda: table)
    monkeypatch.setattr(toll_engine, "load_toll_segments_seed", lambda: (_seed(),))
    monkeypatch.setattr(toll_engine, "load_toll_confidence_calibration", _calibration)
    monkeypatch.setattr(toll_engine, "_segment_overlap_km", lambda _seed_obj, _pts: 1.2)

    result = toll_engine.compute_toll_cost(
        route=_route(),
        distance_km=8.0,
        vehicle_type="rigid_hgv",
        departure_time_utc=datetime(2026, 2, 23, 9, 30, tzinfo=UTC),
        use_tolls=True,
    )

    assert result.contains_toll is True
    assert result.toll_distance_km > 0.0
    assert result.toll_cost_gbp > 0.0
    assert result.confidence > 0.0
    assert result.details["pricing_unresolved"] is False


def test_compute_toll_cost_unresolved_when_no_tariff_match(monkeypatch) -> None:
    table = TollTariffTable(
        default_crossing_fee_gbp=0.0,
        default_distance_fee_gbp_per_km=0.0,
        rules=(),
        source="unit",
    )
    monkeypatch.setattr(toll_engine, "load_toll_tariffs", lambda: table)
    monkeypatch.setattr(toll_engine, "load_toll_segments_seed", lambda: (_seed(),))
    monkeypatch.setattr(toll_engine, "load_toll_confidence_calibration", _calibration)
    monkeypatch.setattr(toll_engine, "_segment_overlap_km", lambda _seed_obj, _pts: 1.0)

    result = toll_engine.compute_toll_cost(
        route=_route(),
        distance_km=8.0,
        vehicle_type="rigid_hgv",
        departure_time_utc=datetime(2026, 2, 23, 9, 30, tzinfo=UTC),
        use_tolls=True,
    )

    assert result.source == "unpriced_toll"
    assert result.toll_cost_gbp == 0.0
    assert result.confidence == 0.0
    assert result.details["pricing_unresolved"] is True
