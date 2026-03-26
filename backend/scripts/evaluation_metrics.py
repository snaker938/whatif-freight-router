from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


OBJECTIVE_FIELDS: tuple[str, str, str] = ("duration_s", "monetary_cost", "emissions_kg")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(parsed)


# Linear sample-quantile interpolation follows Hyndman & Fan (1996):
# https://robjhyndman.com/papers/sample_quantiles.pdf
def percentile(values: Sequence[float], quantile: float) -> float | None:
    cleaned = [as_float(value, float("nan")) for value in values]
    cleaned = sorted(value for value in cleaned if math.isfinite(value))
    if not cleaned:
        return None
    q = min(max(as_float(quantile, 0.0), 0.0), 1.0)
    if len(cleaned) == 1:
        return round(cleaned[0], 6)
    position = q * (len(cleaned) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return round(cleaned[lower], 6)
    fraction = position - lower
    interpolated = cleaned[lower] + (cleaned[upper] - cleaned[lower]) * fraction
    return round(interpolated, 6)


def route_metrics(route: dict[str, Any]) -> dict[str, float]:
    metrics = route.get("metrics") or {}
    return {
        "distance_km": as_float(metrics.get("distance_km")),
        "duration_s": as_float(metrics.get("duration_s")),
        "monetary_cost": as_float(metrics.get("monetary_cost")),
        "emissions_kg": as_float(metrics.get("emissions_kg")),
    }


def dominates(lhs: dict[str, float], rhs: dict[str, float]) -> bool:
    return (
        lhs["duration_s"] <= rhs["duration_s"]
        and lhs["monetary_cost"] <= rhs["monetary_cost"]
        and lhs["emissions_kg"] <= rhs["emissions_kg"]
        and (
            lhs["duration_s"] < rhs["duration_s"]
            or lhs["monetary_cost"] < rhs["monetary_cost"]
            or lhs["emissions_kg"] < rhs["emissions_kg"]
        )
    )


def pairwise_weighted_sum_score(
    route_a: dict[str, float],
    route_b: dict[str, float],
    *,
    weights: tuple[float, float, float],
) -> tuple[float, float]:
    mins = {
        field: min(route_a[field], route_b[field])
        for field in OBJECTIVE_FIELDS
    }
    maxs = {
        field: max(route_a[field], route_b[field])
        for field in OBJECTIVE_FIELDS
    }

    def _score(route: dict[str, float]) -> float:
        total = 0.0
        for idx, field in enumerate(OBJECTIVE_FIELDS):
            span = maxs[field] - mins[field]
            norm = 0.0 if span <= 0.0 else (route[field] - mins[field]) / span
            total += as_float(weights[idx]) * norm
        return float(total)

    return _score(route_a), _score(route_b)


def balanced_gain_score(route: dict[str, float], baseline: dict[str, float]) -> float:
    improvements = []
    for field in OBJECTIVE_FIELDS:
        denom = max(abs(baseline[field]), 1e-9)
        improvements.append((baseline[field] - route[field]) / denom)
    clipped = [max(-1.0, min(1.0, value)) for value in improvements]
    return float(sum(clipped) / len(clipped))


def robust_win(route: dict[str, Any], baseline: dict[str, Any]) -> bool | None:
    route_unc = route.get("uncertainty")
    baseline_unc = baseline.get("uncertainty")
    if not isinstance(route_unc, dict) or not isinstance(baseline_unc, dict):
        return None
    route_cvar = as_float(route_unc.get("cvar95_duration_s"), float("nan"))
    base_cvar = as_float(baseline_unc.get("cvar95_duration_s"), float("nan"))
    if not math.isfinite(route_cvar) or not math.isfinite(base_cvar):
        return None
    return route_cvar < base_cvar


def frontier_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        route_id = str(row.get("route_id") or row.get("id") or "")
        if route_id:
            unique.setdefault(route_id, dict(row))
    ordered = [unique[key] for key in sorted(unique)]
    frontier: list[dict[str, Any]] = []
    for route in ordered:
        metrics = {
            "duration_s": as_float(route.get("duration_s")),
            "monetary_cost": as_float(route.get("monetary_cost")),
            "emissions_kg": as_float(route.get("emissions_kg")),
        }
        if any(
            dominates(
                {
                    "duration_s": as_float(other.get("duration_s")),
                    "monetary_cost": as_float(other.get("monetary_cost")),
                    "emissions_kg": as_float(other.get("emissions_kg")),
                },
                metrics,
            )
            for other in ordered
            if other is not route
        ):
            continue
        frontier.append(route)
    return frontier


def hypervolume_3d(frontier: list[dict[str, Any]], *, reference: dict[str, float]) -> float:
    points = [
        (
            min(as_float(row.get("duration_s")), reference["duration_s"]),
            min(as_float(row.get("monetary_cost")), reference["monetary_cost"]),
            min(as_float(row.get("emissions_kg")), reference["emissions_kg"]),
        )
        for row in frontier
    ]
    points = sorted(set(points))
    if not points:
        return 0.0
    xs = sorted({point[0] for point in points} | {reference["duration_s"]})
    volume = 0.0
    for idx in range(len(xs) - 1):
        x_left = xs[idx]
        x_right = xs[idx + 1]
        if x_right <= x_left:
            continue
        yz_points = [(y, z) for x, y, z in points if x <= x_left]
        if not yz_points:
            continue
        area = _area_2d(yz_points, ref_y=reference["monetary_cost"], ref_z=reference["emissions_kg"])
        volume += (x_right - x_left) * area
    return round(volume, 6)


def _area_2d(points: list[tuple[float, float]], *, ref_y: float, ref_z: float) -> float:
    ordered = sorted({(min(y, ref_y), min(z, ref_z)) for y, z in points})
    if not ordered:
        return 0.0
    ys = sorted({point[0] for point in ordered} | {ref_y})
    area = 0.0
    for idx in range(len(ys) - 1):
        y_left = ys[idx]
        y_right = ys[idx + 1]
        if y_right <= y_left:
            continue
        z_best = min((z for y, z in ordered if y <= y_left), default=ref_z)
        area += (y_right - y_left) * max(0.0, ref_z - z_best)
    return float(area)


def additive_epsilon_indicator(frontier: list[dict[str, Any]], baseline: dict[str, float]) -> float:
    if not frontier:
        return float("inf")
    epsilon = min(
        max(
            as_float(row.get("duration_s")) - baseline["duration_s"],
            as_float(row.get("monetary_cost")) - baseline["monetary_cost"],
            as_float(row.get("emissions_kg")) - baseline["emissions_kg"],
        )
        for row in frontier
    )
    return round(float(epsilon), 6)


def coverage_of_baseline(frontier: list[dict[str, Any]], baseline: dict[str, float]) -> float:
    return 1.0 if any(
        dominates(
            {
                "duration_s": as_float(row.get("duration_s")),
                "monetary_cost": as_float(row.get("monetary_cost")),
                "emissions_kg": as_float(row.get("emissions_kg")),
            },
            baseline,
        )
        for row in frontier
    ) else 0.0


def frontier_diversity(frontier: list[dict[str, Any]]) -> tuple[float, float]:
    if len(frontier) <= 1:
        return 0.0, 0.0
    vectors = [
        (
            as_float(row.get("duration_s")),
            as_float(row.get("monetary_cost")),
            as_float(row.get("emissions_kg")),
        )
        for row in frontier
    ]
    mins = [min(vec[idx] for vec in vectors) for idx in range(3)]
    maxs = [max(vec[idx] for vec in vectors) for idx in range(3)]
    scales = [max(1e-9, maxs[idx] - mins[idx]) for idx in range(3)]

    def _distance(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
        return math.sqrt(
            sum((((lhs[idx] - rhs[idx]) / scales[idx]) ** 2) for idx in range(3))
        )

    pairwise: list[float] = []
    nearest: list[float] = []
    for idx, vec in enumerate(vectors):
        distances = [_distance(vec, other) for jdx, other in enumerate(vectors) if jdx != idx]
        pairwise.extend(distances)
        nearest.append(min(distances))
    spread = sum(pairwise) / len(pairwise) if pairwise else 0.0
    crowding = sum(nearest) / len(nearest) if nearest else 0.0
    return round(spread, 6), round(crowding, 6)


def frontier_diversity_index(frontier: list[dict[str, Any]]) -> float | None:
    spread, crowding = frontier_diversity(frontier)
    total = spread + crowding
    if total <= 0.0:
        return 0.0 if frontier else None
    return round(spread / total, 6)


def frontier_entropy(frontier: list[dict[str, Any]]) -> float | None:
    ranked = normalized_weighted_ranking(frontier)
    if not ranked:
        return None
    strengths = [1.0 / (1.0 + max(as_float(row.get("score"), 0.0), 0.0)) for row in ranked]
    return normalized_entropy(strengths)


def pearson_binary_correlation(scores: list[float], labels: list[int]) -> float | None:
    if len(scores) != len(labels) or len(scores) < 2:
        return None
    mean_x = sum(scores) / len(scores)
    mean_y = sum(labels) / len(labels)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(scores, labels, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in scores)
    var_y = sum((y - mean_y) ** 2 for y in labels)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return round(cov / math.sqrt(var_x * var_y), 6)


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    pairs = [
        (as_float(x, float("nan")), as_float(y, float("nan")))
        for x, y in zip(xs, ys, strict=True)
    ]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    var_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    var_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return round(cov / math.sqrt(var_x * var_y), 6)


def _rank_sequence(values: Sequence[float]) -> list[float] | None:
    cleaned = [(as_float(value, float("nan")), idx) for idx, value in enumerate(values)]
    cleaned = [(value, idx) for value, idx in cleaned if math.isfinite(value)]
    if len(cleaned) < 2:
        return None
    ordered = sorted(cleaned, key=lambda item: (item[0], item[1]))
    ranks = [0.0] * len(ordered)
    position = 0
    while position < len(ordered):
        value = ordered[position][0]
        end = position + 1
        while end < len(ordered) and ordered[end][0] == value:
            end += 1
        # Average tied ranks using 1-based positions.
        average_rank = ((position + 1) + end) / 2.0
        for idx in range(position, end):
            ranks[idx] = average_rank
        position = end
    # Map back to the original order.
    restored = [0.0] * len(ordered)
    for ordered_idx, (_, original_idx) in enumerate(ordered):
        restored[original_idx] = ranks[ordered_idx]
    return restored


def rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    paired = [
        (as_float(x, float("nan")), as_float(y, float("nan")))
        for x, y in zip(xs, ys, strict=True)
    ]
    paired = [(x, y) for x, y in paired if math.isfinite(x) and math.isfinite(y)]
    if len(paired) < 2:
        return None
    x_ranks = _rank_sequence([x for x, _ in paired])
    y_ranks = _rank_sequence([y for _, y in paired])
    if x_ranks is None or y_ranks is None:
        return None
    return pearson_correlation(x_ranks, y_ranks)


def normalized_weighted_ranking(
    frontier_rows: Sequence[dict[str, Any]],
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in frontier_rows if isinstance(row, dict)]
    if not rows:
        return []
    mins = {field: min(as_float(row.get(field), 0.0) for row in rows) for field in OBJECTIVE_FIELDS}
    maxs = {field: max(as_float(row.get(field), 0.0) for row in rows) for field in OBJECTIVE_FIELDS}
    ranked: list[dict[str, Any]] = []
    for row in rows:
        score = 0.0
        for idx, field in enumerate(OBJECTIVE_FIELDS):
            low = mins[field]
            high = maxs[field]
            value = as_float(row.get(field), 0.0)
            if high > low:
                score += max(as_float(weights[idx], 0.0), 0.0) * ((value - low) / (high - low))
        ranked.append(
            {
                "route_id": str(row.get("route_id") or row.get("id") or ""),
                "score": round(score, 6),
                "duration_s": as_float(row.get("duration_s"), 0.0),
                "monetary_cost": as_float(row.get("monetary_cost"), 0.0),
                "emissions_kg": as_float(row.get("emissions_kg"), 0.0),
            }
        )
    return sorted(ranked, key=lambda item: (item["score"], item["route_id"]))


def nominal_winner_margin(
    frontier_rows: Sequence[dict[str, Any]],
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float | None, str | None]:
    ranked = normalized_weighted_ranking(frontier_rows, weights=weights)
    if not ranked:
        return None, None
    if len(ranked) == 1:
        return 1.0, ranked[0]["route_id"]
    margin = max(0.0, ranked[1]["score"] - ranked[0]["score"])
    return round(min(1.0, margin), 6), ranked[0]["route_id"]


def near_tie_mass(
    frontier_rows: Sequence[dict[str, Any]],
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tolerance: float = 0.05,
) -> float | None:
    ranked = normalized_weighted_ranking(frontier_rows, weights=weights)
    if not ranked:
        return None
    if len(ranked) == 1:
        return 0.0
    best = ranked[0]["score"]
    threshold = best + max(float(tolerance), 0.0)
    near_ties = sum(1 for row in ranked[1:] if row["score"] <= threshold)
    return round(near_ties / max(1, len(ranked) - 1), 6)


def certificate_margin(certificate: float | None, *, threshold: float) -> float | None:
    if certificate is None or not math.isfinite(float(certificate)):
        return None
    return round(float(certificate) - float(threshold), 6)


def certificate_runner_up_gap(
    certificate_by_route: dict[str, float],
    *,
    winner_id: str,
) -> float | None:
    winner = as_float(certificate_by_route.get(winner_id), float("nan"))
    if not math.isfinite(winner):
        return None
    competitors = [
        as_float(value, float("nan"))
        for route_id, value in certificate_by_route.items()
        if str(route_id) != str(winner_id)
    ]
    competitors = [value for value in competitors if math.isfinite(value)]
    if not competitors:
        return 1.0
    return round(winner - max(competitors), 6)


# Shannon entropy for discrete distributions: Shannon, 1948.
# https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
def normalized_entropy(weights: Sequence[float]) -> float | None:
    cleaned = [max(as_float(value, 0.0), 0.0) for value in weights]
    total = sum(cleaned)
    active = [value for value in cleaned if value > 0.0]
    if not cleaned:
        return None
    if total <= 0.0 or len(active) <= 1:
        return 0.0
    probs = [value / total for value in active]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0.0:
        return 0.0
    return round(entropy / max_entropy, 6)


def fragility_entropy(route_fragility: dict[str, float]) -> float | None:
    if not isinstance(route_fragility, dict):
        return None
    return normalized_entropy([as_float(value, 0.0) for value in route_fragility.values()])


def competitor_turnover_rate(competitor_breakdown: dict[str, dict[str, float]]) -> float | None:
    if not isinstance(competitor_breakdown, dict) or not competitor_breakdown:
        return None
    totals: list[float] = []
    for family_map in competitor_breakdown.values():
        if not isinstance(family_map, dict):
            continue
        totals.append(sum(max(as_float(value, 0.0), 0.0) for value in family_map.values()))
    totals = [value for value in totals if value > 0.0]
    if not totals:
        return 0.0
    return round(len(totals) / sum(totals), 6)


def score_ranked_recall(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    budget: int,
    positive_labels: set[str],
    score_key: str = "final_score",
) -> float | None:
    rows = [dict(row) for row in candidate_rows if isinstance(row, dict)]
    if not rows:
        return None
    positive = [
        row
        for row in rows
        if str(
            row.get("outcome_label")
            or row.get("decision_reason")
            or row.get("post_refine_label")
            or ""
        ) in positive_labels
    ]
    if not positive:
        return 0.0
    ordered = sorted(
        rows,
        key=lambda row: (
            -as_float(row.get(score_key), float("-inf")),
            str(row.get("candidate_id") or row.get("route_id") or ""),
        ),
    )
    chosen = ordered[: max(0, int(budget))]
    chosen_ids = {
        str(row.get("candidate_id") or row.get("route_id") or "")
        for row in chosen
    }
    positive_ids = {
        str(row.get("candidate_id") or row.get("route_id") or "")
        for row in positive
    }
    hits = len(chosen_ids & positive_ids)
    return round(hits / max(1, len(positive_ids)), 6)


def corridor_family_recall(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    budget: int,
    positive_labels: set[str],
    score_key: str = "final_score",
) -> float | None:
    rows = [dict(row) for row in candidate_rows if isinstance(row, dict)]
    if not rows:
        return None

    def _family(row: dict[str, Any]) -> str:
        for key in ("corridor_family", "family_id", "mechanism_family", "signature", "graph_signature"):
            value = str(row.get(key) or "").strip()
            if value:
                return value
        mechanism = row.get("mechanism_descriptor")
        if isinstance(mechanism, dict):
            for key in ("family", "cluster", "signature"):
                value = str(mechanism.get(key) or "").strip()
                if value:
                    return value
        return str(row.get("candidate_id") or row.get("route_id") or "")

    positive = [
        row
        for row in rows
        if str(
            row.get("outcome_label")
            or row.get("decision_reason")
            or row.get("post_refine_label")
            or ""
        ) in positive_labels
    ]
    if not positive:
        return 0.0
    ordered = sorted(
        rows,
        key=lambda row: (
            -as_float(row.get(score_key), float("-inf")),
            str(row.get("candidate_id") or row.get("route_id") or ""),
        ),
    )
    chosen = ordered[: max(0, int(budget))]
    chosen_families = {_family(row) for row in chosen if _family(row)}
    positive_families = {_family(row) for row in positive if _family(row)}
    if not positive_families:
        return 0.0
    return round(len(chosen_families & positive_families) / len(positive_families), 6)


def _valid_refine_cost_pairs(candidate_rows: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        predicted = as_float(row.get("predicted_refine_cost"), float("nan"))
        observed = as_float(row.get("observed_refine_cost"), float("nan"))
        if not math.isfinite(predicted) or not math.isfinite(observed):
            continue
        pairs.append((predicted, observed))
    return pairs


def refine_cost_sample_count(candidate_rows: Sequence[dict[str, Any]]) -> int:
    return len(_valid_refine_cost_pairs(candidate_rows))


def refine_cost_positive_sample_count(candidate_rows: Sequence[dict[str, Any]]) -> int:
    return sum(1 for _, observed in _valid_refine_cost_pairs(candidate_rows) if observed > 0.0)


def refine_cost_zero_observed_count(candidate_rows: Sequence[dict[str, Any]]) -> int:
    return sum(1 for _, observed in _valid_refine_cost_pairs(candidate_rows) if observed == 0.0)


def refine_cost_mape(candidate_rows: Sequence[dict[str, Any]]) -> float | None:
    errors: list[float] = []
    for predicted, observed in _valid_refine_cost_pairs(candidate_rows):
        if observed <= 0.0:
            continue
        errors.append(abs(predicted - observed) / abs(observed))
    if not errors:
        return None
    return round(sum(errors) / len(errors), 6)


def refine_cost_mae_ms(candidate_rows: Sequence[dict[str, Any]]) -> float | None:
    pairs = _valid_refine_cost_pairs(candidate_rows)
    if not pairs:
        return None
    errors: list[float] = []
    for predicted, observed in pairs:
        errors.append(abs(predicted - observed))
    if not errors:
        return None
    return round(sum(errors) / len(errors), 6)


def refine_cost_rank_correlation(candidate_rows: Sequence[dict[str, Any]]) -> float | None:
    pairs = _valid_refine_cost_pairs(candidate_rows)
    if not pairs:
        return None
    predicted: list[float] = []
    observed: list[float] = []
    for predicted_value, observed_value in pairs:
        predicted.append(predicted_value)
        observed.append(observed_value)
    return rank_correlation(predicted, observed)


def refine_cost_prediction_error(candidate_rows: Sequence[dict[str, Any]]) -> float | None:
    # Deprecated alias retained for compatibility with older summaries.
    return refine_cost_mape(candidate_rows)


def time_to_best_iteration(
    action_trace: Sequence[dict[str, Any]],
    *,
    selected_route_id: str | None = None,
) -> int | None:
    rows = [dict(row) for row in action_trace if isinstance(row, dict)]
    if not rows:
        return None
    target = str(selected_route_id or "").strip()
    if not target:
        return 0
    for row in rows:
        if str(row.get("selected_route_id") or "").strip() == target:
            return int(as_float(row.get("iteration"), 0.0))
    return len(rows)


def frontier_action_gain(
    *,
    frontier_count: float | None,
    frontier_diversity_index: float | None,
) -> float | None:
    signals: list[float] = []
    count = as_float(frontier_count, float("nan"))
    if math.isfinite(count):
        # Bounded thesis-specific frontier richness proxy; this is deliberately
        # normalized so downstream efficiency metrics remain unitless.
        signals.append(min(1.0, max(0.0, (count - 1.0) / 4.0)))
    diversity = as_float(frontier_diversity_index, float("nan"))
    if math.isfinite(diversity):
        signals.append(min(1.0, max(0.0, diversity)))
    if not signals:
        return None
    return round(sum(signals) / len(signals), 6)


def action_efficiency(
    *,
    certificate_lift: float | None,
    frontier_gain: float | None,
    action_count: int | None,
    search_budget_used: int | None,
    evidence_budget_used: int | None,
) -> float | None:
    if certificate_lift is None and frontier_gain is None:
        return None
    if int(action_count or 0) <= 0 and int(search_budget_used or 0) <= 0 and int(evidence_budget_used or 0) <= 0:
        return None
    gain_terms = []
    if certificate_lift is not None:
        gain_terms.append(min(1.0, max(as_float(certificate_lift, 0.0), 0.0)))
    if frontier_gain is not None:
        gain_terms.append(min(1.0, max(as_float(frontier_gain, 0.0), 0.0)))
    if not gain_terms:
        return None
    numerator = sum(gain_terms)
    denominator = max(1, int(action_count or 0) + int(search_budget_used or 0) + int(evidence_budget_used or 0))
    return round(numerator / denominator, 6)


def runtime_share(part_ms: float | None, total_ms: float | None) -> float | None:
    part = as_float(part_ms, float("nan"))
    total = as_float(total_ms, float("nan"))
    if not math.isfinite(part) or not math.isfinite(total) or total <= 0.0:
        return None
    return round(max(0.0, min(1.0, part / total)), 6)


def runtime_ratio(lhs_ms: float | None, rhs_ms: float | None) -> float | None:
    lhs = as_float(lhs_ms, float("nan"))
    rhs = as_float(rhs_ms, float("nan"))
    if not math.isfinite(lhs) or not math.isfinite(rhs) or rhs <= 0.0:
        return None
    return round(max(0.0, lhs / rhs), 6)


def runtime_per_unit(runtime_ms: float | None, unit_count: float | None) -> float | None:
    runtime = as_float(runtime_ms, float("nan"))
    units = as_float(unit_count, float("nan"))
    if not math.isfinite(runtime) or not math.isfinite(units) or units <= 0.0:
        return None
    return round(max(0.0, runtime / units), 6)


def value_per_second(value: float | None, runtime_ms: float | None) -> float | None:
    score = as_float(value, float("nan"))
    runtime = as_float(runtime_ms, float("nan"))
    if not math.isfinite(score) or not math.isfinite(runtime) or runtime <= 0.0:
        return None
    return round(score / max(1e-9, runtime / 1000.0), 6)


def quality_per_second(
    *,
    weighted_margin: float | None,
    balanced_gain: float | None,
    runtime_ms: float | None,
) -> float | None:
    weighted = as_float(weighted_margin, float("nan"))
    balanced = as_float(balanced_gain, float("nan"))
    gain_terms = [
        max(0.0, value)
        for value in (weighted, balanced)
        if math.isfinite(value)
    ]
    if not gain_terms:
        return None
    return value_per_second(sum(gain_terms) / len(gain_terms), runtime_ms)


def route_improvement_per_second(
    *,
    weighted_margin: float | None,
    balanced_gain: float | None,
    runtime_ms: float | None,
) -> float | None:
    weighted = as_float(weighted_margin, float("nan"))
    balanced = as_float(balanced_gain, float("nan"))
    if math.isfinite(weighted):
        return value_per_second(max(0.0, weighted), runtime_ms)
    if not math.isfinite(balanced):
        return None
    return value_per_second(max(0.0, balanced), runtime_ms)


def frontier_gain_per_ms(frontier_gain: float | None, runtime_ms: float | None) -> float | None:
    gain = as_float(frontier_gain, float("nan"))
    runtime = as_float(runtime_ms, float("nan"))
    if not math.isfinite(gain) or not math.isfinite(runtime) or runtime <= 0.0:
        return None
    return round(max(0.0, gain / runtime), 9)


def certificate_gain_per_world(
    certificate_margin_value: float | None,
    effective_world_count: float | None,
) -> float | None:
    margin = as_float(certificate_margin_value, float("nan"))
    worlds = as_float(effective_world_count, float("nan"))
    if not math.isfinite(margin) or not math.isfinite(worlds) or worlds <= 0.0:
        return None
    return round(max(0.0, margin) / worlds, 9)


def controller_cost_per_certificate_point(
    controller_runtime_ms: float | None,
    certificate_lift: float | None,
) -> float | None:
    runtime = as_float(controller_runtime_ms, float("nan"))
    lift = as_float(certificate_lift, float("nan"))
    if not math.isfinite(runtime) or not math.isfinite(lift):
        return None
    if lift <= 0.0:
        return None
    return round(max(0.0, runtime) / lift, 6)


def productive_action_rate(
    productive_actions: float | None,
    total_actions: float | None,
) -> float | None:
    productive = as_float(productive_actions, float("nan"))
    total = as_float(total_actions, float("nan"))
    if not math.isfinite(productive) or not math.isfinite(total) or total <= 0.0:
        return None
    return round(max(0.0, min(1.0, productive / total)), 6)


def cache_reuse_ratio(*ratios: float | None) -> float | None:
    values = [
        as_float(value, float("nan"))
        for value in ratios
    ]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return round(sum(max(0.0, min(1.0, value)) for value in values) / len(values), 6)


def memory_per_unit(memory_mb: float | None, unit_count: float | None) -> float | None:
    memory = as_float(memory_mb, float("nan"))
    units = as_float(unit_count, float("nan"))
    if not math.isfinite(memory) or not math.isfinite(units) or units <= 0.0:
        return None
    return round(max(0.0, memory / units), 6)


def ambiguity_alignment(prior: float | None, observed: float | None) -> float | None:
    prior_value = as_float(prior, float("nan"))
    observed_value = as_float(observed, float("nan"))
    if not math.isfinite(prior_value) or not math.isfinite(observed_value):
        return None
    return round(max(0.0, 1.0 - min(1.0, abs(prior_value - observed_value))), 6)


def ambiguity_absolute_error(prior: float | None, observed: float | None) -> float | None:
    prior_value = as_float(prior, float("nan"))
    observed_value = as_float(observed, float("nan"))
    if not math.isfinite(prior_value) or not math.isfinite(observed_value):
        return None
    return round(abs(prior_value - observed_value), 6)


def ambiguity_prior_top_k_precision(
    priors: Sequence[float | None],
    observed: Sequence[float | None],
    *,
    k: int,
    observed_positive_threshold: float = 0.10,
) -> float | None:
    paired: list[tuple[float, float]] = []
    for prior, realized in zip(priors, observed, strict=False):
        prior_value = as_float(prior, float("nan"))
        realized_value = as_float(realized, float("nan"))
        if not math.isfinite(prior_value) or not math.isfinite(realized_value):
            continue
        paired.append((prior_value, realized_value))
    if not paired:
        return None
    top_k = max(1, min(int(k), len(paired)))
    ranked = sorted(paired, key=lambda item: (-item[0], -item[1]))
    positives = sum(1 for _, realized in ranked[:top_k] if realized >= observed_positive_threshold)
    return round(positives / float(top_k), 6)


def ambiguity_prior_overtrigger_rate(
    priors: Sequence[float | None],
    observed: Sequence[float | None],
    *,
    prior_trigger_threshold: float = 0.45,
    observed_low_threshold: float = 0.08,
) -> float | None:
    triggers = 0
    overtriggered = 0
    for prior, realized in zip(priors, observed, strict=False):
        prior_value = as_float(prior, float("nan"))
        realized_value = as_float(realized, float("nan"))
        if not math.isfinite(prior_value) or not math.isfinite(realized_value):
            continue
        if prior_value < prior_trigger_threshold:
            continue
        triggers += 1
        if realized_value <= observed_low_threshold:
            overtriggered += 1
    if triggers <= 0:
        return None
    return round(overtriggered / float(triggers), 6)


def supported_ambiguity_alignment(
    prior: float | None,
    observed: float | None,
    *,
    confidence: float | None,
    support_ratio: float | None,
    source_mix_count: float | None,
) -> float | None:
    alignment = ambiguity_alignment(prior, observed)
    if alignment is None:
        return None
    confidence_value = max(0.0, min(1.0, as_float(confidence, 0.0)))
    support_value = max(0.0, min(1.0, as_float(support_ratio, 0.0)))
    source_mix_value = max(0.0, min(1.0, as_float(source_mix_count, 0.0) / 3.0))
    support_weight = max(
        0.0,
        min(
            1.0,
            (0.50 * confidence_value) + (0.35 * support_value) + (0.15 * source_mix_value),
        ),
    )
    return round(alignment * support_weight, 6)


def controller_activation_on_high_ambiguity(
    prior: float | None,
    controller_engaged: bool | None,
    *,
    threshold: float = 0.55,
) -> bool | None:
    prior_value = as_float(prior, float("nan"))
    if not math.isfinite(prior_value):
        return None
    if prior_value < float(threshold):
        return None
    return bool(controller_engaged)


def bytes_to_megabytes(value: Any) -> float | None:
    raw = as_float(value, float("nan"))
    if not math.isfinite(raw) or raw < 0.0:
        return None
    return round(raw / (1024.0 * 1024.0), 6)


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
