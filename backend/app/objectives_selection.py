from __future__ import annotations

from .models import RouteOption


def normalise_weights(w_time: float, w_money: float, w_co2: float) -> tuple[float, float, float]:
    s = w_time + w_money + w_co2
    if s <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (w_time / s, w_money / s, w_co2 / s)


def pick_best_by_weighted_sum(
    options: list[RouteOption], *, w_time: float, w_money: float, w_co2: float
) -> RouteOption:
    """Pick best option by weighted sum after min-max normalisation per objective."""
    wt, wm, we = normalise_weights(w_time, w_money, w_co2)

    times = [o.metrics.duration_s for o in options]
    moneys = [o.metrics.monetary_cost for o in options]
    co2s = [o.metrics.emissions_kg for o in options]

    tmin, tmax = min(times), max(times)
    mmin, mmax = min(moneys), max(moneys)
    emin, emax = min(co2s), max(co2s)

    def norm(v: float, mn: float, mx: float) -> float:
        return 0.0 if mx <= mn else (v - mn) / (mx - mn)

    best = options[0]
    best_score = float("inf")

    for o in options:
        score = (
            wt * norm(o.metrics.duration_s, tmin, tmax)
            + wm * norm(o.metrics.monetary_cost, mmin, mmax)
            + we * norm(o.metrics.emissions_kg, emin, emax)
        )
        if score < best_score:
            best = o
            best_score = score

    return best
