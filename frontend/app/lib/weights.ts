import type { RouteOption } from './types';

export type WeightState = { time: number; money: number; co2: number };

export function normaliseWeights(w: WeightState) {
  const sum = w.time + w.money + w.co2;
  if (sum <= 0) return { time: 1 / 3, money: 1 / 3, co2: 1 / 3 };
  return { time: w.time / sum, money: w.money / sum, co2: w.co2 / sum };
}

export function pickBestByWeightedSum(routes: RouteOption[], weights: WeightState): string | null {
  if (!routes.length) return null;

  const { time: wt, money: wm, co2: we } = normaliseWeights(weights);

  const times = routes.map((r) => r.metrics.duration_s);
  const moneys = routes.map((r) => r.metrics.monetary_cost);
  const co2s = routes.map((r) => r.metrics.emissions_kg);

  const tmin = Math.min(...times), tmax = Math.max(...times);
  const mmin = Math.min(...moneys), mmax = Math.max(...moneys);
  const emin = Math.min(...co2s), emax = Math.max(...co2s);

  const norm = (v: number, mn: number, mx: number) => (mx <= mn ? 0 : (v - mn) / (mx - mn));

  let bestId = routes[0].id;
  let bestScore = Infinity;

  for (const r of routes) {
    const score =
      wt * norm(r.metrics.duration_s, tmin, tmax) +
      wm * norm(r.metrics.monetary_cost, mmin, mmax) +
      we * norm(r.metrics.emissions_kg, emin, emax);

    if (score < bestScore) {
      bestScore = score;
      bestId = r.id;
    }
  }

  return bestId;
}
