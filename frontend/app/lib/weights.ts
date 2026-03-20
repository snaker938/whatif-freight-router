import type { RouteOption } from './types';

export type WeightState = { time: number; money: number; co2: number };
export type SelectionMathProfile =
  | 'academic_reference'
  | 'academic_tchebycheff'
  | 'academic_vikor'
  | 'modified_hybrid'
  | 'modified_distance_aware'
  | 'modified_vikor_distance';

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
  let bestScore = Number.POSITIVE_INFINITY;

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

export function pickBestByModifiedHybrid(routes: RouteOption[], weights: WeightState): string | null {
  return pickBestBySelectionProfile(routes, weights, 'modified_hybrid');
}

export function pickBestBySelectionProfile(
  routes: RouteOption[],
  weights: WeightState,
  profile: SelectionMathProfile,
): string | null {
  if (!routes.length) return null;

  const { time: wt, money: wm, co2: we } = normaliseWeights(weights);
  const regretWeight = 0.35;
  const balanceWeight = 0.1;
  const distanceWeight = 0.22;
  const etaDistanceWeight = 0.18;
  const entropyWeight = 0.08;
  const kneeWeight = 0.12;
  const tchebycheffRho = 0.001;
  const vikorV = 0.5;

  const times = routes.map((r) => r.metrics.duration_s);
  const moneys = routes.map((r) => r.metrics.monetary_cost);
  const co2s = routes.map((r) => r.metrics.emissions_kg);
  const distances = routes.map((r) => r.metrics.distance_km);

  const tmin = Math.min(...times), tmax = Math.max(...times);
  const mmin = Math.min(...moneys), mmax = Math.max(...moneys);
  const emin = Math.min(...co2s), emax = Math.max(...co2s);
  const dmin = Math.min(...distances), dmax = Math.max(...distances);

  const norm = (v: number, mn: number, mx: number) => (mx <= mn ? 0 : (v - mn) / (mx - mn));
  const safeScale = (v: number, mn: number, mx: number) => (mx <= mn ? 0 : (v - mn) / (mx - mn));

  // Academic and modified selection formulas mirrored from backend:
  // 1) Weighted-sum: Marler & Arora (2010) https://doi.org/10.1007/s00158-009-0460-7
  // 2) Augmented Tchebycheff: Steuer & Choo (1983) https://doi.org/10.1007/BF02591962
  // 3) VIKOR compromise ranking: Opricovic & Tzeng (2004) https://doi.org/10.1016/S0377-2217(03)00020-1
  // 4) Distance criterion in multi-objective routing: Martins (1984) https://doi.org/10.1016/0377-2217(84)90202-2
  // 5) Knee-oriented compromise signal: Branke et al. (2004) https://doi.org/10.1007/978-3-540-30217-9_73
  // 6) Entropy reward: Shannon (1948) https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
  //
  // The modified profiles are practical engineering blends for selecting one
  // representative route from an existing Pareto set. They are transparent and
  // intentionally configurable, not presented as novel optimization theory.
  const normRows = routes.map((r) => {
    const nt = norm(r.metrics.duration_s, tmin, tmax);
    const nm = norm(r.metrics.monetary_cost, mmin, mmax);
    const ne = norm(r.metrics.emissions_kg, emin, emax);
    const nd = norm(r.metrics.distance_km, dmin, dmax);
    return { nt, nm, ne, nd };
  });
  const weightedRows = normRows.map((row) => wt * row.nt + wm * row.nm + we * row.ne);
  const regretRows = normRows.map((row) => Math.max(wt * row.nt, wm * row.nm, we * row.ne));
  const vikorSMin = Math.min(...weightedRows);
  const vikorSMax = Math.max(...weightedRows);
  const vikorRMin = Math.min(...regretRows);
  const vikorRMax = Math.max(...regretRows);

  let bestId = routes[0].id;
  let bestScore = Number.POSITIVE_INFINITY;

  for (let idx = 0; idx < routes.length; idx += 1) {
    const route = routes[idx];
    const row = normRows[idx];
    const nt = row.nt;
    const nm = row.nm;
    const ne = row.ne;
    const nd = row.nd;
    const weighted = weightedRows[idx];
    const regret = regretRows[idx];
    const mean = (nt + nm + ne) / 3;
    const balance = Math.sqrt(((nt - mean) ** 2 + (nm - mean) ** 2 + (ne - mean) ** 2) / 3);
    const kneePenalty = (Math.abs(nt - nm) + Math.abs(nt - ne) + Math.abs(nm - ne)) / 3;
    const etaDistancePenalty = Math.sqrt(Math.max(0, nt * nd));
    const improveTime = Math.max(1e-6, 1 - nt);
    const improveMoney = Math.max(1e-6, 1 - nm);
    const improveCo2 = Math.max(1e-6, 1 - ne);
    const improveSum = improveTime + improveMoney + improveCo2;
    const pTime = improveTime / improveSum;
    const pMoney = improveMoney / improveSum;
    const pCo2 = improveCo2 / improveSum;
    const entropyReward = -(
      pTime * Math.log(pTime) +
      pMoney * Math.log(pMoney) +
      pCo2 * Math.log(pCo2)
    ) / Math.log(3);
    const vikorQ =
      vikorV * safeScale(weighted, vikorSMin, vikorSMax) +
      (1 - vikorV) * safeScale(regret, vikorRMin, vikorRMax);
    let score = weighted;
    if (profile === 'academic_tchebycheff') {
      score = regret + tchebycheffRho * weighted;
    } else if (profile === 'academic_vikor') {
      score = vikorQ;
    } else if (profile === 'modified_hybrid') {
      score = weighted + regretWeight * regret + balanceWeight * balance;
    } else if (profile === 'modified_vikor_distance') {
      score =
        vikorQ +
        balanceWeight * balance +
        distanceWeight * nd +
        etaDistanceWeight * etaDistancePenalty +
        kneeWeight * kneePenalty -
        entropyWeight * entropyReward;
    } else if (profile === 'modified_distance_aware') {
      score =
        weighted +
        regretWeight * regret +
        balanceWeight * balance +
        distanceWeight * nd +
        etaDistanceWeight * etaDistancePenalty +
        kneeWeight * kneePenalty -
        entropyWeight * entropyReward;
    } else {
      score = weighted;
    }
    if (score < bestScore) {
      bestScore = score;
      bestId = route.id;
    }
  }
  return bestId;
}
