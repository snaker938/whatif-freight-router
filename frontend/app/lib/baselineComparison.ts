import type { RouteOption } from './types';

export type BaselineComparison = {
  etaDeltaS: number;
  etaPct: number;
  costDelta: number;
  costPct: number;
  co2Delta: number;
  co2Pct: number;
  distanceDeltaKm: number;
  distancePct: number;
  distanceGainPct: number;
  epicScore: number;
  epicTier: 'Epic' | 'Strong' | 'Positive' | 'Limited gain';
};

function ratioPctLowerIsBetter(baseline: number, smart: number): number {
  const denom = Math.max(Math.abs(baseline), 1e-6);
  return ((baseline - smart) / denom) * 100;
}

function ratioPctDirect(baseline: number, smart: number): number {
  const denom = Math.max(Math.abs(baseline), 1e-6);
  return ((smart - baseline) / denom) * 100;
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function epicTier(score: number): BaselineComparison['epicTier'] {
  if (score >= 85) return 'Epic';
  if (score >= 70) return 'Strong';
  if (score >= 55) return 'Positive';
  return 'Limited gain';
}

export function buildBaselineComparison(
  smartRoute: RouteOption | null,
  baselineRoute: RouteOption | null,
): BaselineComparison | null {
  if (!smartRoute || !baselineRoute) return null;
  const smart = smartRoute.metrics;
  const base = baselineRoute.metrics;

  const etaDeltaS = base.duration_s - smart.duration_s;
  const costDelta = base.monetary_cost - smart.monetary_cost;
  const co2Delta = base.emissions_kg - smart.emissions_kg;
  const distanceDeltaKm = smart.distance_km - base.distance_km;

  const etaPct = ratioPctLowerIsBetter(base.duration_s, smart.duration_s);
  const costPct = ratioPctLowerIsBetter(base.monetary_cost, smart.monetary_cost);
  const co2Pct = ratioPctLowerIsBetter(base.emissions_kg, smart.emissions_kg);
  const distancePct = ratioPctDirect(base.distance_km, smart.distance_km);
  const distanceGainPct = -distancePct;

  const normEta = clamp(etaPct / 60, -1, 1);
  const normCost = clamp(costPct / 60, -1, 1);
  const normCo2 = clamp(co2Pct / 60, -1, 1);
  const epicScore = Math.round(clamp(50 + ((normEta + normCost + normCo2) / 3) * 50, 0, 100));

  return {
    etaDeltaS,
    etaPct,
    costDelta,
    costPct,
    co2Delta,
    co2Pct,
    distanceDeltaKm,
    distancePct,
    distanceGainPct,
    epicScore,
    epicTier: epicTier(epicScore),
  };
}
