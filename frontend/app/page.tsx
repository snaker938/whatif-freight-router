'use client';
// frontend/app/page.tsx

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useTransition } from 'react';

import BatchRunner from './components/devtools/BatchRunner';
import CollapsibleCard from './components/CollapsibleCard';
import CounterfactualPanel from './components/CounterfactualPanel';
import CustomVehicleManager from './components/devtools/CustomVehicleManager';
import DepartureOptimizerChart from './components/DepartureOptimizerChart';
import DutyChainPlanner from './components/DutyChainPlanner';
import ExperimentManager from './components/ExperimentManager';
import FieldInfo from './components/FieldInfo';
import OpsDiagnosticsPanel from './components/devtools/OpsDiagnosticsPanel';
import OracleQualityDashboard from './components/OracleQualityDashboard';
import PinManager from './components/PinManager';
import RunInspector from './components/devtools/RunInspector';
import ScenarioParameterEditor, {
  type ScenarioAdvancedParams,
} from './components/ScenarioParameterEditor';
import ScenarioTimeLapse from './components/ScenarioTimeLapse';
import Select, { type SelectOption } from './components/Select';
import SignatureVerifier from './components/devtools/SignatureVerifier';
import { deleteJSON, getJSON, getText, postJSON, postJSONWithMeta, postNDJSON, putJSON } from './lib/api';
import { formatNumber } from './lib/format';
import { LOCALE_OPTIONS, createTranslator, type Locale } from './lib/i18n';
import { buildManagedPinNodes } from './lib/mapOverlays';
import {
  buildDepartureOptimizeRequest,
  buildDutyChainRequest,
  buildParetoRequest,
  buildRouteRequest,
  buildScenarioCompareRequest as buildScenarioCompareRequestPayload,
  type RoutingAdvancedPatch,
} from './lib/requestBuilders';
import {
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
  vehicleDescriptionFromId,
} from './lib/sidebarHelpText';
import {
  TUTORIAL_CANONICAL_DESTINATION,
  TUTORIAL_CANONICAL_DUTY_STOPS,
  TUTORIAL_CANONICAL_ORIGIN,
  TUTORIAL_COMPLETED_KEY,
  TUTORIAL_PROGRESS_KEY,
  defaultDepartureWindow,
  nextUtcHourLocalInput,
} from './lib/tutorial/prefills';
import {
  clearTutorialProgress,
  loadTutorialCompleted,
  loadTutorialProgress,
  saveTutorialCompleted,
  saveTutorialProgress,
} from './lib/tutorial/progress';
import { TUTORIAL_CHAPTERS, TUTORIAL_STEPS } from './lib/tutorial/steps';
import type {
  TutorialLockScope,
  TutorialProgress,
  TutorialStep,
  TutorialTargetRect,
} from './lib/tutorial/types';
import type {
  CostToggles,
  ComputeMode,
  CustomVehicleListResponse,
  EmissionsContext,
  FuelType,
  EuroClass,
  HealthResponse,
  HealthReadyResponse,
  IncidentSimulatorConfig,
  MetricsResponse,
  CacheStatsResponse,
  CacheClearResponse,
  RouteRequest,
  RouteResponse,
  ParetoRequest,
  BatchParetoRequest,
  BatchCSVImportRequest,
  BatchParetoResponse,
  RunArtifactsListResponse,
  SignatureVerificationRequest,
  SignatureVerificationResponse,
  DutyChainRequest,
  DutyChainResponse,
  DepartureOptimizeRequest,
  DepartureOptimizeResponse,
  EpsilonConstraints,
  ExperimentCatalogSort,
  ExperimentBundle,
  ExperimentListResponse,
  LatLng,
  OracleFeedCheckInput,
  OracleFeedCheckRecord,
  OracleQualityDashboardResponse,
  OptimizationMode,
  ParetoResponse,
  ParetoMethod,
  ParetoStreamEvent,
  ManagedStop,
  LiveCallEntry,
  LiveCallTraceResponse,
  MapFailureOverlay,
  PinFocusRequest,
  PinSelectionId,
  RouteOption,
  ScenarioMode,
  ScenarioCompareResponse,
  ScenarioCompareRequest,
  StochasticConfig,
  TerrainProfile,
  TimeWindowConstraints,
  TutorialGuideTarget,
  VehicleListResponse,
  VehicleProfile,
  Waypoint,
  WeatherImpactConfig,
  WeatherProfile,
} from './lib/types';
import { normaliseWeights, pickBestByWeightedSum, type WeightState } from './lib/weights';

type MarkerKind = 'origin' | 'destination';
type ProgressState = { done: number; total: number };
type ComputeTraceLevel = 'info' | 'warn' | 'error' | 'success';
type ComputeTraceEntry = {
  id: number;
  at: string;
  elapsedMs: number;
  level: ComputeTraceLevel;
  step: string;
  detail: string;
  reasonCode?: string;
  recoveries?: string[];
  attempt?: number;
  endpoint?: '/api/pareto/stream' | '/api/pareto' | '/api/route';
  requestId?: string;
  stage?: string;
  stageDetail?: string;
  backendElapsedMs?: number;
  stageElapsedMs?: number;
  alternativesUsed?: number;
  timeoutMs?: number;
  abortReason?: string;
  candidateDiagnostics?: Record<string, unknown> | null;
  failureChain?: Record<string, unknown> | null;
};

type ComputeSession = {
  seq: number;
  startedAt: string;
  mode: ComputeMode;
  scenario: ScenarioMode;
  vehicle: string;
  waypointCount: number;
  maxAlternatives: number;
  streamStallTimeoutMs: number;
  origin: LatLng;
  destination: LatLng;
  attemptTimeoutMs: number;
  routeFallbackTimeoutMs: number;
  degradeSteps: number[];
};

function inferComputeRecoveryHints(message: string): string[] {
  const lower = message.toLowerCase();
  const hints: string[] = [];

  if (
    lower.includes('econnreset') ||
    lower.includes('fetch failed') ||
    lower.includes('body timeout') ||
    lower.includes('und_err_body_timeout') ||
    lower.includes('streaming response stalled')
  ) {
    hints.push('Network stream dropped between frontend and backend. Try stream mode again or keep JSON mode.');
    hints.push('Check backend logs for provider timeouts while the route was computing.');
    hints.push('Use JSON compute mode for large requests, or lower max alternatives for faster completion.');
  }
  if (lower.includes('backend_pareto_unreachable')) {
    hints.push('Pareto JSON request likely exceeded proxy wait limits before backend returned headers.');
    hints.push('Use stream mode for long runs or lower max alternatives.');
  }
  if (lower.includes('backend_headers_timeout')) {
    hints.push('Backend did not return response headers within timeout budget.');
    hints.push('Retry with fewer alternatives, or run stream mode with reduced candidate budget.');
  }
  if (lower.includes('backend_body_timeout')) {
    hints.push('Backend started responding but body streaming timed out.');
    hints.push('Retry with fewer alternatives or inspect backend logs for long-running stage stalls.');
  }
  if (lower.includes('backend_connection_reset')) {
    hints.push('Connection to backend was reset mid-request. Check backend process stability and memory pressure.');
  }
  if (lower.includes('route_compute_timeout')) {
    hints.push('Server-side route compute exceeded the bounded attempt timeout.');
    hints.push('Retry with lower max alternatives or use smaller OD spans for interactive runs.');
  }
  if (lower.includes('routing_graph_warming_up')) {
    hints.push('Routing graph is still warming up in memory.');
    hints.push('Check GET /health/ready and retry when strict_route_ready=true.');
  }
  if (lower.includes('routing_graph_warmup_failed')) {
    hints.push('Routing graph warmup failed. Rebuild routing_graph_uk.json and restart backend.');
    hints.push('Use GET /health/ready to inspect warmup phase, timeout, and asset diagnostics.');
  }
  if (lower.includes('routing_graph_unavailable')) {
    hints.push('Routing graph assets are unavailable for strict routing.');
    hints.push('Rebuild/publish model assets and verify backend /health/ready before retrying.');
  }
  if (lower.includes('routing_graph_no_path')) {
    hints.push('Routing graph search exhausted without a feasible strict path for this OD.');
    hints.push('Tune adaptive hop budget or verify graph connectivity for this corridor before retrying.');
  }
  if (lower.includes('routing_graph_fragmented')) {
    hints.push('Loaded route graph is fragmented under strict caps.');
    hints.push('Rebuild routing graph artifacts and verify /health/ready largest_component_ratio is healthy.');
  }
  if (lower.includes('routing_graph_disconnected_od')) {
    hints.push('Origin and destination are in different graph components in the loaded runtime graph.');
    hints.push('Try a nearer OD pair or refresh/rebuild graph assets to restore long-corridor connectivity.');
  }
  if (lower.includes('routing_graph_coverage_gap')) {
    hints.push('Nearest graph node is too far from origin/destination for strict routing.');
    hints.push('Check graph coverage for this region or increase graph coverage quality before retrying.');
  }
  if (lower.includes('routing_graph_precheck_timeout')) {
    hints.push('Route graph feasibility precheck timed out before candidate search could continue.');
    hints.push('Increase ROUTE_GRAPH_OD_FEASIBILITY_TIMEOUT_MS for full graph OD checks, then retry.');
  }
  if (lower.includes('terrain_dem_coverage_insufficient')) {
    hints.push('Terrain probe fetched a tile but sampling reported insufficient coverage for strict validation.');
    hints.push('This is usually terrain sampling/coverage validation, not a live API outage. Check prefetch_failed_details for probe counts and points.');
  }
  const terrainGateFailure =
    lower.includes('live_source_refresh_failed') &&
    (lower.includes('terrain_live_tile') || lower.includes('terrain_dem_coverage_insufficient'));
  const scenarioSemanticGateFailure =
    lower.includes('live_source_refresh_failed') &&
    lower.includes('scenario_live_context:scenario_profile_unavailable') &&
    lower.includes('missing_expected_sources=none');
  if (lower.includes('live_source_refresh_failed')) {
    if (terrainGateFailure) {
      hints.push('Strict live-refresh gate failed because terrain validation did not meet strict coverage thresholds.');
      hints.push('Inspect Graph Diagnostics -> prefetch_failed_details/prefetch_rows_json for terrain probe coverage and sampled points.');
    } else if (scenarioSemanticGateFailure) {
      hints.push('Strict live-refresh failed on scenario semantic coverage, not transport reachability.');
      hints.push('Expected source-family matrix is satisfied; inspect Scenario Coverage Gate for source_ok vs required thresholds.');
      hints.push('When local observations are sparse, verify road-hint alignment and scenario_live_context waiver diagnostics.');
    } else {
      hints.push('Strict live-refresh gate failed before candidate search completed.');
      hints.push('Check Live API calls panel for blocked/miss source families and upstream provider response errors.');
    }
  }
  if (lower.includes('backend_route_unreachable')) {
    hints.push('Single-route endpoint is unreachable. Verify backend is running and healthy on port 8000.');
  }
  if (lower.includes('stream_proxy_interrupted')) {
    hints.push('Backend stream proxy was interrupted before final response; this is usually a long-running stream timeout.');
    hints.push('Automatic fallback should now run. If it still fails, reduce max alternatives and retry.');
  }
  if (
    lower.includes('scenario_profile_unavailable') ||
    lower.includes('scenario coefficient payload is stale') ||
    lower.includes('live scenario context incomplete')
  ) {
    const scenarioCoeffStale =
      lower.includes('scenario coefficient payload is stale') ||
      (lower.includes('as_of_utc') && lower.includes('max_age_minutes'));
    if (scenarioCoeffStale) {
      hints.push(
        'Scenario coefficient payload is stale for strict mode; verify as_of_utc age against LIVE_SCENARIO_COEFFICIENT_MAX_AGE_MINUTES.',
      );
      hints.push(
        'Use backend preflight to confirm strict live readiness and check LIVE_SCENARIO_COEFFICIENT_URL points to the tracked main artifact.',
      );
    } else {
      hints.push('Run strict preflight and check scenario_live_context result for missing live sources.');
      hints.push(
        'If a source is intermittently unavailable, tune LIVE_SCENARIO_MIN_SOURCE_COUNT_STRICT / LIVE_SCENARIO_MIN_COVERAGE_OVERALL_STRICT.',
      );
    }
  }
  if (lower.includes('toll_tariff_unavailable')) {
    hints.push('Republish toll tariffs JSON artifact and confirm LIVE_TOLL_TARIFFS_URL points to JSON, not YAML.');
  }
  if (lower.includes('stochastic_calibration_unavailable')) {
    hints.push('Rebuild stochastic residuals and stochastic_regimes_uk.json, then republish live artifacts.');
  }
  if (lower.includes('weights') && (lower.includes('money') || lower.includes('co2'))) {
    hints.push('Use weights keys { time, money, co2 }. Backend also accepts { cost, emissions } now.');
  }
  if (lower.includes('osrm') || lower.includes('connection refused')) {
    hints.push('Verify OSRM container is healthy and reachable on the configured backend URL.');
  }
  if (!hints.length) {
    hints.push('Check backend logs around the same timestamp for the exact failure source.');
    hints.push('Retry with compute mode set to JSON to avoid transport-stream instability.');
  }

  return hints;
}

function parsePositiveIntEnv(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return parsed;
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value;
}

function clampUnit(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function formatDurationCompactMs(ms: number | null): string {
  if (ms === null || !Number.isFinite(ms) || ms < 0) return 'n/a';
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes >= 10 || seconds === 0) return `${minutes}m`;
  return `${minutes}m ${seconds}s`;
}

function warmupPhaseLabel(phaseRaw: string): string {
  const phase = phaseRaw.trim().toLowerCase();
  if (!phase) return 'Unknown phase';
  if (phase === 'initializing') return 'Initializing loader';
  if (phase === 'parsing_nodes') return 'Parsing graph nodes';
  if (phase === 'parsing_edges') return 'Parsing graph edges';
  if (phase === 'finalizing') return 'Finalizing graph index';
  if (phase === 'ready') return 'Ready';
  if (phase === 'failed') return 'Warmup failed';
  return phase.replace(/_/g, ' ');
}

function safeJsonString(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function safeParseJsonObject(raw: unknown): Record<string, unknown> | null {
  if (typeof raw !== 'string') return null;
  const text = raw.trim();
  if (!text) return null;
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}

function parseDegradeSteps(raw: string | undefined): number[] {
  const source = (raw ?? '12,6,3').trim();
  const parts = source
    .split(',')
    .map((token) => Number.parseInt(token.trim(), 10))
    .filter((value) => Number.isFinite(value) && value > 0);
  const unique = Array.from(new Set(parts));
  return unique.length ? unique : [12, 6, 3];
}

function extractReasonCodeFromMessage(message: string): string | undefined {
  const raw = String(message ?? '').trim();
  if (!raw) return undefined;
  const reasonMatch = raw.match(/reason_code[:=]\s*([a-z0-9_]+)/i);
  if (reasonMatch?.[1]) return reasonMatch[1].toLowerCase();
  const parenMatch = raw.match(/\(([a-z0-9_]+)\)(?:\s|$)/i);
  if (parenMatch?.[1]) return parenMatch[1].toLowerCase();
  return undefined;
}

function buildAlternativesSequence(requested: number, defaults: number[]): number[] {
  const boundedRequested = Math.max(1, Math.floor(requested));
  const boundedDefaults = defaults
    .map((step) => Math.max(1, Math.min(boundedRequested, Math.floor(step))))
    .filter((step) => Number.isFinite(step) && step > 0);
  const seed = boundedDefaults[0] ?? boundedRequested;
  const out: number[] = [seed];
  for (const step of boundedDefaults.slice(1)) {
    if (!out.includes(step)) out.push(step);
    if (out.length >= 3) break;
  }
  while (out.length < 3) {
    const next = Math.max(1, Math.floor(out[out.length - 1] / 2));
    if (out.includes(next)) break;
    out.push(next);
  }
  return out.slice(0, 3);
}

type MapViewProps = {
  origin: LatLng | null;
  destination: LatLng | null;
  tutorialDraftOrigin?: LatLng | null;
  tutorialDraftDestination?: LatLng | null;
  tutorialDragDraftOrigin?: LatLng | null;
  tutorialDragDraftDestination?: LatLng | null;
  managedStop?: ManagedStop | null;
  originLabel?: string;
  destinationLabel?: string;

  selectedPinId?: 'origin' | 'destination' | 'stop-1' | null;
  focusPinRequest?: PinFocusRequest | null;
  fitAllRequestNonce?: number;

  route: RouteOption | null;
  failureOverlay?: MapFailureOverlay | null;
  timeLapsePosition?: LatLng | null;
  dutyStops?: DutyChainRequest['stops'];
  showStopOverlay?: boolean;
  showIncidentOverlay?: boolean;
  showSegmentTooltips?: boolean;
  showPreviewConnector?: boolean;
  overlayLabels?: {
    stopLabel: string;
    segmentLabel: string;
    incidentTypeLabels: Record<'dwell' | 'accident' | 'closure', string>;
  };
  onTutorialAction?: (actionId: string) => void;
  onTutorialTargetState?: (state: { hasSegmentTooltipPath: boolean; hasIncidentMarkers: boolean }) => void;
  tutorialMapLocked?: boolean;
  tutorialMapDimmed?: boolean;
  tutorialViewportLocked?: boolean;
  tutorialHideZoomControls?: boolean;
  tutorialRelaxBounds?: boolean;
  tutorialExpectedAction?: string | null;
  tutorialGuideTarget?: TutorialGuideTarget | null;
  tutorialGuideVisible?: boolean;
  tutorialConfirmPin?: 'origin' | 'destination' | null;
  tutorialDragConfirmPin?: 'origin' | 'destination' | null;

  onMapClick: (lat: number, lon: number) => void;
  onSelectPinId?: (id: 'origin' | 'destination' | 'stop-1' | null) => void;
  onMoveMarker: (kind: MarkerKind, lat: number, lon: number) => boolean;
  onMoveStop?: (lat: number, lon: number) => boolean;
  onAddStopFromPin?: () => void;
  onRenameStop?: (name: string) => void;
  onDeleteStop?: () => void;
  onFocusPin?: (id: 'origin' | 'destination' | 'stop-1') => void;
  onSwapMarkers?: () => void;
  onTutorialConfirmPin?: (kind: 'origin' | 'destination') => void;
  onTutorialConfirmDrag?: (kind: 'origin' | 'destination') => void;
};

const MapView = dynamic<MapViewProps>(() => import('./components/MapView'), {
  ssr: false,
  loading: () => <div className="mapPane" />,
});

type ParetoChartProps = {
  routes: RouteOption[];
  selectedId: string | null;
  labelsById: Record<string, string>;
  onSelect: (routeId: string) => void;
};

const ParetoChart = dynamic<ParetoChartProps>(() => import('./components/ParetoChart'), {
  ssr: false,
  loading: () => (
    <div className="chartWrap chartWrap--loading" aria-hidden="true">
      <div className="skeletonBar skeletonBar--lg" />
      <div className="skeletonBar skeletonBar--md" />
    </div>
  ),
});

const EtaTimelineChart = dynamic<{ route: RouteOption | null }>(
  () => import('./components/EtaTimelineChart'),
  {
    ssr: false,
    loading: () => (
      <div className="chartWrap chartWrap--loading" aria-hidden="true">
        <div className="skeletonBar skeletonBar--md" />
        <div className="skeletonBar skeletonBar--sm" />
      </div>
    ),
  },
);

const ScenarioComparison = dynamic<
  {
    data: ScenarioCompareResponse | null;
    loading: boolean;
    error: string | null;
    locale: Locale;
    onInspectScenarioManifest?: (runId: string) => void;
    onInspectScenarioSignature?: (runId: string) => void;
    onOpenRunInspector?: (runId: string) => void;
  }
>(() => import('./components/ScenarioComparison'), {
  ssr: false,
  loading: () => (
    <div className="cardSkeleton" aria-hidden="true">
      <div className="skeletonBar skeletonBar--md" />
      <div className="skeletonBar skeletonBar--sm" />
      <div className="skeletonBar skeletonBar--sm" />
    </div>
  ),
});

const SegmentBreakdown = dynamic<{ route: RouteOption | null; onTutorialAction?: (actionId: string) => void }>(
  () => import('./components/SegmentBreakdown'),
  {
    ssr: false,
    loading: () => (
      <div className="cardSkeleton cardSkeleton--compact" aria-hidden="true">
        <div className="skeletonBar skeletonBar--md" />
      </div>
    ),
  },
);

const TutorialOverlay = dynamic<
  {
    open: boolean;
    mode: 'blocked' | 'chooser' | 'running' | 'completed';
    isDesktop: boolean;
    hasSavedProgress: boolean;
    chapterTitle: string;
    chapterDescription: string;
    chapterIndex: number;
    chapterCount: number;
    stepTitle: string;
    stepWhat: string;
    stepImpact: string;
    stepIndex: number;
    stepCount: number;
    canGoNext: boolean;
    atStart: boolean;
    atEnd: boolean;
    checklist: Array<{
      actionId: string;
      label: string;
      details?: string;
      done: boolean;
      kind?: 'ui' | 'manual';
    }>;
    currentTaskOverride?: string | null;
    optionalDecision: {
      id: string;
      label: string;
      resolved: boolean;
      defaultLabel: string;
      actionTouched: boolean;
    } | null;
    targetRect: TutorialTargetRect | null;
    targetMissing: boolean;
    requiresTargetRect?: boolean;
    runningScope?: TutorialLockScope;
    onClose: () => void;
    onStartNew: () => void;
    onResume: () => void;
    onRestart: () => void;
    onBack: () => void;
    onNext: () => void;
    onFinish: () => void;
    onMarkManual: (actionId: string) => void;
    onUseOptionalDefault: (optionalDecisionId: string) => void;
  }
>(() => import('./components/TutorialOverlay'), {
  ssr: false,
  loading: () => null,
});

const DEFAULT_ADVANCED_PARAMS: ScenarioAdvancedParams = {
  maxAlternatives: '24',
  paretoMethod: 'dominance',
  epsilonDurationS: '',
  epsilonMonetaryCost: '',
  epsilonEmissionsKg: '',
  departureTimeUtcLocal: '',
  useTolls: true,
  fuelPriceMultiplier: '1.0',
  carbonPricePerKg: '0.0',
  tollCostPerKm: '0.0',
  terrainProfile: 'flat',
  optimizationMode: 'expected_value',
  riskAversion: '1.0',
  stochasticEnabled: false,
  stochasticSeed: '',
  stochasticSigma: '0.08',
  stochasticSamples: '25',
  fuelType: 'diesel',
  euroClass: 'euro6',
  ambientTempC: '15',
  weatherEnabled: false,
  weatherProfile: 'clear',
  weatherIntensity: '1.0',
  weatherIncidentUplift: true,
  incidentSimulationEnabled: false,
  incidentSeed: '',
  incidentDwellRatePer100km: '0.8',
  incidentAccidentRatePer100km: '0.25',
  incidentClosureRatePer100km: '0.05',
  incidentDwellDelayS: '120',
  incidentAccidentDelayS: '480',
  incidentClosureDelayS: '900',
  incidentMaxEventsPerRoute: '12',
};

const LOCALE_STORAGE_KEY = 'ui_locale_v1';

function sortRoutesDeterministic(routes: RouteOption[]): RouteOption[] {
  return [...routes].sort((a, b) => {
    const byDuration = a.metrics.duration_s - b.metrics.duration_s;
    if (byDuration !== 0) return byDuration;
    return a.id.localeCompare(b.id);
  });
}

function dedupeWarnings(items: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];

  for (const item of items) {
    const trimmed = item.trim();
    if (!trimmed || seen.has(trimmed)) continue;
    seen.add(trimmed);
    out.push(trimmed);
  }

  return out;
}

function sameLatLng(a: LatLng | null, b: LatLng | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return Math.abs(a.lat - b.lat) <= 1e-6 && Math.abs(a.lon - b.lon) <= 1e-6;
}

function sameManagedStop(a: ManagedStop | null, b: ManagedStop | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.id === b.id &&
    Math.abs(a.lat - b.lat) <= 1e-6 &&
    Math.abs(a.lon - b.lon) <= 1e-6 &&
    (a.label ?? '') === (b.label ?? '')
  );
}

function normalizeSelectedPinId(
  selectedPinId: PinSelectionId | null,
  points: {
    origin: LatLng | null;
    destination: LatLng | null;
    stop: ManagedStop | null;
  },
): PinSelectionId | null {
  if (selectedPinId === 'origin' && !points.origin) return null;
  if (selectedPinId === 'destination' && !points.destination) return null;
  if (selectedPinId === 'stop-1' && !points.stop) return null;
  return selectedPinId;
}

function toDutyLine(lat: number, lon: number, label?: string): string {
  const base = `${lat.toFixed(6)},${lon.toFixed(6)}`;
  const trimmed = (label ?? '').trim();
  return trimmed ? `${base},${trimmed}` : base;
}

function serializePinsToDutyText(
  origin: LatLng | null,
  stops: Array<{ lat: number; lon: number; label?: string | null }>,
  destination: LatLng | null,
): string {
  const lines: string[] = [];
  if (origin) {
    lines.push(toDutyLine(origin.lat, origin.lon, 'Start'));
  }
  for (let idx = 0; idx < stops.length; idx += 1) {
    const stop = stops[idx];
    lines.push(toDutyLine(stop.lat, stop.lon, stop.label || `Stop #${idx + 1}`));
  }
  if (destination) {
    lines.push(toDutyLine(destination.lat, destination.lon, 'End'));
  }
  return lines.join('\n');
}

type ParsedPinSync =
  | {
      ok: true;
      origin: LatLng | null;
      destination: LatLng | null;
      stop: ManagedStop | null;
      stops: Array<{ lat: number; lon: number; label: string }>;
    }
  | {
      ok: false;
      error: string;
    };

function parseDutyTextToPins(text: string): ParsedPinSync {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return {
      ok: true,
      origin: null,
      destination: null,
      stop: null,
      stops: [],
    };
  }

  const parsed = lines.map((line, idx) => {
    const parts = line.split(',');
    if (parts.length < 2) {
      throw new Error(`Line ${idx + 1} must be "lat,lon,label(optional)".`);
    }
    const lat = Number(parts[0].trim());
    const lon = Number(parts[1].trim());
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      throw new Error(`Line ${idx + 1} has invalid latitude/longitude.`);
    }
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      throw new Error(`Line ${idx + 1} is out of bounds.`);
    }
    const label = parts
      .slice(2)
      .join(',')
      .trim();
    return { lat, lon, label };
  });

  if (parsed.length === 1) {
    const first = parsed[0];
    return {
      ok: true,
      origin: { lat: first.lat, lon: first.lon },
      destination: null,
      stop: null,
      stops: [],
    };
  }

  const first = parsed[0];
  const last = parsed[parsed.length - 1];
  const stopRows = parsed.slice(1, -1).map((row, idx) => ({
    lat: row.lat,
    lon: row.lon,
    label: row.label || `Stop #${idx + 1}`,
  }));
  const firstStop = stopRows[0] ?? null;
  return {
    ok: true,
    origin: { lat: first.lat, lon: first.lon },
    destination: { lat: last.lat, lon: last.lon },
    stop: firstStop
      ? {
          id: 'stop-1',
          lat: firstStop.lat,
          lon: firstStop.lon,
          label: firstStop.label,
        }
      : null,
    stops: stopRows,
  };
}

function extractIntermediateStops(text: string): Array<{ lat: number; lon: number; label: string }> {
  const parsed = parseDutyTextToPins(text);
  if (!parsed.ok) return [];
  return parsed.stops;
}

function buildDutyStopsForTextSync(
  dutyStopsText: string,
  managedStop: ManagedStop | null,
): Array<{ lat: number; lon: number; label: string }> {
  const existing = extractIntermediateStops(dutyStopsText);
  if (!managedStop) return [];
  if (existing.length === 0) {
    return [
      {
        lat: managedStop.lat,
        lon: managedStop.lon,
        label: managedStop.label || 'Stop #1',
      },
    ];
  }
  return [
    {
      lat: managedStop.lat,
      lon: managedStop.lon,
      label: managedStop.label || existing[0].label || 'Stop #1',
    },
    ...existing.slice(1),
  ];
}

function RoutesSkeleton() {
  return (
    <div className="routesSkeleton" aria-hidden="true">
      {[0, 1, 2].map((idx) => (
        <div className="routeSkeleton" key={idx}>
          <div className="routeSkeleton__row">
            <div className="routeSkeleton__line routeSkeleton__line--title shimmer" />
            <div className="routeSkeleton__line routeSkeleton__line--pill shimmer" />
          </div>
          <div className="routeSkeleton__line routeSkeleton__line--meta shimmer" />
        </div>
      ))}
    </div>
  );
}

function SidebarToggleIcon({ collapsed }: { collapsed: boolean }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.9"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`sidebarToggle__icon ${collapsed ? 'isCollapsed' : ''}`}
    >
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.8" />
      <path d="M14 5v14" />
      <path d="M10.5 10l-2.5 2 2.5 2" />
    </svg>
  );
}

function TutorialSparkIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.9"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 3l1.6 4.2L18 8.8l-4.4 1.6L12 14.6l-1.6-4.2L6 8.8l4.4-1.6L12 3z" />
      <path d="M19 14l.8 2 .2.8.8.2 2 .8-2 .8-.8.2-.2.8-.8 2-.8-2-.2-.8-.8-.2-2-.8 2-.8.8-.2.2-.8.8-2z" />
      <path d="M5 14l.6 1.5.2.5.5.2 1.5.6-1.5.6-.5.2-.2.5L5 20l-.6-1.5-.2-.5-.5-.2-1.5-.6 1.5-.6.5-.2.2-.5L5 14z" />
    </svg>
  );
}

const WEIGHT_PRESETS: Array<{ id: string; label: string; value: WeightState }> = [
  { id: 'balanced', label: 'Balanced', value: { time: 60, money: 20, co2: 20 } },
  { id: 'fastest', label: 'Fastest', value: { time: 85, money: 10, co2: 5 } },
  { id: 'cheapest', label: 'Cheapest', value: { time: 10, money: 85, co2: 5 } },
  { id: 'cleanest', label: 'Cleanest', value: { time: 10, money: 10, co2: 80 } },
];

const TUTORIAL_SECTION_IDS = [
  'setup.section',
  'pins.section',
  'advanced.section',
  'preferences.section',
  'selected.route_panel',
  'routes.list',
  'compare.section',
  'departure.section',
  'duty.section',
  'oracle.section',
  'experiments.section',
  'timelapse.section',
] as const;

type TutorialSectionId = (typeof TUTORIAL_SECTION_IDS)[number];
type TutorialSectionControl = {
  isOpen?: boolean;
  lockToggle?: boolean;
  tutorialLocked?: boolean;
};

function inferTutorialSectionId(step: TutorialStep | null): TutorialSectionId | null {
  if (!step) return null;
  if (step.activeSectionId) {
    return TUTORIAL_SECTION_IDS.includes(step.activeSectionId as TutorialSectionId)
      ? (step.activeSectionId as TutorialSectionId)
      : null;
  }
  const match = step.targetIds.find((targetId) =>
    TUTORIAL_SECTION_IDS.includes(targetId as TutorialSectionId),
  );
  if (match) return match as TutorialSectionId;

  const hintFromTarget = step.targetIds.find(Boolean) ?? '';
  if (hintFromTarget.startsWith('setup.')) return 'setup.section';
  if (hintFromTarget.startsWith('pins.')) return 'pins.section';
  if (hintFromTarget.startsWith('advanced.')) return 'advanced.section';
  if (hintFromTarget.startsWith('preferences.') || hintFromTarget.startsWith('pref.')) {
    return 'preferences.section';
  }
  if (hintFromTarget.startsWith('selected.')) return 'selected.route_panel';
  if (hintFromTarget.startsWith('routes.')) return 'routes.list';
  if (hintFromTarget.startsWith('compare.')) return 'compare.section';
  if (hintFromTarget.startsWith('departure.') || hintFromTarget.startsWith('dep.')) {
    return 'departure.section';
  }
  if (hintFromTarget.startsWith('duty.')) return 'duty.section';
  if (hintFromTarget.startsWith('oracle.')) return 'oracle.section';
  if (hintFromTarget.startsWith('experiments.') || hintFromTarget.startsWith('exp.')) {
    return 'experiments.section';
  }
  if (hintFromTarget.startsWith('timelapse.')) return 'timelapse.section';
  return null;
}

function inferTutorialLockScope(step: TutorialStep | null): TutorialLockScope {
  if (!step) return 'free';
  if (step.lockScope) return step.lockScope;
  const activeSection = inferTutorialSectionId(step);
  if (activeSection) return 'sidebar_section_only';
  if (step.targetIds.includes('map.interactive')) return 'map_only';
  return 'free';
}

function isTutorialActionAllowed(actionId: string, allowSet: Set<string>, allowPrefixes: string[]): boolean {
  if (!actionId) return false;
  if (allowSet.has(actionId)) return true;
  if (actionId.endsWith(':open')) {
    const stem = actionId.slice(0, -5);
    if (allowSet.has(stem)) return true;
    for (const allowedId of allowSet) {
      if (allowedId.startsWith(`${stem}:`)) return true;
    }
  }
  return allowPrefixes.some((prefix) => actionId.startsWith(prefix));
}

type TutorialPlacementStage =
  | 'newcastle_origin'
  | 'london_destination'
  | 'post_confirm_marker_actions'
  | 'done';

const TUTORIAL_CITY_TARGETS = {
  newcastle: { lat: 54.9783, lon: -1.6178, radiusKm: 20, zoom: 11 },
  london: { lat: 51.5072, lon: -0.1276, radiusKm: 20, zoom: 12 },
  stoke: { lat: 53.0027, lon: -2.1794, radiusKm: 8, zoom: 9 },
} as const;
const TUTORIAL_FORCE_ONLY_ACTIONS = new Set([
  'map.confirm_origin_newcastle',
  'map.confirm_destination_london',
  'map.confirm_drag_destination_marker',
  'map.confirm_drag_origin_marker',
]);
const TUTORIAL_VALUE_CONSTRAINED_ACTIONS = new Set([
  'advanced.risk_aversion_input',
  'advanced.epsilon_duration_input',
  'advanced.epsilon_money_input',
  'advanced.epsilon_emissions_input',
  'advanced.stochastic_seed_input',
  'advanced.stochastic_sigma_input',
  'advanced.stochastic_samples_input',
  'advanced.use_tolls_toggle',
  'advanced.fuel_multiplier_input',
  'advanced.carbon_price_input',
  'advanced.toll_per_km_input',
  'pref.weight_time',
  'pref.weight_money',
  'pref.weight_co2',
]);

function haversineDistanceKm(a: LatLng, b: LatLng): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lon - a.lon);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(h)));
}

export default function Page() {
  const [origin, setOrigin] = useState<LatLng | null>(null);
  const [destination, setDestination] = useState<LatLng | null>(null);
  const [tutorialDraftOrigin, setTutorialDraftOrigin] = useState<LatLng | null>(null);
  const [tutorialDraftDestination, setTutorialDraftDestination] = useState<LatLng | null>(null);
  const [tutorialDragDraftOrigin, setTutorialDragDraftOrigin] = useState<LatLng | null>(null);
  const [tutorialDragDraftDestination, setTutorialDragDraftDestination] = useState<LatLng | null>(null);
  const [managedStop, setManagedStop] = useState<ManagedStop | null>(null);
  const [selectedPinId, setSelectedPinId] = useState<'origin' | 'destination' | 'stop-1' | null>(null);
  const [focusPinRequest, setFocusPinRequest] = useState<PinFocusRequest | null>(null);
  const [fitAllRequestNonce, setFitAllRequestNonce] = useState(0);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  const [locale, setLocale] = useState<Locale>('en');

  const [vehicles, setVehicles] = useState<VehicleProfile[]>([]);
  const [vehicleType, setVehicleType] = useState<string>('rigid_hgv');
  const [scenarioMode, setScenarioMode] = useState<ScenarioMode>('no_sharing');

  const [weights, setWeights] = useState<WeightState>({ time: 60, money: 20, co2: 20 });
  const [advancedParams, setAdvancedParams] = useState<ScenarioAdvancedParams>(DEFAULT_ADVANCED_PARAMS);
  const [advancedError, setAdvancedError] = useState<string | null>(null);
  const [computeMode, setComputeMode] = useState<ComputeMode>('pareto_stream');
  const [routeSort, setRouteSort] = useState<'duration' | 'cost' | 'co2'>('duration');

  const [paretoRoutes, setParetoRoutes] = useState<RouteOption[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [routeNames, setRouteNames] = useState<Record<string, string>>({});
  const [editingRouteId, setEditingRouteId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const [loading, setLoading] = useState(false);
  const [appBootReady, setAppBootReady] = useState(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [computeTraceOpen, setComputeTraceOpen] = useState(false);
  const [computeTrace, setComputeTrace] = useState<ComputeTraceEntry[]>([]);
  const [computeSession, setComputeSession] = useState<ComputeSession | null>(null);
  const [computeLiveCallsByAttempt, setComputeLiveCallsByAttempt] = useState<
    Record<number, LiveCallTraceResponse>
  >({});
  const [computeRequestIdByAttempt, setComputeRequestIdByAttempt] = useState<Record<number, string>>({});
  const [computeTraceNowMs, setComputeTraceNowMs] = useState(() => Date.now());
  const [showWarnings, setShowWarnings] = useState(false);
  const [scenarioCompare, setScenarioCompare] = useState<ScenarioCompareResponse | null>(null);
  const [scenarioCompareLoading, setScenarioCompareLoading] = useState(false);
  const [scenarioCompareError, setScenarioCompareError] = useState<string | null>(null);
  const [experiments, setExperiments] = useState<ExperimentBundle[]>([]);
  const [experimentsLoading, setExperimentsLoading] = useState(true);
  const [experimentsError, setExperimentsError] = useState<string | null>(null);
  const [expCatalogQuery, setExpCatalogQuery] = useState('');
  const [expCatalogVehicleType, setExpCatalogVehicleType] = useState('');
  const [expCatalogScenarioMode, setExpCatalogScenarioMode] = useState<'' | ScenarioMode>('');
  const [expCatalogSort, setExpCatalogSort] = useState<ExperimentCatalogSort>('updated_desc');
  const [depWindowStartLocal, setDepWindowStartLocal] = useState('');
  const [depWindowEndLocal, setDepWindowEndLocal] = useState('');
  const [depEarliestArrivalLocal, setDepEarliestArrivalLocal] = useState('');
  const [depLatestArrivalLocal, setDepLatestArrivalLocal] = useState('');
  const [depStepMinutes, setDepStepMinutes] = useState(60);
  const [depOptimizeLoading, setDepOptimizeLoading] = useState(false);
  const [depOptimizeError, setDepOptimizeError] = useState<string | null>(null);
  const [depOptimizeData, setDepOptimizeData] = useState<DepartureOptimizeResponse | null>(null);
  const [dutyStopsText, setDutyStopsText] = useState('');
  const [dutySyncError, setDutySyncError] = useState<string | null>(null);
  const [dutyChainLoading, setDutyChainLoading] = useState(false);
  const [dutyChainError, setDutyChainError] = useState<string | null>(null);
  const [dutyChainData, setDutyChainData] = useState<DutyChainResponse | null>(null);
  const [oracleDashboard, setOracleDashboard] = useState<OracleQualityDashboardResponse | null>(null);
  const [oracleDashboardLoading, setOracleDashboardLoading] = useState(true);
  const [oracleIngestLoading, setOracleIngestLoading] = useState(false);
  const [oracleError, setOracleError] = useState<string | null>(null);
  const [oracleLatestCheck, setOracleLatestCheck] = useState<OracleFeedCheckRecord | null>(null);
  const [opsHealth, setOpsHealth] = useState<HealthResponse | null>(null);
  const [opsHealthReady, setOpsHealthReady] = useState<HealthReadyResponse | null>(null);
  const [opsMetrics, setOpsMetrics] = useState<MetricsResponse | null>(null);
  const [opsCacheStats, setOpsCacheStats] = useState<CacheStatsResponse | null>(null);
  const [opsLoading, setOpsLoading] = useState(false);
  const [opsClearing, setOpsClearing] = useState(false);
  const [opsError, setOpsError] = useState<string | null>(null);
  const [customVehicles, setCustomVehicles] = useState<VehicleProfile[]>([]);
  const [customVehiclesLoading, setCustomVehiclesLoading] = useState(false);
  const [customVehicleSaving, setCustomVehicleSaving] = useState(false);
  const [customVehicleError, setCustomVehicleError] = useState<string | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchParetoResponse | null>(null);
  const [signatureLoading, setSignatureLoading] = useState(false);
  const [signatureError, setSignatureError] = useState<string | null>(null);
  const [signatureResult, setSignatureResult] = useState<SignatureVerificationResponse | null>(null);
  const [runInspectorRunId, setRunInspectorRunId] = useState('');
  const [runInspectorLoading, setRunInspectorLoading] = useState(false);
  const [runInspectorError, setRunInspectorError] = useState<string | null>(null);
  const [runManifest, setRunManifest] = useState<unknown | null>(null);
  const [runScenarioManifest, setRunScenarioManifest] = useState<unknown | null>(null);
  const [runProvenance, setRunProvenance] = useState<unknown | null>(null);
  const [runSignature, setRunSignature] = useState<unknown | null>(null);
  const [runScenarioSignature, setRunScenarioSignature] = useState<unknown | null>(null);
  const [runArtifacts, setRunArtifacts] = useState<RunArtifactsListResponse | null>(null);
  const [runArtifactPreviewName, setRunArtifactPreviewName] = useState<string | null>(null);
  const [runArtifactPreviewText, setRunArtifactPreviewText] = useState<string | null>(null);
  const [timeLapsePosition, setTimeLapsePosition] = useState<LatLng | null>(null);
  const [showStopOverlay, setShowStopOverlay] = useState(true);
  const [showIncidentOverlay, setShowIncidentOverlay] = useState(true);
  const [showSegmentTooltips, setShowSegmentTooltips] = useState(true);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [tutorialMode, setTutorialMode] = useState<'blocked' | 'chooser' | 'running' | 'completed'>(
    'chooser',
  );
  const [tutorialStepIndex, setTutorialStepIndex] = useState(0);
  const [tutorialStepActionsById, setTutorialStepActionsById] = useState<Record<string, string[]>>({});
  const [tutorialOptionalDecisionsByStep, setTutorialOptionalDecisionsByStep] = useState<Record<string, string[]>>(
    {},
  );
  const [tutorialTargetRect, setTutorialTargetRect] = useState<TutorialTargetRect | null>(null);
  const [tutorialTargetMissing, setTutorialTargetMissing] = useState(false);
  const [tutorialSavedProgress, setTutorialSavedProgress] = useState<TutorialProgress | null>(null);
  const [tutorialCompleted, setTutorialCompleted] = useState(false);
  const [tutorialPrefilledSteps, setTutorialPrefilledSteps] = useState<string[]>([]);
  const [tutorialIsDesktop, setTutorialIsDesktop] = useState(() =>
    typeof window === 'undefined' ? true : window.innerWidth >= 1100,
  );
  const [tutorialExperimentPrefill, setTutorialExperimentPrefill] = useState<{
    name: string;
    description: string;
  } | null>(null);
  const [tutorialResetNonce, setTutorialResetNonce] = useState(0);
  const [liveMessage, setLiveMessage] = useState('');

  const [error, setError] = useState<string | null>(null);

  const [isPending, startTransition] = useTransition();

  const abortRef = useRef<AbortController | null>(null);
  const activeAttemptAbortRef = useRef<AbortController | null>(null);
  const requestSeqRef = useRef(0);
  const computeTraceSeqRef = useRef(0);
  const computeStartedAtMsRef = useRef<number | null>(null);
  const progressRef = useRef<ProgressState | null>(null);
  const routeBufferRef = useRef<RouteOption[]>([]);
  const flushTimerRef = useRef<number | null>(null);
  const tutorialBootstrappedRef = useRef(false);
  const dutySyncSourceRef = useRef<'pins' | 'text' | null>(null);
  const focusPinNonceRef = useRef(0);
  const tutorialPlacementStageRef = useRef<TutorialPlacementStage>('done');
  const tutorialMapClickGuardUntilRef = useRef(0);
  const tutorialStepEnterRef = useRef<string | null>(null);
  const tutorialAutoAdvanceTokenRef = useRef('');
  const tutorialPreviousStepIdRef = useRef<string | null>(null);
  const tutorialRouteCenteredStepRef = useRef<string | null>(null);
  const tutorialLockNoticeAtRef = useRef(0);
  const tutorialConfirmedOriginRef = useRef<LatLng | null>(null);
  const tutorialConfirmedDestinationRef = useRef<LatLng | null>(null);
  const tutorialPulseLogSeqRef = useRef(0);
  const logTutorialMidpoint = useCallback(
    (_event: string, _payload?: Record<string, unknown>) => {},
    [],
  );
  const logTutorialPulse = useCallback((event: string, payload?: Record<string, unknown>) => {
    if (typeof window === 'undefined') return;
    tutorialPulseLogSeqRef.current += 1;
    const elapsed = window.performance.now().toFixed(1);
    if (payload) {
      console.log(
        `[tutorial-pulse-debug][${tutorialPulseLogSeqRef.current}] +${elapsed}ms ${event}`,
        payload,
      );
      return;
    }
    console.log(
      `[tutorial-pulse-debug][${tutorialPulseLogSeqRef.current}] +${elapsed}ms ${event}`,
    );
  }, []);
  const logMidpointFitFlow = useCallback(
    (_event: string, _payload?: Record<string, unknown>) => {},
    [],
  );

  useEffect(() => {
    let cancelled = false;
    let minDelayTimer = 0;
    let rafA = 0;
    let rafB = 0;
    let onLoad: (() => void) | null = null;

    const markBootReady = () => {
      if (cancelled) return;
      rafA = window.requestAnimationFrame(() => {
        rafB = window.requestAnimationFrame(() => {
          if (!cancelled) {
            setAppBootReady(true);
          }
        });
      });
    };

    const fontsReady =
      typeof document !== 'undefined' && 'fonts' in document
        ? (document as Document & { fonts: FontFaceSet }).fonts.ready.catch(() => undefined)
        : Promise.resolve();

    const loadReady =
      document.readyState === 'complete'
        ? Promise.resolve()
        : new Promise<void>((resolve) => {
            onLoad = () => resolve();
            window.addEventListener('load', onLoad, { once: true });
          });

    const minDelay = new Promise<void>((resolve) => {
      minDelayTimer = window.setTimeout(resolve, 700);
    });

    Promise.all([fontsReady, loadReady, minDelay]).then(markBootReady);

    return () => {
      cancelled = true;
      window.clearTimeout(minDelayTimer);
      if (rafA) window.cancelAnimationFrame(rafA);
      if (rafB) window.cancelAnimationFrame(rafB);
      if (onLoad) {
        window.removeEventListener('load', onLoad);
      }
    };
  }, []);

  const t = useMemo(() => createTranslator(locale), [locale]);
  const tutorialStep = useMemo<TutorialStep | null>(() => {
    if (tutorialStepIndex < 0 || tutorialStepIndex >= TUTORIAL_STEPS.length) return null;
    return TUTORIAL_STEPS[tutorialStepIndex] ?? null;
  }, [tutorialStepIndex]);
  const tutorialRunning = tutorialOpen && tutorialMode === 'running';
  const tutorialStepId = tutorialStep?.id ?? '';
  const tutorialActionSet = useMemo(
    () => new Set(tutorialStepId ? tutorialStepActionsById[tutorialStepId] ?? [] : []),
    [tutorialStepActionsById, tutorialStepId],
  );
  const tutorialOptionalSet = useMemo(
    () => new Set(tutorialStepId ? tutorialOptionalDecisionsByStep[tutorialStepId] ?? [] : []),
    [tutorialOptionalDecisionsByStep, tutorialStepId],
  );
  const tutorialChapter = useMemo(() => {
    if (!tutorialStep) return null;
    return TUTORIAL_CHAPTERS.find((chapter) => chapter.id === tutorialStep.chapterId) ?? null;
  }, [tutorialStep]);
  const tutorialChapterIndex = useMemo(() => {
    if (!tutorialChapter) return 0;
    return Math.max(0, TUTORIAL_CHAPTERS.findIndex((chapter) => chapter.id === tutorialChapter.id)) + 1;
  }, [tutorialChapter]);
  const tutorialChecklist = useMemo(() => {
    if (!tutorialStep) return [];
    return tutorialStep.required.map((item) => ({
      actionId: item.actionId,
      label: item.label,
      details: item.details,
      done: tutorialActionSet.has(item.actionId),
      kind: item.kind,
    }));
  }, [tutorialActionSet, tutorialStep]);
  const tutorialNextRequiredActionId = useMemo(() => {
    if (!tutorialStep) return null;
    const pending = tutorialStep.required.find((item) => !tutorialActionSet.has(item.actionId));
    return pending?.actionId ?? null;
  }, [tutorialActionSet, tutorialStep]);
  const tutorialOriginConfirmed = useMemo(
    () => tutorialActionSet.has('map.confirm_origin_newcastle'),
    [tutorialActionSet],
  );
  const tutorialDestinationConfirmed = useMemo(
    () => tutorialActionSet.has('map.confirm_destination_london'),
    [tutorialActionSet],
  );
  const tutorialOriginPlaced = useMemo(
    () => tutorialActionSet.has('map.set_origin_newcastle'),
    [tutorialActionSet],
  );
  const tutorialDestinationPlaced = useMemo(
    () => tutorialActionSet.has('map.set_destination_london'),
    [tutorialActionSet],
  );
  const tutorialPlacementStage = useMemo<TutorialPlacementStage>(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return 'done';
    if (!tutorialOriginConfirmed) return 'newcastle_origin';
    if (!tutorialDestinationConfirmed) return 'london_destination';
    if (tutorialNextRequiredActionId) return 'post_confirm_marker_actions';
    return 'done';
  }, [
    tutorialNextRequiredActionId,
    tutorialDestinationConfirmed,
    tutorialOriginConfirmed,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  const isTutorialPinDraftMode = useMemo(
    () => tutorialRunning && tutorialStep?.id === 'map_set_pins' && tutorialPlacementStage !== 'done',
    [tutorialPlacementStage, tutorialRunning, tutorialStep?.id],
  );
  const tutorialBlockingActionId = useMemo(() => {
    if (!tutorialRunning || !tutorialStep) return null;
    if (tutorialStep.id !== 'map_set_pins') return tutorialNextRequiredActionId;
    if (tutorialPlacementStage === 'newcastle_origin') {
      if (!tutorialDraftOrigin && !origin) return 'map.set_origin_newcastle';
      return 'map.confirm_origin_newcastle';
    }
    if (tutorialPlacementStage === 'london_destination') {
      if (!tutorialDraftDestination && !destination) return 'map.set_destination_london';
      return 'map.confirm_destination_london';
    }
    return tutorialNextRequiredActionId;
  }, [
    destination,
    origin,
    tutorialDraftDestination,
    tutorialDraftOrigin,
    tutorialNextRequiredActionId,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep,
  ]);
  const tutorialGuidePanNonce = useMemo(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return tutorialStepIndex;
    if (tutorialPlacementStage === 'newcastle_origin') return tutorialStepIndex * 1000 + 1;
    if (tutorialPlacementStage === 'london_destination') return tutorialStepIndex * 1000 + 2;
    if (tutorialPlacementStage === 'post_confirm_marker_actions') {
      const actionToOffset: Record<string, number> = {
        'map.click_destination_marker': 31,
        'map.popup_close_destination_marker': 32,
        'map.click_origin_marker': 33,
        'map.popup_close_origin_marker': 34,
        'map.drag_destination_marker': 35,
        'map.confirm_drag_destination_marker': 36,
        'map.drag_origin_marker': 37,
        'map.confirm_drag_origin_marker': 38,
      };
      return tutorialStepIndex * 1000 + (actionToOffset[tutorialBlockingActionId ?? ''] ?? 39);
    }
    return tutorialStepIndex * 1000 + 4;
  }, [tutorialBlockingActionId, tutorialPlacementStage, tutorialRunning, tutorialStep?.id, tutorialStepIndex]);
  const tutorialOptionalState = useMemo(() => {
    if (!tutorialStep?.optional) return null;
    const resolved = tutorialOptionalSet.has(tutorialStep.optional.id);
    const actionTouched = tutorialStep.optional.actionIds.some((id) => tutorialActionSet.has(id));
    return {
      id: tutorialStep.optional.id,
      label: tutorialStep.optional.label,
      resolved,
      defaultLabel: tutorialStep.optional.defaultLabel,
      actionTouched,
    };
  }, [tutorialOptionalSet, tutorialActionSet, tutorialStep]);
  const tutorialBaseLockScope = useMemo(() => inferTutorialLockScope(tutorialStep), [tutorialStep]);
  const tutorialLockScope = useMemo<TutorialLockScope>(() => {
    if (!tutorialRunning || !tutorialStep) return tutorialBaseLockScope;
    if (tutorialStep.id !== 'map_stop_lifecycle') return tutorialBaseLockScope;
    const blockingAction = tutorialBlockingActionId ?? tutorialNextRequiredActionId ?? '';
    if (blockingAction === 'pins.add_stop') {
      return 'sidebar_section_only';
    }
    if (blockingAction.startsWith('map.')) {
      return 'map_only';
    }
    return tutorialBaseLockScope;
  }, [
    tutorialBaseLockScope,
    tutorialBlockingActionId,
    tutorialNextRequiredActionId,
    tutorialRunning,
    tutorialStep,
  ]);
  const tutorialActiveSectionId = useMemo(() => inferTutorialSectionId(tutorialStep), [tutorialStep]);
  const tutorialAllowedActionSet = useMemo(() => {
    if (!tutorialStep) return new Set<string>();
    const orderedAction = tutorialBlockingActionId;
    const fromRequired = orderedAction ? [orderedAction] : [];
    const fromOptional = orderedAction ? [] : tutorialStep.optional?.actionIds ?? [];
    const fromCustom = tutorialStep.allowedActions ?? [];
    return new Set<string>([...fromRequired, ...fromOptional, ...fromCustom]);
  }, [tutorialBlockingActionId, tutorialStep]);
  const tutorialAllowedActionPrefixes = useMemo(() => {
    return [...tutorialAllowedActionSet]
      .filter((actionId) => actionId.endsWith('*'))
      .map((actionId) => actionId.slice(0, -1));
  }, [tutorialAllowedActionSet]);
  const tutorialAllowedActionExact = useMemo(() => {
    return new Set(
      [...tutorialAllowedActionSet].filter((actionId) => !actionId.endsWith('*')),
    );
  }, [tutorialAllowedActionSet]);
  const tutorialTargetIdSet = useMemo(() => {
    return new Set<string>(tutorialStep?.targetIds ?? []);
  }, [tutorialStep]);
  const tutorialUsesSectionTarget = useMemo(
    () =>
      (tutorialStep?.targetIds ?? []).some((targetId) =>
        TUTORIAL_SECTION_IDS.includes(targetId as TutorialSectionId),
      ),
    [tutorialStep],
  );
  const tutorialSidebarLocked = tutorialRunning && tutorialLockScope === 'map_only';
  const tutorialSectionControlFor = useCallback(
    (sectionId: TutorialSectionId): TutorialSectionControl | undefined => {
      if (!tutorialRunning) return undefined;
      if (tutorialLockScope === 'map_only') {
        const keepPinsOpen = tutorialStep?.id === 'map_stop_lifecycle' && sectionId === 'pins.section';
        return {
          isOpen: keepPinsOpen,
          lockToggle: true,
          tutorialLocked: true,
        };
      }
      if (tutorialLockScope === 'sidebar_section_only') {
        const isActive = tutorialActiveSectionId === sectionId;
        return {
          isOpen: isActive,
          lockToggle: true,
          tutorialLocked: !isActive,
        };
      }
      return undefined;
    },
    [tutorialActiveSectionId, tutorialLockScope, tutorialRunning, tutorialStep?.id],
  );
  const tutorialSectionControl = useMemo(
    () => ({
      setup: tutorialSectionControlFor('setup.section'),
      pins: tutorialSectionControlFor('pins.section'),
      advanced: tutorialSectionControlFor('advanced.section'),
      preferences: tutorialSectionControlFor('preferences.section'),
      selectedRoute: tutorialSectionControlFor('selected.route_panel'),
      routes: tutorialSectionControlFor('routes.list'),
      compare: tutorialSectionControlFor('compare.section'),
      departure: tutorialSectionControlFor('departure.section'),
      duty: tutorialSectionControlFor('duty.section'),
      oracle: tutorialSectionControlFor('oracle.section'),
      experiments: tutorialSectionControlFor('experiments.section'),
      timelapse: tutorialSectionControlFor('timelapse.section'),
    }),
    [tutorialSectionControlFor],
  );
  const tutorialMapLocked =
    tutorialRunning &&
    tutorialLockScope === 'sidebar_section_only';
  const tutorialViewportLocked = tutorialRunning;
  const tutorialHideZoomControls = tutorialRunning;
  const tutorialRelaxBounds =
    tutorialRunning &&
    (tutorialStep?.id === 'map_stop_lifecycle' || tutorialMapLocked);
  const tutorialMapDimmed = false;
  const tutorialGuideTarget = useMemo<TutorialGuideTarget | null>(() => {
    if (!tutorialRunning) return null;
    if (tutorialStep?.id === 'map_stop_lifecycle') {
      if (tutorialBlockingActionId !== 'map.drag_stop_marker') return null;
      return {
        lat: TUTORIAL_CITY_TARGETS.stoke.lat,
        lon: TUTORIAL_CITY_TARGETS.stoke.lon,
        radius_km: TUTORIAL_CITY_TARGETS.stoke.radiusKm,
        label: 'Drag Midpoint Toward Stoke-on-Trent',
        stage: 3,
        pan_nonce: tutorialGuidePanNonce,
        zoom: TUTORIAL_CITY_TARGETS.stoke.zoom,
      };
    }
    if (tutorialStep?.id !== 'map_set_pins') return null;
    if (tutorialPlacementStage === 'done') return null;
    if (tutorialPlacementStage === 'newcastle_origin') {
      return {
        lat: TUTORIAL_CITY_TARGETS.newcastle.lat,
        lon: TUTORIAL_CITY_TARGETS.newcastle.lon,
        radius_km: TUTORIAL_CITY_TARGETS.newcastle.radiusKm,
        label: 'Place Start Near Newcastle',
        stage: 1,
        pan_nonce: tutorialGuidePanNonce,
        zoom: TUTORIAL_CITY_TARGETS.newcastle.zoom,
      };
    }
    if (tutorialPlacementStage === 'post_confirm_marker_actions') {
      const action = tutorialBlockingActionId ?? '';
      const isOriginTask =
        action === 'map.click_origin_marker' ||
        action === 'map.popup_close_origin_marker' ||
        action === 'map.drag_origin_marker' ||
        action === 'map.confirm_drag_origin_marker';
      const city = isOriginTask ? TUTORIAL_CITY_TARGETS.newcastle : TUTORIAL_CITY_TARGETS.london;
      let label = isOriginTask ? 'Click Start Marker' : 'Click End Marker';
      if (action === 'map.popup_close_destination_marker') label = 'Close End Popup';
      if (action === 'map.popup_close_origin_marker') label = 'Close Start Popup';
      if (action === 'map.drag_destination_marker') label = 'Drag End In London Zone';
      if (action === 'map.drag_origin_marker') label = 'Drag Start In Newcastle Zone';
      if (action === 'map.confirm_drag_destination_marker') label = 'Confirm End Drag';
      if (action === 'map.confirm_drag_origin_marker') label = 'Confirm Start Drag';
      const stage = isOriginTask ? 1 : 2;
      return {
        lat: city.lat,
        lon: city.lon,
        radius_km: city.radiusKm,
        label,
        stage,
        pan_nonce: tutorialGuidePanNonce,
        zoom: city.zoom,
      };
    }
    return {
      lat: TUTORIAL_CITY_TARGETS.london.lat,
      lon: TUTORIAL_CITY_TARGETS.london.lon,
      radius_km: TUTORIAL_CITY_TARGETS.london.radiusKm,
      label: 'Place End Near London',
      stage: 2,
      pan_nonce: tutorialGuidePanNonce,
      zoom: TUTORIAL_CITY_TARGETS.london.zoom,
    };
  }, [tutorialBlockingActionId, tutorialGuidePanNonce, tutorialPlacementStage, tutorialRunning, tutorialStep?.id]);
  const tutorialGuideVisible = Boolean(tutorialGuideTarget);
  const showPreviewConnector = !tutorialRunning || tutorialStep?.id !== 'map_set_pins';
  const mapOriginForRender = useMemo(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return origin;
    if (tutorialPlacementStage === 'post_confirm_marker_actions' && tutorialDragDraftOrigin) {
      return tutorialDragDraftOrigin;
    }
    return tutorialOriginPlaced ? origin : null;
  }, [
    origin,
    tutorialDragDraftOrigin,
    tutorialOriginPlaced,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  const mapDestinationForRender = useMemo(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return destination;
    if (tutorialPlacementStage === 'post_confirm_marker_actions' && tutorialDragDraftDestination) {
      return tutorialDragDraftDestination;
    }
    return tutorialDestinationPlaced ? destination : null;
  }, [
    destination,
    tutorialDestinationPlaced,
    tutorialDragDraftDestination,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  const tutorialConfirmPin = useMemo<'origin' | 'destination' | null>(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return null;
    if (
      tutorialPlacementStage === 'newcastle_origin' &&
      (tutorialDraftOrigin || (tutorialOriginPlaced && origin))
    ) {
      return 'origin';
    }
    if (
      tutorialPlacementStage === 'london_destination' &&
      (tutorialDraftDestination || (tutorialDestinationPlaced && destination))
    ) {
      return 'destination';
    }
    return null;
  }, [
    destination,
    origin,
    tutorialDraftDestination,
    tutorialDraftOrigin,
    tutorialDestinationPlaced,
    tutorialOriginPlaced,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  const tutorialDragConfirmPin = useMemo<'origin' | 'destination' | null>(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return null;
    if (tutorialPlacementStage !== 'post_confirm_marker_actions') return null;
    if (
      tutorialBlockingActionId === 'map.confirm_drag_destination_marker' &&
      tutorialDragDraftDestination
    ) {
      return 'destination';
    }
    if (
      tutorialBlockingActionId === 'map.confirm_drag_origin_marker' &&
      tutorialDragDraftOrigin
    ) {
      return 'origin';
    }
    return null;
  }, [
    tutorialBlockingActionId,
    tutorialDragDraftDestination,
    tutorialDragDraftOrigin,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  const tutorialCurrentTaskOverride = useMemo(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return null;
    if (tutorialBlockingActionId === 'map.confirm_origin_newcastle') {
      return 'Confirm Start using the marker-adjacent button on the map.';
    }
    if (tutorialBlockingActionId === 'map.confirm_destination_london') {
      return 'Confirm End using the marker-adjacent button on the map.';
    }
    return null;
  }, [tutorialBlockingActionId, tutorialRunning, tutorialStep?.id]);

  useEffect(() => {
    if (!tutorialRunning || !tutorialStep?.id) {
      tutorialStepEnterRef.current = null;
      return;
    }
    if (tutorialStepEnterRef.current === tutorialStep.id) {
      return;
    }
    tutorialStepEnterRef.current = tutorialStep.id;

    if (tutorialStep.id === 'map_stop_lifecycle') {
      const stableOrigin = origin ?? tutorialConfirmedOriginRef.current;
      const stableDestination = destination ?? tutorialConfirmedDestinationRef.current;
      logMidpointFitFlow('step-enter', {
        stepId: tutorialStep.id,
        origin,
        destination,
        stableOrigin,
        stableDestination,
        managedStop,
        tutorialBlockingActionId,
        tutorialNextRequiredActionId,
        tutorialLockScope,
        tutorialMapLocked,
        tutorialMapDimmed,
        tutorialViewportLocked,
      });
      logTutorialMidpoint('step-enter:map_stop_lifecycle', {
        origin,
        destination,
        stableOrigin,
        stableDestination,
        confirmedOriginRef: tutorialConfirmedOriginRef.current,
        confirmedDestinationRef: tutorialConfirmedDestinationRef.current,
        tutorialBlockingActionId,
        tutorialNextRequiredActionId,
      });
      if (!origin && stableOrigin) {
        setOrigin(stableOrigin);
      }
      if (!destination && stableDestination) {
        setDestination(stableDestination);
      }
      if (stableOrigin && stableDestination) {
        const canonicalDutyText = serializePinsToDutyText(
          stableOrigin,
          buildDutyStopsForTextSync(dutyStopsText, managedStop),
          stableDestination,
        );
        dutySyncSourceRef.current = 'pins';
        setDutyStopsText(canonicalDutyText);
        setDutySyncError(null);
        logTutorialMidpoint('step-enter:map_stop_lifecycle-sync-duty-text', {
          canonicalDutyText,
        });
      }
      setSelectedPinId(null);
      setFocusPinRequest(null);
      setLiveMessage('Map locked. Use Pins & Stops and click Add Stop to place a midpoint.');
      window.requestAnimationFrame(() => {
        setFitAllRequestNonce((prev) => {
          const next = prev + 1;
          logMidpointFitFlow('fit-nonce:step-enter-raf', {
            prev,
            next,
            originAfterStabilize: stableOrigin,
            destinationAfterStabilize: stableDestination,
            hasStop: Boolean(managedStop),
          });
          return next;
        });
      });
    }
  }, [
    destination,
    logMidpointFitFlow,
    logTutorialMidpoint,
    dutyStopsText,
    managedStop,
    origin,
    tutorialBlockingActionId,
    tutorialLockScope,
    tutorialMapDimmed,
    tutorialMapLocked,
    tutorialNextRequiredActionId,
    tutorialRunning,
    tutorialStep?.id,
    tutorialViewportLocked,
  ]);

  useEffect(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_stop_lifecycle') return;
    if (tutorialBlockingActionId === 'pins.add_stop') {
      setLiveMessage('Map locked. Use Pins & Stops and click Add Stop to place a midpoint.');
      return;
    }
    if (tutorialBlockingActionId === 'map.click_stop_marker') {
      setLiveMessage('Map unlocked for this task. Click the midpoint marker once.');
      return;
    }
    if (tutorialBlockingActionId === 'map.popup_close') {
      setLiveMessage('Close the midpoint popup to continue.');
      return;
    }
    if (tutorialBlockingActionId === 'map.drag_stop_marker') {
      setLiveMessage('Drag the midpoint marker once to complete this task.');
      return;
    }
  }, [tutorialBlockingActionId, tutorialRunning, tutorialStep?.id]);

  useEffect(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_stop_lifecycle') return;
    if (tutorialBlockingActionId !== 'map.drag_stop_marker') return;
    if (selectedPinId !== null) {
      setSelectedPinId(null);
    }
  }, [selectedPinId, tutorialBlockingActionId, tutorialRunning, tutorialStep?.id]);

  useEffect(() => {
    if (!tutorialRunning) {
      tutorialRouteCenteredStepRef.current = null;
      return;
    }
    if (!origin || !destination) return;
    if (!tutorialStep?.id) return;
    if (tutorialStep.id === 'map_set_pins') return;
    if (tutorialRouteCenteredStepRef.current === tutorialStep.id) return;
    tutorialRouteCenteredStepRef.current = tutorialStep.id;
    setFitAllRequestNonce((prev) => prev + 1);
    logTutorialMidpoint('route-center:sidebar-step-refit', {
      stepId: tutorialStep.id,
      origin,
      destination,
      managedStop,
    });
  }, [
    destination,
    logTutorialMidpoint,
    managedStop,
    origin,
    tutorialRunning,
    tutorialStep?.id,
  ]);

  useEffect(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_stop_lifecycle') return;
    if (!origin || !destination) {
      logMidpointFitFlow('fit-nonce:timer-skip-missing-pins', {
        hasOrigin: Boolean(origin),
        hasDestination: Boolean(destination),
        managedStop,
      });
      return;
    }
    logMidpointFitFlow('fit-nonce:timer-scheduled', {
      origin,
      destination,
      managedStop,
      delayMs: 280,
    });
    const timer = window.setTimeout(() => {
      setFitAllRequestNonce((prev) => {
        const next = prev + 1;
        logMidpointFitFlow('fit-nonce:timer-fired', {
          prev,
          next,
          origin,
          destination,
          managedStop,
        });
        return next;
      });
    }, 280);
    return () => {
      window.clearTimeout(timer);
      logMidpointFitFlow('fit-nonce:timer-cleared');
    };
  }, [destination, logMidpointFitFlow, origin, tutorialRunning, tutorialStep?.id, managedStop]);

  useEffect(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') {
      tutorialPlacementStageRef.current = 'done';
      return;
    }

    const previousStage = tutorialPlacementStageRef.current;
    if (previousStage === tutorialPlacementStage) return;
    tutorialPlacementStageRef.current = tutorialPlacementStage;

    if (tutorialPlacementStage === 'london_destination' && !tutorialDestinationPlaced) {
      tutorialMapClickGuardUntilRef.current = Date.now() + 700;
      setDestination(null);
      setTutorialDraftDestination(null);
      setTutorialDragDraftDestination(null);
      setTutorialDragDraftOrigin(null);
      setSelectedPinId(null);
      setFocusPinRequest(null);
      return;
    }

    if (tutorialPlacementStage === 'post_confirm_marker_actions') {
      setSelectedPinId(null);
      setFocusPinRequest(null);
      setTutorialDragDraftOrigin(null);
      setTutorialDragDraftDestination(null);
    }
  }, [
    destination,
    tutorialDestinationPlaced,
    tutorialDragDraftDestination,
    tutorialDraftDestination,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep?.id,
  ]);

  useEffect(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return;
    if (tutorialPlacementStage === 'newcastle_origin' && !tutorialDraftOrigin && !origin) {
      setLiveMessage('Place Start inside the Newcastle guided zone.');
      return;
    }
    if (tutorialPlacementStage === 'newcastle_origin' && (tutorialDraftOrigin || origin) && !tutorialOriginConfirmed) {
      setLiveMessage('Start placed. Click the Confirm Start button near the marker to continue.');
      return;
    }
    if (tutorialPlacementStage === 'london_destination' && !tutorialDraftDestination && !destination) {
      setLiveMessage('Great. Now place End inside the London guided zone.');
      return;
    }
    if (
      tutorialPlacementStage === 'london_destination' &&
      (tutorialDraftDestination || destination) &&
      !tutorialDestinationConfirmed
    ) {
      setLiveMessage('End placed. Click the Confirm End button near the marker to continue.');
      return;
    }
    if (tutorialPlacementStage === 'post_confirm_marker_actions') {
      if (tutorialBlockingActionId === 'map.click_destination_marker') {
        setLiveMessage('End confirmed. Click the End marker once.');
        return;
      }
      if (tutorialBlockingActionId === 'map.popup_close_destination_marker') {
        setLiveMessage('Close the End marker popup to continue.');
        return;
      }
      if (tutorialBlockingActionId === 'map.click_origin_marker') {
        setLiveMessage('Now click the Start marker once.');
        return;
      }
      if (tutorialBlockingActionId === 'map.popup_close_origin_marker') {
        setLiveMessage('Close the Start marker popup to continue.');
        return;
      }
      if (tutorialBlockingActionId === 'map.popup_copy') {
        setLiveMessage('Copy coordinates from the marker popup.');
        return;
      }
      if (tutorialBlockingActionId === 'map.drag_destination_marker') {
        setLiveMessage('Drag End inside the London guided zone.');
        return;
      }
      if (tutorialBlockingActionId === 'map.confirm_drag_destination_marker') {
        setLiveMessage('Use Confirm End Drag near the marker.');
        return;
      }
      if (tutorialBlockingActionId === 'map.drag_origin_marker') {
        setLiveMessage('Drag Start inside the Newcastle guided zone.');
        return;
      }
      if (tutorialBlockingActionId === 'map.confirm_drag_origin_marker') {
        setLiveMessage('Use Confirm Start Drag near the marker.');
        return;
      }
      setLiveMessage('Complete the remaining marker workflow in order.');
      return;
    }
  }, [
    destination,
    origin,
    tutorialBlockingActionId,
    tutorialDraftDestination,
    tutorialDraftOrigin,
    tutorialDestinationConfirmed,
    tutorialOriginConfirmed,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  useEffect(() => {
    if (isTutorialPinDraftMode) return;
    if (!tutorialDraftOrigin && !tutorialDraftDestination) return;
    setTutorialDraftOrigin(null);
    setTutorialDraftDestination(null);
  }, [isTutorialPinDraftMode, tutorialDraftDestination, tutorialDraftOrigin]);
  useEffect(() => {
    const dragDraftMode =
      tutorialRunning &&
      tutorialStep?.id === 'map_set_pins' &&
      tutorialPlacementStage === 'post_confirm_marker_actions';
    if (dragDraftMode) return;
    if (!tutorialDragDraftOrigin && !tutorialDragDraftDestination) return;
    setTutorialDragDraftOrigin(null);
    setTutorialDragDraftDestination(null);
  }, [
    tutorialDragDraftDestination,
    tutorialDragDraftOrigin,
    tutorialPlacementStage,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  const tutorialCanAdvance = useMemo(() => {
    if (!tutorialStep) return false;
    const requiredDone = tutorialStep.required.every((item) => tutorialActionSet.has(item.actionId));
    if (!requiredDone) return false;
    if (
      tutorialStep.id === 'map_set_pins' &&
      (!tutorialOriginConfirmed || !tutorialDestinationConfirmed)
    ) {
      return false;
    }
    if (!tutorialOptionalState) return true;
    return tutorialOptionalState.resolved || tutorialOptionalState.actionTouched;
  }, [
    tutorialActionSet,
    tutorialDestinationConfirmed,
    tutorialOptionalState,
    tutorialOriginConfirmed,
    tutorialStep,
  ]);
  const tutorialAtStart = tutorialStepIndex <= 0;
  const tutorialAtEnd = tutorialStepIndex >= TUTORIAL_STEPS.length - 1;
  const weightSum = weights.time + weights.money + weights.co2;

  useEffect(() => {
    if (!tutorialRunning || !tutorialStep?.id) {
      tutorialAutoAdvanceTokenRef.current = '';
      return;
    }
    if (!tutorialCanAdvance) {
      tutorialAutoAdvanceTokenRef.current = '';
      return;
    }
    const token = `${tutorialStep.id}:${tutorialStepIndex}:${tutorialActionSet.size}:${tutorialOptionalState?.resolved ? '1' : '0'}:${tutorialOptionalState?.actionTouched ? '1' : '0'}`;
    if (tutorialAutoAdvanceTokenRef.current === token) return;
    tutorialAutoAdvanceTokenRef.current = token;

    const timer = window.setTimeout(() => {
      setSelectedPinId(null);
      setFocusPinRequest(null);
      if (tutorialAtEnd) {
        tutorialFinish();
      } else {
        tutorialNext();
      }
    }, 420);

    return () => {
      window.clearTimeout(timer);
    };
  }, [
    tutorialActionSet.size,
    tutorialAtEnd,
    tutorialCanAdvance,
    tutorialOptionalState?.actionTouched,
    tutorialOptionalState?.resolved,
    tutorialRunning,
    tutorialStep?.id,
    tutorialStepIndex,
  ]);

  useEffect(() => {
    if (!tutorialRunning) {
      tutorialPreviousStepIdRef.current = null;
      return;
    }
    const currentStepId = tutorialStep?.id ?? null;
    const previousStepId = tutorialPreviousStepIdRef.current;
    if (currentStepId && previousStepId === 'map_stop_lifecycle' && currentStepId !== 'map_stop_lifecycle') {
      setSelectedPinId(null);
      setFocusPinRequest(null);
      setFitAllRequestNonce((prev) => prev + 1);
    }
    tutorialPreviousStepIdRef.current = currentStepId;
  }, [tutorialRunning, tutorialStep?.id]);

  const clearFlushTimer = useCallback(() => {
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }, []);

  const tutorialActionValueSatisfied = useCallback(
    (actionId: string): boolean => {
      const parseNumber = (raw: string): number | null => {
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
      };
      const nearlyEqual = (a: number | null, b: number, tolerance = 1e-3): boolean =>
        a !== null && Math.abs(a - b) <= tolerance;

      switch (actionId) {
        case 'advanced.risk_aversion_input':
          return nearlyEqual(parseNumber(advancedParams.riskAversion), 1.4, 1e-4);
        case 'advanced.epsilon_duration_input':
          return nearlyEqual(parseNumber(advancedParams.epsilonDurationS), 9000, 1e-3);
        case 'advanced.epsilon_money_input':
          return nearlyEqual(parseNumber(advancedParams.epsilonMonetaryCost), 250, 1e-3);
        case 'advanced.epsilon_emissions_input':
          return nearlyEqual(parseNumber(advancedParams.epsilonEmissionsKg), 900, 1e-3);
        case 'advanced.stochastic_seed_input':
          return nearlyEqual(parseNumber(advancedParams.stochasticSeed), 42, 1e-6);
        case 'advanced.stochastic_sigma_input':
          return nearlyEqual(parseNumber(advancedParams.stochasticSigma), 0.08, 1e-5);
        case 'advanced.stochastic_samples_input':
          return nearlyEqual(parseNumber(advancedParams.stochasticSamples), 25, 1e-6);
        case 'advanced.use_tolls_toggle':
          return advancedParams.useTolls;
        case 'advanced.fuel_multiplier_input':
          return nearlyEqual(parseNumber(advancedParams.fuelPriceMultiplier), 1.1, 1e-5);
        case 'advanced.carbon_price_input':
          return nearlyEqual(parseNumber(advancedParams.carbonPricePerKg), 0.08, 1e-5);
        case 'advanced.toll_per_km_input':
          return nearlyEqual(parseNumber(advancedParams.tollCostPerKm), 0.12, 1e-5);
        case 'pref.weight_time':
          return weights.time === 55;
        case 'pref.weight_money':
          return weights.money === 25;
        case 'pref.weight_co2':
          return weights.co2 === 20;
        default:
          return true;
      }
    },
    [
      advancedParams.carbonPricePerKg,
      advancedParams.epsilonDurationS,
      advancedParams.epsilonEmissionsKg,
      advancedParams.epsilonMonetaryCost,
      advancedParams.fuelPriceMultiplier,
      advancedParams.riskAversion,
      advancedParams.stochasticSamples,
      advancedParams.stochasticSeed,
      advancedParams.stochasticSigma,
      advancedParams.tollCostPerKm,
      advancedParams.useTolls,
      weights.co2,
      weights.money,
      weights.time,
    ],
  );

  const markTutorialAction = useCallback(
    (actionId: string, options?: { force?: boolean }) => {
      if (!actionId) return;
      if (!tutorialRunning || !tutorialStepId) {
        return;
      }
      if (TUTORIAL_FORCE_ONLY_ACTIONS.has(actionId) && !options?.force) {
        return;
      }
      if (!options?.force && !tutorialActionValueSatisfied(actionId)) {
        return;
      }
      const nextRequiredAction = tutorialBlockingActionId ?? tutorialNextRequiredActionId;
      const actionAlreadyDone = tutorialActionSet.has(actionId);
      if (!options?.force && !actionAlreadyDone && nextRequiredAction && actionId !== nextRequiredAction) {
        return;
      }
      setTutorialStepActionsById((prev) => {
        const existing = prev[tutorialStepId] ?? [];
        if (existing.includes(actionId)) {
          return prev;
        }
        return { ...prev, [tutorialStepId]: [...existing, actionId] };
      });
    },
    [
      tutorialActionSet,
      tutorialBlockingActionId,
      tutorialNextRequiredActionId,
      tutorialActionValueSatisfied,
      tutorialRunning,
      tutorialStepId,
    ],
  );

  useEffect(() => {
    if (!tutorialRunning) return;
    const actionId = tutorialBlockingActionId;
    if (!actionId) return;
    if (!TUTORIAL_VALUE_CONSTRAINED_ACTIONS.has(actionId)) return;
    if (tutorialActionSet.has(actionId)) return;
    if (!tutorialActionValueSatisfied(actionId)) return;
    markTutorialAction(actionId);
  }, [
    markTutorialAction,
    tutorialActionSet,
    tutorialActionValueSatisfied,
    tutorialBlockingActionId,
    tutorialRunning,
  ]);

  const markTutorialOptionalDefault = useCallback(
    (decisionId: string) => {
      if (!decisionId || !tutorialRunning || !tutorialStepId) return;
      setTutorialOptionalDecisionsByStep((prev) => {
        const existing = prev[tutorialStepId] ?? [];
        if (existing.includes(decisionId)) return prev;
        return { ...prev, [tutorialStepId]: [...existing, decisionId] };
      });
    },
    [tutorialRunning, tutorialStepId],
  );

  const applyTutorialPrefill = useCallback(
    (prefillId: TutorialStep['prefillId']) => {
      if (!prefillId) return;
      if (tutorialPrefilledSteps.includes(prefillId)) return;

      if (prefillId === 'clear_map') {
        dutySyncSourceRef.current = 'text';
        setOrigin(null);
        setDestination(null);
        setTutorialDraftOrigin(null);
        setTutorialDraftDestination(null);
        setTutorialDragDraftOrigin(null);
        setTutorialDragDraftDestination(null);
        setManagedStop(null);
        setSelectedPinId(null);
        setFocusPinRequest(null);
        setDutyStopsText('');
        setDutySyncError(null);
      }

      if (prefillId === 'canonical_map') {
        setOrigin(TUTORIAL_CANONICAL_ORIGIN);
        setDestination(TUTORIAL_CANONICAL_DESTINATION);
        setTutorialDraftOrigin(null);
        setTutorialDraftDestination(null);
        setTutorialDragDraftOrigin(null);
        setTutorialDragDraftDestination(null);
        setManagedStop(null);
      }

      if (prefillId === 'canonical_setup') {
        setVehicleType('');
        setScenarioMode('partial_sharing');
      }

      if (prefillId === 'canonical_advanced') {
        setAdvancedParams((prev) => ({
          ...prev,
          optimizationMode: 'expected_value',
          riskAversion: '1.0',
          paretoMethod: 'dominance',
          epsilonDurationS: '',
          epsilonMonetaryCost: '',
          epsilonEmissionsKg: '',
          departureTimeUtcLocal: '',
          stochasticEnabled: false,
          stochasticSeed: '',
          stochasticSigma: '0.05',
          stochasticSamples: '10',
          terrainProfile: 'flat',
          useTolls: false,
          fuelPriceMultiplier: '1.0',
          carbonPricePerKg: '0.0',
          tollCostPerKm: '0.0',
        }));
      }

      if (prefillId === 'canonical_preferences') {
        setWeights({ time: 60, money: 20, co2: 20 });
      }

      if (prefillId === 'canonical_departure') {
        const win = defaultDepartureWindow();
        setDepWindowStartLocal(win.start);
        setDepWindowEndLocal(win.end);
        setDepStepMinutes(60);
        setDepEarliestArrivalLocal('');
        setDepLatestArrivalLocal('');
      }

      if (prefillId === 'canonical_duty') {
        dutySyncSourceRef.current = 'text';
        setDutyStopsText(TUTORIAL_CANONICAL_DUTY_STOPS);
      }

      if (prefillId === 'canonical_oracle') {
        // Oracle panel carries canonical defaults in-component for tutorial mode.
      }

      if (prefillId === 'canonical_experiment') {
        setTutorialExperimentPrefill({
          name: 'Tutorial Full Walkthrough',
          description: 'Created by guided tutorial',
        });
      }

      setTutorialPrefilledSteps((prev) => (prev.includes(prefillId) ? prev : [...prev, prefillId]));
    },
    [tutorialPrefilledSteps],
  );

  const flushRouteBuffer = useCallback(
    (seq: number) => {
      if (seq !== requestSeqRef.current) return;
      const pending = routeBufferRef.current;
      if (!pending.length) return;

      routeBufferRef.current = [];
      startTransition(() => {
        setParetoRoutes((prev) => {
          if (seq !== requestSeqRef.current) return prev;
          const seen = new Set(prev.map((route) => route.id));
          const merged = [...prev];

          for (const route of pending) {
            if (seen.has(route.id)) continue;
            seen.add(route.id);
            merged.push(route);
          }

          return merged;
        });
      });
    },
    [startTransition],
  );

  const scheduleRouteFlush = useCallback(
    (seq: number) => {
      if (flushTimerRef.current !== null) return;
      flushTimerRef.current = window.setTimeout(() => {
        flushTimerRef.current = null;
        flushRouteBuffer(seq);
      }, 80);
    },
    [flushRouteBuffer],
  );

  const flushRouteBufferNow = useCallback(
    (seq: number) => {
      clearFlushTimer();
      flushRouteBuffer(seq);
    },
    [clearFlushTimer, flushRouteBuffer],
  );

  const abortActiveCompute = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (activeAttemptAbortRef.current) {
      activeAttemptAbortRef.current.abort();
      activeAttemptAbortRef.current = null;
    }
    routeBufferRef.current = [];
    clearFlushTimer();
  }, [clearFlushTimer]);

  const resetComputeTrace = useCallback(() => {
    computeTraceSeqRef.current = 0;
    computeStartedAtMsRef.current = null;
    setComputeTrace([]);
    setComputeSession(null);
    setComputeLiveCallsByAttempt({});
    setComputeRequestIdByAttempt({});
  }, []);

  const appendComputeTrace = useCallback(
    (entry: Omit<ComputeTraceEntry, 'id' | 'at' | 'elapsedMs'>) => {
      computeTraceSeqRef.current += 1;
      const row: ComputeTraceEntry = {
        ...entry,
        id: computeTraceSeqRef.current,
        at: new Date().toISOString(),
        elapsedMs: computeStartedAtMsRef.current ? Date.now() - computeStartedAtMsRef.current : 0,
      };
      setComputeTrace((prev) => {
        const next = [...prev, row];
        return next.length > 300 ? next.slice(next.length - 300) : next;
      });
    },
    [],
  );

  const clearComputed = useCallback(() => {
    requestSeqRef.current += 1;
    abortActiveCompute();

    setLoading(false);
    setProgress(null);
    setWarnings([]);
    setShowWarnings(false);
    setError(null);
    setParetoRoutes([]);
    setSelectedId(null);

    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');
    setScenarioCompare(null);
    setScenarioCompareError(null);
    setScenarioCompareLoading(false);
    setDepOptimizeData(null);
    setDepOptimizeError(null);
    setDepOptimizeLoading(false);
    setDutyChainData(null);
    setDutyChainError(null);
    setTimeLapsePosition(null);
    setAdvancedError(null);
    setComputeTraceOpen(false);
    resetComputeTrace();
  }, [abortActiveCompute, resetComputeTrace]);

  const clearComputedFromUi = useCallback(() => {
    markTutorialAction('pref.clear_results_click');
    clearComputed();
  }, [clearComputed, markTutorialAction]);

  const selectRouteFromChart = useCallback(
    (routeId: string) => {
      setSelectedId(routeId);
      markTutorialAction('routes.select_chart');
    },
    [markTutorialAction],
  );

  const selectRouteFromCard = useCallback(
    (routeId: string) => {
      setSelectedId(routeId);
      markTutorialAction('routes.select_card');
    },
    [markTutorialAction],
  );

  const cancelCompute = useCallback(() => {
    requestSeqRef.current += 1;
    abortActiveCompute();
    setLoading(false);
    setProgress(null);
    appendComputeTrace({
      level: 'warn',
      step: 'Compute cancelled',
      detail: 'User cancelled the in-flight route computation.',
    });
  }, [abortActiveCompute, appendComputeTrace]);

  useEffect(() => {
    return () => {
      requestSeqRef.current += 1;
      abortActiveCompute();
    };
  }, [abortActiveCompute]);

  useEffect(() => {
    clearComputed();
  }, [vehicleType, scenarioMode, clearComputed]);

  useEffect(() => {
    const controller = new AbortController();

    void (async () => {
      try {
        await loadVehicles(controller.signal);
      } catch (e) {
        if (!controller.signal.aborted) {
          console.error('Failed to load vehicles:', e);
        }
      }
    })();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    void Promise.all([
      loadExperiments(),
      refreshOracleDashboard(),
      refreshOpsDiagnostics(),
      refreshCustomVehicles(),
    ]);
  }, []);

  useEffect(() => {
    const warmupState = String(opsHealthReady?.route_graph?.state ?? '').trim().toLowerCase();
    const pollMs = warmupState === 'loading' ? 1_500 : 5_000;
    const timer = window.setInterval(() => {
      void refreshRouteReadiness();
    }, pollMs);
    return () => window.clearInterval(timer);
  }, [opsHealthReady?.route_graph?.state]);

  useEffect(() => {
    if (depWindowStartLocal || depWindowEndLocal) return;
    const now = new Date();
    const start = new Date(now.getTime() + 60 * 60 * 1000);
    const end = new Date(now.getTime() + 6 * 60 * 60 * 1000);
    setDepWindowStartLocal(start.toISOString().slice(0, 16));
    setDepWindowEndLocal(end.toISOString().slice(0, 16));
  }, [depWindowStartLocal, depWindowEndLocal]);

  useEffect(() => {
    if (!isTutorialPinDraftMode) return;
    dutySyncSourceRef.current = null;
  }, [isTutorialPinDraftMode]);

  useEffect(() => {
    const preload = () => {
      void import('./components/ParetoChart');
      void import('./components/EtaTimelineChart');
      void import('./components/ScenarioComparison');
      void import('./components/SegmentBreakdown');
      void import('./components/TutorialOverlay');
    };
    const browserWindow = window as Window & {
      requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number;
      cancelIdleCallback?: (handle: number) => void;
    };
    if (browserWindow.requestIdleCallback) {
      const idleId = browserWindow.requestIdleCallback(preload, { timeout: 1200 });
      return () => browserWindow.cancelIdleCallback?.(idleId);
    }
    const timer = window.setTimeout(preload, 350);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (isTutorialPinDraftMode) return;
    if (dutySyncSourceRef.current === 'text') {
      logTutorialMidpoint('duty-sync:pins-to-text-skip:text-source');
      dutySyncSourceRef.current = null;
      return;
    }

    const syncedStops = buildDutyStopsForTextSync(dutyStopsText, managedStop);
    const nextText = serializePinsToDutyText(origin, syncedStops, destination);
    if (nextText === dutyStopsText) return;
    logTutorialMidpoint('duty-sync:pins-to-text-write', {
      nextText,
      previousText: dutyStopsText,
      origin,
      destination,
      managedStop,
      syncedStops,
      tutorialStepId: tutorialStep?.id ?? null,
    });
    dutySyncSourceRef.current = 'pins';
    setDutyStopsText(nextText);
    setDutySyncError(null);
  }, [
    destination,
    dutyStopsText,
    isTutorialPinDraftMode,
    logTutorialMidpoint,
    managedStop,
    origin,
    tutorialRunning,
    tutorialStep?.id,
  ]);

  useEffect(() => {
    if (isTutorialPinDraftMode) return;
    if (tutorialRunning && tutorialStep?.id === 'map_stop_lifecycle') {
      logTutorialMidpoint('duty-sync:text-to-pins-skipped-midpoint-step', {
        dutyStopsText,
        tutorialStepId: tutorialStep?.id ?? null,
        tutorialBlockingActionId,
      });
      return;
    }
    if (dutySyncSourceRef.current === 'pins') {
      logTutorialMidpoint('duty-sync:text-to-pins-skip:pins-source');
      dutySyncSourceRef.current = null;
      return;
    }
    let parsed: ParsedPinSync;
    try {
      parsed = parseDutyTextToPins(dutyStopsText);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Invalid duty input.';
      setDutySyncError(msg);
      return;
    }
    if (!parsed.ok) {
      setDutySyncError(parsed.error);
      return;
    }
    setDutySyncError(null);
    let changed = false;
    if (!sameLatLng(origin, parsed.origin)) {
      setOrigin(parsed.origin);
      changed = true;
    }
    if (!sameLatLng(destination, parsed.destination)) {
      setDestination(parsed.destination);
      changed = true;
    }
    if (!sameManagedStop(managedStop, parsed.stop)) {
      setManagedStop(parsed.stop);
      changed = true;
    }
    logTutorialMidpoint('duty-sync:text-to-pins-applied', {
      parsedOrigin: parsed.origin,
      parsedDestination: parsed.destination,
      parsedStop: parsed.stop,
      changed,
      previousOrigin: origin,
      previousDestination: destination,
      previousStop: managedStop,
      dutyStopsText,
    });
    const canonicalText = serializePinsToDutyText(parsed.origin, parsed.stops, parsed.destination);
    if (canonicalText !== dutyStopsText) {
      dutySyncSourceRef.current = 'pins';
      setDutyStopsText(canonicalText);
    }
    if (changed) {
      clearComputed();
    }
  }, [
    clearComputed,
    dutyStopsText,
    isTutorialPinDraftMode,
    logTutorialMidpoint,
    tutorialBlockingActionId,
    tutorialRunning,
    tutorialStep?.id,
  ]);

  useEffect(() => {
    const normalizedSelection = normalizeSelectedPinId(selectedPinId, {
      origin,
      destination,
      stop: managedStop,
    });
    if (normalizedSelection !== selectedPinId) {
      setSelectedPinId(normalizedSelection);
    }
  }, [selectedPinId, origin, destination, managedStop]);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(LOCALE_STORAGE_KEY);
      if (saved === 'en' || saved === 'es') {
        setLocale(saved);
      }
    } catch {
      // Ignore localStorage access errors.
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
    } catch {
      // Ignore localStorage access errors.
    }
    document.documentElement.lang = locale;
  }, [locale]);

  useEffect(() => {
    const evaluateDesktop = () => {
      setTutorialIsDesktop(window.innerWidth >= 1100);
    };
    evaluateDesktop();
    window.addEventListener('resize', evaluateDesktop);
    return () => window.removeEventListener('resize', evaluateDesktop);
  }, []);

  useEffect(() => {
    if (!tutorialOpen) return;
    if (!tutorialIsDesktop && tutorialMode === 'running') {
      setTutorialMode('blocked');
      return;
    }
    if (tutorialIsDesktop && tutorialMode === 'blocked') {
      setTutorialMode(tutorialSavedProgress ? 'chooser' : 'running');
    }
  }, [tutorialIsDesktop, tutorialMode, tutorialOpen, tutorialSavedProgress]);

  useEffect(() => {
    if (!tutorialRunning) return;
    const shouldForcePanelOpen =
      tutorialLockScope === 'sidebar_section_only' || tutorialStep?.id === 'map_stop_lifecycle';
    if (shouldForcePanelOpen && isPanelCollapsed) {
      setIsPanelCollapsed(false);
    }
  }, [isPanelCollapsed, tutorialLockScope, tutorialRunning, tutorialStep?.id]);

  useEffect(() => {
    if (tutorialBootstrappedRef.current) return;
    tutorialBootstrappedRef.current = true;
    const savedProgress = loadTutorialProgress(TUTORIAL_PROGRESS_KEY);
    const isCompleted = loadTutorialCompleted(TUTORIAL_COMPLETED_KEY);

    setTutorialSavedProgress(savedProgress);
    setTutorialCompleted(isCompleted);
    if (!isCompleted) {
      setTutorialMode(savedProgress ? 'chooser' : tutorialIsDesktop ? 'running' : 'blocked');
      // Keep tutorial closed by default so map HUD remains available on load.
      setTutorialOpen(false);
      if (!savedProgress) {
        setTutorialStepActionsById({});
        setTutorialOptionalDecisionsByStep({});
        setTutorialStepIndex(0);
      }
    }
  }, [tutorialIsDesktop]);

  useEffect(() => {
    if (!tutorialRunning || !tutorialStep) return;
    applyTutorialPrefill(tutorialStep.prefillId);
  }, [applyTutorialPrefill, tutorialRunning, tutorialStep]);

  useEffect(() => {
    if (!tutorialRunning) return;

    const isTargetAllowedByStep = (target: HTMLElement): boolean => {
      for (const targetId of tutorialTargetIdSet) {
        const node = document.querySelector<HTMLElement>(`[data-tutorial-id="${targetId}"]`);
        if (node?.contains(target)) {
          return true;
        }
      }
      return false;
    };

    const handleActionEvent = (event: Event) => {
      if (event.defaultPrevented) return;
      const target = event.target as HTMLElement | null;
      if (!target) return;
      if (target.closest('.tutorialOverlay__card')) return;

      const actionable = target?.closest<HTMLElement>('[data-tutorial-action]');
      const actionId = actionable?.dataset.tutorialAction;
      if (!actionId) return;
      if (actionable?.matches(':disabled')) return;

      if (actionId === 'pins.add_stop') {
        logTutorialMidpoint('action-event:received', {
          actionId,
          tutorialStepId: tutorialStep?.id ?? null,
          tutorialRunning,
          tutorialLockScope,
          tutorialActiveSectionId,
          tutorialBlockingActionId,
        });
      }

      const inMap = Boolean(target.closest('[data-tutorial-id="map.interactive"]'));
      const inPanel = Boolean(target.closest('.panel'));
      const activeSectionNode = tutorialActiveSectionId
        ? document.querySelector<HTMLElement>(`[data-tutorial-id="${tutorialActiveSectionId}"]`)
        : null;

      if (tutorialLockScope === 'map_only' && !inMap) {
        if (actionId === 'pins.add_stop') {
          logTutorialMidpoint('action-event:blocked-map-only', { actionId });
        }
        return;
      }
      if (tutorialLockScope === 'sidebar_section_only') {
        if (!inPanel) {
          if (actionId === 'pins.add_stop') {
            logTutorialMidpoint('action-event:blocked-not-in-panel', { actionId });
          }
          return;
        }
        if (activeSectionNode && !activeSectionNode.contains(target)) {
          if (actionId === 'pins.add_stop') {
            logTutorialMidpoint('action-event:blocked-not-in-active-section', {
              actionId,
              tutorialActiveSectionId,
            });
          }
          return;
        }
        if (
          !isTutorialActionAllowed(actionId, tutorialAllowedActionExact, tutorialAllowedActionPrefixes)
        ) {
          if (actionId === 'pins.add_stop') {
            logTutorialMidpoint('action-event:blocked-allowlist', {
              actionId,
              tutorialAllowedActionExact: [...tutorialAllowedActionExact],
              tutorialAllowedActionPrefixes,
              tutorialUsesSectionTarget,
            });
          }
          return;
        }
      }

      if (!isTutorialActionAllowed(actionId, tutorialAllowedActionExact, tutorialAllowedActionPrefixes)) {
        if (actionId === 'pins.add_stop') {
          logTutorialMidpoint('action-event:blocked-allowlist-global', {
            actionId,
            tutorialAllowedActionExact: [...tutorialAllowedActionExact],
            tutorialAllowedActionPrefixes,
          });
        }
        return;
      }

      if (actionId === 'pins.add_stop') {
        logTutorialMidpoint('action-event:accepted', {
          actionId,
          tutorialBlockingActionId,
          tutorialStepId: tutorialStep?.id ?? null,
        });
        // Important: do not mark pins.add_stop here.
        // The actual midpoint creation flow is in addStopFromMidpoint().
        // Marking here (capture-phase) can lock the panel before button onClick runs.
        logTutorialMidpoint('action-event:defer-pins.add_stop-to-handler', {
          actionId,
          tutorialStepId: tutorialStep?.id ?? null,
        });
        return;
      }
      markTutorialAction(actionId);
    };

    document.addEventListener('click', handleActionEvent, true);
    document.addEventListener('change', handleActionEvent, true);
    document.addEventListener('input', handleActionEvent, true);
    return () => {
      document.removeEventListener('click', handleActionEvent, true);
      document.removeEventListener('change', handleActionEvent, true);
      document.removeEventListener('input', handleActionEvent, true);
    };
  }, [
    markTutorialAction,
    tutorialActiveSectionId,
    tutorialAllowedActionExact,
    tutorialAllowedActionPrefixes,
    logTutorialMidpoint,
    tutorialLockScope,
    tutorialRunning,
    tutorialStep?.id,
    tutorialTargetIdSet,
    tutorialBlockingActionId,
    tutorialUsesSectionTarget,
  ]);

  useEffect(() => {
    if (!tutorialRunning) return;

    const isTargetAllowedByStep = (target: HTMLElement): boolean => {
      for (const targetId of tutorialTargetIdSet) {
        const node = document.querySelector<HTMLElement>(`[data-tutorial-id="${targetId}"]`);
        if (node?.contains(target)) {
          return true;
        }
      }
      return false;
    };

    const blockInteraction = (event: Event) => {
      event.preventDefault();
      event.stopPropagation();
      (event as { stopImmediatePropagation?: () => void }).stopImmediatePropagation?.();
      const now = Date.now();
      if (now - tutorialLockNoticeAtRef.current > 800) {
        tutorialLockNoticeAtRef.current = now;
        setLiveMessage('Complete the highlighted tutorial action first.');
      }
    };

    const onGuardEvent = (event: Event) => {
      const target = event.target as HTMLElement | null;
      if (!target) return;
      if (target.closest('.tutorialOverlay__card')) return;
      const actionId =
        target.closest<HTMLElement>('[data-tutorial-action]')?.dataset.tutorialAction ?? '';
      const isMidpointAction = actionId === 'pins.add_stop';

      const inMap = Boolean(target.closest('[data-tutorial-id="map.interactive"]'));
      const inPanel = Boolean(target.closest('.panel'));
      const activeSectionNode = tutorialActiveSectionId
        ? document.querySelector<HTMLElement>(`[data-tutorial-id="${tutorialActiveSectionId}"]`)
        : null;

      if (tutorialLockScope === 'map_only') {
        if (!inMap) {
          if (isMidpointAction) {
            logTutorialMidpoint('guard:block-map-only', {
              actionId,
              eventType: event.type,
            });
          }
          blockInteraction(event);
        }
        return;
      }

      if (tutorialLockScope === 'sidebar_section_only') {
        if (!inPanel) {
          if (isMidpointAction) {
            logTutorialMidpoint('guard:block-not-panel', {
              actionId,
              eventType: event.type,
            });
          }
          blockInteraction(event);
          return;
        }
        if (activeSectionNode && !activeSectionNode.contains(target)) {
          if (isMidpointAction) {
            logTutorialMidpoint('guard:block-not-active-section', {
              actionId,
              eventType: event.type,
              tutorialActiveSectionId,
            });
          }
          blockInteraction(event);
          return;
        }

        if (
          actionId &&
          isTutorialActionAllowed(actionId, tutorialAllowedActionExact, tutorialAllowedActionPrefixes)
        ) {
          if (isMidpointAction) {
            logTutorialMidpoint('guard:allowed-action', {
              actionId,
              eventType: event.type,
              tutorialAllowedActionExact: [...tutorialAllowedActionExact],
            });
          }
          return;
        }

        if (!actionId && isTargetAllowedByStep(target) && !tutorialUsesSectionTarget) {
          if (isMidpointAction) {
            logTutorialMidpoint('guard:allowed-target-step-fallback', {
              actionId,
              eventType: event.type,
            });
          }
          return;
        }

        if (isMidpointAction) {
          logTutorialMidpoint('guard:block-non-allowed-action', {
            actionId,
            eventType: event.type,
            tutorialAllowedActionExact: [...tutorialAllowedActionExact],
            tutorialAllowedActionPrefixes,
            tutorialUsesSectionTarget,
          });
        }
        blockInteraction(event);
      }
    };

    const guardEvents: Array<keyof DocumentEventMap> = ['pointerdown', 'click', 'change', 'input'];
    for (const name of guardEvents) {
      document.addEventListener(name, onGuardEvent, true);
    }
    return () => {
      for (const name of guardEvents) {
        document.removeEventListener(name, onGuardEvent, true);
      }
    };
  }, [
    tutorialActiveSectionId,
    tutorialAllowedActionExact,
    tutorialAllowedActionPrefixes,
    logTutorialMidpoint,
    tutorialLockScope,
    tutorialRunning,
    tutorialTargetIdSet,
    tutorialUsesSectionTarget,
  ]);

  useEffect(() => {
    const actionNodes = Array.from(document.querySelectorAll<HTMLElement>('[data-tutorial-action]'));
    const targetNodes = Array.from(document.querySelectorAll<HTMLElement>('[data-tutorial-id]'));
    const sidebarControlNodes = Array.from(
      document.querySelectorAll<HTMLElement>(
        '.panel button, .panel input, .panel select, .panel textarea, .panel [role="button"], .panel [role="option"]',
      ),
    );
    const resetNode = (node: HTMLElement) => {
      node.classList.remove('tutorialActionPulse', 'tutorialActionCurrent', 'tutorialActionBlocked');
      node.removeAttribute('data-tutorial-state');
    };
    const resetTargetNode = (node: HTMLElement) => {
      node.classList.remove('tutorialTargetPulse', 'tutorialTargetCurrent');
    };
    const resetControlNode = (node: HTMLElement) => {
      node.classList.remove('tutorialControlBlocked');
    };
    actionNodes.forEach(resetNode);
    targetNodes.forEach(resetTargetNode);
    sidebarControlNodes.forEach(resetControlNode);
    let styleProbeRaf = 0;

    if (!tutorialRunning) return;

    const activeSectionNode = tutorialActiveSectionId
      ? document.querySelector<HTMLElement>(`[data-tutorial-id="${tutorialActiveSectionId}"]`)
      : null;
    const currentActionIds = new Set<string>();
    if (tutorialBlockingActionId) {
      currentActionIds.add(tutorialBlockingActionId);
      if (!tutorialBlockingActionId.endsWith(':open')) {
        currentActionIds.add(`${tutorialBlockingActionId}:open`);
      }
      const lastColonIndex = tutorialBlockingActionId.lastIndexOf(':');
      if (lastColonIndex > 0) {
        const actionPrefix = tutorialBlockingActionId.slice(0, lastColonIndex);
        const actionValue = tutorialBlockingActionId.slice(lastColonIndex + 1);
        if (actionValue !== 'open') {
          currentActionIds.add(`${actionPrefix}:open`);
        }
      }
    }
    const interestingActionIds = new Set<string>([
      'map.popup_close_origin_marker',
      'map.popup_close_destination_marker',
      'map.popup_close',
      tutorialBlockingActionId ?? '',
    ]);

    logTutorialPulse('highlight-pass:start', {
      tutorialStepId: tutorialStep?.id ?? null,
      tutorialBlockingActionId,
      tutorialLockScope,
      tutorialActiveSectionId,
      tutorialTargetIds: [...tutorialTargetIdSet],
      allowedActionExact: [...tutorialAllowedActionExact],
      allowedActionPrefixes: tutorialAllowedActionPrefixes,
      actionNodeCount: actionNodes.length,
      targetNodeCount: targetNodes.length,
      sidebarControlNodeCount: sidebarControlNodes.length,
    });

    const isInScope = (node: HTMLElement): boolean => {
      const inMap = Boolean(node.closest('[data-tutorial-id="map.interactive"]'));
      const inPanel = Boolean(node.closest('.panel'));
      if (tutorialLockScope === 'map_only') return inMap;
      if (tutorialLockScope === 'sidebar_section_only') {
        if (!inPanel) return false;
        if (activeSectionNode && !activeSectionNode.contains(node)) return false;
      }
      return true;
    };

    for (const node of actionNodes) {
      const actionId = node.dataset.tutorialAction ?? '';
      const inScope = isInScope(node);
      const allowed = actionId
        ? isTutorialActionAllowed(actionId, tutorialAllowedActionExact, tutorialAllowedActionPrefixes)
        : false;
      const isCurrent = Boolean(actionId && currentActionIds.has(actionId));
      const isBlocked = !inScope || !allowed || node.matches(':disabled');

      if (isBlocked) {
        node.classList.add('tutorialActionBlocked');
        node.setAttribute('data-tutorial-state', 'blocked');
        if (interestingActionIds.has(actionId)) {
          logTutorialPulse('highlight-pass:node-blocked', {
            actionId,
            isCurrent,
            inScope,
            allowed,
            disabled: node.matches(':disabled'),
            nodeClassName: node.className,
            state: node.getAttribute('data-tutorial-state'),
          });
        }
        continue;
      }

      node.setAttribute('data-tutorial-state', 'active');
      if (isCurrent) {
        node.classList.add('tutorialActionPulse');
        logTutorialPulse('highlight-pass:node-current', {
          actionId,
          inScope,
          allowed,
          disabled: node.matches(':disabled'),
          nodeClassName: node.className,
          state: node.getAttribute('data-tutorial-state'),
        });
      } else if (interestingActionIds.has(actionId)) {
        logTutorialPulse('highlight-pass:node-allowed-not-current', {
          actionId,
          inScope,
          allowed,
          disabled: node.matches(':disabled'),
          nodeClassName: node.className,
          state: node.getAttribute('data-tutorial-state'),
        });
      }
    }

    for (const node of sidebarControlNodes) {
      let blocked = false;
      if (node.matches(':disabled')) {
        blocked = true;
      }
      if (!blocked && tutorialLockScope === 'map_only') {
        blocked = true;
      }
      if (!blocked && tutorialLockScope === 'sidebar_section_only') {
        if (activeSectionNode && !activeSectionNode.contains(node)) {
          blocked = true;
        } else {
          const actionHost = node.closest<HTMLElement>('[data-tutorial-action]');
          const actionId = actionHost?.dataset.tutorialAction ?? '';
          const actionAllowed = actionId
            ? isTutorialActionAllowed(actionId, tutorialAllowedActionExact, tutorialAllowedActionPrefixes)
            : false;
          if (!actionAllowed) {
            blocked = true;
          }
        }
      }
      if (blocked) {
        node.classList.add('tutorialControlBlocked');
      }
    }

    const currentActionNodes = actionNodes.filter((node) => node.classList.contains('tutorialActionCurrent'));
    const pulsingActionNodes = actionNodes.filter((node) => node.classList.contains('tutorialActionPulse'));
    const blockedActionNodes = actionNodes.filter((node) => node.classList.contains('tutorialActionBlocked'));
    logTutorialPulse('highlight-pass:summary', {
      currentCount: currentActionNodes.length,
      pulsingCount: pulsingActionNodes.length,
      blockedCount: blockedActionNodes.length,
      currentActionIds: currentActionNodes.map((node) => node.dataset.tutorialAction ?? ''),
      pulsingActionIds: pulsingActionNodes.map((node) => node.dataset.tutorialAction ?? ''),
      blockedActionIds: blockedActionNodes.map((node) => node.dataset.tutorialAction ?? ''),
    });

    styleProbeRaf = window.requestAnimationFrame(() => {
      const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const probeNodes = actionNodes.filter((node) => {
        const id = node.dataset.tutorialAction ?? '';
        return interestingActionIds.has(id);
      });
      const probePayload = probeNodes.map((node) => {
        const styles = window.getComputedStyle(node);
        const actionId = node.dataset.tutorialAction ?? '';
        const rect = node.getBoundingClientRect();
        return {
          actionId,
          className: node.className,
          state: node.getAttribute('data-tutorial-state'),
          animationName: styles.animationName,
          animationDuration: styles.animationDuration,
          animationPlayState: styles.animationPlayState,
          opacity: styles.opacity,
          boxShadow: styles.boxShadow,
          filter: styles.filter,
          display: styles.display,
          visibility: styles.visibility,
          pointerEvents: styles.pointerEvents,
          rect: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          },
          closestPopupCardClass: node.closest('.markerPopup__card')?.className ?? null,
        };
      });
      logTutorialPulse('highlight-pass:computed-style-probe', {
        probeCount: probePayload.length,
        reducedMotion,
        probePayload,
      });
      for (const entry of probePayload) {
        if (!entry.actionId.includes('popup_close')) continue;
        console.log(
          `[tutorial-pulse-debug] close-probe action=${entry.actionId} state=${entry.state} reducedMotion=${String(reducedMotion)} anim=${entry.animationName} duration=${entry.animationDuration} play=${entry.animationPlayState} boxShadow=${entry.boxShadow}`,
        );
      }

      const closeButtons = Array.from(
        document.querySelectorAll<HTMLElement>(
          '[data-tutorial-action="map.popup_close_origin_marker"], [data-tutorial-action="map.popup_close_destination_marker"], [data-tutorial-action="map.popup_close"]',
        ),
      );
      logTutorialPulse('highlight-pass:close-button-dom-scan', {
        closeButtonCount: closeButtons.length,
        closeButtons: closeButtons.map((node) => {
          const styles = window.getComputedStyle(node);
          const rect = node.getBoundingClientRect();
          return {
            actionId: node.dataset.tutorialAction ?? '',
            className: node.className,
            state: node.getAttribute('data-tutorial-state'),
            animationName: styles.animationName,
            animationDuration: styles.animationDuration,
            animationPlayState: styles.animationPlayState,
            boxShadow: styles.boxShadow,
            opacity: styles.opacity,
            filter: styles.filter,
            pointerEvents: styles.pointerEvents,
            rect: {
              x: Math.round(rect.x),
              y: Math.round(rect.y),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
            },
          };
        }),
      });
    });

    return () => {
      if (styleProbeRaf) {
        window.cancelAnimationFrame(styleProbeRaf);
      }
      actionNodes.forEach(resetNode);
      targetNodes.forEach(resetTargetNode);
      sidebarControlNodes.forEach(resetControlNode);
    };
  }, [
    tutorialActiveSectionId,
    tutorialAllowedActionExact,
    tutorialAllowedActionPrefixes,
    tutorialBlockingActionId,
    logTutorialPulse,
    tutorialLockScope,
    tutorialRunning,
    tutorialStep?.id,
    tutorialTargetIdSet,
  ]);

  useEffect(() => {
    if (!tutorialRunning) return;
    const payload: TutorialProgress = {
      version: 3,
      stepIndex: tutorialStepIndex,
      stepActionsById: tutorialStepActionsById,
      optionalDecisionsByStep: tutorialOptionalDecisionsByStep,
      updatedAt: new Date().toISOString(),
    };
    saveTutorialProgress(TUTORIAL_PROGRESS_KEY, payload);
    setTutorialSavedProgress(payload);
  }, [tutorialOptionalDecisionsByStep, tutorialRunning, tutorialStepActionsById, tutorialStepIndex]);

  useLayoutEffect(() => {
    if (!tutorialRunning) return;
    setTutorialTargetRect(null);
    setTutorialTargetMissing(false);
  }, [tutorialMode, tutorialRunning, tutorialStepId, tutorialStepIndex]);

  useEffect(() => {
    if (!tutorialRunning || !tutorialStep) return;

    let raf = 0;
    let timeoutId = 0;
    let mutationObserver: MutationObserver | null = null;
    let lastTargetSignature = '';
    let lastMissingState: boolean | null = null;
    const resolveTarget = (scrollIntoViewTarget: boolean) => {
      if (!tutorialStep.targetIds.length) {
        if (lastTargetSignature !== 'none') {
          lastTargetSignature = 'none';
          setTutorialTargetRect(null);
        }
        if (lastMissingState !== false) {
          lastMissingState = false;
          setTutorialTargetMissing(false);
        }
        return;
      }
      const element = tutorialStep.targetIds
        .map((targetId) =>
          document.querySelector<HTMLElement>(`[data-tutorial-id=\"${targetId}\"]`),
        )
        .find((candidate) => Boolean(candidate));

      if (!element) {
        if (lastTargetSignature !== 'none') {
          lastTargetSignature = 'none';
          setTutorialTargetRect(null);
        }
        if (lastMissingState !== true) {
          lastMissingState = true;
          setTutorialTargetMissing(true);
        }
        return;
      }

      if (isPanelCollapsed && element.closest('.panel')) {
        setIsPanelCollapsed(false);
        raf = window.requestAnimationFrame(() => resolveTarget(scrollIntoViewTarget));
        return;
      }
      if (scrollIntoViewTarget) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
      }
      const rect = element.getBoundingClientRect();
      if (rect.width <= 1 || rect.height <= 1) {
        if (lastTargetSignature !== 'none') {
          lastTargetSignature = 'none';
          setTutorialTargetRect(null);
        }
        if (lastMissingState !== true) {
          lastMissingState = true;
          setTutorialTargetMissing(true);
        }
        return;
      }
      const signature = `${Math.round(rect.top)}:${Math.round(rect.left)}:${Math.round(rect.width)}:${Math.round(rect.height)}`;
      if (signature !== lastTargetSignature) {
        lastTargetSignature = signature;
        setTutorialTargetRect({
          top: rect.top,
          left: rect.left,
          width: rect.width,
          height: rect.height,
        });
      }
      if (lastMissingState !== false) {
        lastMissingState = false;
        setTutorialTargetMissing(false);
      }
    };

    resolveTarget(false);
    raf = window.requestAnimationFrame(() => resolveTarget(false));
    timeoutId = window.setTimeout(() => {
      resolveTarget(true);
    }, 40);
    const onResize = () => resolveTarget(false);
    const onScroll = () => resolveTarget(false);
    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onScroll, true);
    if (typeof MutationObserver !== 'undefined') {
      mutationObserver = new MutationObserver(() => resolveTarget(false));
      mutationObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style'],
      });
    }
    return () => {
      window.clearTimeout(timeoutId);
      window.cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onScroll, true);
      mutationObserver?.disconnect();
    };
  }, [isPanelCollapsed, tutorialRunning, tutorialStep]);

  useEffect(() => {
    const best = pickBestByWeightedSum(paretoRoutes, weights);
    setSelectedId(best);
  }, [paretoRoutes, weights]);

  useEffect(() => {
    if (loading) return;

    startTransition(() => {
      setParetoRoutes((prev) => {
        const sorted = sortRoutesDeterministic(prev);
        const unchanged =
          prev.length === sorted.length && prev.every((route, idx) => route.id === sorted[idx]?.id);
        return unchanged ? prev : sorted;
      });
    });
  }, [loading, startTransition]);

  useEffect(() => {
    if (warnings.length === 0) {
      setShowWarnings(false);
    }
  }, [warnings.length]);

  useEffect(() => {
    if (advancedError) {
      setAdvancedError(null);
    }
  }, [advancedParams, advancedError]);

  const selectedRoute = useMemo(() => {
    if (!selectedId) return null;
    return paretoRoutes.find((route) => route.id === selectedId) ?? null;
  }, [paretoRoutes, selectedId]);

  const parsedDutyStops = useMemo(() => {
    const parsed = parseDutyTextToPins(dutyStopsText);
    if (!parsed.ok) return [] as Array<{ lat: number; lon: number; label: string }>;
    return parsed.stops;
  }, [dutyStopsText]);
  const dutyStopsForOverlay = useMemo(
    () =>
      parsedDutyStops.map((stop) => ({
        lat: stop.lat,
        lon: stop.lon,
        label: stop.label,
      })),
    [parsedDutyStops],
  );
  const requestWaypoints = useMemo<Waypoint[]>(
    () =>
      parsedDutyStops.map((stop) => ({
        lat: stop.lat,
        lon: stop.lon,
        label: stop.label,
      })),
    [parsedDutyStops],
  );
  const pinNodes = useMemo(
    () => buildManagedPinNodes(origin, destination, managedStop, { origin: 'Start', destination: 'End' }),
    [origin, destination, managedStop],
  );
  const midpointBaseOrigin =
    origin ??
    (tutorialRunning && tutorialStep?.id === 'map_stop_lifecycle'
      ? tutorialConfirmedOriginRef.current
      : null);
  const midpointBaseDestination =
    destination ??
    (tutorialRunning && tutorialStep?.id === 'map_stop_lifecycle'
      ? tutorialConfirmedDestinationRef.current
      : null);
  const canAddStop = Boolean(midpointBaseOrigin && midpointBaseDestination);
  const tutorialSidebarActionLocked =
    tutorialRunning &&
    tutorialStep?.id === 'map_stop_lifecycle' &&
    tutorialActionSet.has('pins.add_stop');
  useEffect(() => {
    if (!tutorialRunning) return;
    const isMidpointStep = tutorialStep?.id === 'map_stop_lifecycle';
    const isAddStopPending = tutorialBlockingActionId === 'pins.add_stop';
    if (!isMidpointStep && !isAddStopPending) return;
    logTutorialMidpoint('midpoint-state-snapshot', {
      tutorialStepId: tutorialStep?.id ?? null,
      tutorialBlockingActionId,
      tutorialNextRequiredActionId,
      tutorialLockScope,
      tutorialActiveSectionId,
      tutorialSidebarLocked,
      tutorialMapLocked,
      pinsSectionOpen: tutorialSectionControl.pins?.isOpen ?? null,
      pinsSectionLocked: tutorialSectionControl.pins?.tutorialLocked ?? null,
      loading,
      isPending,
      busyLike: loading || isPending,
      canAddStop,
      showStopOverlay,
      hasOrigin: Boolean(origin),
      hasDestination: Boolean(destination),
      hasManagedStop: Boolean(managedStop),
      dutyStopsText,
      dutySyncError,
      origin,
      destination,
      managedStop,
      confirmedOriginRef: tutorialConfirmedOriginRef.current,
      confirmedDestinationRef: tutorialConfirmedDestinationRef.current,
      allowedActions: [...tutorialAllowedActionExact],
      allowedActionPrefixes: tutorialAllowedActionPrefixes,
    });
  }, [
    canAddStop,
    destination,
    dutyStopsText,
    dutySyncError,
    isPending,
    loading,
    logTutorialMidpoint,
    managedStop,
    origin,
    showStopOverlay,
    tutorialActiveSectionId,
    tutorialAllowedActionExact,
    tutorialAllowedActionPrefixes,
    tutorialBlockingActionId,
    tutorialLockScope,
    tutorialMapLocked,
    tutorialNextRequiredActionId,
    tutorialRunning,
    tutorialSectionControl.pins?.isOpen,
    tutorialSectionControl.pins?.tutorialLocked,
    tutorialSidebarLocked,
    tutorialStep?.id,
  ]);

  useEffect(() => {
    if (!tutorialRunning) return;
    if (tutorialStep?.id !== 'map_stop_lifecycle') return;
    logTutorialMidpoint('midpoint-stop-watch', {
      tutorialBlockingActionId,
      tutorialNextRequiredActionId,
      tutorialActionSet: [...tutorialActionSet],
      origin,
      destination,
      managedStop,
      midpointBaseOrigin,
      midpointBaseDestination,
      canAddStop,
      dutyStopsText,
      dutySyncError,
      selectedPinId,
      focusPinRequest,
      showStopOverlay,
      selectedRouteId: selectedRoute?.id ?? null,
    });
  }, [
    canAddStop,
    destination,
    dutyStopsText,
    dutySyncError,
    focusPinRequest,
    logTutorialMidpoint,
    managedStop,
    midpointBaseDestination,
    midpointBaseOrigin,
    origin,
    selectedPinId,
    selectedRoute?.id,
    showStopOverlay,
    tutorialActionSet,
    tutorialBlockingActionId,
    tutorialNextRequiredActionId,
    tutorialRunning,
    tutorialStep?.id,
  ]);
  const mapOverlayLabels = useMemo(
    () => ({
      stopLabel: t('stop_label'),
      segmentLabel: t('segment_label'),
      incidentTypeLabels: {
        dwell: t('incident_dwell'),
        accident: t('incident_accident'),
        closure: t('incident_closure'),
      },
    }),
    [t],
  );

  useEffect(() => {
    if (!selectedRoute) {
      setTimeLapsePosition(null);
    }
  }, [selectedRoute]);

  useEffect(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'selected_read_panels' || !selectedRoute) return;
    markTutorialAction('selected.panel_data_ready');
    markTutorialAction('selected.timeline_panel_visible');
    markTutorialAction('selected.counterfactual_panel_visible');
  }, [markTutorialAction, selectedRoute, tutorialRunning, tutorialStep?.id]);

  const defaultLabelsById = useMemo(() => {
    const labels: Record<string, string> = {};
    for (let idx = 0; idx < paretoRoutes.length; idx += 1) {
      labels[paretoRoutes[idx].id] = `Route ${idx + 1}`;
    }
    return labels;
  }, [paretoRoutes]);

  const labelsById = useMemo(() => {
    const merged: Record<string, string> = { ...defaultLabelsById };
    for (const [routeId, name] of Object.entries(routeNames)) {
      const trimmed = name.trim();
      if (trimmed) merged[routeId] = trimmed;
    }
    return merged;
  }, [defaultLabelsById, routeNames]);
  const sortedDisplayRoutes = useMemo(() => {
    const sorted = [...paretoRoutes];
    sorted.sort((a, b) => {
      if (routeSort === 'duration') {
        const byDuration = a.metrics.duration_s - b.metrics.duration_s;
        if (byDuration !== 0) return byDuration;
      }
      if (routeSort === 'cost') {
        const byCost = a.metrics.monetary_cost - b.metrics.monetary_cost;
        if (byCost !== 0) return byCost;
      }
      if (routeSort === 'co2') {
        const byCo2 = a.metrics.emissions_kg - b.metrics.emissions_kg;
        if (byCo2 !== 0) return byCo2;
      }
      return a.id.localeCompare(b.id);
    });
    return sorted;
  }, [paretoRoutes, routeSort]);

  const selectedLabel = selectedRoute ? labelsById[selectedRoute.id] ?? selectedRoute.id : null;

  const busy = loading || isPending;
  const routeGraphWarmup = opsHealthReady?.route_graph ?? null;
  const routeGraphState = String(routeGraphWarmup?.state ?? '').trim().toLowerCase();
  const routeGraphPhase = String(routeGraphWarmup?.phase ?? '').trim().toLowerCase();
  const strictRouteReady = opsHealthReady?.strict_route_ready === true;
  const routeWarmupBaselineMs = Math.max(
    60_000,
    parsePositiveIntEnv(process.env.NEXT_PUBLIC_ROUTE_GRAPH_WARMUP_BASELINE_MS, 900_000),
  );
  const routeGraphElapsedMs = toFiniteNumber(routeGraphWarmup?.elapsed_ms);
  const routeGraphNodesSeen = Math.max(0, Math.floor(toFiniteNumber(routeGraphWarmup?.nodes_seen) ?? 0));
  const routeGraphNodesKept = Math.max(0, Math.floor(toFiniteNumber(routeGraphWarmup?.nodes_kept) ?? 0));
  const routeGraphEdgesSeen = Math.max(0, Math.floor(toFiniteNumber(routeGraphWarmup?.edges_seen) ?? 0));
  const routeGraphEdgesKept = Math.max(0, Math.floor(toFiniteNumber(routeGraphWarmup?.edges_kept) ?? 0));
  const routeWarmupPhaseLabel = warmupPhaseLabel(routeGraphPhase || routeGraphState);
  const routeWarmupProgressFraction = strictRouteReady
    ? 1
    : routeGraphState === 'loading' && routeGraphElapsedMs !== null
      ? clampUnit(routeGraphElapsedMs / routeWarmupBaselineMs)
      : routeGraphState === 'failed'
        ? 1
        : 0;
  const routeWarmupProgressPct = strictRouteReady
    ? 100
    : routeGraphState === 'loading'
      ? Math.max(1, Math.min(100, Math.round(routeWarmupProgressFraction * 100)))
      : 0;
  const routeWarmupBaselineRemainingMs =
    routeGraphState === 'loading' && routeGraphElapsedMs !== null
      ? Math.max(0, routeWarmupBaselineMs - routeGraphElapsedMs)
      : null;
  const routeWarmupOverrunMs =
    routeGraphState === 'loading' &&
    routeGraphElapsedMs !== null &&
    routeGraphElapsedMs > routeWarmupBaselineMs
      ? routeGraphElapsedMs - routeWarmupBaselineMs
      : null;
  const routeWarmupElapsedLabel = formatDurationCompactMs(routeGraphElapsedMs);
  const routeWarmupBaselineLabel = formatDurationCompactMs(routeWarmupBaselineMs);
  const routeWarmupRemainingLabel =
    routeGraphState !== 'loading'
      ? 'n/a'
      : routeWarmupOverrunMs !== null
        ? '0s (baseline reached)'
        : routeWarmupBaselineRemainingMs !== null
          ? `${formatDurationCompactMs(routeWarmupBaselineRemainingMs)} (baseline)`
          : `${routeWarmupBaselineLabel} (baseline)`;
  const routeWarmupOverrunLabel =
    routeWarmupOverrunMs !== null
      ? `Over baseline by ${formatDurationCompactMs(routeWarmupOverrunMs)}`
      : null;
  const routeWarmupShowCard = !strictRouteReady;
  const routeReadinessCardClassName =
    routeGraphState === 'failed'
      ? 'routeReadinessCard isFailed'
      : routeGraphState === 'loading'
        ? 'routeReadinessCard isLoading'
        : 'routeReadinessCard';
  const routeReadinessReasonCode =
    routeGraphState === 'failed'
      ? 'routing_graph_warmup_failed'
      : routeGraphState === 'loading'
        ? 'routing_graph_warming_up'
        : undefined;
  const routeReadinessSummary = strictRouteReady
    ? 'Ready'
    : routeGraphState === 'failed'
      ? 'Warmup failed'
      : routeGraphState === 'loading'
        ? 'Warming up route graph...'
        : 'Checking route graph readiness...';
  const routeReadinessDetail = strictRouteReady
    ? 'strict_route_ready=true'
    : routeGraphState === 'failed'
      ? String(routeGraphWarmup?.last_error ?? 'unknown warmup failure')
      : routeGraphState === 'loading'
        ? `phase=${routeWarmupPhaseLabel}; progress~${routeWarmupProgressPct}% (${routeWarmupBaselineLabel} baseline); elapsed=${routeWarmupElapsedLabel}; remaining~${routeWarmupRemainingLabel}${routeWarmupOverrunLabel ? `; ${routeWarmupOverrunLabel.toLowerCase()}` : ''}`
        : `state=${routeGraphWarmup?.state ?? 'unknown'}; phase=${routeGraphWarmup?.phase ?? 'unknown'}`;
  const canCompute = Boolean(origin && destination) && !busy && strictRouteReady;

  const normalisedWeights = useMemo(() => normaliseWeights(weights), [weights]);
  const latestTraceEntry = computeTrace.length ? computeTrace[computeTrace.length - 1] : null;
  const computeElapsedSeconds = latestTraceEntry ? (latestTraceEntry.elapsedMs / 1000).toFixed(1) : null;
  const traceEntriesByAttempt = useMemo(() => {
    const groups = new Map<number, ComputeTraceEntry[]>();
    for (const entry of computeTrace) {
      const key = entry.attempt ?? 0;
      const bucket = groups.get(key);
      if (bucket) {
        bucket.push(entry);
      } else {
        groups.set(key, [entry]);
      }
    }
    return Array.from(groups.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([attempt, entries]) => ({ attempt, entries }));
  }, [computeTrace]);
  const activeAttemptEntry = useMemo(
    () =>
      [...computeTrace]
        .reverse()
        .find((entry) => typeof entry.attempt === 'number' && entry.attempt > 0) ?? null,
    [computeTrace],
  );
  const activeAttemptLiveCallTrace = useMemo(() => {
    const attempt = activeAttemptEntry?.attempt;
    if (!attempt) return null;
    return computeLiveCallsByAttempt[attempt] ?? null;
  }, [activeAttemptEntry?.attempt, computeLiveCallsByAttempt]);
  const liveCallFailureHints = useMemo(() => {
    if (!error || !activeAttemptLiveCallTrace) return [];
    const blocked = (activeAttemptLiveCallTrace.expected_rollup ?? [])
      .filter((row) => row.status === 'blocked' || row.blocked)
      .slice(0, 4)
      .map((row) => {
        const why = row.blocked_reason || 'blocked';
        const where = row.blocked_stage ? ` at ${row.blocked_stage}` : '';
        return `Live source not attempted: ${row.source_key} (${why}${where}).`;
      });
    const failures = (activeAttemptLiveCallTrace.observed_calls ?? [])
      .filter((row) => row.requested && !row.success)
      .slice(0, 4)
      .map((row) => {
        const why = row.fetch_error || (typeof row.status_code === 'number' ? `status ${row.status_code}` : 'unknown');
        return `Live source failed: ${row.source_key} -> ${row.url} (${why}).`;
      });
    return [...blocked, ...failures];
  }, [activeAttemptLiveCallTrace, error]);
  const latestDiagnosticEntry = useMemo(
    () =>
      [...computeTrace]
        .reverse()
        .find((entry) => Boolean(entry.candidateDiagnostics || entry.failureChain)) ?? null,
    [computeTrace],
  );
  const diagnosticAwareHints = useMemo(() => {
    const hints: string[] = [];
    if (!error) return hints;
    const diag = (latestDiagnosticEntry?.candidateDiagnostics ?? null) as Record<string, unknown> | null;
    const reasonCode = (
      latestDiagnosticEntry?.reasonCode ??
      extractReasonCodeFromMessage(error) ??
      ''
    ).trim();
    if (reasonCode === 'routing_graph_precheck_timeout' && diag) {
      const prefetchSuccess = Number(diag.prefetch_success_sources ?? 0);
      const precheckGateAction = String(diag.precheck_gate_action ?? '').trim();
      if (Number.isFinite(prefetchSuccess) && prefetchSuccess > 0) {
        hints.push('Live-source prefetch completed before the precheck timeout; this is a graph precheck bottleneck.');
      }
      if (precheckGateAction) {
        hints.push(`Precheck gate action for this run: ${precheckGateAction}.`);
      }
    }
    if (diag) {
      const retryAttempted = Boolean(diag.graph_retry_attempted ?? false);
      const retryOutcome = String(diag.graph_retry_outcome ?? '').trim();
      const retryBudget = Number(diag.graph_retry_state_budget ?? 0);
      if (retryAttempted) {
        hints.push(
          `Graph retry pass executed (outcome=${retryOutcome || 'unknown'}; state_budget=${Number.isFinite(retryBudget) ? retryBudget : 0}).`,
        );
      }
      const rescueAttempted = Boolean(diag.graph_rescue_attempted ?? false);
      const rescueOutcome = String(diag.graph_rescue_outcome ?? '').trim();
      const rescueMode = String(diag.graph_rescue_mode ?? '').trim();
      const rescueBudget = Number(diag.graph_rescue_state_budget ?? 0);
      if (rescueAttempted) {
        hints.push(
          `Graph rescue pass executed (mode=${rescueMode || 'reduced'}; outcome=${rescueOutcome || 'unknown'}; state_budget=${Number.isFinite(rescueBudget) ? rescueBudget : 0}).`,
        );
      }
    }
    return hints;
  }, [error, latestDiagnosticEntry]);
  const computeErrorHints = useMemo(() => {
    const base = error ? inferComputeRecoveryHints(error) : [];
    const merged = [...liveCallFailureHints, ...diagnosticAwareHints, ...base];
    return Array.from(new Set(merged));
  }, [diagnosticAwareHints, error, liveCallFailureHints]);
  const mapFailureOverlay = useMemo<MapFailureOverlay | null>(() => {
    if (!error || loading || !origin || !destination || selectedRoute) return null;
    const reasonCode =
      latestDiagnosticEntry?.reasonCode ??
      extractReasonCodeFromMessage(error) ??
      'route_compute_failed';
    const message = String(latestDiagnosticEntry?.detail || error).trim() || 'Route computation failed.';
    return {
      reason_code: reasonCode,
      message,
      stage: latestDiagnosticEntry?.stage ?? null,
      stage_detail: latestDiagnosticEntry?.stageDetail ?? null,
    };
  }, [destination, error, latestDiagnosticEntry, loading, origin, selectedRoute]);
  const latestAiDiagnosticBundle = useMemo(() => {
    const attempt =
      typeof latestDiagnosticEntry?.attempt === 'number' && latestDiagnosticEntry.attempt > 0
        ? latestDiagnosticEntry.attempt
        : null;
    const liveTrace = attempt ? computeLiveCallsByAttempt[attempt] ?? null : null;
    const requestId = attempt
      ? (computeRequestIdByAttempt[attempt] ?? liveTrace?.request_id ?? null)
      : null;
    const observed = [...(liveTrace?.observed_calls ?? [])];
    const slowestObserved = observed
      .sort((a, b) => Number(b.duration_ms ?? 0) - Number(a.duration_ms ?? 0))
      .slice(0, 6);
    if (!latestDiagnosticEntry && !liveTrace && !error) return null;
    return {
      attempt,
      endpoint: latestDiagnosticEntry?.endpoint ?? null,
      request_id: requestId,
      latest_reason_code:
        latestDiagnosticEntry?.reasonCode ??
        (error ? extractReasonCodeFromMessage(error) ?? null : null),
      message: error ?? null,
      summary: liveTrace?.summary ?? null,
      expected_rollup: liveTrace?.expected_rollup ?? null,
      slowest_calls: slowestObserved,
      candidate_diagnostics: latestDiagnosticEntry?.candidateDiagnostics ?? null,
      failure_chain: latestDiagnosticEntry?.failureChain ?? null,
    };
  }, [computeLiveCallsByAttempt, computeRequestIdByAttempt, error, latestDiagnosticEntry]);
  const progressText = useMemo(() => {
    if (!progress) return null;
    const fallbackTotal =
      typeof activeAttemptEntry?.alternativesUsed === 'number' && activeAttemptEntry.alternativesUsed > 0
        ? activeAttemptEntry.alternativesUsed
        : progress.total;
    const total = Math.max(1, fallbackTotal);
    const done = Math.max(0, Math.min(progress.done, total));
    return `${formatNumber(done, locale, {
      maximumFractionDigits: 0,
    })}/${formatNumber(total, locale, { maximumFractionDigits: 0 })}`;
  }, [activeAttemptEntry?.alternativesUsed, locale, progress]);
  const activeAttemptNumberForTrace =
    typeof activeAttemptEntry?.attempt === 'number' && activeAttemptEntry.attempt > 0
      ? activeAttemptEntry.attempt
      : null;
  const latestBackendMetaEntry = useMemo(
    () => {
      if (activeAttemptNumberForTrace === null) return null;
      return (
        [...computeTrace]
          .reverse()
          .find(
            (entry) =>
              entry.attempt === activeAttemptNumberForTrace &&
              entry.endpoint === '/api/pareto/stream' &&
              typeof entry.backendElapsedMs === 'number' &&
              typeof entry.requestId === 'string',
          ) ?? null
      );
    },
    [activeAttemptNumberForTrace, computeTrace],
  );
  const lastBackendHeartbeatAgeSec = latestBackendMetaEntry
    ? ((computeTraceNowMs - Date.parse(latestBackendMetaEntry.at)) / 1000).toFixed(1)
    : null;

  const copyComputeDiagnostics = useCallback(async () => {
    const header = computeSession
      ? [
          `seq=${computeSession.seq}`,
          `started_at=${computeSession.startedAt}`,
          `mode=${computeSession.mode}`,
          `scenario=${computeSession.scenario}`,
          `vehicle=${computeSession.vehicle}`,
          `max_alternatives=${computeSession.maxAlternatives}`,
          `waypoints=${computeSession.waypointCount}`,
          `origin=${computeSession.origin.lat},${computeSession.origin.lon}`,
          `destination=${computeSession.destination.lat},${computeSession.destination.lon}`,
          `stream_stall_timeout_ms=${computeSession.streamStallTimeoutMs}`,
          `attempt_timeout_ms=${computeSession.attemptTimeoutMs}`,
          `route_fallback_timeout_ms=${computeSession.routeFallbackTimeoutMs}`,
          `degrade_steps=${computeSession.degradeSteps.join('->')}`,
        ].join('\n')
      : 'no active compute session';
    const body = computeTrace
      .map(
        (entry) =>
          `[${entry.level.toUpperCase()}] ${entry.at} (+${(entry.elapsedMs / 1000).toFixed(1)}s) ${entry.step} :: ${entry.detail}${
            entry.reasonCode ? ` | reason_code=${entry.reasonCode}` : ''
          }${
            typeof entry.attempt === 'number' ? ` | attempt=${entry.attempt}` : ''
          }${
            entry.endpoint ? ` | endpoint=${entry.endpoint}` : ''
          }${
            entry.requestId ? ` | request_id=${entry.requestId}` : ''
          }${
            entry.stage ? ` | stage=${entry.stage}` : ''
          }${
            entry.stageDetail ? ` | stage_detail=${entry.stageDetail}` : ''
          }${
            typeof entry.backendElapsedMs === 'number' ? ` | backend_elapsed_ms=${entry.backendElapsedMs}` : ''
          }${
            typeof entry.stageElapsedMs === 'number' ? ` | stage_elapsed_ms=${entry.stageElapsedMs}` : ''
          }${
            typeof entry.alternativesUsed === 'number' ? ` | alternatives=${entry.alternativesUsed}` : ''
          }${
            typeof entry.timeoutMs === 'number' ? ` | timeout_ms=${entry.timeoutMs}` : ''
          }${
            entry.candidateDiagnostics ? ` | candidate_diagnostics=${safeJsonString(entry.candidateDiagnostics)}` : ''
          }${
            entry.failureChain ? ` | failure_chain=${safeJsonString(entry.failureChain)}` : ''
          }`,
      )
      .join('\n');
    const liveCallsText = Object.entries(computeLiveCallsByAttempt)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([attemptKey, trace]) => {
        const requestId = computeRequestIdByAttempt[Number(attemptKey)] ?? trace.request_id;
        return `Attempt ${attemptKey} request_id=${requestId}\n${JSON.stringify(trace, null, 2)}`;
      })
      .join('\n\n');
    const text = `# Compute Session\n${header}\n\n# Trace\n${body}\n\n# Live API Calls\n${
      liveCallsText || 'No live-call trace rows captured.'
    }`;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      setError('Failed to copy diagnostics to clipboard.');
    }
  }, [computeLiveCallsByAttempt, computeRequestIdByAttempt, computeSession, computeTrace]);

  const copyLiveCallJson = useCallback(async () => {
    const payload = {
      request_ids: computeRequestIdByAttempt,
      traces: computeLiveCallsByAttempt,
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      appendComputeTrace({
        level: 'info',
        step: 'Live-call JSON copied',
        detail: `Copied ${Object.keys(computeLiveCallsByAttempt).length} attempt trace payload(s).`,
      });
    } catch {
      setError('Failed to copy live-call JSON to clipboard.');
    }
  }, [appendComputeTrace, computeLiveCallsByAttempt, computeRequestIdByAttempt]);
  const copyLatestAiDiagnosticBundle = useCallback(async () => {
    if (!latestAiDiagnosticBundle) {
      setError('No diagnostic bundle is available yet.');
      return;
    }
    try {
      await navigator.clipboard.writeText(JSON.stringify(latestAiDiagnosticBundle, null, 2));
      appendComputeTrace({
        level: 'info',
        step: 'Diagnostic bundle copied',
        detail: 'Copied latest AI diagnostic bundle JSON to clipboard.',
      });
    } catch {
      setError('Failed to copy AI diagnostic bundle to clipboard.');
    }
  }, [appendComputeTrace, latestAiDiagnosticBundle]);

  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  useEffect(() => {
    if (!computeTraceOpen) return;
    if (!loading && latestBackendMetaEntry === null) return;
    const timer = window.setInterval(() => {
      setComputeTraceNowMs(Date.now());
    }, 1000);
    return () => {
      window.clearInterval(timer);
    };
  }, [computeTraceOpen, latestBackendMetaEntry, loading]);

  useEffect(() => {
    if (!loading) return;
    if (progress && progress.total > 0) {
      setLiveMessage(
        t('live_computing_progress', {
          done: formatNumber(Math.min(progress.done, progress.total), locale, {
            maximumFractionDigits: 0,
          }),
          total: formatNumber(progress.total, locale, { maximumFractionDigits: 0 }),
        }),
      );
      return;
    }
    setLiveMessage(t('live_computing'));
  }, [loading, progress, t, locale]);

  const hasNameOverrides = Object.keys(routeNames).length > 0;

  const beginRename = useCallback(
    (routeId: string) => {
      if (busy) return;
      setEditingRouteId(routeId);
      setEditingName(routeNames[routeId] ?? labelsById[routeId] ?? '');
      markTutorialAction('routes.rename_start');
    },
    [busy, labelsById, markTutorialAction, routeNames],
  );

  const cancelRename = useCallback(() => {
    setEditingRouteId(null);
    setEditingName('');
  }, []);

  const commitRename = useCallback(() => {
    if (!editingRouteId) return;

    const trimmed = editingName.trim();
    setRouteNames((prev) => {
      const next = { ...prev };
      const defaultName = defaultLabelsById[editingRouteId];
      if (!trimmed || trimmed === defaultName) delete next[editingRouteId];
      else next[editingRouteId] = trimmed;
      return next;
    });

    setEditingRouteId(null);
    setEditingName('');
    markTutorialAction('routes.rename_save');
  }, [defaultLabelsById, editingName, editingRouteId, markTutorialAction]);

  const resetRouteNames = useCallback(() => {
    if (busy) return;
    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');
    markTutorialAction('routes.reset_names');
  }, [busy, markTutorialAction]);

  function handleMapClick(lat: number, lon: number) {
    setError(null);
    if (tutorialRunning && tutorialStep?.id === 'map_set_pins') {
      const now = Date.now();
      if (now < tutorialMapClickGuardUntilRef.current) {
        return;
      }
    }

    if (tutorialRunning && tutorialStep?.id === 'map_set_pins') {
      const canPlaceOrAdjustOrigin = tutorialPlacementStage === 'newcastle_origin';
      const canPlaceOrAdjustDestination = tutorialPlacementStage === 'london_destination';

      if (
        !canPlaceOrAdjustOrigin &&
        !canPlaceOrAdjustDestination
      ) {
        if (tutorialBlockingActionId === 'map.confirm_origin_newcastle') {
          setLiveMessage('Confirm Start with the map button near the Start marker before moving to London.');
          return;
        }
        if (tutorialBlockingActionId === 'map.confirm_destination_london') {
          setLiveMessage('Confirm End with the map button near the End marker before continuing this step.');
          return;
        }
        setLiveMessage('Follow the checklist order. Complete the current marker action first.');
        return;
      }
      const adjustingOrigin = canPlaceOrAdjustOrigin;
      const targetCity = adjustingOrigin
        ? TUTORIAL_CITY_TARGETS.newcastle
        : TUTORIAL_CITY_TARGETS.london;
      const cityLabel = adjustingOrigin ? 'Newcastle' : 'London';
      const pinLabel = adjustingOrigin ? 'Start' : 'End';
      const distanceKm = haversineDistanceKm(
        { lat, lon },
        { lat: targetCity.lat, lon: targetCity.lon },
      );

      if (distanceKm > targetCity.radiusKm) {
        setLiveMessage(
          `Place ${pinLabel} within ${targetCity.radiusKm} km of ${cityLabel}. Current click was outside the guided area.`,
        );
        return;
      }

      if (adjustingOrigin) {
        setTutorialDraftOrigin({ lat, lon });
        setSelectedPinId(null);
        markTutorialAction('map.set_origin_newcastle', { force: true });
        setLiveMessage('Start draft set near Newcastle. Use Confirm Start when ready, or click again to reposition.');
        return;
      }

      setTutorialDraftDestination({ lat, lon });
      setSelectedPinId(null);
      markTutorialAction('map.set_destination_london', { force: true });
      setLiveMessage('End draft set near London. Use Confirm End when ready, or click again to reposition.');
      return;
    }

    if (!origin) {
      setOrigin({ lat, lon });
      setSelectedPinId('origin');
      clearComputed();
      markTutorialAction('map.set_origin');
      return;
    }

    if (!destination) {
      setDestination({ lat, lon });
      setSelectedPinId('destination');
      clearComputed();
      markTutorialAction('map.set_destination');
      return;
    }

    setLiveMessage('Pins move by drag only. Drag Start, End, or Stop #1 to reposition.');
  }

  function handleMoveMarker(kind: MarkerKind, lat: number, lon: number): boolean {
    setError(null);

    if (tutorialRunning && tutorialStep?.id === 'map_set_pins' && tutorialPlacementStage === 'post_confirm_marker_actions') {
      const dragActionId = kind === 'origin' ? 'map.drag_origin_marker' : 'map.drag_destination_marker';
      const confirmActionId =
        kind === 'origin'
          ? 'map.confirm_drag_origin_marker'
          : 'map.confirm_drag_destination_marker';
      if (
        tutorialBlockingActionId !== dragActionId &&
        tutorialBlockingActionId !== confirmActionId
      ) {
        setLiveMessage('Follow the checklist order. Complete the current marker action first.');
        return false;
      }

      const targetCity = kind === 'origin' ? TUTORIAL_CITY_TARGETS.newcastle : TUTORIAL_CITY_TARGETS.london;
      const cityLabel = kind === 'origin' ? 'Newcastle' : 'London';
      const pinLabel = kind === 'origin' ? 'Start' : 'End';
      const distanceKm = haversineDistanceKm(
        { lat, lon },
        { lat: targetCity.lat, lon: targetCity.lon },
      );
      if (distanceKm > targetCity.radiusKm) {
        setLiveMessage(
          `Keep ${pinLabel} within ${targetCity.radiusKm} km of ${cityLabel} when dragging during tutorial.`,
        );
        return false;
      }

      if (kind === 'origin') {
        setTutorialDragDraftOrigin({ lat, lon });
      } else {
        setTutorialDragDraftDestination({ lat, lon });
      }
      setSelectedPinId(null);
      setFocusPinRequest(null);
      markTutorialAction(dragActionId, { force: true });
      setLiveMessage(
        kind === 'origin'
          ? 'Start drag draft set. Use Confirm Start Drag near the marker.'
          : 'End drag draft set. Use Confirm End Drag near the marker.',
      );
      return true;
    }

    if (kind === 'origin') setOrigin({ lat, lon });
    else setDestination({ lat, lon });

    clearComputed();
    markTutorialAction('map.drag_marker');
    if (kind === 'origin') {
      markTutorialAction('map.drag_origin_marker');
    } else {
      markTutorialAction('map.drag_destination_marker');
    }
    return true;
  }

  function handleMoveStop(lat: number, lon: number): boolean {
    setError(null);
    if (!managedStop) return false;
    if (
      tutorialRunning &&
      tutorialStep?.id === 'map_stop_lifecycle' &&
      tutorialBlockingActionId === 'map.drag_stop_marker'
    ) {
      const target = TUTORIAL_CITY_TARGETS.stoke;
      const distanceKm = haversineDistanceKm(
        { lat, lon },
        { lat: target.lat, lon: target.lon },
      );
      if (distanceKm > target.radiusKm) {
        const approxMiles = Math.round(target.radiusKm * 0.621371);
        setLiveMessage(
          `Keep midpoint stop inside the Stoke guided zone (~${approxMiles} miles radius).`,
        );
        logTutorialMidpoint('stop-drag:rejected-outside-guided-zone', {
          lat,
          lon,
          distanceKm,
          radiusKm: target.radiusKm,
          tutorialBlockingActionId,
        });
        return false;
      }
    }
    setManagedStop((prev) => (prev ? { ...prev, lat, lon } : prev));
    setSelectedPinId(null);
    setFocusPinRequest(null);
    clearComputed();
    return true;
  }

  function addStopFromMidpoint() {
    logTutorialMidpoint('add-stop:invoked', {
      tutorialRunning,
      tutorialStepId: tutorialStep?.id ?? null,
      tutorialBlockingActionId,
      tutorialLockScope,
      busy,
      canAddStop: Boolean(origin && destination),
      hasOrigin: Boolean(origin),
      hasDestination: Boolean(destination),
      hasManagedStop: Boolean(managedStop),
      origin,
      destination,
      managedStop,
      confirmedOriginRef: tutorialConfirmedOriginRef.current,
      confirmedDestinationRef: tutorialConfirmedDestinationRef.current,
    });
    const baseOrigin =
      origin ??
      (tutorialRunning && tutorialStep?.id === 'map_stop_lifecycle'
        ? tutorialConfirmedOriginRef.current
        : null);
    const baseDestination =
      destination ??
      (tutorialRunning && tutorialStep?.id === 'map_stop_lifecycle'
        ? tutorialConfirmedDestinationRef.current
        : null);
    if (!baseOrigin || !baseDestination) {
      logTutorialMidpoint('add-stop:exit-missing-base-pins', {
        baseOrigin,
        baseDestination,
      });
      return;
    }
    if (!origin) {
      logTutorialMidpoint('add-stop:set-origin-from-base', {
        baseOrigin,
      });
      setOrigin(baseOrigin);
    }
    if (!destination) {
      logTutorialMidpoint('add-stop:set-destination-from-base', {
        baseDestination,
      });
      setDestination(baseDestination);
    }
    const existingStops = extractIntermediateStops(dutyStopsText);
    const nextLabel = `Stop #${existingStops.length + 1}`;
    const midpoint = {
      id: 'stop-1' as const,
      lat: (baseOrigin.lat + baseDestination.lat) / 2,
      lon: (baseOrigin.lon + baseDestination.lon) / 2,
      label: nextLabel,
    };
    logTutorialMidpoint('add-stop:computed-midpoint', midpoint);
    const nextStops = [...existingStops, { lat: midpoint.lat, lon: midpoint.lon, label: midpoint.label }];
    dutySyncSourceRef.current = 'pins';
    setDutyStopsText(serializePinsToDutyText(baseOrigin, nextStops, baseDestination));
    const firstStop = nextStops[0];
    setManagedStop(
      firstStop
        ? {
            id: 'stop-1',
            lat: firstStop.lat,
            lon: firstStop.lon,
            label: firstStop.label || 'Stop #1',
          }
        : null,
    );
    setSelectedPinId('stop-1');
    clearComputed();
    logTutorialMidpoint('add-stop:mark-action', {
      actionId: 'pins.add_stop',
      force: true,
    });
    markTutorialAction('pins.add_stop', { force: true });
    logTutorialMidpoint('add-stop:mark-action', {
      actionId: 'map.add_stop_midpoint',
      force: false,
    });
    markTutorialAction('map.add_stop_midpoint');
    logTutorialMidpoint('add-stop:completed');
  }

  function renameStop(name: string) {
    const nextLabel = name.trim() || 'Stop #1';
    setManagedStop((prev) => {
      if (!prev) return prev;
      if (prev.label === nextLabel) return prev;
      return { ...prev, label: nextLabel };
    });
    markTutorialAction('map.rename_stop');
  }

  function deleteStop() {
    const existingStops = extractIntermediateStops(dutyStopsText);
    if (existingStops.length === 0) return;
    const nextStops = existingStops.slice(1).map((stop, idx) => ({
      lat: stop.lat,
      lon: stop.lon,
      label: stop.label || `Stop #${idx + 1}`,
    }));
    dutySyncSourceRef.current = 'pins';
    setDutyStopsText(serializePinsToDutyText(origin, nextStops, destination));
    const nextManagedStop = nextStops[0];
    setManagedStop(
      nextManagedStop
        ? {
            id: 'stop-1',
            lat: nextManagedStop.lat,
            lon: nextManagedStop.lon,
            label: nextManagedStop.label || 'Stop #1',
          }
        : null,
    );
    setSelectedPinId((prev) =>
      normalizeSelectedPinId(prev, {
        origin,
        destination,
        stop: nextManagedStop
          ? {
              id: 'stop-1',
              lat: nextManagedStop.lat,
              lon: nextManagedStop.lon,
              label: nextManagedStop.label || 'Stop #1',
            }
          : null,
      }),
    );
    clearComputed();
    markTutorialAction('map.delete_stop');
  }

  function handleTutorialConfirmPin(kind: 'origin' | 'destination') {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return;

    if (kind === 'origin') {
      if (tutorialPlacementStage !== 'newcastle_origin') return;
      const nextOrigin = tutorialDraftOrigin ?? (tutorialOriginPlaced ? origin : null);
      if (!nextOrigin) return;
      tutorialMapClickGuardUntilRef.current = Date.now() + 900;
      if (!tutorialActionSet.has('map.set_origin_newcastle')) {
        markTutorialAction('map.set_origin_newcastle', { force: true });
      }
      setOrigin(nextOrigin);
      tutorialConfirmedOriginRef.current = nextOrigin;
      setTutorialDraftOrigin(null);
      setDestination(null);
      tutorialConfirmedDestinationRef.current = null;
      setTutorialDraftDestination(null);
      setTutorialDragDraftOrigin(null);
      setTutorialDragDraftDestination(null);
      setTutorialStepActionsById((prev) => {
        const existing = prev[tutorialStep.id] ?? [];
        if (!existing.length) return prev;
        const resetActions = new Set([
          'map.set_destination_london',
          'map.confirm_destination_london',
          'map.click_destination_marker',
          'map.click_origin_marker',
          'map.popup_close_destination_marker',
          'map.popup_close_origin_marker',
          'map.popup_copy',
          'map.popup_close',
          'map.drag_destination_marker',
          'map.confirm_drag_destination_marker',
          'map.drag_origin_marker',
          'map.confirm_drag_origin_marker',
        ]);
        const next = existing.filter((actionId) => !resetActions.has(actionId));
        if (next.length === existing.length) return prev;
        return { ...prev, [tutorialStep.id]: next };
      });
      clearComputed();
      setSelectedPinId(null);
      window.requestAnimationFrame(() => {
        markTutorialAction('map.confirm_origin_newcastle', { force: true });
        setLiveMessage('Start confirmed. Now place End near London.');
      });
      return;
    }

    if (tutorialPlacementStage !== 'london_destination') return;
    const nextDestination = tutorialDraftDestination ?? (tutorialDestinationPlaced ? destination : null);
    if (!nextDestination) return;
    tutorialMapClickGuardUntilRef.current = Date.now() + 450;
    if (!tutorialActionSet.has('map.set_destination_london')) {
      markTutorialAction('map.set_destination_london', { force: true });
    }
    setDestination(nextDestination);
    tutorialConfirmedDestinationRef.current = nextDestination;
    setTutorialDraftDestination(null);
    setTutorialDragDraftDestination(null);
    clearComputed();
    markTutorialAction('map.confirm_destination_london', { force: true });
    setSelectedPinId(null);
    setLiveMessage('End confirmed. Click the End marker first, then Start.');
  }

  function handleTutorialConfirmDrag(kind: 'origin' | 'destination') {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return;
    if (tutorialPlacementStage !== 'post_confirm_marker_actions') return;

    if (kind === 'destination') {
      if (tutorialBlockingActionId !== 'map.confirm_drag_destination_marker') return;
      const nextDestination = tutorialDragDraftDestination;
      if (!nextDestination) return;
      setDestination(nextDestination);
      tutorialConfirmedDestinationRef.current = nextDestination;
      setTutorialDragDraftDestination(null);
      clearComputed();
      tutorialMapClickGuardUntilRef.current = Date.now() + 350;
      markTutorialAction('map.confirm_drag_destination_marker', { force: true });
      setSelectedPinId(null);
      setLiveMessage('End drag confirmed. Now drag Start within Newcastle and confirm.');
      return;
    }

    if (tutorialBlockingActionId !== 'map.confirm_drag_origin_marker') return;
    const nextOrigin = tutorialDragDraftOrigin;
    if (!nextOrigin) return;
    setOrigin(nextOrigin);
    tutorialConfirmedOriginRef.current = nextOrigin;
    setTutorialDragDraftOrigin(null);
    clearComputed();
    tutorialMapClickGuardUntilRef.current = Date.now() + 350;
    markTutorialAction('map.confirm_drag_origin_marker', { force: true });
    setSelectedPinId(null);
    setLiveMessage('Start drag confirmed. Step 1 marker workflow complete.');
    setTutorialStepIndex((prev) => Math.min(TUTORIAL_STEPS.length - 1, prev + 1));
  }

  function selectPinFromSidebar(id: 'origin' | 'destination' | 'stop-1') {
    setSelectedPinId((prev) => {
      const next = prev === id ? null : id;
      if (next) {
        focusPinNonceRef.current += 1;
        setFocusPinRequest({ id: next, nonce: focusPinNonceRef.current });
      } else {
        setFocusPinRequest(null);
        setFitAllRequestNonce((n) => n + 1);
      }
      return next;
    });
    markTutorialAction('pins.sidebar_select');
  }

  function swapMarkers() {
    if (!origin || !destination) return;
    setOrigin(destination);
    setDestination(origin);
    setSelectedPinId((prev) =>
      normalizeSelectedPinId(prev, {
        origin: destination,
        destination: origin,
        stop: managedStop,
      }),
    );
    clearComputed();
    markTutorialAction('setup.swap_pins_button');
    markTutorialAction('map.popup_swap');
  }

  function reset() {
    setOrigin(null);
    setDestination(null);
    setTutorialDraftOrigin(null);
    setTutorialDraftDestination(null);
    setTutorialDragDraftOrigin(null);
    setTutorialDragDraftDestination(null);
    setManagedStop(null);
    setSelectedPinId(null);
    setFocusPinRequest(null);
    tutorialConfirmedOriginRef.current = null;
    tutorialConfirmedDestinationRef.current = null;
    setFitAllRequestNonce((n) => n + 1);
    clearComputed();
    setError(null);
    setDutySyncError(null);
    markTutorialAction('setup.clear_pins_button');
  }

  function closeTutorial() {
    if (tutorialMode === 'running') {
      const snapshot: TutorialProgress = {
        version: 3,
        stepIndex: tutorialStepIndex,
        stepActionsById: tutorialStepActionsById,
        optionalDecisionsByStep: tutorialOptionalDecisionsByStep,
        updatedAt: new Date().toISOString(),
      };
      saveTutorialProgress(TUTORIAL_PROGRESS_KEY, snapshot);
      setTutorialSavedProgress(snapshot);
    }
    setTutorialOpen(false);
  }

  function startTutorialFresh() {
    clearComputed();
    setOrigin(null);
    setDestination(null);
    setTutorialDraftOrigin(null);
    setTutorialDraftDestination(null);
    setTutorialDragDraftOrigin(null);
    setTutorialDragDraftDestination(null);
    setManagedStop(null);
    setSelectedPinId(null);
    setFocusPinRequest(null);
    tutorialConfirmedOriginRef.current = null;
    tutorialConfirmedDestinationRef.current = null;
    setFitAllRequestNonce((n) => n + 1);
    setVehicleType('rigid_hgv');
    setScenarioMode('no_sharing');
    setWeights({ time: 60, money: 20, co2: 20 });
    setAdvancedParams(DEFAULT_ADVANCED_PARAMS);
    setComputeMode('pareto_stream');
    setDepEarliestArrivalLocal('');
    setDepLatestArrivalLocal('');
    setDutyStopsText('');
    setDutySyncError(null);
    setShowStopOverlay(true);
    setShowIncidentOverlay(true);
    setShowSegmentTooltips(true);
    setTutorialMode(tutorialIsDesktop ? 'running' : 'blocked');
    setTutorialStepActionsById({});
    setTutorialOptionalDecisionsByStep({});
    setTutorialStepIndex(0);
    setTutorialTargetRect(null);
    setTutorialTargetMissing(false);
    setTutorialPrefilledSteps([]);
    setTutorialExperimentPrefill(null);
    setTutorialSavedProgress(null);
    setTutorialResetNonce((prev) => prev + 1);
    clearTutorialProgress(TUTORIAL_PROGRESS_KEY);
    saveTutorialCompleted(TUTORIAL_COMPLETED_KEY, false);
    setTutorialCompleted(false);
    setTutorialOpen(true);
  }

  function resumeTutorialProgress() {
    if (!tutorialSavedProgress) {
      startTutorialFresh();
      return;
    }
    setTutorialStepActionsById(tutorialSavedProgress.stepActionsById ?? {});
    setTutorialOptionalDecisionsByStep(tutorialSavedProgress.optionalDecisionsByStep ?? {});
    setTutorialStepIndex(
      Math.max(0, Math.min(tutorialSavedProgress.stepIndex, Math.max(0, TUTORIAL_STEPS.length - 1))),
    );
    setTutorialMode(tutorialIsDesktop ? 'running' : 'blocked');
    setTutorialOpen(true);
  }

  function restartTutorialProgress() {
    startTutorialFresh();
  }

  function tutorialBack() {
    setTutorialStepIndex((prev) => Math.max(0, prev - 1));
  }

  function tutorialNext() {
    if (!tutorialCanAdvance) return;
    setTutorialStepIndex((prev) => Math.min(TUTORIAL_STEPS.length - 1, prev + 1));
  }

  function tutorialFinish() {
    if (!tutorialCanAdvance) return;
    setTutorialMode('completed');
    setTutorialCompleted(true);
    saveTutorialCompleted(TUTORIAL_COMPLETED_KEY, true);
    clearTutorialProgress(TUTORIAL_PROGRESS_KEY);
    setTutorialSavedProgress(null);
  }

  function parseNonNegativeOrDefault(raw: string, field: string, fallback: number): number {
    const trimmed = raw.trim();
    if (!trimmed) return fallback;
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed) || parsed < 0) {
      throw new Error(`${field} must be a non-negative number.`);
    }
    return parsed;
  }

  function parseOptionalNonNegative(raw: string, field: string): number | undefined {
    const trimmed = raw.trim();
    if (!trimmed) return undefined;
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed) || parsed < 0) {
      throw new Error(`${field} must be a non-negative number.`);
    }
    return parsed;
  }

  function parseOptionalInteger(raw: string, field: string): number | undefined {
    const trimmed = raw.trim();
    if (!trimmed) return undefined;
    const parsed = Number(trimmed);
    if (!Number.isInteger(parsed)) {
      throw new Error(`${field} must be an integer.`);
    }
    return parsed;
  }

  function parseBounded(raw: string, field: string, min: number, max: number): number {
    const parsed = Number(raw.trim());
    if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
      throw new Error(`${field} must be between ${min} and ${max}.`);
    }
    return parsed;
  }

  function buildAdvancedRequestPatch(): {
    maxAlternatives: number;
    advancedPatch: RoutingAdvancedPatch;
  } {
    const patch: RoutingAdvancedPatch = {};
    const maxAlternatives = Math.round(
      parseBounded(advancedParams.maxAlternatives, 'Max alternatives', 1, 48),
    );

    if (advancedParams.optimizationMode !== 'expected_value') {
      patch.optimization_mode = advancedParams.optimizationMode;
    }

    const riskAversion = parseNonNegativeOrDefault(
      advancedParams.riskAversion,
      'Risk aversion',
      1.0,
    );
    if (riskAversion !== 1.0) {
      patch.risk_aversion = riskAversion;
    }

    if (advancedParams.paretoMethod !== 'dominance') {
      patch.pareto_method = advancedParams.paretoMethod;
    }

    if (advancedParams.paretoMethod === 'epsilon_constraint') {
      const durationS = parseOptionalNonNegative(advancedParams.epsilonDurationS, 'Epsilon duration');
      const monetaryCost = parseOptionalNonNegative(
        advancedParams.epsilonMonetaryCost,
        'Epsilon monetary cost',
      );
      const emissionsKg = parseOptionalNonNegative(
        advancedParams.epsilonEmissionsKg,
        'Epsilon emissions',
      );

      if (durationS !== undefined || monetaryCost !== undefined || emissionsKg !== undefined) {
        patch.epsilon = {};
        if (durationS !== undefined) patch.epsilon.duration_s = durationS;
        if (monetaryCost !== undefined) patch.epsilon.monetary_cost = monetaryCost;
        if (emissionsKg !== undefined) patch.epsilon.emissions_kg = emissionsKg;
      }
    }

    if (advancedParams.departureTimeUtcLocal.trim()) {
      const dt = new Date(advancedParams.departureTimeUtcLocal);
      if (Number.isNaN(dt.getTime())) {
        throw new Error('Departure time must be a valid date/time.');
      }
      patch.departure_time_utc = dt.toISOString();
    }

    if (advancedParams.stochasticEnabled) {
      const sigma = parseBounded(advancedParams.stochasticSigma, 'Stochastic sigma', 0, 0.5);
      const samples = parseBounded(advancedParams.stochasticSamples, 'Stochastic samples', 5, 200);
      const seed = parseOptionalInteger(advancedParams.stochasticSeed, 'Stochastic seed');
      patch.stochastic = {
        enabled: true,
        sigma,
        samples: Math.round(samples),
        ...(seed !== undefined ? { seed } : {}),
      };
    }

    const fuelPriceMultiplier = parseNonNegativeOrDefault(
      advancedParams.fuelPriceMultiplier,
      'Fuel price multiplier',
      1.0,
    );
    const carbonPricePerKg = parseNonNegativeOrDefault(
      advancedParams.carbonPricePerKg,
      'Carbon price',
      0.0,
    );
    const tollCostPerKm = parseNonNegativeOrDefault(
      advancedParams.tollCostPerKm,
      'Toll cost per km',
      0.0,
    );

    if (
      !advancedParams.useTolls ||
      fuelPriceMultiplier !== 1.0 ||
      carbonPricePerKg !== 0.0 ||
      tollCostPerKm !== 0.0
    ) {
      patch.cost_toggles = {
        use_tolls: advancedParams.useTolls,
        fuel_price_multiplier: fuelPriceMultiplier,
        carbon_price_per_kg: carbonPricePerKg,
        toll_cost_per_km: tollCostPerKm,
      };
    }

    if (advancedParams.terrainProfile !== 'flat') {
      patch.terrain_profile = advancedParams.terrainProfile;
    }

    const ambientTempC = parseBounded(advancedParams.ambientTempC, 'Ambient temperature', -60, 70);
    patch.emissions_context = {
      fuel_type: advancedParams.fuelType as FuelType,
      euro_class: advancedParams.euroClass as EuroClass,
      ambient_temp_c: ambientTempC,
    } satisfies EmissionsContext;

    const weatherIntensity = parseBounded(advancedParams.weatherIntensity, 'Weather intensity', 0, 2);
    if (
      advancedParams.weatherEnabled ||
      advancedParams.weatherProfile !== 'clear' ||
      Math.abs(weatherIntensity - 1.0) > 1e-9 ||
      !advancedParams.weatherIncidentUplift
    ) {
      patch.weather = {
        enabled: advancedParams.weatherEnabled,
        profile: advancedParams.weatherProfile as WeatherProfile,
        intensity: weatherIntensity,
        apply_incident_uplift: advancedParams.weatherIncidentUplift,
      } satisfies WeatherImpactConfig;
    }

    const incidentSeed = parseOptionalInteger(advancedParams.incidentSeed, 'Incident seed');
    const incidentDwellRate = parseNonNegativeOrDefault(
      advancedParams.incidentDwellRatePer100km,
      'Incident dwell rate',
      0.8,
    );
    const incidentAccidentRate = parseNonNegativeOrDefault(
      advancedParams.incidentAccidentRatePer100km,
      'Incident accident rate',
      0.25,
    );
    const incidentClosureRate = parseNonNegativeOrDefault(
      advancedParams.incidentClosureRatePer100km,
      'Incident closure rate',
      0.05,
    );
    const incidentDwellDelay = parseNonNegativeOrDefault(
      advancedParams.incidentDwellDelayS,
      'Incident dwell delay',
      120,
    );
    const incidentAccidentDelay = parseNonNegativeOrDefault(
      advancedParams.incidentAccidentDelayS,
      'Incident accident delay',
      480,
    );
    const incidentClosureDelay = parseNonNegativeOrDefault(
      advancedParams.incidentClosureDelayS,
      'Incident closure delay',
      900,
    );
    const incidentMaxEvents = Math.round(
      parseBounded(advancedParams.incidentMaxEventsPerRoute, 'Incident max events', 0, 1000),
    );

    if (
      advancedParams.incidentSimulationEnabled ||
      incidentSeed !== undefined ||
      Math.abs(incidentDwellRate - 0.8) > 1e-9 ||
      Math.abs(incidentAccidentRate - 0.25) > 1e-9 ||
      Math.abs(incidentClosureRate - 0.05) > 1e-9 ||
      Math.abs(incidentDwellDelay - 120) > 1e-9 ||
      Math.abs(incidentAccidentDelay - 480) > 1e-9 ||
      Math.abs(incidentClosureDelay - 900) > 1e-9 ||
      incidentMaxEvents !== 12
    ) {
      patch.incident_simulation = {
        enabled: advancedParams.incidentSimulationEnabled,
        dwell_rate_per_100km: incidentDwellRate,
        accident_rate_per_100km: incidentAccidentRate,
        closure_rate_per_100km: incidentClosureRate,
        dwell_delay_s: incidentDwellDelay,
        accident_delay_s: incidentAccidentDelay,
        closure_delay_s: incidentClosureDelay,
        max_events_per_route: incidentMaxEvents,
        ...(incidentSeed !== undefined ? { seed: incidentSeed } : {}),
      } satisfies IncidentSimulatorConfig;
    }

    return { maxAlternatives, advancedPatch: patch };
  }

  function buildScenarioCompareRequest(originPoint: LatLng, destinationPoint: LatLng): ScenarioCompareRequest {
    const { maxAlternatives, advancedPatch } = buildAdvancedRequestPatch();
    return buildScenarioCompareRequestPayload({
      origin: originPoint,
      destination: destinationPoint,
      waypoints: requestWaypoints,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: maxAlternatives,
      weights: {
        time: weights.time,
        money: weights.money,
        co2: weights.co2,
      },
      advanced: advancedPatch,
    });
  }

  async function computePareto() {
    if (!origin || !destination) {
      setError('Click the map to set Start, then End.');
      return;
    }
    let readinessSnapshot: HealthReadyResponse | null = null;
    try {
      readinessSnapshot = await getJSON<HealthReadyResponse>('/api/health/ready');
      setOpsHealthReady(readinessSnapshot);
    } catch (e: unknown) {
      const cause = e instanceof Error ? e.message : 'Unable to reach backend /health/ready endpoint.';
      const reasonCode = 'backend_unreachable';
      const blockedMessage = `Unable to reach backend /health/ready endpoint. (${reasonCode}) cause=${cause}`;
      setOpsHealthReady({
        status: 'not_ready',
        strict_route_ready: false,
        recommended_action: 'retry',
        route_graph: {
          ok: false,
          state: 'failed',
          phase: 'backend_unreachable',
          last_error: blockedMessage,
        },
      });
      setError(blockedMessage);
      resetComputeTrace();
      computeStartedAtMsRef.current = Date.now();
      setComputeTraceOpen(true);
      appendComputeTrace({
        level: 'warn',
        step: 'Compute blocked',
        detail: blockedMessage,
        reasonCode,
        recoveries: inferComputeRecoveryHints(blockedMessage),
      });
      return;
    }
    const strictLive = readinessSnapshot?.strict_live;
    if (strictLive && strictLive.ok === false) {
      const strictLiveStatus = String(strictLive.status ?? '').trim().toLowerCase();
      const reasonCode = String(strictLive.reason_code ?? 'scenario_profile_unavailable').trim() || 'scenario_profile_unavailable';
      const asOf = String(strictLive.as_of_utc ?? '').trim();
      const ageMinutes = toFiniteNumber(strictLive.age_minutes);
      const maxAgeMinutes = toFiniteNumber(strictLive.max_age_minutes);
      const staleDetailParts: string[] = [];
      if (asOf) staleDetailParts.push(`as_of_utc=${asOf}`);
      if (ageMinutes !== null) staleDetailParts.push(`age_minutes=${ageMinutes.toFixed(2)}`);
      if (maxAgeMinutes !== null) staleDetailParts.push(`max_age_minutes=${Math.round(maxAgeMinutes)}`);
      const staleDetail = staleDetailParts.length ? ` (${staleDetailParts.join('; ')})` : '';
      const blockedMessage =
        strictLiveStatus === 'stale'
          ? `Strict live scenario coefficients are stale${staleDetail}. Refresh live scenario coefficients before retrying.`
          : String(strictLive.message ?? 'Strict live source readiness failed.').trim() ||
            'Strict live source readiness failed.';
      setError(blockedMessage);
      resetComputeTrace();
      computeStartedAtMsRef.current = Date.now();
      setComputeTraceOpen(true);
      appendComputeTrace({
        level: 'warn',
        step: 'Compute blocked',
        detail: blockedMessage,
        reasonCode,
        recoveries: inferComputeRecoveryHints(`${reasonCode} ${blockedMessage}`),
      });
      return;
    }
    if (!readinessSnapshot || readinessSnapshot.strict_route_ready !== true) {
      const routeGraph = readinessSnapshot?.route_graph ?? {};
      const routeGraphStateNow = String(routeGraph.state ?? '').trim().toLowerCase();
      const routeGraphPhaseNow = String(routeGraph.phase ?? '').trim().toLowerCase();
      const elapsedLabel = formatDurationCompactMs(toFiniteNumber(routeGraph.elapsed_ms));
      const phaseLabel = warmupPhaseLabel(routeGraphPhaseNow || routeGraphStateNow);
      const reasonCode =
        routeGraphStateNow === 'failed' ? 'routing_graph_warmup_failed' : 'routing_graph_warming_up';
      const blockedMessage =
        reasonCode === 'routing_graph_warmup_failed'
          ? `Routing graph warmup failed: ${String(routeGraph.last_error ?? 'unknown warmup failure')}`
          : `Routing graph is still warming up (${phaseLabel}; elapsed=${elapsedLabel}). Retry when GET /health/ready reports strict_route_ready=true.`;
      setError(blockedMessage);
      resetComputeTrace();
      computeStartedAtMsRef.current = Date.now();
      setComputeTraceOpen(true);
      appendComputeTrace({
        level: 'warn',
        step: 'Compute blocked',
        detail: blockedMessage,
        reasonCode,
        recoveries: inferComputeRecoveryHints(reasonCode),
      });
      return;
    }
    markTutorialAction('pref.compute_pareto_click');

    const seq = requestSeqRef.current + 1;
    requestSeqRef.current = seq;

    abortActiveCompute();

    const runController = new AbortController();
    abortRef.current = runController;
    const streamStallTimeoutMs = 90_000;
    const attemptTimeoutMs = parsePositiveIntEnv(process.env.NEXT_PUBLIC_COMPUTE_ATTEMPT_TIMEOUT_MS, 1_200_000);
    const routeFallbackTimeoutMs = parsePositiveIntEnv(
      process.env.NEXT_PUBLIC_COMPUTE_ROUTE_FALLBACK_TIMEOUT_MS,
      900_000,
    );
    const degradeDefaults = parseDegradeSteps(process.env.NEXT_PUBLIC_COMPUTE_DEGRADE_STEPS);
    let heartbeatTimer: number | null = null;
    let activeAttemptNumber: number | null = null;
    let activeAttemptEndpoint: '/api/pareto/stream' | '/api/pareto' | '/api/route' | null = null;
    let activeAttemptAlternatives = 0;
    let activeAttemptTimeoutMs = 0;
    const liveCallPollTimers = new Map<number, number>();
    const liveCallRequestByAttempt = new Map<number, string>();
    const liveCallSummaryByAttempt = new Map<number, string>();

    resetComputeTrace();
    setComputeLiveCallsByAttempt({});
    setComputeRequestIdByAttempt({});
    computeStartedAtMsRef.current = Date.now();
    setComputeTraceOpen(true);
    appendComputeTrace({
      level: 'info',
      step: 'Compute initialised',
      detail: `seq=${seq}; mode=${computeMode}; scenario=${scenarioMode}; vehicle=${vehicleType}; origin=(${origin.lat.toFixed(5)},${origin.lon.toFixed(5)}); destination=(${destination.lat.toFixed(5)},${destination.lon.toFixed(5)})`,
    });

    routeBufferRef.current = [];
    clearFlushTimer();

    setLoading(true);
    setError(null);
    setProgress(null);
    setWarnings([]);
    setShowWarnings(false);
    setParetoRoutes([]);
    setSelectedId(null);
    setRouteNames({});
    setEditingRouteId(null);
    setEditingName('');
    setScenarioCompare(null);
    setScenarioCompareError(null);
    setScenarioCompareLoading(false);
    setAdvancedError(null);

    heartbeatTimer = window.setInterval(() => {
      const latestProgress = progressRef.current;
      appendComputeTrace({
        level: 'info',
        step: 'Compute heartbeat',
        detail: `Still running; mode=${computeMode}; attempt=${activeAttemptNumber ?? 0}; endpoint=${activeAttemptEndpoint ?? 'n/a'}; alternatives=${activeAttemptAlternatives || 'n/a'}; timeout_ms=${activeAttemptTimeoutMs || 'n/a'}; progress=${latestProgress ? `${latestProgress.done}/${latestProgress.total}` : 'n/a'}`,
        attempt: activeAttemptNumber ?? undefined,
        endpoint: activeAttemptEndpoint ?? undefined,
        alternativesUsed: activeAttemptAlternatives || undefined,
        timeoutMs: activeAttemptTimeoutMs || undefined,
      });
    }, 15_000);

    let advancedPatch: RoutingAdvancedPatch;
    let maxAlternatives = 5;
    let alternativesSequence: number[] = [maxAlternatives];
    try {
      const parsed = buildAdvancedRequestPatch();
      advancedPatch = parsed.advancedPatch;
      maxAlternatives = parsed.maxAlternatives;
      alternativesSequence = buildAlternativesSequence(maxAlternatives, degradeDefaults);
      appendComputeTrace({
        level: 'info',
        step: 'Advanced parameters validated',
        detail: `max_alternatives=${maxAlternatives}; degradation=${alternativesSequence.join('->')}`,
      });
      setComputeSession({
        seq,
        startedAt: new Date().toISOString(),
        mode: computeMode,
        scenario: scenarioMode,
        vehicle: vehicleType,
        waypointCount: requestWaypoints.length,
        maxAlternatives,
        streamStallTimeoutMs,
        origin,
        destination,
        attemptTimeoutMs,
        routeFallbackTimeoutMs,
        degradeSteps: alternativesSequence,
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Invalid advanced parameter values.';
      appendComputeTrace({
        level: 'error',
        step: 'Advanced parameter validation failed',
        detail: msg,
        recoveries: inferComputeRecoveryHints(msg),
      });
      setAdvancedError(msg);
      setError(msg);
      setLoading(false);
      abortRef.current = null;
      if (heartbeatTimer !== null) {
        window.clearInterval(heartbeatTimer);
      }
      return;
    }

    const buildParetoBody = (alternatives: number): ParetoRequest =>
      buildParetoRequest({
        origin,
        destination,
        waypoints: requestWaypoints,
        vehicle_type: vehicleType,
        scenario_mode: scenarioMode,
        max_alternatives: alternatives,
        weights: {
          time: weights.time,
          money: weights.money,
          co2: weights.co2,
        },
        advanced: advancedPatch,
      });
    const buildRouteBody = (alternatives: number): RouteRequest =>
      buildRouteRequest({
        origin,
        destination,
        waypoints: requestWaypoints,
        vehicle_type: vehicleType,
        scenario_mode: scenarioMode,
        max_alternatives: alternatives,
        weights: {
          time: weights.time,
          money: weights.money,
          co2: weights.co2,
        },
        advanced: advancedPatch,
      });
    appendComputeTrace({
      level: 'info',
      step: 'Payload prepared',
      detail: `weights(time=${weights.time}, money=${weights.money}, co2=${weights.co2}); normalised(time=${normalisedWeights.time.toFixed(3)}, money=${normalisedWeights.money.toFixed(3)}, co2=${normalisedWeights.co2.toFixed(3)}); waypoints=${requestWaypoints.length}; attempt_timeout_ms=${attemptTimeoutMs}; route_fallback_timeout_ms=${routeFallbackTimeoutMs}`,
    });

    type AttemptPlan = {
      attempt: number;
      kind: ComputeMode;
      endpoint: '/api/pareto/stream' | '/api/pareto' | '/api/route';
      alternatives: number;
      timeoutMs: number;
    };
    type AttemptRuntime = {
      controller: AbortController;
      signal: AbortSignal;
      timeoutTriggered: () => boolean;
      cleanup: () => void;
    };
    type AttemptResult = {
      ok: boolean;
      aborted: boolean;
      message?: string;
      reasonCode?: string;
      statusCode?: number;
    };

    const stopAttemptLiveCallPolling = (attemptNumber: number) => {
      const timer = liveCallPollTimers.get(attemptNumber);
      if (typeof timer === 'number') {
        window.clearInterval(timer);
        liveCallPollTimers.delete(attemptNumber);
      }
    };

    const stopAllLiveCallPolling = () => {
      liveCallPollTimers.forEach((timer) => {
        window.clearInterval(timer);
      });
      liveCallPollTimers.clear();
    };

    const isIgnorableLiveCallError = (message: string): boolean => {
      const lower = message.toLowerCase();
      return (
        lower.includes('no live-call trace found') ||
        lower.includes('debug live-call diagnostics are disabled')
      );
    };

    const syncAttemptLiveCalls = async (
      attempt: AttemptPlan,
      requestId: string,
      opts?: { step?: string; forceTrace?: boolean; traceErrors?: boolean },
    ): Promise<LiveCallTraceResponse | null> => {
      if (seq !== requestSeqRef.current || runController.signal.aborted) return null;
      try {
        const payload = await getJSON<LiveCallTraceResponse>(
          `/api/debug/live-calls/${encodeURIComponent(requestId)}`,
        );
        if (seq !== requestSeqRef.current || runController.signal.aborted) return null;
        setComputeLiveCallsByAttempt((prev) => ({ ...prev, [attempt.attempt]: payload }));
        const summary = payload.summary;
        const summaryKey = `${summary?.total_calls ?? 0}|${summary?.failed_calls ?? 0}|${summary?.expected_satisfied ?? 0}|${summary?.expected_miss_count ?? 0}|${summary?.expected_blocked_count ?? 0}|${payload.status}`;
        const changed = liveCallSummaryByAttempt.get(attempt.attempt) !== summaryKey;
        if (changed || opts?.forceTrace) {
          liveCallSummaryByAttempt.set(attempt.attempt, summaryKey);
          const blockedExpected = (payload.expected_rollup ?? []).filter((row) => row.status === 'blocked').length;
          const notReachedExpected = (payload.expected_rollup ?? []).filter((row) => row.status === 'not_reached').length;
          const missedExpected = (payload.expected_rollup ?? []).filter((row) => row.status === 'miss').length;
          appendComputeTrace({
            level: 'info',
            step: opts?.step ?? 'Live API trace update',
            detail: `request_id=${requestId}; calls=${summary?.total_calls ?? 0}; requested=${summary?.requested_calls ?? 0}; success=${summary?.successful_calls ?? 0}; failed=${summary?.failed_calls ?? 0}; blocked=${blockedExpected}; not_reached=${notReachedExpected}; miss=${missedExpected}; cache_hits=${summary?.cache_hit_calls ?? 0}; expected=${summary?.expected_satisfied ?? 0}/${summary?.expected_total ?? 0}`,
            attempt: attempt.attempt,
            endpoint: attempt.endpoint,
            requestId,
            alternativesUsed: attempt.alternatives,
            timeoutMs: attempt.timeoutMs,
          });
        }
        return payload;
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : 'Failed to fetch live-call diagnostics.';
        if (isIgnorableLiveCallError(message)) return null;
        if (opts?.traceErrors === false) return null;
        const errorKey = `error:${message}`;
        if (liveCallSummaryByAttempt.get(attempt.attempt) === errorKey) return null;
        liveCallSummaryByAttempt.set(attempt.attempt, errorKey);
        appendComputeTrace({
          level: 'warn',
          step: 'Live API trace unavailable',
          detail: message,
          attempt: attempt.attempt,
          endpoint: attempt.endpoint,
          requestId,
          alternativesUsed: attempt.alternatives,
          timeoutMs: attempt.timeoutMs,
        });
        return null;
      }
    };

    const attachAttemptRequestId = (attempt: AttemptPlan, requestIdRaw: string | null | undefined) => {
      const requestId = String(requestIdRaw || '').trim();
      if (!requestId) return;
      const existing = liveCallRequestByAttempt.get(attempt.attempt);
      if (existing === requestId) return;
      liveCallRequestByAttempt.set(attempt.attempt, requestId);
      setComputeRequestIdByAttempt((prev) => ({ ...prev, [attempt.attempt]: requestId }));
      appendComputeTrace({
        level: 'info',
        step: 'Attempt request correlated',
        detail: `Captured backend request_id for live-call tracing.`,
        attempt: attempt.attempt,
        endpoint: attempt.endpoint,
        requestId,
        alternativesUsed: attempt.alternatives,
        timeoutMs: attempt.timeoutMs,
      });

      stopAttemptLiveCallPolling(attempt.attempt);
      const pollLiveCalls = async () => {
        await syncAttemptLiveCalls(attempt, requestId, { step: 'Live API trace update' });
      };

      void pollLiveCalls();
      const timer = window.setInterval(() => {
        void pollLiveCalls();
      }, 1500);
      liveCallPollTimers.set(attempt.attempt, timer);
    };

    const applyParetoPayload = (
      payload: ParetoResponse,
      source: 'pareto_json' | 'stream_done' | 'stream_fallback',
      attempt: AttemptPlan,
    ) => {
      if (seq !== requestSeqRef.current) return;
      const finalRoutes = sortRoutesDeterministic(payload.routes ?? []);
      startTransition(() => {
        setParetoRoutes(finalRoutes);
      });
      setProgress({ done: finalRoutes.length, total: finalRoutes.length });
      if (payload.warnings?.length) {
        const payloadWarnings = payload.warnings ?? [];
        setWarnings((prev) => dedupeWarnings([...prev, ...payloadWarnings]));
      }
      markTutorialAction('pref.compute_pareto_done');
      appendComputeTrace({
        level: 'success',
        step:
          source === 'pareto_json'
            ? 'JSON compute complete'
            : source === 'stream_done'
              ? 'Stream compute complete'
              : 'Fallback JSON compute complete',
        detail: `routes=${finalRoutes.length}`,
        attempt: attempt.attempt,
        endpoint: attempt.endpoint,
        alternativesUsed: attempt.alternatives,
        timeoutMs: attempt.timeoutMs,
      });
    };

    const applyRoutePayload = (
      payload: RouteResponse,
      source: 'route_single' | 'json_fallback_single' | 'stream_fallback_single',
      attempt: AttemptPlan,
    ) => {
      if (seq !== requestSeqRef.current) return;
      const routeMap = new Map<string, RouteOption>();
      [payload.selected, ...(payload.candidates ?? [])].forEach((route) => routeMap.set(route.id, route));
      const finalRoutes = sortRoutesDeterministic(Array.from(routeMap.values()));
      startTransition(() => {
        setParetoRoutes(finalRoutes);
      });
      setSelectedId(payload.selected.id);
      setProgress({ done: finalRoutes.length, total: finalRoutes.length });
      markTutorialAction('pref.compute_pareto_done');
      appendComputeTrace({
        level: 'success',
        step:
          source === 'route_single'
            ? 'Single-route compute complete'
            : source === 'json_fallback_single'
              ? 'JSON fallback converted to single-route complete'
              : 'Stream fallback converted to single-route complete',
        detail: `selected=${payload.selected.id}; candidates=${finalRoutes.length}`,
        attempt: attempt.attempt,
        endpoint: attempt.endpoint,
        alternativesUsed: attempt.alternatives,
        timeoutMs: attempt.timeoutMs,
      });
    };

    const startAttemptRuntime = (attempt: AttemptPlan): AttemptRuntime => {
      const controller = new AbortController();
      let timeoutTriggered = false;
      const onRunAbort = () => {
        try {
          controller.abort(runController.signal.reason ?? new Error('compute_run_aborted'));
        } catch {
          controller.abort();
        }
      };
      if (runController.signal.aborted) {
        onRunAbort();
      } else {
        runController.signal.addEventListener('abort', onRunAbort, { once: true });
      }
      const timeoutId = window.setTimeout(() => {
        timeoutTriggered = true;
        try {
          controller.abort(new Error(`attempt_timeout:${attempt.timeoutMs}`));
        } catch {
          controller.abort();
        }
      }, attempt.timeoutMs);
      activeAttemptAbortRef.current = controller;
      return {
        controller,
        signal: controller.signal,
        timeoutTriggered: () => timeoutTriggered,
        cleanup: () => {
          window.clearTimeout(timeoutId);
          runController.signal.removeEventListener('abort', onRunAbort);
          stopAttemptLiveCallPolling(attempt.attempt);
          if (activeAttemptAbortRef.current === controller) {
            activeAttemptAbortRef.current = null;
          }
        },
      };
    };

    const transitionDetail = (from: AttemptPlan, to: AttemptPlan): string => {
      const alternativesNote =
        from.alternatives === to.alternatives
          ? `alternatives=${to.alternatives}`
          : `degraded alternatives ${from.alternatives} -> ${to.alternatives}`;
      return `Switching from ${from.endpoint} to ${to.endpoint}; ${alternativesNote}; timeout_ms=${to.timeoutMs}.`;
    };

    const attemptPlans: AttemptPlan[] =
      computeMode === 'pareto_stream'
        ? [
            {
              attempt: 1,
              kind: 'pareto_stream',
              endpoint: '/api/pareto/stream',
              alternatives: alternativesSequence[0] ?? Math.max(1, maxAlternatives),
              timeoutMs: attemptTimeoutMs,
            },
            {
              attempt: 2,
              kind: 'pareto_json',
              endpoint: '/api/pareto',
              alternatives: alternativesSequence[1] ?? alternativesSequence[0] ?? Math.max(1, maxAlternatives),
              timeoutMs: attemptTimeoutMs,
            },
            {
              attempt: 3,
              kind: 'route_single',
              endpoint: '/api/route',
              alternatives:
                alternativesSequence[2] ??
                alternativesSequence[1] ??
                alternativesSequence[0] ??
                Math.max(1, maxAlternatives),
              timeoutMs: routeFallbackTimeoutMs,
            },
          ]
        : computeMode === 'pareto_json'
          ? [
              {
                attempt: 1,
                kind: 'pareto_json',
                endpoint: '/api/pareto',
                alternatives: alternativesSequence[0] ?? Math.max(1, maxAlternatives),
                timeoutMs: attemptTimeoutMs,
              },
              {
                attempt: 2,
                kind: 'route_single',
                endpoint: '/api/route',
                alternatives: alternativesSequence[1] ?? alternativesSequence[0] ?? Math.max(1, maxAlternatives),
                timeoutMs: routeFallbackTimeoutMs,
              },
            ]
          : [
              {
                attempt: 1,
                kind: 'route_single',
                endpoint: '/api/route',
                alternatives: alternativesSequence[0] ?? Math.max(1, maxAlternatives),
                timeoutMs: routeFallbackTimeoutMs,
              },
            ];

    try {
      let succeeded = false;
      let finalFailure: AttemptResult | null = null;

      for (let idx = 0; idx < attemptPlans.length; idx += 1) {
        const attempt = attemptPlans[idx];
        if (seq !== requestSeqRef.current || runController.signal.aborted) return;

        const runtime = startAttemptRuntime(attempt);
        activeAttemptNumber = attempt.attempt;
        activeAttemptEndpoint = attempt.endpoint;
        activeAttemptAlternatives = attempt.alternatives;
        activeAttemptTimeoutMs = attempt.timeoutMs;
        setProgress({ done: 0, total: Math.max(1, attempt.alternatives) });
        setComputeTraceNowMs(Date.now());

        appendComputeTrace({
          level: 'info',
          step: 'Attempt started',
          detail: `attempt=${attempt.attempt}; endpoint=${attempt.endpoint}; alternatives=${attempt.alternatives}; timeout_ms=${attempt.timeoutMs}`,
          attempt: attempt.attempt,
          endpoint: attempt.endpoint,
          alternativesUsed: attempt.alternatives,
          timeoutMs: attempt.timeoutMs,
        });

        const runAttempt = async (): Promise<AttemptResult> => {
          if (attempt.kind === 'pareto_stream') {
            let sawDone = false;
            let streamFatalMessage: string | null = null;
            let streamFatalReasonCode: string | undefined;
            let previousMetaStage = '';
            let previousMetaDetail = '';
            let previousMetaHeartbeat = -1;
            let previousMetaDone = -1;
            let previousMetaTotal = -1;

            appendComputeTrace({
              level: 'info',
              step: 'Starting stream request',
              detail: `POST /api/pareto/stream; idle_timeout_ms=${streamStallTimeoutMs}`,
              attempt: attempt.attempt,
              endpoint: attempt.endpoint,
              alternativesUsed: attempt.alternatives,
              timeoutMs: attempt.timeoutMs,
            });

            try {
              await postNDJSON<ParetoStreamEvent>('/api/pareto/stream', buildParetoBody(attempt.alternatives), {
                signal: runtime.signal,
                stallTimeoutMs: streamStallTimeoutMs,
                onEvent: (event) => {
                  if (seq !== requestSeqRef.current) return;
                  if (runController.signal.aborted) return;

                  switch (event.type) {
                    case 'meta': {
                      const done =
                        typeof event.candidate_done === 'number'
                          ? event.candidate_done
                          : typeof event.done === 'number'
                            ? event.done
                            : 0;
                      const total =
                        typeof event.candidate_total === 'number' && event.candidate_total > 0
                          ? event.candidate_total
                          : event.total;
                      setProgress({ done, total });
                      const stage = event.stage ?? 'collecting_candidates';
                      const stageDetail = event.stage_detail ?? 'n/a';
                      const heartbeat = typeof event.heartbeat === 'number' ? event.heartbeat : -1;
                      const requestId = event.request_id;
                      if (requestId) {
                        attachAttemptRequestId(attempt, requestId);
                      }
                      const shouldTrace =
                        stage !== previousMetaStage ||
                        stageDetail !== previousMetaDetail ||
                        heartbeat !== previousMetaHeartbeat ||
                        done !== previousMetaDone ||
                        total !== previousMetaTotal;
                      if (shouldTrace) {
                        appendComputeTrace({
                          level: 'info',
                          step: heartbeat > 0 ? 'Compute heartbeat' : 'Stream metadata received',
                          detail:
                            `stage=${stage}; stage_detail=${stageDetail}; heartbeat=${Math.max(0, heartbeat)}; progress=${done}/${total}` +
                            (requestId ? `; request_id=${requestId}` : ''),
                          attempt: attempt.attempt,
                          endpoint: attempt.endpoint,
                          requestId,
                          stage,
                          stageDetail,
                          backendElapsedMs: event.elapsed_ms,
                          stageElapsedMs: event.stage_elapsed_ms,
                          candidateDiagnostics: event.candidate_diagnostics ?? null,
                          alternativesUsed: attempt.alternatives,
                          timeoutMs: attempt.timeoutMs,
                        });
                        previousMetaStage = stage;
                        previousMetaDetail = stageDetail;
                        previousMetaHeartbeat = heartbeat;
                        previousMetaDone = done;
                        previousMetaTotal = total;
                      }
                      return;
                    }

                    case 'route': {
                      routeBufferRef.current.push(event.route);
                      scheduleRouteFlush(seq);
                      setProgress({ done: event.done, total: event.total });
                      appendComputeTrace({
                        level: 'info',
                        step: `Candidate route ${event.done}/${event.total}`,
                        detail: `${event.route.id} duration=${event.route.metrics.duration_s.toFixed(1)}s cost=${event.route.metrics.monetary_cost.toFixed(2)} co2=${event.route.metrics.emissions_kg.toFixed(3)}kg`,
                        attempt: attempt.attempt,
                        endpoint: attempt.endpoint,
                        alternativesUsed: attempt.alternatives,
                        timeoutMs: attempt.timeoutMs,
                      });
                      return;
                    }

                    case 'error': {
                      setProgress({ done: event.done, total: event.total });
                      setWarnings((prev) => dedupeWarnings([...prev, event.message]));
                      appendComputeTrace({
                        level: 'warn',
                        step: `Candidate warning ${event.done}/${event.total}`,
                        detail: event.message,
                        attempt: attempt.attempt,
                        endpoint: attempt.endpoint,
                        alternativesUsed: attempt.alternatives,
                        timeoutMs: attempt.timeoutMs,
                      });
                      return;
                    }

                    case 'fatal': {
                      const fatalMessage = event.message || 'Route computation failed.';
                      streamFatalMessage = fatalMessage;
                      streamFatalReasonCode = event.reason_code;
                      if (event.request_id) {
                        attachAttemptRequestId(attempt, event.request_id);
                      }
                      setWarnings((prev) =>
                        dedupeWarnings([
                          ...prev,
                          `Stream returned fatal event (${event.reason_code ?? 'unknown'}): ${fatalMessage}`,
                        ]),
                      );
                      appendComputeTrace({
                        level: 'error',
                        step: 'Stream fatal error',
                        detail: fatalMessage,
                        reasonCode: event.reason_code,
                        recoveries: inferComputeRecoveryHints(fatalMessage),
                        attempt: attempt.attempt,
                        endpoint: attempt.endpoint,
                        requestId: event.request_id,
                        stage: event.stage,
                        stageDetail: event.stage_detail,
                        backendElapsedMs: event.elapsed_ms,
                        stageElapsedMs: event.stage_elapsed_ms,
                        candidateDiagnostics: event.candidate_diagnostics ?? null,
                        failureChain: event.failure_chain ?? null,
                        alternativesUsed: attempt.alternatives,
                        timeoutMs: attempt.timeoutMs,
                      });
                      return;
                    }

                    case 'done': {
                      sawDone = true;
                      flushRouteBufferNow(seq);
                      if (seq !== requestSeqRef.current) return;
                      setProgress({ done: event.done, total: event.total });
                      const doneWarnings = event.warnings ?? [];
                      if (doneWarnings.length) {
                        setWarnings((prev) => dedupeWarnings([...prev, ...doneWarnings]));
                      }
                      applyParetoPayload(
                        { routes: event.routes ?? [], warnings: doneWarnings },
                        'stream_done',
                        attempt,
                      );
                      if (event.candidate_diagnostics) {
                        appendComputeTrace({
                          level: 'info',
                          step: 'Stream diagnostics snapshot',
                          detail: `Graph and prefetch diagnostics captured for ${event.done}/${event.total} routes.`,
                          attempt: attempt.attempt,
                          endpoint: attempt.endpoint,
                          candidateDiagnostics: event.candidate_diagnostics,
                          alternativesUsed: attempt.alternatives,
                          timeoutMs: attempt.timeoutMs,
                        });
                      }
                      return;
                    }

                    default:
                      return;
                  }
                },
              });

              if (seq !== requestSeqRef.current || runController.signal.aborted) {
                return { ok: false, aborted: true };
              }
              flushRouteBufferNow(seq);
              if (streamFatalMessage !== null) {
                return {
                  ok: false,
                  aborted: false,
                  message: streamFatalMessage,
                  reasonCode: streamFatalReasonCode,
                };
              }
              if (!sawDone) {
                return {
                  ok: false,
                  aborted: false,
                  message: 'Streaming ended without a terminal done event.',
                  reasonCode: 'stream_incomplete',
                };
              }
              return { ok: true, aborted: false };
            } catch (streamError: unknown) {
              if (seq !== requestSeqRef.current || runController.signal.aborted) {
                return { ok: false, aborted: true };
              }
              const streamMessage = runtime.timeoutTriggered()
                ? `Stream attempt timed out after ${attempt.timeoutMs}ms`
                : streamError instanceof Error
                  ? streamError.message
                  : 'Unknown streaming error';
              const streamReasonCode = extractReasonCodeFromMessage(streamMessage);
              appendComputeTrace({
                level: 'error',
                step: 'Stream transport failed',
                detail: streamMessage,
                reasonCode: streamReasonCode,
                recoveries: inferComputeRecoveryHints(streamMessage),
                attempt: attempt.attempt,
                endpoint: attempt.endpoint,
                alternativesUsed: attempt.alternatives,
                timeoutMs: attempt.timeoutMs,
              });
              return {
                ok: false,
                aborted: false,
                message: streamMessage,
                reasonCode: streamReasonCode,
              };
            }
          }

          if (attempt.kind === 'pareto_json') {
            appendComputeTrace({
              level: 'info',
              step: 'Starting JSON Pareto request',
              detail: 'POST /api/pareto',
              attempt: attempt.attempt,
              endpoint: attempt.endpoint,
              alternativesUsed: attempt.alternatives,
              timeoutMs: attempt.timeoutMs,
            });
            try {
              const { data: payload, response: rawResponse } = await postJSONWithMeta<ParetoResponse>(
                '/api/pareto',
                buildParetoBody(attempt.alternatives),
                runtime.signal,
              );
              const requestId = rawResponse.headers.get('x-route-request-id');
              if (requestId) {
                attachAttemptRequestId(attempt, requestId);
              }
              if (seq !== requestSeqRef.current || runController.signal.aborted) {
                return { ok: false, aborted: true };
              }
              applyParetoPayload(payload, 'pareto_json', attempt);
              return { ok: true, aborted: false };
            } catch (jsonError: unknown) {
              if (seq !== requestSeqRef.current || runController.signal.aborted) {
                return { ok: false, aborted: true };
              }
              const jsonResponse =
                jsonError &&
                typeof jsonError === 'object' &&
                'response' in jsonError &&
                (jsonError as { response?: Response }).response instanceof Response
                  ? (jsonError as { response: Response }).response
                  : null;
              const jsonRequestId = jsonResponse?.headers.get('x-route-request-id');
              const jsonStatusCode = jsonResponse?.status;
              if (jsonRequestId) {
                attachAttemptRequestId(attempt, jsonRequestId);
              }
              const jsonMessage = runtime.timeoutTriggered()
                ? `JSON attempt timed out after ${attempt.timeoutMs}ms`
                : jsonError instanceof Error
                  ? jsonError.message
                  : 'Unknown JSON compute error';
              const jsonReasonCode = extractReasonCodeFromMessage(jsonMessage);
              appendComputeTrace({
                level: 'error',
                step: 'JSON Pareto request failed',
                detail: jsonMessage,
                reasonCode: jsonReasonCode,
                recoveries: inferComputeRecoveryHints(jsonMessage),
                attempt: attempt.attempt,
                endpoint: attempt.endpoint,
                alternativesUsed: attempt.alternatives,
                timeoutMs: attempt.timeoutMs,
              });
              return {
                ok: false,
                aborted: false,
                message: jsonMessage,
                reasonCode: jsonReasonCode,
                statusCode: jsonStatusCode,
              };
            }
          }

          appendComputeTrace({
            level: 'info',
            step: 'Starting single-route request',
            detail: 'POST /api/route',
            attempt: attempt.attempt,
            endpoint: attempt.endpoint,
            alternativesUsed: attempt.alternatives,
            timeoutMs: attempt.timeoutMs,
          });
          try {
            const { data: payload, response: rawResponse } = await postJSONWithMeta<RouteResponse>(
              '/api/route',
              buildRouteBody(attempt.alternatives),
              runtime.signal,
            );
            const requestId = rawResponse.headers.get('x-route-request-id');
            if (requestId) {
              attachAttemptRequestId(attempt, requestId);
            }
            if (seq !== requestSeqRef.current || runController.signal.aborted) {
              return { ok: false, aborted: true };
            }
            applyRoutePayload(payload, 'route_single', attempt);
            return { ok: true, aborted: false };
          } catch (routeError: unknown) {
            if (seq !== requestSeqRef.current || runController.signal.aborted) {
              return { ok: false, aborted: true };
            }
            const routeResponse =
              routeError &&
              typeof routeError === 'object' &&
              'response' in routeError &&
              (routeError as { response?: Response }).response instanceof Response
                ? (routeError as { response: Response }).response
                : null;
            const routeRequestId = routeResponse?.headers.get('x-route-request-id');
            const routeStatusCode = routeResponse?.status;
            if (routeRequestId) {
              attachAttemptRequestId(attempt, routeRequestId);
            }
            const routeMessage = runtime.timeoutTriggered()
              ? `Single-route attempt timed out after ${attempt.timeoutMs}ms`
              : routeError instanceof Error
                ? routeError.message
                : 'Unknown single-route compute error';
            const routeReasonCode = extractReasonCodeFromMessage(routeMessage);
            appendComputeTrace({
              level: 'error',
              step: 'Single-route request failed',
              detail: routeMessage,
              reasonCode: routeReasonCode,
              recoveries: inferComputeRecoveryHints(routeMessage),
              attempt: attempt.attempt,
              endpoint: attempt.endpoint,
              alternativesUsed: attempt.alternatives,
              timeoutMs: attempt.timeoutMs,
            });
            return {
              ok: false,
              aborted: false,
              message: routeMessage,
              reasonCode: routeReasonCode,
              statusCode: routeStatusCode,
            };
          }
        };

        const result = await runAttempt();

        if (!result.ok && !result.aborted && idx < attemptPlans.length - 1) {
          if (!runtime.signal.aborted) {
            try {
              runtime.controller.abort(new Error('fallback_transition'));
            } catch {
              runtime.controller.abort();
            }
          }
          appendComputeTrace({
            level: 'warn',
            step: 'Attempt aborted before fallback',
            detail: `Cancelled attempt ${attempt.attempt} (${attempt.endpoint}) before starting fallback.`,
            attempt: attempt.attempt,
            endpoint: attempt.endpoint,
            alternativesUsed: attempt.alternatives,
            timeoutMs: attempt.timeoutMs,
            abortReason: 'fallback_transition',
          });
        }

        const attemptRequestId = liveCallRequestByAttempt.get(attempt.attempt) ?? null;
        if (attemptRequestId) {
          await syncAttemptLiveCalls(attempt, attemptRequestId, {
            step: 'Live API trace final sync',
            forceTrace: true,
            traceErrors: false,
          });
        }
        runtime.cleanup();

        if (result.ok) {
          succeeded = true;
          break;
        }
        if (result.aborted) {
          return;
        }

        finalFailure = result;
        const transportReasonCodes = new Set([
          'backend_headers_timeout',
          'backend_body_timeout',
          'backend_connection_reset',
          'backend_unreachable',
          'stream_incomplete',
        ]);
        const isStrictBusiness4xx =
          typeof result.statusCode === 'number' &&
          result.statusCode >= 400 &&
          result.statusCode < 500 &&
          !(
            typeof result.reasonCode === 'string' &&
            transportReasonCodes.has(result.reasonCode)
          );
        if (isStrictBusiness4xx) {
          const strictDetail = `Strict business failure returned HTTP ${result.statusCode}${
            result.reasonCode ? ` (reason_code=${result.reasonCode})` : ''
          }; skipping additional fallback attempts.`;
          appendComputeTrace({
            level: 'warn',
            step: 'Fallback halted (strict business failure)',
            detail: strictDetail,
            reasonCode: result.reasonCode,
            recoveries: inferComputeRecoveryHints(result.reasonCode ?? strictDetail),
            attempt: attempt.attempt,
            endpoint: attempt.endpoint,
            alternativesUsed: attempt.alternatives,
            timeoutMs: attempt.timeoutMs,
          });
          break;
        }
        const stopFallbackReasonCodes = new Set([
          'routing_graph_no_path',
          'routing_graph_unavailable',
          'routing_graph_fragmented',
          'routing_graph_disconnected_od',
          'routing_graph_coverage_gap',
          'routing_graph_precheck_timeout',
          'routing_graph_warming_up',
          'routing_graph_warmup_failed',
          'live_source_refresh_failed',
          'scenario_profile_unavailable',
          'scenario_profile_invalid',
        ]);
        if (result.reasonCode && stopFallbackReasonCodes.has(result.reasonCode)) {
          const haltDetail =
            result.reasonCode === 'routing_graph_warming_up'
              ? 'Routing graph warmup is still in progress; skipping additional fallback attempts.'
              : result.reasonCode === 'routing_graph_warmup_failed'
                ? 'Routing graph warmup failed; skipping additional fallback attempts.'
                : result.reasonCode === 'routing_graph_fragmented'
                  ? 'Routing graph is fragmented under strict caps; skipping additional fallback attempts.'
                  : result.reasonCode === 'routing_graph_disconnected_od'
                    ? 'Origin and destination are disconnected in the loaded graph; skipping additional fallback attempts.'
                    : result.reasonCode === 'routing_graph_coverage_gap'
                      ? 'Graph coverage gap detected near origin/destination; skipping additional fallback attempts.'
                      : result.reasonCode === 'routing_graph_precheck_timeout'
                        ? 'Graph feasibility precheck timed out; skipping additional fallback attempts.'
                      : result.reasonCode === 'routing_graph_no_path'
                        ? 'Routing graph search exhausted without a feasible path; skipping additional fallback attempts.'
                      : result.reasonCode === 'routing_graph_unavailable'
                        ? 'Routing graph assets are unavailable; skipping additional fallback attempts.'
                      : result.reasonCode === 'live_source_refresh_failed'
                        ? 'Strict live-source refresh gate failed; skipping additional fallback attempts.'
                : 'Strict live scenario data is unavailable/invalid; skipping additional fallback attempts.';
          appendComputeTrace({
            level: 'warn',
            step: 'Fallback halted',
            detail: haltDetail,
            reasonCode: result.reasonCode,
            recoveries: inferComputeRecoveryHints(result.reasonCode),
            attempt: attempt.attempt,
            endpoint: attempt.endpoint,
            alternativesUsed: attempt.alternatives,
            timeoutMs: attempt.timeoutMs,
          });
          break;
        }
        if (idx < attemptPlans.length - 1) {
          const nextAttempt = attemptPlans[idx + 1];
          appendComputeTrace({
            level: 'warn',
            step:
              nextAttempt.kind === 'pareto_json'
                ? 'Retrying with JSON fallback'
                : nextAttempt.kind === 'route_single'
                  ? 'Retrying with single-route fallback'
                  : 'Retrying stream attempt',
            detail: transitionDetail(attempt, nextAttempt),
            attempt: nextAttempt.attempt,
            endpoint: nextAttempt.endpoint,
            alternativesUsed: nextAttempt.alternatives,
            timeoutMs: nextAttempt.timeoutMs,
          });
        }
      }

      if (!succeeded) {
        if (finalFailure?.message) {
          const tagged = finalFailure.reasonCode
            ? `${finalFailure.message} (${finalFailure.reasonCode})`
            : finalFailure.message;
          throw new Error(tagged);
        }
        throw new Error('Route computation failed after all fallback attempts.');
      }
    } catch (e: unknown) {
      if (seq !== requestSeqRef.current) return;
      if (runController.signal.aborted) return;
      const message = e instanceof Error ? e.message : 'Unknown error';
      const reasonCode = extractReasonCodeFromMessage(message);
      const recoveries = inferComputeRecoveryHints(message);
      appendComputeTrace({
        level: 'error',
        step: 'Compute failed',
        detail: message,
        reasonCode,
        recoveries,
        attempt: activeAttemptNumber ?? undefined,
        endpoint: activeAttemptEndpoint ?? undefined,
        alternativesUsed: activeAttemptAlternatives || undefined,
        timeoutMs: activeAttemptTimeoutMs || undefined,
      });
      setError(message);
    } finally {
      stopAllLiveCallPolling();
      if (heartbeatTimer !== null) {
        window.clearInterval(heartbeatTimer);
      }
      const finalAttempt =
        typeof activeAttemptNumber === 'number' && activeAttemptNumber > 0
          ? attemptPlans.find((item) => item.attempt === activeAttemptNumber) ?? null
          : null;
      if (finalAttempt) {
        const finalRequestId = liveCallRequestByAttempt.get(finalAttempt.attempt) ?? null;
        if (finalRequestId) {
          await syncAttemptLiveCalls(finalAttempt, finalRequestId, {
            step: 'Live API trace terminal sync',
            forceTrace: true,
            traceErrors: false,
          });
        }
      }
      if (seq === requestSeqRef.current) {
        abortRef.current = null;
        activeAttemptAbortRef.current = null;
        setLoading(false);
        appendComputeTrace({
          level: 'info',
          step: 'Compute finished',
          detail: `Request lifecycle ended; aborted=${runController.signal.aborted}`,
          attempt: activeAttemptNumber ?? undefined,
          endpoint: activeAttemptEndpoint ?? undefined,
          alternativesUsed: activeAttemptAlternatives || undefined,
          timeoutMs: activeAttemptTimeoutMs || undefined,
        });
      }
    }
  }

  async function compareScenarios() {
    if (!origin || !destination) {
      setScenarioCompareError('Set Start and End before comparing scenarios.');
      return;
    }
    markTutorialAction('compare.run_click');

    let requestBody: ScenarioCompareRequest;
    try {
      setAdvancedError(null);
      requestBody = buildScenarioCompareRequest(origin, destination);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Invalid advanced parameter values.';
      setAdvancedError(msg);
      setScenarioCompareError(msg);
      return;
    }

    setScenarioCompareLoading(true);
    setScenarioCompareError(null);
    try {
      const payload = await postJSON<ScenarioCompareResponse>(
        '/api/scenario/compare',
        requestBody,
        undefined,
      );
      setScenarioCompare(payload);
      setRunInspectorRunId(payload.run_id);
      markTutorialAction('compare.run_done');
      setLiveMessage(t('live_compare_complete'));
    } catch (e: unknown) {
      setScenarioCompareError(e instanceof Error ? e.message : 'Failed to compare scenarios');
      setLiveMessage(t('live_compare_failed'));
    } finally {
      setScenarioCompareLoading(false);
    }
  }

  function toDatetimeLocalValue(isoUtc: string | undefined): string {
    if (!isoUtc) return '';
    const dt = new Date(isoUtc);
    if (Number.isNaN(dt.getTime())) return '';
    return dt.toISOString().slice(0, 16);
  }

  function applyScenarioRequestToState(req: ScenarioCompareRequest) {
    setOrigin(req.origin);
    setDestination(req.destination);
    const incomingStops =
      Array.isArray(req.waypoints) && req.waypoints.length > 0
        ? req.waypoints.map((waypoint, idx) => ({
            lat: waypoint.lat,
            lon: waypoint.lon,
            label: waypoint.label?.trim() || `Stop #${idx + 1}`,
          }))
        : [];
    if (Array.isArray(req.waypoints) && req.waypoints.length > 0) {
      const firstWaypoint = req.waypoints[0];
      setManagedStop({
        id: 'stop-1',
        lat: firstWaypoint.lat,
        lon: firstWaypoint.lon,
        label: firstWaypoint.label?.trim() || 'Stop #1',
      });
    } else {
      setManagedStop(null);
    }
    dutySyncSourceRef.current = 'pins';
    setDutyStopsText(serializePinsToDutyText(req.origin, incomingStops, req.destination));
    setDutySyncError(null);
    if (req.vehicle_type) {
      setVehicleType(req.vehicle_type);
    }
    if (req.scenario_mode) {
      setScenarioMode(req.scenario_mode);
    }
    if (req.weights) {
      setWeights({
        time: Number(req.weights.time ?? 60),
        money: Number(req.weights.money ?? 20),
        co2: Number(req.weights.co2 ?? 20),
      });
    }

    const cost = req.cost_toggles;
    const stochastic = req.stochastic;
    const emissionsContext = req.emissions_context;
    const weather = req.weather;
    const incident = req.incident_simulation;
    setAdvancedParams({
      maxAlternatives: String(req.max_alternatives ?? 24),
      paretoMethod: req.pareto_method ?? 'dominance',
      epsilonDurationS: req.epsilon?.duration_s !== undefined ? String(req.epsilon.duration_s) : '',
      epsilonMonetaryCost:
        req.epsilon?.monetary_cost !== undefined ? String(req.epsilon.monetary_cost) : '',
      epsilonEmissionsKg: req.epsilon?.emissions_kg !== undefined ? String(req.epsilon.emissions_kg) : '',
      departureTimeUtcLocal: toDatetimeLocalValue(req.departure_time_utc),
      useTolls: cost?.use_tolls ?? true,
      fuelPriceMultiplier: String(cost?.fuel_price_multiplier ?? 1.0),
      carbonPricePerKg: String(cost?.carbon_price_per_kg ?? 0.0),
      tollCostPerKm: String(cost?.toll_cost_per_km ?? 0.0),
      terrainProfile: req.terrain_profile ?? 'flat',
      optimizationMode: req.optimization_mode ?? 'expected_value',
      riskAversion: String(req.risk_aversion ?? 1.0),
      stochasticEnabled: Boolean(stochastic?.enabled),
      stochasticSeed:
        stochastic?.seed !== undefined && stochastic?.seed !== null ? String(stochastic.seed) : '',
      stochasticSigma: String(stochastic?.sigma ?? 0.08),
      stochasticSamples: String(stochastic?.samples ?? 25),
      fuelType: (emissionsContext?.fuel_type as FuelType | undefined) ?? 'diesel',
      euroClass: (emissionsContext?.euro_class as EuroClass | undefined) ?? 'euro6',
      ambientTempC: String(emissionsContext?.ambient_temp_c ?? 15),
      weatherEnabled: Boolean(weather?.enabled),
      weatherProfile: (weather?.profile as WeatherProfile | undefined) ?? 'clear',
      weatherIntensity: String(weather?.intensity ?? 1.0),
      weatherIncidentUplift: weather?.apply_incident_uplift ?? true,
      incidentSimulationEnabled: Boolean(incident?.enabled),
      incidentSeed:
        incident?.seed !== undefined && incident?.seed !== null ? String(incident.seed) : '',
      incidentDwellRatePer100km: String(incident?.dwell_rate_per_100km ?? 0.8),
      incidentAccidentRatePer100km: String(incident?.accident_rate_per_100km ?? 0.25),
      incidentClosureRatePer100km: String(incident?.closure_rate_per_100km ?? 0.05),
      incidentDwellDelayS: String(incident?.dwell_delay_s ?? 120),
      incidentAccidentDelayS: String(incident?.accident_delay_s ?? 480),
      incidentClosureDelayS: String(incident?.closure_delay_s ?? 900),
      incidentMaxEventsPerRoute: String(incident?.max_events_per_route ?? 12),
    });
  }

  async function loadVehicles(signal?: AbortSignal) {
    const payload = await getJSON<VehicleListResponse>('/api/vehicles', signal);
    setVehicles(Array.isArray(payload.vehicles) ? payload.vehicles : []);
  }

  async function loadExperiments(
    params: {
      q?: string;
      vehicleType?: string;
      scenarioMode?: '' | ScenarioMode;
      sort?: ExperimentCatalogSort;
    } = {},
  ) {
    setExperimentsLoading(true);
    setExperimentsError(null);
    try {
      const q = (params.q ?? expCatalogQuery).trim();
      const vehicleType = (params.vehicleType ?? expCatalogVehicleType).trim();
      const scenarioMode = (params.scenarioMode ?? expCatalogScenarioMode).trim();
      const sort = params.sort ?? expCatalogSort;
      const qs = new URLSearchParams();
      if (q) qs.set('q', q);
      if (vehicleType) qs.set('vehicle_type', vehicleType);
      if (scenarioMode) qs.set('scenario_mode', scenarioMode);
      if (sort) qs.set('sort', sort);
      const path = qs.toString() ? `/api/experiments?${qs.toString()}` : '/api/experiments';
      const parsed = await getJSON<ExperimentListResponse>(path);
      setExperiments(Array.isArray(parsed.experiments) ? parsed.experiments : []);
    } catch (e: unknown) {
      setExperimentsError(e instanceof Error ? e.message : 'Failed to load experiments');
    } finally {
      setExperimentsLoading(false);
    }
  }

  async function saveCurrentExperiment(name: string, description: string) {
    if (!origin || !destination) {
      setExperimentsError('Set Start and End before saving an experiment.');
      return;
    }
    try {
      setExperimentsError(null);
      markTutorialAction('exp.save_click');
      const request = buildScenarioCompareRequest(origin, destination);
      await postJSON<ExperimentBundle>(
        '/api/experiments',
        {
          name,
          description: description || null,
          request,
        },
        undefined,
      );
      await loadExperiments();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to save experiment';
      setAdvancedError(msg);
      setExperimentsError(msg);
    }
  }

  async function deleteExperimentById(experimentId: string) {
    try {
      setExperimentsError(null);
      markTutorialAction('exp.delete_click');
      await deleteJSON<{ experiment_id: string; deleted: boolean }>(`/api/experiments/${experimentId}`);
      await loadExperiments();
    } catch (e: unknown) {
      setExperimentsError(e instanceof Error ? e.message : 'Failed to delete experiment');
    }
  }

  async function openExperimentById(experimentId: string) {
    try {
      setExperimentsError(null);
      const payload = await getJSON<ExperimentBundle>(`/api/experiments/${experimentId}`);
      applyScenarioRequestToState(payload.request);
    } catch (e: unknown) {
      setExperimentsError(e instanceof Error ? e.message : 'Failed to open experiment');
    }
  }

  async function updateExperimentMetadata(
    bundle: ExperimentBundle,
    next: { name: string; description?: string | null },
  ) {
    try {
      setExperimentsError(null);
      await putJSON<ExperimentBundle>(`/api/experiments/${bundle.id}`, {
        name: next.name,
        description: next.description ?? null,
        request: bundle.request,
      });
      await loadExperiments();
    } catch (e: unknown) {
      setExperimentsError(e instanceof Error ? e.message : 'Failed to update experiment');
    }
  }

  async function replayExperimentById(experimentId: string) {
    setScenarioCompareLoading(true);
    setScenarioCompareError(null);
    try {
      markTutorialAction('exp.replay_click');
      const payload = await postJSON<ScenarioCompareResponse>(
        `/api/experiments/${experimentId}/compare`,
        {
          overrides: {},
        },
        undefined,
      );
      setScenarioCompare(payload);
      setRunInspectorRunId(payload.run_id);
    } catch (e: unknown) {
      setScenarioCompareError(e instanceof Error ? e.message : 'Failed to replay experiment');
    } finally {
      setScenarioCompareLoading(false);
    }
  }

  async function refreshOpsDiagnostics() {
    setOpsLoading(true);
    setOpsError(null);
    try {
      const [health, healthReady, metrics, cacheStats] = await Promise.all([
        getJSON<HealthResponse>('/api/health'),
        getJSON<HealthReadyResponse>('/api/health/ready'),
        getJSON<MetricsResponse>('/api/metrics'),
        getJSON<CacheStatsResponse>('/api/cache/stats'),
      ]);
      setOpsHealth(health);
      setOpsHealthReady(healthReady);
      setOpsMetrics(metrics);
      setOpsCacheStats(cacheStats);
    } catch (e: unknown) {
      setOpsError(e instanceof Error ? e.message : 'Failed to load ops diagnostics.');
    } finally {
      setOpsLoading(false);
    }
  }

  async function refreshRouteReadiness() {
    try {
      const readiness = await getJSON<HealthReadyResponse>('/api/health/ready');
      setOpsHealthReady(readiness);
    } catch (e: unknown) {
      const cause = e instanceof Error ? e.message : 'Unable to reach backend /health/ready endpoint.';
      setOpsHealthReady({
        status: 'not_ready',
        strict_route_ready: false,
        recommended_action: 'retry',
        route_graph: {
          ok: false,
          state: 'failed',
          phase: 'backend_unreachable',
          last_error: cause,
        },
      });
    }
  }

  async function clearOpsCache() {
    const confirmed = window.confirm('Clear backend route cache now?');
    if (!confirmed) return;
    setOpsClearing(true);
    setOpsError(null);
    try {
      await deleteJSON<CacheClearResponse>('/api/cache');
      await refreshOpsDiagnostics();
    } catch (e: unknown) {
      setOpsError(e instanceof Error ? e.message : 'Failed to clear backend cache.');
    } finally {
      setOpsClearing(false);
    }
  }

  async function refreshCustomVehicles() {
    setCustomVehiclesLoading(true);
    setCustomVehicleError(null);
    try {
      const payload = await getJSON<CustomVehicleListResponse>('/api/vehicles/custom');
      setCustomVehicles(payload.vehicles ?? []);
    } catch (e: unknown) {
      setCustomVehicleError(e instanceof Error ? e.message : 'Failed to load custom vehicles.');
    } finally {
      setCustomVehiclesLoading(false);
    }
  }

  async function createCustomVehicle(vehicle: VehicleProfile) {
    setCustomVehicleSaving(true);
    setCustomVehicleError(null);
    try {
      await postJSON('/api/vehicles/custom', vehicle);
      await refreshCustomVehicles();
      await loadVehicles();
    } catch (e: unknown) {
      setCustomVehicleError(e instanceof Error ? e.message : 'Failed to create custom vehicle.');
    } finally {
      setCustomVehicleSaving(false);
    }
  }

  async function updateCustomVehicle(vehicleId: string, vehicle: VehicleProfile) {
    setCustomVehicleSaving(true);
    setCustomVehicleError(null);
    try {
      await putJSON(`/api/vehicles/custom/${vehicleId}`, vehicle);
      await refreshCustomVehicles();
      await loadVehicles();
    } catch (e: unknown) {
      setCustomVehicleError(e instanceof Error ? e.message : 'Failed to update custom vehicle.');
    } finally {
      setCustomVehicleSaving(false);
    }
  }

  async function removeCustomVehicle(vehicleId: string) {
    setCustomVehicleSaving(true);
    setCustomVehicleError(null);
    try {
      await deleteJSON(`/api/vehicles/custom/${vehicleId}`);
      await refreshCustomVehicles();
      await loadVehicles();
    } catch (e: unknown) {
      setCustomVehicleError(e instanceof Error ? e.message : 'Failed to delete custom vehicle.');
    } finally {
      setCustomVehicleSaving(false);
    }
  }

  async function runBatchParetoRequest(req: BatchParetoRequest) {
    setBatchLoading(true);
    setBatchError(null);
    try {
      const payload = await postJSON<BatchParetoResponse>('/api/batch/pareto', req);
      setBatchResult(payload);
      setRunInspectorRunId(payload.run_id);
    } catch (e: unknown) {
      setBatchError(e instanceof Error ? e.message : 'Failed to run batch pareto.');
    } finally {
      setBatchLoading(false);
    }
  }

  async function runBatchCsvRequest(req: BatchCSVImportRequest) {
    setBatchLoading(true);
    setBatchError(null);
    try {
      const payload = await postJSON<BatchParetoResponse>('/api/batch/import/csv', req);
      setBatchResult(payload);
      setRunInspectorRunId(payload.run_id);
    } catch (e: unknown) {
      setBatchError(e instanceof Error ? e.message : 'Failed to run batch CSV import.');
    } finally {
      setBatchLoading(false);
    }
  }

  async function runSignatureVerification(req: SignatureVerificationRequest) {
    setSignatureLoading(true);
    setSignatureError(null);
    try {
      const payload = await postJSON<SignatureVerificationResponse>('/api/verify/signature', req);
      setSignatureResult(payload);
    } catch (e: unknown) {
      setSignatureError(e instanceof Error ? e.message : 'Failed to verify signature.');
    } finally {
      setSignatureLoading(false);
    }
  }

  async function loadRunCoreDocs() {
    if (!runInspectorRunId.trim()) {
      setRunInspectorError('Enter a run_id first.');
      return;
    }
    setRunInspectorLoading(true);
    setRunInspectorError(null);
    try {
      const runId = runInspectorRunId.trim();
      const [manifest, scenarioManifest, provenance, signature, scenarioSignature] = await Promise.all([
        getJSON<unknown>(`/api/runs/${runId}/manifest`),
        getJSON<unknown>(`/api/runs/${runId}/scenario-manifest`),
        getJSON<unknown>(`/api/runs/${runId}/provenance`),
        getJSON<unknown>(`/api/runs/${runId}/signature`),
        getJSON<unknown>(`/api/runs/${runId}/scenario-signature`),
      ]);
      setRunManifest(manifest);
      setRunScenarioManifest(scenarioManifest);
      setRunProvenance(provenance);
      setRunSignature(signature);
      setRunScenarioSignature(scenarioSignature);
    } catch (e: unknown) {
      setRunInspectorError(e instanceof Error ? e.message : 'Failed to load run core docs.');
    } finally {
      setRunInspectorLoading(false);
    }
  }

  async function loadRunArtifactsList() {
    if (!runInspectorRunId.trim()) {
      setRunInspectorError('Enter a run_id first.');
      return;
    }
    setRunInspectorLoading(true);
    setRunInspectorError(null);
    try {
      const runId = runInspectorRunId.trim();
      const payload = await getJSON<RunArtifactsListResponse>(`/api/runs/${runId}/artifacts`);
      setRunArtifacts(payload);
    } catch (e: unknown) {
      setRunInspectorError(e instanceof Error ? e.message : 'Failed to load run artifacts.');
    } finally {
      setRunInspectorLoading(false);
    }
  }

  async function previewRunArtifact(name: string) {
    if (!runInspectorRunId.trim()) {
      setRunInspectorError('Enter a run_id first.');
      return;
    }
    setRunInspectorLoading(true);
    setRunInspectorError(null);
    try {
      const runId = runInspectorRunId.trim();
      const text = await getText(`/api/runs/${runId}/artifacts/${name}`);
      setRunArtifactPreviewName(name);
      setRunArtifactPreviewText(text);
    } catch (e: unknown) {
      setRunInspectorError(e instanceof Error ? e.message : `Failed to preview artifact ${name}.`);
    } finally {
      setRunInspectorLoading(false);
    }
  }

  async function downloadFromApi(path: string, fallbackName: string) {
    const resp = await fetch(path, {
      method: 'GET',
      cache: 'no-store',
    });
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const blob = await resp.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    const disposition = resp.headers.get('content-disposition');
    const fileNameMatch = disposition?.match(/filename=\"?([^\";]+)\"?/i);
    link.download = fileNameMatch?.[1] ?? fallbackName;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  }

  async function downloadRunCore(kind: 'manifest' | 'scenarioManifest' | 'provenance' | 'signature' | 'scenarioSignature') {
    if (!runInspectorRunId.trim()) {
      setRunInspectorError('Enter a run_id first.');
      return;
    }
    const runId = runInspectorRunId.trim();
    const endpointByKind: Record<typeof kind, { path: string; file: string }> = {
      manifest: { path: `/api/runs/${runId}/manifest`, file: `${runId}-manifest.json` },
      scenarioManifest: {
        path: `/api/runs/${runId}/scenario-manifest`,
        file: `${runId}-scenario-manifest.json`,
      },
      provenance: { path: `/api/runs/${runId}/provenance`, file: `${runId}-provenance.json` },
      signature: { path: `/api/runs/${runId}/signature`, file: `${runId}-signature.json` },
      scenarioSignature: {
        path: `/api/runs/${runId}/scenario-signature`,
        file: `${runId}-scenario-signature.json`,
      },
    };
    const target = endpointByKind[kind];
    try {
      await downloadFromApi(target.path, target.file);
    } catch (e: unknown) {
      setRunInspectorError(e instanceof Error ? e.message : `Failed to download ${kind}.`);
    }
  }

  async function downloadRunArtifact(name: string) {
    if (!runInspectorRunId.trim()) {
      setRunInspectorError('Enter a run_id first.');
      return;
    }
    const runId = runInspectorRunId.trim();
    try {
      await downloadFromApi(`/api/runs/${runId}/artifacts/${name}`, name);
    } catch (e: unknown) {
      setRunInspectorError(e instanceof Error ? e.message : `Failed to download artifact ${name}.`);
    }
  }

  async function inspectScenarioManifestForRun(runId: string) {
    setRunInspectorRunId(runId);
    setRunInspectorLoading(true);
    setRunInspectorError(null);
    try {
      const payload = await getJSON<unknown>(`/api/runs/${runId}/scenario-manifest`);
      setRunScenarioManifest(payload);
    } catch (e: unknown) {
      setRunInspectorError(e instanceof Error ? e.message : 'Failed to inspect scenario manifest.');
    } finally {
      setRunInspectorLoading(false);
    }
  }

  async function inspectScenarioSignatureForRun(runId: string) {
    setRunInspectorRunId(runId);
    setRunInspectorLoading(true);
    setRunInspectorError(null);
    try {
      const payload = await getJSON<unknown>(`/api/runs/${runId}/scenario-signature`);
      setRunScenarioSignature(payload);
    } catch (e: unknown) {
      setRunInspectorError(e instanceof Error ? e.message : 'Failed to inspect scenario signature.');
    } finally {
      setRunInspectorLoading(false);
    }
  }

  function openRunInspectorForRun(runId: string) {
    setRunInspectorRunId(runId);
  }

  function parseDutyStops(text: string): DutyChainRequest['stops'] {
    let parsed: ParsedPinSync;
    try {
      parsed = parseDutyTextToPins(text);
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Invalid duty chain input.';
      throw new Error(msg);
    }
    if (!parsed.ok) {
      throw new Error(parsed.error);
    }
    if (!parsed.origin || !parsed.destination) {
      throw new Error('Duty chain requires both Start and End rows.');
    }
    if (parsed.stops.length + 2 > 50) {
      throw new Error('Duty chain supports at most 50 stops including Start and End.');
    }

    const out: DutyChainRequest['stops'] = [
      {
        lat: parsed.origin.lat,
        lon: parsed.origin.lon,
        label: 'Start',
      },
    ];
    for (let idx = 0; idx < parsed.stops.length; idx += 1) {
      const stop = parsed.stops[idx];
      out.push({
        lat: stop.lat,
        lon: stop.lon,
        label: stop.label || `Stop #${idx + 1}`,
      });
    }
    out.push({
      lat: parsed.destination.lat,
      lon: parsed.destination.lon,
      label: 'End',
    });
    return out;
  }

  function handleDutyStopsTextChange(nextValue: string) {
    dutySyncSourceRef.current = 'text';
    setDutyStopsText(nextValue);
    if (dutySyncError) {
      setDutySyncError(null);
    }
    markTutorialAction('duty.stops_input');
  }

  async function runDutyChain() {
    let advancedPatch: RoutingAdvancedPatch;
    let maxAlternatives = 5;
    let stops: DutyChainRequest['stops'];
    try {
      setAdvancedError(null);
      setDutyChainError(null);
      const parsed = buildAdvancedRequestPatch();
      advancedPatch = parsed.advancedPatch;
      maxAlternatives = parsed.maxAlternatives;
      stops = parseDutyStops(dutyStopsText);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Invalid duty chain request.';
      setAdvancedError(msg);
      setDutyChainError(msg);
      return;
    }

    const req = buildDutyChainRequest({
      stops,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: maxAlternatives,
      weights: {
        time: weights.time,
        money: weights.money,
        co2: weights.co2,
      },
      advanced: advancedPatch,
    });

    setDutyChainLoading(true);
    try {
      markTutorialAction('duty.run_click');
      const payload = await postJSON<DutyChainResponse>(
        '/api/duty/chain',
        req,
        undefined,
      );
      setDutyChainData(payload);
      markTutorialAction('duty.run_done');
      setLiveMessage(t('live_duty_complete'));
    } catch (e: unknown) {
      setDutyChainError(e instanceof Error ? e.message : 'Failed to run duty chain.');
      setLiveMessage(t('live_duty_failed'));
    } finally {
      setDutyChainLoading(false);
    }
  }

  async function refreshOracleDashboard(opts: { silent?: boolean } = {}) {
    if (!opts.silent) {
      setOracleDashboardLoading(true);
      markTutorialAction('oracle.refresh_click');
    }
    setOracleError(null);
    try {
      const resp = await fetch('/api/oracle/quality/dashboard', {
        cache: 'no-store',
      });
      const text = await resp.text();
      if (!resp.ok) {
        throw new Error(text.trim() || 'Failed to fetch oracle quality dashboard.');
      }
      const parsed = JSON.parse(text) as OracleQualityDashboardResponse;
      setOracleDashboard(parsed);
    } catch (e: unknown) {
      setOracleError(e instanceof Error ? e.message : 'Failed to fetch oracle quality dashboard.');
    } finally {
      if (!opts.silent) {
        setOracleDashboardLoading(false);
      }
    }
  }

  async function ingestOracleCheck(payload: OracleFeedCheckInput) {
    setOracleIngestLoading(true);
    setOracleError(null);
    try {
      markTutorialAction('oracle.record_check_click');
      const record = await postJSON<OracleFeedCheckRecord>(
        '/api/oracle/quality/check',
        payload,
        undefined,
      );
      setOracleLatestCheck(record);
      markTutorialAction('oracle.record_check_done');
      await refreshOracleDashboard({ silent: true });
    } catch (e: unknown) {
      setOracleError(e instanceof Error ? e.message : 'Failed to record oracle quality check.');
    } finally {
      setOracleIngestLoading(false);
    }
  }

  async function optimizeDepartures() {
    if (!origin || !destination) {
      setDepOptimizeError('Set Start and End before optimizing departures.');
      return;
    }

    let advancedPatch: RoutingAdvancedPatch;
    let maxAlternatives = 5;
    try {
      setAdvancedError(null);
      const parsed = buildAdvancedRequestPatch();
      advancedPatch = parsed.advancedPatch;
      maxAlternatives = parsed.maxAlternatives;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Invalid advanced parameter values.';
      setAdvancedError(msg);
      setDepOptimizeError(msg);
      return;
    }

    const startDate = new Date(depWindowStartLocal);
    const endDate = new Date(depWindowEndLocal);
    if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
      setDepOptimizeError('Departure window must include valid start and end times.');
      return;
    }
    if (endDate <= startDate) {
      setDepOptimizeError('Departure window end must be later than start.');
      return;
    }

    let timeWindow: TimeWindowConstraints | undefined;
    if (depEarliestArrivalLocal.trim() || depLatestArrivalLocal.trim()) {
      const earliestDate = depEarliestArrivalLocal.trim()
        ? new Date(depEarliestArrivalLocal)
        : undefined;
      const latestDate = depLatestArrivalLocal.trim() ? new Date(depLatestArrivalLocal) : undefined;
      if (earliestDate && Number.isNaN(earliestDate.getTime())) {
        setDepOptimizeError('Earliest arrival must be a valid date/time.');
        return;
      }
      if (latestDate && Number.isNaN(latestDate.getTime())) {
        setDepOptimizeError('Latest arrival must be a valid date/time.');
        return;
      }
      if (earliestDate && latestDate && latestDate < earliestDate) {
        setDepOptimizeError('Latest arrival must be later than or equal to earliest arrival.');
        return;
      }
      timeWindow = {
        ...(earliestDate ? { earliest_arrival_utc: earliestDate.toISOString() } : {}),
        ...(latestDate ? { latest_arrival_utc: latestDate.toISOString() } : {}),
      };
    }

    const req = buildDepartureOptimizeRequest({
      origin,
      destination,
      waypoints: requestWaypoints,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: maxAlternatives,
      weights: {
        time: weights.time,
        money: weights.money,
        co2: weights.co2,
      },
      window_start_utc: startDate.toISOString(),
      window_end_utc: endDate.toISOString(),
      step_minutes: depStepMinutes,
      ...(timeWindow ? { time_window: timeWindow } : {}),
      advanced: advancedPatch,
    });

    setDepOptimizeLoading(true);
    setDepOptimizeError(null);
    try {
      markTutorialAction('dep.optimize_click');
      const payload = await postJSON<DepartureOptimizeResponse>(
        '/api/departure/optimize',
        req,
        undefined,
      );
      setDepOptimizeData(payload);
      markTutorialAction('dep.optimize_done');
      setLiveMessage(t('live_departure_complete'));
    } catch (e: unknown) {
      setDepOptimizeError(e instanceof Error ? e.message : 'Failed to optimize departures');
      setLiveMessage(t('live_departure_failed'));
    } finally {
      setDepOptimizeLoading(false);
    }
  }

  function applyOptimizedDeparture(isoUtc: string) {
    setAdvancedParams((prev) => ({
      ...prev,
      departureTimeUtcLocal: toDatetimeLocalValue(isoUtc),
    }));
    markTutorialAction('dep.apply_departure');
  }

  const m = selectedRoute?.metrics ?? null;
  const terrainSummary = selectedRoute?.terrain_summary ?? null;
  const terrainBadgeLabel =
    terrainSummary?.source === 'dem_real'
      ? 'DEM Real'
      : terrainSummary
        ? 'Terrain Missing'
        : null;
  const terrainBadgeClass =
    terrainSummary?.source === 'dem_real'
      ? 'terrainBadge isDemReal'
      : terrainSummary
        ? 'terrainBadge isMissing'
        : 'terrainBadge';
  const terrainTooltip =
    terrainSummary
      ? `Source: ${terrainSummary.source} | Coverage: ${(terrainSummary.coverage_ratio * 100).toFixed(1)}% | Ascent: ${terrainSummary.ascent_m.toFixed(0)}m | Descent: ${terrainSummary.descent_m.toFixed(0)}m | Spacing: ${terrainSummary.sample_spacing_m.toFixed(0)}m | Confidence: ${(terrainSummary.confidence * 100).toFixed(0)}% | Version: ${terrainSummary.version}`
      : null;

  const vehicleOptions = vehicles.length
    ? vehicles.map((v) => ({
        value: v.id,
        label: v.label,
        description: vehicleDescriptionFromId(v.id),
      }))
    : [
        { value: 'van', label: 'Van', description: vehicleDescriptionFromId('van') },
        {
          value: 'rigid_hgv',
          label: 'Rigid HGV',
          description: vehicleDescriptionFromId('rigid_hgv'),
        },
        {
          value: 'artic_hgv',
          label: 'Articulated HGV',
          description: vehicleDescriptionFromId('artic_hgv'),
        },
      ];

  const scenarioOptions: SelectOption<ScenarioMode | ''>[] = [
    {
      value: 'no_sharing',
      label: 'No sharing',
      description: 'Baseline assumptions with no sharing policy uplift.',
    },
    {
      value: 'partial_sharing',
      label: 'Partial sharing',
      description: 'Moderate coordination with moderate delay effects.',
    },
    {
      value: 'full_sharing',
      label: 'Full sharing',
      description: 'Maximum coordination with strongest sharing assumptions.',
    },
  ];
  const vehicleSelectionPending =
    tutorialRunning &&
    tutorialStep?.id === 'setup_vehicle' &&
    !tutorialActionSet.has('setup.vehicle_option:rigid_hgv');
  const scenarioSelectionPending =
    tutorialRunning &&
    tutorialStep?.id === 'setup_scenario' &&
    !tutorialActionSet.has('setup.scenario_option:no_sharing');
  const vehicleSelectValue = vehicleSelectionPending ? '' : vehicleType;
  const scenarioSelectValue: ScenarioMode | '' = scenarioSelectionPending ? '' : scenarioMode;

  const localeOptions: SelectOption<Locale>[] = LOCALE_OPTIONS.map((option) => ({
    value: option.value,
    label: option.label,
    description: option.value === 'en' ? 'English locale formatting.' : 'Spanish locale formatting.',
  }));

  const routeSortOptions: SelectOption<'duration' | 'cost' | 'co2'>[] = [
    { value: 'duration', label: 'Sort By ETA', description: 'Show fastest routes first.' },
    { value: 'cost', label: 'Sort By Cost', description: 'Show lowest-cost routes first.' },
    { value: 'co2', label: 'Sort By CO2', description: 'Show lowest-emission routes first.' },
  ];
  const computeModeOptions: SelectOption<ComputeMode>[] = [
    {
      value: 'pareto_stream',
      label: t('compute_mode_pareto_stream'),
      description: 'Streams candidates as they are computed.',
    },
    {
      value: 'pareto_json',
      label: t('compute_mode_pareto_json'),
      description: 'Fetches full Pareto set in one JSON response.',
    },
    {
      value: 'route_single',
      label: t('compute_mode_route_single'),
      description: 'Calls single-route endpoint and returns selected+candidates.',
    },
  ];

  const showRoutesSection = loading || paretoRoutes.length > 0 || warnings.length > 0;
  const canCompareScenarios = Boolean(origin && destination) && !busy && !scenarioCompareLoading;
  const canSaveExperiment = Boolean(origin && destination) && !busy;
  const sidebarToggleLabel = isPanelCollapsed ? 'Extend sidebar' : 'Collapse sidebar';

  return (
    <div className={`app ${appBootReady ? 'isBootReady' : 'isBooting'}`}>
      <div className={`appBootOverlay ${appBootReady ? 'isHidden' : ''}`} aria-hidden={appBootReady}>
        <div className="appBootOverlay__veil" />
        <div className="appBootOverlay__spinner" role="status" aria-live="polite" aria-label="Loading">
          <span className="srOnly">Loading interface</span>
        </div>
      </div>
      <a href="#app-sidebar-content" className="skipLink">
        {t('skip_to_controls')}
      </a>
      <div className="srOnly" role="status" aria-live="polite" aria-atomic="true">
        {liveMessage}
      </div>
      <div
        className={`mapStage ${tutorialMapDimmed ? 'isTutorialLocked' : ''} ${tutorialGuideVisible ? 'isTutorialGuided' : ''}`.trim()}
      >
        <MapView
          origin={mapOriginForRender}
          destination={mapDestinationForRender}
          tutorialDraftOrigin={tutorialDraftOrigin}
          tutorialDraftDestination={tutorialDraftDestination}
          tutorialDragDraftOrigin={tutorialDragDraftOrigin}
          tutorialDragDraftDestination={tutorialDragDraftDestination}
          managedStop={managedStop}
          originLabel="Start"
          destinationLabel="End"
          selectedPinId={selectedPinId}
          focusPinRequest={focusPinRequest}
          fitAllRequestNonce={fitAllRequestNonce}
          route={selectedRoute}
          failureOverlay={mapFailureOverlay}
          timeLapsePosition={timeLapsePosition}
          dutyStops={dutyStopsForOverlay}
          showStopOverlay={showStopOverlay}
          showIncidentOverlay={showIncidentOverlay}
          showSegmentTooltips={showSegmentTooltips}
          showPreviewConnector={showPreviewConnector}
          overlayLabels={mapOverlayLabels}
          tutorialMapLocked={tutorialMapLocked}
          tutorialMapDimmed={tutorialMapDimmed}
          tutorialViewportLocked={tutorialViewportLocked}
          tutorialHideZoomControls={tutorialHideZoomControls}
          tutorialRelaxBounds={tutorialRelaxBounds}
          tutorialExpectedAction={tutorialRunning ? tutorialBlockingActionId : null}
          tutorialGuideTarget={tutorialGuideTarget}
          tutorialGuideVisible={tutorialGuideVisible}
          tutorialConfirmPin={tutorialConfirmPin}
          tutorialDragConfirmPin={tutorialDragConfirmPin}
          onMapClick={handleMapClick}
          onSelectPinId={setSelectedPinId}
          onMoveMarker={handleMoveMarker}
          onMoveStop={handleMoveStop}
          onAddStopFromPin={addStopFromMidpoint}
          onRenameStop={renameStop}
          onDeleteStop={deleteStop}
          onFocusPin={setSelectedPinId}
          onSwapMarkers={swapMarkers}
          onTutorialConfirmPin={handleTutorialConfirmPin}
          onTutorialConfirmDrag={handleTutorialConfirmDrag}
          onTutorialAction={markTutorialAction}
          onTutorialTargetState={(state) => {
            if (state.hasSegmentTooltipPath) {
              markTutorialAction('map.segment_tooltip_available');
            }
            if (state.hasIncidentMarkers) {
              markTutorialAction('map.incidents_available');
            }
          }}
        />
        {mapFailureOverlay ? (
          <section className="mapFailureCard" role="status" aria-live="polite" aria-label="Route compute failure">
            <div className="mapFailureCard__title">Route compute failed</div>
            <div className="mapFailureCard__reason">reason_code={mapFailureOverlay.reason_code}</div>
            {mapFailureOverlay.stage ? (
              <div className="mapFailureCard__stage">
                stage={mapFailureOverlay.stage}
                {mapFailureOverlay.stage_detail ? `; detail=${mapFailureOverlay.stage_detail}` : ''}
              </div>
            ) : null}
            <div className="mapFailureCard__message">{mapFailureOverlay.message}</div>
            <div className="mapFailureCard__actions">
              <button
                type="button"
                className="ghostButton"
                onClick={() => {
                  setComputeTraceOpen((prev) => !prev);
                }}
              >
                {computeTraceOpen ? 'Hide compute log' : 'Show compute log'}
              </button>
              <button
                type="button"
                className="ghostButton"
                onClick={() => {
                  void copyLatestAiDiagnosticBundle();
                }}
              >
                Copy diagnostic bundle
              </button>
            </div>
          </section>
        ) : null}
      </div>

      {!tutorialRunning ? (
        <button
          type="button"
          className={`sidebarToggle ${isPanelCollapsed ? 'isCollapsed' : ''}`}
          onClick={() => setIsPanelCollapsed((prev) => !prev)}
          aria-label={sidebarToggleLabel}
          aria-pressed={isPanelCollapsed}
          title={sidebarToggleLabel}
        >
          <SidebarToggleIcon collapsed={isPanelCollapsed} />
        </button>
      ) : null}

      <aside
        className={`panel ${isPanelCollapsed ? 'isCollapsed' : ''} ${tutorialSidebarLocked ? 'isTutorialLocked' : ''} ${tutorialSidebarActionLocked ? 'isTutorialActionLocked' : ''}`.trim()}
        aria-hidden={isPanelCollapsed}
      >
            <header className="panelHeader">
              <div className="panelHeader__top">
                <h1>{t('panel_title')}</h1>
              </div>
            </header>

            <div id="app-sidebar-content" className="panelBody">
              <CollapsibleCard
                title={t('setup')}
                hint={SIDEBAR_SECTION_HINTS.setup}
                dataTutorialId="setup.section"
                isOpen={tutorialSectionControl.setup?.isOpen}
                lockToggle={tutorialSectionControl.setup?.lockToggle}
                tutorialLocked={tutorialSectionControl.setup?.tutorialLocked}
              >

                <div className="fieldLabelRow">
                  <div className="fieldLabel">Vehicle type</div>
                  <FieldInfo text={SIDEBAR_FIELD_HELP.vehicleType} />
                </div>
                <Select
                  ariaLabel="Vehicle type"
                  value={vehicleSelectValue}
                  options={vehicleOptions}
                  placeholder="Select vehicle profile"
                  onChange={(next) => {
                    if (!next) return;
                    setVehicleType(next);
                    markTutorialAction('setup.vehicle_select');
                    markTutorialAction(`setup.vehicle_option:${next}`);
                  }}
                  disabled={busy}
                  tutorialId="setup.vehicle"
                  tutorialActionPrefix="setup.vehicle_option"
                />

                <div className="fieldLabelRow">
                  <div className="fieldLabel">Scenario mode</div>
                  <FieldInfo text={SIDEBAR_FIELD_HELP.scenarioMode} />
                </div>
                <Select
                  ariaLabel="Scenario mode"
                  value={scenarioSelectValue}
                  options={scenarioOptions}
                  placeholder="Select scenario mode"
                  onChange={(next) => {
                    if (!next) return;
                    setScenarioMode(next);
                    markTutorialAction('setup.scenario_select');
                    markTutorialAction(`setup.scenario_option:${next}`);
                  }}
                  disabled={busy}
                  tutorialId="setup.scenario"
                  tutorialActionPrefix="setup.scenario_option"
                />

                <div className="fieldLabelRow">
                  <label className="fieldLabel" htmlFor="ui-language">
                    {t('language')}
                  </label>
                  <FieldInfo text={SIDEBAR_FIELD_HELP.language} />
                </div>
                <Select
                  id="ui-language"
                  ariaLabel="Language"
                  value={locale}
                  options={localeOptions}
                  disabled={busy}
                  onChange={setLocale}
                  tutorialId="setup.language"
                  tutorialAction="setup.language_select"
                />

                {scenarioMode !== 'no_sharing' ? (
                  <div className="contextNote">
                    Sharing Mode Is Active. Scenario Profiles Now Apply Strict Policy Multipliers
                    Across ETA, Incidents, Fuel/Energy, Emissions, And Uncertainty.
                  </div>
                ) : null}

                <div className="setupActionGroup">
                  <div className="setupActionGroup__head">
                    <div>
                      <div className="setupActionGroup__title">Map Actions</div>
                      <div className="setupActionGroup__hint">
                        Quick controls for pin setup, viewport fit, and reset.
                      </div>
                    </div>
                    <button
                      type="button"
                      className="iconActionButton"
                      onClick={() => {
                        if (!tutorialIsDesktop) {
                          setTutorialMode('blocked');
                          setTutorialOpen(true);
                          return;
                        }
                        if (tutorialSavedProgress && !tutorialCompleted) {
                          setTutorialMode('chooser');
                          setTutorialOpen(true);
                          return;
                        }
                        startTutorialFresh();
                      }}
                      disabled={busy}
                      title={t('start_tutorial')}
                      aria-label={t('start_tutorial')}
                    >
                      <TutorialSparkIcon />
                    </button>
                  </div>

                  <div className="actionGrid u-mt12">
                    <button type="button"
                      className="secondary"
                      onClick={() => {
                        setOrigin(TUTORIAL_CANONICAL_ORIGIN);
                        setDestination(TUTORIAL_CANONICAL_DESTINATION);
                        setManagedStop(null);
                        setSelectedPinId(null);
                        clearComputed();
                        setFitAllRequestNonce((prev) => prev + 1);
                      }}
                      disabled={busy}
                      data-tutorial-action="setup.sample_pins_button"
                    >
                      Use Sample Pins
                    </button>
                    <button type="button"
                      className="secondary"
                      onClick={() => setFitAllRequestNonce((prev) => prev + 1)}
                      disabled={busy || (!origin && !destination && !managedStop)}
                      data-tutorial-action="setup.fit_map_button"
                    >
                      Fit Map To Pins
                    </button>
                    <button type="button"
                      className="secondary"
                      onClick={swapMarkers}
                      disabled={!origin || !destination || busy}
                      title="Swap Start and End"
                      data-tutorial-action="setup.swap_pins_button"
                    >
                      Swap Pins
                    </button>
                    <button type="button"
                      className="secondary"
                      onClick={reset}
                      disabled={busy}
                      data-tutorial-action="setup.clear_pins_button"
                    >
                      Clear Pins
                    </button>
                  </div>
                </div>
              </CollapsibleCard>

              <PinManager
                nodes={pinNodes}
                selectedPinId={selectedPinId}
                disabled={busy || tutorialSidebarActionLocked}
                hasStop={Boolean(managedStop)}
                canAddStop={canAddStop}
                tutorialRunning={tutorialRunning}
                tutorialStepId={tutorialStep?.id ?? null}
                tutorialBlockingActionId={tutorialBlockingActionId}
                oneStopHint={dutySyncError}
                onSelectPin={selectPinFromSidebar}
                onRenameStop={renameStop}
                onAddStop={addStopFromMidpoint}
                onDeleteStop={deleteStop}
                onSwapPins={swapMarkers}
                onClearPins={reset}
                sectionControl={tutorialSectionControl.pins}
              />

              <ScenarioParameterEditor
                value={advancedParams}
                onChange={setAdvancedParams}
                disabled={busy}
                validationError={advancedError}
                sectionControl={tutorialSectionControl.advanced}
              />

              <CollapsibleCard
                title={t('preferences')}
                hint={SIDEBAR_SECTION_HINTS.preferences}
                dataTutorialId="preferences.section"
                isOpen={tutorialSectionControl.preferences?.isOpen}
                lockToggle={tutorialSectionControl.preferences?.lockToggle}
                tutorialLocked={tutorialSectionControl.preferences?.tutorialLocked}
              >

                <div className="sliderField" data-tutorial-id="preferences.weights">
                  <div className="sliderField__head">
                    <label htmlFor="weight-time">Time</label>
                    <span className="sliderField__value">{weights.time}</span>
                  </div>
                  <p id="weight-time-help" className="sliderField__desc">
                    Prioritises shorter delivery duration. Increase this to favour faster routes.
                  </p>
                  <input
                    id="weight-time"
                    type="range"
                    min={0}
                    max={100}
                    value={weights.time}
                    aria-describedby="weight-time-help"
                    onChange={(e) => setWeights((w) => ({ ...w, time: Number(e.target.value) }))}
                    data-tutorial-action="pref.weight_time"
                  />
                </div>

                <div className="sliderField">
                  <div className="sliderField__head">
                    <label htmlFor="weight-money">Money</label>
                    <span className="sliderField__value">{weights.money}</span>
                  </div>
                  <p id="weight-money-help" className="sliderField__desc">
                    Prioritises lower operating cost. Increase this to favour cheaper routes.
                  </p>
                  <input
                    id="weight-money"
                    type="range"
                    min={0}
                    max={100}
                    value={weights.money}
                    aria-describedby="weight-money-help"
                    onChange={(e) => setWeights((w) => ({ ...w, money: Number(e.target.value) }))}
                    data-tutorial-action="pref.weight_money"
                  />
                </div>

                <div className="sliderField">
                  <div className="sliderField__head">
                    <label htmlFor="weight-co2">CO2</label>
                    <span className="sliderField__value">{weights.co2}</span>
                  </div>
                  <p id="weight-co2-help" className="sliderField__desc">
                    Prioritises lower carbon emissions. Increase this to favour cleaner routes.
                  </p>
                  <input
                    id="weight-co2"
                    type="range"
                    min={0}
                    max={100}
                    value={weights.co2}
                    aria-describedby="weight-co2-help"
                    onChange={(e) => setWeights((w) => ({ ...w, co2: Number(e.target.value) }))}
                    data-tutorial-action="pref.weight_co2"
                  />
                </div>

                <div className="actionGrid u-mt12">
                  {WEIGHT_PRESETS.map((preset) => (
                    <button type="button"
                      key={preset.id}
                      className="ghostButton"
                      onClick={() => setWeights(preset.value)}
                      disabled={busy}
                      data-tutorial-action={`pref.preset_${preset.id}`}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>

                <div className="fieldLabelRow u-mt12">
                  <div className="fieldLabel">{t('compute_mode_label')}</div>
                  <FieldInfo text={SIDEBAR_FIELD_HELP.computeMode} />
                </div>
                <Select
                  id="compute-mode"
                  ariaLabel="Compute mode"
                  value={computeMode}
                  options={computeModeOptions}
                  disabled={busy}
                  onChange={setComputeMode}
                />

                {routeWarmupShowCard && (
                  <div className={routeReadinessCardClassName} role="status" aria-live="polite">
                    <div className="routeReadinessCard__head">
                      <span className="routeReadinessCard__title">Route Graph Warmup</span>
                      <span className="routeReadinessCard__state">{routeReadinessSummary}</span>
                    </div>
                    <div className="routeReadinessCard__phase">
                      Phase: {routeWarmupPhaseLabel}
                    </div>
                    {routeGraphState === 'loading' && (
                      <>
                        <div
                          className="routeReadinessCard__meter"
                          role="progressbar"
                          aria-label="Route graph warmup progress"
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={routeWarmupProgressPct}
                        >
                          <span
                            className="routeReadinessCard__meterFill"
                            style={{ width: `${routeWarmupProgressPct}%` }}
                          />
                        </div>
                        <div className="routeReadinessCard__meta">
                          <div>
                            <div className="routeReadinessCard__metaLabel">Progress</div>
                            <div className="routeReadinessCard__metaValue">{routeWarmupProgressPct}%</div>
                          </div>
                          <div>
                            <div className="routeReadinessCard__metaLabel">Elapsed</div>
                            <div className="routeReadinessCard__metaValue">{routeWarmupElapsedLabel}</div>
                          </div>
                          <div>
                            <div className="routeReadinessCard__metaLabel">Remaining</div>
                            <div className="routeReadinessCard__metaValue">{routeWarmupRemainingLabel}</div>
                          </div>
                          <div>
                            <div className="routeReadinessCard__metaLabel">Baseline</div>
                            <div className="routeReadinessCard__metaValue">{routeWarmupBaselineLabel}</div>
                          </div>
                        </div>
                        {routeWarmupOverrunLabel && (
                          <div className="routeReadinessCard__overrun">{routeWarmupOverrunLabel}</div>
                        )}
                      </>
                    )}
                    <div className="routeReadinessCard__diag">
                      nodes_seen={formatNumber(routeGraphNodesSeen, locale, { maximumFractionDigits: 0 })}; nodes_kept=
                      {formatNumber(routeGraphNodesKept, locale, { maximumFractionDigits: 0 })}; edges_seen=
                      {formatNumber(routeGraphEdgesSeen, locale, { maximumFractionDigits: 0 })}; edges_kept=
                      {formatNumber(routeGraphEdgesKept, locale, { maximumFractionDigits: 0 })}
                    </div>
                    {routeGraphState === 'failed' && (
                      <div className="routeReadinessCard__error">
                        {String(routeGraphWarmup?.last_error ?? 'unknown warmup failure')}
                      </div>
                    )}
                  </div>
                )}

                <div className="tiny u-mt8">
                  Route readiness: {routeReadinessSummary}
                  {routeReadinessDetail ? ` | ${routeReadinessDetail}` : ''}
                </div>

                <div className="actionGrid u-mt10">
                  <button type="button"
                    className="primary"
                    onClick={computePareto}
                    disabled={!canCompute}
                    data-tutorial-id="preferences.compute_button"
                    data-tutorial-action="pref.compute_pareto_click"
                  >
                    {busy ? (
                      <span className="buttonLabel">
                        <span className="spinner spinner--inline" />
                        <span>
                          {t('computing')}
                          {progressText ? ` ${progressText}` : ''}
                        </span>
                      </span>
                    ) : (
                      t('compute_pareto')
                    )}
                  </button>

                  <button type="button"
                    className="secondary"
                    onClick={clearComputedFromUi}
                    disabled={busy || paretoRoutes.length === 0}
                    data-tutorial-action="pref.clear_results_click"
                  >
                    {t('clear_results')}
                  </button>

                  {loading && (
                    <button type="button" className="secondary" onClick={cancelCompute}>
                      Cancel
                    </button>
                  )}
                </div>

                <div className="u-mt8">
                  <button
                    type="button"
                    className="ghostButton"
                    onClick={() => setComputeTraceOpen((prev) => !prev)}
                    disabled={!loading && computeTrace.length === 0}
                  >
                    {computeTraceOpen ? 'Hide compute log' : 'Show compute log'}
                  </button>
                </div>

                <div className="tiny">
                  Weight sum: {formatNumber(weightSum, locale, { maximumFractionDigits: 0 })} |{' '}
                  Relative influence: Time{' '}
                  {formatNumber(normalisedWeights.time * 100, locale, {
                    maximumFractionDigits: 0,
                  })}
                  % | Money{' '}
                  {formatNumber(normalisedWeights.money * 100, locale, {
                    maximumFractionDigits: 0,
                  })}
                  % | CO2{' '}
                  {formatNumber(normalisedWeights.co2 * 100, locale, {
                    maximumFractionDigits: 0,
                  })}
                  %
                </div>

                {error && <div className="error">{error}</div>}
              </CollapsibleCard>

          {m && (
            <CollapsibleCard
              title="Selected Route"
              hint={SIDEBAR_SECTION_HINTS.selectedRoute}
              dataTutorialId="selected.route_panel"
              className="selectedRouteCard"
              isOpen={tutorialSectionControl.selectedRoute?.isOpen}
              lockToggle={tutorialSectionControl.selectedRoute?.lockToggle}
              tutorialLocked={tutorialSectionControl.selectedRoute?.tutorialLocked}
            >
              <div data-tutorial-action="selected.panel_click">
              <div className="metrics">
                <div className="metric">
                  <div className="metric__label">Distance</div>
                  <div className="metric__value">
                    {formatNumber(m.distance_km, locale, { maximumFractionDigits: 2 })} km
                  </div>
                </div>
                <div className="metric">
                  <div className="metric__label">Duration</div>
                  <div className="metric__value">
                    {formatNumber(m.duration_s / 60, locale, { maximumFractionDigits: 1 })} min
                  </div>
                </div>
                <div className="metric">
                  <div className="metric__label">£ (proxy)</div>
                  <div className="metric__value">
                    {formatNumber(m.monetary_cost, locale, { maximumFractionDigits: 2 })}
                  </div>
                </div>
                <div className="metric">
                  <div className="metric__label">CO2</div>
                  <div className="metric__value">
                    {formatNumber(m.emissions_kg, locale, { maximumFractionDigits: 3 })} kg
                  </div>
                </div>
                <div className="metric">
                  <div className="metric__label">Avg speed</div>
                  <div className="metric__value">
                    {formatNumber(m.avg_speed_kmh, locale, { maximumFractionDigits: 1 })} km/h
                  </div>
                </div>
                {selectedLabel && (
                  <div className="metric">
                    <div className="metric__label">Route</div>
                    <div className="metric__value">
                      {selectedLabel}
                      {selectedRoute?.is_knee ? ' (knee)' : ''}
                    </div>
                  </div>
                )}
              </div>
              {terrainSummary && terrainBadgeLabel ? (
                <div className="terrainBadgeRow">
                  <span className={terrainBadgeClass} title={terrainTooltip ?? undefined}>
                    {terrainBadgeLabel}
                  </span>
                  <span className="terrainBadgeMeta">
                    Coverage {(terrainSummary.coverage_ratio * 100).toFixed(1)}% | Confidence{' '}
                    {(terrainSummary.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ) : null}

              <div className="actionGrid u-mt10">
                <button type="button"
                  className="secondary"
                  onClick={() => setFitAllRequestNonce((prev) => prev + 1)}
                  disabled={!selectedRoute}
                  data-tutorial-action="selected.fit_route"
                >
                  Fit Map To Route
                </button>
                <button type="button"
                  className="secondary"
                  onClick={async () => {
                    if (!selectedRoute) return;
                    const summary = [
                      `Route: ${selectedLabel ?? selectedRoute.id}`,
                      `Distance km: ${selectedRoute.metrics.distance_km.toFixed(2)}`,
                      `Duration s: ${selectedRoute.metrics.duration_s.toFixed(1)}`,
                      `Cost: ${selectedRoute.metrics.monetary_cost.toFixed(2)}`,
                      `CO2 kg: ${selectedRoute.metrics.emissions_kg.toFixed(3)}`,
                    ].join('\n');
                    try {
                      await navigator.clipboard.writeText(summary);
                    } catch {
                      // no-op
                    }
                  }}
                  disabled={!selectedRoute}
                  data-tutorial-action="selected.copy_summary"
                >
                  Copy Summary
                </button>
              </div>

              {selectedRoute?.eta_explanations?.length ? (
                <div className="u-mt12">
                  <div className="fieldLabel u-mb6">
                    ETA explanation
                  </div>
                  <ul className="u-m0 u-pl16">
                    {selectedRoute.eta_explanations.map((item, idx) => (
                      <li key={`${idx}-${item}`} className="tiny">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <div className="u-mt12">
                <EtaTimelineChart route={selectedRoute} />
              </div>

              <SegmentBreakdown route={selectedRoute} onTutorialAction={markTutorialAction} />
              <CounterfactualPanel route={selectedRoute} />
              </div>
            </CollapsibleCard>
          )}

          <ScenarioTimeLapse
            route={selectedRoute}
            onPositionChange={setTimeLapsePosition}
            sectionControl={tutorialSectionControl.timelapse}
          />

          {showRoutesSection && (
            <CollapsibleCard
              className={`routesSection ${isPending ? 'isUpdating' : ''}`}
              title="Routes"
              hint={SIDEBAR_SECTION_HINTS.routes}
              dataTutorialId="routes.list"
              isOpen={tutorialSectionControl.routes?.isOpen}
              lockToggle={tutorialSectionControl.routes?.lockToggle}
              tutorialLocked={tutorialSectionControl.routes?.tutorialLocked}
            >
              <div className="sectionTitleRow">
                <div className="sectionTitleMeta">
                  {loading && <span className="statusPill">Computing {progressText ?? '...'}</span>}

                  {warnings.length > 0 && (
                    <button
                      type="button"
                      className={`warningPill warningPill--button ${showWarnings ? 'isOpen' : ''}`}
                      onClick={() => setShowWarnings((prev) => !prev)}
                      aria-expanded={showWarnings}
                      aria-controls="route-warning-list"
                    >
                      {warnings.length} warning{warnings.length === 1 ? '' : 's'}
                    </button>
                  )}

                  {hasNameOverrides && (
                    <button
                      type="button"
                      className="ghostButton"
                      onClick={resetRouteNames}
                      disabled={busy}
                      data-tutorial-id="routes.reset_names"
                      data-tutorial-action="routes.reset_names"
                    >
                      Reset names
                    </button>
                  )}
                </div>
              </div>

              {warnings.length > 0 && showWarnings && (
                <div
                  id="route-warning-list"
                  className="warningPanel"
                  role="region"
                  aria-labelledby="route-warning-title"
                >
                  <div id="route-warning-title" className="warningPanel__title">
                    Route generation warnings
                  </div>
                  <div className="warningPanel__hint">
                    Routing succeeded, but some candidate requests were skipped or failed.
                  </div>
                  <ul className="warningPanel__list">
                    {warnings.map((warning, idx) => (
                      <li
                        key={`${idx}-${warning}`}
                        className={
                          warning.includes('terrain_fail_closed')
                            ? 'warningPanel__item warningPanel__item--terrainFailClosed'
                            : 'warningPanel__item'
                        }
                      >
                        {warning}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {loading && paretoRoutes.length === 0 ? (
                <RoutesSkeleton />
              ) : (
                <>
                  {paretoRoutes.length > 0 && (
                    <>
                      <div className="actionGrid actionGrid--single u-mb10">
                        <Select
                          id="route-sort"
                          ariaLabel="Sort routes"
                          value={routeSort}
                          options={routeSortOptions}
                          onChange={setRouteSort}
                          tutorialAction="routes.sort_select"
                        />
                      </div>

                      <div className="chartWrap" data-tutorial-id="routes.chart">
                        <ParetoChart
                          routes={paretoRoutes}
                          selectedId={selectedId}
                          labelsById={labelsById}
                          onSelect={selectRouteFromChart}
                        />
                      </div>

                      <div className="helper u-mt10">
                        Tip: click a point (or a route card) to lock a specific route. Moving
                        sliders will re-select the best route for your weights.
                        {paretoRoutes.length === 1 && (
                          <>
                            <br />
                            Only one route was generated for this pair - try moving the pins a
                            little.
                          </>
                        )}
                      </div>

                      <ul className="routeList">
                        {sortedDisplayRoutes.map((route, idx) => {
                          const label = labelsById[route.id] ?? `Route ${idx + 1}`;
                          const isEditing = editingRouteId === route.id;
                          const isSelected = route.id === selectedId;

                          return (
                            <li
                              key={route.id}
                              className={`routeCard ${isSelected ? 'isSelected' : ''}`}
                              role="button"
                              tabIndex={0}
                              aria-label={`Select ${label}`}
                              aria-pressed={isSelected}
                              onClick={() => selectRouteFromCard(route.id)}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter' || event.key === ' ') {
                                  event.preventDefault();
                                  selectRouteFromCard(route.id);
                                }
                              }}
                              data-tutorial-action="routes.select_card"
                            >
                              <div className="routeCard__top">
                                {isEditing ? (
                                  <div
                                    className="routeRename"
                                    onClick={(event) => event.stopPropagation()}
                                  >
                                    <input
                                      className="routeRename__input"
                                      value={editingName}
                                      onChange={(event) => setEditingName(event.target.value)}
                                      onKeyDown={(event) => {
                                        if (event.key === 'Enter') {
                                          event.preventDefault();
                                          commitRename();
                                        }
                                        if (event.key === 'Escape') {
                                          event.preventDefault();
                                          cancelRename();
                                        }
                                      }}
                                      autoFocus
                                      aria-label="Route name"
                                    />
                                    <button
                                      type="button"
                                      className="routeRename__btn"
                                      onClick={commitRename}
                                      data-tutorial-action="routes.rename_save"
                                    >
                                      Save
                                    </button>
                                    <button
                                      type="button"
                                      className="routeRename__btn routeRename__btn--secondary"
                                      onClick={cancelRename}
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                ) : (
                                  <div className="routeCard__titleWrap">
                                    <div
                                      className="routeCard__id"
                                      onDoubleClick={(event) => {
                                        event.stopPropagation();
                                        beginRename(route.id);
                                      }}
                                    >
                                      {label}
                                    </div>

                                    <button
                                      type="button"
                                      className="routeCard__renameBtn"
                                      disabled={busy}
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        beginRename(route.id);
                                      }}
                                      data-tutorial-action="routes.rename_start"
                                    >
                                      Rename
                                    </button>
                                  </div>
                                )}

                                <div className="routeCard__pill">
                                  {formatNumber(route.metrics.duration_s / 60, locale, {
                                    maximumFractionDigits: 1,
                                  })}{' '}
                                  min
                                  {route.is_knee ? ' • knee' : ''}
                                </div>
                              </div>

                              <div className="routeCard__meta">
                                <span>
                                  {formatNumber(route.metrics.emissions_kg, locale, {
                                    maximumFractionDigits: 3,
                                  })}{' '}
                                  kg CO2
                                </span>
                                <span>
                                  £
                                  {formatNumber(route.metrics.monetary_cost, locale, {
                                    maximumFractionDigits: 2,
                                  })}
                                </span>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    </>
                  )}
                </>
              )}
            </CollapsibleCard>
          )}

          <CollapsibleCard
            title={t('compare_scenarios')}
            hint={SIDEBAR_SECTION_HINTS.compareScenarios}
            dataTutorialId="compare.section"
            isOpen={tutorialSectionControl.compare?.isOpen}
            lockToggle={tutorialSectionControl.compare?.lockToggle}
            tutorialLocked={tutorialSectionControl.compare?.tutorialLocked}
          >
            <div className="sectionTitleRow">
              <button type="button"
                className="secondary"
                onClick={compareScenarios}
                disabled={!canCompareScenarios}
                data-tutorial-action="compare.run_click"
              >
                {scenarioCompareLoading ? t('comparing_scenarios') : t('compare_scenarios')}
              </button>
            </div>
            <ScenarioComparison
              data={scenarioCompare}
              loading={scenarioCompareLoading}
              error={scenarioCompareError}
              locale={locale}
              onInspectScenarioManifest={(runId) => {
                void inspectScenarioManifestForRun(runId);
              }}
              onInspectScenarioSignature={(runId) => {
                void inspectScenarioSignatureForRun(runId);
              }}
              onOpenRunInspector={openRunInspectorForRun}
            />
          </CollapsibleCard>

          <DepartureOptimizerChart
            windowStartLocal={depWindowStartLocal}
            windowEndLocal={depWindowEndLocal}
            earliestArrivalLocal={depEarliestArrivalLocal}
            latestArrivalLocal={depLatestArrivalLocal}
            stepMinutes={depStepMinutes}
            loading={depOptimizeLoading}
            error={depOptimizeError}
            data={depOptimizeData}
            disabled={!origin || !destination || busy}
            onWindowStartChange={setDepWindowStartLocal}
            onWindowEndChange={setDepWindowEndLocal}
            onEarliestArrivalChange={setDepEarliestArrivalLocal}
            onLatestArrivalChange={setDepLatestArrivalLocal}
            onStepMinutesChange={setDepStepMinutes}
            onRun={optimizeDepartures}
            onApplyDepartureTime={applyOptimizedDeparture}
            locale={locale}
            sectionControl={tutorialSectionControl.departure}
          />

          <DutyChainPlanner
            stopsText={dutyStopsText}
            onStopsTextChange={handleDutyStopsTextChange}
            onRun={runDutyChain}
            loading={dutyChainLoading}
            error={dutyChainError ?? dutySyncError}
            data={dutyChainData}
            disabled={busy}
            locale={locale}
            sectionControl={tutorialSectionControl.duty}
          />

          <OracleQualityDashboard
            dashboard={oracleDashboard}
            loading={oracleDashboardLoading}
            ingesting={oracleIngestLoading}
            error={oracleError}
            latestCheck={oracleLatestCheck}
            disabled={busy}
            onRefresh={() => {
              void refreshOracleDashboard();
            }}
            onIngest={ingestOracleCheck}
            locale={locale}
            tutorialResetNonce={tutorialResetNonce}
            sectionControl={tutorialSectionControl.oracle}
          />

          <ExperimentManager
            experiments={experiments}
            loading={experimentsLoading}
            error={experimentsError}
            canSave={canSaveExperiment}
            disabled={busy}
            onRefresh={() => {
              void loadExperiments();
            }}
            onSave={saveCurrentExperiment}
            onLoad={(bundle) => {
              markTutorialAction('exp.load_click');
              void openExperimentById(bundle.id);
              clearComputed();
            }}
            onOpen={(experimentId) => {
              void openExperimentById(experimentId);
            }}
            onUpdateMetadata={(bundle, next) => {
              void updateExperimentMetadata(bundle, next);
            }}
            onDelete={deleteExperimentById}
            onReplay={replayExperimentById}
            catalogQuery={expCatalogQuery}
            catalogVehicleType={expCatalogVehicleType}
            catalogScenarioMode={expCatalogScenarioMode}
            catalogSort={expCatalogSort}
            vehicleOptions={vehicleOptions}
            onCatalogQueryChange={setExpCatalogQuery}
            onCatalogVehicleTypeChange={setExpCatalogVehicleType}
            onCatalogScenarioModeChange={setExpCatalogScenarioMode}
            onCatalogSortChange={setExpCatalogSort}
            onApplyCatalogFilters={() => {
              void loadExperiments({
                q: expCatalogQuery,
                vehicleType: expCatalogVehicleType,
                scenarioMode: expCatalogScenarioMode,
                sort: expCatalogSort,
              });
            }}
            locale={locale}
            defaultName={tutorialExperimentPrefill?.name}
            defaultDescription={tutorialExperimentPrefill?.description}
            tutorialResetNonce={tutorialResetNonce}
            sectionControl={tutorialSectionControl.experiments}
          />

          <CollapsibleCard title="Dev Tools" hint={SIDEBAR_SECTION_HINTS.devTools} defaultCollapsed={true}>
            <OpsDiagnosticsPanel
              health={opsHealth}
              metrics={opsMetrics}
              cacheStats={opsCacheStats}
              loading={opsLoading}
              clearing={opsClearing}
              error={opsError}
              onRefresh={() => {
                void refreshOpsDiagnostics();
              }}
              onClearCache={() => {
                void clearOpsCache();
              }}
            />

            <CustomVehicleManager
              vehicles={customVehicles}
              loading={customVehiclesLoading}
              saving={customVehicleSaving}
              error={customVehicleError}
              onRefresh={() => {
                void refreshCustomVehicles();
              }}
              onCreate={(vehicle) => {
                void createCustomVehicle(vehicle);
              }}
              onUpdate={(vehicleId, vehicle) => {
                void updateCustomVehicle(vehicleId, vehicle);
              }}
              onDelete={(vehicleId) => {
                void removeCustomVehicle(vehicleId);
              }}
            />

            <BatchRunner
              loading={batchLoading}
              error={batchError}
              result={batchResult}
              onRunPairs={(request) => {
                void runBatchParetoRequest(request);
              }}
              onRunCsv={(request) => {
                void runBatchCsvRequest(request);
              }}
            />

            <RunInspector
              runId={runInspectorRunId}
              onRunIdChange={setRunInspectorRunId}
              loading={runInspectorLoading}
              error={runInspectorError}
              manifest={runManifest}
              scenarioManifest={runScenarioManifest}
              provenance={runProvenance}
              signature={runSignature}
              scenarioSignature={runScenarioSignature}
              artifacts={runArtifacts}
              artifactPreviewName={runArtifactPreviewName}
              artifactPreviewText={runArtifactPreviewText}
              onLoadCore={() => {
                void loadRunCoreDocs();
              }}
              onLoadArtifacts={() => {
                void loadRunArtifactsList();
              }}
              onPreviewArtifact={(name) => {
                void previewRunArtifact(name);
              }}
              onDownloadCore={(kind) => {
                void downloadRunCore(kind);
              }}
              onDownloadArtifact={(name) => {
                void downloadRunArtifact(name);
              }}
            />

            <SignatureVerifier
              loading={signatureLoading}
              error={signatureError}
              result={signatureResult}
              onVerify={(request) => {
                void runSignatureVerification(request);
              }}
            />
          </CollapsibleCard>
            </div>
      </aside>

      {computeTraceOpen && (
        <div className="computeTraceOverlay" role="dialog" aria-modal="false" aria-labelledby="compute-trace-title">
          <button
            type="button"
            className="computeTraceOverlay__backdrop"
            aria-label="Close compute diagnostics"
            onClick={() => setComputeTraceOpen(false)}
          />
          <section className="computeTraceOverlay__panel">
            <header className="computeTraceOverlay__header">
              <div>
                <h2 id="compute-trace-title" className="computeTraceOverlay__title">
                  Route compute diagnostics
                </h2>
                <p className="computeTraceOverlay__subtitle">
                  Live execution trace, fallback attempts, and recovery suggestions.
                </p>
              </div>
              <div className="computeTraceOverlay__headerActions">
                {loading ? (
                  <span className="statusPill">
                    <span className="spinner spinner--inline" /> Running
                  </span>
                ) : (
                  <span className="statusPill">Idle</span>
                )}
                <button
                  type="button"
                  className="ghostButton"
                  onClick={() => {
                    resetComputeTrace();
                  }}
                  disabled={loading || computeTrace.length === 0}
                >
                  Clear
                </button>
                <button
                  type="button"
                  className="ghostButton"
                  onClick={() => {
                    void copyComputeDiagnostics();
                  }}
                  disabled={computeTrace.length === 0}
                >
                  Copy log
                </button>
                <button
                  type="button"
                  className="ghostButton"
                  onClick={() => {
                    void copyLiveCallJson();
                  }}
                  disabled={Object.keys(computeLiveCallsByAttempt).length === 0}
                >
                  Copy live JSON
                </button>
                <button type="button" className="ghostButton" onClick={() => setComputeTraceOpen(false)}>
                  Close
                </button>
              </div>
            </header>

            <div className="computeTraceOverlay__meta">
              <span>Mode: {computeMode}</span>
              <span>Entries: {computeTrace.length}</span>
              {computeElapsedSeconds ? <span>Elapsed: {computeElapsedSeconds}s</span> : null}
              {progressText ? <span>Progress: {progressText}</span> : null}
              {activeAttemptEntry ? (
                <span>
                  Active attempt: #{activeAttemptEntry.attempt} {activeAttemptEntry.endpoint ?? ''}
                </span>
              ) : null}
              {activeAttemptEntry?.stage ? <span>Active stage: {activeAttemptEntry.stage}</span> : null}
              {activeAttemptEntry?.stageDetail ? <span>Stage detail: {activeAttemptEntry.stageDetail}</span> : null}
              {lastBackendHeartbeatAgeSec ? <span>Last backend heartbeat age: {lastBackendHeartbeatAgeSec}s</span> : null}
            </div>

            {computeSession ? (
              <section className="computeTraceOverlay__session" aria-label="Session details">
                <div className="computeTraceOverlay__sessionGrid">
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Session</span>
                    <span className="computeTraceOverlay__sessionValue">#{computeSession.seq}</span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Started</span>
                    <span className="computeTraceOverlay__sessionValue">{computeSession.startedAt}</span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Scenario</span>
                    <span className="computeTraceOverlay__sessionValue">{computeSession.scenario}</span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Vehicle</span>
                    <span className="computeTraceOverlay__sessionValue">{computeSession.vehicle}</span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Max alternatives</span>
                    <span className="computeTraceOverlay__sessionValue">{computeSession.maxAlternatives}</span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Waypoints</span>
                    <span className="computeTraceOverlay__sessionValue">{computeSession.waypointCount}</span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Origin</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {computeSession.origin.lat.toFixed(5)}, {computeSession.origin.lon.toFixed(5)}
                    </span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Destination</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {computeSession.destination.lat.toFixed(5)}, {computeSession.destination.lon.toFixed(5)}
                    </span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Stream idle timeout</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {computeSession.streamStallTimeoutMs} ms
                    </span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Attempt timeout</span>
                    <span className="computeTraceOverlay__sessionValue">{computeSession.attemptTimeoutMs} ms</span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Route fallback timeout</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {computeSession.routeFallbackTimeoutMs} ms
                    </span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Degrade steps</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {computeSession.degradeSteps.join(' -> ')}
                    </span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Active attempt</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {activeAttemptEntry
                        ? `#${activeAttemptEntry.attempt} ${activeAttemptEntry.endpoint ?? ''} alt=${activeAttemptEntry.alternativesUsed ?? 'n/a'}`
                        : 'none'}
                    </span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Backend heartbeat age</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {lastBackendHeartbeatAgeSec ? `${lastBackendHeartbeatAgeSec}s` : 'n/a'}
                    </span>
                  </div>
                  <div>
                    <span className="computeTraceOverlay__sessionLabel">Last event</span>
                    <span className="computeTraceOverlay__sessionValue">
                      {latestTraceEntry ? `${latestTraceEntry.step} (${latestTraceEntry.level})` : 'none'}
                    </span>
                  </div>
                </div>
              </section>
            ) : null}

            {error && computeErrorHints.length > 0 ? (
              <section className="computeTraceOverlay__hints" aria-label="Recovery suggestions">
                <div className="computeTraceOverlay__hintsTitle">Suggested recovery steps</div>
                <ul className="computeTraceOverlay__hintsList">
                  {computeErrorHints.map((hint) => (
                    <li key={hint}>{hint}</li>
                  ))}
                </ul>
              </section>
            ) : null}

            <div className="computeTraceOverlay__log" role="log" aria-live="polite" aria-relevant="additions text">
              {computeTrace.length === 0 ? (
                <div className="computeTraceOverlay__empty">No compute events yet.</div>
              ) : (
                [...traceEntriesByAttempt].reverse().map((group) => {
                  const headerEntry = [...group.entries]
                    .reverse()
                    .find((entry) => entry.endpoint || typeof entry.alternativesUsed === 'number');
                  const liveTrace = group.attempt > 0 ? computeLiveCallsByAttempt[group.attempt] ?? null : null;
                  const liveTraceRequestId =
                    (group.attempt > 0 ? computeRequestIdByAttempt[group.attempt] : null) ??
                    liveTrace?.request_id ??
                    null;
                  const liveObserved = (liveTrace?.observed_calls ?? []) as LiveCallEntry[];
                  const expectedRows = liveTrace?.expected_rollup ?? [];
                  const blockedExpectedRows = expectedRows.filter(
                    (row) => row.status === 'blocked' || row.blocked,
                  );
                  const notReachedExpectedRows = expectedRows.filter((row) => row.status === 'not_reached');
                  const missedExpectedRows = expectedRows.filter(
                    (row) => row.status === 'miss' || (!row.satisfied && row.status !== 'blocked' && row.status !== 'not_reached'),
                  );
                  const okExpectedRows = expectedRows.filter((row) => row.status === 'ok' || row.satisfied);
                  const slowestObserved = [...liveObserved]
                    .sort((a, b) => (Number(b.duration_ms ?? 0) - Number(a.duration_ms ?? 0)))
                    .slice(0, 6);
                  const latestDiagEntry =
                    [...group.entries]
                      .reverse()
                      .find((entry) => entry.candidateDiagnostics || entry.failureChain) ?? null;
                  const candidateDiagnostics = latestDiagEntry?.candidateDiagnostics ?? null;
                  const failureChain = latestDiagEntry?.failureChain ?? null;
                  const scenarioGateSignal = safeParseJsonObject(
                    candidateDiagnostics?.scenario_gate_source_signal_json,
                  );
                  const scenarioGateReachability = safeParseJsonObject(
                    candidateDiagnostics?.scenario_gate_source_reachability_json,
                  );
                  const scenarioGateRequiredConfigured = Number(
                    candidateDiagnostics?.scenario_gate_required_configured ?? 0,
                  );
                  const scenarioGateRequiredEffective = Number(
                    candidateDiagnostics?.scenario_gate_required_effective ?? 0,
                  );
                  const scenarioGateSourceOkCount = Number(
                    candidateDiagnostics?.scenario_gate_source_ok_count ?? 0,
                  );
                  const scenarioGateWaiverApplied = Boolean(
                    candidateDiagnostics?.scenario_gate_waiver_applied ?? false,
                  );
                  const scenarioGateWaiverReason = String(
                    candidateDiagnostics?.scenario_gate_waiver_reason ?? '',
                  ).trim();
                  const scenarioGateRoadHint = String(candidateDiagnostics?.scenario_gate_road_hint ?? '').trim();
                  const scenarioGate =
                    scenarioGateRequiredConfigured > 0 ||
                    scenarioGateRequiredEffective > 0 ||
                    scenarioGateSourceOkCount > 0 ||
                    scenarioGateWaiverApplied ||
                    Boolean(scenarioGateWaiverReason) ||
                    Boolean(scenarioGateRoadHint) ||
                    Boolean(scenarioGateSignal) ||
                    Boolean(scenarioGateReachability)
                      ? {
                          source_ok_count: Number.isFinite(scenarioGateSourceOkCount)
                            ? scenarioGateSourceOkCount
                            : 0,
                          required_configured: Number.isFinite(scenarioGateRequiredConfigured)
                            ? scenarioGateRequiredConfigured
                            : 0,
                          required_effective: Number.isFinite(scenarioGateRequiredEffective)
                            ? scenarioGateRequiredEffective
                            : 0,
                          waiver_applied: scenarioGateWaiverApplied,
                          waiver_reason: scenarioGateWaiverReason || null,
                          road_hint: scenarioGateRoadHint || null,
                          source_signal_set: scenarioGateSignal,
                          source_reachability_set: scenarioGateReachability,
                        }
                      : null;
                  const aiDiagnosticBundle = {
                    attempt: group.attempt,
                    endpoint: headerEntry?.endpoint ?? 'n/a',
                    request_id: liveTraceRequestId ?? null,
                    trace_status: liveTrace?.status ?? 'unavailable',
                    latest_reason_code:
                      [...group.entries]
                        .reverse()
                        .find((entry) => Boolean(entry.reasonCode))
                        ?.reasonCode ?? null,
                    summary: liveTrace?.summary ?? null,
                    expected_rollup: expectedRows,
                    slowest_calls: slowestObserved,
                    candidate_diagnostics: candidateDiagnostics,
                    scenario_gate: scenarioGate,
                    failure_chain: failureChain,
                  };
                  return (
                    <section key={`attempt-${group.attempt}`} className="computeTraceAttempt">
                      <header className="computeTraceAttempt__header">
                        <span className="computeTraceAttempt__title">
                          Attempt {group.attempt > 0 ? `#${group.attempt}` : 'unscoped'}
                        </span>
                        <span className="computeTraceAttempt__meta">
                          endpoint={headerEntry?.endpoint ?? 'n/a'}; alternatives=
                          {typeof headerEntry?.alternativesUsed === 'number'
                            ? headerEntry.alternativesUsed
                            : 'n/a'}
                          ; timeout_ms={typeof headerEntry?.timeoutMs === 'number' ? headerEntry.timeoutMs : 'n/a'}
                        </span>
                      </header>
                      {[...group.entries].reverse().map((entry) => (
                        <article
                          key={entry.id}
                          className={`computeTraceItem computeTraceItem--${entry.level}`}
                          aria-label={`${entry.level} ${entry.step}`}
                        >
                          <div className="computeTraceItem__row">
                            <span className="computeTraceItem__level">
                              #{entry.id} {entry.level.toUpperCase()}
                            </span>
                            <span className="computeTraceItem__time">
                              +{(entry.elapsedMs / 1000).toFixed(1)}s | {entry.at}
                            </span>
                          </div>
                          <div className="computeTraceItem__step">{entry.step}</div>
                          <div className="computeTraceItem__detail">{entry.detail}</div>
                          <div className="computeTraceItem__reason">
                            attempt={entry.attempt ?? 'n/a'}; endpoint={entry.endpoint ?? 'n/a'}
                            {entry.requestId ? `; request_id=${entry.requestId}` : ''}
                            {entry.stage ? `; stage=${entry.stage}` : ''}
                            {entry.stageDetail ? `; stage_detail=${entry.stageDetail}` : ''}
                            {typeof entry.backendElapsedMs === 'number'
                              ? `; backend_elapsed_ms=${entry.backendElapsedMs}`
                              : ''}
                            {typeof entry.stageElapsedMs === 'number'
                              ? `; stage_elapsed_ms=${entry.stageElapsedMs}`
                              : ''}
                            {typeof entry.alternativesUsed === 'number'
                              ? `; alternatives=${entry.alternativesUsed}`
                              : ''}
                            {typeof entry.timeoutMs === 'number' ? `; timeout_ms=${entry.timeoutMs}` : ''}
                            {entry.abortReason ? `; abort_reason=${entry.abortReason}` : ''}
                            {entry.reasonCode ? `; reason_code=${entry.reasonCode}` : ''}
                          </div>
                          {entry.recoveries?.length ? (
                            <ul className="computeTraceItem__recoveryList">
                              {entry.recoveries.map((item) => (
                                <li key={`${entry.id}-${item}`}>{item}</li>
                              ))}
                            </ul>
                          ) : null}
                        </article>
                      ))}
                      {group.attempt > 0 ? (
                        <section className="computeTraceLiveCalls" aria-label={`Live API calls for attempt ${group.attempt}`}>
                          <header className="computeTraceLiveCalls__header">
                            <span className="computeTraceLiveCalls__title">Live API calls</span>
                            <span className="computeTraceLiveCalls__meta">
                              request_id={liveTraceRequestId ?? 'pending'}
                            </span>
                          </header>
                          {liveTrace ? (
                            <>
                              <div className="computeTraceLiveCalls__summary">
                                <span>calls={liveTrace.summary?.total_calls ?? 0}</span>
                                <span>requested={liveTrace.summary?.requested_calls ?? 0}</span>
                                <span>success={liveTrace.summary?.successful_calls ?? 0}</span>
                                <span>failed={liveTrace.summary?.failed_calls ?? 0}</span>
                                <span>blocked={blockedExpectedRows.length}</span>
                                <span>not_reached={notReachedExpectedRows.length}</span>
                                <span>miss={missedExpectedRows.length}</span>
                                <span>cache_hits={liveTrace.summary?.cache_hit_calls ?? 0}</span>
                                <span>expected={liveTrace.summary?.expected_satisfied ?? 0}/{liveTrace.summary?.expected_total ?? 0}</span>
                              </div>
                              {blockedExpectedRows.length > 0 && liveObserved.length === 0 ? (
                                <div className="computeTraceLiveCalls__blockedHint">
                                  Expected sources were blocked before execution could reach their phase.
                                </div>
                              ) : null}
                              {liveTrace.expected_rollup?.length ? (
                                <div className="computeTraceLiveCalls__expected">
                                  {liveTrace.expected_rollup.map((row) => (
                                    <div key={`${group.attempt}-${row.source_key}-${row.url}`} className="computeTraceLiveCalls__expectedRow">
                                      <span
                                        className={`computeTraceLiveCalls__badge ${
                                          row.status === 'blocked' || row.blocked
                                            ? 'is-blocked'
                                            : row.status === 'not_reached'
                                              ? 'is-not-reached'
                                              : row.status === 'ok' || row.satisfied
                                                ? 'is-ok'
                                                : 'is-fail'
                                        }`}
                                      >
                                        {row.status === 'blocked' || row.blocked
                                          ? 'BLOCKED'
                                          : row.status === 'not_reached'
                                            ? 'NOT_REACHED'
                                            : row.status === 'ok' || row.satisfied
                                              ? 'OK'
                                              : 'MISS'}
                                      </span>
                                      <span>{row.source_key}</span>
                                      <span className="computeTraceLiveCalls__url">{row.url}</span>
                                      <span>
                                        observed={row.observed_calls}; success={row.success_count}; failure={row.failure_count}
                                        {row.phase ? `; phase=${row.phase}` : ''}
                                        {row.gate ? `; gate=${row.gate}` : ''}
                                        {row.blocked_reason ? `; blocked_reason=${row.blocked_reason}` : ''}
                                        {row.blocked_stage ? `; blocked_stage=${row.blocked_stage}` : ''}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              ) : null}
                              {failureChain ? (
                                <section className="computeTraceDiagCard" aria-label="Failure chain">
                                  <div className="computeTraceDiagCard__title">Failure Chain</div>
                                  <pre className="computeTraceDiagCard__code">
                                    {safeJsonString(failureChain)}
                                  </pre>
                                </section>
                              ) : null}
                              {candidateDiagnostics ? (
                                <section className="computeTraceDiagCard" aria-label="Graph diagnostics">
                                  <div className="computeTraceDiagCard__title">Graph Diagnostics</div>
                                  <pre className="computeTraceDiagCard__code">
                                    {safeJsonString(candidateDiagnostics)}
                                  </pre>
                                </section>
                              ) : null}
                              {scenarioGate ? (
                                <section className="computeTraceDiagCard" aria-label="Scenario coverage gate">
                                  <div className="computeTraceDiagCard__title">Scenario Coverage Gate</div>
                                  <pre className="computeTraceDiagCard__code">
                                    {safeJsonString(scenarioGate)}
                                  </pre>
                                </section>
                              ) : null}
                              {expectedRows.length > 0 ? (
                                <section className="computeTraceDiagCard" aria-label="Live refresh gate matrix">
                                  <div className="computeTraceDiagCard__title">Live Refresh Gate Matrix</div>
                                  <div className="computeTraceGateMatrix">
                                    <span>ok={okExpectedRows.length}</span>
                                    <span>blocked={blockedExpectedRows.length}</span>
                                    <span>not_reached={notReachedExpectedRows.length}</span>
                                    <span>miss={missedExpectedRows.length}</span>
                                  </div>
                                </section>
                              ) : null}
                              {slowestObserved.length > 0 ? (
                                <section className="computeTraceDiagCard" aria-label="Slowest calls">
                                  <div className="computeTraceDiagCard__title">Slowest Calls</div>
                                  <div className="computeTraceDiagRows">
                                    {slowestObserved.map((row) => (
                                      <div key={`${group.attempt}-slow-${row.entry_id}`} className="computeTraceDiagRows__row">
                                        <span>{row.source_key}</span>
                                        <span>{typeof row.duration_ms === 'number' ? `${row.duration_ms.toFixed(2)}ms` : '-'}</span>
                                        <span>{row.status_code ?? '-'}</span>
                                      </div>
                                    ))}
                                  </div>
                                </section>
                              ) : null}
                              <section className="computeTraceDiagCard" aria-label="AI diagnostic bundle">
                                <div className="computeTraceDiagCard__title">AI Diagnostic Bundle</div>
                                <pre className="computeTraceDiagCard__code">
                                  {safeJsonString(aiDiagnosticBundle)}
                                </pre>
                              </section>
                              <div className="computeTraceLiveCalls__tableWrap">
                                <table className="computeTraceLiveCalls__table">
                                  <thead>
                                    <tr>
                                      <th>#</th>
                                      <th>Source</th>
                                      <th>URL</th>
                                      <th>Requested</th>
                                      <th>Success</th>
                                      <th>Status</th>
                                      <th>Error</th>
                                      <th>Cache</th>
                                      <th>Retries</th>
                                      <th>Duration</th>
                                      <th>Req Headers</th>
                                      <th>Resp Headers</th>
                                      <th>Resp Body</th>
                                      <th>Extra</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {[...liveObserved].reverse().map((row) => (
                                      <tr key={`${group.attempt}-${row.entry_id}`}>
                                        <td>{row.entry_id}</td>
                                        <td>{row.source_key}</td>
                                        <td className="computeTraceLiveCalls__urlCell">{row.url}</td>
                                        <td>{row.requested ? 'yes' : 'no'}</td>
                                        <td>{row.success ? 'yes' : 'no'}</td>
                                        <td>{typeof row.status_code === 'number' ? row.status_code : '-'}</td>
                                        <td>{row.fetch_error || '-'}</td>
                                        <td>
                                          hit={row.cache_hit ? '1' : '0'}; stale={row.stale_cache_used ? '1' : '0'}
                                        </td>
                                        <td>
                                          attempts={row.retry_attempts ?? 0}; count={row.retry_count ?? 0}; backoff_ms=
                                          {row.retry_total_backoff_ms ?? 0}
                                        </td>
                                        <td>{typeof row.duration_ms === 'number' ? `${row.duration_ms.toFixed(2)}ms` : '-'}</td>
                                        <td className="computeTraceLiveCalls__jsonCell">
                                          {row.request_headers_raw
                                            ? safeJsonString(row.request_headers_raw)
                                            : row.headers
                                              ? safeJsonString(row.headers)
                                              : '-'}
                                        </td>
                                        <td className="computeTraceLiveCalls__jsonCell">
                                          {row.response_headers_raw ? safeJsonString(row.response_headers_raw) : '-'}
                                        </td>
                                        <td className="computeTraceLiveCalls__jsonCell">
                                          {row.response_body_raw
                                            ? row.response_body_raw
                                            : typeof row.response_body_bytes === 'number'
                                              ? `bytes=${row.response_body_bytes}; content_type=${row.response_body_content_type ?? 'n/a'}`
                                              : '-'}
                                        </td>
                                        <td className="computeTraceLiveCalls__jsonCell">
                                          {row.extra ? safeJsonString(row.extra) : '-'}
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </>
                          ) : (
                            <div className="computeTraceLiveCalls__empty">Waiting for backend live-call trace rows...</div>
                          )}
                        </section>
                      ) : null}
                    </section>
                  );
                })
              )}
            </div>
          </section>
        </div>
      )}

      <TutorialOverlay
        open={appBootReady && tutorialOpen}
        mode={tutorialMode}
        isDesktop={tutorialIsDesktop}
        hasSavedProgress={Boolean(tutorialSavedProgress)}
        chapterTitle={tutorialChapter?.title ?? 'Tutorial'}
        chapterDescription={tutorialChapter?.description ?? ''}
        chapterIndex={tutorialChapterIndex}
        chapterCount={TUTORIAL_CHAPTERS.length}
        stepTitle={tutorialStep?.title ?? 'Tutorial'}
        stepWhat={tutorialStep?.what ?? 'Follow the guided checklist.'}
        stepImpact={tutorialStep?.impact ?? 'Each step explains how controls affect outcomes.'}
        stepIndex={tutorialStepIndex + 1}
        stepCount={TUTORIAL_STEPS.length}
        canGoNext={tutorialCanAdvance}
        atStart={tutorialAtStart}
        atEnd={tutorialAtEnd}
        checklist={tutorialChecklist}
        currentTaskOverride={tutorialCurrentTaskOverride}
        optionalDecision={tutorialOptionalState}
        targetRect={tutorialTargetRect}
        targetMissing={tutorialTargetMissing && !(tutorialStep?.allowMissingTarget ?? false)}
        requiresTargetRect={Boolean(tutorialStep?.targetIds?.length)}
        runningScope={tutorialLockScope}
        onClose={closeTutorial}
        onStartNew={startTutorialFresh}
        onResume={resumeTutorialProgress}
        onRestart={restartTutorialProgress}
        onBack={tutorialBack}
        onNext={tutorialNext}
        onFinish={tutorialFinish}
        onMarkManual={markTutorialAction}
        onUseOptionalDefault={markTutorialOptionalDefault}
      />
    </div>
  );
}

