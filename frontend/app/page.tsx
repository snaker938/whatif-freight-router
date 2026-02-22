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
import { deleteJSON, getJSON, getText, postJSON, postNDJSON, putJSON } from './lib/api';
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
  const requestSeqRef = useRef(0);
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
    routeBufferRef.current = [];
    clearFlushTimer();
  }, [clearFlushTimer]);

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
  }, [abortActiveCompute]);

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
  }, [abortActiveCompute]);

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
  const canCompute = Boolean(origin && destination) && !busy;

  const progressText = progress
    ? `${formatNumber(Math.min(progress.done, progress.total), locale, {
        maximumFractionDigits: 0,
      })}/${formatNumber(progress.total, locale, { maximumFractionDigits: 0 })}`
    : null;
  const normalisedWeights = useMemo(() => normaliseWeights(weights), [weights]);

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
    markTutorialAction('pref.compute_pareto_click');

    const seq = requestSeqRef.current + 1;
    requestSeqRef.current = seq;

    abortActiveCompute();

    const controller = new AbortController();
    abortRef.current = controller;

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

    let advancedPatch: RoutingAdvancedPatch;
    let maxAlternatives = 5;
    try {
      const parsed = buildAdvancedRequestPatch();
      advancedPatch = parsed.advancedPatch;
      maxAlternatives = parsed.maxAlternatives;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Invalid advanced parameter values.';
      setAdvancedError(msg);
      setError(msg);
      setLoading(false);
      abortRef.current = null;
      return;
    }

    const paretoBody = buildParetoRequest({
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
      advanced: advancedPatch,
    });
    const routeBody = buildRouteRequest({
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
      advanced: advancedPatch,
    });

    let sawDone = false;

    try {
      if (computeMode === 'pareto_stream') {
        await postNDJSON<ParetoStreamEvent>('/api/pareto/stream', paretoBody, {
          signal: controller.signal,
          onEvent: (event) => {
            if (seq !== requestSeqRef.current) return;

            switch (event.type) {
              case 'meta': {
                setProgress({ done: 0, total: event.total });
                return;
              }

              case 'route': {
                routeBufferRef.current.push(event.route);
                scheduleRouteFlush(seq);
                setProgress({ done: event.done, total: event.total });
                return;
              }

              case 'error': {
                setProgress({ done: event.done, total: event.total });
                setWarnings((prev) => dedupeWarnings([...prev, event.message]));
                return;
              }

              case 'fatal': {
                setError(event.message || 'Route computation failed.');
                return;
              }

              case 'done': {
                sawDone = true;
                flushRouteBufferNow(seq);
                markTutorialAction('pref.compute_pareto_done');

                const finalRoutes = sortRoutesDeterministic(event.routes ?? []);
                startTransition(() => {
                  setParetoRoutes(finalRoutes);
                });

                setProgress({ done: event.done, total: event.total });
                const doneWarnings = event.warnings ?? [];
                if (doneWarnings.length) {
                  setWarnings((prev) => dedupeWarnings([...prev, ...doneWarnings]));
                }
                return;
              }

              default:
                return;
            }
          },
        });

        if (seq === requestSeqRef.current) {
          flushRouteBufferNow(seq);
          if (!sawDone) {
            startTransition(() => {
              setParetoRoutes((prev) => sortRoutesDeterministic(prev));
            });
          }
        }
      } else if (computeMode === 'pareto_json') {
        const payload = await postJSON<ParetoResponse>('/api/pareto', paretoBody, controller.signal);
        if (seq !== requestSeqRef.current) return;
        const finalRoutes = sortRoutesDeterministic(payload.routes ?? []);
        startTransition(() => {
          setParetoRoutes(finalRoutes);
        });
        setProgress({ done: finalRoutes.length, total: finalRoutes.length });
        if (payload.warnings?.length) {
          setWarnings(dedupeWarnings(payload.warnings));
        }
        markTutorialAction('pref.compute_pareto_done');
      } else {
        const payload = await postJSON<RouteResponse>('/api/route', routeBody, controller.signal);
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
      }
    } catch (e: unknown) {
      if (seq !== requestSeqRef.current) return;
      if (controller.signal.aborted) return;
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      if (seq === requestSeqRef.current) {
        abortRef.current = null;
        setLoading(false);
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
      const [health, metrics, cacheStats] = await Promise.all([
        getJSON<HealthResponse>('/api/health'),
        getJSON<MetricsResponse>('/api/metrics'),
        getJSON<CacheStatsResponse>('/api/cache/stats'),
      ]);
      setOpsHealth(health);
      setOpsMetrics(metrics);
      setOpsCacheStats(cacheStats);
    } catch (e: unknown) {
      setOpsError(e instanceof Error ? e.message : 'Failed to load ops diagnostics.');
    } finally {
      setOpsLoading(false);
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

          <CollapsibleCard title="Dev Tools" hint={SIDEBAR_SECTION_HINTS.devTools} isOpen={true}>
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

