'use client';
// frontend/app/page.tsx

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useTransition } from 'react';

import CollapsibleCard from './components/CollapsibleCard';
import CounterfactualPanel from './components/CounterfactualPanel';
import DepartureOptimizerChart from './components/DepartureOptimizerChart';
import DutyChainPlanner from './components/DutyChainPlanner';
import ExperimentManager from './components/ExperimentManager';
import FieldInfo from './components/FieldInfo';
import OracleQualityDashboard from './components/OracleQualityDashboard';
import PinManager from './components/PinManager';
import ScenarioParameterEditor, {
  type ScenarioAdvancedParams,
} from './components/ScenarioParameterEditor';
import ScenarioTimeLapse from './components/ScenarioTimeLapse';
import Select, { type SelectOption } from './components/Select';
import { postJSON, postNDJSON } from './lib/api';
import { formatNumber } from './lib/format';
import { LOCALE_OPTIONS, createTranslator, type Locale } from './lib/i18n';
import { buildManagedPinNodes } from './lib/mapOverlays';
import {
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
  vehicleDescriptionFromId,
} from './lib/sidebarHelpText';
import {
  LEGACY_TUTORIAL_COMPLETED_KEYS,
  LEGACY_TUTORIAL_PROGRESS_KEYS,
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
  onMoveStop?: (lat: number, lon: number) => void;
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
  stop: ManagedStop | null,
  destination: LatLng | null,
): string {
  const lines: string[] = [];
  if (origin) {
    lines.push(toDutyLine(origin.lat, origin.lon, 'Start'));
  }
  if (stop) {
    lines.push(toDutyLine(stop.lat, stop.lon, stop.label || 'Stop #1'));
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
    };
  }

  if (lines.length > 3) {
    return { ok: false, error: 'One-stop mode allows at most one intermediate stop (three lines total).' };
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
    };
  }

  const first = parsed[0];
  const last = parsed[parsed.length - 1];
  const mid = parsed.length === 3 ? parsed[1] : null;
  return {
    ok: true,
    origin: { lat: first.lat, lon: first.lon },
    destination: { lat: last.lat, lon: last.lon },
    stop: mid
      ? {
          id: 'stop-1',
          lat: mid.lat,
          lon: mid.lon,
          label: mid.label || 'Stop #1',
        }
      : null,
  };
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
} as const;
const TUTORIAL_FORCE_ONLY_ACTIONS = new Set([
  'map.confirm_origin_newcastle',
  'map.confirm_destination_london',
  'map.confirm_drag_destination_marker',
  'map.confirm_drag_origin_marker',
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
  const [apiToken, setApiToken] = useState('');
  const [advancedParams, setAdvancedParams] = useState<ScenarioAdvancedParams>(DEFAULT_ADVANCED_PARAMS);
  const [advancedError, setAdvancedError] = useState<string | null>(null);
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
  const tutorialLockNoticeAtRef = useRef(0);
  const tutorialConfirmedOriginRef = useRef<LatLng | null>(null);
  const tutorialConfirmedDestinationRef = useRef<LatLng | null>(null);
  const tutorialMidpointLogSeqRef = useRef(0);
  const tutorialMidpointLogStartRef = useRef(
    typeof performance !== 'undefined' ? performance.now() : 0,
  );
  const midpointFitLogSeqRef = useRef(0);
  const midpointFitLogStartRef = useRef(
    typeof performance !== 'undefined' ? performance.now() : 0,
  );
  const logTutorialMidpoint = useCallback(
    (event: string, payload?: Record<string, unknown>) => {
      if (typeof window === 'undefined') return;
      tutorialMidpointLogSeqRef.current += 1;
      const now = typeof performance !== 'undefined' ? performance.now() : 0;
      const elapsed = (now - tutorialMidpointLogStartRef.current).toFixed(1);
      console.log(
        `[tutorial-midpoint-debug][${tutorialMidpointLogSeqRef.current}] +${elapsed}ms ${event}`,
        payload ?? {},
      );
    },
    [],
  );
  const logMidpointFitFlow = useCallback(
    (event: string, payload?: Record<string, unknown>) => {
      if (typeof window === 'undefined') return;
      midpointFitLogSeqRef.current += 1;
      const now = typeof performance !== 'undefined' ? performance.now() : 0;
      const elapsed = (now - midpointFitLogStartRef.current).toFixed(1);
      console.log(
        `[midpoint-fit-flow][${midpointFitLogSeqRef.current}] +${elapsed}ms ${event}`,
        payload ?? {},
      );
    },
    [],
  );
  const authHeaders = useMemo(() => {
    const token = apiToken.trim();
    return token ? ({ 'x-api-token': token } as Record<string, string>) : undefined;
  }, [apiToken]);

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
  const tutorialLockScope = useMemo(() => inferTutorialLockScope(tutorialStep), [tutorialStep]);
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
        return {
          isOpen: false,
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
    [tutorialActiveSectionId, tutorialLockScope, tutorialRunning],
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
    (tutorialLockScope === 'sidebar_section_only' || tutorialStep?.id === 'map_stop_lifecycle');
  const tutorialViewportLocked =
    tutorialRunning &&
    (tutorialStep?.id === 'map_set_pins' || tutorialStep?.id === 'map_stop_lifecycle');
  const tutorialHideZoomControls =
    tutorialRunning &&
    (tutorialLockScope === 'map_only' || tutorialStep?.id === 'map_stop_lifecycle');
  const tutorialRelaxBounds =
    tutorialRunning &&
    tutorialStep?.id === 'map_stop_lifecycle';
  const tutorialMapDimmed =
    tutorialRunning &&
    tutorialLockScope === 'sidebar_section_only' &&
    tutorialStep?.id !== 'map_stop_lifecycle';
  useEffect(() => {
    if (!tutorialRunning) return;
    if (tutorialStep?.id !== 'map_stop_lifecycle') return;
    logTutorialMidpoint('midpoint-dim-state', {
      tutorialStepId: tutorialStep?.id ?? null,
      tutorialLockScope,
      tutorialMapDimmed,
      tutorialMapLocked,
      tutorialViewportLocked,
      tutorialHideZoomControls,
      tutorialPlacementStage,
      tutorialSidebarLocked,
      tutorialActiveSectionId,
      tutorialBlockingActionId,
      tutorialNextRequiredActionId,
      tutorialMode,
      tutorialOpen,
    });
  }, [
    logTutorialMidpoint,
    tutorialActiveSectionId,
    tutorialBlockingActionId,
    tutorialHideZoomControls,
    tutorialLockScope,
    tutorialMapDimmed,
    tutorialMapLocked,
    tutorialMode,
    tutorialNextRequiredActionId,
    tutorialOpen,
    tutorialRunning,
    tutorialSidebarLocked,
    tutorialStep?.id,
    tutorialPlacementStage,
    tutorialViewportLocked,
  ]);
  useEffect(() => {
    if (!tutorialRunning) return;
    if (tutorialStep?.id !== 'map_stop_lifecycle') return;
    if (typeof window === 'undefined') return;

    let raf = 0;
    let t1 = 0;
    let t2 = 0;

    const probe = (phase: string) => {
      const mapStage = document.querySelector<HTMLElement>('.mapStage');
      const mapPane = document.querySelector<HTMLElement>('.mapPane');
      const overlay = document.querySelector<HTMLElement>('.tutorialOverlay');
      const backdrop = overlay?.querySelector<HTMLElement>('.tutorialOverlay__backdrop') ?? null;
      const spotlight = overlay?.querySelector<HTMLElement>('.tutorialOverlay__spotlight') ?? null;
      const panel = document.querySelector<HTMLElement>('.panel');
      const mapStageAfter = mapStage ? window.getComputedStyle(mapStage, '::after') : null;
      const mapPaneStyle = mapPane ? window.getComputedStyle(mapPane) : null;
      const backdropStyle = backdrop ? window.getComputedStyle(backdrop) : null;
      const overlayStyle = overlay ? window.getComputedStyle(overlay) : null;
      const panelStyle = panel ? window.getComputedStyle(panel) : null;

      logTutorialMidpoint(`midpoint-dim-dom-probe:${phase}`, {
        mapStageClass: mapStage?.className ?? null,
        mapPaneClass: mapPane?.className ?? null,
        overlayClass: overlay?.className ?? null,
        panelClass: panel?.className ?? null,
        hasSpotlight: Boolean(spotlight),
        spotlightDisplay: spotlight ? window.getComputedStyle(spotlight).display : null,
        mapStageAfterBackground: mapStageAfter?.backgroundColor ?? null,
        mapStageAfterContent: mapStageAfter?.content ?? null,
        mapPaneFilter: mapPaneStyle?.filter ?? null,
        mapPaneOpacity: mapPaneStyle?.opacity ?? null,
        backdropBackground: backdropStyle?.backgroundColor ?? null,
        backdropOpacity: backdropStyle?.opacity ?? null,
        backdropDisplay: backdropStyle?.display ?? null,
        overlayPointerEvents: overlayStyle?.pointerEvents ?? null,
        panelFilter: panelStyle?.filter ?? null,
      });
    };

    probe('effect-start');
    raf = window.requestAnimationFrame(() => probe('raf'));
    t1 = window.setTimeout(() => probe('timeout-120'), 120);
    t2 = window.setTimeout(() => probe('timeout-450'), 450);

    return () => {
      window.cancelAnimationFrame(raf);
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [
    logTutorialMidpoint,
    tutorialMapDimmed,
    tutorialMapLocked,
    tutorialOpen,
    tutorialRunning,
    tutorialStep?.id,
    tutorialViewportLocked,
  ]);
  const tutorialGuideTarget = useMemo<TutorialGuideTarget | null>(() => {
    if (!tutorialRunning || tutorialStep?.id !== 'map_set_pins') return null;
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
        const canonicalDutyText = serializePinsToDutyText(stableOrigin, managedStop, stableDestination);
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

  const clearFlushTimer = useCallback(() => {
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }, []);

  const markTutorialAction = useCallback(
    (actionId: string, options?: { force?: boolean }) => {
      if (!actionId) return;
      if (!tutorialRunning || !tutorialStepId) {
        return;
      }
      if (TUTORIAL_FORCE_ONLY_ACTIONS.has(actionId) && !options?.force) {
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
      tutorialRunning,
      tutorialStepId,
    ],
  );

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
        setVehicleType('rigid_hgv');
        setScenarioMode('no_sharing');
      }

      if (prefillId === 'canonical_advanced') {
        setAdvancedParams((prev) => ({
          ...prev,
          optimizationMode: 'robust',
          riskAversion: '1.4',
          paretoMethod: 'epsilon_constraint',
          epsilonDurationS: '9000',
          epsilonMonetaryCost: '250',
          epsilonEmissionsKg: '900',
          departureTimeUtcLocal: nextUtcHourLocalInput(),
          stochasticEnabled: true,
          stochasticSeed: '42',
          stochasticSigma: '0.08',
          stochasticSamples: '25',
          terrainProfile: 'rolling',
          useTolls: true,
          fuelPriceMultiplier: '1.1',
          carbonPricePerKg: '0.08',
          tollCostPerKm: '0.12',
        }));
      }

      if (prefillId === 'canonical_preferences') {
        setWeights({ time: 55, money: 25, co2: 20 });
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
        const resp = await fetch('/api/vehicles', {
          headers: authHeaders,
          signal: controller.signal,
          cache: 'no-store',
        });
        if (!resp.ok) return;

        const payload = (await resp.json()) as Partial<VehicleListResponse>;
        setVehicles(Array.isArray(payload.vehicles) ? payload.vehicles : []);
      } catch (e) {
        if (!controller.signal.aborted) {
          console.error('Failed to load vehicles:', e);
        }
      }
    })();

    return () => controller.abort();
  }, []);

  useEffect(() => {
    void Promise.all([loadExperiments(), refreshOracleDashboard()]);
  }, [authHeaders]);

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

    const nextText = serializePinsToDutyText(origin, managedStop, destination);
    if (nextText === dutyStopsText) return;
    logTutorialMidpoint('duty-sync:pins-to-text-write', {
      nextText,
      previousText: dutyStopsText,
      origin,
      destination,
      managedStop,
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
    const canonicalText = serializePinsToDutyText(parsed.origin, parsed.stop, parsed.destination);
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
    if (tutorialLockScope === 'sidebar_section_only' && isPanelCollapsed) {
      setIsPanelCollapsed(false);
    }
  }, [isPanelCollapsed, tutorialLockScope, tutorialRunning]);

  useEffect(() => {
    if (tutorialBootstrappedRef.current) return;
    tutorialBootstrappedRef.current = true;
    let savedProgress = loadTutorialProgress(TUTORIAL_PROGRESS_KEY);
    if (!savedProgress) {
      for (const legacyKey of LEGACY_TUTORIAL_PROGRESS_KEYS) {
        const legacyProgress = loadTutorialProgress(legacyKey);
        if (!legacyProgress) continue;
        savedProgress = legacyProgress;
        saveTutorialProgress(TUTORIAL_PROGRESS_KEY, legacyProgress);
        break;
      }
    }

    let isCompleted = loadTutorialCompleted(TUTORIAL_COMPLETED_KEY);
    if (!isCompleted) {
      for (const legacyCompletedKey of LEGACY_TUTORIAL_COMPLETED_KEYS) {
        if (!loadTutorialCompleted(legacyCompletedKey)) continue;
        isCompleted = true;
        saveTutorialCompleted(TUTORIAL_COMPLETED_KEY, true);
        break;
      }
    }

    for (const legacyKey of LEGACY_TUTORIAL_PROGRESS_KEYS) {
      clearTutorialProgress(legacyKey);
    }
    for (const legacyCompletedKey of LEGACY_TUTORIAL_COMPLETED_KEYS) {
      saveTutorialCompleted(legacyCompletedKey, false);
    }

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
          !isTutorialActionAllowed(actionId, tutorialAllowedActionExact, tutorialAllowedActionPrefixes) &&
          !(isTargetAllowedByStep(target) && !tutorialUsesSectionTarget)
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

        if (isTargetAllowedByStep(target) && !tutorialUsesSectionTarget) {
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

  const dutyStopsForOverlay = useMemo(
    () =>
      managedStop
        ? [
            {
              lat: managedStop.lat,
              lon: managedStop.lon,
              label: managedStop.label,
            },
          ]
        : [],
    [managedStop],
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

  function handleMoveStop(lat: number, lon: number) {
    setError(null);
    setManagedStop((prev) => (prev ? { ...prev, lat, lon } : prev));
    clearComputed();
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
    if (managedStop) {
      logTutorialMidpoint('add-stop:replace-confirm-required', {
        currentStop: managedStop,
      });
      const shouldReplace = window.confirm(
        'A stop already exists in one-stop mode. Replace it with a new midpoint stop?',
      );
      logTutorialMidpoint('add-stop:replace-confirm-result', {
        shouldReplace,
      });
      if (!shouldReplace) {
        logTutorialMidpoint('add-stop:exit-replace-cancelled');
        return;
      }
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
    const nextLabel = managedStop?.label?.trim() || 'Stop #1';
    const midpoint = {
      id: 'stop-1' as const,
      lat: (baseOrigin.lat + baseDestination.lat) / 2,
      lon: (baseOrigin.lon + baseDestination.lon) / 2,
      label: nextLabel,
    };
    logTutorialMidpoint('add-stop:computed-midpoint', midpoint);
    setManagedStop({
      ...midpoint,
    });
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
    if (!managedStop) return;
    setManagedStop(null);
    setSelectedPinId((prev) =>
      normalizeSelectedPinId(prev, {
        origin,
        destination,
        stop: null,
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
    setApiToken('');
    setAdvancedParams(DEFAULT_ADVANCED_PARAMS);
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
    pareto_method?: ParetoMethod;
    epsilon?: EpsilonConstraints;
    departure_time_utc?: string;
    cost_toggles?: CostToggles;
    terrain_profile?: TerrainProfile;
    stochastic?: StochasticConfig;
    optimization_mode?: OptimizationMode;
    risk_aversion?: number;
  } {
    const patch: {
      pareto_method?: ParetoMethod;
      epsilon?: EpsilonConstraints;
      departure_time_utc?: string;
      cost_toggles?: CostToggles;
      terrain_profile?: TerrainProfile;
      stochastic?: StochasticConfig;
      optimization_mode?: OptimizationMode;
      risk_aversion?: number;
    } = {};

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

    return patch;
  }

  function buildScenarioCompareRequest(originPoint: LatLng, destinationPoint: LatLng): ScenarioCompareRequest {
    const advancedPatch = buildAdvancedRequestPatch();
    return {
      origin: originPoint,
      destination: destinationPoint,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: 5,
      weights: {
        time: weights.time,
        money: weights.money,
        co2: weights.co2,
      },
      ...advancedPatch,
    };
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

    let advancedPatch: ReturnType<typeof buildAdvancedRequestPatch>;
    try {
      advancedPatch = buildAdvancedRequestPatch();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Invalid advanced parameter values.';
      setAdvancedError(msg);
      setError(msg);
      setLoading(false);
      abortRef.current = null;
      return;
    }

    const body = {
      origin,
      destination,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: 5,
      ...advancedPatch,
    };

    let sawDone = false;

    try {
      await postNDJSON<ParetoStreamEvent>('/api/pareto/stream', body, {
        signal: controller.signal,
        headers: authHeaders,
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
        authHeaders,
      );
      setScenarioCompare(payload);
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
    setAdvancedParams({
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
    });
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

      const payload = await fetch(path, {
        cache: 'no-store',
        headers: authHeaders,
      });
      if (!payload.ok) {
        throw new Error(await payload.text());
      }
      const parsed = (await payload.json()) as ExperimentListResponse;
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
        authHeaders,
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
      const resp = await fetch(`/api/experiments/${experimentId}`, {
        method: 'DELETE',
        cache: 'no-store',
        headers: authHeaders,
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      await loadExperiments();
    } catch (e: unknown) {
      setExperimentsError(e instanceof Error ? e.message : 'Failed to delete experiment');
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
        authHeaders,
      );
      setScenarioCompare(payload);
    } catch (e: unknown) {
      setScenarioCompareError(e instanceof Error ? e.message : 'Failed to replay experiment');
    } finally {
      setScenarioCompareLoading(false);
    }
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

    const out: DutyChainRequest['stops'] = [
      {
        lat: parsed.origin.lat,
        lon: parsed.origin.lon,
        label: 'Start',
      },
    ];
    if (parsed.stop) {
      out.push({
        lat: parsed.stop.lat,
        lon: parsed.stop.lon,
        label: parsed.stop.label || 'Stop #1',
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
    let advancedPatch: ReturnType<typeof buildAdvancedRequestPatch>;
    let stops: DutyChainRequest['stops'];
    try {
      setAdvancedError(null);
      setDutyChainError(null);
      advancedPatch = buildAdvancedRequestPatch();
      stops = parseDutyStops(dutyStopsText);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Invalid duty chain request.';
      setAdvancedError(msg);
      setDutyChainError(msg);
      return;
    }

    const req: DutyChainRequest = {
      stops,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: 5,
      weights: {
        time: weights.time,
        money: weights.money,
        co2: weights.co2,
      },
      ...(advancedPatch.cost_toggles ? { cost_toggles: advancedPatch.cost_toggles } : {}),
      ...(advancedPatch.terrain_profile ? { terrain_profile: advancedPatch.terrain_profile } : {}),
      ...(advancedPatch.stochastic ? { stochastic: advancedPatch.stochastic } : {}),
      ...(advancedPatch.optimization_mode ? { optimization_mode: advancedPatch.optimization_mode } : {}),
      ...(advancedPatch.risk_aversion !== undefined ? { risk_aversion: advancedPatch.risk_aversion } : {}),
      ...(advancedPatch.departure_time_utc ? { departure_time_utc: advancedPatch.departure_time_utc } : {}),
      ...(advancedPatch.pareto_method ? { pareto_method: advancedPatch.pareto_method } : {}),
      ...(advancedPatch.epsilon ? { epsilon: advancedPatch.epsilon } : {}),
    };

    setDutyChainLoading(true);
    try {
      markTutorialAction('duty.run_click');
      const payload = await postJSON<DutyChainResponse>(
        '/api/duty/chain',
        req,
        undefined,
        authHeaders,
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
        headers: authHeaders,
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
        authHeaders,
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

    let advancedPatch: ReturnType<typeof buildAdvancedRequestPatch>;
    try {
      setAdvancedError(null);
      advancedPatch = buildAdvancedRequestPatch();
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

    const req: DepartureOptimizeRequest = {
      origin,
      destination,
      vehicle_type: vehicleType,
      scenario_mode: scenarioMode,
      max_alternatives: 5,
      weights: {
        time: weights.time,
        money: weights.money,
        co2: weights.co2,
      },
      window_start_utc: startDate.toISOString(),
      window_end_utc: endDate.toISOString(),
      step_minutes: depStepMinutes,
      ...(advancedPatch.cost_toggles ? { cost_toggles: advancedPatch.cost_toggles } : {}),
      ...(advancedPatch.terrain_profile ? { terrain_profile: advancedPatch.terrain_profile } : {}),
      ...(advancedPatch.stochastic ? { stochastic: advancedPatch.stochastic } : {}),
      ...(advancedPatch.optimization_mode ? { optimization_mode: advancedPatch.optimization_mode } : {}),
      ...(advancedPatch.risk_aversion !== undefined ? { risk_aversion: advancedPatch.risk_aversion } : {}),
      ...(advancedPatch.pareto_method ? { pareto_method: advancedPatch.pareto_method } : {}),
      ...(advancedPatch.epsilon ? { epsilon: advancedPatch.epsilon } : {}),
      ...(timeWindow ? { time_window: timeWindow } : {}),
    };

    setDepOptimizeLoading(true);
    setDepOptimizeError(null);
    try {
      markTutorialAction('dep.optimize_click');
      const payload = await postJSON<DepartureOptimizeResponse>(
        '/api/departure/optimize',
        req,
        undefined,
        authHeaders,
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

  const scenarioOptions: { value: ScenarioMode; label: string; description: string }[] = [
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
                  value={vehicleType}
                  options={vehicleOptions}
                  onChange={(next) => {
                    setVehicleType(next);
                    markTutorialAction('setup.vehicle_select');
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
                  value={scenarioMode}
                  options={scenarioOptions}
                  onChange={(next) => {
                    setScenarioMode(next);
                    markTutorialAction('setup.scenario_select');
                  }}
                  disabled={busy}
                  tutorialId="setup.scenario"
                  tutorialActionPrefix="setup.scenario_option"
                />

                <div className="fieldLabelRow">
                  <div className="fieldLabel">API token (optional)</div>
                  <FieldInfo text={SIDEBAR_FIELD_HELP.apiToken} />
                </div>
                <input
                  className="input"
                  type="password"
                  placeholder="x-api-token for RBAC-enabled backends"
                  value={apiToken}
                  onChange={(event) => setApiToken(event.target.value)}
                  data-tutorial-id="setup.api_token"
                  data-tutorial-action="setup.api_token_input"
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
                    Sharing Mode Is Active. Current Scenario Logic Applies Policy-Based Delay And
                    Idle Emissions As A Lightweight Simulation Stub.
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
                      <li key={`${idx}-${warning}`}>{warning}</li>
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
              applyScenarioRequestToState(bundle.request);
              clearComputed();
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

