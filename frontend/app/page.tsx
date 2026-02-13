'use client';
// frontend/app/page.tsx

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from 'react';

import FieldInfo from './components/FieldInfo';
import Select from './components/Select';
import { postJSON, postNDJSON } from './lib/api';
import { formatNumber } from './lib/format';
import { LOCALE_OPTIONS, createTranslator, type Locale } from './lib/i18n';
import { buildManagedPinNodes, buildStopOverlayPoints } from './lib/mapOverlays';
import {
  SIDEBAR_DROPDOWN_OPTIONS_HELP,
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
import type { TutorialProgress, TutorialStep, TutorialTargetRect } from './lib/tutorial/types';
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
  RouteOption,
  ScenarioMode,
  ScenarioCompareResponse,
  ScenarioCompareRequest,
  StochasticConfig,
  TerrainProfile,
  TimeWindowConstraints,
  VehicleListResponse,
  VehicleProfile,
} from './lib/types';
import { normaliseWeights, pickBestByWeightedSum, type WeightState } from './lib/weights';
import type { ScenarioAdvancedParams } from './components/ScenarioParameterEditor';

type MarkerKind = 'origin' | 'destination';
type ProgressState = { done: number; total: number };

type MapViewProps = {
  origin: LatLng | null;
  destination: LatLng | null;
  managedStop?: ManagedStop | null;
  originLabel?: string;
  destinationLabel?: string;

  selectedPinId?: 'origin' | 'destination' | 'stop-1' | null;

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

  onMapClick: (lat: number, lon: number) => void;
  onSelectPinId?: (id: 'origin' | 'destination' | 'stop-1' | null) => void;
  onMoveMarker: (kind: MarkerKind, lat: number, lon: number) => void;
  onAddStopFromPin?: (kind: MarkerKind) => void;
  onRenameStop?: (name: string) => void;
  onDeleteStop?: () => void;
  onFocusPin?: (id: 'origin' | 'destination' | 'stop-1') => void;
  onSwapMarkers?: () => void;
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
  loading: () => null,
});

const EtaTimelineChart = dynamic<{ route: RouteOption | null }>(
  () => import('./components/EtaTimelineChart'),
  {
    ssr: false,
    loading: () => null,
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
  loading: () => null,
});

const SegmentBreakdown = dynamic<{ route: RouteOption | null; onTutorialAction?: (actionId: string) => void }>(
  () => import('./components/SegmentBreakdown'),
  {
    ssr: false,
    loading: () => null,
  },
);

const ScenarioParameterEditor = dynamic<
  { value: ScenarioAdvancedParams; onChange: (next: ScenarioAdvancedParams) => void; disabled: boolean; validationError: string | null }
>(() => import('./components/ScenarioParameterEditor'), {
  ssr: false,
  loading: () => null,
});

const ExperimentManager = dynamic<
  {
    experiments: ExperimentBundle[];
    loading: boolean;
    error: string | null;
    canSave: boolean;
    disabled: boolean;
    onRefresh: () => void;
    onSave: (name: string, description: string) => Promise<void> | void;
    onLoad: (bundle: ExperimentBundle) => void;
    onDelete: (experimentId: string) => Promise<void> | void;
    onReplay: (experimentId: string) => Promise<void> | void;
    catalogQuery: string;
    catalogVehicleType: string;
    catalogScenarioMode: '' | ScenarioMode;
    catalogSort: ExperimentCatalogSort;
    vehicleOptions: Array<{ value: string; label: string }>;
    onCatalogQueryChange: (value: string) => void;
    onCatalogVehicleTypeChange: (value: string) => void;
    onCatalogScenarioModeChange: (value: '' | ScenarioMode) => void;
    onCatalogSortChange: (value: ExperimentCatalogSort) => void;
    onApplyCatalogFilters: () => void;
    locale: Locale;
    defaultName?: string;
    defaultDescription?: string;
    tutorialResetNonce?: number;
  }
>(() => import('./components/ExperimentManager'), {
  ssr: false,
  loading: () => null,
});

const DepartureOptimizerChart = dynamic<
  {
    windowStartLocal: string;
    windowEndLocal: string;
    earliestArrivalLocal: string;
    latestArrivalLocal: string;
    stepMinutes: number;
    loading: boolean;
    error: string | null;
    data: DepartureOptimizeResponse | null;
    disabled: boolean;
    onWindowStartChange: (value: string) => void;
    onWindowEndChange: (value: string) => void;
    onEarliestArrivalChange: (value: string) => void;
    onLatestArrivalChange: (value: string) => void;
    onStepMinutesChange: (value: number) => void;
    onRun: () => void;
    onApplyDepartureTime: (isoUtc: string) => void;
    locale: Locale;
  }
>(() => import('./components/DepartureOptimizerChart'), {
  ssr: false,
  loading: () => null,
});

const CounterfactualPanel = dynamic<{ route: RouteOption | null }>(
  () => import('./components/CounterfactualPanel'),
  {
    ssr: false,
    loading: () => null,
  },
);

const ScenarioTimeLapse = dynamic<
  { route: RouteOption | null; onPositionChange: (position: LatLng | null) => void }
>(() => import('./components/ScenarioTimeLapse'), {
  ssr: false,
  loading: () => null,
});

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
    optionalDecision: {
      id: string;
      label: string;
      resolved: boolean;
      defaultLabel: string;
      actionTouched: boolean;
    } | null;
    targetRect: TutorialTargetRect | null;
    targetMissing: boolean;
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

const DutyChainPlanner = dynamic<
  {
    stopsText: string;
    onStopsTextChange: (value: string) => void;
    onRun: () => void;
    loading: boolean;
    error: string | null;
    data: DutyChainResponse | null;
    disabled: boolean;
    locale: Locale;
  }
>(() => import('./components/DutyChainPlanner'), {
  ssr: false,
  loading: () => null,
});

const PinManager = dynamic<
  {
    nodes: import('./lib/types').PinDisplayNode[];
    selectedPinId: 'origin' | 'destination' | 'stop-1' | null;
    disabled: boolean;
    hasStop: boolean;
    canAddStop: boolean;
    onSelectPin: (id: 'origin' | 'destination' | 'stop-1') => void;
    onRenameStart: (name: string) => void;
    onRenameDestination: (name: string) => void;
    onRenameStop: (name: string) => void;
    onAddStop: () => void;
    onDeleteStop: () => void;
    onSwapPins: () => void;
    onClearPins: () => void;
  }
>(() => import('./components/PinManager'), {
  ssr: false,
  loading: () => null,
});

const OracleQualityDashboard = dynamic<
  {
    dashboard: OracleQualityDashboardResponse | null;
    loading: boolean;
    ingesting: boolean;
    error: string | null;
    latestCheck: OracleFeedCheckRecord | null;
    disabled: boolean;
    onRefresh: () => void;
    onIngest: (payload: OracleFeedCheckInput) => Promise<void> | void;
    locale: Locale;
    tutorialResetNonce?: number;
  }
>(() => import('./components/OracleQualityDashboard'), {
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
  return Math.abs(a.lat - b.lat) <= 1e-9 && Math.abs(a.lon - b.lon) <= 1e-9;
}

function sameManagedStop(a: ManagedStop | null, b: ManagedStop | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.id === b.id &&
    Math.abs(a.lat - b.lat) <= 1e-9 &&
    Math.abs(a.lon - b.lon) <= 1e-9 &&
    (a.label ?? '') === (b.label ?? '')
  );
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
  labels: {
    startLabel: string;
    destinationLabel: string;
  },
): string {
  const lines: string[] = [];
  if (origin) {
    lines.push(toDutyLine(origin.lat, origin.lon, labels.startLabel || 'Start'));
  }
  if (stop) {
    lines.push(toDutyLine(stop.lat, stop.lon, stop.label || 'Stop #1'));
  }
  if (destination) {
    lines.push(toDutyLine(destination.lat, destination.lon, labels.destinationLabel || 'Destination'));
  }
  return lines.join('\n');
}

type ParsedPinSync =
  | {
      ok: true;
      origin: LatLng | null;
      destination: LatLng | null;
      stop: ManagedStop | null;
      startLabel: string;
      destinationLabel: string;
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
      startLabel: 'Start',
      destinationLabel: 'Destination',
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
      startLabel: first.label || 'Start',
      destinationLabel: 'Destination',
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
    startLabel: first.label || 'Start',
    destinationLabel: last.label || 'Destination',
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

export default function Page() {
  const [origin, setOrigin] = useState<LatLng | null>(null);
  const [destination, setDestination] = useState<LatLng | null>(null);
  const [managedStop, setManagedStop] = useState<ManagedStop | null>(null);
  const [startLabel, setStartLabel] = useState('Start');
  const [destinationLabel, setDestinationLabel] = useState('Destination');
  const [selectedPinId, setSelectedPinId] = useState<'origin' | 'destination' | 'stop-1' | null>(null);
  const [isPanelCollapsed, setIsPanelCollapsed] = useState(false);
  const [locale, setLocale] = useState<Locale>('en');

  const [vehicles, setVehicles] = useState<VehicleProfile[]>([]);
  const [vehicleType, setVehicleType] = useState<string>('rigid_hgv');
  const [scenarioMode, setScenarioMode] = useState<ScenarioMode>('no_sharing');

  const [weights, setWeights] = useState<WeightState>({ time: 60, money: 20, co2: 20 });
  const [apiToken, setApiToken] = useState('');
  const [advancedParams, setAdvancedParams] = useState<ScenarioAdvancedParams>(DEFAULT_ADVANCED_PARAMS);
  const [advancedError, setAdvancedError] = useState<string | null>(null);

  const [paretoRoutes, setParetoRoutes] = useState<RouteOption[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [routeNames, setRouteNames] = useState<Record<string, string>>({});
  const [editingRouteId, setEditingRouteId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [showWarnings, setShowWarnings] = useState(false);
  const [scenarioCompare, setScenarioCompare] = useState<ScenarioCompareResponse | null>(null);
  const [scenarioCompareLoading, setScenarioCompareLoading] = useState(false);
  const [scenarioCompareError, setScenarioCompareError] = useState<string | null>(null);
  const [experiments, setExperiments] = useState<ExperimentBundle[]>([]);
  const [experimentsLoading, setExperimentsLoading] = useState(false);
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
  const [oracleDashboardLoading, setOracleDashboardLoading] = useState(false);
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
  const authHeaders = useMemo(() => {
    const token = apiToken.trim();
    return token ? ({ 'x-api-token': token } as Record<string, string>) : undefined;
  }, [apiToken]);
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
  const tutorialCanAdvance = useMemo(() => {
    if (!tutorialStep) return false;
    const requiredDone = tutorialStep.required.every((item) => tutorialActionSet.has(item.actionId));
    if (!requiredDone) return false;
    if (!tutorialOptionalState) return true;
    return tutorialOptionalState.resolved || tutorialOptionalState.actionTouched;
  }, [tutorialActionSet, tutorialOptionalState, tutorialStep]);
  const tutorialAtStart = tutorialStepIndex <= 0;
  const tutorialAtEnd = tutorialStepIndex >= TUTORIAL_STEPS.length - 1;

  const clearFlushTimer = useCallback(() => {
    if (flushTimerRef.current !== null) {
      window.clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }, []);

  const markTutorialAction = useCallback(
    (actionId: string) => {
      if (!actionId || !tutorialRunning || !tutorialStepId) return;
      setTutorialStepActionsById((prev) => {
        const existing = prev[tutorialStepId] ?? [];
        if (existing.includes(actionId)) return prev;
        return { ...prev, [tutorialStepId]: [...existing, actionId] };
      });
    },
    [tutorialRunning, tutorialStepId],
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
        setOrigin(null);
        setDestination(null);
        setManagedStop(null);
        setStartLabel('Start');
        setDestinationLabel('Destination');
        setSelectedPinId(null);
      }

      if (prefillId === 'canonical_map') {
        setOrigin(TUTORIAL_CANONICAL_ORIGIN);
        setDestination(TUTORIAL_CANONICAL_DESTINATION);
        setManagedStop(null);
        setStartLabel('Start');
        setDestinationLabel('Destination');
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
    void loadExperiments();
  }, [authHeaders]);

  useEffect(() => {
    void refreshOracleDashboard();
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
    if (dutySyncSourceRef.current === 'text') {
      dutySyncSourceRef.current = null;
      return;
    }

    const nextText = serializePinsToDutyText(origin, managedStop, destination, {
      startLabel,
      destinationLabel,
    });
    if (nextText === dutyStopsText) return;
    dutySyncSourceRef.current = 'pins';
    setDutyStopsText(nextText);
    setDutySyncError(null);
  }, [origin, destination, managedStop, startLabel, destinationLabel, dutyStopsText]);

  useEffect(() => {
    if (dutySyncSourceRef.current === 'pins') {
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
    dutySyncSourceRef.current = 'text';
    if (!sameLatLng(origin, parsed.origin)) {
      setOrigin(parsed.origin);
    }
    if (!sameLatLng(destination, parsed.destination)) {
      setDestination(parsed.destination);
    }
    if (!sameManagedStop(managedStop, parsed.stop)) {
      setManagedStop(parsed.stop);
    }
    if (selectedPinId === 'origin' && !parsed.origin) {
      setSelectedPinId(null);
    } else if (selectedPinId === 'destination' && !parsed.destination) {
      setSelectedPinId(null);
    } else if (selectedPinId === 'stop-1' && !parsed.stop) {
      setSelectedPinId(null);
    }
    if (startLabel !== parsed.startLabel) {
      setStartLabel(parsed.startLabel);
    }
    if (destinationLabel !== parsed.destinationLabel) {
      setDestinationLabel(parsed.destinationLabel);
    }
  }, [dutyStopsText, origin, destination, managedStop, startLabel, destinationLabel, selectedPinId]);

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
    if (tutorialBootstrappedRef.current) return;
    tutorialBootstrappedRef.current = true;
    const savedProgress = loadTutorialProgress(TUTORIAL_PROGRESS_KEY);
    const isCompleted = loadTutorialCompleted(TUTORIAL_COMPLETED_KEY);
    setTutorialSavedProgress(savedProgress);
    setTutorialCompleted(isCompleted);
    if (!isCompleted) {
      setTutorialMode(savedProgress ? 'chooser' : tutorialIsDesktop ? 'running' : 'blocked');
      setTutorialOpen(true);
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
    const handleActionEvent = (event: Event) => {
      const target = event.target as HTMLElement | null;
      const actionable = target?.closest<HTMLElement>('[data-tutorial-action]');
      const actionId = actionable?.dataset.tutorialAction;
      if (actionId) {
        markTutorialAction(actionId);
      }
    };

    document.addEventListener('click', handleActionEvent, true);
    document.addEventListener('change', handleActionEvent, true);
    document.addEventListener('input', handleActionEvent, true);
    return () => {
      document.removeEventListener('click', handleActionEvent, true);
      document.removeEventListener('change', handleActionEvent, true);
      document.removeEventListener('input', handleActionEvent, true);
    };
  }, [markTutorialAction, tutorialRunning]);

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

  useEffect(() => {
    if (!tutorialRunning || !tutorialStep) return;

    let raf = 0;
    let timeoutId = 0;
    let intervalId = 0;
    const resolveTarget = (scrollIntoViewTarget: boolean) => {
      if (!tutorialStep.targetIds.length) {
        setTutorialTargetRect(null);
        setTutorialTargetMissing(false);
        return;
      }
      const element = tutorialStep.targetIds
        .map((targetId) =>
          document.querySelector<HTMLElement>(`[data-tutorial-id=\"${targetId}\"]`),
        )
        .find((candidate) => Boolean(candidate));

      if (!element) {
        setTutorialTargetRect(null);
        setTutorialTargetMissing(true);
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
        setTutorialTargetRect(null);
        setTutorialTargetMissing(true);
        return;
      }
      setTutorialTargetRect({
        top: rect.top,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      });
      setTutorialTargetMissing(false);
    };

    timeoutId = window.setTimeout(() => {
      resolveTarget(true);
      raf = window.requestAnimationFrame(() => resolveTarget(false));
    }, 120);
    intervalId = window.setInterval(() => resolveTarget(false), 450);
    const onResize = () => resolveTarget(false);
    const onScroll = () => resolveTarget(false);
    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onScroll, true);
    return () => {
      window.clearTimeout(timeoutId);
      window.clearInterval(intervalId);
      window.cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onScroll, true);
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
    () =>
      buildManagedPinNodes(origin, destination, managedStop, {
        origin: startLabel,
        destination: destinationLabel,
      }),
    [origin, destination, managedStop, startLabel, destinationLabel],
  );
  const canAddStop = Boolean(origin && destination);
  const stopOverlayCount = useMemo(
    () => buildStopOverlayPoints(origin, destination, dutyStopsForOverlay).length,
    [origin, destination, dutyStopsForOverlay],
  );
  const incidentOverlayCount = selectedRoute?.incident_events?.length ?? 0;
  const segmentOverlayCount = Math.min(120, selectedRoute?.segment_breakdown?.length ?? 0);
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

  const selectedLabel = selectedRoute ? labelsById[selectedRoute.id] ?? selectedRoute.id : null;

  const busy = loading || isPending;
  const canCompute = Boolean(origin && destination) && !busy;

  const progressText = progress
    ? `${formatNumber(Math.min(progress.done, progress.total), locale, {
        maximumFractionDigits: 0,
      })}/${formatNumber(progress.total, locale, { maximumFractionDigits: 0 })}`
    : null;
  const progressPct =
    progress && progress.total > 0
      ? Math.max(0, Math.min(100, (progress.done / progress.total) * 100))
      : 0;
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

    if (selectedPinId === 'origin') {
      setOrigin({ lat, lon });
      clearComputed();
      markTutorialAction('map.set_origin');
      return;
    }

    setDestination({ lat, lon });
    setSelectedPinId('destination');
    clearComputed();
    markTutorialAction('map.set_destination');
  }

  function handleMoveMarker(kind: MarkerKind, lat: number, lon: number) {
    setError(null);

    if (kind === 'origin') setOrigin({ lat, lon });
    else setDestination({ lat, lon });

    setSelectedPinId(kind);
    clearComputed();
    markTutorialAction('map.drag_marker');
    if (kind === 'origin') {
      markTutorialAction('map.drag_origin_marker');
    } else {
      markTutorialAction('map.drag_destination_marker');
    }
  }

  function addStopFromMidpoint(_kind: MarkerKind) {
    if (!origin || !destination) return;
    if (managedStop) {
      const shouldReplace = window.confirm(
        'A stop already exists in one-stop mode. Replace it with a new midpoint stop?',
      );
      if (!shouldReplace) return;
    }
    const nextLabel = managedStop?.label?.trim() || 'Stop #1';
    setManagedStop({
      id: 'stop-1',
      lat: (origin.lat + destination.lat) / 2,
      lon: (origin.lon + destination.lon) / 2,
      label: nextLabel,
    });
    setSelectedPinId('stop-1');
    clearComputed();
    markTutorialAction('map.add_stop_midpoint');
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
    if (selectedPinId === 'stop-1') {
      setSelectedPinId(null);
    }
    clearComputed();
    markTutorialAction('map.delete_stop');
  }

  function renameStart(name: string) {
    const next = name.trim() || 'Start';
    setStartLabel(next);
    markTutorialAction('pins.rename_start');
  }

  function renameDestination(name: string) {
    const next = name.trim() || 'Destination';
    setDestinationLabel(next);
    markTutorialAction('pins.rename_destination');
  }

  function selectPinFromSidebar(id: 'origin' | 'destination' | 'stop-1') {
    setSelectedPinId(id);
    markTutorialAction('pins.sidebar_select');
  }

  function swapMarkers() {
    if (!origin || !destination) return;
    setOrigin(destination);
    setDestination(origin);
    setSelectedPinId(null);
    clearComputed();
    markTutorialAction('setup.swap_pins_button');
    markTutorialAction('map.popup_swap');
  }

  function reset() {
    setOrigin(null);
    setDestination(null);
    setManagedStop(null);
    setStartLabel('Start');
    setDestinationLabel('Destination');
    setSelectedPinId(null);
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
    setManagedStop(null);
    setStartLabel('Start');
    setDestinationLabel('Destination');
    setSelectedPinId(null);
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
      setError('Click the map to set Start, then Destination.');
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
      setScenarioCompareError('Set origin and destination before comparing scenarios.');
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
      setExperimentsError('Set origin and destination before saving an experiment.');
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
      throw new Error('Duty chain requires both start and destination rows.');
    }

    const out: DutyChainRequest['stops'] = [
      {
        lat: parsed.origin.lat,
        lon: parsed.origin.lon,
        label: parsed.startLabel || 'Start',
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
      label: parsed.destinationLabel || 'Destination',
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
      setDepOptimizeError('Set origin and destination before optimizing departures.');
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

  const mapHint = (() => {
    if (!origin) return t('route_hint_start');
    if (origin && !destination) return t('route_hint_destination');
    if (loading) {
      return progressText
        ? `${t('live_computing')} (${progressText})`
        : t('live_computing');
    }
    return t('route_hint_default');
  })();

  const showRoutesSection = loading || paretoRoutes.length > 0 || warnings.length > 0;
  const canCompareScenarios = Boolean(origin && destination) && !busy && !scenarioCompareLoading;
  const canSaveExperiment = Boolean(origin && destination) && !busy;
  const sidebarToggleLabel = isPanelCollapsed ? 'Extend sidebar' : 'Collapse sidebar';

  return (
    <div className="app">
      <a href="#app-sidebar-content" className="skipLink">
        {t('skip_to_controls')}
      </a>
      <div className="srOnly" role="status" aria-live="polite" aria-atomic="true">
        {liveMessage}
      </div>
      <div className="mapStage" data-tutorial-id="map.interactive">
        <MapView
          origin={origin}
          destination={destination}
          managedStop={managedStop}
          originLabel={startLabel}
          destinationLabel={destinationLabel}
          selectedPinId={selectedPinId}
          route={selectedRoute}
          timeLapsePosition={timeLapsePosition}
          dutyStops={dutyStopsForOverlay}
          showStopOverlay={showStopOverlay}
          showIncidentOverlay={showIncidentOverlay}
          showSegmentTooltips={showSegmentTooltips}
          overlayLabels={mapOverlayLabels}
          onMapClick={handleMapClick}
          onSelectPinId={setSelectedPinId}
          onMoveMarker={handleMoveMarker}
          onAddStopFromPin={addStopFromMidpoint}
          onRenameStop={renameStop}
          onDeleteStop={deleteStop}
          onFocusPin={setSelectedPinId}
          onSwapMarkers={swapMarkers}
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

        <div className="mapHUD">
          {busy && (
            <div className="mapHUD__status" role="status" aria-live="polite">
              <div className="mapHUD__statusLine">
                <span className="spinner" />
                <span>
                  {t('live_computing')}
                  {progressText ? ` (${progressText})` : '...'}
                </span>
              </div>

              {progress?.total ? (
                <div className="hudProgress" aria-hidden="true">
                  <div className="hudProgress__fill" style={{ width: `${progressPct}%` }} />
                </div>
              ) : null}
            </div>
          )}

          <div className="mapHUD__hint">{mapHint}</div>
          <div
            className="mapHUD__overlayCard"
            role="group"
            aria-label={t('map_overlays')}
            data-tutorial-id="map.overlay_controls"
          >
            <div className="mapHUD__overlayTitle">{t('map_overlays')}</div>
            <div className="mapHUD__overlayControls">
              <button
                type="button"
                className={`mapOverlayToggle ${showStopOverlay ? 'isOn' : ''}`}
                onClick={() => {
                  setShowStopOverlay((prev) => !prev);
                  markTutorialAction('map.overlay_stops_toggle');
                }}
                aria-pressed={showStopOverlay}
                aria-label={`${t('overlay_stops')} (${stopOverlayCount})`}
                data-tutorial-action="map.overlay_stops_toggle"
              >
                <span>{t('overlay_stops')}</span>
                <span className="mapOverlayToggle__count">{stopOverlayCount}</span>
              </button>
              <button
                type="button"
                className={`mapOverlayToggle ${showIncidentOverlay ? 'isOn' : ''}`}
                onClick={() => {
                  setShowIncidentOverlay((prev) => !prev);
                  markTutorialAction('map.overlay_incidents_toggle');
                }}
                aria-pressed={showIncidentOverlay}
                aria-label={`${t('overlay_incidents')} (${incidentOverlayCount})`}
                data-tutorial-action="map.overlay_incidents_toggle"
              >
                <span>{t('overlay_incidents')}</span>
                <span className="mapOverlayToggle__count">{incidentOverlayCount}</span>
              </button>
              <button
                type="button"
                className={`mapOverlayToggle ${showSegmentTooltips ? 'isOn' : ''}`}
                onClick={() => {
                  setShowSegmentTooltips((prev) => !prev);
                  markTutorialAction('map.overlay_segments_toggle');
                }}
                aria-pressed={showSegmentTooltips}
                aria-label={`${t('overlay_segments')} (${segmentOverlayCount})`}
                data-tutorial-action="map.overlay_segments_toggle"
              >
                <span>{t('overlay_segments')}</span>
                <span className="mapOverlayToggle__count">{segmentOverlayCount}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

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

      <aside className={`panel ${isPanelCollapsed ? 'isCollapsed' : ''}`} aria-hidden={isPanelCollapsed}>
            <header className="panelHeader">
              <div className="panelHeader__top">
                <h1>{t('panel_title')}</h1>
                <div className="panelHeader__actions">
                  <span className="badge">v0</span>
                </div>
              </div>
              <p className="subtitle">{t('panel_subtitle')}</p>
            </header>

            <div id="app-sidebar-content" className="panelBody">
              <section className="card" data-tutorial-id="setup.section">
                <div className="sectionTitle">{t('setup')}</div>
                <div className="sectionHint">{SIDEBAR_SECTION_HINTS.setup}</div>

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
                  showSelectionHint={true}
                  tutorialId="setup.vehicle"
                  tutorialActionPrefix="setup.vehicle_option"
                />

                <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.scenarioMode}</div>

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
                  showSelectionHint={true}
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
                <select
                  id="ui-language"
                  className="input"
                  value={locale}
                  disabled={busy}
                  onChange={(event) => setLocale(event.target.value as Locale)}
                  data-tutorial-id="setup.language"
                  data-tutorial-action="setup.language_select"
                >
                  {LOCALE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>

                <div className="helper">
                  Scenario mode applies a policy-based delay multiplier and adds idle emissions (and
                  driver time cost) for the extra delay. This is a lightweight stub that can be
                  swapped for a detailed simulator later.
                </div>

                <div className="row" style={{ marginTop: 12 }}>
                  <button
                    className="secondary"
                    onClick={swapMarkers}
                    disabled={!origin || !destination || busy}
                    title="Swap start and destination"
                    data-tutorial-action="setup.swap_pins_button"
                  >
                    Swap pins
                  </button>
                  <button
                    className="secondary"
                    onClick={reset}
                    disabled={busy}
                    data-tutorial-action="setup.clear_pins_button"
                  >
                    Clear pins
                  </button>
                  <button
                    className="secondary"
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
                  >
                    {t('start_tutorial')}
                  </button>
                </div>
              </section>

              <PinManager
                nodes={pinNodes}
                selectedPinId={selectedPinId}
                disabled={busy}
                hasStop={Boolean(managedStop)}
                canAddStop={canAddStop}
                onSelectPin={selectPinFromSidebar}
                onRenameStart={renameStart}
                onRenameDestination={renameDestination}
                onRenameStop={renameStop}
                onAddStop={() => addStopFromMidpoint('origin')}
                onDeleteStop={deleteStop}
                onSwapPins={swapMarkers}
                onClearPins={reset}
              />

              <ScenarioParameterEditor
                value={advancedParams}
                onChange={setAdvancedParams}
                disabled={busy}
                validationError={advancedError}
              />

              <section className="card" data-tutorial-id="preferences.section">
                <div className="sectionTitle">{t('preferences')}</div>
                <div className="sectionHint">{SIDEBAR_SECTION_HINTS.preferences}</div>

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

                <div className="row row--actions" style={{ marginTop: 12 }}>
                  <button
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

                  <button
                    className="secondary"
                    onClick={clearComputedFromUi}
                    disabled={busy || paretoRoutes.length === 0}
                    data-tutorial-action="pref.clear_results_click"
                  >
                    {t('clear_results')}
                  </button>

                  {loading && (
                    <button className="secondary" onClick={cancelCompute}>
                      Cancel
                    </button>
                  )}
                </div>

                <div className="tiny">
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
              </section>

          {m && (
            <section
              className="card"
              data-tutorial-id="selected.route_panel"
              data-tutorial-action="selected.panel_click"
            >
              <div className="sectionTitle">Selected route</div>
              <div className="sectionHint">{SIDEBAR_SECTION_HINTS.selectedRoute}</div>
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

              {selectedRoute?.eta_explanations?.length ? (
                <div style={{ marginTop: 12 }}>
                  <div className="fieldLabel" style={{ marginBottom: 6 }}>
                    ETA explanation
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 16 }}>
                    {selectedRoute.eta_explanations.map((item, idx) => (
                      <li key={`${idx}-${item}`} className="tiny">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <div style={{ marginTop: 12 }}>
                <EtaTimelineChart route={selectedRoute} />
              </div>

              <SegmentBreakdown route={selectedRoute} onTutorialAction={markTutorialAction} />
              <CounterfactualPanel route={selectedRoute} />
            </section>
          )}

          <ScenarioTimeLapse route={selectedRoute} onPositionChange={setTimeLapsePosition} />

          {showRoutesSection && (
            <section className={`card routesSection ${isPending ? 'isUpdating' : ''}`} data-tutorial-id="routes.list">
              <div className="sectionTitleRow">
                <div className="sectionTitle">Routes</div>

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
              <div className="sectionHint">{SIDEBAR_SECTION_HINTS.routes}</div>

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
                      <div className="chartWrap" data-tutorial-id="routes.chart">
                        <ParetoChart
                          routes={paretoRoutes}
                          selectedId={selectedId}
                          labelsById={labelsById}
                          onSelect={selectRouteFromChart}
                        />
                      </div>

                      <div className="helper" style={{ marginTop: 10 }}>
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
                        {paretoRoutes.map((route, idx) => {
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
            </section>
          )}

          <section className="card" data-tutorial-id="compare.section">
            <div className="sectionTitleRow">
              <div className="sectionTitle">{t('compare_scenarios')}</div>
              <button
                className="secondary"
                onClick={compareScenarios}
                disabled={!canCompareScenarios}
                data-tutorial-action="compare.run_click"
              >
                {scenarioCompareLoading ? t('comparing_scenarios') : t('compare_scenarios')}
              </button>
            </div>
            <div className="sectionHint">{SIDEBAR_SECTION_HINTS.compareScenarios}</div>
            <ScenarioComparison
              data={scenarioCompare}
              loading={scenarioCompareLoading}
              error={scenarioCompareError}
              locale={locale}
            />
          </section>

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
          />
            </div>
      </aside>

      <TutorialOverlay
        open={tutorialOpen}
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
        optionalDecision={tutorialOptionalState}
        targetRect={tutorialTargetRect}
        targetMissing={tutorialTargetMissing && !(tutorialStep?.allowMissingTarget ?? false)}
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
