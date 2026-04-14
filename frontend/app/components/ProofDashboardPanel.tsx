'use client';

import FieldInfo from './FieldInfo';
import type { BaselineComparison } from '../lib/baselineComparison';
import type {
  ActionTraceSummary,
  PipelineMode,
  ProofArtifactLink,
  ProofDashboardSliceId,
  RouteCertificationSummary,
  RouteOption,
  WitnessSummary,
  WorldSupportSummary,
} from '../lib/types';

type ComparatorMeta = {
  method: string;
  compute_ms: number;
  notes: string[];
} | null;

type AcademicComparisonSummary = {
  same: boolean;
  durationPct: number;
  costPct: number;
  co2Pct: number;
  distancePct: number;
} | null;

type MetricTooltip = {
  definition: string;
  direction: string;
  unit: string;
  note?: string;
};

type TileMetric = {
  label: string;
  value: string;
  tooltip: MetricTooltip;
};

type ProofTile = {
  id: ProofDashboardSliceId;
  title: string;
  badge: string;
  summary: string;
  metrics: TileMetric[];
  links: ProofArtifactLink[];
  note?: string | null;
};

type TheoremFamilyCard = {
  title: string;
  summary: string;
  links: ProofArtifactLink[];
};

type Props = {
  locale: string;
  runId?: string | null;
  pipelineMode?: PipelineMode | null;
  terminalType?: string | null;
  selectedRoute: RouteOption | null;
  selectedRouteLabel?: string | null;
  manifestEndpoint?: string | null;
  artifactsEndpoint?: string | null;
  provenanceEndpoint?: string | null;
  artifactPointers?: Record<string, string | null> | null;
  selectedCertificate?: RouteCertificationSummary | null;
  selectedCertificateBasis?: string | null;
  actionTraceSummary?: ActionTraceSummary | null;
  witnessSummary?: WitnessSummary | null;
  worldSupportSummary?: WorldSupportSummary | null;
  supportSummary?: Record<string, unknown> | null;
  candidateCount?: number;
  baselineComparison?: BaselineComparison | null;
  baselineMeta?: ComparatorMeta;
  orsComparison?: BaselineComparison | null;
  orsMeta?: ComparatorMeta;
  academicComparisonLabel?: string;
  academicComparison?: AcademicComparisonSummary;
  onOpenRunInspector?: (runId: string) => void;
};

const tileGridStyle: React.CSSProperties = {
  display: 'grid',
  gap: '12px',
  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
};

const tileStyle: React.CSSProperties = {
  border: '1px solid rgba(15, 23, 42, 0.12)',
  borderRadius: '14px',
  padding: '14px',
  display: 'grid',
  gap: '12px',
  background:
    'linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.98) 100%)',
};

const metricListStyle: React.CSSProperties = {
  display: 'grid',
  gap: '8px',
};

const metricRowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: '10px',
  alignItems: 'flex-start',
};

const metricLabelStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  fontWeight: 600,
};

const metricValueStyle: React.CSSProperties = {
  textAlign: 'right',
  fontVariantNumeric: 'tabular-nums',
};

const theoremGridStyle: React.CSSProperties = {
  display: 'grid',
  gap: '12px',
  gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
};

function n(locale: string, value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  return new Intl.NumberFormat(locale, { maximumFractionDigits: 2 }).format(value);
}

function pct(locale: string, value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'n/a';
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    maximumFractionDigits: 1,
    minimumFractionDigits: 0,
  }).format(value);
}

function text(value: unknown): string {
  if (typeof value !== 'string') return 'n/a';
  const trimmed = value.trim();
  return trimmed || 'n/a';
}

function textOrNull(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed || null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function supportLabel(flag: boolean | null | undefined): string {
  if (flag === true) return 'In support';
  if (flag === false) return 'Out of support';
  return 'Support unknown';
}

function artifactHref(
  artifactsEndpoint: string | null | undefined,
  artifactPointer: string | null | undefined,
  fallbackName?: string,
  options?: { allowBaseResolution?: boolean },
): string | null {
  const allowBaseResolution = options?.allowBaseResolution ?? true;
  const raw = textOrNull(artifactPointer) ?? fallbackName ?? null;
  if (!raw) return null;
  if (/^(https?:)?\/\//i.test(raw) || raw.startsWith('/')) return raw;
  if (!allowBaseResolution || !artifactsEndpoint) return null;
  return `${artifactsEndpoint.replace(/\/$/, '')}/${encodeURIComponent(raw)}`;
}

function linkDefined(links: ProofArtifactLink[]): ProofArtifactLink[] {
  return links.filter((link) => Boolean(link.href));
}

function tooltipText(tooltip: MetricTooltip): string {
  return [
    `Definition: ${tooltip.definition}`,
    `Direction: ${tooltip.direction}`,
    `Unit: ${tooltip.unit}`,
    tooltip.note ? `Note: ${tooltip.note}` : null,
  ]
    .filter(Boolean)
    .join(' ');
}

function renderMetric(metric: TileMetric) {
  return (
    <div key={metric.label} style={metricRowStyle}>
      <div style={metricLabelStyle}>
        <span>{metric.label}</span>
        <FieldInfo text={tooltipText(metric.tooltip)} />
      </div>
      <div style={metricValueStyle}>{metric.value}</div>
    </div>
  );
}

function metric(
  label: string,
  value: string,
  definition: string,
  direction: string,
  unit: string,
  note?: string,
): TileMetric {
  return {
    label,
    value,
    tooltip: {
      definition,
      direction,
      unit,
      ...(note ? { note } : {}),
    },
  };
}

function dashboardTileTone(title: ProofDashboardSliceId): string {
  switch (title) {
    case 'v0':
      return 'V0';
    case 'a':
      return 'A';
    case 'b':
      return 'B';
    case 'c':
      return 'C';
    case 'broad':
      return 'Broad';
    case 'focused':
      return 'Focused';
    case 'cold_hot':
      return 'Cold/Hot';
    case 'osrm_ors':
      return 'OSRM/ORS';
    case 'theorem_artifact':
      return 'Proof';
    default:
      return 'Proof';
  }
}

function terminalLabel(terminalType: string | null | undefined): string {
  switch (terminalType) {
    case 'certified_singleton':
      return 'Certified singleton';
    case 'certified_set':
      return 'Certified set';
    case 'typed_abstention':
      return 'Typed abstention';
    default:
      return 'Decision payload';
  }
}

function comparatorDeltaSummary(
  locale: string,
  comparison: BaselineComparison | null | undefined,
  meta: ComparatorMeta | undefined,
): string {
  if (!comparison) return 'Unavailable';
  const method = meta?.method ? ` via ${meta.method}` : '';
  return `ETA ${comparison.etaPct.toFixed(1)}%${method}`;
}

function witnessNote(
  witnessSummary: WitnessSummary | null | undefined,
  selectedCertificateBasis: string | null | undefined,
  terminalType: string | null | undefined,
): string | null {
  if (!witnessSummary && !selectedCertificateBasis && !terminalType) return null;
  const parts: string[] = [];
  if (terminalType) parts.push(terminalLabel(terminalType));
  if (selectedCertificateBasis) parts.push(`basis ${selectedCertificateBasis}`);
  if (witnessSummary?.witness_size !== undefined && witnessSummary?.witness_size !== null) {
    parts.push(`witness size ${witnessSummary.witness_size}`);
  }
  if (witnessSummary?.active_challenger_ids?.length) {
    parts.push(`challengers ${witnessSummary.active_challenger_ids.join(', ')}`);
  }
  if (witnessSummary?.active_evidence_families?.length) {
    parts.push(`evidence ${witnessSummary.active_evidence_families.join(', ')}`);
  }
  return parts.length ? parts.join('; ') : null;
}

function csvCell(value: string): string {
  const escaped = value.replace(/"/g, '""');
  return `"${escaped}"`;
}

function downloadTextFile(contents: string, fileName: string, mimeType: string): void {
  const blob = new Blob([contents], { type: mimeType });
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = href;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(href);
}

export default function ProofDashboardPanel({
  locale,
  runId,
  pipelineMode,
  terminalType,
  selectedRoute,
  selectedRouteLabel,
  manifestEndpoint,
  artifactsEndpoint,
  provenanceEndpoint,
  artifactPointers,
  selectedCertificate,
  selectedCertificateBasis,
  actionTraceSummary,
  witnessSummary,
  worldSupportSummary,
  supportSummary,
  candidateCount = 0,
  baselineComparison,
  baselineMeta,
  orsComparison,
  orsMeta,
  academicComparisonLabel,
  academicComparison,
  onOpenRunInspector,
}: Props) {
  const supportFlag =
    worldSupportSummary?.support_flag ??
    (typeof supportSummary?.support_flag === 'boolean' ? (supportSummary.support_flag as boolean) : null);
  const worldReuseRate =
    worldSupportSummary?.world_reuse_rate ??
    numberOrNull(supportSummary?.world_reuse_rate) ??
    worldSupportSummary?.world_bundle_summary?.probabilistic_world_bundle?.world_reuse_rate ??
    null;
  const worldCount =
    worldSupportSummary?.world_count ??
    worldSupportSummary?.world_bundle_summary?.probabilistic_world_bundle?.world_count ??
    null;
  const searchCompleteness = actionTraceSummary?.search_completeness_score ?? null;
  const searchGap = actionTraceSummary?.search_completeness_gap ?? null;
  const selectedCandidateCount = actionTraceSummary?.selected_candidate_count ?? candidateCount ?? null;
  const bundleIndexJsonHref = artifactHref(artifactsEndpoint, null, 'index.json');
  const bundleIndexMdHref = artifactHref(artifactsEndpoint, null, 'index.md');
  const reportPdfHref = artifactHref(artifactsEndpoint, null, 'report.pdf');
  const dccsSummaryHref = artifactHref(artifactsEndpoint, artifactPointers?.dccs_summary ?? null, 'dccs_summary.json');
  const dccsCandidatesHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.dccs_candidates ?? null,
    'dccs_candidates.jsonl',
  );
  const strictFrontierHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.strict_frontier ?? null,
    'strict_frontier.jsonl',
  );
  const refinedRoutesHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.refined_routes ?? null,
    'refined_routes.jsonl',
  );
  const certificateSummaryHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.certificate_summary ?? null,
    'certificate_summary.json',
  );
  const fragilityHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.route_fragility_map ?? null,
    'route_fragility_map.json',
  );
  const worldManifestHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.sampled_world_manifest ?? null,
    'sampled_world_manifest.json',
  );
  const valueOfRefreshHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.value_of_refresh ?? null,
    'value_of_refresh.json',
  );
  const voiActionTraceHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.voi_action_trace ?? null,
    'voi_action_trace.json',
  );
  const voiActionScoresHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.voi_action_scores ?? null,
    'voi_action_scores.csv',
  );
  const voiStopHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.voi_stop_certificate ?? null,
    'voi_stop_certificate.json',
  );
  const resultsCsvHref = artifactHref(artifactsEndpoint, null, 'results.csv');
  const methodsAppendixHref = artifactHref(artifactsEndpoint, null, 'methods_appendix.md');
  const thesisReportHref = artifactHref(artifactsEndpoint, null, 'thesis_report.md');
  const orsSnapshotHref = artifactHref(artifactsEndpoint, null, 'ors_snapshot.json');
  const theoremFamilies: TheoremFamilyCard[] = [
    {
      title: 'Search Deficiency / DCCS',
      summary: 'Envelope, safe-prune, frontier, and refinement proof artifacts.',
      links: linkDefined([
        { label: 'DCCS summary', href: dccsSummaryHref },
        { label: 'Candidates', href: dccsCandidatesHref },
        { label: 'Strict frontier', href: strictFrontierHref },
      ]),
    },
    {
      title: 'Certification / REFC',
      summary: 'Certificate, fragility, and world-manifest proof artifacts.',
      links: linkDefined([
        { label: 'Certificate summary', href: certificateSummaryHref },
        { label: 'Fragility map', href: fragilityHref },
        { label: 'World manifest', href: worldManifestHref },
      ]),
    },
    {
      title: 'VOI / Controller',
      summary: 'Chosen actions, action scores, and the terminal stop certificate.',
      links: linkDefined([
        { label: 'Action trace', href: voiActionTraceHref },
        { label: 'Action scores', href: voiActionScoresHref },
        { label: 'Stop certificate', href: voiStopHref },
      ]),
    },
    {
      title: 'Reviewer Bundle',
      summary: 'Bundle-level documentation and reviewer-readable proof surfaces.',
      links: linkDefined([
        { label: 'index.json', href: bundleIndexJsonHref },
        { label: 'index.md', href: bundleIndexMdHref },
        { label: 'methods appendix', href: methodsAppendixHref },
        { label: 'thesis report', href: thesisReportHref },
      ]),
    },
  ];
  const vSeriesTiles: ProofTile[] = [
    {
      id: 'v0',
      title: 'V0 Comparator Slice',
      badge: dashboardTileTone('v0'),
      summary:
        'Comparator slice keeps baseline proof visible for the current run without leaving the route surface.',
      metrics: [
        metric(
          'OSRM delta',
          comparatorDeltaSummary(locale, baselineComparison, baselineMeta),
          'Current ETA-improvement summary against the OSRM baseline panel for this run.',
          'Higher positive percentages are better because they mean the selected route improved over the OSRM baseline.',
          'percent ETA improvement vs OSRM baseline',
          'The rendered string may also include the baseline method when it is available.',
        ),
        metric(
          'ORS delta',
          comparatorDeltaSummary(locale, orsComparison, orsMeta),
          'Current ETA-improvement summary against the OpenRouteService baseline panel for this run.',
          'Higher positive percentages are better because they mean the selected route improved over the ORS baseline.',
          'percent ETA improvement vs ORS baseline',
          'The rendered string may also include the comparator method when it is available.',
        ),
        metric(
          'Academic profile',
          academicComparison && !academicComparison.same && academicComparisonLabel
            ? `${academicComparisonLabel} (${academicComparison.durationPct.toFixed(1)}% ETA)`
            : 'No split from academic selector',
          'Academic-selector comparison status for the same candidate set.',
          'Context only; this is categorical when selectors converge and favorable when a positive ETA split is shown.',
          'profile label plus optional percent ETA difference vs academic selector',
        ),
      ],
      links: linkDefined([
        { label: 'results.csv', href: resultsCsvHref },
        { label: 'ORS snapshot', href: orsSnapshotHref },
      ]),
      note:
        baselineMeta?.notes?.length || orsMeta?.notes?.length
          ? [...(baselineMeta?.notes ?? []), ...(orsMeta?.notes ?? [])].join(' | ')
          : null,
    },
    {
      id: 'a',
      title: 'A Search / DCCS Slice',
      badge: dashboardTileTone('a'),
      summary: 'Search slice exposes candidate pressure, frontier hooks, and DCCS artifact paths.',
      metrics: [
        metric(
          'Candidates',
          n(locale, selectedCandidateCount),
          'Number of candidates still represented by the current action trace or route list.',
          'Lower is usually better once certification is stable because fewer survivors remain unresolved.',
          'candidate count',
        ),
        metric(
          'Search completeness',
          n(locale, searchCompleteness),
          'Reported search-completeness score at the terminal decision.',
          'Higher is better because more search deficiency has been resolved.',
          'search-completeness score',
        ),
        metric(
          'Search gap',
          n(locale, searchGap),
          'Reported unresolved search-side shortfall at the terminal decision.',
          'Lower is better because less unresolved search gap remains.',
          'search-gap score',
        ),
      ],
      links: linkDefined([
        { label: 'DCCS summary', href: dccsSummaryHref },
        { label: 'DCCS candidates', href: dccsCandidatesHref },
        { label: 'Strict frontier', href: strictFrontierHref },
        { label: 'Refined routes', href: refinedRoutesHref },
      ]),
    },
    {
      id: 'b',
      title: 'B Certification Slice',
      badge: dashboardTileTone('b'),
      summary: 'Certification slice keeps certificate, support, fragility, and world-count proof visible.',
      metrics: [
        metric(
          'Certificate',
          pct(locale, selectedCertificate?.certificate ?? null),
          'Empirical certificate level for the selected route or decision package.',
          'Higher is better because stronger certificate mass supports termination.',
          'probability or share',
        ),
        metric(
          'Threshold',
          pct(locale, selectedCertificate?.threshold ?? null),
          'Threshold the certification engine compared against for the current decision.',
          'Context only; this is the bar the certificate must meet rather than a better/worse score by itself.',
          'probability or share threshold',
        ),
        metric(
          'Support',
          supportLabel(supportFlag),
          'Whether the current decision remains inside the world-model support regime.',
          'In-support is better for trustworthiness; out-of-support indicates degraded certification claims.',
          'categorical support status',
        ),
        metric(
          'World count',
          n(locale, worldCount),
          'World-count signal carried by the support summary or probabilistic world bundle summary.',
          'Context only; larger counts mean more sampled worlds but not automatically a better outcome.',
          'world count',
        ),
      ],
      links: linkDefined([
        { label: 'Certificate summary', href: certificateSummaryHref },
        { label: 'Fragility map', href: fragilityHref },
        { label: 'World manifest', href: worldManifestHref },
      ]),
      note:
        selectedCertificateBasis && selectedRoute
          ? `Focused on ${selectedRouteLabel ?? selectedRoute.id} with basis ${selectedCertificateBasis}.`
          : selectedCertificateBasis
            ? `Certificate basis ${selectedCertificateBasis}.`
            : null,
    },
    {
      id: 'c',
      title: 'C Controller Slice',
      badge: dashboardTileTone('c'),
      summary: 'Controller slice keeps VOI actioning, stop reason, and refresh evidence adjacent to the decision.',
      metrics: [
        metric(
          'Terminal outcome',
          terminalLabel(terminalType),
          'Terminal class returned by the live DecisionPackage payload.',
          'Context only; singleton, set, or abstention are different stop modes rather than higher/lower scores.',
          'categorical terminal class',
        ),
        metric(
          'Stop reason',
          text(actionTraceSummary?.stop_reason),
          'Recorded reason for why the controller stopped or abstained.',
          'Context only; it explains the stop decision rather than ranking the run.',
          'categorical stop reason',
        ),
        metric(
          'Witness size',
          n(locale, witnessSummary?.witness_size ?? null),
          'Compactness of the active proof set for the current decision.',
          'Lower is usually better because a smaller witness is easier to inspect and explain.',
          'atomic witness item count',
        ),
      ],
      links: linkDefined([
        { label: 'Action trace', href: voiActionTraceHref },
        { label: 'Action scores', href: voiActionScoresHref },
        { label: 'Stop certificate', href: voiStopHref },
        { label: 'Value of refresh', href: valueOfRefreshHref },
      ]),
      note: witnessNote(witnessSummary, selectedCertificateBasis, terminalType),
    },
  ];
  const proofLensTiles: ProofTile[] = [
    {
      id: 'broad',
      title: 'Broad Proof Lens',
      badge: dashboardTileTone('broad'),
      summary: 'Broad lens keeps bundle-level proof entrypoints visible for the whole run, not only the selected route.',
      metrics: [
        metric(
          'Run ID',
          text(runId),
          'Stable run identifier used by Run Inspector and bundle-level artifact endpoints.',
          'Context only; identifiers are for lookup, not better/worse scoring.',
          'run identifier string',
        ),
        metric(
          'Pipeline mode',
          text(pipelineMode),
          'Pipeline mode reported by the run bundle for the current request.',
          'Context only; this is a runtime-path label rather than a quality metric.',
          'categorical pipeline identifier',
        ),
        metric(
          'Bundle docs',
          bundleIndexMdHref || bundleIndexJsonHref ? 'Available' : 'Unavailable',
          'Whether the run bundle exposes reviewer-readable and machine-readable index surfaces.',
          'Available is better for inspectability, but this is primarily a documentation-availability flag.',
          'categorical availability flag',
        ),
      ],
      links: linkDefined([
        { label: 'Manifest', href: manifestEndpoint },
        { label: 'Provenance', href: provenanceEndpoint },
        { label: 'index.md', href: bundleIndexMdHref },
        { label: 'index.json', href: bundleIndexJsonHref },
      ]),
    },
    {
      id: 'focused',
      title: 'Focused Proof Lens',
      badge: dashboardTileTone('focused'),
      summary: 'Focused lens compresses the live route-level proof into the route, witness, and certificate basis now on screen.',
      metrics: [
        metric(
          'Focused route',
          selectedRouteLabel ?? selectedRoute?.id ?? terminalLabel(terminalType),
          'Selected route label when a singleton is available, otherwise the terminal decision class.',
          'Context only; this names the currently focused proof target.',
          'route label or terminal class',
        ),
        metric(
          'Certificate basis',
          text(selectedCertificateBasis),
          'Selected certificate basis for the live decision package.',
          'Context only; this names the basis rather than scoring it.',
          'categorical certificate basis label',
        ),
        metric(
          'Challengers',
          n(locale, witnessSummary?.active_challenger_ids?.length ?? null),
          'How many active challenger routes are still recorded in the witness summary.',
          'Lower is usually better because fewer unresolved challengers remain on the proof boundary.',
          'challenger count',
        ),
      ],
      links: linkDefined([
        { label: 'Certificate summary', href: certificateSummaryHref },
        { label: 'Fragility map', href: fragilityHref },
        { label: 'Refined routes', href: refinedRoutesHref },
      ]),
      note: witnessNote(witnessSummary, selectedCertificateBasis, terminalType),
    },
    {
      id: 'cold_hot',
      title: 'Cold / Hot Lens',
      badge: dashboardTileTone('cold_hot'),
      summary: 'Cold/hot lens stays honest by anchoring reuse to current run evidence and bundle provenance links.',
      metrics: [
        metric(
          'World reuse',
          pct(locale, worldReuseRate),
          'Reuse share visible from the inline support/world summary for the current run.',
          'Context only; higher reuse indicates hotter reruns, not necessarily better routing quality.',
          'reuse fraction',
        ),
        metric(
          'Reuse state',
          worldReuseRate === null
            ? 'No reuse signal'
            : worldReuseRate > 0
              ? 'Hot-leaning'
              : 'Cold-leaning',
          'Dashboard heuristic derived from world reuse rate only.',
          'Context only; this indicates reuse posture, not route superiority.',
          'categorical reuse state',
          'This intentionally avoids inventing backend cache labels that are not emitted by the run payload.',
        ),
        metric(
          'Bundle provenance',
          manifestEndpoint || provenanceEndpoint ? 'Linked' : 'Missing',
          'Whether bundle-level manifest/provenance endpoints are available for current cold/hot inspection.',
          'Linked is better for auditability, but this is an availability flag rather than a performance metric.',
          'categorical availability flag',
        ),
      ],
      links: linkDefined([
        { label: 'Manifest', href: manifestEndpoint },
        { label: 'Provenance', href: provenanceEndpoint },
        { label: 'World manifest', href: worldManifestHref },
      ]),
    },
    {
      id: 'osrm_ors',
      title: 'OSRM / ORS Lens',
      badge: dashboardTileTone('osrm_ors'),
      summary: 'Comparator lens keeps both baseline families visible from one proof tile before opening the fuller comparison cards.',
      metrics: [
        metric(
          'OSRM ETA',
          baselineComparison ? `${baselineComparison.etaPct.toFixed(1)}%` : 'Unavailable',
          'ETA-improvement percentage versus OSRM for the active selected route.',
          'Higher positive percentages are better because they mean the selected route is faster than OSRM.',
          'percent ETA improvement vs OSRM',
        ),
        metric(
          'ORS ETA',
          orsComparison ? `${orsComparison.etaPct.toFixed(1)}%` : 'Unavailable',
          'ETA-improvement percentage versus OpenRouteService for the active selected route.',
          'Higher positive percentages are better because they mean the selected route is faster than ORS.',
          'percent ETA improvement vs ORS',
        ),
        metric(
          'Comparator methods',
          [baselineMeta?.method, orsMeta?.method]
            .filter((value): value is string => Boolean(value))
            .join(' | ') || 'Unavailable',
          'Current baseline engine identities surfaced by the page-level baseline panels.',
          'Context only; this names comparator sources rather than ranking them.',
          'engine identity label',
        ),
      ],
      links: linkDefined([
        { label: 'results.csv', href: resultsCsvHref },
        { label: 'ORS snapshot', href: orsSnapshotHref },
      ]),
    },
  ];
  const allTiles = [...vSeriesTiles, ...proofLensTiles];
  const generatedExplanation = [
    `This dashboard is currently focused on ${selectedRouteLabel ?? selectedRoute?.id ?? terminalLabel(terminalType)}.`,
    selectedCertificateBasis ? `The active certificate basis is ${selectedCertificateBasis}.` : null,
    supportFlag !== null && supportFlag !== undefined ? `Support status is ${supportLabel(supportFlag)}.` : null,
    actionTraceSummary?.stop_reason ? `The controller stopped because ${text(actionTraceSummary.stop_reason).toLowerCase()}.` : null,
    witnessSummary?.witness_size !== null && witnessSummary?.witness_size !== undefined
      ? `The active witness currently spans ${n(locale, witnessSummary.witness_size)} atomic items.`
      : null,
    witnessSummary?.active_challenger_ids?.length
      ? `Active challengers: ${witnessSummary.active_challenger_ids.join(', ')}.`
      : null,
    witnessSummary?.active_evidence_families?.length
      ? `Active evidence families: ${witnessSummary.active_evidence_families.join(', ')}.`
      : null,
  ]
    .filter((value): value is string => Boolean(value))
    .join(' ');
  const dashboardSummaryText = [
    `Proof dashboard for ${selectedRouteLabel ?? selectedRoute?.id ?? terminalLabel(terminalType)}`,
    `Pipeline mode: ${text(pipelineMode)}`,
    `Run id: ${text(runId)}`,
    '',
    ...allTiles.flatMap((tile) => [
      `${tile.title} [${tile.badge}]`,
      `Summary: ${tile.summary}`,
      ...tile.metrics.map((metric) => `${metric.label}: ${metric.value}`),
      ...(tile.note ? [`Note: ${tile.note}`] : []),
      ...tile.links.map((link) => `${link.label}: ${link.href ?? 'n/a'}`),
      '',
    ]),
    'Generated explanation:',
    generatedExplanation || 'No witness-driven explanation available.',
  ].join('\n');
  const dashboardCsv = [
    ['tile', 'badge', 'metric', 'value', 'link_label', 'link_href']
      .map(csvCell)
      .join(','),
    ...allTiles.flatMap((tile) => {
      const metricRows = tile.metrics.map((metric) =>
        [
          csvCell(tile.title),
          csvCell(tile.badge),
          csvCell(metric.label),
          csvCell(metric.value),
          csvCell(''),
          csvCell(''),
        ].join(','),
      );
      const linkRows = tile.links.map((link) =>
        [
          csvCell(tile.title),
          csvCell(tile.badge),
          csvCell('artifact_link'),
          csvCell(link.label),
          csvCell(link.label),
          csvCell(link.href ?? ''),
        ].join(','),
      );
      return [...metricRows, ...linkRows];
    }),
  ].join('\n');
  const visible =
    Boolean(runId) ||
    Boolean(selectedRoute) ||
    Boolean(manifestEndpoint) ||
    Boolean(artifactsEndpoint) ||
    Boolean(provenanceEndpoint) ||
    Boolean(selectedCertificate) ||
    Boolean(witnessSummary) ||
    Boolean(actionTraceSummary);

  if (!visible) return null;

  return (
    <section className="baselineComparePanel">
      <div className="baselineComparePanel__head">
        <div className="baselineComparePanel__title">Proof Dashboard</div>
        <div className="baselineEpicScore baselineEpicScore--mixed">
          {pipelineMode ? pipelineMode.toUpperCase() : 'PROOF'}
        </div>
      </div>
      <div className="baselineComparePanel__epicNote">
        Separate dashboard for proof slices and demo navigation. It complements the selected-route proof cards
        instead of replacing them.
      </div>
      <div className="actionGrid u-mt10">
        <button
          type="button"
          className="secondary"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(dashboardSummaryText);
            } catch {
              // no-op
            }
          }}
        >
          Copy Proof Summary
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => downloadTextFile(dashboardCsv, `proof-dashboard-${runId ?? 'current'}.csv`, 'text/csv;charset=utf-8')}
        >
          Export Dashboard CSV
        </button>
        {reportPdfHref ? (
          <a className="secondary" href={reportPdfHref} target="_blank" rel="noreferrer">
            Bundle PDF
          </a>
        ) : null}
      </div>

      <div className="fieldLabel u-mb6 u-mt12">
        V0 / A / B / C Proof Slices
        <FieldInfo text="These tiles keep comparator, search, certification, and controller proof families visible in one dashboard, each with direct links to the backing artifacts or bundle entrypoints." />
      </div>
      <div style={tileGridStyle}>
        {vSeriesTiles.map((tile) => (
          <article key={tile.id} style={tileStyle}>
            <div className="baselineComparePanel__head">
              <div className="baselineComparePanel__title">{tile.title}</div>
              <div className="baselineEpicScore baselineEpicScore--mixed">{tile.badge}</div>
            </div>
            <div className="baselineComparePanel__tradeoff">{tile.summary}</div>
            <div style={metricListStyle}>{tile.metrics.map(renderMetric)}</div>
            {tile.note ? <div className="baselineComparePanel__tradeoff">{tile.note}</div> : null}
            <div className="actionGrid">
              {tile.links.map((link) => (
                <a key={`${tile.id}-${link.label}`} className="secondary" href={link.href ?? undefined} target="_blank" rel="noreferrer">
                  {link.label}
                </a>
              ))}
              {runId && onOpenRunInspector ? (
                <button type="button" className="secondary" onClick={() => onOpenRunInspector(runId)}>
                  Open Run Inspector
                </button>
              ) : null}
            </div>
          </article>
        ))}
      </div>

      <div className="fieldLabel u-mb6 u-mt12">
        Proof Lenses
        <FieldInfo text="These lenses reorganize the same run into bundle-wide versus focused proof, reuse-aware cold/hot inspection, and baseline-family comparison without leaving the main page." />
      </div>
      <div style={tileGridStyle}>
        {proofLensTiles.map((tile) => (
          <article key={tile.id} style={tileStyle}>
            <div className="baselineComparePanel__head">
              <div className="baselineComparePanel__title">{tile.title}</div>
              <div className="baselineEpicScore baselineEpicScore--mixed">{tile.badge}</div>
            </div>
            <div className="baselineComparePanel__tradeoff">{tile.summary}</div>
            <div style={metricListStyle}>{tile.metrics.map(renderMetric)}</div>
            {tile.note ? <div className="baselineComparePanel__tradeoff">{tile.note}</div> : null}
            <div className="actionGrid">
              {tile.links.map((link) => (
                <a key={`${tile.id}-${link.label}`} className="secondary" href={link.href ?? undefined} target="_blank" rel="noreferrer">
                  {link.label}
                </a>
              ))}
              {runId && onOpenRunInspector ? (
                <button type="button" className="secondary" onClick={() => onOpenRunInspector(runId)}>
                  Open Run Inspector
                </button>
              ) : null}
            </div>
          </article>
        ))}
      </div>

      <div className="fieldLabel u-mb6 u-mt12">
        Theorem-to-Artifact Navigation
        <FieldInfo text="Frontend theorem-to-artifact navigation groups the current run's proof families by artifact bundle so reviewers can jump from search, certification, or controller claims to the emitted files directly." />
      </div>
      <div style={theoremGridStyle}>
        {theoremFamilies.map((family) => (
          <article key={family.title} style={tileStyle}>
            <div className="baselineComparePanel__head">
              <div className="baselineComparePanel__title">{family.title}</div>
              <div className="baselineEpicScore baselineEpicScore--mixed">Artifact</div>
            </div>
            <div className="baselineComparePanel__tradeoff">{family.summary}</div>
            <div className="actionGrid">
              {family.links.map((link) => (
                <a key={`${family.title}-${link.label}`} className="secondary" href={link.href ?? undefined} target="_blank" rel="noreferrer">
                  {link.label}
                </a>
              ))}
              {runId && onOpenRunInspector ? (
                <button type="button" className="secondary" onClick={() => onOpenRunInspector(runId)}>
                  Open Run Inspector
                </button>
              ) : null}
            </div>
          </article>
        ))}
      </div>

      <div className="fieldLabel u-mb6 u-mt12">
        Generated Proof Explanation
        <FieldInfo text="This explanation is synthesized deterministically from witness, terminal, support, and controller fields already visible in the run payload. It does not rely on free-form LLM generation." />
      </div>
      <div className="baselineComparePanel__tradeoff">
        {generatedExplanation || 'No witness-driven explanation was available for the current run payload.'}
      </div>
    </section>
  );
}
