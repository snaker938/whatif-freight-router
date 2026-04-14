'use client';

import { useEffect, useState, type CSSProperties } from 'react';

import FieldInfo from './FieldInfo';
import type { BaselineComparison } from '../lib/baselineComparison';
import type {
  ActionTraceSummary,
  DecisionProofContext,
  PipelineMode,
  ProofArtifactLink,
  ProofDashboardSliceId,
  RouteCertificationSummary,
  RouteOption,
  VoiStopCertificateArtifact,
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

type ExportReadinessItem = {
  label: string;
  href: string | null | undefined;
  actionLabel: string;
  description: string;
  tooltip: MetricTooltip;
  onAction?: (() => void) | null;
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
  proofContext?: DecisionProofContext | null;
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

const tileGridStyle: CSSProperties = {
  display: 'grid',
  gap: '12px',
  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
};

const tileStyle: CSSProperties = {
  border: '1px solid rgba(15, 23, 42, 0.12)',
  borderRadius: '14px',
  padding: '14px',
  display: 'grid',
  gap: '12px',
  background:
    'linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.98) 100%)',
};

const metricListStyle: CSSProperties = {
  display: 'grid',
  gap: '8px',
};

const metricRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: '10px',
  alignItems: 'flex-start',
};

const metricLabelStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  fontWeight: 600,
};

const metricValueStyle: CSSProperties = {
  textAlign: 'right',
  fontVariantNumeric: 'tabular-nums',
};

const theoremGridStyle: CSSProperties = {
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

async function fetchOptionalJson<T>(href: string, signal: AbortSignal): Promise<T | null> {
  const response = await fetch(href, { cache: 'no-store', signal });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Failed to load ${href} (${response.status})`);
  return (await response.json()) as T;
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

function renderExportReadinessLabel(item: ExportReadinessItem) {
  return (
    <div style={metricLabelStyle}>
      <span>{item.label}</span>
      <FieldInfo text={tooltipText(item.tooltip)} />
    </div>
  );
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

function escapeXml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function wrapExportText(value: string, maxCharsPerLine: number): string[] {
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (!normalized) return [];
  const words = normalized.split(' ');
  const lines: string[] = [];
  let currentLine = '';

  words.forEach((word) => {
    const candidate = currentLine ? `${currentLine} ${word}` : word;
    if (candidate.length <= maxCharsPerLine || !currentLine) {
      currentLine = candidate;
      return;
    }
    lines.push(currentLine);
    currentLine = word;
  });

  if (currentLine) lines.push(currentLine);
  return lines;
}

function dashboardExportStem(runId: string | null | undefined): string {
  const normalized = textOrNull(runId)?.replace(/[^a-zA-Z0-9_-]+/g, '-') ?? 'current';
  return `proof-dashboard-${normalized}`;
}

function buildDashboardSvg(args: {
  runId?: string | null;
  pipelineMode?: PipelineMode | null;
  terminalType?: string | null;
  selectedRouteLabel?: string | null;
  generatedExplanation: string;
  tiles: ProofTile[];
}): string {
  const { runId, pipelineMode, terminalType, selectedRouteLabel, generatedExplanation, tiles } = args;
  const columns = 2;
  const cardWidth = 610;
  const cardHeight = 176;
  const gap = 18;
  const left = 28;
  const top = 112;
  const width = 1280;
  const rows = Math.ceil(tiles.length / columns);
  const explanationLines = wrapExportText(generatedExplanation || 'No witness-driven explanation was available.', 120);
  const height = top + rows * (cardHeight + gap) + 96 + explanationLines.length * 18;
  const elements = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#f8fafc"/>`,
    `<text x="${left}" y="38" font-size="30" font-family="Georgia, 'Times New Roman', serif" font-weight="700" fill="#0f172a">Proof Dashboard Export</text>`,
    `<text x="${left}" y="64" font-size="14" font-family="'Segoe UI', Arial, sans-serif" fill="#475569">Route: ${escapeXml(selectedRouteLabel ?? terminalLabel(terminalType))} | Pipeline: ${escapeXml(text(pipelineMode))} | Run: ${escapeXml(text(runId))}</text>`,
    `<text x="${left}" y="86" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#64748b">SVG export generated from the visible decision, controller, evidence, and comparator dashboard state.</text>`,
  ];

  tiles.forEach((tile, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = left + column * (cardWidth + gap);
    const y = top + row * (cardHeight + gap);
    const summaryLines = wrapExportText(tile.summary, 56).slice(0, 2);
    const noteLines = tile.note ? wrapExportText(tile.note, 58).slice(0, 2) : [];
    const linkLabels = tile.links.slice(0, 3).map((link) => link.label).join(' | ');

    elements.push(
      `<rect x="${x}" y="${y}" width="${cardWidth}" height="${cardHeight}" rx="16" fill="#ffffff" stroke="#cbd5e1"/>`,
      `<text x="${x + 16}" y="${y + 26}" font-size="18" font-family="'Segoe UI', Arial, sans-serif" font-weight="700" fill="#0f172a">${escapeXml(tile.title)}</text>`,
      `<text x="${x + cardWidth - 80}" y="${y + 26}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" font-weight="700" fill="#1d4ed8">${escapeXml(tile.badge)}</text>`,
    );

    summaryLines.forEach((line, lineIndex) => {
      elements.push(
        `<text x="${x + 16}" y="${y + 48 + lineIndex * 16}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#334155">${escapeXml(line)}</text>`,
      );
    });

    tile.metrics.slice(0, 3).forEach((metricRow, metricIndex) => {
      elements.push(
        `<text x="${x + 16}" y="${y + 88 + metricIndex * 18}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" font-weight="700" fill="#0f172a">${escapeXml(metricRow.label)}:</text>`,
        `<text x="${x + 210}" y="${y + 88 + metricIndex * 18}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#0f172a">${escapeXml(metricRow.value)}</text>`,
      );
    });

    if (noteLines.length) {
      noteLines.forEach((line, lineIndex) => {
        elements.push(
          `<text x="${x + 16}" y="${y + 146 + lineIndex * 14}" font-size="11" font-family="'Segoe UI', Arial, sans-serif" fill="#7c2d12">${escapeXml(line)}</text>`,
        );
      });
    } else if (linkLabels) {
      elements.push(
        `<text x="${x + 16}" y="${y + 150}" font-size="11" font-family="'Segoe UI', Arial, sans-serif" fill="#475569">Linked artifacts: ${escapeXml(linkLabels)}</text>`,
      );
    }
  });

  const explanationTop = top + rows * (cardHeight + gap) + 28;
  elements.push(
    `<text x="${left}" y="${explanationTop}" font-size="16" font-family="'Segoe UI', Arial, sans-serif" font-weight="700" fill="#0f172a">Generated explanation</text>`,
  );
  explanationLines.forEach((line, lineIndex) => {
    elements.push(
      `<text x="${left}" y="${explanationTop + 24 + lineIndex * 18}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#334155">${escapeXml(line)}</text>`,
    );
  });

  elements.push('</svg>');
  return elements.join('\n');
}

function buildDashboardPrintHtml(args: {
  title: string;
  subtitle: string;
  svg: string;
  tiles: ProofTile[];
  exportReadinessItems: ExportReadinessItem[];
  generatedExplanation: string;
}): string {
  const { title, subtitle, svg, tiles, exportReadinessItems, generatedExplanation } = args;
  const exportRows = exportReadinessItems
    .map((item) => {
      const availability = item.href || item.onAction ? 'Available' : 'Missing';
      return `<tr><td>${item.label}</td><td>${availability}</td><td>${item.description}</td></tr>`;
    })
    .join('');
  const tileSections = tiles
    .map((tile) => {
      const metricRows = tile.metrics
        .map((metricRow) => `<tr><td>${metricRow.label}</td><td>${metricRow.value}</td></tr>`)
        .join('');
      const artifactRows = tile.links.length
        ? `<ul>${tile.links
            .map((link) => `<li>${link.label}${link.href ? `: ${link.href}` : ''}</li>`)
            .join('')}</ul>`
        : '<p>No linked artifacts were available for this tile.</p>';
      return `<section class="tile"><h2>${tile.title}</h2><p class="summary">${tile.summary}</p><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${metricRows}</tbody></table><div class="links"><strong>Artifact links</strong>${artifactRows}</div>${tile.note ? `<p class="note">${tile.note}</p>` : ''}</section>`;
    })
    .join('');

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${title}</title>
  <style>
    body { margin: 24px; font-family: Georgia, "Times New Roman", serif; color: #0f172a; background: #f8fafc; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    p.meta { margin: 0 0 18px; color: #475569; font-family: "Segoe UI", Arial, sans-serif; }
    .hero { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 16px; padding: 16px; }
    .hero svg { width: 100%; height: auto; display: block; }
    .sectionTitle { margin: 24px 0 10px; font-size: 22px; }
    table { width: 100%; border-collapse: collapse; margin: 0; font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; background: #ffffff; }
    th, td { border: 1px solid #cbd5e1; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #dbeafe; color: #1e3a8a; }
    .tileGrid { display: grid; gap: 16px; }
    .tile { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 16px; padding: 16px; break-inside: avoid; }
    .tile h2 { margin: 0 0 8px; font-size: 18px; }
    .summary, .note, .links, .explanation { font-family: "Segoe UI", Arial, sans-serif; color: #334155; }
    .note { color: #7c2d12; }
    ul { margin: 8px 0 0 18px; }
    @page { size: A4 landscape; margin: 14mm; }
  </style>
</head>
<body>
  <h1>${title}</h1>
  <p class="meta">${subtitle}</p>
  <div class="hero">${svg}</div>
  <h2 class="sectionTitle">Export readiness</h2>
  <table><thead><tr><th>Surface</th><th>Status</th><th>Description</th></tr></thead><tbody>${exportRows}</tbody></table>
  <h2 class="sectionTitle">Generated explanation</h2>
  <p class="explanation">${generatedExplanation}</p>
  <h2 class="sectionTitle">Decision / controller / evidence slices</h2>
  <div class="tileGrid">${tileSections}</div>
</body>
</html>`;
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
  proofContext,
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
  const proofSelectedCertificateBasis = proofContext?.selected_certificate_basis ?? selectedCertificateBasis ?? null;
  const supportFlag =
    proofContext?.support_flag ??
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
  const decisionPackageHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.decision_package ?? null,
  );
  const preferenceStateHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.preference_state ?? null,
  );
  const preferenceQueryTraceHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.preference_query_trace ?? null,
  );
  const worldSupportSummaryHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.world_support_summary ?? null,
  );
  const certifiedSetSummaryHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.certified_set_summary ?? null,
  );
  const certificateWitnessHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.certificate_witness ?? null,
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
  const voiControllerStateHref = artifactHref(
    artifactsEndpoint,
    artifactPointers?.voi_controller_state ?? null,
    'voi_controller_state.jsonl',
  );
  const resultsCsvHref = artifactHref(artifactsEndpoint, null, 'results.csv');
  const methodsAppendixHref = artifactHref(artifactsEndpoint, null, 'methods_appendix.md');
  const thesisReportHref = artifactHref(artifactsEndpoint, null, 'thesis_report.md');
  const orsSnapshotHref = artifactHref(artifactsEndpoint, null, 'ors_snapshot.json');
  const [voiStopCertificate, setVoiStopCertificate] = useState<VoiStopCertificateArtifact | null>(null);

  useEffect(() => {
    if (!voiStopHref) {
      setVoiStopCertificate(null);
      return;
    }
    const controller = new AbortController();
    void fetchOptionalJson<VoiStopCertificateArtifact>(voiStopHref, controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) {
          setVoiStopCertificate(payload);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setVoiStopCertificate(null);
        }
      });
    return () => controller.abort();
  }, [voiStopHref]);

  const controllerState = proofContext?.controller_state ?? voiStopCertificate?.controller_state ?? null;
  const controllerCertificateLcb = controllerState?.certificate_lcb ?? selectedCertificate?.certificate_lcb ?? null;
  const controllerCertifiedSetSize = controllerState?.certified_set_size ?? null;
  const controllerBoundarySummary =
    proofContext?.controller_boundary_summary ?? controllerState?.active_certificate_boundary_summary ?? null;
  const controllerBoundaryChallenger = controllerBoundarySummary?.active_challenger_id ?? null;
  const controllerWeightSetShrinkage = controllerState?.weight_set_shrinkage ?? null;
  const controllerUnresolvedWinnerMass = controllerState?.unresolved_possible_winner_mass ?? null;
  const controllerProxyOnlyFraction = controllerState?.proxy_only_fraction ?? null;
  const controllerSupportReason = proofContext?.out_of_support_reason ?? controllerState?.out_of_support_reason ?? null;
  const controllerEvaluationTag = controllerState?.audit_propensity_summary?.certification_evaluation_tag ?? null;
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
        { label: 'Controller state', href: voiControllerStateHref },
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
        metric(
          'Certificate LCB',
          pct(locale, controllerCertificateLcb),
          'Lower certificate bound emitted by the controller-state or selected certificate payload.',
          'Higher is better because the terminal decision has stronger conservative support.',
          'probability or share lower bound',
        ),
        metric(
          'Certified-set size',
          n(locale, controllerCertifiedSetSize),
          'Surviving certified-set size emitted on the controller state when available.',
          'Lower is usually better because fewer routes remain in the certified set once the proof is sharper.',
          'certified-set size',
        ),
      ],
      links: linkDefined([
        { label: 'Certificate summary', href: certificateSummaryHref },
        { label: 'Fragility map', href: fragilityHref },
        { label: 'World manifest', href: worldManifestHref },
      ]),
      note:
        proofSelectedCertificateBasis && selectedRoute
          ? `Focused on ${selectedRouteLabel ?? selectedRoute.id} with basis ${proofSelectedCertificateBasis}.`
          : proofSelectedCertificateBasis
            ? `Certificate basis ${proofSelectedCertificateBasis}.`
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
        metric(
          'Boundary challenger',
          text(controllerBoundaryChallenger),
          'Active challenger emitted by the literal controller-state boundary summary.',
          'Context only; this identifies the route currently pressing on the certificate boundary.',
          'route identifier',
        ),
        metric(
          'Weight shrinkage',
          pct(locale, controllerWeightSetShrinkage),
          'Weight-set shrinkage emitted by the terminal controller state.',
          'Higher is usually better because more of the compatible proof region has been collapsed.',
          'shrinkage fraction',
        ),
        metric(
          'Unresolved winner mass',
          pct(locale, controllerUnresolvedWinnerMass),
          'Remaining unresolved possible-winner mass emitted on the literal controller state.',
          'Lower is better because less winner-side ambiguity remains.',
          'probability mass',
        ),
        metric(
          'Proxy-only fraction',
          pct(locale, controllerProxyOnlyFraction),
          'Share of the decision still resting on proxy-only evidence according to the controller state.',
          'Lower is usually better because less of the proof relies on proxy-only evidence.',
          'proxy-only fraction',
        ),
      ],
      links: linkDefined([
        { label: 'Action trace', href: voiActionTraceHref },
        { label: 'Action scores', href: voiActionScoresHref },
        { label: 'Stop certificate', href: voiStopHref },
        { label: 'Controller state', href: voiControllerStateHref },
        { label: 'Value of refresh', href: valueOfRefreshHref },
      ]),
      note: witnessNote(witnessSummary, proofSelectedCertificateBasis, terminalType),
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
          text(proofSelectedCertificateBasis),
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
        { label: 'Decision package', href: decisionPackageHref },
        { label: 'Preference state', href: preferenceStateHref },
        { label: 'Preference query trace', href: preferenceQueryTraceHref },
        { label: 'World support summary', href: worldSupportSummaryHref },
        { label: 'Certified set summary', href: certifiedSetSummaryHref },
        { label: 'Certificate witness', href: certificateWitnessHref },
        { label: 'VOI stop certificate', href: voiStopHref },
        { label: 'VOI controller state', href: voiControllerStateHref },
        { label: 'Fragility map', href: fragilityHref },
        { label: 'Refined routes', href: refinedRoutesHref },
      ]),
      note: witnessNote(witnessSummary, proofSelectedCertificateBasis, terminalType),
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
    proofSelectedCertificateBasis ? `The active certificate basis is ${proofSelectedCertificateBasis}.` : null,
    supportFlag !== null && supportFlag !== undefined ? `Support status is ${supportLabel(supportFlag)}.` : null,
    actionTraceSummary?.stop_reason ? `The controller stopped because ${text(actionTraceSummary.stop_reason).toLowerCase()}.` : null,
    controllerBoundaryChallenger ? `The current boundary challenger is ${controllerBoundaryChallenger}.` : null,
    controllerSupportReason ? `Controller support note: ${controllerSupportReason}.` : null,
    terminalType === 'typed_abstention' && proofContext?.typed_abstention?.reason_code
      ? `Typed abstention provenance: ${proofContext.typed_abstention.reason_code}.`
      : null,
    controllerProxyOnlyFraction !== null
      ? `Proxy-only share is ${pct(locale, controllerProxyOnlyFraction)}.`
      : null,
    controllerEvaluationTag ? `Evaluation tag: ${controllerEvaluationTag}.` : null,
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
  const proofSurfaceSelectedRoute = selectedRouteLabel ?? selectedRoute?.id ?? null;
  const proofSurfaceCsv = [
    ['surface', 'summary', 'metrics', 'artifact_hrefs'].map(csvCell).join(','),
    [
      csvCell('decision'),
      csvCell(
        [
          `terminal=${terminalLabel(terminalType)}`,
          proofSurfaceSelectedRoute ? `route=${proofSurfaceSelectedRoute}` : null,
          `certificate=${pct(locale, selectedCertificate?.certificate ?? null)}`,
          `threshold=${pct(locale, selectedCertificate?.threshold ?? null)}`,
          `certificate_lcb=${pct(locale, selectedCertificate?.certificate_lcb ?? null)}`,
          `certificate_ucb=${pct(locale, selectedCertificate?.certificate_ucb ?? null)}`,
          proofSelectedCertificateBasis ? `basis=${proofSelectedCertificateBasis}` : null,
        ]
          .filter((value): value is string => Boolean(value))
          .join('; '),
      ),
      csvCell(
        [
          `selected_route=${proofSurfaceSelectedRoute ?? 'n/a'}`,
          `certificate_basis=${proofSelectedCertificateBasis ?? 'n/a'}`,
          `certified=${selectedCertificate?.certified === true ? 'true' : selectedCertificate?.certified === false ? 'false' : 'n/a'}`,
        ].join('; '),
      ),
      csvCell(
        [
          decisionPackageHref,
          certificateSummaryHref,
        ]
          .filter((href): href is string => Boolean(href))
          .join(' | '),
      ),
    ].join(','),
    [
      csvCell('controller'),
      csvCell(
        [
          `stop_reason=${text(actionTraceSummary?.stop_reason)}`,
          `search_completeness=${n(locale, actionTraceSummary?.search_completeness_score ?? null)}`,
          `search_gap=${n(locale, actionTraceSummary?.search_completeness_gap ?? null)}`,
          `selected_candidates=${n(locale, actionTraceSummary?.selected_candidate_count ?? candidateCount ?? null)}`,
          `witness_size=${n(locale, witnessSummary?.witness_size ?? null)}`,
          `proxy_only_fraction=${pct(locale, controllerProxyOnlyFraction)}`,
        ].join('; '),
      ),
      csvCell(
        [
          witnessSummary?.active_challenger_ids?.length
            ? `challengers=${witnessSummary.active_challenger_ids.join('|')}`
            : null,
          proofSelectedCertificateBasis ? `basis=${proofSelectedCertificateBasis}` : null,
          controllerBoundaryChallenger ? `boundary_challenger=${controllerBoundaryChallenger}` : null,
        ]
          .filter((value): value is string => Boolean(value))
          .join('; '),
      ),
      csvCell(
        [
          voiActionTraceHref,
          voiControllerStateHref,
          voiStopHref,
        ]
          .filter((href): href is string => Boolean(href))
          .join(' | '),
      ),
    ].join(','),
    [
      csvCell('evidence'),
      csvCell(
        [
          `support=${supportLabel(supportFlag)}`,
          controllerSupportReason ? `support_reason=${controllerSupportReason}` : null,
          `world_count=${n(locale, worldCount)}`,
          `world_reuse_rate=${pct(locale, worldReuseRate)}`,
          worldSupportSummary?.calibration_bin ? `calibration_bin=${worldSupportSummary.calibration_bin}` : null,
          worldSupportSummary?.support_bin ? `support_bin=${worldSupportSummary.support_bin}` : null,
        ]
          .filter((value): value is string => Boolean(value))
          .join('; '),
      ),
      csvCell(
        [
          worldSupportSummary?.active_families?.length
            ? `active_families=${worldSupportSummary.active_families.join('|')}`
            : null,
          witnessSummary?.active_evidence_families?.length
            ? `evidence_families=${witnessSummary.active_evidence_families.join('|')}`
            : null,
        ]
          .filter((value): value is string => Boolean(value))
          .join('; '),
      ),
      csvCell(
        [
          worldSupportSummaryHref,
          worldManifestHref,
          certificateWitnessHref,
          fragilityHref,
        ]
          .filter((href): href is string => Boolean(href))
          .join(' | '),
      ),
    ].join(','),
  ].join('\n');
  const dashboardExportFileStem = dashboardExportStem(runId);
  const bundleExportReadinessItems: ExportReadinessItem[] = [
    {
      label: 'report.pdf',
      href: reportPdfHref,
      actionLabel: 'Open PDF',
      description: 'Bundle-level PDF document emitted for the current run when available.',
      tooltip: {
        definition: 'Bundle-level PDF document emitted for the current run when available.',
        direction: 'Availability is better because the current run published a reviewer-facing PDF surface.',
        unit: 'availability state',
      },
    },
    {
      label: 'results.csv',
      href: resultsCsvHref,
      actionLabel: 'Open CSV',
      description: 'Raw results table exported by the current bundle when available.',
      tooltip: {
        definition: 'Raw results table exported by the current bundle when available.',
        direction: 'Availability is better because reviewers can inspect or reuse emitted row-level results directly.',
        unit: 'availability state',
      },
    },
    {
      label: 'index.md',
      href: bundleIndexMdHref,
      actionLabel: 'Open Markdown',
      description: 'Reviewer-readable bundle index for the current run.',
      tooltip: {
        definition: 'Reviewer-readable bundle index for the current run.',
        direction: 'Availability is better because the bundle exposes a human-readable entrypoint to emitted surfaces.',
        unit: 'availability state',
      },
    },
    {
      label: 'index.json',
      href: bundleIndexJsonHref,
      actionLabel: 'Open JSON',
      description: 'Machine-readable bundle index for the current run.',
      tooltip: {
        definition: 'Machine-readable bundle index for the current run.',
        direction: 'Availability is better because the bundle exposes a structured entrypoint to emitted surfaces.',
        unit: 'availability state',
      },
    },
    {
      label: 'methods_appendix.md',
      href: methodsAppendixHref,
      actionLabel: 'Open appendix',
      description: 'Methods appendix document published by the bundle when available.',
      tooltip: {
        definition: 'Methods appendix document published by the bundle when available.',
        direction: 'Availability is better because the bundle includes a reader-facing methods companion.',
        unit: 'availability state',
      },
    },
    {
      label: 'thesis_report.md',
      href: thesisReportHref,
      actionLabel: 'Open report',
      description: 'Long-form Markdown report emitted by the bundle when available.',
      tooltip: {
        definition: 'Long-form Markdown report emitted by the bundle when available.',
        direction: 'Availability is better because the current run includes a narrative report surface in addition to raw artifacts.',
        unit: 'availability state',
      },
    },
    {
      label: 'voi_stop_certificate.json',
      href: voiStopHref,
      actionLabel: 'Open stop certificate',
      description: 'Terminal VOI stop certificate emitted for the current run when available.',
      tooltip: {
        definition: 'Terminal VOI stop certificate emitted for the current run when available.',
        direction: 'Availability is better because the reviewer can inspect the terminal controller decision directly.',
        unit: 'availability state',
      },
    },
    {
      label: 'voi_controller_state.jsonl',
      href: voiControllerStateHref,
      actionLabel: 'Open controller state',
      description: 'Per-iteration controller-state trace emitted for the current run when available.',
      tooltip: {
        definition: 'Per-iteration controller-state trace emitted for the current run when available.',
        direction: 'Availability is better because the reviewer can inspect proof-boundary state evolution directly.',
        unit: 'availability state',
      },
    },
    {
      label: 'manifest',
      href: manifestEndpoint,
      actionLabel: 'Open manifest',
      description: 'Run manifest endpoint for bundle metadata and emitted-surface discovery.',
      tooltip: {
        definition: 'Run manifest endpoint for bundle metadata and emitted-surface discovery.',
        direction: 'Availability is better because it provides the primary endpoint for discovering emitted run surfaces.',
        unit: 'availability state',
      },
    },
    {
      label: 'provenance',
      href: provenanceEndpoint,
      actionLabel: 'Open provenance',
      description: 'Bundle provenance endpoint for run lineage and source context.',
      tooltip: {
        definition: 'Bundle provenance endpoint for run lineage and source context.',
        direction: 'Availability is better because it exposes source lineage and run-context auditability for the current proof bundle.',
        unit: 'availability state',
      },
    },
  ];
  const dashboardSvg = buildDashboardSvg({
    runId,
    pipelineMode,
    terminalType,
    selectedRouteLabel,
    generatedExplanation:
      generatedExplanation || 'No witness-driven explanation was available for the current run payload.',
    tiles: allTiles,
  });
  const dashboardPrintHtml = buildDashboardPrintHtml({
    title: 'Proof Dashboard Print Layout',
    subtitle: `${selectedRouteLabel ?? selectedRoute?.id ?? terminalLabel(terminalType)} | ${text(pipelineMode)} | run ${text(runId)}`,
    svg: dashboardSvg,
    tiles: allTiles,
    exportReadinessItems: bundleExportReadinessItems,
    generatedExplanation:
      generatedExplanation || 'No witness-driven explanation was available for the current run payload.',
  });
  const openDashboardPrintLayout = () => {
    const popup = window.open('', '_blank');
    if (!popup) {
      downloadTextFile(
        dashboardPrintHtml,
        `${dashboardExportFileStem}.print.html`,
        'text/html;charset=utf-8',
      );
      return;
    }
    popup.document.open();
    popup.document.write(dashboardPrintHtml);
    popup.document.close();
    popup.focus();
    window.setTimeout(() => {
      try {
        popup.print();
      } catch {
        // no-op
      }
    }, 250);
  };
  const exportReadinessItems: ExportReadinessItem[] = [
    ...bundleExportReadinessItems,
    {
      label: 'proof-surfaces.csv',
      href: null,
      onAction: () =>
        downloadTextFile(
          proofSurfaceCsv,
          `proof-surfaces-${runId ?? 'current'}.csv`,
          'text/csv;charset=utf-8',
        ),
      actionLabel: 'Export proof surfaces CSV',
      description: 'Client-side CSV export for the visible decision, controller, and evidence proof slices.',
      tooltip: {
        definition: 'Client-side CSV export for the visible decision, controller, and evidence proof slices.',
        direction: 'Availability is better because the reviewer can extract the proof-facing slices without leaving the dashboard.',
        unit: 'availability state',
      },
    },
    {
      label: 'dashboard.svg',
      href: null,
      onAction: () =>
        downloadTextFile(
          dashboardSvg,
          `${dashboardExportFileStem}.svg`,
          'image/svg+xml;charset=utf-8',
        ),
      actionLabel: 'Export SVG',
      description: 'Client-side SVG export for the currently visible decision, controller, and evidence dashboard.',
      tooltip: {
        definition: 'Client-side SVG export for the currently visible decision, controller, and evidence dashboard.',
        direction: 'Availability is better because the reviewer can capture a vector figure from the current dashboard state without leaving the page.',
        unit: 'availability state',
      },
    },
    {
      label: 'dashboard.print.html',
      href: null,
      onAction: () =>
        downloadTextFile(
          dashboardPrintHtml,
          `${dashboardExportFileStem}.print.html`,
          'text/html;charset=utf-8',
        ),
      actionLabel: 'Export print HTML',
      description: 'Print-ready HTML layout for browser Save-as-PDF or direct printing of the current proof dashboard.',
      tooltip: {
        definition: 'Print-ready HTML layout for browser Save-as-PDF or direct printing of the current proof dashboard.',
        direction: 'Availability is better because the reviewer can produce a PDF-ready dashboard figure from the current visible proof state.',
        unit: 'availability state',
      },
    },
  ];
  const explanationSources = [
    {
      label: 'Decision package',
      href: decisionPackageHref,
      description: 'Primary decision payload containing terminal state and certificate-basis context.',
    },
    {
      label: 'World support summary',
      href: worldSupportSummaryHref,
      description: 'Support regime and world-bundle summary used by the explanation when emitted.',
    },
    {
      label: 'Certificate witness',
      href: certificateWitnessHref,
      description: 'Witness-side challenger and evidence-family source when the bundle publishes it.',
    },
    {
      label: 'VOI stop certificate',
      href: voiStopHref,
      description: 'Controller stop certificate for the terminal proof state when present.',
    },
    {
      label: 'VOI controller state',
      href: voiControllerStateHref,
      description: 'Per-iteration controller-state trace for the current proof state when present.',
    },
  ] as const;
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
          onClick={() =>
            downloadTextFile(
              dashboardSummaryText,
              `proof-dashboard-${runId ?? 'current'}.txt`,
              'text/plain;charset=utf-8',
            )
          }
        >
          Export Proof Summary TXT
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => downloadTextFile(dashboardCsv, `proof-dashboard-${runId ?? 'current'}.csv`, 'text/csv;charset=utf-8')}
        >
          Export Dashboard CSV
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() =>
            downloadTextFile(
              proofSurfaceCsv,
              `proof-surfaces-${runId ?? 'current'}.csv`,
              'text/csv;charset=utf-8',
            )
          }
        >
          Export Proof Surfaces CSV
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() =>
            downloadTextFile(
              dashboardSvg,
              `${dashboardExportFileStem}.svg`,
              'image/svg+xml;charset=utf-8',
            )
          }
        >
          Export Dashboard SVG
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() =>
            downloadTextFile(
              dashboardPrintHtml,
              `${dashboardExportFileStem}.print.html`,
              'text/html;charset=utf-8',
            )
          }
        >
          Export Print HTML
        </button>
        <button type="button" className="secondary" onClick={openDashboardPrintLayout}>
          Open Print Layout
        </button>
        {reportPdfHref ? (
          <a className="secondary" href={reportPdfHref} target="_blank" rel="noreferrer">
            Bundle PDF
          </a>
        ) : null}
      </div>

      <div className="fieldLabel u-mb6 u-mt12">
        Export Readiness
        <FieldInfo text="This section reports both bundle-emitted documents and client-side dashboard exports. The current dashboard can always export TXT, CSV, SVG, and print-ready HTML, while linked bundle PDFs and documents depend on what the active run actually emitted." />
      </div>
      <div className="baselineComparePanel__tradeoff">
        Reflects both emitted bundle surfaces and on-demand dashboard exports. Missing entries mean the
        current run did not publish that bundle document or endpoint; dashboard CSV, proof-surface CSV,
        SVG, and print-ready HTML remain available from the controls above.
      </div>
      <div className="baselineKpiGrid">
        {exportReadinessItems.map((item) => (
          <div key={item.label} className={`baselineKpi ${item.href || item.onAction ? 'isPositive' : 'isNegative'}`}>
            <div className="baselineKpi__label">{renderExportReadinessLabel(item)}</div>
            <div className="baselineKpi__value">{item.href || item.onAction ? 'Available' : 'Missing'}</div>
            <div className="baselineKpi__meta">
              {item.href ? (
                <>
                  <a className="secondary" href={item.href} target="_blank" rel="noreferrer">
                    {item.actionLabel}
                  </a>{' '}
                  · {item.description}
                </>
              ) : item.onAction ? (
                <>
                  <button type="button" className="secondary" onClick={item.onAction}>
                    {item.actionLabel}
                  </button>{' '}
                  · {item.description}
                </>
              ) : (
                `Not emitted. ${item.description}`
              )}
            </div>
          </div>
        ))}
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
      <div className="fieldLabel u-mb6 u-mt12">
        Explanation Sources
        <FieldInfo text="Direct emitted sources for the deterministic explanation above. Availability depends on which artifacts or bundle endpoints the current run actually published." />
      </div>
      <div className="baselineComparePanel__tradeoff">
        These links point to the concrete emitted sources behind the explanation above when they are available.
      </div>
      <div className="actionGrid">
        {explanationSources.map((source) =>
          source.href ? (
            <a
              key={source.label}
              className="secondary"
              href={source.href}
              target="_blank"
              rel="noreferrer"
              title={source.description}
            >
              {source.label}
            </a>
          ) : (
            <span
              key={source.label}
              className="secondary"
              aria-disabled="true"
              title={`Unavailable. ${source.description}`}
              style={{ opacity: 0.65, cursor: 'not-allowed' }}
            >
              {source.label}: unavailable
            </span>
          ),
        )}
      </div>
    </section>
  );
}
