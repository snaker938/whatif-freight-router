import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const artifactIndexPath = path.join(repoRoot, 'paper_artifact_index.json');
const outputDir = path.join(repoRoot, 'out', 'headline_exports', 'current_checked');
const ARTIFACT_ROOT_CANDIDATES = [
  path.join(repoRoot, 'backend', 'out', 'artifacts'),
  path.join(repoRoot, 'out', 'artifacts'),
  path.resolve('C:\\app\\out\\artifacts'),
];

function resolveArtifactBundle(...dirNames) {
  for (const root of ARTIFACT_ROOT_CANDIDATES) {
    for (const dirName of dirNames) {
      const candidate = path.join(root, dirName);
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
  }
  return path.join(ARTIFACT_ROOT_CANDIDATES[0], dirNames[0]);
}

const FULL_SUITE_SOURCE_BUNDLE = resolveArtifactBundle(
  'full_suite_curated_latest_20260411',
);
const FULL_SUITE_COMPANION_DIR_NAME = 'full_suite_curated_latest_20260411';
const FULL_SUITE_BROAD_COLD_SOURCE_BUNDLE = resolveArtifactBundle(
  'full_suite_curated_latest_20260411_broad_cold_proof',
);
const FULL_SUITE_BROAD_COLD_COMPANION_DIR_NAME =
  'full_suite_curated_latest_20260411_broad_cold_proof';
const FULL_SUITE_BROAD_COLD_COMPANION_FILES = [
  'evaluation_manifest.json',
  'lane_metadata.json',
  'metadata.json',
  'thesis_metrics.json',
  'thesis_metrics.summary.md',
  'thesis_plots.json',
  'thesis_results.json',
  'thesis_summary.csv',
  'thesis_summary.json',
];
const FULL_SUITE_BROAD_COLD_REQUIRED_COMPANION_FILES = [
  'evaluation_manifest.json',
  'lane_metadata.json',
  'metadata.json',
  'thesis_metrics.json',
  'thesis_plots.json',
  'thesis_results.json',
  'thesis_summary.csv',
  'thesis_summary.json',
];
const FULL_SUITE_BASELINE_IDENTITY_SOURCE_ROOT = path.join(
  repoRoot,
  'backend',
  'out',
  'thesis_campaigns',
  'dominance_cluster5_cardiff_bath_corr12p5_r2',
  'tranche_001',
  'artifacts',
  'dominance_cluster5_cardiff_bath_corr12p5_r2_t001',
);
const FULL_SUITE_BASELINE_IDENTITY_FILES = [
  'osrm_baseline_identity_manifest.json',
  'ors_baseline_identity_manifest.json',
];
const FULL_SUITE_COMPANION_FILES = [
  'index.json',
  'index.md',
  'publishability_verdict.json',
  'publishability_verdict.summary.md',
  'publishability_assessment.md',
  'lane_publishability_summary.csv',
  'lane_publishability_summary.json',
  'lane_publishability_summary.summary.md',
  'sample_size_gate_summary.csv',
  'sample_size_gate_summary.json',
  'sample_size_gate_summary.summary.md',
  'headline_seed_claims_summary.csv',
  'headline_seed_claims_summary.json',
  'headline_seed_claims_summary.summary.md',
  'failure_atlas_lane_metadata.json',
  'failure_atlas.json',
  'failure_atlas.md',
  'failure_atlas.summary.md',
  'universal_baseline_audit.csv',
  'universal_baseline_audit.json',
  'universal_baseline_audit.summary.md',
  'metadata.json',
  'metadata.summary.md',
  'suite_sources.json',
  'suite_sources.summary.md',
  'suite_progress.json',
  'suite_progress.summary.md',
];
const FULL_SUITE_REQUIRED_COMPANION_FILES = [
  'index.json',
  'publishability_verdict.json',
  'publishability_assessment.md',
  'lane_publishability_summary.csv',
  'lane_publishability_summary.json',
  'sample_size_gate_summary.csv',
  'sample_size_gate_summary.json',
  'headline_seed_claims_summary.csv',
  'headline_seed_claims_summary.json',
  'failure_atlas_lane_metadata.json',
  'failure_atlas.json',
  'failure_atlas.md',
  'universal_baseline_audit.csv',
  'universal_baseline_audit.json',
];
const THRESHOLD_SENSITIVITY_SOURCE_BUNDLE = resolveArtifactBundle(
  'full_suite_curated_latest_20260411_threshold_sensitivity',
);
const THRESHOLD_SENSITIVITY_COMPANION_DIR_NAME =
  'full_suite_curated_latest_20260411_threshold_sensitivity';
const THRESHOLD_SENSITIVITY_COMPANION_FILES = [
  'index.json',
  'index.md',
  'evaluation_manifest.json',
  'evaluation_manifest.summary.md',
  'lane_metadata.json',
  'lane_metadata.summary.md',
  'metadata.json',
  'metadata.summary.md',
  'results.json',
  'results.summary.md',
  'thesis_metrics.json',
  'thesis_metrics.summary.md',
  'thesis_summary.csv',
  'thesis_summary.json',
  'thesis_summary.summary.md',
  'thesis_summary_by_cohort.csv',
  'thesis_summary_by_cohort.json',
  'thesis_summary_by_cohort.summary.md',
  'cohort_composition.json',
  'cohort_composition.summary.md',
  'threshold_sensitivity_summary.csv',
  'threshold_sensitivity_summary.json',
  'threshold_sensitivity_summary.summary.md',
  'threshold_sensitivity_report.md',
  'thesis_plots.json',
  'thesis_plots.summary.md',
  'thesis_report.md',
];
const THRESHOLD_SENSITIVITY_REQUIRED_COMPANION_FILES = [
  'index.json',
  'lane_metadata.json',
  'thesis_metrics.json',
  'threshold_sensitivity_summary.csv',
  'threshold_sensitivity_summary.json',
  'threshold_sensitivity_report.md',
  'thesis_plots.json',
];
const PUBLIC_TRANSFER_SOURCE_BUNDLE = resolveArtifactBundle(
  'full_suite_curated_latest_20260411_public_transfer',
);
const PUBLIC_TRANSFER_COMPANION_DIR_NAME = 'full_suite_curated_latest_20260411_public_transfer';
const PUBLIC_TRANSFER_COMPANION_FILES = [
  'index.json',
  'index.md',
  'evaluation_manifest.json',
  'evaluation_manifest.summary.md',
  'lane_metadata.json',
  'lane_metadata.summary.md',
  'metadata.json',
  'metadata.summary.md',
  'results.json',
  'results.summary.md',
  'thesis_metrics.json',
  'thesis_metrics.summary.md',
  'thesis_summary.csv',
  'thesis_summary.json',
  'thesis_summary.summary.md',
  'thesis_summary_by_cohort.csv',
  'thesis_summary_by_cohort.json',
  'thesis_summary_by_cohort.summary.md',
  'cohort_composition.json',
  'cohort_composition.summary.md',
  'thesis_summary_by_transfer_slice.csv',
  'thesis_summary_by_transfer_slice.json',
  'thesis_summary_by_transfer_slice.summary.md',
  'thesis_summary_by_weather_regime_transfer_slice.csv',
  'thesis_summary_by_weather_regime_transfer_slice.json',
  'thesis_summary_by_weather_regime_transfer_slice.summary.md',
  'thesis_plots.json',
  'thesis_plots.summary.md',
  'thesis_report.md',
];
const PUBLIC_TRANSFER_REQUIRED_COMPANION_FILES = [
  'index.json',
  'lane_metadata.json',
  'thesis_metrics.json',
  'thesis_summary_by_transfer_slice.csv',
  'thesis_summary_by_transfer_slice.json',
  'thesis_summary_by_weather_regime_transfer_slice.csv',
  'thesis_summary_by_weather_regime_transfer_slice.json',
  'thesis_plots.json',
];
const PROOF_SURFACE_COMPANION_FILES = [
  'index.json',
  'index.md',
  'baseline_smoke_summary.json',
  'baseline_smoke_summary.summary.md',
  'cohort_composition.json',
  'cohort_composition.summary.md',
  'evaluation_manifest.json',
  'evaluation_manifest.summary.md',
  'lane_metadata.json',
  'lane_metadata.summary.md',
  'metadata.json',
  'metadata.summary.md',
  'methods_appendix.md',
  'od_corpus.csv',
  'od_corpus.json',
  'od_corpus.summary.md',
  'od_corpus_summary.json',
  'od_corpus_summary.summary.md',
  'repo_asset_preflight.json',
  'results.json',
  'results.summary.md',
  'thesis_metrics.json',
  'thesis_metrics.summary.md',
  'thesis_plots.json',
  'thesis_plots.summary.md',
  'thesis_report.md',
  'thesis_results.csv',
  'thesis_results.json',
  'thesis_results.summary.md',
  'thesis_summary.csv',
  'thesis_summary.json',
  'thesis_summary.summary.md',
  'thesis_summary_by_cohort.csv',
  'thesis_summary_by_cohort.json',
  'thesis_summary_by_cohort.summary.md',
];
const PROOF_SURFACE_REQUIRED_COMPANION_FILES = [
  'index.json',
  'lane_metadata.json',
  'results.json',
  'thesis_metrics.json',
  'thesis_plots.json',
  'thesis_results.json',
  'thesis_summary.csv',
  'thesis_summary.json',
];
const OPTIONAL_STOPPING_SOURCE_BUNDLE = resolveArtifactBundle(
  'full_suite_curated_latest_20260411_optional_stopping_coverage',
);
const OPTIONAL_STOPPING_COMPANION_DIR_NAME =
  'full_suite_curated_latest_20260411_optional_stopping_coverage';
const OPTIONAL_STOPPING_COMPANION_FILES = PROOF_SURFACE_COMPANION_FILES;
const OPTIONAL_STOPPING_REQUIRED_COMPANION_FILES =
  PROOF_SURFACE_REQUIRED_COMPANION_FILES;
const PERTURBATION_SOURCE_BUNDLE = resolveArtifactBundle(
  'full_suite_curated_latest_20260411_perturbation_flip_radius',
);
const PERTURBATION_COMPANION_DIR_NAME =
  'full_suite_curated_latest_20260411_perturbation_flip_radius';
const PERTURBATION_COMPANION_FILES = PROOF_SURFACE_COMPANION_FILES;
const PERTURBATION_REQUIRED_COMPANION_FILES =
  PROOF_SURFACE_REQUIRED_COMPANION_FILES;
const HOT_RERUN_SOURCE_BUNDLE = resolveArtifactBundle(
  'full_suite_curated_latest_20260411_hot_rerun_pair_hot',
  'full_suite_curated_latest_20260411_hot_rerun_hot',
);
const HOT_RERUN_COMPANION_DIR_NAME = 'full_suite_curated_latest_20260411_hot_rerun_hot';
const HOT_RERUN_COMPANION_FILES = [
  'index.json',
  'index.md',
  'baseline_smoke_summary.json',
  'baseline_smoke_summary.summary.md',
  'cohort_composition.json',
  'cohort_composition.summary.md',
  'evaluation_manifest.json',
  'evaluation_manifest.summary.md',
  'hot_rerun_gate.json',
  'hot_rerun_gate.summary.md',
  'hot_rerun_report.md',
  'hot_rerun_vs_cold_comparison.csv',
  'hot_rerun_vs_cold_comparison.json',
  'hot_rerun_vs_cold_comparison.summary.md',
  'lane_metadata.json',
  'lane_metadata.summary.md',
  'metadata.json',
  'metadata.summary.md',
  'methods_appendix.md',
  'od_corpus.csv',
  'od_corpus.json',
  'od_corpus.summary.md',
  'od_corpus_summary.json',
  'od_corpus_summary.summary.md',
  'repo_asset_preflight.json',
  'results.json',
  'results.summary.md',
  'thesis_metrics.json',
  'thesis_metrics.summary.md',
  'thesis_plots.json',
  'thesis_plots.summary.md',
  'thesis_report.md',
  'thesis_results.csv',
  'thesis_results.json',
  'thesis_results.summary.md',
  'thesis_summary.csv',
  'thesis_summary.json',
  'thesis_summary.summary.md',
  'thesis_summary_by_cohort.csv',
  'thesis_summary_by_cohort.json',
  'thesis_summary_by_cohort.summary.md',
];
const HOT_RERUN_REQUIRED_COMPANION_FILES = [
  'index.json',
  'lane_metadata.json',
  'hot_rerun_gate.json',
  'hot_rerun_report.md',
  'hot_rerun_vs_cold_comparison.json',
  'thesis_metrics.json',
  'thesis_report.md',
];

const TABLE_COLUMNS = {
  'table.focused_voi.variant_summary': [
    'variant_id',
    'pipeline_mode',
    'weighted_win_rate_best_baseline',
    'mean_runtime_ms',
    'mean_certificate',
  ],
  'table.focused_voi.cohort_summary': [
    'cohort_label',
    'variant_id',
    'pipeline_mode',
    'weighted_win_rate_best_baseline',
    'mean_runtime_ms',
    'mean_certificate',
  ],
  'table.focused_voi.aggregate_variant_evidence': [
    'variant_id',
    'pipeline_mode',
    'row_count',
    'success_rate',
    'weighted_win_rate_best_baseline',
    'mean_runtime_ms',
    'mean_certificate',
    'mean_frontier_count',
    'mean_search_budget_utilization',
    'mean_evidence_budget_utilization',
    'median_preference_query_count',
    'p90_preference_query_count',
    'max_preference_query_count',
    'preference_certification_success_rate',
  ],
  'table.focused_voi.preference_burden_summary': [
    'variant_id',
    'pipeline_mode',
    'row_count',
    'success_rate',
    'mean_runtime_ms',
    'mean_certificate',
    'median_preference_query_count',
    'p90_preference_query_count',
    'max_preference_query_count',
    'preference_certification_success_rate',
    'mean_search_budget_utilization',
    'mean_evidence_budget_utilization',
  ],
  'table.focused_voi.preference_burden_by_cohort': [
    'variant_id',
    'cohort_label',
    'pipeline_mode',
    'row_count',
    'success_rate',
    'mean_runtime_ms',
    'mean_certificate',
    'median_preference_query_count',
    'p90_preference_query_count',
    'max_preference_query_count',
    'preference_certification_success_rate',
    'mean_search_budget_utilization',
    'mean_evidence_budget_utilization',
  ],
  'table.latest_checked_campaign.summary_and_metrics': [
    'metric',
    'V0',
    'A',
    'B',
    'C',
    'A-V0',
    'B-A',
    'C-B',
  ],
  'table.latest_checked_campaign.runtime_observability_summary': [
    'variant_id',
    'row_count',
    'mean_runtime_p50_ms',
    'mean_runtime_p90_ms',
    'mean_runtime_p95_ms',
    'mean_process_rss_p90_mb',
    'mean_process_vms_p90_mb',
    'max_process_rss_mb',
    'max_process_vms_mb',
    'mean_search_budget_utilization_p90',
    'mean_evidence_budget_utilization_p90',
    'mean_graph_low_ambiguity_fast_path_rate',
    'mean_graph_supported_ambiguity_fast_fallback_rate',
  ],
  'table.latest_checked_campaign.runtime_action_observability_summary': [
    'variant_id',
    'row_count',
    'mean_runtime_ms',
    'mean_search_budget_used',
    'mean_evidence_budget_used',
    'mean_search_budget_utilization',
    'mean_evidence_budget_utilization',
    'mean_action_family_budget_share_search',
    'mean_action_family_budget_share_evidence',
    'mean_action_family_budget_share_preference',
    'mean_voi_action_count',
    'mean_voi_refine_action_count',
    'mean_voi_refresh_action_count',
    'mean_voi_resample_action_count',
    'mean_graph_low_ambiguity_fast_path_rate',
    'mean_graph_supported_ambiguity_fast_fallback_rate',
    'graph_low_ambiguity_fast_path_precision',
    'graph_low_ambiguity_fast_path_precision_denominator',
    'graph_low_ambiguity_fast_path_recall',
    'graph_low_ambiguity_fast_path_recall_denominator',
  ],
  'table.latest_checked_campaign.runtime_stage_quantiles': [
    'stage',
    'row_count',
    'p50_ms',
    'p90_ms',
    'p95_ms',
  ],
  'table.focused_voi.cohort_support_composition': [
    'scope',
    'variant_id',
    'composition_family',
    'composition_label',
    'row_count',
    'reference_row_count',
  ],
  'table.latest_checked_campaign.cohort_support_composition': [
    'scope',
    'variant_id',
    'composition_family',
    'composition_label',
    'row_count',
    'reference_row_count',
  ],
};

const SECTION_EXPORT_ROOT = 'out/headline_exports/current_checked';
const QUICKSTART_EXPORT_REFERENCE =
  'docs/reviewer_quickstart.md#headline-svg-and-pdf-export-commands';
const EXPLICIT_REVIEWER_EXPORT_SURFACE_IDS = [
  'table.focused_voi.aggregate_variant_evidence',
  'table.focused_voi.preference_burden_summary',
  'table.focused_voi.preference_burden_by_cohort',
  'table.latest_checked_campaign.runtime_observability_summary',
  'table.latest_checked_campaign.runtime_action_observability_summary',
  'table.latest_checked_campaign.runtime_stage_quantiles',
  'table.focused_voi.cohort_support_composition',
  'table.latest_checked_campaign.cohort_support_composition',
];
const CONTAINER_DIGEST_ENV_VARS = [
  'CONTAINER_DIGEST',
  'IMAGE_DIGEST',
  'OCI_IMAGE_DIGEST',
  'DOCKER_IMAGE_DIGEST',
];
const HEADLINE_IDENTITY_SCHEMA_VERSION = 'route-artifact-identity-v1';
const VARIANT_ORDER = ['ALL', 'V0', 'A', 'B', 'C'];
const SUPPORT_BIN_DEFINITIONS = {
  unknown_support: 'No finite support-richness estimate was available for the row.',
  weak_support: 'Support richness is at or below 0.45 and the row is support-fragile.',
  mid_support: 'Support richness lies between 0.45 and 0.75.',
  strong_support: 'Support richness is at or above 0.75 and the row is strong-support.',
};
const SUPPORT_BIN_ORDER = [
  'unknown_support',
  'weak_support',
  'mid_support',
  'strong_support',
];

let cachedGitCommitHash;
let cachedEnvironmentLockfile;
let cachedContainerIdentity;

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function removeIfExists(filePath) {
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
}

function stageCompanionBundle({
  sourceRoot,
  targetDirName,
  companionFiles,
  requiredCompanionFiles,
  errorLabel,
}) {
  const targetRoot = path.join(outputDir, targetDirName);
  const sourceAvailable = fs.existsSync(sourceRoot);
  ensureDir(targetRoot);

  if (!sourceAvailable && !fs.existsSync(targetRoot)) {
    throw new Error(`Missing ${errorLabel} source bundle at ${sourceRoot}`);
  }

  const relativeFiles = [];
  for (const fileName of companionFiles) {
    const sourcePath = path.join(sourceRoot, fileName);
    const targetPath = path.join(targetRoot, fileName);
    if (sourceAvailable && fs.existsSync(sourcePath)) {
      ensureDir(path.dirname(targetPath));
      fs.copyFileSync(sourcePath, targetPath);
    }
    if (fs.existsSync(targetPath)) {
      relativeFiles.push(relativeOutputPath(path.join(targetDirName, fileName)));
    }
  }

  for (const fileName of requiredCompanionFiles) {
    const targetPath = path.join(targetRoot, fileName);
    if (!fs.existsSync(targetPath)) {
      throw new Error(`Checked ${errorLabel} companion is incomplete. Missing ${targetPath}`);
    }
  }

  return {
    relativeBundlePath: relativeOutputPath(targetDirName),
    relativeFiles,
    sourceRoot,
  };
}

function stageFullSuiteVerdictCompanion() {
  const bundle = stageCompanionBundle({
    sourceRoot: FULL_SUITE_SOURCE_BUNDLE,
    targetDirName: FULL_SUITE_COMPANION_DIR_NAME,
    companionFiles: FULL_SUITE_COMPANION_FILES,
    requiredCompanionFiles: FULL_SUITE_REQUIRED_COMPANION_FILES,
    errorLabel: 'checked full-suite verdict',
  });
  const targetRoot = path.join(outputDir, FULL_SUITE_COMPANION_DIR_NAME);
  for (const fileName of FULL_SUITE_BASELINE_IDENTITY_FILES) {
    const sourcePath = path.join(FULL_SUITE_BASELINE_IDENTITY_SOURCE_ROOT, fileName);
    const targetPath = path.join(targetRoot, fileName);
    if (!fs.existsSync(sourcePath)) {
      throw new Error(`Missing baseline identity manifest source at ${sourcePath}`);
    }
    fs.copyFileSync(sourcePath, targetPath);
  }
  const indexPath = path.join(targetRoot, 'index.json');
  if (fs.existsSync(indexPath)) {
    const index = JSON.parse(fs.readFileSync(indexPath, 'utf8'));
    index.osrm_baseline_identity_manifest_json = relativeOutputPath(
      path.join(FULL_SUITE_COMPANION_DIR_NAME, 'osrm_baseline_identity_manifest.json'),
    );
    index.ors_baseline_identity_manifest_json = relativeOutputPath(
      path.join(FULL_SUITE_COMPANION_DIR_NAME, 'ors_baseline_identity_manifest.json'),
    );
    fs.writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`, 'utf8');
  }
  return bundle;
}

function stageRuntimeObservabilityCompanion() {
  return stageCompanionBundle({
    sourceRoot: FULL_SUITE_BROAD_COLD_SOURCE_BUNDLE,
    targetDirName: FULL_SUITE_BROAD_COLD_COMPANION_DIR_NAME,
    companionFiles: FULL_SUITE_BROAD_COLD_COMPANION_FILES,
    requiredCompanionFiles: FULL_SUITE_BROAD_COLD_REQUIRED_COMPANION_FILES,
    errorLabel: 'checked runtime-observability',
  });
}

function stageThresholdSensitivityCompanion() {
  return stageCompanionBundle({
    sourceRoot: THRESHOLD_SENSITIVITY_SOURCE_BUNDLE,
    targetDirName: THRESHOLD_SENSITIVITY_COMPANION_DIR_NAME,
    companionFiles: THRESHOLD_SENSITIVITY_COMPANION_FILES,
    requiredCompanionFiles: THRESHOLD_SENSITIVITY_REQUIRED_COMPANION_FILES,
    errorLabel: 'checked threshold-sensitivity',
  });
}

function stagePublicTransferCompanion() {
  return stageCompanionBundle({
    sourceRoot: PUBLIC_TRANSFER_SOURCE_BUNDLE,
    targetDirName: PUBLIC_TRANSFER_COMPANION_DIR_NAME,
    companionFiles: PUBLIC_TRANSFER_COMPANION_FILES,
    requiredCompanionFiles: PUBLIC_TRANSFER_REQUIRED_COMPANION_FILES,
    errorLabel: 'checked public-transfer',
  });
}

function stageOptionalStoppingCompanion() {
  return stageCompanionBundle({
    sourceRoot: OPTIONAL_STOPPING_SOURCE_BUNDLE,
    targetDirName: OPTIONAL_STOPPING_COMPANION_DIR_NAME,
    companionFiles: OPTIONAL_STOPPING_COMPANION_FILES,
    requiredCompanionFiles: OPTIONAL_STOPPING_REQUIRED_COMPANION_FILES,
    errorLabel: 'checked optional-stopping',
  });
}

function stagePerturbationCompanion() {
  return stageCompanionBundle({
    sourceRoot: PERTURBATION_SOURCE_BUNDLE,
    targetDirName: PERTURBATION_COMPANION_DIR_NAME,
    companionFiles: PERTURBATION_COMPANION_FILES,
    requiredCompanionFiles: PERTURBATION_REQUIRED_COMPANION_FILES,
    errorLabel: 'checked perturbation',
  });
}

function stageHotRerunCompanion() {
  return stageCompanionBundle({
    sourceRoot: HOT_RERUN_SOURCE_BUNDLE,
    targetDirName: HOT_RERUN_COMPANION_DIR_NAME,
    companionFiles: HOT_RERUN_COMPANION_FILES,
    requiredCompanionFiles: HOT_RERUN_REQUIRED_COMPANION_FILES,
    errorLabel: 'checked hot-rerun',
  });
}

const LANE_ARTIFACT_GENERATION_SOURCES = [
  {
    lane_role: 'broad_cold_proof',
    sourceRoot: FULL_SUITE_BROAD_COLD_SOURCE_BUNDLE,
    companionDirName: FULL_SUITE_BROAD_COLD_COMPANION_DIR_NAME,
  },
  {
    lane_role: 'threshold_sensitivity',
    sourceRoot: THRESHOLD_SENSITIVITY_SOURCE_BUNDLE,
    companionDirName: THRESHOLD_SENSITIVITY_COMPANION_DIR_NAME,
  },
  {
    lane_role: 'public_transfer',
    sourceRoot: PUBLIC_TRANSFER_SOURCE_BUNDLE,
    companionDirName: PUBLIC_TRANSFER_COMPANION_DIR_NAME,
  },
  {
    lane_role: 'optional_stopping_coverage',
    sourceRoot: OPTIONAL_STOPPING_SOURCE_BUNDLE,
    companionDirName: OPTIONAL_STOPPING_COMPANION_DIR_NAME,
  },
  {
    lane_role: 'perturbation_flip_radius',
    sourceRoot: PERTURBATION_SOURCE_BUNDLE,
    companionDirName: PERTURBATION_COMPANION_DIR_NAME,
  },
  {
    lane_role: 'hot_rerun',
    sourceRoot: HOT_RERUN_SOURCE_BUNDLE,
    companionDirName: HOT_RERUN_COMPANION_DIR_NAME,
  },
];

function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function escapeHtml(value) {
  return escapeXml(value);
}

function sanitizeFileStem(surfaceId) {
  return surfaceId.replaceAll('/', '_').replaceAll('\\', '_');
}

function toPosixPath(relativePath) {
  return String(relativePath ?? '').replaceAll('\\', '/');
}

function fileExists(relativePath) {
  try {
    const stat = fs.statSync(path.join(repoRoot, relativePath));
    return stat.isFile();
  } catch {
    return false;
  }
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function formatValue(value) {
  if (value === null || value === undefined || value === '') return 'n/a';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return 'n/a';
    if (Math.abs(value) >= 1000) return value.toFixed(2);
    if (Math.abs(value) >= 1) return value.toFixed(3).replace(/\.?0+$/, '');
    return value.toFixed(4).replace(/\.?0+$/, '');
  }
  return String(value);
}

function sortVariantIds(variantIds) {
  return [...new Set(variantIds.filter(Boolean))].sort((left, right) => {
    const leftIndex = VARIANT_ORDER.indexOf(left);
    const rightIndex = VARIANT_ORDER.indexOf(right);
    if (leftIndex !== -1 && rightIndex !== -1) return leftIndex - rightIndex;
    if (leftIndex !== -1) return -1;
    if (rightIndex !== -1) return 1;
    return String(left).localeCompare(String(right));
  });
}

function incrementCount(counts, key, amount = 1) {
  counts[key] = (counts[key] ?? 0) + amount;
}

function quantile(values, percentile) {
  const sorted = values
    .filter((value) => Number.isFinite(value))
    .map((value) => Number(value))
    .sort((left, right) => left - right);
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * percentile;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  if (lowerIndex === upperIndex) return sorted[lowerIndex];
  const lowerWeight = upperIndex - position;
  const upperWeight = position - lowerIndex;
  return sorted[lowerIndex] * lowerWeight + sorted[upperIndex] * upperWeight;
}

function sanitizeMetricKey(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'value';
}

function classifySupportBin(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'unknown_support';
  if (value <= 0.45) return 'weak_support';
  if (value >= 0.75) return 'strong_support';
  return 'mid_support';
}

function orderedCountEntries(counts, preferredOrder = []) {
  const orderedKeys = [];
  for (const key of preferredOrder) {
    if (!orderedKeys.includes(key)) orderedKeys.push(key);
  }
  for (const key of Object.keys(counts ?? {})) {
    if (!orderedKeys.includes(key)) orderedKeys.push(key);
  }
  return orderedKeys.map((key) => [key, counts?.[key] ?? 0]);
}

function resolveArtifactPath(surface, artifactName) {
  const candidates = dedupePaths([
    ...surface.source_files.filter((sourceFile) =>
      toPosixPath(sourceFile).endsWith(`/${artifactName}`),
    ),
    ...surface.source_files.map((sourceFile) =>
      path.posix.join(path.posix.dirname(toPosixPath(sourceFile)), artifactName),
    ),
    (() => {
      const evaluationManifestPath = resolveEvaluationManifestPath(surface);
      if (!evaluationManifestPath) return null;
      return path.posix.join(path.posix.dirname(evaluationManifestPath), artifactName);
    })(),
    path.posix.join(surface.bundle_path, artifactName),
  ]);
  return candidates.find((candidate) => fileExists(candidate)) ?? null;
}

function deriveSupportBinCountsFromResults(surface) {
  const resultsPath = resolveArtifactPath(surface, 'results.json');
  const byVariantCounts = {};
  const totalCounts = {};
  if (!resultsPath) {
    return {
      resultsPath: null,
      totalCounts,
      byVariantCounts,
    };
  }

  const payload = readJsonIfExists(resultsPath) ?? {};
  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  for (const row of rows) {
    const variantId = firstNonemptyText(row?.variant_id) ?? 'unknown_variant';
    const supportBin = classifySupportBin(row?.support_richness);
    incrementCount(totalCounts, supportBin);
    if (!isPlainObject(byVariantCounts[variantId])) {
      byVariantCounts[variantId] = {};
    }
    incrementCount(byVariantCounts[variantId], supportBin);
  }

  return {
    resultsPath,
    totalCounts,
    byVariantCounts,
  };
}

function accumulateNestedCounts(byVariant, fieldName) {
  const aggregate = {};
  for (const payload of Object.values(byVariant ?? {})) {
    if (!isPlainObject(payload?.[fieldName])) continue;
    for (const [key, value] of Object.entries(payload[fieldName])) {
      incrementCount(aggregate, key, Number.isFinite(value) ? Number(value) : 0);
    }
  }
  return aggregate;
}

function isCompositionTableSurface(surface) {
  return [
    'table.focused_voi.cohort_support_composition',
    'table.latest_checked_campaign.cohort_support_composition',
  ].includes(surface.surface_id);
}

function buildCompositionTable(surface) {
  const compositionPath = resolveArtifactPath(surface, 'cohort_composition.json');
  if (!compositionPath) {
    return {
      csvPath: null,
      columns: TABLE_COLUMNS[surface.surface_id] ?? [],
      rows: [],
      subtitle: 'No cohort-composition artifact was available.',
      sourceContext: {
        composition_path: null,
        results_path: null,
        support_bin_definitions: SUPPORT_BIN_DEFINITIONS,
        support_bin_counts_source: 'missing_composition_artifact',
      },
    };
  }

  const compositionPayload = readJsonIfExists(compositionPath) ?? {};
  const byVariant = isPlainObject(compositionPayload.by_variant) ? compositionPayload.by_variant : {};
  const derivedSupport = deriveSupportBinCountsFromResults(surface);
  const totalRowCount =
    Number.isFinite(compositionPayload.total_row_count)
      ? Number(compositionPayload.total_row_count)
      : Object.values(byVariant).reduce(
          (sum, payload) => sum + (Number.isFinite(payload?.row_count) ? Number(payload.row_count) : 0),
          0,
        );

  const aggregateCohortCounts = accumulateNestedCounts(byVariant, 'cohort_counts');
  const aggregateSupportCounts = isPlainObject(compositionPayload.support_bin_counts)
    ? compositionPayload.support_bin_counts
    : derivedSupport.totalCounts;
  const variantIds = sortVariantIds([
    ...Object.keys(byVariant),
    ...Object.keys(derivedSupport.byVariantCounts),
  ]);

  const rows = [];
  const pushCountRows = ({
    scope,
    variantId,
    compositionFamily,
    counts,
    referenceRowCount,
    preferredOrder = [],
  }) => {
    for (const [compositionLabel, rowCount] of orderedCountEntries(counts, preferredOrder)) {
      rows.push({
        scope,
        variant_id: variantId,
        composition_family: compositionFamily,
        composition_label: compositionLabel,
        row_count: Number.isFinite(rowCount) ? Number(rowCount) : 0,
        reference_row_count: Number.isFinite(referenceRowCount) ? Number(referenceRowCount) : 0,
      });
    }
  };

  pushCountRows({
    scope: 'lane_total',
    variantId: 'ALL',
    compositionFamily: 'cohort',
    counts: aggregateCohortCounts,
    referenceRowCount: totalRowCount,
  });
  pushCountRows({
    scope: 'lane_total',
    variantId: 'ALL',
    compositionFamily: 'support_bin',
    counts: aggregateSupportCounts,
    referenceRowCount: totalRowCount,
    preferredOrder: SUPPORT_BIN_ORDER,
  });

  for (const variantId of variantIds) {
    const payload = isPlainObject(byVariant[variantId]) ? byVariant[variantId] : {};
    const variantRowCount = Number.isFinite(payload.row_count)
      ? Number(payload.row_count)
      : orderedCountEntries(
          payload.support_bin_counts ?? derivedSupport.byVariantCounts[variantId] ?? {},
          SUPPORT_BIN_ORDER,
        ).reduce((sum, [, value]) => sum + (Number.isFinite(value) ? Number(value) : 0), 0);
    pushCountRows({
      scope: 'variant',
      variantId,
      compositionFamily: 'cohort',
      counts: payload.cohort_counts ?? {},
      referenceRowCount: variantRowCount,
    });
    pushCountRows({
      scope: 'variant',
      variantId,
      compositionFamily: 'support_bin',
      counts: payload.support_bin_counts ?? derivedSupport.byVariantCounts[variantId] ?? {},
      referenceRowCount: variantRowCount,
      preferredOrder: SUPPORT_BIN_ORDER,
    });
  }

  const supportCountSource = isPlainObject(compositionPayload.support_bin_counts)
    ? 'cohort_composition.json'
    : derivedSupport.resultsPath
      ? 'results.json_support_richness_derivation'
      : 'support_bins_unavailable';
  const subtitleParts = [`Rendered from ${compositionPath}`];
  if (supportCountSource === 'results.json_support_richness_derivation') {
    subtitleParts.push(
      `support-bin counts derived from ${derivedSupport.resultsPath} using weak<=0.45, mid<0.75, strong>=0.75`,
    );
  } else if (supportCountSource === 'cohort_composition.json') {
    subtitleParts.push('support-bin counts taken directly from cohort_composition.json');
  }

  return {
    csvPath: null,
    columns: TABLE_COLUMNS[surface.surface_id] ?? [],
    rows,
    subtitle: subtitleParts.join('; '),
    sourceContext: {
      composition_path: compositionPath,
      results_path: derivedSupport.resultsPath,
      support_bin_definitions:
        compositionPayload.support_bin_definitions ?? SUPPORT_BIN_DEFINITIONS,
      support_bin_counts_source: supportCountSource,
      total_row_count: totalRowCount,
    },
  };
}

function buildCompositionFigureSeries(surface) {
  const compositionTable = buildCompositionTable(surface);
  const laneTotalRows = compositionTable.rows.filter((row) => row.scope === 'lane_total');
  const topCohort = [...laneTotalRows]
    .filter((row) => row.composition_family === 'cohort')
    .sort((left, right) => right.row_count - left.row_count)[0];
  const topSupportBin = [...laneTotalRows]
    .filter((row) => row.composition_family === 'support_bin')
    .sort((left, right) => right.row_count - left.row_count)[0];
  const cohortMetricKey = topCohort
    ? `cohort_${sanitizeMetricKey(topCohort.composition_label)}_count`
    : null;
  const supportMetricKey = topSupportBin
    ? `support_${sanitizeMetricKey(topSupportBin.composition_label)}_count`
    : null;
  const variantRows = [];
  const variantIds = sortVariantIds(
    compositionTable.rows
      .filter((row) => row.scope === 'variant')
      .map((row) => row.variant_id),
  );

  for (const variantId of variantIds) {
    const variantRowsForId = compositionTable.rows.filter(
      (row) => row.scope === 'variant' && row.variant_id === variantId,
    );
    const variantRowCount =
      variantRowsForId.find(
        (row) =>
          row.composition_family === 'cohort' || row.composition_family === 'support_bin',
      )?.reference_row_count ?? 0;
    const row = {
      variant_id: variantId,
      row_count: variantRowCount,
    };
    if (topCohort && cohortMetricKey) {
      row[cohortMetricKey] =
        variantRowsForId.find(
          (item) =>
            item.composition_family === 'cohort' &&
            item.composition_label === topCohort.composition_label,
        )?.row_count ?? 0;
    }
    if (topSupportBin && supportMetricKey) {
      row[supportMetricKey] =
        variantRowsForId.find(
          (item) =>
            item.composition_family === 'support_bin' &&
            item.composition_label === topSupportBin.composition_label,
        )?.row_count ?? 0;
    }
    variantRows.push(row);
  }

  const subtitleParts = [compositionTable.subtitle];
  if (topCohort) subtitleParts.push(`cohort focus: ${topCohort.composition_label}`);
  if (topSupportBin) subtitleParts.push(`support-bin focus: ${topSupportBin.composition_label}`);
  return {
    selector: 'cohort_composition',
    jsonPath:
      compositionTable.sourceContext?.composition_path ??
      resolveArtifactPath(surface, 'cohort_composition.json'),
    labelKey: 'variant_id',
    numericKeys: ['row_count', cohortMetricKey, supportMetricKey].filter(Boolean),
    rows: variantRows,
    subtitle: subtitleParts.join('; '),
    sourceContext: compositionTable.sourceContext,
  };
}

function parseJsonText(text) {
  try {
    return JSON.parse(text);
  } catch (error) {
    const sanitized = String(text).replace(
      /([:\[,]\s*)(?:NaN|-Infinity|Infinity)(?=\s*[,}\]])/g,
      '$1null',
    );
    return JSON.parse(sanitized);
  }
}

function readJson(relativePath) {
  return parseJsonText(fs.readFileSync(path.join(repoRoot, relativePath), 'utf8'));
}

function readJsonIfExists(relativePath) {
  if (!relativePath || !fileExists(relativePath)) return null;
  try {
    const payload = readJson(relativePath);
    return isPlainObject(payload) ? payload : null;
  } catch {
    return null;
  }
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
    } else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n') {
      row.push(field.replace(/\r$/, ''));
      rows.push(row);
      row = [];
      field = '';
    } else {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field.replace(/\r$/, ''));
    rows.push(row);
  }

  if (rows.length === 0) return [];
  const header = rows[0];
  return rows.slice(1).map((cells) =>
    Object.fromEntries(header.map((key, index) => [key, cells[index] ?? ''])),
  );
}

function readCsv(relativePath) {
  return parseCsv(fs.readFileSync(path.join(repoRoot, relativePath), 'utf8'));
}

function isFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatTableNumber(value) {
  return Number.isFinite(value) ? Number(value.toFixed(6)) : 'n/a';
}

function csvCell(value) {
  const text = String(value ?? '');
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function serializeCsv(columns, rows) {
  return `${[columns, ...rows.map((row) => columns.map((column) => csvCell(row[column])))]
    .map((row) => row.join(','))
    .join('\n')}\n`;
}

function findBrowserExecutable() {
  const candidates = [
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

function relativeOutputPath(fileName) {
  return `${SECTION_EXPORT_ROOT}/${fileName}`.replaceAll('\\', '/');
}

function stripNoRenderClaim(note) {
  return note
    .replace(/\s*No committed PDF or SVG render is claimed for this slice\./gi, '')
    .replace(/\s*This slice does not claim a committed rendered figure\./gi, '')
    .replace(/\s*The checked local source is plot-ready JSON, not a committed rendered PDF\/SVG\./gi, '')
    .replace(
      /\s*Checked SVG, PDF, and print-ready HTML exports exist under out\/headline_exports\/current_checked\/ for this headline surface\./gi,
      '',
    )
    .replace(
      /\s*Checked SVG, PDF, (?:and )?print-ready HTML[^.]*for this headline surface\./gi,
      '',
    )
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function firstNonemptyText(...values) {
  for (const value of values) {
    if (value === null || value === undefined) continue;
    const text = String(value).trim();
    if (text) return text;
  }
  return null;
}

function normalizeStableJsonValue(value) {
  if (value === undefined) return null;
  if (Array.isArray(value)) return value.map((item) => normalizeStableJsonValue(item));
  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, normalizeStableJsonValue(value[key])]),
    );
  }
  return value;
}

function stableJson(payload) {
  return JSON.stringify(normalizeStableJsonValue(payload ?? {}));
}

function normalizePolicyName(policyName) {
  const key = String(policyName ?? '').trim().toLowerCase();
  return key || 'unspecified';
}

function buildPolicyHash(policyName, { version = 'v1', configuration = null } = {}) {
  const material = [
    normalizePolicyName(policyName),
    String(version ?? 'v1').trim().toLowerCase() || 'v1',
    stableJson(configuration),
  ].join('|');
  return crypto.createHash('sha1').update(material, 'utf8').digest('hex');
}

function stablePolicyHash(policyName, { version, configuration = null } = {}) {
  const normalizedVersion = firstNonemptyText(version);
  if (!normalizedVersion) return null;
  if (['unknown', 'untracked', 'inactive'].includes(normalizedVersion.toLowerCase())) {
    return null;
  }
  return buildPolicyHash(policyName, {
    version: normalizedVersion,
    configuration,
  });
}

function hexFileHash(relativePath, algorithm = 'sha256') {
  try {
    return crypto
      .createHash(algorithm)
      .update(fs.readFileSync(path.join(repoRoot, relativePath)))
      .digest('hex');
  } catch {
    return null;
  }
}

function resolveGitCommitHash() {
  if (cachedGitCommitHash !== undefined) return cachedGitCommitHash;
  try {
    const result = spawnSync('git', ['rev-parse', 'HEAD'], {
      cwd: repoRoot,
      encoding: 'utf8',
      timeout: 5000,
    });
    const commitHash = String(result.stdout ?? '').trim().toLowerCase();
    cachedGitCommitHash =
      /^[0-9a-f]{40}$/.test(commitHash) && result.status === 0 ? commitHash : null;
  } catch {
    cachedGitCommitHash = null;
  }
  return cachedGitCommitHash;
}

function resolveEnvironmentLockfileHash() {
  if (cachedEnvironmentLockfile !== undefined) return cachedEnvironmentLockfile;
  for (const candidate of ['backend/uv.lock', 'backend/poetry.lock', 'backend/requirements.lock']) {
    if (!fileExists(candidate)) continue;
    cachedEnvironmentLockfile = {
      environment_lockfile_hash: hexFileHash(candidate, 'sha256'),
      environment_lockfile_path: toPosixPath(candidate),
    };
    return cachedEnvironmentLockfile;
  }
  cachedEnvironmentLockfile = {
    environment_lockfile_hash: null,
    environment_lockfile_path: null,
  };
  return cachedEnvironmentLockfile;
}

function resolveContainerIdentity() {
  if (cachedContainerIdentity !== undefined) return cachedContainerIdentity;
  for (const envName of CONTAINER_DIGEST_ENV_VARS) {
    const value = String(process.env[envName] ?? '').trim();
    if (value) {
      cachedContainerIdentity = {
        container_digest: value,
        container_digest_source: `env:${envName}`,
      };
      return cachedContainerIdentity;
    }
  }
  cachedContainerIdentity = {
    container_digest: null,
    container_digest_source: 'unavailable_local_runtime',
  };
  return cachedContainerIdentity;
}

function thesisPolicyHashes(metadata, evaluationManifest) {
  const evaluationSuite =
    (isPlainObject(metadata?.evaluation_suite) && metadata.evaluation_suite) ||
    (isPlainObject(evaluationManifest?.evaluation_suite) && evaluationManifest.evaluation_suite) ||
    {};
  const laneRole = firstNonemptyText(evaluationSuite.role, 'thesis_evaluation') ?? 'thesis_evaluation';
  const strictEvidencePolicy = firstNonemptyText(
    metadata?.strict_evidence_policy,
    evaluationManifest?.strict_evidence_policy,
  );
  const orsBaselinePolicy = firstNonemptyText(
    metadata?.ors_baseline_policy,
    evaluationManifest?.ors_baseline_policy,
  );
  const cacheMode = firstNonemptyText(
    metadata?.cache_mode,
    evaluationManifest?.cache_mode,
    'mixed',
  ) ?? 'mixed';
  const cacheResetScope = firstNonemptyText(
    metadata?.cache_reset_scope,
    evaluationManifest?.cache_reset_scope,
    'none',
  ) ?? 'none';
  const cacheResetPolicy = firstNonemptyText(
    metadata?.cache_reset_policy,
    evaluationManifest?.cache_reset_policy,
    'none',
  ) ?? 'none';
  const modelVersion =
    firstNonemptyText(evaluationManifest?.model_version, 'thesis-script-untracked') ??
    'thesis-script-untracked';
  const snapshotMode =
    firstNonemptyText(evaluationManifest?.ors_snapshot_mode, 'off') ?? 'off';

  return {
    strict_evidence_policy_hash: stablePolicyHash('strict_evidence_policy', {
      version: strictEvidencePolicy,
      configuration: {
        lane_id: laneRole,
      },
    }),
    baseline_engine_policy_hash: stablePolicyHash('baseline_engine_policy', {
      version: orsBaselinePolicy,
      configuration: {
        lane_id: laneRole,
        ors_snapshot_mode: snapshotMode,
      },
    }),
    evaluation_lane_policy_hash: stablePolicyHash('evaluation_lane_policy', {
      version: modelVersion,
      configuration: {
        role: laneRole,
        scope: evaluationSuite.scope ?? null,
        focus: evaluationSuite.focus ?? null,
        strict_proxy_ors_allowed: metadata?.strict_proxy_ors_allowed ?? null,
        strict_evidence_fallbacks_allowed: metadata?.strict_evidence_fallbacks_allowed ?? null,
      },
    }),
    cache_policy_hash: buildPolicyHash('evaluation_cache_policy', {
      version: 'thesis-eval-cache-policy-v1',
      configuration: {
        cache_mode: cacheMode,
        cache_reset_scope: cacheResetScope,
        cache_reset_policy: cacheResetPolicy,
        cache_carryover_expected: metadata?.cache_carryover_expected ?? null,
      },
    }),
  };
}

function normalizeHeadlineIdentity(headlineIdentity) {
  const containerDigestSource =
    firstNonemptyText(headlineIdentity?.container_digest_source, 'unavailable_local_runtime') ??
    'unavailable_local_runtime';
  return {
    schema_version:
      firstNonemptyText(headlineIdentity?.schema_version, HEADLINE_IDENTITY_SCHEMA_VERSION) ??
      HEADLINE_IDENTITY_SCHEMA_VERSION,
    git_commit_hash: headlineIdentity?.git_commit_hash ?? null,
    environment_lockfile_hash: headlineIdentity?.environment_lockfile_hash ?? null,
    environment_lockfile_path: headlineIdentity?.environment_lockfile_path ?? null,
    container_digest: headlineIdentity?.container_digest ?? null,
    container_digest_source: containerDigestSource,
    policy_hashes: isPlainObject(headlineIdentity?.policy_hashes)
      ? headlineIdentity.policy_hashes
      : {},
  };
}

function dedupePaths(paths) {
  return [...new Set(paths.filter(Boolean).map((relativePath) => toPosixPath(relativePath)))];
}

function resolveEvaluationManifestPath(surface) {
  const indexedPath = surface.source_files.find((sourceFile) =>
    toPosixPath(sourceFile).endsWith('/evaluation_manifest.json'),
  );
  if (indexedPath && fileExists(indexedPath)) return toPosixPath(indexedPath);
  const bundleCandidate = toPosixPath(path.posix.join(surface.bundle_path, 'evaluation_manifest.json'));
  if (fileExists(bundleCandidate)) return bundleCandidate;
  return null;
}

function resolveMetadataPath(surface, evaluationManifestPath) {
  const indexedPath = surface.source_files.find((sourceFile) =>
    toPosixPath(sourceFile).endsWith('/metadata.json'),
  );
  if (indexedPath && fileExists(indexedPath)) return toPosixPath(indexedPath);
  if (evaluationManifestPath) {
    const siblingCandidate = toPosixPath(
      path.posix.join(path.posix.dirname(evaluationManifestPath), 'metadata.json'),
    );
    if (fileExists(siblingCandidate)) return siblingCandidate;
  }
  const bundleCandidate = toPosixPath(path.posix.join(surface.bundle_path, 'metadata.json'));
  if (fileExists(bundleCandidate)) return bundleCandidate;
  return null;
}

function resolveHeadlineProvenance(surface) {
  const embeddedCandidates = dedupePaths([
    ...surface.source_files.filter((sourceFile) => toPosixPath(sourceFile).endsWith('.json')),
    path.posix.join(surface.bundle_path, 'evaluation_manifest.json'),
    path.posix.join(surface.bundle_path, 'thesis_summary.json'),
    path.posix.join(surface.bundle_path, 'thesis_summary_by_cohort.json'),
    path.posix.join(surface.bundle_path, 'thesis_plots.json'),
  ]);

  for (const relativePath of embeddedCandidates) {
    const payload = readJsonIfExists(relativePath);
    const headlineIdentity = isPlainObject(payload?.artifact_provenance?.headline_identity)
      ? normalizeHeadlineIdentity(payload.artifact_provenance.headline_identity)
      : null;
    if (headlineIdentity) {
      return {
        headlineIdentity,
        provenanceSourceMode: 'copied_from_artifact_provenance_headline_identity',
        provenanceSourceFiles: [relativePath],
      };
    }
  }

  const evaluationManifestPath = resolveEvaluationManifestPath(surface);
  if (!evaluationManifestPath) {
    throw new Error(`No evaluation_manifest.json available for headline surface ${surface.surface_id}`);
  }
  const metadataPath = resolveMetadataPath(surface, evaluationManifestPath);
  const evaluationManifest = readJsonIfExists(evaluationManifestPath) ?? {};
  const metadata = metadataPath ? readJsonIfExists(metadataPath) ?? {} : {};
  const { environment_lockfile_hash, environment_lockfile_path } =
    resolveEnvironmentLockfileHash();
  const { container_digest, container_digest_source } = resolveContainerIdentity();

  return {
    headlineIdentity: {
      schema_version: HEADLINE_IDENTITY_SCHEMA_VERSION,
      git_commit_hash: resolveGitCommitHash(),
      environment_lockfile_hash,
      environment_lockfile_path,
      container_digest,
      container_digest_source,
      policy_hashes: thesisPolicyHashes(metadata, evaluationManifest),
    },
    provenanceSourceMode: 'reconstructed_from_evaluation_manifest_and_metadata',
    provenanceSourceFiles: dedupePaths([evaluationManifestPath, metadataPath]),
  };
}

function runtimeMemoryExtremaByVariant(surface) {
  const resultsPath = resolveArtifactPath(surface, 'thesis_results.json');
  if (!resultsPath) {
    return new Map();
  }
  const payload = readJsonIfExists(resultsPath) ?? {};
  const resultRows = Array.isArray(payload.rows) ? payload.rows : [];
  const extremaByVariant = new Map();
  for (const row of resultRows) {
    const variantId = typeof row?.variant_id === 'string' ? row.variant_id : '';
    if (!variantId) continue;
    const current = extremaByVariant.get(variantId) ?? {
      max_process_rss_mb: null,
      max_process_vms_mb: null,
    };
    const rss = typeof row?.process_rss_mb === 'number' && Number.isFinite(row.process_rss_mb)
      ? row.process_rss_mb
      : null;
    const vms = typeof row?.process_vms_mb === 'number' && Number.isFinite(row.process_vms_mb)
      ? row.process_vms_mb
      : null;
    if (rss !== null) {
      current.max_process_rss_mb =
        current.max_process_rss_mb === null
          ? rss
          : Math.max(current.max_process_rss_mb, rss);
    }
    if (vms !== null) {
      current.max_process_vms_mb =
        current.max_process_vms_mb === null
          ? vms
          : Math.max(current.max_process_vms_mb, vms);
    }
    extremaByVariant.set(variantId, current);
  }
  return extremaByVariant;
}

function buildTableRows(surface) {
  if (surface.surface_id === 'table.latest_checked_campaign.summary_and_metrics') {
    const csvPath = surface.source_files.find((sourceFile) => sourceFile.endsWith('.csv'));
    if (!csvPath) {
      return {
        csvPath: null,
        columns: TABLE_COLUMNS[surface.surface_id] ?? [],
        rows: [],
        subtitle: 'No CSV source was available; exported source-file inventory instead.',
        sourceContext: null,
      };
    }

    const summaryRows = readCsv(csvPath);
    const rowsByVariant = new Map(summaryRows.map((row) => [String(row.variant_id ?? ''), row]));
    const metricDefinitions = [
      ['Weighted win vs OSRM', 'weighted_win_rate_osrm'],
      ['Weighted win vs ORS', 'weighted_win_rate_ors'],
      ['Weighted win vs `V0`', 'weighted_win_rate_v0'],
      ['Mean runtime ms', 'mean_runtime_ms'],
      ['Mean algorithm ms', 'mean_algorithm_runtime_ms'],
      ['Mean certificate', 'mean_certificate'],
      ['Mean frontier count', 'mean_frontier_count'],
    ];
    const variantIds = ['V0', 'A', 'B', 'C'];
    const columns = TABLE_COLUMNS[surface.surface_id] ?? [
      'metric',
      'V0',
      'A',
      'B',
      'C',
      'A-V0',
      'B-A',
      'C-B',
    ];
    const rows = metricDefinitions.map(([metric, field]) => {
      const values = Object.fromEntries(
        variantIds.map((variantId) => [variantId, isFiniteNumber(rowsByVariant.get(variantId)?.[field])]),
      );
      return {
        metric,
        V0: formatTableNumber(values.V0),
        A: formatTableNumber(values.A),
        B: formatTableNumber(values.B),
        C: formatTableNumber(values.C),
        'A-V0': formatTableNumber(values.V0 !== null && values.A !== null ? values.A - values.V0 : null),
        'B-A': formatTableNumber(values.A !== null && values.B !== null ? values.B - values.A : null),
        'C-B': formatTableNumber(values.B !== null && values.C !== null ? values.C - values.B : null),
      };
    });
    return {
      csvPath: null,
      sourceCsvText: serializeCsv(columns, rows),
      columns,
      rows,
      subtitle: `Rendered from ${csvPath} as a variant comparison matrix with adjacent delta columns.`,
      sourceContext: {
        source_csv_path: csvPath,
        metric_count: rows.length,
        variant_order: variantIds,
      },
    };
  }

  if (surface.surface_id === 'table.latest_checked_campaign.runtime_stage_quantiles') {
    const resultsPath = resolveArtifactPath(surface, 'thesis_results.json');
    if (!resultsPath) {
      return {
        csvPath: null,
        columns: TABLE_COLUMNS[surface.surface_id] ?? [],
        rows: [],
        subtitle: 'No stage-timing artifact was available; exported source-file inventory instead.',
        sourceContext: null,
      };
    }

    const payload = readJson(resultsPath) ?? {};
    const resultRows = Array.isArray(payload.rows) ? payload.rows : [];
    const stageDefinitions = [
      ['option_build', 'stage_option_build_ms'],
      ['k_raw', 'stage_k_raw_ms'],
      ['dccs', 'stage_dccs_ms'],
      ['refinement', 'stage_refinement_ms'],
      ['pareto', 'stage_pareto_ms'],
      ['refc', 'stage_refc_ms'],
      ['voi', 'stage_voi_ms'],
      ['supplemental_rescue', 'stage_supplemental_rescue_ms'],
      ['preemptive_comparator_seed', 'stage_preemptive_comparator_seed_ms'],
    ];
    const columns = TABLE_COLUMNS[surface.surface_id] ?? ['stage', 'row_count', 'p50_ms', 'p90_ms', 'p95_ms'];
    const rows = stageDefinitions.map(([label, field]) => {
      const values = resultRows
        .map((row) => Number(row?.[field]))
        .filter((value) => Number.isFinite(value));
      return {
        stage: label,
        row_count: values.length,
        p50_ms: formatTableNumber(quantile(values, 0.5)),
        p90_ms: formatTableNumber(quantile(values, 0.9)),
        p95_ms: formatTableNumber(quantile(values, 0.95)),
      };
    });
    return {
      csvPath: null,
      sourceCsvText: serializeCsv(columns, rows),
      columns,
      rows,
      subtitle: `Aggregated per-stage timing quantiles rendered from ${resultsPath}.`,
      sourceContext: {
        results_path: resultsPath,
        stage_fields: stageDefinitions.map(([, field]) => field),
        stage_count: rows.length,
      },
    };
  }

  if (isCompositionTableSurface(surface)) {
    return buildCompositionTable(surface);
  }

  const csvPath = surface.source_files.find((sourceFile) => sourceFile.endsWith('.csv'));
  if (!csvPath) {
    return {
      csvPath: null,
      columns: ['source_files'],
      rows: [{ source_files: surface.source_files.join(' | ') }],
      subtitle: 'No CSV source was available; exported source-file inventory instead.',
      sourceContext: null,
    };
  }

  const rows = readCsv(csvPath);
  const runtimeMemoryExtrema = surface.surface_id.startsWith(
    'table.latest_checked_campaign.runtime_',
  )
    ? runtimeMemoryExtremaByVariant(surface)
    : null;
  const configuredColumns =
    TABLE_COLUMNS[surface.surface_id] ?? Object.keys(rows[0] ?? {}).slice(0, 6);
  return {
    csvPath,
    columns: configuredColumns,
    rows: rows.map((row) => {
      const variantId = typeof row.variant_id === 'string' ? row.variant_id : '';
      const runtimeExtrema = runtimeMemoryExtrema?.get(variantId) ?? null;
      return Object.fromEntries(
        configuredColumns.map((column) => {
          if ((column === 'max_process_rss_mb' || column === 'max_process_vms_mb') && runtimeExtrema) {
            return [column, runtimeExtrema[column] ?? ''];
          }
          return [column, row[column] ?? ''];
        }),
      );
    }),
    subtitle: `Rendered from ${csvPath}`,
    sourceContext: null,
  };
}

function buildFigureSeries(surface) {
  const selector = surface.source_selector?.replace(/^\$\./, '');
  if (selector === 'runtime_stage_quantiles_vs_stage') {
    const resultsPath = surface.source_files.find((sourceFile) =>
      sourceFile.endsWith('thesis_results.json'),
    );
    if (!resultsPath) {
      return {
        selector,
        jsonPath: null,
        labelKey: 'stage',
        numericKeys: [],
        rows: [],
        subtitle: 'No stage-timing artifact was available; exported source-file inventory instead.',
        sourceContext: null,
      };
    }

    const payload = readJson(resultsPath) ?? {};
    const resultRows = Array.isArray(payload.rows) ? payload.rows : [];
    const stageDefinitions = [
      ['option_build', 'stage_option_build_ms'],
      ['k_raw', 'stage_k_raw_ms'],
      ['dccs', 'stage_dccs_ms'],
      ['refinement', 'stage_refinement_ms'],
      ['pareto', 'stage_pareto_ms'],
      ['refc', 'stage_refc_ms'],
      ['voi', 'stage_voi_ms'],
      ['supplemental_rescue', 'stage_supplemental_rescue_ms'],
      ['preemptive_comparator_seed', 'stage_preemptive_comparator_seed_ms'],
    ];
    return {
      selector,
      jsonPath: resultsPath,
      labelKey: 'stage',
      numericKeys: ['p50_ms', 'p90_ms', 'p95_ms'],
      rows: stageDefinitions.map(([label, field]) => {
        const values = resultRows
          .map((row) => Number(row?.[field]))
          .filter((value) => Number.isFinite(value));
        return {
          stage: label,
          row_count: values.length,
          p50_ms: quantile(values, 0.5),
          p90_ms: quantile(values, 0.9),
          p95_ms: quantile(values, 0.95),
        };
      }),
      subtitle: `Aggregated from ${resultsPath} across ${resultRows.length} lane rows.`,
      sourceContext: {
        results_path: resultsPath,
        stage_fields: stageDefinitions.map(([, field]) => field),
        stage_count: stageDefinitions.length,
      },
    };
  }
  const jsonPath = surface.source_files.find((sourceFile) => sourceFile.endsWith('thesis_plots.json'));
  if (!selector || !jsonPath) {
    return {
      selector: selector ?? 'unknown',
      jsonPath: jsonPath ?? null,
      labelKey: 'index',
      numericKeys: [],
      rows: [],
      subtitle: 'No plot selector source was available; exported source-file inventory instead.',
      sourceContext: null,
    };
  }

  const plots = readJson(jsonPath);
  if (selector === 'cohort_composition' && isPlainObject(plots?.[selector])) {
    return buildCompositionFigureSeries(surface);
  }
  let rows = Array.isArray(plots?.[selector]) ? plots[selector] : [];
  if (selector === 'runtime_distribution_vs_variant') {
    const runtimeMemoryExtrema = runtimeMemoryExtremaByVariant(surface);
    rows = rows.map((row) => {
      const variantId = typeof row?.variant_id === 'string' ? row.variant_id : '';
      const runtimeExtrema = runtimeMemoryExtrema.get(variantId) ?? {};
      return {
        ...row,
        max_process_rss_mb:
          row?.max_process_rss_mb ?? runtimeExtrema.max_process_rss_mb ?? null,
        max_process_vms_mb:
          row?.max_process_vms_mb ?? runtimeExtrema.max_process_vms_mb ?? null,
      };
    });
  }
  const labelKey =
    ['variant_id', 'cohort_label', 'cohort', 'label', 'id'].find((key) =>
      rows.some((row) => row && typeof row[key] === 'string' && row[key].trim()),
    ) ?? 'index';
  const configuredNumericKeysBySelector = {
    runtime_distribution_vs_variant: [
      'mean_runtime_p50_ms',
      'mean_runtime_p90_ms',
      'mean_runtime_p95_ms',
      'mean_process_rss_p90_mb',
      'mean_process_vms_p90_mb',
      'max_process_rss_mb',
      'max_process_vms_mb',
    ],
    runtime_breakdown_vs_variant: [
      'mean_stage_k_raw_ms',
      'mean_stage_dccs_ms',
      'mean_stage_refinement_ms',
      'mean_stage_pareto_ms',
      'mean_stage_refc_ms',
      'mean_stage_voi_ms',
      'mean_runtime_per_refined_candidate_ms',
      'mean_runtime_per_frontier_member_ms',
      'mean_process_rss_mb',
      'mean_process_vms_mb',
      'mean_search_budget_utilization_p90',
      'mean_evidence_budget_utilization_p90',
      'mean_graph_low_ambiguity_fast_path_rate',
      'mean_graph_supported_ambiguity_fast_fallback_rate',
    ],
  };
  const numericKeys = (configuredNumericKeysBySelector[selector] ?? Object.keys(rows[0] ?? {})).filter(
    (key) =>
      key !== labelKey &&
      rows.some((row) => typeof row?.[key] === 'number' && Number.isFinite(row[key])),
  );
  return {
    selector,
    jsonPath,
    labelKey,
    numericKeys,
    rows,
    subtitle: `Rendered from ${jsonPath} selector $.${selector}`,
    sourceContext: null,
  };
}

function writeSourceCompanionForTable(surface, table, stem) {
  const shouldWriteJsonCompanion = surface.surface_id.startsWith(
    'table.latest_checked_campaign.runtime_',
  );
  if (table.sourceCsvText) {
    const companionName = `${stem}.source.csv`;
    const companionPath = path.join(outputDir, companionName);
    fs.writeFileSync(
      companionPath,
      table.sourceCsvText.endsWith('\n') ? table.sourceCsvText : `${table.sourceCsvText}\n`,
      'utf8',
    );
    if (shouldWriteJsonCompanion) {
      writeTableSourceJsonCompanion(surface, table, stem);
    }
    return {
      relativePath: relativeOutputPath(companionName),
      format: 'csv',
      label: 'CSV',
    };
  }

  if (table.csvPath) {
    const csvText = fs.readFileSync(path.join(repoRoot, table.csvPath), 'utf8');
    const companionName = `${stem}.source.csv`;
    const companionPath = path.join(outputDir, companionName);
    fs.writeFileSync(
      companionPath,
      csvText.endsWith('\n') ? csvText : `${csvText}\n`,
      'utf8',
    );
    if (shouldWriteJsonCompanion) {
      writeTableSourceJsonCompanion(surface, table, stem);
    }
    return {
      relativePath: relativeOutputPath(companionName),
      format: 'csv',
      label: 'CSV',
    };
  }

  removeIfExists(path.join(outputDir, `${stem}.source.csv`));
  const companionName = `${stem}.source.json`;
  const companionPath = path.join(outputDir, companionName);
  fs.writeFileSync(
    companionPath,
    `${JSON.stringify(
      {
        surface_id: surface.surface_id,
        source_files: surface.source_files,
        columns: table.columns,
        subtitle: table.subtitle,
        source_context: table.sourceContext ?? null,
        rows: table.rows,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
  return {
    relativePath: relativeOutputPath(companionName),
    format: 'json',
    label: 'JSON',
  };
}

function writeTableSourceJsonCompanion(surface, table, stem) {
  const companionName = `${stem}.source.json`;
  const companionPath = path.join(outputDir, companionName);
  const payload = {
    surface_id: surface.surface_id,
    source_files: surface.source_files,
    columns: table.columns,
    subtitle: table.subtitle,
    source_context: table.sourceContext ?? null,
    rows: table.rows,
  };
  fs.writeFileSync(companionPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  return {
    relativePath: relativeOutputPath(companionName),
    format: 'json',
    label: 'JSON',
  };
}

function writeSourceCompanionForFigure(surface, figure, stem) {
  removeIfExists(path.join(outputDir, `${stem}.source.csv`));
  const companionName = `${stem}.source.json`;
  const companionPath = path.join(outputDir, companionName);
  fs.writeFileSync(
    companionPath,
    `${JSON.stringify(
      {
        surface_id: surface.surface_id,
        source_files: surface.source_files,
        source_selector: surface.source_selector ?? null,
        rendered_from: figure.jsonPath,
        label_key: figure.labelKey,
        numeric_keys: figure.numericKeys,
        subtitle: figure.subtitle,
        source_context: figure.sourceContext ?? null,
        rows: figure.rows,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
  return {
    relativePath: relativeOutputPath(companionName),
    format: 'json',
    label: 'JSON',
  };
}

function writeProvenanceCompanionForSurface(surface, stem) {
  const {
    headlineIdentity,
    provenanceSourceMode,
    provenanceSourceFiles,
  } = resolveHeadlineProvenance(surface);
  const companionName = `${stem}.provenance.json`;
  const companionPath = path.join(outputDir, companionName);
  const payload = {
    schema_version: 'headline-export-provenance-sidecar-v1',
    surface_id: surface.surface_id,
    title: surface.title,
    bundle_path: surface.bundle_path,
    provenance_source_mode: provenanceSourceMode,
    provenance_source_files: provenanceSourceFiles,
    git_commit_hash: headlineIdentity.git_commit_hash,
    environment_lockfile_hash: headlineIdentity.environment_lockfile_hash,
    environment_lockfile_path: headlineIdentity.environment_lockfile_path,
    container_identity: {
      container_digest: headlineIdentity.container_digest,
      container_digest_source: headlineIdentity.container_digest_source,
    },
    policy_hashes: headlineIdentity.policy_hashes,
    headline_identity: headlineIdentity,
  };
  fs.writeFileSync(companionPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  return {
    relativePath: relativeOutputPath(companionName),
    format: 'json',
    label: 'JSON',
  };
}

function writeLaneArtifactGenerationSummary(targetRoot) {
  const rows = [];
  for (const source of LANE_ARTIFACT_GENERATION_SOURCES) {
    const thesisMetricsPath = path.join(source.sourceRoot, 'thesis_metrics.json');
    if (!fs.existsSync(thesisMetricsPath)) continue;
    const thesisMetrics = parseJsonText(fs.readFileSync(thesisMetricsPath, 'utf8'));
    const publishabilitySummary = isPlainObject(thesisMetrics?.publishability_summary)
      ? thesisMetrics.publishability_summary
      : {};
    const artifactGenerationMs =
      typeof publishabilitySummary.artifact_generation_ms === 'number' &&
      Number.isFinite(publishabilitySummary.artifact_generation_ms)
        ? publishabilitySummary.artifact_generation_ms
        : null;
    rows.push({
      lane_role:
        typeof publishabilitySummary.lane_role === 'string' && publishabilitySummary.lane_role.trim()
          ? publishabilitySummary.lane_role
          : source.lane_role,
      run_id:
        typeof thesisMetrics?.run_id === 'string' && thesisMetrics.run_id.trim()
          ? thesisMetrics.run_id
          : null,
      artifact_generation_ms: artifactGenerationMs,
      artifact_generation_seconds:
        artifactGenerationMs === null ? null : Number((artifactGenerationMs / 1000).toFixed(6)),
      artifact_generation_scope:
        typeof publishabilitySummary.artifact_generation_scope === 'string'
          ? publishabilitySummary.artifact_generation_scope
          : null,
      publishability_plot_family:
        typeof publishabilitySummary.publishability_plot_family === 'string'
          ? publishabilitySummary.publishability_plot_family
          : null,
      thesis_metrics_json: relativeOutputPath(
        path.join(source.companionDirName, 'thesis_metrics.json'),
      ),
    });
  }

  const columns = [
    'lane_role',
    'run_id',
    'artifact_generation_ms',
    'artifact_generation_seconds',
    'artifact_generation_scope',
    'publishability_plot_family',
    'thesis_metrics_json',
  ];
  fs.writeFileSync(
    path.join(targetRoot, 'lane_artifact_generation_summary.csv'),
    serializeCsv(columns, rows),
    'utf8',
  );
  fs.writeFileSync(
    path.join(targetRoot, 'lane_artifact_generation_summary.json'),
    `${JSON.stringify({ rows }, null, 2)}\n`,
    'utf8',
  );
  const markdownLines = [
    '# Lane Artifact-Generation Summary',
    '',
    'Per-lane artifact-generation time from the checked reviewer-package companion bundles. This separates evaluator artifact/report rendering cost from route/controller runtime cost.',
    '',
    '| Lane role | Run ID | Artifact generation (ms) | Artifact generation (s) | Scope | Plot family | Thesis metrics |',
    '| --- | --- | ---: | ---: | --- | --- | --- |',
    ...rows.map(
      (row) =>
        `| ${row.lane_role ?? ''} | ${row.run_id ?? ''} | ${formatValue(row.artifact_generation_ms)} | ${formatValue(row.artifact_generation_seconds)} | ${row.artifact_generation_scope ?? ''} | ${row.publishability_plot_family ?? ''} | \`${row.thesis_metrics_json ?? ''}\` |`,
    ),
    '',
  ];
  fs.writeFileSync(
    path.join(targetRoot, 'lane_artifact_generation_summary.md'),
    `${markdownLines.join('\n')}\n`,
    'utf8',
  );
}

function buildTableSvg({ title, subtitle, columns, rows }) {
  const rowHeight = 30;
  const left = 24;
  const top = 88;
  const columnWidth = 190;
  const width = Math.max(980, left * 2 + columns.length * columnWidth);
  const visibleRows = rows.slice(0, 18);
  const height = top + (visibleRows.length + 1) * rowHeight + 70;
  const elements = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#f8fafc"/>`,
    `<text x="${left}" y="38" font-size="24" font-family="Georgia, 'Times New Roman', serif" font-weight="700" fill="#0f172a">${escapeXml(title)}</text>`,
    `<text x="${left}" y="62" font-size="13" font-family="'Segoe UI', Arial, sans-serif" fill="#475569">${escapeXml(subtitle)}</text>`,
    `<rect x="${left}" y="${top}" width="${columns.length * columnWidth}" height="${rowHeight}" fill="#dbeafe" stroke="#93c5fd"/>`,
  ];

  columns.forEach((column, columnIndex) => {
    const x = left + columnIndex * columnWidth;
    elements.push(
      `<text x="${x + 10}" y="${top + 20}" font-size="13" font-family="'Segoe UI', Arial, sans-serif" font-weight="700" fill="#1e3a8a">${escapeXml(column)}</text>`,
    );
    elements.push(
      `<line x1="${x}" y1="${top}" x2="${x}" y2="${height - 32}" stroke="#cbd5e1" stroke-width="1"/>`,
    );
  });
  elements.push(
    `<line x1="${left + columns.length * columnWidth}" y1="${top}" x2="${left + columns.length * columnWidth}" y2="${height - 32}" stroke="#cbd5e1" stroke-width="1"/>`,
  );

  visibleRows.forEach((row, rowIndex) => {
    const y = top + rowHeight * (rowIndex + 1);
    const fill = rowIndex % 2 === 0 ? '#ffffff' : '#f8fafc';
    elements.push(
      `<rect x="${left}" y="${y}" width="${columns.length * columnWidth}" height="${rowHeight}" fill="${fill}" stroke="#e2e8f0"/>`,
    );
    columns.forEach((column, columnIndex) => {
      const x = left + columnIndex * columnWidth;
      elements.push(
        `<text x="${x + 10}" y="${y + 20}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#0f172a">${escapeXml(formatValue(row[column]))}</text>`,
      );
    });
  });

  if (rows.length > visibleRows.length) {
    elements.push(
      `<text x="${left}" y="${height - 14}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#64748b">Showing ${visibleRows.length} of ${rows.length} rows. Use the CSV source for the full table payload.</text>`,
    );
  } else {
    elements.push(
      `<text x="${left}" y="${height - 14}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#64748b">Checked SVG export generated from the current headline table source.</text>`,
    );
  }

  elements.push('</svg>');
  return elements.join('\n');
}

function buildFigureSvg({ title, subtitle, selector, rows, labelKey, numericKeys }) {
  if (!rows.length || !numericKeys.length) {
    return [
      `<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="220" viewBox="0 0 1100 220">`,
      `<rect width="1100" height="220" fill="#f8fafc"/>`,
      `<text x="24" y="38" font-size="24" font-family="Georgia, 'Times New Roman', serif" font-weight="700" fill="#0f172a">${escapeXml(title)}</text>`,
      `<text x="24" y="62" font-size="13" font-family="'Segoe UI', Arial, sans-serif" fill="#475569">${escapeXml(subtitle)}</text>`,
      `<text x="24" y="120" font-size="14" font-family="'Segoe UI', Arial, sans-serif" fill="#0f172a">No numeric plot series were available for selector ${escapeXml(selector)}.</text>`,
      `</svg>`,
    ].join('\n');
  }

  const sectionHeight = 180;
  const width = 1200;
  const height = 90 + numericKeys.length * sectionHeight;
  const colors = ['#0f766e', '#2563eb', '#b45309'];
  const elements = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#f8fafc"/>`,
    `<text x="24" y="38" font-size="24" font-family="Georgia, 'Times New Roman', serif" font-weight="700" fill="#0f172a">${escapeXml(title)}</text>`,
    `<text x="24" y="62" font-size="13" font-family="'Segoe UI', Arial, sans-serif" fill="#475569">${escapeXml(subtitle)}</text>`,
  ];

  numericKeys.forEach((metric, metricIndex) => {
    const sectionTop = 96 + metricIndex * sectionHeight;
    const chartLeft = 210;
    const chartRight = width - 36;
    const chartWidth = chartRight - chartLeft;
    const chartHeight = 110;
    const chartBottom = sectionTop + chartHeight;
    const values = rows.map((row) => {
      const raw = row?.[metric];
      return typeof raw === 'number' && Number.isFinite(raw) ? raw : 0;
    });
    const minValue = Math.min(0, ...values);
    const maxValue = Math.max(0, ...values, 1e-9);
    const span = maxValue - minValue || 1;
    const zeroX = chartLeft + ((0 - minValue) / span) * chartWidth;
    const barHeight = Math.max(16, Math.floor(chartHeight / Math.max(rows.length, 1)) - 6);

    elements.push(
      `<text x="24" y="${sectionTop + 16}" font-size="15" font-family="'Segoe UI', Arial, sans-serif" font-weight="700" fill="#0f172a">${escapeXml(metric)}</text>`,
    );
    elements.push(
      `<line x1="${zeroX}" y1="${sectionTop + 22}" x2="${zeroX}" y2="${chartBottom + 4}" stroke="#94a3b8" stroke-dasharray="4 4"/>`,
    );
    elements.push(
      `<rect x="${chartLeft}" y="${sectionTop + 22}" width="${chartWidth}" height="${chartHeight}" fill="#ffffff" stroke="#cbd5e1"/>`,
    );

    rows.forEach((row, rowIndex) => {
      const label = row?.[labelKey] ?? `row_${rowIndex + 1}`;
      const value = values[rowIndex];
      const y = sectionTop + 28 + rowIndex * (barHeight + 6);
      const scaledX = chartLeft + ((value - minValue) / span) * chartWidth;
      const rectX = Math.min(zeroX, scaledX);
      const rectWidth = Math.max(1, Math.abs(scaledX - zeroX));

      elements.push(
        `<text x="34" y="${y + barHeight - 2}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#0f172a">${escapeXml(String(label))}</text>`,
      );
      elements.push(
        `<rect x="${rectX}" y="${y}" width="${rectWidth}" height="${barHeight}" rx="3" fill="${colors[metricIndex % colors.length]}" opacity="0.85"/>`,
      );
      elements.push(
        `<text x="${Math.min(chartRight - 80, Math.max(chartLeft + 8, scaledX + 8))}" y="${y + barHeight - 2}" font-size="11" font-family="'Segoe UI', Arial, sans-serif" fill="#0f172a">${escapeXml(formatValue(value))}</text>`,
      );
    });
  });

  elements.push(
    `<text x="24" y="${height - 14}" font-size="12" font-family="'Segoe UI', Arial, sans-serif" fill="#64748b">Checked SVG export generated from selector $.${escapeXml(selector)}. Additional source fields remain available in the print HTML and JSON source.</text>`,
  );
  elements.push('</svg>');
  return elements.join('\n');
}

function buildPrintHtml({ title, subtitle, sourceFiles, inlineSvg, tableColumns, tableRows }) {
  const headerCells = (tableColumns ?? [])
    .map((column) => `<th>${escapeHtml(column)}</th>`)
    .join('');
  const rowMarkup = (tableRows ?? [])
    .map(
      (row) =>
        `<tr>${(tableColumns ?? [])
          .map((column) => `<td>${escapeHtml(formatValue(row[column]))}</td>`)
          .join('')}</tr>`,
    )
    .join('');
  const sourceList = sourceFiles.map((sourceFile) => `<li>${escapeHtml(sourceFile)}</li>`).join('');

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(title)}</title>
  <style>
    body { font-family: Georgia, "Times New Roman", serif; margin: 24px; color: #0f172a; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p.meta { margin: 0 0 18px; color: #475569; font-family: "Segoe UI", Arial, sans-serif; }
    .figure-shell { border: 1px solid #cbd5e1; padding: 12px; border-radius: 12px; background: #fff; }
    svg { width: 100%; height: auto; display: block; }
    table { width: 100%; border-collapse: collapse; margin-top: 18px; font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; }
    th, td { border: 1px solid #cbd5e1; padding: 8px; text-align: left; }
    th { background: #dbeafe; color: #1e3a8a; }
    ul { font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; color: #334155; }
    @page { size: A4 landscape; margin: 16mm; }
  </style>
</head>
<body>
  <h1>${escapeHtml(title)}</h1>
  <p class="meta">${escapeHtml(subtitle)}</p>
  <div class="figure-shell">
    ${inlineSvg}
  </div>
  ${
    tableColumns && tableColumns.length
      ? `<table><thead><tr>${headerCells}</tr></thead><tbody>${rowMarkup}</tbody></table>`
      : ''
  }
  <h2 style="font-size:18px; margin-top: 20px;">Source files</h2>
  <ul>${sourceList}</ul>
</body>
</html>`;
}

function writePdf(htmlPath, pdfPath, browserPath) {
  const fileUrl = pathToFileURL(htmlPath).href;
  const result = spawnSync(
    browserPath,
    [
      '--headless=new',
      '--disable-gpu',
      '--allow-file-access-from-files',
      '--no-pdf-header-footer',
      `--print-to-pdf=${pdfPath}`,
      fileUrl,
    ],
    { stdio: 'ignore' },
  );

  if (result.status !== 0 || !fs.existsSync(pdfPath)) {
    throw new Error(`Failed to generate PDF for ${htmlPath}`);
  }
}

function updateArtifactIndex(
  index,
  exportedSurfaceIds,
  sourceCompanionBySurfaceId,
  provenanceCompanionBySurfaceId,
) {
  const exportedSet = new Set(exportedSurfaceIds);
  index.notes[1] =
    'Headline surfaces in this slice, plus the indexed focused-VOI and latest-checked-campaign cohort/support-bin composition tables, now have checked SVG, PDF, print-ready HTML, co-packaged CSV/JSON source companions, and co-packaged JSON provenance companions under out/headline_exports/current_checked/. The same reviewer-package boundary now also carries checked local companion bundles for the full-suite verdict, the broad-cold runtime-observability lane, threshold-sensitivity lane, optional-stopping lane, perturbation lane, and public-transfer lane under out/headline_exports/current_checked/.';
  index.notes[2] =
    'Checked local companion bundles now exist for the threshold_sensitivity, broad_cold_proof runtime-observability, optional_stopping_coverage, perturbation_flip_radius, and public_transfer suite roles. The threshold bundle backs the one-factor-at-a-time sweep surfaces for certificate threshold, fast-path threshold, and certified-set cap; the broad-cold bundle backs the runtime quantiles, action-family budget-share surfaces, and stage-timing surfaces; the optional-stopping bundle backs the route-level CS method/delta, validity, and lane-size proof surfaces; the perturbation bundle backs the real-lane and exact-synthetic flip-radius proof surfaces; and the public-transfer bundle backs both the leave-one-corridor-family-out and leave-one-weather-regime-out transfer slices. These checked bundles prove the evaluator surfaces exist, but they do not by themselves close the corresponding G11 or P14 gates unless the emitted metrics are green.';
  index.notes[3] =
    'The quickstart command sections at docs/reviewer_quickstart.md#focused-voi-headline-table-and-figure-commands, docs/reviewer_quickstart.md#focused-voi-additional-table-commands, docs/reviewer_quickstart.md#latest-checked-campaign-table-and-figure-commands, docs/reviewer_quickstart.md#checked-full-suite-verdict-companion, docs/reviewer_quickstart.md#checked-threshold-sensitivity-lane, docs/reviewer_quickstart.md#checked-optional-stopping-coverage-lane, docs/reviewer_quickstart.md#checked-perturbation-flip-radius-lane, docs/reviewer_quickstart.md#checked-public-transfer-lane, docs/reviewer_quickstart.md#checked-runtime-observability-lane, and docs/reviewer_quickstart.md#headline-svg-and-pdf-export-commands now give one documented command block for every indexed headline table and figure source surface in this slice together with the explicit focused-VOI and latest-checked-campaign cohort/support-bin composition tables, the runtime-observability reviewer surfaces, and the checked reviewer-package companion bundles.';

  for (const surface of index.surfaces) {
    if (!exportedSet.has(surface.surface_id)) continue;
    const stem = sanitizeFileStem(surface.surface_id);
    surface.rendered_outputs = [
      relativeOutputPath(`${stem}.svg`),
      relativeOutputPath(`${stem}.pdf`),
      relativeOutputPath(`${stem}.print.html`),
    ];
    const sourceCompanion = sourceCompanionBySurfaceId.get(surface.surface_id);
    const provenanceCompanion = provenanceCompanionBySurfaceId.get(surface.surface_id);
    surface.packaged_source_companions = sourceCompanion ? [sourceCompanion.relativePath] : [];
    surface.packaged_provenance_companions = provenanceCompanion
      ? [provenanceCompanion.relativePath]
      : [];
    const formats = new Set(surface.export_formats_available ?? []);
    formats.add('svg');
    formats.add('pdf');
    formats.add('html');
    if (sourceCompanion?.format) {
      formats.add(sourceCompanion.format);
    }
    if (provenanceCompanion?.format) {
      formats.add(provenanceCompanion.format);
    }
    surface.export_formats_available = [...formats];
    surface.quickstart_reference = QUICKSTART_EXPORT_REFERENCE;
    const cleaned = stripNoRenderClaim(surface.status_note ?? '');
    const sourcePhrase = sourceCompanion
      ? `a co-packaged ${sourceCompanion.label} source companion`
      : 'the indexed source files';
    const provenancePhrase = provenanceCompanion
      ? 'a co-packaged JSON provenance companion with git, lockfile, container, and policy identity'
      : 'the indexed provenance sources';
    const suffix = ` Checked SVG, PDF, print-ready HTML, ${sourcePhrase}, and ${provenancePhrase} exist under out/headline_exports/current_checked/ for this headline surface.`;
    surface.status_note = `${cleaned}${suffix}`.trim();
  }
}

function main() {
  ensureDir(outputDir);
  const browserPath = findBrowserExecutable();
  if (!browserPath) {
    throw new Error('Could not find Microsoft Edge or Google Chrome for PDF export.');
  }

  stageFullSuiteVerdictCompanion();
  stageRuntimeObservabilityCompanion();
  stageThresholdSensitivityCompanion();
  stageOptionalStoppingCompanion();
  stagePerturbationCompanion();
  stagePublicTransferCompanion();
  stageHotRerunCompanion();
  writeLaneArtifactGenerationSummary(path.join(outputDir, FULL_SUITE_COMPANION_DIR_NAME));
  const artifactIndex = JSON.parse(fs.readFileSync(artifactIndexPath, 'utf8'));
  const headlineSurfaceIds =
    artifactIndex.inventory_sections.find((section) => section.section_id === 'headline_surfaces')
      ?.surface_ids ?? [];
  const exportedSurfaceIds = [...new Set([...headlineSurfaceIds, ...EXPLICIT_REVIEWER_EXPORT_SURFACE_IDS])];
  const exportedSurfaces = artifactIndex.surfaces.filter((surface) =>
    exportedSurfaceIds.includes(surface.surface_id),
  );
  const sourceCompanionBySurfaceId = new Map();
  const provenanceCompanionBySurfaceId = new Map();

  for (const surface of exportedSurfaces) {
    const stem = sanitizeFileStem(surface.surface_id);
    const svgPath = path.join(outputDir, `${stem}.svg`);
    const htmlPath = path.join(outputDir, `${stem}.print.html`);
    const pdfPath = path.join(outputDir, `${stem}.pdf`);

    if (surface.surface_type === 'table') {
      const table = buildTableRows(surface);
      const svg = buildTableSvg({
        title: surface.title,
        subtitle: table.subtitle,
        columns: table.columns,
        rows: table.rows,
      });
      const html = buildPrintHtml({
        title: surface.title,
        subtitle: table.subtitle,
        sourceFiles: surface.source_files,
        inlineSvg: svg,
        tableColumns: table.columns,
        tableRows: table.rows.slice(0, 24),
      });
      fs.writeFileSync(svgPath, `${svg}\n`, 'utf8');
      fs.writeFileSync(htmlPath, html, 'utf8');
      writePdf(htmlPath, pdfPath, browserPath);
      sourceCompanionBySurfaceId.set(
        surface.surface_id,
        writeSourceCompanionForTable(surface, table, stem),
      );
      provenanceCompanionBySurfaceId.set(
        surface.surface_id,
        writeProvenanceCompanionForSurface(surface, stem),
      );
    } else if (surface.surface_type === 'figure') {
      const figure = buildFigureSeries(surface);
      const tableColumns = [figure.labelKey, ...figure.numericKeys];
      const tableRows = figure.rows.map((row, index) =>
        Object.fromEntries(
          tableColumns.map((column) => [column, column === figure.labelKey ? row?.[column] ?? `row_${index + 1}` : row?.[column]]),
        ),
      );
      const svg = buildFigureSvg({
        title: surface.title,
        subtitle: figure.subtitle,
        selector: figure.selector,
        rows: figure.rows,
        labelKey: figure.labelKey,
        numericKeys: figure.numericKeys,
      });
      const html = buildPrintHtml({
        title: surface.title,
        subtitle: figure.subtitle,
        sourceFiles: surface.source_files,
        inlineSvg: svg,
        tableColumns,
        tableRows,
      });
      fs.writeFileSync(svgPath, `${svg}\n`, 'utf8');
      fs.writeFileSync(htmlPath, html, 'utf8');
      writePdf(htmlPath, pdfPath, browserPath);
      sourceCompanionBySurfaceId.set(
        surface.surface_id,
        writeSourceCompanionForFigure(surface, figure, stem),
      );
      provenanceCompanionBySurfaceId.set(
        surface.surface_id,
        writeProvenanceCompanionForSurface(surface, stem),
      );
    }
  }

  updateArtifactIndex(
    artifactIndex,
    exportedSurfaceIds,
    sourceCompanionBySurfaceId,
    provenanceCompanionBySurfaceId,
  );
  fs.writeFileSync(artifactIndexPath, `${JSON.stringify(artifactIndex, null, 2)}\n`, 'utf8');

  console.log(
    `Exported ${exportedSurfaces.length} reviewer surfaces to ${path.relative(repoRoot, outputDir)}`,
  );
}

main();
