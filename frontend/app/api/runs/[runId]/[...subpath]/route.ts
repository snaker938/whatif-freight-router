const CORE_ALLOWED = new Set([
  'manifest',
  'scenario-manifest',
  'provenance',
  'signature',
  'scenario-signature',
  'artifacts',
]);

const ARTIFACT_ALLOWED = new Set([
  'results.json',
  'results.csv',
  'metadata.json',
  'routes.geojson',
  'results_summary.csv',
  'report.pdf',
  'dccs_candidates.jsonl',
  'dccs_summary.json',
  'refined_routes.jsonl',
  'strict_frontier.jsonl',
  'winner_summary.json',
  'certificate_summary.json',
  'route_fragility_map.json',
  'competitor_fragility_breakdown.json',
  'value_of_refresh.json',
  'sampled_world_manifest.json',
  'voi_action_trace.json',
  'voi_action_scores.csv',
  'voi_stop_certificate.json',
  'final_route_trace.json',
  'od_corpus.csv',
  'od_corpus_summary.json',
  'ors_snapshot.json',
  'thesis_results.csv',
  'thesis_summary.csv',
  'methods_appendix.md',
  'thesis_report.md',
]);

function backendBase(): string {
  return process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

function isAllowedSubpath(subpath: string[]): boolean {
  if (subpath.length === 0) return false;
  if (subpath.length === 1) return CORE_ALLOWED.has(subpath[0]);
  if (subpath[0] !== 'artifacts') return false;
  if (subpath.length !== 2) return false;
  return ARTIFACT_ALLOWED.has(subpath[1]);
}

type Ctx = { params: Promise<{ runId: string; subpath: string[] }> };

export async function GET(_req: Request, ctx: Ctx): Promise<Response> {
  const { runId, subpath } = await ctx.params;
  if (!isAllowedSubpath(subpath)) {
    return new Response(JSON.stringify({ detail: 'unsupported run artifact path' }), {
      status: 404,
      headers: { 'content-type': 'application/json' },
    });
  }

  const targetPath = subpath.join('/');
  const resp = await fetch(`${backendBase()}/runs/${runId}/${targetPath}`, {
    method: 'GET',
    cache: 'no-store',
  });

  if (!resp.body) {
    const text = await resp.text();
    return new Response(text, {
      status: resp.status,
      headers: {
        'content-type': resp.headers.get('content-type') ?? 'application/json',
        'cache-control': 'no-store',
      },
    });
  }

  const headers: Record<string, string> = {
    'content-type': resp.headers.get('content-type') ?? 'application/octet-stream',
    'cache-control': 'no-store',
  };
  const contentDisposition = resp.headers.get('content-disposition');
  if (contentDisposition) headers['content-disposition'] = contentDisposition;

  return new Response(resp.body, {
    status: resp.status,
    headers,
  });
}
