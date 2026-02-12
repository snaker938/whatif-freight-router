export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

function backendBaseUrl(): string {
  return (
    process.env.BACKEND_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    'http://localhost:8000'
  );
}

function forwardedAuthHeaders(req: Request): Record<string, string> {
  const headers: Record<string, string> = {};
  const auth = req.headers.get('authorization');
  const token = req.headers.get('x-api-token');
  if (auth) headers.authorization = auth;
  if (token) headers['x-api-token'] = token;
  return headers;
}

export async function POST(req: Request): Promise<Response> {
  const body = await req.text();

  const resp = await fetch(`${backendBaseUrl()}/api/pareto/stream`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...forwardedAuthHeaders(req) },
    body,
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

  return new Response(resp.body, {
    status: resp.status,
    headers: {
      'content-type': resp.headers.get('content-type') ?? 'application/x-ndjson',
      'cache-control': 'no-store',
    },
  });
}
