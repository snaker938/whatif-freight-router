export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

function backendBaseUrl(): string {
  return (
    process.env.BACKEND_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    'http://localhost:8000'
  );
}

export async function POST(req: Request): Promise<Response> {
  const body = await req.text();

  const resp = await fetch(`${backendBaseUrl()}/api/pareto/stream`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
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
