import { NextResponse } from 'next/server';

function forwardedAuthHeaders(req: Request): Record<string, string> {
  const headers: Record<string, string> = {};
  const auth = req.headers.get('authorization');
  const token = req.headers.get('x-api-token');
  if (auth) headers.authorization = auth;
  if (token) headers['x-api-token'] = token;
  return headers;
}

export async function GET(req: Request) {
  const backendBase =
    process.env.BACKEND_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    'http://localhost:8000';

  const resp = await fetch(`${backendBase}/vehicles`, {
    cache: 'no-store',
    headers: forwardedAuthHeaders(req),
  });
  const text = await resp.text();

  return new NextResponse(text, { status: resp.status, headers: { 'content-type': 'application/json' } });
}
