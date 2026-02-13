import { NextResponse } from 'next/server';

function backendBase(): string {
  return process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

function forwardedAuthHeaders(req: Request): Record<string, string> {
  const headers: Record<string, string> = {};
  const auth = req.headers.get('authorization');
  const token = req.headers.get('x-api-token');
  if (auth) headers.authorization = auth;
  if (token) headers['x-api-token'] = token;
  return headers;
}

export async function GET(req: Request) {
  const resp = await fetch(`${backendBase()}/oracle/quality/dashboard.csv`, {
    cache: 'no-store',
    headers: forwardedAuthHeaders(req),
  });
  const csv = await resp.text();
  return new NextResponse(csv, {
    status: resp.status,
    headers: {
      'content-type': resp.headers.get('content-type') ?? 'text/csv; charset=utf-8',
      'content-disposition': resp.headers.get('content-disposition') ?? 'attachment; filename="oracle_quality_dashboard.csv"',
    },
  });
}
