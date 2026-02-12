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

type Ctx = { params: Promise<{ experimentId: string }> };

export async function POST(req: Request, ctx: Ctx) {
  const { experimentId } = await ctx.params;
  const body = await req.text();
  const resp = await fetch(`${backendBase()}/experiments/${experimentId}/compare`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...forwardedAuthHeaders(req) },
    body,
    cache: 'no-store',
  });
  const text = await resp.text();
  return new NextResponse(text, { status: resp.status, headers: { 'content-type': 'application/json' } });
}
