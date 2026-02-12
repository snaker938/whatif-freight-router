import { NextResponse } from 'next/server';

function backendBase(): string {
  return process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

type Ctx = { params: Promise<{ experimentId: string }> };

export async function POST(req: Request, ctx: Ctx) {
  const { experimentId } = await ctx.params;
  const body = await req.text();
  const resp = await fetch(`${backendBase()}/experiments/${experimentId}/compare`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
    cache: 'no-store',
  });
  const text = await resp.text();
  return new NextResponse(text, { status: resp.status, headers: { 'content-type': 'application/json' } });
}
