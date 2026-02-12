import { NextResponse } from 'next/server';

function backendBase(): string {
  return process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

type Ctx = { params: Promise<{ experimentId: string }> };

export async function GET(_: Request, ctx: Ctx) {
  const { experimentId } = await ctx.params;
  const resp = await fetch(`${backendBase()}/experiments/${experimentId}`, { cache: 'no-store' });
  const text = await resp.text();
  return new NextResponse(text, { status: resp.status, headers: { 'content-type': 'application/json' } });
}

export async function PUT(req: Request, ctx: Ctx) {
  const { experimentId } = await ctx.params;
  const body = await req.text();
  const resp = await fetch(`${backendBase()}/experiments/${experimentId}`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body,
    cache: 'no-store',
  });
  const text = await resp.text();
  return new NextResponse(text, { status: resp.status, headers: { 'content-type': 'application/json' } });
}

export async function DELETE(_: Request, ctx: Ctx) {
  const { experimentId } = await ctx.params;
  const resp = await fetch(`${backendBase()}/experiments/${experimentId}`, {
    method: 'DELETE',
    cache: 'no-store',
  });
  const text = await resp.text();
  return new NextResponse(text, { status: resp.status, headers: { 'content-type': 'application/json' } });
}
