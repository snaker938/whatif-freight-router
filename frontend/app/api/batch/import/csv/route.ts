import { NextResponse } from 'next/server';

function backendBase(): string {
  return process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

export async function POST(req: Request) {
  const body = await req.text();
  const resp = await fetch(`${backendBase()}/batch/import/csv`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
    cache: 'no-store',
  });
  const text = await resp.text();
  return new NextResponse(text, {
    status: resp.status,
    headers: { 'content-type': 'application/json' },
  });
}
