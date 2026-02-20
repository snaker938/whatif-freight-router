import { NextResponse } from 'next/server';

function backendBase(): string {
  return process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

export async function GET() {
  const resp = await fetch(`${backendBase()}/cache/stats`, { cache: 'no-store' });
  const text = await resp.text();
  return new NextResponse(text, {
    status: resp.status,
    headers: { 'content-type': 'application/json' },
  });
}
