import { NextResponse } from 'next/server';

export async function GET() {
  const backendBase =
    process.env.BACKEND_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    'http://localhost:8000';

  const resp = await fetch(`${backendBase}/vehicles`, { cache: 'no-store' });
  const text = await resp.text();

  return new NextResponse(text, { status: resp.status, headers: { 'content-type': 'application/json' } });
}
