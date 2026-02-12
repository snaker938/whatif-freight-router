import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  const backendBase =
    process.env.BACKEND_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    'http://localhost:8000';
  const body = await req.text();

  const resp = await fetch(`${backendBase}/scenario/compare`, {
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

