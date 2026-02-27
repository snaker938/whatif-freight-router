import { NextResponse } from 'next/server';

function backendBase(): string {
  return process.env.BACKEND_INTERNAL_URL ?? process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

export async function GET() {
  try {
    const resp = await fetch(`${backendBase()}/health`, { cache: 'no-store' });
    const text = await resp.text();
    return new NextResponse(text, {
      status: resp.status,
      headers: { 'content-type': 'application/json' },
    });
  } catch (error: unknown) {
    const cause = error instanceof Error ? error.message : String(error ?? 'Unknown backend transport error');
    return NextResponse.json(
      {
        detail: {
          message: 'Unable to reach backend /health endpoint.',
          reason_code: 'backend_unreachable',
          cause,
        },
      },
      { status: 503 },
    );
  }
}
