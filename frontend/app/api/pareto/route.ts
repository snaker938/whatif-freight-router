import { NextResponse } from 'next/server';

import {
  classifyBackendTransportError,
  computeAttemptTimeoutMs,
  fetchBackend,
} from '../../lib/backendFetch';

export async function POST(req: Request) {
  const body = await req.text();

  let resp: Response;
  try {
    resp = await fetchBackend('/pareto', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
      requestSignal: req.signal,
      timeoutMs: computeAttemptTimeoutMs(),
    });
  } catch (error: unknown) {
    const classified = classifyBackendTransportError(error);
    return new NextResponse(
      JSON.stringify({
        detail: {
          message: 'Unable to reach backend /pareto endpoint.',
          reason_code: classified.reasonCode,
          cause: classified.cause || classified.detail,
        },
      }),
      {
        status: 502,
        headers: { 'content-type': 'application/json' },
      },
    );
  }

  const text = await resp.text();
  const outHeaders: Record<string, string> = {
    'content-type': 'application/json',
    'cache-control': 'no-store',
  };
  const requestId = resp.headers.get('x-route-request-id');
  if (requestId) {
    outHeaders['x-route-request-id'] = requestId;
  }
  return new NextResponse(text, {
    status: resp.status,
    headers: outHeaders,
  });
}
