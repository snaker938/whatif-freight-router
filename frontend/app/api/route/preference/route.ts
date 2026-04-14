import { NextResponse } from 'next/server';

import {
  classifyBackendTransportError,
  computeRouteFallbackTimeoutMs,
  fetchBackend,
} from '../../../lib/backendFetch';

export async function POST(req: Request) {
  const body = await req.text();

  let resp: Response;
  try {
    resp = await fetchBackend('/route/preference', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
      requestSignal: req.signal,
      timeoutMs: computeRouteFallbackTimeoutMs(),
    });
  } catch (error: unknown) {
    const classified = classifyBackendTransportError(error);
    return new NextResponse(
      JSON.stringify({
        detail: {
          message: 'Unable to reach backend /route/preference endpoint.',
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
  return new NextResponse(text, {
    status: resp.status,
    headers: {
      'content-type': 'application/json',
      'cache-control': 'no-store',
    },
  });
}
