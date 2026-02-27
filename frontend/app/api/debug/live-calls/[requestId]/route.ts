export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

import { NextResponse } from 'next/server';

import {
  classifyBackendTransportError,
  computeAttemptTimeoutMs,
  fetchBackend,
} from '../../../../lib/backendFetch';

export async function GET(
  _req: Request,
  context: { params: Promise<{ requestId: string }> },
): Promise<Response> {
  const params = await context.params;
  const requestId = String(params.requestId || '').trim();
  if (!requestId) {
    return new NextResponse(
      JSON.stringify({
        detail: {
          message: 'requestId is required.',
          reason_code: 'invalid_request_id',
        },
      }),
      {
        status: 400,
        headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
      },
    );
  }

  let resp: Response;
  try {
    resp = await fetchBackend(`/debug/live-calls/${encodeURIComponent(requestId)}`, {
      method: 'GET',
      requestSignal: _req.signal,
      timeoutMs: computeAttemptTimeoutMs(),
    });
  } catch (error: unknown) {
    const classified = classifyBackendTransportError(error);
    return new NextResponse(
      JSON.stringify({
        detail: {
          message: 'Unable to reach backend live-call debug endpoint.',
          reason_code: classified.reasonCode,
          cause: classified.cause || classified.detail,
        },
      }),
      {
        status: 502,
        headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
      },
    );
  }

  const text = await resp.text();
  return new NextResponse(text, {
    status: resp.status,
    headers: {
      'content-type': resp.headers.get('content-type') ?? 'application/json',
      'cache-control': 'no-store',
    },
  });
}
