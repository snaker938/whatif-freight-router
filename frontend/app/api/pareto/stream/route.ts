export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

import {
  classifyBackendTransportError,
  computeAttemptTimeoutMs,
  fetchBackend,
} from '../../../lib/backendFetch';

function isInvalidStreamStateError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const code =
    'code' in error && typeof (error as { code?: unknown }).code === 'string'
      ? String((error as { code?: string }).code)
      : '';
  const message =
    'message' in error && typeof (error as { message?: unknown }).message === 'string'
      ? String((error as { message?: string }).message)
      : '';
  const lowered = `${code} ${message}`.toLowerCase();
  return (
    lowered.includes('err_invalid_state') ||
    lowered.includes('controller is already closed') ||
    lowered.includes('reader is not attached to a stream')
  );
}

export async function POST(req: Request): Promise<Response> {
  const body = await req.text();

  let resp: Response;
  try {
    resp = await fetchBackend('/api/pareto/stream', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
      requestSignal: req.signal,
      timeoutMs: computeAttemptTimeoutMs(),
    });
  } catch (error: unknown) {
    const classified = classifyBackendTransportError(error);
    return new Response(
      JSON.stringify({
        detail: {
          message: 'Unable to reach backend streaming endpoint.',
          reason_code: classified.reasonCode,
          cause: classified.cause || classified.detail,
        },
      }),
      {
        status: 502,
        headers: {
          'content-type': 'application/json',
          'cache-control': 'no-store',
        },
      },
    );
  }

  if (!resp.ok) {
    const text = await resp.text();
    const outHeaders: Record<string, string> = {
      'content-type': resp.headers.get('content-type') ?? 'application/json',
      'cache-control': 'no-store',
    };
    const requestId = resp.headers.get('x-route-request-id');
    if (requestId) {
      outHeaders['x-route-request-id'] = requestId;
    }
    return new Response(text, {
      status: resp.status,
      headers: outHeaders,
    });
  }

  if (!resp.body) {
    const text = await resp.text();
    const outHeaders: Record<string, string> = {
      'content-type': resp.headers.get('content-type') ?? 'application/json',
      'cache-control': 'no-store',
    };
    const requestId = resp.headers.get('x-route-request-id');
    if (requestId) {
      outHeaders['x-route-request-id'] = requestId;
    }
    return new Response(text, {
      status: resp.status,
      headers: outHeaders,
    });
  }

  const reader = resp.body.getReader();
  const encoder = new TextEncoder();
  let streamClosed = false;
  let readerCancelled = false;
  let readerReleased = false;
  let cancelInFlight = false;
  let readLoopDone = false;

  const safeReleaseLock = (): void => {
    if (readerReleased) return;
    try {
      reader.releaseLock();
    } catch {
      // no-op: lock release is best-effort
    } finally {
      readerReleased = true;
    }
  };

  const safeCancelReader = async (): Promise<void> => {
    if (readLoopDone || readerCancelled || readerReleased || cancelInFlight) return;
    cancelInFlight = true;
    try {
      await reader.cancel();
      readerCancelled = true;
    } catch (error: unknown) {
      if (!isInvalidStreamStateError(error)) {
        // no-op: cancellation is best-effort and transport can close underneath us
      }
      readerCancelled = true;
    } finally {
      cancelInFlight = false;
    }
  };

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const safeEnqueue = (payload: Uint8Array): void => {
        if (streamClosed) return;
        try {
          controller.enqueue(payload);
        } catch (error: unknown) {
          if (isInvalidStreamStateError(error)) {
            streamClosed = true;
            return;
          }
          throw error;
        }
      };

      const safeClose = (): void => {
        if (streamClosed) return;
        try {
          controller.close();
        } catch (error: unknown) {
          if (!isInvalidStreamStateError(error)) throw error;
        } finally {
          streamClosed = true;
        }
      };

      void (async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              readLoopDone = true;
              break;
            }
            if (value) safeEnqueue(value);
          }
        } catch (error: unknown) {
          const classified = classifyBackendTransportError(error);
          const fatal = JSON.stringify({
            type: 'fatal',
            message: 'Streaming proxy interrupted while reading backend response.',
            reason_code: classified.reasonCode,
            warnings: [classified.cause || classified.detail],
          });
          safeEnqueue(encoder.encode(`${fatal}\n`));
        } finally {
          void safeCancelReader().catch(() => {
            // no-op: best effort cancellation only
          });
          safeReleaseLock();
          safeClose();
        }
      })();
    },
    cancel() {
      void safeCancelReader()
        .catch(() => {
          // no-op: best effort cancellation only
        })
        .finally(() => {
          try {
            safeReleaseLock();
          } catch {
            // no-op
          }
        });
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'content-type': 'application/x-ndjson',
      'cache-control': 'no-store',
      ...(resp.headers.get('x-route-request-id')
        ? { 'x-route-request-id': resp.headers.get('x-route-request-id') as string }
        : {}),
    },
  });
}
