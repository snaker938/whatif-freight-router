function extractErrorMessage(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return 'Request failed';

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (typeof parsed === 'string' && parsed.trim()) return parsed.trim();
    if (parsed && typeof parsed === 'object') {
      const detail = (parsed as Record<string, unknown>).detail;
      if (typeof detail === 'string' && detail.trim()) return detail.trim();
      if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
        const detailObj = detail as Record<string, unknown>;
        const detailMessage =
          typeof detailObj.message === 'string' ? detailObj.message.trim() : '';
        const reasonCode =
          typeof detailObj.reason_code === 'string' ? detailObj.reason_code.trim() : '';
        const cause = typeof detailObj.cause === 'string' ? detailObj.cause.trim() : '';
        const warningSummary = Array.isArray(detailObj.warnings)
          ? detailObj.warnings
              .map((warning) => (typeof warning === 'string' ? warning.trim() : ''))
              .filter(Boolean)
              .slice(0, 2)
              .join('; ')
          : '';
        if (detailMessage && warningSummary) {
          const core = reasonCode ? `${detailMessage} (${reasonCode})` : detailMessage;
          return cause ? `${core} ${warningSummary} cause=${cause}` : `${core} ${warningSummary}`;
        }
        if (detailMessage) {
          const core = reasonCode ? `${detailMessage} (${reasonCode})` : detailMessage;
          return cause ? `${core} cause=${cause}` : core;
        }
        if (reasonCode) return cause ? `${reasonCode} cause=${cause}` : reasonCode;
        if (cause) return cause;
      }
      if (Array.isArray(detail) && detail.length) {
        const joined = detail
          .map((entry) => {
            if (typeof entry === 'string') return entry;
            if (entry && typeof entry === 'object') {
              const msg = (entry as Record<string, unknown>).msg;
              return typeof msg === 'string' ? msg : '';
            }
            return '';
          })
          .filter(Boolean)
          .join('; ')
          .trim();
        if (joined) return joined;
      }
      const message = (parsed as Record<string, unknown>).message;
      if (typeof message === 'string' && message.trim()) return message.trim();
    }
  } catch {
    // Response was not JSON.
  }

  return trimmed;
}

async function readJSON<T>(res: Response): Promise<T> {
  const text = await res.text();
  if (!res.ok) {
    throw new Error(extractErrorMessage(text));
  }
  if (!text.trim()) {
    throw new Error('Server returned an empty response');
  }
  return JSON.parse(text) as T;
}

async function readText(res: Response): Promise<string> {
  const text = await res.text();
  if (!res.ok) {
    throw new Error(extractErrorMessage(text));
  }
  return text;
}

export async function postJSON<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(headers ?? {}) },
    body: JSON.stringify(body),
    cache: 'no-store',
    signal,
  });

  return readJSON<T>(res);
}

export async function postJSONWithMeta<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<{ data: T; response: Response }> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(headers ?? {}) },
    body: JSON.stringify(body),
    cache: 'no-store',
    signal,
  });
  const text = await res.text();
  if (!res.ok) {
    const err = new Error(extractErrorMessage(text)) as Error & {
      response?: Response;
      responseText?: string;
    };
    err.response = res;
    err.responseText = text;
    throw err;
  }
  if (!text.trim()) {
    throw new Error('Server returned an empty response');
  }
  return { data: JSON.parse(text) as T, response: res };
}

export async function getJSON<T>(
  path: string,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<T> {
  const res = await fetch(path, {
    method: 'GET',
    headers: { ...(headers ?? {}) },
    cache: 'no-store',
    signal,
  });
  return readJSON<T>(res);
}

export async function getText(
  path: string,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<string> {
  const res = await fetch(path, {
    method: 'GET',
    headers: { ...(headers ?? {}) },
    cache: 'no-store',
    signal,
  });
  return readText(res);
}

export async function putJSON<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<T> {
  const res = await fetch(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...(headers ?? {}) },
    body: JSON.stringify(body),
    cache: 'no-store',
    signal,
  });
  return readJSON<T>(res);
}

export async function deleteJSON<T>(
  path: string,
  signal?: AbortSignal,
  headers?: HeadersInit,
): Promise<T> {
  const res = await fetch(path, {
    method: 'DELETE',
    headers: { ...(headers ?? {}) },
    cache: 'no-store',
    signal,
  });
  return readJSON<T>(res);
}

type PostNDJSONOptions<TEvent> = {
  signal?: AbortSignal;
  headers?: HeadersInit;
  onEvent: (event: TEvent) => void;
  stallTimeoutMs?: number;
};

export async function postNDJSON<TEvent extends object>(
  path: string,
  body: unknown,
  options: PostNDJSONOptions<TEvent>,
): Promise<void> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
    body: JSON.stringify(body),
    cache: 'no-store',
    signal: options.signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(extractErrorMessage(text));
  }

  if (!res.body) {
    throw new Error('Streaming response was empty');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const stallTimeoutMs =
    typeof options.stallTimeoutMs === 'number' && options.stallTimeoutMs > 0
      ? Math.floor(options.stallTimeoutMs)
      : 0;

  async function readWithOptionalTimeout(): Promise<ReadableStreamReadResult<Uint8Array>> {
    if (stallTimeoutMs <= 0) {
      return reader.read();
    }
    return new Promise<ReadableStreamReadResult<Uint8Array>>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error(`Streaming response stalled for ${stallTimeoutMs}ms`));
      }, stallTimeoutMs);
      void reader
        .read()
        .then((result) => {
          clearTimeout(timeoutId);
          resolve(result);
        })
        .catch((err) => {
          clearTimeout(timeoutId);
          reject(err);
        });
    });
  }

  try {
    while (true) {
      const { done, value } = await readWithOptionalTimeout();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        options.onEvent(JSON.parse(trimmed) as TEvent);
      }
    }

    buffer += decoder.decode();
    const tail = buffer.trim();
    if (tail) {
      options.onEvent(JSON.parse(tail) as TEvent);
    }
  } catch (error: unknown) {
    try {
      await reader.cancel();
    } catch {
      // noop: cancellation best-effort only.
    }
    throw error;
  }
}
